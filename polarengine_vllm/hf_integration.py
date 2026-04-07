"""
PolarQuant HuggingFace Transformers Integration.

Registers PolarQuant as a native quantization method in transformers
via the external registration API (no fork needed).

Usage:
    import polarengine_vllm  # auto-registers
    from transformers import AutoModelForCausalLM

    # Load pre-quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "caiovicentino1/Qwen3.5-9B-PolarQuant-Q5",
        device_map="auto"
    )

    # Or quantize on-the-fly
    from polarengine_vllm.hf_integration import PolarQuantConfig
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B",
        quantization_config=PolarQuantConfig(weight_bits=5),
        device_map="auto",
    )
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
#  PolarQuant Math (self-contained)
# ============================================================

_CENTROIDS_CACHE = {}


def _get_centroids(bits: int) -> torch.Tensor:
    if bits in _CENTROIDS_CACHE:
        return _CENTROIDS_CACHE[bits]
    from scipy.stats import norm as sp_norm
    n = 1 << bits
    bd = torch.linspace(-4.0, 4.0, n + 1)
    ct = torch.zeros(n)
    for _ in range(100):
        for i in range(n):
            lo, hi = bd[i].item(), bd[i + 1].item()
            plo, phi = sp_norm.cdf(lo), sp_norm.cdf(hi)
            ct[i] = (sp_norm.pdf(lo) - sp_norm.pdf(hi)) / (phi - plo) if phi - plo > 1e-12 else (lo + hi) / 2
        for i in range(1, n):
            bd[i] = (ct[i - 1] + ct[i]) / 2
    _CENTROIDS_CACHE[bits] = ct
    return ct


def _build_hadamard(n: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([[1.0]])
    h = _build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def _unpack_5bit(packed: torch.Tensor, total: int) -> torch.Tensor:
    p = packed.long().reshape(-1, 5)
    b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    return torch.stack([
        (b0 >> 3) & 31, ((b0 & 7) << 2) | ((b1 >> 6) & 3),
        (b1 >> 1) & 31, ((b1 & 1) << 4) | ((b2 >> 4) & 15),
        ((b2 & 15) << 1) | ((b3 >> 7) & 1), (b3 >> 2) & 31,
        ((b3 & 3) << 3) | ((b4 >> 5) & 7), b4 & 31,
    ], dim=-1).reshape(-1)[:total].to(torch.uint8)


def _bitpack_5(codes_flat: torch.Tensor):
    total = codes_flat.shape[0]
    pad = (8 - total % 8) % 8
    c = codes_flat.long()
    if pad:
        c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
    c = c.reshape(-1, 8)
    packed = torch.stack([
        ((c[:, 0] << 3) | (c[:, 1] >> 2)).to(torch.uint8),
        (((c[:, 1] & 3) << 6) | (c[:, 2] << 1) | (c[:, 3] >> 4)).to(torch.uint8),
        (((c[:, 3] & 15) << 4) | (c[:, 4] >> 1)).to(torch.uint8),
        (((c[:, 4] & 1) << 7) | (c[:, 5] << 2) | (c[:, 6] >> 3)).to(torch.uint8),
        (((c[:, 6] & 7) << 5) | c[:, 7]).to(torch.uint8),
    ], dim=-1).reshape(-1)
    return packed, total


# ============================================================
#  PolarQuantConfig
# ============================================================

try:
    from transformers.utils.quantization_config import QuantizationConfigMixin
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    QuantizationConfigMixin = object


@dataclass
class PolarQuantConfig(QuantizationConfigMixin):
    """PolarQuant quantization config for HuggingFace transformers.

    Args:
        weight_bits: Bit width for weights (2-8, default 5).
        kv_bits: Bit width for KV cache (2-4, or None to disable).
        block_size: Hadamard block size (power of 2, default 128).
        skip_patterns: Layer name patterns to keep in BF16.
    """

    def __init__(
        self,
        weight_bits: int = 5,
        kv_bits: Optional[int] = 3,
        block_size: int = 128,
        skip_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = "polar"
        self.weight_bits = weight_bits
        self.kv_bits = kv_bits
        self.block_size = block_size
        self.skip_patterns = skip_patterns or [
            "norm", "layernorm", "rmsnorm", "bias",
            "gate.weight", "router",
        ]
        if hasattr(self, "post_init"):
            self.post_init()


# ============================================================
#  PolarQuantHfQuantizer
# ============================================================

try:
    from transformers.quantizers.base import HfQuantizer
    _HAS_HF_QUANTIZER = True
except ImportError:
    _HAS_HF_QUANTIZER = False
    HfQuantizer = object


class PolarQuantHfQuantizer(HfQuantizer if _HAS_HF_QUANTIZER else object):
    """HuggingFace quantizer for PolarQuant.

    Hooks into from_pretrained to automatically load/quantize models.
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        if _HAS_HF_QUANTIZER:
            super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ImportError("PolarQuant requires scipy: pip install scipy")

    def update_torch_dtype(self, torch_dtype):
        return torch_dtype or torch.bfloat16

    def _process_model_before_weight_loading(self, model, **kwargs):
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        model._is_polarquant = True
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False


# ============================================================
#  Auto-register with transformers
# ============================================================

def register_with_transformers():
    """Register PolarQuant as a quantization method in transformers."""
    if not _HAS_TRANSFORMERS:
        logger.debug("transformers not installed, skipping registration")
        return False

    try:
        from transformers.quantizers.auto import (
            register_quantization_config,
            register_quantizer,
        )
        register_quantization_config("polar")(PolarQuantConfig)
        register_quantizer("polar")(PolarQuantHfQuantizer)
        register_quantization_config("polarengine")(PolarQuantConfig)
        register_quantizer("polarengine")(PolarQuantHfQuantizer)
        logger.info("PolarQuant registered with transformers (polar + polarengine)")
        return True
    except (ImportError, AttributeError):
        logger.debug("transformers version too old for external registration")
        return False


# Auto-register on import
_registered = register_with_transformers()
