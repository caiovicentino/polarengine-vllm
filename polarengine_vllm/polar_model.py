"""
PolarQuantModel — standalone HuggingFace loader for PolarQuant models.

Downloads a PolarQuant model from HuggingFace Hub, dequantizes PQ codes
to BF16 on-the-fly, and loads into a standard transformers model. Supports
all PolarQuant formats: dot-separated (.codes/.norms), double-underscore
(__codes/__norms), and bit-packed (__packed/__norms/__meta).

Usage::

    from polarengine_vllm import PolarQuantModel
    model = PolarQuantModel.from_pretrained("caiovicentino1/Qwen3-8B-PQ5")
    out = model.generate("Hello, world!", max_new_tokens=64)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


# ===================================================================
# Helpers
# ===================================================================

def _parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """Split model_id into (repo_id, subfolder) for nested HF paths.

    Examples:
        "caiovicentino1/Model"        -> ("caiovicentino1/Model", None)
        "caiovicentino1/Model/PQ5"    -> ("caiovicentino1/Model", "PQ5")
        "caiovicentino1/Model/a/b"    -> ("caiovicentino1/Model", "a/b")
    """
    parts = model_id.split("/")
    if len(parts) <= 2:
        return model_id, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder


def _load_polar_config(model_dir: str) -> Optional[dict]:
    """Load polar_config.json, returning None if not found."""
    path = os.path.join(model_dir, "polar_config.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _detect_format(keys: list[str]) -> str:
    """Auto-detect tensor naming format from safetensor keys.

    Returns one of: "dot", "dunder", "packed", "bf16_only"
    """
    for k in keys:
        if k.endswith("__packed"):
            return "packed"
        if k.endswith("__codes"):
            return "dunder"
        if k.endswith(".codes"):
            return "dot"
    return "bf16_only"


def _iter_safetensors(model_dir: str):
    """Yield (name, tensor) pairs from all safetensors files in model_dir."""
    st_files = sorted(Path(model_dir).glob("*.safetensors"))
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _collect_all_keys(model_dir: str) -> list[str]:
    """Collect all tensor keys across safetensors files without loading data."""
    keys = []
    for st_file in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def _dequant_dot_format(
    model_dir: str, polar_config: Optional[dict], dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Dequant .codes/.norms/.ct_scaled format using weight_converter."""
    from polarengine_vllm.weight_converter import polar_dequant_iterator

    state = {}
    for name, tensor in polar_dequant_iterator(_iter_safetensors(model_dir), model_dir):
        state[name] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    return state


def _dequant_dunder_format(
    model_dir: str, polar_config: Optional[dict], dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Dequant __codes/__norms format (PQ Multi naming)."""
    from polarengine_vllm.weight_converter import _dequant_weight
    from polarengine_vllm.utils import get_centroids

    block_size = polar_config.get("block_size", 128) if polar_config else 128

    # Collect all tensors grouped by prefix
    raw: Dict[str, torch.Tensor] = {}
    for name, tensor in _iter_safetensors(model_dir):
        raw[name] = tensor

    state: Dict[str, torch.Tensor] = {}
    seen_prefixes: set = set()

    for key in list(raw.keys()):
        if key.endswith("__codes"):
            prefix = key[:-7]  # strip __codes
            seen_prefixes.add(prefix)

            codes = raw[key]
            norms_key = prefix + "__norms"
            norms = raw.get(norms_key)
            if norms is None:
                logger.warning("Missing norms for %s, skipping", prefix)
                continue

            # Recover original layer name: undo __ -> .
            layer_name = prefix.replace("__", ".")

            # Get bits from polar_config or infer from centroid count
            bits = 5
            if polar_config:
                layers_meta = polar_config.get("layers", {})
                meta = layers_meta.get(layer_name, {})
                bits = meta.get("bits", polar_config.get("bits", 5))

            ct = get_centroids(bits)
            weight = _dequant_weight(codes, norms, ct, block_size=block_size)

            # Trim padding
            if polar_config:
                meta = polar_config.get("layers", {}).get(layer_name, {})
                in_f = meta.get("in_features")
                if in_f and weight.shape[1] > in_f:
                    weight = weight[:, :in_f]

            state[layer_name + ".weight"] = weight.to(dtype)

        elif key.endswith("__norms"):
            continue  # handled above
        else:
            # BF16 passthrough tensor
            t = raw[key]
            state[key] = t.to(dtype) if t.is_floating_point() else t

    return state


def _dequant_packed_format(
    model_dir: str, polar_config: Optional[dict], dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Dequant __packed/__norms/__meta format (bit-packed PQ Multi)."""
    from polarengine_vllm.weight_converter import _dequant_weight
    from polarengine_vllm.utils import get_centroids
    from polarengine_vllm.kv_cache.cache import BitPacker

    block_size = polar_config.get("block_size", 128) if polar_config else 128

    raw: Dict[str, torch.Tensor] = {}
    for name, tensor in _iter_safetensors(model_dir):
        raw[name] = tensor

    state: Dict[str, torch.Tensor] = {}

    for key in list(raw.keys()):
        if key.endswith("__packed"):
            prefix = key[:-8]  # strip __packed
            packed = raw[key]
            norms = raw.get(prefix + "__norms")
            meta = raw.get(prefix + "__meta")

            if norms is None:
                logger.warning("Missing norms for %s, skipping", prefix)
                continue

            layer_name = prefix.replace("__", ".")

            # Extract bits from meta tensor or config
            bits = 5
            if meta is not None and meta.numel() >= 1:
                bits = int(meta[0].item())
            elif polar_config:
                layers_meta = polar_config.get("layers", {})
                lm = layers_meta.get(layer_name, {})
                bits = lm.get("bits", polar_config.get("bits", 5))

            # Unpack codes
            out_f = norms.shape[0]
            in_f_padded = norms.shape[1] * block_size if norms.ndim == 2 else block_size
            if meta is not None and meta.numel() >= 2:
                in_f_padded = int(meta[1].item())

            codes = BitPacker.unpack(packed, bits, in_f_padded)

            ct = get_centroids(bits)
            weight = _dequant_weight(codes, norms, ct, block_size=block_size)

            # Trim padding
            if polar_config:
                lm = polar_config.get("layers", {}).get(layer_name, {})
                in_f = lm.get("in_features")
                if in_f and weight.shape[1] > in_f:
                    weight = weight[:, :in_f]

            state[layer_name + ".weight"] = weight.to(dtype)

        elif key.endswith(("__norms", "__meta")):
            continue
        else:
            t = raw[key]
            state[key] = t.to(dtype) if t.is_floating_point() else t

    return state


# ===================================================================
# PolarQuantModel
# ===================================================================

class PolarQuantModel:
    """Wrapper around a HuggingFace model loaded from PolarQuant weights.

    Attributes:
        model: The underlying ``AutoModelForCausalLM`` instance.
        tokenizer: The loaded tokenizer (if available).
        kv_cache: Optional ``PolarKVCache`` for compressed KV inference.
        polar_config: Parsed ``polar_config.json`` (or None).
    """

    def __init__(self, model, tokenizer, kv_cache=None, polar_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.polar_config = polar_config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        kv_cache_nbits: Optional[int] = None,
        trust_remote_code: bool = False,
    ) -> "PolarQuantModel":
        """Load a PolarQuant model from HuggingFace Hub or local directory.

        Args:
            model_id: HF repo id (e.g. "caiovicentino1/Qwen3-8B-PQ5") or
                local path.  Nested paths like "user/repo/PQ5" are handled
                by splitting at the second slash for subfolder download.
            device_map: Device map for model placement (default "auto").
            dtype: Target dtype for dequantized weights (default BF16).
            kv_cache_nbits: If set (2, 3, or 4), create a PolarKVCache.
            trust_remote_code: Pass to transformers for custom architectures.

        Returns:
            PolarQuantModel with .model, .tokenizer, .generate() attributes.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # -- Step 1: Resolve model directory ----------------------------------
        if os.path.isdir(model_id):
            model_dir = model_id
        else:
            from huggingface_hub import snapshot_download
            repo_id, subfolder = _parse_model_id(model_id)
            logger.info("Downloading %s (subfolder=%s)", repo_id, subfolder)
            model_dir = snapshot_download(
                repo_id,
                allow_patterns=[
                    f"{subfolder}/**" if subfolder else "**",
                ],
            )
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)

        logger.info("Model directory: %s", model_dir)

        # -- Step 2: Load configs ---------------------------------------------
        polar_config = _load_polar_config(model_dir)

        # Load HF config for base model architecture
        hf_config_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(hf_config_path):
            hf_config = AutoConfig.from_pretrained(
                model_dir, trust_remote_code=trust_remote_code,
            )
        else:
            # Fall back to base_model from polar_config
            base_model = polar_config.get("base_model", "") if polar_config else ""
            if not base_model:
                raise FileNotFoundError(
                    f"No config.json or base_model found in {model_dir}"
                )
            hf_config = AutoConfig.from_pretrained(
                base_model, trust_remote_code=trust_remote_code,
            )

        # -- Step 3: Detect format and dequantize -----------------------------
        all_keys = _collect_all_keys(model_dir)
        fmt = _detect_format(all_keys)
        logger.info("Detected PolarQuant format: %s (%d tensors)", fmt, len(all_keys))

        if fmt == "dot":
            state_dict = _dequant_dot_format(model_dir, polar_config, dtype)
        elif fmt == "dunder":
            state_dict = _dequant_dunder_format(model_dir, polar_config, dtype)
        elif fmt == "packed":
            state_dict = _dequant_packed_format(model_dir, polar_config, dtype)
        elif fmt == "bf16_only":
            # No quantized tensors -- just load everything
            state_dict = {}
            for name, tensor in _iter_safetensors(model_dir):
                state_dict[name] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        else:
            raise ValueError(f"Unknown PolarQuant format: {fmt}")

        logger.info("Dequantized state dict: %d tensors", len(state_dict))

        # -- Step 4: Load into transformers model -----------------------------
        model = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        # Load dequantized weights (strict=False to handle padding mismatches)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])

        model = model.to(dtype)
        if device_map == "auto":
            model = model.cuda() if torch.cuda.is_available() else model
        elif device_map != "cpu":
            model = model.to(device_map)

        model.eval()

        # -- Step 5: Load tokenizer -------------------------------------------
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Tokenizer might be in the base model repo
            base_model = polar_config.get("base_model", "") if polar_config else ""
            if base_model:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model, trust_remote_code=trust_remote_code,
                    )
                except Exception:
                    logger.warning("Could not load tokenizer from %s", base_model)

        # -- Step 6: Optional KV cache ----------------------------------------
        kv_cache = None
        if kv_cache_nbits is not None:
            from polarengine_vllm.kv_cache.config import PolarKVConfig
            from polarengine_vllm.kv_cache.cache import PolarKVCache

            head_dim = getattr(hf_config, "head_dim", 128)
            num_kv_heads = getattr(hf_config, "num_key_value_heads",
                                   getattr(hf_config, "num_attention_heads", 8))
            num_layers = getattr(hf_config, "num_hidden_layers", 32)

            kv_config = PolarKVConfig(
                nbits=kv_cache_nbits,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                num_layers=num_layers,
            )
            kv_cache = PolarKVCache(kv_config)
            logger.info(
                "PolarKVCache: %d-bit, %dx compression, head_dim=%d",
                kv_cache_nbits, int(kv_config.compression_ratio), head_dim,
            )

        return cls(model, tokenizer, kv_cache=kv_cache, polar_config=polar_config)

    # -- convenience methods --------------------------------------------------

    def generate(self, prompt: str, *, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text from a prompt string.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Passed to model.generate().

        Returns:
            Generated text (decoded, prompt excluded).
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Pass tokenizer manually.")

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **kwargs,
            )

        # Strip prompt tokens from output
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        kv_str = f", kv_cache={self.kv_cache.config.nbits}bit" if self.kv_cache else ""
        return f"PolarQuantModel({n_params:.1f}B params{kv_str})"
