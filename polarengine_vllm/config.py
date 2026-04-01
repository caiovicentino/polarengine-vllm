"""
PolarQuant quantization configuration for vLLM.

Defines how vLLM discovers, parses, and applies PolarEngine quantization
to model layers during inference. The config reads ``polar_config.json``
shipped alongside each quantized model and maps layers to their bit widths
and block parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional vLLM imports -- the package should remain importable (for unit
# tests, config parsing, etc.) even when vLLM is not installed.
# ---------------------------------------------------------------------------

_VLLM_AVAILABLE = True

try:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
except ImportError:
    _VLLM_AVAILABLE = False

    # Provide lightweight stubs so the module can still be imported and the
    # config-parsing logic exercised without a full vLLM installation.
    class QuantizationConfig:  # type: ignore[no-redef]
        """Stub base class used when vLLM is not installed."""
        pass

    class QuantizeMethodBase:  # type: ignore[no-redef]
        """Stub base class used when vLLM is not installed."""
        pass

# ---------------------------------------------------------------------------
# Optional decorator import -- register_quantization_config was added in
# vLLM 0.8.x. If it is not present we skip automatic registration and fall
# back to manual insertion (see bottom of file).
# ---------------------------------------------------------------------------

_register_decorator = None
try:
    from vllm.model_executor.layers.quantization import (
        register_quantization_config,
    )
    _register_decorator = register_quantization_config
except ImportError:
    pass


# ---- helpers for building the class with or without the decorator ---------

def _build_config_class(decorator):
    """Construct PolarQuantConfig, optionally applying the vLLM decorator."""

    # We define the class inside a factory so that the decorator (if present)
    # is applied at class-creation time, matching vLLM's expected pattern.

    @decorator("polarengine") if decorator else (lambda cls: cls)
    class PolarQuantConfig(QuantizationConfig):
        """vLLM quantization config for PolarEngine (PolarQuant) models.

        A PolarQuant model ships with a ``polar_config.json`` that contains:

        * ``block_size`` -- number of weights per quantization block (default 128).
        * ``format`` -- serialisation format tag (default ``"polar_engine_v4"``).
        * ``bit_assignment`` -- dict mapping layer-name patterns to bit widths.
        * ``layers`` -- per-layer metadata (``in_features``, ``out_features``,
          ``bits``, ``block_size``, ``scale_dtype``, etc.).
        """

        # -- construction -----------------------------------------------------

        def __init__(
            self,
            block_size: int = 128,
            bit_assignment: Optional[Dict[str, int]] = None,
            layers_meta: Optional[Dict[str, Dict[str, Any]]] = None,
        ) -> None:
            self.block_size: int = block_size
            self.bit_assignment: Dict[str, int] = bit_assignment or {}
            self.layers_meta: Dict[str, Dict[str, Any]] = layers_meta or {}
            self.packed_modules_mapping: Dict[str, Any] = {}

        def __repr__(self) -> str:
            n_layers = len(self.layers_meta)
            bits = sorted(set(self.bit_assignment.values())) if self.bit_assignment else []
            return (
                f"PolarQuantConfig(block_size={self.block_size}, "
                f"bits={bits}, layers={n_layers})"
            )

        # -- required interface ------------------------------------------------

        @staticmethod
        def get_name() -> str:
            return "polarengine"

        @staticmethod
        def get_supported_act_dtypes() -> List[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @staticmethod
        def get_min_capability() -> int:
            # Volta (SM 7.0) and newer.
            return 70

        @staticmethod
        def get_config_filenames() -> List[str]:
            return ["polar_config.json"]

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "PolarQuantConfig":
            """Parse the contents of ``polar_config.json``.

            Parameters
            ----------
            config:
                The parsed JSON dictionary from the model's config file.

            Returns
            -------
            PolarQuantConfig
                A fully-initialised config instance.
            """
            block_size = config.get("block_size", 128)
            fmt = config.get("format", "polar_engine_v4")

            if fmt not in ("polar_engine_v4", "polar_engine_v5"):
                logger.warning(
                    "Unrecognised PolarEngine format '%s'. "
                    "Proceeding, but weight loading may fail.",
                    fmt,
                )

            bit_assignment: Dict[str, int] = config.get("bit_assignment", {})
            layers_meta: Dict[str, Dict[str, Any]] = config.get("layers", {})

            logger.info(
                "PolarQuantConfig: block_size=%d, format=%s, "
                "%d bit-assignment rules, %d layer entries",
                block_size, fmt, len(bit_assignment), len(layers_meta),
            )

            return cls(
                block_size=block_size,
                bit_assignment=bit_assignment,
                layers_meta=layers_meta,
            )

        def get_quant_method(
            self,
            layer: torch.nn.Module,
            prefix: str,
        ) -> Optional["QuantizeMethodBase"]:
            """Return the quantization method for a given layer.

            Only ``torch.nn.Linear`` layers (or vLLM equivalents) are
            quantized. All other layer types return ``None`` so that vLLM
            falls back to the default (unquantized) path.
            """
            from polarengine_vllm.linear_method import PolarQuantLinearMethod
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
            from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase

            # Check if this is a FusedMoE layer — needs FusedMoEMethodBase
            class_name = type(layer).__name__
            is_moe = "MoE" in class_name or "Moe" in class_name or "moe" in class_name or hasattr(layer, 'num_experts')
            if is_moe:
                # Return a deferred unquantized MoE method that initializes lazily
                # (can't pass moe=layer here because layer.__init__ hasn't finished)
                class _DeferredUnquantMoE(FusedMoEMethodBase):
                    def __init__(self):
                        # Skip FusedMoEMethodBase.__init__ which needs moe
                        pass
                    def create_weights(self, layer, **kwargs):
                        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
                        self._inner = UnquantizedFusedMoEMethod(moe=layer)
                        return self._inner.create_weights(layer, **kwargs)
                    def apply(self, *args, **kwargs):
                        return self._inner.apply(*args, **kwargs)
                    def get_fused_moe_quant_config(self):
                        if hasattr(self, '_inner'):
                            return self._inner.get_fused_moe_quant_config()
                        return None
                return _DeferredUnquantMoE()

            # Guard: only quantize Linear layers.
            is_linear = isinstance(layer, torch.nn.Linear)
            if not is_linear:
                if "Linear" not in class_name:
                    return UnquantizedLinearMethod()

            # Determine the bit width for this specific layer.
            bits = self._resolve_bits(prefix)
            if bits is None:
                return UnquantizedLinearMethod()

            return PolarQuantLinearMethod(
                block_size=self.block_size,
                bits=bits,
                layer_meta=self.layers_meta.get(prefix, {}),
            )

        # -- internal helpers --------------------------------------------------

        def _resolve_bits(self, prefix: str) -> Optional[int]:
            """Look up the bit width for *prefix* using ``bit_assignment``.

            The lookup order is:
            1. Exact match in ``layers_meta`` (per-layer ``bits`` field).
            2. Pattern match against ``bit_assignment`` keys (longest match wins).
            3. ``None`` if no rule matches (layer will not be quantized).
            """
            # 1. Exact per-layer metadata.
            meta = self.layers_meta.get(prefix)
            if meta and "bits" in meta:
                return int(meta["bits"])

            # 2. Pattern matching -- keys in bit_assignment are treated as
            #    simple substring patterns. The longest matching key wins so
            #    that more-specific rules override general ones.
            best_match: Optional[str] = None
            for pattern in self.bit_assignment:
                if pattern in prefix:
                    if best_match is None or len(pattern) > len(best_match):
                        best_match = pattern

            if best_match is not None:
                return int(self.bit_assignment[best_match])

            return None

    return PolarQuantConfig


# Build and export the class.
PolarQuantConfig = _build_config_class(_register_decorator)

# ---------------------------------------------------------------------------
# Fallback manual registration for older vLLM versions that lack the
# ``register_quantization_config`` decorator.
# ---------------------------------------------------------------------------
if _VLLM_AVAILABLE and _register_decorator is None:
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        if "polarengine" not in QUANTIZATION_METHODS:
            QUANTIZATION_METHODS["polarengine"] = PolarQuantConfig
            logger.debug(
                "Registered PolarQuantConfig via QUANTIZATION_METHODS dict "
                "(legacy path)."
            )
    except ImportError:
        logger.warning(
            "Could not register PolarEngine quantization: "
            "vLLM QUANTIZATION_METHODS dict not found."
        )
