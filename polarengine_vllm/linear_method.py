"""
PolarQuant linear method for vLLM quantized inference.

Implements vLLM's LinearMethodBase interface:
  - create_weights()                  -- register quantized buffers on the layer
  - apply()                           -- forward pass (FWHT + Triton GEMV)
  - process_weights_after_loading()   -- post-load validation and optional INT4 packing

Weights stay as int8 codes in VRAM throughout inference -- no dequantization
step is needed.  The FWHT transform is cached across Q/K/V projections that
share the same hidden state, saving ~33% of FWHT overhead per attention block.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from polarengine_vllm.kernels.fwht import FWHTCache, build_hadamard, fwht_matmul
from polarengine_vllm.kernels.polar_gemv import (
    pack_codes_int4,
    polar_gemv,
    polar_gemv_packed,
)
from polarengine_vllm.kernels.polar_gemm import polar_matmul
from polarengine_vllm.utils import get_centroids

# ---------------------------------------------------------------------------
# vLLM utility import -- graceful fallback when vLLM is not installed so
# that the module remains importable for unit testing and offline tooling.
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.layers.quantization.utils import set_weight_attrs
except ImportError:

    def set_weight_attrs(weight: torch.Tensor, attrs: Dict[str, Any]) -> None:
        """Fallback: attach arbitrary attributes directly on a tensor.

        vLLM's ``set_weight_attrs`` stores metadata (input_dim, output_dim,
        etc.) that the weight loader inspects when sharding across tensor-
        parallel ranks.  Without vLLM we just set them as regular attributes.
        """
        for key, value in attrs.items():
            setattr(weight, key, value)


logger = logging.getLogger(__name__)


# ===================================================================
# PolarQuantLinearMethod
# ===================================================================

class PolarQuantLinearMethod:
    """vLLM linear method for PolarQuant quantized layers.

    Implements create_weights(), apply(), and process_weights_after_loading()
    as required by vLLM's QuantizeMethodBase / LinearMethodBase interface.

    Each quantized linear layer stores:
      - ``codes``     (int8)   -- per-weight centroid indices
      - ``norms``     (fp16)   -- per-block L2 norms
      - ``ct_scaled`` (fp32)   -- centroids pre-divided by sqrt(block_size)

    The forward path is:
      1. Pad + FWHT(x)          (via fwht_matmul, cached)
      2. Triton GEMV kernel     (unpacked or packed INT4 nibble)
      3. Cast to fp16 + bias
    """

    def __init__(
        self,
        block_size: int = 128,
        bits: int = 4,
        layer_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.block_size = block_size
        self.bits = bits
        self.layer_meta = layer_meta or {}
        self._fwht_cache = FWHTCache()

    # ----------------------------------------------------------------
    # create_weights
    # ----------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Register quantized weight buffers on *layer*.

        Buffers created:
          - ``codes``     -- int8, shape (out_f, in_f_padded)
          - ``norms``     -- fp16, shape (out_f, n_blocks)
          - ``ct_scaled`` -- fp32, shape (n_levels,)

        Also stores dimension metadata as plain attributes so that
        ``apply()`` and ``process_weights_after_loading()`` can access
        them without re-deriving.
        """
        bs = self.block_size

        in_f = input_size
        out_f = sum(output_partition_sizes)
        in_f_padded = ((in_f + bs - 1) // bs) * bs
        n_blocks = in_f_padded // bs

        # --- quantized weight storage ---
        layer.register_buffer(
            "codes",
            torch.zeros(out_f, in_f_padded, dtype=torch.int8),
        )
        layer.register_buffer(
            "norms",
            torch.zeros(out_f, n_blocks, dtype=torch.float16),
        )

        # Centroids pre-scaled by 1/sqrt(block_size) so the kernel does not
        # need an extra multiply.
        ct = get_centroids(self.bits)
        layer.register_buffer(
            "ct_scaled",
            (ct / math.sqrt(bs)).clone(),
        )

        # --- dimension metadata ---
        layer.in_f = in_f
        layer.out_f = out_f
        layer.in_f_padded = in_f_padded
        layer.n_blocks = n_blocks
        layer.bits = self.bits
        layer.block_size = bs
        layer.packed = False

        # --- vLLM weight-loader attributes ---
        # ``input_dim`` / ``output_dim`` tell the weight loader which tensor
        # axes correspond to the input and output feature dimensions so that
        # tensor-parallel sharding works correctly.
        set_weight_attrs(layer.codes, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(layer.norms, {"input_dim": 1, "output_dim": 0})

    # ----------------------------------------------------------------
    # apply (forward pass)
    # ----------------------------------------------------------------

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: FWHT(x) then Triton GEMV kernel.

        Replaces the standard ``nn.Linear`` forward with quantized
        inference.  Weights stay as int8 codes in VRAM -- no
        dequantization is performed.

        Steps:
          1. Flatten x to (batch, in_f) and cast to float32.
          2. Pad to in_f_padded if necessary.
          3. Apply blockwise FWHT (cached across Q/K/V).
          4. Launch the Triton GEMV kernel (packed or unpacked).
          5. Cast back to fp16, reshape, and add bias.

        Args:
            layer: The nn.Module carrying the quantized buffers.
            x:     Input activation, shape (..., in_f).
            bias:  Optional bias vector, shape (out_f,).

        Returns:
            Output tensor, shape (..., out_f), dtype float16.
        """
        # --- flatten to 2-D ---
        orig_shape = x.shape
        x_flat = x.view(-1, layer.in_f).float()
        batch = x_flat.shape[0]

        # --- FWHT with caching (Q/K/V share the same hidden state) ---
        cached = self._fwht_cache.get(x, layer.in_f)
        if cached is not None:
            x_tf = cached
        else:
            pad = layer.in_f_padded - layer.in_f
            x_p = F.pad(x_flat, (0, pad)) if pad > 0 else x_flat
            x_tf = fwht_matmul(x_p, layer.block_size).view(batch, -1)
            self._fwht_cache.put(x, layer.in_f, x_tf)

        # --- Triton GEMV/GEMM (adaptive: GEMV for batch=1, GEMM for batch>1) ---
        is_packed = getattr(layer, "packed", False)
        output = polar_matmul(
            codes=layer.codes if not is_packed else None,
            x_transformed=x_tf,
            norms=layer.norms,
            ct_scaled=layer.ct_scaled,
            out_f=layer.out_f,
            in_f_padded=layer.in_f_padded,
            n_blocks=layer.n_blocks,
            block_size=layer.block_size,
            packed=is_packed,
            packed_codes=getattr(layer, "codes_packed", None) if is_packed else None,
            in_f_half=getattr(layer, "in_f_half", None) if is_packed else None,
        )
        # Ensure output is (batch, out_f) for reshape below
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # --- reshape and bias ---
        result = output.half().view(*orig_shape[:-1], layer.out_f)
        if bias is not None:
            result = result + bias.half()
        return result

    # ----------------------------------------------------------------
    # process_weights_after_loading
    # ----------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-processing after weights are loaded from safetensors.

        Responsibilities:
          1. Move centroids to the same device as the codes.
          2. For Q3/Q4 layers (bits <= 4): pack int8 codes into INT4
             nibbles (halves memory traffic at the cost of two cheap
             bitwise ops in the kernel).
          3. Free the original unpacked codes buffer after packing.
        """
        device = layer.codes.device

        # Ensure centroids live on the correct device.
        layer.ct_scaled = layer.ct_scaled.to(device)

        # Pack into INT4 nibbles for Q2/Q3/Q4 layers.
        if layer.bits <= 4 and not getattr(layer, "packed", False):
            packed = pack_codes_int4(layer.codes, layer.block_size)
            layer.codes_packed = packed
            layer.in_f_half = layer.n_blocks * (layer.block_size // 2)
            layer.packed = True

            # Free unpacked codes -- they are no longer needed.
            del layer.codes
            logger.debug(
                "Packed layer codes to INT4 nibbles: "
                "out_f=%d, in_f_half=%d, bits=%d",
                layer.out_f,
                layer.in_f_half,
                layer.bits,
            )


# ===================================================================
# FWHT cache clear hook
# ===================================================================

def create_fwht_clear_hook(linear_method: PolarQuantLinearMethod):
    """Create a forward pre-hook that clears the FWHT cache.

    Attach to the top-level model so the cache is invalidated at the
    start of each forward pass::

        hook = create_fwht_clear_hook(my_linear_method)
        model.register_forward_pre_hook(hook)

    This prevents stale cache entries from leaking across successive
    forward calls (different input sequences, different data pointers).

    Args:
        linear_method: The ``PolarQuantLinearMethod`` whose FWHT cache
                       should be cleared.

    Returns:
        A callable suitable for ``register_forward_pre_hook``.
    """

    def hook(module: torch.nn.Module, args):
        linear_method._fwht_cache.clear()

    return hook
