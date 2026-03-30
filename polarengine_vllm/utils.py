"""
PolarEngine utilities: Lloyd-Max centroids, bit assignment, and packing helpers.

These are shared across the quantizer (offline) and the vLLM linear method
(online inference).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch


# ===================================================================
# LLOYD-MAX OPTIMAL CENTROIDS FOR N(0,1)
# ===================================================================

def compute_lloyd_max_centroids(n_levels: int, n_iter: int = 100) -> torch.Tensor:
    """Compute MSE-optimal quantization levels for N(0,1) distribution.

    Uses the Lloyd-Max iterative algorithm:
      1. Start with uniform quantizer boundaries over [-4, 4].
      2. Compute optimal centroids given boundaries (conditional expectation):
         E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
      3. Compute optimal boundaries given centroids (midpoints).
      4. Repeat until convergence.

    where phi = N(0,1) PDF, Phi = N(0,1) CDF.

    Args:
        n_levels: Number of quantization levels (2^bits).
        n_iter:   Number of Lloyd-Max iterations.

    Returns:
        Tensor of shape (n_levels,) with optimal centroid values.
    """
    from scipy.stats import norm

    # Initialize with uniform boundaries
    boundaries = np.linspace(-4, 4, n_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    for _ in range(n_iter):
        # Step 1: Optimal centroids given boundaries
        centroids = np.zeros(n_levels)
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            # E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
            prob = norm.cdf(b) - norm.cdf(a)
            if prob > 1e-10:
                centroids[i] = (norm.pdf(a) - norm.pdf(b)) / prob
            else:
                centroids[i] = (a + b) / 2

        # Step 2: Optimal boundaries given centroids (midpoints)
        new_boundaries = np.zeros(n_levels + 1)
        new_boundaries[0] = -np.inf
        new_boundaries[-1] = np.inf
        for i in range(1, n_levels):
            new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        boundaries = new_boundaries

    return torch.tensor(centroids, dtype=torch.float32)


# ===================================================================
# CENTROID CACHE
# ===================================================================

_centroids_cache: dict[int, torch.Tensor] = {}


def get_centroids(bits: int) -> torch.Tensor:
    """Get cached Lloyd-Max centroids for given bit width (2-8).

    First call for a given bit width computes centroids via the Lloyd-Max
    algorithm and caches the result. Subsequent calls return the cached tensor.

    Args:
        bits: Quantization bit width (2 through 8).

    Returns:
        Tensor of shape (2^bits,) with optimal centroid values for N(0,1).

    Raises:
        ValueError: If bits is outside the supported range [2, 8].
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")

    if bits not in _centroids_cache:
        n_levels = 1 << bits
        _centroids_cache[bits] = compute_lloyd_max_centroids(n_levels)

    return _centroids_cache[bits]


# Pre-warm the cache for the most common bit widths.
for _b in (2, 3, 4, 5, 6):
    get_centroids(_b)


# ===================================================================
# MIXED-BIT ASSIGNMENT
# ===================================================================

DEFAULT_BIT_ASSIGNMENT: Dict[str, int] = {
    "embed": 5,
    "lm_head": 6,
    "q_proj": 5,
    "k_proj": 5,
    "v_proj": 5,
    "o_proj": 6,
    "out_proj": 6,
    "gate_up_proj": 3,
    "gate_proj": 3,
    "up_proj": 3,
    "down_proj": 4,
}

# Layer-name fragments that should never be quantized (stay FP16).
_SKIP_PATTERNS = (
    "norm",
    "layernorm",
    "rmsnorm",
    "a_log",
    "dt_bias",
    "conv1d",
)

# Checked CASE-SENSITIVE (not lowered) because ".D" lowered to ".d"
# would falsely match "down_proj", "decoder", etc.
_SKIP_PATTERNS_CASE_SENSITIVE = (
    ".D",       # Mamba D buffer (model.layers.X.mamba.D)
    "A_log",    # Mamba A_log (already in lowercase patterns too)
)


def get_bits_for_layer(
    name: str,
    param: torch.Tensor,
    assignment: Optional[Dict[str, int]] = None,
) -> int:
    """Determine quantization bits for a named parameter.

    Returns 16 for layers that should stay in FP16:
      - Scalars, 1-D tensors, or very small tensors (< 256 elements)
      - Norms (layernorm, rmsnorm, ...)
      - Mamba-specific buffers (A_log, .D, dt_bias, conv1d)
      - Bias vectors
      - MoE router / gating weights

    For all other layers the bit width is resolved by matching the layer
    name against ``assignment`` (or ``DEFAULT_BIT_ASSIGNMENT``).  If no
    rule matches, falls back to 5 bits.

    Args:
        name:       Fully-qualified parameter name (e.g. "model.layers.0.self_attn.q_proj.weight").
        param:      The parameter tensor (used for shape / numel checks).
        assignment: Optional custom bit-assignment dict. Defaults to ``DEFAULT_BIT_ASSIGNMENT``.

    Returns:
        Quantization bit width (2-8) or 16 if the layer should remain FP16.
    """
    if assignment is None:
        assignment = DEFAULT_BIT_ASSIGNMENT

    # --- layers that must stay FP16 ---
    if param.ndim < 2 or param.numel() < 256:
        return 16

    name_lower = name.lower()
    if any(k in name_lower for k in _SKIP_PATTERNS):
        return 16

    # Case-sensitive checks (e.g. ".D" must not match "down_proj")
    if any(k in name for k in _SKIP_PATTERNS_CASE_SENSITIVE):
        return 16

    if "bias" in name and param.ndim == 1:
        return 16

    # MoE router / gating -- typically a small weight that must be precise
    if name.endswith(".gate.weight") or "router" in name:
        return 16

    # --- resolve bit width from assignment dict ---
    # Check patterns in a deterministic order; first match wins.
    # We check longer (more specific) keys first so that "gate_up_proj"
    # matches before "gate_proj".
    sorted_keys = sorted(assignment.keys(), key=len, reverse=True)
    for pattern in sorted_keys:
        if pattern in name:
            return assignment[pattern]

    # Default fallback
    return 5


# ===================================================================
# NIBBLE PACKING (HALF-BLOCK ORDER)
# ===================================================================

def pack_codes_half_block(
    codes: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Pack int8 codes into nibbles using half-block order.

    For each block of ``block_size`` codes, the first half goes into the low
    nibble and the second half into the high nibble:

        byte[i] = (code[block_size//2 + i] << 4) | code[i]

    This layout lets the inference kernel load the activation vector ``x``
    in two contiguous half-block chunks, avoiding strided memory accesses.

    The input ``codes`` tensor must have its last dimension equal to a
    multiple of ``block_size``.  The output has half the elements along the
    last dimension (one byte stores two codes).

    Only safe for layers with bits <= 4 (code values fit in 4 bits).

    Args:
        codes:      Int8 tensor of quantization codes.  Shape ``(..., K)``
                    where ``K`` is a multiple of ``block_size``.
        block_size: Block size used during quantization (default 128).

    Returns:
        Uint8 tensor of shape ``(..., K // 2)`` with nibble-packed codes.

    Raises:
        ValueError: If the last dimension is not a multiple of block_size.
    """
    *leading, K = codes.shape
    if K % block_size != 0:
        raise ValueError(
            f"Last dimension ({K}) must be a multiple of block_size ({block_size})"
        )

    half = block_size // 2
    n_blocks = K // block_size

    # Reshape to expose blocks: (..., n_blocks, block_size)
    blocked = codes.view(*leading, n_blocks, block_size).to(torch.uint8)
    first_half = blocked[..., :half]       # low nibble source
    second_half = blocked[..., half:]      # high nibble source

    # Pack: high nibble << 4 | low nibble
    packed = (second_half << 4) | first_half  # (..., n_blocks, half)

    # Flatten the block dimension back: (..., n_blocks * half)
    return packed.reshape(*leading, n_blocks * half)


def unpack_codes_half_block(
    packed: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Unpack nibble-packed codes back to int8 (inverse of pack_codes_half_block).

    Args:
        packed:     Uint8 tensor of shape ``(..., K // 2)``.
        block_size: Block size used during quantization (default 128).

    Returns:
        Int8 tensor of shape ``(..., K)`` with unpacked codes.
    """
    half = block_size // 2
    *leading, K_half = packed.shape
    n_blocks = K_half // half

    blocked = packed.view(*leading, n_blocks, half)
    low = (blocked & 0x0F).to(torch.int8)
    high = ((blocked >> 4) & 0x0F).to(torch.int8)

    # Reassemble: first half from low nibble, second half from high nibble
    unpacked = torch.cat([low, high], dim=-1)  # (..., n_blocks, block_size)
    return unpacked.reshape(*leading, n_blocks * block_size)
