"""PolarEngine Triton GEMM kernels for batched quantized inference.

Extends GEMV to GEMM: processes all batch elements in parallel instead of
looping per-element.  Critical for prefill and batch decode where batch > 1.

Two kernels:
  1. polar_gemm_kernel        -- unpacked (int8 codes, 1 byte per code)
  2. polar_gemm_packed_kernel -- packed INT4 nibble (2 codes per byte)

Computes:  Y[batch, out_f] = X_transformed[batch, in_f_padded] @ W^T
where W is reconstructed on-the-fly from PolarQuant codes + centroids + norms.

The FWHT transform of X is applied ONCE before the kernel launch, and the
transformed X is reused across all output tiles -- no redundant transforms.

Tiling strategy:
  - BLOCK_M: batch dimension (rows of X)
  - BLOCK_N: output dimension (rows of W / columns of Y)
  - BLOCK_K: reduction dimension (= block_size = 128, matches head_dim)

For each (M, N) tile, the kernel iterates over K-blocks:
  load codes[N_tile, K_block] -> lookup centroids -> scale by norms ->
  multiply by X[M_tile, K_block] -> accumulate into Y[M_tile, N_tile].

Derived from the proven polar_gemv_kernel (EOQ_POLAR_ENGINE_V4, PPL 6.43).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ===================================================================
# Kernel 1: Unpacked GEMM (int8 codes)
# ===================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            # --- Small batch (decode-like, 1-8 tokens) ---
            triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            # --- Medium batch (chunked prefill, 16-64 tokens) ---
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            # --- Large batch (full prefill, 64-512 tokens) ---
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
            # --- Pipeline depth variants (helps hide memory latency) ---
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        ],
        key=['batch_size', 'out_f', 'in_f_padded'],
    )
    @triton.jit
    def polar_gemm_kernel(
        # --- Data pointers ---
        codes_ptr,      # int8 codes, shape (out_f, in_f_padded), row-major
        x_ptr,          # FWHT-transformed input, shape (batch_size, in_f_padded), float32
        norms_ptr,      # per-block norms, shape (out_f, n_blocks), float16
        ct_ptr,         # pre-scaled centroids (incl. 1/sqrt(bs)), float32
        out_ptr,        # output, shape (batch_size, out_f), float32
        # --- Dimensions ---
        batch_size,     # M dimension (number of input vectors)
        out_f,          # N dimension (number of output features / weight rows)
        in_f_padded,    # K dimension (padded input features, multiple of BLOCK_K)
        n_blocks,       # number of K-blocks per weight row = in_f_padded / BLOCK_K
        # --- Strides ---
        stride_xb,      # x_ptr stride for batch dim (= in_f_padded)
        stride_ob,      # out_ptr stride for batch dim (= out_f)
        # --- Tile sizes ---
        BLOCK_K: tl.constexpr,    # must equal block_size (128)
        BLOCK_M: tl.constexpr,    # batch tile (autotuned)
        BLOCK_N: tl.constexpr,    # output tile (autotuned)
    ):
        """Tiled GEMM: Y = X_transformed @ W^T  (W reconstructed on-the-fly).

        2D grid: (ceil(batch/BLOCK_M), ceil(out_f/BLOCK_N)).
        Each program computes a (BLOCK_M, BLOCK_N) output tile by iterating
        over n_blocks in the K dimension.

        For each K-block:
          1. Load X tile:      (BLOCK_M, BLOCK_K) from x_ptr
          2. Load codes tile:  (BLOCK_N, BLOCK_K) from codes_ptr
          3. Centroid lookup:  codes -> float32 values via ct_ptr gather
          4. Norm scaling:     values *= norms[n, block_idx]
          5. Accumulate:       acc += X_tile @ values^T   (BLOCK_M, BLOCK_N)

        Centroids are pre-scaled by 1/sqrt(block_size) so no extra factor
        is needed in the kernel.
        """
        # --- Program IDs ---
        pid_m = tl.program_id(0)  # batch tile index
        pid_n = tl.program_id(1)  # output tile index

        # --- Offset ranges ---
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # batch indices
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output indices
        m_mask = m_offs < batch_size
        n_mask = n_offs < out_f

        # --- Accumulator ---
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- Main K-block loop ---
        for block_idx in range(n_blocks):
            k_offs = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)

            # 1. Load X tile: (BLOCK_M, BLOCK_K)
            #    x_ptr[m, k] = x_ptr + m * stride_xb + k
            x_ptrs = m_offs[:, None] * stride_xb + k_offs[None, :]
            x_tile = tl.load(x_ptr + x_ptrs, mask=m_mask[:, None], other=0.0)

            # 2. Load codes tile: (BLOCK_N, BLOCK_K)
            #    codes_ptr[n, k] = codes_ptr + n * in_f_padded + k
            code_ptrs = n_offs[:, None] * in_f_padded + k_offs[None, :]
            codes_tile = tl.load(
                codes_ptr + code_ptrs,
                mask=n_mask[:, None],
                other=0,
            )

            # 3. Centroid lookup: int8 code -> float32 value
            #    Pre-scaled centroids (already divided by sqrt(block_size))
            w_tile = tl.load(ct_ptr + codes_tile.to(tl.int32))  # (BLOCK_N, BLOCK_K)

            # 4. Per-block norm scaling: norms[n, block_idx]
            #    norms layout: (out_f, n_blocks), row-major
            norm_ptrs = n_offs * n_blocks + block_idx
            norms_val = tl.load(
                norms_ptr + norm_ptrs,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            # Broadcast norm across K: (BLOCK_N,) -> (BLOCK_N, BLOCK_K)
            w_tile = w_tile * norms_val[:, None]

            # 5. Matrix multiply: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            #    x_tile is (BLOCK_M, BLOCK_K), w_tile is (BLOCK_N, BLOCK_K)
            #    We want (BLOCK_M, BLOCK_N) = x_tile @ w_tile^T
            acc += tl.dot(x_tile, tl.trans(w_tile))

        # --- Store output tile ---
        out_ptrs = m_offs[:, None] * stride_ob + n_offs[None, :]
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptr + out_ptrs, acc, mask=out_mask)


# ===================================================================
# Kernel 2: Packed INT4 nibble GEMM (half-block packing)
# ===================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            # --- Small batch ---
            triton.Config({'BLOCK_M': 1,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 1,  'BLOCK_N': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 2,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 4,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 4,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
            # --- Medium batch ---
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            # --- Large batch ---
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
            # --- Pipeline depth variants ---
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        ],
        key=['batch_size', 'out_f', 'in_f_half'],
    )
    @triton.jit
    def polar_gemm_packed_kernel(
        # --- Data pointers ---
        packed_ptr,     # packed uint8 codes, shape (out_f, in_f_half)
        x_ptr,          # FWHT-transformed input, shape (batch_size, in_f_padded), float32
        norms_ptr,      # per-block norms, shape (out_f, n_blocks), float16
        ct_ptr,         # pre-scaled centroids (incl. 1/sqrt(bs)), float32
        out_ptr,        # output, shape (batch_size, out_f), float32
        # --- Dimensions ---
        batch_size,     # M dimension
        out_f,          # N dimension
        in_f_half,      # packed K dimension = n_blocks * HALF_BK
        n_blocks,       # number of K-blocks per weight row
        # --- Strides ---
        stride_xb,      # x_ptr stride for batch dim (= in_f_padded)
        stride_ob,      # out_ptr stride for batch dim (= out_f)
        # --- Tile sizes ---
        HALF_BK: tl.constexpr,    # half block size (64 for block_size=128)
        BLOCK_M: tl.constexpr,    # batch tile (autotuned)
        BLOCK_N: tl.constexpr,    # output tile (autotuned)
    ):
        """Packed INT4 GEMM: loads half the bytes of unpacked kernel.

        Half-block packing layout:
            byte[i] = (code[HALF_BK + i] << 4) | code[i]   for i in [0, HALF_BK)
            Low nibble  = first half of block  (codes 0..63)
            High nibble = second half of block (codes 64..127)

        For each K-block, this kernel:
          1. Loads HALF_BK packed bytes per weight row (vs BLOCK_K=128 for unpacked)
          2. Unpacks two nibbles with cheap bitwise ops
          3. Reconstructs two half-block weight tiles
          4. Loads two contiguous X half-tiles (no strided access)
          5. Accumulates two half-block matmuls into the output tile
        """
        # --- Program IDs ---
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # --- Offset ranges ---
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < batch_size
        n_mask = n_offs < out_f

        # --- Accumulator ---
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # --- Main K-block loop ---
        for block_idx in range(n_blocks):
            # X indices for this block: two contiguous halves
            base_k = block_idx * HALF_BK * 2
            h_offs = tl.arange(0, HALF_BK)

            # 1. Load X halves: (BLOCK_M, HALF_BK) each
            x_lo_ptrs = m_offs[:, None] * stride_xb + (base_k + h_offs[None, :])
            x_hi_ptrs = m_offs[:, None] * stride_xb + (base_k + HALF_BK + h_offs[None, :])
            x_lo = tl.load(x_ptr + x_lo_ptrs, mask=m_mask[:, None], other=0.0)
            x_hi = tl.load(x_ptr + x_hi_ptrs, mask=m_mask[:, None], other=0.0)

            # 2. Load packed codes: (BLOCK_N, HALF_BK)
            pack_offs = block_idx * HALF_BK + h_offs
            pack_ptrs = n_offs[:, None] * in_f_half + pack_offs[None, :]
            packed = tl.load(
                packed_ptr + pack_ptrs,
                mask=n_mask[:, None],
                other=0,
            )

            # 3. Unpack nibbles (2 cheap bitwise ops)
            lo_codes = packed & 0xF              # first half of block
            hi_codes = (packed >> 4) & 0xF       # second half of block

            # 4. Centroid lookup (pre-scaled)
            w_lo = tl.load(ct_ptr + lo_codes.to(tl.int32))   # (BLOCK_N, HALF_BK)
            w_hi = tl.load(ct_ptr + hi_codes.to(tl.int32))   # (BLOCK_N, HALF_BK)

            # 5. Per-block norm scaling
            norm_ptrs = n_offs * n_blocks + block_idx
            norms_val = tl.load(
                norms_ptr + norm_ptrs,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            w_lo = w_lo * norms_val[:, None]
            w_hi = w_hi * norms_val[:, None]

            # 6. Two half-block matmuls:
            #    (BLOCK_M, HALF_BK) @ (HALF_BK, BLOCK_N) for each half
            acc += tl.dot(x_lo, tl.trans(w_lo))
            acc += tl.dot(x_hi, tl.trans(w_hi))

        # --- Store output tile ---
        out_ptrs = m_offs[:, None] * stride_ob + n_offs[None, :]
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptr + out_ptrs, acc, mask=out_mask)


# ===================================================================
# Wrapper functions
# ===================================================================

def polar_gemm(
    codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_padded: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Launch unpacked GEMM kernel for batched inference.

    Args:
        codes: int8 codes, shape (out_f, in_f_padded)
        x_transformed: FWHT-transformed input, shape (batch_size, in_f_padded), float32
        norms: per-block norms, shape (out_f, n_blocks), float16 or float32
        ct_scaled: pre-scaled centroids (centroids / sqrt(block_size)), float32
        out_f: number of output features
        in_f_padded: padded input features (multiple of block_size)
        n_blocks: blocks per row = in_f_padded / block_size
        block_size: quantization block size (default 128, must match BLOCK_K)

    Returns:
        output tensor, shape (batch_size, out_f), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemm kernel")

    batch_size = x_transformed.shape[0]

    # Ensure norms is 2D (out_f, n_blocks) -- the GEMV kernel uses flat
    # (out_f * n_blocks,) layout, but GEMM needs row-major 2D for efficient
    # per-row-per-block indexing.
    if norms.dim() == 1:
        norms = norms.view(out_f, n_blocks)

    output = torch.zeros(
        batch_size, out_f, device=codes.device, dtype=torch.float32,
    )

    # 2D grid: (batch tiles, output tiles)
    grid = lambda meta: (
        (batch_size + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (out_f + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    polar_gemm_kernel[grid](
        codes, x_transformed, norms, ct_scaled, output,
        batch_size, out_f, in_f_padded, n_blocks,
        x_transformed.stride(0),  # stride_xb
        output.stride(0),         # stride_ob
        BLOCK_K=block_size,
    )
    return output


def polar_gemm_packed(
    packed_codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_half: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Launch packed (INT4 nibble) GEMM kernel for batched inference.

    Args:
        packed_codes: uint8 packed codes, shape (out_f, in_f_half)
        x_transformed: FWHT-transformed input, shape (batch_size, in_f_padded), float32
        norms: per-block norms, shape (out_f, n_blocks), float16 or float32
        ct_scaled: pre-scaled centroids (centroids / sqrt(block_size)), float32
        out_f: number of output features
        in_f_half: packed dimension = n_blocks * (block_size // 2)
        n_blocks: blocks per row
        block_size: quantization block size (default 128)

    Returns:
        output tensor, shape (batch_size, out_f), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemm_packed kernel")

    batch_size = x_transformed.shape[0]
    half_bk = block_size // 2

    if norms.dim() == 1:
        norms = norms.view(out_f, n_blocks)

    output = torch.zeros(
        batch_size, out_f, device=packed_codes.device, dtype=torch.float32,
    )

    grid = lambda meta: (
        (batch_size + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (out_f + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    polar_gemm_packed_kernel[grid](
        packed_codes, x_transformed, norms, ct_scaled, output,
        batch_size, out_f, in_f_half, n_blocks,
        x_transformed.stride(0),  # stride_xb
        output.stride(0),         # stride_ob
        HALF_BK=half_bk,
    )
    return output


# ===================================================================
# Adaptive dispatch: GEMV for batch=1, GEMM for batch>1
# ===================================================================

def polar_matmul(
    codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_padded: int,
    n_blocks: int,
    block_size: int = 128,
    packed: bool = False,
    packed_codes: Optional[torch.Tensor] = None,
    in_f_half: Optional[int] = None,
) -> torch.Tensor:
    """Adaptive dispatch: GEMV/SplitK for single vectors, GEMM for batches.

    Dispatch logic:
      - batch > 1  --> GEMM kernel (batched matmul)
      - batch = 1 AND n_blocks >= 15  --> SplitK GEMV (better SM saturation)
      - batch = 1 AND n_blocks < 15   --> regular GEMV (proven, low overhead)

    The SplitK threshold of 15 blocks targets models with hidden_size >= 1920
    (e.g., Nemotron with hidden_size=2688 has 21 blocks of 128).  For these
    shapes, the regular GEMV only launches ceil(out_f/128) threadblocks,
    which is too few to saturate modern GPUs (96+ SMs).  SplitK adds a
    second grid dimension to split the K-reduction across multiple
    threadblocks, improving GPU utilization.

    This is the recommended entry point for the linear_method.py apply()
    function -- it handles decode (batch=1) and prefill (batch>1)
    transparently.

    Args:
        codes: int8 codes, shape (out_f, in_f_padded)
        x_transformed: FWHT-transformed input, shape (in_f_padded,) or
                       (batch_size, in_f_padded), float32
        norms: per-block norms, shape (out_f * n_blocks,) or (out_f, n_blocks)
        ct_scaled: pre-scaled centroids, float32
        out_f: number of output features
        in_f_padded: padded input features
        n_blocks: blocks per row
        block_size: quantization block size
        packed: whether to use INT4 packed variant
        packed_codes: uint8 packed codes (required if packed=True)
        in_f_half: packed dimension (required if packed=True)

    Returns:
        output tensor, float32. Shape matches input batch dimension.
    """
    # Import GEMV and SplitK here to avoid circular imports
    from polarengine_vllm.kernels.polar_gemv import polar_gemv, polar_gemv_packed
    from polarengine_vllm.kernels.polar_gemv_splitk import (
        polar_gemv_splitk,
        polar_gemv_packed_splitk,
    )

    # Threshold: use SplitK when n_blocks >= 15 (hidden_size >= 1920)
    # Nemotron hidden_size=2688 -> 21 blocks, well above threshold.
    _SPLITK_THRESHOLD = 15

    # Determine if this is a single vector or a batch
    is_single = x_transformed.dim() == 1 or (
        x_transformed.dim() == 2 and x_transformed.shape[0] == 1
    )

    if is_single:
        x_vec = x_transformed.view(-1) if x_transformed.dim() == 2 else x_transformed
        norms_flat = norms.view(-1) if norms.dim() == 2 else norms

        if n_blocks >= _SPLITK_THRESHOLD:
            # --- SplitK GEMV path (large K, better GPU saturation) ---
            if packed and packed_codes is not None and in_f_half is not None:
                result = polar_gemv_packed_splitk(
                    packed_codes, x_vec, norms_flat, ct_scaled,
                    out_f, in_f_half, n_blocks, block_size,
                )
            else:
                result = polar_gemv_splitk(
                    codes, x_vec, norms_flat, ct_scaled,
                    out_f, in_f_padded, n_blocks, block_size,
                )
        else:
            # --- Regular GEMV path (small K, proven, low overhead) ---
            if packed and packed_codes is not None and in_f_half is not None:
                result = polar_gemv_packed(
                    packed_codes, x_vec, norms_flat, ct_scaled,
                    out_f, in_f_half, n_blocks, block_size,
                )
            else:
                result = polar_gemv(
                    codes, x_vec, norms_flat, ct_scaled,
                    out_f, in_f_padded, n_blocks, block_size,
                )

        # Return with batch dimension if input had one
        if x_transformed.dim() == 2:
            result = result.unsqueeze(0)
        return result

    else:
        # --- GEMM path (batched) ---
        if packed and packed_codes is not None and in_f_half is not None:
            return polar_gemm_packed(
                packed_codes, x_transformed, norms, ct_scaled,
                out_f, in_f_half, n_blocks, block_size,
            )
        else:
            return polar_gemm(
                codes, x_transformed, norms, ct_scaled,
                out_f, in_f_padded, n_blocks, block_size,
            )


# ===================================================================
# Standalone test
# ===================================================================

if __name__ == '__main__':
    import sys
    import time

    if not HAS_TRITON:
        print("ERROR: Triton is required to run tests.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required to run tests.")
        sys.exit(1)

    from polarengine_vllm.kernels.polar_gemv import polar_gemv, polar_gemv_packed, pack_codes_int4

    device = 'cuda'
    block_size = 128
    bits = 4
    n_levels = 1 << bits

    torch.manual_seed(42)
    centroids = torch.linspace(-1.5, 1.5, n_levels, dtype=torch.float32, device=device)
    ct_scaled = centroids / math.sqrt(block_size)

    # Build Hadamard matrix for reference path
    def _hadamard(n: int) -> torch.Tensor:
        if n == 1:
            return torch.tensor([[1.0]])
        h = _hadamard(n // 2)
        return torch.cat([
            torch.cat([h, h], 1),
            torch.cat([h, -h], 1),
        ], 0) / math.sqrt(2)

    H = _hadamard(block_size).to(device)

    test_shapes = [
        (4096, 4096),
        (2048, 8192),
        (512, 4096),
        (12288, 4096),
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 70)
    print("  PolarEngine GEMM Kernel Tests")
    print("=" * 70)

    all_pass = True

    # ---- Test 1: GEMM correctness vs reference dequantized matmul ----
    print("\n--- Test 1: Unpacked GEMM correctness ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size

        for batch in batch_sizes:
            codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
            norms = (torch.randn(out_f, n_blocks, dtype=torch.float32, device=device).abs() + 0.1)
            x = torch.randn(batch, in_f, dtype=torch.float32, device=device)

            # Pad and transform input
            x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
            x_transformed = torch.matmul(
                x_padded.view(-1, block_size), H
            ).reshape(batch, -1)

            # Reference: dequantize weights and do full matmul
            codes_blocked = codes.view(out_f, n_blocks, block_size)
            norms_blocked = norms.view(out_f, n_blocks, 1)
            values = ct_scaled[codes_blocked.long()]
            values = values * norms_blocked
            values_flat = values.view(out_f * n_blocks, block_size)
            values_iht = torch.matmul(values_flat, H)
            w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
            ref_out = torch.matmul(x[:, :in_f], w_deq.T)  # (batch, out_f)

            # GEMM kernel output
            kernel_out = polar_gemm(
                codes, x_transformed, norms, ct_scaled,
                out_f, in_f_padded, n_blocks, block_size,
            )

            # Check
            cos = F.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                      kernel_out.flatten().unsqueeze(0)).item()
            rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()

            status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] Unpacked ({out_f:>5}x{in_f:>5}, batch={batch:>3}): "
                  f"cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 2: Packed GEMM correctness ----
    print("\n--- Test 2: Packed INT4 GEMM correctness ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size
        half_bk = block_size // 2
        in_f_half = n_blocks * half_bk

        for batch in batch_sizes:
            codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
            norms = (torch.randn(out_f, n_blocks, dtype=torch.float32, device=device).abs() + 0.1)
            x = torch.randn(batch, in_f, dtype=torch.float32, device=device)

            x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
            x_transformed = torch.matmul(
                x_padded.view(-1, block_size), H
            ).reshape(batch, -1)

            packed_codes = pack_codes_int4(codes, block_size)

            # Reference
            codes_blocked = codes.view(out_f, n_blocks, block_size)
            norms_blocked = norms.view(out_f, n_blocks, 1)
            values = ct_scaled[codes_blocked.long()]
            values = values * norms_blocked
            values_flat = values.view(out_f * n_blocks, block_size)
            values_iht = torch.matmul(values_flat, H)
            w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
            ref_out = torch.matmul(x[:, :in_f], w_deq.T)

            # Packed GEMM kernel output
            kernel_out = polar_gemm_packed(
                packed_codes, x_transformed, norms, ct_scaled,
                out_f, in_f_half, n_blocks, block_size,
            )

            cos = F.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                      kernel_out.flatten().unsqueeze(0)).item()
            rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()

            status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] Packed   ({out_f:>5}x{in_f:>5}, batch={batch:>3}): "
                  f"cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 3: GEMM matches GEMV element-by-element ----
    print("\n--- Test 3: GEMM matches GEMV for each batch element ---")
    for out_f, in_f in [(4096, 4096), (2048, 8192)]:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size
        batch = 8

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms_flat = (torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1)
        norms_2d = norms_flat.view(out_f, n_blocks)
        x = torch.randn(batch, in_f, dtype=torch.float32, device=device)

        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(
            x_padded.view(-1, block_size), H
        ).reshape(batch, -1)

        # GEMM
        gemm_out = polar_gemm(
            codes, x_transformed, norms_2d, ct_scaled,
            out_f, in_f_padded, n_blocks, block_size,
        )

        # GEMV per element
        gemv_out = torch.zeros_like(gemm_out)
        for b in range(batch):
            gemv_out[b] = polar_gemv(
                codes, x_transformed[b], norms_flat, ct_scaled,
                out_f, in_f_padded, n_blocks, block_size,
            )

        cos = F.cosine_similarity(gemm_out.flatten().unsqueeze(0),
                                  gemv_out.flatten().unsqueeze(0)).item()
        max_diff = (gemm_out - gemv_out).abs().max().item()
        status = "PASS" if cos > 0.99999 and max_diff < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] ({out_f:>5}x{in_f:>5}, batch={batch}): "
              f"cos={cos:.7f}, max_diff={max_diff:.2e}")

    # ---- Test 4: polar_matmul adaptive dispatch ----
    print("\n--- Test 4: polar_matmul adaptive dispatch ---")
    out_f, in_f = 4096, 4096
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size

    codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
    norms = (torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1)

    # Single vector (1D) -> should use GEMV
    x1 = torch.randn(in_f, dtype=torch.float32, device=device)
    x1_padded = F.pad(x1, (0, in_f_padded - in_f))
    x1_tf = torch.matmul(x1_padded.view(-1, block_size), H).reshape(-1)
    out_1d = polar_matmul(codes, x1_tf, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    assert out_1d.dim() == 1 and out_1d.shape[0] == out_f, f"1D dispatch shape wrong: {out_1d.shape}"

    # Single vector (2D, batch=1) -> should use GEMV, return (1, out_f)
    x1_tf_2d = x1_tf.unsqueeze(0)
    out_1b = polar_matmul(codes, x1_tf_2d, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    assert out_1b.dim() == 2 and out_1b.shape == (1, out_f), f"batch=1 dispatch shape wrong: {out_1b.shape}"

    # Batch > 1 -> should use GEMM
    x4 = torch.randn(4, in_f, dtype=torch.float32, device=device)
    x4_padded = F.pad(x4, (0, in_f_padded - in_f))
    x4_tf = torch.matmul(x4_padded.view(-1, block_size), H).reshape(4, -1)
    out_4b = polar_matmul(codes, x4_tf, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    assert out_4b.dim() == 2 and out_4b.shape == (4, out_f), f"batch=4 dispatch shape wrong: {out_4b.shape}"

    # Verify 1D and batch=1 produce the same result
    cos_1d_1b = F.cosine_similarity(out_1d.unsqueeze(0), out_1b).item()
    status = "PASS" if cos_1d_1b > 0.99999 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{status}] 1D vs batch=1 dispatch: cos={cos_1d_1b:.7f}")
    print(f"  [PASS] Shape checks: 1D={out_1d.shape}, batch=1={out_1b.shape}, batch=4={out_4b.shape}")

    # ---- Benchmark: GEMM vs serial GEMV ----
    print("\n--- Benchmark: GEMM vs serial GEMV ---")
    out_f, in_f = 4096, 4096
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size

    codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
    norms_flat = (torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1)
    norms_2d = norms_flat.view(out_f, n_blocks)

    for batch in [1, 4, 8, 16, 32, 64]:
        x = torch.randn(batch, in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f))
        x_transformed = torch.matmul(
            x_padded.view(-1, block_size), H
        ).reshape(batch, -1)

        n_warmup = 10
        n_iters = 50

        # Warmup GEMM
        for _ in range(n_warmup):
            _ = polar_gemm(codes, x_transformed, norms_2d, ct_scaled,
                           out_f, in_f_padded, n_blocks, block_size)
        torch.cuda.synchronize()

        # Time GEMM
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = polar_gemm(codes, x_transformed, norms_2d, ct_scaled,
                           out_f, in_f_padded, n_blocks, block_size)
        torch.cuda.synchronize()
        t_gemm = (time.perf_counter() - t0) / n_iters * 1000

        # Warmup serial GEMV
        for _ in range(n_warmup):
            for b in range(batch):
                _ = polar_gemv(codes, x_transformed[b], norms_flat, ct_scaled,
                               out_f, in_f_padded, n_blocks, block_size)
        torch.cuda.synchronize()

        # Time serial GEMV
        t0 = time.perf_counter()
        for _ in range(n_iters):
            for b in range(batch):
                _ = polar_gemv(codes, x_transformed[b], norms_flat, ct_scaled,
                               out_f, in_f_padded, n_blocks, block_size)
        torch.cuda.synchronize()
        t_gemv = (time.perf_counter() - t0) / n_iters * 1000

        speedup = t_gemv / t_gemm if t_gemm > 0 else float("inf")
        print(f"  batch={batch:>3}: GEMM={t_gemm:.3f}ms, serial_GEMV={t_gemv:.3f}ms, "
              f"speedup={speedup:.2f}x")

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  SOME TESTS FAILED.")
        sys.exit(1)
