"""PolarEngine SplitK Triton GEMV kernels for quantized inference.

SplitK variant of polar_gemv_kernel and polar_gemv_packed_kernel.

Problem with the original kernel:
  - A single threadblock accumulates over the entire K dimension (all n_blocks).
  - For a 2048x8192 layer with block_size=128, that's 64 blocks iterated
    sequentially by one threadblock per output row tile.
  - The GPU has thousands of SMs but only ceil(out_f / BLOCK_M) threadblocks
    in the grid. For out_f=2048 and BLOCK_M=128, that's only 16 threadblocks --
    far too few to saturate a modern GPU (RTX PRO 6000 has 96 SMs).
  - cuBLAS avoids this by splitting the K accumulation across multiple
    threadblocks (SplitK), achieving 4.5x better throughput.

Solution: SplitK
  - Grid becomes 2D: (ceil(out_f / BLOCK_M), SPLIT_K)
  - Each threadblock in the K dimension handles n_blocks / SPLIT_K blocks
  - Partial sums written to a workspace buffer: (SPLIT_K, out_f)
  - A lightweight reduction kernel sums the SPLIT_K partial results

Why the previous SplitK attempt caused "CUDA device-side assert":
  The v4 notebook used `range(tl.program_id(1), n_blocks, SPLIT_K)` which is
  a strided range starting from the program_id. In Triton, `tl.program_id()`
  returns a scalar that works fine in most expressions, but using it as the
  *start* argument of `range()` in a `for` loop is unreliable -- Triton's
  compiler sometimes fails to properly handle non-zero, non-constant start
  values in range(), leading to invalid memory accesses when the loop bounds
  are computed incorrectly. The device-side assert fires because codes/norms
  are accessed at out-of-bounds indices.

  Additionally, the strided pattern `range(pid_k, n_blocks, SPLIT_K)` creates
  non-contiguous memory access across K blocks, which defeats coalescing.

Fix: Use a contiguous slice pattern instead:
  - block_start = pid_k * blocks_per_split
  - block_end   = min(block_start + blocks_per_split, n_blocks)
  - for block_idx in range(block_start, block_end):  <-- SAFE: constant start
    (Triton handles `range(const, var)` correctly; the trick is that
     block_start is computed from program_id but stored in a local variable
     that Triton treats as a loop-invariant constant after CSE.)

  But even safer: use `range(blocks_per_split)` with offset:
  - for i in range(blocks_per_split):
        block_idx = block_start + i
        if block_idx < n_blocks: ...
  This guarantees the loop trip count is a compile-time constant (known from
  the constexpr BLOCKS_PER_SPLIT), which Triton can always handle.

Performance notes:
  - SPLIT_K=4 for 8192-dim: 64 blocks / 4 = 16 blocks per split, 4x more
    threadblocks, GPU saturation goes from 16 to 64 (closer to 96 SMs).
  - SPLIT_K=8 for even larger dims is possible but adds reduction overhead.
  - The reduction kernel is trivial: sum SPLIT_K floats per output element.
  - Workspace is SPLIT_K * out_f * 4 bytes = 4 * 2048 * 4 = 32 KB (negligible).
  - atomicAdd alternative: saves one kernel launch but causes contention
    when SPLIT_K is large. We provide both paths.
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
# SplitK Kernel 1: Unpacked GEMV (int8 codes)
# ===================================================================

if HAS_TRITON:

    @triton.jit
    def polar_gemv_splitk_kernel(
        codes_ptr,       # int8 codes, shape (out_f, in_f_padded)
        x_ptr,           # Hadamard-transformed input, shape (in_f_padded,), float32
        norms_ptr,       # per-block norms, shape (out_f * n_blocks,), float32 or float16
        ct_ptr,          # pre-scaled centroids, float32
        partial_ptr,     # workspace for partial sums, shape (SPLIT_K, out_f), float32
        out_f,           # number of output features (rows)
        in_f_padded,     # padded input features (multiple of BLOCK_K)
        n_blocks,        # number of blocks per row = in_f_padded / BLOCK_K
        BLOCK_K: tl.constexpr,        # block_size (128)
        BLOCK_M: tl.constexpr,        # rows per program (autotuned)
        SPLIT_K: tl.constexpr,        # number of K splits
        BLOCKS_PER_SPLIT: tl.constexpr,  # ceil(n_blocks / SPLIT_K), loop trip count
    ):
        """SplitK GEMV: each program handles BLOCK_M rows over a K-slice.

        Grid: (ceil(out_f / BLOCK_M), SPLIT_K)
        Axis 0: output row tiles
        Axis 1: K-dimension splits

        Each program iterates over its assigned contiguous slice of K blocks,
        accumulates partial dot products, and writes them to partial_ptr.
        """
        pid_m = tl.program_id(0)  # which tile of output rows
        pid_k = tl.program_id(1)  # which K-split

        row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Contiguous K-slice for this split
        block_start = pid_k * BLOCKS_PER_SPLIT

        # Safe loop: trip count is the constexpr BLOCKS_PER_SPLIT,
        # with a dynamic guard for the last split (which may have fewer blocks)
        for i in range(BLOCKS_PER_SPLIT):
            block_idx = block_start + i

            # Guard: last split may have fewer than BLOCKS_PER_SPLIT blocks
            if block_idx < n_blocks:
                # Load transformed input for this block
                k_offs = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
                x_vals = tl.load(x_ptr + k_offs)

                # Load codes tile: (BLOCK_M, BLOCK_K)
                code_ptrs = row_offs[:, None] * in_f_padded + k_offs[None, :]
                codes_tile = tl.load(
                    codes_ptr + code_ptrs,
                    mask=row_mask[:, None],
                    other=0,
                )

                # Centroid lookup (pre-scaled, no 1/sqrt(bs) needed)
                values = tl.load(ct_ptr + codes_tile.to(tl.int32))

                # Per-block norm scaling
                norms_val = tl.load(
                    norms_ptr + row_offs * n_blocks + block_idx,
                    mask=row_mask,
                    other=0.0,
                )
                values = values * norms_val[:, None].to(tl.float32)

                # Dot product along K dimension
                dots = tl.sum(values * x_vals[None, :], axis=1)
                acc += dots

        # Store partial sum to workspace: partial_ptr[pid_k, row_offs]
        partial_ptrs = pid_k * out_f + row_offs
        tl.store(partial_ptr + partial_ptrs, acc, mask=row_mask)


    @triton.jit
    def polar_gemv_splitk_reduce_kernel(
        partial_ptr,    # shape (SPLIT_K, out_f), float32
        out_ptr,        # shape (out_f,), float32
        out_f,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Reduce SPLIT_K partial sums into final output.

        Grid: (ceil(out_f / BLOCK_M),)
        Each program reduces SPLIT_K values for BLOCK_M output elements.
        """
        pid = tl.program_id(0)
        row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k in range(SPLIT_K):
            vals = tl.load(
                partial_ptr + k * out_f + row_offs,
                mask=row_mask,
                other=0.0,
            )
            acc += vals

        tl.store(out_ptr + row_offs, acc, mask=row_mask)


    # ===================================================================
    # SplitK Kernel 1b: Unpacked GEMV with atomicAdd (no reduction pass)
    # ===================================================================

    @triton.jit
    def polar_gemv_splitk_atomic_kernel(
        codes_ptr,
        x_ptr,
        norms_ptr,
        ct_ptr,
        out_ptr,         # output, shape (out_f,), float32 -- MUST be zeroed
        out_f,
        in_f_padded,
        n_blocks,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """SplitK GEMV with atomicAdd -- single kernel, no workspace needed.

        Grid: (ceil(out_f / BLOCK_M), SPLIT_K)
        Each program atomicAdd's its partial sum directly to the output.

        Trade-off: atomicAdd contention at low SPLIT_K (2-4) is negligible
        on modern GPUs (hardware fp32 atomics since Volta). Saves one kernel
        launch and the workspace allocation.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        block_start = pid_k * BLOCKS_PER_SPLIT

        for i in range(BLOCKS_PER_SPLIT):
            block_idx = block_start + i
            if block_idx < n_blocks:
                k_offs = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
                x_vals = tl.load(x_ptr + k_offs)

                code_ptrs = row_offs[:, None] * in_f_padded + k_offs[None, :]
                codes_tile = tl.load(
                    codes_ptr + code_ptrs,
                    mask=row_mask[:, None],
                    other=0,
                )

                values = tl.load(ct_ptr + codes_tile.to(tl.int32))

                norms_val = tl.load(
                    norms_ptr + row_offs * n_blocks + block_idx,
                    mask=row_mask,
                    other=0.0,
                )
                values = values * norms_val[:, None].to(tl.float32)

                dots = tl.sum(values * x_vals[None, :], axis=1)
                acc += dots

        # AtomicAdd partial sums directly to output
        tl.atomic_add(out_ptr + row_offs, acc, mask=row_mask)


    # ===================================================================
    # SplitK Kernel 2: Packed INT4 nibble GEMV
    # ===================================================================

    @triton.jit
    def polar_gemv_packed_splitk_kernel(
        packed_ptr,      # packed uint8 codes, shape (out_f, in_f_half)
        x_ptr,           # Hadamard-transformed input, shape (in_f_padded,), float32
        norms_ptr,       # per-block norms, shape (out_f * n_blocks,), float32 or float16
        ct_ptr,          # pre-scaled centroids, float32
        partial_ptr,     # workspace for partial sums, shape (SPLIT_K, out_f), float32
        out_f,
        in_f_half,       # packed dimension = n_blocks * HALF_BK
        n_blocks,
        HALF_BK: tl.constexpr,        # half block size (64)
        BLOCK_M: tl.constexpr,        # rows per program
        SPLIT_K: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """SplitK packed INT4 GEMV.

        Same SplitK strategy as unpacked, but with nibble unpacking.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        block_start = pid_k * BLOCKS_PER_SPLIT

        for i in range(BLOCKS_PER_SPLIT):
            block_idx = block_start + i
            if block_idx < n_blocks:
                # Load x: two contiguous halves
                base_k = block_idx * HALF_BK * 2
                x_first = tl.load(x_ptr + base_k + tl.arange(0, HALF_BK))
                x_second = tl.load(x_ptr + base_k + HALF_BK + tl.arange(0, HALF_BK))

                # Load packed codes
                pack_offs = block_idx * HALF_BK + tl.arange(0, HALF_BK)
                pack_ptrs = row_offs[:, None] * in_f_half + pack_offs[None, :]
                packed = tl.load(
                    packed_ptr + pack_ptrs,
                    mask=row_mask[:, None],
                    other=0,
                )

                # Unpack nibbles
                low_codes = packed & 0xF
                high_codes = (packed >> 4) & 0xF

                # Centroid lookup
                low_vals = tl.load(ct_ptr + low_codes.to(tl.int32))
                high_vals = tl.load(ct_ptr + high_codes.to(tl.int32))

                # Per-block norm scaling
                norms_val = tl.load(
                    norms_ptr + row_offs * n_blocks + block_idx,
                    mask=row_mask,
                    other=0.0,
                )
                low_vals = low_vals * norms_val[:, None].to(tl.float32)
                high_vals = high_vals * norms_val[:, None].to(tl.float32)

                # Two contiguous dot products
                dots = (
                    tl.sum(low_vals * x_first[None, :], axis=1)
                    + tl.sum(high_vals * x_second[None, :], axis=1)
                )
                acc += dots

        # Store partial sum
        partial_ptrs = pid_k * out_f + row_offs
        tl.store(partial_ptr + partial_ptrs, acc, mask=row_mask)


    @triton.jit
    def polar_gemv_packed_splitk_atomic_kernel(
        packed_ptr,
        x_ptr,
        norms_ptr,
        ct_ptr,
        out_ptr,         # output, shape (out_f,), float32 -- MUST be zeroed
        out_f,
        in_f_half,
        n_blocks,
        HALF_BK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """SplitK packed INT4 GEMV with atomicAdd (no workspace)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        block_start = pid_k * BLOCKS_PER_SPLIT

        for i in range(BLOCKS_PER_SPLIT):
            block_idx = block_start + i
            if block_idx < n_blocks:
                base_k = block_idx * HALF_BK * 2
                x_first = tl.load(x_ptr + base_k + tl.arange(0, HALF_BK))
                x_second = tl.load(x_ptr + base_k + HALF_BK + tl.arange(0, HALF_BK))

                pack_offs = block_idx * HALF_BK + tl.arange(0, HALF_BK)
                pack_ptrs = row_offs[:, None] * in_f_half + pack_offs[None, :]
                packed = tl.load(
                    packed_ptr + pack_ptrs,
                    mask=row_mask[:, None],
                    other=0,
                )

                low_codes = packed & 0xF
                high_codes = (packed >> 4) & 0xF

                low_vals = tl.load(ct_ptr + low_codes.to(tl.int32))
                high_vals = tl.load(ct_ptr + high_codes.to(tl.int32))

                norms_val = tl.load(
                    norms_ptr + row_offs * n_blocks + block_idx,
                    mask=row_mask,
                    other=0.0,
                )
                low_vals = low_vals * norms_val[:, None].to(tl.float32)
                high_vals = high_vals * norms_val[:, None].to(tl.float32)

                dots = (
                    tl.sum(low_vals * x_first[None, :], axis=1)
                    + tl.sum(high_vals * x_second[None, :], axis=1)
                )
                acc += dots

        tl.atomic_add(out_ptr + row_offs, acc, mask=row_mask)


# ===================================================================
# Wrapper functions
# ===================================================================

def _choose_split_k(n_blocks: int) -> int:
    """Choose SPLIT_K based on number of K-blocks.

    Heuristic: we want each split to process at least 4 blocks
    (enough work to amortize kernel launch overhead), and SPLIT_K
    should be a power of 2 for clean division.

    n_blocks=32 (4096-dim) -> SPLIT_K=4  (8 blocks/split)
    n_blocks=64 (8192-dim) -> SPLIT_K=8  (8 blocks/split)
    n_blocks=16 (2048-dim) -> SPLIT_K=4  (4 blocks/split)
    n_blocks=8  (1024-dim) -> SPLIT_K=2  (4 blocks/split)
    """
    if n_blocks <= 4:
        return 1  # too few blocks, no benefit from splitting
    elif n_blocks <= 8:
        return 2
    elif n_blocks <= 32:
        return 4
    else:
        return 8


def polar_gemv_splitk(
    codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_padded: int,
    n_blocks: int,
    block_size: int = 128,
    split_k: Optional[int] = None,
    use_atomic: bool = False,
    block_m: int = 128,
) -> torch.Tensor:
    """Launch SplitK unpacked GEMV kernel.

    Args:
        codes: int8 codes, shape (out_f, in_f_padded)
        x_transformed: Hadamard-transformed input vector, shape (in_f_padded,), float32
        norms: per-block norms, shape (out_f * n_blocks,), float16 or float32
        ct_scaled: pre-scaled centroids (centroids / sqrt(block_size)), float32
        out_f: number of output features
        in_f_padded: padded input features
        n_blocks: blocks per row = in_f_padded / block_size
        block_size: quantization block size (default 128)
        split_k: number of K-splits (None = auto-choose)
        use_atomic: if True, use atomicAdd variant (no workspace)
        block_m: rows per threadblock (default 128)

    Returns:
        output tensor, shape (out_f,), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemv_splitk kernel")

    if split_k is None:
        split_k = _choose_split_k(n_blocks)

    blocks_per_split = (n_blocks + split_k - 1) // split_k
    grid_m = (out_f + block_m - 1) // block_m

    if use_atomic:
        output = torch.zeros(out_f, device=codes.device, dtype=torch.float32)
        polar_gemv_splitk_atomic_kernel[(grid_m, split_k)](
            codes, x_transformed, norms, ct_scaled, output,
            out_f, in_f_padded, n_blocks,
            BLOCK_K=block_size,
            BLOCK_M=block_m,
            SPLIT_K=split_k,
            BLOCKS_PER_SPLIT=blocks_per_split,
        )
        return output
    else:
        # Two-pass: compute partial sums, then reduce
        partial = torch.empty(
            (split_k, out_f), device=codes.device, dtype=torch.float32
        )
        polar_gemv_splitk_kernel[(grid_m, split_k)](
            codes, x_transformed, norms, ct_scaled, partial,
            out_f, in_f_padded, n_blocks,
            BLOCK_K=block_size,
            BLOCK_M=block_m,
            SPLIT_K=split_k,
            BLOCKS_PER_SPLIT=blocks_per_split,
        )
        # Reduction pass
        output = torch.empty(out_f, device=codes.device, dtype=torch.float32)
        reduce_block_m = 256
        reduce_grid = (out_f + reduce_block_m - 1) // reduce_block_m
        polar_gemv_splitk_reduce_kernel[(reduce_grid,)](
            partial, output, out_f,
            SPLIT_K=split_k,
            BLOCK_M=reduce_block_m,
        )
        return output


def polar_gemv_packed_splitk(
    packed_codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_half: int,
    n_blocks: int,
    block_size: int = 128,
    split_k: Optional[int] = None,
    use_atomic: bool = False,
    block_m: int = 128,
) -> torch.Tensor:
    """Launch SplitK packed (INT4) GEMV kernel.

    Args:
        packed_codes: uint8 packed codes, shape (out_f, in_f_half)
        x_transformed: Hadamard-transformed input vector, shape (in_f_padded,), float32
        norms: per-block norms, shape (out_f * n_blocks,), float16 or float32
        ct_scaled: pre-scaled centroids, float32
        out_f: number of output features
        in_f_half: packed dimension = n_blocks * (block_size // 2)
        n_blocks: blocks per row
        block_size: quantization block size (default 128)
        split_k: number of K-splits (None = auto-choose)
        use_atomic: if True, use atomicAdd variant
        block_m: rows per threadblock (default 128)

    Returns:
        output tensor, shape (out_f,), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemv_packed_splitk kernel")

    if split_k is None:
        split_k = _choose_split_k(n_blocks)

    half_bk = block_size // 2
    blocks_per_split = (n_blocks + split_k - 1) // split_k
    grid_m = (out_f + block_m - 1) // block_m

    if use_atomic:
        output = torch.zeros(out_f, device=packed_codes.device, dtype=torch.float32)
        polar_gemv_packed_splitk_atomic_kernel[(grid_m, split_k)](
            packed_codes, x_transformed, norms, ct_scaled, output,
            out_f, in_f_half, n_blocks,
            HALF_BK=half_bk,
            BLOCK_M=block_m,
            SPLIT_K=split_k,
            BLOCKS_PER_SPLIT=blocks_per_split,
        )
        return output
    else:
        partial = torch.empty(
            (split_k, out_f), device=packed_codes.device, dtype=torch.float32
        )
        polar_gemv_packed_splitk_kernel[(grid_m, split_k)](
            packed_codes, x_transformed, norms, ct_scaled, partial,
            out_f, in_f_half, n_blocks,
            HALF_BK=half_bk,
            BLOCK_M=block_m,
            SPLIT_K=split_k,
            BLOCKS_PER_SPLIT=blocks_per_split,
        )
        output = torch.empty(out_f, device=packed_codes.device, dtype=torch.float32)
        reduce_block_m = 256
        reduce_grid = (out_f + reduce_block_m - 1) // reduce_block_m
        polar_gemv_splitk_reduce_kernel[(reduce_grid,)](
            partial, output, out_f,
            SPLIT_K=split_k,
            BLOCK_M=reduce_block_m,
        )
        return output


# ===================================================================
# Packing utility (re-exported for convenience)
# ===================================================================

def pack_codes_int4(codes: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Pack int8 codes into INT4 nibble format (half-block packing).

    Same as polar_gemv.pack_codes_int4 -- duplicated here so this module
    is self-contained for testing.
    """
    half_bk = block_size // 2
    out_f = codes.shape[0]
    n_blocks = codes.shape[1] // block_size
    codes_blocked = codes.view(out_f, n_blocks, block_size)
    first_half = codes_blocked[:, :, :half_bk].to(torch.uint8)
    second_half = codes_blocked[:, :, half_bk:].to(torch.uint8)
    packed = ((second_half << 4) | first_half).reshape(out_f, -1)
    return packed


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
        (2048, 8192),    # the problematic shape (4.5x slower than cuBLAS)
        (512, 4096),
        (12288, 4096),
        (2048, 16384),   # extra-large K to stress SplitK
    ]

    print("=" * 70)
    print("  PolarEngine SplitK GEMV Kernel Tests")
    print("=" * 70)

    all_pass = True

    # ---- Test 1: Correctness of unpacked SplitK kernels ----
    print("\n--- Unpacked SplitK (two-pass) ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

        # Reference
        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]
        values = values * norms_blocked
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        for sk in [2, 4, 8]:
            kernel_out = polar_gemv_splitk(
                codes, x_transformed, norms, ct_scaled,
                out_f, in_f_padded, n_blocks, block_size,
                split_k=sk, use_atomic=False,
            )
            cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
            rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()
            status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] ({out_f:>5}x{in_f:>5}) SK={sk}: cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 2: Correctness of unpacked SplitK atomic ----
    print("\n--- Unpacked SplitK (atomic) ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]
        values = values * norms_blocked
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        sk = 4
        kernel_out = polar_gemv_splitk(
            codes, x_transformed, norms, ct_scaled,
            out_f, in_f_padded, n_blocks, block_size,
            split_k=sk, use_atomic=True,
        )
        cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
        rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()
        # atomicAdd with float32 may have slight non-determinism but should still be very close
        status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] ({out_f:>5}x{in_f:>5}) SK={sk} atomic: cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 3: Correctness of packed SplitK ----
    print("\n--- Packed SplitK (two-pass) ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size
        half_bk = block_size // 2
        in_f_half = n_blocks * half_bk

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

        packed_codes = pack_codes_int4(codes, block_size)

        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]
        values = values * norms_blocked
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        for sk in [2, 4, 8]:
            kernel_out = polar_gemv_packed_splitk(
                packed_codes, x_transformed, norms, ct_scaled,
                out_f, in_f_half, n_blocks, block_size,
                split_k=sk, use_atomic=False,
            )
            cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
            rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()
            status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] ({out_f:>5}x{in_f:>5}) SK={sk}: cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 4: Packed SplitK atomic ----
    print("\n--- Packed SplitK (atomic) ---")
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size
        half_bk = block_size // 2
        in_f_half = n_blocks * half_bk

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

        packed_codes = pack_codes_int4(codes, block_size)

        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]
        values = values * norms_blocked
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        sk = 4
        kernel_out = polar_gemv_packed_splitk(
            packed_codes, x_transformed, norms, ct_scaled,
            out_f, in_f_half, n_blocks, block_size,
            split_k=sk, use_atomic=True,
        )
        cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
        rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()
        status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] ({out_f:>5}x{in_f:>5}) SK={sk} atomic: cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Test 5: SplitK=1 matches original kernel ----
    print("\n--- SplitK=1 (should match original kernel exactly) ---")
    from polarengine_vllm.kernels.polar_gemv import polar_gemv, polar_gemv_packed

    for out_f, in_f in [(4096, 4096), (2048, 8192)]:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size

        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

        # Original kernel
        orig_out = polar_gemv(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)

        # SplitK=1 (should be identical path)
        sk1_out = polar_gemv_splitk(
            codes, x_transformed, norms, ct_scaled,
            out_f, in_f_padded, n_blocks, block_size,
            split_k=1, use_atomic=False,
        )
        cos = F.cosine_similarity(orig_out.unsqueeze(0), sk1_out.unsqueeze(0)).item()
        rel_err = (torch.abs(orig_out - sk1_out) / (torch.abs(orig_out) + 1e-8)).mean().item()
        status = "PASS" if cos > 0.99999 and rel_err < 0.0001 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] Unpacked ({out_f:>5}x{in_f:>5}) SK=1 vs orig: cos={cos:.6f}, rel_err={rel_err:.6f}")

    # ---- Benchmark: SplitK vs original ----
    print("\n--- Benchmark: SplitK vs original (2048x8192) ---")
    out_f, in_f = 2048, 8192
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size

    codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
    norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
    x = torch.randn(in_f, dtype=torch.float32, device=device)
    x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
    x_transformed = torch.matmul(x_padded.view(-1, block_size), H).reshape(-1)

    n_warmup = 50
    n_bench = 200

    # Warmup
    for _ in range(n_warmup):
        _ = polar_gemv(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    for sk in [4, 8]:
        for _ in range(n_warmup):
            _ = polar_gemv_splitk(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size, split_k=sk)
    torch.cuda.synchronize()

    # Original
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = polar_gemv(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    torch.cuda.synchronize()
    t_orig = (time.perf_counter() - t0) / n_bench * 1000

    print(f"  Original (SK=1):   {t_orig:.4f} ms")

    for sk in [2, 4, 8]:
        for use_atomic in [False, True]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_bench):
                _ = polar_gemv_splitk(
                    codes, x_transformed, norms, ct_scaled,
                    out_f, in_f_padded, n_blocks, block_size,
                    split_k=sk, use_atomic=use_atomic,
                )
            torch.cuda.synchronize()
            t_sk = (time.perf_counter() - t0) / n_bench * 1000
            mode = "atomic" if use_atomic else "2-pass"
            speedup = t_orig / t_sk if t_sk > 0 else float("inf")
            print(f"  SplitK={sk} ({mode:6s}): {t_sk:.4f} ms  ({speedup:.2f}x)")

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  SOME TESTS FAILED.")
        sys.exit(1)
