"""PolarEngine Triton GEMV kernels for quantized inference.

Two kernels:
  1. polar_gemv_kernel        -- unpacked (int8 codes, 1 byte per code)
  2. polar_gemv_packed_kernel -- packed INT4 nibble (2 codes per byte, half-block packing)

Both use pre-scaled centroids (ct already includes 1/sqrt(block_size)) and
SPLIT_K=1 (SplitK causes PPL regression in full model inference).

Extracted from EOQ_POLAR_ENGINE_V4 notebook (proven on Qwen-2.5-9B, PPL 6.43).
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
# Kernel 1: Unpacked GEMV (proven v4, SPLIT_K=1)
# ===================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        ],
        key=['out_f', 'in_f_padded'],
    )
    @triton.jit
    def polar_gemv_kernel(
        codes_ptr,    # int8 codes, shape (out_f, in_f_padded)
        x_ptr,        # Hadamard-transformed input, shape (in_f_padded,), float32
        norms_ptr,    # per-block norms, shape (out_f * n_blocks,), float32 or float16
        ct_ptr,       # pre-scaled centroids (includes 1/sqrt(block_size)), float32
        out_ptr,      # output, shape (out_f,), float32
        out_f,        # number of output features (rows)
        in_f_padded,  # padded input features (multiple of BLOCK_K)
        n_blocks,     # number of blocks per row = in_f_padded / BLOCK_K
        BLOCK_K: tl.constexpr,   # must equal block_size (128)
        BLOCK_M: tl.constexpr,   # rows per program (autotuned)
    ):
        """Tiled GEMV: centroid lookup + norm scaling + dot product.

        Each program handles BLOCK_M output rows. Iterates over n_blocks
        in the K dimension. Centroids are pre-scaled by 1/sqrt(block_size)
        so no additional scale factor is needed.
        """
        pid = tl.program_id(0)
        row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for block_idx in range(n_blocks):
            # Load transformed input for this block
            k_offs = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            x_vals = tl.load(x_ptr + k_offs)

            # Load codes tile: (BLOCK_M, BLOCK_K)
            code_ptrs = row_offs[:, None] * in_f_padded + k_offs[None, :]
            codes_tile = tl.load(codes_ptr + code_ptrs, mask=row_mask[:, None], other=0)

            # Centroid lookup (pre-scaled, no 1/sqrt(bs) needed)
            values = tl.load(ct_ptr + codes_tile.to(tl.int32))

            # Per-block norm scaling
            norms_val = tl.load(
                norms_ptr + row_offs * n_blocks + block_idx,
                mask=row_mask, other=0.0,
            )
            values = values * norms_val[:, None].to(tl.float32)

            # Dot product along K dimension
            dots = tl.sum(values * x_vals[None, :], axis=1)
            acc += dots

        tl.store(out_ptr + row_offs, acc, mask=row_mask)


# ===================================================================
# Kernel 2: Packed INT4 nibble GEMV (half-block packing)
# ===================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
        ],
        key=['out_f', 'in_f_half'],
    )
    @triton.jit
    def polar_gemv_packed_kernel(
        packed_ptr,   # packed uint8 codes, shape (out_f, in_f_half)
        x_ptr,        # Hadamard-transformed input, shape (in_f_padded,), float32
        norms_ptr,    # per-block norms, shape (out_f * n_blocks,), float32 or float16
        ct_ptr,       # pre-scaled centroids (includes 1/sqrt(block_size)), float32
        out_ptr,      # output, shape (out_f,), float32
        out_f,        # number of output features (rows)
        in_f_half,    # packed dimension = n_blocks * HALF_BK (half of unpacked)
        n_blocks,     # number of blocks per row
        HALF_BK: tl.constexpr,   # half block size (64 for block_size=128)
        BLOCK_M: tl.constexpr,   # rows per program (autotuned)
    ):
        """Packed INT4 GEMV: loads half the bytes of unpacked kernel.

        Half-block packing layout:
            byte[i] = (code[HALF_BK + i] << 4) | code[i]   for i in [0, HALF_BK)
            Low nibble  = first half of block  (codes 0..63)
            High nibble = second half of block (codes 64..127)

        This allows two contiguous x loads per block (no strided access).
        """
        pid = tl.program_id(0)
        row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < out_f
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for block_idx in range(n_blocks):
            # Load x: two contiguous halves (no strided loads)
            base_k = block_idx * HALF_BK * 2
            x_first = tl.load(x_ptr + base_k + tl.arange(0, HALF_BK))
            x_second = tl.load(x_ptr + base_k + HALF_BK + tl.arange(0, HALF_BK))

            # Load packed codes: HALF_BK bytes per row (half of unpacked)
            pack_offs = block_idx * HALF_BK + tl.arange(0, HALF_BK)
            pack_ptrs = row_offs[:, None] * in_f_half + pack_offs[None, :]
            packed = tl.load(packed_ptr + pack_ptrs, mask=row_mask[:, None], other=0)

            # Unpack nibbles (2 cheap bitwise ops)
            low_codes = packed & 0xF              # first half of block
            high_codes = (packed >> 4) & 0xF      # second half of block

            # Centroid lookup (pre-scaled)
            low_vals = tl.load(ct_ptr + low_codes.to(tl.int32))
            high_vals = tl.load(ct_ptr + high_codes.to(tl.int32))

            # Per-block norm scaling
            norms_val = tl.load(
                norms_ptr + row_offs * n_blocks + block_idx,
                mask=row_mask, other=0.0,
            )
            low_vals = low_vals * norms_val[:, None].to(tl.float32)
            high_vals = high_vals * norms_val[:, None].to(tl.float32)

            # Two contiguous dot products
            dots = (
                tl.sum(low_vals * x_first[None, :], axis=1)
                + tl.sum(high_vals * x_second[None, :], axis=1)
            )
            acc += dots

        tl.store(out_ptr + row_offs, acc, mask=row_mask)


# ===================================================================
# Wrapper functions
# ===================================================================

def polar_gemv(
    codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_padded: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Launch unpacked GEMV kernel.

    Args:
        codes: int8 codes, shape (out_f, in_f_padded) -- flat row-major
        x_transformed: Hadamard-transformed input vector, shape (in_f_padded,), float32
        norms: per-block norms, shape (out_f * n_blocks,), float16 or float32
        ct_scaled: pre-scaled centroids (centroids / sqrt(block_size)), float32
        out_f: number of output features
        in_f_padded: padded input features (multiple of block_size)
        n_blocks: blocks per row = in_f_padded / block_size
        block_size: quantization block size (default 128, must match BLOCK_K)

    Returns:
        output tensor, shape (out_f,), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemv kernel")

    output = torch.zeros(out_f, device=codes.device, dtype=torch.float32)
    grid = lambda meta: ((out_f + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
    polar_gemv_kernel[grid](
        codes, x_transformed, norms, ct_scaled, output,
        out_f, in_f_padded, n_blocks,
        BLOCK_K=block_size,
    )
    return output


def polar_gemv_packed(
    packed_codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_half: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Launch packed (INT4 nibble) GEMV kernel.

    Args:
        packed_codes: uint8 packed codes, shape (out_f, in_f_half)
            Packing: byte[i] = (code[half_bk+i] << 4) | code[i]
        x_transformed: Hadamard-transformed input vector, shape (in_f_padded,), float32
        norms: per-block norms, shape (out_f * n_blocks,), float16 or float32
        ct_scaled: pre-scaled centroids (centroids / sqrt(block_size)), float32
        out_f: number of output features
        in_f_half: packed dimension = n_blocks * (block_size // 2)
        n_blocks: blocks per row
        block_size: quantization block size (default 128)

    Returns:
        output tensor, shape (out_f,), float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for polar_gemv_packed kernel")

    half_bk = block_size // 2
    output = torch.zeros(out_f, device=packed_codes.device, dtype=torch.float32)
    grid = lambda meta: ((out_f + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
    polar_gemv_packed_kernel[grid](
        packed_codes, x_transformed, norms, ct_scaled, output,
        out_f, in_f_half, n_blocks,
        HALF_BK=half_bk,
    )
    return output


# ===================================================================
# Packing utility
# ===================================================================

def pack_codes_int4(codes: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Pack int8 codes into INT4 nibble format (half-block packing).

    Args:
        codes: int8 codes, shape (out_f, in_f_padded) where in_f_padded is
               a multiple of block_size.
        block_size: quantization block size (default 128).

    Returns:
        packed: uint8 tensor, shape (out_f, in_f_padded // 2).
            byte[i] = (code[half_bk + i] << 4) | code[i]
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

    # Pre-compute Lloyd-Max-style centroids for N(0,1) (simplified for test)
    # In production these come from the quantizer.
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

    print("=" * 60)
    print("  PolarEngine GEMV Kernel Tests")
    print("=" * 60)

    all_pass = True

    for out_f, in_f in test_shapes:
        # Pad to multiple of block_size
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size

        # Random quantized data
        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)

        # Pad and transform input
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(
            x_padded.view(-1, block_size), H
        ).reshape(-1)

        # ----- Reference: torch matmul -----
        # Dequantize: centroid lookup + norm + inverse Hadamard
        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]          # (out_f, n_blocks, block_size)
        values = values * norms_blocked                   # scale by norms
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)         # inverse Hadamard
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        # ----- Kernel output -----
        kernel_out = polar_gemv(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)

        # ----- Check -----
        cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
        rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()

        status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] Unpacked ({out_f:>5}x{in_f:>5}): cosine={cos:.6f}, rel_err={rel_err:.6f}")

    # ----- Packed kernel tests -----
    print()
    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        n_blocks = in_f_padded // block_size
        half_bk = block_size // 2
        in_f_half = n_blocks * half_bk

        # Random quantized data
        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device)
        norms = torch.randn(out_f * n_blocks, dtype=torch.float32, device=device).abs() + 0.1
        x = torch.randn(in_f, dtype=torch.float32, device=device)

        # Pad and transform input
        x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
        x_transformed = torch.matmul(
            x_padded.view(-1, block_size), H
        ).reshape(-1)

        # Pack codes
        packed_codes = pack_codes_int4(codes, block_size)

        # ----- Reference: torch matmul (same as unpacked) -----
        codes_blocked = codes.view(out_f, n_blocks, block_size)
        norms_blocked = norms.view(out_f, n_blocks, 1)
        values = ct_scaled[codes_blocked.long()]
        values = values * norms_blocked
        values_flat = values.view(out_f * n_blocks, block_size)
        values_iht = torch.matmul(values_flat, H)
        w_deq = values_iht.view(out_f, in_f_padded)[:, :in_f]
        ref_out = torch.matmul(w_deq, x[:in_f])

        # ----- Packed kernel output -----
        kernel_out = polar_gemv_packed(packed_codes, x_transformed, norms, ct_scaled, out_f, in_f_half, n_blocks, block_size)

        # ----- Check -----
        cos = F.cosine_similarity(ref_out.unsqueeze(0), kernel_out.unsqueeze(0)).item()
        rel_err = (torch.abs(ref_out - kernel_out) / (torch.abs(ref_out) + 1e-8)).mean().item()

        status = "PASS" if cos > 0.9999 and rel_err < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] Packed   ({out_f:>5}x{in_f:>5}): cosine={cos:.6f}, rel_err={rel_err:.6f}")

    # ----- Pack/unpack roundtrip -----
    print()
    print("  Pack/unpack roundtrip:")
    codes_rt = torch.randint(0, n_levels, (256, 1024), dtype=torch.int8, device=device)
    packed_rt = pack_codes_int4(codes_rt, block_size)
    # Unpack and verify
    codes_blocked_rt = codes_rt.view(256, -1, block_size)
    first_half_rt = codes_blocked_rt[:, :, :64]
    second_half_rt = codes_blocked_rt[:, :, 64:]
    low_rt = (packed_rt.view(256, -1, 64) & 0xF).to(torch.int8)
    high_rt = ((packed_rt.view(256, -1, 64) >> 4) & 0xF).to(torch.int8)
    match_low = (low_rt == first_half_rt).all().item()
    match_high = (high_rt == second_half_rt).all().item()
    status = "PASS" if match_low and match_high else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{status}] low nibble match={match_low}, high nibble match={match_high}")

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  SOME TESTS FAILED.")
        sys.exit(1)
