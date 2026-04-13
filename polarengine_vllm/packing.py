"""INT4 nibble packing utilities for PolarEngine.

Half-block packing order:
    byte[i] = (code[block_size//2 + i] << 4) | code[i]

First half of each block -> low nibble
Second half -> high nibble

This enables contiguous x loads in the packed kernel
(no strided access required).
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F


def pack_codes_half_block(codes: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Pack int8 codes into nibbles using half-block order.

    Args:
        codes: (out_f, in_f_padded) int8 tensor with values 0-15
        block_size: must be even (default 128)

    Returns:
        packed: (out_f, in_f_padded // 2) uint8 tensor
    """
    assert block_size % 2 == 0, f"block_size must be even, got {block_size}"
    out_f, in_f_padded = codes.shape
    assert in_f_padded % block_size == 0, (
        f"in_f_padded ({in_f_padded}) must be a multiple of block_size ({block_size})"
    )

    # Validate code range: must be 0-15 for nibble packing
    codes_u8 = codes.to(torch.uint8)
    assert (codes_u8 <= 15).all(), (
        "All code values must be in [0, 15] for INT4 nibble packing. "
        f"Found max={codes_u8.max().item()}"
    )

    half_bk = block_size // 2
    n_blocks = in_f_padded // block_size

    # Reshape into blocks: (out_f, n_blocks, block_size)
    codes_blocked = codes_u8.view(out_f, n_blocks, block_size)

    # Split each block into first half and second half
    first_half = codes_blocked[:, :, :half_bk]    # -> low nibble
    second_half = codes_blocked[:, :, half_bk:]    # -> high nibble

    # Pack: low nibble from first half, high nibble from second half
    packed = (second_half << 4) | first_half

    # Flatten blocks back: (out_f, n_blocks * half_bk) = (out_f, in_f_padded // 2)
    packed = packed.reshape(out_f, -1)
    return packed


def unpack_codes_half_block(packed: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Unpack nibble-packed codes back to int8.

    Args:
        packed: (out_f, in_f_padded // 2) uint8 tensor
        block_size: must be even

    Returns:
        codes: (out_f, in_f_padded) int8 tensor with values 0-15
    """
    assert block_size % 2 == 0, f"block_size must be even, got {block_size}"
    half_bk = block_size // 2
    out_f, in_f_half = packed.shape
    assert in_f_half % half_bk == 0, (
        f"Packed dim ({in_f_half}) must be a multiple of half_bk ({half_bk})"
    )

    n_blocks = in_f_half // half_bk

    # Reshape to (out_f, n_blocks, half_bk) for per-block unpacking
    packed_blocked = packed.view(out_f, n_blocks, half_bk)

    # Extract nibbles
    low_codes = packed_blocked & 0xF              # first half of block
    high_codes = (packed_blocked >> 4) & 0xF      # second half of block

    # Reconstruct original block order: first_half then second_half
    codes_blocked = torch.cat([low_codes, high_codes], dim=2)  # (out_f, n_blocks, block_size)

    # Flatten and convert to int8
    codes = codes_blocked.reshape(out_f, -1).to(torch.int8)
    return codes


def pack_codes_q5(codes: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Pack Q5 codes (0-31) from int8 into 5-bit packed uint8.

    Input: (N, n_blocks * block_size) int8, values 0-31
    Output: (N, n_blocks * packed_block_bytes) uint8

    Each block of 128 codes (640 bits) packs into 80 bytes.

    Packing layout per block of 128 codes:
    - Process 8 codes at a time into 5 bytes (8 x 5 bits = 40 bits = 5 bytes)
    - 128 codes / 8 = 16 groups -> 16 x 5 = 80 bytes per block

    For 8 codes [c0..c7], each 5 bits, packed into 5 bytes [b0..b4]:
      b0 = (c0) | (c1 << 5)           -> c0[4:0] + c1[2:0]
      b1 = (c1 >> 3) | (c2 << 2) | (c3 << 7)  -> c1[4:3] + c2[4:0] + c3[0]
      b2 = (c3 >> 1) | (c4 << 4)      -> c3[4:1] + c4[3:0]
      b3 = (c4 >> 4) | (c5 << 1) | (c6 << 6)  -> c4[4] + c5[4:0] + c6[1:0]
      b4 = (c6 >> 2) | (c7 << 3)      -> c6[4:2] + c7[4:0]

    Args:
        codes: (N, K) int8 tensor with values 0-31, K must be a multiple of block_size
        block_size: must be a multiple of 8 (default 128)

    Returns:
        packed: (N, n_blocks * (block_size * 5 // 8)) uint8 tensor
    """
    assert block_size % 8 == 0, f"block_size must be a multiple of 8, got {block_size}"
    *leading, K = codes.shape
    assert K % block_size == 0, (
        f"Last dimension ({K}) must be a multiple of block_size ({block_size})"
    )

    # Validate code range: must be 0-31 for 5-bit packing
    codes_long = codes.to(torch.long) & 0xFF  # ensure unsigned interpretation
    assert (codes_long <= 31).all(), (
        "All code values must be in [0, 31] for Q5 packing. "
        f"Found max={codes_long.max().item()}"
    )

    # Reshape so that the last dimension is groups of 8
    # (*leading, K) -> (*leading, K // 8, 8)
    grouped = codes_long.reshape(*leading, K // 8, 8)

    c0 = grouped[..., 0]
    c1 = grouped[..., 1]
    c2 = grouped[..., 2]
    c3 = grouped[..., 3]
    c4 = grouped[..., 4]
    c5 = grouped[..., 5]
    c6 = grouped[..., 6]
    c7 = grouped[..., 7]

    b0 = (c0 | (c1 << 5)) & 0xFF
    b1 = ((c1 >> 3) | (c2 << 2) | (c3 << 7)) & 0xFF
    b2 = ((c3 >> 1) | (c4 << 4)) & 0xFF
    b3 = ((c4 >> 4) | (c5 << 1) | (c6 << 6)) & 0xFF
    b4 = ((c6 >> 2) | (c7 << 3)) & 0xFF

    # Stack: (*leading, K // 8, 5)
    packed = torch.stack([b0, b1, b2, b3, b4], dim=-1).to(torch.uint8)

    # Flatten the last two dims: (*leading, K * 5 // 8)
    packed = packed.reshape(*leading, K * 5 // 8)
    return packed


def unpack_codes_q5(packed: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Inverse of pack_codes_q5. Returns int8 tensor with values 0-31.

    Args:
        packed: (*leading, K_packed) uint8 tensor where K_packed = original_K * 5 // 8
        block_size: must be a multiple of 8 (default 128)

    Returns:
        codes: (*leading, K_packed * 8 // 5) int8 tensor with values 0-31
    """
    assert block_size % 8 == 0, f"block_size must be a multiple of 8, got {block_size}"
    *leading, K_packed = packed.shape
    assert (K_packed * 8) % 5 == 0, (
        f"Packed dimension ({K_packed}) is not valid for Q5 unpacking "
        f"(K_packed * 8 must be divisible by 5)"
    )

    packed_block_bytes = block_size * 5 // 8
    assert K_packed % packed_block_bytes == 0, (
        f"Packed dimension ({K_packed}) must be a multiple of "
        f"packed_block_bytes ({packed_block_bytes})"
    )

    # Reshape to groups of 5 bytes: (*leading, K_packed // 5, 5)
    p = packed.to(torch.long).reshape(*leading, K_packed // 5, 5)

    b0 = p[..., 0]
    b1 = p[..., 1]
    b2 = p[..., 2]
    b3 = p[..., 3]
    b4 = p[..., 4]

    c0 = b0 & 0x1F
    c1 = ((b0 >> 5) | (b1 << 3)) & 0x1F
    c2 = (b1 >> 2) & 0x1F
    c3 = ((b1 >> 7) | (b2 << 1)) & 0x1F
    c4 = ((b2 >> 4) | (b3 << 4)) & 0x1F
    c5 = (b3 >> 1) & 0x1F
    c6 = ((b3 >> 6) | (b4 << 2)) & 0x1F
    c7 = (b4 >> 3) & 0x1F

    # Stack: (*leading, K_packed // 5, 8)
    codes = torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1).to(torch.int8)

    # Flatten last two dims: (*leading, K_packed * 8 // 5)
    codes = codes.reshape(*leading, K_packed * 8 // 5)
    return codes


# ─────────────────────────────────────────────────────────────────────────
# HLWQ bit-packed format (used by Gemopus / MiniMax HLWQ-Q5 skill output)
# ─────────────────────────────────────────────────────────────────────────
#
# Different bit layout than pack_codes_q5. 8 codes (5 bits each, 40 bits)
# pack into 5 bytes with MSB-first placement:
#
#   b0 = (c0 << 3) | (c1 >> 2)                  # c0[4:0] + c1[4:2]
#   b1 = ((c1 & 3) << 6) | (c2 << 1) | (c3 >> 4) # c1[1:0] + c2[4:0] + c3[4]
#   b2 = ((c3 & 15) << 4) | (c4 >> 1)           # c3[3:0] + c4[4:1]
#   b3 = ((c4 & 1) << 7) | (c5 << 2) | (c6 >> 3) # c4[0] + c5[4:0] + c6[4:2]
#   b4 = ((c6 & 7) << 5) | c7                   # c6[2:0] + c7[4:0]
#
# This matches the `bitpack_5` function used in the /polarquant skill
# notebooks for Gemopus-4-26B-A4B-it-HLWQ-Q5 and MiniMax-M2.7-HLWQ-Q5.


def unpack_codes_q5_hlwq(packed: torch.Tensor, total: int | None = None) -> torch.Tensor:
    """Unpack HLWQ-bit-packed Q5 codes (bitpack_5 layout) to int8.

    This is the inverse of the ``bitpack_5`` helper in the /polarquant
    skill notebooks (see Gemopus/MiniMax Etapa 1). Distinct from
    ``unpack_codes_q5`` which decodes a different bit layout.

    Args:
        packed: uint8 tensor of arbitrary shape; last dim must be
                divisible by 5 (5 bytes per 8-code group).
        total:  optional exact code count to return. If None, returns
                (K // 5) * 8 codes including any padding.

    Returns:
        codes: int8 tensor with values 0-31. Flat on the last dim.
    """
    *leading, K_packed = packed.shape
    assert K_packed % 5 == 0, (
        f"HLWQ bit-packed codes require K_packed % 5 == 0, got {K_packed}"
    )
    p = packed.to(torch.long).reshape(*leading, K_packed // 5, 5)

    b0 = p[..., 0]
    b1 = p[..., 1]
    b2 = p[..., 2]
    b3 = p[..., 3]
    b4 = p[..., 4]

    c0 = (b0 >> 3) & 0x1F
    c1 = ((b0 & 7) << 2) | ((b1 >> 6) & 3)
    c2 = (b1 >> 1) & 0x1F
    c3 = ((b1 & 1) << 4) | ((b2 >> 4) & 0xF)
    c4 = ((b2 & 0xF) << 1) | ((b3 >> 7) & 1)
    c5 = (b3 >> 2) & 0x1F
    c6 = ((b3 & 3) << 3) | ((b4 >> 5) & 7)
    c7 = b4 & 0x1F

    codes = torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1).to(torch.int8)
    # Flatten the last two dims (K_packed//5, 8) → K_packed*8//5
    codes = codes.reshape(*codes.shape[:-2], -1)
    if total is not None:
        codes = codes[..., :total]
    return codes


def pack_codes_q5_hlwq(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of unpack_codes_q5_hlwq. Not used by the loader, but kept
    here for round-trip testing against the /polarquant skill output."""
    *leading, K = codes.shape
    pad = (8 - K % 8) % 8
    if pad:
        codes = torch.cat([codes, torch.zeros(*leading, pad, dtype=codes.dtype)], dim=-1)
        K = K + pad
    c = codes.to(torch.long).reshape(*leading, K // 8, 8)
    c0, c1, c2, c3, c4, c5, c6, c7 = [c[..., i] for i in range(8)]
    b0 = ((c0 << 3) | (c1 >> 2)) & 0xFF
    b1 = (((c1 & 3) << 6) | (c2 << 1) | (c3 >> 4)) & 0xFF
    b2 = (((c3 & 0xF) << 4) | (c4 >> 1)) & 0xFF
    b3 = (((c4 & 1) << 7) | (c5 << 2) | (c6 >> 3)) & 0xFF
    b4 = (((c6 & 7) << 5) | c7) & 0xFF
    packed = torch.stack([b0, b1, b2, b3, b4], dim=-1).to(torch.uint8)
    return packed.reshape(*leading, -1)


def pack_model_codes(model: Any, block_size: int = 128) -> dict:
    """Pack all Q3/Q4 layer codes in a model in-place.

    Finds layers with .bits <= 4 and .codes buffer,
    packs codes into nibbles, replaces buffer with packed version.

    Args:
        model: nn.Module with quantized layers that have .bits and .codes attributes.
        block_size: quantization block size (default 128).

    Returns:
        stats: dict with 'layers_packed', 'vram_before', 'vram_after', 'saved_gb'
    """
    layers_packed = 0
    vram_before = 0
    vram_after = 0

    for name, module in model.named_modules():
        # Check if this module is a quantized layer with packable codes
        bits = getattr(module, "bits", None)
        if bits is None or bits > 4:
            continue
        if not hasattr(module, "codes"):
            continue

        codes = module.codes
        if codes is None:
            continue

        # Track VRAM usage before packing
        bytes_before = codes.numel() * codes.element_size()
        vram_before += bytes_before

        # Ensure codes are on the right dtype for packing
        codes_for_pack = codes
        if codes.dtype == torch.int8:
            pass  # Already correct input type
        elif codes.dtype == torch.uint8:
            pass  # Also fine
        else:
            codes_for_pack = codes.to(torch.int8)

        # Pack in half-block order
        packed = pack_codes_half_block(codes_for_pack, block_size)

        # Track VRAM usage after packing
        bytes_after = packed.numel() * packed.element_size()
        vram_after += bytes_after

        # Replace buffer in-place
        # Try deleting the old buffer and registering the packed one
        if hasattr(module, "_buffers") and "codes" in module._buffers:
            del module._buffers["codes"]
            module.register_buffer("codes_packed", packed)
        else:
            # Fallback: direct attribute replacement
            module.codes = packed
            module.codes_packed = packed

        # Mark the module as packed so downstream code knows
        module._codes_packed = True

        layers_packed += 1

    saved_bytes = vram_before - vram_after
    stats = {
        "layers_packed": layers_packed,
        "vram_before": vram_before,
        "vram_after": vram_after,
        "saved_gb": saved_bytes / (1024 ** 3),
    }
    return stats


def verify_packing_roundtrip(codes: torch.Tensor, block_size: int = 128) -> bool:
    """Verify pack -> unpack gives back original codes.

    Args:
        codes: (out_f, in_f_padded) int8 tensor with values 0-15
        block_size: must be even

    Returns:
        True if roundtrip is lossless, False otherwise
    """
    packed = pack_codes_half_block(codes, block_size)
    unpacked = unpack_codes_half_block(packed, block_size)

    # Compare as uint8 to avoid sign issues
    original_u8 = codes.to(torch.uint8)
    unpacked_u8 = unpacked.to(torch.uint8)

    return torch.equal(original_u8, unpacked_u8)


def verify_packing_roundtrip_q5(codes: torch.Tensor, block_size: int = 128) -> bool:
    """Verify Q5 pack -> unpack gives back original codes.

    Args:
        codes: (..., K) int8 tensor with values 0-31
        block_size: must be a multiple of 8

    Returns:
        True if roundtrip is lossless, False otherwise
    """
    packed = pack_codes_q5(codes, block_size)
    unpacked = unpack_codes_q5(packed, block_size)

    # Compare as uint8 to avoid sign issues
    original_u8 = codes.to(torch.uint8)
    unpacked_u8 = unpacked.to(torch.uint8)

    return torch.equal(original_u8, unpacked_u8)


# ===================================================================
# Standalone tests
# ===================================================================

if __name__ == "__main__":
    import sys

    device = "cpu"
    block_size = 128
    half_bk = block_size // 2
    n_levels = 16  # INT4: 0-15
    all_pass = True

    print("=" * 60)
    print("  PolarEngine INT4 Nibble Packing Tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Roundtrip test: pack -> unpack == original
    # ------------------------------------------------------------------
    print("\n  [1] Roundtrip tests (pack -> unpack == original)")

    test_shapes = [
        (4096, 4096),
        (12288, 4096),
        (24576, 4096),
    ]

    for out_f, in_f in test_shapes:
        # Pad to multiple of block_size
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8)

        passed = verify_packing_roundtrip(codes, block_size)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"      [{status}] shape=({out_f:>5}, {in_f_padded:>5})")

    # ------------------------------------------------------------------
    # 2. Shape correctness: packed is exactly half the columns
    # ------------------------------------------------------------------
    print("\n  [2] Shape correctness (packed dim = in_f_padded // 2)")

    for out_f, in_f in test_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8)
        packed = pack_codes_half_block(codes, block_size)

        expected_shape = (out_f, in_f_padded // 2)
        shape_ok = packed.shape == expected_shape
        dtype_ok = packed.dtype == torch.uint8
        status = "PASS" if (shape_ok and dtype_ok) else "FAIL"
        if not (shape_ok and dtype_ok):
            all_pass = False
        print(f"      [{status}] ({out_f:>5}, {in_f_padded:>5}) -> packed {packed.shape}, dtype={packed.dtype}")

    # ------------------------------------------------------------------
    # 3. Edge cases: codes with values 0 and 15, boundary values
    # ------------------------------------------------------------------
    print("\n  [3] Edge cases")

    # All zeros
    codes_zeros = torch.zeros(256, 1024, dtype=torch.int8)
    passed = verify_packing_roundtrip(codes_zeros, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All zeros")

    # All fifteens
    codes_max = torch.full((256, 1024), 15, dtype=torch.int8)
    passed = verify_packing_roundtrip(codes_max, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All 15s")

    # Alternating 0 and 15
    codes_alt = torch.zeros(256, 1024, dtype=torch.int8)
    codes_alt[:, ::2] = 15
    passed = verify_packing_roundtrip(codes_alt, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] Alternating 0/15")

    # First half = 0, second half = 15 within each block (tests nibble placement)
    codes_half = torch.zeros(128, 512, dtype=torch.int8)
    codes_blocked_h = codes_half.view(128, -1, block_size)
    codes_blocked_h[:, :, :half_bk] = 0
    codes_blocked_h[:, :, half_bk:] = 15
    packed_half = pack_codes_half_block(codes_half, block_size)
    packed_view = packed_half.view(128, -1, half_bk)
    # Low nibble should be 0, high nibble should be 15 -> byte = 0xF0 = 240
    expected_byte = 15 << 4 | 0  # 240
    nibble_ok = (packed_view == expected_byte).all().item()
    status = "PASS" if nibble_ok else "FAIL"
    if not nibble_ok:
        all_pass = False
    print(f"      [{status}] Half-block nibble placement (low=0, high=15 -> byte=0x{expected_byte:02X})")

    # Reverse: first half = 15, second half = 0 within each block
    codes_rev = torch.zeros(128, 512, dtype=torch.int8)
    codes_blocked_r = codes_rev.view(128, -1, block_size)
    codes_blocked_r[:, :, :half_bk] = 15
    codes_blocked_r[:, :, half_bk:] = 0
    packed_rev = pack_codes_half_block(codes_rev, block_size)
    packed_rev_view = packed_rev.view(128, -1, half_bk)
    expected_byte_rev = 0 << 4 | 15  # 0x0F = 15
    nibble_rev_ok = (packed_rev_view == expected_byte_rev).all().item()
    status = "PASS" if nibble_rev_ok else "FAIL"
    if not nibble_rev_ok:
        all_pass = False
    print(f"      [{status}] Reverse nibble placement (low=15, high=0 -> byte=0x{expected_byte_rev:02X})")

    # Non-standard block sizes (still even)
    for bs in [32, 64, 256]:
        hbk = bs // 2
        in_f_pad = bs * 4  # 4 blocks
        codes_bs = torch.randint(0, n_levels, (64, in_f_pad), dtype=torch.int8)
        passed = verify_packing_roundtrip(codes_bs, bs)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"      [{status}] block_size={bs}")

    # ------------------------------------------------------------------
    # 4. Consistency with polar_gemv.py pack_codes_int4
    # ------------------------------------------------------------------
    print("\n  [4] Consistency with polar_gemv.pack_codes_int4")

    try:
        from polarengine_vllm.kernels.polar_gemv import pack_codes_int4

        for out_f, in_f in [(512, 1024), (2048, 4096)]:
            in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
            codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8)
            packed_ref = pack_codes_int4(codes, block_size)
            packed_new = pack_codes_half_block(codes, block_size)
            match = torch.equal(packed_ref, packed_new)
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False
            print(f"      [{status}] ({out_f:>5}, {in_f_padded:>5}) matches polar_gemv.pack_codes_int4")
    except ImportError:
        print("      [SKIP] Could not import polar_gemv.pack_codes_int4 (not in path)")

    # ------------------------------------------------------------------
    # 5. Benchmark: packing speed
    # ------------------------------------------------------------------
    print("\n  [5] Packing speed benchmark (CPU)")

    bench_shapes = [
        (4096, 4096),
        (12288, 4096),
        (24576, 4096),
    ]

    for out_f, in_f in bench_shapes:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes = torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8)

        # Warmup
        _ = pack_codes_half_block(codes, block_size)
        _ = unpack_codes_half_block(_, block_size)

        # Time packing
        n_iters = 10
        t0 = time.perf_counter()
        for _ in range(n_iters):
            packed = pack_codes_half_block(codes, block_size)
        t_pack = (time.perf_counter() - t0) / n_iters

        # Time unpacking
        t0 = time.perf_counter()
        for _ in range(n_iters):
            unpacked = unpack_codes_half_block(packed, block_size)
        t_unpack = (time.perf_counter() - t0) / n_iters

        size_mb = codes.numel() / (1024 * 1024)
        print(f"      ({out_f:>5}, {in_f_padded:>5}) [{size_mb:>6.1f} MB]  "
              f"pack={t_pack*1000:>6.1f}ms  unpack={t_unpack*1000:>6.1f}ms")

    # ==================================================================
    # Q5 BIT-PACKING TESTS
    # ==================================================================
    print()
    print("=" * 60)
    print("  PolarEngine Q5 Bit-Packing Tests")
    print("=" * 60)

    n_levels_q5 = 32  # Q5: 0-31

    # ------------------------------------------------------------------
    # Q5-1. Roundtrip test: pack -> unpack == original
    # ------------------------------------------------------------------
    print("\n  [Q5-1] Roundtrip tests (pack -> unpack == original)")

    test_shapes_q5 = [
        (1, 128),
        (64, 2688),
        (2688, 1856),
        (4096, 4096),
        (12288, 4096),
    ]

    for out_f, in_f in test_shapes_q5:
        # Pad to multiple of block_size
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes_q5 = torch.randint(0, n_levels_q5, (out_f, in_f_padded), dtype=torch.int8)

        passed = verify_packing_roundtrip_q5(codes_q5, block_size)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"      [{status}] shape=({out_f:>5}, {in_f_padded:>5})")

    # ------------------------------------------------------------------
    # Q5-2. Size verification: output is exactly 5/8 of input
    # ------------------------------------------------------------------
    print("\n  [Q5-2] Size correctness (packed dim = K * 5 / 8)")

    for out_f, in_f in test_shapes_q5:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes_q5 = torch.randint(0, n_levels_q5, (out_f, in_f_padded), dtype=torch.int8)
        packed_q5 = pack_codes_q5(codes_q5, block_size)

        expected_packed_cols = in_f_padded * 5 // 8
        expected_shape = (out_f, expected_packed_cols)
        shape_ok = packed_q5.shape == expected_shape
        dtype_ok = packed_q5.dtype == torch.uint8
        status = "PASS" if (shape_ok and dtype_ok) else "FAIL"
        if not (shape_ok and dtype_ok):
            all_pass = False
        print(f"      [{status}] ({out_f:>5}, {in_f_padded:>5}) -> packed {packed_q5.shape}, "
              f"expected {expected_shape}, dtype={packed_q5.dtype}")

    # ------------------------------------------------------------------
    # Q5-3. Edge cases: all zeros, all 31s, boundary values
    # ------------------------------------------------------------------
    print("\n  [Q5-3] Edge cases")

    # All zeros
    codes_q5_zeros = torch.zeros(256, 1024, dtype=torch.int8)
    passed = verify_packing_roundtrip_q5(codes_q5_zeros, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All zeros")

    # All 31s (max Q5 value)
    codes_q5_max = torch.full((256, 1024), 31, dtype=torch.int8)
    passed = verify_packing_roundtrip_q5(codes_q5_max, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All 31s")

    # Value 1 everywhere
    codes_q5_one = torch.ones(256, 1024, dtype=torch.int8)
    passed = verify_packing_roundtrip_q5(codes_q5_one, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All 1s")

    # Value 30 everywhere
    codes_q5_30 = torch.full((256, 1024), 30, dtype=torch.int8)
    passed = verify_packing_roundtrip_q5(codes_q5_30, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] All 30s")

    # Alternating 0 and 31
    codes_q5_alt = torch.zeros(256, 1024, dtype=torch.int8)
    codes_q5_alt[:, ::2] = 31
    passed = verify_packing_roundtrip_q5(codes_q5_alt, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] Alternating 0/31")

    # Sequential pattern: 0,1,2,...,31,0,1,... (covers all values)
    codes_q5_seq = torch.zeros(128, 1024, dtype=torch.int8)
    for i in range(1024):
        codes_q5_seq[:, i] = i % 32
    passed = verify_packing_roundtrip_q5(codes_q5_seq, block_size)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"      [{status}] Sequential 0-31 pattern")

    # Non-standard block sizes (still multiple of 8)
    for bs_q5 in [8, 16, 32, 64, 256]:
        in_f_pad_q5 = bs_q5 * 4  # 4 blocks
        codes_bs_q5 = torch.randint(0, n_levels_q5, (64, in_f_pad_q5), dtype=torch.int8)
        passed = verify_packing_roundtrip_q5(codes_bs_q5, bs_q5)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"      [{status}] block_size={bs_q5}")

    # ------------------------------------------------------------------
    # Q5-4. Compression ratio verification
    # ------------------------------------------------------------------
    print("\n  [Q5-4] Compression ratio (5/8 = 62.5% of original)")

    codes_cr = torch.randint(0, n_levels_q5, (4096, 4096), dtype=torch.int8)
    packed_cr = pack_codes_q5(codes_cr, block_size)
    original_bytes = codes_cr.numel() * codes_cr.element_size()
    packed_bytes = packed_cr.numel() * packed_cr.element_size()
    ratio = packed_bytes / original_bytes
    ratio_ok = abs(ratio - 5.0 / 8.0) < 1e-6
    status = "PASS" if ratio_ok else "FAIL"
    if not ratio_ok:
        all_pass = False
    print(f"      [{status}] Ratio: {ratio:.6f} (expected {5/8:.6f}), "
          f"{original_bytes / 1024**2:.1f} MB -> {packed_bytes / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Q5-5. Benchmark: packing speed
    # ------------------------------------------------------------------
    print("\n  [Q5-5] Q5 Packing speed benchmark (CPU)")

    bench_shapes_q5 = [
        (4096, 4096),
        (12288, 4096),
    ]

    for out_f, in_f in bench_shapes_q5:
        in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
        codes_bench = torch.randint(0, n_levels_q5, (out_f, in_f_padded), dtype=torch.int8)

        # Warmup
        _ = pack_codes_q5(codes_bench, block_size)
        _ = unpack_codes_q5(_, block_size)

        # Time packing
        n_iters = 10
        t0 = time.perf_counter()
        for _ in range(n_iters):
            packed_bench = pack_codes_q5(codes_bench, block_size)
        t_pack = (time.perf_counter() - t0) / n_iters

        # Time unpacking
        t0 = time.perf_counter()
        for _ in range(n_iters):
            unpacked_bench = unpack_codes_q5(packed_bench, block_size)
        t_unpack = (time.perf_counter() - t0) / n_iters

        size_mb = codes_bench.numel() / (1024 * 1024)
        print(f"      ({out_f:>5}, {in_f_padded:>5}) [{size_mb:>6.1f} MB]  "
              f"pack={t_pack*1000:>6.1f}ms  unpack={t_unpack*1000:>6.1f}ms")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if all_pass:
        print("All packing tests passed! (INT4 + Q5)")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
