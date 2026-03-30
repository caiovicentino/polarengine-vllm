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

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if all_pass:
        print("All packing tests passed!")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
