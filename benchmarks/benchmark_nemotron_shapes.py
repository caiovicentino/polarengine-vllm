"""Benchmark PolarEngine kernels on Nemotron-Ultra-253B layer shapes.

Nemotron hidden_size=2688, block_size=128 -> 21 K-blocks per row.
This script benchmarks all Nemotron layer shapes against:
  - Regular GEMV (baseline)
  - SplitK GEMV with SK=2, 4, 8
  - GEMM with batch=1

Each kernel is timed over 100 iterations (after warmup) and reported in
microseconds.

Usage:
    python -m benchmarks.benchmark_nemotron_shapes
    # or
    python benchmarks/benchmark_nemotron_shapes.py
"""

from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

try:
    import triton
except ImportError:
    print("ERROR: Triton is required to run benchmarks.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: CUDA is required to run benchmarks.")
    sys.exit(1)

from polarengine_vllm.kernels.polar_gemv import polar_gemv, polar_gemv_packed, pack_codes_int4
from polarengine_vllm.kernels.polar_gemv_splitk import polar_gemv_splitk, polar_gemv_packed_splitk
from polarengine_vllm.kernels.polar_gemm import polar_gemm, polar_gemm_packed


# ===================================================================
# Configuration
# ===================================================================

DEVICE = "cuda"
BLOCK_SIZE = 128
BITS = 4
N_LEVELS = 1 << BITS
N_WARMUP = 50
N_ITERS = 100

# Nemotron-Ultra-253B layer shapes: (out_features, in_features)
NEMOTRON_SHAPES = {
    # Attention projections
    "attn (4096x2688)":  (4096, 2688),
    "attn (256x2688)":   (256, 2688),
    "attn (2688x4096)":  (2688, 4096),
    # MoE expert projections
    "moe  (1856x2688)":  (1856, 2688),
    "moe  (2688x1856)":  (2688, 1856),
    # Shared expert projections
    "sexp (3712x2688)":  (3712, 2688),
    "sexp (2688x3712)":  (2688, 3712),
}


# ===================================================================
# Benchmark utilities
# ===================================================================

def build_hadamard(n: int) -> torch.Tensor:
    """Build n x n Hadamard matrix (normalized)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h = build_hadamard(n // 2)
    return torch.cat([
        torch.cat([h, h], 1),
        torch.cat([h, -h], 1),
    ], 0) / math.sqrt(2)


def time_fn(fn, n_warmup: int, n_iters: int) -> float:
    """Time a function, return mean time in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / n_iters) * 1e6  # microseconds


def setup_layer(out_f: int, in_f: int):
    """Create synthetic quantized layer data for benchmarking."""
    in_f_padded = ((in_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    n_blocks = in_f_padded // BLOCK_SIZE
    half_bk = BLOCK_SIZE // 2
    in_f_half = n_blocks * half_bk

    torch.manual_seed(42)
    centroids = torch.linspace(-1.5, 1.5, N_LEVELS, dtype=torch.float32, device=DEVICE)
    ct_scaled = centroids / math.sqrt(BLOCK_SIZE)

    codes = torch.randint(0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8, device=DEVICE)
    norms = (torch.randn(out_f * n_blocks, dtype=torch.float32, device=DEVICE).abs() + 0.1)
    norms_2d = norms.view(out_f, n_blocks)

    # Packed codes for INT4 variants
    packed_codes = pack_codes_int4(codes, BLOCK_SIZE)

    # Input vector (batch=1)
    x = torch.randn(in_f, dtype=torch.float32, device=DEVICE)
    x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x

    H = build_hadamard(BLOCK_SIZE).to(DEVICE)
    x_transformed = torch.matmul(x_padded.view(-1, BLOCK_SIZE), H).reshape(-1)
    # For GEMM: need (1, in_f_padded)
    x_transformed_2d = x_transformed.unsqueeze(0)

    return {
        "codes": codes,
        "packed_codes": packed_codes,
        "norms": norms,
        "norms_2d": norms_2d,
        "ct_scaled": ct_scaled,
        "x_transformed": x_transformed,
        "x_transformed_2d": x_transformed_2d,
        "out_f": out_f,
        "in_f_padded": in_f_padded,
        "n_blocks": n_blocks,
        "in_f_half": in_f_half,
    }


# ===================================================================
# Main benchmark
# ===================================================================

def run_benchmark():
    print("=" * 90)
    print("  PolarEngine Nemotron-Ultra-253B Shape Benchmark")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  block_size={BLOCK_SIZE}, bits={BITS}, warmup={N_WARMUP}, iters={N_ITERS}")
    print("=" * 90)

    # Table header
    header = (
        f"{'Layer':<22} | {'out_f':>5} x {'in_f':>5} | {'blocks':>6} | "
        f"{'GEMV':>8} | {'SK=2':>8} | {'SK=4':>8} | {'SK=8':>8} | {'GEMM b1':>8}"
    )
    separator = "-" * len(header)
    print()
    print("  Unpacked (int8 codes) - times in microseconds")
    print(f"  {separator}")
    print(f"  {header}")
    print(f"  {separator}")

    # --- Unpacked benchmarks ---
    for name, (out_f, in_f) in NEMOTRON_SHAPES.items():
        d = setup_layer(out_f, in_f)

        # Regular GEMV
        t_gemv = time_fn(
            lambda: polar_gemv(
                d["codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                d["out_f"], d["in_f_padded"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        # SplitK variants
        splitk_times = {}
        for sk in [2, 4, 8]:
            t_sk = time_fn(
                lambda sk=sk: polar_gemv_splitk(
                    d["codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                    d["out_f"], d["in_f_padded"], d["n_blocks"], BLOCK_SIZE,
                    split_k=sk,
                ),
                N_WARMUP, N_ITERS,
            )
            splitk_times[sk] = t_sk

        # GEMM with batch=1
        t_gemm = time_fn(
            lambda: polar_gemm(
                d["codes"], d["x_transformed_2d"], d["norms_2d"], d["ct_scaled"],
                d["out_f"], d["in_f_padded"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        print(
            f"  {name:<22} | {out_f:>5} x {in_f:>5} | {d['n_blocks']:>6} | "
            f"{t_gemv:>7.1f}u | {splitk_times[2]:>7.1f}u | "
            f"{splitk_times[4]:>7.1f}u | {splitk_times[8]:>7.1f}u | "
            f"{t_gemm:>7.1f}u"
        )

    print(f"  {separator}")

    # --- Packed INT4 benchmarks ---
    print()
    print("  Packed INT4 (nibble codes) - times in microseconds")
    print(f"  {separator}")
    print(f"  {header}")
    print(f"  {separator}")

    for name, (out_f, in_f) in NEMOTRON_SHAPES.items():
        d = setup_layer(out_f, in_f)

        # Regular GEMV packed
        t_gemv = time_fn(
            lambda: polar_gemv_packed(
                d["packed_codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        # SplitK packed variants
        splitk_times = {}
        for sk in [2, 4, 8]:
            t_sk = time_fn(
                lambda sk=sk: polar_gemv_packed_splitk(
                    d["packed_codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                    d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
                    split_k=sk,
                ),
                N_WARMUP, N_ITERS,
            )
            splitk_times[sk] = t_sk

        # GEMM packed with batch=1
        t_gemm = time_fn(
            lambda: polar_gemm_packed(
                d["packed_codes"], d["x_transformed_2d"], d["norms_2d"], d["ct_scaled"],
                d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        print(
            f"  {name:<22} | {out_f:>5} x {in_f:>5} | {d['n_blocks']:>6} | "
            f"{t_gemv:>7.1f}u | {splitk_times[2]:>7.1f}u | "
            f"{splitk_times[4]:>7.1f}u | {splitk_times[8]:>7.1f}u | "
            f"{t_gemm:>7.1f}u"
        )

    print(f"  {separator}")

    # --- Summary: best kernel per shape ---
    print()
    print("  Summary: best kernel per shape (packed INT4)")
    print(f"  {'-' * 60}")

    for name, (out_f, in_f) in NEMOTRON_SHAPES.items():
        d = setup_layer(out_f, in_f)

        results = {}

        results["GEMV"] = time_fn(
            lambda: polar_gemv_packed(
                d["packed_codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        for sk in [2, 4, 8]:
            results[f"SK={sk}"] = time_fn(
                lambda sk=sk: polar_gemv_packed_splitk(
                    d["packed_codes"], d["x_transformed"], d["norms"], d["ct_scaled"],
                    d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
                    split_k=sk,
                ),
                N_WARMUP, N_ITERS,
            )

        results["GEMM"] = time_fn(
            lambda: polar_gemm_packed(
                d["packed_codes"], d["x_transformed_2d"], d["norms_2d"], d["ct_scaled"],
                d["out_f"], d["in_f_half"], d["n_blocks"], BLOCK_SIZE,
            ),
            N_WARMUP, N_ITERS,
        )

        best_name = min(results, key=results.get)
        best_time = results[best_name]
        gemv_time = results["GEMV"]
        speedup = gemv_time / best_time if best_time > 0 else float("inf")

        print(
            f"  {name:<22} -> best: {best_name:<6} "
            f"({best_time:>7.1f}us, {speedup:.2f}x vs GEMV)"
        )

    print()
    print("  Done.")


if __name__ == "__main__":
    run_benchmark()
