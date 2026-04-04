"""Fused Triton kernel for PolarLinear nearest-centroid quantization.

Eliminates the massive intermediate 4D broadcast tensor that the naive
PyTorch implementation creates during training:

    # OLD (PyTorch broadcast -- ~367 MB per layer for 4096x4096 Q5):
    idx = (codes.unsqueeze(-1) - ct.view(1,1,1,-1)).abs().argmin(-1)
    hard = ct[idx]

    # NEW (Triton fused -- zero intermediate memory):
    hard = polar_quantize(codes, centroids)

The kernel computes, for each element in ``codes``, the nearest centroid
from the ``centroids`` table entirely in-register, writing only the final
centroid value to ``hard``. No 4D diff tensor is ever materialized.

After the kernel, the STE (Straight-Through Estimator) gradient is handled
by standard PyTorch autograd::

    q = codes + (hard - codes).detach()

Supports Q2 (4 levels), Q3 (8), Q4 (16), Q5 (32), Q6 (64).
Supports float32 and bfloat16 input codes.

Usage::

    from polarengine_vllm.kernels.polar_quantize import polar_quantize

    # codes: (out_features, n_blocks, block_size) -- float32 or bf16
    # centroids: (n_levels,) -- float32
    hard = polar_quantize(codes, centroids)
    q = codes + (hard - codes).detach()  # STE
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ===================================================================
# Triton Kernel: fused nearest-centroid lookup
# ===================================================================

if HAS_TRITON:

    @triton.jit
    def _polar_quantize_kernel(
        # --- Pointers ---
        codes_ptr,       # input codes, shape (N_total,), contiguous flat
        centroids_ptr,   # centroid table, shape (n_levels,), float32
        hard_ptr,        # output, same shape as codes (N_total,)
        # --- Dimensions ---
        N_total,         # total number of elements (out_f * n_blocks * block_size)
        n_levels: tl.constexpr,  # number of centroids (4, 8, 16, 32, 64)
        # --- Tile size ---
        BLOCK_SIZE: tl.constexpr,  # elements per program (autotuned)
    ):
        """For each element in codes, find nearest centroid in-register.

        Strategy: each program processes BLOCK_SIZE elements. For each
        element, we iterate over all n_levels centroids (max 64 for Q6)
        and keep track of the minimum distance and corresponding centroid
        value. Since n_levels is a compile-time constant (constexpr),
        the centroid loop is fully unrolled by the compiler.

        The centroids table (max 64 floats = 256 bytes) fits entirely
        in registers, so there is zero global memory traffic for the
        centroid lookup -- only the codes load and hard store touch DRAM.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_total

        # Load codes for this tile
        vals = tl.load(codes_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Load ALL centroids into registers (n_levels is constexpr, so this
        # is a compile-time-sized register array -- no dynamic indexing).
        # For Q5 (32 centroids), this is 32 x 4 bytes = 128 bytes of registers.

        # Initialize best distance and best centroid value
        best_dist = tl.full([BLOCK_SIZE], value=float("inf"), dtype=tl.float32)
        best_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Iterate over centroids. Since n_levels is constexpr, Triton
        # unrolls this loop completely. Each iteration is just:
        #   dist = |val - centroid|; if dist < best: update
        for c in range(n_levels):
            cent = tl.load(centroids_ptr + c)
            dist = tl.abs(vals - cent)
            # Branchless update: where dist < best_dist, replace
            improve = dist < best_dist
            best_dist = tl.where(improve, dist, best_dist)
            best_val = tl.where(improve, cent, best_val)

        # Cast back to the original dtype is handled by the wrapper
        # (we always compute in float32 for accuracy)
        tl.store(hard_ptr + offs, best_val, mask=mask)


    # ---------------------------------------------------------------
    # Specializations with explicit n_levels for each bit width.
    # Having n_levels as constexpr enables full loop unrolling.
    # We use a dispatch wrapper rather than separate @jit functions
    # because Triton handles constexpr specialization automatically
    # when the value is passed as a Python int at launch time.
    # ---------------------------------------------------------------


# ===================================================================
# Autograd Function wrapper
# ===================================================================

class _PolarQuantizeFn(torch.autograd.Function):
    """Forward-only autograd wrapper for the Triton nearest-centroid kernel.

    Backward is NOT needed here because the caller applies STE in PyTorch::

        hard = polar_quantize(codes, centroids)
        q = codes + (hard - codes).detach()  # .detach() blocks grad through hard

    So ``hard`` never requires grad propagation through the kernel itself.
    """

    @staticmethod
    def forward(ctx, codes: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        # Validate inputs
        assert codes.is_cuda, "codes must be on CUDA"
        assert centroids.is_cuda, "centroids must be on CUDA"
        assert centroids.ndim == 1, f"centroids must be 1-D, got {centroids.ndim}-D"

        n_levels = centroids.shape[0]
        assert n_levels in (4, 8, 16, 32, 64), (
            f"n_levels must be 4/8/16/32/64 (Q2-Q6), got {n_levels}"
        )

        orig_shape = codes.shape
        orig_dtype = codes.dtype

        # Flatten codes for the 1-D kernel
        codes_flat = codes.contiguous().view(-1)
        N_total = codes_flat.numel()

        # Ensure centroids are float32 and contiguous
        centroids_f32 = centroids.float().contiguous()

        # Allocate output (always float32 for kernel accuracy)
        hard_flat = torch.empty(N_total, dtype=torch.float32, device=codes.device)

        # Grid: one program per BLOCK_SIZE elements
        # BLOCK_SIZE=1024 is a good default for this memory-bound kernel.
        # For very small tensors we clamp down.
        BLOCK_SIZE = 1024
        grid = ((N_total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _polar_quantize_kernel[grid](
            codes_flat if codes_flat.dtype == torch.float32 else codes_flat.float(),
            centroids_f32,
            hard_flat,
            N_total,
            n_levels=n_levels,  # constexpr -- enables full unrolling
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape and cast back to original dtype
        hard = hard_flat.view(orig_shape)
        if orig_dtype != torch.float32:
            hard = hard.to(orig_dtype)

        return hard

    @staticmethod
    def backward(ctx, grad_output):
        # Not needed -- STE is handled externally:
        #   q = codes + (hard - codes).detach()
        # grad flows through codes, not through hard.
        return None, None


# ===================================================================
# Public API
# ===================================================================

def polar_quantize(codes: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Fused nearest-centroid quantization -- zero intermediate memory.

    For each element in ``codes``, finds the nearest value in ``centroids``
    and writes it to the output. The entire computation happens in GPU
    registers with no intermediate tensor allocation.

    Args:
        codes:      Tensor of shape ``(out_features, n_blocks, block_size)``
                    or any shape. float32 or bfloat16.
        centroids:  1-D tensor of shape ``(n_levels,)`` where n_levels is
                    one of 4 (Q2), 8 (Q3), 16 (Q4), 32 (Q5), 64 (Q6).

    Returns:
        ``hard`` tensor of same shape and dtype as ``codes``, containing
        the nearest centroid value for each element.

    Example::

        hard = polar_quantize(codes, centroids)
        q = codes + (hard - codes).detach()  # STE gradient
    """
    if not HAS_TRITON or not codes.is_cuda:
        # Fallback: PyTorch broadcast (original slow path)
        return _polar_quantize_pytorch(codes, centroids)

    return _PolarQuantizeFn.apply(codes, centroids)


def _polar_quantize_pytorch(
    codes: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """PyTorch fallback for CPU or when Triton is unavailable.

    This IS the slow broadcast approach -- kept as a reference and for
    CPU-only testing. Uses chunking to avoid OOM on very large tensors.
    """
    orig_shape = codes.shape
    codes_flat = codes.reshape(-1)
    n_levels = centroids.shape[0]

    # Chunk to limit peak memory: process 1M elements at a time
    CHUNK = 1_000_000
    hard_flat = torch.empty_like(codes_flat)

    for i in range(0, codes_flat.numel(), CHUNK):
        j = min(i + CHUNK, codes_flat.numel())
        chunk = codes_flat[i:j]
        # Broadcast: (chunk_size, 1) vs (1, n_levels)
        idx = (chunk.unsqueeze(-1) - centroids.unsqueeze(0)).abs().argmin(-1)
        hard_flat[i:j] = centroids[idx]

    return hard_flat.view(orig_shape)


# ===================================================================
# Benchmark
# ===================================================================

def benchmark(
    out_features: int = 4096,
    n_blocks: int = 32,
    block_size: int = 128,
    bits: int = 5,
    n_warmup: int = 10,
    n_iters: int = 100,
    device: str = "cuda",
) -> dict:
    """Compare Triton fused kernel vs PyTorch broadcast approach.

    Args:
        out_features: Number of output features (rows).
        n_blocks:     Number of blocks per row.
        block_size:   Elements per block.
        bits:         Quantization bits (2-6).
        n_warmup:     Warmup iterations.
        n_iters:      Timed iterations.
        device:       CUDA device string.

    Returns:
        Dict with timing results and memory comparison.
    """
    import time

    n_levels = 1 << bits
    shape = (out_features, n_blocks, block_size)
    total_elements = out_features * n_blocks * block_size

    # Simulate codes as float32 values in the range of centroids
    codes = torch.randn(*shape, device=device, dtype=torch.float32)
    centroids = torch.linspace(-3.0, 3.0, n_levels, device=device, dtype=torch.float32)

    results = {}

    # --- PyTorch broadcast baseline ---
    def pytorch_broadcast():
        ct_view = centroids.view(1, 1, 1, -1)
        idx = (codes.unsqueeze(-1) - ct_view).abs().argmin(-1)
        return centroids[idx]

    # Warmup
    for _ in range(n_warmup):
        _ = pytorch_broadcast()
    torch.cuda.synchronize()

    # Measure PyTorch broadcast
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        hard_pt = pytorch_broadcast()
    torch.cuda.synchronize()
    t_pytorch = (time.perf_counter() - t0) / n_iters * 1000  # ms
    mem_peak_pt = torch.cuda.max_memory_allocated() - mem_before

    # --- Triton fused kernel ---
    if not HAS_TRITON:
        print("Triton not available, skipping Triton benchmark.")
        results["pytorch_ms"] = t_pytorch
        results["pytorch_peak_MB"] = mem_peak_pt / 1024**2
        return results

    # Warmup
    for _ in range(n_warmup):
        _ = polar_quantize(codes, centroids)
    torch.cuda.synchronize()

    # Measure Triton fused
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        hard_tr = polar_quantize(codes, centroids)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / n_iters * 1000  # ms
    mem_peak_tr = torch.cuda.max_memory_allocated() - mem_before

    # --- Correctness check ---
    hard_pt_ref = pytorch_broadcast()
    hard_tr_ref = polar_quantize(codes, centroids)
    max_diff = (hard_pt_ref - hard_tr_ref).abs().max().item()
    match = max_diff == 0.0

    # --- Theoretical intermediate memory ---
    # PyTorch broadcast creates: (out_f, n_blocks, block_size, n_levels) float32
    intermediate_bytes = total_elements * n_levels * 4  # float32

    # --- Results ---
    results = {
        "shape": shape,
        "bits": bits,
        "n_levels": n_levels,
        "total_elements": total_elements,
        "pytorch_ms": t_pytorch,
        "triton_ms": t_triton,
        "speedup": t_pytorch / t_triton if t_triton > 0 else float("inf"),
        "pytorch_peak_MB": mem_peak_pt / 1024**2,
        "triton_peak_MB": mem_peak_tr / 1024**2,
        "theoretical_intermediate_MB": intermediate_bytes / 1024**2,
        "memory_saved_MB": (mem_peak_pt - mem_peak_tr) / 1024**2,
        "exact_match": match,
        "max_diff": max_diff,
    }

    return results


def print_benchmark(
    configs: list[dict] | None = None,
    device: str = "cuda",
):
    """Run and print benchmark results for multiple configurations.

    Args:
        configs: List of config dicts (keys: out_features, n_blocks, block_size, bits).
                 If None, uses a representative set of PolarQuant layer sizes.
        device:  CUDA device string.
    """
    if configs is None:
        configs = [
            # Typical Qwen-9B / Llama-8B layer sizes with block_size=128
            {"out_features": 4096, "n_blocks": 32, "block_size": 128, "bits": 5},
            {"out_features": 4096, "n_blocks": 32, "block_size": 128, "bits": 3},
            {"out_features": 11008, "n_blocks": 32, "block_size": 128, "bits": 3},
            {"out_features": 4096, "n_blocks": 32, "block_size": 128, "bits": 4},
            {"out_features": 4096, "n_blocks": 32, "block_size": 128, "bits": 6},
            {"out_features": 4096, "n_blocks": 32, "block_size": 128, "bits": 2},
            # Large layer (down_proj-like)
            {"out_features": 11008, "n_blocks": 32, "block_size": 128, "bits": 4},
        ]

    print("=" * 90)
    print("PolarLinear Fused Quantization Kernel Benchmark")
    print("  Triton fused (zero intermediate) vs PyTorch broadcast (4D intermediate)")
    print("=" * 90)

    for cfg in configs:
        r = benchmark(device=device, **cfg)
        print(
            f"\n  Q{r['bits']} ({r['n_levels']:2d} levels) | "
            f"shape={r['shape']} | {r['total_elements']/1e6:.1f}M elements"
        )
        print(
            f"    PyTorch broadcast:  {r['pytorch_ms']:7.3f} ms | "
            f"peak {r.get('pytorch_peak_MB', 0):8.1f} MB"
        )
        if "triton_ms" in r:
            print(
                f"    Triton fused:       {r['triton_ms']:7.3f} ms | "
                f"peak {r.get('triton_peak_MB', 0):8.1f} MB"
            )
            print(
                f"    Speedup: {r['speedup']:.2f}x | "
                f"Memory saved: {r.get('memory_saved_MB', 0):.1f} MB | "
                f"Theoretical intermediate: {r['theoretical_intermediate_MB']:.1f} MB"
            )
            status = "PASS" if r["exact_match"] else f"FAIL (max_diff={r['max_diff']:.6e})"
            print(f"    Correctness: {status}")

    print("\n" + "=" * 90)


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    print_benchmark()
