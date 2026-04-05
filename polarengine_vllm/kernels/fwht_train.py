"""Training-compatible Fast Walsh-Hadamard Transform (FWHT) with Triton kernel.

Provides a Triton-accelerated FWHT that supports both forward and backward
passes via torch.autograd.Function. The FWHT is self-inverse (orthogonal),
so the backward pass is simply another FWHT applied to the incoming gradient.

Key properties of the normalized Walsh-Hadamard matrix H:
    H @ H = I          (self-inverse / involution)
    H^T = H             (symmetric)
    d/dx (x @ H) = H^T = H   (gradient is just H again)

Therefore: backward(grad_output) = grad_output @ H = FWHT(grad_output).

Three implementations:
    1. fwht_triton   -- Triton kernel, fastest, supports autograd
    2. fwht_matmul   -- cuBLAS GEMM fallback with autograd
    3. fwht_train    -- convenience alias for fwht_triton

Works on 2D tensors (batch, n) where n is a power of 2.
Supports BF16 and FP32.
"""

from __future__ import annotations

import math
import time
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =====================================================================
# Hadamard matrix builder (cached, for matmul fallback + validation)
# =====================================================================

_hadamard_cache: dict[tuple, torch.Tensor] = {}


def build_hadamard(n: int, device: torch.device = None) -> torch.Tensor:
    """Build normalized Walsh-Hadamard matrix of size n (power of 2).

    H @ H = I (self-inverse, orthogonal).
    Cached globally per (size, device).
    """
    assert n >= 1 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"

    dev_key = str(device) if device is not None else "cpu"
    cache_key = (n, dev_key)

    if cache_key not in _hadamard_cache:
        cpu_key = (n, "cpu")
        if cpu_key not in _hadamard_cache:
            if n == 1:
                H = torch.tensor([[1.0]])
            else:
                h = build_hadamard(n // 2)
                H = torch.cat([
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ], dim=0) / math.sqrt(2)
            _hadamard_cache[cpu_key] = H

        if device is not None:
            _hadamard_cache[cache_key] = _hadamard_cache[cpu_key].to(device)
        else:
            return _hadamard_cache[cpu_key]

    return _hadamard_cache[cache_key]


# =====================================================================
# Triton FWHT kernel (single launch, compile-time unrolled butterfly)
# =====================================================================

if HAS_TRITON:

    def _make_fwht_single_kernel(log2_n: int):
        """Factory: build a single-kernel FWHT for a specific n = 2^log2_n.

        Triton requires loop bounds to be constexpr. We work around this
        by generating a specialized kernel for each n via closure + jit.
        This is cached by Triton's JIT cache keyed on the function object.
        """
        n = 1 << log2_n

        @triton.jit
        def _kernel(
            buf_ptr,
            stride_row,
            BLOCK_N: tl.constexpr,
            LOG2_N: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK_N)
            ptrs = buf_ptr + row * stride_row + cols
            x = tl.load(ptrs).to(tl.float32)

            for stage in tl.static_range(LOG2_N):
                half = 1 << stage
                pair_cols = cols ^ half
                pair_ptrs = buf_ptr + row * stride_row + pair_cols
                x_pair = tl.load(pair_ptrs).to(tl.float32)
                is_low = (cols & half) == 0
                new_x = tl.where(is_low, x + x_pair, x_pair - x)
                # Store so pair reads in next stage see updated values
                tl.store(ptrs, new_x)
                # Reload for next stage (ensures coherence)
                x = tl.load(ptrs).to(tl.float32)

            # Final normalization: 1/sqrt(n)
            x = x * (1.0 / tl.sqrt(float(1 << LOG2_N)))
            tl.store(ptrs, x)

        return _kernel

    # Cache of compiled single-kernel FWHT per log2_n
    _fwht_kernel_cache: dict[int, object] = {}

    def _triton_fwht_single(x: torch.Tensor) -> torch.Tensor:
        """Single-kernel Triton FWHT: all butterfly stages in one launch.

        Works for n up to ~1024 (limited by Triton block size).
        Input: 2D tensor (batch, n) where n is power of 2.
        Returns: FWHT(x), same shape.
        """
        assert x.ndim == 2, f"Expected 2D, got {x.ndim}D"
        batch, n = x.shape
        assert n >= 2 and (n & (n - 1)) == 0, f"n must be power of 2 >= 2, got {n}"
        log2_n = int(math.log2(n))
        assert log2_n <= 10, f"n={n} too large for single-kernel FWHT (max 1024)"

        # Get or compile kernel for this n
        if log2_n not in _fwht_kernel_cache:
            _fwht_kernel_cache[log2_n] = _make_fwht_single_kernel(log2_n)
        kernel = _fwht_kernel_cache[log2_n]

        # Work in FP32 for butterfly accuracy, cast back at end
        buf = x.contiguous().to(torch.float32).clone()

        kernel[(batch,)](
            buf,
            buf.stride(0),
            BLOCK_N=n,
            LOG2_N=log2_n,
        )

        return buf.to(x.dtype)


# =====================================================================
# Autograd Functions
# =====================================================================

class FWHTTritonFunction(torch.autograd.Function):
    """Autograd wrapper for Triton FWHT.

    Forward:  y = FWHT(x)
    Backward: grad_x = FWHT(grad_y)

    The FWHT is self-inverse and symmetric, so the Jacobian is H itself,
    and the VJP (backward) is simply another FWHT application.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Save shape info for backward -- no need to save tensors
        # since backward just applies the same transform
        ctx.input_dtype = x.dtype
        return _triton_fwht_single(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # FWHT is its own inverse: d/dx(x @ H) = H^T = H
        # So grad_input = grad_output @ H = FWHT(grad_output)
        return _triton_fwht_single(grad_output.contiguous())


class FWHTMatmulFunction(torch.autograd.Function):
    """Autograd wrapper for matmul-based FWHT (fallback when Triton unavailable)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        ctx.n = n
        H = build_hadamard(n, device=x.device).to(x.dtype)
        return x @ H

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        n = ctx.n
        H = build_hadamard(n, device=grad_output.device).to(grad_output.dtype)
        # H is symmetric: H^T = H
        return grad_output @ H


# =====================================================================
# Public API
# =====================================================================

def fwht_triton(x: torch.Tensor) -> torch.Tensor:
    """Training-compatible FWHT using Triton kernel.

    Input:  2D tensor (batch, n) where n is power of 2.
    Output: FWHT(x), same shape. Supports autograd.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available. Use fwht_matmul_train() instead.")
    return FWHTTritonFunction.apply(x)


def fwht_matmul_train(x: torch.Tensor) -> torch.Tensor:
    """Training-compatible FWHT using cuBLAS matmul (fallback).

    Input:  2D tensor (batch, n) where n is power of 2.
    Output: FWHT(x), same shape. Supports autograd.
    """
    return FWHTMatmulFunction.apply(x)


def fwht_train(x: torch.Tensor) -> torch.Tensor:
    """Training-compatible FWHT. Uses Triton if available, else matmul.

    Input:  2D tensor (batch, n) where n is power of 2.
    Output: FWHT(x), same shape. Supports autograd.
    """
    if HAS_TRITON:
        return FWHTTritonFunction.apply(x)
    else:
        return FWHTMatmulFunction.apply(x)


# =====================================================================
# nn.Module wrapper (for use in model definitions)
# =====================================================================

class FWHTLayer(nn.Module):
    """FWHT as an nn.Module for use in model architectures.

    Replaces a dense matrix multiply x @ H (O(n^2)) with FWHT (O(n log n)).
    No learnable parameters -- the Hadamard matrix is fixed.

    Args:
        n: Transform size (power of 2). Default 64.
        use_triton: Use Triton kernel if available. Default True.

    Example:
        layer = FWHTLayer(n=64)
        x = torch.randn(32, 64, requires_grad=True)
        y = layer(x)      # forward FWHT
        y.sum().backward() # backward is also FWHT
    """

    def __init__(self, n: int = 64, use_triton: bool = True):
        super().__init__()
        assert n >= 2 and (n & (n - 1)) == 0, f"n must be power of 2 >= 2, got {n}"
        self.n = n
        self.use_triton = use_triton and HAS_TRITON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.n, (
            f"Expected last dim = {self.n}, got {x.shape[-1]}"
        )
        if self.use_triton:
            return FWHTTritonFunction.apply(x)
        else:
            return FWHTMatmulFunction.apply(x)

    def extra_repr(self) -> str:
        return f"n={self.n}, triton={self.use_triton}"


# =====================================================================
# Tests and benchmarks
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FWHT Training Kernel -- Tests and Benchmarks")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Triton available: {HAS_TRITON}")

    # ------------------------------------------------------------------
    # Test 1: Correctness -- Triton vs matmul reference
    # ------------------------------------------------------------------
    print("\n--- Test 1: Triton FWHT vs matmul reference ---")
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x = torch.randn(64, n, device=device)
        ref = x.float() @ build_hadamard(n, device=device)
        if HAS_TRITON:
            out_triton = _triton_fwht_single(x)
            match = torch.allclose(out_triton.float(), ref, atol=1e-4, rtol=1e-4)
            print(f"  n={n:4d}: triton_match={match}")
            assert match, f"Triton FWHT mismatch at n={n}! max_diff={torch.max(torch.abs(out_triton.float() - ref))}"
        else:
            print(f"  n={n:4d}: skipped (no Triton)")
    print("  Correctness PASSED.")

    # ------------------------------------------------------------------
    # Test 2: Self-inverse: FWHT(FWHT(x)) = x
    # ------------------------------------------------------------------
    print("\n--- Test 2: Self-inverse property ---")
    for n in [32, 64, 128, 256]:
        x = torch.randn(32, n, device=device)
        if HAS_TRITON:
            y = _triton_fwht_single(x)
            recon = _triton_fwht_single(y)
            match = torch.allclose(recon, x.float(), atol=1e-4)
            print(f"  n={n:4d}: FWHT(FWHT(x))=x: {match}")
            assert match, f"Self-inverse failed at n={n}"
    print("  Self-inverse PASSED.")

    # ------------------------------------------------------------------
    # Test 3: Autograd -- gradients flow correctly
    # ------------------------------------------------------------------
    print("\n--- Test 3: Autograd gradient correctness ---")
    for n in [32, 64, 128]:
        x = torch.randn(16, n, device=device, dtype=torch.float32, requires_grad=True)

        # Forward with our autograd function
        y = fwht_train(x)
        loss = y.sum()
        loss.backward()
        grad_ours = x.grad.clone()

        # Reference: explicit matmul
        x2 = x.detach().clone().requires_grad_(True)
        H = build_hadamard(n, device=device)
        y2 = x2 @ H
        loss2 = y2.sum()
        loss2.backward()
        grad_ref = x2.grad.clone()

        match = torch.allclose(grad_ours, grad_ref, atol=1e-4, rtol=1e-4)
        print(f"  n={n:4d}: grad_match={match}")
        if not match:
            diff = (grad_ours - grad_ref).abs().max()
            print(f"           max_diff={diff:.6e}")
        assert match, f"Gradient mismatch at n={n}"
    print("  Autograd PASSED.")

    # ------------------------------------------------------------------
    # Test 4: Autograd with gradcheck (numerical gradient verification)
    # ------------------------------------------------------------------
    print("\n--- Test 4: torch.autograd.gradcheck ---")
    for n in [8, 16, 32]:
        x = torch.randn(4, n, device=device, dtype=torch.float64, requires_grad=True)
        # gradcheck needs float64 for numerical stability
        # Use matmul version since it natively supports float64
        ok = torch.autograd.gradcheck(FWHTMatmulFunction.apply, (x,), eps=1e-6, atol=1e-4)
        print(f"  n={n:4d}: gradcheck={ok}")
    print("  gradcheck PASSED.")

    # ------------------------------------------------------------------
    # Test 5: BF16 support
    # ------------------------------------------------------------------
    print("\n--- Test 5: BF16 support ---")
    if device.type == "cuda":
        for n in [64, 128]:
            x_bf16 = torch.randn(32, n, device=device, dtype=torch.bfloat16)
            if HAS_TRITON:
                y = fwht_triton(x_bf16)
                assert y.dtype == torch.bfloat16, f"Expected bf16 output, got {y.dtype}"
                # Compare with FP32 reference
                ref = x_bf16.float() @ build_hadamard(n, device=device)
                match = torch.allclose(y.float(), ref, atol=5e-2, rtol=1e-2)
                print(f"  n={n}: bf16 output dtype={y.dtype}, match_fp32={match}")
                assert match, f"BF16 mismatch at n={n}"
            y_mm = fwht_matmul_train(x_bf16)
            assert y_mm.dtype == torch.bfloat16
            print(f"  n={n}: matmul bf16 dtype={y_mm.dtype}: OK")
    else:
        print("  Skipped (no CUDA)")
    print("  BF16 PASSED.")

    # ------------------------------------------------------------------
    # Test 6: FWHTLayer module
    # ------------------------------------------------------------------
    print("\n--- Test 6: FWHTLayer nn.Module ---")
    layer = FWHTLayer(n=64).to(device)
    print(f"  {layer}")
    x = torch.randn(8, 64, device=device, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None, "No gradient computed!"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    print(f"  forward: {y.shape}, backward: grad.shape={x.grad.shape}")
    print("  FWHTLayer PASSED.")

    # ------------------------------------------------------------------
    # Benchmark: matmul vs Triton FWHT
    # ------------------------------------------------------------------
    if device.type == "cuda":
        print("\n--- Benchmark: matmul vs Triton FWHT ---")
        print(f"  {'n':>6s}  {'batch':>6s}  {'matmul_ms':>10s}  {'triton_ms':>10s}  {'speedup':>8s}")
        print("  " + "-" * 50)

        for n in [32, 64, 128, 256, 512]:
            for batch in [256, 1024, 4096]:
                x = torch.randn(batch, n, device=device)
                H = build_hadamard(n, device=device)
                n_iters = 500

                # Warmup
                for _ in range(50):
                    _ = x @ H
                    if HAS_TRITON:
                        _ = _triton_fwht_single(x)
                torch.cuda.synchronize()

                # Matmul
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    _ = x @ H
                torch.cuda.synchronize()
                t_mm = (time.perf_counter() - t0) / n_iters * 1000

                # Triton
                if HAS_TRITON:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n_iters):
                        _ = _triton_fwht_single(x)
                    torch.cuda.synchronize()
                    t_tr = (time.perf_counter() - t0) / n_iters * 1000
                else:
                    t_tr = float("nan")

                speedup = t_mm / t_tr if t_tr > 0 else float("nan")
                print(f"  {n:6d}  {batch:6d}  {t_mm:10.4f}  {t_tr:10.4f}  {speedup:7.2f}x")

        # ------------------------------------------------------------------
        # Benchmark: Forward + Backward (training scenario)
        # ------------------------------------------------------------------
        print("\n--- Benchmark: Forward+Backward (training) ---")
        print(f"  {'n':>6s}  {'batch':>6s}  {'matmul_ms':>10s}  {'fwht_ms':>10s}  {'speedup':>8s}")
        print("  " + "-" * 50)

        for n in [64, 128, 256]:
            for batch in [256, 1024, 4096]:
                n_iters = 200

                # Warmup
                for _ in range(20):
                    x = torch.randn(batch, n, device=device, requires_grad=True)
                    y = fwht_train(x)
                    y.sum().backward()

                    x2 = torch.randn(batch, n, device=device, requires_grad=True)
                    H = build_hadamard(n, device=device).to(x2.dtype)
                    y2 = x2 @ H
                    y2.sum().backward()
                torch.cuda.synchronize()

                # Matmul forward+backward
                H = build_hadamard(n, device=device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    x = torch.randn(batch, n, device=device, requires_grad=True)
                    y = x @ H
                    y.sum().backward()
                torch.cuda.synchronize()
                t_mm = (time.perf_counter() - t0) / n_iters * 1000

                # FWHT forward+backward
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    x = torch.randn(batch, n, device=device, requires_grad=True)
                    y = fwht_train(x)
                    y.sum().backward()
                torch.cuda.synchronize()
                t_fwht = (time.perf_counter() - t0) / n_iters * 1000

                speedup = t_mm / t_fwht if t_fwht > 0 else float("nan")
                print(f"  {n:6d}  {batch:6d}  {t_mm:10.4f}  {t_fwht:10.4f}  {speedup:7.2f}x")

    else:
        print("\n--- Benchmarks skipped (no CUDA) ---")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
