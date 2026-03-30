"""Fast Walsh-Hadamard Transform (FWHT) for PolarEngine.

Two implementations:
  1. fwht_matmul  -- single GEMM kernel launch, 25x faster than butterfly
  2. fwht_butterfly -- reference implementation via butterfly pattern (29 kernels)

Both produce identical results: FWHT(x) = x @ H_normalized.
The transform is self-inverse: FWHT(FWHT(x)) = x.

H128 is 128x128 float32 = 64 KB, which fits entirely in GPU L2 cache.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch

# =====================================================================
# Hadamard matrix builder (cached, normalized)
# =====================================================================

_hadamard_cache: dict[tuple, torch.Tensor] = {}


def build_hadamard(n: int, device: torch.device = None) -> torch.Tensor:
    """Build normalized Walsh-Hadamard matrix of size n (power of 2).

    H @ H = I (self-inverse, orthogonal).
    Cached globally per (size, device) -- 64 KB for n=128, fits in GPU L2.

    The matrix is cached on the requested device so repeated calls
    with the same (n, device) are free (no CPU->GPU copy).

    Args:
        n: Matrix dimension, must be a power of 2.
        device: Target device. If None, returns CPU tensor.

    Returns:
        Normalized Hadamard matrix of shape (n, n).
    """
    assert n >= 1 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"

    # Normalize device for cache key
    dev_key = str(device) if device is not None else "cpu"
    cache_key = (n, dev_key)

    if cache_key not in _hadamard_cache:
        # Build on CPU first (recursive)
        cpu_key = (n, "cpu")
        if cpu_key not in _hadamard_cache:
            if n == 1:
                H = torch.tensor([[1.0]])
            else:
                h = build_hadamard(n // 2)  # recursive, hits cache
                H = torch.cat([
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ], dim=0) / math.sqrt(2)
            _hadamard_cache[cpu_key] = H

        # Move to target device and cache there
        if device is not None:
            _hadamard_cache[cache_key] = _hadamard_cache[cpu_key].to(device)
        else:
            return _hadamard_cache[cpu_key]

    return _hadamard_cache[cache_key]


# =====================================================================
# Matmul FWHT (production path -- 1 kernel launch)
# =====================================================================

def fwht_matmul(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform via matrix multiply.

    25x faster than butterfly (1 kernel launch vs 29).

    Input:  x of shape (..., in_features) where in_features is a
            multiple of block_size.
    Output: same shape, each contiguous block of block_size elements
            is transformed independently.

    The Hadamard matrix is cached and moved to x's device on first call.
    Since H is symmetric and orthogonal, FWHT(x) = x @ H.

    Args:
        x: Input tensor, last dimension must be divisible by block_size.
        block_size: Size of each FWHT block. Must be a power of 2.

    Returns:
        Transformed tensor with the same shape as x.
    """
    in_features = x.shape[-1]
    assert in_features % block_size == 0, (
        f"in_features ({in_features}) must be divisible by block_size ({block_size})"
    )

    H = build_hadamard(block_size, device=x.device).to(x.dtype)

    orig_shape = x.shape
    # Flatten all leading dims + split last dim into blocks
    x_flat = x.reshape(-1, block_size)
    out = x_flat @ H
    return out.reshape(orig_shape)


# =====================================================================
# FWHT Cache (reuse across Q/K/V sharing the same hidden state)
# =====================================================================

class FWHTCache:
    """Cache FWHT results across layers sharing the same input.

    In a transformer, Q/K/V projections receive the same hidden state.
    Cache hit saves 33% of FWHT calls.

    Keyed by (data_ptr, in_features). Must be cleared between
    model.forward() calls via a forward pre-hook:

        cache = FWHTCache()
        model.register_forward_pre_hook(lambda m, inp: cache.clear())
    """

    def __init__(self):
        self._ptr: int = -1
        self._in_f: int = -1
        self._result: Optional[torch.Tensor] = None

    def get(self, x: torch.Tensor, in_f: int) -> Optional[torch.Tensor]:
        """Return cached FWHT result if x matches the cached input.

        Match is determined by data_ptr equality (same storage) and
        in_features equality (same block structure). Returns None on miss.
        """
        if x.data_ptr() == self._ptr and in_f == self._in_f:
            return self._result
        return None

    def put(self, x: torch.Tensor, in_f: int, result: torch.Tensor):
        """Store an FWHT result for the given input tensor."""
        self._ptr = x.data_ptr()
        self._in_f = in_f
        self._result = result

    def clear(self):
        """Clear the cache. Call this between forward passes."""
        self._ptr = -1
        self._in_f = -1
        self._result = None


# =====================================================================
# Butterfly FWHT (reference -- 29 kernel launches for n=128)
# =====================================================================

def fwht_butterfly(x: torch.Tensor) -> torch.Tensor:
    """Reference FWHT via butterfly pattern. Slow (29 kernels) but correct.

    Implements the standard in-place butterfly decomposition:
        for each stage h = 1, 2, 4, ..., n/2:
            for each pair separated by h:
                (a, b) -> (a+b, a-b)
        then normalize by 1/sqrt(n).

    This produces the same result as x @ H_normalized.

    Args:
        x: Input tensor, last dimension must be a power of 2.

    Returns:
        Transformed tensor with the same shape as x.
    """
    n = x.shape[-1]
    assert n >= 1 and (n & (n - 1)) == 0, f"last dim must be power of 2, got {n}"

    h = 1
    while h < n:
        leading = x.shape[:-1]
        r = x.view(*leading, -1, 2 * h)
        a = r[..., :h].clone()
        b = r[..., h:].clone()
        r[..., :h] = a + b
        r[..., h:] = a - b
        x = r.view(*leading, -1)
        h *= 2
    return x / math.sqrt(n)


# =====================================================================
# Standalone tests and benchmarks
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FWHT Module -- Tests and Benchmarks")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Verify matmul == butterfly for various block sizes
    # ------------------------------------------------------------------
    print("\n--- Test 1: matmul vs butterfly equivalence ---")
    for bs in [2, 4, 8, 16, 32, 64, 128, 256]:
        x = torch.randn(64, bs, device=device)
        out_mm = fwht_matmul(x, block_size=bs)
        out_bf = fwht_butterfly(x.clone())
        match = torch.allclose(out_mm, out_bf, atol=1e-5)
        print(f"  block_size={bs:4d}: match={match}")
        assert match, f"FWHT mismatch at block_size={bs}!"

    # Also test with in_features > block_size (multiple blocks)
    for bs in [64, 128]:
        x = torch.randn(32, bs * 4, device=device)
        out_mm = fwht_matmul(x, block_size=bs)
        # butterfly on each block separately
        x_blocks = x.reshape(-1, bs)
        out_bf = fwht_butterfly(x_blocks).reshape(x.shape)
        match = torch.allclose(out_mm, out_bf, atol=1e-5)
        print(f"  multi-block bs={bs}, in_f={bs*4}: match={match}")
        assert match, f"FWHT mismatch at multi-block bs={bs}!"

    print("  All equivalence tests PASSED.")

    # ------------------------------------------------------------------
    # 2. Verify self-inverse: FWHT(FWHT(x)) = x
    # ------------------------------------------------------------------
    print("\n--- Test 2: self-inverse property ---")
    for bs in [64, 128, 256]:
        x = torch.randn(32, bs, device=device)
        reconstructed = fwht_matmul(fwht_matmul(x, block_size=bs), block_size=bs)
        match = torch.allclose(x, reconstructed, atol=1e-5)
        print(f"  block_size={bs}: self-inverse={match}")
        assert match, f"Self-inverse failed at block_size={bs}!"
    print("  Self-inverse PASSED.")

    # ------------------------------------------------------------------
    # 3. Verify H @ H = I (orthogonality)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Hadamard orthogonality ---")
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        H = build_hadamard(n, device=device)
        eye = torch.eye(n, device=device)
        match = torch.allclose(H @ H, eye, atol=1e-5)
        print(f"  n={n:4d}: H@H=I: {match}")
        assert match, f"Orthogonality failed at n={n}!"
    print("  Orthogonality PASSED.")

    # ------------------------------------------------------------------
    # 4. Benchmark: matmul vs butterfly
    # ------------------------------------------------------------------
    if device.type == "cuda":
        print("\n--- Benchmark: matmul vs butterfly (block_size=128) ---")
        x = torch.randn(4096, 128, device=device)
        n_iters = 200

        # Warmup
        for _ in range(20):
            _ = fwht_matmul(x, block_size=128)
            _ = fwht_butterfly(x.clone())
        torch.cuda.synchronize()

        # Matmul timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fwht_matmul(x, block_size=128)
        torch.cuda.synchronize()
        t_mm = (time.perf_counter() - t0) / n_iters * 1000

        # Butterfly timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fwht_butterfly(x.clone())
        torch.cuda.synchronize()
        t_bf = (time.perf_counter() - t0) / n_iters * 1000

        speedup = t_bf / t_mm if t_mm > 0 else float("inf")
        print(f"  Matmul:    {t_mm:.4f} ms/iter")
        print(f"  Butterfly: {t_bf:.4f} ms/iter")
        print(f"  Speedup:   {speedup:.1f}x")
    else:
        print("\n--- Benchmark skipped (no CUDA) ---")

    # ------------------------------------------------------------------
    # 5. Test FWHTCache hit/miss behavior
    # ------------------------------------------------------------------
    print("\n--- Test 5: FWHTCache ---")
    cache = FWHTCache()

    x = torch.randn(32, 128, device=device)
    result = fwht_matmul(x, block_size=128)

    # Miss on empty cache
    assert cache.get(x, 128) is None, "Expected cache miss on empty cache"
    print("  Empty cache -> miss: OK")

    # Put and hit
    cache.put(x, 128, result)
    cached = cache.get(x, 128)
    assert cached is not None, "Expected cache hit after put"
    assert torch.equal(cached, result), "Cached result should be identical"
    print("  After put  -> hit:  OK")

    # Miss with different in_features
    assert cache.get(x, 64) is None, "Expected miss with different in_f"
    print("  Wrong in_f -> miss: OK")

    # Miss with different tensor (different data_ptr)
    y = torch.randn(32, 128, device=device)
    assert cache.get(y, 128) is None, "Expected miss with different tensor"
    print("  Diff tensor -> miss: OK")

    # Clear
    cache.clear()
    assert cache.get(x, 128) is None, "Expected miss after clear"
    print("  After clear -> miss: OK")

    print("  FWHTCache tests PASSED.")

    # ------------------------------------------------------------------
    # 6. Test various block sizes with larger input
    # ------------------------------------------------------------------
    print("\n--- Test 6: various block sizes with multi-block input ---")
    for bs in [64, 128, 256]:
        x = torch.randn(16, 1024, device=device)
        out = fwht_matmul(x, block_size=bs)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        # Verify self-inverse
        recon = fwht_matmul(out, block_size=bs)
        match = torch.allclose(x, recon, atol=1e-5)
        print(f"  bs={bs}, in_f=1024: shape OK, self-inverse={match}")
        assert match

    print("  All block size tests PASSED.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
