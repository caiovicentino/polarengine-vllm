"""Test FWHT implementation."""

import math

import pytest
import torch

from polarengine_vllm.kernels.fwht import (
    FWHTCache,
    build_hadamard,
    fwht_butterfly,
    fwht_matmul,
)


class TestFWHT:
    """Core FWHT correctness tests (CPU-safe)."""

    def test_matmul_equals_butterfly(self):
        """fwht_matmul and fwht_butterfly produce identical results."""
        x = torch.randn(64, 128)
        out_mm = fwht_matmul(x, block_size=128)
        out_bf = fwht_butterfly(x.clone())
        assert torch.allclose(out_mm, out_bf, atol=1e-5), (
            "fwht_matmul and fwht_butterfly diverge"
        )

    def test_matmul_equals_butterfly_multiblock(self):
        """Equivalence holds when in_features > block_size (multiple blocks)."""
        bs = 128
        x = torch.randn(32, bs * 4)
        out_mm = fwht_matmul(x, block_size=bs)

        # Butterfly applies to each block independently
        x_blocks = x.reshape(-1, bs)
        out_bf = fwht_butterfly(x_blocks).reshape(x.shape)

        assert torch.allclose(out_mm, out_bf, atol=1e-5), (
            "Multi-block matmul vs butterfly mismatch"
        )

    def test_self_inverse(self):
        """FWHT(FWHT(x)) = x (the transform is its own inverse)."""
        x = torch.randn(32, 128)
        reconstructed = fwht_matmul(fwht_matmul(x, block_size=128), block_size=128)
        assert torch.allclose(x, reconstructed, atol=1e-5), (
            "Self-inverse property violated"
        )

    @pytest.mark.parametrize("bs", [64, 128, 256])
    def test_self_inverse_various_sizes(self, bs):
        """Self-inverse property holds for various block sizes."""
        x = torch.randn(16, bs)
        reconstructed = fwht_matmul(fwht_matmul(x, block_size=bs), block_size=bs)
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_hadamard_orthogonal(self):
        """H @ H = I for the normalized Hadamard matrix."""
        for n in [2, 4, 8, 16, 32, 64, 128, 256]:
            H = build_hadamard(n)
            eye = torch.eye(n)
            assert torch.allclose(H @ H, eye, atol=1e-5), (
                f"H@H != I for n={n}"
            )

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128])
    def test_hadamard_symmetric(self, n):
        """H is symmetric (H = H^T) for the normalized Hadamard matrix."""
        H = build_hadamard(n)
        assert torch.allclose(H, H.T, atol=1e-6), f"H is not symmetric for n={n}"

    @pytest.mark.parametrize("bs", [32, 64, 128, 256])
    def test_block_sizes(self, bs):
        """fwht_matmul works correctly for various block sizes."""
        x = torch.randn(16, 1024)
        out = fwht_matmul(x, block_size=bs)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

        # Verify self-inverse as a correctness check
        recon = fwht_matmul(out, block_size=bs)
        assert torch.allclose(x, recon, atol=1e-5), (
            f"Self-inverse failed for bs={bs}"
        )

    def test_hadamard_power_of_two_assertion(self):
        """build_hadamard rejects non-power-of-2 sizes."""
        with pytest.raises(AssertionError):
            build_hadamard(3)
        with pytest.raises(AssertionError):
            build_hadamard(0)
        with pytest.raises(AssertionError):
            build_hadamard(7)

    def test_fwht_matmul_divisibility_assertion(self):
        """fwht_matmul raises when in_features is not divisible by block_size."""
        x = torch.randn(4, 100)
        with pytest.raises(AssertionError):
            fwht_matmul(x, block_size=128)

    def test_butterfly_power_of_two_assertion(self):
        """fwht_butterfly raises when last dim is not a power of 2."""
        x = torch.randn(4, 100)
        with pytest.raises(AssertionError):
            fwht_butterfly(x)

    def test_output_preserves_shape(self):
        """FWHT does not change the tensor shape."""
        shapes = [(1, 128), (32, 256), (8, 64), (100, 128)]
        for shape in shapes:
            x = torch.randn(*shape)
            out = fwht_matmul(x, block_size=shape[1])
            assert out.shape == x.shape


class TestFWHTCache:
    """Test FWHTCache hit/miss semantics."""

    def test_cache_miss_empty(self):
        """Empty cache always returns None."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        assert cache.get(x, 128) is None

    def test_cache_hit_after_put(self):
        """After put, get returns the cached result."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        result = fwht_matmul(x, block_size=128)

        cache.put(x, 128, result)
        cached = cache.get(x, 128)

        assert cached is not None
        assert torch.equal(cached, result)

    def test_cache_miss_different_in_f(self):
        """Cache miss when in_features does not match."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        result = fwht_matmul(x, block_size=128)

        cache.put(x, 128, result)
        assert cache.get(x, 64) is None

    def test_cache_miss_different_tensor(self):
        """Cache miss when a different tensor is queried."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        y = torch.randn(32, 128)
        result = fwht_matmul(x, block_size=128)

        cache.put(x, 128, result)
        assert cache.get(y, 128) is None

    def test_cache_clear(self):
        """After clear, previously cached entry is gone."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        result = fwht_matmul(x, block_size=128)

        cache.put(x, 128, result)
        assert cache.get(x, 128) is not None

        cache.clear()
        assert cache.get(x, 128) is None

    def test_cache_overwrite(self):
        """A second put overwrites the first entry."""
        cache = FWHTCache()
        x = torch.randn(32, 128)
        y = torch.randn(32, 128)
        result_x = fwht_matmul(x, block_size=128)
        result_y = fwht_matmul(y, block_size=128)

        cache.put(x, 128, result_x)
        cache.put(y, 128, result_y)

        # Old entry is gone (single-entry cache)
        assert cache.get(x, 128) is None
        assert cache.get(y, 128) is not None


class TestBuildHadamard:
    """Test the Hadamard matrix builder."""

    def test_n1(self):
        """build_hadamard(1) returns [[1.0]]."""
        H = build_hadamard(1)
        assert torch.allclose(H, torch.tensor([[1.0]]))

    def test_n2(self):
        """build_hadamard(2) returns the 2x2 normalized Hadamard matrix."""
        H = build_hadamard(2)
        expected = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32) / math.sqrt(2)
        assert torch.allclose(H, expected, atol=1e-6)

    def test_device_transfer(self):
        """build_hadamard moves the matrix to the requested device."""
        H = build_hadamard(16, device=torch.device("cpu"))
        assert H.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )
    def test_device_cuda(self):
        """build_hadamard moves the matrix to CUDA when requested."""
        H = build_hadamard(16, device=torch.device("cuda"))
        assert H.device.type == "cuda"

    def test_caching(self):
        """Repeated calls return tensors from the cache."""
        H1 = build_hadamard(64)
        H2 = build_hadamard(64)
        # They should be the exact same object in CPU cache
        assert torch.equal(H1, H2)
