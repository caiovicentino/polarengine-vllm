"""Tests for PolarQuant KV cache.

Run: pytest polarengine_vllm/kv_cache/test_polar_kv.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from .config import PolarKVConfig
from .cache import (
    PolarKVQuantizer,
    PolarKVLayer,
    PolarKVCache,
    BitPacker,
    get_centroids,
    build_hadamard,
)


# ═══════════════════════════════════════════════════════════════════
# BitPacker Tests
# ═══════════════════════════════════════════════════════════════════


class TestBitPacker:
    @pytest.mark.parametrize("nbits", [2, 3, 4])
    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_roundtrip(self, nbits, D):
        """pack → unpack must be lossless."""
        N = 128
        n_levels = 1 << nbits
        codes = torch.randint(0, n_levels, (N, D), dtype=torch.int8)
        packed = BitPacker.pack(codes, nbits)
        unpacked = BitPacker.unpack(packed, nbits, D)
        assert torch.equal(unpacked, codes.long())

    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_packed_size(self, nbits):
        """Packed tensor must have expected size."""
        D = 256
        N = 64
        codes = torch.randint(0, 1 << nbits, (N, D), dtype=torch.int8)
        packed = BitPacker.pack(codes, nbits)
        expected_cols = BitPacker.packed_size(D, nbits)
        assert packed.shape == (N, expected_cols)
        assert packed.dtype == torch.uint8

    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_edge_values(self, nbits):
        """All-zeros and all-max-value roundtrip correctly."""
        D = 128
        N = 32
        max_val = (1 << nbits) - 1

        for val in [0, max_val]:
            codes = torch.full((N, D), val, dtype=torch.int8)
            packed = BitPacker.pack(codes, nbits)
            unpacked = BitPacker.unpack(packed, nbits, D)
            assert torch.equal(unpacked, codes.long())


# ═══════════════════════════════════════════════════════════════════
# Centroids Tests
# ═══════════════════════════════════════════════════════════════════


class TestCentroids:
    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_centroid_count(self, nbits):
        ct = get_centroids(nbits)
        assert ct.shape == (1 << nbits,)

    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_centroids_sorted(self, nbits):
        ct = get_centroids(nbits)
        assert (ct[1:] > ct[:-1]).all(), "Centroids must be strictly increasing"

    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_centroids_symmetric(self, nbits):
        """Lloyd-Max centroids for N(0,1) should be approximately symmetric."""
        ct = get_centroids(nbits)
        n = len(ct)
        for i in range(n // 2):
            assert abs(ct[i].item() + ct[n - 1 - i].item()) < 0.01

    def test_caching(self):
        ct1 = get_centroids(3)
        ct2 = get_centroids(3)
        assert ct1 is ct2, "Centroids should be cached"


# ═══════════════════════════════════════════════════════════════════
# Hadamard Tests
# ═══════════════════════════════════════════════════════════════════


class TestHadamard:
    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_orthogonal(self, n):
        H = build_hadamard(n)
        I = torch.eye(n)
        product = H @ H.T
        assert torch.allclose(product, I, atol=1e-5), "H must be orthogonal"

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_shape(self, n):
        H = build_hadamard(n)
        assert H.shape == (n, n)

    def test_caching(self):
        H1 = build_hadamard(128, "cpu")
        H2 = build_hadamard(128, "cpu")
        assert H1 is H2, "Hadamard matrix should be cached"


# ═══════════════════════════════════════════════════════════════════
# Quantizer Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPolarKVQuantizer:
    @pytest.mark.parametrize("head_dim", [128, 256])
    @pytest.mark.parametrize("nbits", [2, 3, 4])
    def test_roundtrip_quality(self, head_dim, nbits):
        """Quantize → dequantize should have high cosine similarity."""
        quantizer = PolarKVQuantizer(head_dim, nbits, "cuda")
        N = 256
        x = torch.randn(N, head_dim, device="cuda", dtype=torch.bfloat16)

        packed, norms = quantizer.quantize(x)
        recon = quantizer.dequantize(packed, norms, x.shape)

        cos_sim = torch.nn.functional.cosine_similarity(
            x.float(), recon.float(), dim=1
        ).mean().item()

        min_quality = {2: 0.90, 3: 0.97, 4: 0.99}
        assert cos_sim > min_quality[nbits], (
            f"Q{nbits} cosine sim {cos_sim:.4f} < {min_quality[nbits]}"
        )

    @pytest.mark.parametrize("head_dim", [128, 256])
    def test_output_dtype(self, head_dim):
        quantizer = PolarKVQuantizer(head_dim, 3, "cuda")
        x = torch.randn(32, head_dim, device="cuda", dtype=torch.bfloat16)
        packed, norms = quantizer.quantize(x)
        recon = quantizer.dequantize(packed, norms, x.shape)

        assert packed.dtype == torch.uint8
        assert norms.dtype == torch.bfloat16
        assert recon.dtype == torch.bfloat16

    def test_4d_input(self):
        """Test (B, H, S, D) shaped input."""
        quantizer = PolarKVQuantizer(128, 3, "cuda")
        x = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        packed, norms = quantizer.quantize(x)
        recon = quantizer.dequantize(packed, norms, x.shape)
        assert recon.shape == x.shape


# ═══════════════════════════════════════════════════════════════════
# KV Layer Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPolarKVLayer:
    def test_residual_only(self):
        """Below residual_length, no quantization occurs."""
        quantizer = PolarKVQuantizer(128, 3, "cuda")
        layer = PolarKVLayer(quantizer, residual_length=128)

        x = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        out = layer.update(x)

        assert out.shape == x.shape
        assert layer._packed is None  # No quantization yet
        assert layer.get_seq_length() == 64

    def test_quantization_triggers(self):
        """Beyond residual_length, older tokens get quantized."""
        quantizer = PolarKVQuantizer(128, 3, "cuda")
        layer = PolarKVLayer(quantizer, residual_length=64)

        # Add 128 tokens — should quantize 64 and keep 64 in residual
        x = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        out = layer.update(x)

        assert out.shape == (1, 8, 128, 128)
        assert layer._packed is not None
        assert layer.get_seq_length() == 128

    def test_incremental_updates(self):
        """Token-by-token generation should work correctly."""
        quantizer = PolarKVQuantizer(128, 3, "cuda")
        layer = PolarKVLayer(quantizer, residual_length=32)

        total_len = 0
        for _ in range(100):
            x = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
            out = layer.update(x)
            total_len += 1
            assert out.shape[2] == total_len
            assert layer.get_seq_length() == total_len

    def test_memory_savings(self):
        """Quantized cache should use less memory than FP16."""
        quantizer = PolarKVQuantizer(128, 3, "cuda")
        layer = PolarKVLayer(quantizer, residual_length=32)

        # Add 512 tokens
        x = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
        layer.update(x)

        actual_bytes = layer.memory_bytes()
        fp16_bytes = 1 * 8 * 512 * 128 * 2
        assert actual_bytes < fp16_bytes, "Quantized cache should be smaller than FP16"


# ═══════════════════════════════════════════════════════════════════
# Full Cache Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPolarKVCache:
    def test_creation(self):
        config = PolarKVConfig(nbits=3, head_dim=128, num_layers=4, num_kv_heads=8)
        cache = PolarKVCache(config)
        assert len(cache.k_layers) == 4
        assert len(cache.v_layers) == 4

    def test_update_returns_kv(self):
        config = PolarKVConfig(nbits=3, head_dim=128, num_layers=2, num_kv_heads=8)
        cache = PolarKVCache(config)

        k = torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.bfloat16)

        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_skip_layers(self):
        config = PolarKVConfig(
            nbits=3, head_dim=128, num_layers=4, num_kv_heads=8,
            skip_layers=[1, 3],
        )
        cache = PolarKVCache(config)
        assert cache.k_layers[0] is not None
        assert cache.k_layers[1] is None  # skipped
        assert cache.k_layers[2] is not None
        assert cache.k_layers[3] is None  # skipped

    def test_stats(self):
        config = PolarKVConfig(nbits=3, head_dim=128, num_layers=2, num_kv_heads=8)
        cache = PolarKVCache(config)

        k = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        cache.update(k, v, 0)
        cache.update(k, v, 1)

        stats = cache.stats()
        assert stats["seq_length"] == 64
        assert stats["memory_mb"] > 0

    def test_reset(self):
        config = PolarKVConfig(nbits=3, head_dim=128, num_layers=2, num_kv_heads=8)
        cache = PolarKVCache(config)

        k = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
        cache.update(k, v, 0)

        cache.reset()
        assert cache.get_seq_length() == 0
        assert cache.seen_tokens == 0


# ═══════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════


class TestPolarKVConfig:
    def test_gemma4_preset(self):
        config = PolarKVConfig.for_gemma4_31b()
        assert config.head_dim == 256
        assert config.num_layers == 60
        assert config.num_kv_heads == 16

    def test_compression_ratio(self):
        config = PolarKVConfig(nbits=3, head_dim=128)
        assert abs(config.compression_ratio - 16 / 3) < 0.01

    def test_max_context(self):
        config = PolarKVConfig.for_gemma4_31b(nbits=3)
        max_ctx = config.max_context(4.0)
        assert max_ctx > 10000, "Q3 with 4GB should support >10K context"

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="power of 2"):
            PolarKVConfig(head_dim=72)

    def test_invalid_nbits(self):
        with pytest.raises(AssertionError):
            PolarKVConfig(nbits=5)
