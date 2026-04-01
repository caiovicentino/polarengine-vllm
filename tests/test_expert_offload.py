"""Tests for expert cache and expert offload store.

These tests run entirely on CPU (no CUDA required) using mock data
that simulates PolarQuant expert tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from polarengine_vllm.expert_cache import ExpertCacheManager
from polarengine_vllm.expert_offload import ExpertOffloadStore


# ===================================================================
# Helpers
# ===================================================================

def _make_expert_tensors(
    out_f: int = 1856,
    in_f: int = 2688,
    block_size: int = 128,
    bits: int = 3,
) -> dict:
    """Create mock PolarQuant tensors for a single projection."""
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size
    n_levels = 1 << bits

    return {
        "codes": torch.randint(0, n_levels, (out_f, in_f_padded), dtype=torch.int8),
        "norms": torch.randn(out_f, n_blocks, dtype=torch.float16).abs() + 0.1,
        "ct_scaled": torch.randn(n_levels, dtype=torch.float32),
    }


def _make_flat_expert_tensors() -> dict:
    """Create a simple flat dict with codes/norms/ct_scaled for cache tests."""
    return {
        "codes": torch.randint(0, 8, (16, 32), dtype=torch.int8),
        "norms": torch.randn(16, 4, dtype=torch.float16),
        "ct_scaled": torch.randn(8, dtype=torch.float32),
    }


# ===================================================================
# ExpertCacheManager tests
# ===================================================================

class TestExpertCacheManager:
    """Tests for ExpertCacheManager LRU cache."""

    def test_init(self):
        cache = ExpertCacheManager(num_layers=4, cache_size_per_layer=8, device="cpu")
        assert cache.num_layers == 4
        assert cache.cache_size_per_layer == 8
        assert cache.device == "cpu"

    def test_put_and_get(self):
        cache = ExpertCacheManager(num_layers=2, cache_size_per_layer=4, device="cpu")
        tensors = _make_flat_expert_tensors()

        cache.put(0, 5, tensors)
        result = cache.get(0, 5)

        assert result is not None
        assert "codes" in result
        assert "norms" in result
        assert "ct_scaled" in result
        assert torch.equal(result["codes"], tensors["codes"])
        assert torch.equal(result["norms"], tensors["norms"])
        assert torch.equal(result["ct_scaled"], tensors["ct_scaled"])

    def test_get_miss(self):
        cache = ExpertCacheManager(num_layers=2, cache_size_per_layer=4, device="cpu")
        result = cache.get(0, 99)
        assert result is None

    def test_cache_isolation_between_layers(self):
        cache = ExpertCacheManager(num_layers=3, cache_size_per_layer=4, device="cpu")
        t0 = _make_flat_expert_tensors()
        t1 = _make_flat_expert_tensors()

        cache.put(0, 1, t0)
        cache.put(1, 1, t1)

        r0 = cache.get(0, 1)
        r1 = cache.get(1, 1)

        assert r0 is not None
        assert r1 is not None
        # They should be different tensors (from different layers)
        assert not torch.equal(r0["codes"], r1["codes"])

    def test_lru_eviction(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=3, device="cpu")

        # Fill to capacity
        cache.put(0, 0, _make_flat_expert_tensors())
        cache.put(0, 1, _make_flat_expert_tensors())
        cache.put(0, 2, _make_flat_expert_tensors())

        # All three should be present
        assert cache.get(0, 0) is not None
        assert cache.get(0, 1) is not None
        assert cache.get(0, 2) is not None

        # Access expert 0 to make it most-recently used.
        # Current LRU order (oldest first): 1, 2, 0
        cache.get(0, 0)

        # Insert a 4th expert -- should evict expert 1 (LRU)
        cache.put(0, 3, _make_flat_expert_tensors())

        assert cache.get(0, 1) is None  # evicted
        assert cache.get(0, 0) is not None  # was accessed, not evicted
        assert cache.get(0, 2) is not None  # still present
        assert cache.get(0, 3) is not None  # newly inserted

    def test_lru_order_respects_access(self):
        """Verify that accessing an expert moves it to most-recently-used."""
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=2, device="cpu")

        cache.put(0, 10, _make_flat_expert_tensors())
        cache.put(0, 20, _make_flat_expert_tensors())

        # Access expert 10 (making 20 the LRU)
        cache.get(0, 10)

        # Insert 30 -- should evict 20 (LRU), not 10
        cache.put(0, 30, _make_flat_expert_tensors())

        assert cache.get(0, 20) is None  # evicted
        assert cache.get(0, 10) is not None  # kept (was accessed)
        assert cache.get(0, 30) is not None  # newly inserted

    def test_update_existing(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=4, device="cpu")
        t1 = _make_flat_expert_tensors()
        t2 = _make_flat_expert_tensors()

        cache.put(0, 5, t1)
        cache.put(0, 5, t2)  # update

        result = cache.get(0, 5)
        assert result is not None
        assert torch.equal(result["codes"], t2["codes"])

    def test_stats(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=3, device="cpu")

        # Miss
        cache.get(0, 0)
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 1

        # Put then hit
        cache.put(0, 0, _make_flat_expert_tensors())
        cache.get(0, 0)
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["total_cached"] == 1

        # Fill and evict
        cache.put(0, 1, _make_flat_expert_tensors())
        cache.put(0, 2, _make_flat_expert_tensors())
        cache.put(0, 3, _make_flat_expert_tensors())  # evicts expert 0
        s = cache.stats()
        assert s["evictions"] == 1

    def test_stats_hit_rate(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=4, device="cpu")
        cache.put(0, 0, _make_flat_expert_tensors())

        # 3 hits, 1 miss
        cache.get(0, 0)
        cache.get(0, 0)
        cache.get(0, 0)
        cache.get(0, 99)

        s = cache.stats()
        assert s["hits"] == 3
        assert s["misses"] == 1
        assert s["hit_rate_pct"] == 75.0

    def test_reset_stats(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=4, device="cpu")
        cache.put(0, 0, _make_flat_expert_tensors())
        cache.get(0, 0)
        cache.get(0, 99)

        cache.reset_stats()
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["evictions"] == 0

    def test_contains(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=4, device="cpu")
        cache.put(0, 5, _make_flat_expert_tensors())

        assert cache.contains(0, 5) is True
        assert cache.contains(0, 99) is False

    def test_cached_experts(self):
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=4, device="cpu")
        cache.put(0, 3, _make_flat_expert_tensors())
        cache.put(0, 7, _make_flat_expert_tensors())
        cache.put(0, 1, _make_flat_expert_tensors())

        experts = cache.cached_experts(0)
        assert set(experts) == {1, 3, 7}

    def test_clear_single_layer(self):
        cache = ExpertCacheManager(num_layers=2, cache_size_per_layer=4, device="cpu")
        cache.put(0, 1, _make_flat_expert_tensors())
        cache.put(1, 1, _make_flat_expert_tensors())

        cache.clear(layer_idx=0)

        assert cache.get(0, 1) is None
        # Layer 1 should be unaffected -- but get() increments miss counter,
        # so use contains() to avoid side effects.
        assert cache.contains(1, 1) is True

    def test_clear_all(self):
        cache = ExpertCacheManager(num_layers=2, cache_size_per_layer=4, device="cpu")
        cache.put(0, 1, _make_flat_expert_tensors())
        cache.put(1, 1, _make_flat_expert_tensors())

        cache.clear()

        assert cache.contains(0, 1) is False
        assert cache.contains(1, 1) is False

    def test_repr(self):
        cache = ExpertCacheManager(num_layers=3, cache_size_per_layer=16, device="cpu")
        r = repr(cache)
        assert "num_layers=3" in r
        assert "cache_size_per_layer=16" in r

    def test_capacity_one(self):
        """Edge case: cache with capacity 1 per layer."""
        cache = ExpertCacheManager(num_layers=1, cache_size_per_layer=1, device="cpu")

        cache.put(0, 0, _make_flat_expert_tensors())
        assert cache.contains(0, 0) is True

        cache.put(0, 1, _make_flat_expert_tensors())
        assert cache.contains(0, 0) is False  # evicted
        assert cache.contains(0, 1) is True


# ===================================================================
# ExpertOffloadStore tests
# ===================================================================

class TestExpertOffloadStore:
    """Tests for ExpertOffloadStore with mock safetensors data."""

    @staticmethod
    def _make_mock_weight_loader(expert_keys: dict):
        """Create a mock PolarWeightLoader with predefined tensors.

        Args:
            expert_keys: dict mapping tensor key -> torch.Tensor
        """
        loader = MagicMock()
        loader.weight_map = {k: "shard-00001.safetensors" for k in expert_keys}
        loader.load_tensor = MagicMock(
            side_effect=lambda key, device="cpu": expert_keys[key]
        )
        return loader

    @staticmethod
    def _build_expert_keys(
        layer_idx: int = 0,
        expert_id: int = 0,
        prefix: str = "backbone",
    ) -> dict:
        """Build a dict of tensor keys for a single expert."""
        base = f"{prefix}.layers.{layer_idx}.mixer.experts.{expert_id}"
        keys = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t = _make_expert_tensors()
            for component in ("codes", "norms", "ct_scaled"):
                full_key = f"{base}.{proj}.{component}"
                keys[full_key] = t[component]
        return keys

    def test_load_single_expert(self):
        expert_keys = self._build_expert_keys(layer_idx=2, expert_id=5)
        loader = self._make_mock_weight_loader(expert_keys)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        assert store.num_experts_loaded == 1
        assert store.has_expert(2, 5)
        assert not store.has_expert(2, 6)

        expert = store.get_expert(2, 5)
        assert "gate_proj" in expert
        assert "up_proj" in expert
        assert "down_proj" in expert
        assert "codes" in expert["gate_proj"]
        assert "norms" in expert["gate_proj"]

    def test_load_multiple_experts(self):
        keys = {}
        keys.update(self._build_expert_keys(layer_idx=0, expert_id=0))
        keys.update(self._build_expert_keys(layer_idx=0, expert_id=1))
        keys.update(self._build_expert_keys(layer_idx=1, expert_id=0))

        loader = self._make_mock_weight_loader(keys)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        assert store.num_experts_loaded == 3
        assert store.has_expert(0, 0)
        assert store.has_expert(0, 1)
        assert store.has_expert(1, 0)
        assert not store.has_expert(1, 1)

    def test_moe_layer_indices(self):
        keys = {}
        keys.update(self._build_expert_keys(layer_idx=3, expert_id=0))
        keys.update(self._build_expert_keys(layer_idx=7, expert_id=0))
        keys.update(self._build_expert_keys(layer_idx=12, expert_id=0))

        loader = self._make_mock_weight_loader(keys)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        assert store.moe_layer_indices == [3, 7, 12]

    def test_experts_for_layer(self):
        keys = {}
        for eid in (0, 5, 12, 127):
            keys.update(self._build_expert_keys(layer_idx=0, expert_id=eid))

        loader = self._make_mock_weight_loader(keys)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        assert store.experts_for_layer(0) == [0, 5, 12, 127]
        assert store.experts_for_layer(1) == []

    def test_get_expert_missing_raises(self):
        loader = self._make_mock_weight_loader({})
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        with pytest.raises(KeyError, match="not found in offload store"):
            store.get_expert(0, 0)

    def test_transfer_to_gpu_on_cpu(self):
        """Test transfer_to_gpu targeting CPU (works without CUDA)."""
        keys = self._build_expert_keys(layer_idx=0, expert_id=0)
        loader = self._make_mock_weight_loader(keys)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        # Transfer to "cpu" device (CUDA not required for this test)
        result = store.transfer_to_gpu(0, 0, device="cpu")
        assert "gate_proj" in result
        assert result["gate_proj"]["codes"].device == torch.device("cpu")

    def test_model_prefix_pattern(self):
        """Test that model.layers.X.mlp.experts.Y pattern also works."""
        keys = self._build_expert_keys(
            layer_idx=5, expert_id=42, prefix="model"
        )
        # Rewrite keys to use mlp instead of mixer
        rewritten = {}
        for k, v in keys.items():
            new_key = k.replace(".mixer.", ".mlp.")
            rewritten[new_key] = v

        loader = self._make_mock_weight_loader(rewritten)
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        store.load_all_experts()

        assert store.num_experts_loaded == 1
        assert store.has_expert(5, 42)

    def test_no_experts_warning(self, caplog):
        """When no expert keys are found, a warning should be logged."""
        # Weight map with non-expert keys only
        loader = self._make_mock_weight_loader({
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        })
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )

        import logging
        with caplog.at_level(logging.WARNING):
            store.load_all_experts()

        assert store.num_experts_loaded == 0
        assert "No expert tensors found" in caplog.text

    def test_repr(self):
        loader = self._make_mock_weight_loader({})
        config = {"layers": {}, "block_size": 128, "format": "polar_engine_v5"}

        store = ExpertOffloadStore(
            polar_config=config, weight_loader=loader, device="cpu"
        )
        r = repr(store)
        assert "ExpertOffloadStore" in r
        assert "experts=0" in r
