"""
LRU expert cache for PolarQuant MoE expert offloading.

Manages GPU-side expert tensor caching with per-layer LRU eviction.
Each cached expert stores its PolarQuant components (codes, norms,
ct_scaled) so the Triton kernel can consume them directly without
any dequantization step.

Designed for Nemotron-Cascade-2-30B-A3B: 23 MoE layers, 128 routed
experts per layer, top-6 active per token.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ExpertCacheManager:
    """Per-layer LRU cache for PolarQuant expert tensors on GPU.

    Each cache entry holds a dict with:
      - ``codes``     (int8 or uint8 packed) -- quantised weight indices
      - ``norms``     (fp16) -- per-block L2 norms
      - ``ct_scaled`` (fp32) -- pre-scaled centroids

    Uses :class:`collections.OrderedDict` for O(1) LRU get/put/evict.
    A separate cache is maintained per MoE layer so that eviction
    pressure is localised -- a cold expert in layer 5 does not evict
    a hot expert in layer 12.

    Args:
        num_layers:          Number of MoE layers in the model
                             (e.g. 23 for Nemotron-Cascade-2-30B-A3B).
        cache_size_per_layer: Maximum experts cached per layer (default 16).
        device:              GPU device for cached tensors (default ``'cuda'``).
    """

    def __init__(
        self,
        num_layers: int,
        cache_size_per_layer: int = 16,
        device: str = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.cache_size_per_layer = cache_size_per_layer
        self.device = device

        # Per-layer OrderedDict: expert_id -> dict of tensors
        # OrderedDict maintains insertion order; move_to_end on access
        # gives O(1) LRU semantics.
        self._caches: list[OrderedDict[int, Dict[str, torch.Tensor]]] = [
            OrderedDict() for _ in range(num_layers)
        ]

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def get(
        self, layer_idx: int, expert_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Look up a cached expert.

        If found, the entry is marked as most-recently-used.

        Args:
            layer_idx: MoE layer index (0-based).
            expert_id: Expert index within the layer.

        Returns:
            Dict with ``'codes'``, ``'norms'``, ``'ct_scaled'`` tensors
            on ``self.device``, or ``None`` on cache miss.
        """
        cache = self._caches[layer_idx]
        if expert_id in cache:
            # Move to end = most recently used
            cache.move_to_end(expert_id)
            self._hits += 1
            return cache[expert_id]

        self._misses += 1
        return None

    def put(
        self,
        layer_idx: int,
        expert_id: int,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Insert or update an expert in the cache.

        If the cache is full, the least-recently-used expert is evicted.
        Tensors are moved to ``self.device`` if not already there.

        Args:
            layer_idx: MoE layer index (0-based).
            expert_id: Expert index within the layer.
            tensors:   Dict with ``'codes'``, ``'norms'``, ``'ct_scaled'``
                       tensors (will be moved to ``self.device``).
        """
        cache = self._caches[layer_idx]

        # If already present, update in place and mark as most recent.
        if expert_id in cache:
            cache.move_to_end(expert_id)
            cache[expert_id] = self._to_device(tensors)
            return

        # Evict LRU entry if at capacity.
        if len(cache) >= self.cache_size_per_layer:
            # popitem(last=False) removes the *oldest* (least recently used)
            evicted_id, evicted_tensors = cache.popitem(last=False)
            self._evictions += 1
            # Let the evicted tensors be garbage-collected to free GPU memory.
            del evicted_tensors
            logger.debug(
                "Evicted expert %d from layer %d cache", evicted_id, layer_idx
            )

        cache[expert_id] = self._to_device(tensors)

    def _to_device(
        self, tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move tensors to the cache device if not already there."""
        result: Dict[str, torch.Tensor] = {}
        for key, t in tensors.items():
            if t.device != torch.device(self.device):
                result[key] = t.to(self.device, non_blocking=True)
            else:
                result[key] = t
        return result

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Check if an expert is in the cache without affecting LRU order."""
        return expert_id in self._caches[layer_idx]

    def cached_experts(self, layer_idx: int) -> list[int]:
        """Return list of expert IDs currently cached for a layer.

        Ordered from least-recently-used to most-recently-used.
        """
        return list(self._caches[layer_idx].keys())

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear the cache.

        Args:
            layer_idx: If given, clear only this layer's cache.
                       If ``None``, clear all layers.
        """
        if layer_idx is not None:
            self._caches[layer_idx].clear()
        else:
            for cache in self._caches:
                cache.clear()

    def stats(self) -> Dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with ``'hits'``, ``'misses'``, ``'evictions'``, and
            ``'total_cached'`` (sum of cached experts across all layers).
        """
        total_cached = sum(len(c) for c in self._caches)
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100.0) if total_requests > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "total_cached": total_cached,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters to zero."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __repr__(self) -> str:
        total_cached = sum(len(c) for c in self._caches)
        return (
            f"ExpertCacheManager(num_layers={self.num_layers}, "
            f"cache_size_per_layer={self.cache_size_per_layer}, "
            f"total_cached={total_cached}, device='{self.device}')"
        )
