"""
LFRU (Least Frequently + Recently Used) expert cache.

Solves the deep-layer cache starvation problem in MoE models:
- Standard LRU: early layers execute first, monopolize cache slots
- LFRU: tracks access frequency, high-frequency experts survive eviction

Result on GPT-OSS-20B (from @e1n00r PR #37190):
  LRU:  layers 18-23 hit rate 0-8%
  LFRU: layers 18-23 hit rate 52-94%

For Nemotron-Cascade-2-30B-A3B (128 experts/layer, top-6, 23 MoE layers),
LFRU is critical because the large expert space causes severe LRU thrashing.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class LFRUCache:
    """Least-Frequently-Recently-Used cache for MoE expert weights.

    Eviction policy: evict the expert with the lowest access frequency.
    On tie, evict the least recently used (standard LRU tiebreaker).

    Frequency counts decay by half every `decay_interval` calls to prevent
    stale experts from permanently occupying cache slots.

    Args:
        capacity: Maximum number of experts in cache.
        decay_interval: Number of prepare() calls between frequency decay.
                        Set to 0 to disable decay.
    """

    def __init__(self, capacity: int, decay_interval: int = 1000):
        self.capacity = capacity
        self.decay_interval = decay_interval

        # expert_id → slot_index
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free_slots: list[int] = list(range(capacity))

        # Frequency tracking
        self._freq: dict[int, int] = {}  # expert_id → access count
        self._call_count: int = 0

        # Stats
        self.hits: int = 0
        self.misses: int = 0

    def _decay_frequencies(self):
        """Halve all frequencies to prevent stale dominance."""
        for eid in list(self._freq.keys()):
            self._freq[eid] >>= 1  # integer divide by 2
            if self._freq[eid] == 0 and eid not in self._lru:
                del self._freq[eid]  # cleanup

    def _find_victim(self) -> int:
        """Find the expert to evict: lowest frequency, then LRU tiebreaker."""
        min_freq = float('inf')
        victim_id = None

        # Iterate in LRU order (oldest first) for tiebreaking
        for eid in self._lru:
            freq = self._freq.get(eid, 0)
            if freq < min_freq:
                min_freq = freq
                victim_id = eid

        return victim_id

    def access(self, expert_id: int) -> tuple[bool, Optional[int]]:
        """Record access to an expert.

        Returns:
            (is_hit, evicted_slot_or_None)
            - hit: (True, None) — expert already in cache
            - miss with free slot: (False, None) — loaded into free slot
            - miss with eviction: (False, evicted_expert_id) — evicted + loaded
        """
        # Update frequency
        self._freq[expert_id] = self._freq.get(expert_id, 0) + 1

        if expert_id in self._lru:
            # Cache hit
            self._lru.move_to_end(expert_id)
            self.hits += 1
            return True, None

        # Cache miss
        self.misses += 1
        evicted = None

        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            # Evict: lowest frequency, LRU tiebreak
            victim_id = self._find_victim()
            slot = self._lru.pop(victim_id)
            evicted = victim_id

        self._lru[expert_id] = slot

        # Periodic frequency decay
        self._call_count += 1
        if self.decay_interval > 0 and self._call_count % self.decay_interval == 0:
            self._decay_frequencies()

        return False, evicted

    def get_slot(self, expert_id: int) -> Optional[int]:
        """Get the GPU buffer slot for an expert, or None if not cached."""
        return self._lru.get(expert_id)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CachedWeightProviderLFRU:
    """Expert weight provider with LFRU eviction policy.

    Drop-in replacement for CachedWeightProvider (LRU) with better
    hit rates on deep MoE layers.

    Args:
        capacity: Number of expert slots in GPU buffer.
        w13_weight: Full gate+up weights (num_experts, intermediate*2, hidden).
        w2_weight: Full down weights (num_experts, hidden, intermediate).
        decay_interval: Frequency decay interval (default 1000).
    """

    def __init__(self, capacity, w13_weight, w2_weight, decay_interval=1000):
        self.capacity = capacity
        self.num_experts = w13_weight.shape[0]

        logger.info("CachedWeightProviderLFRU: %d slots, w13=%s, decay=%d",
                     capacity, list(w13_weight.shape), decay_interval)

        # CPU pinned storage
        self._cpu_w13 = w13_weight.detach().cpu().pin_memory()
        self._cpu_w2 = w2_weight.detach().cpu().pin_memory()
        logger.info("Copied %.1f GB to CPU pinned memory",
                     (self._cpu_w13.numel() + self._cpu_w2.numel()) * 2 / 1e9)

        # GPU buffer (fixed size)
        self._buf_w13 = torch.empty(
            (capacity, *w13_weight.shape[1:]),
            dtype=w13_weight.dtype, device="cuda")
        self._buf_w2 = torch.empty(
            (capacity, *w2_weight.shape[1:]),
            dtype=w2_weight.dtype, device="cuda")

        # LFRU cache
        self._cache = LFRUCache(capacity, decay_interval)

        # Expert ID → GPU slot mapping
        self._mapping = torch.full(
            (self.num_experts,), -1, dtype=torch.int32, device="cuda")

        self._overflow_warned = False
        self._last_log = time.time()

    def prepare(self, topk_ids):
        """Ensure experts in topk_ids are on GPU. Returns (buf_w1, buf_w2, remapped_ids)."""
        unique = topk_ids.unique().tolist()

        if len(unique) > self.capacity:
            if not self._overflow_warned:
                logger.warning(
                    "LFRU prepare: %d unique experts > capacity %d. Truncating.",
                    len(unique), self.capacity)
                self._overflow_warned = True
            unique = unique[-self.capacity:]

        for eid in unique:
            is_hit, evicted = self._cache.access(eid)

            if is_hit:
                continue

            # Cache miss — load expert to GPU
            slot = self._cache.get_slot(eid)

            if evicted is not None:
                self._mapping[evicted] = -1

            self._buf_w13[slot].copy_(self._cpu_w13[eid], non_blocking=True)
            self._buf_w2[slot].copy_(self._cpu_w2[eid], non_blocking=True)
            self._mapping[eid] = slot

        torch.cuda.current_stream().synchronize()
        remapped = self._mapping[topk_ids.long()]

        # Log stats periodically
        now = time.time()
        if now - self._last_log > 30:
            rate = self._cache.hit_rate * 100
            logger.info("LFRU cache: %.1f%% hit (%d/%d)",
                         rate, self._cache.hits, self._cache.hits + self._cache.misses)
            self._last_log = now

        return self._buf_w13, self._buf_w2, remapped
