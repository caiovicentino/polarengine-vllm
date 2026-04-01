"""
Nemotron MoE Expert Offloading — custom model layer for vLLM.

Replaces SharedFusedMoE with a scatter-based MoE that loads experts
on-demand from CPU pinned memory. Only top-K experts reside on GPU
at any time, enabling RTX 4090 (24 GB) inference for Nemotron 30B.

Usage:
    1. polarquant-convert creates BF16 model + this file
    2. vLLM loads with trust_remote_code=True
    3. OffloadedNemotronMoE replaces SharedFusedMoE in forward pass

VRAM budget (cache_size=8 per layer, 23 MoE layers):
    Experts: 8 × 30 MB × 23 layers × 2 (w13+w2) ≈ 11 GB
    Non-expert (Mamba, attn, norms, router): ≈ 2.5 GB
    Total: ≈ 13.5 GB → fits RTX 4090!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Optional, Dict, List
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ExpertGPUCache:
    """Per-layer LRU cache for MoE expert weights on GPU."""

    def __init__(self, capacity: int = 8, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self._cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, expert_id: int) -> Optional[Dict[str, torch.Tensor]]:
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self.hits += 1
            return self._cache[expert_id]
        self.misses += 1
        return None

    def put(self, expert_id: int, tensors: Dict[str, torch.Tensor]):
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self._cache[expert_id] = tensors
            return
        if len(self._cache) >= self.capacity:
            evicted_id, evicted = self._cache.popitem(last=False)
            del evicted  # free GPU memory
        self._cache[expert_id] = tensors

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._cache

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class OffloadedNemotronMoE(nn.Module):
    """MoE layer with expert offloading for consumer GPU inference.

    Experts are stored in CPU pinned memory and loaded to GPU on-demand.
    An LRU cache keeps the most-used experts on GPU.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: Expert FFN intermediate dimension
        num_experts: Total number of routed experts (e.g., 128)
        top_k: Number of experts selected per token (e.g., 6)
        num_shared_experts: Number of always-resident shared experts
        shared_intermediate_size: Shared expert FFN dimension
        cache_size: Max experts cached on GPU per layer (default 8)
        scoring_func: Router scoring function ("softmax" or "sigmoid")
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int = 1,
        shared_intermediate_size: Optional[int] = None,
        cache_size: int = 8,
        scoring_func: str = "softmax",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.shared_intermediate_size = shared_intermediate_size or intermediate_size
        self.cache_size = cache_size
        self.scoring_func = scoring_func
        self.dtype = dtype

        # Router (always on GPU, tiny)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)
        # e_score_correction_bias (always on GPU)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32, device=device)
        )

        # Shared expert (always on GPU)
        self.shared_w13 = nn.Linear(hidden_size, shared_intermediate_size * 2,
                                     bias=False, dtype=dtype, device=device)
        self.shared_w2 = nn.Linear(shared_intermediate_size, hidden_size,
                                    bias=False, dtype=dtype, device=device)

        # Expert cache (GPU LRU)
        self.expert_cache = ExpertGPUCache(capacity=cache_size, device=device)

        # CPU storage for expert weights (populated during load_weights)
        self._cpu_experts: Dict[int, Dict[str, torch.Tensor]] = {}
        self._loaded = False

    def load_expert_weights(self, expert_id: int,
                            w13_weight: torch.Tensor,
                            w2_weight: torch.Tensor):
        """Store expert weights in CPU pinned memory."""
        self._cpu_experts[expert_id] = {
            "w13": w13_weight.pin_memory() if w13_weight.device.type == "cpu" else w13_weight.cpu().pin_memory(),
            "w2": w2_weight.pin_memory() if w2_weight.device.type == "cpu" else w2_weight.cpu().pin_memory(),
        }

    def _ensure_expert_on_gpu(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """Get expert tensors on GPU, loading from CPU if needed."""
        cached = self.expert_cache.get(expert_id)
        if cached is not None:
            return cached

        # Cache miss — transfer from CPU
        cpu_expert = self._cpu_experts.get(expert_id)
        if cpu_expert is None:
            raise KeyError(f"Expert {expert_id} not found in CPU storage")

        gpu_expert = {
            "w13": cpu_expert["w13"].to(self.gate.weight.device, non_blocking=True),
            "w2": cpu_expert["w2"].to(self.gate.weight.device, non_blocking=True),
        }
        self.expert_cache.put(expert_id, gpu_expert)
        return gpu_expert

    def forward(self, hidden_states: torch.Tensor,
                router_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with expert offloading.

        Args:
            hidden_states: (batch*seq, hidden_size)
            router_logits: pre-computed router logits (optional)

        Returns:
            output: (batch*seq, hidden_size)
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        num_tokens = hidden_states.shape[0]

        # Router
        if router_logits is None:
            router_logits = self.gate(hidden_states)  # (num_tokens, num_experts)

        # Apply e_score_correction_bias
        router_logits = router_logits + self.e_score_correction_bias

        # Scoring
        if self.scoring_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=-1)
        else:
            routing_weights = torch.sigmoid(router_logits)

        # Top-K selection
        topk_weights, topk_ids = routing_weights.topk(self.top_k, dim=-1)

        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Prefetch unique experts
        unique_experts = topk_ids.unique().tolist()
        for eid in unique_experts:
            self._ensure_expert_on_gpu(eid)

        # Synchronize async transfers
        torch.cuda.current_stream().synchronize()

        # Per-expert scatter computation
        output = torch.zeros(num_tokens, self.hidden_size,
                             dtype=hidden_states.dtype, device=hidden_states.device)

        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]  # (num_tokens,)
            weights = topk_weights[:, k]  # (num_tokens,)

            for eid in expert_ids.unique().tolist():
                mask = expert_ids == eid
                if not mask.any():
                    continue

                x = hidden_states[mask]  # (n, hidden)
                expert = self.expert_cache.get(eid)

                # w13 = gate_up fused: (intermediate*2, hidden)
                gate_up = x @ expert["w13"].T  # (n, intermediate*2)
                gate = gate_up[:, :self.intermediate_size]
                up = gate_up[:, self.intermediate_size:]

                # SwiGLU activation
                activated = F.silu(gate) * up  # (n, intermediate)

                # w2 = down: (hidden, intermediate)
                down = activated @ expert["w2"].T  # (n, hidden)

                output[mask] += weights[mask].unsqueeze(1) * down

        # Shared expert (always on GPU)
        shared_gate_up = self.shared_w13(hidden_states)
        shared_gate = shared_gate_up[:, :self.shared_intermediate_size]
        shared_up = shared_gate_up[:, self.shared_intermediate_size:]
        shared_out = F.silu(shared_gate) * shared_up
        shared_out = self.shared_w2(shared_out)
        output = output + shared_out

        if len(orig_shape) == 3:
            output = output.view(orig_shape)

        return output

    def stats(self) -> str:
        hr = self.expert_cache.hit_rate * 100
        return (f"ExpertOffload: {len(self._cpu_experts)} experts on CPU, "
                f"cache_size={self.cache_size}, "
                f"hit_rate={hr:.1f}%")
