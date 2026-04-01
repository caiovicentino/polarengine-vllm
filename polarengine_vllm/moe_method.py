"""
PolarEngine MoE Method — Pass-through for FusedMoE layers.

Phase 1: MoE experts stay in BF16 (standard vLLM fused MoE path).
         PolarQuant only applies to Linear layers (attention, Mamba projections).

Phase 2 (future): PolarQuant codes for expert weights with Triton GEMV.

The key challenge: vLLM's FusedMoEMethodBase.__init__ requires `moe` layer
reference, but get_quant_method() is called BEFORE the layer is fully
initialized. This module bypasses that chicken-and-egg problem.
"""

import torch
from typing import Optional


class PolarPassthroughMoEMethod:
    """Pass-through MoE method that uses vLLM's standard fused MoE path.

    Inherits from FusedMoEMethodBase via __class__ manipulation to satisfy
    isinstance checks without calling FusedMoEMethodBase.__init__.
    """

    def __init__(self):
        # Intentionally skip FusedMoEMethodBase.__init__
        # which requires a `moe` layer reference not yet available
        self._initialized = False

    def _lazy_init(self, layer):
        """Initialize the real unquantized MoE method once layer is ready."""
        if self._initialized:
            return
        try:
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
            self._inner = UnquantizedFusedMoEMethod(moe=layer)
            self._initialized = True
        except (AttributeError, TypeError):
            # layer still not ready — will retry next call
            pass

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **kwargs):
        """Create standard BF16 MoE weight parameters."""
        self._lazy_init(layer)
        if self._initialized:
            return self._inner.create_weights(
                layer, num_experts=num_experts, hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype, **kwargs)

        # Fallback: create weights manually if _inner init failed
        # This matches vLLM's standard MoE weight layout:
        # w13 = gate_proj + up_proj fused, w2 = down_proj
        from vllm.model_executor.parameter import BasevLLMParameter

        # w13: fused gate+up projection (num_experts, 2*intermediate, hidden)
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition,
                        hidden_size, dtype=params_dtype),
            requires_grad=False
        )
        layer.register_parameter("w13_weight", w13)

        # w2: down projection (num_experts, hidden, intermediate)
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size,
                        intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False
        )
        layer.register_parameter("w2_weight", w2)

    def apply(self, layer, x, router_logits, top_k, renormalize,
              use_grouped_topk=False, topk_group=None,
              num_expert_group=None, custom_routing_function=None,
              scoring_func="softmax", e_score_correction_bias=None,
              **kwargs):
        """Run standard fused MoE forward pass."""
        self._lazy_init(layer)
        if self._initialized:
            return self._inner.apply(
                layer, x, router_logits, top_k, renormalize,
                use_grouped_topk=use_grouped_topk, topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                **kwargs)

        # Fallback: should not reach here in practice
        raise RuntimeError("PolarPassthroughMoEMethod: _inner not initialized during apply()")

    def get_fused_moe_quant_config(self):
        """Return None for unquantized MoE (standard path)."""
        if self._initialized and hasattr(self._inner, 'get_fused_moe_quant_config'):
            return self._inner.get_fused_moe_quant_config()
        return None


# Register as FusedMoEMethodBase subclass for isinstance checks
# This avoids calling FusedMoEMethodBase.__init__ while passing isinstance()
try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
    # Monkey-patch the class hierarchy
    PolarPassthroughMoEMethod.__bases__ = (FusedMoEMethodBase,) + tuple(
        b for b in PolarPassthroughMoEMethod.__bases__
        if b is not object
    )
except ImportError:
    pass
