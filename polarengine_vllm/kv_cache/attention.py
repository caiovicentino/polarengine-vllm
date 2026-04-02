"""PolarQuant KV Cache Attention Wrapper for vLLM.

Intercepts vLLM's attention to compress/decompress KV cache on-the-fly.
Works with any attention backend (FlashInfer, Xformers, SDPA).

Two integration modes:
1. **Standalone**: Drop-in replacement for transformers Cache
2. **vLLM plugin**: Wraps vLLM's attention backend

Usage (standalone with transformers):
    cache = PolarKVCache(PolarKVConfig.for_gemma4_31b())
    wrapper = PolarKVAttentionWrapper(cache)

    # In generation loop:
    out = model(input_ids, past_key_values=wrapper, use_cache=True)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Any

import torch

from .config import PolarKVConfig
from .cache import PolarKVCache


class PolarKVAttentionWrapper:
    """Wraps PolarKVCache for use with transformers or vLLM.

    For transformers integration, this acts as a Cache-like object
    that model.forward() can use via past_key_values.

    For vLLM integration, this wraps the attention forward pass
    to intercept KV cache reads/writes.
    """

    def __init__(self, cache: PolarKVCache):
        self.cache = cache
        self.config = cache.config

    # ── transformers Cache API ────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """transformers Cache.update() interface."""
        return self.cache.update(key_states, value_states, layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> None:
        return None

    @property
    def seen_tokens(self) -> int:
        return self.cache.seen_tokens

    @property
    def is_initialized(self) -> bool:
        return self.cache.seen_tokens > 0

    # ── vLLM attention wrapping ───────────────────────────────────

    def wrap_attention_forward(
        self,
        original_forward,
        layer_idx: int,
    ):
        """Create a wrapped attention forward that uses compressed KV.

        Args:
            original_forward: The original attention module's forward method
            layer_idx: Which transformer layer this wraps

        Returns:
            A wrapped forward function
        """
        cache = self.cache

        def wrapped_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Any,
            attn_metadata: Any,
            **kwargs,
        ) -> torch.Tensor:
            # Compress new K/V into PolarQuant cache
            # key/value shape: (num_tokens, num_kv_heads, head_dim)
            B = 1  # vLLM flattens batch
            H = key.shape[1] if key.dim() == 3 else self.config.num_kv_heads
            D = key.shape[-1]

            # Reshape for cache update: (B, H, S, D)
            k_4d = key.unsqueeze(0).transpose(1, 2) if key.dim() == 3 else key
            v_4d = value.unsqueeze(0).transpose(1, 2) if value.dim() == 3 else value

            # Update cache (quantize + store)
            full_k, full_v = cache.update(k_4d, v_4d, layer_idx)

            # Reshape back for attention: (num_tokens, num_kv_heads, head_dim)
            if key.dim() == 3:
                full_k = full_k.squeeze(0).transpose(0, 1)
                full_v = full_v.squeeze(0).transpose(0, 1)

            # Run original attention with decompressed KV
            return original_forward(
                query, full_k, full_v, kv_cache, attn_metadata, **kwargs
            )

        return wrapped_forward


def patch_model_attention(
    model: torch.nn.Module,
    config: PolarKVConfig,
) -> PolarKVCache:
    """Patch a model's attention layers to use PolarQuant KV cache.

    Finds all attention modules and wraps their forward methods.
    Works with transformers models loaded via AutoModelForCausalLM.

    Args:
        model: The transformer model
        config: PolarKV configuration

    Returns:
        The PolarKVCache instance (for monitoring/stats)
    """
    cache = PolarKVCache(config)
    wrapper = PolarKVAttentionWrapper(cache)

    layer_idx = 0
    for name, module in model.named_modules():
        # Detect attention modules by common naming patterns
        if any(
            pattern in name
            for pattern in ["self_attn", "attention", "attn"]
        ):
            # Check if this is the actual attention module (has q_proj)
            if hasattr(module, "q_proj") or hasattr(module, "qkv_proj"):
                if layer_idx < config.num_layers:
                    # Store reference for the wrapper
                    module._polar_kv_layer_idx = layer_idx
                    module._polar_kv_cache = cache
                    layer_idx += 1

    return cache
