"""PolarQuant KV Cache Compression for vLLM.

Adds Q2/Q3/Q4 KV cache compression to vLLM's attention pipeline
using Hadamard rotation + Lloyd-Max optimal centroids.

Usage:
    from polarengine_vllm.kv_cache import PolarKVConfig, PolarKVCache

    config = PolarKVConfig(nbits=3, head_dim=256, num_layers=60, num_kv_heads=16)
    cache = PolarKVCache(config)
"""

from .config import PolarKVConfig
from .cache import PolarKVCache, PolarKVQuantizer
from .attention import PolarKVAttentionWrapper

__all__ = [
    "PolarKVConfig",
    "PolarKVCache",
    "PolarKVQuantizer",
    "PolarKVAttentionWrapper",
]
