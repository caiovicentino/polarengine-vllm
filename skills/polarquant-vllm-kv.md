# PolarQuant KV Cache vLLM Integration Generator

Generate a complete implementation that adds PolarQuant Q3 KV cache compression to a vLLM-served model, packaged as an extension to the existing `polarengine-vllm` plugin.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `google/gemma-4-31B-it`
- `https://huggingface.co/caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check the model page for: parameter count, architecture, num_layers, num_kv_heads, head_dim, hidden_size, attention type (standard, GQA, MQA, hybrid sliding/global).

2. **Check vLLM support**: Verify the model architecture is supported in vLLM. Check for special attention patterns (sliding window, hybrid, MoE).

3. **Determine head_dim**: Critical for Hadamard matrix size. Common values: 64, 96, 128, 256. Must be power of 2 for Walsh-Hadamard. If not power of 2 → skip PolarQuant for those layers, fall back to FP16 KV.

4. **Generate the implementation**: Write files to `~/Desktop/polarquant-kv-vllm-{SHORT_NAME}/`.

## Architecture Overview

PolarQuant KV cache integrates into vLLM's attention pipeline:

```
Standard vLLM:
  Q, K, V → PagedAttention(Q, K_cache, V_cache) → Output

PolarQuant KV:
  Q, K, V → Quantize(K, V) → PagedAttention(Q, Dequant(K_cache), Dequant(V_cache)) → Output

  Quantize: Hadamard rotate → normalize → Lloyd-Max Q3 → bit-pack
  Dequant:  unpack → centroid lookup → inverse Hadamard → denormalize
```

## Files to Generate

### 1. `polar_kv_cache.py` — Core KV cache quantization

```python
"""PolarQuant KV Cache for vLLM.

Wraps vLLM's cache operations to compress K/V tensors on-the-fly.
Supports Q2/Q3/Q4 with Hadamard rotation + Lloyd-Max centroids.
"""

import torch
import math
from typing import Optional, Tuple

class PolarKVQuantizer:
    """Quantizes/dequantizes KV cache tensors for a specific head_dim."""

    def __init__(self, head_dim: int, nbits: int = 3, device: str = 'cuda'):
        self.head_dim = head_dim
        self.nbits = nbits
        self.device = device
        self.n_levels = 1 << nbits

        # Precompute Lloyd-Max centroids
        self.centroids = self._compute_centroids().to(device)

        # Precompute Hadamard matrix (cached per head_dim)
        self.H = self._build_hadamard(head_dim).to(device)
        self.scale = math.sqrt(head_dim)

    @staticmethod
    def _compute_centroids(nbits):
        """Lloyd-Max optimal centroids for N(0,1) via iterative refinement."""
        # [Include full Lloyd-Max computation from get_centroids()]
        pass

    @staticmethod
    def _build_hadamard(n):
        """Recursive Walsh-Hadamard matrix construction."""
        # [Include _build_H() function]
        pass

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV tensor: (B, H, S, D) → (packed_codes, norms).

        Args:
            tensor: BF16 tensor of shape (batch, num_heads, seq_len, head_dim)

        Returns:
            packed: uint8 bit-packed codes
            norms: BF16 per-vector norms
        """
        B, H, S, D = tensor.shape
        flat = tensor.reshape(-1, D).float()

        # L2 normalize
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normalized = flat / norms * self.scale

        # Hadamard rotation
        rotated = normalized @ self.H

        # Lloyd-Max quantize
        codes = (rotated.unsqueeze(-1) - self.centroids.view(1, 1, -1)).abs().argmin(-1)

        # Bit-pack
        packed = self._pack(codes, self.nbits)

        return packed, norms.bfloat16().squeeze(1)

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor,
                   shape: Tuple[int, ...]) -> torch.Tensor:
        """Dequantize: (packed_codes, norms) → (B, H, S, D) BF16 tensor.

        Args:
            packed: uint8 bit-packed codes
            norms: BF16 per-vector norms
            shape: original (B, H, S, D) shape

        Returns:
            BF16 tensor of shape (B, H, S, D)
        """
        B, H, S, D = shape

        # Unpack
        codes = self._unpack(packed, self.nbits, D)

        # Centroid lookup + inverse Hadamard
        values = self.centroids[codes] / self.scale
        values = (values @ self.H) * norms.float().unsqueeze(1)

        return values.bfloat16().reshape(B, H, S, D)
```

**CRITICAL — Include complete BitPacker with pack/unpack for 2/3/4 bit:**
- 2-bit: pack 4 codes per byte
- 3-bit: pack 8 codes into 3 bytes
- 4-bit: pack 2 codes per byte

**CRITICAL — Hadamard matrix size = head_dim:**
- head_dim=128 → H128 (standard Llama/Qwen)
- head_dim=256 → H256 (Gemma 4)
- head_dim=64 → H64 (some smaller models)
- Non-power-of-2 → SKIP quantization, keep FP16

### 2. `polar_kv_attention.py` — vLLM attention wrapper

```python
"""Attention wrapper that uses PolarQuant compressed KV cache.

Intercepts vLLM's attention computation to:
1. Quantize new K/V entries before writing to cache
2. Dequantize cached K/V when reading for attention
"""

class PolarKVAttentionWrapper:
    """Wraps vLLM attention backend with PolarQuant KV compression."""

    def __init__(self, original_attention, head_dim, nbits=3):
        self.original = original_attention
        self.quantizer = PolarKVQuantizer(head_dim, nbits)
        self.compressed_cache = {}  # layer_idx → (packed_k, norms_k, packed_v, norms_v)

    def forward(self, query, key, value, kv_cache, attn_metadata):
        """Modified attention forward with compressed KV cache."""
        # Quantize new K/V
        # Dequantize for attention computation
        # Use original attention kernel
        pass
```

### 3. `polar_kv_triton.py` — Triton kernels for fused quantize/dequantize

```python
"""Triton kernels for PolarQuant KV cache operations.

Two main kernels:
1. polar_kv_quantize_kernel: fused Hadamard + normalize + quantize + pack
2. polar_kv_dequantize_kernel: fused unpack + lookup + inverse Hadamard + denormalize

These run during cache write/read and must be FAST (< 0.1ms per layer).
"""

import triton
import triton.language as tl

@triton.jit
def polar_kv_quantize_kernel(
    input_ptr,     # (N, D) float input
    packed_ptr,    # output packed codes
    norms_ptr,     # output norms
    centroids_ptr, # (n_levels,) centroids
    H_ptr,         # (D, D) Hadamard matrix
    N, D, n_levels, nbits,
    BLOCK_D: tl.constexpr,
):
    """Fused quantize: rotate → normalize → quantize → pack."""
    pass

@triton.jit
def polar_kv_dequantize_kernel(
    packed_ptr,    # input packed codes
    norms_ptr,     # input norms
    output_ptr,    # (N, D) float output
    centroids_ptr, # (n_levels,) centroids
    H_ptr,         # (D, D) Hadamard matrix
    N, D, n_levels, nbits,
    BLOCK_D: tl.constexpr,
):
    """Fused dequantize: unpack → lookup → inverse rotate → denormalize."""
    pass
```

### 4. `polar_kv_config.py` — Configuration

```python
"""Configuration for PolarQuant KV cache compression."""

from dataclasses import dataclass

@dataclass
class PolarKVConfig:
    nbits: int = 3              # Quantization bits (2, 3, or 4)
    residual_length: int = 128  # Keep last N tokens in FP16
    head_dim: int = 128         # Model's KV head dimension
    num_kv_heads: int = 8       # Number of KV heads
    num_layers: int = 32        # Number of transformer layers
    enabled: bool = True        # Enable/disable compression
```

### 5. `benchmark_kv.py` — Benchmark script

Benchmark PolarQuant KV cache vs FP16 KV:
- Throughput (tok/s) at different context lengths (1K, 4K, 16K, 64K)
- VRAM usage at different context lengths
- Quality: token match rate vs FP16
- Latency: per-token quantize + dequantize time

### 6. `test_polar_kv.py` — Tests

- Roundtrip test: quantize → dequantize ≈ original (within tolerance)
- BitPacker roundtrip: pack → unpack == original (exact)
- Shape correctness for different head_dims (64, 128, 256)
- Memory savings verification
- Integration test with vLLM attention

### 7. `README.md` — Documentation

Include:
- Installation: `pip install polarengine-vllm[kv-cache]`
- Quick start with vLLM
- Benchmark results
- Architecture diagram
- Supported models

## Key Patterns (MUST FOLLOW)

### Lloyd-Max Centroids
```python
# scipy.stats.norm, 100 iterations, EXACT same as weight quantization
from scipy.stats import norm as sp_norm
# Precompute for bits [2, 3, 4]
```

### Hadamard Matrix
```python
def _build_H(n):
    if n == 1: return torch.tensor([[1.0]])
    half = _build_H(n // 2)
    return torch.cat([
        torch.cat([half, half], 1),
        torch.cat([half, -half], 1)
    ], 0) / math.sqrt(2)
```

### BitPacker (EXACT copy from PolarQuant unified)
Must support 2-bit, 3-bit, and 4-bit packing. Use the same BitPacker class from the unified notebook.

### BF16 Native
- All dequantized outputs must be `.bfloat16()` NOT `.half()`
- Modern models (Gemma 4, Qwen3.5, Llama 3+) are BF16 native
- FP16 causes numerical instability

### head_dim Detection
```python
self._can_quantize = (head_dim & (head_dim - 1) == 0)  # must be power of 2
```
If head_dim is NOT power of 2, skip PolarQuant for those layers, keep FP16 KV.

### Residual Buffer
Keep last `residual_length` tokens in FP16 (not quantized). This:
1. Avoids quantizing very recent tokens (most important for attention)
2. Allows batching quantization (process in chunks)
3. Handles streaming token-by-token generation

### Cache API Compatibility
For vLLM integration, the cache must support:
- `update(key_states, value_states, layer_idx)` — add new KV entries
- `get_seq_length(layer_idx)` — current sequence length
- PagedAttention compatibility (block-level quantization)

### Performance Targets
- Quantize latency: < 0.1 ms per layer per token
- Dequantize latency: < 0.1 ms per layer per token
- Memory compression: 5.3x for Q3, 4x for Q4, 8x for Q2
- Speed overhead: < 5% vs FP16 KV
- Quality: > 90% token match for Q3, > 95% for Q4

## Model-Specific Adaptations

### Gemma 4 (head_dim=256, hybrid attention)
- H256 Hadamard matrix (256×256)
- Sliding window layers (head_dim=256) + global layers (may differ)
- `_can_quantize` guard for layers with different head_dim

### Llama 3 / Qwen3.5 (head_dim=128)
- H128 Hadamard matrix (standard)
- GQA: num_kv_heads < num_attention_heads

### Qwen3.5 Hybrid (attention + linear attention)
- Some layers use linear attention (Mamba-style) → no KV cache
- `_HybridCacheLayer` for `update_conv_state` / `update_recurrent_state`

## Existing Code References

- **PolarEngine vLLM plugin**: `/Volumes/SSD Major/fish/polarengine-vllm/`
- **Weight quantization**: `polarengine_vllm/linear_method.py`
- **FWHT kernels**: `polarengine_vllm/kernels/fwht.py`
- **Triton GEMV**: `polarengine_vllm/kernels/polar_gemv.py`
- **Config system**: `polarengine_vllm/config.py`
- **Weight converter**: `polarengine_vllm/weight_converter.py`
- **PolarQuant unified notebook**: PolarQuantKVCache class with full BitPacker
- **arXiv paper**: `https://arxiv.org/abs/2603.29078`
- **GitHub**: `https://github.com/caiovicentino/eoq-quantization`

## Output

Tell the user:
1. Files generated and their locations
2. Model specs (head_dim, num_layers, num_kv_heads, attention type)
3. Which head_dims are PolarQuant-compatible (power of 2)
4. Estimated memory savings (e.g., "Q3 KV: 5.3x compression → 64K context in 4 GB")
5. Next steps (integration with existing PolarEngine plugin, testing)
