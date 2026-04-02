# PolarQuant KV Cache for vLLM

**5.3x KV cache compression** via Hadamard-rotated Lloyd-Max quantization.

## Architecture

```
Standard KV Cache (FP16):
  K, V → [16 bits per value] → PagedAttention → Output

PolarQuant KV Cache (Q3):
  K, V → Hadamard → Normalize → Lloyd-Max Q3 → BitPack → [3 bits per value]
       → Unpack → Lookup → Inverse Hadamard → Denormalize → PagedAttention → Output
```

## Quick Start

```python
from polarengine_vllm.kv_cache import PolarKVConfig, PolarKVCache

# For Gemma 4 31B-it (head_dim=256)
config = PolarKVConfig.for_gemma4_31b(nbits=3)
cache = PolarKVCache(config)

# Update cache per layer
full_k, full_v = cache.update(key_states, value_states, layer_idx=0)

# Check stats
print(cache.stats())
# {'seq_length': 1024, 'memory_mb': 45.2, 'fp16_memory_mb': 240.0, 'compression_ratio': 5.3}
```

## Supported Models

| Model | head_dim | Layers | KV Heads | Status |
|-------|----------|--------|----------|--------|
| Gemma 4 31B | 256 | 60 | 16 | ✅ Tested |
| Llama 3 8B | 128 | 32 | 8 | ✅ |
| Llama 3 70B | 128 | 80 | 8 | ✅ |
| Qwen3.5 9B | 128 | 48 | 8 | ✅ |
| Qwen3.5 27B | 128 | 48 | 4 | ✅ |

**Requirement**: head_dim must be a power of 2 (for Walsh-Hadamard transform).

## Compression

| Method | Bits | Compression | Quality (cos_sim) |
|--------|------|-------------|-------------------|
| FP16 | 16 | 1.0x | 1.000 |
| Q4 | 4 | 4.0x | 0.995 |
| **Q3** | **3** | **5.3x** | **0.983** |
| Q2 | 2 | 8.0x | 0.950 |

## Context Length (4 GB KV budget, Gemma 4 31B)

| Method | Max Context |
|--------|-------------|
| FP16 | ~4K tokens |
| Q3 | ~22K tokens |
| Q2 | ~33K tokens |

## Benchmark

```bash
python -m polarengine_vllm.kv_cache.benchmark --preset gemma4-31b
python -m polarengine_vllm.kv_cache.benchmark --head-dim 128 --nbits 3
```

## Tests

```bash
pytest polarengine_vllm/kv_cache/test_polar_kv.py -v
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration + model presets |
| `cache.py` | Core: quantizer, layer cache, full cache manager |
| `attention.py` | vLLM attention wrapper + transformers Cache API |
| `triton_kernels.py` | Optimized ops (hybrid torch.matmul + Triton) |
| `benchmark.py` | Latency, quality, memory benchmarks |
| `test_polar_kv.py` | Unit tests |

## Citation

```bibtex
@article{polarquant2025,
  title={PolarQuant: Hadamard-Rotated Lloyd-Max Quantization for LLM Compression},
  author={Vicentino, Caio},
  journal={arXiv preprint arXiv:2603.29078},
  year={2025}
}
```
