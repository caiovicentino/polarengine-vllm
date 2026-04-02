# PolarQuant Q3 — Hadamard-Rotated KV Cache Quantization

PolarQuant Q3 (`pq3_0`) is a 3-bit KV cache quantization type that uses Walsh-Hadamard rotation before Lloyd-Max optimal quantization, achieving better reconstruction quality than standard 3-bit approaches at the same compression ratio.

## Usage

```bash
./llama-cli -m model.gguf \
    --cache-type-k pq3_0 \
    --cache-type-v pq3_0 \
    -c 131072
```

## How It Works

### Standard 3-bit (Q3_K)
```
weight → find scale → round to nearest 3-bit value → store
```

### PolarQuant Q3 (pq3_0)
```
weight → compute L2 norm → normalize → Walsh-Hadamard rotate →
Lloyd-Max optimal quantize → 3-bit bit-pack → store
```

The Walsh-Hadamard transform **decorrelates** the values within each block of 128 elements. After rotation, the distribution becomes more Gaussian, making Lloyd-Max quantization optimal. This preserves more information than simple round-to-nearest.

### Block Structure

Each block stores 128 values in 50 bytes (3.125 bits per value):

| Field | Size | Description |
|-------|------|-------------|
| `d` | 2 bytes | FP16 L2 norm of the block |
| `qs` | 48 bytes | 128 × 3-bit codes, bit-packed |
| **Total** | **50 bytes** | **3.125 bits/value** |

### Lloyd-Max Centroids

The 8 reconstruction levels are optimal for the standard normal distribution:

```
[-2.152, -1.344, -0.756, -0.245, +0.245, +0.756, +1.344, +2.152]
```

### Fast Walsh-Hadamard Transform

The transform is computed in-place using the butterfly algorithm in O(n log n) time — 7 stages for n=128. The transform is self-inverse, so the same function is used for both quantization (forward) and dequantization (inverse).

## Compression Comparison

| Cache Type | Bits/Value | Compression | Quality (cos_sim) |
|------------|-----------|-------------|-------------------|
| f16 | 16 | 1x | 1.000 |
| q8_0 | 8.5 | 1.9x | 0.999+ |
| q4_0 | 4.5 | 3.6x | 0.99+ |
| **pq3_0** | **3.125** | **5.1x** | **0.983** |
| q4_1 | 4.5 | 3.6x | 0.99+ |

## Context Length Impact

With PolarQuant Q3, the same GPU memory supports ~5x longer context:

| GPU | FP16 KV Context | PQ3_0 KV Context |
|-----|-----------------|-------------------|
| RTX 4090 (24 GB) | ~32K tokens | ~160K tokens |
| RTX 3090 (24 GB) | ~32K tokens | ~160K tokens |
| A100 (80 GB) | ~128K tokens | ~640K tokens |

## Technical Reference

- **Paper**: [PolarQuant: Hadamard-Rotated Lloyd-Max Quantization](https://arxiv.org/abs/2603.7424577)
- **Implementation**: [github.com/caiovicentino/polarengine-vllm](https://github.com/caiovicentino/polarengine-vllm)
- **GGML Type**: `GGML_TYPE_PQ3_0` (index 41)
- **Block size**: 128 (`QK_PQ3_0`)
