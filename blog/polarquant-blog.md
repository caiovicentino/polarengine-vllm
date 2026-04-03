---
title: "PolarQuant: Running Gemma 4 31B on a RTX 4090 with Hadamard-Rotated Quantization"
thumbnail: /blog/assets/polarquant/thumbnail.png
authors:
- user: caiovicentino1
date: 2026-04-03
tags:
- quantization
- inference
- gemma
- llm
- kv-cache
---

# PolarQuant: Running Gemma 4 31B on a RTX 4090

**TL;DR**: We fit Google's Gemma 4 31B-it (62.5 GB in BF16) into 21.5 GB using PolarQuant — a quantization method based on Walsh-Hadamard rotation and Lloyd-Max optimal centroids. It runs on consumer GPUs (RTX 4090, L4) with full chat quality and 5.3x longer context via compressed KV cache.

## The Problem

Large language models keep getting bigger. Gemma 4 31B needs 62.5 GB just to load in BF16 — that's more than any consumer GPU can handle. Even an A100 40GB can't fit it.

Standard INT4 quantization (torchao, BnB NF4) can reduce this, but they lose quality because they quantize weights directly without accounting for the statistical structure of the data.

## The Idea: Rotate First, Then Quantize Optimally

PolarQuant is based on a simple mathematical insight:

1. **Walsh-Hadamard rotation** decorrelates the weight values within each block of 128 elements. After rotation, the values follow an approximately Gaussian distribution.

2. **Lloyd-Max quantization** is the provably optimal quantizer for Gaussian distributions — it minimizes mean squared error.

By combining these two steps, we get better reconstruction quality at the same bit-width than methods that quantize raw weights directly.

```
Standard: weight → round to nearest INT4 → store
PolarQuant: weight → Hadamard rotate → normalize → Lloyd-Max Q5 → dequant → INT4
```

The extra rotation + optimal centroids step takes only ~25 seconds for a 31B model and requires no calibration data.

## Results

### Weight Compression (Qwen3.5-9B, WikiText-2 PPL)

| Method | Bits | PPL | Δ vs FP16 |
|--------|------|-----|-----------|
| FP16 Baseline | 16 | 6.37 | — |
| **PolarQuant Q5+INT4** | **~4** | **6.54** | **+0.17** |
| torchao INT4 | 4 | 6.68 | +0.31 |
| BnB NF4 | 4 | ~6.7 | +0.33 |

PolarQuant beats torchao by 0.14 PPL — the gap between a 7B and 9B model in quality.

### Gemma 4 31B-it on Consumer GPUs

| Metric | Value |
|--------|-------|
| **VRAM** | 21.5 GB (text) / 21.9 GB (multimodal) |
| **Speed** | 24.9 tok/s |
| **Min GPU** | RTX 4090 / L4 (24 GB) |
| **Original size** | 62.5 GB (BF16) |
| **Compression** | 2.9x |

### KV Cache Compression

PolarQuant also compresses the KV cache — the memory that grows with context length:

| Method | Compression | Max Context (4GB budget) |
|--------|-------------|--------------------------|
| FP16 | 1.0x | 4K tokens |
| **PolarQuant Q3** | **5.3x** | **22K tokens** |
| PolarQuant Q2 | 8.0x | 33K tokens |

## How the Streaming Loader Works

The key innovation for fitting on 24GB GPUs is the **per-module streaming loader**:

```python
# Never loads full BF16 on GPU!
model = AutoModelForMultimodalLM.from_pretrained(MODEL, device_map='cpu')

for name, child in model.named_modules():
    if isinstance(child, nn.Linear):
        # 1. Move one layer to GPU
        w = child.weight.data.float().to('cuda')
        # 2. PQ5 quantize + dequant on GPU (fast!)
        # 3. INT4 via nn.Sequential wrapper
        # 4. Delete BF16, keep INT4
```

Peak VRAM is just the accumulated INT4 weights (~21 GB), never the full BF16 model.

## Try It

### CLI
```bash
pip install polarquant[all]

# Info about any model
polarquant info google/gemma-4-31B-it

# Start chatting immediately
polarquant chat google/gemma-4-31B-it --vision

# Quantize and upload
polarquant quantize google/gemma-4-31B-it --upload
```

### Python
```python
from transformers import AutoModelForCausalLM
# Coming soon as native transformers integration (issue #45203)
```

### Colab
- [Inference notebook](https://huggingface.co/caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5/blob/main/POLARQUANT_GEMMA4_31B_INFERENCE.ipynb) — Gradio chat, streaming loader
- [Quantization notebook](https://huggingface.co/caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5/blob/main/POLARQUANT_UNIFIED_GEMMA4_31B.ipynb) — Full pipeline

## Models Available

36 models quantized with PolarQuant, including:

| Model | VRAM | Speed | Pipeline |
|-------|------|-------|----------|
| [Gemma 4 31B-it](https://huggingface.co/caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5) | 21.5 GB | 24.9 tok/s | text-generation |
| [Gemma 4 31B-it Vision](https://huggingface.co/caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5-Vision) | 21.9 GB | 24.9 tok/s | image-text-to-text |
| [Qwen3.5 9B](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-Q5) | 6.5 GB | 43.1 tok/s | text-generation |
| [Qwen3.5 27B](https://huggingface.co/caiovicentino1/Qwen3.5-27B-PolarQuant-Q5) | 17.7 GB | 22 tok/s | text-generation |

Full collection: [PolarQuant Models](https://huggingface.co/collections/caiovicentino1/polarquant-models-69cbc96292c5174df2088b08)

## Citation

```bibtex
@article{polarquant2025,
  title={PolarQuant: Hadamard-Rotated Lloyd-Max Quantization for LLM Compression},
  author={Vicentino, Caio},
  journal={arXiv preprint arXiv:2603.29078},
  year={2025},
  url={https://arxiv.org/abs/2603.29078}
}
```

## Links

- 📄 [Paper](https://arxiv.org/abs/2603.29078)
- 💻 [GitHub](https://github.com/caiovicentino/polarengine-vllm)
- 📦 [PyPI](https://pypi.org/project/polarquant/)
- 🤗 [Models](https://huggingface.co/collections/caiovicentino1/polarquant-models-69cbc96292c5174df2088b08)
- 🔧 [Claude Code Skills](https://huggingface.co/caiovicentino1/polarquant-skills)
