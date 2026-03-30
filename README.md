# PolarEngine for vLLM

Custom quantization plugin for vLLM using PolarQuant -- optimal Gaussian quantization with Walsh-Hadamard rotation.

## Results

| Method | tok/s | VRAM | PPL (WikiText-2) |
|--------|-------|------|------------------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 |
| torchao INT4 | 43.3 | 6.3 GB | 6.68 |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 |
| **PolarEngine v4** | **34.2** | **7.9 GB** | **6.89** |

*Benchmarked on Qwen3.5-9B, NVIDIA RTX PRO 6000 Blackwell Server Edition.*

## How It Works

PolarQuant quantization:
1. **Normalize** weight blocks by L2 norm
2. **Rotate** via Walsh-Hadamard Transform (makes weights Gaussian)
3. **Quantize** using Lloyd-Max optimal centroids for N(0,1)
4. **Store** codes (int8/nibble-packed) + per-block norms (fp16)

Inference keeps weights quantized in GPU VRAM:
- Triton kernel does centroid lookup + GEMV in one operation
- FWHT applied to input (not weights) -- 25x faster via matmul
- FWHT cached across Q/K/V projections (69x total speedup)
- INT4 nibble packing for Q3/Q4 layers (36% VRAM savings)

## Installation

```bash
pip install polarengine-vllm
```

Or from source:
```bash
git clone https://github.com/caiovicentino/polarengine-vllm
cd polarengine-vllm
pip install -e .
```

Optional CUDA kernels (for CUDA graph support):
```bash
pip install -e ".[cuda]"
```

## Quick Start

### 1. Quantize a model
```bash
python -m polarengine_vllm.quantize \
    --model Qwen/Qwen3.5-9B \
    --output ./Qwen3.5-9B-PolarEngine/
```

### 2. Serve with vLLM
```bash
vllm serve ./Qwen3.5-9B-PolarEngine/ --quantization polarengine
```

### 3. Use from Python
```python
from vllm import LLM
model = LLM("./Qwen3.5-9B-PolarEngine/", quantization="polarengine")
output = model.generate("Explain quantum computing:")
```

## Mixed-Bit Assignment

| Layer Type | Bits | Rationale |
|-----------|------|-----------|
| gate/up proj (MLP) | Q3 | Tolerant to quantization |
| down proj (MLP) | Q4 | Moderate sensitivity |
| Q/K/V proj (Attention) | Q5 | Higher precision for attention |
| O proj (Attention) | Q6 | Output projection needs quality |
| Embeddings | Q5 | Large, benefits from compression |
| LM Head | Q6 | Critical for token prediction |
| Norms, biases, router | FP16 | Too small to quantize |

## Architecture

```
Input x -> Pad -> FWHT(x) via matmul -> Triton GEMV Kernel -> Output
                  ^                        ^
          H128 (cached, 64KB)    codes + norms + centroids
                                 (quantized, in VRAM)
```

## Citation

```bibtex
@software{eoq2026,
    title={EOQ: Entropy-Optimal Quantization},
    author={Vicentino, Caio},
    url={https://github.com/caiovicentino/eoq-quantization},
    year={2026}
}
```

## License

Apache 2.0
