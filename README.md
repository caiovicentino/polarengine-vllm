> [!IMPORTANT]
> **Naming notice (2026-04-10).** The "PolarQuant" technique referenced throughout this README is being rebranded to **HLWQ (Hadamard-Lloyd Weight Quantization)**. The change is only the name; the algorithm and the code in this repository are unchanged.
>
> The rebrand resolves a name collision with an unrelated, earlier KV cache quantization method also named PolarQuant ([Han et al., arXiv:2502.02617, 2025](https://arxiv.org/abs/2502.02617)). HLWQ addresses **weight** quantization with a **deterministic Walsh-Hadamard rotation** and Lloyd-Max scalar codebook; Han et al.'s PolarQuant addresses **KV cache** quantization with a **random polar transformation**. The two methods are technically distinct.
>
> The PyPI package `polarquant` will be superseded by [`hlwq`](https://pypi.org/project/hlwq/); the `polarquant` package and this repository's name will continue to work during the transition period. Reference paper: [arXiv:2603.29078](https://arxiv.org/abs/2603.29078) (v2 in preparation under the new name).

# PolarEngine for vLLM

Custom quantization plugin for vLLM using PolarQuant -- optimal Gaussian quantization via Walsh-Hadamard rotation + Lloyd-Max centroids.

**arXiv preprint**: [arXiv:2603.29078](https://arxiv.org/abs/2603.29078)

> **Recommended path**: For best quality-per-VRAM, use **PolarQuant Q5 + torchao INT4** (43.1 tok/s, 6.5 GB VRAM, PPL 6.56). PolarEngine's custom Triton kernel is available for environments where torchao is not an option.

---

## Results (Qwen3.5-9B, RTX PRO 6000 Blackwell)

| Method | tok/s | VRAM | PPL (WikiText-2) | Notes |
|--------|-------|------|-------------------|-------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 | Reference |
| **PolarQuant Q5 + torchao INT4** | **43.1** | **6.5 GB** | **6.56** | **Recommended** |
| torchao INT4 (absmax) | 43.3 | 6.3 GB | 6.68 | |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 | |
| PolarEngine v4 (Triton) | 34.2 | 7.9 GB | 6.89 | Custom kernel |
| PolarQuant Q5 dequant FP16 | 45.9 | 18.1 GB | 6.39 | Near-lossless |
| PolarQuant MLX Q4 | 19.7 | 4.8 GB | 6.90 | Mac mini M4 16 GB |

### PolarQuant Ablation (Q5, Qwen3.5-9B)

| Configuration | PPL | Delta vs FP16 |
|---------------|-----|---------------|
| Absmax Q5 (baseline) | 6.9030 | +0.53 |
| + Hadamard rotation | 6.4010 | +0.03 |
| + Lloyd-Max centroids | 6.9139 | +0.54 |
| + Both (PolarQuant Q5) | 6.3909 | +0.02 |

Hadamard rotation accounts for 98% of the improvement. The Walsh-Hadamard transform makes weight distributions approximately Gaussian, enabling near-optimal uniform quantization.

---

## How It Works

PolarQuant quantization:
1. **Normalize** weight blocks by L2 norm
2. **Rotate** via Walsh-Hadamard Transform (makes weights Gaussian -- 98% of quality gain)
3. **Quantize** using Lloyd-Max optimal centroids for N(0,1)
4. **Store** codes (int8/nibble-packed) + per-block norms (fp16)

Inference keeps weights quantized in GPU VRAM:
- Triton kernel does centroid lookup + GEMV in one operation
- FWHT applied to input (not weights) -- 25x faster via matmul
- FWHT cached across Q/K/V projections (69x total speedup)
- INT4 nibble packing for Q3/Q4 layers (36% VRAM savings)

---

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

---

## Quick Start

### Option A: PolarQuant Q5 + torchao (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int4WeightOnlyConfig
import torch

# Load PolarQuant Q5 model (auto-dequantizes to FP16)
model = AutoModelForCausalLM.from_pretrained(
    "caiovicentino1/Qwen3.5-9B-PolarQuant-Q5",
    dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("caiovicentino1/Qwen3.5-9B-PolarQuant-Q5")

# Apply torchao INT4 for fast inference (43 tok/s, 6.5 GB VRAM)
quantize_(model, Int4WeightOnlyConfig(group_size=128))

inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Option B: PolarEngine Triton Kernel

#### 1. Quantize a model
```bash
python -m polarengine_vllm.quantize \
    --model Qwen/Qwen3.5-9B \
    --output ./Qwen3.5-9B-PolarEngine/
```

#### 2. Serve with vLLM
```bash
vllm serve ./Qwen3.5-9B-PolarEngine/ --quantization polarengine
```

#### 3. Use from Python
```python
from vllm import LLM
model = LLM("./Qwen3.5-9B-PolarEngine/", quantization="polarengine")
output = model.generate("Explain quantum computing:")
```

---

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

---

## Published Models

| Model | Link | Notes |
|-------|------|-------|
| Qwen3.5-9B PolarQuant Q5 | [HuggingFace](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-Q5) | Recommended, 9.1 GB |
| Qwen3.5-9B PolarQuant MLX 4-bit | [HuggingFace](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-MLX-4bit) | Apple Silicon |
| Qwen3.5-9B PolarEngine v4 | [HuggingFace](https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarEngine-v4) | Triton kernel |

See the [main EOQ repository](https://github.com/caiovicentino/eoq-quantization) for additional models and full documentation.

---

## Citation

```bibtex
@article{vicentino2026polarquant,
    title={PolarQuant: Near-Lossless LLM Quantization via Walsh-Hadamard Rotation
           and Entropy-Optimal Coding},
    author={Vicentino, Caio},
    journal={arXiv preprint arXiv:2603.29078},
    year={2026}
}
```

## License

Apache 2.0
