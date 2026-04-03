# PolarQuant GGUF Converter

Generate a Colab notebook (.ipynb) that converts a PolarQuant model to GGUF format for ollama/llama.cpp users. Produces multiple quantization levels (Q4_K_M, Q5_K_M, Q8_0) and uploads to HuggingFace.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `google/gemma-4-31B-it`
- `caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check: parameter count, architecture, license, chat template format.

2. **Determine base model**: If PolarQuant repo given, find `base_model` from config. If base model given, use directly.

3. **Generate notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_GGUF_{SHORT_NAME}.ipynb`.

## Notebook Structure (7 cells)

### Cell 0: Markdown header
```markdown
# PolarQuant → GGUF: {Model-Name}

Convert PolarQuant Q5 model to GGUF format for **ollama** and **llama.cpp**.

Pipeline: PolarQuant Q5 codes → Dequant to BF16 → llama.cpp convert → GGUF quantize
```

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q accelerate safetensors sentencepiece scipy

# Clone llama.cpp for conversion tools
!git clone https://github.com/ggerganov/llama.cpp.git /content/llama.cpp
!cd /content/llama.cpp && pip install -r requirements/requirements-convert_hf_to_gguf.txt -q
```

### Cell 2: Load + PolarQuant Q5 Dequant to BF16

**Flow:**
1. Load base model in BF16 on CPU: `device_map='cpu'`
2. PQ5 quantize+dequant each nn.Linear on GPU (per-layer, streaming)
3. Result: BF16 model with PolarQuant-conditioned weights on CPU

```python
# Move one layer at a time to GPU for fast dequant, then back to CPU
for name, child in model.named_modules():
    if not isinstance(child, nn.Linear): continue
    w = child.weight.data.float().to('cuda')
    # PQ5 quantize + dequant on GPU (fast)
    # ...
    child.weight.data = result.to('cpu').to(torch.bfloat16)
    del w; torch.cuda.empty_cache()
```

**Do NOT apply torchao INT4** — we need BF16 weights for GGUF conversion.

### Cell 3: Save BF16 model in HuggingFace format

```python
SAVE_DIR = '/content/model_bf16_for_gguf'
model.save_pretrained(SAVE_DIR, safe_serialization=True, max_shard_size='5GB')
tokenizer.save_pretrained(SAVE_DIR)
```

### Cell 4: Convert to GGUF

```python
# Convert HF model to GGUF F16
!python /content/llama.cpp/convert_hf_to_gguf.py {SAVE_DIR} \
    --outfile /content/{short_name}-f16.gguf \
    --outtype f16

# Quantize to multiple formats
quantize_bin = '/content/llama.cpp/build/bin/llama-quantize'

# Build llama.cpp if quantize binary doesn't exist
import os
if not os.path.exists(quantize_bin):
    !cd /content/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc) --target llama-quantize

for quant_type in ['Q4_K_M', 'Q5_K_M', 'Q8_0']:
    input_gguf = f'/content/{short_name}-f16.gguf'
    output_gguf = f'/content/{short_name}-{quant_type}.gguf'
    !{quantize_bin} {input_gguf} {output_gguf} {quant_type}
    size_gb = os.path.getsize(output_gguf) / 1e9
    print(f'{quant_type}: {size_gb:.1f} GB')
```

### Cell 5: Test with llama.cpp

```python
# Build llama-cli if needed
!cd /content/llama.cpp && cmake --build build --config Release -j$(nproc) --target llama-cli

# Quick test with Q4_K_M
!cd /content/llama.cpp && ./build/bin/llama-cli \
    -m /content/{short_name}-Q4_K_M.gguf \
    -p "Hello! What is 2+2?" \
    -n 100 --temp 0.7 --top-p 0.9 --repeat-penalty 1.3 \
    -ngl 99
```

### Cell 6: Upload to HuggingFace

```python
from huggingface_hub import HfApi, login
login(token=userdata.get('HF_TOKEN'))
api = HfApi()

REPO = 'caiovicentino1/{Model-Name}-GGUF'
api.create_repo(REPO, exist_ok=True)

# Upload each GGUF file
for quant in ['Q4_K_M', 'Q5_K_M', 'Q8_0']:
    gguf_path = f'/content/{short_name}-{quant}.gguf'
    if os.path.exists(gguf_path):
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=f'{short_name}-{quant}.gguf',
            repo_id=REPO,
            repo_type='model',
        )
        print(f'Uploaded {quant}')

# Model card with ollama instructions
card = f"""---
license: {license}
tags:
- gguf
- polarquant
- quantized
- {architecture}
base_model: {base_model}
---

# {Model-Name} GGUF (PolarQuant-conditioned)

GGUF quantizations from PolarQuant Q5 dequantized weights.

## Quick Start (ollama)

```bash
ollama run hf.co/{REPO}:{short_name}-Q4_K_M
```

## Available Quantizations

| Quant | Size | Use Case |
|-------|------|----------|
| Q4_K_M | X GB | Best balance (recommended) |
| Q5_K_M | X GB | Higher quality |
| Q8_0 | X GB | Near-lossless |

## Why PolarQuant-conditioned?

These GGUFs are created from PolarQuant Q5 dequantized weights, not raw BF16.
PolarQuant's Hadamard rotation + Lloyd-Max quantization conditions the weights
to be more quantization-friendly, potentially improving GGUF quality.

📄 [Paper](https://arxiv.org/abs/2603.29078) · 💻 [GitHub](https://github.com/caiovicentino/eoq-quantization)
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo='README.md',
                repo_id=REPO, repo_type='model')
```

## Key Patterns

- **PQ5 dequant BEFORE GGUF** — conditions weights for better quantization
- **Do NOT apply torchao INT4** — GGUF needs BF16 input
- **GPU-accelerated dequant** — move one layer to GPU, dequant, move back to CPU
- **Multiple quant levels** — Q4_K_M (recommended), Q5_K_M, Q8_0
- **Build llama.cpp from source** — Colab has cmake and make available
- **Test with llama-cli** — verify output before uploading
- **Ollama-compatible** — users can `ollama run hf.co/repo:quant` directly

## Output

Tell the user:
1. File path of the notebook
2. GGUF sizes for each quantization level
3. Ollama command to run the model
4. HF repo URL for the GGUF files

## Argument: $ARGUMENTS
