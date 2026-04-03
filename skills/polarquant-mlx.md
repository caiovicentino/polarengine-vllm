# PolarQuant MLX Converter

Generate a Python script that converts a HuggingFace model to PolarQuant MLX 4-bit for Apple Silicon, runs PPL benchmark, and uploads to HF.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `Jackrong/Qwopus3.5-9B-v3`
- `https://huggingface.co/Qwen/Qwen3.5-9B`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check the model page for: parameter count, architecture, license, model_type in config.json.

2. **Estimate memory**: params × 2 bytes for BF16. Must fit in Mac unified memory (16/32/64 GB). The MLX 4-bit output will be ~params × 0.5 bytes.

3. **Generate the script**: Write to `/Volumes/SSD Major/fish/convert_mlx_{short_name}.py`.

## Script Structure

```python
#!/usr/bin/env python3
"""PolarQuant MLX Converter — {MODEL_NAME}"""

import os
os.environ['HF_HOME'] = '/Volumes/SSD Major/fish/.hf_cache'
os.environ['TMPDIR'] = '/Volumes/SSD Major/fish/.tmp'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TMPDIR'], exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, gc, time, json, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import norm
from huggingface_hub import HfApi, login

MODEL = '{owner/model-name}'
HF_TOKEN = 'YOUR_HF_TOKEN'
WORK_DIR = '/Volumes/SSD Major/fish/polar_mlx_{short_name}'
DEQUANT_DIR = os.path.join(WORK_DIR, 'dequanted_bf16')
MLX_DIR = os.path.join(WORK_DIR, 'mlx_4bit')
REPO_ID = 'caiovicentino1/{Model-Name}-PolarQuant-MLX-4bit'
BS = 128

os.makedirs(DEQUANT_DIR, exist_ok=True)
os.makedirs(MLX_DIR, exist_ok=True)
login(token=HF_TOKEN)
```

### Step 1: Load BF16 + PolarQuant Q5 dequant (CPU)

**Include (copy exactly):**
- `get_centroids(bits)` — Lloyd-Max, precompute [2,3,4,5,6]
- `_build_H(n)` + `H128 = _build_H(128)` (CPU, no .to(DEVICE))
- `get_bits_q5(name, param)` — skip norms, biases, router, SSM tensors

**Load model:**
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True
)
```

**PolarQuant Q5 dequant loop (CPU):**
```python
for name, module in list(model.named_modules()):
    for child_name, child in list(module.named_children()):
        if not isinstance(child, nn.Linear) or child.weight.numel() < 256: continue
        full = f'{name}.{child_name}' if name else child_name
        bits = get_bits_q5(full + '.weight', child.weight)
        if bits >= 16: continue
        # ... quantize + dequant with CORRECT reshape:
        values = ct[all_codes.long()] / math.sqrt(BS)
        for i in range(0, out_f, 64):
            end = min(i + 64, out_f)
            v = values[i:end].reshape(-1, BS)
            values[i:end] = (v @ H128).reshape(end - i, n_blocks, BS)
        values = values * norms.unsqueeze(2)
        child.weight.data = values.reshape(out_f, -1)[:, :in_f].to(torch.bfloat16)
```

Print progress every 50 layers.

### Step 2: Save dequanted BF16

```python
model.save_pretrained(DEQUANT_DIR, safe_serialization=True)
tokenizer.save_pretrained(DEQUANT_DIR)
del model; gc.collect()
```

### Step 3: Fix config.json model_type

**CRITICAL for Qwen3.5 hybrid models:**
```python
import json
cfg_path = os.path.join(DEQUANT_DIR, 'config.json')
with open(cfg_path) as f:
    cfg = json.load(f)
# Fix model_type: qwen3_5_text → qwen3_5 (mlx-lm doesn't support _text suffix)
if cfg.get('model_type', '').endswith('_text'):
    cfg['model_type'] = cfg['model_type'].replace('_text', '')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Fixed model_type: {cfg["model_type"]}')
```

### Step 4: Convert to MLX 4-bit

```python
ret = os.system(
    f'python3 -m mlx_lm convert '
    f'--hf-path "{DEQUANT_DIR}" '
    f'--mlx-path "{MLX_DIR}" '
    f'--quantize --q-bits 4 --q-group-size 64'
)
```

If `mlx_lm` uses old CLI format, try:
```python
ret = os.system(
    f'python3 -m mlx_lm.convert '
    f'--hf-path "{DEQUANT_DIR}" '
    f'--mlx-path "{MLX_DIR}" '
    f'--quantize --q-bits 4 --q-group-size 64'
)
```

### Step 5: Test generation

```python
os.system(
    f'python3 -m mlx_lm generate '
    f'--model "{MLX_DIR}" '
    f'--prompt "What is 2+3? Think step by step." '
    f'--max-tokens 100'
)
```

Or with `mlx_lm.generate` (new CLI format).

### Step 6: PPL benchmark (MLX)

```python
import mlx.core as mx
import mlx.nn as mnn
from mlx_lm import load
from datasets import load_dataset

model, tokenizer = load(MLX_DIR)
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
wiki = '\n\n'.join([t for t in ds['text'] if t.strip()])[:150000]
ids = tokenizer.encode(wiki)

nlls = []; total = 0; t0 = time.time()
seq_len = 2048; stride = 512; mask_len = 1536

for i in range(0, min(len(ids) - seq_len, 15000), stride):
    chunk = mx.array(ids[i:i+seq_len]).reshape(1, -1)
    logits = model(chunk)
    shift_logits = logits[:, mask_len-1:-1, :]
    shift_labels = chunk[:, mask_len:]
    log_probs = mnn.log_softmax(shift_logits, axis=-1)
    gather_idx = mx.expand_dims(shift_labels, axis=-1)
    token_log_probs = mx.take_along_axis(log_probs, gather_idx, axis=-1).squeeze(-1)
    nll = -mx.sum(token_log_probs).item()
    nlls.append(nll)
    total += seq_len - mask_len
    if ((i // stride) + 1) % 10 == 0:
        print(f'  {total} tok | PPL: {math.exp(sum(nlls)/total):.2f} | {time.time()-t0:.0f}s')

final_ppl = math.exp(sum(nlls) / total)
print(f'PPL = {final_ppl:.4f}')
```

### Step 7: Upload to HF with pro model card

```python
api = HfApi()
api.create_repo(REPO_ID, repo_type='model', exist_ok=True)
```

**Model card (adapt for this model):**

```markdown
# 🍎 PolarQuant MLX 4-bit — {MODEL_NAME}

**PolarQuant Q5 dequant → MLX 4-bit** for Apple Silicon inference.

PPL **{ppl}** — Mac mini M4 16GB, {tok_s} tok/s, {mem} GB peak.

## 🎯 Key Results
| Metric | Value |
|---|---|
| **Perplexity** | **{ppl}** |
| **Speed** | **{tok_s} tok/s** (Mac mini M4 16GB) |
| **Memory** | **{mem} GB** peak |
| **Size** | **{size} GB** |

## 🚀 Quick Start
\```bash
pip install mlx-lm
\```
\```python
from mlx_lm import load, generate
model, tokenizer = load("{REPO_ID}")
response = generate(model, tokenizer, prompt="...", max_tokens=500)
\```

## 🔧 How It Was Made
Base BF16 → PolarQuant Q5 dequant (Hadamard + Lloyd-Max) → MLX 4-bit (group_size=64)

## 🔗 Resources
- 🧊 [CUDA version](https://huggingface.co/caiovicentino1/{Model-Name}-PolarQuant-Q5)
- 📄 [GitHub](https://github.com/caiovicentino/eoq-quantization)

## 📖 Citation
\```bibtex
@misc{polarquant2025, ...}
\```
```

**Upload folder + add to collections:**
```python
api.upload_folder(folder_path=MLX_DIR, repo_id=REPO_ID, repo_type='model')

# Add to both MLX and unified collections
for slug in [
    'caiovicentino1/polarquant-mlx-apple-silicon-69cbd314b96ed3a93085efb0',
    'caiovicentino1/polarquant-unified-weights-q5-kv-cache-q3-69cc5b78d516b9de96a0205b',
    'caiovicentino1/polarquant-models-69cbc96292c5174df2088b08',
]:
    try:
        api.add_collection_item(collection_slug=slug, item_id=REPO_ID, item_type='model')
    except: pass
```

### Step 8: Cleanup

```python
shutil.rmtree(DEQUANT_DIR, ignore_errors=True)
```

## Key Patterns (MUST FOLLOW)

- **HF_HOME on SSD**: `os.environ['HF_HOME'] = '/Volumes/SSD Major/fish/.hf_cache'` — system disk is small (228 GB), cache fills it fast.
- **CPU dequant**: No GPU on Mac for PyTorch. Use `device_map='cpu'`. Slower (~3 min for 9B) but works.
- **model_type fix**: Qwen3.5 hybrid saves as `qwen3_5_text` but mlx-lm only supports `qwen3_5`. Strip `_text` suffix.
- **BF16 native**: Load with `dtype=torch.bfloat16`. Never FP16.
- **Dequant reshape**: `(v @ H128).reshape(end-i, n_blocks, BS)`. NEVER the buggy `values.view(out_f, n_blocks*BS, BS)`.
- **group_size=64**: MLX with group_size=64 gives better PPL than torchao group_size=128. This is why MLX PPL can beat CUDA.
- **Use python3.12**: System python3 is 3.9.6, too old for transformers. Use `/opt/homebrew/bin/python3.12`.
- **Run with**: `cd "/Volumes/SSD Major/fish" && /opt/homebrew/bin/python3.12 convert_mlx_{name}.py`
- **Hardware**: Mac mini M4 16GB for benchmarks. Specify in model card.
- **Collections**: Add to MLX, Unified, and PolarQuant Models collections.
- **arXiv**: `https://arxiv.org/abs/2603.7424577`
- **GitHub**: `https://github.com/caiovicentino/eoq-quantization`

## Expected Results (reference from Qwopus3.5-9B-v3)

| Platform | Method | PPL | tok/s | Memory | Size |
|---|---|---|---|---|---|
| Mac mini M4 | MLX 4-bit | 6.44 | 20.7 | 5.1 GB | 4.7 GB |
| RTX PRO 6000 | PolarQuant Q5 + INT4 | 6.48 | 43 | 7.1 GB | 9.1 GB |

MLX 4-bit on PolarQuant dequanted weights beats CUDA torchao on PPL (6.44 vs 6.48) at lower memory (5.1 vs 7.1 GB).

## Output

Tell the user:
1. Script path
2. Model specs (params, architecture)
3. Memory requirement (must fit in Mac RAM)
4. Run command: `cd "/Volumes/SSD Major/fish" && /opt/homebrew/bin/python3.12 convert_mlx_{name}.py`

## Argument: $ARGUMENTS
