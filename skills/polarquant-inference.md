# PolarQuant Inference Notebook Generator

Generate a user-facing Colab notebook (.ipynb) that loads a PolarQuant model and provides a Gradio chat UI. Designed for end-users, not researchers — simple, fast, one-click.

## Input

The user provides a model name, HuggingFace URL, or PolarQuant repo. Examples:
- `google/gemma-4-31B-it`
- `caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5`
- `https://huggingface.co/caiovicentino1/Qwen3.5-9B-PolarQuant-Q5`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check the model page for: parameter count, architecture, base model name, head_dim, num_layers, num_kv_heads, chat vs base model, license.

2. **Determine base model**: If a PolarQuant repo is given, find `base_model` from config.json or polar_config.json. If a base model is given, the PolarQuant repo is `caiovicentino1/{Model-Name}-PolarQuant-Q5`.

3. **Estimate VRAM**: INT4 model size ≈ params × 0.6 bytes + embeddings. Pick GPU tier.

4. **Generate the Colab notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_INFERENCE_{SHORT_NAME}.ipynb`.

## Notebook Structure (7 cells)

### Cell 0: Markdown header
```markdown
# 🧊 {Model-Name} — PolarQuant Q5+INT4 Inference

**Run {Model-Name} on consumer GPUs** with PolarQuant full-stack compression.

| Component | Method | Effect |
|---|---|---|
| **Weights** | PolarQuant Q5 → torchao INT4 | BF16 → ~3x reduction |
| **KV Cache** | PolarQuant Q3 (Hadamard + Lloyd-Max) | 5.3x longer context |

| GPU | VRAM | Status |
|---|---|---|
| {GPU table based on model size} |

📄 [Paper](https://arxiv.org/abs/2603.29078) · 💻 [GitHub](https://github.com/caiovicentino/eoq-quantization) · 🤗 [Model]({PQ_REPO_URL})
```

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q accelerate safetensors sentencepiece scipy torchao gradio
```
**NEVER include flash-attn.**

### Cell 2: Imports + GPU detection
```python
import torch, math, gc, time, os
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as sp_norm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

DEVICE = 'cuda'
MODEL = '{base_model}'  # Base model for architecture
BS = 128
HEAD_DIM = {head_dim}  # From model config

# GPU auto-detection
gpu_name = torch.cuda.get_device_name(0)
gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu_name} ({gpu_vram:.0f} GB)')
```

### Cell 3: PolarQuant Core Code
Include ALL of these (same as /polarquant skill):
- `get_centroids(bits)` — Lloyd-Max centroids
- `_build_H(n)` — Hadamard matrix
- `BitPacker` class — 2/3/4 bit packing
- `PolarQuantLayer` class — KV cache layer
- `_HybridCacheLayer` — hybrid model support
- `PolarQuantKVCache` class — full cache manager with `is_initialized` property
- `get_bits_q5()` — weight bit assignment
- torchao guard patch

**CRITICAL adaptations for the specific model:**
- Set `HEAD_DIM` correctly (128 for Llama/Qwen, 256 for Gemma 4)
- Build `H_W = _build_H(BS)` for weights and `H_KV = _build_H(HEAD_DIM)` for KV cache
- Use `.bfloat16()` NOT `.half()` in KV cache quantize/dequantize
- Override `is_initialized` property: `return self._seen_tokens > 0`
- `apply_chat_template` returns BatchEncoding — extract `.input_ids`

### Cell 4: Streaming Model Loader (THE KEY CELL)

**This is what makes it fit on consumer GPUs.** Per-module loading:

1. **Load BF16 on CPU** — `device_map='cpu'`
2. **For each nn.Linear**: move weight to GPU → PQ5 quantize+dequant → INT4 via `nn.Sequential` wrapper
3. **Move non-quantized params** (norms, embeddings) to GPU
4. **Peak VRAM**: accumulated INT4 only (~60% of param count in bytes)

```python
# Streaming pattern:
for name, child in list(model.named_modules()):
    if not isinstance(child, nn.Linear): continue
    if child.weight.device.type == 'meta': continue
    if get_bits_q5(name, child.weight) == 16: continue

    # GPU-accelerated dequant
    w = child.weight.data.float().to(DEVICE)
    # ... PQ5 quantize+dequant on GPU (fast, ~25s total) ...
    bf16_weight = result.to(torch.bfloat16)

    # Per-module INT4 via vLLM pattern
    with torch.device('meta'):
        dummy = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    dummy[0].weight = nn.Parameter(bf16_weight)
    quantize_(dummy, Int4WeightOnlyConfig(group_size=128))
    child.weight = dummy[0].weight

    del dummy, bf16_weight; torch.cuda.empty_cache()
```

**CRITICAL patterns:**
- Chunked argmin (QC=256) to avoid OOM on large layers
- In-place normalize: `w.div_(norms).mul_(math.sqrt(BS))`
- `del w; torch.cuda.empty_cache()` after each layer
- `nn.Sequential` wrapper for torchao module swap compatibility
- Move remaining params with: `for name, param in model.named_parameters(): if param.device.type == 'cpu': param.data = param.data.to(DEVICE)`

### Cell 5: Quick Sanity Test
```python
msgs = [{'role': 'user', 'content': 'Hello! What is 2+2?'}]
chat_out = tokenizer.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True)
test_ids = chat_out['input_ids'].to(DEVICE) if hasattr(chat_out, 'input_ids') else chat_out.to(DEVICE)
with torch.no_grad():
    out = model.generate(test_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0][test_ids.shape[1]:], skip_special_tokens=True))
```

### Cell 6: Gradio Chat UI

**MUST use `model.generate()` + `TextIteratorStreamer`** — NOT manual argmax loop (degenerates into repetition). model.generate() handles stop tokens from generation_config.json correctly.

```python
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread

@torch.no_grad()
def chat_fn(message, history):
    messages = list(history) + [{'role': 'user', 'content': message}]
    chat_out = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True)
    input_ids = chat_out['input_ids'].to(DEVICE) if hasattr(chat_out, 'input_ids') else chat_out.to(DEVICE)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    partial = ''
    for text in streamer:
        partial += text
        yield partial
    thread.join()

demo = gr.ChatInterface(
    chat_fn,
    title='🧊 {Model-Name} — PolarQuant Q5+INT4',
    description=f'VRAM: {torch.cuda.memory_allocated()/1e9:.0f} GB | temp=0.7 top_p=0.9',
    examples=[...],
    type='messages',
)
demo.launch(share=True, quiet=True)
```

**CRITICAL Gradio patterns:**
- `type='messages'` (NOT 'tuples' — deprecated)
- `TextIteratorStreamer` for streaming (NOT manual token loop)
- `model.generate()` handles stop tokens (`<end_of_turn>`, EOS, etc.)
- `repetition_penalty=1.3` prevents degenerate repetition
- `share=True` generates public URL

## Key Patterns (MUST FOLLOW)

- **BF16 native**: `dtype=torch.bfloat16` everywhere. NEVER `.half()`.
- **No flash-attn**: Use `attn_implementation='sdpa'`.
- **Streaming loader**: Load BF16 on CPU → per-module GPU dequant + INT4. Peak VRAM = final INT4 model only.
- **`nn.Sequential` wrapper**: Required for torchao `quantize_()` per-module (module swap issue).
- **`apply_chat_template`**: Returns BatchEncoding in newer transformers — always extract `.input_ids`.
- **`model.generate()`**: Use instead of manual loop for proper stop token handling.
- **GPU detection**: `torch.cuda.get_device_properties(0).total_memory` (NOT `total_mem`).
- **torchao INT4 tensors**: Can't move to CPU (`TensorCoreTiledAQTTensorImpl` GPU-only). Save with `torch.save()` not safetensors.
- **Instruct models**: WikiText-2 PPL is meaningless (Gemma 4 BF16 baseline = 1002). Skip PPL, show generation quality.
- **arXiv**: `https://arxiv.org/abs/2603.29078`
- **GitHub**: `https://github.com/caiovicentino/eoq-quantization`

## GPU Tier Table (by model size)

| Model Size | INT4 VRAM | Min GPU |
|---|---|---|
| ≤ 4B | ~3 GB | T4 (16 GB) |
| 7-9B | ~6-7 GB | T4 (16 GB) |
| 13-14B | ~9-10 GB | T4 (16 GB) |
| 27-31B | ~18-22 GB | L4/RTX 4090 (24 GB) |
| 70B | ~40-45 GB | A100 80 GB |

## Output

Tell the user:
1. File path of the notebook
2. Model specs (params, head_dim, layers, chat/base)
3. Estimated VRAM with streaming loader
4. Minimum GPU required
5. Gradio features included

## Argument: $ARGUMENTS
