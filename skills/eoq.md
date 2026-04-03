# EOQ Quantization Colab Generator

Generate a complete Colab notebook that quantizes a HuggingFace model with EOQ Q5, uploads to HF, and runs full benchmarks.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill`
- `https://huggingface.co/Qwen/Qwen3.5-35B-A3B`
- `https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` (GGUF-only repo)

## Instructions

1. **Parse the model**: Extract the HuggingFace `owner/model-name` from the input. Use WebFetch to check the model page to confirm: parameter count, architecture, license, base model, **what file formats are available** (safetensors vs GGUF-only), and **whether it's a chat/instruct model or a base model**. Also check if it uses **custom code** (trust_remote_code=True with custom modeling_*.py files like Mamba/Nemotron/etc) and if it needs **extra pip packages** (e.g. mamba-ssm, causal-conv1d).

2. **Detect model type for PPL measurement**:
   - **Base models** (e.g. Qwen3.5-9B, Llama-3.1-8B): Use raw WikiText-2 text directly
   - **Chat/instruct models** (name contains "Instruct", "Chat", "Reasoning", "Distill", "Uncensored", or model card says it's post-trained for chat): Use **chat-template PPL** — wrap WikiText-2 chunks in the model's chat template and measure loss only on assistant response tokens
   - Set `IS_CHAT_MODEL = True/False` at the top of the script

3. **Detect format**: Check the model's file listing:
   - **If safetensors/pytorch files exist**: Load normally with `AutoModelForCausalLM.from_pretrained(MODEL, ...)`
   - **If GGUF-only** (no safetensors, no config.json, no tokenizer files):
     - Add `gguf` to pip install
     - Load model with `from_pretrained(MODEL, gguf_file="MODEL-BF16.gguf", ...)`
     - Load tokenizer from the **base model** (e.g. `Qwen/Qwen3.5-9B`)
     - Save config from the **base model**
     - Set both `MODEL` (GGUF repo), `GGUF_FILE` (the BF16 gguf filename), and `BASE_MODEL` (for tokenizer/config)

4. **Generate the Colab script**: Write a Python file to `~/Desktop/EOQ_{SHORT_NAME}.py` following EXACTLY this structure with 3 cells:

### Cell 1: Install dependencies
```
!pip install -q git+https://github.com/huggingface/transformers.git datasets accelerate safetensors sentencepiece tiktoken huggingface_hub
```
Add extra packages as needed:
- `gguf` if the model is GGUF-only
- `causal-conv1d mamba-ssm` if the model uses Mamba/SSM architecture (e.g. Nemotron)
- Any other model-specific dependencies

### Cell 2: Load model + Baseline + Quantize + Upload
- Login to HF with token `YOUR_HF_TOKEN`
- HF_USER = `caiovicentino1`
- BITS = 5, BLOCK_SIZE = 128
- `IS_CHAT_MODEL = True/False` (set based on model type detection)
- Load model:
  - **Safetensors**: `AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto', trust_remote_code=True)`
  - **GGUF-only**: `AutoModelForCausalLM.from_pretrained(MODEL, gguf_file=GGUF_FILE, dtype=torch.float16, device_map='auto', trust_remote_code=True)` + tokenizer from BASE_MODEL
- Measure FP16 tok/s (warmup + 3 runs of 100 tokens)
- **Measure FP16 PPL** using the appropriate method:
  - Include BOTH PPL functions in the script, select based on IS_CHAT_MODEL:

  ```python
  def measure_ppl_base(model, tokenizer, device):
      """PPL for base models: raw WikiText-2 text."""
      ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
      wiki = '\n\n'.join([t for t in ds['text'] if t.strip()])[:150000]
      ids = tokenizer(wiki, return_tensors='pt').input_ids.to(device)
      print(f'  Total tokens: {ids.size(1)}')
      nlls = []; total = 0; t0 = time.time()
      with torch.no_grad():
          for i in range(0, min(ids.size(1) - 2048, 15000), 512):
              c = ids[:, i:i+2048]
              t = c.clone(); t[:, :1536] = -100
              loss = model(c, labels=t).loss
              nlls.append(loss.item() * 512); total += 512
              if (total // 512) % 10 == 0:
                  print(f'  {total} tokens | PPL: {math.exp(sum(nlls)/total):.2f} | {time.time()-t0:.0f}s', flush=True)
      return math.exp(sum(nlls) / total)

  def measure_ppl_chat(model, tokenizer, device):
      """PPL for chat/instruct models: wrap text in chat template, measure loss on assistant tokens only."""
      ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
      paragraphs = [t.strip() for t in ds['text'] if len(t.strip()) > 100]
      nlls = []; total = 0; t0 = time.time()
      with torch.no_grad():
          for i, para in enumerate(paragraphs[:200]):
              # Format as chat: user asks to continue, assistant "responds" with the text
              messages = [
                  {"role": "user", "content": "Continue the following text naturally:\n" + para[:200]},
                  {"role": "assistant", "content": para[200:]}
              ]
              try:
                  formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
              except Exception:
                  formatted = f"User: Continue the following text naturally:\n{para[:200]}\nAssistant: {para[200:]}"
              ids = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=2048).input_ids.to(device)
              if ids.size(1) < 10: continue

              # Find where assistant response starts (roughly after the user message)
              user_part = tokenizer.apply_chat_template(
                  [{"role": "user", "content": "Continue the following text naturally:\n" + para[:200]}],
                  tokenize=False, add_generation_prompt=True
              ) if hasattr(tokenizer, 'apply_chat_template') else ""
              user_len = len(tokenizer(user_part, return_tensors='pt').input_ids[0]) if user_part else ids.size(1) // 2

              labels = ids.clone()
              labels[:, :user_len] = -100  # mask user tokens, only measure assistant

              loss = model(ids, labels=labels).loss
              n_tokens = (labels != -100).sum().item()
              if n_tokens > 0:
                  nlls.append(loss.item() * n_tokens)
                  total += n_tokens

              if (i+1) % 20 == 0:
                  print(f'  {i+1} samples | {total} tokens | PPL: {math.exp(sum(nlls)/total):.2f} | {time.time()-t0:.0f}s', flush=True)

      return math.exp(sum(nlls) / total)

  # Use the right method
  ppl_method = 'chat-template' if IS_CHAT_MODEL else 'wikitext-raw'
  fp16_ppl = measure_ppl_chat(model, tokenizer, DEVICE) if IS_CHAT_MODEL else measure_ppl_base(model, tokenizer, DEVICE)
  ```
- Quantize all 2D tensors >= 256 elements with absmax Q5
- Save as sharded safetensors (5 GB per shard): `{name}.codes` (int8) + `{name}.scales` (fp16)
- Save `eoq_metadata.json`, tokenizer, config, `eoq_loader.py`, `README.md`
  - For GGUF-only models, save config from BASE_MODEL
- **IMPORTANT: Copy custom model code files** to the save directory before upload:
  ```python
  # Copy custom modeling/config files for trust_remote_code models
  import glob, shutil
  from huggingface_hub import snapshot_download
  cache_dir = snapshot_download(MODEL)
  for pattern in ['modeling_*.py', 'configuration_*.py', 'generation_config.json']:
      for f in glob.glob(os.path.join(cache_dir, pattern)):
          shutil.copy2(f, save_dir)
  ```
  This ensures models with custom code (Nemotron, etc.) can be loaded from our compressed repo without needing the original repo.
- Upload to `caiovicentino1/{model-short-name}-EOQ-Q5-compressed`

### Cell 3: Test compressed model from HF
- Download from HF
- Load shard by shard with dequantization (saves RAM)
- Create model architecture using this pattern (avoids downloading full FP16 weights):
  ```python
  try:
      model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
      model.load_state_dict(state_dict, strict=False)
  except Exception:
      # Meta device: creates empty model (zero RAM), then assigns our weights
      with torch.device('meta'):
          model = AutoModelForCausalLM.from_pretrained(
              meta['base_model'], dtype=torch.float16, trust_remote_code=True
          )
      model.load_state_dict(state_dict, strict=False, assign=True)
  ```
  Note: `assign=True` is required with meta device to replace meta tensors with real data
- Generate text with 3 prompts
- Measure EOQ PPL (same method as Cell 2 — chat or raw based on IS_CHAT_MODEL)
- Measure EOQ tok/s (warmup + 3 runs)
- Print comparison table (FP16 vs EOQ Q5), include PPL method used
- Save results JSON (include `ppl_method` field)

### Key patterns to follow:
- Use `dtype=` not `torch_dtype=` (transformers 5.x)
- Always use `trust_remote_code=True`
- Handle `from_config` failure with `meta` device fallback (NOT full `from_pretrained` which wastes bandwidth downloading FP16 weights)
- **Always copy custom code files** (modeling_*.py, configuration_*.py) to the upload directory for trust_remote_code models
- **Detect chat vs base model** and use appropriate PPL method
- Shard-by-shard loading in test cell to save RAM
- `tokenizer.pad_token = tokenizer.eos_token` if None
- `torch.cuda.synchronize()` around speed measurements
- PPL with running progress
- Final comparison table + JSON with all metrics (include ppl_method)
- For GGUF-only models: tokenizer and config come from BASE_MODEL, weights from GGUF

### README.md template for HF:
Include YAML frontmatter (tags: eoq, quantized, entropy-coding, compressed + model-specific tags), base_model, license. Include benchmark table with PPL method noted, usage with eoq_loader, link to GitHub (https://github.com/caiovicentino/eoq-quantization) and base model.

### eoq_loader.py:
Include the universal loader that handles sharded safetensors, shard-by-shard dequantization, from_config with meta device fallback.

5. **Output**: Tell the user the file path and a brief summary of the model specs (including whether it's chat or base).

### EOQ Dynamic Mode (Mixed-bit)

When WebFetch reveals the model would benefit from mixed-bit quantization (or user requests it), generate the script with dynamic bit allocation instead of uniform Q5:

**Bit allocation by tensor type:**
- `gate_proj`, `up_proj`: Q3 (most robust)
- `down_proj`: Q4
- `q_proj`, `k_proj`, `v_proj`: Q5
- `o_proj`, `out_proj`: Q6 (sensitive, no AWQ fix)
- `embed_tokens`: Q5
- `lm_head`: Q6
- norms, biases, routing gates: FP16
- SSM tensors (A_log, D, dt_bias, conv1d): FP16

**Key difference in quantization loop:**
```python
def get_bits_for_tensor(name, param):
    if param.ndim < 2 or param.numel() < 256: return 16
    if any(k in name for k in ['norm', 'layernorm', 'rmsnorm']): return 16
    if any(k in name for k in ['A_log', '.D', 'dt_bias', 'conv1d']): return 16
    if any(k in name for k in ['bias']): return 16
    if 'embed' in name: return 5
    if 'lm_head' in name: return 6
    if any(k in name for k in ['o_proj', 'out_proj']): return 6
    if any(k in name for k in ['q_proj', 'k_proj', 'v_proj']): return 5
    if any(k in name for k in ['gate_proj', 'up_proj']): return 3
    if 'down_proj' in name: return 4
    return 5
```

**eoq_metadata.json** must include per-tensor bits.
**eoq_loader.py** must handle mixed bits during dequantization.
**HF repo name** uses `-EOQ-Dynamic-compressed` suffix.

## Argument: $ARGUMENTS
