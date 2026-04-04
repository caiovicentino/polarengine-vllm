# PolarQuant Video Model Quantizer

Quantize video generation/diffusion models with PolarQuant Q5. Works on any diffusion transformer — Wan, CogVideo, Mochi, LTX, HunyuanVideo, etc.

## Input

The user provides a video model name or URL. Examples:
- `Wan-AI/Wan2.2-Animate-14B`
- `THUDM/CogVideoX-5b`
- `genmo/mochi-1-preview`

## Instructions

1. **Parse the model**: Use WebFetch to check: parameter count, architecture (diffusion transformer, UNet, DiT?), components (VAE, text encoder, CLIP), total repo size, file structure.

2. **Identify components**:
   - **Transformer/DiT** — the main backbone, quantizable
   - **VAE** — keep BF16 (quality sensitive for visual output)
   - **Text encoders** (T5, CLIP, etc.) — keep BF16 (small, important)
   - **ONNX/preprocessing** — copy as-is

3. **Determine loading strategy**:
   - If diffusers pipeline exists: try `DiffusionPipeline.from_pretrained()`
   - If custom model (like Wan): load state_dict directly from safetensors
   - Check `config.json` for `_class_name`, `num_layers`, `dim`, `num_heads`

4. **Generate notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_VIDEO_{SHORT_NAME}.ipynb`

## Notebook Structure (8 cells)

### Cell 0: Markdown header
```markdown
# 🧊🎬 PolarQuant Video: {MODEL}

**First PolarQuant quantized version** of {MODEL}.

| Component | Strategy | Size |
|---|---|---|
| **Transformer** | PolarQuant Q5 codes | ~50% of original |
| **VAE** | BF16 (preserved) | unchanged |
| **Text encoders** | BF16 (preserved) | unchanged |
```

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q diffusers accelerate safetensors scipy torchao imageio[ffmpeg] opencv-python pillow
```

### Cell 2: Explore model structure

**CRITICAL — explore BEFORE quantizing:**
```python
# Check repo files
api = HfApi()
files = list(api.list_repo_files(MODEL))

# Check config
config = json.load(open(hf_hub_download(MODEL, 'config.json')))
print(f'Class: {config.get("_class_name")}')
print(f'Layers: {config.get("num_layers")}')
print(f'Dim: {config.get("dim")}')
print(f'Heads: {config.get("num_heads")}')
# head_dim = dim / num_heads — must be power of 2 for Hadamard
```

### Cell 3: Load state_dict from safetensors

**Load directly from safetensors — do NOT rely on model class:**
```python
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

state_dict = {}
for shard_file in [f for f in files if f.endswith('.safetensors') and 'diffusion' in f]:
    path = hf_hub_download(MODEL, shard_file)
    state_dict.update(load_file(path))

# Analyze
n_2d = sum(1 for k,v in state_dict.items() if v.ndim == 2 and v.numel() > 256)
n_3d = sum(1 for k,v in state_dict.items() if v.ndim == 3)
total = sum(v.numel() for v in state_dict.values())
print(f'{total/1e9:.1f}B params, {n_2d} quantizable 2D, {n_3d} 3D')
```

**Why state_dict instead of model class?**
- Video models often have custom classes NOT in diffusers
- WanAnimateModel, CogVideoXTransformer3DModel may have version mismatches
- State dict works universally — no class dependency

### Cell 4: PolarQuant Q5 quantize

```python
# PolarQuant core (same as LLMs — works on any 2D weight)
# Quantize all 2D weights, skip norms/biases/embeddings/modulation

for key, tensor in state_dict.items():
    if tensor.ndim != 2 or tensor.numel() < 256 or 'norm' in key:
        new_state[key] = tensor  # keep as-is
        continue
    # PQ5: Hadamard rotate → normalize → Lloyd-Max Q5 → dequant
    # ... (standard PQ5 code) ...
```

**Key patterns for video models:**
- `'norm' in key` — skip (layer norms, critical for quality)
- `'modulation' in key` — skip (adaptive norm, often 3D)
- `'patch_embedding' in key` — skip (input projection, shape-critical)
- `'time_' in key` — skip (timestep embeddings)
- Everything else with ndim==2 and numel>256 → quantize

### Cell 5: Verify quality (cosine similarity)

**CRITICAL — always verify before saving:**
```python
# Reload original shard and compare
orig_shard = load_file(hf_hub_download(MODEL, first_shard))
for key in list(orig_shard.keys())[:10]:
    if key in new_state and orig_shard[key].ndim == 2:
        cos = cosine_similarity(orig_shard[key], new_state[key])
        print(f'{key}: cos_sim={cos:.6f}')
# Should be >0.999 for Q5
```

### Cell 6: Save as PQ5 codes (smaller than BF16 dequant)

Two options:
- **PQ5 codes** (int8 codes + fp16 norms): ~50% of original transformer
- **BF16 dequant**: same size as original (no benefit for download)

Always save as PQ5 codes for smallest download.

### Cell 7: Copy non-transformer files + Upload

```python
# Copy ALL non-transformer files from original repo
# VAE, text encoders, ONNX checkpoints, configs, tokenizers
for f in orig_files:
    if 'diffusion_pytorch_model' in f: continue  # skip original transformer
    path = hf_hub_download(ORIG, f)
    api.upload_file(path, f, REPO)
```

**Use upload_large_folder for repos with 200+ files** to avoid rate limiting.

### Cell 8: Model card

Include: original size, PQ5 size, compression ratio, cos_sim, architecture details, how to load codes.

## Key Patterns (MUST FOLLOW)

### Universal for ALL video models:
- **Load state_dict, not model class** — avoids version/class mismatches
- **Quantize 2D weights only** — skip norms, biases, embeddings, modulation
- **head_dim must be power of 2** — check dim/num_heads before quantizing
- **Verify cos_sim >0.999** before uploading
- **Keep VAE in BF16** — visual quality depends on it
- **Keep text encoders in BF16** — small relative to transformer
- **Save PQ5 codes** (not BF16 dequant) for smallest download
- **Copy ALL non-transformer files** — repo must be self-contained

### Common video model architectures:
| Model | Transformer file | VAE | Text Encoder |
|---|---|---|---|
| Wan2.2 | `diffusion_pytorch_model-*.safetensors` | `Wan2.1_VAE.pth` | `models_t5_umt5-xxl-enc-bf16.pth` |
| CogVideoX | `transformer/` | `vae/` | `text_encoder/` |
| Mochi | `dit/` | `vae/` | `text_encoder/` |
| HunyuanVideo | `transformer/` | `vae/` | `text_encoder/` |

### Config key names vary:
- Wan: `dim`, `num_heads`, `num_layers`
- CogVideoX: `hidden_size`, `num_attention_heads`, `num_layers`
- Standard diffusers: `sample_size`, `in_channels`, `num_layers`

Always check config.json first.

## Expected Results

| Model | Original | PQ5 Codes | Compression | cos_sim |
|---|---|---|---|---|
| Wan2.2-Animate-14B | 34.5 GB transformer | 17.6 GB | 2.0x | >0.999 |
| CogVideoX-5B | ~10 GB | ~5 GB | 2.0x | >0.999 |

PQ5 on diffusion transformers gives identical compression and quality as on LLMs.

## Output

Tell the user:
1. File path of the notebook
2. Model architecture (layers, dim, heads, head_dim)
3. Components identified (transformer, VAE, encoders)
4. Estimated compression
5. Whether head_dim is power of 2 (Hadamard compatible)

## Argument: $ARGUMENTS
