# PolarQuant v5 Colab Generator

Generate a complete Colab notebook (.ipynb) that quantizes a HuggingFace model with PolarQuant Q5 (Hadamard + Lloyd-Max) weights + PolarQuant Q3 KV cache, benchmarks with torchao INT4, and uploads to HF.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `Jackrong/Qwopus3.5-9B-v3`
- `https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check the model page for: parameter count, architecture, license, file formats, chat vs base model, custom code, extra pip packages.

2. **Estimate VRAM**: params × 2 bytes for BF16. If > 48 GB → A100 80 GB. If > 80 GB → multi-GPU.

3. **Generate the Colab notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_UNIFIED_{SHORT_NAME}.ipynb`.

## Notebook Structure (9 cells)

### Cell 0: Markdown header
```markdown
# PolarQuant Unified: Weights Q5 + KV Cache Q3

**Full stack compression** for consumer GPU inference.

Model: `{MODEL}`

| Component | Technique | Effect |
|---|---|---|
| **Weights** | PolarQuant Q5 → dequant → torchao INT4 | BF16 → ~3x reduction |
| **KV Cache** | PolarQuant Q3 (bit-packed Hadamard + Lloyd-Max) | 5x longer context |

Pipeline: Load BF16 → PolarQuant Q5 dequant → torchao INT4 → Generate with PolarQuant Q3 KV Cache
```

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q datasets accelerate safetensors sentencepiece tiktoken scipy torchao
```
**NEVER include flash-attn** — takes too long to compile, SDPA is identical speed.

### Cell 2: Restart check
```python
import transformers; print(f'transformers: {transformers.__version__}')
import torchao; print(f'torchao: {torchao.__version__}')
```

### Cell 3: Core (all classes + functions)

**Constants:**
```python
DEVICE = 'cuda'
MODEL = '{owner/model-name}'
HEAD_DIM = 128
BS = 128
```

**Include ALL of these (copy exactly):**

**A) PolarQuant Math:**
- `get_centroids(bits)` — Lloyd-Max via scipy.stats.norm, 100 iterations, cache in `_C` dict, precompute for bits [2,3,4,5,6]
- `_build_H(n)` — Recursive Walsh-Hadamard, `H128 = _build_H(BS)` (NOT `.to(DEVICE)` here — let each user move it)

**B) BitPacker (real bit-packing, NOT fake int8):**
```python
class BitPacker:
    @staticmethod
    def pack(codes, nbits):
        c = codes.long(); N = c.shape[0]
        if nbits == 2:
            c = c.reshape(N, -1, 4)
            return ((c[:,:,0]<<6)|(c[:,:,1]<<4)|(c[:,:,2]<<2)|c[:,:,3]).to(torch.uint8)
        elif nbits == 3:
            c = c.reshape(N, -1, 8)
            b0 = (c[:,:,0]<<5)|(c[:,:,1]<<2)|(c[:,:,2]>>1)
            b1 = ((c[:,:,2]&1)<<7)|(c[:,:,3]<<4)|(c[:,:,4]<<1)|(c[:,:,5]>>2)
            b2 = ((c[:,:,5]&3)<<6)|(c[:,:,6]<<3)|c[:,:,7]
            return torch.stack([b0,b1,b2], dim=-1).reshape(N,-1).to(torch.uint8)
        elif nbits == 4:
            return ((c[:,0::2]<<4)|c[:,1::2]).to(torch.uint8)
        return codes.to(torch.uint8)

    @staticmethod
    def unpack(packed, nbits, D):
        p = packed.long(); N = p.shape[0]
        if nbits == 2:
            return torch.stack([(p>>6)&3,(p>>4)&3,(p>>2)&3,p&3], dim=-1).reshape(N, D)
        elif nbits == 3:
            p3 = p.reshape(N, -1, 3)
            b0, b1, b2 = p3[:,:,0], p3[:,:,1], p3[:,:,2]
            return torch.stack([
                (b0>>5)&7, (b0>>2)&7, ((b0&3)<<1)|((b1>>7)&1),
                (b1>>4)&7, (b1>>1)&7, ((b1&1)<<2)|((b2>>6)&3),
                (b2>>3)&7, b2&7
            ], dim=-1).reshape(N, D)
        elif nbits == 4:
            return torch.stack([(p>>4)&0xF, p&0xF], dim=-1).reshape(N, D)
        return p

    @staticmethod
    def packed_bytes_per_vec(D, nbits):
        if nbits == 2: return D // 4
        elif nbits == 3: return (D // 8) * 3
        elif nbits == 4: return D // 2
        return D
```

**C) PolarQuant KV Cache (MUST include all of this for hybrid model support):**

```python
class PolarQuantLayer:
    def __init__(self, nbits=3, residual_length=128, device='cuda'):
        self.nbits = nbits
        self.residual_length = residual_length
        self.device = device
        self.ct = get_centroids(nbits).to(device)
        self.H = H128.to(device)
        self.scale = math.sqrt(HEAD_DIM)
        self._packed = None; self._norms = None; self._q_seq = 0
        self._B = None; self._NH = None; self._D = None
        self._can_quantize = True; self.residual = None

    def _quantize(self, tensor):
        flat = tensor.reshape(-1, HEAD_DIM).float()
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        rotated = (flat / norms) @ self.H * self.scale
        codes = (rotated.unsqueeze(-1) - self.ct.view(1,1,-1)).abs().argmin(-1)
        return BitPacker.pack(codes.to(torch.uint8), self.nbits), norms.half().squeeze(1)

    def _dequantize(self, packed, norms, B, H):
        codes = BitPacker.unpack(packed, self.nbits, HEAD_DIM)
        values = self.ct[codes] / self.scale
        values = (values @ self.H) * norms.float().unsqueeze(1)
        S = packed.shape[0] // (B * H)  # derive S from data, not tracker
        return values.half().reshape(B, H, S, HEAD_DIM)

    def update(self, new_tensor):
        if self._B is None:
            self._B, self._NH = new_tensor.shape[0], new_tensor.shape[1]
            self._D = new_tensor.shape[3]
            self._can_quantize = (self._D == HEAD_DIM)  # skip if head_dim != 128
        self.residual = new_tensor if self.residual is None else torch.cat([self.residual, new_tensor], dim=2)
        if self._can_quantize and self.residual.shape[2] > self.residual_length:
            n_q = self.residual.shape[2] - self.residual_length
            packed, norms = self._quantize(self.residual[:, :, :n_q, :])
            self.residual = self.residual[:, :, n_q:, :].contiguous()
            if self._packed is None:
                self._packed, self._norms = packed, norms
            else:
                self._packed = torch.cat([self._packed, packed], dim=0)
                self._norms = torch.cat([self._norms, norms], dim=0)
            self._q_seq += n_q
        if self._packed is not None:
            return torch.cat([self._dequantize(self._packed, self._norms, self._B, self._NH), self.residual], dim=2)
        return self.residual

    def get_seq_length(self):
        q = 0
        if self._packed is not None and self._B and self._NH:
            q = self._packed.shape[0] // (self._B * self._NH)
        return q + (self.residual.shape[2] if self.residual is not None else 0)

    def memory_bytes(self):
        t = 0
        if self._packed is not None: t += self._packed.numel() + self._norms.numel() * 2
        if self.residual is not None: t += self.residual.numel() * 2
        return t
```

**CRITICAL — HybridCacheLayer for Qwen3.5 linear attention:**
```python
try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    _la_base = LinearAttentionCacheLayerMixin
except ImportError:
    _la_base = object

class _HybridCacheLayer(_la_base):
    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None
    def lazy_initialization(self, *args, **kwargs): pass
    def update_conv_state(self, conv_states, **kwargs):
        self.conv_states = conv_states
        return conv_states
    def update_recurrent_state(self, recurrent_states, **kwargs):
        self.recurrent_states = recurrent_states
        return recurrent_states
```

**CRITICAL — PolarQuantKVCache (inherits Cache, full hybrid API):**
```python
class PolarQuantKVCache(Cache):
    def __init__(self, num_layers, nbits=3, residual_length=128, device='cuda'):
        try:
            super().__init__()
        except (TypeError, ValueError):
            pass  # latest transformers Cache requires layers param
        self.num_layers = num_layers; self.nbits = nbits
        self.kl = [PolarQuantLayer(nbits, residual_length, device) for _ in range(num_layers)]
        self.vl = [PolarQuantLayer(nbits, residual_length, device) for _ in range(num_layers)]
        self._seen_tokens = 0
        if not hasattr(self, 'layers'):
            self.layers = [None] * num_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0: self._seen_tokens += key_states.shape[2]
        return self.kl[layer_idx].update(key_states), self.vl[layer_idx].update(value_states)

    def get_seq_length(self, layer_idx=0): return self.kl[layer_idx].get_seq_length()
    def get_max_cache_shape(self): return None
    def get_mask_sizes(self, query_length, layer_idx):
        return query_length, self.get_seq_length(layer_idx)
    def has_previous_state(self, layer_idx=0):
        has_kv = self.get_seq_length(layer_idx) > 0
        has_linear = self.layers[layer_idx] is not None and isinstance(self.layers[layer_idx], _HybridCacheLayer)
        return has_kv or has_linear

    # Hybrid model support (Qwen3.5 linear attention layers)
    def update_conv_state(self, conv_states, layer_idx, **kwargs):
        if self.layers[layer_idx] is None:
            self.layers[layer_idx] = _HybridCacheLayer()
        self.layers[layer_idx].conv_states = conv_states
        return conv_states
    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        if self.layers[layer_idx] is None:
            self.layers[layer_idx] = _HybridCacheLayer()
        self.layers[layer_idx].recurrent_states = recurrent_states
        return recurrent_states

    @property
    def seen_tokens(self): return self._seen_tokens
    def __getitem__(self, idx):
        k, v = self.kl[idx], self.vl[idx]
        k_out, v_out = k.residual, v.residual
        if k._packed is not None and k_out is not None:
            k_out = torch.cat([k._dequantize(k._packed, k._norms, k._B, k._NH), k_out], dim=2)
            v_out = torch.cat([v._dequantize(v._packed, v._norms, v._B, v._NH), v_out], dim=2)
        return (k_out, v_out)
    def __len__(self): return self.num_layers
    def __iter__(self):
        for i in range(self.num_layers): yield self[i]
    def memory_bytes(self):
        return sum(k.memory_bytes() + v.memory_bytes() for k, v in zip(self.kl, self.vl))
```

**D) Weight quantization config:**
```python
def get_bits_q5(name, param):
    if param.ndim < 2 or param.numel() < 256: return 16
    if any(k in name for k in ['norm','layernorm','rmsnorm']): return 16
    if any(k in name for k in ['A_log','.D','dt_bias','conv1d']): return 16
    if 'bias' in name and param.ndim == 1: return 16
    if name.endswith('.gate.weight') or 'router' in name: return 16
    return 5
```

**E) torchao guard patch:**
```python
import torchao.quantization.utils as _tao_utils
_orig_guard = _tao_utils.guard_dtype_size
def _patched_guard(tensor_arg, arg_name, dtype=None, size=None):
    if dtype is not None and tensor_arg.dtype != dtype:
        tensor_arg.data = tensor_arg.data.to(dtype)
    if size is not None and tensor_arg.size() != size:
        raise ValueError(f"Expected size {size}, got {tensor_arg.size()}")
_tao_utils.guard_dtype_size = _patched_guard
```

**F) Load tokenizer at end of cell.**

### Cell 4: Step 1 — Load + PolarQuant Q5 + torchao INT4

**Flow:**
1. Load model in **BF16** (`dtype=torch.bfloat16`, `attn_implementation='sdpa'`)
2. PolarQuant Q5 dequant each nn.Linear in-place:
   ```python
   # CORRECT dequant (DO NOT use the old buggy view with n_blocks*BS*BS):
   values = ct[all_codes.long()] / math.sqrt(BS)  # (out_f, n_blocks, BS)
   for i in range(0, out_f, 64):
       end = min(i + 64, out_f)
       v = values[i:end].reshape(-1, BS)
       values[i:end] = (v @ H_dev).reshape(end - i, n_blocks, BS)
   values = values * norms.unsqueeze(2)
   child.weight.data = values.reshape(out_f, -1)[:, :in_f].to(child.weight.dtype)
   ```
3. Test generation (quick sanity)
4. Apply torchao INT4 **without `.half()`** — keep BF16:
   ```python
   # Do NOT call model.half()! Qwen3.5 is BF16 native, FP16 causes garbage.
   quantize_(model, Int4WeightOnlyConfig(group_size=128))
   _tao_utils.guard_dtype_size = _orig_guard
   ```
5. Test generation again
6. If torchao BF16 fails → fallback to bitsandbytes NF4

### Cell 5: Step 2 — Generation with PolarQuant KV Cache

**Flow:**
1. Get `num_layers`, `num_kv_heads` from model config
2. Manual generation loop:
   ```python
   @torch.no_grad()
   def generate_with_cache(model, input_ids, max_new_tokens, cache=None):
       out = model(input_ids, past_key_values=cache, use_cache=True)
       cache = out.past_key_values
       next_tok = out.logits[:, -1:].argmax(-1)
       tokens = [next_tok]
       for _ in range(max_new_tokens - 1):
           out = model(next_tok, past_key_values=cache, use_cache=True)
           cache = out.past_key_values
           next_tok = out.logits[:, -1:].argmax(-1)
           tokens.append(next_tok)
           if next_tok.item() == tokenizer.eos_token_id: break
       return torch.cat([input_ids] + tokens, dim=1), cache
   ```
3. Use `tokenizer.apply_chat_template()` for chat models
4. FP16 KV baseline (cache=None) → 300 tokens
5. PolarQuant Q3 KV (cache=PolarQuantKVCache(...)) → 300 tokens
6. Token match comparison

### Cell 6: Step 3 — Speed Benchmark

FP16 KV vs Q4/Q3/Q2 KV: 3 runs × 100 tokens each. Peak VRAM measurement.

### Cell 7: Step 4 — PPL (WikiText-2)

Sliding window: 2048 context, 512 stride, mask first 1536.

### Cell 8: Step 5 — Results Table

**Include:**
- Weights table (vs FP16, torchao, BnB references)
- KV Cache table (compression, speed, max context)
- Combined VRAM budget
- Consumer GPU projections (pick appropriate GPUs based on model size)

### Cell 9: Step 6 — Save + Upload to HF

**Flow:**
1. `login(token="YOUR_HF_TOKEN")`
2. Re-quantize on CPU (current model has torchao, can't extract codes)
3. Save: `{key_prefix}.codes` (int8), `{key_prefix}.norms` (fp16), `{key_prefix}.ct` (fp32, **`.clone()`** — shared memory fix)
4. Shard safetensors at 5 GB
5. Save polar_config.json, tokenizer, config
6. Upload to `caiovicentino1/{Model-Name}-PolarQuant-Q5`

### Cell 10: Step 7 — Pro Model Card + Charts

**Charts (dark theme, PolarQuant palette):**
- PPL comparison bar (#4FC3F7=PolarQuant, #FF8A65=torchao, #AED581=BnB, #CE93D8=FP16)
- Speed vs VRAM scatter
- KV Context horizontal bar

**Model card sections (emoji headers):**
🧊 Title, 🎯 Key Results, 📊 Benchmark Comparison, 🚀 Quick Start, 🔬 KV Cache Compression, 🏆 Consumer GPU Support, 🔧 Technical Details, 📖 Citation, 🔗 Resources, 🙏 Acknowledgements

**CRITICAL — Update ALL values for this specific model:**
- Title with correct model name and size (9B/27B/etc)
- PPL, tok/s, VRAM from actual benchmark results
- Architecture (num_layers, KV heads)
- GPU table appropriate for model size
- Code example with correct MODEL and REPO strings
- Dequant time from actual measurement

**Add to collection:** `caiovicentino1/polarquant-models-{slug}`

## Key Patterns (MUST FOLLOW)

- **BF16 native**: `dtype=torch.bfloat16` (NEVER float16). Qwen3.5 crashes in FP16.
- **No `.half()`**: torchao INT4 works directly on BF16. FP16 cast produces garbage.
- **No flash-attn**: Use `attn_implementation='sdpa'`. SDPA uses FlashAttention kernel natively.
- **Dequant reshape**: Use chunked `(v @ H_dev).reshape(end-i, n_blocks, BS)`. NEVER `values.view(out_f, n_blocks*BS, BS)` — this is a bug.
- **`.clone().contiguous()`** on centroids before saving (shared memory crash).
- **head_dim detection**: `self._can_quantize = (self._D == HEAD_DIM)` — skip PolarQuant for layers with head_dim != 128.
- **S from data**: `S = packed.shape[0] // (B * H)` in dequantize, NOT from `_q_seq` tracker.
- **Cache API compatibility**: `try/except` on `super().__init__()`, `self.layers = [None] * num_layers`, override `get_mask_sizes`, `has_previous_state(layer_idx=0)`.
- **Hybrid model support**: `_HybridCacheLayer` with `lazy_initialization`, `update_conv_state`, `update_recurrent_state` for Qwen3.5's linear attention layers.
- **Model card auto-update**: Always use actual benchmark values, never values from a previous model.
- **arXiv link**: `https://arxiv.org/abs/2603.29078`
- **GitHub link**: `https://github.com/caiovicentino/eoq-quantization`

## Expected Results (reference)

| Model | Method | tok/s | VRAM | PPL |
|-------|--------|-------|------|-----|
| 9B | FP16 baseline | 45.7 | 17.9 GB | 6.37 |
| 9B | PolarQuant Q5 + INT4 | 43 | 7.1 GB | 6.48-6.54 |
| 9B | torchao INT4 (absmax) | 43.3 | 6.3 GB | 6.68 |
| 27B | PolarQuant Q5 + INT4 | 22 | 17.7 GB | 5.37 |

PolarQuant beats torchao absmax by ~0.14-0.20 PPL with same speed/VRAM.

## Output

Tell the user:
1. File path of the notebook
2. Model specs (params, architecture, chat/base, hybrid?)
3. Estimated VRAM requirement
4. Which GPU tier is needed (T4/A100/H100)

## Argument: $ARGUMENTS
