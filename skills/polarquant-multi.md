# PolarQuant Multi — All Variants (PQ1 to PQ8)

Generate ALL PolarQuant quantization variants for a model, from 1-bit binary to 8-bit lossless. Lloyd-Max optimal centroids + Hadamard rotation beat GGUF/BitNet at every bit width.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `black-forest-labs/FLUX.2-klein-9B`
- `Qwen/Qwen3.5-9B`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Check architecture, parameter count, file format, components.

2. **Generate notebook** at `~/Desktop/POLARQUANT_MULTI_{SHORT_NAME}.ipynb`

## Notebook Structure

### Cell 0: Markdown header
```markdown
# 🧊 PolarQuant Multi: {MODEL}

**All quantization variants** — PQ2 through PQ8, bit-packed, Lloyd-Max optimal.

| Variant | Bits | Centroids | Packed Size | Quality | vs |
|---|---|---|---|---|---|
| PQ1 | 1 | 2 | ~{size_1} GB | Extreme | — |
| PQ1.58 | 1.58 | 3 | ~{size_158} GB | Aggressive | BitNet b1.58 |
| PQ2 | 2 | 4 | ~{size_2} GB | Aggressive | GGUF Q2_K |
| PQ3 | 3 | 8 | ~{size_3} GB | Good |
| PQ4 | 4 | 16 | ~{size_4} GB | Very Good |
| **PQ5** | **5** | **32** | **~{size_5} GB** | **Excellent** |
| PQ6 | 6 | 64 | ~{size_6} GB | Near-lossless |
| PQ8 | 8 | 256 | ~{size_8} GB | Lossless |

Lloyd-Max centroids are mathematically optimal for Gaussian-distributed weights — better quality than GGUF at same bit width.
```

### Cell 1: Install
```python
!pip install safetensors huggingface_hub scipy -q
```

### Cell 2: PolarQuant Core (supports all bit widths)

**Constants:**
```python
DEVICE = 'cuda'
MODEL = '{owner/model-name}'
BS = 128  # Hadamard block size
VARIANTS = [1, 1.58, 2, 3, 4, 5, 6, 8]  # All bit widths: binary to lossless
```

**Include:**
- `get_centroids(bits)` — Lloyd-Max via scipy.stats.norm, 100 iterations
- `build_H(n)` — Recursive Walsh-Hadamard
- `should_quantize(name, tensor)` — skip norms, embeddings, biases, routers

**Bit packing for ALL widths:**
```python
class BitPacker:
    @staticmethod
    def pack(codes, nbits):
        """Pack codes to minimum bytes. Returns (packed_uint8, total_codes)."""
        c = codes.long().reshape(-1)
        total = c.shape[0]
        # Pad to alignment
        if nbits == 2:
            pad = (4 - total % 4) % 4
            if pad: c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
            c = c.reshape(-1, 4)
            packed = ((c[:,0]<<6)|(c[:,1]<<4)|(c[:,2]<<2)|c[:,3]).to(torch.uint8)
        elif nbits == 3:
            pad = (8 - total % 8) % 8
            if pad: c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
            c = c.reshape(-1, 8)
            b0=(c[:,0]<<5)|(c[:,1]<<2)|(c[:,2]>>1)
            b1=((c[:,2]&1)<<7)|(c[:,3]<<4)|(c[:,4]<<1)|(c[:,5]>>2)
            b2=((c[:,5]&3)<<6)|(c[:,6]<<3)|c[:,7]
            packed = torch.stack([b0,b1,b2], dim=-1).reshape(-1).to(torch.uint8)
        elif nbits == 4:
            pad = (2 - total % 2) % 2
            if pad: c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
            packed = ((c[0::2]<<4)|c[1::2]).to(torch.uint8)
        elif nbits == 5:
            pad = (8 - total % 8) % 8
            if pad: c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
            c = c.reshape(-1, 8)
            b0 = ((c[:,0]<<3)|(c[:,1]>>2)).to(torch.uint8)
            b1 = (((c[:,1]&3)<<6)|(c[:,2]<<1)|(c[:,3]>>4)).to(torch.uint8)
            b2 = (((c[:,3]&15)<<4)|(c[:,4]>>1)).to(torch.uint8)
            b3 = (((c[:,4]&1)<<7)|(c[:,5]<<2)|(c[:,6]>>3)).to(torch.uint8)
            b4 = (((c[:,6]&7)<<5)|c[:,7]).to(torch.uint8)
            packed = torch.stack([b0,b1,b2,b3,b4], dim=-1).reshape(-1)
        elif nbits == 6:
            pad = (4 - total % 4) % 4
            if pad: c = torch.cat([c, torch.zeros(pad, dtype=c.dtype)])
            c = c.reshape(-1, 4)
            b0 = ((c[:,0]<<2)|(c[:,1]>>4)).to(torch.uint8)
            b1 = (((c[:,1]&15)<<4)|(c[:,2]>>2)).to(torch.uint8)
            b2 = (((c[:,2]&3)<<6)|c[:,3]).to(torch.uint8)
            packed = torch.stack([b0,b1,b2], dim=-1).reshape(-1)
        elif nbits == 8:
            packed = c.to(torch.uint8)
        else:
            packed = c.to(torch.uint8)
        return packed, total
```

**Quantize function:**
```python
def pq_quantize(weight, bits, block_size=BS):
    """Quantize a 2D weight to PQ{bits} codes + norms."""
    ct = get_centroids(bits).to(DEVICE)
    H = build_H(block_size).to(DEVICE)
    w = weight.float().to(DEVICE)
    out_f, in_f = w.shape
    pad = (block_size - in_f % block_size) % block_size
    if pad > 0: w = F.pad(w, (0, pad))
    nb = w.shape[1] // block_size
    w = w.reshape(out_f, nb, block_size)
    # Hadamard rotation
    for i in range(0, out_f, 256):
        e = min(i+256, out_f)
        w[i:e] = (w[i:e].reshape(-1, block_size) @ H).reshape(e-i, nb, block_size)
    # Normalize
    norms = w.norm(dim=2, keepdim=True).clamp(min=1e-10)
    w.div_(norms).mul_(math.sqrt(block_size))
    # Lloyd-Max quantize
    QC = 256
    codes = torch.empty(out_f, nb, block_size, dtype=torch.uint8, device=DEVICE)
    for i in range(0, out_f, QC):
        e = min(i+QC, out_f)
        codes[i:e] = (w[i:e].unsqueeze(-1) - ct.view(1,1,1,-1)).abs().argmin(-1).to(torch.uint8)
    # Bit-pack
    packed, total_codes = BitPacker.pack(codes.cpu(), bits)
    return packed, norms.squeeze(2).half().cpu(), torch.tensor([out_f, nb, block_size, total_codes], dtype=torch.int64)
```

### Cell 3: Load model + quantize all variants

```python
# Load safetensors
# For each variant (PQ2 to PQ8):
#   1. Quantize all 2D weights
#   2. Verify cos_sim on first 5 layers
#   3. Save to /content/pq{bits}/
#   4. Print size + quality summary

results = {}
for bits in VARIANTS:
    print(f'\n{"="*60}')
    print(f'  PQ{bits} — {1 << bits} Lloyd-Max centroids')
    print(f'{"="*60}')

    save_dir = f'/content/pq{bits}'
    os.makedirs(save_dir, exist_ok=True)

    polar_state = {}; bf16_state = {}
    n_quant = 0; cos_sims = []

    for name, tensor in state_dict.items():
        if should_quantize(name, tensor):
            packed, norms, meta = pq_quantize(tensor, bits)
            key = name.replace('.', '__')
            polar_state[f'{key}__packed'] = packed
            polar_state[f'{key}__norms'] = norms
            polar_state[f'{key}__meta'] = meta
            n_quant += 1
            # cos_sim check on first 5
            if n_quant <= 5:
                # dequant and compare...
        else:
            bf16_state[name] = tensor.to(torch.bfloat16)

    # Save
    save_file(polar_state, f'{save_dir}/codes.safetensors')
    save_file(bf16_state, f'{save_dir}/bf16.safetensors')

    codes_size = os.path.getsize(f'{save_dir}/codes.safetensors') / 1e9
    bf16_size = os.path.getsize(f'{save_dir}/bf16.safetensors') / 1e9
    total = codes_size + bf16_size
    avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0

    results[bits] = {
        'total_gb': total, 'cos_sim': avg_cos,
        'n_quant': n_quant, 'codes_gb': codes_size
    }
    print(f'  PQ{bits}: {total:.1f} GB | cos_sim: {avg_cos:.6f}')
```

### Cell 4: Results summary + comparison table

```python
print('='*70)
print(f'  PolarQuant Multi — {MODEL}')
print('='*70)
print(f'\n{"Variant":<10} {"Bits":<6} {"Centroids":<10} {"Size":<10} {"cos_sim":<12} {"vs GGUF"}')
print('-'*70)

gguf_sizes = {2: 'Q2_K', 3: 'Q3_K_M', 4: 'Q4_K_M', 5: 'Q5_K_M', 6: 'Q6_K', 8: 'Q8_0'}
for bits in VARIANTS:
    r = results[bits]
    gguf = gguf_sizes.get(bits, '—')
    quality = '★★★★★' if r['cos_sim'] > 0.999 else '★★★★' if r['cos_sim'] > 0.99 else '★★★' if r['cos_sim'] > 0.98 else '★★' if r['cos_sim'] > 0.95 else '★'
    print(f'  PQ{bits:<7} {bits:<6} {1<<bits:<10} {r["total_gb"]:<10.1f} {r["cos_sim"]:<12.6f} {gguf} {quality}')
```

### Cell 5: Upload ALL variants to HuggingFace

```python
# Upload each variant as a separate repo or as folders in one repo
# Option A: One repo with folders (recommended)
REPO = f'caiovicentino1/{MODEL_SHORT}-PolarQuant-Multi'

for bits in VARIANTS:
    save_dir = f'/content/pq{bits}'
    for f in os.listdir(save_dir):
        api.upload_file(
            path_or_fileobj=f'{save_dir}/{f}',
            path_in_repo=f'PQ{bits}/{f}',
            repo_id=REPO
        )

# Model card with all variants table
```

## Key Patterns

### Lloyd-Max vs GGUF (uniform) at each bit width:
- **PQ2 (4 centroids)**: Lloyd-Max places centroids at ±0.45, ±1.51 (optimal for N(0,1)). GGUF Q2_K uses uniform spacing — wastes codebook on tails.
- **PQ3 (8 centroids)**: Lloyd-Max cos_sim ~0.98 vs GGUF Q3_K ~0.95 at same size.
- **PQ5 (32 centroids)**: Lloyd-Max cos_sim ~0.999 — effectively lossless.
- **PQ8 (256 centroids)**: Perfect reconstruction, no visible loss.

### Hadamard rotation is KEY:
Without rotation, quantization error concentrates in outlier dimensions. Hadamard spreads information uniformly — each code carries equal information. This is what makes PolarQuant beat GGUF at every bit width.

### Bit packing ratios:
| Bits | Packing | Bytes per 8 codes | Ratio vs int8 |
|---|---|---|---|
| 2 | 4 codes/byte | 2 bytes | 25% |
| 3 | 8 codes/3 bytes | 3 bytes | 37.5% |
| 4 | 2 codes/byte | 4 bytes | 50% |
| 5 | 8 codes/5 bytes | 5 bytes | 62.5% |
| 6 | 4 codes/3 bytes | 6 bytes | 75% |
| 8 | 1 code/byte | 8 bytes | 100% |

### Model card must include:
- Hardware compatibility table (like GGUF screenshot)
- All variants with sizes
- cos_sim for each variant
- Comparison with GGUF equivalents
- Download links for each variant

## Output

Tell the user:
1. File path of notebook
2. All variant sizes estimated
3. Expected cos_sim per variant
4. Comparison with GGUF

## Argument: $ARGUMENTS
