# PolarQuant Benchmark Comparison Generator

Generate a Colab notebook (.ipynb) that benchmarks PolarQuant against torchao INT4, bitsandbytes NF4, and FP16 baseline on the same model. Produces comparison tables, charts, and auto-updates the HF model card.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `google/gemma-4-31B-it`
- `Qwen/Qwen3.5-9B`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check: parameter count, architecture, license, chat vs base.

2. **Estimate VRAM**: Determine GPU tier needed. FP16 baseline needs full BF16 model.

3. **Generate notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_BENCH_{SHORT_NAME}.ipynb`.

## Notebook Structure (8 cells)

### Cell 0: Markdown header
```markdown
# PolarQuant Benchmark: {Model-Name}

Comparison of quantization methods on {Model-Name}:
- **FP16 Baseline** — unquantized reference
- **PolarQuant Q5 + torchao INT4** — Hadamard + Lloyd-Max + INT4
- **torchao INT4** — absmax INT4 (no PolarQuant)
- **bitsandbytes NF4** — QLoRA-style NF4
```

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q datasets accelerate safetensors sentencepiece scipy torchao bitsandbytes
```

### Cell 2: Common setup
- Load tokenizer
- Prepare WikiText-2 test set (for base models) or generation prompts (for instruct models)
- Define `quick_ppl()` function (manual cross-entropy from logits, NOT model.forward(labels=...))
- Define `benchmark_speed()` function (3 runs × 100 tokens, torch.cuda.synchronize)
- Define `measure_vram()` function

**CRITICAL for instruct models:**
- Check if model is instruct (`-it`, `-Instruct`, `-Chat` in name)
- If instruct: skip PPL (meaningless), use generation quality instead
- If base: run PPL on WikiText-2

**CRITICAL PPL function:**
```python
# ALWAYS compute loss manually from logits — model.forward(labels=...) gives
# wrong results for multimodal/conditional generation architectures
logits = model(input_slice).logits
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = target_ids[:, 1:].contiguous()
mask = shift_labels != -100
loss = loss_fct(shift_logits[mask], shift_labels[mask])
```

### Cell 3: Method 1 — FP16 Baseline
- Load BF16: `dtype=torch.bfloat16, device_map='auto'`
- Measure: PPL (base) or generation quality (instruct), speed, VRAM
- Store results in dict
- Delete model, gc.collect(), torch.cuda.empty_cache()

### Cell 4: Method 2 — PolarQuant Q5 + torchao INT4
- Load BF16 on CPU → streaming PQ5 dequant + INT4 per-module
- Same benchmarks
- Store results
- Delete model

### Cell 5: Method 3 — torchao INT4 (no PolarQuant)
- Load BF16 → `quantize_(model, Int4WeightOnlyConfig(group_size=128))`
- Same benchmarks
- Store results
- Delete model

### Cell 6: Method 4 — bitsandbytes NF4
- Load with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- Same benchmarks
- Store results
- Delete model

### Cell 7: Results + Charts

**Comparison table:**
```
Method              | tok/s | VRAM    | PPL   | Delta PPL
─────────────────────────────────────────────────────────
FP16 Baseline       | XX.X  | XX.X GB | X.XX  | —
PolarQuant Q5+INT4  | XX.X  | XX.X GB | X.XX  | +0.XX
torchao INT4        | XX.X  | XX.X GB | X.XX  | +0.XX
bitsandbytes NF4    | XX.X  | XX.X GB | X.XX  | +0.XX
```

**Charts (dark theme, PolarQuant palette):**
- Bar chart: PPL comparison (#4FC3F7=PolarQuant, #FF8A65=torchao, #AED581=BnB, #CE93D8=FP16)
- Bar chart: Speed comparison
- Scatter: Speed vs VRAM (each method as a point)
- Bar chart: VRAM comparison

**Save charts** to `/content/` as PNG.

### Cell 8: Upload results
- Update HF model card with benchmark results
- Upload charts to repo
- Print summary

## Key Patterns

- **Always delete model between methods** to get accurate VRAM measurements
- **FP16 first** as baseline, then quantized methods
- **Same prompt/data** for all methods (fair comparison)
- **torch.cuda.reset_peak_memory_stats()** before each measurement
- **3 runs minimum** for speed, report average
- **For instruct models**: use 3 diverse generation prompts, compare output quality side-by-side
- **PolarQuant streaming loader** for the PQ5+INT4 method (per-module INT4)
- **BF16 native** everywhere, never `.half()`

## Output

Tell the user:
1. File path of the notebook
2. Methods being compared
3. Estimated total runtime
4. GPU tier needed (must fit FP16 for baseline)

## Argument: $ARGUMENTS
