# PolarQuant Arena — Public Benchmark Runner

Generate a Colab notebook (.ipynb) that evaluates a PolarQuant model on public benchmarks (MMLU, HumanEval, GSM8K, ARC, HellaSwag) and formats results for leaderboard submission.

## Input

The user provides a model name. Examples:
- `google/gemma-4-31B-it`
- `caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5`

## Instructions

1. **Parse the model**: Extract specs via WebFetch.
2. **Select benchmarks**: Based on model type (chat vs base, size).
3. **Generate notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_ARENA_{SHORT_NAME}.ipynb`.

## Notebook Structure (7 cells)

### Cell 0: Markdown header

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q accelerate safetensors sentencepiece scipy torchao
!pip install -q lm-eval  # EleutherAI evaluation harness
```

### Cell 2: Load model with streaming loader
Same PQ5+INT4 streaming loader from /polarquant-inference.

### Cell 3: Run benchmarks via lm-eval

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

# Wrap our model for lm-eval
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)

# Select benchmarks based on model type
benchmarks_chat = ['mmlu', 'gsm8k', 'arc_challenge', 'hellaswag']
benchmarks_code = ['humaneval', 'mbpp']

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=benchmarks_chat,
    num_fewshot=5,  # standard for MMLU
    batch_size=4,
)
```

**Alternative if lm-eval doesn't work with torchao INT4:**
Run benchmarks manually with generation:
```python
# MMLU: multiple choice (A/B/C/D)
# GSM8K: chain-of-thought math
# ARC: science reasoning
# Each uses model.generate() with appropriate prompts
```

### Cell 4: HumanEval (code generation)
```python
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

problems = read_problems()
# Generate completions for each problem
# Run functional tests
# Report pass@1
```

### Cell 5: Results table + comparison

```python
print(f'{"Benchmark":<20} {"PolarQuant Q5+INT4":>20} {"FP16 Reference":>15}')
print('-' * 60)
for bench, score in results.items():
    print(f'{bench:<20} {score:>20.1f}')
```

Compare against:
- Original model's reported scores (from model card)
- Other quantized versions (GGUF Q4, AWQ, GPTQ)

### Cell 6: Generate leaderboard submission + update model card

Format results for:
- Open LLM Leaderboard v2
- HuggingFace model card (benchmark section)
- Social media summary (Twitter/X format)

```python
# Auto-update model card with benchmark results
card_update = f"""
## 🏆 Benchmark Results

| Benchmark | Score | FP16 Ref | Delta |
|---|---|---|---|
| MMLU (5-shot) | {mmlu:.1f} | {ref_mmlu:.1f} | {mmlu-ref_mmlu:+.1f} |
| GSM8K | {gsm8k:.1f} | {ref_gsm8k:.1f} | {gsm8k-ref_gsm8k:+.1f} |
| ARC-C | {arc:.1f} | {ref_arc:.1f} | {arc-ref_arc:+.1f} |
| HumanEval | {humaneval:.1f} | {ref_humaneval:.1f} | {humaneval-ref_humaneval:+.1f} |
"""
```

## Key Patterns

- **lm-eval harness** preferred (standard, reproducible)
- **Fallback to manual eval** if lm-eval incompatible with torchao
- **Same streaming loader** as /polarquant-inference
- **Compare vs FP16** reference from original model card
- **Report delta** (quantization quality loss)
- **5-shot for MMLU**, 0-shot for others (standard protocol)
- **BF16 native**, no flash-attn

## Output

Tell the user:
1. File path of the notebook
2. Benchmarks selected
3. Estimated runtime
4. Expected quality delta (typically < 1% for Q5+INT4)

## Argument: $ARGUMENTS
