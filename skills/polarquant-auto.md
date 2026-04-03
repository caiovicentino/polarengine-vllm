# PolarQuant Auto-Quantize — Trending Model Bot

Automatically find trending HuggingFace models, quantize with PolarQuant, and publish.

## Input

No input needed — runs autonomously. Or optionally:
- `scan` — find new models to quantize
- `run` — quantize the top opportunity
- `run-all` — quantize all opportunities

## Instructions

### Step 1: Scan for opportunities

Use HF API to find high-value models to quantize:

```python
from huggingface_hub import HfApi
api = HfApi()

# Check trending models
trending = list(api.list_models(sort="trending", direction=-1, limit=50))

# Filter:
# - > 7B params (worth quantizing)
# - Permissive license (Apache 2.0, MIT, Llama)
# - No existing PolarQuant version
# - Released in last 7 days (fresh)
# - Text-generation or image-text-to-text pipeline

# Cross-reference with existing PolarQuant models
existing = {m.id for m in api.list_models(author="caiovicentino1")}
```

### Step 2: Prioritize

Score = downloads × recency × (1 if no quant exists else 0.1)

Top orgs to watch:
- google (Gemma)
- Qwen (Qwen3.5)
- meta-llama (Llama)
- nvidia (Nemotron)
- mistralai (Mistral)
- microsoft (Phi)
- deepseek-ai (DeepSeek)

### Step 3: Quantize

For each opportunity:
1. Run `polarquant info {model}` to check specs
2. If dense model: use `/polarquant` skill (streaming PQ5+INT4)
3. If MoE model: use expert offloading + PQ5 codes approach
4. If multimodal: use `/polarquant` with `--vision` (BF16 vision encoder)
5. Generate inference notebook + Gradio Space
6. Upload to HF with model card + charts
7. Add to PolarQuant collection
8. Post to r/LocalLLaMA if model is popular

### Step 4: Report

Print summary:
```
New models quantized today:
1. google/gemma-4-12B-it → caiovicentino1/Gemma-4-12B-it-PolarQuant-Q5 (8 GB)
2. Qwen/Qwen3.5-32B → caiovicentino1/Qwen35-32B-PolarQuant-Q5 (18 GB)
...
```

## Key Patterns

- **Always check existing**: Don't re-quantize what we already have
- **Prioritize by impact**: Large models + high downloads = most value
- **License check**: Only Apache 2.0, MIT, Llama Community, Gemma
- **Vision support**: If model is multimodal, always include vision
- **MoE detection**: If num_experts > 1, use expert offloading approach
- **Speed**: Use streaming loader for dense, expert offload for MoE
- **Quality**: Always test generation before uploading

## Output

1. List of opportunities found
2. Models quantized (with links)
3. Suggested r/LocalLLaMA post text

## Argument: $ARGUMENTS
