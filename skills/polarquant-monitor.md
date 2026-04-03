# PolarQuant Monitor

Monitor PolarQuant models and base model ecosystem. Track downloads, detect new base model releases, and suggest re-quantization opportunities.

## Input

The user provides an action. Examples:
- `status` — overview of all PolarQuant models (downloads, likes, age)
- `trending` — which models are growing fastest
- `new-models` — check if base models have new versions to quantize
- `opportunities` — suggest high-value models to quantize next

## Instructions

### Action: `status`
1. List all caiovicentino1 models via HF API
2. For each, get: downloads (last 30 days), likes, last modified, model size
3. Print sorted by downloads:
   ```
   Model                              Downloads  Likes  Size   Updated
   ─────────────────────────────────────────────────────────────────────
   Qwen3.5-9B-PolarQuant-Q5            1,234     12    6.5GB  2d ago
   Gemma-4-31B-it-PolarQuant-Q5          567      8   21.4GB  today
   ...
   ```

### Action: `trending`
1. Get download stats for last 7 days vs previous 7 days
2. Calculate growth rate
3. Highlight models with > 50% growth
4. Suggest promotion actions (tweet, HF Space, blog post)

### Action: `new-models`
Check if base models have newer versions. Use WebFetch on:
- `https://huggingface.co/google` — new Gemma releases
- `https://huggingface.co/Qwen` — new Qwen releases
- `https://huggingface.co/meta-llama` — new Llama releases
- `https://huggingface.co/nvidia` — new Nemotron releases
- `https://huggingface.co/mistralai` — new Mistral releases

For each, check:
- Any model released in last 7 days?
- Is it > 7B params? (worth quantizing)
- Do we already have a PolarQuant version?
- If not → suggest `/polarquant {model}` command

### Action: `opportunities`
Analyze HuggingFace trending models and suggest high-value quantization targets:
1. Check `https://huggingface.co/models?sort=trending` for top trending
2. Filter: > 7B params, no existing PolarQuant version, permissive license
3. Rank by: downloads × trending_score
4. Print top 10 opportunities with estimated VRAM after PQ5+INT4

## Implementation

```python
from huggingface_hub import HfApi, ModelInfo
api = HfApi()

# Get download stats
models = api.list_models(author='caiovicentino1', sort='downloads', direction=-1)
for m in models:
    info = api.model_info(m.id)
    print(f'{m.id}: {info.downloads} downloads, {info.likes} likes')

# Check for new releases
for org in ['google', 'Qwen', 'meta-llama', 'nvidia', 'mistralai']:
    new_models = api.list_models(author=org, sort='lastModified', direction=-1, limit=10)
    for m in new_models:
        if m.lastModified > week_ago:
            print(f'NEW: {m.id}')
```

## Key Patterns

- **HF API for stats** — `api.model_info()` for downloads/likes
- **WebFetch for trending** — check HF trending page
- **Cross-reference** our models vs base models to find gaps
- **Prioritize by impact** — large models + high downloads = most value
- **License check** — only suggest models with permissive licenses (Apache 2.0, MIT, Llama)
- **Size estimation** — params × 0.6 bytes for INT4 VRAM estimate

## Output

Print a clear, actionable report with specific next steps (e.g., "Run `/polarquant google/gemma-4-12B-it` to quantize the new Gemma 12B").

## Argument: $ARGUMENTS
