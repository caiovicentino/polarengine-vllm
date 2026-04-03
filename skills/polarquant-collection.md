# PolarQuant Collection Manager

Manage the PolarQuant HuggingFace collection: audit model cards, sync collection membership, generate comparison tables, and batch-update metadata.

## Input

The user provides an action. Examples:
- `audit` — check all models for missing notebooks/benchmarks/cards
- `sync` — ensure all PolarQuant models are in the collection
- `compare` — generate comparison table across all models
- `update-cards` — batch update model cards with consistent formatting
- `stats` — download/like stats for all models

## Instructions

### Action: `audit`
1. List all models by `caiovicentino1` via HF API
2. For each PolarQuant/EOQ model, check:
   - Has README.md with benchmark results?
   - Has inference notebook?
   - Has quantization notebook?
   - Has polar_config.json?
   - Has charts (PNG files)?
   - Is in the PolarQuant collection?
3. Print report with missing items per model

### Action: `sync`
1. List all caiovicentino1 models
2. Get current PolarQuant collection members
3. Add any missing PolarQuant/EOQ models
4. Print changes made

### Action: `compare`
1. For each model, read polar_config.json for metrics
2. Generate markdown comparison table:
   ```
   | Model | Params | Method | VRAM | tok/s | PPL |
   |-------|--------|--------|------|-------|-----|
   ```
3. Generate charts (bar chart of VRAM, speed)
4. Optionally upload to collection README

### Action: `update-cards`
1. For each model, read current README.md
2. Ensure consistent sections (emoji headers, citation, links)
3. Update arXiv link, GitHub link, collection link
4. Fix any stale benchmark values
5. Print diff for each model before applying

### Action: `stats`
1. For each model, get download count and likes via HF API
2. Sort by downloads (descending)
3. Print summary table
4. Identify trending models

## Implementation

Use HuggingFace Hub API (`pip install huggingface_hub`):

```python
from huggingface_hub import HfApi
api = HfApi()

# List models
models = list(api.list_models(author='caiovicentino1'))

# Get collection
col = api.get_collection('caiovicentino1/polarquant-models-69cbc96292c5174df2088b08')

# List repo files
files = api.list_repo_files('caiovicentino1/Model-Name')

# Read file
content = api.hf_hub_download('repo', 'file.json')
```

## Key Patterns

- **Collection slug**: `caiovicentino1/polarquant-models-69cbc96292c5174df2088b08`
- **PolarQuant detection**: model ID contains 'polarquant', 'eoq', or 'polarengine'
- **Standard model card sections**: 🧊 Title, 🎯 Key Results, 📊 Benchmarks, 🚀 Quick Start, 🏆 GPU Support, 🔧 Technical Details, 📖 Citation, 🔗 Resources
- **arXiv**: `https://arxiv.org/abs/2603.29078`
- **GitHub**: `https://github.com/caiovicentino/eoq-quantization`
- **NEVER modify model weights** — only metadata/README/config
- **Show diff before applying** any changes to model cards

## Output

Print a clear report of the action taken, changes made, and any issues found.

## Argument: $ARGUMENTS
