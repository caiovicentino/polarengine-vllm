# PolarQuant Skills for Claude Code

**14 slash commands** for the complete PolarQuant model compression workflow — from discovery to deployment.

## Installation

Copy the `.md` files to your Claude Code commands directory:

```bash
# Claude Code CLI
cp skills/*.md ~/.claude/commands/

# Or for project-scoped commands
cp skills/*.md .claude/commands/
```

Then use any skill via `/skill-name` in Claude Code.

## Skills Overview

### Quantize
| Skill | Description |
|-------|-------------|
| `/polarquant {model}` | Full PQ5+INT4+Q3 KV quantization Colab notebook |
| `/polarquant-mlx {model}` | MLX 4-bit for Apple Silicon |
| `/polarquant-gguf {model}` | GGUF for ollama/llama.cpp |
| `/polarquant-llamacpp {model}` | KV cache Q3 in llama.cpp |

### Evaluate
| Skill | Description |
|-------|-------------|
| `/polarquant-bench {model}` | Compare PQ vs torchao vs BnB vs FP16 |
| `/polarquant-arena {model}` | MMLU, HumanEval, GSM8K benchmarks |

### Serve
| Skill | Description |
|-------|-------------|
| `/polarquant-inference {model}` | Colab Gradio chat (streaming loader, fits 24GB) |
| `/polarquant-serve {model}` | Docker + OpenAI-compatible API |
| `/polarquant-vllm-kv {model}` | vLLM KV cache compression module |

### Advanced
| Skill | Description |
|-------|-------------|
| `/polarquant-finetune {model}` | QLoRA + re-quantize pipeline |
| `/polarquant-collection {action}` | Manage HF collection (audit/sync/compare) |
| `/polarquant-monitor {action}` | Track downloads, find new models to quantize |

### Legacy
| Skill | Description |
|-------|-------------|
| `/eoq {model}` | EOQ quantization |
| `/nemotron-offload {model}` | Nemotron expert offloading |

## Workflow

```
/polarquant-monitor opportunities     # Find hot models
/polarquant google/gemma-5-48B-it     # Quantize
/polarquant-bench google/gemma-5-48B  # Benchmark vs alternatives
/polarquant-inference caiovicentino1/... # Create user notebook
/polarquant-gguf caiovicentino1/...   # GGUF for ollama users
/polarquant-serve caiovicentino1/...  # Deploy API
/polarquant-collection sync           # Update HF collection
```

## References

- [arXiv Paper](https://arxiv.org/abs/2603.29078)
- [GitHub](https://github.com/caiovicentino/eoq-quantization)
- [HuggingFace Collection](https://huggingface.co/collections/caiovicentino1/polarquant-models-69cbc96292c5174df2088b08)
