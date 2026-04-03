# PolarQuant vLLM Serve Generator

Generate deployment files for serving a PolarQuant model via vLLM with an OpenAI-compatible API endpoint. Produces a Colab notebook or Docker setup.

## Input

The user provides a model name or HuggingFace URL. Examples:
- `google/gemma-4-31B-it`
- `caiovicentino1/Gemma-4-31B-it-PolarQuant-Q5`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch for specs.

2. **Determine serving strategy**:
   - Small models (≤ 9B): Single GPU, direct vLLM serve
   - Medium models (10-35B): Single GPU with PolarQuant streaming loader
   - Large models (70B+): Multi-GPU tensor parallel

3. **Generate files** to `~/Desktop/polarquant-serve-{SHORT_NAME}/`:

## Files to Generate

### 1. `serve.py` — Main serving script

```python
"""Serve PolarQuant model with vLLM-compatible API.

Usage:
    python serve.py --model google/gemma-4-31B-it --port 8000
    curl http://localhost:8000/v1/chat/completions -d '{...}'
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from polarengine_vllm.kv_cache import PolarKVConfig, PolarKVCache

# ... streaming loader code ...
# ... FastAPI/uvicorn OpenAI-compatible endpoint ...
```

Include:
- Streaming loader (per-module INT4)
- PolarQuant KV cache
- OpenAI-compatible `/v1/chat/completions` endpoint
- SSE streaming responses
- Health check `/health`
- Model info `/v1/models`

### 2. `Dockerfile`

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN pip install torch transformers torchao safetensors scipy sentencepiece \
    fastapi uvicorn polarengine-vllm

COPY serve.py /app/serve.py
WORKDIR /app

EXPOSE 8000
CMD ["python", "serve.py", "--model", "{MODEL}", "--port", "8000"]
```

### 3. `docker-compose.yml`

```yaml
services:
  polarquant:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL={MODEL}
```

### 4. `serve_colab.ipynb` — Colab notebook version

Same as serve.py but in Colab format with ngrok tunnel for public API access:
```python
!pip install pyngrok fastapi uvicorn
from pyngrok import ngrok
# ... start server + ngrok tunnel ...
print(f'API endpoint: {ngrok_url}/v1/chat/completions')
```

### 5. `client_example.py` — Usage example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="{MODEL}",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### 6. `README.md`

Include:
- Quick start (Docker, Colab, local Python)
- API documentation (OpenAI-compatible endpoints)
- GPU requirements
- Performance benchmarks
- Configuration options

## Key Patterns

- **Streaming loader**: per-module INT4 for < 24 GB VRAM
- **PolarQuant KV cache**: Q3 compression for longer context
- **OpenAI-compatible API**: drop-in replacement for OpenAI client
- **SSE streaming**: Server-Sent Events for token streaming
- **Health check**: `/health` endpoint for load balancers
- **Docker GPU**: `nvidia-container-toolkit` for GPU passthrough
- **BF16 native**: all computation in bfloat16

## Output

Tell the user:
1. Files generated and locations
2. Docker command to start serving
3. Curl command to test the endpoint
4. Estimated VRAM and throughput
5. OpenAI client example

## Argument: $ARGUMENTS
