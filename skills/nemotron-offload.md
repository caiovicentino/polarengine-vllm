# Nemotron Expert Offloading — Colab Notebook Generator

Generate a complete Colab notebook (.ipynb) that runs Nemotron MoE models with expert offloading on consumer GPUs via our vLLM fork.

## Input

The user provides a Nemotron model name. Examples:
- `nvidia/Nemotron-Cascade-2-30B-A3B`
- `nvidia/Nemotron-Ultra-253B-v1`

## Instructions

1. **Parse the model**: Extract `owner/model-name`. Use WebFetch to check the HF page for: parameter count, MoE architecture (num_experts, top_k, num_layers), hybrid components (Mamba/attention), expert weight size.

2. **Estimate resources**:
   - Expert weights: count experts × layers × (intermediate × hidden × 2 × 2 bytes) for w13+w2
   - Non-expert weights: total - expert weights
   - GPU VRAM: non-expert + cache_buffer (cache_size × expert_size × layers × 2)
   - CPU RAM: expert weights (pinned memory)
   - Recommend cache_size based on target GPU

3. **Generate the Colab notebook**: Write `.ipynb` to `~/Desktop/NEMOTRON_OFFLOAD_{SHORT_NAME}.ipynb`.

## Notebook Structure (6 cells)

### Cell 0: Markdown header
```markdown
# Nemotron Expert Offloading — {MODEL_NAME}

**{TOTAL_PARAMS}B MoE model at ~{GPU_VRAM} GB VRAM, {EST_SPEED} tok/s.**

| Component | Location | Size |
|-----------|----------|------|
| Non-expert weights | GPU | {NON_EXPERT} GB |
| Expert cache ({CACHE_SIZE} slots) | GPU | {CACHE_BUFFER} GB |
| Expert weights (pinned) | CPU | {EXPERT_WEIGHTS} GB |
| **Total GPU** | | **{GPU_VRAM} GB** |

Architecture: {ARCH_DESCRIPTION}
```

### Cell 1: Install
```python
# Install vLLM fork with expert offloading
import subprocess, sys, os
os.environ["VLLM_USE_PRECOMPILED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install",
    "git+https://github.com/caiovicentino/vllm-expert-offload.git@nemotron-expert-offload",
    "-q"], check=True)
!pip install transformers -q
print("Installed!")
```

### Cell 2: Load Model
```python
import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "{MODEL}"
CACHE_SIZE = {CACHE_SIZE}  # Adjust: 8=~{VRAM_8}GB, 16=~{VRAM_16}GB, 32=~{VRAM_32}GB

llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    enforce_eager=True,
    moe_expert_cache_size=CACHE_SIZE,
    kernel_config={"moe_backend": "triton"},
    gpu_memory_utilization=0.95,
)
print("MODEL LOADED!")

import subprocess
smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True)
print(f"VRAM: {int(smi.stdout.strip())/1024:.1f} GB")
```

### Cell 3: Test Generation
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

prompts = [
    "What is 2+3? Think step by step.",
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to check if a number is prime.",
]

for prompt in prompts:
    p = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    out = llm.generate([p], SamplingParams(max_tokens=200, temperature=0))
    text = out[0].outputs[0].text
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {text[:300]}")
```

### Cell 4: Speed Benchmark
```python
import time

p = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a detailed essay about artificial intelligence."}],
    tokenize=False, add_generation_prompt=True)

# Warmup
_ = llm.generate([p], SamplingParams(max_tokens=10, temperature=0))

speeds = []
for run in range(3):
    t0 = time.time()
    out = llm.generate([p], SamplingParams(max_tokens=200, temperature=0))
    n = len(out[0].outputs[0].token_ids)
    tps = n / (time.time() - t0)
    speeds.append(tps)
    print(f"Run {run+1}: {tps:.1f} tok/s ({n} tokens)")

import subprocess
smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True)
print(f"\nAverage: {sum(speeds)/len(speeds):.1f} tok/s")
print(f"VRAM: {int(smi.stdout.strip())/1024:.1f} GB")
```

### Cell 5: PPL Test (WikiText-2)
```python
import math, time
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join([t for t in ds["text"] if t.strip()])
input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
print(f"WikiText-2: {len(input_ids)} tokens")

CTX = 2048
STRIDE = 512
nlls = []
t0 = time.time()

for i in range(0, len(input_ids) - CTX, STRIDE):
    chunk = input_ids[i:i+CTX]
    target = tokenizer.decode(chunk)
    out = llm.generate([target], SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0))
    logprobs = out[0].prompt_logprobs
    for j in range(CTX-STRIDE, CTX):
        if j < len(logprobs) and logprobs[j] is not None:
            token_id = chunk[j].item()
            if token_id in logprobs[j]:
                nlls.append(-logprobs[j][token_id].logprob)
    if len(nlls) >= 5000:
        break
    if (i // STRIDE) % 20 == 0 and nlls:
        print(f"  [{i//STRIDE}] {len(nlls)} tokens, PPL={math.exp(sum(nlls)/len(nlls)):.2f}")

ppl = math.exp(sum(nlls) / len(nlls))
print(f"\nPPL: {ppl:.4f} ({len(nlls)} tokens, {time.time()-t0:.0f}s)")
```

## Key Configuration Patterns

**Nemotron-Cascade-2-30B-A3B:**
- 52 layers: 23 Mamba + 23 MoE + 6 Attention
- 128 experts/layer, top-6, relu2 activation
- Expert weights: 58.7 GB (92.9%)
- Non-expert: 4.4 GB (7.1%)
- cache=8: ~7.6 GB GPU, ~60 GB CPU RAM
- Tested: 14.6-16.9 tok/s, PPL 6.09

**For other Nemotron MoE models:**
- Calculate expert_weight_ratio = expert_params / total_params
- If ratio > 80%, expert offloading is highly effective
- Cache size = 8 is good default (top_k=6, temporal locality)
- Larger cache = faster but more VRAM

**Required flags (always):**
- `enforce_eager=True` (no CUDA graph support)
- `kernel_config={"moe_backend": "triton"}` (FlashInfer CUTLASS may not support all GPUs)
- `FLASHINFER_DISABLE_VERSION_CHECK=1` (version mismatch workaround)
- `VLLM_USE_PRECOMPILED=1` during install (avoids CUDA compilation)

**Model card template:**
- Use emoji headers for HF
- Include before/after VRAM chart
- Include speed vs cache size table
- Link to fork and collection
- Add to collection: `caiovicentino1/nemotron-30b-consumer-gpu-inference-{slug}`

## Output

Tell the user:
1. File path of the notebook
2. Model specs (params, architecture, MoE details)
3. Estimated VRAM per cache size
4. Required GPU tier
5. CPU RAM requirement
