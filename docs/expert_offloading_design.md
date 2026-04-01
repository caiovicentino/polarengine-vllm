# PolarQuant Expert Offloading for Nemotron-Cascade-2-30B-A3B

## Design Document v1.0

**Target:** Run Nemotron-Cascade-2-30B-A3B on RTX 4090 (24 GB VRAM)
**Method:** PolarQuant-compressed MoE expert offloading with sparsity-aware caching
**Status:** Design phase

---

## 1. Architecture Analysis

### 1.1 Nemotron-Cascade-2-30B-A3B Structure

The model uses the `NemotronHForCausalLM` architecture -- a hybrid Mamba-Transformer
with Mixture of Experts. Key config values from HuggingFace:

| Parameter | Value |
|---|---|
| `model_type` | `nemotron_h` |
| `num_hidden_layers` | 52 |
| `hidden_size` | 2688 |
| `moe_intermediate_size` | 1856 |
| `moe_shared_expert_intermediate_size` | 3712 |
| `n_routed_experts` | 128 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` (top-K) | **6** |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 2 (GQA 16:1) |
| `head_dim` | 128 |
| `mamba_num_heads` | 64 |
| `mamba_head_dim` | 64 |
| `ssm_state_size` | 128 |
| `vocab_size` | 131072 |
| `max_position_embeddings` | 262144 |
| `hybrid_override_pattern` | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` |

### 1.2 Layer Type Distribution

The `hybrid_override_pattern` encodes layer types: M=Mamba, E=MoE, *=Attention.

| Layer Type | Count | Description |
|---|---|---|
| Mamba-2 (M) | 23 | SSM layers with selective scan |
| MoE (E) | 23 | 128 routed + 1 shared expert each |
| Attention (*) | 6 | GQA with 32 heads, 2 KV heads |
| **Total** | **52** | |

### 1.3 Per-Expert Parameter Count

Each routed expert is a SwiGLU-style FFN with squared ReLU activation:

| Projection | Shape | Parameters |
|---|---|---|
| `gate_proj` | (1856, 2688) | 4,988,928 |
| `up_proj` | (1856, 2688) | 4,988,928 |
| `down_proj` | (2688, 1856) | 4,988,928 |
| **Total per expert** | | **14,966,784 (14.97M)** |

Shared expert uses `moe_shared_expert_intermediate_size=3712`, so 29.93M params each.

### 1.4 Model Parameter Breakdown

| Component | Parameters | BF16 Size |
|---|---|---|
| Mamba layers (23) | 0.78B | 1.56 GB |
| Attention layers (6) | 0.14B | 0.28 GB |
| Shared experts (23 layers x 1) | 0.69B | 1.38 GB |
| Routers (23 layers) | 7.9M | 0.016 GB |
| Layer norms | 0.3M | 0.001 GB |
| Embeddings | 0.35B | 0.70 GB |
| LM head | 0.35B | 0.70 GB |
| **Non-expert subtotal** | **2.32B** | **4.6 GB** |
| Routed experts (23 x 128) | 44.06B | 88.1 GB |
| **Total** | **46.38B** | **92.8 GB** |

The routed experts are **95%** of total parameters but only **top-6 of 128** (4.7%) are active per token.

---

## 2. PolarQuant Expert Compression

### 2.1 Quantization Strategy

Using PolarEngine's existing mixed-bit assignment:

| Expert Projection | Bits | Method |
|---|---|---|
| `gate_proj` | Q3 | Hadamard rotate + 8-level Lloyd-Max, nibble-packed |
| `up_proj` | Q3 | Same as gate_proj |
| `down_proj` | Q4 | Hadamard rotate + 16-level Lloyd-Max, nibble-packed |

### 2.2 Per-Expert Size After PolarQuant

For block_size=128:

| Component | gate_proj Q3 | up_proj Q3 | down_proj Q4 |
|---|---|---|---|
| codes (nibble-packed) | 2.49 MB | 2.49 MB | 2.58 MB |
| norms (fp16) | 0.08 MB | 0.08 MB | 0.08 MB |
| ct_scaled (fp32) | 32 B | 32 B | 64 B |
| **Subtotal** | **2.57 MB** | **2.57 MB** | **2.66 MB** |

| Metric | Value |
|---|---|
| **Expert PolarQuant total** | **7.81 MB** |
| Expert BF16 total | 29.93 MB |
| **Compression ratio** | **3.8x** |

### 2.3 Key Advantage: No Dequantization Step

PolarQuant codes are consumed directly by the Triton GEMV kernel during inference.
There is no separate "dequantize to FP16" step when loading an expert. The kernel
reads int8/packed-int4 codes, looks up pre-scaled centroids, and multiplies by
per-block norms in a single fused operation. This means:

- CPU-to-GPU transfer sends only 7.81 MB of compressed data per expert
- No GPU-side dequantization latency
- No temporary FP16 buffer allocation

---

## 3. VRAM Projections

### 3.1 Non-Expert VRAM (Always Resident on GPU)

| Component | Quantization | VRAM |
|---|---|---|
| Mamba layers (23) | BF16 (too sensitive) | 1.56 GB |
| Attention layers (6) | Q5 PolarQuant | 0.10 GB |
| Shared experts (23) | Q3/Q4 PolarQuant | 0.36 GB |
| Routers (23) | FP16 (must be precise) | 0.016 GB |
| Layer norms | FP16 | 0.001 GB |
| Embeddings | Q5 PolarQuant | 0.25 GB |
| LM head | Q6 PolarQuant | 0.30 GB |
| **Total non-expert** | | **2.59 GB** |

### 3.2 Expert Cache Scenarios

| Scenario | Experts on GPU | Expert VRAM | Total Model | KV Cache | **Total** | Fits 24 GB? |
|---|---|---|---|---|---|---|
| S1: top-6 hot | 6/layer (138 total) | 1.08 GB | 3.67 GB | ~2 GB | **5.7 GB** | YES (+18.3 GB) |
| S2: top-12 hot | 12/layer (276 total) | 2.15 GB | 4.75 GB | ~2 GB | **6.8 GB** | YES (+17.2 GB) |
| S3: top-16 hot | 16/layer (368 total) | 2.87 GB | 5.47 GB | ~2 GB | **7.5 GB** | YES (+16.5 GB) |
| S4: top-32 hot | 32/layer (736 total) | 5.75 GB | 8.34 GB | ~2 GB | **10.3 GB** | YES (+13.7 GB) |
| S5: ALL experts | 128/layer (2944 total) | 22.98 GB | 25.57 GB | ~2 GB | **27.6 GB** | NO |

### 3.3 Recommended Configuration: Scenario S3 (top-16)

- **Total VRAM: 7.5 GB** on RTX 4090 (24 GB)
- Leaves **16.5 GB** for KV cache, CUDA context, and batch processing
- At 262K context with GQA 16:1, KV cache for a single sequence is manageable
- 16 cached experts per layer covers **12.5%** of experts, but research shows
  expert activation follows a Zipf-like distribution -- the top 12-16 experts
  typically cover **60-80%** of activations

### 3.4 CPU RAM for Offloaded Experts

| Scenario | Offloaded Experts | CPU RAM |
|---|---|---|
| S3 (top-16 hot) | 112/layer (2576 total) | 20.1 GB |
| S4 (top-32 hot) | 96/layer (2208 total) | 17.2 GB |

This is well within typical system RAM (32-128 GB).

---

## 4. Expert Offloading System Design

### 4.1 Architecture Overview

```
                    GPU (24 GB VRAM)
    +------------------------------------------+
    |  Always Resident:                        |
    |    - Mamba layers (BF16)       1.56 GB   |
    |    - Attention layers (Q5)     0.10 GB   |
    |    - Shared experts (Q3/Q4)    0.36 GB   |
    |    - Routers + Norms + Embeds  0.57 GB   |
    |                                          |
    |  Expert Cache (hot experts):             |
    |    - LRU cache: 16 experts/layer         |
    |    - Total: 368 experts        2.87 GB   |
    |                                          |
    |  Expert Swap Buffer:                     |
    |    - Double-buffered           0.06 GB   |
    |    - (2 experts pre-allocated)           |
    |                                          |
    |  KV Cache + CUDA context       ~14 GB   |
    +------------------------------------------+
              |  PCIe 4.0/5.0  |
    +------------------------------------------+
    |  CPU Pinned Memory:                      |
    |    - All 2944 experts (PolarQuant codes)  |
    |    - ~23 GB pinned RAM                   |
    |    - Expert Activation Matrix Collection  |
    +------------------------------------------+
```

### 4.2 Component Design

#### 4.2.1 ExpertCacheManager

Manages the GPU-side expert cache using a hybrid eviction policy:

```python
class ExpertCacheManager:
    """Per-layer LRU cache for PolarQuant expert codes on GPU.

    Each cache entry holds:
      - codes (int8/uint8 packed): 7.81 MB per expert
      - norms (fp16): included in codes entry
      - ct_scaled (fp32): shared across all experts in a layer (32-64 bytes)

    Cache capacity: configurable (default 16 experts per layer).
    Eviction: frequency-weighted LRU (combines recency + activation count).
    """

    def __init__(self, n_layers, cache_size_per_layer=16):
        self.n_layers = n_layers
        self.cache_size = cache_size_per_layer
        # Per-layer: expert_id -> (codes, norms, last_used, use_count)
        self.cache = [{} for _ in range(n_layers)]
        self.swap_buffer = None  # Double-buffered GPU memory for async transfers

    def get_expert(self, layer_idx, expert_id):
        """Return cached expert tensors or None if cache miss."""

    def evict_and_load(self, layer_idx, expert_id, cpu_expert_store):
        """Evict least-valuable expert, async-copy new expert from CPU."""

    def prefetch(self, layer_idx, expert_ids, cpu_expert_store):
        """Async prefetch experts predicted by router lookahead."""
```

#### 4.2.2 ExpertOffloadStore

Manages CPU-side storage of all experts in pinned memory:

```python
class ExpertOffloadStore:
    """CPU pinned memory store for all PolarQuant expert codes.

    At initialization, loads ALL expert weights from safetensors into
    pinned CPU memory. Pinned memory enables DMA transfers to GPU
    without going through the CPU page cache, achieving near-peak
    PCIe bandwidth.

    Memory layout: contiguous per-expert blocks for efficient DMA.
    """

    def __init__(self, model_dir, polar_config):
        # Load all expert codes/norms into pinned memory
        self.experts = {}  # (layer_idx, expert_id) -> {codes, norms}
        self._load_all_experts(model_dir, polar_config)

    def get_expert(self, layer_idx, expert_id):
        """Return CPU-pinned expert tensors for async GPU transfer."""

    def _load_all_experts(self, model_dir, polar_config):
        """Load from safetensors into torch.cuda.pin_memory() tensors."""
```

#### 4.2.3 RouterPrefetcher

Exploits the router's output to prefetch experts before they are needed:

```python
class RouterPrefetcher:
    """Predict and prefetch experts using router logits.

    Key insight: MoE layers are interleaved with Mamba layers in the
    hybrid_override_pattern (MEMEM...). While the Mamba layer at
    position i+1 is computing, we can read the router logits from
    MoE layer i and prefetch experts for MoE layer i+2.

    The Mamba computation takes ~0.5-1ms, which is enough time to
    transfer 2-3 experts via PCIe 4.0 (0.31ms each).
    """

    def __init__(self, expert_cache, expert_store):
        self.cache = expert_cache
        self.store = expert_store
        self.stream = torch.cuda.Stream()  # Dedicated copy stream

    def on_router_output(self, layer_idx, router_logits):
        """Called after router computes expert selection.

        Uses the logits to predict which experts the NEXT MoE layer
        will need (based on activation correlation patterns).
        Also prefetches any cache-missing experts for the CURRENT
        layer's top-K selection.
        """

    def prefetch_current_layer(self, layer_idx, selected_expert_ids):
        """Immediately async-copy any cache-missing experts for this layer."""
```

### 4.3 Inference Flow

```
For each token:
  1. Embedding lookup (GPU, cached)

  For each layer i in [0..51]:
    if layer_type[i] == 'mamba':
      2a. Run Mamba-2 SSM (GPU, always resident)
      2b. OVERLAP: prefetch experts for next MoE layer (async PCIe copy)

    elif layer_type[i] == 'attention':
      3a. Run GQA attention (GPU, Q5 PolarQuant)

    elif layer_type[i] == 'moe':
      4a. Run router -> get top-6 expert IDs + weights
      4b. For each selected expert:
          - Check GPU cache -> HIT: use cached codes directly
          - MISS: wait for prefetched data (or sync-copy from CPU)
      4c. Run PolarQuant Triton GEMV for all 6 experts (fused)
      4d. Run shared expert (always resident)
      4e. Weighted sum of expert outputs + shared output
      4f. OVERLAP: start prefetching for next MoE layer based on
          router logit patterns

  5. LM head -> next token logits
```

### 4.4 Prefetching Strategy

#### 4.4.1 Same-Layer Prefetching

When the router selects top-6 experts, any cache misses trigger immediate
async copies on a dedicated CUDA stream. While the first cached experts
compute, the missing experts are being transferred.

With top-16 cache and 6 active experts per token:
- Expected cache hit rate: 60-80% (Zipf distribution of expert activations)
- Average misses per layer: 1-2 experts
- Transfer time for 2 misses: 0.62 ms (PCIe 4.0)

#### 4.4.2 Cross-Layer Prefetching

The hybrid pattern `MEMEM*E...` means every MoE layer is followed by a
Mamba layer (or attention). While the Mamba/attention layer computes,
we can prefetch experts for the next MoE layer.

**Temporal overlap budget:**
- Mamba layer compute: ~0.5-1.0 ms
- Expert transfer: ~0.31 ms each
- Can prefetch 1-3 experts during Mamba compute

**Prediction methods:**
1. **Greedy repeat:** Assume the next MoE layer uses similar experts.
   Research shows ~40-60% overlap between adjacent MoE layers.
2. **Expert Activation Matrix (EAM):** Track per-request activation patterns
   and match against a small collection of historical patterns (MoE-Infinity approach).
   This achieves 80%+ prediction accuracy with negligible overhead (~21us per query).
3. **Router logit forwarding:** Run a lightweight projection of the current
   hidden state through the next layer's router to get early expert predictions.

#### 4.4.3 Expected Performance

| Cache Hit Rate | Avg Misses/Layer | Sync Wait | Impact on tok/s |
|---|---|---|---|
| 80% (expected) | 1.2 | ~0 ms (prefetched) | < 5% slowdown |
| 60% (worst typical) | 2.4 | ~0.3 ms | ~10% slowdown |
| 0% (cold start) | 6.0 | ~1.9 ms | ~40% slowdown |

Cold start penalty applies only to the first token of a new request.

---

## 5. Integration with vLLM

### 5.1 Current vLLM Limitations

vLLM does **not** natively support MoE expert offloading as of early 2026.
The only mechanism is `cpu_offload_gb`, which indiscriminately moves weight
slabs to CPU memory and swaps them back synchronously -- no caching, no
prefetching, no awareness of MoE sparsity.

There is an active RFC for DeepSeek-R1 MoE offloading (issue #33869) that
proposes a Dual Batch Overlap (DBO) mode with expert caching, but this is
not yet merged.

### 5.2 Integration Options

#### Option A: Custom vLLM Model Runner (Recommended)

Implement expert offloading as part of a custom `NemotronHForCausalLM` model
definition that hooks into vLLM's existing model runner:

```python
# In the custom model definition:
class NemotronHMoELayer(nn.Module):
    def __init__(self, config, layer_idx, expert_cache, expert_store):
        self.router = nn.Linear(config.hidden_size, config.n_routed_experts)
        self.shared_expert = PolarQuantLinearMethod(...)  # Always on GPU
        self.expert_cache = expert_cache
        self.expert_store = expert_store

    def forward(self, hidden_states):
        # Router
        router_logits = self.router(hidden_states)
        topk_ids, topk_weights = top_k_softmax(router_logits, k=6)

        # Ensure experts are on GPU (cache or load)
        expert_tensors = []
        for expert_id in topk_ids:
            cached = self.expert_cache.get(self.layer_idx, expert_id)
            if cached is None:
                cached = self.expert_cache.evict_and_load(
                    self.layer_idx, expert_id, self.expert_store
                )
            expert_tensors.append(cached)

        # Run PolarQuant GEMV for each expert
        outputs = []
        for expert_codes, weight in zip(expert_tensors, topk_weights):
            out = polar_gemv_on_codes(hidden_states, expert_codes)
            outputs.append(out * weight)

        # Shared expert (always on GPU)
        shared_out = self.shared_expert(hidden_states)
        return sum(outputs) + shared_out
```

#### Option B: Standalone Inference Engine

Build a dedicated inference engine outside vLLM that uses PolarEngine's
kernels directly, without vLLM's scheduler and memory manager. This gives
full control over memory management but loses vLLM's batching, continuous
batching, and API server.

#### Option C: vLLM Plugin with Memory Hooks

Use PolarEngine's existing vLLM plugin architecture but extend the
`PolarQuantLinearMethod` to support lazy loading:

```python
class PolarQuantExpertMethod(PolarQuantLinearMethod):
    """Extended linear method with expert offloading support.

    Instead of loading all expert codes at initialization, loads
    them on-demand from CPU pinned memory with LRU caching.
    """

    def apply(self, layer, x, bias=None):
        # Check if this expert's codes are on GPU
        if not hasattr(layer, 'codes') or layer.codes is None:
            # Load from CPU store
            self._load_expert_to_gpu(layer)
        return super().apply(layer, x, bias)
```

### 5.3 Recommendation

**Option A** is the most practical path:
- Works within vLLM's existing model registry (`trust_remote_code=True`)
- Gets vLLM's continuous batching, API server, and scheduling for free
- Expert cache management is self-contained in the model definition
- No changes to vLLM core required
- Can be shipped as part of the `polarengine-vllm` package

---

## 6. Prior Art and Comparison

### 6.1 Existing MoE Offloading Systems

| System | Approach | Compression | GPU Memory | Speed |
|---|---|---|---|---|
| **mixtral-offloading** | HQQ quantization + LRU cache | ~2x (INT4) | 12-16 GB | Slow (no prefetch) |
| **MoE-Infinity** | Sparsity-aware EAM cache | None (FP16) | ~24 GB | 3-17x vs baselines |
| **FloE** | INT8 + predictive prefetch | ~2x (INT8) | 11 GB | 48.7x vs DeepSpeed |
| **DeepSpeed-MoE** | CPU offload + pipeline | None | Varies | High latency |
| **vLLM RFC #33869** | DBO + expert cache | None planned | TBD | TBD |
| **PolarQuant (ours)** | Q3/Q4 + LRU + prefetch | **3.8x** | **7.5 GB** | **Projected: ~30 tok/s** |

### 6.2 Our Advantage: Higher Compression = Smaller Transfers

PolarQuant achieves **3.8x compression** vs 2x for typical INT4/INT8 approaches.
This translates directly to:

1. **Smaller per-expert transfer:** 7.81 MB vs 15 MB (INT8) or 30 MB (FP16)
2. **More experts cached in same VRAM:** 16 experts = 2.87 GB vs 7.5 GB (INT8)
3. **Faster PCIe transfers:** 0.31 ms vs 0.60 ms (INT8) per expert
4. **No dequantization overhead:** Triton kernel consumes codes directly

### 6.3 Lessons from MoE-Infinity (Most Relevant Prior Work)

MoE-Infinity's Expert Activation Matrix (EAM) approach is directly applicable:

1. **Track per-request activation patterns** using an L x E matrix
2. **Match against historical patterns** via cosine similarity (21us overhead)
3. **Use layer proximity decay** to prioritize near-future layers for prefetching
4. **Cluster historical patterns** with K-means (120 clusters covers 1000 sequences)

We should implement EAM-based prediction as Phase 2, starting with simple
LRU + greedy-repeat prefetching in Phase 1.

---

## 7. Implementation Plan

### Phase 1: Basic Expert Offloading (MVP)

**Goal:** Run Nemotron on RTX 4090 with basic cache, validate quality.

1. Extend `PolarQuantizer` to handle MoE experts:
   - Iterate over `model.layers[i].mlp.experts[j].{gate,up,down}_proj`
   - Quantize each expert independently (Q3/Q4)
   - Store expert codes with layer/expert indexing

2. Implement `ExpertOffloadStore`:
   - Load all expert codes into CPU pinned memory at init
   - Provide async GPU transfer via `torch.cuda.Stream`

3. Implement `ExpertCacheManager`:
   - Simple LRU cache (16 experts/layer)
   - Synchronous fallback when cache misses

4. Implement custom `NemotronHMoELayer`:
   - Hook into vLLM via `trust_remote_code=True`
   - Router runs on GPU (FP16), selects top-6
   - Cache lookup + sync load on miss

**Expected result:** ~20-25 tok/s on RTX 4090, 7-8 GB VRAM.

### Phase 2: Async Prefetching

**Goal:** Hide transfer latency, approach full-speed inference.

1. Implement cross-layer prefetching:
   - Dedicated CUDA copy stream
   - Prefetch during Mamba layer compute
   - Greedy-repeat prediction (assume next layer uses same experts)

2. Implement same-layer async loading:
   - Start loading cache-miss experts immediately
   - Process cached experts first, sync-wait only if needed

**Expected result:** ~28-33 tok/s on RTX 4090, 7-8 GB VRAM.

### Phase 3: Intelligent Prediction (EAM)

**Goal:** Maximize cache hit rate with sparsity-aware prediction.

1. Implement Expert Activation Matrix tracking
2. Build EAMC (clustered historical patterns)
3. Cosine-similarity matching for expert prediction
4. Layer proximity decay for prefetch prioritization

**Expected result:** 80%+ cache hit rate, ~30-35 tok/s.

### Phase 4: Fused Expert Kernel

**Goal:** Eliminate per-expert kernel launch overhead.

1. Fused Triton kernel that processes all 6 selected experts in one launch
2. Batch the FWHT transform across experts (shared input)
3. Fused weighted sum of expert outputs

**Expected result:** ~33-40 tok/s on RTX 4090.

---

## 8. Mamba-Aware Quantization Considerations

From previous experiments, Mamba SSM tensors are extremely sensitive to
quantization. The following MUST stay in BF16/FP16:

| Tensor | Reason |
|---|---|
| `mamba.A_log` | SSM eigenvalues; quantization destroys dynamics |
| `mamba.D` | Skip connection scaling; high sensitivity |
| `mamba.dt_bias` | Discretization timing; must be precise |
| `mamba.conv1d.weight` | Small conv kernel (4096 x 4); too few params |
| All layer norms | Normalization precision critical |
| Router weights | Expert selection must be accurate |

The existing `_SKIP_PATTERNS` in `utils.py` already handles these correctly:
```python
_SKIP_PATTERNS = ("norm", "layernorm", "rmsnorm", "a_log", "dt_bias", "conv1d")
_SKIP_PATTERNS_CASE_SENSITIVE = (".D", "A_log")
```

Router weights are also skipped:
```python
if name.endswith(".gate.weight") or "router" in name:
    return 16
```

---

## 9. Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| Expert activation is uniform (not Zipf) | Low cache hit rate, frequent swaps | Increase cache size to 32/layer (still fits 24 GB at 10.3 GB) |
| PCIe bandwidth contention with KV cache | Slower transfers | Use dedicated CUDA copy stream, separate from compute |
| Mamba SSM state size limits batch size | Reduced throughput | Mamba state = 128 x 64 x 64 = 0.5 MB/layer, manageable |
| Quality degradation from Q3 experts | Higher perplexity | Test Q4/Q5 fallback (still fits: Q4 all = 9.5 GB, Q5 all = 11.2 GB) |
| vLLM model runner incompatibility | Cannot use vLLM features | Fall back to standalone engine (Option B) |
| Cross-layer prediction inaccuracy | Prefetch misses | Start with same-layer async (no prediction needed) |

---

## 10. Summary

**Can Nemotron-Cascade-2-30B-A3B fit in 24 GB with PolarQuant expert offloading? YES.**

| Metric | Value |
|---|---|
| Total model (BF16) | 92.8 GB |
| Non-expert VRAM (PolarQuant) | 2.59 GB |
| Expert cache (16/layer, PolarQuant) | 2.87 GB |
| KV cache + CUDA overhead | ~2-3 GB |
| **Total VRAM** | **~7.5 GB** |
| **VRAM headroom on RTX 4090** | **~16.5 GB** |
| CPU RAM for offloaded experts | 20.1 GB |
| Per-expert transfer latency (PCIe 4.0) | 0.31 ms |
| Expected cache hit rate | 60-80% |
| **Projected throughput** | **28-35 tok/s** |

The combination of PolarQuant's 3.8x compression and sparsity-aware expert
caching makes this feasible. The key insight is that PolarQuant codes are
consumed directly by the Triton kernel -- there is no dequantization step,
so the transfer size IS the compute-ready size. This is a fundamental
advantage over INT4/INT8 approaches that must dequantize after transfer.

---

## Sources

- [NVIDIA Nemotron-Cascade-2 Research Page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- [Nemotron-Cascade-2-30B-A3B on HuggingFace](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
- [Nemotron 3 Nano Technical Report](https://arxiv.org/html/2512.20848v1)
- [MoE-Infinity: Sparsity-Aware Expert Cache](https://arxiv.org/html/2401.14361v3)
- [FloE: On-the-Fly MoE Inference](https://arxiv.org/pdf/2505.05950)
- [mixtral-offloading (GitHub)](https://github.com/dvmazur/mixtral-offloading)
- [vLLM Expert Offloading Discussion](https://discuss.vllm.ai/t/enable-expert-offloading/1884)
- [vLLM DeepSeek-R1 MoE Offload RFC](https://github.com/vllm-project/vllm/issues/33869)
- [Mixture of Lookup Experts (MoLE)](https://arxiv.org/html/2503.15798v2)
- [NVIDIA Nemotron 3 Family Paper](https://arxiv.org/pdf/2512.20856)
