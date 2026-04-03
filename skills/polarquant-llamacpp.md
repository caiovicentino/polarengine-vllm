# PolarQuant llama.cpp Integration

Apply PolarQuant Q3 KV cache quantization to a llama.cpp clone for `--cache-type-k pq3_0 --cache-type-v pq3_0`.

## Input

The user provides either:
- A path to an existing llama.cpp clone: `/path/to/llama.cpp`
- Or "fresh" to clone from upstream

## Instructions

1. **Setup**: If "fresh", clone `https://github.com/ggml-org/llama.cpp`. Otherwise use the provided path.

2. **Apply the patch**: The patch is at `polarengine-vllm/ggml-integration/polarquant-q3.patch`. If not available locally, download from GitHub:
   ```bash
   curl -L https://raw.githubusercontent.com/caiovicentino/polarengine-vllm/main/ggml-integration/polarquant-q3.patch -o /tmp/pq3.patch
   git apply /tmp/pq3.patch
   ```

3. **If patch fails** (llama.cpp version mismatch), apply changes manually to these 14 files:

### Files to Modify

**GGML Type Registration (3 files):**

1. `ggml/include/ggml.h` — Add before GGML_TYPE_COUNT:
```c
GGML_TYPE_PQ3_0 = <next_id>,  // PolarQuant Q3: Hadamard-rotated Lloyd-Max 3-bit
```
Bump GGML_TYPE_COUNT. Add `GGML_FTYPE_MOSTLY_PQ3_0` to ggml_ftype enum.

2. `ggml/src/ggml-common.h` — Add block struct:
```c
#define QK_PQ3_0 128
typedef struct {
    ggml_half d;                          // block L2 norm
    uint8_t qs[QK_PQ3_0 * 3 / 8];        // 3-bit packed codes (48 bytes)
} block_pq3_0;
static_assert(sizeof(block_pq3_0) == sizeof(ggml_half) + QK_PQ3_0 * 3 / 8, "wrong pq3_0 block size");
```

3. `ggml/src/ggml.c` — Add type_traits entry:
```c
[GGML_TYPE_PQ3_0] = {
    .type_name      = "pq3_0",
    .blck_size      = QK_PQ3_0,
    .type_size      = sizeof(block_pq3_0),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_pq3_0,
    .from_float_ref = (ggml_from_float_t) quantize_row_pq3_0_ref,
},
```

**Core Quantization (2 files):**

4. `ggml/src/ggml-quants.h` — Declare:
```c
void quantize_row_pq3_0_ref(const float * GGML_RESTRICT x, block_pq3_0 * GGML_RESTRICT y, int64_t k);
void dequantize_row_pq3_0(const block_pq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
size_t quantize_pq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void ggml_vec_dot_pq3_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
```

5. `ggml/src/ggml-quants.c` — Implement (port from polarengine-vllm/ggml-integration/polar_quants.c):
   - `pq3_fwht_128()` — Fast Walsh-Hadamard Transform, in-place, O(n log n), 7 butterfly stages
   - `PQ3_CENTROIDS[8]` — Lloyd-Max optimal: {-2.152, -1.344, -0.756, -0.245, +0.245, +0.756, +1.344, +2.152}
   - `pq3_pack_3bit()` / `pq3_unpack_3bit()` — 8 codes ↔ 3 bytes
   - `quantize_row_pq3_0_ref()` — norm → normalize → FWHT → scale → nearest centroid → pack
   - `dequantize_row_pq3_0()` — unpack → lookup → descale → IFWHT → denorm
   - `quantize_pq3_0()` — batch wrapper (imatrix ignored)

**CPU Backend (3 files):**

6. `ggml/src/ggml-cpu/ggml-cpu.c` — Add type_traits_cpu:
```c
[GGML_TYPE_PQ3_0] = {
    .from_float   = quantize_row_pq3_0,
    .vec_dot      = ggml_vec_dot_pq3_0_q8_0,
    .vec_dot_type = GGML_TYPE_Q8_0,
    .nrows        = 1,
},
```

7. `ggml/src/ggml-cpu/quants.c` — Implement:
   - `quantize_row_pq3_0()` — void* wrapper for quantize_row_pq3_0_ref
   - `ggml_vec_dot_pq3_0_q8_0()` — dequant PQ3 block, dot with Q8_0 blocks

8. `ggml/src/ggml-cpu/quants.h` — Declare both functions

**CUDA Scaffold (4 files):**

9. `ggml/src/ggml-cuda/common.cuh` — Type traits template:
```cpp
template<>
struct ggml_cuda_type_traits<GGML_TYPE_PQ3_0> {
    static constexpr int qk = QK_PQ3_0;
    static constexpr int qr = 1;
    static constexpr int qi = QK_PQ3_0 / 4;
};
```

10. `ggml/src/ggml-cuda/convert.cu` — Add simplified dequant kernel (element-wise, no FWHT) + dispatch case

11. `ggml/src/ggml-cuda/mmvq.cu` — Add `case GGML_TYPE_PQ3_0: return nullptr;` (CPU fallback)

12. `ggml/src/ggml-cuda/ggml-cuda.cu` — Add PQ3_0 to type dispatch switch

**CLI + Docs (2 files):**

13. `common/arg.cpp` — Add `GGML_TYPE_PQ3_0` to `kv_cache_types` vector

14. `docs/polar-quant.md` — Documentation (copy from ggml-integration/)

### Build & Test

```bash
cmake -B build -DGGML_CUDA=OFF  # CPU-only first
cmake --build build -j$(nproc)
./build/bin/llama-cli -m model.gguf --cache-type-k pq3_0 --cache-type-v pq3_0 -c 65536
```

### Key Technical Details

- **Block size**: 128 (QK_PQ3_0) — larger than standard Q types (32) for Hadamard efficiency
- **FWHT**: Self-inverse, O(n log n), 7 butterfly stages for n=128
- **Lloyd-Max**: Optimal for Gaussian — after Hadamard rotation, distribution is ~N(0,1)
- **Bit-packing**: 8 codes × 3 bits = 24 bits = 3 bytes per group
- **Compression**: 3.125 bits/value = 10.2x from FP32, 5.1x from FP16
- **Quality**: cos_sim 0.983 on random data (tested)
- **vec_dot**: Dequant-then-dot strategy; FWHT fusion is future optimization
- **CUDA**: Simplified kernel (no FWHT) for initial support; full kernel is TODO

### Standalone C Library

For testing/embedding without llama.cpp:
```
polarengine-vllm/ggml-integration/
├── polar_quants.h    — header (block struct, API)
├── polar_quants.c    — implementation (FWHT, pack, quant/dequant)
├── test_polar.c      — tests (roundtrip, packing, dot product)
└── polar-quant.md    — documentation
```

Build: `gcc -O2 -Iinclude -o test src/polar_quants.c src/test_polar.c -lm && ./test`

## Output

Tell the user:
1. Whether patch applied cleanly or manual integration was needed
2. Build status (any errors to fix)
3. How to run: `--cache-type-k pq3_0 --cache-type-v pq3_0`
4. Expected compression: 5.1x KV cache, context ~5x longer
