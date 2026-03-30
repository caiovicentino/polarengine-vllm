// polar_gemv_cuda.cu -- PolarEngine CUDA GEMV kernels
//
// Replaces the Triton polar_gemv_v4_kernel and polar_gemv_v4_packed_kernel.
// CUDA graph compatible: no host-side autotuning, fixed grid/block dims.
//
// Two variants:
//   1) polar_gemv_kernel        -- int8 codes (1 byte per code)
//   2) polar_gemv_packed_kernel -- nibble-packed uint8 (2 codes per byte, Q4)
//
// Compute capability: 7.0+ (Volta / Turing / Ampere / Hopper / Blackwell)

#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Kernel 1: unpacked codes (int8, 1 byte per code)
// ---------------------------------------------------------------------------
// Template parameter BLOCK_M: number of output rows per thread block.
// BLOCK_K is always 128 (matches block_size).
//
// Grid:  (ceil(out_f / BLOCK_M), 1, 1)
// Block: (BLOCK_K, 1, 1)   -- 128 threads
//
// Each thread handles one K-position across all BLOCK_M rows.
// For every K-block:
//   1. Cooperatively load BLOCK_K x-values into shared memory.
//   2. Load the centroid table into shared memory (done once).
//   3. Each thread loads BLOCK_M codes (one per row at its K-position),
//      performs centroid lookup, multiplies by norm, multiplies by x,
//      and accumulates into a register per row.
//   4. After all K-blocks: warp-reduce the partial sums across K and
//      atomicAdd into global output (only the first warp does the store).
//
// Shared memory layout:
//   float smem_x[BLOCK_K];               -- input vector tile
//   float smem_ct[MAX_LEVELS];            -- centroid LUT (max 64 for 6-bit)
//   float smem_acc[BLOCK_M * WARPS_PER_BLOCK]; -- partial accumulators
//      (not needed if we reduce differently)
//
// We actually use a simpler strategy: 128 threads, each thread owns one k-lane.
// Each thread accumulates BLOCK_M partial products across all n_blocks, then we
// reduce across the 128 k-lanes using shared memory.

template <int BLOCK_M, int BLOCK_K>
__global__ void polar_gemv_kernel(
    const int8_t*  __restrict__ codes,       // (out_f, in_f_padded)
    const float*   __restrict__ x,           // (in_f_padded,) pre-FWHT transformed
    const __half*  __restrict__ norms,       // (out_f, n_blocks)
    const float*   __restrict__ ct_scaled,   // (n_levels,) pre-scaled centroids
    float*         __restrict__ output,      // (out_f,)
    int out_f,
    int in_f_padded,
    int n_blocks,
    int n_levels                             // number of centroid entries (e.g. 16)
)
{
    // Thread indexing
    const int tid = threadIdx.x;               // 0..BLOCK_K-1 (k-lane)
    const int bid = blockIdx.x;                // block row index
    const int row_base = bid * BLOCK_M;        // first output row for this block

    // Shared memory: centroid table + reduction buffer
    // Centroid table is tiny (max 64 floats = 256 bytes for 6-bit).
    // Reduction buffer: BLOCK_M floats per warp => BLOCK_M * (BLOCK_K/32) floats.
    constexpr int N_WARPS = BLOCK_K / 32;
    extern __shared__ float smem[];
    // Layout: [0 .. 63] centroid table  |  [64 .. 64 + BLOCK_M*N_WARPS - 1] reduction
    float* smem_ct  = smem;                                     // 64 floats max
    float* smem_red = smem + 64;                                // BLOCK_M * N_WARPS

    // Load centroid table into shared memory (all threads cooperate)
    if (tid < n_levels) {
        smem_ct[tid] = ct_scaled[tid];
    }
    __syncthreads();

    // Per-thread accumulators: one per output row handled by this block
    float acc[BLOCK_M];
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        acc[m] = 0.0f;
    }

    // Main loop over K-blocks
    for (int blk = 0; blk < n_blocks; ++blk) {
        // Global K offset for this thread in the current block
        const int k_global = blk * BLOCK_K + tid;

        // Load this thread's x value
        const float xv = x[k_global];

        // For each output row in BLOCK_M
        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            const int row = row_base + m;
            if (row < out_f) {
                // Load code (int8 -> int for indexing)
                int code = static_cast<int>(codes[row * in_f_padded + k_global]);
                // Centroid lookup from shared memory
                float cv = smem_ct[code];
                // Load norm (fp16 -> fp32)
                float nv = __half2float(norms[row * n_blocks + blk]);
                // Accumulate: centroid * norm * x
                acc[m] += cv * nv * xv;
            }
        }
    }

    // Warp-level reduction across the 32 threads in each warp
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        float val = acc[m];
        // Warp shuffle reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        // Lane 0 of each warp writes partial sum to shared memory
        if (lane_id == 0) {
            smem_red[m * N_WARPS + warp_id] = val;
        }
    }
    __syncthreads();

    // Final reduction: first warp reduces across N_WARPS partial sums and writes output
    if (warp_id == 0) {
        #pragma unroll
        for (int m = lane_id; m < BLOCK_M; m += 32) {
            const int row = row_base + m;
            if (row < out_f) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    sum += smem_red[m * N_WARPS + w];
                }
                output[row] = sum;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Kernel 2: nibble-packed codes (uint8, 2 codes per byte, Q4 only)
// ---------------------------------------------------------------------------
// Packing order (matches Triton packed kernel):
//   byte[i] = (code[64+i] << 4) | code[i]   for i in [0, 64)
//   Low nibble  -> first half of block  (k in [0,  HALF_K))
//   High nibble -> second half of block (k in [HALF_K, BLOCK_K))
//
// Each thread block has HALF_K (=64) threads. Each thread handles two k-lanes
// (one from each half-block) across all BLOCK_M rows.

template <int BLOCK_M, int HALF_K>
__global__ void polar_gemv_packed_kernel(
    const uint8_t* __restrict__ packed_codes,  // (out_f, in_f_padded/2)
    const float*   __restrict__ x,             // (in_f_padded,) pre-FWHT transformed
    const __half*  __restrict__ norms,         // (out_f, n_blocks)
    const float*   __restrict__ ct_scaled,     // (n_levels,) pre-scaled centroids
    float*         __restrict__ output,        // (out_f,)
    int out_f,
    int in_f_half,                             // in_f_padded / 2
    int n_blocks,
    int n_levels
)
{
    const int tid = threadIdx.x;                  // 0..HALF_K-1
    const int bid = blockIdx.x;
    const int row_base = bid * BLOCK_M;

    constexpr int BLOCK_K = HALF_K * 2;           // 128
    constexpr int N_WARPS = HALF_K / 32;          // 2

    extern __shared__ float smem[];
    float* smem_ct  = smem;
    float* smem_red = smem + 64;                  // BLOCK_M * N_WARPS

    // Load centroid table
    if (tid < n_levels) {
        smem_ct[tid] = ct_scaled[tid];
    }
    __syncthreads();

    float acc[BLOCK_M];
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        acc[m] = 0.0f;
    }

    for (int blk = 0; blk < n_blocks; ++blk) {
        // Load two x values for this thread: first half and second half
        const int k_base = blk * BLOCK_K;
        const float xv_lo = x[k_base + tid];
        const float xv_hi = x[k_base + HALF_K + tid];

        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            const int row = row_base + m;
            if (row < out_f) {
                // Load packed byte
                uint8_t packed = packed_codes[row * in_f_half + blk * HALF_K + tid];
                int lo_code = packed & 0xF;
                int hi_code = (packed >> 4) & 0xF;

                float cv_lo = smem_ct[lo_code];
                float cv_hi = smem_ct[hi_code];

                float nv = __half2float(norms[row * n_blocks + blk]);
                acc[m] += nv * (cv_lo * xv_lo + cv_hi * xv_hi);
            }
        }
    }

    // Warp-level reduction
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        float val = acc[m];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            smem_red[m * N_WARPS + warp_id] = val;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int m = lane_id; m < BLOCK_M; m += 32) {
            const int row = row_base + m;
            if (row < out_f) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    sum += smem_red[m * N_WARPS + w];
                }
                output[row] = sum;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Host-side launch wrappers (called from Python via pybind11 / torch extension)
// ---------------------------------------------------------------------------

torch::Tensor polar_gemv_cuda(
    torch::Tensor codes,        // int8  (out_f, in_f_padded)
    torch::Tensor x,            // float (in_f_padded,)
    torch::Tensor norms,        // half  (out_f, n_blocks)
    torch::Tensor ct_scaled,    // float (n_levels,)
    int out_f,
    int in_f_padded,
    int n_blocks,
    int block_size               // 128
)
{
    TORCH_CHECK(codes.dtype() == torch::kInt8,   "codes must be int8");
    TORCH_CHECK(x.dtype() == torch::kFloat32,    "x must be float32");
    TORCH_CHECK(norms.dtype() == torch::kFloat16, "norms must be float16");
    TORCH_CHECK(ct_scaled.dtype() == torch::kFloat32, "ct_scaled must be float32");
    TORCH_CHECK(block_size == 128, "block_size must be 128");

    auto output = torch::zeros({out_f}, x.options());
    const int n_levels = ct_scaled.size(0);

    // Choose BLOCK_M based on problem size
    // For large out_f, use BLOCK_M=4 for better occupancy;
    // register pressure with BLOCK_M>8 is too high for 128-thread blocks.
    // We use BLOCK_M=4 as a good balance between ILP and register usage.
    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_K = 128;
    constexpr int N_WARPS = BLOCK_K / 32;  // 4

    const int grid_x = cdiv(out_f, BLOCK_M);
    const dim3 grid(grid_x);
    const dim3 block(BLOCK_K);  // 128 threads

    // Shared memory: 64 floats (centroid LUT) + BLOCK_M * N_WARPS floats (reduction)
    const int smem_bytes = (64 + BLOCK_M * N_WARPS) * sizeof(float);

    polar_gemv_kernel<BLOCK_M, BLOCK_K><<<grid, block, smem_bytes>>>(
        codes.data_ptr<int8_t>(),
        x.data_ptr<float>(),
        reinterpret_cast<const __half*>(norms.data_ptr<at::Half>()),
        ct_scaled.data_ptr<float>(),
        output.data_ptr<float>(),
        out_f, in_f_padded, n_blocks, n_levels
    );

    return output;
}


torch::Tensor polar_gemv_packed_cuda(
    torch::Tensor packed_codes,  // uint8 (out_f, in_f_padded/2)
    torch::Tensor x,             // float (in_f_padded,)
    torch::Tensor norms,         // half  (out_f, n_blocks)
    torch::Tensor ct_scaled,     // float (n_levels,)
    int out_f,
    int in_f_half,
    int n_blocks,
    int block_size                // 128
)
{
    TORCH_CHECK(packed_codes.dtype() == torch::kByte, "packed_codes must be uint8");
    TORCH_CHECK(x.dtype() == torch::kFloat32,         "x must be float32");
    TORCH_CHECK(norms.dtype() == torch::kFloat16,      "norms must be float16");
    TORCH_CHECK(ct_scaled.dtype() == torch::kFloat32,  "ct_scaled must be float32");
    TORCH_CHECK(block_size == 128, "block_size must be 128");

    auto output = torch::zeros({out_f}, x.options());
    const int n_levels = ct_scaled.size(0);

    constexpr int BLOCK_M = 4;
    constexpr int HALF_K  = 64;
    constexpr int N_WARPS = HALF_K / 32;  // 2

    const int grid_x = cdiv(out_f, BLOCK_M);
    const dim3 grid(grid_x);
    const dim3 block(HALF_K);  // 64 threads

    const int smem_bytes = (64 + BLOCK_M * N_WARPS) * sizeof(float);

    polar_gemv_packed_kernel<BLOCK_M, HALF_K><<<grid, block, smem_bytes>>>(
        packed_codes.data_ptr<uint8_t>(),
        x.data_ptr<float>(),
        reinterpret_cast<const __half*>(norms.data_ptr<at::Half>()),
        ct_scaled.data_ptr<float>(),
        output.data_ptr<float>(),
        out_f, in_f_half, n_blocks, n_levels
    );

    return output;
}


// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

// Forward declarations (fwht_cuda.cu provides these)
torch::Tensor fwht_cuda(torch::Tensor data, int block_size);
void fwht_cuda_inplace(torch::Tensor data, int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polar_gemv",        &polar_gemv_cuda,        "PolarEngine GEMV (int8 codes)");
    m.def("polar_gemv_packed", &polar_gemv_packed_cuda,  "PolarEngine GEMV (nibble-packed codes)");
    m.def("fwht",              &fwht_cuda,               "Fast Walsh-Hadamard Transform (butterfly, returns copy)");
    m.def("fwht_",             &fwht_cuda_inplace,       "Fast Walsh-Hadamard Transform (butterfly, in-place)");
}
