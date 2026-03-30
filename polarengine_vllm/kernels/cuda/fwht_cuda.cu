// fwht_cuda.cu -- Fast Walsh-Hadamard Transform using butterfly pattern
//
// O(n log n) butterfly implementation: for block_size=128 this is 7 stages
// with 128*7 = 896 butterfly ops per block, vs 128*128 = 16384 for matmul.
//
// Each thread block processes one data block (128 floats).
// Processing is done entirely in shared memory.
//
// Compute capability: 7.0+ (Volta / Turing / Ampere / Hopper / Blackwell)

#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// FWHT butterfly kernel
// ---------------------------------------------------------------------------
// Grid:  (n_total_blocks, 1, 1)   -- one thread block per data block
// Block: (block_size / 2, 1, 1)   -- 64 threads for block_size=128
//
// Each thread handles one butterfly pair per stage. With 64 threads and 128
// elements, every element is covered exactly once per stage.
//
// 7 stages for block_size=128:
//   stage 0: stride = 1  (pairs:  0,1  2,3  4,5 ...)
//   stage 1: stride = 2  (pairs:  0,2  1,3  4,6 ...)
//   stage 2: stride = 4  (pairs:  0,4  1,5  2,6 ...)
//   ...
//   stage 6: stride = 64 (pairs:  0,64  1,65  2,66 ...)
//
// After all stages, divide by sqrt(block_size) to get the normalized transform.

template <int BLOCK_SIZE>
__global__ void fwht_butterfly_kernel(
    float* __restrict__ data,    // (n_total_blocks, BLOCK_SIZE) -- modified in-place
    int n_total_blocks
)
{
    const int block_idx = blockIdx.x;
    if (block_idx >= n_total_blocks) return;

    const int tid = threadIdx.x;   // 0 .. BLOCK_SIZE/2 - 1
    constexpr int HALF = BLOCK_SIZE / 2;

    // Shared memory for the data block
    __shared__ float smem[BLOCK_SIZE];

    // Each thread loads 2 elements into shared memory
    float* base = data + block_idx * BLOCK_SIZE;
    smem[tid]        = base[tid];
    smem[tid + HALF] = base[tid + HALF];
    __syncthreads();

    // Butterfly stages: log2(BLOCK_SIZE) stages
    // For block_size=128: 7 stages (stride 1, 2, 4, 8, 16, 32, 64)
    #pragma unroll
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        // Each thread computes one butterfly:
        // Pair elements at positions (lo, hi) where hi = lo + stride
        // Within groups of size 2*stride, the first `stride` elements
        // are the "lo" positions.
        int group_size = stride << 1;
        int group_id   = tid / stride;
        int in_group   = tid % stride;
        int lo = group_id * group_size + in_group;
        int hi = lo + stride;

        float a = smem[lo];
        float b = smem[hi];
        smem[lo] = a + b;
        smem[hi] = a - b;
        __syncthreads();
    }

    // Normalize by 1/sqrt(BLOCK_SIZE)
    const float inv_sqrt_n = rsqrtf(static_cast<float>(BLOCK_SIZE));
    smem[tid]        *= inv_sqrt_n;
    smem[tid + HALF] *= inv_sqrt_n;
    __syncthreads();

    // Write back to global memory
    base[tid]        = smem[tid];
    base[tid + HALF] = smem[tid + HALF];
}


// ---------------------------------------------------------------------------
// Host-side launch wrapper
// ---------------------------------------------------------------------------

torch::Tensor fwht_cuda(torch::Tensor data, int block_size) {
    // data: (N, block_size) or (N * block_size,) -- will be treated as
    //       (n_total_blocks, block_size) where n_total_blocks = numel / block_size.
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(block_size == 128, "Only block_size=128 is supported");

    const int64_t total_elements = data.numel();
    TORCH_CHECK(total_elements % block_size == 0,
                "Total elements must be divisible by block_size");

    const int n_total_blocks = static_cast<int>(total_elements / block_size);

    // Clone to avoid modifying the input in-place (caller can pass the same
    // tensor if in-place is desired via fwht_cuda_(data, block_size)).
    auto output = data.clone();

    constexpr int BS = 128;
    const int threads = BS / 2;  // 64 threads per block

    fwht_butterfly_kernel<BS><<<n_total_blocks, threads>>>(
        output.data_ptr<float>(),
        n_total_blocks
    );

    return output;
}


// In-place variant
void fwht_cuda_inplace(torch::Tensor data, int block_size) {
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(block_size == 128, "Only block_size=128 is supported");

    const int64_t total_elements = data.numel();
    TORCH_CHECK(total_elements % block_size == 0,
                "Total elements must be divisible by block_size");

    const int n_total_blocks = static_cast<int>(total_elements / block_size);

    constexpr int BS = 128;
    const int threads = BS / 2;

    fwht_butterfly_kernel<BS><<<n_total_blocks, threads>>>(
        data.data_ptr<float>(),
        n_total_blocks
    );
}
