/**
 * PolarQuant Q3 for GGML — Hadamard-rotated Lloyd-Max 3-bit quantization.
 *
 * Block structure: 128 values → Hadamard rotate → Lloyd-Max Q3 → bit-pack
 *
 * Storage per block of 128 values:
 *   - 48 bytes: 128 × 3-bit codes, bit-packed
 *   - 2 bytes:  1 × FP16 norm
 *   Total: 50 bytes / 128 values = 3.125 bits per value
 *
 * Designed for KV cache compression in llama.cpp:
 *   --cache-type-k polar_q3 --cache-type-v polar_q3
 *
 * (c) 2025 PolarQuant — github.com/caiovicentino/polarengine-vllm
 */

#ifndef POLAR_QUANTS_H
#define POLAR_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Block structure ─── */

#define POLAR_Q3_BLOCK_SIZE 128
#define POLAR_Q3_CODES_BYTES 48   /* 128 × 3 bits = 384 bits = 48 bytes */
#define POLAR_Q3_NUM_CENTROIDS 8

typedef struct {
    uint16_t norm;                         /* FP16 block norm */
    uint8_t  codes[POLAR_Q3_CODES_BYTES];  /* bit-packed 3-bit codes */
} block_polar_q3;
/* sizeof(block_polar_q3) = 50 bytes per 128 values */

/* ─── Lloyd-Max optimal centroids for N(0,1) at 3 bits ─── */
static const float POLAR_Q3_CENTROIDS[8] = {
    -2.1519461f, -1.3439096f, -0.7560055f, -0.2450942f,
     0.2450942f,  0.7560055f,  1.3439096f,  2.1519461f
};

/* ─── API ─── */

/**
 * Quantize a row of floats using PolarQuant Q3.
 * @param src   Input float array (must be multiple of 128)
 * @param dst   Output block_polar_q3 array
 * @param nelem Number of elements (must be multiple of 128)
 */
void polar_q3_quantize_row(const float * src, block_polar_q3 * dst, size_t nelem);

/**
 * Dequantize a row of PolarQuant Q3 blocks back to floats.
 * @param src   Input block_polar_q3 array
 * @param dst   Output float array
 * @param nelem Number of elements (must be multiple of 128)
 */
void polar_q3_dequantize_row(const block_polar_q3 * src, float * dst, size_t nelem);

/**
 * Quantize a row from floats, storing as PolarQuant Q3 (void* interface for GGML).
 */
void quantize_row_polar_q3(const float * x, void * y, int64_t k);

/**
 * Dequantize a row from PolarQuant Q3 to floats (void* interface for GGML).
 */
void dequantize_row_polar_q3(const void * x, float * y, int64_t k);

/**
 * Dot product: float vec · PolarQuant Q3 vec.
 * Used for attention score computation in KV cache.
 */
void polar_q3_vec_dot(int n, float * s, size_t bs, const void * vx, size_t bx,
                      const void * vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* POLAR_QUANTS_H */
