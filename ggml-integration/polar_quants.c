/**
 * PolarQuant Q3 — Hadamard-rotated Lloyd-Max 3-bit quantization.
 *
 * Core operations:
 *   Quantize:   float[128] → norm + Hadamard → Lloyd-Max Q3 → bit-pack
 *   Dequantize: unpack → centroid lookup → inverse Hadamard → denorm
 *
 * The Walsh-Hadamard transform decorrelates values before quantization,
 * significantly improving reconstruction quality vs. simple round-to-nearest.
 */

#include "polar_quants.h"
#include <math.h>
#include <string.h>
#include <float.h>

#define BS POLAR_Q3_BLOCK_SIZE  /* 128 */

/* ─── Fast Walsh-Hadamard Transform (in-place, O(n log n)) ─── */

static void fwht_128(float * x) {
    /* 7 stages for n=128 (log2(128) = 7) */
    for (int len = 1; len < BS; len <<= 1) {
        for (int i = 0; i < BS; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    /* Normalize by 1/sqrt(128) */
    const float inv_sqrt_n = 1.0f / sqrtf((float)BS);
    for (int i = 0; i < BS; i++) {
        x[i] *= inv_sqrt_n;
    }
}

/* fwht_128 is its own inverse (orthogonal + symmetric) */
#define ifwht_128 fwht_128

/* ─── 3-bit packing: 8 codes → 3 bytes ─── */

static void pack_3bit(const uint8_t * codes, uint8_t * packed, int n) {
    /* Pack groups of 8 codes (3 bits each) into 3 bytes */
    for (int i = 0; i < n; i += 8) {
        uint8_t c0 = codes[i+0], c1 = codes[i+1], c2 = codes[i+2], c3 = codes[i+3];
        uint8_t c4 = codes[i+4], c5 = codes[i+5], c6 = codes[i+6], c7 = codes[i+7];
        int j = (i / 8) * 3;
        packed[j+0] = (c0 << 5) | (c1 << 2) | (c2 >> 1);
        packed[j+1] = ((c2 & 1) << 7) | (c3 << 4) | (c4 << 1) | (c5 >> 2);
        packed[j+2] = ((c5 & 3) << 6) | (c6 << 3) | c7;
    }
}

static void unpack_3bit(const uint8_t * packed, uint8_t * codes, int n) {
    /* Unpack 3 bytes → 8 codes (3 bits each) */
    for (int i = 0; i < n; i += 8) {
        int j = (i / 8) * 3;
        uint8_t b0 = packed[j+0], b1 = packed[j+1], b2 = packed[j+2];
        codes[i+0] = (b0 >> 5) & 7;
        codes[i+1] = (b0 >> 2) & 7;
        codes[i+2] = ((b0 & 3) << 1) | ((b1 >> 7) & 1);
        codes[i+3] = (b1 >> 4) & 7;
        codes[i+4] = (b1 >> 1) & 7;
        codes[i+5] = ((b1 & 1) << 2) | ((b2 >> 6) & 3);
        codes[i+6] = (b2 >> 3) & 7;
        codes[i+7] = b2 & 7;
    }
}

/* ─── FP16 conversion helpers ──�� */

static inline uint16_t fp32_to_fp16(float f) {
    /* Simple conversion — for production use the platform's intrinsic */
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (v.u >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static inline float fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        v.u = sign;
        return v.f;
    }
    if (exp == 31) {
        v.u = sign | 0x7F800000 | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    return v.f;
}

/* ─── Quantize one block of 128 floats ─── */

static void quantize_block_polar_q3(const float * x, block_polar_q3 * blk) {
    float buf[BS];

    /* 1. Compute L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < BS; i++) {
        norm_sq += x[i] * x[i];
    }
    float norm = sqrtf(norm_sq);
    if (norm < 1e-10f) norm = 1e-10f;
    blk->norm = fp32_to_fp16(norm);

    /* 2. Normalize */
    float inv_norm = 1.0f / norm;
    for (int i = 0; i < BS; i++) {
        buf[i] = x[i] * inv_norm;
    }

    /* 3. Walsh-Hadamard transform (decorrelate) */
    fwht_128(buf);

    /* 4. Scale by sqrt(block_size) for Lloyd-Max */
    const float scale = sqrtf((float)BS);  /* 11.3137... */
    for (int i = 0; i < BS; i++) {
        buf[i] *= scale;
    }

    /* 5. Lloyd-Max quantize: find nearest centroid */
    uint8_t codes[BS];
    for (int i = 0; i < BS; i++) {
        float val = buf[i];
        int best = 0;
        float best_dist = fabsf(val - POLAR_Q3_CENTROIDS[0]);
        for (int c = 1; c < POLAR_Q3_NUM_CENTROIDS; c++) {
            float dist = fabsf(val - POLAR_Q3_CENTROIDS[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best = c;
            }
        }
        codes[i] = (uint8_t)best;
    }

    /* 6. Bit-pack 128 codes → 48 bytes */
    pack_3bit(codes, blk->codes, BS);
}

/* ─── Dequantize one block of 128 values ─── */

static void dequantize_block_polar_q3(const block_polar_q3 * blk, float * x) {
    float buf[BS];

    /* 1. Unpack 3-bit codes */
    uint8_t codes[BS];
    unpack_3bit(blk->codes, codes, BS);

    /* 2. Centroid lookup + descale */
    const float inv_scale = 1.0f / sqrtf((float)BS);
    for (int i = 0; i < BS; i++) {
        buf[i] = POLAR_Q3_CENTROIDS[codes[i]] * inv_scale;
    }

    /* 3. Inverse Walsh-Hadamard transform */
    ifwht_128(buf);

    /* 4. Denormalize */
    float norm = fp16_to_fp32(blk->norm);
    for (int i = 0; i < BS; i++) {
        x[i] = buf[i] * norm;
    }
}

/* ─── Public API ─── */

void polar_q3_quantize_row(const float * src, block_polar_q3 * dst, size_t nelem) {
    size_t nblocks = nelem / BS;
    for (size_t b = 0; b < nblocks; b++) {
        quantize_block_polar_q3(src + b * BS, dst + b);
    }
}

void polar_q3_dequantize_row(const block_polar_q3 * src, float * dst, size_t nelem) {
    size_t nblocks = nelem / BS;
    for (size_t b = 0; b < nblocks; b++) {
        dequantize_block_polar_q3(src + b, dst + b * BS);
    }
}

void quantize_row_polar_q3(const float * x, void * y, int64_t k) {
    polar_q3_quantize_row(x, (block_polar_q3 *)y, (size_t)k);
}

void dequantize_row_polar_q3(const void * x, float * y, int64_t k) {
    polar_q3_dequantize_row((const block_polar_q3 *)x, y, (size_t)k);
}

/* ─── Vec dot product: float · polar_q3 ─── */

void polar_q3_vec_dot(int n, float * s, size_t bs,
                      const void * vx, size_t bx,
                      const void * vy, size_t by, int nrc) {
    /*
     * Compute dot product between float vector (vx) and polar_q3 vector (vy).
     * For KV cache attention: Q (float) · K (polar_q3) = attention score.
     *
     * Optimization: dequantize-then-dot. For production, fuse the operations.
     */
    const float * x = (const float *)vx;
    const block_polar_q3 * y = (const block_polar_q3 *)vy;

    float sum = 0.0f;
    int nblocks = n / BS;

    for (int b = 0; b < nblocks; b++) {
        float dequant[BS];
        dequantize_block_polar_q3(y + b, dequant);
        for (int i = 0; i < BS; i++) {
            sum += x[b * BS + i] * dequant[i];
        }
    }

    *s = sum;
}
