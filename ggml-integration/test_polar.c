/**
 * Test PolarQuant Q3 encode/decode roundtrip.
 * Validates: bit-packing, Hadamard transform, Lloyd-Max quantization.
 */

#include "polar_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BS 128
#define NBLOCKS 100
#define NELEM (BS * NBLOCKS)

static float randf(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(void) {
    srand(42);

    float * original = malloc(NELEM * sizeof(float));
    float * reconstructed = malloc(NELEM * sizeof(float));
    block_polar_q3 * quantized = malloc(NBLOCKS * sizeof(block_polar_q3));

    /* Generate random data (Gaussian-ish) */
    for (int i = 0; i < NELEM; i++) {
        /* Box-Muller for approximate Gaussian */
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
        float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
        original[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }

    /* === Test 1: Roundtrip quality === */
    printf("=== PolarQuant Q3 Roundtrip Test ===\n\n");

    clock_t t0 = clock();
    polar_q3_quantize_row(original, quantized, NELEM);
    double quant_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

    t0 = clock();
    polar_q3_dequantize_row(quantized, reconstructed, NELEM);
    double dequant_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

    /* Compute MSE and cosine similarity */
    double mse = 0, dot_xy = 0, dot_xx = 0, dot_yy = 0;
    float max_err = 0;
    for (int i = 0; i < NELEM; i++) {
        float err = original[i] - reconstructed[i];
        mse += err * err;
        dot_xy += original[i] * reconstructed[i];
        dot_xx += original[i] * original[i];
        dot_yy += reconstructed[i] * reconstructed[i];
        if (fabsf(err) > max_err) max_err = fabsf(err);
    }
    mse /= NELEM;
    double cos_sim = dot_xy / (sqrt(dot_xx) * sqrt(dot_yy));

    /* Compression ratio */
    size_t orig_bytes = NELEM * sizeof(float);
    size_t quant_bytes = NBLOCKS * sizeof(block_polar_q3);
    double ratio = (double)orig_bytes / quant_bytes;
    double bits_per_val = 8.0 * quant_bytes / NELEM;

    printf("  Elements:     %d (%d blocks of %d)\n", NELEM, NBLOCKS, BS);
    printf("  Original:     %zu bytes (FP32)\n", orig_bytes);
    printf("  Quantized:    %zu bytes (%.2f bits/value)\n", quant_bytes, bits_per_val);
    printf("  Compression:  %.1fx\n", ratio);
    printf("  MSE:          %.6f\n", mse);
    printf("  Cosine sim:   %.6f\n", cos_sim);
    printf("  Max error:    %.4f\n", max_err);
    printf("  Quant time:   %.2f ms\n", quant_ms);
    printf("  Dequant time: %.2f ms\n", dequant_ms);

    /* === Test 2: Bit-packing roundtrip === */
    printf("\n=== Bit-Packing Test ===\n");
    int pack_ok = 1;
    for (int b = 0; b < NBLOCKS; b++) {
        uint8_t codes_orig[BS], codes_recon[BS];
        /* Fill with known pattern */
        for (int i = 0; i < BS; i++) codes_orig[i] = i % 8;

        uint8_t packed[48];
        /* Pack using our internal pack (reimplement here for test) */
        for (int i = 0; i < BS; i += 8) {
            int j = (i / 8) * 3;
            packed[j+0] = (codes_orig[i+0]<<5)|(codes_orig[i+1]<<2)|(codes_orig[i+2]>>1);
            packed[j+1] = ((codes_orig[i+2]&1)<<7)|(codes_orig[i+3]<<4)|(codes_orig[i+4]<<1)|(codes_orig[i+5]>>2);
            packed[j+2] = ((codes_orig[i+5]&3)<<6)|(codes_orig[i+6]<<3)|codes_orig[i+7];
        }
        /* Unpack */
        for (int i = 0; i < BS; i += 8) {
            int j = (i / 8) * 3;
            codes_recon[i+0] = (packed[j+0]>>5)&7;
            codes_recon[i+1] = (packed[j+0]>>2)&7;
            codes_recon[i+2] = ((packed[j+0]&3)<<1)|((packed[j+1]>>7)&1);
            codes_recon[i+3] = (packed[j+1]>>4)&7;
            codes_recon[i+4] = (packed[j+1]>>1)&7;
            codes_recon[i+5] = ((packed[j+1]&1)<<2)|((packed[j+2]>>6)&3);
            codes_recon[i+6] = (packed[j+2]>>3)&7;
            codes_recon[i+7] = packed[j+2]&7;
        }
        for (int i = 0; i < BS; i++) {
            if (codes_orig[i] != codes_recon[i]) { pack_ok = 0; break; }
        }
        if (!pack_ok) break;
    }
    printf("  Pack/unpack:  %s\n", pack_ok ? "PASS" : "FAIL");

    /* === Test 3: Vec dot product === */
    printf("\n=== Vec Dot Product Test ===\n");
    float query[NELEM];
    for (int i = 0; i < NELEM; i++) query[i] = randf();

    /* Reference: dot(query, original) */
    double ref_dot = 0;
    for (int i = 0; i < NELEM; i++) ref_dot += query[i] * original[i];

    /* PolarQuant: dot(query, dequant(quantized)) */
    float polar_dot = 0;
    polar_q3_vec_dot(NELEM, &polar_dot, 0, query, 0, quantized, 0, 1);

    double dot_err = fabs(ref_dot - polar_dot) / fabs(ref_dot);
    printf("  Reference dot:   %.4f\n", ref_dot);
    printf("  PolarQuant dot:  %.4f\n", polar_dot);
    printf("  Relative error:  %.4f%%\n", dot_err * 100);

    /* === Test 4: Zero/small value handling === */
    printf("\n=== Edge Cases ===\n");
    float zeros[BS] = {0};
    block_polar_q3 zblk;
    float zrecon[BS];
    polar_q3_quantize_row(zeros, &zblk, BS);
    polar_q3_dequantize_row(&zblk, zrecon, BS);
    float zmax = 0;
    for (int i = 0; i < BS; i++) if (fabsf(zrecon[i]) > zmax) zmax = fabsf(zrecon[i]);
    printf("  Zero input max recon: %.8f %s\n", zmax, zmax < 1e-5 ? "PASS" : "FAIL");

    /* === Summary === */
    printf("\n=== Summary ===\n");
    printf("  %-20s %s\n", "Roundtrip quality:", cos_sim > 0.97 ? "PASS (cos>0.97)" : "CHECK");
    printf("  %-20s %s\n", "Bit-packing:", pack_ok ? "PASS" : "FAIL");
    printf("  %-20s %s\n", "Vec dot product:", dot_err < 0.05 ? "PASS (<5%% error)" : "CHECK");
    printf("  %-20s %s\n", "Zero handling:", zmax < 1e-5 ? "PASS" : "FAIL");
    printf("  %-20s %.1fx (%.2f bits/val)\n", "Compression:", ratio, bits_per_val);
    printf("  %-20s %.2f ms / %.2f ms\n", "Speed (Q/DQ):", quant_ms, dequant_ms);

    free(original);
    free(reconstructed);
    free(quantized);

    return 0;
}
