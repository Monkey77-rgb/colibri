/* ggml_dequant.h — bit-exact K-quant + F32/F16/BF16 dequantizers (header-only, all
 * static). Pure compute, no Model/QT/shards dependency -- matches quant.h's
 * existing header-only convention in this tree.
 *
 * WHY THIS EXISTS
 * ----------------
 * Stage 1 of the GGUF weight loader. Given a pointer to raw on-disk tensor bytes
 * (as located by gguf_reader.h's GgufIndex) and a ggml type, decode to f32. Five
 * types cover the measured minimum set for the two target models:
 *   - NetSec v9 Q4_K_M    : Q4_K, F32, Q6_K
 *   - Qwen3-30B-A3B Q3_K_M: F32, Q3_K, Q4_K, Q5_K, Q6_K
 *
 * PROVENANCE — TRANSCRIBED, not written from memory
 * Block layouts:  /home/monkey/.unsloth/llama.cpp/ggml/src/ggml-common.h
 *   block_q3_K  (lines 283-289), block_q4_K (295-306), block_q5_K (312-324),
 *   block_q6_K  (330-336). QK_K=256 (line 89), K_SCALE_SIZE=12 (line 90).
 * Decoders:       /home/monkey/.unsloth/llama.cpp/ggml/src/ggml-quants.c
 *   get_scale_min_k4    : lines 703-710
 *   dequantize_row_q3_K : lines 1128-1176
 *   dequantize_row_q4_K : lines 1352-1374
 *   dequantize_row_q5_K : lines 1554-1579
 *   dequantize_row_q6_K : lines 1762-1791
 * All of the above are (c) 2023-2026 The ggml authors, MIT licence
 * (https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).
 *
 * F16/BF16 -> F32 conversion is NOT reimplemented here: it is reused from st.h's
 * existing `f16_to_f32` / `bf16_to_f32` (both are the unique, lossless IEEE-754
 * half/bfloat16 -> single-precision widening -- any correct implementation, table-
 * or arithmetic-based, produces the same bits, which is why reuse rather than
 * re-derivation is safe here).
 *
 * BIT-EXACTNESS
 * Dequantization is a fixed integer-to-float decode with no reduction or
 * reassociation (unlike a matmul), so a correct port reproduces llama.cpp's f32
 * output to the last bit. That is the test bar in c/tests/, not a tolerance.
 *
 * This header does NOT depend on ggml headers -- the block structs below are
 * redeclared locally with `gguf_` / `GgufBlock` naming so this stays a standalone,
 * header-only unit like quant.h. Each struct's sizeof is asserted against the
 * upstream static_assert value at the bottom of this file.
 */
#ifndef COLI_GGML_DEQUANT_H
#define COLI_GGML_DEQUANT_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* must precede ANY system header in this TU -- st.h needs it
                         (pread/strdup/posix_fadvise); same guard pattern as st.h itself */
#endif

#include <stdint.h>
#include <string.h>

#include "st.h"   /* f16_to_f32 / bf16_to_f32 -- reused, not reimplemented */

#define GGUF_QK_K          256
#define GGUF_K_SCALE_SIZE  12

/* ---- block layouts, transcribed from ggml-common.h (see file header for exact
 * line ranges). Field order and sizes are copied verbatim; only the type/struct
 * names are prefixed to avoid colliding with any future ggml headers in this tree. */

typedef struct {
    uint8_t hmask[GGUF_QK_K/8];   /* quants - high bit */
    uint8_t qs[GGUF_QK_K/4];      /* quants - low 2 bits */
    uint8_t scales[12];           /* scales, quantized with 6 bits */
    uint16_t d;                   /* super-block scale, ggml_half (f16) */
} GgufBlockQ3K;

typedef struct {
    uint16_t d;                        /* super-block scale for quantized scales (f16) */
    uint16_t dmin;                     /* super-block scale for quantized mins (f16) */
    uint8_t  scales[GGUF_K_SCALE_SIZE]; /* scales and mins, quantized with 6 bits */
    uint8_t  qs[GGUF_QK_K/2];          /* 4-bit quants */
} GgufBlockQ4K;

typedef struct {
    uint16_t d;                        /* f16 */
    uint16_t dmin;                     /* f16 */
    uint8_t  scales[GGUF_K_SCALE_SIZE];
    uint8_t  qh[GGUF_QK_K/8];          /* quants, high bit */
    uint8_t  qs[GGUF_QK_K/2];          /* quants, low 4 bits */
} GgufBlockQ5K;

typedef struct {
    uint8_t ql[GGUF_QK_K/2];      /* quants, lower 4 bits */
    uint8_t qh[GGUF_QK_K/4];      /* quants, upper 2 bits */
    int8_t  scales[GGUF_QK_K/16]; /* scales, quantized with 8 bits */
    uint16_t d;                   /* f16 */
} GgufBlockQ6K;

/* Sizes asserted against upstream's static_assert (ggml-common.h lines 289, 306,
 * 324, 336): Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 bytes. A C99 file-scope trick
 * (negative-size array in an unused typedef) since this header must build as C99
 * without relying on C11 _Static_assert. */
typedef char gguf_assert_q3k_size[(sizeof(GgufBlockQ3K) == 110) ? 1 : -1];
typedef char gguf_assert_q4k_size[(sizeof(GgufBlockQ4K) == 144) ? 1 : -1];
typedef char gguf_assert_q5k_size[(sizeof(GgufBlockQ5K) == 176) ? 1 : -1];
typedef char gguf_assert_q6k_size[(sizeof(GgufBlockQ6K) == 210) ? 1 : -1];

/* ---- f32 / f16 / bf16: no quantization, just a widening copy/convert -------- */

static void gguf_dequant_f32(const void *src, float *dst, int64_t nelem) {
    memcpy(dst, src, (size_t)nelem * sizeof(float));
}

static void gguf_dequant_f16(const void *src, float *dst, int64_t nelem) {
    const uint16_t *s = (const uint16_t *)src;
    for (int64_t i = 0; i < nelem; i++) dst[i] = f16_to_f32(s[i]);
}

static void gguf_dequant_bf16(const void *src, float *dst, int64_t nelem) {
    const uint16_t *s = (const uint16_t *)src;
    for (int64_t i = 0; i < nelem; i++) dst[i] = bf16_to_f32(s[i]);
}

/* ---- get_scale_min_k4 -- transcribed verbatim from ggml-quants.c:703-710 ----
 * Shared by Q4_K and Q5_K: unpacks one of the 8 six-bit (scale,min) pairs packed
 * into the 12-byte `scales` array of a K4-family super-block. */
static inline void gguf_scale_min_k4(int j, const uint8_t q[12], uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

/* ---- Q3_K -- transcribed verbatim from ggml-quants.c:1128-1176 (dequantize_row_q3_K)
 * x = a * q, 16 sub-blocks of 16 elements, 3-bit quants + 1 high bit from hmask. */
static void gguf_dequant_q3_K(const void *src, float *dst, int64_t nblk) {
    const GgufBlockQ3K *x = (const GgufBlockQ3K *)src;
    float *y = dst;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int64_t i = 0; i < nblk; i++) {
        const float d_all = f16_to_f32(x[i].d);

        const uint8_t *q  = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < GGUF_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/* ---- Q4_K -- transcribed verbatim from ggml-quants.c:1352-1374 (dequantize_row_q4_K)
 * x = a*q + b, 8 sub-blocks of 32 elements, 4-bit quants, per-sub-block scale+min
 * from get_scale_min_k4. */
static void gguf_dequant_q4_K(const void *src, float *dst, int64_t nblk) {
    const GgufBlockQ4K *x = (const GgufBlockQ4K *)src;
    float *y = dst;

    for (int64_t i = 0; i < nblk; i++) {
        const uint8_t *q = x[i].qs;

        const float d   = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < GGUF_QK_K; j += 64) {
            gguf_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            gguf_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

/* ---- Q5_K -- transcribed verbatim from ggml-quants.c:1554-1579 (dequantize_row_q5_K)
 * x = a*q + b, 8 sub-blocks of 32 elements, 4-bit quants + 1 high bit from qh. */
static void gguf_dequant_q5_K(const void *src, float *dst, int64_t nblk) {
    const GgufBlockQ5K *x = (const GgufBlockQ5K *)src;
    float *y = dst;

    for (int64_t i = 0; i < nblk; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;

        const float d   = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < GGUF_QK_K; j += 64) {
            gguf_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            gguf_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

/* ---- Q6_K -- transcribed verbatim from ggml-quants.c:1762-1791 (dequantize_row_q6_K)
 * x = a*q, 16 sub-blocks of 16 elements, 4-bit low + 2-bit high quants, int8 scale. */
static void gguf_dequant_q6_K(const void *src, float *dst, int64_t nblk) {
    const GgufBlockQ6K *x = (const GgufBlockQ6K *)src;
    float *y = dst;

    for (int64_t i = 0; i < nblk; i++) {
        const float d = f16_to_f32(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        for (int n = 0; n < GGUF_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

#endif /* COLI_GGML_DEQUANT_H */
