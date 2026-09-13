/* gemm_mxfp4.cpp — see gemm_mxfp4.h for the layout, numerics and provenance
 * this kernel is built on.
 *
 * -ffp-contract=off is load-bearing here for the same reason gemm_i8.cpp pins
 * it (see that file's header comment): coli_gemm_mxfp4 and coli_gemm_mxfp4_ref
 * run the identical float accumulation `acc += wscale * (as0*(float)d0 +
 * as1*(float)d1)`, and an FMA-contracted build is free to round that
 * differently between two call sites even with nothing else different. */
#define _GNU_SOURCE
#include "gemm_mxfp4.h"
#include <stdlib.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* e2m1 values, DOUBLED — identical table to c/ggml_dequant.h's
 * gguf_kvalues_mxfp4 (transcribed from the same upstream source, see that
 * file's PROVENANCE-2). Not #included from there: this is a small standalone
 * C++ TU, matching gemm_i8.cpp's convention of not depending on ../c headers. */
static const int8_t coli_mxfp4_lut[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* GGML_E8M0_TO_FP32_HALF, bit-identical to c/ggml_dequant.h's
 * gguf_e8m0_to_fp32_half — see that file's PROVENANCE-2 for the upstream
 * source. Kept as a second transcription rather than a shared include so this
 * kernel has no dependency on ../c, matching every other file in engine/src/. */
static inline float coli_mxfp4_e8m0_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = (uint32_t)0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

void coli_free_mxfp4(coli_w_mxfp4 *w) {
    if (!w) return;
    if (w->owned) free((void *)w->blocks);
    w->blocks = NULL;
}

/* One output row against every activation row. `nblk` = I/COLI_MXFP4_BLK,
 * `rowbytes` = nblk*COLI_MXFP4_BYTES -- both hoisted by the caller so this
 * function has no division in the hot loop. Shared, byte-identical, by
 * coli_gemm_mxfp4 and coli_gemm_mxfp4_ref: the ONLY difference between those
 * two entry points is whether the o-loop around this call carries an
 * `#pragma omp parallel for`, and each row's reduction is independent of every
 * other row's, so parallelizing the outer loop cannot change a single bit of
 * any one row's result. */
static inline void mxfp4_row(float *y, int64_t O, int64_t o,
                              const coli_a_i8 *a, int64_t I,
                              const uint8_t *wr, int64_t nblk) {
    const int64_t anb = I / COLI_ABLK;
    for (int r = 0; r < a->n; r++) {
        const int8_t *xr = a->q + (int64_t)r * I;
        const float  *as = a->scale + (int64_t)r * anb;
        float acc = 0.f;
        for (int64_t g = 0; g < nblk; g++) {
            const uint8_t *blk = wr + g * COLI_MXFP4_BYTES;
            const float wscale = coli_mxfp4_e8m0_half(blk[0]);
            const uint8_t *qs = blk + 1;
            const int64_t ab = g * 2;              /* two COLI_ABLK=16 act. blocks per group */
            int32_t d0 = 0, d1 = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t byte = qs[j];
                d0 += (int32_t)xr[ab*16       + j] * (int32_t)coli_mxfp4_lut[byte & 0x0F];
                d1 += (int32_t)xr[(ab+1)*16   + j] * (int32_t)coli_mxfp4_lut[byte >>   4];
            }
            acc += wscale * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
        }
        y[(int64_t)r * O + o] = acc;
    }
}

void coli_gemm_mxfp4(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w) {
    const int64_t I = w->I, O = w->O;
    const int64_t nblk = I / COLI_MXFP4_BLK;
    const int64_t rowbytes = nblk * COLI_MXFP4_BYTES;
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        mxfp4_row(y, O, o, a, I, w->blocks + o*rowbytes, nblk);
    }
}

void coli_gemm_mxfp4_ref(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w) {
    const int64_t I = w->I, O = w->O;
    const int64_t nblk = I / COLI_MXFP4_BLK;
    const int64_t rowbytes = nblk * COLI_MXFP4_BYTES;
    for (int64_t o = 0; o < O; o++) {
        mxfp4_row(y, O, o, a, I, w->blocks + o*rowbytes, nblk);
    }
}

/* See gemm_i8.h's coli_gemm_i4_multi for the pattern this mirrors: one
 * concatenated row-space, one parallel region, one thread-local scratch-free
 * pass (MXFP4 needs no unpack scratch -- each block is decoded and consumed
 * immediately -- so unlike the int4 multi kernel there is nothing to
 * pre-allocate per thread). */
void coli_gemm_mxfp4_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                            const coli_w_mxfp4 *const *ws, int cnt) {
    if (cnt <= 0) return;
    /* Fixed-size offset table, same MAXM convention as coli_gemm_i4_multi. */
    enum { MAXM = 256 };
    if (cnt <= MAXM) {
        int64_t off[MAXM+1];
        off[0] = 0;
        for (int j = 0; j < cnt; j++) off[j+1] = off[j] + ws[j]->O;
        const int64_t R = off[cnt];
        #pragma omp parallel
        {
            int j = 0;
            #pragma omp for schedule(static)
            for (int64_t rr = 0; rr < R; rr++) {
                while (rr >= off[j+1]) j++;
                while (rr <  off[j])   j--;   /* first row of a static chunk may sit behind */
                const coli_w_mxfp4 *w = ws[j];
                const int64_t o = rr - off[j];
                const int64_t I = w->I, nblk = I / COLI_MXFP4_BLK, rowbytes = nblk * COLI_MXFP4_BYTES;
                const int r = arow[j];
                coli_a_i8 v = *a;
                /* one-row view of activation row `r`, matching the fallback
                 * path in coli_gemm_i4_multi -- mxfp4_row indexes a->q /
                 * a->scale by [r][*] against a->I, so give it I=w->I and
                 * n=1 with q/scale advanced to row r's data. */
                const int64_t anb = I / COLI_ABLK;
                v.q     = a->q     + (int64_t)r * I;
                v.scale = a->scale + (int64_t)r * anb;
                v.n = 1; v.I = I;
                mxfp4_row(ys[j], w->O, o, &v, I, w->blocks + o*rowbytes, nblk);
            }
        }
        return;
    }
    /* Fallback for cnt > MAXM: one matrix at a time through the standard
     * dispatch, on a one-row view of the activation (same shape as
     * coli_gemm_i4_multi's fallback). */
    for (int j = 0; j < cnt; j++) {
        const int64_t I = ws[j]->I, anb = I / COLI_ABLK;
        const int r = arow[j];
        coli_a_i8 v = *a;
        v.q = a->q + (int64_t)r*I; v.scale = a->scale + (int64_t)r*anb;
        v.n = 1; v.I = I;
        coli_gemm_mxfp4(ys[j], &v, ws[j]);
    }
}
