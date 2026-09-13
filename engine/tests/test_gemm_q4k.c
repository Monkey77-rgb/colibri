#include "../src/platform.h"
/* test_gemm_q4k — does the native Q4_K GEMV agree with (a) its own scalar
 * reference and (b) an independent dequant-to-f32 dot product, and can a
 * perturbed kernel be told apart from a correct one?
 *
 * THIS FILE IS A Q4_K QUANTIZER, ON PURPOSE. gemm_q4k.h has no quantize-to-
 * Q4_K entry point -- Q4_K blocks come from a GGUF file in production, not
 * from this engine -- so proving the KERNEL against real data needs a small,
 * self-contained encoder here. It follows the exact bit layout
 * c/ggml_dequant.h's gguf_dequant_q4_K decodes (GgufBlockQ4K, get_scale_min_k4)
 * -- see pack_scale_min_k4 below, which is the byte-for-byte INVERSE of
 * gguf_scale_min_k4, derived directly from that function's own bit
 * expressions rather than from a second, independent reading of upstream, so
 * that a hand-encoded block here is understood-and-checked against the same
 * unpack the kernel/shim uses, not against an assumption about it.
 *
 * THREE-WAY COMPARISON, not two:
 *   1. dispatched kernel   (coli_gemm_q4k         -- AVX2 on this box)
 *   2. scalar reference    (coli_gemm_q4k_ref)
 *   3. dequant -> f32 dot  (this file's own plain float loop over the SAME
 *                           encoded blocks, decoded by hand here using the
 *                           identical (d,dmin,sc,mn) this test just packed --
 *                           an INDEPENDENT re-derivation of the algebra in
 *                           gemm_q4k.h's header comment, not a call into the
 *                           kernel's own decode helpers)
 * All three must agree to within a relative tolerance (int8-activation
 * quantization plus float summation-order differences, not exactness -- see
 * gemm_q4k.h's numerics note).
 *
 * NEGATIVE CONTROL: -DCOLI_BREAK_Q4K (gemm_q4k.cpp) drops the min-term
 * correction in the dispatched AVX2 kernel ONLY, leaving coli_gemm_q4k_ref
 * correct -- so kernel-vs-ref is the arm that must diverge. Perturbing one
 * IMPLEMENTATION rather than a shared input buffer matters: an earlier
 * version of this control instead corrupted one nibble of the block bytes
 * BOTH kernel and ref read, which left them agreeing with each other and
 * disagreeing with `truth` by an amount the rtol+atol bound could not
 * distinguish from ordinary quantization noise -- a control that could not
 * produce the opposite result (gemm_i8.h's own header names this exact
 * failure mode). See main()'s comment where the control is invoked.
 */
#include "../src/gemm_q4k.h"
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

/* ---- f16 <-> f32, IEEE754 round-to-nearest-even, copied verbatim from
 * c/tools/iq3_encode.c (already proven against numpy's astype(float16) there)
 * -- not re-derived, so this test's own encoder cannot disagree with the
 * engine about what a given f16 bit pattern means. */
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign = (x>>16)&0x8000u, e=(x>>23)&0xffu, man=x&0x7fffffu;
    if (e==0xff) return (uint16_t)(sign|0x7c00u|(man?(0x200u|(man>>13)):0u));
    int32_t ex=(int32_t)e-127+15;
    if (ex>=0x1f) return (uint16_t)(sign|0x7c00u);
    if (ex<=0) {
        if (ex<-10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t sh=(uint32_t)(14-ex), r=man>>sh, rem=man&((1u<<sh)-1u), half=1u<<(sh-1);
        if (rem>half || (rem==half && (r&1u))) r++;
        return (uint16_t)(sign|r);
    }
    uint32_t r=man>>13, rem=man&0x1fffu, o=((uint32_t)ex<<10)|r;
    if (rem>0x1000u || (rem==0x1000u && (r&1u))) o++;
    return (uint16_t)(sign|o);
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000u)<<16, e=(h>>10)&0x1fu, man=h&0x3ffu; uint32_t x;
    if (e==0) { if (man==0) x=sign; else { int sh=0; while(!(man&0x400u)){man<<=1;sh++;} man&=0x3ffu;
                x=sign|((uint32_t)(127-15-sh)<<23)|(man<<13); } }
    else if (e==0x1f) x=sign|0x7f800000u|(man<<13);
    else x=sign|((e-15+127)<<23)|(man<<13);
    float f; memcpy(&f,&x,4); return f;
}

/* Byte-for-byte inverse of gguf_scale_min_k4 (c/ggml_dequant.h). Derived from
 * that function's own bit expressions -- see this file's header comment. */
static void pack_scale_min_k4(uint8_t out[12], const uint8_t sc[8], const uint8_t mn[8]) {
    memset(out, 0, 12);
    for (int j = 0; j < 4; j++) { out[j] |= (uint8_t)(sc[j] & 0x3F); out[4+j] |= (uint8_t)(mn[j] & 0x3F); }
    for (int k = 0; k < 4; k++) {
        int j = 4+k;
        out[8+k] = (uint8_t)((sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4));
        out[k]   |= (uint8_t)(((sc[j] >> 4) & 0x3) << 6);
        out[4+k] |= (uint8_t)(((mn[j] >> 4) & 0x3) << 6);
    }
}

/* Encode ONE super-block from 256 f32 weights, following the exact structure
 * gemm_q4k.h documents (4 segments of 32 bytes, sub-block b's low/high nibble
 * split). `truth` gets back the actual dequantized weight this block encodes
 * -- i.e. AFTER f16 and 6-bit rounding -- so the "dequant->f32 dot" arm of
 * the test compares against what the block really says, not against the
 * pre-quantization float. Returns the packed 144-byte block in `blkbytes`. */
static void encode_q4k_block(const float *x256, uint8_t blkbytes[COLI_Q4K_BLOCK_BYTES], float truth[256]) {
    float d1[8], m1[8];
    for (int b = 0; b < 8; b++) {
        const float *xs = x256 + b*32;
        float mn = xs[0], mx = xs[0];
        for (int i = 1; i < 32; i++) { if (xs[i] < mn) mn = xs[i]; if (xs[i] > mx) mx = xs[i]; }
        float d = (mx > mn) ? (mx - mn) / 15.0f : 0.f;
        if (d < 1e-9f) d = 1e-9f;
        d1[b] = d; m1[b] = -mn;   /* w = d*q - m1 = d*q + mn, so m1 = -mn */
    }
    float dscale = 0.f, dmin = 0.f;
    for (int b = 0; b < 8; b++) { if (d1[b] > dscale) dscale = d1[b]; if (m1[b] > dmin) dmin = m1[b]; }
    dscale = (dscale > 1e-9f) ? dscale/63.0f : 1e-9f;
    dmin   = (dmin   > 1e-9f) ? dmin/63.0f   : 1e-9f;
    uint16_t d_f16 = f32_to_f16(dscale), dmin_f16 = f32_to_f16(dmin);
    float d_act = f16_to_f32(d_f16), dmin_act = f16_to_f32(dmin_f16);

    uint8_t sc[8], mn8[8]; float deff[8], meff[8];
    for (int b = 0; b < 8; b++) {
        int s = (int)lrintf(d1[b]/d_act); if (s < 0) s = 0; if (s > 63) s = 63;
        int m = (int)lrintf(m1[b]/dmin_act); if (m < 0) m = 0; if (m > 63) m = 63;
        sc[b] = (uint8_t)s; mn8[b] = (uint8_t)m;
        deff[b] = d_act * s; meff[b] = dmin_act * m;
    }
    uint8_t packed_sm[12]; pack_scale_min_k4(packed_sm, sc, mn8);

    /* blkbytes layout matches GgufBlockQ4K: d(2) dmin(2) scales(12) qs(128) */
    memcpy(blkbytes+0, &d_f16, 2);
    memcpy(blkbytes+2, &dmin_f16, 2);
    memcpy(blkbytes+4, packed_sm, 12);
    uint8_t *qs = blkbytes+16;
    for (int j = 0; j < 4; j++) {
        int blo = 2*j, bhi = 2*j+1;
        const float *xlo = x256 + j*64, *xhi = x256 + j*64 + 32;
        float *tlo = truth + j*64, *thi = truth + j*64 + 32;
        for (int i = 0; i < 32; i++) {
            float invlo = 1.f/deff[blo], invhi = 1.f/deff[bhi];
            int qlo = (int)lrintf((xlo[i]+meff[blo])*invlo); if (qlo<0)qlo=0; if(qlo>15)qlo=15;
            int qhi = (int)lrintf((xhi[i]+meff[bhi])*invhi); if (qhi<0)qhi=0; if(qhi>15)qhi=15;
            qs[j*32+i] = (uint8_t)((qlo & 0xF) | ((qhi & 0xF) << 4));
            tlo[i] = deff[blo]*(float)qlo - meff[blo];
            thi[i] = deff[bhi]*(float)qhi - meff[bhi];
        }
    }
}

/* Build an O x I native Q4_K matrix (I must be a multiple of 256) from random
 * f32 weights with a plausible outlier distribution (same shape gemm_i4's own
 * test uses), returning both the packed matrix and its ground-truth f32
 * dequant (row-major, O*I) for the third comparison arm. */
static void build_random_q4k(int64_t I, int64_t O, coli_w_q4k *w, float **truth_out, unsigned seed) {
    int64_t nsb = I / COLI_Q4K_SUPERBLOCK;
    uint8_t *blocks = (uint8_t*)malloc((size_t)O*(size_t)nsb*COLI_Q4K_BLOCK_BYTES);
    float *truth = (float*)malloc((size_t)O*(size_t)I*sizeof(float));
    float *row = (float*)malloc((size_t)I*sizeof(float));
    srand(seed);
    for (int64_t o = 0; o < O; o++) {
        for (int64_t i = 0; i < I; i++) { float u=(float)(rand()%20001-10000)/10000.f; row[i]=u*u*u*0.6f; }
        for (int64_t sb = 0; sb < nsb; sb++) {
            uint8_t *blk = blocks + ((size_t)o*nsb+sb)*COLI_Q4K_BLOCK_BYTES;
            encode_q4k_block(row + sb*COLI_Q4K_SUPERBLOCK, blk, truth + o*I + sb*COLI_Q4K_SUPERBLOCK);
        }
    }
    free(row);
    w->blocks = blocks; w->owns = 1; w->I = I; w->O = O;
    *truth_out = truth;
}

int main(int argc, char **argv) {
    int64_t I = argc>1 ? atoll(argv[1]) : 2048, O = argc>2 ? atoll(argv[2]) : 2048;
    if (I % COLI_Q4K_SUPERBLOCK) { printf("I must be a multiple of %d\n", COLI_Q4K_SUPERBLOCK); return 2; }
    int64_t nb = I / COLI_ABLK;
    printf("test_gemm_q4k: I=%lld O=%lld kernel=%s\n",
           (long long)I, (long long)O, coli_gemm_q4k_kernel(1));

    coli_w_q4k w; float *truth;
    build_random_q4k(I, O, &w, &truth, 1234);
    /* -DCOLI_BREAK_Q4K (build-time, see gemm_q4k.cpp) drops the min-term
     * correction in ONE of the two dispatched-kernel implementations only,
     * leaving coli_gemm_q4k_ref correct -- so kernel-vs-ref is the arm that
     * must catch it below. An EARLIER version of this control instead poked
     * one nibble of the shared block buffer both kernel and ref read: kernel
     * and ref agreed with each other (0 diff) and disagreed with `truth` by
     * an amount the rtol+atol bound could not distinguish from ordinary
     * int8-activation noise (0 cells flagged, measured 2026-09-13) -- a
     * control that could not produce the opposite result. Perturbing one
     * IMPLEMENTATION instead of one shared INPUT is what gemm_i8.cpp's own
     * COLI_BREAK_WIDE/COLI_BREAK_I4 already do, for the same reason. */

    /* ---- correctness: n=3 activation rows, compare 3 ways ---- */
    int NM = 3;
    coli_a_i8 a = {0}; a.I = I; a.n = NM;
    a.q = (int8_t*)malloc((size_t)I*NM);
    a.scale = (float*)malloc((size_t)nb*NM*sizeof(float));
    a.sum = (int32_t*)malloc((size_t)nb*NM*sizeof(int32_t));
    float *X = (float*)malloc((size_t)I*NM*sizeof(float));
    srand(77);
    for (int64_t i = 0; i < I*NM; i++) X[i] = (float)((rand()%2001)-1000)/450.0f;
    coli_quantize_a(&a, X, NM, I);

    float *y_kernel = (float*)malloc((size_t)O*NM*sizeof(float));
    float *y_ref    = (float*)malloc((size_t)O*NM*sizeof(float));
    float *y_dequant= (float*)malloc((size_t)O*NM*sizeof(float));
    coli_gemm_q4k(y_kernel, &a, &w);
    coli_gemm_q4k_ref(y_ref, &a, &w);
    /* Independent dequant->f32 dot: plain float loop over `truth` (the
     * decoded-by-THIS-FILE weight matrix) against the ORIGINAL f32
     * activations X -- no int8 quantization at all on this arm, so it also
     * exercises the int8-activation quantization error, not just the kernel
     * arithmetic. */
    for (int r = 0; r < NM; r++)
        for (int64_t o = 0; o < O; o++) {
            const float *xr = X + (int64_t)r*I, *wr = truth + o*I;
            float acc = 0.f; for (int64_t i = 0; i < I; i++) acc += xr[i]*wr[i];
            y_dequant[(int64_t)r*O+o] = acc;
        }

    /* A plain relative bound (|a-b|/|b|) is the wrong test near a
     * near-cancellation output: at I=2048 the true dot is a sum of ~2048
     * signed terms that partially cancel, so |y_dequant| can land close to
     * zero for a given (r,o) even though the ABSOLUTE quantization noise
     * floor (set by int8 activation rounding x the weight scale x sqrt(I))
     * does not shrink with it -- dividing by a near-zero denominator then
     * manufactures an arbitrarily large "relative error" out of an entirely
     * normal amount of noise. numpy.allclose's rtol+atol combination exists
     * for exactly this reason; atol here is one int8 activation quantization
     * step (~1/127) times a representative weight scale times sqrt(I), i.e.
     * the expected RMS noise floor of an I-term int8 dot product, not a
     * tuned fudge factor. */
    double sum_abs_w = 0.0; for (int64_t i = 0; i < O*I; i++) sum_abs_w += fabs(truth[i]);
    double mean_abs_w = sum_abs_w / (double)(O*I);
    double atol = mean_abs_w * (1.0/127.0) * sqrt((double)I) * 4.0;   /* 4x headroom, not tuned per run */
    double max_rel_kr = 0, max_rel_kd = 0, max_abs_kd = 0;
    int64_t worst_o = -1;
    int viol = 0;
    for (int64_t idx = 0; idx < O*(int64_t)NM; idx++) {
        double denom = fabs(y_dequant[idx]) > 1e-6 ? fabs(y_dequant[idx]) : 1e-6;
        double rel_kr = fabs(y_kernel[idx]-y_ref[idx]) / denom;
        double abs_kd = fabs(y_kernel[idx]-y_dequant[idx]);
        double rel_kd = abs_kd / denom;
        if (rel_kr > max_rel_kr) max_rel_kr = rel_kr;
        if (rel_kd > max_rel_kd) max_rel_kd = rel_kd;
        if (abs_kd > max_abs_kd) { max_abs_kd = abs_kd; worst_o = idx; }
        double bound = 0.10 * fabs(y_dequant[idx]) + atol;   /* rtol=10% + atol noise floor */
        if (abs_kd > bound) viol++;
    }
    printf("kernel vs ref        max relative diff: %.3e (must be ~0, same integer arithmetic)\n", max_rel_kr);
    printf("kernel vs dequant.f32 max relative diff: %.3e, max absolute diff: %.3e, atol(noise floor)=%.3e, "
           "cells over rtol=10%%+atol bound: %d/%lld [worst idx=%lld]\n",
           max_rel_kd, max_abs_kd, atol, viol, (long long)(O*NM), (long long)worst_o);

    /* ONE check, unconditional -- no separate ifdef branch for the control
     * build, same convention test_gemm_i4.c uses: a build compiled with
     * -DCOLI_BREAK_Q4K makes the AVX2 kernel wrong (see gemm_q4k.cpp), and
     * that wrongness is expected to trip THIS SAME check naturally. A
     * negative control that needed its own separate pass/fail logic to
     * "detect itself" would not actually be testing the comparison -- it
     * would be testing whether the ifdef fired, which is not the same
     * question. Exit nonzero here means "kernel disagrees with the
     * definition"; the Makefile's `make test` reads that exit code directly
     * (nonzero from the *_broken binary == "control failed as required",
     * matching test_gemm_i8_broken / test_gemm_i4_broken). */
    int fail = 0;
    /* kernel-vs-ref must be exact (same accumulation order per (r,o), see
     * gemm_q4k.cpp's comment): treat >1e-5 as a real dispatch bug, not noise. */
    if (max_rel_kr > 1e-5) { printf("FAIL: kernel disagrees with its own scalar reference\n"); fail = 1; }
    /* kernel-vs-dequant bound: rtol=10%% of the true value PLUS the int8
     * activation quantization noise floor (atol, see above) -- a plain
     * relative bound fails near cancellation outputs for reasons that have
     * nothing to do with kernel correctness (see the comment above `atol`). */
    if (viol > 0) { printf("FAIL: %d cell(s) exceed the rtol=10%%+atol bound\n", viol); fail = 1; }
    if (fail) { printf("(this is EXPECTED and correct under -DCOLI_BREAK_Q4K)\n"); return 1; }
    printf("PASS\n");

    /* ---- timing: decode shape (n=1) at the two requested expert shapes,
     * plus a same-matrix comparison against coli_gemm_i4 so the two formats
     * are timed on IDENTICAL weight VALUES (the dequantized `truth`), not
     * independent random draws. ---- */
    printf("\n--- timing, n=1 (decode), %lldx%lld, %s ---\n", (long long)I, (long long)O, coli_gemm_q4k_kernel(1));
    coli_a_i8 a1 = {0}; a1.I = I; a1.n = 1;
    a1.q = (int8_t*)malloc((size_t)I); a1.scale = (float*)malloc((size_t)nb*sizeof(float));
    a1.sum = (int32_t*)malloc((size_t)nb*sizeof(int32_t));
    float *x1 = (float*)malloc((size_t)I*sizeof(float));
    for (int64_t i = 0; i < I; i++) x1[i] = (float)((rand()%2001)-1000)/450.0f;
    coli_quantize_a(&a1, x1, 1, I);
    float *y1 = (float*)malloc((size_t)O*sizeof(float));

    coli_w_i4 w4; coli_quantize_w4(&w4, truth, I, O);   /* same VALUES, engine int4 format */

    enum { REPS = 15 };
    double best_q4k = 1e9, best_i4 = 1e9;
    for (int rp = 0; rp < REPS; rp++) { double t0=now(); coli_gemm_q4k(y1,&a1,&w); double dt=now()-t0; if(dt<best_q4k)best_q4k=dt; }
    for (int rp = 0; rp < REPS; rp++) { double t0=now(); coli_gemm_i4(y1,&a1,&w4); double dt=now()-t0; if(dt<best_i4)best_i4=dt; }
    double bytes_q4k = (double)O*I/2.0 + (double)O*(I/32)*0 ; /* nibbles only, scales are tiny/derived on the fly */
    bytes_q4k = (double)O * (I/COLI_Q4K_SUPERBLOCK) * COLI_Q4K_BLOCK_BYTES;
    double bytes_i4 = (double)O*I/2.0 + (double)O*(I/32)*sizeof(float);
    printf("native q4k : best of %d = %.4f ms  (%.2f GB/s, %.0f bytes/row-matrix)\n",
           REPS, best_q4k*1000.0, bytes_q4k/best_q4k/1e9, bytes_q4k);
    printf("engine int4: best of %d = %.4f ms  (%.2f GB/s, %.0f bytes/row-matrix)  ratio q4k/i4=%.3fx\n",
           REPS, best_i4*1000.0, bytes_i4/best_i4/1e9, bytes_i4, best_q4k/best_i4);

    return 0;
}
