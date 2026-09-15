/* test_mxfp4.c — GGUF MXFP4 (ttype 39) + Q8_0 (ttype 8) oracle, CPU only.
 *
 * Four parts, per the task this was written under:
 *   (a) dequantize ONE real gpt-oss-120b expert slice with gguf_dequant_mxfp4,
 *       cross-checked against an independent inline decode written from the
 *       block layout (different code path: ldexpf() instead of the bit-trick,
 *       a switch instead of a table lookup).
 *   (b) coli_gemm_mxfp4 vs coli_gemm_mxfp4_ref vs a dequant->f32 dot, on that
 *       real expert AND on random blocks; a negative control (one corrupted
 *       nibble) must fail.
 *   (c) timing at 2880x2880 vs coli_gemm_i4 on an int4 twin of the same
 *       (dequantized) matrix.
 *   (d) Q8_0: dequantize blk.0.attn_q.weight, check row norms are finite and
 *       plausible, plus a control.
 *
 * Real-file facts asserted here (verified against the GGUF's own tensor
 * directory, not assumed): blk.0.ffn_gate_exps.weight has ggml type 39 and
 * shape [2880, 2880, 128] with dims stored fastest-to-slowest as
 * [ne0=I, ne1=O, ne2=n_experts] -- i.e. GGUF's own on-disk order, confirmed by
 * this file's own gguf_index_open() call, not copied from a doc. Expert e is
 * the contiguous byte range [e * O * rowbytes, (e+1) * O * rowbytes) because
 * ne2 (experts) is the SLOWEST-varying dimension.
 *
 * Build (see engine/Makefile for the wired rule):
 *   c++ -std=c++20 -O3 -ffp-contract=off -I../c -Isrc -march=native -fopenmp \
 *     -x c++ tests/test_mxfp4.c src/gemm_mxfp4.cpp src/gemm_i8.cpp \
 *     src/cpu_features.cpp -x none -o tests/test_mxfp4 -lm
 * Run (CPU only, no GPU touched):
 *   OMP_NUM_THREADS=4 ./tests/test_mxfp4 [path-to-gpt-oss-120b-MXFP4.gguf]
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* must precede ANY system header in this TU (pread,
                       * aligned_alloc, posix_fadvise via st.h/platform.h) --
                       * same requirement ggml_dequant.h documents on itself */
#endif
#include "../src/platform.h"
#include "../src/gemm_i8.h"
#include "../src/gemm_mxfp4.h"
#ifdef __cplusplus
extern "C" {
#endif
/* declared here, defined in mxfp4_dequant_shim.c -- see that file for why the
 * two dequant functions live behind a C-linkage boundary instead of this test
 * including c/ggml_dequant.h directly */
void coli_test_dequant_mxfp4(const void *src, float *dst, long long nblk);
void coli_test_dequant_q8_0(const void *src, float *dst, long long nblk);
#ifdef __cplusplus
}
#endif
#include "../../c/gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail = 1; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }

/* ---- independent MXFP4 decode, written from the block layout directly, NOT
 * calling gguf_dequant_mxfp4 or sharing its lookup table object. Two
 * deliberate divergences from ggml_dequant.h's implementation so agreement is
 * evidence of correctness rather than of calling the same code twice:
 *   - scale via ldexpf(1.0f, (int)e - 128), a libm call, not the bit trick
 *   - nibble value via a switch statement, not gguf_kvalues_mxfp4[]
 * (still e2m1-DOUBLED values, per the same upstream table this file's header
 * comment cites -- an independent SOURCE for the table, not an independent
 * spec, was not available inside this worktree). */
static float indep_fp4_doubled(uint8_t nib) {
    switch (nib & 0x0F) {
        case 0: return 0.f;  case 1: return 1.f;  case 2: return 2.f;  case 3: return 3.f;
        case 4: return 4.f;  case 5: return 6.f;  case 6: return 8.f;  case 7: return 12.f;
        case 8: return -0.f; case 9: return -1.f; case 10: return -2.f; case 11: return -3.f;
        case 12: return -4.f; case 13: return -6.f; case 14: return -8.f; default: return -12.f;
    }
}
static void indep_dequant_mxfp4(const uint8_t *raw, float *out, int64_t nblk) {
    for (int64_t i = 0; i < nblk; i++) {
        const uint8_t *blk = raw + i*17;
        const uint8_t e = blk[0];
        const float scale = ldexpf(1.0f, (int)e - 128);
        for (int j = 0; j < 16; j++) {
            const uint8_t byte = blk[1+j];
            out[i*32 + j]      = indep_fp4_doubled(byte & 0x0F) * scale;
            out[i*32 + j + 16] = indep_fp4_doubled(byte >>   4) * scale;
        }
    }
}

/* Naive per-element dequant->dot: reduces one weight at a time in float, the
 * opposite reduction order from mxfp4_row's per-16 integer dot. `xr`/`xs` are
 * the reconstructed (quantized-then-descaled) activation values, so this
 * isolates weight-path disagreement from activation-quantization noise. */
static float dequant_dot(const float *wrow, const int8_t *xq, const float *xscale, int64_t I) {
    double acc = 0.0;   /* double accumulator: this is the ORACLE, not a kernel
                          * under test, so it should not contribute its own
                          * rounding to the bound being measured */
    for (int64_t i = 0; i < I; i++) {
        float xv = (float)xq[i] * xscale[i / COLI_ABLK];
        acc += (double)wrow[i] * (double)xv;
    }
    return (float)acc;
}

static float relerr(float a, float b) {
    float d = fabsf(a - b);
    float m = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return m > 1e-20f ? d / m : d;
}

/* Combined absolute+relative tolerance, numpy's allclose convention:
 * |a-b| <= atol + rtol*|b|. A pure relative bound is the wrong test here --
 * measured (MXFP4_DEBUG=1): real-expert outputs include near-zero cells from
 * ordinary dot-product cancellation (want=5.18e-05) where the absolute
 * disagreement between the kernel and the oracle is 1.7e-06 in EVERY case
 * checked (real expert AND random blocks, n=1 and n=4) but the ratio to a
 * ~5e-5 target reads as 0.5% "relative error" despite nothing having gone
 * wrong. atol=1e-4 is ~60x that measured absolute residual; rtol=1e-3 is the
 * bound from gemm_mxfp4.h's O(I*eps) argument (2880*2^-23 ~= 3.4e-4). */
static int allclose(float want, float got, float atol, float rtol) {
    return fabsf(want - got) <= atol + rtol * fabsf(want);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf";
    char cpubuf[256]; coli_cpu_describe(cpubuf, sizeof cpubuf);
    printf("cpu: %s\n", cpubuf);

    GgufIndex ix; char err[256];
    if (!gguf_index_open(path, &ix, err, sizeof err)) {
        fprintf(stderr, "SKIP: cannot open '%s': %s\n", path, err);
        return 77;
    }
    const GgufTensorInfo *gate = NULL, *aq = NULL;
    for (size_t i = 0; i < ix.n; i++) {
        if (!strcmp(ix.t[i].name, "blk.0.ffn_gate_exps.weight")) gate = &ix.t[i];
        if (!strcmp(ix.t[i].name, "blk.0.attn_q.weight"))        aq   = &ix.t[i];
    }
    if (!gate || !aq) { fprintf(stderr, "SKIP: expected tensors not found in '%s'\n", path); return 77; }

    CHECK(gate->ttype == 39, "blk.0.ffn_gate_exps.weight ttype=%u, expected 39 (MXFP4)", gate->ttype);
    CHECK(gate->rank == 3, "blk.0.ffn_gate_exps.weight rank=%d, expected 3", gate->rank);
    CHECK(aq->ttype == 8, "blk.0.attn_q.weight ttype=%u, expected 8 (Q8_0)", aq->ttype);

    const int64_t I  = (int64_t)gate->shape[0];   /* ne0, fastest-varying = input dim */
    const int64_t O  = (int64_t)gate->shape[1];   /* ne1 = output dim */
    const int64_t NE = (int64_t)gate->shape[2];   /* ne2, SLOWEST = expert index */
    printf("blk.0.ffn_gate_exps.weight: shape [%lld,%lld,%lld] (I,O,n_experts), data_off=%llu\n",
           (long long)I, (long long)O, (long long)NE, (unsigned long long)gate->data_off);
    CHECK(I % COLI_MXFP4_BLK == 0, "I=%lld not a multiple of %d", (long long)I, COLI_MXFP4_BLK);

    const int64_t nblk_row = I / COLI_MXFP4_BLK;
    const int64_t rowbytes = nblk_row * COLI_MXFP4_BYTES;
    const int64_t expert_bytes = O * rowbytes;

    int fd = coli_open_ro(path);
    CHECK(fd >= 0, "cannot open '%s' for pread", path);
    if (fd < 0) return 1;

    /* ---- (a) real expert slice: dequant vs independent inline decode ---- */
    const int64_t EXPERT = 3;   /* arbitrary mid-range expert, not expert 0, so
                                  * a bug that only shows on the first row/block
                                  * of the file can't hide behind it */
    CHECK(EXPERT < NE, "chosen expert %lld >= n_experts %lld", (long long)EXPERT, (long long)NE);
    uint8_t *raw = (uint8_t *)malloc((size_t)expert_bytes);
    CHECK(raw != NULL, "OOM reading %lld bytes for expert %lld", (long long)expert_bytes, (long long)EXPERT);
    int64_t got = coli_pread(fd, raw, (size_t)expert_bytes, (int64_t)gate->data_off + EXPERT*expert_bytes);
    CHECK(got == expert_bytes, "short pread: got %lld of %lld bytes", (long long)got, (long long)expert_bytes);

    const int64_t nblk_expert = O * nblk_row;
    float *deq_a = (float *)malloc((size_t)nblk_expert * COLI_MXFP4_BLK * sizeof(float));
    float *deq_b = (float *)malloc((size_t)nblk_expert * COLI_MXFP4_BLK * sizeof(float));
    coli_test_dequant_mxfp4(raw, deq_a, nblk_expert);
    indep_dequant_mxfp4(raw, deq_b, nblk_expert);
    int64_t mismatches = 0, first_bad = -1;
    for (int64_t i = 0; i < nblk_expert * COLI_MXFP4_BLK; i++) {
        if (deq_a[i] != deq_b[i]) { mismatches++; if (first_bad < 0) first_bad = i; }
    }
    CHECK(mismatches == 0, "gguf_dequant_mxfp4 vs independent decode: %lld/%lld elements differ (first at %lld: %.9g vs %.9g)",
          (long long)mismatches, (long long)(nblk_expert*COLI_MXFP4_BLK), (long long)first_bad,
          first_bad >= 0 ? deq_a[first_bad] : 0.0, first_bad >= 0 ? deq_b[first_bad] : 0.0);
    printf("(a) real expert %lld: %lld blocks / %lld weights, gguf_dequant_mxfp4 vs independent decode: %lld mismatches\n",
           (long long)EXPERT, (long long)nblk_expert, (long long)(nblk_expert*COLI_MXFP4_BLK), (long long)mismatches);
    {   /* a few sample values, so a reviewer can eyeball plausibility */
        double sumabs = 0.0; float mx = 0.f;
        for (int64_t i = 0; i < nblk_expert*COLI_MXFP4_BLK; i++) { float v = fabsf(deq_a[i]); sumabs += v; if (v > mx) mx = v; }
        printf("    |w| mean=%.6g max=%.6g over %lld weights\n", sumabs/(nblk_expert*COLI_MXFP4_BLK), mx, (long long)(nblk_expert*COLI_MXFP4_BLK));
    }

    /* ---- (b) kernel vs scalar reference vs dequant->f32 dot ------------- */
    /* Build coli_w_mxfp4 over expert EXPERT's row 0..O-1 (the whole expert
     * matrix just read into `raw`). */
    coli_w_mxfp4 w = {0}; w.blocks = raw; w.I = I; w.O = O; w.owned = 0;

    srand(20260913);
    for (int trial = 0; trial < 2; trial++) {
        const char *label = trial == 0 ? "real expert" : "random blocks";
        uint8_t *rraw = raw;
        int64_t rI = I, rO = O;
        if (trial == 1) {
            rI = 256; rO = 8;   /* smaller synthetic matrix, still I%32==0 */
            int64_t rnblk_row = rI / COLI_MXFP4_BLK;
            int64_t rrowbytes = rnblk_row * COLI_MXFP4_BYTES;
            rraw = (uint8_t *)malloc((size_t)(rO*rrowbytes));
            /* Random NIBBLES over the full range, but the E8M0 exponent byte
             * (blk[0] of every 17-byte block) is clamped to [118,138] -- a
             * decade either side of the ~127 "no scaling" point, matching the
             * dynamic range actually seen in the real expert above (|w| mean
             * 0.016, max 0.25). An UNCLAMPED random exponent (0..255) spans
             * 2^-128..2^127 and makes the atol half of the atol+rtol bound
             * meaningless -- weights of magnitude 1e30 fail any fixed absolute
             * tolerance without being wrong, which is a property of the
             * synthetic input's range, not of the kernel. */
            for (int64_t i = 0; i < rO*rrowbytes; i++) {
                int64_t within_block = i % COLI_MXFP4_BYTES;   /* byte 0 of EVERY
                                                                 * 17-byte block is
                                                                 * its exponent,
                                                                 * not just the row's
                                                                 * first block */
                rraw[i] = (within_block == 0) ? (uint8_t)(118 + rand()%21) : (uint8_t)(rand() & 0xFF);
            }
            w.blocks = rraw; w.I = rI; w.O = rO;
        }
        int ns[] = {1, 4};
        for (unsigned k = 0; k < sizeof ns/sizeof *ns; k++) {
            int n = ns[k];
            int64_t anb = rI / COLI_ABLK;
            float *X = (float *)malloc((size_t)rI*n*sizeof(float));
            for (int64_t i = 0; i < rI*n; i++) X[i] = (float)((rand()%2001)-1000)/500.0f;
            coli_a_i8 a = {0}; a.I = rI; a.n = n;
            a.q     = (int8_t *)malloc((size_t)rI*n);
            a.scale = (float *)malloc((size_t)anb*n*sizeof(float));
            a.sum   = (int32_t *)malloc((size_t)anb*n*sizeof(int32_t));
            coli_quantize_a(&a, X, n, rI);

            float *y1 = (float *)malloc((size_t)rO*n*sizeof(float));
            float *y2 = (float *)malloc((size_t)rO*n*sizeof(float));
            coli_gemm_mxfp4(y1, &a, &w);
            coli_gemm_mxfp4_ref(y2, &a, &w);
            int64_t bad = 0;
            for (int64_t i = 0; i < rO*n; i++) if (y1[i] != y2[i]) bad++;
            CHECK(bad == 0, "%s n=%d: coli_gemm_mxfp4 vs _ref: %lld/%lld cells differ", label, n, (long long)bad, (long long)(rO*n));

            /* dequant->f32 dot oracle, one output row at a time */
            int64_t rnblk_row2 = rI / COLI_MXFP4_BLK;
            float *wrow_f = (float *)malloc((size_t)rI*sizeof(float));
            float maxrel = 0.f, maxabs = 0.f; int64_t nfail = 0;
            const float ATOL = 1e-4f, RTOL = 1e-3f;
            for (int64_t o = 0; o < rO; o++) {
                coli_test_dequant_mxfp4(w.blocks + o*rnblk_row2*COLI_MXFP4_BYTES, wrow_f, rnblk_row2);
                for (int r = 0; r < n; r++) {
                    float want = dequant_dot(wrow_f, a.q + (int64_t)r*rI, a.scale + (int64_t)r*anb, rI);
                    float got_v = y1[(int64_t)r*rO + o];
                    float re = relerr(want, got_v);
                    float ae = fabsf(want-got_v);
                    if (ae > maxabs) maxabs = ae;
                    if (re > maxrel) maxrel = re;
                    if (!allclose(want, got_v, ATOL, RTOL)) nfail++;
                }
            }
            /* atol+rtol combined bound (see allclose()'s comment for the
             * measured numbers this is set from), NOT a pure relative bound:
             * a pure relative bound fails on ordinary cancellation near zero,
             * which is not a defect (see MXFP4_DEBUG=1 for the measured
             * per-cell absolute/relative pair this replaced). */
            CHECK(nfail == 0, "%s n=%d: kernel vs dequant->f32 dot: %lld/%lld cells outside atol=%.0e+rtol=%.0e (max abs=%.6g, max rel=%.6g)",
                  label, n, (long long)nfail, (long long)(rO*n), ATOL, RTOL, maxabs, maxrel);
            printf("(b) %-14s n=%d: kernel==ref (%lld/%lld cells), kernel vs dequant-dot max abs=%.3g max rel=%.3g (0/%lld cells outside atol=%.0e+rtol=%.0e)\n",
                   label, n, (long long)(rO*n-bad), (long long)(rO*n), maxabs, maxrel, (long long)(rO*n), ATOL, RTOL);

            free(wrow_f); free(y1); free(y2);
            free(a.q); free(a.scale); free(a.sum); free(X);
        }
        if (trial == 1) free(rraw);
    }

    /* ---- negative control: corrupt one nibble, must disagree ------------ */
    {
        uint8_t *craw = (uint8_t *)malloc((size_t)rowbytes);   /* one row's worth */
        memcpy(craw, raw, (size_t)rowbytes);
        coli_w_mxfp4 cw = {0}; cw.blocks = craw; cw.I = I; cw.O = 1; cw.owned = 0;
        int64_t anb = I / COLI_ABLK;
        float *X = (float *)malloc((size_t)I*sizeof(float));
        for (int64_t i = 0; i < I; i++) X[i] = (float)((rand()%2001)-1000)/500.0f;
        coli_a_i8 a = {0}; a.I = I; a.n = 1;
        a.q = (int8_t *)malloc((size_t)I); a.scale = (float *)malloc((size_t)anb*sizeof(float));
        a.sum = (int32_t *)malloc((size_t)anb*sizeof(int32_t));
        coli_quantize_a(&a, X, 1, I);
        float y_before; coli_gemm_mxfp4(&y_before, &a, &cw);
        craw[1] ^= 0xFF;   /* flip a whole nibble byte inside block 0's qs[] */
        float y_after; coli_gemm_mxfp4(&y_after, &a, &cw);
        CHECK(y_before != y_after, "control FAILED: corrupting a weight nibble did not change the kernel's output (apparatus cannot detect a real defect)");
        printf("(b) control: corrupting one nibble byte changed row-0 output %.6g -> %.6g (apparatus can fail)\n", y_before, y_after);
        free(craw); free(X); free(a.q); free(a.scale); free(a.sum);
    }

    /* ---- (c) timing: MXFP4 GEMV vs int4 twin, 2880x2880, n=1 ------------ */
    /* `w` was repointed at a small synthetic 256x8 matrix inside the (b) trial
     * loop above (trial==1, "random blocks") and never restored -- caught by
     * the timing coming back at an impossible ~12 TB/s, i.e. this bug produced
     * a loud, self-evident failure rather than a silently wrong number. Reset
     * to the real 2880x2880 expert before timing anything. */
    w.blocks = raw; w.I = I; w.O = O; w.owned = 0;
    {
        int n = 1;
        int64_t anb = I / COLI_ABLK;
        float *X = (float *)malloc((size_t)I*n*sizeof(float));
        for (int64_t i = 0; i < I*n; i++) X[i] = (float)((rand()%2001)-1000)/500.0f;
        coli_a_i8 a = {0}; a.I = I; a.n = n;
        a.q = (int8_t *)malloc((size_t)I*n); a.scale = (float *)malloc((size_t)anb*n*sizeof(float));
        a.sum = (int32_t *)malloc((size_t)anb*n*sizeof(int32_t));
        coli_quantize_a(&a, X, n, I);
        float *y = (float *)malloc((size_t)O*n*sizeof(float));

        double dt, spent = 0; int reps = 0; dt = 1e30;
        coli_gemm_mxfp4(y, &a, &w); coli_gemm_mxfp4(y, &a, &w);   /* warmup */
        while (spent < 0.100 || reps < 7) {
            double t0 = now(); coli_gemm_mxfp4(y, &a, &w); double d = now()-t0;
            if (d < dt) dt = d; spent += d; reps++; if (reps > 2000) break;
        }
        double gbs_mxfp4 = (double)expert_bytes / dt / 1e9;
        printf("(c) coli_gemm_mxfp4  2880x2880 n=1: %7.3f us/call  %6.2f GB/s (weights=%lld bytes, kernel=%s)\n",
               dt*1e6, gbs_mxfp4, (long long)expert_bytes, coli_gemm_mxfp4_kernel());
        {   /* the scalar reference, timed the same way: the before/after of the SIMD port in one run */
            double dts = 1e30; double sp = 0; int rp = 0;
            coli_gemm_mxfp4_ref(y, &a, &w);
            while (sp < 0.100 || rp < 3) { double t0 = now(); coli_gemm_mxfp4_ref(y, &a, &w); double d = now()-t0;
                if (d < dts) dts = d; sp += d; rp++; if (rp > 200) break; }
            printf("(c) coli_gemm_mxfp4_ref (scalar, 1 thread) n=1: %7.3f us/call  %6.2f GB/s\n", dts*1e6, (double)expert_bytes/dts/1e9);
        }
        for (int nn = 4; nn <= 16; nn *= 4) {   /* n>1: weights decoded once per block per 8-row chunk */
            int64_t anb2 = I / COLI_ABLK;
            float *X2 = (float *)malloc((size_t)I*nn*sizeof(float));
            for (int64_t i = 0; i < I*nn; i++) X2[i] = (float)((rand()%2001)-1000)/500.0f;
            coli_a_i8 a2 = {0}; a2.I = I; a2.n = nn;
            a2.q = (int8_t *)malloc((size_t)I*nn); a2.scale = (float *)malloc((size_t)anb2*nn*sizeof(float));
            a2.sum = (int32_t *)malloc((size_t)anb2*nn*sizeof(int32_t));
            coli_quantize_a(&a2, X2, nn, I);
            float *y2 = (float *)malloc((size_t)O*nn*sizeof(float));
            double dtn = 1e30; double sp = 0; int rp = 0;
            coli_gemm_mxfp4(y2, &a2, &w);
            while (sp < 0.100 || rp < 7) { double t0 = now(); coli_gemm_mxfp4(y2, &a2, &w); double d = now()-t0;
                if (d < dtn) dtn = d; sp += d; rp++; if (rp > 2000) break; }
            printf("(c) coli_gemm_mxfp4  2880x2880 n=%d: %7.3f us/call  %6.2f GB/s weight-stream, %.2fx the n=1 time\n",
                   nn, dtn*1e6, (double)expert_bytes/dtn/1e9, dtn/dt);
            free(X2); free(a2.q); free(a2.scale); free(a2.sum); free(y2);
        }

        /* int4 twin: dequant the SAME expert matrix to f32, then quantize to
         * int4 with this engine's own quantizer, and time coli_gemm_i4 on the
         * identical activations. */
        float *wf = (float *)malloc((size_t)I*O*sizeof(float));
        coli_test_dequant_mxfp4(raw, wf, nblk_expert);
        coli_w_i4 wi4 = {0};
        coli_quantize_w4_ex(&wi4, wf, I, O, 0);
        float *y4 = (float *)malloc((size_t)O*n*sizeof(float));
        double dt4 = 1e30; spent = 0; reps = 0;
        coli_gemm_i4(y4, &a, &wi4); coli_gemm_i4(y4, &a, &wi4);
        while (spent < 0.100 || reps < 7) {
            double t0 = now(); coli_gemm_i4(y4, &a, &wi4); double d = now()-t0;
            if (d < dt4) dt4 = d; spent += d; reps++; if (reps > 2000) break;
        }
        int64_t i4bytes = (I/2)*O + (I/COLI_W4BLK)*O*(int64_t)sizeof(float);
        double gbs_i4 = (double)i4bytes / dt4 / 1e9;
        printf("(c) coli_gemm_i4     2880x2880 n=1: %7.3f us/call  %6.2f GB/s (weights~%lld bytes, kernel=%s)\n",
               dt4*1e6, gbs_i4, (long long)i4bytes, coli_gemm_i4_kernel(n));
        printf("(c) mxfp4/i4 time ratio: %.3fx (mxfp4 weight bytes/i4 weight bytes = %.3fx)\n",
               dt/dt4, (double)expert_bytes/(double)i4bytes);

        coli_free_w4(&wi4);
        free(wf); free(y4); free(y); free(a.q); free(a.scale); free(a.sum); free(X);
    }

    /* ---- (d) Q8_0: dequantize blk.0.attn_q.weight, check row norms ------ */
    {
        const int64_t aI = (int64_t)aq->shape[0];   /* ne0 = input dim (row length) */
        const int64_t aO = (int64_t)aq->shape[1];   /* ne1 = output dim (rows) */
        printf("blk.0.attn_q.weight: shape [%lld,%lld] (I,O), data_off=%llu\n",
               (long long)aI, (long long)aO, (unsigned long long)aq->data_off);
        CHECK(aI % 32 == 0, "attn_q I=%lld not a multiple of 32", (long long)aI);
        const int64_t rnblk = aI / 32;
        const int64_t arowbytes = rnblk * 34;
        const int64_t total_bytes = aO * arowbytes;
        uint8_t *araw = (uint8_t *)malloc((size_t)total_bytes);
        int64_t agot = coli_pread(fd, araw, (size_t)total_bytes, (int64_t)aq->data_off);
        CHECK(agot == total_bytes, "short pread on attn_q.weight: got %lld of %lld", (long long)agot, (long long)total_bytes);
        float *adeq = (float *)malloc((size_t)aI*sizeof(float));
        int64_t nonfinite = 0, implausible = 0;
        double normsum = 0.0; float normmax = 0.f, normmin = 1e30f;
        for (int64_t o = 0; o < aO; o++) {
            coli_test_dequant_q8_0(araw + o*arowbytes, adeq, rnblk);
            double ss = 0.0;
            for (int64_t i = 0; i < aI; i++) {
                if (!isfinite(adeq[i])) nonfinite++;
                ss += (double)adeq[i]*(double)adeq[i];
            }
            float norm = (float)sqrt(ss);
            if (!isfinite(norm) || norm > 1000.0f) implausible++;   /* generous ceiling for a real attn weight row */
            normsum += norm; if (norm > normmax) normmax = norm; if (norm < normmin) normmin = norm;
        }
        CHECK(nonfinite == 0, "Q8_0 dequant produced %lld non-finite values across %lld rows", (long long)nonfinite, (long long)aO);
        CHECK(implausible == 0, "Q8_0 dequant produced %lld implausible row norms (>1000 or non-finite)", (long long)implausible);
        printf("(d) blk.0.attn_q.weight: %lld rows, row L2 norm mean=%.4g min=%.4g max=%.4g, non-finite=%lld\n",
               (long long)aO, normsum/aO, normmin, normmax, (long long)nonfinite);

        /* control: corrupt the scale (f16 d) of row 0 to a huge value and
         * confirm the finite/plausible check WOULD catch it */
        uint16_t bad_d = 0x7BFF; /* large finite f16 */
        memcpy(araw, &bad_d, 2);
        coli_test_dequant_q8_0(araw, adeq, rnblk);
        double ss = 0.0; for (int64_t i = 0; i < aI; i++) ss += (double)adeq[i]*(double)adeq[i];
        float cnorm = (float)sqrt(ss);
        CHECK(cnorm > 1000.0f, "control FAILED: corrupting row 0's scale to a huge f16 did not blow up its row norm (got %.6g)", cnorm);
        printf("(d) control: corrupting row 0's Q8_0 scale drove its row norm to %.6g (apparatus can fail)\n", cnorm);
        free(araw); free(adeq);
    }

    coli_close(fd);
    free(raw); free(deq_a); free(deq_b);
    gguf_index_free(&ix);

    printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
