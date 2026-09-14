#include "../src/platform.h"
/* test_gemm_q6k -- native Q6_K GEMV on REAL expert bytes (2026-09-13).
 *
 * Why real data instead of a hand encoder (test_gemm_q4k's approach): Q6_K
 * expert tensors exist on disk in both target models (Qwen3-30B-A3B: 24
 * ffn_down_exps, Qwen3-235B-A22B: 46), and the reference dequant already lives
 * in c/ggml_dequant.h. So the oracle is the file plus the reference, not a
 * second reading of the format written by the same author as the kernel.
 *
 * Per model (skipped if absent), one expert slice of blk.0.ffn_down_exps:
 *   (1) decode: ds[i/16]*q[i] from coli_q6k_decode vs gguf_dequant_q6_K, every
 *       weight, EXACT (same operands, same order). Control: flip 4 bits of byte 0
 *       in a copy of block 0 -> the same comparison must now find mismatches.
 *   (2) dispatched kernel vs scalar reference, max relative diff <= 1e-5.
 *   (3) kernel vs dequant->f32 weights dotted with the DEQUANTIZED activations
 *       (scale*q, i.e. exactly what the kernel sees), bound I*FLT_EPSILON*S per
 *       cell where S = sum_i |a_i*w_i|: the most rounding I float additions can
 *       add. This arm isolates kernel arithmetic, so it FAILS the test.
 *   (3b) the same against the ORIGINAL float activations with test_gemm_q4k's
 *       rtol 10% + atol heuristic -- REPORTED, not failed. Its error is int8
 *       activation rounding, which no GEMV kernel controls. Measured 2026-09-13
 *       on the 235B blk.0 down expert 5: 1 of 12,288 cells over that heuristic
 *       (abs 1.02e-2 vs atol 5.4e-3) while (1) and (2) were exact -- the reason
 *       (3) was added rather than the heuristic loosened until it passed.
 * Negative control for (2): the _broken build (-DCOLI_BREAK_Q6K) uses the next
 * sub-block's scale in the AVX2 kernel only; that binary must exit nonzero.
 * Exit 77 = SKIP (neither model on disk). */
#include "../src/loader.h"
#include "../src/gemm_q4k.h"
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static int run_case(const char *path, const char *nm, int64_t e_pick, int *skipped) {
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { printf("SKIP %s: %s\n", path, err); *skipped = 1; return 0; }
    if (!coli_gguf_has(g, nm)) { printf("SKIP %s: no %s\n", path, nm); coli_gguf_close(g); *skipped = 1; return 0; }
    int64_t I = coli_gguf_shape(g, nm, 0), O = coli_gguf_shape(g, nm, 1), NE = coli_gguf_shape(g, nm, 2);
    if (e_pick < 0 || e_pick >= NE) e_pick = NE - 1;
    coli_gguf_slice sl;
    if (!coli_gguf_tensor_slice(g, nm, e_pick, NE, &sl)) { printf("FAIL: cannot resolve slice\n"); coli_gguf_close(g); return 1; }
    if (sl.ttype != 14) { printf("SKIP %s: %s is ggml type %d, not Q6_K\n", path, nm, sl.ttype); coli_gguf_close(g); *skipped = 1; return 0; }
    int64_t nsb = I / 256, nblk = O * nsb;
    if (I % 256 || sl.nbytes != nblk * COLI_Q6K_BLOCK_BYTES) {
        printf("FAIL: I=%lld O=%lld slice %lld bytes, expected %lld\n", (long long)I, (long long)O,
               (long long)sl.nbytes, (long long)(nblk*COLI_Q6K_BLOCK_BYTES));
        coli_gguf_close(g); return 1;
    }
    uint8_t *buf = (uint8_t*)malloc((size_t)sl.nbytes);
    int rd = coli_gguf_slice_pread(&sl, buf);
    coli_gguf_close(g);
    if (!rd) { printf("FAIL: short read\n"); free(buf); return 1; }
    printf("%s\n  %s expert %lld of %lld: I=%lld O=%lld, %lld Q6_K blocks, kernel=%s\n",
           path, nm, (long long)e_pick, (long long)NE, (long long)I, (long long)O, (long long)nblk,
           coli_gemm_q6k_kernel(1));
    int fail = 0;

    /* (1) decode vs reference dequant, exact */
    float *truth = (float*)malloc((size_t)(O*I)*sizeof(float));
    coli_q6k_dequant_ref(buf, truth, nblk);
    int64_t mism = 0;
    for (int64_t b = 0; b < nblk; b++) {
        int8_t q[256]; float ds[16];
        coli_q6k_decode(buf + b*COLI_Q6K_BLOCK_BYTES, q, ds);
        for (int i = 0; i < 256; i++) if (ds[i/16]*(float)q[i] != truth[b*256+i]) mism++;
    }
    uint8_t blk0[COLI_Q6K_BLOCK_BYTES]; memcpy(blk0, buf, sizeof blk0); blk0[0] ^= 0x0F;
    int8_t qc[256]; float dc[16]; coli_q6k_decode(blk0, qc, dc);
    int64_t cmism = 0; for (int i = 0; i < 256; i++) if (dc[i/16]*(float)qc[i] != truth[i]) cmism++;
    printf("  (1) decode vs gguf_dequant_q6_K: %lld mismatches over %lld weights; control (byte 0 ^ 0x0F): %lld mismatches\n",
           (long long)mism, (long long)(nblk*256), (long long)cmism);
    if (mism != 0) { printf("FAIL: decode disagrees with the reference dequant\n"); fail = 1; }
    if (cmism == 0) { printf("FAIL: decode control did not change the output -- the comparison proves nothing\n"); fail = 1; }

    /* (2)(3) */
    int NM = 3; int64_t nb = I / COLI_ABLK;
    coli_a_i8 a; memset(&a, 0, sizeof a); a.I = I; a.n = NM;
    a.q = (int8_t*)malloc((size_t)I*NM);
    a.scale = (float*)malloc((size_t)nb*NM*sizeof(float));
    a.sum = (int32_t*)malloc((size_t)nb*NM*sizeof(int32_t));
    float *X = (float*)malloc((size_t)I*NM*sizeof(float));
    srand(77);
    for (int64_t i = 0; i < I*NM; i++) X[i] = (float)((rand()%2001)-1000)/450.0f;
    coli_quantize_a(&a, X, NM, I);
    coli_w_q4k w; w.blocks = buf; w.owns = 0; w.I = I; w.O = O;
    float *yk = (float*)malloc((size_t)O*NM*sizeof(float));
    float *yr = (float*)malloc((size_t)O*NM*sizeof(float));
    float *yt = (float*)malloc((size_t)O*NM*sizeof(float));
    coli_gemm_q6k(yk, &a, &w);
    coli_gemm_q6k_ref(yr, &a, &w);
    for (int r = 0; r < NM; r++)
        for (int64_t o = 0; o < O; o++) {
            const float *xr = X + (int64_t)r*I, *wr = truth + o*I;
            float acc = 0.f; for (int64_t i = 0; i < I; i++) acc += xr[i]*wr[i];
            yt[(int64_t)r*O+o] = acc;
        }
    double sum_abs_w = 0; for (int64_t i = 0; i < O*I; i++) sum_abs_w += fabs(truth[i]);
    double atol = (sum_abs_w/(double)(O*I)) * (1.0/127.0) * sqrt((double)I) * 4.0;
    double max_rel_kr = 0, max_abs_kt = 0, worst_ratio3 = 0; int64_t viol = 0, viol3 = 0;
    for (int r = 0; r < NM; r++) {
        const int8_t *qa = a.q + (int64_t)r*I; const float *as = a.scale + (int64_t)r*nb;
        for (int64_t o = 0; o < O; o++) {
            const float *wr = truth + o*I;
            double acc = 0, S = 0;
            for (int64_t i = 0; i < I; i++) { double t = (double)as[i/COLI_ABLK]*(double)qa[i]*(double)wr[i]; acc += t; S += fabs(t); }
            int64_t idx = (int64_t)r*O + o;
            double d3 = fabs((double)yk[idx] - acc), bound3 = (double)I * FLT_EPSILON * S;
            if (bound3 > 0 && d3/bound3 > worst_ratio3) worst_ratio3 = d3/bound3;
            if (d3 > bound3) viol3++;
        }
    }
    for (int64_t idx = 0; idx < O*(int64_t)NM; idx++) {
        double denom = fabs(yt[idx]) > 1e-6 ? fabs(yt[idx]) : 1e-6;
        double rkr = fabs(yk[idx]-yr[idx]) / denom, akt = fabs(yk[idx]-yt[idx]);
        if (rkr > max_rel_kr) max_rel_kr = rkr;
        if (akt > max_abs_kt) max_abs_kt = akt;
        if (akt > 0.10*fabs(yt[idx]) + atol) viol++;
    }
    printf("  (2) kernel vs ref max relative diff %.3e\n", max_rel_kr);
    printf("  (3) kernel vs dequantized weights . dequantized activations: cells over I*FLT_EPSILON*S: %lld/%lld (worst diff/bound %.3f)\n",
           (long long)viol3, (long long)(O*NM), worst_ratio3);
    printf("  (3b) REPORT ONLY, activation rounding: vs float activations max abs diff %.3e, atol %.3e, cells over rtol 10%%+atol: %lld/%lld\n",
           max_abs_kt, atol, (long long)viol, (long long)(O*NM));
    if (max_rel_kr > 1e-5) { printf("FAIL: kernel disagrees with its scalar reference\n"); fail = 1; }
    if (viol3 > 0) { printf("FAIL: kernel disagrees with the dequantized-weight x dequantized-activation dot\n"); fail = 1; }

    if (!fail) {
        coli_a_i8 a1; memset(&a1, 0, sizeof a1); a1.I = I; a1.n = 1;
        a1.q = a.q; a1.scale = a.scale; a1.sum = a.sum;
        double best = 1e9;
        for (int rp = 0; rp < 15; rp++) { double t0 = now(); coli_gemm_q6k(yk, &a1, &w); double dt = now()-t0; if (dt < best) best = dt; }
        printf("  timing n=1: best of 15 = %.4f ms (%.2f GB/s over %lld weight bytes)\n",
               best*1e3, (double)sl.nbytes/best/1e9, (long long)sl.nbytes);
    }
    free(a.q); free(a.scale); free(a.sum); free(X); free(yk); free(yr); free(yt); free(truth); free(buf);
    return fail;
}

int main(int argc, char **argv) {
    const char *m30  = argc > 1 ? argv[1] : "/home/monkey/Documents/Ai_Models/qwen3moe/Qwen3-30B-A3B-Q4_K_M.gguf";
    const char *m235 = argc > 2 ? argv[2] : "/home/monkey/Documents/Ai_Models/qwen3moe/Qwen3-235B-A22B-Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf";
    int fail = 0, s30 = 0, s235 = 0;
    fail |= run_case(m30,  "blk.0.ffn_down_exps.weight", -1, &s30);
    fail |= run_case(m235, "blk.0.ffn_down_exps.weight",  5, &s235);
    if (s30 && s235) { printf("SKIP: no Q6_K expert model on disk\n"); return 77; }
    if (fail) { printf("FAIL (this is EXPECTED and correct under -DCOLI_BREAK_Q6K)\n"); return 1; }
    printf("PASS\n");
    return 0;
}
