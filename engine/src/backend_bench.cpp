/* backend_bench.cpp -- startup calibration of the backend choice (2026-09-15).
 *
 * WHY. The planner's "vulkan before cuda" order was a measured fact of 09-14
 * (CUDA had CPU attention then). With attention and async fills on CUDA (09-15)
 * the two are level end to end (2.1-2.2 tok/s both), and the next kernel change
 * on either side can flip the order. A fixed rule would carry a stale
 * measurement forward; a 20 ms benchmark at startup carries today's.
 *
 * WHAT. Open the named backend, upload one 2880x2880 int4 matrix (the gpt-oss
 * expert/o_proj shape), time gemm4 n=1 best-of-N after a warm-up, close. The
 * number is a proxy for the memory-bound GEMV that dominates every backend's
 * share of a token; it is NOT the whole-token speed (attention, fills and the
 * CPU experts are not in it), so the caller reports it as "gemv 2880^2 n=1". */
#include "backend.h"
#include "gemm_i8.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double mono(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

extern "C" double coli_backend_bench_gemv_us(const char *name, int reps, char *err, size_t errcap) {
    coli_backend *be = coli_backend_open(name, err, errcap);
    if (!be) return -1;
    if (!be->has_i4(be->ctx)) { snprintf(err, errcap, "%s: no int4 GEMM", name); coli_backend_close(be); return -1; }
    const int64_t I = 2880, O = 2880;
    unsigned s = 11; auto frand = [&]{ s = s*1664525u+1013904223u; return ((s>>8)&0xffff)/65536.0f-0.5f; };
    float *W = (float*)malloc((size_t)I*O*4), *X = (float*)malloc((size_t)I*4);
    for (int64_t i=0;i<I*O;i++) W[i]=frand(); for (int64_t i=0;i<I;i++) X[i]=frand();
    coli_w_i4 w4; coli_quantize_w4(&w4, W, I, O); free(W);
    int64_t nb = I/COLI_ABLK; coli_a_i8 a; memset(&a,0,sizeof a); a.I=I; a.n=1;
    a.q=(int8_t*)aligned_alloc(64,(size_t)I); a.scale=(float*)aligned_alloc(64,(size_t)nb*4); a.sum=(int32_t*)aligned_alloc(64,(size_t)nb*4);
    coli_quantize_a(&a, X, 1, I); free(X);
    float *Y = (float*)calloc((size_t)O, 4);
    double best = -1;
    int h = be->upload_w4(be->ctx, &w4);
    if (h < 0 || be->gemm4(be->ctx, h, &a, Y) != 0) { snprintf(err, errcap, "%s: upload/gemm4 failed", name); }
    else {
        for (int i = 0; i < 3; i++) be->gemm4(be->ctx, h, &a, Y);            /* warm-up: first-dispatch costs */
        if (reps < 1) reps = 20;
        for (int i = 0; i < reps; i++) { double t0 = mono(); be->gemm4(be->ctx, h, &a, Y); double us = (mono()-t0)*1e6; if (best < 0 || us < best) best = us; }
    }
    free(Y); free(a.q); free(a.scale); free(a.sum);
    coli_backend_close(be);
    return best;
}
