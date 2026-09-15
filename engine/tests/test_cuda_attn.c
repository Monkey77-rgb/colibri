/* test_cuda_attn -- does backend_cuda.cu's KV cache + attn_ex compute
 * attention, through the coli_backend seam (backend.h), the same way
 * tests/test_cuda_gemm.c checks the GEMM entries and tests/test_vk_attn.c
 * checks the Vulkan attention kernel.
 *
 * REFERENCE: a two-pass softmax (score every position, find the max,
 * exponentiate, weight), accumulated in DOUBLE precision, over the resident
 * cache's own K/V -- not the running-max/running-denominator identity the
 * kernel uses. attn_decode's k_attn_decode and attend_online() in model.cpp
 * both use the online form; testing one against the other would only prove
 * they made the same mistake (same reasoning as test_vk_attn.c's header).
 * Attention sinks and the sliding window are folded into the SAME two-pass
 * form here (see attn_ref below) rather than reused from model.cpp, which
 * does not export attend_online() for a test to link against.
 *
 * NOT BIT-EXACT with the kernel, and stated as a relative bound rather than
 * hidden: k_attn_decode reduces the hd-wide dot product across one warp via
 * __shfl_xor_sync (a tree), this reference sums sequentially in double. A
 * CONTROL (the reference perturbed by 1.001) must exceed the bound, or a
 * kernel that silently returned the reference unmodified would pass.
 *
 * TWO FILL PATHS are exercised, because backend_cuda.cu's kv_write and
 * kv_put are different code paths that must independently land in the same
 * device buffers: slot 0 is bulk-loaded through kv_write (the prefill path),
 * slot 1 is built one row at a time through kv_put, applied by attn_ex's
 * pending-ring flush (the decode path) -- see backend_cuda.cu's struct
 * comment on pend_* for why kv_put alone cannot apply a row.
 */
#include "../src/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static unsigned long rs = 987654321UL;
/* Cast to long is load-bearing -- see test_vk_attn.c's identical comment:
 * without it `%20001` on an unsigned intermediate wraps every value below
 * 10000 to ~2^64, and reference and kernel then silently agree on garbage. */
static float frnd(void){ rs = rs*6364136223846793005UL + 1442695040888963407UL;
                         return (float)((long)((rs>>33)%20001) - 10000) / 10000.0f; }

/* Two-pass reference, folding in sink and window exactly as attend_online()
 * does: sink is one extra logit with no value row, NOT scaled; window t0
 * masks positions t < tmax-window+1. hd<=256 kept on the stack, matching the
 * kernel's own ceiling. */
static void attn_ref(float *o, const float *qv, const float *Kb, const float *Vb,
                      int tmax, int hd, float scale, int window, int has_sink, float sink) {
    int t0 = 0;
    if (window > 0) { t0 = tmax - window + 1; if (t0 < 0) t0 = 0; }
    double sc[8192];
    double mx = has_sink ? (double)sink : -1e300;
    for (int t = t0; t <= tmax; t++) {
        double d = 0.0;
        const float *k = Kb + (long)t*hd;
        for (int i = 0; i < hd; i++) d += (double)qv[i]*k[i];
        d *= scale;
        sc[t - t0] = d;
        if (d > mx) mx = d;
    }
    double den = has_sink ? exp((double)sink - mx) : 0.0;
    for (int t = t0; t <= tmax; t++) den += exp(sc[t - t0] - mx);
    for (int i = 0; i < hd; i++) {
        double num = 0.0;
        for (int t = t0; t <= tmax; t++) {
            const float *v = Vb + (long)t*hd;
            num += exp(sc[t - t0] - mx) * (double)v[i];
        }
        o[i] = (float)(num / den);
    }
}

typedef struct { int H, KVH, hd, kv_ctx, slots; } shape;

/* rel-error over a whole [n][H*hd] buffer vs a reference of the same shape */
static double worst_rel(const float *a, const float *b, size_t n) {
    double w = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]) / (fabs((double)b[i]) + 1e-3);
        if (d > w) w = d;
    }
    return w;
}

/* Builds a resident cache for `sh`, fills slot 0 via kv_write and slot 1 via
 * kv_put (one row at a time, flushed by attn_ex), then compares attn_ex
 * against attn_ref for both slots, with/without sinks, with/without window.
 * Returns 1 pass / 0 fail. TOL is the same order as test_vk_attn.c's 1e-4 --
 * both kernels accumulate in float32 against a float64 reference. */
static int run_shape(coli_backend *be, shape sh, const char *label, double TOL) {
    int H=sh.H, KVH=sh.KVH, hd=sh.hd, kv_ctx=sh.kv_ctx, slots=sh.slots;
    int grp = H/KVH;
    printf("-- %s: H=%d KVH=%d hd=%d kv_ctx=%d slots=%d --\n", label, H, KVH, hd, kv_ctx, slots);

    if (be->kv_init(be->ctx, 1, slots, KVH, kv_ctx, hd) != 0) { printf("  FAIL: kv_init\n"); return 0; }

    size_t kvn = (size_t)slots*KVH*kv_ctx*hd;
    float *K = (float*)malloc(kvn*4), *V = (float*)malloc(kvn*4);
    for (size_t i=0;i<kvn;i++) { K[i]=frnd(); V[i]=frnd(); }

    /* slot 0: bulk kv_write, whole context in one call -- the prefill path */
    if (be->kv_write(be->ctx, 0, 0, 0, kv_ctx, K, V) != 0) { printf("  FAIL: kv_write slot0\n"); return 0; }

    /* slot 1: one row at a time via kv_put, flushed by a throwaway attn_ex
     * per position -- the decode path. A tiny 1-row batch at slot 0 is used
     * as the flush vehicle so slot 1's own rows land before this shape's
     * real comparison runs; its output is discarded. */
    { float *dummy_q = (float*)calloc((size_t)H*hd, 4);
      float *dummy_o = (float*)malloc((size_t)H*hd*4);
      int dm[2] = {0, 0};
      for (int pos=0; pos<kv_ctx; pos++) {
          for (int kh=0; kh<KVH; kh++) {
              size_t off = (((size_t)1*KVH + kh)*kv_ctx + pos)*hd;
              if (be->kv_put(be->ctx, 1, kh, pos, 0, K+off) != 0) { printf("  FAIL: kv_put K\n"); return 0; }
              if (be->kv_put(be->ctx, 1, kh, pos, 1, V+off) != 0) { printf("  FAIL: kv_put V\n"); return 0; }
          }
          if (be->attn_ex(be->ctx, 0, dummy_q, dummy_o, dm, 1, H, 1.0f, 0, -1) != 0) {
              printf("  FAIL: attn_ex flush at pos %d\n", pos); return 0; }
      }
      free(dummy_q); free(dummy_o);
    }

    float scale = 1.0f/sqrtf((float)hd);
    int n = 3;
    float *q = (float*)malloc((size_t)n*H*hd*4);
    for (size_t i=0;i<(size_t)n*H*hd;i++) q[i] = frnd();
    int meta[6] = { 0, kv_ctx-1,  1, kv_ctx-1,  1, kv_ctx/2 };  /* rows over both slots, ragged tmax */

    float *sinks = (float*)malloc((size_t)H*4);
    for (int h=0;h<H;h++) sinks[h] = frnd();
    if (be->attn_sinks_upload(be->ctx, sinks, (size_t)H) != 0) { printf("  FAIL: attn_sinks_upload\n"); return 0; }

    float *og = (float*)malloc((size_t)n*H*hd*4);
    float *oc = (float*)malloc((size_t)n*H*hd*4);
    int pass = 1;

    /* four combinations: {no window, window=kv_ctx/3} x {no sink, sink} */
    int windows[2] = { 0, kv_ctx/3 > 0 ? kv_ctx/3 : 1 };
    for (int wi = 0; wi < 2; wi++) {
        for (int si = 0; si < 2; si++) {
            int window = windows[wi];
            int sink_off = si ? 0 : -1;
            if (be->attn_ex(be->ctx, 0, q, og, meta, n, H, scale, window, sink_off) != 0) {
                printf("  FAIL: attn_ex window=%d sink=%d\n", window, si); return 0; }
            for (int r=0;r<n;r++) {
                int slot = meta[r*2], tmax = meta[r*2+1];
                for (int h=0;h<H;h++) {
                    int kvh = h/grp;
                    const float *qv = q + (size_t)r*H*hd + (size_t)h*hd;
                    const float *Kb = K + (((size_t)slot*KVH+kvh)*kv_ctx)*hd;
                    const float *Vb = V + (((size_t)slot*KVH+kvh)*kv_ctx)*hd;
                    attn_ref(oc + (size_t)r*H*hd + (size_t)h*hd, qv, Kb, Vb, tmax, hd, scale,
                             window, si, sinks[h]);
                }
            }
            double rel = worst_rel(og, oc, (size_t)n*H*hd);
            int ok = rel <= TOL;
            printf("  window=%-4d sink=%d : rel=%.3e  %s\n", window, si, rel, ok?"ok":"BAD");
            pass = pass && ok;
        }
    }

    /* POSITIVE CONTROL: perturb the reference (last combination run above,
     * sink=1 window=windows[1]) by 1.001 and require it to exceed TOL --
     * without this, a kernel returning the reference unmodified, or a
     * comparison loop that never executed, would pass silently. */
    { float *oc2 = (float*)malloc((size_t)n*H*hd*4);
      for (size_t i=0;i<(size_t)n*H*hd;i++) oc2[i] = oc[i]*1.001f;
      double crel = worst_rel(og, oc2, (size_t)n*H*hd);
      printf("  control (reference x1.001): rel=%.3e -> %s\n", crel,
             crel > TOL ? "exceeds tolerance, as required" : "DID NOT EXCEED -- test is inert");
      pass = pass && (crel > TOL);
      free(oc2);
    }

    free(K); free(V); free(q); free(sinks); free(og); free(oc);
    return pass;
}

int main(void) {
    char err[256];
    coli_backend *be = coli_backend_cuda_open(err, sizeof err);
    if (!be) { printf("no CUDA device (%s) -- SKIP\n", err); return 0; }
    printf("device: %s   mem=%s/%s\n", be->device_name(be->ctx), be->memdesc(be->ctx), be->memdesc2(be->ctx));
    if (!be->has_attn(be->ctx)) { printf("has_attn()==0 -- SKIP\n"); return 0; }

    const double TOL = 1e-4;   /* set from test_vk_attn.c's measurement on the same class of hardware;
                                 * re-measured below rather than assumed to hold for this kernel too. */
    int pass = 1;

    /* small case */
    pass = run_shape(be, (shape){ .H=8, .KVH=2, .hd=64, .kv_ctx=257, .slots=2 }, "small", TOL) && pass;
    /* hd=128 case (Qwen-class dense) */
    pass = run_shape(be, (shape){ .H=28, .KVH=4, .hd=128, .kv_ctx=513, .slots=2 }, "hd128", TOL) && pass;
    /* gpt-oss shape: H=64, KVH=8, hd=64 */
    pass = run_shape(be, (shape){ .H=64, .KVH=8, .hd=64, .kv_ctx=1025, .slots=2 }, "gpt-oss-shape", TOL) && pass;

    /* ------------------------------------------------------- timing, gpt-oss shape
     * attn_ex at tmax=512 and tmax=4096, decode batch n=1 (the shape that
     * actually matters: prefill amortizes the fixed per-call cost over many
     * rows, decode does not). Best-of-N after a warm-up call; N stated next
     * to the number, per the measurement discipline this doctrine requires. */
    {
        int H=64, KVH=8, hd=64;
        int kv_ctx = 4097;   /* covers both tmax=512 and tmax=4096 in one cache */
        if (be->kv_init(be->ctx, 1, 1, KVH, kv_ctx, hd) != 0) { printf("FAIL: kv_init (timing)\n"); return 1; }
        size_t kvn = (size_t)KVH*kv_ctx*hd;
        float *K = (float*)malloc(kvn*4), *V = (float*)malloc(kvn*4);
        for (size_t i=0;i<kvn;i++) { K[i]=frnd(); V[i]=frnd(); }
        if (be->kv_write(be->ctx, 0, 0, 0, kv_ctx, K, V) != 0) { printf("FAIL: kv_write (timing)\n"); return 1; }

        float scale = 1.0f/sqrtf((float)hd);
        float *q = (float*)malloc((size_t)H*hd*4);
        for (size_t i=0;i<(size_t)H*hd;i++) q[i] = frnd();
        float *o = (float*)malloc((size_t)H*hd*4);

        int N = 200;
        int tmaxes[2] = { 512, 4096 };
        for (int ti = 0; ti < 2; ti++) {
            int meta[2] = { 0, tmaxes[ti] };
            be->attn_ex(be->ctx, 0, q, o, meta, 1, H, scale, 0, -1);   /* warm */
            double best = 1e300;
            for (int rep = 0; rep < N; rep++) {
                double t0 = now();
                be->attn_ex(be->ctx, 0, q, o, meta, 1, H, scale, 0, -1);
                double d = now() - t0;
                if (d < best) best = d;
            }
            printf("TIMING gpt-oss-shape attn_ex n=1 H=%d KVH=%d hd=%d tmax=%d: best-of-%d = %.2f us/call\n",
                   H, KVH, hd, tmaxes[ti], N, best*1e6);
        }
        free(K); free(V); free(q); free(o);
    }

    printf("%s\n", pass ? "PASS (cuda attn matches the two-pass reference, and the control does not)" : "FAIL");
    return pass ? 0 : 1;
}
