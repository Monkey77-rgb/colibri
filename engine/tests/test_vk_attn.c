/* test_vk_attn -- does attn_decode.comp compute attention?
 *
 * The reference here is the TWO-PASS form: score every position, find the max,
 * exponentiate, normalise, weight. That is deliberately NOT the algorithm the
 * kernel uses. attend_online() in model.cpp and the shader both use the running
 * -max/running-denominator form, and testing one against the other would only
 * prove they made the same mistake. The two-pass version is the definition, so
 * it validates the identity as well as the implementation.
 *
 * NOT BIT-EXACT, and the tolerance says so rather than hiding it: the shader
 * splits t across four subgroups and merges their softmax states at the end, so
 * the rescales happen in a different order from any sequential walk.
 *
 * The control multiplies the reference by 1.001 and MUST exceed the tolerance.
 * Without it a kernel that returned the reference buffer unmodified -- or a
 * comparison loop that never executed -- would pass silently. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../src/vk_backend.h"

static unsigned long rs = 12345;
/* The cast to long is load-bearing. Without it `%20001 - 10000` is evaluated in
 * UNSIGNED arithmetic, so every value below 10000 wraps to ~2^64 and the whole
 * input set becomes ~9.2e14. Both the reference and the kernel then consume the
 * same garbage and partially agree, which reads as a kernel bug. */
static float frnd(void){ rs = rs*6364136223846793005ULL + 1442695040888963407ULL;
                         return (float)((long)((rs>>33)%20001) - 10000) / 10000.0f; }

int main(int argc, char **argv) {
    int H = 28, KVH = 4, hd = 128, n = 2, slots = 2;
    int kv_ctx = (argc > 1) ? atoi(argv[1]) : 700;
    if (kv_ctx < 8) kv_ctx = 8;

    char err[256] = {0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { printf("no Vulkan device (%s) -- SKIP\n", err); return 0; }
    if (!coli_vk_has_attn(v)) { printf("shaders/attn_decode.spv absent -- SKIP\n"); coli_vk_free(v); return 0; }
    printf("device: %s   H=%d KVH=%d hd=%d n=%d kv_ctx=%d\n",
           coli_vk_device_name(v), H, KVH, hd, n, kv_ctx);

    size_t qn  = (size_t)n*H*hd;
    size_t kvn = (size_t)slots*KVH*kv_ctx*hd;
    float *q  = (float*)malloc(qn*4), *K = (float*)malloc(kvn*4), *V = (float*)malloc(kvn*4);
    float *og = (float*)malloc(qn*4), *oc = (float*)malloc(qn*4);
    int   *meta = (int*)malloc((size_t)n*2*sizeof(int));
    if (!q||!K||!V||!og||!oc||!meta) { printf("OOM\n"); return 1; }

    for (size_t i=0;i<qn;i++)  q[i] = frnd();
    for (size_t i=0;i<kvn;i++) { K[i] = frnd(); V[i] = frnd(); }
    /* Two rows in DIFFERENT slots with DIFFERENT lengths: a kernel that ignored
     * meta and used slot 0 / the full context would still match on a uniform
     * batch, so the batch is made non-uniform on purpose. tmax is not a multiple
     * of 4 either, which exercises the subgroups' ragged tail. */
    meta[0]=0; meta[1]=kv_ctx-1;
    meta[2]=1; meta[3]=(kv_ctx/2)+1;

    float scale = 1.0f/sqrtf((float)hd);
    int grp = H/KVH;

    /* ---- reference: two-pass softmax ---- */
    float *sc = (float*)malloc((size_t)kv_ctx*4);
    for (int r=0;r<n;r++) {
        int slot=meta[r*2], tmax=meta[r*2+1];
        for (int h=0;h<H;h++) {
            int kvh=h/grp;
            const float *qv = q + (size_t)r*H*hd + (size_t)h*hd;
            const float *Kb = K + (((size_t)slot*KVH + kvh)*kv_ctx)*hd;
            const float *Vb = V + (((size_t)slot*KVH + kvh)*kv_ctx)*hd;
            float mx = -1e30f;
            for (int t=0;t<=tmax;t++) {
                double d=0; for (int i=0;i<hd;i++) d += (double)qv[i]*Kb[(size_t)t*hd+i];
                sc[t] = (float)(d*scale); if (sc[t]>mx) mx=sc[t];
            }
            double den=0; for (int t=0;t<=tmax;t++) { sc[t]=expf(sc[t]-mx); den+=sc[t]; }
            float *o = oc + (size_t)r*H*hd + (size_t)h*hd;
            for (int i=0;i<hd;i++) {
                double a=0; for (int t=0;t<=tmax;t++) a += (double)sc[t]*Vb[(size_t)t*hd+i];
                o[i] = (float)(a/den);
            }
        }
    }

    if (coli_vk_attn_ref(v,q,K,V,og,meta,n,H,KVH,hd,kv_ctx,slots,scale)!=0) {
        printf("FAIL: dispatch returned an error\n"); return 1; }

    double worst=0; size_t at=0;
    for (size_t i=0;i<qn;i++) {
        double d = fabs((double)og[i]-oc[i]) / (fabs((double)oc[i]) + 1e-3);
        if (d>worst) { worst=d; at=i; }
    }
    /* 1e-4, set from measurement rather than taste. The kernel accumulates in
     * float32 while this reference accumulates in double, so a gap of that order
     * is the reference being more precise, not the kernel being wrong. Measured
     * 2026-08-22 on an RTX 4070 across a 128x range of context length:
     *   kv_ctx   128     700    2048    8192   16384
     *   rel    1.574   1.595   1.450   1.498   1.370   (x 1e-5)
     * FLAT, not growing -- which is the point of sweeping it. A tolerance that
     * merely fits the one length someone happened to test would pass here and
     * fail silently at a production context. The control lands at ~9.9e-4, so
     * this still leaves an order of magnitude between pass and fail. */
    const double TOL = 1e-4;
    printf("  gpu vs two-pass reference: rel=%.3e (worst at %zu)  %s\n",
           worst, at, worst<=TOL ? "ok" : "BAD");

    double wc=0;
    for (size_t i=0;i<qn;i++) {
        double d = fabs((double)og[i]-oc[i]*1.001) / (fabs((double)oc[i]*1.001) + 1e-3);
        if (d>wc) wc=d;
    }
    printf("  control (reference x1.001): rel=%.3e -> %s\n", wc,
           wc>TOL ? "exceeds tolerance, as required" : "DID NOT EXCEED -- test is inert");

    int pass = (worst<=TOL) && (wc>TOL);

    /* ---------------------------------------------------------- resident KV
     * The path production will use: rows arrive one at a time and the kernel
     * reads them out of device memory. Same reference, same tolerance.
     *
     * The last position's rows are deliberately staged AFTER every other flush,
     * so the final dispatch reads a row written in its OWN command buffer --
     * the case a missing vkCmdPipelineBarrier would break.
     *
     * ⚠️ IT DOES NOT ACTUALLY CATCH THAT. Measured 2026-08-22: with the barrier
     * deleted from coli_vk_attn this test still passed, at a rel identical to
     * the barrier build, because the NVIDIA driver orders transfer-before-
     * compute on its own. The structure above is still the right shape and may
     * catch it on another driver, but nobody should read a pass here as
     * evidence the synchronisation is correct. Stated rather than quietly left
     * to be assumed. */
    if (coli_vk_kv_init(v,1,slots,KVH,kv_ctx,hd)!=0) {
        printf("  FAIL: coli_vk_kv_init\n"); coli_vk_free(v); return 1; }
    printf("  resident KV allocated: %.2f MiB\n", coli_vk_kv_bytes(v)/1048576.0);

    float *ostep = (float*)malloc(qn*4);
    int *m1 = (int*)malloc((size_t)n*2*sizeof(int));
    int staged_ok = 1;
    for (int pos=0; pos<kv_ctx-1 && staged_ok; pos++) {
        for (int sl=0; sl<slots; sl++) for (int kh=0; kh<KVH; kh++) {
            size_t off = (((size_t)sl*KVH + kh)*kv_ctx + pos)*hd;
            if (coli_vk_kv_put(v,sl,kh,pos,0,K+off)!=0) staged_ok=0;
            if (coli_vk_kv_put(v,sl,kh,pos,1,V+off)!=0) staged_ok=0;
        }
        /* flush: staging holds 64 rows and each position stages slots*KVH*2 */
        for (int r2=0;r2<n;r2++){ m1[r2*2]=meta[r2*2]; m1[r2*2+1]=pos; }
        if (coli_vk_attn(v,0,q,ostep,m1,n,H,scale)!=0) {
            printf("  FAIL: coli_vk_attn at pos %d\n", pos); coli_vk_free(v); return 1; }
    }
    for (int sl=0; sl<slots; sl++) for (int kh=0; kh<KVH; kh++) {
        size_t off = (((size_t)sl*KVH + kh)*kv_ctx + (kv_ctx-1))*hd;
        if (coli_vk_kv_put(v,sl,kh,kv_ctx-1,0,K+off)!=0) staged_ok=0;
        if (coli_vk_kv_put(v,sl,kh,kv_ctx-1,1,V+off)!=0) staged_ok=0;
    }
    if (!staged_ok) { printf("  FAIL: a row was not staged\n"); coli_vk_free(v); return 1; }

    if (coli_vk_attn(v,0,q,og,meta,n,H,scale)!=0) {
        printf("  FAIL: coli_vk_attn (final)\n"); coli_vk_free(v); return 1; }

    double rw=0;
    for (size_t i=0;i<qn;i++) {
        double d = fabs((double)og[i]-oc[i]) / (fabs((double)oc[i]) + 1e-3);
        if (d>rw) rw=d;
    }
    printf("  resident KV vs two-pass reference: rel=%.3e  %s\n", rw, rw<=TOL?"ok":"BAD");

    /* ------------------------------------------------- PREFILL shape
     * The kernel is used at n=683 in coli_forward and was tested only at n=2.
     * Those differ in more than size: prefill's meta is a causal RAMP, every
     * row a different tmax including tmax=0, and the rows arrive through
     * coli_vk_kv_write (one contiguous run per kv head) rather than the 64-row
     * staging ring, which caps n at 32 and so could never have been used here.
     * coli_vk_kv_write had no test at all before this block.
     *
     * The cache is re-initialised first, on purpose. If it were left holding the
     * rows the staged loop already wrote, this section would pass whether or not
     * kv_write did anything -- see the pre-write control below, which asserts a
     * fresh cache does NOT produce the reference. */
    {
        int pn = 192;                       /* < kv_ctx; the reference is O(pn^2) */
        if (pn > kv_ctx) pn = kv_ctx;
        float *pq = (float*)malloc((size_t)pn*H*hd*4);
        float *pg = (float*)malloc((size_t)pn*H*hd*4);
        float *pc = (float*)malloc((size_t)pn*H*hd*4);
        int   *pm = (int*)malloc((size_t)pn*2*sizeof(int));
        if (!pq||!pg||!pc||!pm) { printf("  OOM in prefill block\n"); coli_vk_free(v); return 1; }
        for (size_t i=0;i<(size_t)pn*H*hd;i++) pq[i] = frnd();
        /* slot 0, tmax = r: the causal ramp. Row 0 attends to exactly one
         * position, which is the degenerate case the n=2 batch never covers. */
        for (int r=0;r<pn;r++) { pm[r*2]=0; pm[r*2+1]=r; }

        if (coli_vk_kv_init(v,1,slots,KVH,kv_ctx,hd)!=0) {
            printf("  FAIL: kv_init (prefill)\n"); coli_vk_free(v); return 1; }

        /* CONTROL, and it runs BEFORE the write: on a freshly allocated cache
         * the kernel must NOT reproduce the reference. Without this, a kv_write
         * that silently did nothing would be indistinguishable from one that
         * worked, because the section that follows would still be comparing the
         * kernel against the same K and V the reference used. */
        if (coli_vk_attn(v,0,pq,pg,pm,pn,H,scale)!=0) {
            printf("  FAIL: coli_vk_attn (prefill, pre-write control)\n"); coli_vk_free(v); return 1; }

        /* reference, two-pass, over the SAME K/V the write is about to send */
        for (int r=0;r<pn;r++) {
            int tmax=pm[r*2+1];
            for (int h=0;h<H;h++) {
                int kvh=h/grp;
                const float *qv = pq + (size_t)r*H*hd + (size_t)h*hd;
                const float *Kb = K + (((size_t)0*KVH + kvh)*kv_ctx)*hd;
                const float *Vb = V + (((size_t)0*KVH + kvh)*kv_ctx)*hd;
                float mx = -1e30f;
                for (int t=0;t<=tmax;t++) {
                    double d=0; for (int i=0;i<hd;i++) d += (double)qv[i]*Kb[(size_t)t*hd+i];
                    sc[t] = (float)(d*scale); if (sc[t]>mx) mx=sc[t];
                }
                double den=0; for (int t=0;t<=tmax;t++) { sc[t]=expf(sc[t]-mx); den+=sc[t]; }
                float *o = pc + (size_t)r*H*hd + (size_t)h*hd;
                for (int i=0;i<hd;i++) {
                    double a=0; for (int t=0;t<=tmax;t++) a += (double)sc[t]*Vb[(size_t)t*hd+i];
                    o[i] = (float)(a/den);
                }
            }
        }

        double pre=0;
        for (size_t i=0;i<(size_t)pn*H*hd;i++) {
            double d = fabs((double)pg[i]-pc[i]) / (fabs((double)pc[i]) + 1e-3);
            if (d>pre) pre=d;
        }
        printf("  prefill pre-write control: rel=%.3e -> %s\n", pre,
               pre>TOL ? "differs, as required" : "MATCHES -- the write is not being tested");

        /* the thing under test: one bulk write of positions 0..pn-1 for slot 0 */
        if (coli_vk_kv_write(v,0,0,0,pn,K,V)!=0) {
            printf("  FAIL: coli_vk_kv_write\n"); coli_vk_free(v); return 1; }
        if (coli_vk_attn(v,0,pq,pg,pm,pn,H,scale)!=0) {
            printf("  FAIL: coli_vk_attn (prefill)\n"); coli_vk_free(v); return 1; }

        double pw=0; size_t pat=0;
        for (size_t i=0;i<(size_t)pn*H*hd;i++) {
            double d = fabs((double)pg[i]-pc[i]) / (fabs((double)pc[i]) + 1e-3);
            if (d>pw) { pw=d; pat=i; }
        }
        printf("  prefill n=%d (causal ramp, bulk write): rel=%.3e (worst at %zu)  %s\n",
               pn, pw, pat, pw<=TOL ? "ok" : "BAD");
        pass = pass && (pw<=TOL) && (pre>TOL);
        free(pq); free(pg); free(pc); free(pm);

        /* restore the resident rows the timing block below expects */
        if (coli_vk_kv_write(v,0,0,0,kv_ctx,K,V)!=0 ||
            coli_vk_kv_write(v,0,1,0,kv_ctx,K+(size_t)KVH*kv_ctx*hd,V+(size_t)KVH*kv_ctx*hd)!=0) {
            printf("  FAIL: kv_write (restore)\n"); coli_vk_free(v); return 1; }
    }

    /* Cost of the dispatch alone, at two very different context lengths. If the
     * two are close the kernel is dominated by fixed per-dispatch overhead; if
     * they scale with tmax it is reading K and V. Guessing between those two
     * has already cost this project four failed optimisations. */
    { struct timespec a,b; int REP=200;
      for (int shortctx=0; shortctx<2; shortctx++) {
        int tm = shortctx ? 63 : kv_ctx-1;
        for (int r2=0;r2<n;r2++){ m1[r2*2]=meta[r2*2]; m1[r2*2+1]=tm; }
        coli_vk_attn(v,0,q,ostep,m1,n,H,scale);           /* warm */
        clock_gettime(CLOCK_MONOTONIC,&a);
        for (int i=0;i<REP;i++) coli_vk_attn(v,0,q,ostep,m1,n,H,scale);
        clock_gettime(CLOCK_MONOTONIC,&b);
        double us = ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec))/1e3/REP;
        double mb = (double)n*H*(tm+1)*hd*2*4/1048576.0;
        printf("  dispatch tmax=%5d : %7.1f us   (%6.2f MiB of K+V -> %6.1f GB/s)\n",
               tm, us, mb, mb*1048576.0/(us*1e-6)/1e9);
      } }
    pass = pass && (rw<=TOL);

    printf("%s\n", pass ? "PASS (gpu matches the definition, and the control does not)" : "FAIL");
    coli_vk_free(v);
    return pass?0:1;
}
