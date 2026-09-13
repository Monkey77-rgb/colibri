/* test_attn_split -- does the split-K decode attention kernel pair
 * (attn_decode_split.comp + attn_decode_merge.comp) compute the same thing as
 * the TWO-PASS reference, and is it actually faster than attn_decode.comp at
 * the shape that motivated it?
 *
 * Modelled on tests/test_vk_attn.c: the reference is the two-pass softmax
 * (score everything, find the max, exponentiate, normalise, weight), which is
 * deliberately NOT the online-softmax identity either kernel uses -- testing
 * one online kernel against another online kernel would only prove they made
 * the same mistake. See test_vk_attn.c's header for the fuller version of
 * this argument; it is not repeated here.
 *
 * BOUND: relative L2 norm over the WHOLE batch output (n*H*hd elements),
 * ||gpu - ref||_2 / (||ref||_2 + 1e-6), not the per-element worst-case
 * test_vk_attn.c uses. Stated here rather than assumed identical to that
 * file's 1e-4 per-element bound: this kernel adds a SECOND rescale-and-merge
 * (across NCHUNK chunks, on top of attn_decode.comp's existing 4-subgroup
 * merge), so more independent roundings are summed and an L2 norm is the
 * fairer summary of "how far off, in aggregate" than a single worst element.
 * TOL is set from measurement below (see the printed table), not from taste.
 *
 * CONTROL: perturbs ONE V element after staging and re-attends; the L2 gap
 * MUST exceed TOL, or a comparison that always passes -- because it never
 * actually reads the corrupted data, or because the perturbation is too
 * small to move an L2 norm over n*H*hd elements -- would be proving nothing.
 *
 * SHAPES: H=32, KVH=4, hd=128 (Qwen3-30B-A3B's decode shape, per the
 * 2026-09-13 measurement in attn_decode_split.comp's header) x tmax in
 * {0,1,7,31,266,1023} x n in {1,4}. tmax=0 is the single-position degenerate
 * case; 1023 exercises a context an order of magnitude past the 266 the
 * kernel was tuned at; n=4 exercises more than one row per dispatch, which
 * split_nchunk's denominator (n*KVH) treats differently than n=1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../src/vk_backend.h"

static unsigned long rs = 987654321ULL;
/* Same load-bearing cast to long as test_vk_attn.c's frnd -- see that file's
 * comment. Without it the unsigned wraparound below 10000 poisons the whole
 * input set with ~9.2e14, and the reference and kernel would then agree on
 * having both consumed garbage. */
static float frnd(void){ rs = rs*6364136223846793005ULL + 1442695040888963407ULL;
                         return (float)((long)((rs>>33)%20001) - 10000) / 10000.0f; }

/* Two-pass reference over ONE (row, head): identical structure to
 * test_vk_attn.c's inline reference, factored into a function here because
 * this file calls it for every (n, tmax) combination in the sweep rather than
 * once. */
static void ref_two_pass(float *o, const float *qv, const float *Kb, const float *Vb,
                         int tmax, int hd, float scale, float *sc) {
    float mx = -1e30f;
    for (int t=0;t<=tmax;t++) {
        double d=0; for (int i=0;i<hd;i++) d += (double)qv[i]*Kb[(size_t)t*hd+i];
        sc[t] = (float)(d*scale); if (sc[t]>mx) mx=sc[t];
    }
    double den=0; for (int t=0;t<=tmax;t++) { sc[t]=expf(sc[t]-mx); den+=sc[t]; }
    for (int i=0;i<hd;i++) {
        double a=0; for (int t=0;t<=tmax;t++) a += (double)sc[t]*Vb[(size_t)t*hd+i];
        o[i] = (float)(a/den);
    }
}

/* relative L2 over `count` floats */
static double rel_l2(const float *a, const float *b, size_t count) {
    double num=0, den=0;
    for (size_t i=0;i<count;i++) { double d=(double)a[i]-(double)b[i]; num += d*d; den += (double)b[i]*(double)b[i]; }
    return sqrt(num) / (sqrt(den) + 1e-6);
}

int main(int argc, char **argv) {
    int H = 32, KVH = 4, hd = 128;
    int kv_ctx = 2048;   /* > every tmax below */
    float scale = 1.0f/sqrtf((float)hd);
    int grp = H/KVH;

    char err[256] = {0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { printf("no Vulkan device (%s) -- SKIP\n", err); return 0; }
    if (!coli_vk_has_attn(v))       { printf("shaders/attn_decode.spv absent -- SKIP\n"); coli_vk_free(v); return 0; }
    if (!coli_vk_has_attn_split(v)) { printf("shaders/attn_decode_split.spv or _merge.spv absent -- SKIP\n"); coli_vk_free(v); return 0; }
    printf("device: %s   H=%d KVH=%d hd=%d\n", coli_vk_device_name(v), H, KVH, hd);
    printf("  subgroup: native=%d, attn pipeline pinned to 32 = %s\n",
           coli_vk_subgroup_size(v), coli_vk_attn_pinned32(v) ? "yes" : "no (native is 32)");

    if (coli_vk_kv_init(v,1,1,KVH,kv_ctx,hd)!=0) { printf("FAIL: kv_init\n"); coli_vk_free(v); return 1; }

    /* Measured 2026-09-13 on an RTX 4070, this exact sweep (H=32,KVH=4,hd=128,
     * n in {1,4}, tmax in {0,1,7,31,266,1023}): worst rel_l2 was 3.331e-07 for
     * the un-split kernel and 1.762e-07 for split-K, both at n=1 tmax=1023.
     * FLAT across a 1024x range of context length in both arms, which is the
     * point of sweeping it -- same argument test_vk_attn.c's TOL comment
     * makes. 1e-4 leaves ~300x headroom over the worst observed value and the
     * control lands at 1.7e-01, so pass/fail is not a close call either way. */
    const double TOL = 1e-4;
    int pass = 1;

    int tmaxes[] = {0,1,7,31,266,1023};
    int ns[] = {1,4};
    /* saved for the timing block and the control, both at n=1, tmax=266 */
    float *Ksave=NULL, *Vsave=NULL; size_t kvn_save=0;

    for (int ni=0; ni<2; ni++) {
        int n = ns[ni];
        for (int ti=0; ti<6; ti++) {
            int tmax = tmaxes[ti];

            size_t qn  = (size_t)n*H*hd;
            size_t kvn = (size_t)KVH*kv_ctx*hd;   /* one slot */
            float *q = (float*)malloc(qn*4), *K = (float*)malloc(kvn*4), *V = (float*)malloc(kvn*4);
            float *o_old = (float*)malloc(qn*4), *o_new = (float*)malloc(qn*4), *o_ref = (float*)malloc(qn*4);
            int *meta = (int*)malloc((size_t)n*2*sizeof(int));
            float *sc = (float*)malloc((size_t)(tmax+1)*4);
            if (!q||!K||!V||!o_old||!o_new||!o_ref||!meta||!sc) { printf("OOM\n"); return 1; }

            for (size_t i=0;i<qn;i++) q[i] = frnd();
            for (size_t i=0;i<kvn;i++) { K[i]=frnd(); V[i]=frnd(); }
            for (int r=0;r<n;r++) { meta[r*2]=0; meta[r*2+1]=tmax; }

            if (coli_vk_kv_write(v,0,0,0,tmax+1,K,V)!=0) { printf("FAIL: kv_write\n"); return 1; }

            for (int r=0;r<n;r++) for (int h=0;h<H;h++) {
                int kvh = h/grp;
                const float *qv = q + (size_t)r*H*hd + (size_t)h*hd;
                const float *Kb = K + (size_t)kvh*kv_ctx*hd;
                const float *Vb = V + (size_t)kvh*kv_ctx*hd;
                ref_two_pass(o_ref + (size_t)r*H*hd + (size_t)h*hd, qv, Kb, Vb, tmax, hd, scale, sc);
            }

            if (coli_vk_attn(v,0,q,o_old,meta,n,H,scale)!=0)       { printf("FAIL: coli_vk_attn\n"); return 1; }
            if (coli_vk_attn_split(v,0,q,o_new,meta,n,H,scale)!=0) { printf("FAIL: coli_vk_attn_split\n"); return 1; }

            double rold = rel_l2(o_old,o_ref,qn);
            double rnew = rel_l2(o_new,o_ref,qn);
            printf("  n=%2d tmax=%5d : old rel_l2=%.3e %-4s  split rel_l2=%.3e %-4s\n",
                   n, tmax, rold, rold<=TOL?"ok":"BAD", rnew, rnew<=TOL?"ok":"BAD");
            pass = pass && (rold<=TOL) && (rnew<=TOL);

            if (n==1 && tmax==266) {
                Ksave = (float*)malloc(kvn*4); Vsave = (float*)malloc(kvn*4);
                if (Ksave && Vsave) { memcpy(Ksave,K,kvn*4); memcpy(Vsave,V,kvn*4); kvn_save=kvn; }
            }

            free(q);free(K);free(V);free(o_old);free(o_new);free(o_ref);free(meta);free(sc);
        }
    }

    /* ---------------- CONTROL: perturb one V element, must FAIL ---------------- */
    if (Ksave && Vsave) {
        int n=1, tmax=266;
        size_t qn = (size_t)n*H*hd;
        float *q = (float*)malloc(qn*4), *o_new = (float*)malloc(qn*4), *o_ref = (float*)malloc(qn*4);
        int meta[2] = {0,tmax};
        float *sc = (float*)malloc((size_t)(tmax+1)*4);
        for (size_t i=0;i<qn;i++) q[i]=frnd();
        for (int h=0;h<H;h++) {
            int kvh=h/grp;
            ref_two_pass(o_ref+(size_t)h*hd, q+(size_t)h*hd,
                        Ksave+(size_t)kvh*kv_ctx*hd, Vsave+(size_t)kvh*kv_ctx*hd, tmax, hd, scale, sc);
        }
        /* corrupt V at position tmax/2, kv-head 0, dim 0 -- squarely inside the
         * live range every row and every merge pass reads */
        float *Vbad = (float*)malloc(kvn_save*4);
        memcpy(Vbad, Vsave, kvn_save*4);
        Vbad[(size_t)(tmax/2)*hd + 0] += 37.0f;
        if (coli_vk_kv_write(v,0,0,0,tmax+1,Ksave,Vbad)!=0) { printf("FAIL: kv_write (control)\n"); return 1; }
        if (coli_vk_attn_split(v,0,q,o_new,meta,n,H,scale)!=0) { printf("FAIL: coli_vk_attn_split (control)\n"); return 1; }
        double rc = rel_l2(o_new,o_ref,qn);
        printf("  control (V perturbed by +37 at t=%d): rel_l2=%.3e -> %s\n",
               tmax/2, rc, rc>TOL ? "exceeds tolerance, as required" : "DID NOT EXCEED -- test is inert");
        pass = pass && (rc>TOL);
        free(Vbad); free(q); free(o_new); free(o_ref); free(sc);

        /* restore the uncorrupted cache for the timing block below */
        if (coli_vk_kv_write(v,0,0,0,tmax+1,Ksave,Vsave)!=0) { printf("FAIL: kv_write (restore)\n"); return 1; }
    } else {
        printf("  control SKIPPED -- n=1,tmax=266 case did not save K/V\n");
        pass = 0;
    }

    /* ---------------- TIMING: old vs new, n=1, tmax=266, 200 dispatches ---------------- */
    if (Ksave && Vsave) {
        int n=1, tmax=266;
        size_t qn = (size_t)n*H*hd;
        float *q = (float*)malloc(qn*4), *o = (float*)malloc(qn*4);
        int meta[2] = {0,tmax};
        for (size_t i=0;i<qn;i++) q[i]=frnd();

        int REP = 200;
        struct timespec a,b;

        coli_vk_attn(v,0,q,o,meta,n,H,scale);                       /* warm */
        clock_gettime(CLOCK_MONOTONIC,&a);
        for (int i=0;i<REP;i++) coli_vk_attn(v,0,q,o,meta,n,H,scale);
        clock_gettime(CLOCK_MONOTONIC,&b);
        double us_old = ((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/1e3/REP;

        coli_vk_attn_split(v,0,q,o,meta,n,H,scale);                  /* warm */
        clock_gettime(CLOCK_MONOTONIC,&a);
        for (int i=0;i<REP;i++) coli_vk_attn_split(v,0,q,o,meta,n,H,scale);
        clock_gettime(CLOCK_MONOTONIC,&b);
        double us_new = ((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/1e3/REP;

        printf("  TIMING n=%d tmax=%d, %d reps, submit+fence each call:\n", n, tmax, REP);
        printf("    old (attn_decode.comp)              : %8.2f us/call\n", us_old);
        printf("    new (split+merge, nchunk=%2d)        : %8.2f us/call   -> %.2fx %s\n",
               128/(n*KVH) < 1 ? 1 : (128/(n*KVH) > 64 ? 64 : 128/(n*KVH)),
               us_new, us_old>us_new ? us_old/us_new : us_new/us_old,
               us_old>us_new ? "FASTER" : "SLOWER");
        free(q); free(o);
    }

    free(Ksave); free(Vsave);
    printf("%s\n", pass ? "PASS (split-K matches the two-pass reference, and the control does not)" : "FAIL");
    coli_vk_free(v);
    return pass?0:1;
}
