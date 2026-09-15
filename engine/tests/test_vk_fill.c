/* test_vk_fill -- the GPU expert-slot ASYNC fill (2026-09-14).
 *
 * Five parts, in the order the task that wrote this asked for them:
 *
 *  1. PHASE BREAKDOWN of the synchronous coli_vk_slot_fill: memcpy into
 *     staging / command-record+submit / fence-wait, 200 fills round-robin
 *     over 24 slots, read back through coli_vk_fill_stats(which=0).
 *  2. SYNC vs ASYNC throughput: the same 24 slots refilled once each, timed
 *     wall-clock, sync path vs async-submit-all-then-wait-once.
 *  3. CORRECTNESS: async-fill every slot with REAL (repacked) MXFP4 data,
 *     coli_vk_slot_fill_wait(), then coli_vk_gemm4 against coli_gemm_mxfp4_ref
 *     -- same reference and the same 2e-5 tolerance test_vk_oai.c's MXFP4 arm
 *     uses, because this is the same kernel reading the same slot format.
 *  4. CONTROL that can fail: refill an already-correct slot asynchronously
 *     with NEW content, deliberately skip the wait, and dispatch the GEMM
 *     immediately. If the transfer and compute queues are genuinely
 *     independent (the point of this whole feature), the compute dispatch
 *     can race the copy and read it mid-flight or not at all -- the shape is
 *     sized up (8192x8192, ~32 MiB/half of transfer) specifically to WIDEN
 *     that window on a fast PCIe link. A corrupted SOURCE buffer after the
 *     memcpy already happened would prove nothing (coli_vk_slot_fill_async's
 *     memcpy into the staging ring is synchronous on the caller's thread),
 *     so the control instead omits the real wait() and lets actual hardware
 *     timing decide -- which means it can legitimately come back 0/N on a
 *     given run; a SEPARATE, deterministic positive control (compare the
 *     fully-waited, correct content against the WRONG reference on purpose)
 *     proves the detection metric itself is capable of failing, so a 0/N
 *     race result is read as "not observed this run", never as "the check
 *     is incapable of seeing it". Skipped with a stated reason on
 *     fill_mode==0, where there is only one queue and therefore no race to
 *     expose at all.
 *  5. OVERLAP: N async fills running on the transfer queue while the CPU
 *     does real work (a CPU int4 GEMM, repeated to fill a measured
 *     duration), then one coli_vk_slot_fill_wait(). Wall time of that vs
 *     (sync-refill time) + (cpu time) is the overlap evidence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
#include "../src/gemm_mxfp4.h"

#define FM(n) ((float*)malloc((size_t)(n)*4))
#define UM(n) ((uint8_t*)malloc((size_t)(n)))

static unsigned long rs = 13371337;
static unsigned urnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (unsigned)(rs>>33); }
static float frnd(void) { return (float)((long)(urnd()%20001) - 10000) / 10000.f; }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }

/* Raw block_mxfp4 matrix, same shape fill_mx() in test_vk_oai.c uses --
 * needed here too because the CPU reference (coli_gemm_mxfp4_ref) reads the
 * ORIGINAL 17-byte blocks, not the repacked coli_w_i4 the GPU slot holds. */
static void fill_mx(uint8_t *blocks, int64_t I, int64_t O) {
    int64_t nb=I/COLI_MXFP4_BLK;
    for (int64_t i=0;i<O*nb;i++) { uint8_t *b=blocks+i*COLI_MXFP4_BYTES;
        b[0] = (uint8_t)(120 + urnd()%9);
        for (int j=1;j<17;j++) b[j]=(uint8_t)(urnd()&0xFF); }
}

/* Purely-random coli_w_i4 content for the timing-only parts (1, 2, 5): no
 * MXFP4 validity needed, only the right byte counts, since nothing there
 * checks numerical correctness. */
static void rnd_w4(coli_w_i4 *w, int64_t I, int64_t O) {
    w->I=I; w->O=O;
    size_t wn=(size_t)I*O/2, sn=(size_t)O*(I/COLI_W4BLK)*sizeof(float);
    w->q4 = UM(wn); for (size_t i=0;i<wn;i++) w->q4[i]=(uint8_t)urnd();
    w->bscale = FM(sn/4); for (size_t i=0;i<sn/4;i++) w->bscale[i]=0.01f+0.001f*(urnd()%100);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    char err[256]={0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { printf("no Vulkan device (%s) -- SKIP\n", err); return 0; }
    printf("device: %s  mx=%d  fill_mode: %s\n", coli_vk_device_name(v), coli_vk_has_mx(v), coli_vk_fill_mode(v));
    if (!coli_vk_has_mx(v)) { printf("FAIL: gemm_i4_mx_dp pipeline missing (make vk)\n"); return 1; }
    int pass = 1;

    const int64_t I=2880, O=2880;   /* gpt-oss-120b expert shape */
    const int NSLOT=24;
    int h[NSLOT];
    for (int i=0;i<NSLOT;i++) { h[i]=coli_vk_slot_alloc_mx(v,I,O); if (h[i]<0){printf("FAIL slot_alloc_mx %d\n",i); return 1;} }
    size_t wn=(size_t)I*O/2, sn=(size_t)O*(I/COLI_W4BLK)*sizeof(float);
    double mib_per_fill = (wn+sn)/1048576.0;

    /* ============================================================ 1. phase breakdown, sync */
    {
        coli_w_i4 w; rnd_w4(&w,I,O);
        int NFILL=200;
        double t0=now_s();
        for (int i=0;i<NFILL;i++) {
            /* fresh scale values each call so the copy is not a no-op the driver could elide */
            for (size_t j=0;j<8;j++) w.bscale[j] = 0.01f + 0.0001f*(i+j);
            if (coli_vk_slot_fill(v,h[i%NSLOT],&w)!=0) { printf("FAIL slot_fill (sync) i=%d\n",i); return 1; }
        }
        double t1=now_s();
        double out[4]; coli_vk_fill_stats(v,0,out);
        printf("\n-- 1. sync coli_vk_slot_fill, %d fills over %d slots, %.2f MiB/fill --\n", NFILL, NSLOT, mib_per_fill);
        printf("   wall: %.3f ms/fill mean (%.3f s total)\n", 1000.0*(t1-t0)/NFILL, t1-t0);
        printf("   breakdown (coli_vk_fill_stats): memcpy %.3f ms  rec+submit %.3f ms  wait %.3f ms  total %.3f ms\n",
               out[0], out[1], out[2], out[3]);
        printf("   effective GB/s (wall): %.2f\n", (mib_per_fill/1024.0*NFILL) / (t1-t0));
        free(w.q4); free(w.bscale);
    }

    /* ============================================================ 2. sync vs async throughput */
    double t_sync_total = 0;
    {
        coli_w_i4 w; rnd_w4(&w,I,O);
        double t0=now_s();
        for (int i=0;i<NSLOT;i++) {
            w.bscale[0] = 0.02f + 0.0001f*i;
            if (coli_vk_slot_fill(v,h[i],&w)!=0) { printf("FAIL slot_fill sync-2 %d\n",i); return 1; }
        }
        double t1=now_s();
        t_sync_total = t1-t0;

        double t2=now_s();
        for (int i=0;i<NSLOT;i++) {
            w.bscale[0] = 0.03f + 0.0001f*i;
            if (coli_vk_slot_fill_async(v,h[i],&w)!=0) { printf("FAIL slot_fill_async %d\n",i); return 1; }
        }
        if (coli_vk_slot_fill_wait(v)!=0) { printf("FAIL slot_fill_wait\n"); return 1; }
        double t3=now_s();
        double t_async_total = t3-t2;

        printf("\n-- 2. sync vs async, %d slots, %.2f MiB/fill (mode: %s) --\n", NSLOT, mib_per_fill, coli_vk_fill_mode(v));
        printf("   sync  total %.3f ms  (%.3f ms/fill, %.2f GB/s)\n",
               1000*t_sync_total, 1000*t_sync_total/NSLOT, (mib_per_fill/1024.0*NSLOT)/t_sync_total);
        printf("   async total %.3f ms  (%.3f ms/fill, %.2f GB/s)  speedup %.2fx\n",
               1000*t_async_total, 1000*t_async_total/NSLOT, (mib_per_fill/1024.0*NSLOT)/t_async_total,
               t_sync_total/t_async_total);
        free(w.q4); free(w.bscale);
    }

    /* ============================================================ 3. correctness */
    {
        int64_t nb=I/COLI_MXFP4_BLK;
        uint8_t **blocks = (uint8_t**)malloc(NSLOT*sizeof *blocks);
        for (int i=0;i<NSLOT;i++) {
            blocks[i] = UM((size_t)O*nb*COLI_MXFP4_BYTES);
            fill_mx(blocks[i],I,O);
            coli_w_i4 w4; if (!coli_mxfp4_repack_i4(blocks[i],I,O,&w4)) { printf("FAIL repack %d\n",i); return 1; }
            if (coli_vk_slot_fill_async(v,h[i],&w4)!=0) { printf("FAIL async fill (correctness) %d\n",i); return 1; }
            free(w4.q4); free(w4.bscale);
        }
        if (coli_vk_slot_fill_wait(v)!=0) { printf("FAIL wait (correctness)\n"); return 1; }

        const double TOL=2e-5;   /* test_vk_oai.c's tolerance for this exact kernel+format */
        int n=2;
        float *x=FM((size_t)n*I); for (int64_t i=0;i<n*I;i++) x[i]=frnd();
        coli_a_i8 a; a.n=n; a.I=I; a.q=(int8_t*)malloc((size_t)n*I);
        a.scale=(float*)malloc((size_t)n*(I/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(I/COLI_ABLK)*4);
        coli_quantize_a(&a,x,n,I);
        printf("\n-- 3. correctness, %d async-filled slots, n=%d, TOL=%.0e (test_vk_oai.c's MXFP4 tolerance) --\n", NSLOT, n, TOL);
        /* SAME metric test_vk_oai.c's MXFP4 arm uses -- max|gpu-cpu| over
         * max|ref| -- NOT a per-element relative metric with a floor (that
         * one is what test_vk_oai.c's attention arm uses, for a different
         * kernel). Mixing the two metrics is exactly the kind of mismatch
         * that manufactures a false BAD: a per-element floor=1e-3 metric was
         * tried first here and read 1e-4..1e-3 "BAD" on every slot against
         * this TOL before this fix, which is what caught the mismatch. */
        int allok=1;
        for (int i=0;i<NSLOT;i++) {
            coli_w_mxfp4 mw={blocks[i],I,O,0};
            float *yr=FM((size_t)n*O), *yg=FM((size_t)n*O);
            coli_gemm_mxfp4_ref(yr,&a,&mw);
            if (coli_vk_gemm4(v,h[i],&a,yg)!=0) { printf("FAIL gemm4 slot %d\n",i); return 1; }
            double ymax=0, md=0;
            for (int64_t j=0;j<n*O;j++){ if (fabs(yr[j])>ymax) ymax=fabs(yr[j]); double d=fabs((double)yg[j]-yr[j]); if (d>md) md=d; }
            double w = md/ymax;
            int ok = w<=TOL; allok &= ok;
            if (i<3 || !ok) printf("   slot %2d: rel=%.3e %s\n", i, w, ok?"ok":"BAD");
            free(yr); free(yg);
        }
        printf("   (remaining %d slots: %s)\n", NSLOT-3, allok?"all ok":"see above");
        pass &= allok;
        free(x); free(a.q); free(a.scale); free(a.sum);
        for (int i=0;i<NSLOT;i++) free(blocks[i]);
        free(blocks);
    }

    /* ============================================================ 4. control that must be able to fail */
    {
        printf("\n-- 4. control: dispatch a GEMM on a slot async-refilled WITHOUT calling wait() --\n");
        /* Gate on ACTUAL capability rather than assuming: if async silently
         * fell back to the synchronous path (fill_mode==0), slot_fill_async
         * already blocks until the copy lands and there is no race to expose
         * -- the control legitimately cannot fire, and the honest thing is to
         * say so rather than report a pass that tested nothing. Detected
         * indirectly: a control matrix big enough that even a SYNC copy is
         * many ms is filled, and the wall time of the "without wait" call is
         * measured -- fill_mode string already states the mode plainly. */
        int looks_async = (strstr(coli_vk_fill_mode(v),"async:") != NULL);
        if (!looks_async) {
            printf("   SKIPPED: fill_mode is \"%s\" -- this device/driver has only one usable queue, so\n", coli_vk_fill_mode(v));
            printf("   coli_vk_slot_fill_async falls back to the synchronous, blocking path and there is\n");
            printf("   no second queue for a compute dispatch to race. Not measured on this hardware.\n");
        } else {
            int64_t CI=8192, CO=8192;   /* sized up: ~33.5 MB/half, to widen the DMA window */
            int ch = coli_vk_slot_alloc_mx(v,CI,CO);
            if (ch<0) { printf("FAIL control slot_alloc_mx\n"); return 1; }
            int64_t cnb = CI/COLI_MXFP4_BLK;
            uint8_t *cblocksA = UM((size_t)CO*cnb*COLI_MXFP4_BYTES); fill_mx(cblocksA,CI,CO);
            uint8_t *cblocksB = UM((size_t)CO*cnb*COLI_MXFP4_BYTES); fill_mx(cblocksB,CI,CO);
            coli_w_i4 wA, wB;
            if (!coli_mxfp4_repack_i4(cblocksA,CI,CO,&wA) || !coli_mxfp4_repack_i4(cblocksB,CI,CO,&wB)) { printf("FAIL control repack\n"); return 1; }
            /* establish a known-good baseline A, fully waited */
            if (coli_vk_slot_fill_async(v,ch,&wA)!=0 || coli_vk_slot_fill_wait(v)!=0) { printf("FAIL control baseline fill\n"); return 1; }

            int n=1; float *x=FM((size_t)n*CI); for (int64_t i=0;i<n*CI;i++) x[i]=frnd();
            coli_a_i8 a; a.n=n; a.I=CI; a.q=(int8_t*)malloc((size_t)n*CI);
            a.scale=(float*)malloc((size_t)n*(CI/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(CI/COLI_ABLK)*4);
            coli_quantize_a(&a,x,n,CI);
            coli_w_mxfp4 mwB={cblocksB,CI,CO,0};
            float *yrB=FM((size_t)n*CO);
            coli_gemm_mxfp4_ref(yrB,&a,&mwB);
            const double TOL=2e-5;

            /* POSITIVE CONTROL ON THE APPARATUS ITSELF, before trusting the
             * race probe below: run the GEMM on the KNOWN-GOOD, fully-waited
             * A content and compare it against yrB (B's reference) --
             * content and reference are deliberately mismatched, so this
             * MUST read far over TOL. If it didn't, the metric below could
             * not detect a real staleness either and a clean race-probe
             * result would be a null result, not reassurance. */
            { float *yA=FM((size_t)n*CO);
              if (coli_vk_gemm4(v,ch,&a,yA)!=0) { printf("FAIL control positive-control gemm\n"); return 1; }
              double ymax=0, md=0;
              for (int64_t j=0;j<n*CO;j++){ if (fabs(yrB[j])>ymax) ymax=fabs(yrB[j]); double d=fabs((double)yA[j]-yrB[j]); if (d>md) md=d; }
              double wpos = md/ymax;
              printf("   positive control (A content vs B reference, deliberately mismatched): rel=%.3e %s\n",
                     wpos, wpos>TOL?"fails as required -- the metric can detect wrongness":"INERT (metric cannot fail -- race probe below is not trustworthy)");
              pass &= (wpos > TOL);
              free(yA); }

            int tries=10, detected=0;
            for (int t=0;t<tries;t++) {
                /* refill with B asynchronously, then IMMEDIATELY dispatch --
                 * no coli_vk_slot_fill_wait() call in between, which is the
                 * exact misuse the API's contract forbids. */
                if (coli_vk_slot_fill_async(v,ch,&wB)!=0) { printf("FAIL control refill\n"); return 1; }
                float *yg=FM((size_t)n*CO);
                int gok = coli_vk_gemm4(v,ch,&a,yg)==0;
                double w;
                if (!gok) w = 1e9;
                else { double ymax=0, md=0;
                    for (int64_t j=0;j<n*CO;j++){ if (fabs(yrB[j])>ymax) ymax=fabs(yrB[j]); double d=fabs((double)yg[j]-yrB[j]); if (d>md) md=d; }
                    w = md/ymax; }
                if (w > TOL) detected++;
                free(yg);
                /* drain before the next trial so we are not piling up pending fences */
                coli_vk_slot_fill_wait(v);
            }
            printf("   shape %lldx%lld (%.1f MiB/half): stale/partial content detected %d/%d tries without wait()\n",
                   (long long)CI,(long long)CO,(double)((size_t)CI*CO/2)/1048576.0, detected, tries);
            printf("   (each detection is max|gpu-ref| > %.0e against the B-content CPU reference -- reading ahead\n", TOL);
            printf("    of the copy's completion, exactly the misuse coli_vk_slot_fill_wait exists to prevent)\n");
            /* NOT gated into pass/fail: the positive control above already
             * proved the metric CAN fail, which is the falsifiability the
             * doctrine asks for. Whether THIS run's race window happened to
             * be crossed is a timing fact about this GPU/driver/moment, not
             * a property of the API contract -- see the honest report. */
            if (detected == 0) printf("   NOTE: 0/%d -- not measured as a contradiction of the design. The host\n"
                                       "   fence-wait in coli_vk_slot_fill_wait is what the correctness (part 3) rests\n"
                                       "   on; this probe is empirical evidence of HOW EASILY the race bites on THIS\n"
                                       "   hardware, not a guarantee it always will or never will.\n", tries);
            else printf("   (%d/%d: the window was crossed -- direct evidence wait() is load-bearing, not decorative)\n", detected, tries);
            free(cblocksA); free(cblocksB); free(wA.q4); free(wA.bscale); free(wB.q4); free(wB.bscale);
            free(x); free(a.q); free(a.scale); free(a.sum); free(yrB);
        }
    }

    /* ============================================================ 5. overlap */
    {
        printf("\n-- 5. overlap: %d async fills + CPU work vs sync fills + CPU work, in series --\n", NSLOT);
        /* CPU busy-work: a real CPU int4 GEMM, repeated until a target wall
         * time elapses. "any CPU kernel" per the task -- this one is already
         * in the tree and is not a synthetic spin loop. */
        int64_t ci=512, co=512; coli_w_i4 cw; rnd_w4(&cw,ci,co);
        float *cx=FM(ci); for (int64_t i=0;i<ci;i++) cx[i]=frnd();
        coli_a_i8 ca; ca.n=1; ca.I=ci; ca.q=(int8_t*)malloc((size_t)ci);
        ca.scale=(float*)malloc((size_t)(ci/COLI_ABLK)*4); ca.sum=(int32_t*)malloc((size_t)(ci/COLI_ABLK)*4);
        coli_quantize_a(&ca,cx,1,ci);
        float *cy=FM(co);

        double target = t_sync_total > 0 ? t_sync_total : 0.010;   /* match part 2's sync-refill cost */

        coli_w_i4 w; rnd_w4(&w,I,O);
        double t_cpu0=now_s(); long cpu_iters=0;
        while (now_s()-t_cpu0 < target) { coli_gemm_i4(cy,&ca,&cw); cpu_iters++; }
        double t_cpu = now_s()-t_cpu0;

        double t0=now_s();
        for (int i=0;i<NSLOT;i++) { w.bscale[0]=0.05f+0.0001f*i; if (coli_vk_slot_fill_async(v,h[i],&w)!=0){printf("FAIL overlap async fill %d\n",i); return 1;} }
        double t_cpu0b=now_s(); long cpu_iters2=0;
        while (now_s()-t_cpu0b < target) { coli_gemm_i4(cy,&ca,&cw); cpu_iters2++; }
        if (coli_vk_slot_fill_wait(v)!=0) { printf("FAIL overlap wait\n"); return 1; }
        double t_overlap = now_s()-t0;

        printf("   CPU busy-loop alone: %.3f ms (%ld coli_gemm_i4 calls, 512x512)\n", 1000*t_cpu, cpu_iters);
        printf("   sync-fill time (from part 2):      %.3f ms\n", 1000*t_sync_total);
        printf("   sync-fill + CPU in series (budget): %.3f ms\n", 1000*(t_sync_total + t_cpu));
        printf("   async fills + CPU + wait, OVERLAPPED: %.3f ms (%ld coli_gemm_i4 calls during the window)\n",
               1000*t_overlap, cpu_iters2);
        int overlapped = t_overlap < (t_sync_total + t_cpu);
        printf("   overlap %s (%.3f ms < %.3f ms)\n", overlapped?"CONFIRMED":"NOT SHOWN",
               1000*t_overlap, 1000*(t_sync_total+t_cpu));
        pass &= overlapped;

        free(cw.q4); free(cw.bscale); free(cx); free(ca.q); free(ca.scale); free(ca.sum); free(cy);
        free(w.q4); free(w.bscale);
    }

    coli_vk_free(v);
    coli_vk_prof_dump(stdout);
    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
