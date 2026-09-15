/* test_cuda_fill -- the CUDA expert-slot ASYNC fill (2026-09-15), mirroring
 * tests/test_vk_fill.c one for one, through the coli_backend seam instead of
 * vk_backend.h directly (backend_cuda.cu exposes only
 * coli_backend_cuda_open / coli_cuda_probe_class + the test-only
 * coli_cuda_fill_stats, per backend.h's contract -- same pattern
 * tests/test_cuda_gemm.c and tests/test_cuda_oai.c already use).
 *
 * Five parts, same order as test_vk_fill.c:
 *
 *  1. PHASE BREAKDOWN of the synchronous slot_fill: memcpy into staging /
 *     command-issue / wait, 200 fills round-robin over 24 slots, read back
 *     through coli_cuda_fill_stats(which=0).
 *  2. SYNC vs ASYNC throughput: the same 24 slots refilled once each, timed
 *     wall-clock, sync path vs async-submit-all-then-wait-once.
 *  3. CORRECTNESS: async-fill every slot with REAL (repacked) MXFP4 data,
 *     slot_fill_wait(), then gemm4 against coli_gemm_mxfp4_ref -- same
 *     reference and TOL=2e-5 tests/test_cuda_oai.c's MXFP4 arm uses, because
 *     this is the same kernel reading the same slot format.
 *  4. CONTROL that can fail: refill an already-correct slot asynchronously
 *     with NEW content, deliberately skip the wait, and dispatch the GEMM
 *     immediately. Sized up (8192x8192) to widen the DMA window on a fast
 *     PCIe link, same rationale as test_vk_fill.c's arm 4. A SEPARATE,
 *     deterministic positive control (known-good, fully-waited content
 *     compared against the WRONG reference on purpose) proves the detection
 *     metric itself is capable of failing, so a 0/N race-probe result reads
 *     as "not observed this run", never as "the check is incapable of seeing
 *     it". Both device queues here are real CUDA streams on the SAME
 *     hardware queue family (there is only one copy engine on consumer
 *     Ada), so unlike Vulkan's fill_mode==0 fallback this control is not
 *     skippable by mode -- cudaMemcpyAsync + a kernel launch on two
 *     DIFFERENT streams are genuinely unordered without the
 *     cudaStreamWaitEvent slot_fill_wait installs, which is exactly what
 *     this control omits on purpose.
 *  5. OVERLAP: N async fills running on the copy stream while the CPU does
 *     real work (a CPU int4 GEMM, repeated to fill a measured duration),
 *     then one slot_fill_wait(). Wall time of that vs (sync-refill time) +
 *     (cpu time) is the overlap evidence.
 */
#include "../src/backend.h"
#include "../src/gemm_i8.h"
#include "../src/gemm_mxfp4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

extern "C" void coli_cuda_fill_stats(void *ctx, int which, double out[4]);

#define FM(n) ((float*)malloc((size_t)(n)*4))
#define UM(n) ((uint8_t*)malloc((size_t)(n)))

static unsigned long rs = 13371337;
static unsigned urnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (unsigned)(rs>>33); }
static float frnd(void) { return (float)((long)(urnd()%20001) - 10000) / 10000.f; }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }

/* Raw block_mxfp4 matrix, same shape test_cuda_oai.c's fill_mx() uses --
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
    coli_backend *be = coli_backend_cuda_open(err, sizeof err);
    if (!be) { printf("no CUDA device (%s) -- SKIP\n", err); return 0; }
    printf("device: %s  mx=%d  fill_mode: %s\n", be->device_name(be->ctx), be->has_mx(be->ctx), be->fill_mode(be->ctx));
    if (!be->has_mx(be->ctx)) { printf("FAIL: MXFP4 pipeline missing\n"); return 1; }
    int pass = 1;

    const int64_t I=2880, O=2880;   /* gpt-oss-120b expert shape, same as test_vk_fill.c */
    const int NSLOT=24;
    int h[NSLOT];
    for (int i=0;i<NSLOT;i++) { h[i]=be->slot_alloc_mx(be->ctx,I,O); if (h[i]<0){printf("FAIL slot_alloc_mx %d\n",i); return 1;} }
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
            if (be->slot_fill(be->ctx,h[i%NSLOT],&w)!=0) { printf("FAIL slot_fill (sync) i=%d\n",i); return 1; }
        }
        double t1=now_s();
        double out[4]; coli_cuda_fill_stats(be->ctx,0,out);
        printf("\n-- 1. sync slot_fill, %d fills over %d slots, %.2f MiB/fill --\n", NFILL, NSLOT, mib_per_fill);
        printf("   wall: %.3f ms/fill mean (%.3f s total)\n", 1000.0*(t1-t0)/NFILL, t1-t0);
        printf("   breakdown (coli_cuda_fill_stats): memcpy %.3f ms  issue %.3f ms  wait %.3f ms  total %.3f ms\n",
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
            if (be->slot_fill(be->ctx,h[i],&w)!=0) { printf("FAIL slot_fill sync-2 %d\n",i); return 1; }
        }
        double t1=now_s();
        t_sync_total = t1-t0;

        /* Warm-up (2026-09-15): the ring's pinned staging buffers and events are
         * created lazily on the FIRST async call, so without this the timed arm
         * measured setup, not fills -- lead's re-run read 2.45 ms/fill here vs
         * 0.28 ms/fill in the ring-full arm that ran after it. One full ring of
         * fills + wait moves that cost outside the clock. */
        for (int i=0;i<NSLOT && i<8;i++) { w.bscale[0]=0.025f; if (be->slot_fill_async(be->ctx,h[i],&w)!=0) { printf("FAIL warm-up async fill %d\n",i); return 1; } }
        if (be->slot_fill_wait(be->ctx)!=0) { printf("FAIL warm-up wait\n"); return 1; }
        double t2=now_s();
        for (int i=0;i<NSLOT;i++) {
            w.bscale[0] = 0.03f + 0.0001f*i;
            if (be->slot_fill_async(be->ctx,h[i],&w)!=0) { printf("FAIL slot_fill_async %d\n",i); return 1; }
        }
        if (be->slot_fill_wait(be->ctx)!=0) { printf("FAIL slot_fill_wait\n"); return 1; }
        double t3=now_s();
        double t_async_total = t3-t2;

        printf("\n-- 2. sync vs async, %d slots, %.2f MiB/fill (mode: %s) --\n", NSLOT, mib_per_fill, be->fill_mode(be->ctx));
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
            if (be->slot_fill_async(be->ctx,h[i],&w4)!=0) { printf("FAIL async fill (correctness) %d\n",i); return 1; }
            free(w4.q4); free(w4.bscale);
        }
        if (be->slot_fill_wait(be->ctx)!=0) { printf("FAIL wait (correctness)\n"); return 1; }

        const double TOL=2e-5;   /* test_cuda_oai.c's tolerance for this exact kernel+format */
        int n=2;
        float *x=FM((size_t)n*I); for (int64_t i=0;i<n*I;i++) x[i]=frnd();
        coli_a_i8 a; a.n=n; a.I=I; a.q=(int8_t*)malloc((size_t)n*I);
        a.scale=(float*)malloc((size_t)n*(I/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(I/COLI_ABLK)*4);
        coli_quantize_a(&a,x,n,I);
        printf("\n-- 3. correctness, %d async-filled slots, n=%d, TOL=%.0e (test_cuda_oai.c's MXFP4 tolerance) --\n", NSLOT, n, TOL);
        int allok=1;
        for (int i=0;i<NSLOT;i++) {
            coli_w_mxfp4 mw={blocks[i],I,O,0};
            float *yr=FM((size_t)n*O), *yg=FM((size_t)n*O);
            coli_gemm_mxfp4_ref(yr,&a,&mw);
            if (be->gemm4(be->ctx,h[i],&a,yg)!=0) { printf("FAIL gemm4 slot %d\n",i); return 1; }
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
        int64_t CI=8192, CO=8192;   /* sized up: ~33.5 MB/half, to widen the DMA window */
        int ch = be->slot_alloc_mx(be->ctx,CI,CO);
        if (ch<0) { printf("FAIL control slot_alloc_mx\n"); return 1; }
        int64_t cnb = CI/COLI_MXFP4_BLK;
        uint8_t *cblocksA = UM((size_t)CO*cnb*COLI_MXFP4_BYTES); fill_mx(cblocksA,CI,CO);
        uint8_t *cblocksB = UM((size_t)CO*cnb*COLI_MXFP4_BYTES); fill_mx(cblocksB,CI,CO);
        coli_w_i4 wA, wB;
        if (!coli_mxfp4_repack_i4(cblocksA,CI,CO,&wA) || !coli_mxfp4_repack_i4(cblocksB,CI,CO,&wB)) { printf("FAIL control repack\n"); return 1; }
        /* establish a known-good baseline A, fully waited */
        if (be->slot_fill_async(be->ctx,ch,&wA)!=0 || be->slot_fill_wait(be->ctx)!=0) { printf("FAIL control baseline fill\n"); return 1; }

        int n=1; float *x=FM((size_t)n*CI); for (int64_t i=0;i<n*CI;i++) x[i]=frnd();
        coli_a_i8 a; a.n=n; a.I=CI; a.q=(int8_t*)malloc((size_t)n*CI);
        a.scale=(float*)malloc((size_t)n*(CI/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(CI/COLI_ABLK)*4);
        coli_quantize_a(&a,x,n,CI);
        coli_w_mxfp4 mwB={cblocksB,CI,CO,0};
        float *yrB=FM((size_t)n*CO);
        coli_gemm_mxfp4_ref(yrB,&a,&mwB);
        const double TOL=2e-5;

        /* POSITIVE CONTROL ON THE APPARATUS ITSELF, before trusting the race
         * probe below: run the GEMM on the KNOWN-GOOD, fully-waited A content
         * and compare it against yrB (B's reference) -- content and
         * reference are deliberately mismatched, so this MUST read far over
         * TOL. If it didn't, the metric below could not detect a real
         * staleness either and a clean race-probe result would be a null
         * result, not reassurance. */
        { float *yA=FM((size_t)n*CO);
          if (be->gemm4(be->ctx,ch,&a,yA)!=0) { printf("FAIL control positive-control gemm\n"); return 1; }
          double ymax=0, md=0;
          for (int64_t j=0;j<n*CO;j++){ if (fabs(yrB[j])>ymax) ymax=fabs(yrB[j]); double d=fabs((double)yA[j]-yrB[j]); if (d>md) md=d; }
          double wpos = md/ymax;
          printf("   positive control (A content vs B reference, deliberately mismatched): rel=%.3e %s\n",
                 wpos, wpos>TOL?"fails as required -- the metric can detect wrongness":"INERT (metric cannot fail -- race probe below is not trustworthy)");
          pass &= (wpos > TOL);
          free(yA); }

        int tries=10, detected=0;
        for (int t=0;t<tries;t++) {
            /* refill with B asynchronously, then IMMEDIATELY dispatch -- no
             * slot_fill_wait() call in between, which is the exact misuse
             * the API's contract forbids. */
            if (be->slot_fill_async(be->ctx,ch,&wB)!=0) { printf("FAIL control refill\n"); return 1; }
            float *yg=FM((size_t)n*CO);
            int gok = be->gemm4(be->ctx,ch,&a,yg)==0;
            double w;
            if (!gok) w = 1e9;
            else { double ymax=0, md=0;
                for (int64_t j=0;j<n*CO;j++){ if (fabs(yrB[j])>ymax) ymax=fabs(yrB[j]); double d=fabs((double)yg[j]-yrB[j]); if (d>md) md=d; }
                w = md/ymax; }
            if (w > TOL) detected++;
            free(yg);
            /* drain before the next trial so we are not piling up pending events */
            be->slot_fill_wait(be->ctx);
        }
        printf("   shape %lldx%lld (%.1f MiB/half): stale/partial content detected %d/%d tries without wait()\n",
               (long long)CI,(long long)CO,(double)((size_t)CI*CO/2)/1048576.0, detected, tries);
        printf("   (each detection is max|gpu-ref| > %.0e against the B-content CPU reference -- reading ahead\n", TOL);
        printf("    of the copy's completion, exactly the misuse slot_fill_wait exists to prevent)\n");
        /* NOT gated into pass/fail: the positive control above already
         * proved the metric CAN fail, which is the falsifiability the
         * doctrine asks for. Whether THIS run's race window happened to be
         * crossed is a timing fact about this GPU/driver/moment, not a
         * property of the API contract. */
        if (detected == 0) printf("   NOTE: 0/%d -- not measured as a contradiction of the design. The host\n"
                                   "   event-wait in slot_fill_wait is what the correctness (part 3) rests on;\n"
                                   "   this probe is empirical evidence of HOW EASILY the race bites on THIS\n"
                                   "   hardware, not a guarantee it always will or never will.\n", tries);
        else printf("   (%d/%d: the window was crossed -- direct evidence wait() is load-bearing, not decorative)\n", detected, tries);
        free(cblocksA); free(cblocksB); free(wA.q4); free(wA.bscale); free(wB.q4); free(wB.bscale);
        free(x); free(a.q); free(a.scale); free(a.sum); free(yrB);
    }

    /* ============================================================ 4b. ring-full case: more fills queued than ring slots */
    {
        printf("\n-- 4b. ring-full: queue more async fills than the ring holds, then wait --\n");
        int NQ = 40;   /* > COLI_CUDA_FILL_INFLIGHT (8); forces backpressure inside slot_fill_async itself */
        coli_w_i4 w; rnd_w4(&w,I,O);
        double t0=now_s();
        for (int q=0;q<NQ;q++) {
            w.bscale[0] = 0.07f + 0.0001f*q;
            if (be->slot_fill_async(be->ctx,h[q%NSLOT],&w)!=0) { printf("FAIL ring-full async fill %d\n",q); return 1; }
        }
        if (be->slot_fill_wait(be->ctx)!=0) { printf("FAIL ring-full wait\n"); return 1; }
        double t1=now_s();
        printf("   %d async fills over an 8-deep ring: %.3f ms total (%.3f ms/fill) -- no deadlock, no growth-without-bound\n",
               NQ, 1000*(t1-t0), 1000*(t1-t0)/NQ);
        free(w.q4); free(w.bscale);
    }

    /* ============================================================ 5. overlap */
    {
        printf("\n-- 5. overlap: one ring (8) of async fills + CPU work vs sync fills + CPU work, in series --\n");
        int64_t ci=512, co=512; coli_w_i4 cw; rnd_w4(&cw,ci,co);
        float *cx=FM(ci); for (int64_t i=0;i<ci;i++) cx[i]=frnd();
        coli_a_i8 ca; ca.n=1; ca.I=ci; ca.q=(int8_t*)malloc((size_t)ci);
        ca.scale=(float*)malloc((size_t)(ci/COLI_ABLK)*4); ca.sum=(int32_t*)malloc((size_t)(ci/COLI_ABLK)*4);
        coli_quantize_a(&ca,cx,1,ci);
        float *cy=FM(co);

        /* CPU work (lead, 2026-09-15): FIXED scalar work, no OpenMP. The first version
         * time-boxed coli_gemm_i4 calls to the sync-fill duration, which cannot fail
         * (overlapped = max(box, fills) + wait is always under box + fills); the second
         * did a fixed count of coli_gemm_i4 calls, whose OpenMP time swung 7x run to run
         * under a loaded machine (94..640 calls in 8 ms) and failed 3 of 6 for noise.
         * Now: a scalar xor-sum over an 8 MB buffer, iterations calibrated once to about
         * the sync-fill time, and BOTH arms are best-of-5 so the comparison is between
         * their floors. Fails if issuing the async fills blocks for their duration. */
        const size_t CPUB = 8u<<20; unsigned char *cbuf=(unsigned char*)malloc(CPUB); for(size_t i=0;i<CPUB;i++) cbuf[i]=(unsigned char)i;
        volatile unsigned long sink=0;
        #define CPU_PASS() do{ unsigned long x=0; for(size_t i=0;i<CPUB;i+=64) x^=cbuf[i]+i; sink+=x; }while(0)
        double tcal0=now_s(); CPU_PASS(); double t_one=now_s()-tcal0;
        double target = t_sync_total > 0 ? t_sync_total : 0.010;
        long cpu_iters = (long)(target / (t_one>0?t_one:1e-6)); if (cpu_iters<1) cpu_iters=1;
        coli_w_i4 w; rnd_w4(&w,I,O);
        /* Overlap over ONE ring of fills (8): fill_async blocks on the ring when it is
         * full, so 24 fills into an 8-deep ring wait for 16 copies before the CPU work
         * can start -- that is backpressure (arm 4b), not overlap. */
        const int NOV = NSLOT < 8 ? NSLOT : 8;
        double t_cpu=1e9, t_overlap=1e9, t_sync5=1e9; long cpu_iters2=cpu_iters;
        for (int rep=0; rep<5; rep++) {
            double c0=now_s(); for (long k=0;k<cpu_iters;k++) CPU_PASS(); double c=now_s()-c0; if (c<t_cpu) t_cpu=c;
            double s0=now_s(); for (int i=0;i<NOV;i++) { w.bscale[0]=0.04f+0.0001f*i; if (be->slot_fill(be->ctx,h[i],&w)!=0){printf("FAIL overlap sync fill %d\n",i); return 1;} }
            double sd=now_s()-s0; if (sd<t_sync5) t_sync5=sd;
            double t0=now_s();
            for (int i=0;i<NOV;i++) { w.bscale[0]=0.05f+0.0001f*i; if (be->slot_fill_async(be->ctx,h[i],&w)!=0){printf("FAIL overlap async fill %d\n",i); return 1;} }
            for (long k=0;k<cpu_iters;k++) CPU_PASS();
            if (be->slot_fill_wait(be->ctx)!=0) { printf("FAIL overlap wait\n"); return 1; }
            double o=now_s()-t0; if (o<t_overlap) t_overlap=o;
        }
        t_sync_total = t_sync5;   /* same-arm sync fills, best-of-5, for the budget */
        (void)cy; (void)ca; (void)cw; free(cbuf);
        printf("   CPU scalar work alone (best of 5): %.3f ms (%ld passes over 8 MB)\n", 1000*t_cpu, cpu_iters);
        printf("   sync fills (best of 5):            %.3f ms\n", 1000*t_sync_total);
        printf("   sync-fill + CPU in series (budget): %.3f ms\n", 1000*(t_sync_total + t_cpu));
        printf("   async fills + CPU + wait, OVERLAPPED (best of 5): %.3f ms (%ld passes during the window)\n",
               1000*t_overlap, cpu_iters2);
        int overlapped = t_overlap < (t_sync_total + t_cpu);
        printf("   overlap %s (%.3f ms < %.3f ms)\n", overlapped?"CONFIRMED":"NOT SHOWN",
               1000*t_overlap, 1000*(t_sync_total+t_cpu));
        pass &= overlapped;

        free(cw.q4); free(cw.bscale); free(cx); free(ca.q); free(ca.scale); free(ca.sum); free(cy);
        free(w.q4); free(w.bscale);
    }

    coli_backend_close(be);
    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
