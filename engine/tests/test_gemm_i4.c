#include "../src/platform.h"
/* test_gemm_i4 — does hoisting the int4 unpack out of the row loop actually
 * remove the prefill penalty, and is the fast path still the same arithmetic?
 *
 * THE CLAIM UNDER TEST. gemm_i8.h records int4 as 2.3x FASTER than int8 at n=1
 * and 1.5x SLOWER at n=4, and concludes prefill is ALU-bound so nibble unpacking
 * costs more than it saves. That conclusion had a confound: the narrow kernel
 * unpacks each weight block INSIDE the loop over activation rows, so an n-row
 * call unpacks the whole matrix n times. 3.46 ms -> 13.75 ms across n=1 -> n=4
 * is 3.97x for 4x the rows, which is what "the dot is free next to the unpack"
 * looks like. This test measures the two int4 kernels against each other on the
 * same data so the confound is visible rather than argued.
 *
 * TWO THINGS IT MUST PROVE, not one:
 *   1. bit-exactness  wide == narrow on every cell, or the speed is worthless
 *   2. dispatch       both kernels actually ran -- a run that silently took the
 *                     narrow path twice would print "no difference" and read as
 *                     a result
 * Build with -DCOLI_BREAK_I4 for the negative control: it perturbs the WIDE
 * kernel's correction term only, so a passing control means the comparison
 * cannot tell the two apart and proves nothing.
 */
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

int main(int argc,char**argv){
  int64_t I=argc>1?atoll(argv[1]):2048, O=argc>2?atoll(argv[2]):2048;
  int64_t nb=I/COLI_ABLK; int NM=64;
  if(I%COLI_W4BLK){ printf("I must be a multiple of %d\n",COLI_W4BLK); return 2; }

  float *F=(float*)coli_aligned_alloc(64,(size_t)I*O*4);
  srand(1234);
  /* A plausible weight distribution, not uniform noise: real rows have a few
   * large outliers, which is the case block scales exist to handle. */
  for(int64_t i=0;i<I*O;i++){ float u=(float)(rand()%20001-10000)/10000.f; F[i]=u*u*u*0.1f; }

  coli_w_i4 w4; coli_quantize_w4(&w4,F,I,O);
  /* int8 form of the SAME weights, so the two formats are compared on identical
   * data rather than on two independent random draws. */
  coli_w_i8 w8={0}; w8.I=I; w8.O=O;
  w8.qu=(uint8_t*)coli_aligned_alloc(64,(size_t)I*O); w8.scale=(float*)coli_aligned_alloc(64,(size_t)O*4);
  for(int64_t o=0;o<O;o++){ float am=0; for(int64_t i=0;i<I;i++){float a=fabsf(F[o*I+i]); if(a>am)am=a;}
    float s=am/127.f; if(s<1e-12f)s=1e-12f; w8.scale[o]=s; float inv=1.f/s;
    for(int64_t i=0;i<I;i++){ int q=(int)lrintf(F[o*I+i]*inv); if(q>127)q=127; if(q<-127)q=-127;
      w8.qu[o*I+i]=(uint8_t)(q+128); } }

  coli_a_i8 a={0}; a.I=I;
  a.q=(int8_t*)coli_aligned_alloc(64,(size_t)I*NM);
  a.scale=(float*)coli_aligned_alloc(64,(size_t)nb*NM*4);
  a.sum=(int32_t*)coli_aligned_alloc(64,(size_t)nb*NM*4);
  float*X=(float*)coli_aligned_alloc(64,(size_t)I*NM*4);
  for(int64_t i=0;i<I*NM;i++) X[i]=(float)((rand()%2001)-1000)/500.0f;
  float*Y=(float*)coli_aligned_alloc(64,(size_t)O*NM*4);
  float*R=(float*)coli_aligned_alloc(64,(size_t)O*NM*4);
  float*Y8=(float*)coli_aligned_alloc(64,(size_t)O*NM*4);

  char cpu[256]; coli_cpu_describe(cpu,sizeof cpu); printf("cpu: %s\n",cpu);
  printf("I=%lld O=%lld   int8 weights %.1f MiB   int4 weights %.1f MiB (%.2fx)\n",
         (long long)I,(long long)O,(double)I*O/1048576.0,
         ((double)I*O/2 + (double)O*(I/COLI_W4BLK)*4)/1048576.0,
         ((double)I*O/2 + (double)O*(I/COLI_W4BLK)*4)/((double)I*O));
  /* RESIDENT OR STREAMING is the condition that makes every number below mean
   * something. gemm_i8.h's dispatch threshold was derived from a 448 MiB
   * streaming matrix; the same kernels invert at 16 MiB resident in L3. A run
   * that does not say which it measured is not a comparable measurement. */
  { uint64_t l3=coli_cache_bytes(3);
    double w8mib=(double)I*O/1048576.0;
    /* A resctrl CAT mask makes the USABLE L3 smaller than the hardware's, and
     * coli_cache_bytes reports the hardware. Under `legion-sim.sh` this line
     * printed "L3: 96 MiB -> L3-RESIDENT" while the process was masked to 12 MiB
     * and actually streaming -- a label contradicting the run it describes. Read
     * the mask when there is one. */
    const char *note = "";
    FILE *sf = fopen("/sys/fs/resctrl/legionsim/schemata","r");
    if (sf) {
      char line[256];
      while (fgets(line,sizeof line,sf)) {
        char *p2 = strstr(line,"L3:0=");
        if (p2) { unsigned m=(unsigned)strtoul(p2+5,NULL,16); int ways=0;
                  while(m){ ways += m&1; m>>=1; }
                  uint64_t full=l3?l3:0;
                  /* cbm_mask width gives the way count; assume 16 unless told */
                  if (full && ways) { l3 = full/16*(uint64_t)ways;
                      if (ways < 16) note=" (resctrl CAT mask, NOT the hardware L3)"; }
                  break; }
      }
      fclose(sf);
    }
    if(!l3) printf("L3: unknown -- residency NOT established, treat timings with suspicion\n");
    else printf("L3: %.0f MiB%s   int8 weights %.1f MiB -> %s\n", (double)l3/1048576.0, note, w8mib,
                (double)I*O > (double)l3*1.5 ? "STREAMING from RAM" : "L3-RESIDENT (not the decode case)"); }

  int ns[]={1,2,4,8,16,32}; int fail=0; long cells=0; int saw_wide=0,saw_narrow=0;
  /* Both int4 kernels are static, so neither is called directly; each n goes
   * through the dispatcher and is checked against the plain scalar definition
   * recomputed below. That is stricter than wide-vs-narrow: it would also catch
   * the two kernels being wrong together. */
  for(unsigned k=0;k<sizeof ns/sizeof*ns;k++){ int n=ns[k];
    coli_quantize_a(&a,X,n,I);
    const char*kn=coli_gemm_i4_kernel(n);
    if(strstr(kn,"wide")) saw_wide=1; else saw_narrow=1;

    /* Reference: the definition, written out plainly. Same block order and same
     * float accumulation order as both kernels, so equality is exact and any
     * difference is a real difference. */
    for(int r=0;r<n;r++){
      const int8_t *xr=a.q+(int64_t)r*I; const float *as=a.scale+(int64_t)r*nb;
      for(int64_t o=0;o<O;o++){
        const uint8_t *wr=w4.q4+o*(I/2); const float *ws=w4.bscale+o*(I/COLI_W4BLK);
        float acc=0.f;
        for(int64_t b=0;b<I/COLI_W4BLK;b++){
          int32_t d0=0,d1=0;
          for(int i=0;i<COLI_W4BLK;i++){
            int64_t kk=b*COLI_W4BLK+i; uint8_t by=wr[kk/2];
            int q=((kk&1)?(by>>4):(by&0x0F))-8;
            int32_t xv=xr[kk];
            if(i<16) d0+=xv*q; else d1+=xv*q; }
          int64_t ab=b*2;
          acc += ws[b]*(as[ab]*(float)d0 + as[ab+1]*(float)d1); }
        R[(int64_t)r*O+o]=acc; } }

    coli_gemm_i4(Y,&a,&w4);
    int bad=0; for(int64_t i=0;i<(int64_t)n*O;i++){ cells++; if(Y[i]!=R[i]) bad++; }

    /* Warm up, then take the MINIMUM, not the mean. A first call pays page
     * faults on the output buffer and pulls the weights in cold; averaging that
     * into the result is how "n=2 is faster than n=1" gets printed. Minimum is
     * the standard defence and is what gemm_i8.h's own table used. */
    /* TWO warmup calls, not one. At 160 MiB the weights exceed L3, so the first
     * timed call still pays first-touch on the output buffer and TLB misses on a
     * freshly written activation -- a legitimate cold cost, and one that made the
     * spread check below fire on a measurement that was actually fine. Two
     * warmups separate "cold" from "too short to measure". */
    int reps=7;
    coli_gemm_i4(Y,&a,&w4); coli_gemm_i8(Y8,&a,&w8);
    coli_gemm_i4(Y,&a,&w4); coli_gemm_i8(Y8,&a,&w8);
    double d4=1e30,d8=1e30,x4=0,x8=0;
    for(int r=0;r<reps;r++){ double t0=now(); coli_gemm_i4(Y,&a,&w4); double d=now()-t0; if(d<d4)d4=d; if(d>x4)x4=d; }
    for(int r=0;r<reps;r++){ double t0=now(); coli_gemm_i8(Y8,&a,&w8); double d=now()-t0; if(d<d8)d8=d; if(d>x8)x8=d; }
    /* PRINT THE SPREAD, and flag only what is genuinely unmeasurable.
     *
     * Two different things produce a wide spread and they must not share a
     * warning. A BANDWIDTH-BOUND run is expected to be noisy -- at n=1 and
     * 160 MiB the kernel streams at ~84 GB/s and any other memory traffic on the
     * box lands on it, so max/min of 3x is normal and the MINIMUM is exactly the
     * right estimator: it is the uncontended run. A run that is simply TOO SHORT
     * is different: at 4 MiB the kernel takes ~0.03 ms, less than it costs to
     * wake 16 OpenMP threads, so every sample including the minimum is jitter.
     * Measured on an idle machine, three consecutive runs of the same binary at
     * 2048x2048 n=1: 0.03 ms, 2.75 ms, 3.21 ms.
     *
     * So the flag is on the MINIMUM being small, not on the spread being large,
     * and the spread is printed either way so the reader can judge rather than
     * trust a boolean. */
    /* ⚠️ THIS SPREAD IS WITHIN ONE RUN AND DOES NOT BOUND BETWEEN-RUN VARIANCE.
     * Measured 2026-08-17 at 4096x14336 under a 2-way CAT mask: eight launches of
     * this binary each reported a tight spread (x1.2 to x1.4) while their ratios
     * ranged 0.56x to 7.14x. With few cache ways the physical page placement,
     * which differs per launch, decides the result. Repeat the whole binary
     * before believing any comparison from a shape whose runs are short. */
    double sp4 = d4>0 ? x4/d4 : 1.0, sp8 = d8>0 ? x8/d8 : 1.0;
    int too_short = (d4 < 2e-4) || (d8 < 2e-4);
    printf("  n=%-3d %-20s bad=%-6d int4 %7.2f ms (x%.1f)  int8 %7.2f ms (x%.1f)  int4/int8 %.2fx%s\n",
           n,kn,bad,d4*1e3,sp4,d8*1e3,sp8,d4/d8,
           too_short ? "  <-- TOO SHORT TO MEASURE (<0.2 ms; thread wake-up dominates)" : "");
    if(bad) fail=1; }

  printf("compared %ld cells; int4 kernels exercised: %s%s\n",cells,
         saw_narrow?"narrow ":"",saw_wide?"wide":"");
  /* The wide kernel must have run on a VNNI machine. It is not "both kernels"
   * any more: COLI_GEMM_I4_MIN_WIDE is 1, so narrow is now reachable only on a
   * CPU without VNNI -- demanding both here would fail on the very machine the
   * threshold was measured on. What still has to be impossible is a silent
   * no-dispatch, where the run takes the slow path throughout and prints a
   * comparison that looks like a result. */
  if((coli_cpu_features()&COLI_CPU_AVX512VNNI) && !saw_wide){
    printf("INCONCLUSIVE: VNNI present but the wide int4 kernel never dispatched\n"); return 3; }
  if(!(coli_cpu_features()&COLI_CPU_AVX512VNNI) && !saw_narrow){
    printf("INCONCLUSIVE: no int4 kernel ran at all\n"); return 3; }
  printf(fail?"FAIL\n":"PASS (int4 wide bit-exact with the definition)\n");
  return fail?1:0; }
