/* bench_gemm_tile — timing sweep for the packed-panel int4 kernel
 * (gemm_i4_panel / "avx512vnni-i4-panel" in gemm_i8.cpp -- name kept from the
 * first, 2x4 register-tile attempt this replaced) against the per-row wide
 * kernel it sits next to, forced via COLI_I4_TILE=0 so both arms run through
 * the SAME dispatcher and SAME weight/activation buffers -- a differential
 * across two binaries could not rule out a compiler-flag difference doing
 * the work instead of the kernel.
 *
 * Correctness is NOT this file's job -- test_gemm_i4.c's run_panel_check()
 * already proves bit-identity, including a control that can fail
 * (test_gemm_i4_panel_broken). This file exists only to print us/call and
 * effective GB/s of int4 weight bytes streamed, for shapes and n values
 * matching gemm_i8.h's own measurement tables.
 *
 * ⚠️ Per the task this was written for: do not run this at more than 4
 * threads, and do not treat its numbers as a result -- the lead re-measures
 * on a controlled box. Label every number "agent-measured, N threads,
 * uncontrolled box" when reporting it. */
#include "../src/platform.h"
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* Minimum of repeated calls until a 100 ms wall-time floor, same discipline
 * as test_gemm_i4.c and test_gemm_i8.c's own timing loops (this box runs a
 * HUD daemon that holds a core long enough to make short fixed-count
 * measurements slow together). */
static double bench_call(void (*call)(float*, const coli_a_i8*, const coli_w_i4*),
                          float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
  call(y,a,w); call(y,a,w);   /* warmup: first-touch + cold weights */
  double dmin=1e30, sp=0; int reps=0;
  while (sp < 0.100 || reps < 5) {
    double t0=now(); call(y,a,w); double d=now()-t0;
    if (d<dmin) dmin=d; sp+=d; reps++; if (reps>2000) break;
  }
  return dmin;
}

static void call_tile(float *y, const coli_a_i8 *a, const coli_w_i4 *w){
  setenv("COLI_I4_TILE","1",1); coli_gemm_i4(y,(coli_a_i8*)a,(coli_w_i4*)w);
}
static void call_ref(float *y, const coli_a_i8 *a, const coli_w_i4 *w){
  setenv("COLI_I4_TILE","0",1); coli_gemm_i4(y,(coli_a_i8*)a,(coli_w_i4*)w);
}

int main(void){
  int nthreads = 4;
#ifdef _OPENMP
  /* Hard-capped here regardless of the environment: the task this kernel was
   * built for is explicit that the agent must not run this sweep above 4
   * threads. omp_set_num_threads, not a hope that OMP_NUM_THREADS was set. */
  omp_set_num_threads(nthreads);
  nthreads = omp_get_max_threads();
#endif
  char cpu[256]; coli_cpu_describe(cpu,sizeof cpu);
  printf("cpu: %s\n", cpu);
  printf("threads: %d (hard-capped by this bench; see file header)\n", nthreads);

  typedef struct { int64_t I,O; } shape_t;
  shape_t shapes[] = { {2048,768}, {768,2048}, {2048,2048}, {4096,14336} };
  int ns[] = {1,4,16,64,682};

  for (unsigned si=0; si<sizeof shapes/sizeof*shapes; si++){
    int64_t I=shapes[si].I, O=shapes[si].O, nb=I/COLI_ABLK;
    float *F=(float*)coli_aligned_alloc(64,(size_t)I*O*4);
    srand(777+(int)si);
    for(int64_t i=0;i<I*O;i++){ float u=(float)(rand()%20001-10000)/10000.f; F[i]=u*u*u*0.1f; }
    coli_w_i4 w4; coli_quantize_w4(&w4,F,I,O);
    double w4mib = ((double)I*O/2 + (double)O*(I/COLI_W4BLK)*4)/1048576.0;

    int NM=682;
    coli_a_i8 a={0}; a.I=I;
    a.q=(int8_t*)coli_aligned_alloc(64,(size_t)I*NM);
    a.scale=(float*)coli_aligned_alloc(64,(size_t)nb*NM*4);
    a.sum=(int32_t*)coli_aligned_alloc(64,(size_t)nb*NM*4);
    float*X=(float*)coli_aligned_alloc(64,(size_t)I*NM*4);
    for(int64_t i=0;i<I*NM;i++) X[i]=(float)((rand()%2001)-1000)/500.0f;
    float*Y=(float*)coli_aligned_alloc(64,(size_t)O*NM*4);

    printf("-- I=%lld O=%lld  int4 weights %.1f MiB --\n",(long long)I,(long long)O,w4mib);
    for (unsigned ni=0; ni<sizeof ns/sizeof*ns; ni++){
      int n=ns[ni];
      coli_quantize_a(&a,X,n,I);
      double dt = bench_call(call_tile, Y, &a, &w4);
      double dr = bench_call(call_ref,  Y, &a, &w4);
      setenv("COLI_I4_TILE","1",1);   /* leave env in its default state */
      double bytes_per_call = (double)I*O/2;   /* int4 weight bytes read per GEMM call */
      double gbps_t = bytes_per_call/dt/1e9, gbps_r = bytes_per_call/dr/1e9;
      printf("  n=%-4d tile %8.1f us (%6.2f GB/s)   ref %8.1f us (%6.2f GB/s)   tile/ref %.2fx\n",
             n, dt*1e6, gbps_t, dr*1e6, gbps_r, dr>0?dt/dr:0.0);
    }
    coli_aligned_free(F); coli_free_w4(&w4);
    coli_aligned_free(a.q); coli_aligned_free(a.scale); coli_aligned_free(a.sum);
    coli_aligned_free(X); coli_aligned_free(Y);
  }
  printf("agent-measured, %d threads, uncontrolled box -- not a lead-quality result, see file header\n", nthreads);
  return 0;
}
