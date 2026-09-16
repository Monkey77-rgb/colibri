/* bench_gemm_n -- batched int4 GEMM (prefill shapes, n activation rows) on the GPU,
 * TWO ways, so the loss can be attributed:
 *   device   : `reps` dispatches back to back in ONE command buffer, one fence
 *              (coli_vk_bench_gemm4) -- the kernel's own rate, no per-call round trip
 *   per-call : coli_vk_gemm4 as model.cpp calls it -- upload activations, dispatch,
 *              submit + fence, download the result -- median of `reps` calls
 * The difference is what the host round trip costs per GEMM. COLI_VK_COOP_MIN_N=0
 * turns the cooperative-matrix (tensor core) kernel off for an A/B on the same
 * binary. Correctness: the last per-call result against coli_gemm_i4.
 * 2026-09-16, for the prefill gap (goss25: dense-GPU prefill GEMM stages 1,292 of
 * 1,680 ms at ~3.7 TFLOPS as timed from the host). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
static float frand(unsigned *s){ *s = *s*1664525u+1013904223u; return ((*s>>8)&0xffff)/65536.0f-0.5f; }
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
int main(int argc,char**argv){
  int n = argc>1?atoi(argv[1]):683, reps = argc>2?atoi(argv[2]):20;
  char err[256]; coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  printf("device: %s  n=%d  reps=%d  COLI_VK_COOP_MIN_N=%s\n", coli_vk_device_name(v), n, reps,
         getenv("COLI_VK_COOP_MIN_N")?getenv("COLI_VK_COOP_MIN_N"):"(default)");
  struct { const char *name; int64_t I,O; } shapes[] = {
    {"8B qkv    4096x6144",  4096, 6144}, {"8B o_proj 4096x4096", 4096, 4096},
    {"8B gate   4096x14336", 4096, 14336}, {"8B down  14336x4096", 14336, 4096},
    {"30B qkv   2048x4096",  2048, 4096}, {"expert gate 2048x768", 2048, 768} };
  unsigned seed=7; int fail=0;
  for (size_t si=0; si<sizeof shapes/sizeof shapes[0]; si++) {
    int64_t I=shapes[si].I, O=shapes[si].O, nb=I/COLI_ABLK;
    float *W=(float*)malloc((size_t)I*O*4), *X=(float*)malloc((size_t)n*I*4);
    for(int64_t i=0;i<I*O;i++) W[i]=frand(&seed); for(int64_t i=0;i<(int64_t)n*I;i++) X[i]=frand(&seed);
    coli_w_i4 w4; coli_quantize_w4(&w4,W,I,O);
    coli_a_i8 a={0}; a.I=I; a.n=n;
    a.q=(int8_t*)aligned_alloc(64,(size_t)n*I); a.scale=(float*)aligned_alloc(64,(size_t)n*nb*4);
    a.sum=(int32_t*)aligned_alloc(64,(size_t)n*nb*4);
    coli_quantize_a(&a,X,n,I);
    int h=coli_vk_upload_w4(v,&w4); if(h<0){ printf("upload failed\n"); return 1; }
    float *Yg=(float*)calloc((size_t)n*O,4), *Yc=(float*)calloc((size_t)n*O,4);
    coli_gemm_i4(Yc,&a,&w4);
    /* device rate: best of 3 batches */
    double best=1e9; for(int k=0;k<3;k++){ double s=coli_vk_bench_gemm4(v,h,&a,Yg,reps); if(s>0&&s<best)best=s; }
    double dev_us = best/reps*1e6;
    /* per-call rate: median of reps */
    double *t=(double*)malloc(sizeof(double)*reps);
    for(int k=0;k<reps;k++){ double t0=now_s(); if(coli_vk_gemm4(v,h,&a,Yg)!=0){ printf("gemm4 failed\n"); return 1; } t[k]=(now_s()-t0)*1e6; }
    qsort(t,reps,sizeof(double),cmpd); double call_us=t[reps/2], call_min=t[0];
    double mx=0,md=0; for(int64_t i=0;i<(int64_t)n*O;i++){ if(fabs(Yc[i])>mx)mx=fabs(Yc[i]); if(fabs(Yc[i]-Yg[i])>md)md=fabs(Yc[i]-Yg[i]); }
    double rel=md/(mx>0?mx:1); int ok=rel<2e-3;   /* fp16-folded coop kernel differs from the int path by design (~1e-3) */
    double flop=2.0*n*I*O;
    printf("  %-22s device %9.1f us (%6.1f TFLOPS)   per-call median %9.1f us min %9.1f (%5.1f TFLOPS)   round-trip %+8.1f us   rel=%.1e %s\n",
           shapes[si].name, dev_us, flop/dev_us/1e6, call_us, call_min, flop/call_us/1e6, call_us-dev_us, rel, ok?"ok":"BAD");
    if(!ok) fail=1;
    free(W);free(X);free(Yg);free(Yc);free(a.q);free(a.scale);free(a.sum);free(t);
  }
  printf(fail?"FAIL\n":"PASS\n"); return fail;
}
