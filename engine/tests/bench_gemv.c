/* bench_gemv -- batch-1 int4 GEMV throughput on the GPU, per shape, WITHOUT the
 * submit+fence floor: `reps` dispatches in one command buffer. Prints us per GEMV
 * and GB/s of int4 weight bytes, plus a correctness check of the last result
 * against coli_gemm_i4 (relative to max|y|). COLI_VK_I4_SPV=<spv> swaps the
 * kernel for an A/B on the same binary. 2026-09-08. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
static float frand(unsigned *s){ *s = *s*1664525u+1013904223u; return ((*s>>8)&0xffff)/65536.0f-0.5f; }
int main(int argc,char**argv){
  int reps = argc>1?atoi(argv[1]):200;
  char err[256]; coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  printf("device: %s  reps=%d  kernel=%s\n", coli_vk_device_name(v), reps, getenv("COLI_VK_I4_SPV")?getenv("COLI_VK_I4_SPV"):"shaders/gemm_i4.spv (default)");
  struct { const char *name; int64_t I,O; } shapes[] = {
    {"expert gate/up  768x2048", 2048, 768}, {"expert down     2048x768", 768, 2048},
    {"qkv-ish        2048x2048", 2048, 2048}, {"o_proj          4096x2048", 4096, 2048},
    {"head          2048x151936", 2048, 151936} };
  unsigned seed=7; int fail=0;
  for (size_t si=0; si<sizeof shapes/sizeof shapes[0]; si++) {
    int64_t I=shapes[si].I, O=shapes[si].O;
    float *W=(float*)malloc((size_t)I*O*4), *X=(float*)malloc((size_t)I*4);
    for(int64_t i=0;i<I*O;i++) W[i]=frand(&seed); for(int64_t i=0;i<I;i++) X[i]=frand(&seed);
    coli_w_i4 w4; coli_quantize_w4(&w4,W,I,O);
    int64_t nb=I/COLI_ABLK;
    coli_a_i8 a={0}; a.I=I; a.n=1;
    a.q=(int8_t*)aligned_alloc(64,(size_t)I); a.scale=(float*)aligned_alloc(64,(size_t)nb*4);
    a.sum=(int32_t*)aligned_alloc(64,(size_t)nb*4);
    coli_quantize_a(&a,X,1,I);
    int h=coli_vk_upload_w4(v,&w4); if(h<0){ printf("upload failed\n"); return 1; }
    float *Yg=(float*)calloc((size_t)O,4), *Yc=(float*)calloc((size_t)O,4);
    coli_gemm_i4(Yc,&a,&w4);
    double best=1e9; for(int k=0;k<3;k++){ double s=coli_vk_bench_gemm4(v,h,&a,Yg,reps); if(s>0&&s<best)best=s; }
    double mx=0,md=0; for(int64_t o=0;o<O;o++){ if(fabs(Yc[o])>mx)mx=fabs(Yc[o]); if(fabs(Yc[o]-Yg[o])>md)md=fabs(Yc[o]-Yg[o]); }
    double rel=md/(mx>0?mx:1); int ok=rel<2e-5;
    double bytes=(double)O*I/2.0; double us=best/reps*1e6;
    printf("  %-26s %8.2f us/gemv  %7.1f GB/s  (%.2f MB int4)  rel=%.1e %s\n", shapes[si].name, us, bytes/(best/reps)/1e9, bytes/1e6, rel, ok?"ok":"BAD");
    if(!ok) fail=1;
    free(W);free(X);free(Yg);free(Yc);free(a.q);free(a.scale);free(a.sum);
  }
  if (getenv("COLI_VK_PROF")) coli_vk_prof_dump(stdout);
  printf(fail?"FAIL\n":"PASS\n"); return fail;
}
