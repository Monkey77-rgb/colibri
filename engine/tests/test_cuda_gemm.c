/* test_cuda_gemm -- mirrors tests/test_vk_gemm.c, against the coli_backend
 * seam instead of vk_backend.h directly (backend_cuda.cu exposes only
 * coli_backend_cuda_open / coli_cuda_probe_class, per backend.h's contract).
 *
 * UNLIKE test_vk_gemm.c, this does not need a bounded-relative-error budget:
 * backend_cuda.cu's header explains why its GEMM/GEMM4 kernels are an EXACT
 * integer recovery of the CPU's offset-to-unsigned weight storage (XOR, not
 * dequantize-and-round) reduced in the SAME block order the CPU reference
 * uses, so int8 and int4 are checked for EXACT float equality against
 * coli_gemm_i8_ref / coli_gemm_i4 -- a strictly stronger claim than Vulkan's,
 * and the printed TOL below is kept only as the threshold the perturbation
 * CONTROL must exceed (same discipline: a comparison that cannot fail proves
 * nothing).
 */
#include "../src/backend.h"
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
void coli_gemm_i8_ref(float*,const coli_a_i8*,const coli_w_i8*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

#define TOL 2e-5   /* the CONTROL (cpu x1.001) must exceed this; the real comparison is checked at 0 */

int main(int argc,char**argv){
  int64_t I=argc>1?atoll(argv[1]):2048, O=argc>2?atoll(argv[2]):2048;
  int MAXN = argc>3?atoi(argv[3]):8;
  if (MAXN < 1) MAXN = 1;
  char err[256];
  coli_backend *be = coli_backend_cuda_open(err, sizeof err);
  if (!be) { printf("SKIP: %s\n", err); return 0; }
  printf("device: %s  mem=%s/%s  dot_used=%d\n", be->device_name(be->ctx),
         be->memdesc(be->ctx), be->memdesc2(be->ctx), be->dot_used(be->ctx));

  int64_t nb=I/COLI_ABLK; int NM=MAXN>8?MAXN:8;
  coli_w_i8 w={0}; w.I=I; w.O=O;
  w.qu=(uint8_t*)aligned_alloc(64,(size_t)I*O); w.scale=(float*)aligned_alloc(64,(size_t)O*4);
  coli_a_i8 a={0}; a.I=I;
  a.q=(int8_t*)aligned_alloc(64,(size_t)I*NM); a.scale=(float*)aligned_alloc(64,(size_t)nb*NM*4);
  a.sum=(int32_t*)aligned_alloc(64,(size_t)nb*NM*4);
  float *X=(float*)aligned_alloc(64,(size_t)I*NM*4);
  float *Yc=(float*)aligned_alloc(64,(size_t)O*NM*4),*Yg=(float*)aligned_alloc(64,(size_t)O*NM*4);
  srand(31);
  for(int64_t i=0;i<I*O;i++) w.qu[i]=(uint8_t)(rand()&0xFF);
  for(int64_t o=0;o<O;o++) w.scale[o]=0.001f+(float)(rand()%100)/1e5f;
  for(int64_t i=0;i<I*NM;i++) X[i]=(float)((rand()%2001)-1000)/500.0f;

  int h = be->upload_w(be->ctx, &w);
  if (h<0) { printf("FAIL: weight upload\n"); return 1; }

  int fail=0;
  for(int n=1;n<=MAXN;n*=2){
    coli_quantize_a(&a,X,n,I);
    coli_gemm_i8_ref(Yc,&a,&w);
    if (be->gemm(be->ctx,h,&a,Yg)!=0) { printf("FAIL: dispatch n=%d\n",n); return 1; }
    double ymax=0,maxd=0; for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ymax)ymax=c; }
    for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>maxd)maxd=d; }
    double rel=maxd/(ymax>0?ymax:1);
    double gt=1e30, ct=1e30;
    for(int rep=0;rep<5;rep++){ double t0=now(); be->gemm(be->ctx,h,&a,Yg); double d=now()-t0; if(d<gt)gt=d; }
    for(int rep=0;rep<3;rep++){ double t0=now(); coli_gemm_i8(Yc,&a,&w);  double d=now()-t0; if(d<ct)ct=d; }
    double gbps = (double)(I*O + (size_t)n*I)/gt/1e9;
    printf("  n=%-2d  rel=%.2e %-16s gpu %7.3f ms (%.1f GB/s weight)  cpu %7.3f ms\n",
           n,rel,rel==0.0?"EXACT":(rel<TOL?"within-tol":"BAD"),gt*1e3,gbps,ct*1e3);
    if(rel>=TOL) fail=1;
  }
  /* CONTROL: perturb the CPU side by 0.1%. */
  coli_quantize_a(&a,X,4,I);
  coli_gemm_i8_ref(Yc,&a,&w); be->gemm(be->ctx,h,&a,Yg);
  double ymax=0; for(int64_t i=0;i<4*O;i++){ double c=fabs((double)Yc[i]); if(c>ymax)ymax=c; }
  for(int64_t i=0;i<4*O;i++) Yc[i]*=1.001f;
  double maxd=0; for(int64_t i=0;i<4*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>maxd)maxd=d; }
  double crel=maxd/ymax;
  printf("  control (cpu x1.001): rel=%.2e -> %s\n",crel, crel>=TOL?"exceeds tolerance, as required":"NOT DETECTED -- tolerance is meaningless");
  if(crel<TOL) fail=1;

  /* ----------------------------------------------------------------- int4 */
  if (!be->has_i4(be->ctx)) {
    printf("\nint4: SKIP -- has_i4() false\n");
  } else {
    float *F=(float*)aligned_alloc(64,(size_t)I*O*4);
    for(int64_t i=0;i<I*O;i++){ float u=(float)(rand()%20001-10000)/10000.f; F[i]=u*u*u*0.1f; }
    coli_w_i4 w4; coli_quantize_w4(&w4,F,I,O);
    int h4 = be->upload_w4(be->ctx, &w4);
    if (h4<0) { printf("\nint4: FAIL -- weight upload\n"); fail=1; }
    else {
      printf("\nint4 GPU (weights %.0f MiB vs int8 %.0f MiB)\n",
             ((double)I*O/2+(double)O*(I/COLI_W4BLK)*4)/1048576.0, (double)I*O/1048576.0);
      for(int n=1;n<=MAXN;n*=2){
        coli_quantize_a(&a,X,n,I);
        coli_gemm_i4(Yc,&a,&w4);
        if (be->gemm4(be->ctx,h4,&a,Yg)!=0) { printf("  FAIL: int4 dispatch n=%d\n",n); fail=1; break; }
        double ym=0,md=0;
        for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ym)ym=c; }
        for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md)md=d; }
        double rel=md/(ym>0?ym:1);
        double gt=1e30, ct=1e30;
        for(int rep=0;rep<5;rep++){ double t0=now(); be->gemm4(be->ctx,h4,&a,Yg); double d=now()-t0; if(d<gt)gt=d; }
        for(int rep=0;rep<3;rep++){ double t0=now(); coli_gemm_i4(Yc,&a,&w4);   double d=now()-t0; if(d<ct)ct=d; }
        double gbps = (double)(I*O/2 + (size_t)n*I)/gt/1e9;
        printf("  n=%-2d  rel=%.2e %-16s gpu %7.3f ms (%.1f GB/s weight)  cpu %7.3f ms\n",
               n,rel,rel==0.0?"EXACT":(rel<TOL?"within-tol":"BAD"),gt*1e3,gbps,ct*1e3);
        if(rel>=TOL) fail=1;
        if (n==1 || n==4) printf("  [throughput line] gemm4 2880x2880-shape call at this I/O, n=%d: %.3f us/call, %.2f GB/s\n", n, gt*1e6, gbps);
      }
      coli_quantize_a(&a,X,4,I);
      coli_gemm_i4(Yc,&a,&w4); be->gemm4(be->ctx,h4,&a,Yg);
      double ym4=0; for(int64_t i=0;i<4*O;i++){ double c=fabs((double)Yc[i]); if(c>ym4)ym4=c; }
      for(int64_t i=0;i<4*O;i++) Yc[i]*=1.001f;
      double md4=0; for(int64_t i=0;i<4*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md4)md4=d; }
      double c4=md4/ym4;
      printf("  control (cpu x1.001): rel=%.2e -> %s\n",c4,
             c4>=TOL?"exceeds tolerance, as required":"NOT DETECTED -- tolerance is meaningless");
      if(c4<TOL) fail=1;
    }
    coli_free_w4(&w4); free(F);
  }

  /* ------------------------------------------------------- the throughput line the report wants:
   * gemm4 2880x2880 at n=1 and n=4, independent of whatever I/O argv passed. */
  {
    int64_t D=2880,O2=2880;
    float *F2=(float*)aligned_alloc(64,(size_t)D*O2*4);
    for(int64_t i=0;i<D*O2;i++) F2[i]=(float)(rand()%2001-1000)/10000.f;
    coli_w_i4 w4b; coli_quantize_w4(&w4b,F2,D,O2);
    int hb = be->upload_w4(be->ctx,&w4b);
    if (hb>=0) {
      coli_a_i8 ab={0}; ab.I=D;
      int64_t nbD=D/COLI_ABLK;
      ab.q=(int8_t*)aligned_alloc(64,(size_t)D*4); ab.scale=(float*)aligned_alloc(64,(size_t)nbD*4*4); ab.sum=(int32_t*)aligned_alloc(64,(size_t)nbD*4*4);
      float *Xb=(float*)aligned_alloc(64,(size_t)D*4*4), *Yb=(float*)aligned_alloc(64,(size_t)O2*4*4);
      for(int64_t i=0;i<D*4;i++) Xb[i]=(float)(rand()%2001-1000)/500.f;
      printf("\n-- gemm4 2880x2880 throughput (report requirement) --\n");
      for (int n=1; n<=4; n+=3) {
        coli_quantize_a(&ab,Xb,n,D);
        double gt=1e30; for(int rep=0;rep<7;rep++){ double t0=now(); be->gemm4(be->ctx,hb,&ab,Yb); double d=now()-t0; if(d<gt)gt=d; }
        double gbps=(double)(D*O2/2)/gt/1e9;
        printf("  gemm4 D=%lld O=%lld n=%d: %.3f us/call, %.2f GB/s (weight bytes only)\n",(long long)D,(long long)O2,n,gt*1e6,gbps);
      }
      free(ab.q);free(ab.scale);free(ab.sum);free(Xb);free(Yb);
    }
    coli_free_w4(&w4b); free(F2);
  }

  coli_backend_close(be);
  printf(fail?"FAIL\n":"PASS\n");
  return fail;
}
