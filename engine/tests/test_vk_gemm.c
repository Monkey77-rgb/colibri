/* GPU vs CPU. NOT bit-exactness: the shader reduces through shared memory in a
 * tree, the CPU accumulates sequentially, so the two round differently. Every
 * other kernel here is held to equality, so this exception gets an explicit
 * bound AND a control that must exceed it -- otherwise "within tolerance" is
 * just a number nobody can fail. */
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
void coli_gemm_i8_ref(float*,const coli_a_i8*,const coli_w_i8*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

#define TOL 2e-5   /* relative to max|y|; float32 tree-vs-sequential over I terms */

int main(int argc,char**argv){
  int64_t I=argc>1?atoll(argv[1]):2048, O=argc>2?atoll(argv[2]):2048;
  /* MAXN was hardcoded to 8, so every GPU GEMM figure in this repo was measured
   * in the DECODE regime. Prefill runs n = prompt length -- 682 on the frozen
   * NLL prompt -- and that is the regime where the gap against llama.cpp is 41x
   * rather than 3.4x. A harness that cannot reach the regime under suspicion
   * cannot exonerate it. Default stays 8 so old invocations are unchanged. */
  int MAXN = argc>3?atoi(argv[3]):8;
  if (MAXN < 1) MAXN = 1;
  char err[256];
  coli_vk *v=coli_vk_init("shaders/gemm_i8.spv",err,sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  printf("device: %s  (%s)\n",coli_vk_device_name(v), coli_vk_is_integrated(v)?"integrated, unified memory":"DISCRETE");

  /* NM sizes every n-indexed buffer below and was pinned at 8 alongside the old
   * hardcoded loop bound. Raising MAXN without raising this segfaults -- it did,
   * exit 139, which is how this line was found. Tie them together so the two
   * cannot drift apart again. */
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

  int h=coli_vk_upload_w(v,&w);
  if(h<0){ printf("FAIL: weight upload\n"); return 1; }
  printf("weight memory: %s -> %s\n", coli_vk_mem_desc(v), coli_vk_weight_mem(v));
  /* The warning must reflect where the weights ACTUALLY are, not where they used
   * to be. It once said "host-visible -> PCIe" unconditionally; after the
   * device-local path landed it was printing that above numbers proving the
   * opposite, which is worse than no warning at all. */
  if(!coli_vk_is_integrated(v)){
    if(strstr(coli_vk_weight_mem(v),"DEVICE_LOCAL"))
      printf("  NOTE: discrete GPU, weights staged into VRAM. The intended target is an\n"
             "        integrated GPU with unified memory, where no staging happens.\n");
    else
      printf("  NOTE: discrete GPU and weights are HOST_VISIBLE -> every read crosses PCIe.\n"
             "        Timings below are NOT representative of the intended target.\n");
  }

  int fail=0;
  for(int n=1;n<=MAXN;n*=2){
    coli_quantize_a(&a,X,n,I);
    coli_gemm_i8_ref(Yc,&a,&w);
    if(coli_vk_gemm(v,h,&a,Yg)!=0){ printf("FAIL: dispatch n=%d\n",n); return 1; }
    double ymax=0,maxd=0; for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ymax)ymax=c; }
    for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>maxd)maxd=d; }
    double rel=maxd/(ymax>0?ymax:1);
    /* MIN of several, not one dispatch. A single GPU submit carries queue and
     * fence latency that varies run to run; one sample cannot tell a 5% kernel
     * difference from scheduling noise, and every number below is compared
     * against another number. */
    double gt=1e30, ct=1e30;
    for(int rep=0;rep<5;rep++){ double t0=now(); coli_vk_gemm(v,h,&a,Yg); double d=now()-t0; if(d<gt)gt=d; }
    for(int rep=0;rep<3;rep++){ double t0=now(); coli_gemm_i8(Yc,&a,&w);  double d=now()-t0; if(d<ct)ct=d; }
    printf("  n=%-2d  rel=%.2e %-4s  gpu %6.2f ms   cpu %6.2f ms\n",
           n,rel,rel<TOL?"ok":"BAD",gt*1e3,ct*1e3);
    if(rel>=TOL) fail=1;
  }
  /* CONTROL: perturb the CPU side by 0.1%. If the comparison cannot see that,
   * the tolerance is too loose to mean anything. */
  coli_quantize_a(&a,X,4,I);
  coli_gemm_i8_ref(Yc,&a,&w); coli_vk_gemm(v,h,&a,Yg);
  double ymax=0; for(int64_t i=0;i<4*O;i++){ double c=fabs((double)Yc[i]); if(c>ymax)ymax=c; }
  for(int64_t i=0;i<4*O;i++) Yc[i]*=1.001f;
  double maxd=0; for(int64_t i=0;i<4*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>maxd)maxd=d; }
  double crel=maxd/ymax;
  printf("  control (cpu x1.001): rel=%.2e -> %s\n",crel, crel>=TOL?"exceeds tolerance, as required":"NOT DETECTED -- tolerance is meaningless");
  if(crel<TOL) fail=1;
  /* ---------------------------------------------------------------- int4 ---
   * Same three obligations as the int8 path above: agree with the CPU inside a
   * stated bound, be timed against it, and carry a control that exceeds the
   * bound. The int4 GPU kernel is checked against the CPU's int4 kernel, not the
   * int8 one -- the two formats do not produce the same answer, so comparing
   * across them would fail for a reason that has nothing to do with the GPU. */
  if(!coli_vk_has_i4(v)){
    printf("\nint4: SKIP -- shaders/gemm_i4.spv absent (build with `make vk`)\n");
  } else {
    float *F=(float*)aligned_alloc(64,(size_t)I*O*4);
    for(int64_t i=0;i<I*O;i++){ float u=(float)(rand()%20001-10000)/10000.f; F[i]=u*u*u*0.1f; }
    coli_w_i4 w4; coli_quantize_w4(&w4,F,I,O);
    int h4=coli_vk_upload_w4(v,&w4);
    if(h4<0){ printf("\nint4: FAIL -- weight upload\n"); fail=1; }
    else {
      printf("\nint4 GPU (weights %.0f MiB vs int8 %.0f MiB)\n",
             ((double)I*O/2+(double)O*(I/COLI_W4BLK)*4)/1048576.0, (double)I*O/1048576.0);
      for(int n=1;n<=MAXN;n*=2){
        coli_quantize_a(&a,X,n,I);
        coli_gemm_i4(Yc,&a,&w4);
        if(coli_vk_gemm4(v,h4,&a,Yg)!=0){ printf("  FAIL: int4 dispatch n=%d\n",n); fail=1; break; }
        double ym=0,md=0;
        for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ym)ym=c; }
        for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md)md=d; }
        double rel=md/(ym>0?ym:1);
        double gt=1e30, ct=1e30;
        for(int rep=0;rep<5;rep++){ double t0=now(); coli_vk_gemm4(v,h4,&a,Yg); double d=now()-t0; if(d<gt)gt=d; }
        for(int rep=0;rep<3;rep++){ double t0=now(); coli_gemm_i4(Yc,&a,&w4);   double d=now()-t0; if(d<ct)ct=d; }
        printf("  n=%-2d  rel=%.2e %-4s  gpu %6.2f ms   cpu %6.2f ms\n",
               n,rel,rel<TOL?"ok":"BAD",gt*1e3,ct*1e3);
        if(rel>=TOL) fail=1;
      }
      /* THE INTEGER-VS-FLOAT QUESTION, measured. Same weights, same buffers,
       * same access pattern -- only where the arithmetic happens differs. See
       * RESEARCH.md 1.2: ggml dequantizes to float because it serves ~20 formats
       * through one templated matmul; we have one, so we kept the nibbles
       * integer. Whether that was worth anything is this table. */
      if(coli_vk_has_i4f(v)){
        printf("  -- same kernel, dequantized to FLOAT (ggml's choice) --\n");
        for(int n=1;n<=MAXN;n*=2){
          coli_quantize_a(&a,X,n,I);
          coli_gemm_i4(Yc,&a,&w4);
          if(coli_vk_gemm4f(v,h4,&a,Yg)!=0){ printf("  FAIL: i4f dispatch n=%d\n",n); fail=1; break; }
          double ym=0,md=0;
          for(int64_t i=0;i<(int64_t)n*O;i++){ double cq=fabs((double)Yc[i]); if(cq>ym)ym=cq; }
          for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md)md=d; }
          double rel=md/(ym>0?ym:1);
          double ft=1e30;
          for(int rep=0;rep<5;rep++){ double t0=now(); coli_vk_gemm4f(v,h4,&a,Yg); double d=now()-t0; if(d<ft)ft=d; }
          printf("  n=%-2d  rel=%.2e %-4s  gpu %6.2f ms\n",n,rel,rel<TOL?"ok":"BAD",ft*1e3);
          if(rel>=TOL) fail=1;
        }
      } else printf("  (float-dequant variant not built)\n");

      coli_quantize_a(&a,X,4,I);
      coli_gemm_i4(Yc,&a,&w4); coli_vk_gemm4(v,h4,&a,Yg);
      double ym4=0; for(int64_t i=0;i<4*O;i++){ double c=fabs((double)Yc[i]); if(c>ym4)ym4=c; }
      for(int64_t i=0;i<4*O;i++) Yc[i]*=1.001f;
      double md4=0; for(int64_t i=0;i<4*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md4)md4=d; }
      double c4=md4/ym4;
      printf("  control (cpu x1.001): rel=%.2e -> %s\n",c4,
             c4>=TOL?"exceeds tolerance, as required":"NOT DETECTED -- tolerance is meaningless");
      if(c4<TOL) fail=1;
    }
    /* ------------------------------------------------------------- FFN ---
     * Does keeping activations on the device actually pay? Same three int4
     * matrices, same input, two ways: three separate GEMM calls (three uploads,
     * three downloads, CPU requantization between) against one fused call that
     * crosses the bus twice in total.
     *
     * The CPU result is the reference for BOTH, so this checks correctness of
     * the fused path and measures the round trips at the same time. */
    if(coli_vk_has_ffn(v) && h4>=0){
      int64_t D=I, EI=1024;                     /* small EI keeps the test quick */
      float *Fg=(float*)aligned_alloc(64,(size_t)D*EI*4);
      float *Fu=(float*)aligned_alloc(64,(size_t)D*EI*4);
      float *Fd=(float*)aligned_alloc(64,(size_t)EI*D*4);
      for(int64_t i=0;i<D*EI;i++){ float t=(float)(rand()%2001-1000)/10000.f; Fg[i]=t; Fu[i]=t*0.9f; }
      for(int64_t i=0;i<EI*D;i++) Fd[i]=(float)(rand()%2001-1000)/10000.f;
      coli_w_i4 wg,wu,wd;
      coli_quantize_w4(&wg,Fg,D,EI); coli_quantize_w4(&wu,Fu,D,EI); coli_quantize_w4(&wd,Fd,EI,D);
      int hg=coli_vk_upload_w4(v,&wg), hu=coli_vk_upload_w4(v,&wu), hd=coli_vk_upload_w4(v,&wd);
      if(hg<0||hu<0||hd<0){ printf("\nFFN: FAIL upload\n"); fail=1; }
      else {
        printf("\nFFN on device (D=%lld EI=%lld): one upload + one download vs three of each\n",
               (long long)D,(long long)EI);
        int64_t nbE=EI/COLI_ABLK;
        coli_a_i8 ha={0}; ha.I=EI;
        ha.q=(int8_t*)aligned_alloc(64,(size_t)EI*NM);
        ha.scale=(float*)aligned_alloc(64,(size_t)nbE*NM*4);
        ha.sum=(int32_t*)aligned_alloc(64,(size_t)nbE*NM*4);
        float *G=(float*)aligned_alloc(64,(size_t)EI*NM*4);
        float *U=(float*)aligned_alloc(64,(size_t)EI*NM*4);
        float *H=(float*)aligned_alloc(64,(size_t)EI*NM*4);
        float *Yr=(float*)aligned_alloc(64,(size_t)D*NM*4);
        float *Yf=(float*)aligned_alloc(64,(size_t)D*NM*4);
        for(int n=1;n<=MAXN;n*=2){
          coli_quantize_a(&a,X,n,D);
          /* CPU reference: the same FFN, entirely on the CPU. */
          coli_gemm_i4(G,&a,&wg); coli_gemm_i4(U,&a,&wu);
          for(int64_t i=0;i<(int64_t)n*EI;i++){ float g=G[i]; H[i]=(g/(1.f+expf(-g)))*U[i]; }
          coli_quantize_a(&ha,H,n,EI);
          coli_gemm_i4(Yr,&ha,&wd);
          /* GPU, fused. */
          if(coli_vk_ffn4(v,hg,hu,hd,&a,Yf)!=0){ printf("  FAIL: ffn dispatch n=%d\n",n); fail=1; break; }
          double ym=0,md=0;
          for(int64_t i=0;i<(int64_t)n*D;i++){ double c=fabs((double)Yr[i]); if(c>ym)ym=c; }
          for(int64_t i=0;i<(int64_t)n*D;i++){ double d=fabs((double)Yr[i]-(double)Yf[i]); if(d>md)md=d; }
          double rel=md/(ym>0?ym:1);
          /* FFN tolerance is looser than the GEMM's on purpose: silu uses the
           * GPU's exp(), and the intermediate is REQUANTIZED on each side
           * independently, so a value near a quantization boundary can land on
           * either side of it. That is a real difference, not noise, and it is
           * bounded rather than pretended away. */
          const double FTOL = 5e-3;
          double gt=1e30, st=1e30;
          for(int rep=0;rep<5;rep++){ double t0=now(); coli_vk_ffn4(v,hg,hu,hd,&a,Yf); double d=now()-t0; if(d<gt)gt=d; }
          for(int rep=0;rep<5;rep++){
            double t0=now();
            coli_vk_gemm4(v,hg,&a,G); coli_vk_gemm4(v,hu,&a,U);
            for(int64_t i=0;i<(int64_t)n*EI;i++){ float g=G[i]; H[i]=(g/(1.f+expf(-g)))*U[i]; }
            coli_quantize_a(&ha,H,n,EI);
            coli_vk_gemm4(v,hd,&ha,Yf);
            double d=now()-t0; if(d<st)st=d;
          }
          printf("  n=%-2d  rel=%.2e %-4s  fused %6.2f ms   3-call %6.2f ms   %.2fx\n",
                 n,rel,rel<FTOL?"ok":"BAD",gt*1e3,st*1e3,st/gt);
          if(rel>=FTOL) fail=1;
        }
        free(ha.q); free(ha.scale); free(ha.sum);
        free(G); free(U); free(H); free(Yr); free(Yf);
      }
      coli_free_w4(&wg); coli_free_w4(&wu); coli_free_w4(&wd);
      free(Fg); free(Fu); free(Fd);
    }

    coli_free_w4(&w4); free(F);
  }

  coli_vk_free(v);
  printf(fail?"FAIL\n":"PASS (gpu within %.0e of cpu, and the control exceeds it)\n",TOL);
  return fail;
}
