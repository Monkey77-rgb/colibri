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
  char err[256];
  coli_vk *v=coli_vk_init("shaders/gemm_i8.spv",err,sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  printf("device: %s  (%s)\n",coli_vk_device_name(v), coli_vk_is_integrated(v)?"integrated, unified memory":"DISCRETE");

  int64_t nb=I/COLI_ABLK; int NM=8;
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
  for(int n=1;n<=8;n*=2){
    coli_quantize_a(&a,X,n,I);
    coli_gemm_i8_ref(Yc,&a,&w);
    if(coli_vk_gemm(v,h,&a,Yg)!=0){ printf("FAIL: dispatch n=%d\n",n); return 1; }
    double ymax=0,maxd=0; for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ymax)ymax=c; }
    for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>maxd)maxd=d; }
    double rel=maxd/(ymax>0?ymax:1);
    double t0=now(); coli_vk_gemm(v,h,&a,Yg); double gt=now()-t0;
    t0=now(); coli_gemm_i8(Yc,&a,&w); double ct=now()-t0;
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
      for(int n=1;n<=8;n*=2){
        coli_quantize_a(&a,X,n,I);
        coli_gemm_i4(Yc,&a,&w4);
        if(coli_vk_gemm4(v,h4,&a,Yg)!=0){ printf("  FAIL: int4 dispatch n=%d\n",n); fail=1; break; }
        double ym=0,md=0;
        for(int64_t i=0;i<(int64_t)n*O;i++){ double c=fabs((double)Yc[i]); if(c>ym)ym=c; }
        for(int64_t i=0;i<(int64_t)n*O;i++){ double d=fabs((double)Yc[i]-(double)Yg[i]); if(d>md)md=d; }
        double rel=md/(ym>0?ym:1);
        double t0=now(); coli_vk_gemm4(v,h4,&a,Yg); double gt=now()-t0;
        t0=now(); coli_gemm_i4(Yc,&a,&w4); double ct=now()-t0;
        printf("  n=%-2d  rel=%.2e %-4s  gpu %6.2f ms   cpu %6.2f ms\n",
               n,rel,rel<TOL?"ok":"BAD",gt*1e3,ct*1e3);
        if(rel>=TOL) fail=1;
      }
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
    coli_free_w4(&w4); free(F);
  }

  coli_vk_free(v);
  printf(fail?"FAIL\n":"PASS (gpu within %.0e of cpu, and the control exceeds it)\n",TOL);
  return fail;
}
