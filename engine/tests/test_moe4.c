/* test_moe4 -- differential oracle for the grouped-expert decode call.
 * 8 random experts (D=2048, EI=768), one random token: coli_vk_moe4 vs a CPU
 * reference (coli_gemm_i4 gate/up, silu*mul, coli_quantize_a, coli_gemm_i4 down)
 * at a loose bound (the GPU quantizes h in silu_mul_q.comp), and prints an
 * exact 64-bit hash of the GPU output so two runs -- COLI_MOE_FUSED=0 and =1 --
 * can be compared bit for bit from the shell. Control that can fail: a second
 * call with two down-projections swapped MUST change the hash. 2026-09-08. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
static float frand(unsigned *s){ *s = *s*1664525u+1013904223u; return ((*s>>8)&0xffff)/65536.0f-0.5f; }
static uint64_t h64(const float *y, size_t n){ uint64_t h=1469598103934665603ull; const unsigned char *b=(const unsigned char*)y; for(size_t i=0;i<n*4;i++){ h^=b[i]; h*=1099511628211ull; } return h; }
int main(void){
  char err[256]; coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  const int NE=8; const int64_t D=2048, EI=768; unsigned seed=11;
  int hg[8],hu[8],hd[8]; coli_w_i4 wg[8],wu[8],wd[8];
  float *W=(float*)malloc((size_t)D*EI*4);
  for(int e=0;e<NE;e++){
    for(int64_t i=0;i<D*EI;i++) W[i]=frand(&seed); coli_quantize_w4(&wg[e],W,D,EI); hg[e]=coli_vk_upload_w4(v,&wg[e]);
    for(int64_t i=0;i<D*EI;i++) W[i]=frand(&seed); coli_quantize_w4(&wu[e],W,D,EI); hu[e]=coli_vk_upload_w4(v,&wu[e]);
    for(int64_t i=0;i<D*EI;i++) W[i]=frand(&seed); coli_quantize_w4(&wd[e],W,EI,D); hd[e]=coli_vk_upload_w4(v,&wd[e]);
    if(hg[e]<0||hu[e]<0||hd[e]<0){ printf("upload failed\n"); return 1; }
  }
  float *X=(float*)malloc((size_t)D*4); for(int64_t i=0;i<D;i++) X[i]=frand(&seed);
  int64_t nbD=D/COLI_ABLK, nbE=EI/COLI_ABLK;
  coli_a_i8 a={0}; a.I=D; a.n=1; a.q=(int8_t*)aligned_alloc(64,(size_t)D); a.scale=(float*)aligned_alloc(64,(size_t)nbD*4); a.sum=(int32_t*)aligned_alloc(64,(size_t)nbD*4);
  coli_quantize_a(&a,X,1,D);
  float *Y=(float*)calloc((size_t)NE*D,4);
  if(coli_vk_moe4(v,hg,hu,hd,NE,&a,Y)!=0){ printf("moe4 declined\n"); return 1; }
  /* CPU reference */
  float *G=(float*)malloc((size_t)EI*4),*U=(float*)malloc((size_t)EI*4),*H=(float*)malloc((size_t)EI*4),*Yc=(float*)malloc((size_t)D*4);
  coli_a_i8 ah={0}; ah.I=EI; ah.n=1; ah.q=(int8_t*)aligned_alloc(64,(size_t)EI); ah.scale=(float*)aligned_alloc(64,(size_t)nbE*4); ah.sum=(int32_t*)aligned_alloc(64,(size_t)nbE*4);
  double worst=0; int bad=0;
  for(int e=0;e<NE;e++){
    coli_gemm_i4(G,&a,&wg[e]); coli_gemm_i4(U,&a,&wu[e]);
    for(int64_t i=0;i<EI;i++){ float g=G[i]; H[i]=g/(1.0f+expf(-g))*U[i]; }
    coli_quantize_a(&ah,H,1,EI); coli_gemm_i4(Yc,&ah,&wd[e]);
    double mx=0,md=0; for(int64_t o=0;o<D;o++){ if(fabs(Yc[o])>mx)mx=fabs(Yc[o]); double d=fabs(Yc[o]-Y[(size_t)e*D+o]); if(d>md)md=d; }
    double rel=md/(mx>0?mx:1); if(rel>worst)worst=rel; if(rel>5e-3) bad=1;
  }
  uint64_t hA=h64(Y,(size_t)NE*D);
  printf("moe4 nexp=%d  gpu-vs-cpu worst rel=%.2e %s  hash=%016llx  fused=%s\n", NE, worst, bad?"BAD":"ok", (unsigned long long)hA, getenv("COLI_MOE_FUSED")?getenv("COLI_MOE_FUSED"):"unset");
  /* control: swap two down-projections; the hash must change */
  int hd2[8]; memcpy(hd2,hd,sizeof hd); hd2[0]=hd[1]; hd2[1]=hd[0];
  float *Y2=(float*)calloc((size_t)NE*D,4);
  if(coli_vk_moe4(v,hg,hu,hd2,NE,&a,Y2)!=0){ printf("moe4 (control) declined\n"); return 1; }
  uint64_t hB=h64(Y2,(size_t)NE*D);
  printf("control (down 0<->1 swapped): hash=%016llx %s\n",(unsigned long long)hB, hB!=hA?"DIFFERS (control ok)":"SAME (control FAILED)");
  /* per-call cost: full submit+fence round trips, what the model pays per layer */
  { int reps=200; struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int r=0;r<reps;r++) coli_vk_moe4(v,hg,hu,hd,NE,&a,Y);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double us=((t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9)/reps*1e6;
    double mb=(double)NE*3*D*EI/2/1e6;
    printf("grouped call: %.1f us/call over %d calls (%.1f MB expert weights, %.0f GB/s incl. submit+fence)\n",us,reps,mb,mb/us*1e3); }
  if (getenv("COLI_VK_PROF")) coli_vk_prof_dump(stdout);
  return bad || hB==hA;
}
