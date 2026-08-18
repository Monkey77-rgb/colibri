#include "../src/platform.h"
/* Every dispatched kernel must equal coli_gemm_i8_ref EXACTLY, and the test
 * must be able to fail: BREAK=1 perturbs the wide kernel's correction term. */
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
void coli_gemm_i8_ref(float*,const coli_a_i8*,const coli_w_i8*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  int64_t I=argc>1?atoll(argv[1]):2048, O=argc>2?atoll(argv[2]):2048;
  int64_t nb=I/COLI_ABLK; int NM=64;
  coli_w_i8 w={0}; w.I=I; w.O=O;
  w.qu=(uint8_t*)coli_aligned_alloc(64,(size_t)I*O); w.scale=(float*)coli_aligned_alloc(64,(size_t)O*4);
  coli_a_i8 a={0}; a.I=I;
  a.q=(int8_t*)coli_aligned_alloc(64,(size_t)I*NM); a.scale=(float*)coli_aligned_alloc(64,(size_t)nb*NM*4);
  a.sum=(int32_t*)coli_aligned_alloc(64,(size_t)nb*NM*4);
  float*X=(float*)coli_aligned_alloc(64,(size_t)I*NM*4);
  float*Y=(float*)coli_aligned_alloc(64,(size_t)O*NM*4),*R=(float*)coli_aligned_alloc(64,(size_t)O*NM*4);
  srand(99);
  for(int64_t i=0;i<I*O;i++) w.qu[i]=(uint8_t)(rand()&0xFF);
  for(int64_t o=0;o<O;o++) w.scale[o]=0.001f+(float)(rand()%100)/1e5f;
  for(int64_t i=0;i<I*NM;i++) X[i]=(float)((rand()%2001)-1000)/500.0f;
  char cpu[256]; coli_cpu_describe(cpu,sizeof cpu); printf("cpu: %s\n",cpu);
  int ns[]={1,2,3,4,8,16,32,64}; int fail=0; long cells=0; int saw_wide=0,saw_narrow=0;
  for(unsigned k=0;k<sizeof ns/sizeof*ns;k++){ int n=ns[k];
    coli_quantize_a(&a,X,n,I);
    const char*kn=coli_gemm_i8_kernel(n,I,O);
    if(strstr(kn,"wide")) saw_wide=1; else saw_narrow=1;
    coli_gemm_i8_ref(R,&a,&w);
    coli_gemm_i8(Y,&a,&w);
    int bad=0; for(int64_t i=0;i<(int64_t)n*O;i++){ cells++; if(Y[i]!=R[i]) bad++; }
    /* MIN and SPREAD, not a mean of 5. A mean folds an outlier straight into the
     * figure, and the spread is the only thing that says whether the figure can
     * be compared to another one. Caught by this test contradicting itself: two
     * builds running the SAME wide kernel at n=4 reported 2.52 ms and 5.36 ms,
     * which is not a difference between builds, it is the noise floor. */
    coli_gemm_i8(Y,&a,&w); coli_gemm_i8(Y,&a,&w);          /* two warmups */
    double dt=1e30, mx=0;
    for(int r=0;r<7;r++){ double t0=now(); coli_gemm_i8(Y,&a,&w); double d=now()-t0;
                          if(d<dt)dt=d; if(d>mx)mx=d; }
    printf("  n=%-3d %-20s bad=%-6d %7.2f ms (x%.1f) %6.1f GB/s%s\n",
           n,kn,bad,dt*1e3,(dt>0?mx/dt:1.0),(double)I*O/dt/1e9,
           dt < 2e-4 ? "  <-- TOO SHORT TO MEASURE" : "");
    if(bad) fail=1; }
  printf("compared %ld cells; kernels exercised: %s%s\n",cells,
         saw_narrow?"narrow ":"",saw_wide?"wide":"");
  if(!saw_wide||!saw_narrow){ printf("INCONCLUSIVE: only one kernel ran, comparison proves nothing\n"); return 3; }
  printf(fail?"FAIL\n":"PASS (all dispatched kernels bit-exact vs reference)\n");
  return fail?1:0; }
