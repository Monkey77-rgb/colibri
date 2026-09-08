/* gpu_keepalive -- EXPERIMENT, not a product. Submits one tiny int4 GEMV (1 KB)
 * every `period_us` microseconds so the GPU never idles long enough to drop its
 * clocks. Measured 2026-09-08: the hybrid decode runs at P3 / MEM 5001 MHz
 * (half rate) because it is bursty; PowerMizer cannot be set without root on
 * this box (nvidia-settings reports the assignment and reads back 0). This
 * tests the hypothesis without root: run it beside coli-gpu, sample clocks,
 * compare tok/s. Usage: gpu_keepalive [period_us=200] [seconds=600] [I=64] [O=32] [reps=1].
 * Measured 2026-09-08: 1 KB every 200 us holds P2 but the 5,000 submits/s of a
 * second context cost the engine 24% (41.1 -> 31.4 tok/s) -- hence the heavy
 * mode: a ~1 ms burst every ~10 ms, ~100 submits/s. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
static float frand(unsigned *s){ *s = *s*1664525u+1013904223u; return ((*s>>8)&0xffff)/65536.0f-0.5f; }
int main(int argc,char**argv){
  int period = argc>1?atoi(argv[1]):200; int secs = argc>2?atoi(argv[2]):600;
  int64_t I = argc>3?atoll(argv[3]):64, O = argc>4?atoll(argv[4]):32; int reps = argc>5?atoi(argv[5]):1;   /* heavy mode: bigger shape, reps per submit, longer period */
  char err[256]; coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
  if(!v){ printf("SKIP: %s\n",err); return 0; }
  unsigned seed=7;
  float *W=(float*)malloc((size_t)I*O*4), *X=(float*)malloc((size_t)I*4);
  for(int64_t i=0;i<I*O;i++) W[i]=frand(&seed); for(int64_t i=0;i<I;i++) X[i]=frand(&seed);
  coli_w_i4 w4; coli_quantize_w4(&w4,W,I,O);
  int64_t nb=I/COLI_ABLK; coli_a_i8 a={0}; a.I=I; a.n=1;
  a.q=(int8_t*)aligned_alloc(64,(size_t)I); a.scale=(float*)aligned_alloc(64,(size_t)nb*4); a.sum=(int32_t*)aligned_alloc(64,(size_t)nb*4);
  coli_quantize_a(&a,X,1,I);
  int h=coli_vk_upload_w4(v,&w4); if(h<0){ printf("upload failed\n"); return 1; }
  float *Y=(float*)calloc((size_t)O,4);
  struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0); long n=0; double busy=0;
  for(;;){ double s=coli_vk_bench_gemm4(v,h,&a,Y,reps); if(s>0) busy+=s; n++; usleep(period);
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); double el=(t.tv_sec-t0.tv_sec)+(t.tv_nsec-t0.tv_nsec)*1e-9;
    if(el>=secs){ printf("keepalive: %ld submits in %.1f s, GPU busy %.3f s (%.2f%%), period %d us, shape %ldx%ld x%d\n",n,el,busy,100*busy/el,period,(long)I,(long)O,reps); break; } }
  return 0;
}
