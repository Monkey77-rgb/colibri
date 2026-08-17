/* Continuous batching: correctness FIRST, then throughput.
 *
 * The claim being tested is that batching changes only the SHAPE of the work,
 * never the result. So k sequences decoded together must emit exactly the same
 * tokens as the same k sequences decoded one at a time. If that fails, the
 * throughput number is worthless.
 *
 * Distinct prompts per slot on purpose: identical prompts would still pass if
 * the code accidentally shared one KV region between slots, which is precisely
 * the bug this must catch. */
#include "../src/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

#define MAXS 8
#define GEN  16

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s <model.gguf>\n",argv[0]);return 2;}
  char err[256];
  coli_model *m=coli_load(argv[1],1024,MAXS,1,err,sizeof err);
  if(!m){fprintf(stderr,"load: %s\n",err);return 1;}
  int V=m->cfg.vocab;

  const char *prompts[MAXS]={
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "In computer science, a binary tree is",
    "The chemical symbol for gold is",
    "A prime number is defined as",
    "The speed of light in a vacuum is",
    "Photosynthesis is the process by which"};

  static int ids[MAXS][512]; int nid[MAXS];
  for(int i=0;i<MAXS;i++) nid[i]=coli_encode(m,prompts[i],ids[i],512);

  static int ref[MAXS][GEN];
  float *lg=(float*)malloc((size_t)MAXS*V*sizeof(float));
  coli_sampler sp; coli_sampler_default(&sp);   /* greedy => deterministic */

  /* reference: one sequence at a time, batch size 1 */
  for(int i=0;i<MAXS;i++){
    float *p=coli_prefill_slot(m,0,ids[i],nid[i]);
    if(!p){printf("FAIL prefill\n");return 1;}
    int cur=coli_sample(&sp,p,V,NULL,0); free(p);
    for(int t=0;t<GEN;t++){
      ref[i][t]=cur;
      coli_seq s={0,nid[i]+t,cur};
      if(coli_decode_batch(m,&s,1,lg)!=0){printf("FAIL decode\n");return 1;}
      cur=coli_sample(&sp,lg,V,NULL,0);
    }
  }
  printf("reference (batch=1) captured for %d sequences\n",MAXS);

  int fail=0;
  for(int B=2;B<=MAXS;B*=2){
    /* prefill B sequences, each into its OWN slot */
    static int cur[MAXS]; static int pos[MAXS];
    for(int i=0;i<B;i++){
      float *p=coli_prefill_slot(m,i,ids[i],nid[i]);
      if(!p){printf("FAIL prefill B=%d\n",B);return 1;}
      cur[i]=coli_sample(&sp,p,V,NULL,0); free(p); pos[i]=nid[i];
    }
    int bad=0; double t0=now();
    for(int t=0;t<GEN;t++){
      coli_seq sq[MAXS];
      for(int i=0;i<B;i++){ sq[i].slot=i; sq[i].pos=pos[i]; sq[i].token=cur[i];
        if(cur[i]!=ref[i][t]) bad++; }
      if(coli_decode_batch(m,sq,B,lg)!=0){printf("FAIL batch decode\n");return 1;}
      for(int i=0;i<B;i++){ cur[i]=coli_sample(&sp,lg+(int64_t)i*V,V,NULL,0); pos[i]++; }
    }
    double dt=now()-t0;
    printf("  B=%-2d  mismatches=%-4d  %6.2f s  %6.1f tok/s total  (%.1f per seq)  %s\n",
      B,bad,dt,(double)B*GEN/dt,(double)GEN/dt, bad?"FAIL":"ok");
    if(bad) fail=1;
  }
  /* baseline throughput at B=1 for the comparison */
  { int c0=ref[0][0]; double t0=now();
    for(int t=0;t<GEN;t++){ coli_seq s={0,nid[0]+t,c0}; coli_decode_batch(m,&s,1,lg);
      c0=coli_sample(&sp,lg,V,NULL,0); }
    double dt=now()-t0;
    printf("  B=1   (baseline)                %6.2f s  %6.1f tok/s total\n",dt,(double)GEN/dt); }

  coli_free(m); free(lg);
  printf(fail?"FAIL\n":"PASS (batched output identical to sequential at every batch size)\n");
  return fail;
}
