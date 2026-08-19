/* Prefix cache: CORRECTNESS FIRST, then the saving.
 *
 * The claim is that reusing cached K/V for a shared leading prefix changes only
 * how much work runs, never the result. So the logits from a cache HIT must
 * equal the logits from the same prefill with the cache disabled -- bit for bit,
 * because nothing about the arithmetic is supposed to differ. Any drift means
 * the cached entries did not actually correspond to the positions they were
 * reused at, and the throughput number would be worthless.
 *
 * The CLI cannot exercise this at all: one prompt per process means cache_len is
 * always 0 and the path never runs. That is exactly why this test exists rather
 * than a command-line A/B. An optimisation whose only test cannot reach it is
 * untested. */
#include "../src/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* Two prompts sharing a long leading prefix and diverging at the end -- the
 * shape a templated judge or a chat system prompt actually produces. */
#define PRE  "You are a strict evaluator. Score the response on accuracy, then on clarity, then give a verdict. Respond only in JSON with keys score and verdict. Here is the response to evaluate: "
static int maxdiff_idx(const float *a, const float *b, int n, double *maxabs) {
    int idx=-1; double mx=0;
    for (int i=0;i<n;i++){ double d=fabs((double)a[i]-(double)b[i]); if(d>mx){mx=d;idx=i;} }
    *maxabs=mx; return idx;
}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s model.gguf\n",argv[0]);return 2;}
  char err[512]={0};
  coli_model *m=coli_load(argv[1],2048,2,1,2,err,sizeof err);
  if(!m){fprintf(stderr,"load failed: %s\n",err);return 1;}

  const char *p1 = PRE "The capital of France is Paris.";
  const char *p2 = PRE "The capital of Japan is Tokyo, a very large city indeed.";
  int idsA[1024], idsB[1024];
  int nA = coli_encode(m, p1, idsA, 1024);
  int nB = coli_encode(m, p2, idsB, 1024);
  if (nA<8 || nB<8) { fprintf(stderr,"tokenizer returned %d/%d\n",nA,nB); return 1; }
  int shared=0; while(shared<nA&&shared<nB&&idsA[shared]==idsB[shared]) shared++;
  printf("prompt A %d tok, B %d tok, shared prefix %d tok\n", nA, nB, shared);
  if (shared < 4) { fprintf(stderr,"FAIL: prompts do not share a prefix; test is vacuous\n"); return 1; }

  int V = m->cfg.vocab;
  float *ref=(float*)malloc((size_t)V*sizeof(float));
  float *hit=(float*)malloc((size_t)V*sizeof(float));
  if(!ref||!hit){fprintf(stderr,"oom\n");return 1;}

  /* ---- reference: cache OFF, cold slot ---- */
  coli_prefix_cache_enable(0);
  double t0=now(); float *l=coli_prefill_slot(m,0,idsB,nB); double tcold=now()-t0;
  if(!l){fprintf(stderr,"prefill failed\n");return 1;}
  memcpy(ref,l,(size_t)V*sizeof(float)); free(l);
  printf("cache OFF : %6.3f s   reused %d/%d\n", tcold, coli_prefix_reused(m), coli_prefix_asked(m));

  /* ---- cache ON: prime with A, then run B so the shared prefix hits ---- */
  coli_prefix_cache_enable(1);
  l=coli_prefill_slot(m,0,idsA,nA); if(!l){fprintf(stderr,"prime failed\n");return 1;} free(l);
  int primed=coli_prefix_reused(m);
  t0=now(); l=coli_prefill_slot(m,0,idsB,nB); double twarm=now()-t0;
  if(!l){fprintf(stderr,"warm prefill failed\n");return 1;}
  memcpy(hit,l,(size_t)V*sizeof(float)); free(l);
  int reused=coli_prefix_reused(m);
  printf("prime run : reused %d\n", primed);
  printf("cache ON  : %6.3f s   reused %d/%d (%.1f%%)\n",
         twarm, reused, coli_prefix_asked(m), 100.0*reused/nB);

  /* ---- the gate ---- */
  int fail=0;
  if (reused == 0) {
    printf("FAIL: cache reused NOTHING -- the path under test never ran, so an\n"
           "      identical-logits result here would prove nothing.\n");
    fail=1;
  }
  if (reused != shared) {
    printf("NOTE: reused %d but prompts share %d (cap is nB-1=%d)\n",reused,shared,nB-1);
  }
  double mx; int idx=maxdiff_idx(ref,hit,V,&mx);
  if (mx != 0.0) {
    printf("FAIL: logits differ. max |delta| = %.3e at vocab index %d\n", mx, idx);
    printf("      A prefix cache must be bit-exact: same positions, same K/V.\n");
    fail=1;
  } else {
    printf("PASS: logits BIT-IDENTICAL across %d vocab entries\n", V);
  }
  if (!fail && twarm < tcold)
      printf("saving: %.3f s -> %.3f s (%.2fx) on a %.1f%% prefix hit\n",
             tcold,twarm,tcold/twarm,100.0*reused/nB);
  else if (!fail)
      printf("NOTE: no wall-clock saving measured (%.3f -> %.3f). Correctness still holds.\n",tcold,twarm);

  free(ref); free(hit); coli_free(m);
  printf(fail?"RESULT: FAIL\n":"RESULT: PASS\n");
  return fail;
}
