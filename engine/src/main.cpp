/* main.c — CLI. Text in, text out. */
#define _GNU_SOURCE
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static void usage(const char*a0){ fprintf(stderr,
  "usage: %s <model.gguf> [options]\n"
  "  -p TEXT     prompt\n  -n N        tokens to generate (default 64)\n"
  "  -c N        context (default: model's)\n  --temp F    (0 = greedy)\n"
  "  --top-k N   --top-p F   --min-p F   --repeat-penalty F   --seed N\n"
  "  --nll       teacher-forced NLL over the prompt (ONE prefill, batch=len)\n"
  "  --nll1      the same NLL but stepping ONE token at a time, so it runs the\n"
  "              DECODE path (batch=1). Use this to measure anything that only\n"
  "              applies at batch 1 -- --nll would silently take the wide path.\n"
  "  --f32       full-precision weights: ~4x memory, much slower, but it\n"
  "              separates an ARCHITECTURE bug from quantization loss.\n"
  "              Validate a NEW architecture with this BEFORE trusting int8.\n"
  "  --w4 2      int4-ONLY weights: 0.68x peak RSS and 1.42x faster decode, for\n"
  "              1.08x slower prefill and +3.87%% NLL (qwen2.5-3b, measured on\n"
  "              tests/nll_prompt.txt). Opt-in: the accuracy cost is a\n"
  "              deployment decision, not an engine default.\n"
  "  --w4 1      carry BOTH formats and choose per batch. Same speed and the\n"
  "              same NLL as --w4 2 for +56%% memory; kept for comparison only.\n"
  "              int4 block scales are chosen by a least-squares SEARCH (after\n"
  "              llama.cpp make_qx_quants): +12 s of load time on a 3B model, and\n"
  "              it recovers 30%% of the quantization gap. Nothing else changes.\n"
  "  --gpu       put the weight matrices on the GPU (Vulkan) and run the GEMMs\n"
  "              there. REQUIRES --w4 2. Dense models only; MoE is refused, not\n"
  "              silently ignored. Falls back to the CPU on any dispatch failure.\n"
  "  --w4 11/12  the same two formats with the older amax/7 scale instead, kept\n"
  "              so the two quantizers can be compared on one build. Any other\n"
  "              value is REJECTED rather than quietly treated as int8.\n", a0); }

int main(int argc,char**argv){
  if(argc<2){ usage(argv[0]); return 2; }
  const char*path=argv[1]; const char*prompt=NULL;
  int n_new=64,ctx=0,nll=0,wq_int8=1,slots=1,w4=0,gpu=0;
  coli_sampler sp; coli_sampler_default(&sp);
  for(int i=2;i<argc;i++){
    if(!strcmp(argv[i],"-p")&&i+1<argc) prompt=argv[++i];
    else if(!strcmp(argv[i],"-n")&&i+1<argc) n_new=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")&&i+1<argc) ctx=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--temp")&&i+1<argc) sp.temp=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) sp.top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) sp.top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--min-p")&&i+1<argc) sp.min_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc){ sp.repeat_penalty=(float)atof(argv[++i]); if(!sp.repeat_last_n) sp.repeat_last_n=64; }
    else if(!strcmp(argv[i],"--seed")&&i+1<argc) sp.seed=strtoull(argv[++i],0,10);
    else if(!strcmp(argv[i],"--nll")) nll=1;
    else if(!strcmp(argv[i],"--nll1")) nll=2;
    else if(!strcmp(argv[i],"--f32")) wq_int8=0;
    else if(!strcmp(argv[i],"--w4")&&i+1<argc) w4=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--gpu")) gpu=1;
    else if(!strcmp(argv[i],"--slots")&&i+1<argc) slots=atoi(argv[++i]);
    else { fprintf(stderr,"unknown option %s\n",argv[i]); usage(argv[0]); return 2; } }

  char err[512]; double t0=now();
  coli_model*m=coli_load(path,ctx,slots,wq_int8,w4,err,sizeof err);
  if(!m){ fprintf(stderr,"load failed: %s\n",err); return 1; }
  coli_cfg*c=&m->cfg;
  fprintf(stderr,"%s: %d layers, d=%d, heads=%d/%d, hd=%d, ffn=%d, vocab=%d, ctx=%d\n",
    c->arch,c->n_layers,c->hidden,c->n_heads,c->n_kv_heads,c->head_dim,c->inter,c->vocab,m->max_ctx);
  fprintf(stderr,"rope=%s theta=%.0f eps=%.1e qkv_bias=%s rope_freqs=%s  loaded in %.1fs\n",
    c->rope==COLI_ROPE_NEOX?"neox":"interleaved",c->rope_theta,c->eps,
    c->qkv_bias?"yes":"no",m->rope_ff?"yes":"no",now()-t0);
  /* Report the format that is actually RESIDENT. A line that only echoed
   * wq_int8 would print "int8 (per-row scale)" for an int4-only model, and a
   * label contradicting the thing it describes is worse than no label -- it is
   * how a run gets filed under the wrong configuration weeks later. */
  { int fmt = w4 % 10, rm = w4 < 10 && fmt;
    fprintf(stderr,"weights: %s%s\n", !wq_int8 ? "f32 (full precision)" :
            fmt==2 ? "int4 ONLY (per-32 block scale), no int8 form" :
            fmt==1 ? "int8 + int4, dispatched per batch size" :
                     "int8 (per-row scale)",
            (wq_int8 && fmt && rm) ? " [RMSE scale search]" : ""); }

  /* COLI_WSUM=1: checksum the loaded weights. Splits "the model loaded
   * differently" from "the arithmetic ran differently" -- without it, a
   * cross-platform output difference has no bisect point. */
  if (getenv("COLI_WSUM")) {
    unsigned long long h1=1469598103934665603ULL, h2=h1;
    const coli_w_i8 *ws[3] = { &m->tok_embd, &m->L[0].wq, &m->out };
    const char *nm[3] = { "tok_embd", "blk0.attn_q", "output" };
    for (int k=0;k<3;k++){ unsigned long long h=1469598103934665603ULL;
      /* --w4 2 frees the int8 form, so there is nothing to checksum here.
       * Saying so beats dereferencing null, and beats printing a zero that
       * would read as "the weights are identical". */
      if (!ws[k]->qu){ fprintf(stderr,"WSUM %-12s int4-only, no int8 form to checksum\n",nm[k]); continue; }
      int64_t n=ws[k]->I*ws[k]->O;
      for (int64_t i=0;i<n;i++){ h^=ws[k]->qu[i]; h*=1099511628211ULL; }
      unsigned long long hs=1469598103934665603ULL;
      for (int64_t o=0;o<ws[k]->O;o++){ unsigned u; memcpy(&u,&ws[k]->scale[o],4); hs^=u; hs*=1099511628211ULL; }
      fprintf(stderr,"WSUM %-12s qu=%016llx scale=%016llx  (%lldx%lld)\n",
              nm[k],h,hs,(long long)ws[k]->I,(long long)ws[k]->O);
      h1^=h; h2^=hs; }
    fprintf(stderr,"WSUM total qu=%016llx scale=%016llx\n",h1,h2);
  }

  if (gpu) {
    char gerr[512]; double tg0=now();
    int nup = coli_gpu_upload(m, gerr, sizeof gerr);
    if (nup < 0) { fprintf(stderr,"--gpu refused: %s\n", gerr); return 1; }
    fprintf(stderr,"gpu: %d weight matrices uploaded in %.1fs\n", nup, now()-tg0);
  }

  static int ids[65536]; int nid=0;
  if(prompt){ if(c->add_bos&&c->bos>=0) ids[nid++]=c->bos;
    nid+=coli_encode(m,prompt,ids+nid,65536-nid); }
  else if(c->bos>=0) ids[nid++]=c->bos;
  fprintf(stderr,"prompt: %d tokens\n",nid);

  if(nll==2){
    /* Decode-path NLL: one token per step, so every GEMM runs at n=1. --nll
     * prefills the whole prompt in one call and therefore measures the WIDE
     * kernel, which is the wrong path for anything decode-specific. */
    double t=now(); double sum=0; int cnt=0;
    coli_seq sq; sq.slot=0;
    float *lg=(float*)malloc((size_t)c->vocab*sizeof(float));
    for(int i=0;i<nid-1;i++){
      sq.pos=i; sq.token=ids[i];
      if(coli_decode_batch(m,&sq,1,lg)!=0){ fprintf(stderr,"decode failed\n"); return 1; }
      float mx=-1e30f; for(int j=0;j<c->vocab;j++) if(lg[j]>mx)mx=lg[j];
      double se=0; for(int j=0;j<c->vocab;j++) se+=exp((double)(lg[j]-mx));
      sum+=-((double)lg[ids[i+1]]-mx-log(se)); cnt++;
    }
    free(lg);
    printf("TF-NLL(decode path, batch=1): %.4f nats/token over %d tokens | ppl=%.3f | %.1fs\n",
           sum/cnt,cnt,exp(sum/cnt),now()-t);
    coli_free(m); return 0;
  }
  if(nll){
    double t=now(); float*lg=coli_forward(m,ids,nid,1);
    if(!lg) return 1;
    double s=0; int cnt=0;
    for(int i=0;i<nid-1;i++){ const float*row=lg+(int64_t)i*c->vocab;
      float mx=-1e30f; for(int j=0;j<c->vocab;j++) if(row[j]>mx)mx=row[j];
      double se=0; for(int j=0;j<c->vocab;j++) se+=exp((double)(row[j]-mx));
      s+=-((double)row[ids[i+1]]-mx-log(se)); cnt++; }
    printf("TF-NLL: %.4f nats/token over %d tokens | ppl=%.3f | %.1fs\n",s/cnt,cnt,exp(s/cnt),now()-t);
    free(lg); coli_free(m); return 0; }

  double tp=now(); float*lg=coli_forward(m,ids,nid,0);
  if(!lg) return 1;
  double prefill=now()-tp;
  fprintf(stderr,"prefill %d tokens in %.2fs (%.1f tok/s)\n",nid,prefill,nid/prefill);

  int cur=coli_sample(&sp,lg,c->vocab,ids,nid); free(lg);
  double tg=now(); int gen=0;
  for(int i=0;i<n_new;i++){
    ids[nid++]=cur; gen++;
    char buf[64]; int nb=coli_decode(m,&cur,1,buf,(int)sizeof buf-1); buf[nb]=0;
    fputs(buf,stdout); fflush(stdout);
    if(cur==c->eos) break;
    float*l2=coli_forward(m,&cur,1,0); if(!l2) break;
    cur=coli_sample(&sp,l2,c->vocab,ids,nid); free(l2); }
  double gt=now()-tg;
  printf("\n");
  fprintf(stderr,"generated %d tokens in %.2fs (%.1f tok/s)\n",gen,gt,gen/gt);
  coli_free(m); return 0; }
