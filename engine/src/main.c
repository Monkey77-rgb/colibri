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
  "  --nll       teacher-forced NLL over the prompt, then exit\n", a0); }

int main(int argc,char**argv){
  if(argc<2){ usage(argv[0]); return 2; }
  const char*path=argv[1]; const char*prompt=NULL;
  int n_new=64,ctx=0,nll=0;
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
    else { fprintf(stderr,"unknown option %s\n",argv[i]); usage(argv[0]); return 2; } }

  char err[512]; double t0=now();
  coli_model*m=coli_load(path,ctx,1,err,sizeof err);
  if(!m){ fprintf(stderr,"load failed: %s\n",err); return 1; }
  coli_cfg*c=&m->cfg;
  fprintf(stderr,"%s: %d layers, d=%d, heads=%d/%d, hd=%d, ffn=%d, vocab=%d, ctx=%d\n",
    c->arch,c->n_layers,c->hidden,c->n_heads,c->n_kv_heads,c->head_dim,c->inter,c->vocab,m->max_ctx);
  fprintf(stderr,"rope=%s theta=%.0f eps=%.1e qkv_bias=%s rope_freqs=%s  loaded in %.1fs\n",
    c->rope==COLI_ROPE_NEOX?"neox":"interleaved",c->rope_theta,c->eps,
    c->qkv_bias?"yes":"no",m->rope_ff?"yes":"no",now()-t0);

  /* COLI_WSUM=1: checksum the loaded weights. Splits "the model loaded
   * differently" from "the arithmetic ran differently" -- without it, a
   * cross-platform output difference has no bisect point. */
  if (getenv("COLI_WSUM")) {
    unsigned long long h1=1469598103934665603ULL, h2=h1;
    const coli_w_i8 *ws[3] = { &m->tok_embd, &m->L[0].wq, &m->out };
    const char *nm[3] = { "tok_embd", "blk0.attn_q", "output" };
    for (int k=0;k<3;k++){ unsigned long long h=1469598103934665603ULL;
      int64_t n=ws[k]->I*ws[k]->O;
      for (int64_t i=0;i<n;i++){ h^=ws[k]->qu[i]; h*=1099511628211ULL; }
      unsigned long long hs=1469598103934665603ULL;
      for (int64_t o=0;o<ws[k]->O;o++){ unsigned u; memcpy(&u,&ws[k]->scale[o],4); hs^=u; hs*=1099511628211ULL; }
      fprintf(stderr,"WSUM %-12s qu=%016llx scale=%016llx  (%lldx%lld)\n",
              nm[k],h,hs,(long long)ws[k]->I,(long long)ws[k]->O);
      h1^=h; h2^=hs; }
    fprintf(stderr,"WSUM total qu=%016llx scale=%016llx\n",h1,h2);
  }

  static int ids[65536]; int nid=0;
  if(prompt){ if(c->add_bos&&c->bos>=0) ids[nid++]=c->bos;
    nid+=coli_encode(m,prompt,ids+nid,65536-nid); }
  else if(c->bos>=0) ids[nid++]=c->bos;
  fprintf(stderr,"prompt: %d tokens\n",nid);

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
