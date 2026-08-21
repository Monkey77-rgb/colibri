/* main.c — CLI. Text in, text out. */
#define _GNU_SOURCE
#include "model.h"
#include <cstdio>
extern "C" void coli_cpu_prof_dump(std::FILE *f);
#ifdef COLI_HAVE_VK
#include "vk_backend.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* Banana -- this homelab's build of the engine (owner-named 2026-08-19). The
 * upstream lineage is Colibri, and the C ABI is still coli_* on purpose; see the
 * README section "Why there are two names in this tree". The binary answers to
 * both names via hard links, so argv[0] is printed rather than hardcoded. */
#define BANANA_NAME "Banana"

static void usage(const char*a0){ fprintf(stderr,
  BANANA_NAME " -- inference engine (upstream lineage: Colibri)\n"
  "usage: %s <model.gguf> [options]\n"
  "  -p TEXT     prompt\n"
  "  --prompt-file F  read the prompt from F instead of the command line. Use\n"
  "              this for any reported NLL: -p \"$(cat f)\" is quoting-sensitive\n"
  "              and has silently lost a baseline before. One trailing newline\n"
  "              is stripped so an editor's final \\n cannot change the count.\n  -n N        tokens to generate (default 64)\n"
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
  "  --awq       calibrate the int4 scales on the PROMPT's own activations\n"
  "              (cheap AWQ). Needs --w4 1: loads both formats, runs one forward\n"
  "              pass, rebuilds int4 weighting the scale search by measured\n"
  "              |activation| per input channel, then drops int8. Load-time cost.\n"
  "  --awq-calib F  calibrate on the text in F instead of on the prompt. USE THIS\n"
  "              for any reported number: calibrating on the text you then measure\n"
  "              is training on the test set, and it flatters the result.\n"
  "  --threads N OpenMP threads (default: all hardware threads).\n"
  "              ⚠️ ON A MACHINE WITH ANY BUSY BACKGROUND PROCESS, USE ONE FEWER\n"
  "              THAN nproc. An OpenMP barrier runs at the speed of its slowest\n"
  "              thread, so a single contended core taxes EVERY parallel region.\n"
  "              Measured on a desktop with a HUD daemon at 37%% of one core:\n"
  "              16 threads 49.5s, 15 threads 28.7s, 14 threads 28.0s -- 1.72x\n"
  "              for using FEWER. On an idle host the opposite holds (Legion Go S,\n"
  "              services idle: 16 -> 45.1s, 15 -> 47.0s), so this is not a rule\n"
  "              to hardcode; it is a knob you must set from the actual machine.\n"
  "  --gpu       put the weight matrices on the GPU (Vulkan) and run the GEMMs\n"
  "              there. REQUIRES --w4 2. Dense models only; MoE is refused, not\n"
  "              silently ignored. Falls back to the CPU on any dispatch failure.\n"
  "              Prints which memory the weights were GRANTED -- not always the\n"
  "              memory that was requested.\n"
  "  --auto      choose CPU or GPU from the DEVICE CLASS and say which and why.\n"
  "              Discrete -> GPU (measured 4.7x). Integrated/UMA -> CPU\n"
  "              (measured 1.13x the other way: on UMA the GPU shares the\n"
  "              CPU's bandwidth and wins nothing back for the round trip).\n"
  "              --gpu still forces the GPU; no flag is still CPU.\n"
  "  env COLI_VK_DEVICE_LOCAL=1\n"
  "              also try DEVICE_LOCAL (staged) weights on an INTEGRATED GPU,\n"
  "              which is otherwise skipped entirely. Off by default: on a UMA\n"
  "              part it is not self-evidently a win, and the code it replaces\n"
  "              asserted it was self-evidently a loss -- both are claims. On\n"
  "              the Radeon 780M RADV exposes TWO heaps, and the default\n"
  "              HOST_VISIBLE|HOST_COHERENT type is heap 0, which is NOT\n"
  "              DEVICE_LOCAL; HOST_COHERENT on AMD is uncached/write-combined.\n"
  "              So `unified memory` does not mean the GPU reads both the same\n"
  "              way. Measure with AND without before believing either. Falls\n"
  "              back silently to HOST_VISIBLE if the allocation does not fit.\n"
  "  env COLI_VK_DEVICE_LOCAL=0\n"
  "              the opposite force: take the HOST_VISIBLE fallback on ANY\n"
  "              device, including a discrete one. Exists so the line above\n"
  "              is falsifiable -- without it the fallback is unreachable on\n"
  "              a discrete GPU and the memory report could only ever print\n"
  "              one of its two values. Unset = per-device default.\n"
  "  --w4 11/12  the same two formats with the older amax/7 scale instead, kept\n"
  "              so the two quantizers can be compared on one build. Any other\n"
  "              value is REJECTED rather than quietly treated as int8.\n", a0); }

int main(int argc,char**argv){
  if(argc<2){ usage(argv[0]); return 2; }
  const char*path=argv[1]; const char*prompt=NULL;
  int n_new=64,ctx=0,nll=0,wq_int8=1,slots=1,w4=0,gpu=0,awq=0;
  const char *awq_file=NULL; int nthreads=0;
  coli_sampler sp; coli_sampler_default(&sp);
  for(int i=2;i<argc;i++){
    if(!strcmp(argv[i],"-p")&&i+1<argc) prompt=argv[++i];
    else if(!strcmp(argv[i],"--prompt-file")&&i+1<argc){
      /* tests/nll_prompt.txt is committed and every NLL baseline is quoted
       * against it, but until now the only way to reach it was
       *   -p "$(cat tests/nll_prompt.txt)"
       * an unrecorded, quoting-sensitive incantation. A previous baseline was
       * lost exactly that way. The file was committed; the invocation was not. */
      const char *pf=argv[++i];
      FILE *f=fopen(pf,"rb");
      if(!f){ fprintf(stderr,"--prompt-file: cannot open %s\n",pf); return 2; }
      fseek(f,0,SEEK_END); long len=ftell(f); fseek(f,0,SEEK_SET);
      if(len<0){ fprintf(stderr,"--prompt-file: cannot size %s\n",pf); fclose(f); return 2; }
      char *buf=(char*)malloc((size_t)len+1);
      if(!buf){ fprintf(stderr,"--prompt-file: out of memory\n"); fclose(f); return 2; }
      size_t rd=fread(buf,1,(size_t)len,f); fclose(f);
      if(rd!=(size_t)len){ fprintf(stderr,"--prompt-file: short read on %s (%zu of %ld)\n",pf,rd,len); free(buf); return 2; }
      buf[len]=0;
      /* Strip ONE trailing newline: every text editor adds it and it would
       * otherwise silently change the token count against a stored baseline. */
      if(len>0&&buf[len-1]=='\n') buf[len-1]=0;
      prompt=buf;
    }
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
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthreads=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--gpu")) gpu=1;
    else if(!strcmp(argv[i],"--auto")) gpu=2;          /* resolved after load */
    else if(!strcmp(argv[i],"--awq")) awq=1;
    else if(!strcmp(argv[i],"--awq-calib")&&i+1<argc){ awq=1; awq_file=argv[++i]; }
    else if(!strcmp(argv[i],"--slots")&&i+1<argc) slots=atoi(argv[++i]);
    else { fprintf(stderr,"unknown option %s\n",argv[i]); usage(argv[0]); return 2; } }

  if (nthreads > 0) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#else
    fprintf(stderr,"--threads ignored: built without OpenMP\n");
#endif
  }

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

  /* --auto: pick the backend from the DEVICE CLASS, measured on both classes
   * 2026-08-20, same engine / model / prompt / 681 scored tokens:
   *
   *   discrete (RTX 4070)   GPU  24.3 s  vs CPU 114.0 s   -> GPU by 4.7x
   *   UMA/iGPU (780M)       GPU 111.3 s  vs CPU  98.2 s   -> CPU by 1.13x
   *
   * The inversion is the whole point: on a discrete part the GPU has its own
   * bandwidth, on UMA it shares the CPU's and wins nothing back for the round
   * trip. Same reason DEVICE_LOCAL is 7.4x on discrete and 2.2% SLOWER on UMA.
   *
   * It prints the class and the decision, because a policy that silently picks
   * the wrong backend is indistinguishable from a slow engine. Explicit --gpu
   * still forces the GPU; the default with neither flag is unchanged (CPU). */
#ifdef COLI_HAVE_VK
  if (gpu == 2) {
    int cls = coli_vk_probe_class("shaders/gemm_i8.spv");
    if (cls < 0) {
      gpu = 0;
      fprintf(stderr,"auto: no usable Vulkan device -> CPU\n");
    } else if (cls == 1) {
      gpu = 0;
      fprintf(stderr,"auto: INTEGRATED (UMA) -> CPU  [measured 2026-08-20: CPU 98.2s vs GPU 111.3s on gfx1103]\n");
    } else {
      gpu = 1;
      fprintf(stderr,"auto: DISCRETE -> GPU  [measured 2026-08-20: GPU 24.3s vs CPU 114.0s on RTX 4070]\n");
    }
  }
#else
  if (gpu == 2) { gpu = 0; fprintf(stderr,"auto: build has no Vulkan backend -> CPU\n"); }
#endif

  if (gpu == 1) {
    char gerr[512]; double tg0=now();
    int nup = coli_gpu_upload(m, gerr, sizeof gerr);
    if (nup < 0) { fprintf(stderr,"--gpu refused: %s\n", gerr); return 1; }
    char gmem[256]; coli_gpu_meminfo(gmem, sizeof gmem);
    fprintf(stderr,"gpu: %d weight matrices uploaded in %.1fs\n", nup, now()-tg0);
    fprintf(stderr,"gpu: weight memory = %s\n", gmem);
  }

  static int ids[65536]; int nid=0;
  if(prompt){ if(c->add_bos&&c->bos>=0) ids[nid++]=c->bos;
    nid+=coli_encode(m,prompt,ids+nid,65536-nid); }
  else if(c->bos>=0) ids[nid++]=c->bos;
  fprintf(stderr,"prompt: %d tokens\n",nid);

  /* AFTER tokenization on purpose: the calibration set is the prompt itself, so
   * there is nothing to calibrate on until the tokens exist. An earlier version
   * of this block sat above the tokenizer and an assertion caught it. */
  if (awq) {
    char aerr[512]; double ta=now();
    static int cids[65536]; int cn = nid; const int *cptr = ids;
    if (awq_file) {
      /* A SEPARATE calibration set. Calibrating on the very text you go on to
       * measure is training on the test set: it improved TF-NLL by 35% of the
       * remaining gap when I first did it, and that figure meant nothing. */
      FILE *cf = fopen(awq_file,"rb");
      if (!cf) { fprintf(stderr,"cannot open %s\n", awq_file); return 1; }
      static char cbuf[1<<20]; size_t cl = fread(cbuf,1,sizeof cbuf-1,cf); fclose(cf); cbuf[cl]=0;
      cn = 0;
      if (c->add_bos && c->bos>=0) cids[cn++] = c->bos;
      cn += coli_encode(m, cbuf, cids+cn, 65536-cn);
      cptr = cids;
      fprintf(stderr,"awq: calibrating on %s (%d tokens), evaluating on the prompt\n", awq_file, cn);
    }
    int nreb = coli_awq_calibrate(m, (int*)cptr, cn, aerr, sizeof aerr);
    if (nreb < 0) { fprintf(stderr,"--awq refused: %s\n", aerr); return 1; }
    fprintf(stderr,"awq: %d matrices recalibrated on %d tokens in %.1fs\n", nreb, cn, now()-ta);
  }

  if(nll==2){
    /* Decode-path NLL: one token per step, so every GEMM runs at n=1. --nll
     * prefills the whole prompt in one call and therefore measures the WIDE
     * kernel, which is the wrong path for anything decode-specific. */
    /* ENGINE time and SCORING time are kept apart on purpose. The log-sum-exp
     * below is a scalar double exp() over the whole vocabulary for every token
     * -- 151,936 exp() calls per token on qwen2.5-3b, single-threaded, and none
     * of it is the engine. Folding it into one number made every "decode"
     * figure engine+harness, which does not flatter a ratio (the term is
     * additive in both arms) but does inflate the absolute seconds and dilute
     * any speedup measured against them. Report both; let the reader divide. */
    double sum=0; int cnt=0; double eng=0, score=0;
    coli_seq sq; sq.slot=0;
    float *lg=(float*)malloc((size_t)c->vocab*sizeof(float));
    for(int i=0;i<nid-1;i++){
      sq.pos=i; sq.token=ids[i];
      double te=now();
      if(coli_decode_batch(m,&sq,1,lg)!=0){ fprintf(stderr,"decode failed\n"); return 1; }
      eng += now()-te;
      double ts=now();
      float mx=-1e30f; for(int j=0;j<c->vocab;j++) if(lg[j]>mx)mx=lg[j];
      double se=0; for(int j=0;j<c->vocab;j++) se+=exp((double)(lg[j]-mx));
      sum+=-((double)lg[ids[i+1]]-mx-log(se)); cnt++;
      score += now()-ts;
    }
    free(lg);
    if(cnt==0){ fprintf(stderr,
      "--nll1: 0 scoreable tokens (prompt has %d token(s); need at least 2).\n"
      "A teacher-forced NLL needs a NEXT token to score against. Pass -p TEXT or\n"
      "--prompt-file FILE. Refusing to print nan, which reads as a result.\n", nid);
      coli_free(m); return 3; }
    printf("TF-NLL(decode path, batch=1): %.4f nats/token over %d tokens | ppl=%.3f | "
           "engine %.1fs + scoring %.1fs = %.1fs\n",
           sum/cnt,cnt,exp(sum/cnt),eng,score,eng+score);
#ifdef COLI_HAVE_VK
    /* The dump used to live ONLY after the generation loop, so it could not fire
     * on --nll1 -- the path this engine is actually benchmarked with. An
     * instrument unreachable from the measured path reports nothing, which reads
     * as "no cost". */
    if (getenv("COLI_VK_PROF")) coli_vk_prof_dump(stderr);
#endif
    if (getenv("COLI_CPU_PROF")) coli_cpu_prof_dump(stderr);
    coli_free(m); return 0;
  }
  if(nll){
    double t=now(); float*lg=coli_forward(m,ids,nid,1);
    if(!lg) return 1;
    double eng=now()-t;                 /* the engine: one wide prefill */
    double ts=now();
    double s=0; int cnt=0;
    for(int i=0;i<nid-1;i++){ const float*row=lg+(int64_t)i*c->vocab;
      float mx=-1e30f; for(int j=0;j<c->vocab;j++) if(row[j]>mx)mx=row[j];
      double se=0; for(int j=0;j<c->vocab;j++) se+=exp((double)(row[j]-mx));
      s+=-((double)row[ids[i+1]]-mx-log(se)); cnt++; }
    double score=now()-ts;              /* the harness: scalar softmax over vocab */
    if(cnt==0){ fprintf(stderr,
      "--nll: 0 scoreable tokens (prompt has %d token(s); need at least 2).\n"
      "A teacher-forced NLL needs a NEXT token to score against. Pass -p TEXT or\n"
      "--prompt-file FILE. Refusing to print nan, which reads as a result.\n", nid);
      free(lg); coli_free(m); return 3; }
    printf("TF-NLL: %.4f nats/token over %d tokens | ppl=%.3f | engine %.1fs + scoring %.1fs = %.1fs\n",
           s/cnt,cnt,exp(s/cnt),eng,score,eng+score);
    free(lg);
#ifdef COLI_HAVE_VK
    /* The dump used to live ONLY after the generation loop, so it could not fire
     * on --nll/--nll1 -- the paths this engine is actually benchmarked with. An
     * instrument that cannot run on the measured path reports nothing and reads
     * as "no cost". */
    if (getenv("COLI_VK_PROF")) coli_vk_prof_dump(stderr);
#endif
    coli_free(m); return 0; }

  double tp=now(); float*lg=coli_forward(m,ids,nid,0);
  if(!lg) return 1;
  double prefill=now()-tp;
  fprintf(stderr,"prefill %d tokens in %.2fs (%.1f tok/s)\n",nid,prefill,nid/prefill);

  int cur=coli_sample(&sp,lg,c->vocab,ids,nid); free(lg);
  /* io accumulates detokenize + fputs + the per-token fflush. That flush is a
   * syscall whose cost depends on where stdout POINTS -- tty, pipe or file --
   * so a tok/s figure containing it is not reproducible unless the redirection
   * is stated. Streaming is worth keeping, so it is measured and reported
   * apart rather than removed. */
  double tg=now(); int gen=0; double io=0;
  for(int i=0;i<n_new;i++){
    ids[nid++]=cur; gen++;
    double ti=now();
    char buf[64]; int nb=coli_decode(m,&cur,1,buf,(int)sizeof buf-1); buf[nb]=0;
    fputs(buf,stdout); fflush(stdout);
    io += now()-ti;
    if(cur==c->eos) break;
    float*l2=coli_forward(m,&cur,1,0); if(!l2) break;
    cur=coli_sample(&sp,l2,c->vocab,ids,nid); free(l2); }
  double gt=now()-tg;
  printf("\n");
  fprintf(stderr,"generated %d tokens in %.2fs (%.1f tok/s) [engine %.2fs = %.1f tok/s, stdout %.2fs]\n",
          gen,gt,gen/gt, gt-io, gen/(gt-io), io);
#ifdef COLI_HAVE_VK
  /* Only meaningful when the GPU actually ran; the dump says so itself when it
   * did not, which is the point -- a silent zero would read as "no cost". */
  if (getenv("COLI_VK_PROF")) coli_vk_prof_dump(stderr);
  if (getenv("COLI_VK_FLOOR") && gpu) {
    double mn = -1;
    double ns = coli_vk_probe_submit_ns(g_vk_handle(), atoi(getenv("COLI_VK_FLOOR")), &mn);
    /* min AND mean: the floor is the minimum, and the gap between them is the
     * contention on this machine right now, not a property of the device. */
    if (ns >= 0) fprintf(stderr,"  empty submit+fence: floor %.1f us, mean %.1f us (%.2fx)\n",
                         mn/1000.0, ns/1000.0, mn>0 ? ns/mn : 0.0);
    else         fprintf(stderr,"  empty submit+fence floor: probe failed\n");
  }
#endif
  coli_free(m); return 0; }
