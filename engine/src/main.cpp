/* main.c — CLI. Text in, text out. */
#define _GNU_SOURCE
#include "model.h"
#include "backend.h"
#include "hw_detect.h"
#include "gpu_keepalive.h"
#include "loader.h"
#include "spec.h"
#include <cstdio>
extern "C" void coli_cpu_prof_dump(std::FILE *f);
extern "C" void coli_prefill_prof_dump(std::FILE *f);
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
  "  --gpu       put the weight matrices on the device and run the GEMMs there\n"
  "  --backend B device backend: auto (hardware probe + plan), vulkan, cuda, torch, cpu\n"
  "  --tune      plan threads/COLI_MOE_VRAM_MB/COLI_EXPERT_GB/COLI_GPU_ATTN from the hardware (unset ones only)\n"
  "  --hw        print the hardware probe and the plan (implies --tune)\n"
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

/* ---- hardware plan helpers (2026-09-14) ----------------------------------
 * The probe result lives for the whole run so phase 2 does not probe twice. */
static coli_hw g_hw;
static unsigned hw_built(const coli_hw *hw) {
  unsigned built = 0;
#ifdef COLI_HAVE_VK
  built |= COLI_BE_VULKAN;
#endif
#ifdef COLI_HAVE_CUDA
  built |= COLI_BE_CUDA;
#endif
  if (hw->torch_plugin_present) built |= COLI_BE_TORCH;
  return built;
}
/* Dense bytes the device upload would pin, read from the GGUF header BEFORE
 * load: per layer attn_q/k/v/output at int8 width (int4 when --w4 halves it)
 * plus the output head (token_embd when the head is tied). KV bytes at the
 * context the run will use (--ctx, else the trained length), for `slots`
 * sequences, at sizeof(coli_kvt). Returns 0 and prints why if the header
 * cannot be read -- the planner then treats the model as size-unknown and
 * says so in its reason. */
static char g_hdr_arch[64]; static long long g_hdr_experts = 0;   /* filled by gguf_dense_estimate */
static uint64_t gguf_dense_estimate(const char *path, int w4, int ctx, int slots, uint64_t *kv_out) {
  char err[256]; char key[128]; coli_gguf *g = coli_gguf_open(path, err, sizeof err);
  if (!g) { fprintf(stderr,"auto: cannot read GGUF header for the size estimate: %s\n", err); return 0; }
  char arch[64] = {0}; coli_gguf_str(g, "general.architecture", arch, sizeof arch);
  snprintf(g_hdr_arch, sizeof g_hdr_arch, "%s", arch);
  snprintf(key, sizeof key, "%s.expert_count", arch); if (!coli_gguf_i64(g, key, &g_hdr_experts)) g_hdr_experts = 0;
  long long nl = 0, nh = 0, nkv = 0, emb = 0, hd = 0, ctxt = 0;
  snprintf(key, sizeof key, "%s.block_count", arch);              coli_gguf_i64(g, key, &nl);
  snprintf(key, sizeof key, "%s.attention.head_count", arch);     coli_gguf_i64(g, key, &nh);
  snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);  coli_gguf_i64(g, key, &nkv);
  snprintf(key, sizeof key, "%s.embedding_length", arch);         coli_gguf_i64(g, key, &emb);
  snprintf(key, sizeof key, "%s.attention.key_length", arch);     coli_gguf_i64(g, key, &hd);
  snprintf(key, sizeof key, "%s.context_length", arch);           coli_gguf_i64(g, key, &ctxt);
  if (!hd && nh) hd = emb / nh;
  if (!nkv) nkv = nh;
  uint64_t dense = 0; const char *nm[4] = { "attn_q", "attn_k", "attn_v", "attn_output" };
  for (long long l = 0; l < nl; l++) for (int t = 0; t < 4; t++) {
    snprintf(key, sizeof key, "blk.%lld.%s.weight", l, nm[t]);
    int64_t d0 = coli_gguf_shape(g, key, 0), d1 = coli_gguf_shape(g, key, 1);
    if (d0 > 0 && d1 > 0) dense += (uint64_t)d0 * (uint64_t)d1;
  }
  { const char *hn = coli_gguf_has(g, "output.weight") ? "output.weight" : "token_embd.weight";
    int64_t d0 = coli_gguf_shape(g, hn, 0), d1 = coli_gguf_shape(g, hn, 1);
    if (d0 > 0 && d1 > 0) dense += (uint64_t)d0 * (uint64_t)d1; }
  if (w4 % 10 == 2) dense /= 2;                      /* int4-only resident width */
  long long use_ctx = ctx > 0 ? ctx : ctxt;
  if (kv_out) *kv_out = (uint64_t)(slots > 0 ? slots : 1) * nl * 2 * nkv * (uint64_t)use_ctx * hd * sizeof(coli_kvt);
  coli_gguf_close(g);
  return dense;
}

int main(int argc,char**argv){
  if(argc<2){ usage(argv[0]); return 2; }
  const char*path=argv[1]; const char*prompt=NULL;
  int n_new=64,ctx=0,nll=0,wq_int8=1,slots=1,w4=0,gpu=0,awq=0,spec_k=0;
  int auto_tune=0, show_hw=0; const char *backend_pref="auto";
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
    else if(!strcmp(argv[i],"--spec")&&i+1<argc) spec_k=atoi(argv[++i]);
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
    else if(!strcmp(argv[i],"--backend")&&i+1<argc){ backend_pref=argv[++i]; if(!strcmp(backend_pref,"auto")) gpu=2; else if(!strcmp(backend_pref,"cpu")) gpu=0; else { coli_gpu_backend(backend_pref); gpu=1; } }
    else if(!strcmp(argv[i],"--auto")) gpu=2;          /* resolved after load */
    else if(!strcmp(argv[i],"--tune")) auto_tune=1;     /* plan knobs from the hardware, keep the backend choice */
    else if(!strcmp(argv[i],"--hw")) { show_hw=1; auto_tune=1; }
    else if(!strcmp(argv[i],"--awq")) awq=1;
    else if(!strcmp(argv[i],"--awq-calib")&&i+1<argc){ awq=1; awq_file=argv[++i]; }
    else if(!strcmp(argv[i],"--slots")&&i+1<argc) slots=atoi(argv[++i]);
    else { fprintf(stderr,"unknown option %s\n",argv[i]); usage(argv[0]); return 2; } }

  /* 2026-09-14: --auto / --backend auto / --tune / --hw plan from a hardware
   * PROBE (src/hw_detect.c: CPU cores + RAM, Vulkan devices, CUDA via dlopen,
   * torch plugin) instead of the old Vulkan-only class test. Two phases, because
   * two of the knobs are consumed by coli_load itself: threads and the expert
   * store size are set HERE from the probe plus the GGUF header's tensor shapes
   * (dense bytes at the resident width, KV at the requested/trained context);
   * the backend and VRAM budget are set after load from the exact counts. What
   * the user set explicitly always wins; every decision is printed with its
   * reason, because a silent policy is indistinguishable from a slow engine. */
  /* An explicit device request (--gpu / --backend vulkan|cuda|torch) on a
   * non-gpt-oss model gets the same int4-only pick as auto: the user asked for
   * the device, not for a lesson about which weight width it serves. */
  if (gpu == 1 && w4 == 0) {
    uint64_t kv_tmp; (void)gguf_dense_estimate(path, w4, ctx, slots, &kv_tmp);
    if (g_hdr_arch[0] && strcmp(g_hdr_arch, "gpt-oss") != 0) {
      w4 = 2; fprintf(stderr,"device path: weights=int4-only (--w4 2) for arch '%s'\n", g_hdr_arch); }
  }
  if (gpu == 2 || auto_tune) {
    coli_hw_probe(&g_hw);
    if (show_hw) coli_hw_print(&g_hw, stderr);
    uint64_t kv_est = 0, dense_est = gguf_dense_estimate(path, w4, ctx, slots, &kv_est);
    coli_hw_plan plan;
    coli_hw_plan_make_ex(&g_hw, backend_pref, dense_est, kv_est, hw_built(&g_hw), &plan);
    fprintf(stderr,"auto(pre-load): threads=%d expert_store_gb=%d (header estimate: dense %.2f GiB, kv %.2f GiB) -- %s\n",
            plan.threads, plan.expert_store_gb, dense_est/1073741824.0, kv_est/1073741824.0, plan.reason);
    if (nthreads <= 0) nthreads = plan.threads;
    /* MoE model (header expert_count > 0): the disk-backed expert store is what
     * lets a model larger than RAM run at all (gpt-oss-120b on this 30 GB box,
     * 2026-09-14). Enable it with the planned budget unless the user decided. */
    if (g_hdr_experts > 0 && !getenv("COLI_EXPERT_STORE")) setenv("COLI_EXPERT_STORE", "1", 0);
    /* MoE router in f32: the 09-13 Qwen3-235B and 09-14 gpt-oss oracles were
     * accepted with COLI_KEEP_F32=router (a quantized router was the larger
     * half of the 235B gap); the router is 2880x128 per layer, so it costs
     * nothing measurable. Auto sets it unless the user chose otherwise. */
    if (g_hdr_experts > 0 && !getenv("COLI_KEEP_F32")) setenv("COLI_KEEP_F32", "router", 0);
    if (!getenv("COLI_EXPERT_GB") && plan.expert_store_gb > 0) {
      char tmp[32]; snprintf(tmp, sizeof tmp, "%d", plan.expert_store_gb); setenv("COLI_EXPERT_GB", tmp, 0); }
    /* The device path for every architecture except gpt-oss serves int4-only
     * weights (--w4 2; gpt-oss keeps int8 dense + native MXFP4 experts). A user
     * who asked for auto should not have to know that: pick it when a device
     * backend was planned and no weight format was requested. */
    if (strcmp(plan.backend, "cpu") != 0 && w4 == 0 && strcmp(g_hdr_arch, "gpt-oss") != 0) {
      w4 = 2; fprintf(stderr,"auto(pre-load): weights=int4-only (--w4 2) for the %s device path on arch '%s'\n", plan.backend, g_hdr_arch); }
    if (g_hdr_experts > 0) fprintf(stderr,"auto(pre-load): MoE (%lld experts): COLI_EXPERT_STORE=%s COLI_EXPERT_GB=%s COLI_KEEP_F32=%s\n",
                                  g_hdr_experts, getenv("COLI_EXPERT_STORE"), getenv("COLI_EXPERT_GB") ? getenv("COLI_EXPERT_GB") : "unset",
                                  getenv("COLI_KEEP_F32") ? getenv("COLI_KEEP_F32") : "unset");
  }
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
  /* Phase 2 of the hardware plan (see hw_plan_pre above): now the exact dense
   * and KV byte counts are known, decide the backend, the expert VRAM budget and
   * GPU attention. Only knobs the user left unset are filled. */
  if (gpu == 2 || auto_tune) {
    coli_hw_plan plan;
    coli_hw_plan_make_ex(&g_hw, backend_pref, coli_model_dense_bytes(m), coli_model_kv_bytes(m), hw_built(&g_hw), &plan);
    fprintf(stderr,"auto: backend=%s moe_vram_mb=%d gpu_attn=%d gpu_keepalive=%d (dense %.2f GiB, kv@max_ctx %.2f GiB) -- %s\n",
            plan.backend, plan.moe_vram_mb, plan.gpu_attn, plan.gpu_keepalive,
            coli_model_dense_bytes(m)/1073741824.0, coli_model_kv_bytes(m)/1073741824.0, plan.reason);
    /* Calibrate by measurement when more than one device backend could serve
     * (2026-09-15): the fixed vulkan>cuda order was 09-14's measurement and the
     * two are level now. COLI_BACKEND_BENCH=0 keeps the planner's order. */
    if (!strcmp(backend_pref, "auto") && strcmp(plan.backend, "cpu") != 0
        && (hw_built(&g_hw) & COLI_BE_VULKAN) && (hw_built(&g_hw) & COLI_BE_CUDA) && g_hw.n_vk > 0 && g_hw.cuda.present
        && !(getenv("COLI_BACKEND_BENCH") && atoi(getenv("COLI_BACKEND_BENCH")) == 0)) {
      char e1[256]="", e2[256]=""; double tb0=now();
      double vk = coli_backend_bench_gemv_us("vulkan", 20, e1, sizeof e1);
      double cu = coli_backend_bench_gemv_us("cuda",   20, e2, sizeof e2);
      const char *pick = plan.backend;
      if (vk > 0 && cu > 0) pick = (cu < vk) ? "cuda" : "vulkan";
      else if (vk > 0) pick = "vulkan"; else if (cu > 0) pick = "cuda";
      fprintf(stderr,"auto: bench gemv 2880^2 n=1 best-of-20: vulkan %.1f us%s%s, cuda %.1f us%s%s -> %s (%.0f ms to measure)\n",
              vk, vk>0?"":" (", vk>0?"":e1, cu, cu>0?"":" (", cu>0?"":e2, pick, (now()-tb0)*1000);
      snprintf(plan.backend, sizeof plan.backend, "%s", pick);
    }
    if (!strcmp(plan.backend, "cpu")) { if (gpu == 2) gpu = 0; }
    else { gpu = 1; coli_gpu_backend(plan.backend); }
    char tmp[32];
    if (!getenv("COLI_MOE_VRAM_MB")) { snprintf(tmp, sizeof tmp, "%d", plan.moe_vram_mb); setenv("COLI_MOE_VRAM_MB", tmp, 0); }
    if (!getenv("COLI_GPU_ATTN"))    { snprintf(tmp, sizeof tmp, "%d", plan.gpu_attn);    setenv("COLI_GPU_ATTN", tmp, 0); }
    if (!getenv("COLI_GPU_KEEPALIVE")) { snprintf(tmp, sizeof tmp, "%d", plan.gpu_keepalive); setenv("COLI_GPU_KEEPALIVE", tmp, 0); }
  }

  if (gpu == 1) {
    char gerr[512]; double tg0=now();
    int nup = coli_gpu_upload(m, gerr, sizeof gerr);
    if (nup < 0) { fprintf(stderr,"--gpu refused: %s\n", gerr); return 1; }
    char gmem[256]; coli_gpu_meminfo(gmem, sizeof gmem);
    fprintf(stderr,"gpu: %d weight matrices uploaded in %.1fs\n", nup, now()-tg0);
    fprintf(stderr,"gpu: weight memory = %s\n", gmem);
    /* Memory-clock keepalive (gpu_keepalive.h): on when COLI_GPU_KEEPALIVE=1, which
     * --tune/--auto set for a discrete GPU. Its own Vulkan context; stopped at exit. */
    if (getenv("COLI_GPU_KEEPALIVE") && atoi(getenv("COLI_GPU_KEEPALIVE")) == 1) {
      char kerr[256]; int us = getenv("COLI_GPU_KEEPALIVE_US") ? atoi(getenv("COLI_GPU_KEEPALIVE_US")) : 2000;
      if (coli_gpu_keepalive_start(us, kerr, sizeof kerr) == 0) { atexit(coli_gpu_keepalive_stop); fprintf(stderr,"gpu: memory-clock keepalive on (period %d us)\n", us); }
      else fprintf(stderr,"gpu: keepalive not started: %s\n", kerr);
    }
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
      /* COLI_NLL_DUMP=<path> (2026-09-13): one line per scored token, "<index> <nll>", so the
       * same tokens can be compared against llama-perplexity, which scores only the second half
       * of each context window -- an aggregate over all 681 tokens cannot be set beside it. */
      { static FILE *nd = NULL; static int tried = 0;
        if (!tried) { tried = 1; const char *e = getenv("COLI_NLL_DUMP"); if (e && *e) nd = fopen(e, "w"); }
        if (nd) { fprintf(nd, "%d %.6f\n", i+1, -((double)lg[ids[i+1]]-mx-log(se))); if (i+2 >= nid) fclose(nd); } }
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
    if (getenv("COLI_CPU_PROF")) { coli_cpu_prof_dump(stderr); coli_prefill_prof_dump(stderr); }
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
      s+=-((double)row[ids[i+1]]-mx-log(se)); cnt++;
      /* COLI_NLL_DUMP on the batched --nll path too (2026-09-13), same "<index> <nll>" format
       * as --nll1. Why: at 0.4 tok/s the 235B needs ~30 min for --nll1 over this prompt, while
       * one wide prefill reads each selected expert once per layer; the per-token dump is what
       * lets either path be set beside llama-perplexity's chunk window. */
      { static FILE *nd = NULL; static int tried = 0;
        if (!tried) { tried = 1; const char *e = getenv("COLI_NLL_DUMP"); if (e && *e) nd = fopen(e, "w"); }
        if (nd) { fprintf(nd, "%d %.6f\n", i+1, -((double)row[ids[i+1]]-mx-log(se))); if (i+2 >= nid) fclose(nd); } } }
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
    /* THIRD time this exact defect shipped. The two comments above say an
     * instrument unreachable from the measured path reports nothing and reads as
     * "no cost" -- and then --nll, the branch this engine is ACTUALLY benchmarked
     * with, returned here with only the vk dump. Measured 2026-08-26: the GPU
     * accounted for 842.1 ms of a 2.8 s engine wall and the missing ~2 s had no
     * breakdown, because this line did not exist. Outside the ifdef on purpose:
     * the CPU profile is not a GPU-build feature. */
    if (getenv("COLI_CPU_PROF")) { coli_cpu_prof_dump(stderr); coli_prefill_prof_dump(stderr); }
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
  long spec_drafted=0, spec_accepted=0, spec_rounds=0;
  if (spec_k>0) {
    /* n-gram speculative decode. Draft up to spec_k tokens from the context
     * n-gram cache, verify all of them in ONE coli_decode_batch, keep the
     * longest correct prefix. Greedy (temp 0) makes the output token-for-token
     * IDENTICAL to the non-spec path -- the control this is checked against.
     * Rollback is implicit: rejected drafts sit at positions the next round's
     * attention (tmax = pos) never reads, and are overwritten in place. */
    NgramCache *nc = ngram_new();
    ngram_update(nc, ids, nid, nid);
    int K = spec_k;
    int *draft = (int*)malloc(sizeof(int)*(size_t)(K+1));
    coli_seq *seq = (coli_seq*)malloc(sizeof(coli_seq)*(size_t)(K+1));
    float *lgb = (float*)malloc(sizeof(float)*(size_t)(K+1)*c->vocab);
    /* ADAPTIVE GATE. Unlike a GPU, a wide verify batch on this CPU engine is NOT
     * free -- decoding nd tokens costs more than 1 -- so a low-acceptance round
     * is a net loss (measured: -14% on unpredictable prose, +71% on repetitive
     * output). So we only draft when a decaying accept-rate estimate says it pays
     * (>=40%), and otherwise fall back to plain decode (keff=0 -> nd=1), while
     * re-probing with a full draft every 8th round so a workload that BECOMES
     * predictable is picked back up. This bounds the downside to ~0 and keeps the
     * upside. */
    double acc_ema = 1.0;
    while (gen < n_new) {
      draft[0] = cur;
      /* Full draft when the accept-rate EMA says it pays; otherwise fall back to
       * plain decode, but re-PROBE cheaply (a 2-draft, every 16th round) so a
       * workload that becomes predictable is picked back up without paying a
       * full wide batch on every low-accept round. */
      int keff = (acc_ema >= 0.40) ? K : ((spec_rounds % 16 == 0) ? 2 : 0);
      int nd = ngram_draft(nc, ids, nid, draft, keff);   /* nd>=1, draft[0]=seed */
      for (int j=0;j<nd;j++){ seq[j].slot=0; seq[j].pos=nid+j; seq[j].token=draft[j]; }
      if (coli_decode_batch(m, seq, nd, lgb)!=0){ fprintf(stderr,"spec decode failed\n"); break; }
      spec_rounds++; spec_drafted += nd-1;
      int stop=0, round_acc=0, round_draft=nd-1;
      for (int j=0;j<nd;j++){
        if (gen>=n_new){ stop=1; break; }
        ids[nid++]=draft[j]; gen++;
        double ti=now();
        char buf[64]; int nb=coli_decode(m,&draft[j],1,buf,(int)sizeof buf-1); buf[nb]=0;
        fputs(buf,stdout); fflush(stdout); io += now()-ti;
        ngram_update(nc, ids, nid, 1);
        if (draft[j]==c->eos){ stop=1; break; }
        int s = coli_sample(&sp, lgb+(int64_t)j*c->vocab, c->vocab, ids, nid);
        if (j+1<nd && s==draft[j+1]){ spec_accepted++; round_acc++; continue; }  /* draft confirmed */
        cur = s; break;                                                          /* mismatch: s = next seed */
      }
      if (round_draft > 0) acc_ema = 0.7*acc_ema + 0.3*((double)round_acc/round_draft);
      if (stop) break;
    }
    ngram_free(nc); free(draft); free(seq); free(lgb);
  } else {
  /* COLI_GEN_BATCH=1 (2026-09-09): step the generation through coli_decode_batch
   * (slot 0, explicit position) instead of coli_forward. Same maths, but it is the
   * path that carries the fused attention block (COLI_GPU_BLOCK) and GPU decode
   * attention, which coli_forward does not -- so until now the block could only be
   * measured on --nll1, never on the tok/s path the h2h is scored on. The spec
   * path below already mixes a coli_forward prefill with decode_batch steps at
   * pos = nid+j, slot 0; this does the same one token at a time. Default off. */
  static int gen_batch = -1;
  if (gen_batch<0){ const char *e=getenv("COLI_GEN_BATCH"); gen_batch=(e&&*e&&*e!='0')?1:0; }
  float *lgb1 = gen_batch ? (float*)malloc((size_t)c->vocab*sizeof(float)) : NULL;
  for(int i=0;i<n_new;i++){
    ids[nid++]=cur; gen++;
    double ti=now();
    char buf[64]; int nb=coli_decode(m,&cur,1,buf,(int)sizeof buf-1); buf[nb]=0;
    fputs(buf,stdout); fflush(stdout);
    io += now()-ti;
    if(cur==c->eos) break;
    float*l2;
    if (gen_batch) {
      coli_seq sq; sq.slot=0; sq.pos=nid-1; sq.token=cur;
      if (coli_decode_batch(m,&sq,1,lgb1)!=0){ fprintf(stderr,"decode_batch failed\n"); break; }
      l2=lgb1;
    } else { l2=coli_forward(m,&cur,1,0); if(!l2) break; }
    cur=coli_sample(&sp,l2,c->vocab,ids,nid); if(!gen_batch) free(l2); }
  free(lgb1);
  }
  double gt=now()-tg;
  printf("\n");
  fprintf(stderr,"generated %d tokens in %.2fs (%.1f tok/s) [engine %.2fs = %.1f tok/s, stdout %.2fs]\n",
          gen,gt,gen/gt, gt-io, gen/(gt-io), io);
  if (spec_k>0)
    fprintf(stderr,"  spec: %ld rounds, %ld/%ld drafts accepted (%.1f%%), %.2f tokens/forward\n",
            spec_rounds, spec_accepted, spec_drafted,
            spec_drafted? 100.0*spec_accepted/spec_drafted : 0.0,
            spec_rounds? (double)gen/spec_rounds : 0.0);
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
  /* Same lesson as the --nll1 dump above: an instrument that is unreachable from
   * the path being measured reports nothing, and nothing reads as "no cost".
   * This one used to sit INSIDE the #ifdef COLI_HAVE_VK above, so the CPU-only
   * ./coli build -- the build where a CPU profile is the ONLY profile there is --
   * printed nothing under COLI_CPU_PROF=1. Hoisted 2026-08-26. */
  if (getenv("COLI_CPU_PROF")) { coli_cpu_prof_dump(stderr); coli_prefill_prof_dump(stderr); }
  coli_free(m); return 0; }
