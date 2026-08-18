/* model.c — see model.h for scope and the architecture-validation warning. */
#define _GNU_SOURCE
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "platform.h"
#include "model.h"
#include "loader.h"
#include "trig.h"
#ifdef COLI_HAVE_VK
#include "vk_backend.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)

static void *xmal(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float *fal(int64_t n){ return (float*)xmal((size_t)n*sizeof(float)); }

/* ------------------------------------------------------------ tensor load */
/* RAII for the f32 staging buffer. Every load path used to be
 *   float *f=NULL; ... if (bad) { free(f); return 0; } ... free(f);
 * repeated per tensor, with a `goto fail` above it that did NOT free. Making the
 * buffer own itself removes that whole class rather than fixing instances. */
namespace {
struct F32Buf {
    float  *p = nullptr;
    int64_t n = 0;
    ~F32Buf(){ if (p) coli_gguf_free_f32(p); }
    F32Buf() = default;
    F32Buf(const F32Buf&) = delete;
    F32Buf &operator=(const F32Buf&) = delete;
    bool load(coli_gguf *g, const char *nm){ n = coli_gguf_load_f32(g,nm,&p); return n>0; }
};
}

static int g_wq_int8 = 1;   /* set from coli_load's wq_int8 */
/* Weight format, from coli_load's w4 argument.
 * 0 = int8 only (default) · 1 = BOTH formats, pick per batch
 *          2 = int4 ONLY, the int8 form is never built
 *
 * Mode 1 was the first answer and it is the wrong one: carrying both formats
 * costs +56% memory to win 1.42x on decode, which on a 16 GiB handheld is a bad
 * trade whatever the speedup. Mode 2 is the trade the other way round -- 0.62x
 * the weight bytes AND the decode win, paid for with 1.1-1.25x on prefill (see
 * the table in gemm_i8.h). It only became possible once the int4 kernel stopped
 * re-unpacking the matrix per activation row; before that, prefill cost 1.7-1.8x
 * and int4-only was not a serious option. */
static int g_w4      = 0;

/* Set for token_embd.weight only. That tensor is a LOOKUP, not a GEMM: the
 * forward pass indexes a row out of it directly (three call sites), so it never
 * reaches mm() and there is no int4 kernel in its path -- in int4-only mode it
 * would be a null dereference on the first token, not a slowdown. It is also
 * the wrong tensor to shrink: one row is read per token, so it contributes
 * ~D bytes of traffic against the ~I*O per matrix that int4 exists to cut.
 * When output.weight is TIED to it, the output head stays int8 with it, which
 * is the same choice llama.cpp makes for its own reasons. */
static int g_no_i4   = 0;

/* Least-squares block-scale search in the int4 quantizer (see
 * coli_quantize_w4_ex). Load-time cost only; nothing in the kernels changes. */
static int g_w4_rmse = 0;

/* AWQ calibration: while this is set, every mm() accumulates the mean |input|
 * per channel for its weight matrix. That vector is what tells the quantizer
 * which input channels actually carry signal -- the whole difference between
 * weighting the scale search by |weight| and by |activation|. */
static int g_calib = 0;


/* Weight matrices that carry an int4 twin, parallel to the coli_w_i8 array in
 * each layer. Kept as a side table rather than widening coli_w_i8, so the
 * kernel ABI and every existing call site are untouched. */
struct W4Side { const coli_w_i8 *key; coli_w_i4 v; int gh; float *imp; int64_t impn; };
static W4Side *g_w4tab = nullptr; static int g_w4n = 0, g_w4cap = 0;
static const coli_w_i4 *w4_find(const coli_w_i8 *k){
    for (int i=0;i<g_w4n;i++) if (g_w4tab[i].key==k) return &g_w4tab[i].v;
    return nullptr; }
static void w4_add(const coli_w_i8 *k, const float *f, int64_t I, int64_t O){
    if (g_w4n==g_w4cap){ g_w4cap = g_w4cap? g_w4cap*2 : 64;
        g_w4tab = (W4Side*)realloc(g_w4tab, sizeof(W4Side)*(size_t)g_w4cap); }
    g_w4tab[g_w4n].key = k;
    g_w4tab[g_w4n].gh  = -1;          /* no GPU handle until coli_gpu_upload */
    g_w4tab[g_w4n].imp = nullptr; g_w4tab[g_w4n].impn = 0;
    coli_quantize_w4_ex(&g_w4tab[g_w4n].v, f, I, O, g_w4_rmse);
    g_w4n++; }
static W4Side *w4_slot(const coli_w_i8 *k){
    for (int i=0;i<g_w4n;i++) if (g_w4tab[i].key==k) return &g_w4tab[i];
    return nullptr; }



/* Online-softmax attention -- "flash attention" without the tiling.
 *
 * The two-pass form scores every position into a buffer, finds the max, then
 * exponentiates and weights. That buffer is O(context) per (row, head) and was
 * being malloc'd and freed INSIDE the parallel loop, so a 36-layer model at 32
 * heads did ~1,150 malloc/free pairs per token purely to hold scores.
 *
 * This keeps a running max and a running denominator instead, rescaling the
 * accumulator whenever a new maximum appears, so it needs O(head_dim) of stack
 * and touches K and V exactly once each. Same identity as FlashAttention
 * (2205.14135); the tiling that paper needs is for GPU SRAM, which is not the
 * constraint here.
 *
 * NOT BIT-EXACT with the two-pass version, and it cannot be: the weights are
 * applied in a different order and rescaled as they go. The difference is
 * measured in the README rather than asserted away. */
static inline void attend_online(float *o, const float *qv, const float *Kb, const float *Vb,
                                 int64_t kstride, int tmax, int hd, float scale) {
    float mx = -1e30f, den = 0.f;
    for (int i = 0; i < hd; i++) o[i] = 0.f;
    for (int t = 0; t <= tmax; t++) {
        const float *kv = Kb + (int64_t)t*hd;
        float d = 0.f;
        for (int i = 0; i < hd; i++) d += qv[i]*kv[i];
        d *= scale;
        const float *vp = Vb + (int64_t)t*hd;
        if (d > mx) {
            /* new maximum: rescale everything accumulated so far */
            float corr = (mx == -1e30f) ? 0.f : expf(mx - d);
            den = den*corr + 1.f;
            for (int i = 0; i < hd; i++) o[i] = o[i]*corr + vp[i];
            mx = d;
        } else {
            float p = expf(d - mx);
            den += p;
            for (int i = 0; i < hd; i++) o[i] += p*vp[i];
        }
    }
    (void)kstride;
    float inv = 1.f/den;
    for (int i = 0; i < hd; i++) o[i] *= inv;
}

/* Grow the KV cache so position `pos` is addressable.
 *
 * The stride is kv_ctx, so growing is not a realloc: every [slot][head] row has
 * to move to a wider pitch. That is a strided copy of live data only -- at most
 * what has actually been generated -- and it happens O(log) times because the
 * capacity doubles. The alternative, reserving max_ctx up front, cost 2.25 GiB
 * on a model whose weights are 2.29 GiB, for a context almost no request uses.
 *
 * Returns 0 on success. On allocation failure the OLD cache is left intact and
 * the caller is told, rather than freeing what it is still using. */
static int kv_grow(coli_model *m, int pos) {
    if (pos < m->kv_ctx) return 0;
    if (pos >= m->max_ctx) return -1;
    int want = m->kv_ctx;
    while (want <= pos) want *= 2;
    if (want > m->max_ctx) want = m->max_ctx;

    coli_cfg *c = &m->cfg;
    int KVH = c->n_kv_heads, hd = c->head_dim, S = m->n_slots;
    int64_t rows = (int64_t)S * KVH;
    for (int l = 0; l < c->n_layers; l++) {
        float *nk = (float*)malloc((size_t)rows*want*hd*sizeof(float));
        float *nv = (float*)malloc((size_t)rows*want*hd*sizeof(float));
        if (!nk || !nv) { free(nk); free(nv); return -1; }
        for (int64_t r = 0; r < rows; r++) {
            memcpy(nk + r*(int64_t)want*hd, m->K[l] + r*(int64_t)m->kv_ctx*hd,
                   (size_t)m->kv_ctx*hd*sizeof(float));
            memcpy(nv + r*(int64_t)want*hd, m->V[l] + r*(int64_t)m->kv_ctx*hd,
                   (size_t)m->kv_ctx*hd*sizeof(float));
        }
        free(m->K[l]); free(m->V[l]);
        m->K[l] = nk; m->V[l] = nv;
    }
    m->kv_ctx = want;
    return 0;
}

/* ------------------------------------------------------------------- GPU ---
 * The Vulkan backend was reachable only from its own test until now: every GPU
 * figure in the README was a benchmark that no token had ever flowed through.
 * This is the wiring that changes that. It stays OPTIONAL and OFF by default --
 * a machine with no Vulkan device is not an error state, and the CPU path is the
 * one that is bit-exact and portable. */
#ifdef COLI_HAVE_VK
static coli_vk *g_vk = nullptr;
/* The profiling probe in main() needs the live device to time an empty submit.
 * Exposed as a function rather than a global so the CPU-only build has no
 * symbol to resolve. */
extern "C" coli_vk *g_vk_handle(void){ return g_vk; }
#else
/* No Vulkan in this build. The GPU entry points still EXIST and refuse politely,
 * so a caller linking the plain library gets a reason rather than a link error
 * -- and the CPU path, which is the bit-exact one, is untouched. */
#define g_vk ((void*)0)
#endif

/* f32 -> per-row int8, stored offset-to-unsigned. See gemm_i8.h for why
 * unsigned: VPDPBUSD is u8 x s8, and offsetting the WEIGHTS makes the correction
 * term depend on activations (n*I) rather than weights (I*O). */
static void quant_rows(const float *f, coli_w_i8 *w, int64_t I, int64_t O) {
    w->I=I; w->O=O;
    if (!g_wq_int8) {                    /* keep full precision, take a copy */
        w->f = (float*)xmal((size_t)I*O*sizeof(float));
        memcpy(w->f, f, (size_t)I*O*sizeof(float));
        return;
    }
    /* int4 needs I to divide evenly into 32-weight blocks. coli_quantize_w4
     * computes nb = I/32 and would SILENTLY leave a tail unquantized -- garbage
     * weights that still produce fluent-looking output, which is the worst
     * failure shape there is. Every shape in qwen2 and llama divides, so this
     * has never fired; it is here because "has never fired" is not "cannot". */
    const int i4_ok = (I % COLI_W4BLK) == 0;
    if (g_w4 && !g_no_i4 && !i4_ok) {
        static int warned = 0;
        if (!warned++) fprintf(stderr,
            "colibri: a weight matrix has I=%lld, not a multiple of %d -- keeping it int8\n",
            (long long)I, COLI_W4BLK);
    }
    const int use4 = g_w4 && !g_no_i4 && i4_ok;
    if (use4) w4_add(w, f, I, O);
    /* int4-only: do not allocate the int8 form at all. Skipping only the
     * DISPATCH and still allocating is what made an earlier "int4 saves memory"
     * probe report 6.45 GiB -- the build under test was carrying both formats,
     * so the measurement could not have shown anything else. */
    if (use4 && g_w4 == 2) { w->qu = nullptr; w->scale = nullptr; return; }
    w->qu = (uint8_t*)xmal((size_t)I*O);
    w->scale = fal(O);
    for (int64_t o=0;o<O;o++) {
        const float *r = f + o*I;
        float am=0.f; for (int64_t i=0;i<I;i++){ float a=fabsf(r[i]); if(a>am)am=a; }
        float s = am/127.f; if (s<1e-12f) s=1e-12f;
        w->scale[o]=s; float inv=1.f/s;
        for (int64_t i=0;i<I;i++) {
            /* CLAMP: lrintf(127.5) = 128 on both glibc and msvcrt, and 128+128
             * wraps a uint8 to 0, which decodes as -128 -- a SIGN FLIP on the
             * largest weight in the row. */
            int q = (int)lrintf(r[i]*inv);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            w->qu[o*I+i] = (uint8_t)(q + 128);
        }
    }
}

static int load_w(coli_gguf *g, const char *nm, coli_w_i8 *w,
                  int64_t I, int64_t O, int req, char *err, size_t errcap) {
    if (!coli_gguf_has(g,nm)) { if (req) MERR("missing tensor %s", nm); return 0; }
    F32Buf b;
    if (!b.load(g,nm)) { MERR("cannot dequantize %s", nm); return 0; }
    if (b.n != I*O) { MERR("%s: %lld elements, expected %lldx%lld",nm,(long long)b.n,(long long)I,(long long)O); return 0; }
    quant_rows(b.p, w, I, O);
    return 1;
}

static int load_vec(coli_gguf *g, const char *nm, float **out, int64_t n,
                    int req, char *err, size_t errcap) {
    if (!coli_gguf_has(g,nm)) { if (req) MERR("missing tensor %s", nm); return 0; }
    F32Buf b;
    if (!b.load(g,nm)) { MERR("cannot dequantize %s", nm); return 0; }
    if (b.n != n) { MERR("%s: %lld elements, expected %lld",nm,(long long)b.n,(long long)n); return 0; }
    *out = (float*)xmal((size_t)n*sizeof(float));
    memcpy(*out, b.p, (size_t)n*sizeof(float));
    return 1;
}

/* ------------------------------------------------------------------- load */
coli_model *coli_load(const char *path, int max_ctx, int n_slots, int wq_int8, int w4, char *err, size_t errcap) {
    g_wq_int8 = wq_int8 ? 1 : 0;
    /* w4 carries the format in the low digit and the quantizer in the 10s digit:
     * 12 / 22 mean "same format, RMSE scale search". Encoding it here rather than
     * adding a parameter keeps coli_load's signature stable for the Go and Python
     * bindings, which is the same reason coli_open_w4 was added rather than
     * coli_open widened. */
    /* WEIGHT MODE. Valid values are exactly:
     *      0  int8 only
     *      1  both formats, chosen per batch      11  the same, amax/7 scales
     *      2  int4 only                           12  the same, amax/7 scales
     * i.e. format in the units digit, "use the OLD amax/7 quantizer" in the tens.
     * The search is the default for int4 because it measured better: it recovers
     * 30-59% of the quantization gap for load time alone (README).
     *
     * THIS ENCODING PRODUCED TWO BUGS IN ONE SESSION, so it now REJECTS anything
     * it does not recognise instead of falling back to int8:
     *   - `w4 -= 10` turned 22 into 12, which was then no valid format, so it ran
     *     int8 while the CLI label still said "int4 [RMSE scale search]". It
     *     reported the int8 NLL to four decimals, which is what gave it away.
     *   - `w4 % 10` then made 21 mean format 1 (BOTH) rather than format 2. The
     *     NLL was identical either way -- dual-format uses int4 at n=1 -- so only
     *     the peak RSS exposed it: 11.56 GiB where int4-only is 5.81 GiB.
     * Both were silent because an unknown value had a plausible meaning. An
     * argument that cannot be wrong out loud will be wrong quietly. */
    {
        int fmt = w4 % 10, legacy = w4 / 10;
        if (w4 < 0 || fmt > 2 || legacy > 1 || (legacy && fmt == 0)) {
            MERR("w4=%d is not a valid weight mode (expected 0,1,2,11,12)", w4);
            return NULL;
        }
        g_w4      = fmt;
        g_w4_rmse = fmt ? !legacy : 0;
    }
    char e[256];
    /* RAII: the old code had `goto fail` paths that leaked the meta handle and
     * the fd. A destructor cannot forget. */
    struct GgufOwner {
        coli_gguf *g;
        explicit GgufOwner(coli_gguf *p):g(p){}
        ~GgufOwner(){ if(g) coli_gguf_close(g); }
        GgufOwner(const GgufOwner&)=delete;
        GgufOwner &operator=(const GgufOwner&)=delete;
    } own(coli_gguf_open(path,e,sizeof e));
    coli_gguf *G = own.g;
    if (!G) { MERR("%s", e); return NULL; }

    coli_model *m = (coli_model*)calloc(1,sizeof *m);
    coli_cfg *c = &m->cfg;
    if (!coli_gguf_str(G,"general.architecture",c->arch,sizeof c->arch)) { MERR("no architecture"); return NULL; }
    if (strcmp(c->arch,"qwen2") && strcmp(c->arch,"llama") &&
        strcmp(c->arch,"qwen3") && strcmp(c->arch,"qwen3moe")) {
        MERR("architecture '%s' is not validated here; refusing rather than guessing "
             "(a wrong arch produces fluent nonsense)", c->arch); return NULL; }

    long long v; float f;
    /* "<arch>.<suffix>" -- hardcoding "llama." silently fails on qwen2. */
    auto arch_i64 = [&](const char *sfx, long long *o){ char k[256];
        snprintf(k,sizeof k,"%s.%s",c->arch,sfx); return coli_gguf_i64(G,k,o); };
    auto arch_f32 = [&](const char *sfx, float *o){ char k[256];
        snprintf(k,sizeof k,"%s.%s",c->arch,sfx); return coli_gguf_f32(G,k,o); };
    #define REQI(sfx,dst) do{ if(!arch_i64(sfx,&v)){MERR("missing %s.%s",c->arch,sfx);return NULL;} dst=(int)v; }while(0)
    REQI("block_count",c->n_layers); REQI("embedding_length",c->hidden);
    REQI("feed_forward_length",c->inter);
    REQI("attention.head_count",c->n_heads); REQI("attention.head_count_kv",c->n_kv_heads);
    if (arch_i64("context_length",&v)) c->ctx_train=(int)v;
    c->rope_theta = arch_f32("rope.freq_base",&f)?f:10000.f;
    c->eps = arch_f32("attention.layer_norm_rms_epsilon",&f)?f:1e-5f;
    c->head_dim = arch_i64("attention.key_length",&v)?(int)v:c->hidden/c->n_heads;
    if (arch_i64("expert_count",&v)) c->n_expert=(int)v;
    if (arch_i64("expert_used_count",&v)) c->n_expert_used=(int)v;
    if (arch_i64("expert_feed_forward_length",&v)) c->expert_inter=(int)v;
    else c->expert_inter = c->inter;

    /* RoPE pairing is per-architecture. Getting this wrong costs ppl 639 vs 29;
     * see model.h. */
    /* RoPE pairing is per-ARCHITECTURE, and getting it wrong is not subtle in
     * effect but is silent in appearance: the llama path once ran at ppl 639
     * against llama.cpp's 29 with no error anywhere. llama.cpp maps
     * LLM_ARCH_QWEN2/QWEN3/QWEN3MOE to ROPE_TYPE_NEOX (pairs i, i+hd/2) and
     * LLM_ARCH_LLAMA to ROPE_TYPE_NORM (pairs 2i, 2i+1). */
    c->rope = (strncmp(c->arch,"qwen",4)==0) ? COLI_ROPE_NEOX : COLI_ROPE_INTERLEAVED;

    { int64_t vs = coli_gguf_shape(G,"token_embd.weight",1);
      if (vs < 0) { MERR("no token_embd.weight"); return NULL; }
      c->vocab = (int)vs; }
    c->qkv_bias = coli_gguf_has(G,"blk.0.attn_q.bias");
    if (c->n_heads % c->n_kv_heads) { MERR("n_heads %d not divisible by n_kv_heads %d",c->n_heads,c->n_kv_heads); return NULL; }

    int D=c->hidden, hd=c->head_dim, qD=c->n_heads*hd, kvD=c->n_kv_heads*hd;

    /* see g_no_i4: this one tensor is read row-wise, not through mm() */
    g_no_i4 = 1;
    int embd_ok = load_w(G,"token_embd.weight",&m->tok_embd,D,c->vocab,1,err,errcap);
    g_no_i4 = 0;
    if (!embd_ok) return NULL;
    if (!load_w(G,"output.weight",&m->out,D,c->vocab,0,err,errcap)) {
        /* tied embeddings: reuse token_embd as the head */
        m->out = m->tok_embd;
    }
    if (!load_vec(G,"output_norm.weight",&m->out_norm,D,1,err,errcap)) return NULL;
    if (coli_gguf_has(G,"rope_freqs.weight"))
        load_vec(G,"rope_freqs.weight",&m->rope_ff,hd/2,0,err,errcap);

    m->L = (coli_layer*)calloc((size_t)c->n_layers,sizeof(coli_layer));
    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l]; char nm[128];
        #define LW(field,suffix,II,OO,req) do{ snprintf(nm,sizeof nm,"blk.%d.%s",l,suffix); \
            if(!load_w(G,nm,&L->field,II,OO,req,err,errcap) && req) return NULL; }while(0)
        #define LV(field,suffix,N,req) do{ snprintf(nm,sizeof nm,"blk.%d.%s",l,suffix); \
            if(!load_vec(G,nm,&L->field,N,req,err,errcap) && req) return NULL; }while(0)
        LV(attn_norm,"attn_norm.weight",D,1);
        LV(ffn_norm ,"ffn_norm.weight" ,D,1);
        LW(wq,"attn_q.weight",D,qD,1);  LW(wk,"attn_k.weight",D,kvD,1);
        LW(wv,"attn_v.weight",D,kvD,1); LW(wo,"attn_output.weight",qD,D,1);
        if (c->qkv_bias) { LV(bq,"attn_q.bias",qD,1); LV(bk,"attn_k.bias",kvD,1); LV(bv,"attn_v.bias",kvD,1); }
        /* qwen3 / qwen3moe only. Length head_dim, NOT hidden: one gain vector
         * shared by every head, applied to each head's own q/k slice. Verified
         * against llama.cpp llama-model.cpp:3683-3684, where both are created
         * with shape {n_embd_head_k}. Optional (req=0) so qwen2 and llama load
         * unchanged and simply leave them NULL. */
        LV(q_norm,"attn_q_norm.weight",hd,0);
        LV(k_norm,"attn_k_norm.weight",hd,0);
        if (c->n_expert > 0) {
            /* GGUF stores the experts of one layer as ONE 3-D tensor,
             * [in, out, n_expert], contiguous in that order, so expert e's
             * matrix starts at e*in*out. Split into per-expert coli_w_i8 so the
             * grouped GEMM can address one expert's weights as a plain matrix. */
            int EI = c->expert_inter;
            snprintf(nm,sizeof nm,"blk.%d.ffn_gate_inp.weight",l);
            if(!load_w(G,nm,&L->router,D,c->n_expert,1,err,errcap)) return NULL;
            L->e_gate=(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            L->e_up  =(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            L->e_down=(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            static const char *exs[3]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
            for (int t3=0;t3<3;t3++) {
                snprintf(nm,sizeof nm,"blk.%d.%s",l,exs[t3]);
                if(!coli_gguf_has(G,nm)){ MERR("missing %s",nm); return NULL; }
                int64_t II = (t3==2)?EI:D, OO = (t3==2)?D:EI;
                F32Buf eb;
                if(!eb.load(G,nm)){ MERR("cannot dequantize %s",nm); return NULL; }
                if(eb.n != II*OO*(int64_t)c->n_expert){
                    MERR("%s: %lld elements, expected %lldx%lldx%d",nm,(long long)eb.n,
                         (long long)II,(long long)OO,c->n_expert); return NULL; }
                for (int e=0;e<c->n_expert;e++) {
                    coli_w_i8 *dst = (t3==0)?&L->e_gate[e]:(t3==1)?&L->e_up[e]:&L->e_down[e];
                    quant_rows(eb.p + (int64_t)e*II*OO, dst, II, OO);
                }
            }
        } else {
        LW(gate,"ffn_gate.weight",D,c->inter,1);
        LW(up  ,"ffn_up.weight"  ,D,c->inter,1);
        LW(down,"ffn_down.weight",c->inter,D,1);
        }
        if (l==0 || l==c->n_layers-1) { fprintf(stderr,"  layer %d loaded\n", l); fflush(stderr); }
    }
    m->max_ctx = max_ctx>0?max_ctx:(c->ctx_train?c->ctx_train:2048);
    m->n_slots = n_slots > 0 ? n_slots : 1;
    /* Start small and grow. See coli_model::kv_ctx. */
    m->kv_ctx = m->max_ctx < 512 ? m->max_ctx : 512;
    m->K=(float**)xmal(sizeof(float*)*(size_t)c->n_layers);
    m->V=(float**)xmal(sizeof(float*)*(size_t)c->n_layers);
    for (int l=0;l<c->n_layers;l++){
        m->K[l]=fal((int64_t)m->n_slots*c->n_kv_heads*m->kv_ctx*hd);
        m->V[l]=fal((int64_t)m->n_slots*c->n_kv_heads*m->kv_ctx*hd); }
    fprintf(stderr,"kv cache: %d slot(s) x %d ctx allocated (grows to %d) = %.2f GiB\n",
            m->n_slots, m->kv_ctx, m->max_ctx,
        (double)m->n_slots*c->n_layers*2*c->n_kv_heads*m->kv_ctx*hd*4/1073741824.0);
    m->n_past = 0;

    { int b,eo,ab; m->tok = coli_tok_load(path,&b,&eo,&ab);
      c->bos=b; c->eos=eo; c->add_bos=ab; }
    return m;
}

/* Activation-aware requantization, in two phases and one process.
 *
 * PHASE 1 runs the calibration tokens through the model with g_calib set, so
 * every mm() accumulates mean |input| per channel for its own weight matrix.
 * PHASE 2 rebuilds each int4 matrix with that importance vector instead of the
 * x*x weighting, dequantizing from the int8 copy rather than re-reading the file.
 *
 * WHY DEQUANTIZING FROM int8 IS FINE HERE, and it is worth being explicit: int8
 * round-trip error is roughly 1/16 of int4's step, so it contributes a small
 * fraction of the error the int4 quantizer is choosing scales against. Keeping
 * the f32 weights instead would cost ~12 GiB on a 3B model, and re-opening the
 * GGUF would make this a loader change rather than a quantizer one.
 *
 * Requires w4=1 (both formats resident): phase 2 reads the int8 form. */
int coli_awq_calibrate(coli_model *m, const int *ids, int n, char *err, size_t errcap) {
    if (g_w4 != 1) { MERR("AWQ calibration needs both formats resident (--w4 1)"); return -1; }
    if (n < 8) { MERR("calibration needs a real prompt; got %d tokens", n); return -1; }

    g_calib = 1;
    float *lg = coli_forward(m, ids, n, 0);
    g_calib = 0;
    if (!lg) { MERR("calibration forward pass failed"); return -1; }
    free(lg);
    m->n_past = 0;

    int done = 0;
    for (int i = 0; i < g_w4n; i++) {
        W4Side *sl = &g_w4tab[i];
        if (!sl->imp || sl->impn == 0) continue;
        const coli_w_i8 *w = sl->key;
        if (!w->qu || !w->scale) continue;          /* nothing to rebuild from */
        int64_t I = w->I, O = w->O;
        float *f = (float*)malloc((size_t)I*O*sizeof(float));
        if (!f) { MERR("out of memory rebuilding matrix %d", i); return -1; }
        for (int64_t o = 0; o < O; o++) {
            float sc = w->scale[o];
            const uint8_t *r = w->qu + o*I;
            for (int64_t k = 0; k < I; k++) f[o*I+k] = ((float)r[k] - 128.f) * sc;
        }
        /* mean, and normalised so the search sees comparable magnitudes whatever
         * the token count was */
        float mean = 0.f;
        for (int64_t k = 0; k < I; k++) { sl->imp[k] /= (float)sl->impn; mean += sl->imp[k]; }
        mean /= (float)I;
        if (mean > 0.f) for (int64_t k = 0; k < I; k++) sl->imp[k] /= mean;
        coli_free_w4(&sl->v);
        coli_quantize_w4_imp(&sl->v, f, I, O, sl->imp);
        free(f);
        done++;
    }
    /* Drop the int8 form: the model is int4-only from here, which is the whole
     * point -- calibration is a load-time cost, not a resident one. */
    g_w4 = 2;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        coli_layer *L = &m->L[l];
        coli_w_i8 *ws[7] = { &L->wq,&L->wk,&L->wv,&L->wo,&L->gate,&L->up,&L->down };
        for (int k = 0; k < 7; k++) {
            if (!ws[k]->qu) continue;
            if (!w4_slot(ws[k])) continue;          /* keep anything with no int4 twin */
            free(ws[k]->qu); ws[k]->qu = nullptr;
            free(ws[k]->scale); ws[k]->scale = nullptr;
        }
    }
    return done;
}

int coli_gpu_upload(coli_model *m, char *err, size_t errcap) {
#ifndef COLI_HAVE_VK
    (void)m; MERR("this build has no Vulkan backend (build with `make cli-gpu`)"); return -1;
#else
    if (g_w4 != 2) { MERR("GPU path requires int4-only weights (w4=2); this model is %s",
                          g_w4==1?"dual-format":"int8"); return -1; }
    if (m->cfg.n_expert > 0) { MERR("MoE is not uploaded: %d experts x %d layers exceeds any "
                                    "sane handle budget, and the grouped GEMM has no GPU form yet",
                                    m->cfg.n_expert, m->cfg.n_layers); return -1; }
    if (!g_vk) {
        char e[256];
        g_vk = coli_vk_init("shaders/gemm_i8.spv", e, sizeof e);
        if (!g_vk) { MERR("no Vulkan device: %s", e); return -1; }
        if (!coli_vk_has_i4(g_vk)) { MERR("Vulkan device present but shaders/gemm_i4.spv is "
                                          "missing -- build it with `make vk`"); return -1; }
    }
    int n = 0;
    for (int i = 0; i < g_w4n; i++) {
        if (g_w4tab[i].gh >= 0) { n++; continue; }
        int h = coli_vk_upload_w4(g_vk, &g_w4tab[i].v);
        if (h < 0) { MERR("upload failed at matrix %d of %d (out of VRAM, or more than the "
                          "backend's handle budget)", i, g_w4n); return -1; }
        g_w4tab[i].gh = h;
        n++;
    }
    return n;
#endif
}

void coli_gpu_release(coli_model *m) {
    (void)m;
#ifdef COLI_HAVE_VK
    if (g_vk) { coli_vk_free(g_vk); g_vk = nullptr; }
#endif
    for (int i = 0; i < g_w4n; i++) g_w4tab[i].gh = -1;
}

void coli_free(coli_model *m) {
    if (!m) return;
    if (m->tok) coli_tok_free_(m->tok);
    free(m->out_norm); free(m->rope_ff);
    if (m->K) for (int l=0;l<m->cfg.n_layers;l++) free(m->K[l]);
    if (m->V) for (int l=0;l<m->cfg.n_layers;l++) free(m->V[l]);
    free(m->K); free(m->V); free(m->L); free(m);
}

int coli_encode(coli_model *m, const char *text, int *out, int max) {
    return coli_tok_encode(m->tok, text, out, max); }
int coli_decode(coli_model *m, const int *ids, int n, char *out, int max) {
    return coli_tok_decode(m->tok, ids, n, out, max); }

/* COLI_TRACE=1: FNV-1a of a float buffer at named points in the forward pass.
 * Exists to BISECT a cross-platform difference: Linux and Windows builds produce
 * different output from the first forward pass, and weights, libm, lrintf and
 * thread count were all ruled out by measurement. Checksumming the bits (not the
 * printed value) is the only way to find the first stage that differs -- a
 * printf comparison hides a 1-ulp change and would point at the wrong place. */
static int g_trace = -1;
static void trace(const char *tag, int layer, const float *v, int64_t n) {
    if (g_trace < 0) { const char *e = getenv("COLI_TRACE"); g_trace = (e && atoi(e)==1) ? 1 : 0; }
    if (!g_trace) return;
    uint64_t h = 1469598103934665603ull;
    for (int64_t i=0;i<n;i++){ uint32_t u; memcpy(&u,&v[i],4); 
        for (int b=0;b<4;b++){ h ^= (u>>(b*8))&0xFF; h *= 1099511628211ull; } }
    fprintf(stderr,"TRACE %-12s layer=%-3d n=%-8lld %016llx\n",
            tag, layer, (long long)n, (unsigned long long)h);
}

/* -------------------------------------------------------------- primitives */
static void rmsnorm(float *o, const float *x, const float *g, int64_t n, float eps) {
    double ss=0; for (int64_t i=0;i<n;i++) ss += (double)x[i]*x[i];
    float inv = 1.f/sqrtf((float)(ss/(double)n)+eps);
    for (int64_t i=0;i<n;i++) o[i]=x[i]*inv*g[i];
}

/* `ff` is llama-3.1's rope_freqs.weight, applied by ggml as theta/ff
 * (ggml-cpu/ops.cpp:5619). NULL when the model has no such tensor. */
/* qwen3's per-head q/k RMSNorm. Rows are laid out [row][head][head_dim], so each
 * head's slice is contiguous and normalises independently. Does nothing when the
 * architecture has no such tensors, which keeps the call unconditional at the
 * three forward sites rather than repeating the test at each. */
static void qk_norm(coli_model *m, coli_layer *L, float *q, float *k, int rows,
                    int H, int KVH, int hd, int qD, int kvD) {
    if (!L->q_norm && !L->k_norm) return;
    float eps = m->cfg.eps;
    for (int r = 0; r < rows; r++) {
        if (L->q_norm) for (int h = 0; h < H; h++) {
            float *v = q + (int64_t)r*qD + h*hd;
            rmsnorm(v, v, L->q_norm, hd, eps);
        }
        if (L->k_norm) for (int h = 0; h < KVH; h++) {
            float *v = k + (int64_t)r*kvD + h*hd;
            rmsnorm(v, v, L->k_norm, hd, eps);
        }
    }
}

static void rope_head(float *v, int pos, int hd, float theta, const float *ff, int neox) {
    int half=hd/2;
    for (int i=0;i<half;i++) {
        /* coli_pow / coli_sincos, not powf / sinf / cosf. libm transcendentals
         * differ by 1 ULP between platforms -- measured: sinf at RoPE frequency
         * index 7 is 0x3e6023da on glibc and 0x3e6023d9 on msvcrt -- and that
         * single bit propagates through attention and eventually flips a
         * near-tie argmax. IEEE-754 does not require correctly-rounded
         * transcendentals, so this is a conforming disagreement, not a bug in
         * either libc. See trig.h. */
        double fr = coli_pow((double)theta, -(double)(2*i)/(double)hd);
        if (ff) fr /= (double)ff[i];
        double sd, cd; coli_sincos((double)pos*fr, &sd, &cd);
        float c=(float)cd, s=(float)sd;
        int ia = neox ? i : 2*i, ib = neox ? i+half : 2*i+1;
        float a=v[ia], b=v[ib];
        v[ia]=a*c-b*s; v[ib]=a*s+b*c;
    }
}

/* alloc scratch for a coli_a_i8 of n rows */
static void a_alloc(coli_a_i8 *a, int n, int64_t I) {
    int64_t nb=I/COLI_ABLK;
    a->q=(int8_t*)xmal((size_t)n*I);
    a->scale=fal((int64_t)n*nb);
    a->sum=(int32_t*)xmal((size_t)n*nb*sizeof(int32_t));
    a->n=n; a->I=I;
}
static void a_free(coli_a_i8 *a){ free(a->q); free(a->scale); free(a->sum); }

static void mm(float *y, const float *x, int n, const coli_w_i8 *w) {
    if (w->f) { coli_gemm_f32(y,x,n,w); return; }
    coli_a_i8 a; a_alloc(&a,n,w->I);
    coli_quantize_a(&a,x,n,w->I);
    /* THE FORMAT DISPATCH.
     *
     * Mode 2 (int4-only) has no choice to make -- there is no int8 form to fall
     * back to. Mode 1 carries both and picks per batch: measured 16384x16384
     * streaming, int4/int8 is 0.62x at n=1 and 0.76x at n=2, then 1.14x at n=4
     * and 1.10x at n=32. The crossover sits at the same n=4 the int8 kernels use,
     * which is coincidence rather than a shared cause -- int8 switches on ISA
     * width against a fixed DRAM ceiling, int4 switches on bytes-vs-ALU.
     *
     * The earlier figures here (2.3x / 1.5x) were measured before the unpack was
     * hoisted and are superseded; do not compare against them. */
    W4Side *slot = g_w4 ? w4_slot(w) : nullptr;
    if (g_calib && slot) {
        if (!slot->imp) { slot->imp = (float*)calloc((size_t)w->I, sizeof(float)); slot->impn = 0; }
        if (slot->imp) {
            for (int r = 0; r < n; r++) {
                const float *xr = x + (int64_t)r*w->I;
                for (int64_t i = 0; i < w->I; i++) slot->imp[i] += fabsf(xr[i]);
            }
            slot->impn += n;
        }
    }
    const coli_w_i4 *w4 = slot ? &slot->v : nullptr;
    /* GPU when the weights are on it. Falls back to the CPU kernel on any
     * dispatch failure rather than returning a wrong answer -- the buffers are
     * already correct for it, and a GPU that fails mid-run should degrade, not
     * corrupt. */
#ifdef COLI_HAVE_VK
    if (g_vk && slot && slot->gh >= 0) {
        if (coli_vk_gemm4(g_vk, slot->gh, &a, y) == 0) { a_free(&a); return; }
    }
#endif
    if (w4 && (g_w4 == 2 || n < COLI_GEMM_MIN_WIDE)) coli_gemm_i4(y,&a,w4);
    else                                             coli_gemm_i8(y,&a,w);
    a_free(&a);
}


/* q, k, v as one GPU submission when all three weights are resident there.
 *
 * They share the input, so they also share the quantization and the upload; the
 * three GEMMs are independent, so they share the command buffer and the fence.
 * Returns 1 when it ran, 0 when the caller should issue three separate mm()
 * calls -- which stays bit-identical, so degrading costs speed and nothing else.
 *
 * Worth it because the fence, not the arithmetic, is what the GPU path was
 * spending its time on: an empty submit+fence measures 22us on an RTX 4070 and
 * ~280us on the Radeon 780M, and this removes two of them per layer. */
static int gpu_qkv(coli_layer *L, float *q, float *k, float *v,
                   const float *xn, int n) {
#ifndef COLI_HAVE_VK
    (void)L;(void)q;(void)k;(void)v;(void)xn;(void)n; return 0;
#else
    if (!g_vk) return 0;
    if (L->wq.f || L->wk.f || L->wv.f) return 0;          /* f32 weights: CPU path */
    W4Side *sq = w4_slot(&L->wq), *sk = w4_slot(&L->wk), *sv = w4_slot(&L->wv);
    if (!sq || !sk || !sv) return 0;
    if (sq->gh < 0 || sk->gh < 0 || sv->gh < 0) return 0;
    if (g_calib) return 0;   /* calibration needs mm()'s per-matrix accumulation */
    int64_t I = L->wq.I;
    if (L->wk.I != I || L->wv.I != I) return 0;
    coli_a_i8 a; a_alloc(&a,n,I);
    coli_quantize_a(&a,xn,n,I);
    int wh[3] = { sq->gh, sk->gh, sv->gh };
    float *ys[3] = { q, k, v };
    int ok = coli_vk_gemm4_qkv(g_vk, wh, &a, ys) == 0;
    a_free(&a);
    return ok;
#endif
}

/* The FFN as ONE GPU submission when the weights are resident there.
 *
 * Three GEMMs and a nonlinearity, with every intermediate staying in device
 * memory and the four dispatches recorded into a single command buffer. Measured
 * standalone at 1.15-1.51x over three separate GPU calls -- and the split of
 * WHERE that came from is the useful part: keeping the data on the device was
 * worth ~5%, batching the submissions was the rest.
 *
 * Returns 1 when it ran, 0 when the caller should use the CPU path. Anything
 * unexpected returns 0 rather than a wrong answer: the CPU path is bit-exact and
 * always available, so degrading to it costs speed and nothing else. */
static int gpu_ffn(coli_model *m, coli_layer *L, float *out, const float *xn, int n) {
#ifndef COLI_HAVE_VK
    (void)m;(void)L;(void)out;(void)xn;(void)n; return 0;
#else
    if (!g_vk || !coli_vk_has_ffn(g_vk)) return 0;
    W4Side *sg = w4_slot(&L->gate), *su = w4_slot(&L->up), *sd = w4_slot(&L->down);
    if (!sg || !su || !sd) return 0;
    if (sg->gh < 0 || su->gh < 0 || sd->gh < 0) return 0;
    int64_t D = L->gate.I;
    coli_a_i8 a; a_alloc(&a, n, D);
    coli_quantize_a(&a, xn, n, D);
    int ok = coli_vk_ffn4(g_vk, sg->gh, su->gh, sd->gh, &a, out) == 0;
    a_free(&a);
    return ok;
#endif
}

/* ------------------------------------------------------------------- MoE */
/* GROUPED GEMM, and why it matters. llama.cpp sorts tokens by expert in
 * ggml_compute_forward_mul_mat_id -- and then calls a GEMV per row. Its repack
 * path is worse: forward_mul_mat_id contains a gemv<> call and NO gemm<> call at
 * all, so 64 tokens routed to one expert become 64 separate GEMVs even where a
 * GEMM kernel exists for that weight type. (Verified in a local b8252 checkout.)
 *
 * Here the tokens for an expert are gathered into one contiguous batch and put
 * through ONE coli_gemm_i8 call, so the expert's weights are read once for the
 * whole group instead of once per token -- and the batch size crosses
 * COLI_GEMM_MIN_WIDE, which is what lets the wide kernel be selected at all.
 * With top-k routing the group size is (tokens * k / n_expert) on average, so
 * this only pays off during prefill; at decode S=1 every group is 1 token and
 * the narrow kernel is correctly chosen. */
static void moe_ffn(coli_model *m, coli_layer *L, float *x, const float *xn, int S) {
    coli_cfg *c=&m->cfg;
    int D=c->hidden, EI=c->expert_inter, NE=c->n_expert, K=c->n_expert_used;
    float *logits = fal((int64_t)S*NE);
    mm(logits, xn, S, &L->router);

    int   *sel = (int*)xmal((size_t)S*K*sizeof(int));
    float *wgt = fal((int64_t)S*K);
    for (int s=0;s<S;s++) {
        const float *r = logits + (int64_t)s*NE;
        /* top-k by value, then softmax over the SELECTED logits only -- which is
         * what llama.cpp does for qwen2moe/mixtral. Softmaxing over all experts
         * first and then taking top-k gives different weights. */
        for (int k=0;k<K;k++) {
            int best=-1; float bv=-1e30f;
            for (int e=0;e<NE;e++) {
                int taken=0; for (int j=0;j<k;j++) if (sel[s*K+j]==e) { taken=1; break; }
                if (!taken && r[e]>bv) { bv=r[e]; best=e; }
            }
            sel[s*K+k]=best; wgt[s*K+k]=bv;
        }
        float mx=-1e30f; for (int k=0;k<K;k++) if (wgt[s*K+k]>mx) mx=wgt[s*K+k];
        float sum=0; for (int k=0;k<K;k++){ wgt[s*K+k]=expf(wgt[s*K+k]-mx); sum+=wgt[s*K+k]; }
        for (int k=0;k<K;k++) wgt[s*K+k]/=sum;
    }

    float *out = fal((int64_t)S*D);
    for (int64_t i=0;i<(int64_t)S*D;i++) out[i]=0.f;

    /* bucket (token,slot) pairs by expert */
    int *cnt=(int*)calloc((size_t)NE,sizeof(int));
    for (int s=0;s<S;s++) for (int k=0;k<K;k++) cnt[sel[s*K+k]]++;
    int maxc=0; for (int e=0;e<NE;e++) if (cnt[e]>maxc) maxc=cnt[e];
    if (maxc>0) {
        int   *idx = (int*)xmal((size_t)maxc*sizeof(int));   /* token index */
        int   *slt = (int*)xmal((size_t)maxc*sizeof(int));   /* which of the k slots */
        float *Xb  = fal((int64_t)maxc*D);
        float *Gb  = fal((int64_t)maxc*EI);
        float *Ub  = fal((int64_t)maxc*EI);
        float *Hb  = fal((int64_t)maxc*D);
        /* COLI_MOE_UNGROUPED=1 runs the same maths one token at a time -- the
         * reference the grouped path must equal. It perturbs the IMPLEMENTATION,
         * which is the only kind of control that can fail a differential test;
         * changing the model or the prompt would move both paths together. */
        int ungrouped = getenv("COLI_MOE_UNGROUPED") && atoi(getenv("COLI_MOE_UNGROUPED"))==1;
        for (int e=0;e<NE;e++) {
            if (!cnt[e]) continue;                     /* never touch an unused expert */
            int n=0;
            for (int s=0;s<S;s++) for (int k=0;k<K;k++)
                if (sel[s*K+k]==e) { idx[n]=s; slt[n]=k; n++; }
            if (ungrouped) {
                for (int r=0;r<n;r++) {
                    memcpy(Xb, xn+(int64_t)idx[r]*D, (size_t)D*sizeof(float));
                    mm(Gb,Xb,1,&L->e_gate[e]); mm(Ub,Xb,1,&L->e_up[e]);
                    for (int64_t i=0;i<EI;i++){ float gv=Gb[i]; Gb[i]=(gv/(1.f+expf(-gv)))*Ub[i]; }
                    mm(Hb,Gb,1,&L->e_down[e]);
                    float w=wgt[idx[r]*K+slt[r]]; float *o=out+(int64_t)idx[r]*D;
                    for (int i=0;i<D;i++) o[i]+=w*Hb[i];
                }
                continue;
            }
            for (int r=0;r<n;r++) memcpy(Xb+(int64_t)r*D, xn+(int64_t)idx[r]*D, (size_t)D*sizeof(float));
            mm(Gb,Xb,n,&L->e_gate[e]);      /* ONE pass over this expert's weights */
            mm(Ub,Xb,n,&L->e_up[e]);
            for (int64_t i=0;i<(int64_t)n*EI;i++){ float gv=Gb[i]; Gb[i]=(gv/(1.f+expf(-gv)))*Ub[i]; }
            mm(Hb,Gb,n,&L->e_down[e]);
            for (int r=0;r<n;r++) {
                float w = wgt[idx[r]*K + slt[r]];
                float *o = out + (int64_t)idx[r]*D;
                const float *h = Hb + (int64_t)r*D;
                for (int i=0;i<D;i++) o[i] += w*h[i];
            }
        }
        free(idx); free(slt); free(Xb); free(Gb); free(Ub); free(Hb);
    }
    for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i] += out[(int64_t)s*D+i];
    free(logits); free(sel); free(wgt); free(out); free(cnt);
}

/* ---------------------------------------------------- continuous batching ---
 * See model.h for why this exists. The shape is the whole argument: every GEMM
 * below runs with n = number of live sequences, so the weights are read ONCE per
 * decode step no matter how many sequences are decoding, and n>=COLI_GEMM_MIN_WIDE
 * additionally selects the wide kernel. Attention is the ONE part that cannot be
 * batched this way, because each sequence attends to a different KV region for a
 * different number of past tokens -- so it stays per-sequence, and that is a
 * property of attention, not a shortcut taken here. */
int coli_decode_batch(coli_model *m, coli_seq *seq, int n, float *logits) {
    coli_cfg *c=&m->cfg;
    int D=c->hidden,H=c->n_heads,KVH=c->n_kv_heads,hd=c->head_dim;
    int qD=H*hd, kvD=KVH*hd, grp=H/KVH;
    if (n<1) return -1;
    int maxpos = 0;
    for (int r=0;r<n;r++) {
        if (seq[r].slot<0 || seq[r].slot>=m->n_slots) { fprintf(stderr,"slot %d out of range [0,%d)\n",seq[r].slot,m->n_slots); return -1; }
        if (seq[r].pos<0 || seq[r].pos>=m->max_ctx)   { fprintf(stderr,"pos %d out of range\n",seq[r].pos); return -1; }
        if (seq[r].token<0 || seq[r].token>=c->vocab) { fprintf(stderr,"token %d out of range\n",seq[r].token); return -1; }
        if (seq[r].pos > maxpos) maxpos = seq[r].pos;
    }
    /* Grow BEFORE the layer loop: every layer writes at the same positions, so
     * growing part-way through would leave earlier layers at the old stride. */
    if (kv_grow(m, maxpos) != 0) { fprintf(stderr,"kv cache cannot grow to %d\n", maxpos); return -1; }
    /* KV region for (slot, layer, kv-head, t) */
    #define KVOFF(slot,h,t) ((((int64_t)(slot)*KVH + (h))*m->kv_ctx + (t))*hd)

    float *x=fal((int64_t)n*D), *xb=fal((int64_t)n*D);
    float *q=fal((int64_t)n*qD), *k=fal((int64_t)n*kvD), *v=fal((int64_t)n*kvD);
    float *att=fal((int64_t)n*qD);
    float *g=fal((int64_t)n*c->inter), *u=fal((int64_t)n*c->inter);

    for (int r=0;r<n;r++) {
        int id=seq[r].token;
        if (m->tok_embd.f) memcpy(x+(int64_t)r*D, m->tok_embd.f+(int64_t)id*D, (size_t)D*sizeof(float));
        else { const uint8_t *e=m->tok_embd.qu+(int64_t)id*D; float sc=m->tok_embd.scale[id];
               for (int i=0;i<D;i++) x[(int64_t)r*D+i]=((int)e[i]-128)*sc; }
    }

    trace("embed",-1,x,(int64_t)n*D);
    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        for (int r=0;r<n;r++) rmsnorm(xb+(int64_t)r*D,x+(int64_t)r*D,L->attn_norm,D,c->eps);
        if (l<2) trace("attn_norm",l,xb,(int64_t)n*D);
        if (!gpu_qkv(L,q,k,v,xb,n)) {
            mm(q,xb,n,&L->wq); mm(k,xb,n,&L->wk); mm(v,xb,n,&L->wv);   /* n rows, ONE weight pass */
        }
        if (l<2) trace("q",l,q,(int64_t)n*qD);
        if (L->bq) for (int r=0;r<n;r++) for (int i=0;i<qD;i++)  q[(int64_t)r*qD+i]+=L->bq[i];
        if (L->bk) for (int r=0;r<n;r++) for (int i=0;i<kvD;i++) k[(int64_t)r*kvD+i]+=L->bk[i];
        if (L->bv) for (int r=0;r<n;r++) for (int i=0;i<kvD;i++) v[(int64_t)r*kvD+i]+=L->bv[i];

        /* qwen3: RMSNorm each head's q and k BEFORE RoPE. Order matters -- after
         * RoPE the vector has been rotated and normalising it is a different
         * function. llama.cpp builds it the same way (build_qwen3moe: norm, then
         * rope). No-op when the tensors are absent, i.e. on qwen2 and llama. */
        qk_norm(m,L,q,k,n,H,KVH,hd,qD,kvD);
        for (int r=0;r<n;r++) {
            int pos=seq[r].pos;
            for (int h=0;h<H;h++)   rope_head(q+(int64_t)r*qD +h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) rope_head(k+(int64_t)r*kvD+h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) {
                memcpy(m->K[l]+KVOFF(seq[r].slot,h,pos), k+(int64_t)r*kvD+h*hd,(size_t)hd*sizeof(float));
                memcpy(m->V[l]+KVOFF(seq[r].slot,h,pos), v+(int64_t)r*kvD+h*hd,(size_t)hd*sizeof(float));
            }
        }

        float scale=1.f/sqrtf((float)hd);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int r=0;r<n;r++) for (int h=0;h<H;h++) {
            int kvh=h/grp, slot=seq[r].slot, tmax=seq[r].pos;
            const float *qv=q+(int64_t)r*qD+h*hd;
            attend_online(att+(int64_t)r*qD+h*hd, qv,
                          m->K[l]+KVOFF(slot,kvh,0), m->V[l]+KVOFF(slot,kvh,0),
                          hd, tmax, hd, scale);
        }
        if (l<2) trace("rope_q",l,q,(int64_t)n*qD);
        if (l<2) trace("att",l,att,(int64_t)n*qD);
        mm(xb,att,n,&L->wo);
        for (int r=0;r<n;r++) for (int i=0;i<D;i++) x[(int64_t)r*D+i]+=xb[(int64_t)r*D+i];
        if (l<2) trace("post_attn",l,x,(int64_t)n*D);

        for (int r=0;r<n;r++) rmsnorm(xb+(int64_t)r*D,x+(int64_t)r*D,L->ffn_norm,D,c->eps);
        if (c->n_expert>0) moe_ffn(m,L,x,xb,n);
        else {
            if (!gpu_ffn(m,L,xb,xb,n)) {
                mm(g,xb,n,&L->gate); mm(u,xb,n,&L->up);
                for (int64_t i=0;i<(int64_t)n*c->inter;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
                mm(xb,g,n,&L->down);
            }
            for (int r=0;r<n;r++) for (int i=0;i<D;i++) x[(int64_t)r*D+i]+=xb[(int64_t)r*D+i];
        }
    }
    trace("final_x",-1,x,(int64_t)n*D);
    float *fin=fal((int64_t)n*D);
    for (int r=0;r<n;r++) rmsnorm(fin+(int64_t)r*D,x+(int64_t)r*D,m->out_norm,D,c->eps);
    mm(logits,fin,n,&m->out);
    trace("logits",-1,logits,(int64_t)n*c->vocab);
    free(fin); free(x); free(xb); free(q); free(k); free(v); free(att); free(g); free(u);
    #undef KVOFF
    return 0;
}

/* Fused prefill: the whole prompt in ONE forward pass, into `slot`.
 *
 * The previous version stepped coli_decode_batch once per token. Same maths,
 * catastrophically wrong shape: it read every weight matrix once PER TOKEN
 * instead of once for the prompt. Measured on 331 tokens, qwen2.5-3b:
 * 34.5 s stepped against 7.1 s fused -- a 4.9x penalty the Go server paid on
 * every single request.
 *
 * Attention is causal WITHIN the prompt and also attends to whatever is already
 * in the slot (pos_base > 0), so this doubles as the path for extending an
 * existing sequence with several tokens at once. */
float *coli_prefill_slot(coli_model *m, int slot, const int *ids, int S) {
    coli_cfg *c=&m->cfg;
    int D=c->hidden,H=c->n_heads,KVH=c->n_kv_heads,hd=c->head_dim;
    int qD=H*hd, kvD=KVH*hd, grp=H/KVH;
    if (S<1) return NULL;
    if (slot<0 || slot>=m->n_slots) { fprintf(stderr,"slot %d out of range\n",slot); return NULL; }
    int pos_base = 0;
    if (pos_base+S > m->max_ctx) { fprintf(stderr,"prefill %d exceeds ctx %d\n",S,m->max_ctx); return NULL; }
    if (kv_grow(m, pos_base+S-1) != 0) { fprintf(stderr,"kv cache cannot grow to %d\n",pos_base+S-1); return NULL; }
    for (int i=0;i<S;i++)
        if (ids[i]<0||ids[i]>=c->vocab){ fprintf(stderr,"token %d out of range\n",ids[i]); return NULL; }

    #define KVOFF(sl,h,t) ((((int64_t)(sl)*KVH + (h))*m->kv_ctx + (t))*hd)

    float *x=fal((int64_t)S*D), *xb=fal((int64_t)S*D);
    float *q=fal((int64_t)S*qD), *k=fal((int64_t)S*kvD), *v=fal((int64_t)S*kvD);
    float *att=fal((int64_t)S*qD);
    float *g=fal((int64_t)S*c->inter), *u=fal((int64_t)S*c->inter);

    for (int s2=0;s2<S;s2++) {
        int id=ids[s2];
        if (m->tok_embd.f) memcpy(x+(int64_t)s2*D, m->tok_embd.f+(int64_t)id*D, (size_t)D*sizeof(float));
        else { const uint8_t *e=m->tok_embd.qu+(int64_t)id*D; float sc=m->tok_embd.scale[id];
               for (int i=0;i<D;i++) x[(int64_t)s2*D+i]=((int)e[i]-128)*sc; }
    }

    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        for (int s2=0;s2<S;s2++) rmsnorm(xb+(int64_t)s2*D,x+(int64_t)s2*D,L->attn_norm,D,c->eps);
        if (!gpu_qkv(L,q,k,v,xb,S)) {
            mm(q,xb,S,&L->wq); mm(k,xb,S,&L->wk); mm(v,xb,S,&L->wv);   /* ONE pass for the whole prompt */
        }
        if (L->bq) for (int s2=0;s2<S;s2++) for (int i=0;i<qD;i++)  q[(int64_t)s2*qD+i]+=L->bq[i];
        if (L->bk) for (int s2=0;s2<S;s2++) for (int i=0;i<kvD;i++) k[(int64_t)s2*kvD+i]+=L->bk[i];
        if (L->bv) for (int s2=0;s2<S;s2++) for (int i=0;i<kvD;i++) v[(int64_t)s2*kvD+i]+=L->bv[i];

        qk_norm(m,L,q,k,S,H,KVH,hd,qD,kvD);        /* qwen3, before RoPE */
        for (int s2=0;s2<S;s2++) {
            int pos=pos_base+s2;
            for (int h=0;h<H;h++)   rope_head(q+(int64_t)s2*qD +h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) rope_head(k+(int64_t)s2*kvD+h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) {
                memcpy(m->K[l]+KVOFF(slot,h,pos), k+(int64_t)s2*kvD+h*hd,(size_t)hd*sizeof(float));
                memcpy(m->V[l]+KVOFF(slot,h,pos), v+(int64_t)s2*kvD+h*hd,(size_t)hd*sizeof(float));
            }
        }

        float scale=1.f/sqrtf((float)hd);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h=0;h<H;h++) for (int s2=0;s2<S;s2++) {
            int kvh=h/grp, tmax=pos_base+s2;              /* causal */
            const float *qv=q+(int64_t)s2*qD+h*hd;
            attend_online(att+(int64_t)s2*qD+h*hd, qv,
                          m->K[l]+KVOFF(slot,kvh,0), m->V[l]+KVOFF(slot,kvh,0),
                          hd, tmax, hd, scale);
        }
        mm(xb,att,S,&L->wo);
        for (int s2=0;s2<S;s2++) for (int i=0;i<D;i++) x[(int64_t)s2*D+i]+=xb[(int64_t)s2*D+i];

        for (int s2=0;s2<S;s2++) rmsnorm(xb+(int64_t)s2*D,x+(int64_t)s2*D,L->ffn_norm,D,c->eps);
        if (c->n_expert>0) moe_ffn(m,L,x,xb,S);
        else {
            if (!gpu_ffn(m,L,xb,xb,S)) {
                mm(g,xb,S,&L->gate); mm(u,xb,S,&L->up);
                for (int64_t i=0;i<(int64_t)S*c->inter;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
                mm(xb,g,S,&L->down);
            }
            for (int s2=0;s2<S;s2++) for (int i=0;i<D;i++) x[(int64_t)s2*D+i]+=xb[(int64_t)s2*D+i];
        }
    }

    /* logits for the LAST token only -- the caller samples from that one. */
    float *fin=fal((int64_t)D);
    rmsnorm(fin,x+(int64_t)(S-1)*D,m->out_norm,D,c->eps);
    float *logits=fal((int64_t)c->vocab);
    mm(logits,fin,1,&m->out);

    free(fin); free(x); free(xb); free(q); free(k); free(v); free(att); free(g); free(u);
    #undef KVOFF
    return logits;
}

/* ---------------------------------------------------------------- forward */
float *coli_forward(coli_model *m, const int *ids, int S, int all_logits) {
    coli_cfg *c=&m->cfg;
    int D=c->hidden,H=c->n_heads,KVH=c->n_kv_heads,hd=c->head_dim;
    int qD=H*hd, kvD=KVH*hd, grp=H/KVH, pos0=m->n_past;
    if (pos0+S > m->max_ctx) { fprintf(stderr,"context overflow: %d+%d > %d\n",pos0,S,m->max_ctx); return NULL; }
    if (kv_grow(m, pos0+S-1) != 0) { fprintf(stderr,"kv cache cannot grow to %d\n",pos0+S-1); return NULL; }

    float *x=fal((int64_t)S*D);
    for (int s=0;s<S;s++) {
        int id=ids[s];
        if (id<0||id>=c->vocab){ fprintf(stderr,"token %d out of range [0,%d)\n",id,c->vocab); free(x); return NULL; }
        if (m->tok_embd.f) memcpy(x+(int64_t)s*D, m->tok_embd.f+(int64_t)id*D, (size_t)D*sizeof(float));
        else { const uint8_t *r=m->tok_embd.qu+(int64_t)id*D; float sc=m->tok_embd.scale[id];
               for (int i=0;i<D;i++) x[(int64_t)s*D+i]=((int)r[i]-128)*sc; }
    }
    float *xb=fal((int64_t)S*D);
    float *q=fal((int64_t)S*qD),*k=fal((int64_t)S*kvD),*vv=fal((int64_t)S*kvD);
    float *att=fal((int64_t)S*qD);
    float *g=fal((int64_t)S*c->inter),*u=fal((int64_t)S*c->inter);

    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->attn_norm,D,c->eps);
        if (!gpu_qkv(L,q,k,vv,xb,S)) {
            mm(q,xb,S,&L->wq); mm(k,xb,S,&L->wk); mm(vv,xb,S,&L->wv);
        }
        if (L->bq) for (int s=0;s<S;s++) for (int i=0;i<qD;i++)  q[(int64_t)s*qD+i]+=L->bq[i];
        if (L->bk) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) k[(int64_t)s*kvD+i]+=L->bk[i];
        if (L->bv) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) vv[(int64_t)s*kvD+i]+=L->bv[i];

        qk_norm(m,L,q,k,S,H,KVH,hd,qD,kvD);        /* qwen3, before RoPE */
        for (int s=0;s<S;s++) {
            int pos=pos0+s;
            for (int h=0;h<H;h++)   rope_head(q+(int64_t)s*qD +h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) rope_head(k+(int64_t)s*kvD+h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
        }
        for (int s=0;s<S;s++) for (int h=0;h<KVH;h++) {
            int64_t row=(int64_t)h*m->kv_ctx+(pos0+s);
            memcpy(m->K[l]+row*hd,k +(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
            memcpy(m->V[l]+row*hd,vv+(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
        }
        float scale=1.f/sqrtf((float)hd);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h=0;h<H;h++) for (int s=0;s<S;s++) {
            int kvh=h/grp;                      /* GQA: many q heads share one kv head */
            const float *qv=q+(int64_t)s*qD+h*hd;
            int tmax=pos0+s;                    /* causal */
            attend_online(att+(int64_t)s*qD+h*hd, qv,
                          m->K[l]+(int64_t)kvh*m->kv_ctx*hd,
                          m->V[l]+(int64_t)kvh*m->kv_ctx*hd,
                          hd, tmax, hd, scale);
        }
        mm(xb,att,S,&L->wo);
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i]+=xb[(int64_t)s*D+i];

        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->ffn_norm,D,c->eps);
        if (c->n_expert > 0) {
            moe_ffn(m,L,x,xb,S);
        } else {
            if (!gpu_ffn(m,L,xb,xb,S)) {
                mm(g,xb,S,&L->gate); mm(u,xb,S,&L->up);
                for (int64_t i=0;i<(int64_t)S*c->inter;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
                mm(xb,g,S,&L->down);
            }
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i]+=xb[(int64_t)s*D+i];
        }
    }

    int rows = all_logits ? S : 1;
    float *fin=fal((int64_t)rows*D);
    for (int r=0;r<rows;r++){ int s = all_logits ? r : S-1;
        rmsnorm(fin+(int64_t)r*D,x+(int64_t)s*D,m->out_norm,D,c->eps); }
    float *logits=fal((int64_t)rows*c->vocab);
    mm(logits,fin,rows,&m->out);

    free(fin); free(x); free(xb); free(q); free(k); free(vv); free(att); free(g); free(u);
    m->n_past += S;
    return logits;
}
