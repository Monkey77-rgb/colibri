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
/* ------------------------------------------------------------- prefix cache
 * ON by default; COLI_NO_PREFIX_CACHE=1 or coli_prefix_cache_enable(0) turns it
 * off so the two can be A/B'd in one binary. A correctness-neutral optimisation
 * that ships disabled is one nobody ever measures. */
static int g_prefix_cache = -1;                 /* -1 = not yet read from env */
static int g_prefix_reused = 0, g_prefix_asked = 0;

static int prefix_cache_on(void) {
    if (g_prefix_cache < 0) g_prefix_cache = getenv("COLI_NO_PREFIX_CACHE") ? 0 : 1;
    return g_prefix_cache;
}
void coli_prefix_cache_enable(int on){ g_prefix_cache = on ? 1 : 0; }
int  coli_prefix_reused(const coli_model *m){ (void)m; return g_prefix_reused; }
int  coli_prefix_asked (const coli_model *m){ (void)m; return g_prefix_asked;  }

/* Grow the per-slot id record alongside the KV it describes. If this ever fails
 * the cache is DROPPED rather than left describing memory it no longer matches:
 * a stale id list is worse than no cache, because it produces a confident wrong
 * prefix match and silently corrupts the output. */
#ifdef COLI_BREAK_CACHE_GROW
/* Negative control (see tests/test_prefix_cache_broken). Forces the allocation
 * in cache_ids_grow to fail on the Nth call so the OOM path is REACHED, which
 * is otherwise unreachable under Linux overcommit. Never defined in a shipping
 * build; `make test` requires the control build to fail. */
static int g_break_cache_grow_after = 0;
void coli_break_cache_grow_after(int n){ g_break_cache_grow_after = n; }
static int break_grow_now(void){
    if (g_break_cache_grow_after <= 0) return 0;
    return (--g_break_cache_grow_after == 0);
}
#else
#define break_grow_now() 0
#endif

static int cache_ids_grow(coli_model *m, int want) {
    if (!m->cache_ids) return 0;
    for (int sl = 0; sl < m->n_slots; sl++) {
        int *ni = break_grow_now() ? NULL : (int*)malloc((size_t)want * sizeof(int));
        if (!ni) {
            /* DROP the cache, do not merely forget its lengths. Zeroing cache_len
             * stops the MATCH loop from reading a stale row, but the RECORD memcpy
             * is not guarded by cache_len -- so a row left allocated at the old,
             * smaller width is a heap overflow waiting for the next prompt that
             * fits the new kv_ctx. Free every row and NULL it: both guards test
             * cache_ids[slot], so a NULL row is unreachable by construction
             * rather than by argument. cache_cap goes to 0 with them.
             * Self-heals: the next successful grow reallocates every row. */
#ifdef COLI_CACHE_UNSAFE_GUARD
            /* The pre-fix behaviour, kept compilable so the control build can
             * demonstrate the overflow. Zeroes the lengths only: rows at and
             * after the failing index keep their OLD width while kv_ctx is
             * published at the new one. Never define this outside tests. */
            for (int q = 0; q < m->n_slots; q++) m->cache_len[q] = 0;
            return -1;
#else
            for (int q = 0; q < m->n_slots; q++) {
                free(m->cache_ids[q]);
                m->cache_ids[q] = NULL;
                m->cache_len[q] = 0;
            }
            m->cache_cap = 0;
            return -1;
#endif
        }
        if (m->cache_ids[sl] && m->cache_len[sl] > 0)
            memcpy(ni, m->cache_ids[sl], (size_t)m->cache_len[sl] * sizeof(int));
        free(m->cache_ids[sl]);
        m->cache_ids[sl] = ni;
    }
    m->cache_cap = want;   /* published only after EVERY row actually grew */
    return 0;
}

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
    /* The id record describes the KV, so it grows with it. The KV grow above has
     * already succeeded by this point, so a cache failure must NOT fail kv_grow --
     * the cache is an optimisation and the KV is not.
     *
     * The previous comment here claimed growing "BEFORE kv_ctx is published"
     * meant a failure could not leave the two disagreeing. That was wrong, and
     * it was the bug: on failure the rows kept their OLD width while kv_ctx was
     * published at the NEW one, and the record memcpy is bounded by kv_ctx. The
     * ordering was never what made it safe. What makes it safe is that
     * cache_ids_grow now frees and NULLs every row on failure, and that the
     * record site bounds itself with cache_cap -- the allocation's own width. */
    if (cache_ids_grow(m, want) != 0) { /* cache dropped; KV valid; kv_ctx still advances */ }
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
    /* Prefix-cache id record, one array per slot, sized like the KV it
     * describes. Costs 4 bytes per cached token per slot -- 2 KiB per slot at
     * the initial 512 ctx, against ~57 KiB per token of actual KV. If it cannot
     * be allocated the engine runs without the cache rather than failing to
     * load: this is an optimisation, and an optimisation must never be the
     * reason a model will not start. */
    m->cache_ids = (int**)calloc((size_t)m->n_slots, sizeof(int*));
    m->cache_len = (int*)calloc((size_t)m->n_slots, sizeof(int));
    if (m->cache_ids && m->cache_len) {
        for (int sl=0; sl<m->n_slots; sl++) {
            m->cache_ids[sl] = (int*)malloc((size_t)m->kv_ctx*sizeof(int));
            if (!m->cache_ids[sl]) {
                for (int q=0;q<sl;q++) free(m->cache_ids[q]);
                free(m->cache_ids); free(m->cache_len);
                m->cache_ids=NULL; m->cache_len=NULL;
                break;
            }
        }
    } else { free(m->cache_ids); free(m->cache_len); m->cache_ids=NULL; m->cache_len=NULL; }
    m->cache_cap = m->cache_ids ? m->kv_ctx : 0;
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

/* Where the weights actually ended up. Reported from the backend rather than
 * inferred from the flags we passed in: the request and the grant are not the
 * same thing, and on an integrated GPU the DEVICE_LOCAL request may silently
 * fall back. Callers print this so an A/B cannot be run blind. */
void coli_gpu_meminfo(char *out, size_t cap) {
#ifdef COLI_HAVE_VK
    /* Name the KERNEL as well as the memory: "the device supports DP4a" and
     * "this run used DP4a" are different claims, and only the second explains a
     * timing. The scalar path is a silent fallback otherwise. */
    if (g_vk) { snprintf(out, cap, "%s -- %s [int4 kernel: %s]",
                         coli_vk_memdesc2(g_vk), coli_vk_memdesc(g_vk),
                         coli_vk_dot_used(g_vk) ? "DP4a" : "scalar"); return; }
#endif
    snprintf(out, cap, "no gpu");
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
    if (m->cache_ids) for (int sl=0;sl<m->n_slots;sl++) free(m->cache_ids[sl]);
    free(m->cache_ids); free(m->cache_len);
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

/* The (c,s) pairs depend ONLY on (pos, i, theta, hd, ff) -- never on the data --
 * so every head at a given position uses the identical table. rope_head computed
 * it per HEAD, which on qwen2.5-7b is 28 q heads + 4 kv heads = 32 rebuilds of
 * the same 64 coli_pow + coli_sincos pairs, per token.
 *
 * Measured 2026-08-20: rope+qknorm was 9,387 ms of a 98.2 s decode on the Legion
 * (37.3% of CPU time) and 1,191 ms on the desktop. The transcendentals dominate
 * it, and 31/32 of them were redundant.
 *
 * Building the table once and reusing it is bit-exact BY CONSTRUCTION -- the same
 * coli_pow/coli_sincos values, computed once instead of 32 times. That matters
 * here specifically: these are the custom double-precision routines that exist
 * because libm sinf differs by 1 ULP across platforms and that bit flips a
 * near-tie argmax. A GPU rope could not reproduce them, which is why this is a
 * caching change and not an offload. */
#define COLI_ROPE_MAXHALF 512
typedef struct { float c[COLI_ROPE_MAXHALF], s[COLI_ROPE_MAXHALF]; int half; } coli_rope_tab;

static void rope_table(coli_rope_tab *t, int pos, int hd, float theta, const float *ff) {
    int half = hd/2; if (half > COLI_ROPE_MAXHALF) half = COLI_ROPE_MAXHALF;
    t->half = half;
    for (int i=0;i<half;i++) {
        double fr = coli_pow((double)theta, -(double)(2*i)/(double)hd);
        if (ff) fr /= (double)ff[i];
        double sd, cd; coli_sincos((double)pos*fr, &sd, &cd);
        t->c[i] = (float)cd; t->s[i] = (float)sd;
    }
}

/* Same rotation as rope_head, reading the prebuilt table. */
static void rope_head_tab(float *v, const coli_rope_tab *t, int hd, int neox) {
    int half = t->half;
    for (int i=0;i<half;i++) {
        float c = t->c[i], s = t->s[i];
        int ia = neox ? i : 2*i, ib = neox ? i+half : 2*i+1;
        float a=v[ia], b=v[ib];
        v[ia]=a*c-b*s; v[ib]=a*s+b*c;
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
/* GPU attention, with the device K/V cache kept in step with the host one.
 *
 * Returns 0 to mean "not done, use the CPU path", exactly like gpu_qkv and
 * gpu_ffn. Every refusal below is a real one: no device, no shader, a batch
 * wider than the staging ring, or a host cache that has grown since the device
 * buffers were built. None of them is a silent partial -- the CPU path runs the
 * whole layer and stays the reference.
 *
 * The kv_ctx comparison is the load-bearing check. When the host cache grows it
 * DOUBLES kv_ctx and re-strides every row, so the same data lives at a new
 * address. A device copy built for the old stride would still index, still
 * return floats, and still produce text -- reading whatever now occupies those
 * offsets. That is the failure this rebuild exists to prevent, and it is why
 * the check is on the stride rather than on a dirty flag someone has to
 * remember to set. */
/* Decided ONCE per batch, before the layer loop, and not per layer: it may
 * allocate and refill the whole device cache, and the answer cannot change
 * between layer 0 and layer 27 of the same token. Separated from gpu_attn for
 * a second reason -- staging happens during the rope loop, BEFORE the dispatch,
 * so whether to stage has to be known first. Folding the two together made
 * readiness depend on a call that only happens once staging has succeeded. */
/* OFF BY DEFAULT, and the measurement says why rather than taste.
 *
 * Wired up and numerically correct -- TF-NLL 2.6758 against the CPU path's
 * 2.6768, the expected non-bit-exact difference -- but as a SEPARATE dispatch it
 * is a net loss: decode wall 13.3 s -> 15.7 s, submissions 57,885 -> 76,953,
 * because it adds a fourth fence per layer without yet removing any of the three
 * that were already there.
 *
 * Measured cost of the dispatch alone (RTX 4070, 2026-08-22, tests/test_vk_attn):
 *     tmax=699   260.1 us   38.28 MiB of K+V   154.3 GB/s
 *     tmax= 63    86.9 us    3.50 MiB          42.2 GB/s
 * which separates to ~69 us of FIXED per-dispatch cost and ~210 GB/s marginal.
 * 69 us is five times the 13.7 us submit floor; it is the q upload, the output
 * download and four map/unmap pairs -- precisely the costs that disappear when
 * the layer becomes one command buffer with activations left resident. The
 * marginal rate reflects 7 query heads each re-reading the same kv head's cache.
 *
 * So this turns on when qkv -> rope -> attention -> o_proj are ONE submission,
 * not before. Until then it stays behind COLI_GPU_ATTN=1 so it can be measured
 * without regressing anyone, and the CPU path remains both the default and the
 * numerical reference. Read once into a static: getenv in a per-layer path cost
 * 4% of every decode the last time it went in a hot loop (52425cc). */
static int gpu_attn_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_GPU_ATTN"); on = (e && *e && *e!='0') ? 1 : 0; }
    return on;
}

/* ---- the fused attention block -------------------------------------------
 *
 * When this is on, ONE submission per layer does qkv, bias, RoPE, the KV
 * scatter, attention, the requantization and o_proj. See coli_vk_attn_block.
 *
 * THE DEVICE KV CACHE BECOMES AUTHORITATIVE, and that is the load-bearing
 * consequence. k and v are produced on the GPU and written to the device cache
 * by kvwrite.comp; they are never on the host, so m->K/m->V go stale from the
 * first token this path handles. Falling back to CPU attention afterwards would
 * read a cache missing every row since then and produce confident garbage, so
 * g_block_stale below makes that a loud refusal instead. This is the same
 * trade llama.cpp makes; what is NOT acceptable is making it silently.
 *
 * NOT FOR qwen3: it RMSNorms q and k between the bias and the rotation and no
 * kernel here does that. Checked per layer, not assumed from the arch string.
 */
static int g_block_stale = 0;      /* device KV has rows the host cache lacks */

static int block_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_GPU_BLOCK"); on = (e && *e && *e!='0') ? 1 : 0; }
    return on;
}

#ifdef COLI_HAVE_VK
/* Every layer's qkv bias in ONE buffer, uploaded once. Per-layer uploads would
 * be 28 map/unmap pairs per token, and mapping is what the 69 us fixed dispatch
 * cost measured on 2026-08-22 turned out to be. Returns the per-layer stride, or
 * 0 if any layer is missing a bias -- in which case the block is not used at all
 * rather than used with a hole. */
static int block_bias_upload(coli_model *m, int qD, int kvD) {
    static int stride = -1;
    if (stride >= 0) return stride;
    stride = 0;
    int L = m->cfg.n_layers, per = qD + 2*kvD;
    for (int l=0;l<L;l++) if (!m->L[l].bq || !m->L[l].bk || !m->L[l].bv) return 0;
    float *all = (float*)malloc((size_t)L*per*sizeof(float));
    if (!all) return 0;
    for (int l=0;l<L;l++) {
        memcpy(all+(size_t)l*per,            m->L[l].bq, (size_t)qD*sizeof(float));
        memcpy(all+(size_t)l*per+qD,         m->L[l].bk, (size_t)kvD*sizeof(float));
        memcpy(all+(size_t)l*per+qD+kvD,     m->L[l].bv, (size_t)kvD*sizeof(float));
    }
    int ok = coli_vk_rope_bias_upload(g_vk, all, (size_t)L*per) == 0;
    free(all);
    stride = ok ? per : 0;
    return stride;
}
#endif

/* Bring the host cache back up to date from the device, at the CURRENT stride.
 * Must run BEFORE kv_grow: growth re-lays the host cache out and copies the old
 * (stale) contents forward, so afterwards the block's rows are unrecoverable. */
static int block_sync_to_host(coli_model *m) {
#ifndef COLI_HAVE_VK
    (void)m; return 1;
#else
    if (!g_block_stale) return 1;
    if (!g_vk || !coli_vk_kv_ready(g_vk)) return 1;
    if (coli_vk_kv_ctx(g_vk) != m->kv_ctx) return 0;   /* strides already diverged */
    for (int l=0;l<m->cfg.n_layers;l++)
        if (coli_vk_kv_get(g_vk, l, m->K[l], m->V[l]) != 0) return 0;
    g_block_stale = 0;
    return 1;
#endif
}

static int gpu_block_ready(coli_model *m, int H, int KVH, int hd, int n) {
#ifndef COLI_HAVE_VK
    (void)m;(void)H;(void)KVH;(void)hd;(void)n; return 0;
#else
    if (!block_enabled()) return 0;
    if (!g_vk || !coli_vk_has_block(g_vk)) return 0;
    if (n > 32 || (H % KVH) || hd > 256 || (hd % 32)) return 0;
    if (g_calib) return 0;
    for (int l=0;l<m->cfg.n_layers;l++) {
        coli_layer *L=&m->L[l];
        if (L->q_norm || L->k_norm) return 0;                 /* qwen3: see header */
        if (L->wq.f || L->wk.f || L->wv.f || L->wo.f) return 0;
        W4Side *a=w4_slot(&L->wq),*b=w4_slot(&L->wk),*c2=w4_slot(&L->wv),*d=w4_slot(&L->wo);
        if (!a||!b||!c2||!d||a->gh<0||b->gh<0||c2->gh<0||d->gh<0) return 0;
    }
    /* The device cache must already hold everything the host cache does. Once
     * the block has run, that direction is reversed and reloading from the host
     * would DELETE rows -- hence the refusal rather than a re-init. */
    if (coli_vk_kv_ready(g_vk) && coli_vk_kv_ctx(g_vk) == m->kv_ctx) return 1;
    if (g_block_stale) {
        /* Reached only if block_sync_to_host() did not run or failed. Reloading
         * the device cache from a host cache that is behind it would delete every
         * row the block wrote, so this stays a refusal rather than a re-init. */
        fprintf(stderr,"fused block: device KV is ahead of the host cache and the stride "
                       "changed without a sync -- refusing to reload and lose rows.\n");
        return 0;
    }
    if (coli_vk_kv_init(g_vk, m->cfg.n_layers, m->n_slots, KVH, m->kv_ctx, hd) != 0) return 0;
    for (int li=0; li<m->cfg.n_layers; li++)
        if (coli_vk_kv_load(g_vk, li, m->K[li], m->V[li]) != 0) return 0;
    return 1;
#endif
}

/* Returns 1 when the layer was done entirely on the device. */
static int gpu_block(coli_model *m, coli_layer *L, int l, float *xn, float *out, int n,
                     int H, int KVH, int hd, int bias_stride, const int *meta, float scale,
                     int stop_attn) {
#ifndef COLI_HAVE_VK
    (void)m;(void)L;(void)l;(void)xn;(void)out;(void)n;(void)H;(void)KVH;(void)hd;
    (void)bias_stride;(void)meta;(void)scale;(void)stop_attn; return 0;
#else
    int64_t I = L->wq.I;
    coli_a_i8 a; a_alloc(&a,n,I);
    coli_quantize_a(&a,xn,n,I);
    int wh[4] = { w4_slot(&L->wq)->gh, w4_slot(&L->wk)->gh,
                  w4_slot(&L->wv)->gh, w4_slot(&L->wo)->gh };
    int ok = coli_vk_attn_block(g_vk, l, wh, &a, meta, n, H, KVH, hd,
                                m->cfg.rope == COLI_ROPE_NEOX,
                                bias_stride ? l*bias_stride : -1, scale, stop_attn, out) == 0;
    a_free(&a);
    if (ok) g_block_stale = 1;
    return ok;
#endif
}

static int gpu_attn_ready(coli_model *m, int KVH, int hd, int n) {
#ifndef COLI_HAVE_VK
    (void)m; (void)KVH; (void)hd; (void)n; return 0;
#else
    if (!gpu_attn_enabled()) return 0;
    if (!g_vk || !coli_vk_has_attn(g_vk)) return 0;
    if (n * KVH * 2 > 64) return 0;              /* staging ring, see kv_put */
    if (n > 32) return 0;                        /* meta[] in gpu_attn */
    if (hd > 256 || (hd % 32)) return 0;         /* the shader strides by 32 */

    if (coli_vk_kv_ready(g_vk) && coli_vk_kv_ctx(g_vk) == m->kv_ctx) return 1;

    if (coli_vk_kv_init(g_vk, m->cfg.n_layers, m->n_slots, KVH,
                        m->kv_ctx, hd) != 0) return 0;
    for (int li = 0; li < m->cfg.n_layers; li++)
        if (coli_vk_kv_load(g_vk, li, m->K[li], m->V[li]) != 0) return 0;
    return 1;
#endif
}

/* PREFILL readiness. Deliberately NOT gpu_attn_ready with the caps removed:
 * the two paths get their KV rows onto the device by different mechanisms and
 * only one of them has a 32-row ceiling.
 *
 * decode  -- one row per token per layer, staged through the 64-row ring
 *            (coli_vk_kv_put), which is where n*KVH*2 > 64 comes from.
 * prefill -- the whole run written contiguously by coli_vk_kv_write, one submit
 *            per layer, no ring and therefore no ceiling.
 *
 * kv_load on init is not optional even though prefill writes its own rows. A
 * continuation prefill has pos0 > 0, and the positions BELOW pos0 have to
 * already be resident or attention reads a freshly-zeroed device buffer for
 * them -- which is not a crash, it is a plausible-looking wrong answer. */
static int gpu_prefill_attn_ready(coli_model *m, int KVH, int hd, int S) {
#ifndef COLI_HAVE_VK
    (void)m; (void)KVH; (void)hd; (void)S; return 0;
#else
    /* DEFAULT: on for a DISCRETE GPU, off for an INTEGRATED one. Measured, not
     * guessed, and the two devices disagree so completely that one default for
     * both would be wrong somewhere.
     *
     * Prefill attention, n rows x 28 heads, causal ramp, kv_ctx 1024, 16 threads
     * (tests/test_vk_attn, 2026-08-27). GPU arm is the full production call --
     * upload, submit, fence, download:
     *
     *     n      RTX 4070 (discrete)      Radeon 780M (integrated)
     *     96     7.96x faster             1.59x faster
     *    192     5.65x faster             1.31x SLOWER
     *    384     4.48x faster             1.04x SLOWER
     *    683     4.85x faster             1.15x SLOWER  (1.15/1.17/1.10, n=3)
     *
     * The mechanism is not a mystery and it is why deviceType is the right
     * discriminator: an iGPU shares ONE memory controller with the CPU it is
     * competing against, so a bandwidth-bound stage moved onto it spends
     * exactly the bandwidth the CPU arm would have used. The 4070 has its own.
     *
     * Two devices is two devices, so this is a heuristic and it says so.
     * COLI_GPU_PREFILL_ATTN=1 forces it on, =0 forces it off, and measuring a
     * new device before trusting either is the whole point of the table above.
     *
     * Empty is OFF, not on. `COLI_GPU_PREFILL_ATTN= ./coli-gpu` exports the
     * variable with an empty value and a bare getenv() != NULL treats that as
     * enabled -- which is exactly how the first A/B of this feature ran both
     * arms with the GPU on and would have reported a 4.3x win as noise. */
    { const char *e = getenv("COLI_GPU_PREFILL_ATTN");
      if (e && *e) { if (!strcmp(e,"0")) return 0; }
      else if (coli_vk_is_integrated(g_vk)) return 0; }
    if (!g_vk || !coli_vk_has_attn(g_vk)) return 0;
    if (hd > 256 || (hd % 32)) return 0;              /* the shader strides by 32 */
    if (m->cfg.n_heads % KVH) return 0;
    if (S < 2) return 0;                             /* decode has its own path */

    if (coli_vk_kv_ready(g_vk) && coli_vk_kv_ctx(g_vk) == m->kv_ctx) return 1;
    if (coli_vk_kv_init(g_vk, m->cfg.n_layers, m->n_slots, KVH,
                        m->kv_ctx, hd) != 0) return 0;
    for (int li = 0; li < m->cfg.n_layers; li++)
        if (coli_vk_kv_load(g_vk, li, m->K[li], m->V[li]) != 0) return 0;
    return 1;
#endif
}

/* Bulk-write one layer's KV rows. Wrapper so the prefill loop carries no #ifdef
 * and the CPU-only build compiles it away, same as gpu_kv_stage.
 *
 * ⚠️ CALLERS PASS pos0=0 AND count=pos0+S, i.e. they rewrite EVERY position up
 * to the end of this batch, not only the new ones. That is not redundancy, it
 * closes a staleness class with no cheap detector: the device cache can fall
 * behind the host whenever anything wrote a row without going through Vulkan --
 * decode running with COLI_GPU_ATTN unset, or a single layer of a previous
 * prefill falling back to the CPU after its write failed. Attention over a
 * stale row is not a crash, it is a fluent wrong answer, and nothing downstream
 * would flag it. Rewriting from 0 makes every prefill self-healing.
 *
 * The cost is bounded by the USED context, not by kv_ctx, and it is zero at
 * pos0=0 where it is already the whole batch. This is also why the readiness
 * check refuses S < 2: at S=1 this would be O(context) per generated token. */
/* The same write, for a slot other than 0. coli_prefill_slot passes the slot's
 * own base pointer, so the K/V argument is already offset; the slot index still
 * has to reach the device because the DEVICE cache is [slot][kvh][kv_ctx][hd]
 * and the host base pointer carries no slot information the backend can see. */
static int gpu_kv_write_slot(int l, int slot, int pos0, int count,
                             const float *K, const float *V) {
#ifndef COLI_HAVE_VK
    (void)l; (void)slot; (void)pos0; (void)count; (void)K; (void)V; return 0;
#else
    if (!g_vk) return 0;
    return coli_vk_kv_write(g_vk, l, slot, pos0, count, K, V) == 0;
#endif
}

static int gpu_kv_write(int l, int pos0, int count, const float *K, const float *V) {
#ifndef COLI_HAVE_VK
    (void)l; (void)pos0; (void)count; (void)K; (void)V; return 0;
#else
    if (!g_vk) return 0;
    /* slot 0: coli_forward indexes m->K[l] as [kvh][kv_ctx][hd] with no slot
     * term, so prefill is slot 0 by construction. */
    return coli_vk_kv_write(g_vk, l, 0, pos0, count, K, V) == 0;
#endif
}

/* Stage one K and one V row for the device cache. Split out so the decode loop
 * carries no #ifdef and the CPU-only build compiles it away to nothing. */
static int gpu_kv_stage(int slot, int kvh, int pos, const float *krow, const float *vrow) {
#ifndef COLI_HAVE_VK
    (void)slot; (void)kvh; (void)pos; (void)krow; (void)vrow; return 0;
#else
    if (!g_vk) return 0;
    return coli_vk_kv_put(g_vk, slot, kvh, pos, 0, krow) == 0
        && coli_vk_kv_put(g_vk, slot, kvh, pos, 1, vrow) == 0;
#endif
}

static int gpu_attn(coli_model *m, int l, const float *q, float *att,
                    int n, int H, float scale, const int *slots, const int *poss) {
    (void)m;
#ifndef COLI_HAVE_VK
    (void)l; (void)q; (void)att; (void)n; (void)H; (void)scale; (void)slots; (void)poss;
    return 0;
#else
    /* meta USED TO BE int[64], which silently made 32 rows the ceiling and was
     * one of the two host-side reasons the GPU attention kernel could not be
     * used for prefill. The kernel itself never had that limit: a workgroup is
     * one (row, head) and meta[r] carries this row's own tmax, which IS causal
     * prefill. Small batches still use the stack; only prefill allocates. */
    int stack[64], *meta = stack;
    if (n * 2 > 64) { meta = (int*)malloc((size_t)n * 2 * sizeof(int)); if (!meta) return 0; }
    for (int r = 0; r < n; r++) { meta[r*2] = slots[r]; meta[r*2+1] = poss[r]; }
    int ok = coli_vk_attn(g_vk, l, q, att, meta, n, H, scale) == 0;
    if (meta != stack) free(meta);
    return ok;
#endif
}

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
/* CPU-SIDE PHASE TIMERS for coli_forward SPECIFICALLY. The vk profile accounted
 * for 14.6 s of a 20.0 s decode; the missing 5.4 s -- 27%, the largest single
 * block -- was absent from every report, and an absent stage reads as free.
 *
 * coli_decode_batch is instrumented, because that is what --nll1 AND the Go
 * scheduler actually call -- verified by the dump reporting "nothing timed" when
 * only coli_forward carried the timers. coli_forward and coli_prefill_slot have
 * their own copies of this loop and are NOT timed; the dump says so rather than
 * printing zeros, because an untimed stage must not read as a free one. */
#include <ctime>
#include <cstdio>
struct coli_cpu_prof { double norm_s, rope_s, kvcopy_s, attn_s; unsigned long long fwd_n; };
static struct coli_cpu_prof CP;
static double cp_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                            return t.tv_sec + 1e-9*t.tv_nsec; }
/* PREFILL profile. coli_cpu_prof above times coli_decode_batch ONLY, which is
 * why a 683-token prefill costing 9.86 s had no breakdown at all -- the timed
 * region did not include it. A profiler that cannot see the phase you are
 * investigating reports nothing and looks like it reported nothing wrong.
 *
 * Prefill is a different shape from decode: ONE call with S rows, so the GEMMs
 * are batched and attention is O(S^2). Which of those dominates is exactly what
 * two falsified guesses (a bigger GPU tile, more CPU threads) failed to settle
 * -- tile 1->4 moved 683 tokens only 11.50 s -> 9.86 s, and threads 2->8 only
 * 15.68 s -> 10.19 s, so neither the GPU's weight reuse nor CPU parallelism is
 * the limiter. Measured 2026-08-23. */
struct coli_pf_prof { double norm, qkv, rope, kvcopy, attn, wo, ffn, head; unsigned long long n; };
static struct coli_pf_prof PF;

extern "C" void coli_prefill_prof_dump(FILE *f) {
    double tot = PF.norm+PF.qkv+PF.rope+PF.kvcopy+PF.attn+PF.wo+PF.ffn+PF.head;
    if (tot <= 0) { fprintf(f,"\nprefill-prof: nothing timed (coli_forward never ran)\n"); return; }
    fprintf(f,"\n--- prefill phase breakdown, coli_forward only (%llu calls) ---\n", PF.n);
    #define PFL(name,v) fprintf(f,"  %-14s %9.1f ms  %5.1f%%\n", name, (v)*1e3, 100*(v)/tot)
    PFL("rmsnorm",  PF.norm);
    PFL("qkv gemm", PF.qkv);
    PFL("rope",     PF.rope);
    PFL("kv copy",  PF.kvcopy);
    PFL("attention",PF.attn);
    PFL("o_proj",   PF.wo);
    PFL("ffn",      PF.ffn);
    PFL("logit head",PF.head);
    #undef PFL
    fprintf(f,"  timed tot     %9.1f ms\n", tot*1e3);
    fprintf(f,"  NOTE: these eight are ALL of coli_forward's per-layer work plus the\n"
              "        head. Embedding lookup and the residual adds are not timed.\n");
}

extern "C" void coli_cpu_prof_dump(FILE *f) {
    double tot = CP.norm_s+CP.rope_s+CP.kvcopy_s+CP.attn_s;
    if (tot <= 0) { fprintf(f,"\ncpu-prof: nothing timed (coli_decode_batch never ran)\n"); return; }
    fprintf(f,"\n--- cpu phase breakdown, coli_decode_batch only (%llu calls) ---\n", CP.fwd_n);
    fprintf(f,"  rmsnorm       %9.1f ms  %5.1f%%\n", CP.norm_s*1e3, 100*CP.norm_s/tot);
    fprintf(f,"  rope+qknorm   %9.1f ms  %5.1f%%\n", CP.rope_s*1e3, 100*CP.rope_s/tot);
    fprintf(f,"  kv cache copy %9.1f ms  %5.1f%%\n", CP.kvcopy_s*1e3, 100*CP.kvcopy_s/tot);
    fprintf(f,"  attention     %9.1f ms  %5.1f%%\n", CP.attn_s*1e3, 100*CP.attn_s/tot);
    fprintf(f,"  timed tot     %9.1f ms\n", tot*1e3);
    fprintf(f,"  NOTE: ONLY these four stages are timed. Sampling, the logit head,\n"
              "        bias adds and FFN glue are NOT here and are not zero.\n");
}

int coli_decode_batch(coli_model *m, coli_seq *seq, int n, float *logits) {
    CP.fwd_n++;
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
    /* Before the host cache can be re-strided, pull back anything the fused
     * block wrote straight to the device. Skipping this is not a slow path, it
     * is a wrong answer: measured 2026-08-23, growth mid-sequence turned a
     * TF-NLL of 2.6768 into 13.6551 because attention then read a cache with a
     * hole for every block-written token. */
    if (maxpos >= m->kv_ctx && !block_sync_to_host(m)) {
        fprintf(stderr,"fused block: could not read the device KV cache back before "
                       "growing the host one; refusing to continue with a stale cache.\n");
        return -1;
    }
    if (kv_grow(m, maxpos) != 0) { fprintf(stderr,"kv cache cannot grow to %d\n", maxpos); return -1; }

    /* AFTER kv_grow, never before: growth doubles kv_ctx and re-strides every
     * row, and a device cache built against the pre-growth stride would index
     * the wrong offsets while still returning plausible floats. */
    int kv_use = gpu_attn_ready(m, KVH, hd, n);
    /* The fused block subsumes GPU attention, so it is decided first and wins.
     * The (c,s) table is uploaded ONCE here, not per layer: it depends only on
     * position, and every layer rotates by the same angles. Uploading it inside
     * the layer loop would be 28 map/unmap pairs per token for identical bytes. */
    int blk_use = gpu_block_ready(m, H, KVH, hd, n);
    /* Say so, once. A path that silently does not engage looks exactly like a
     * path that engaged and changed nothing -- and the first run of this block
     * reported the CPU-attention NLL, which is how the difference was noticed. */
    if (block_enabled()) { static int said=0;
        if (!said) { said=1; fprintf(stderr,"fused block: %s\n",
            blk_use ? "ENGAGED" : "NOT engaged (see gpu_block_ready)"); } }
    int blk_bias = 0;
#ifdef COLI_HAVE_VK
    if (blk_use) {
        blk_bias = block_bias_upload(m, qD, kvD);
        int half = hd/2;
        float *cs = (float*)malloc((size_t)n*half*2*sizeof(float));
        if (!cs) blk_use = 0;
        else {
            for (int r=0;r<n;r++) {
                coli_rope_tab rt; rope_table(&rt, seq[r].pos, hd, c->rope_theta, m->rope_ff);
                for (int i=0;i<half;i++){ cs[((size_t)r*half+i)*2]=rt.c[i]; cs[((size_t)r*half+i)*2+1]=rt.s[i]; }
            }
            if (coli_vk_rope_cs_upload(g_vk, cs, (size_t)n*half*2) != 0) blk_use = 0;
            free(cs);
        }
    }
#endif
    int blk_meta[64];
    for (int r=0;r<n && r<32;r++){ blk_meta[r*2]=seq[r].slot; blk_meta[r*2+1]=seq[r].pos; }
    if (blk_use) kv_use = 0;
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
        int kv_staged = kv_use;      /* per layer: one failed row disables this layer only */
        { double t_=cp_now();
          for (int r=0;r<n;r++) rmsnorm(xb+(int64_t)r*D,x+(int64_t)r*D,L->attn_norm,D,c->eps);
          CP.norm_s += cp_now()-t_; }
        if (l<2) trace("attn_norm",l,xb,(int64_t)n*D);

        /* ONE submission for the whole attention half of the layer. A failure
         * here is not silently absorbed: the device cache already holds rows the
         * host does not, so the CPU path below would attend over a cache with
         * holes. Fail loudly instead of producing a plausible answer. */
        if (blk_use) {
            /* COLI_BLOCK_STOP=attn: the block returns attention's output rather
             * than o_proj's, and the CPU finishes the layer. Bisection aid. */
            static int stopa=-1;
            if (stopa<0){ const char *e=getenv("COLI_BLOCK_STOP"); stopa=(e&&!strcmp(e,"attn"))?1:0; }
            /* COLI_BLOCK_CHECK=1: run the block BOTH ways on the same inputs --
             * once ending at attention with the CPU doing requantize+o_proj, once
             * all the way through -- and report the difference. Safe to run twice:
             * kvwrite writes the same values to the same places and attention only
             * reads. This is what located the o_proj stage as the wrong one. */
            static int chk=-1;
            if (chk<0){ const char *e=getenv("COLI_BLOCK_CHECK"); chk=(e&&*e&&*e!='0')?1:0; }
            if (chk && l<2 && n==1) {
                float *ref=(float*)xmal((size_t)n*D*sizeof(float));
                if (gpu_block(m,L,l,xb,att,n,H,KVH,hd,blk_bias,blk_meta,1.f/sqrtf((float)hd),1)) {
                    mm(ref,att,n,&L->wo);
                    float *gpu2=(float*)xmal((size_t)n*D*sizeof(float));
                    if (gpu_block(m,L,l,xb,gpu2,n,H,KVH,hd,blk_bias,blk_meta,1.f/sqrtf((float)hd),0)) {
                        double num=0,den=0;
                        for (int i=0;i<n*D;i++){ double d=gpu2[i]-ref[i]; num+=d*d; den+=(double)ref[i]*ref[i]; }
                        fprintf(stderr,"BLOCK_CHECK layer=%d  rel(o_proj gpu vs cpu)=%.3e\n",
                                l, sqrt(num/(den>0?den:1)));
                    }
                    free(gpu2);
                }
                free(ref);
            }
            float *bout = stopa ? att : xb;
            if (!gpu_block(m,L,l,xb,bout,n,H,KVH,hd,blk_bias,blk_meta,1.f/sqrtf((float)hd),stopa)) {
                fprintf(stderr,"fused block failed at layer %d; the device KV cache is "
                               "ahead of the host one, so there is no correct fallback.\n", l);
                free(x); free(xb); free(q); free(k); free(v);
                free(att); free(g); free(u);
                return -1;
            }
            if (stopa) mm(xb,att,n,&L->wo);
            for (int r=0;r<n;r++) for (int i=0;i<D;i++) x[(int64_t)r*D+i]+=xb[(int64_t)r*D+i];
            goto ffn_stage;
        }
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
        { double t_=cp_now();
        qk_norm(m,L,q,k,n,H,KVH,hd,qD,kvD);
        for (int r=0;r<n;r++) {
            int pos=seq[r].pos;
            coli_rope_tab rt; rope_table(&rt,pos,hd,c->rope_theta,m->rope_ff);
            for (int h=0;h<H;h++)   rope_head_tab(q+(int64_t)r*qD +h*hd,&rt,hd,c->rope);
            for (int h=0;h<KVH;h++) rope_head_tab(k+(int64_t)r*kvD+h*hd,&rt,hd,c->rope);
            CP.rope_s += cp_now()-t_; t_=cp_now();
            for (int h=0;h<KVH;h++) {
                memcpy(m->K[l]+KVOFF(seq[r].slot,h,pos), k+(int64_t)r*kvD+h*hd,(size_t)hd*sizeof(float));
                memcpy(m->V[l]+KVOFF(seq[r].slot,h,pos), v+(int64_t)r*kvD+h*hd,(size_t)hd*sizeof(float));
                /* Staged for the device cache too. The host write above is NOT
                 * redundant: it costs 1.1% of decode and keeps the CPU path a
                 * correct fallback and the numerical reference. If a stage
                 * fails the flag below sends the whole layer to the CPU rather
                 * than leaving a hole every later position would read. */
                if (kv_staged && !gpu_kv_stage(seq[r].slot, h, pos,
                        k+(int64_t)r*kvD+h*hd, v+(int64_t)r*kvD+h*hd))
                    kv_staged = 0;
            }
            CP.kvcopy_s += cp_now()-t_;
        } }

        {
        float scale=1.f/sqrtf((float)hd);
        double t_attn=cp_now();
        int did_gpu_attn = 0;
        if (kv_staged) {
            int sl[32], ps[32];
            for (int r=0;r<n && r<32;r++) { sl[r]=seq[r].slot; ps[r]=seq[r].pos; }
            did_gpu_attn = gpu_attn(m, l, q, att, n, H, scale, sl, ps);
        }
        if (!did_gpu_attn)
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int r=0;r<n;r++) for (int h=0;h<H;h++) {
            int kvh=h/grp, slot=seq[r].slot, tmax=seq[r].pos;
            const float *qv=q+(int64_t)r*qD+h*hd;
            attend_online(att+(int64_t)r*qD+h*hd, qv,
                          m->K[l]+KVOFF(slot,kvh,0), m->V[l]+KVOFF(slot,kvh,0),
                          hd, tmax, hd, scale);
        }
        CP.attn_s += cp_now()-t_attn;
        if (l<2) trace("rope_q",l,q,(int64_t)n*qD);
        if (l<2) trace("att",l,att,(int64_t)n*qD);
        mm(xb,att,n,&L->wo);
        for (int r=0;r<n;r++) for (int i=0;i<D;i++) x[(int64_t)r*D+i]+=xb[(int64_t)r*D+i];
        if (l<2) trace("post_attn",l,x,(int64_t)n*D);
        }

        ffn_stage:
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
    if (S > m->max_ctx) { fprintf(stderr,"prefill %d exceeds ctx %d\n",S,m->max_ctx); return NULL; }
    if (kv_grow(m, S-1) != 0) { fprintf(stderr,"kv cache cannot grow to %d\n",S-1); return NULL; }
    for (int i=0;i<S;i++)
        if (ids[i]<0||ids[i]>=c->vocab){ fprintf(stderr,"token %d out of range\n",ids[i]); return NULL; }

    /* THE PREFIX MATCH.
     *
     * How much of this prompt is already sitting in the slot's KV from the last
     * call? Cached K/V are position-encoded, and a shared PREFIX occupies the
     * same positions by construction -- so the entries for [0,P) are valid
     * unchanged and only [P,S) has to be computed.
     *
     * P is capped at S-1, never S. The caller needs logits for the final token,
     * and logits come from running that token through the stack; a "complete"
     * hit that computed nothing would have to return the PREVIOUS call's logits,
     * which is a different answer wearing the right shape. One token always runs.
     *
     * Positions [P, old_len) keep stale K/V after this, and that is safe because
     * attention at position p reads only [0,p] and prefill overwrites [P,S)
     * before anything reads there. Nothing ever reads past what it just wrote. */
    int pos_base = 0;
    g_prefix_asked = S; g_prefix_reused = 0;
    if (prefix_cache_on() && m->cache_ids && m->cache_ids[slot]) {
        int have = m->cache_len[slot];
        if (have > S-1) have = S-1;                 /* never reuse the last token */

        int P = 0;
        while (P < have && m->cache_ids[slot][P] == ids[P]) P++;
        pos_base = P;
#ifdef COLI_BREAK_PREFIX
        /* Negative control for tests/test_prefix_cache. Claims ONE token more
         * than actually matched, so the reused KV no longer corresponds to the
         * prompt and the logits must diverge. If the bit-identity assertion
         * still passes with this defined, it is not testing what it claims.
         *
         * NOTE: the first attempt at this breaker incremented `have` instead,
         * which is INERT -- the match loop is bounded by token equality, not by
         * `have`, so it still stopped at the true prefix length. A control that
         * cannot produce the opposite result proves nothing; break pos_base. */
        if (pos_base > 0 && pos_base < S-1) pos_base++;
#endif
        g_prefix_reused = pos_base;
    }
    if (pos_base > 0 && getenv("COLI_PREFIX_VERBOSE"))
        fprintf(stderr,"prefix cache: reused %d of %d tokens (%.1f%%)\n",
                pos_base, S, 100.0*pos_base/S);
    const int S0 = pos_base;                        /* first token actually computed */
    const int NT = S - S0;                          /* how many we run */

    #define KVOFF(sl,h,t) ((((int64_t)(sl)*KVH + (h))*m->kv_ctx + (t))*hd)

    float *x=fal((int64_t)NT*D), *xb=fal((int64_t)NT*D);
    float *q=fal((int64_t)NT*qD), *k=fal((int64_t)NT*kvD), *v=fal((int64_t)NT*kvD);
    float *att=fal((int64_t)NT*qD);
    float *g=fal((int64_t)NT*c->inter), *u=fal((int64_t)NT*c->inter);

    for (int s2=0;s2<NT;s2++) {
        int id=ids[S0+s2];   /* only the tokens the cache did not already hold */
        if (m->tok_embd.f) memcpy(x+(int64_t)s2*D, m->tok_embd.f+(int64_t)id*D, (size_t)D*sizeof(float));
        else { const uint8_t *e=m->tok_embd.qu+(int64_t)id*D; float sc=m->tok_embd.scale[id];
               for (int i=0;i<D;i++) x[(int64_t)s2*D+i]=((int)e[i]-128)*sc; }
    }

    int ps_gpu = gpu_prefill_attn_ready(m, KVH, hd, NT);
    int *ps_slot = NULL, *ps_pos = NULL;
    if (ps_gpu) {
        ps_slot = (int*)xmal((size_t)NT*sizeof(int));
        ps_pos  = (int*)xmal((size_t)NT*sizeof(int));
        for (int s2=0;s2<NT;s2++) { ps_slot[s2]=slot; ps_pos[s2]=pos_base+s2; }
    }
    if (ps_gpu) { static int said=0; if (!said) { said=1;
        fprintf(stderr,"  prefill GPU attention (slot path): ENGAGED\n"); } }

    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        for (int s2=0;s2<NT;s2++) rmsnorm(xb+(int64_t)s2*D,x+(int64_t)s2*D,L->attn_norm,D,c->eps);
        if (!gpu_qkv(L,q,k,v,xb,NT)) {
            mm(q,xb,NT,&L->wq); mm(k,xb,NT,&L->wk); mm(v,xb,NT,&L->wv);   /* ONE pass for the whole prompt */
        }
        if (L->bq) for (int s2=0;s2<NT;s2++) for (int i=0;i<qD;i++)  q[(int64_t)s2*qD+i]+=L->bq[i];
        if (L->bk) for (int s2=0;s2<NT;s2++) for (int i=0;i<kvD;i++) k[(int64_t)s2*kvD+i]+=L->bk[i];
        if (L->bv) for (int s2=0;s2<NT;s2++) for (int i=0;i<kvD;i++) v[(int64_t)s2*kvD+i]+=L->bv[i];

        qk_norm(m,L,q,k,NT,H,KVH,hd,qD,kvD);        /* qwen3, before RoPE */
        for (int s2=0;s2<NT;s2++) {
            int pos=pos_base+s2;
            coli_rope_tab rt; rope_table(&rt,pos,hd,c->rope_theta,m->rope_ff);
            for (int h=0;h<H;h++)   rope_head_tab(q+(int64_t)s2*qD +h*hd,&rt,hd,c->rope);
            for (int h=0;h<KVH;h++) rope_head_tab(k+(int64_t)s2*kvD+h*hd,&rt,hd,c->rope);
            for (int h=0;h<KVH;h++) {
                memcpy(m->K[l]+KVOFF(slot,h,pos), k+(int64_t)s2*kvD+h*hd,(size_t)hd*sizeof(float));
                memcpy(m->V[l]+KVOFF(slot,h,pos), v+(int64_t)s2*kvD+h*hd,(size_t)hd*sizeof(float));
            }
        }

        float scale=1.f/sqrtf((float)hd);
        /* Same GPU arm as coli_forward. It belongs here MORE than there: this is
         * the function coli_api.cpp calls, so it is the one that serves. The
         * differences from coli_forward are that the slot is not always 0 and
         * that pos_base can be non-zero from a prefix-cache hit -- both of which
         * are just arguments, because the kernel takes each row's tmax in meta.
         * Rows 0..pos_base-1 are rewritten too; see gpu_kv_write on why. */
        int did_ps_gpu = 0;
        if (ps_gpu && gpu_kv_write_slot(l, slot, 0, pos_base+NT,
                                        m->K[l]+KVOFF(slot,0,0), m->V[l]+KVOFF(slot,0,0)))
            did_ps_gpu = gpu_attn(m, l, q, att, NT, H, scale, ps_slot, ps_pos);
        if (ps_gpu && !did_ps_gpu) { static int warned=0; if (!warned) { warned=1;
            fprintf(stderr,"  prefill GPU attention: FELL BACK to CPU at layer %d\n", l); } }


        if (!did_ps_gpu) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h=0;h<H;h++) for (int s2=0;s2<NT;s2++) {
            int kvh=h/grp, tmax=pos_base+s2;              /* causal */
            const float *qv=q+(int64_t)s2*qD+h*hd;
            attend_online(att+(int64_t)s2*qD+h*hd, qv,
                          m->K[l]+KVOFF(slot,kvh,0), m->V[l]+KVOFF(slot,kvh,0),
                          hd, tmax, hd, scale);
        }
        }
        mm(xb,att,NT,&L->wo);
        for (int s2=0;s2<NT;s2++) for (int i=0;i<D;i++) x[(int64_t)s2*D+i]+=xb[(int64_t)s2*D+i];

        for (int s2=0;s2<NT;s2++) rmsnorm(xb+(int64_t)s2*D,x+(int64_t)s2*D,L->ffn_norm,D,c->eps);
        if (c->n_expert>0) moe_ffn(m,L,x,xb,NT);
        else {
            if (!gpu_ffn(m,L,xb,xb,NT)) {
                mm(g,xb,NT,&L->gate); mm(u,xb,NT,&L->up);
                for (int64_t i=0;i<(int64_t)NT*c->inter;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
                mm(xb,g,NT,&L->down);
            }
            for (int s2=0;s2<NT;s2++) for (int i=0;i<D;i++) x[(int64_t)s2*D+i]+=xb[(int64_t)s2*D+i];
        }
    }

    /* logits for the LAST token only -- the caller samples from that one. */
    float *fin=fal((int64_t)D);
    rmsnorm(fin,x+(int64_t)(NT-1)*D,m->out_norm,D,c->eps);
    float *logits=fal((int64_t)c->vocab);
    mm(logits,fin,1,&m->out);

    /* RECORD what the slot now holds, so the next call can match against it.
     * Only the prompt is recorded -- tokens generated afterwards go into the KV
     * via coli_forward and are NOT tracked here. That is deliberate and
     * conservative: an untracked token can only cost a cache miss, whereas a
     * WRONGLY tracked one produces a confident false prefix match and silently
     * corrupts the output. Under-claiming is the safe direction. */
#ifdef COLI_CACHE_UNSAFE_GUARD
    if (prefix_cache_on() && m->cache_ids && m->cache_ids[slot] && S <= m->kv_ctx) {
#else
    if (prefix_cache_on() && m->cache_ids && m->cache_ids[slot] && S <= m->cache_cap) {
#endif
        memcpy(m->cache_ids[slot], ids, (size_t)S*sizeof(int));
        m->cache_len[slot] = S;
    } else if (m->cache_len) {
        m->cache_len[slot] = 0;      /* cannot describe it => do not claim it */
    }

    free(fin); free(x); free(xb); free(q); free(k); free(v); free(att); free(g); free(u);
    free(ps_slot); free(ps_pos);
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

    /* GPU prefill attention. Decided ONCE per call, not per layer: readiness
     * can allocate and bulk-load the whole device cache, and doing that inside
     * the layer loop would hide a one-time cost as a per-layer one. */
    int pf_gpu = gpu_prefill_attn_ready(m, KVH, hd, S);
    /* Say so, once. Without this an unchanged attention figure reads as "the GPU
     * did not help" when it may mean "the GPU never ran" -- the same class of
     * false negative as the profiler that was never called from --nll. */
    /* Announce it whenever it is ON, not only when someone set the variable --
     * the default is now device-dependent, so "I did not set anything" no
     * longer tells you which arm ran. An unlabelled timing is how the first A/B
     * of this feature measured the GPU against itself. */
    if (pf_gpu) { static int said=0; if (!said) { said=1;
        fprintf(stderr,"  prefill GPU attention: ENGAGED\n"); } }
    int *pf_slot = NULL, *pf_pos = NULL;
    if (pf_gpu) {
        pf_slot = (int*)xmal((size_t)S*sizeof(int));
        pf_pos  = (int*)xmal((size_t)S*sizeof(int));
        for (int s=0;s<S;s++) { pf_slot[s]=0; pf_pos[s]=pos0+s; }
    }

    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        { double t_=cp_now();
          for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->attn_norm,D,c->eps);
          PF.norm += cp_now()-t_; }
        { double t_=cp_now();
        if (!gpu_qkv(L,q,k,vv,xb,S)) {
            mm(q,xb,S,&L->wq); mm(k,xb,S,&L->wk); mm(vv,xb,S,&L->wv);
        }
          PF.qkv += cp_now()-t_; }
        if (L->bq) for (int s=0;s<S;s++) for (int i=0;i<qD;i++)  q[(int64_t)s*qD+i]+=L->bq[i];
        if (L->bk) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) k[(int64_t)s*kvD+i]+=L->bk[i];
        if (L->bv) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) vv[(int64_t)s*kvD+i]+=L->bv[i];

        { double t_=cp_now();
        qk_norm(m,L,q,k,S,H,KVH,hd,qD,kvD);        /* qwen3, before RoPE */
        for (int s=0;s<S;s++) {
            int pos=pos0+s;
            coli_rope_tab rt; rope_table(&rt,pos,hd,c->rope_theta,m->rope_ff);
            for (int h=0;h<H;h++)   rope_head_tab(q+(int64_t)s*qD +h*hd,&rt,hd,c->rope);
            for (int h=0;h<KVH;h++) rope_head_tab(k+(int64_t)s*kvD+h*hd,&rt,hd,c->rope);
        }
        PF.rope += cp_now()-t_; t_=cp_now();
        for (int s=0;s<S;s++) for (int h=0;h<KVH;h++) {
            int64_t row=(int64_t)h*m->kv_ctx+(pos0+s);
            memcpy(m->K[l]+row*hd,k +(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
            memcpy(m->V[l]+row*hd,vv+(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
        }
        PF.kvcopy += cp_now()-t_; }
        float scale=1.f/sqrtf((float)hd);
        double t_attn=cp_now();
        /* schedule(dynamic,1), NOT static: per-iteration cost is tmax=pos0+s, so it
         * grows LINEARLY across the s dimension and a contiguous static chunk of
         * the flattened (h,s) space is nothing like an equal share of the work.
         * Measured 2026-08-26, 683-token prefill, 9800X3D 8c/16t, interleaved:
         * static 1187.2/1175.1 ms, dynamic,1 955.1/936.0, guided 1039.8/1033.2.
         * 1.25x, and 1.11x on the whole prefill, for a scheduling clause.
         *
         * The diagnosis is imbalance and the control says so: oversubscribing to
         * OMP_NUM_THREADS=28 on 16 hardware threads bought static 1.17x
         * (1175.1 -> 1005.6) and dynamic only 1.04x (933.7 -> 894.8). Letting the
         * OS reshuffle helps exactly when the schedule has misallocated, and
         * stops helping once it has not.
         *
         * No cost on short prompts -- 76 tokens: static 44.4/35.5/36.2 ms,
         * dynamic 35.2/33.8/33.2. Not schedule(runtime): libgomp defaults
         * OMP_SCHEDULE to static, so shipping runtime would ship static in
         * everything that does not set the variable. Decode has its own
         * attention loop and is untouched -- n=1 there, no imbalance to fix. */
        /* The GPU arm writes this layer's new KV rows in one submit, then runs
         * the SAME attn_decode.comp kernel the decode path uses -- unchanged,
         * because a workgroup is one (row, head) and meta[r] carries that row's
         * own tmax, which is exactly causal prefill. Either step failing falls
         * through to the CPU loop below rather than returning a partial result:
         * a dropped layer is a silently wrong answer, not a slow one. */
        int did_pf_gpu = 0;
        if (pf_gpu && gpu_kv_write(l, 0, pos0+S, m->K[l], m->V[l]))
            did_pf_gpu = gpu_attn(m, l, q, att, S, H, scale, pf_slot, pf_pos);
        if (pf_gpu && !did_pf_gpu) { static int warned=0; if (!warned) { warned=1;
            fprintf(stderr,"  prefill GPU attention: FELL BACK to CPU at layer %d\n", l); } }

        if (!did_pf_gpu) {
        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int h=0;h<H;h++) for (int s=0;s<S;s++) {
            int kvh=h/grp;                      /* GQA: many q heads share one kv head */
            const float *qv=q+(int64_t)s*qD+h*hd;
            int tmax=pos0+s;                    /* causal */
            attend_online(att+(int64_t)s*qD+h*hd, qv,
                          m->K[l]+(int64_t)kvh*m->kv_ctx*hd,
                          m->V[l]+(int64_t)kvh*m->kv_ctx*hd,
                          hd, tmax, hd, scale);
        }
        }
        PF.attn += cp_now()-t_attn;
        { double t_=cp_now(); mm(xb,att,S,&L->wo); PF.wo += cp_now()-t_; }
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i]+=xb[(int64_t)s*D+i];

        { double t_=cp_now();
          for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->ffn_norm,D,c->eps);
          PF.norm += cp_now()-t_; }
        double t_ffn=cp_now();
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
        PF.ffn += cp_now()-t_ffn;
    }

    int rows = all_logits ? S : 1;
    float *fin=fal((int64_t)rows*D);
    for (int r=0;r<rows;r++){ int s = all_logits ? r : S-1;
        rmsnorm(fin+(int64_t)r*D,x+(int64_t)s*D,m->out_norm,D,c->eps); }
    float *logits=fal((int64_t)rows*c->vocab);
    { double t_=cp_now(); mm(logits,fin,rows,&m->out); PF.head += cp_now()-t_; }
    PF.n++;

    free(fin); free(x); free(xb); free(q); free(k); free(vv); free(att); free(g); free(u);
    free(pf_slot); free(pf_pos);
    m->n_past += S;
    return logits;
}
