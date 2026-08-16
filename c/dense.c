/* dense.c — a dense-transformer engine that reads GGUF directly.
 *
 * WHY. Every one of Colibri's five engines (GLM, Inkling, Kimi K3, DeepSeek V4,
 * OLMoE) is MoE expert-streaming, and every model this homelab actually serves is
 * DENSE. There was no path from a GGUF file to a forward pass: colibri.c never
 * includes gguf_reader.h, and olmoe.c's attention ignores n_kv_heads entirely
 * (it projects K/V to full hidden width), which is wrong for all four fleet
 * models -- they run GQA at 7:1, 8:1, 4:1 and 4:1.
 *
 * SCOPE, stated so it is not overread: this covers `qwen2` and `llama`
 * architectures, which is what the fleet uses. It is not a general GGUF runtime.
 *
 * WEIGHTS. GGUF stores a linear layer as [ne0=in, ne1=out] with ne0 contiguous,
 * so element (o,i) sits at o*I+i -- exactly the layout matmul_q already expects.
 * Verified against the real files: attn_k.weight on netsec v9 is [3584,512], and
 * 512 = n_kv_heads(4) * head_dim(128), i.e. the GQA shrink is in the weights.
 *
 * TWO WEIGHT MODES, and the reason there are two:
 *   WQ=f32   dequantize GGUF -> f32 and compute in f32. Slow and large, but it
 *            isolates ARCHITECTURE correctness from QUANTIZATION loss. Validate
 *            here first; a mismatch is a bug, not a rounding difference.
 *   WQ=int8  additionally requantize to per-output-row int8 (matmul_q's format)
 *            and reuse matmul_q/matmul_q_mrow. This is lossy by construction, so
 *            it must be judged on NLL, never on "did the text look right".
 * Do not skip the f32 stage. Debugging an architecture bug through a lossy
 * quantizer is how a wrong RoPE base gets explained away as quantization noise.
 *
 * ⚠ VALIDATE EVERY ARCHITECTURE SEPARATELY. The first version of this file was
 * validated on qwen2 alone (96.6% top-1 vs llama.cpp) and the `llama` path was
 * assumed to follow because it loaded and produced fluent-looking text. It did
 * not: llama.cpp gives LLAMA and QWEN2 *different* RoPE pairings, and the llama
 * path was running at ppl 639 against llama.cpp's 29 on the same file. "Covers
 * qwen2 and llama" was true of the loader and false of the maths. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "gguf_reader.h"
#include "gguf_meta.h"
#include "ggml_dequant.h"
#include "tok.h"
#include "tok_gguf.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static void *xmalloc(size_t n){ void *p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float *falloc(int64_t n){ return (float*)xmalloc((size_t)n*sizeof(float)); }

/* ---------------- weight container ---------------- */
/* f != NULL  -> f32 weights.  q != NULL -> int8 rows with one scale per output. */
typedef struct { float *f; int8_t *q; float *s; int64_t I, O; } W;

static void w_free(W *w){ free(w->f); free(w->q); free(w->s); memset(w,0,sizeof *w); }

/* ---------------- config ---------------- */
typedef struct {
    char arch[32];
    int  hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, ctx_train;
    float rope_theta, eps;
    int  qkv_bias;          /* qwen2 carries attn_{q,k,v}.bias; llama does not */
    int  rope_neox;         /* 1 = NeoX pairing (i,i+hd/2); 0 = interleaved (2i,2i+1) */
    int  bos, eos;
} DCfg;

typedef struct {
    float *attn_norm, *ffn_norm;
    W q, k, v, o, gate, up, down;
    float *qb, *kb, *vb;    /* NULL when the architecture has no qkv bias */
} DLayer;

typedef struct {
    DCfg   c;
    DLayer *L;
    W       tok_embd, out;
    float  *out_norm;
    float  *rope_ff;        /* rope_freqs.weight, hd/2 floats; NULL if absent */
    /* KV cache, GQA-shaped: n_kv_heads rows, not n_heads. */
    float **K, **V;
    int     max_t;
} DModel;

/* ---------------- matmul ---------------- */
#if defined(__AVX2__)
#include <immintrin.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    __m256i va16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
    __m256i prod = _mm256_madd_epi16(va16, vb16);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod,1));
    __m128i h = _mm_unpackhi_epi64(s,s); s = _mm_add_epi32(s,h);
    h = _mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1)); s = _mm_add_epi32(s,h);
    return _mm_cvtsi128_si32(s);
}
#define HAVE_FAST_DOT_I8 1
#endif

/* y[n,O] = x[n,I] . W^T   -- one pass over the weights for all n rows, which is
 * the grouped-GEMM shape measured in afc897f. Dense is its ideal case: every row
 * of the batch uses the SAME matrix, so there is nothing to bucket first. */
static void dmatmul(float *y, const float *x, int n, const W *w) {
    int64_t I = w->I, O = w->O;
    if (w->f) {
        #pragma omp parallel for schedule(static)
        for (int64_t o = 0; o < O; o++) {
            const float *wr = w->f + o*I;
            for (int r = 0; r < n; r++) {
                const float *xr = x + (int64_t)r*I;
                float acc = 0.f;
                #pragma omp simd reduction(+:acc)
                for (int64_t i = 0; i < I; i++) acc += xr[i]*wr[i];
                y[(int64_t)r*O + o] = acc;
            }
        }
        return;
    }
#if defined(HAVE_FAST_DOT_I8)
    if (I % 16 == 0) {
        int64_t nb = I/16;
        int8_t *xi = (int8_t*)xmalloc((size_t)n*I);
        float  *xs = falloc((int64_t)n*nb);
        for (int r = 0; r < n; r++) {
            const float *xr = x + (int64_t)r*I;
            for (int64_t b = 0; b < nb; b++) {
                const float *xb = xr + b*16;
                float am = 0.f; for (int i=0;i<16;i++){ float a=fabsf(xb[i]); if(a>am)am=a; }
                float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
                xs[r*nb+b] = s; float inv = 1.f/s;
                for (int i=0;i<16;i++) xi[(int64_t)r*I + b*16+i] = (int8_t)lrintf(xb[i]*inv);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int64_t o = 0; o < O; o++) {
            const int8_t *wr = w->q + o*I; float sc = w->s[o];
            for (int r = 0; r < n; r++) {
                const int8_t *xr = xi + (int64_t)r*I; const float *sr = xs + (int64_t)r*nb;
                float acc = 0.f;
                for (int64_t b = 0; b < nb; b++) acc += sr[b]*(float)dot_i8_16(xr+b*16, wr+b*16);
                y[(int64_t)r*O + o] = acc*sc;
            }
        }
        free(xi); free(xs);
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const int8_t *wr = w->q + o*I; float sc = w->s[o];
        for (int r = 0; r < n; r++) {
            const float *xr = x + (int64_t)r*I;
            float acc = 0.f;
            for (int64_t i = 0; i < I; i++) acc += xr[i]*(float)wr[i];
            y[(int64_t)r*O + o] = acc*sc;
        }
    }
}

static void rmsnorm(float *o, const float *x, const float *g, int64_t n, float eps) {
    double ss = 0; for (int64_t i=0;i<n;i++) ss += (double)x[i]*x[i];
    float inv = 1.f/sqrtf((float)(ss/(double)n) + eps);
    for (int64_t i=0;i<n;i++) o[i] = x[i]*inv*g[i];
}

/* ROPE PAIRING IS PER-ARCHITECTURE, and an earlier version of this file got it
 * wrong by asserting otherwise. llama.cpp maps LLM_ARCH_QWEN2 -> ROPE_TYPE_NEOX
 * (pairs i and i+hd/2) but LLM_ARCH_LLAMA -> ROPE_TYPE_NORM (pairs 2i and 2i+1)
 * -- src/llama-model.cpp, the two return statements of llama_model_rope_type().
 * Applying NeoX to a llama model is not a subtle numeric difference: see the
 * measured table at the rope_neox assignment in main(). */
/* NeoX RoPE. `ff` is Llama-3.1's per-dimension frequency correction, carried in
 * the GGUF as the tensor `rope_freqs.weight` (hd/2 floats) and applied by ggml as
 * `theta/ff` -- ggml/src/ggml-cpu/ops.cpp:5619, `ff = freq_factors[i0/2]`, then
 * rope_yarn(theta/ff, ...). NULL when the model has no such tensor.
 *
 * Ignoring it is NOT a rounding difference. On this WhiteRabbitNeo build the 64
 * factors run 1.0 -> 8.0 with only 29 of 64 equal to 1.0, so 35 of 64 frequency
 * dimensions rotate at the wrong rate. Applying it is what ggml does, so it is
 * correct on those grounds -- but be honest about the size of the effect: on its
 * own it moved TF-NLL only 6.6550 -> 6.5863, because the dominant error at the
 * time was the RoPE PAIRING above, not the frequencies. It is not the fix that
 * mattered, and it was nearly reported as one. */
static void rope_head(float *v, int pos, int hd, float theta, const float *ff, int neox) {
    int half = hd/2;
    for (int i = 0; i < half; i++) {
        float fr = powf(theta, -(float)(2*i)/(float)hd);
        if (ff) fr /= ff[i];
        float c = cosf(pos*fr), s = sinf(pos*fr);
        int ia = neox ? i      : 2*i;
        int ib = neox ? i+half : 2*i+1;
        float a = v[ia], b = v[ib];
        v[ia] = a*c - b*s; v[ib] = a*s + b*c;
    }
}

/* ---------------- forward ---------------- */
/* Returns logits for the LAST row only (rows=1) or all rows. */
static float *dforward(DModel *m, const int *ids, int S, int pos_base, int all_rows) {
    DCfg *c = &m->c;
    int D = c->hidden, H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim;
    int kvD = KVH*hd, qD = H*hd;
    int grp = H/KVH;                 /* query heads per kv head */

    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        int id = ids[s];
        if (id < 0 || id >= c->vocab) { fprintf(stderr,"token id %d out of range [0,%d)\n", id, c->vocab); exit(1); }
        if (m->tok_embd.f) memcpy(x+(int64_t)s*D, m->tok_embd.f + (int64_t)id*D, D*sizeof(float));
        else { const int8_t *r = m->tok_embd.q + (int64_t)id*D; float sc = m->tok_embd.s[id];
               for (int i=0;i<D;i++) x[(int64_t)s*D+i] = r[i]*sc; }
    }

    float *xb = falloc((int64_t)S*D);
    float *q  = falloc((int64_t)S*qD), *k = falloc((int64_t)S*kvD), *v = falloc((int64_t)S*kvD);
    float *att= falloc((int64_t)S*qD);
    float *g  = falloc((int64_t)S*c->inter), *u = falloc((int64_t)S*c->inter);

    for (int l = 0; l < c->n_layers; l++) {
        DLayer *L = &m->L[l];
        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D, x+(int64_t)s*D, L->attn_norm, D, c->eps);

        dmatmul(q, xb, S, &L->q);
        dmatmul(k, xb, S, &L->k);
        dmatmul(v, xb, S, &L->v);
        if (L->qb) for (int s=0;s<S;s++) for (int i=0;i<qD;i++)  q[(int64_t)s*qD+i]  += L->qb[i];
        if (L->kb) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) k[(int64_t)s*kvD+i] += L->kb[i];
        if (L->vb) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) v[(int64_t)s*kvD+i] += L->vb[i];

        for (int s=0;s<S;s++) {
            int pos = pos_base + s;
            for (int h=0;h<H;h++)   rope_head(q + (int64_t)s*qD  + h*hd, pos, hd, c->rope_theta, m->rope_ff, c->rope_neox);
            for (int h=0;h<KVH;h++) rope_head(k + (int64_t)s*kvD + h*hd, pos, hd, c->rope_theta, m->rope_ff, c->rope_neox);
        }
        /* KV cache is GQA-shaped: KVH rows per layer, not H. This is the whole
         * point -- storing H rows would be 4-8x the memory for identical math. */
        for (int s=0;s<S;s++) for (int h=0;h<KVH;h++) {
            int64_t row = (int64_t)h*m->max_t + (pos_base+s);
            memcpy(m->K[l] + row*hd, k + (int64_t)s*kvD + h*hd, hd*sizeof(float));
            memcpy(m->V[l] + row*hd, v + (int64_t)s*kvD + h*hd, hd*sizeof(float));
        }

        int Tk = pos_base + S;
        float scale = 1.f/sqrtf((float)hd);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) for (int s = 0; s < S; s++) {
            int kvh = h/grp;                       /* GQA: many q heads share one kv head */
            const float *qv = q + (int64_t)s*qD + h*hd;
            int tmax = pos_base + s;               /* causal */
            float *sc = (float*)xmalloc((size_t)(tmax+1)*sizeof(float));
            float mx = -1e30f;
            for (int t=0;t<=tmax;t++) {
                const float *kv = m->K[l] + ((int64_t)kvh*m->max_t + t)*hd;
                float d=0; for (int i=0;i<hd;i++) d += qv[i]*kv[i];
                d *= scale; sc[t]=d; if (d>mx) mx=d;
            }
            float sum=0; for (int t=0;t<=tmax;t++){ sc[t]=expf(sc[t]-mx); sum+=sc[t]; }
            float inv = 1.f/sum;
            float *o = att + (int64_t)s*qD + h*hd;
            for (int i=0;i<hd;i++) o[i]=0.f;
            for (int t=0;t<=tmax;t++) {
                const float w = sc[t]*inv;
                const float *vv = m->V[l] + ((int64_t)kvh*m->max_t + t)*hd;
                for (int i=0;i<hd;i++) o[i] += w*vv[i];
            }
            free(sc);
        }
        dmatmul(xb, att, S, &L->o);
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i] += xb[(int64_t)s*D+i];

        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D, x+(int64_t)s*D, L->ffn_norm, D, c->eps);
        dmatmul(g, xb, S, &L->gate);
        dmatmul(u, xb, S, &L->up);
        for (int64_t i=0;i<(int64_t)S*c->inter;i++) { float gv=g[i]; g[i] = (gv/(1.f+expf(-gv)))*u[i]; }
        dmatmul(xb, g, S, &L->down);
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i] += xb[(int64_t)s*D+i];
    }

    int rows = all_rows ? S : 1;
    float *last = falloc((int64_t)rows*D);
    for (int r=0;r<rows;r++) rmsnorm(last+(int64_t)r*D, x+(int64_t)(S-rows+r)*D, m->out_norm, D, c->eps);
    float *logit = falloc((int64_t)rows*c->vocab);
    dmatmul(logit, last, rows, &m->out);

    free(x); free(xb); free(q); free(k); free(v); free(att); free(g); free(u); free(last);
    return logit;
}

/* ---------------- loading ---------------- */
static const GgufTensorInfo *find_t(GgufIndex *ix, const char *name) {
    for (size_t i=0;i<ix->n;i++) if (!strcmp(ix->t[i].name, name)) return &ix->t[i];
    return NULL;
}

/* Dequantize a whole tensor to f32. Returns element count, 0 on failure. */
static int64_t load_f32(int fd, const GgufTensorInfo *t, float **out, long long fsz) {
    const GgmlType *g = ggml_type(t->ttype);
    if (!g || !g->blck) { fprintf(stderr,"unsupported type %u for %s\n", t->ttype, t->name); return 0; }
    int64_t nelem = 1; for (int d=0; d<t->rank; d++) nelem *= (int64_t)t->shape[d];
    int64_t nblk = (g->blck==1) ? nelem : nelem/g->blck;
    long long nbytes = (g->blck==1) ? nelem*(long long)g->bytes : nblk*(long long)g->bytes;
    if ((long long)t->data_off + nbytes > fsz) { fprintf(stderr,"%s runs past EOF\n", t->name); return 0; }
    void *raw = xmalloc((size_t)nbytes);
    if (pread(fd, raw, (size_t)nbytes, (off_t)t->data_off) != (ssize_t)nbytes) {
        fprintf(stderr,"short read on %s\n", t->name); free(raw); return 0; }
    float *dst = falloc(nelem);
    switch (t->ttype) {
        case 0:  gguf_dequant_f32 (raw,dst,nelem); break;
        case 1:  gguf_dequant_f16 (raw,dst,nelem); break;
        case 11: gguf_dequant_q3_K(raw,dst,nblk);  break;
        case 12: gguf_dequant_q4_K(raw,dst,nblk);  break;
        case 13: gguf_dequant_q5_K(raw,dst,nblk);  break;
        case 14: gguf_dequant_q6_K(raw,dst,nblk);  break;
        case 30: gguf_dequant_bf16(raw,dst,nelem); break;
        default: fprintf(stderr,"no dequant for %s (%s)\n", t->name, g->name); free(raw); free(dst); return 0;
    }
    free(raw); *out = dst; return nelem;
}

static int g_int8 = 0;

/* Quantize to per-output-row int8. One scale per row of O, matching matmul_q's
 * format. Symmetric absmax: max|w| maps to 127, so a row of zeros stays zero
 * rather than dividing by zero. */
static void quantize_rows(W *w) {
    int64_t I=w->I, O=w->O;
    w->q = (int8_t*)xmalloc((size_t)I*O);
    w->s = falloc(O);
    for (int64_t o=0;o<O;o++) {
        const float *r = w->f + o*I;
        float am=0.f; for (int64_t i=0;i<I;i++){ float a=fabsf(r[i]); if(a>am)am=a; }
        float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
        w->s[o]=s; float inv=1.f/s;
        for (int64_t i=0;i<I;i++) w->q[o*I+i] = (int8_t)lrintf(r[i]*inv);
    }
    free(w->f); w->f=NULL;
}

static int load_w(int fd, GgufIndex *ix, long long fsz, const char *name, W *w, int64_t I, int64_t O, int required) {
    const GgufTensorInfo *t = find_t(ix, name);
    if (!t) { if (required) fprintf(stderr,"missing required tensor %s\n", name); return 0; }
    float *f=NULL; int64_t n = load_f32(fd,t,&f,fsz);
    if (!n) return 0;
    if (n != I*O) { fprintf(stderr,"%s: %lld elements, expected %lld x %lld\n", name,(long long)n,(long long)I,(long long)O); free(f); return 0; }
    w->f=f; w->I=I; w->O=O;
    if (g_int8) quantize_rows(w);
    return 1;
}

static int load_vec(int fd, GgufIndex *ix, long long fsz, const char *name, float **out, int64_t n, int required) {
    const GgufTensorInfo *t = find_t(ix, name);
    if (!t) { if (required) fprintf(stderr,"missing required tensor %s\n", name); return 0; }
    float *f=NULL; int64_t got = load_f32(fd,t,&f,fsz);
    if (!got) return 0;
    if (got != n) { fprintf(stderr,"%s: %lld elements, expected %lld\n", name,(long long)got,(long long)n); free(f); return 0; }
    *out=f; return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,
        "usage: %s <model.gguf> [ids...]\n"
        "  env: WQ=f32|int8 (default f32)  NEW=<n tokens>  NLL=1  IDS=<comma list>\n", argv[0]); return 2; }
    const char *path = argv[1];
    const char *wq = getenv("WQ"); g_int8 = (wq && !strcmp(wq,"int8"));

    GgufIndex ix; char err[256];
    if (!gguf_index_open(path,&ix,err,sizeof err)) { fprintf(stderr,"gguf open: %s\n",err); return 1; }
    GgufMeta mt;
    if (!gguf_meta_open(path,&mt,err,sizeof err)) { fprintf(stderr,"meta open: %s\n",err); return 1; }

    DModel m; memset(&m,0,sizeof m); DCfg *c=&m.c;
    if (!gguf_meta_str(&mt,"general.architecture",c->arch,sizeof c->arch)) { fprintf(stderr,"no architecture\n"); return 1; }
    if (strcmp(c->arch,"qwen2") && strcmp(c->arch,"llama")) {
        fprintf(stderr,"dense.c supports qwen2 and llama; this file is '%s'. Refusing rather than "
                       "guessing -- a wrong arch silently produces fluent nonsense.\n", c->arch); return 1; }
    long long v; float f;
    #define REQ_I(sfx,dst) do{ if(!gguf_meta_arch_i64(&mt,c->arch,sfx,&v)){fprintf(stderr,"missing %s.%s\n",c->arch,sfx);return 1;} dst=(int)v; }while(0)
    REQ_I("block_count",         c->n_layers);
    REQ_I("embedding_length",    c->hidden);
    REQ_I("feed_forward_length", c->inter);
    REQ_I("attention.head_count",    c->n_heads);
    REQ_I("attention.head_count_kv", c->n_kv_heads);
    if (gguf_meta_arch_i64(&mt,c->arch,"context_length",&v)) c->ctx_train=(int)v;
    c->rope_theta = gguf_meta_arch_f32(&mt,c->arch,"rope.freq_base",&f) ? f : 10000.f;
    c->eps        = gguf_meta_arch_f32(&mt,c->arch,"attention.layer_norm_rms_epsilon",&f) ? f : 1e-5f;
    if (gguf_meta_arch_i64(&mt,c->arch,"attention.key_length",&v)) c->head_dim=(int)v;
    else c->head_dim = c->hidden / c->n_heads;
    /* RoPE pairing, from llama.cpp's llama_model_rope_type(): QWEN2 is NEOX,
     * LLAMA is NORM (interleaved). Measured on WhiteRabbitNeo-2-8B, 911 tokens
     * of the same prose, WQ=int8, and on qwen2.5-3b for the mirror control:
     *
     *   model      pairing              TF-NLL     ppl    top-1 vs llama.cpp
     *   llama 8B   interleaved (right)  3.1589    23.5      21/22 = 95.5%
     *   llama 8B   NeoX        (wrong)  6.4594   638.7      11/22 = 50.0%
     *   qwen2 3B   NeoX        (right)  3.4052    30.1      96.6% (2026-08-15)
     *   qwen2 3B   interleaved (wrong)  6.6490   772.0      --
     *
     * A clean two-way control: each architecture's correct mode works and the
     * other is catastrophic, so neither row can be an artefact of the harness.
     * llama.cpp's own llama-perplexity on the same text and model reports ppl
     * 29.3 (-c 512, 2 chunks); 23.5 here is over one 912-token window, so the
     * two are the same order rather than directly comparable.
     *
     * ROPE=neox|interleaved overrides, so the wrong mode stays available as a
     * negative control. A control you have to patch the source to run is a
     * control nobody runs. */
    c->rope_neox = !strcmp(c->arch,"qwen2");
    { const char *rp = getenv("ROPE");
      if (rp && !strcmp(rp,"neox"))        c->rope_neox = 1;
      else if (rp && !strcmp(rp,"interleaved")) c->rope_neox = 0;
      else if (rp) { fprintf(stderr,"ROPE must be neox or interleaved\n"); return 1; } }
    if (gguf_meta_i64(&mt,"tokenizer.ggml.bos_token_id",&v)) c->bos=(int)v;
    if (gguf_meta_i64(&mt,"tokenizer.ggml.eos_token_id",&v)) c->eos=(int)v;

    const GgufTensorInfo *te = find_t(&ix,"token_embd.weight");
    if (!te) { fprintf(stderr,"no token_embd.weight\n"); return 1; }
    c->vocab = (int)te->shape[1];
    c->qkv_bias = find_t(&ix,"blk.0.attn_q.bias") != NULL;

    printf("== dense engine ==\n");
    printf("arch=%s layers=%d d=%d ffn=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
           c->arch,c->n_layers,c->hidden,c->inter,c->n_heads,c->n_kv_heads,c->head_dim,c->vocab);
    printf("rope_theta=%.1f eps=%.3e qkv_bias=%s ctx_train=%d WQ=%s\n",
           c->rope_theta,c->eps,c->qkv_bias?"yes":"no",c->ctx_train, g_int8?"int8":"f32");
    if (c->n_heads % c->n_kv_heads) { fprintf(stderr,"n_heads %d not divisible by n_kv_heads %d\n",c->n_heads,c->n_kv_heads); return 1; }

    int fd = open(path,O_RDONLY); if (fd<0){perror("open");return 1;}
    struct stat st; fstat(fd,&st); long long fsz=(long long)st.st_size;

    double t0=now_s();
    int D=c->hidden, qD=c->n_heads*c->head_dim, kvD=c->n_kv_heads*c->head_dim;
    if (!load_w(fd,&ix,fsz,"token_embd.weight",&m.tok_embd,D,c->vocab,1)) return 1;
    if (!load_w(fd,&ix,fsz,"output.weight",&m.out,D,c->vocab,1)) return 1;
    if (!load_vec(fd,&ix,fsz,"output_norm.weight",&m.out_norm,D,1)) return 1;
    /* Optional: Llama-3.1 rope frequency correction. Absent on qwen2 -- absence
     * is the normal answer, not an error, so load_vec is called non-required and
     * a NULL result simply disables the correction. */
    if (find_t(&ix,"rope_freqs.weight")) {
        if (!load_vec(fd,&ix,fsz,"rope_freqs.weight",&m.rope_ff,c->head_dim/2,0)) {
            fprintf(stderr,"rope_freqs.weight present but unreadable at %d elements; "
                           "refusing to guess\n", c->head_dim/2); return 1; }
        printf("rope_freqs: %d factors loaded (llama-3.1 rope scaling)\n", c->head_dim/2);
    }
    m.L = (DLayer*)calloc((size_t)c->n_layers,sizeof(DLayer));
    char nm[128];
    for (int l=0;l<c->n_layers;l++) {
        DLayer *L=&m.L[l];
        #define LW(field,tn,I,O) do{ snprintf(nm,sizeof nm,"blk.%d." tn ".weight",l); \
            if(!load_w(fd,&ix,fsz,nm,&L->field,I,O,1)) return 1; }while(0)
        #define LV(field,tn,N,req) do{ snprintf(nm,sizeof nm,"blk.%d." tn,l); \
            if(!load_vec(fd,&ix,fsz,nm,&L->field,N,req) && req) return 1; }while(0)
        LV(attn_norm,"attn_norm.weight",D,1);
        LV(ffn_norm ,"ffn_norm.weight" ,D,1);
        LW(q,"attn_q",D,qD); LW(k,"attn_k",D,kvD); LW(v,"attn_v",D,kvD);
        LW(o,"attn_output",qD,D);
        LW(gate,"ffn_gate",D,c->inter); LW(up,"ffn_up",D,c->inter); LW(down,"ffn_down",c->inter,D);
        if (c->qkv_bias) { LV(qb,"attn_q.bias",qD,1); LV(kb,"attn_k.bias",kvD,1); LV(vb,"attn_v.bias",kvD,1); }
        if (l==0 || l==c->n_layers-1) { printf("  layer %d loaded\n", l); fflush(stdout); }
    }
    printf("weights loaded in %.1fs\n", now_s()-t0);

    /* Tokenizer, from the model file itself. Loaded even when ids are supplied
     * directly, so generated ids can always be rendered back to text -- an id
     * stream is unreadable, and "looks right" is not a check anyone can run on
     * one. TOK=0 opts out for a file whose tokenizer this loader refuses. */
    Tok T; int have_tok = 0, tk_bos = -1, tk_eos = -1, tk_addbos = 0;
    if (!(getenv("TOK") && atoi(getenv("TOK"))==0)) {
        tok_load_gguf(&T, path, &tk_bos, &tk_eos, &tk_addbos);
        have_tok = 1;
        printf("tokenizer: %d ids, %d merges-keyed, %d atomic, bos=%d eos=%d add_bos=%d\n",
               T.n_ids, T.merges.cap, T.nsp, tk_bos, tk_eos, tk_addbos);
    }

    /* token ids: PROMPT text, else IDS env (comma list) or argv, else just BOS */
    int ids[8192]; int nid=0;
    const char *prompt = getenv("PROMPT");
    const char *idsenv = getenv("IDS");
    if (prompt) {
        if (!have_tok) { fprintf(stderr,"PROMPT needs the tokenizer, but TOK=0\n"); return 1; }
        if (tk_addbos && tk_bos >= 0) ids[nid++] = tk_bos;
        nid += tok_encode(&T, prompt, (int)strlen(prompt), ids+nid, 8192-nid);
    }
    else if (idsenv) { char *cp=strdup(idsenv),*sv=NULL; for(char *p=strtok_r(cp,",",&sv);p&&nid<8192;p=strtok_r(NULL,",",&sv)) ids[nid++]=atoi(p); free(cp); }
    else for (int i=2;i<argc && nid<8192;i++) ids[nid++]=atoi(argv[i]);
    if (!nid) ids[nid++]=c->bos;
    if (have_tok) {
        /* Print the ids AND the round-trip. Ids alone hide a tokenizer bug;
         * text alone hides which token boundary moved. */
        printf("prompt: %d tokens |", nid);
        for (int i=0;i<nid && i<64;i++) printf(" %d", ids[i]);
        if (nid > 64) printf(" ...");
        printf("\n");
        /* Size from the actual pieces, not a per-token guess. 4*nid truncated a
         * llama-3 chat prompt at 132 of 150 bytes, because <|start_header_id|>
         * alone is 19 -- and a truncated round-trip reads as a tokenizer bug. */
        size_t need = 1;
        for (int i=0;i<nid;i++) if (ids[i]>=0 && ids[i]<T.n_ids && T.id2str[ids[i]]) need += strlen(T.id2str[ids[i]]);
        char *rt = (char*)malloc(need);
        tok_decode(&T, ids, nid, rt, (int)need-1);
        printf("roundtrip: %s\n", rt);
        free(rt);
    }

    int n_new = getenv("NEW") ? atoi(getenv("NEW")) : 0;
    m.max_t = nid + n_new + 1;
    m.K=(float**)xmalloc(sizeof(float*)*c->n_layers); m.V=(float**)xmalloc(sizeof(float*)*c->n_layers);
    for (int l=0;l<c->n_layers;l++){ m.K[l]=falloc((int64_t)c->n_kv_heads*m.max_t*c->head_dim);
                                     m.V[l]=falloc((int64_t)c->n_kv_heads*m.max_t*c->head_dim); }

    if (getenv("NLL") && atoi(getenv("NLL"))==1) {
        /* Teacher-forced NLL over the given ids. This is the metric that survives
         * quantization: output text does not. */
        double t=now_s();
        float *lg = dforward(&m, ids, nid, 0, 1);
        double nll=0; int scored=0;
        for (int s=0;s<nid-1;s++) {
            const float *row = lg + (int64_t)s*c->vocab;
            float mx=-1e30f; for (int i=0;i<c->vocab;i++) if(row[i]>mx)mx=row[i];
            double se=0; for (int i=0;i<c->vocab;i++) se += exp((double)(row[i]-mx));
            nll += -((double)row[ids[s+1]] - mx - log(se)); scored++;
        }
        printf("TF-NLL: %.4f nats/token over %d tokens | ppl=%.3f | %.1fs\n",
               nll/scored, scored, exp(nll/scored), now_s()-t);
        /* Teacher-forced argmax at EVERY position, from the same single forward
         * pass. This is the comparison that discriminates an architecture bug
         * from numeric drift: a wrong RoPE base or a mis-mapped GQA head is wrong
         * everywhere, while accumulation-order differences only flip near-ties.
         * Printed as ids so a reference can be diffed mechanically rather than
         * by reading the text and deciding it looks plausible. */
        printf("ARGMAX:");
        for (int s = 0; s < nid-1; s++) {
            const float *row = lg + (int64_t)s*c->vocab;
            int b = 0; for (int i = 1; i < c->vocab; i++) if (row[i] > row[b]) b = i;
            printf(" %d", b);
        }
        printf("\n");
        free(lg);
    } else {
        double t=now_s();
        float *lg = dforward(&m, ids, nid, 0, 0);
        int best=0; for (int i=1;i<c->vocab;i++) if (lg[i]>lg[best]) best=i;
        int top[5];
        printf("prefill %d tokens in %.2fs | argmax next = %d (logit %.4f)\n", nid, now_s()-t, best, lg[best]);
        /* Top-5 with logits. A reference comparison must not rest on argmax alone:
         * two implementations can agree on the top token while disagreeing badly
         * underneath, and that shows up here rather than in the generated text. */
        printf("top5:");
        for (int r = 0; r < 5; r++) {
            int b = -1; float bv = -1e30f;
            for (int i = 0; i < c->vocab; i++) {
                int taken = 0;
                for (int j = 0; j < r; j++) if (i == top[j]) { taken = 1; break; }
                if (!taken && lg[i] > bv) { bv = lg[i]; b = i; }
            }
            top[r] = b;
            printf("  %d:%.4f", b, bv);
        }
        printf("\n");
        free(lg);
        int cur = best, pos = nid;
        int *gen = (int*)xmalloc(sizeof(int)*(size_t)(n_new>0?n_new:1)); int ngen=0;
        for (int n=0;n<n_new;n++) {
            gen[ngen++] = cur;
            if (have_tok) {
                /* Stream one token at a time. Decoding the whole run at the end
                 * would be identical output, but a decode-per-token also proves
                 * the id is renderable at the moment it is produced. */
                char buf[64]; int nb2 = tok_decode(&T, &cur, 1, buf, (int)sizeof buf - 1);
                buf[nb2] = 0; fputs(buf, stdout);
            } else { printf("%d ", cur); }
            fflush(stdout);
            if (have_tok && tk_eos >= 0 && cur == tk_eos) { n_new = n+1; break; }
            float *l2 = dforward(&m, &cur, 1, pos, 0);
            int nb=0; for (int i=1;i<c->vocab;i++) if (l2[i]>l2[nb]) nb=i;
            free(l2); pos++; cur=nb;
        }
        if (n_new) printf("\n");
        if (ngen) { printf("gen ids:"); for (int i=0;i<ngen;i++) printf(" %d", gen[i]); printf("\n"); }
        free(gen);
    }
    if (have_tok) tok_free(&T);
    return 0;
}
