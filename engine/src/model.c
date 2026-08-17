/* model.c — see model.h for scope and the architecture-validation warning. */
#define _GNU_SOURCE
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "model.h"
#include "gguf_reader.h"
#include "gguf_meta.h"
#include "ggml_dequant.h"
#include "tok.h"
#include "tok_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)

static void *xmal(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float *fal(int64_t n){ return (float*)xmal((size_t)n*sizeof(float)); }

/* ------------------------------------------------------------ tensor load */
static int64_t deq(int fd, const GgufTensorInfo *t, float **out, long long fsz) {
    const GgmlType *g = ggml_type(t->ttype);
    if (!g || !g->blck) return 0;
    int64_t ne = 1; for (int d=0; d<t->rank; d++) ne *= (int64_t)t->shape[d];
    int64_t nblk = (g->blck==1) ? ne : ne/g->blck;
    long long nb = (g->blck==1) ? ne*(long long)g->bytes : nblk*(long long)g->bytes;
    if ((long long)t->data_off + nb > fsz) return 0;
    void *raw = xmal((size_t)nb);
    if (pread(fd, raw, (size_t)nb, (off_t)t->data_off) != (ssize_t)nb) { free(raw); return 0; }
    float *dst = fal(ne);
    switch (t->ttype) {
        case 0:  gguf_dequant_f32 (raw,dst,ne);   break;
        case 1:  gguf_dequant_f16 (raw,dst,ne);   break;
        case 11: gguf_dequant_q3_K(raw,dst,nblk); break;
        case 12: gguf_dequant_q4_K(raw,dst,nblk); break;
        case 13: gguf_dequant_q5_K(raw,dst,nblk); break;
        case 14: gguf_dequant_q6_K(raw,dst,nblk); break;
        case 30: gguf_dequant_bf16(raw,dst,ne);   break;
        default: free(raw); free(dst); return 0;
    }
    free(raw); *out = dst; return ne;
}

static const GgufTensorInfo *find_t(GgufIndex *ix, const char *nm) {
    for (size_t i=0;i<ix->n;i++) if (!strcmp(ix->t[i].name, nm)) return &ix->t[i];
    return NULL;
}

/* Quantize f32 -> per-row int8 stored offset-to-unsigned. See gemm_i8.h for why
 * unsigned: VPDPBUSD is u8 x s8 and offsetting the weights makes the correction
 * term depend on activations (n*I) rather than weights (I*O). */
static void quant_rows(const float *f, coli_w_i8 *w, int64_t I, int64_t O) {
    w->I=I; w->O=O;
    w->qu = (uint8_t*)xmal((size_t)I*O);
    w->scale = fal(O);
    for (int64_t o=0;o<O;o++) {
        const float *r = f + o*I;
        float am=0.f; for (int64_t i=0;i<I;i++){ float a=fabsf(r[i]); if(a>am)am=a; }
        float s = am/127.f; if (s<1e-12f) s=1e-12f;
        w->scale[o]=s; float inv=1.f/s;
        for (int64_t i=0;i<I;i++) w->qu[o*I+i]=(uint8_t)((int)lrintf(r[i]*inv)+128);
    }
}

static int load_w(int fd, GgufIndex *ix, long long fsz, const char *nm,
                  coli_w_i8 *w, int64_t I, int64_t O, int req, char *err, size_t errcap) {
    const GgufTensorInfo *t = find_t(ix, nm);
    if (!t) { if (req) MERR("missing tensor %s", nm); return 0; }
    float *f=NULL; int64_t n = deq(fd,t,&f,fsz);
    if (!n) { MERR("cannot dequantize %s (type %u)", nm, t->ttype); return 0; }
    if (n != I*O) { MERR("%s: %lld elements, expected %lldx%lld",nm,(long long)n,(long long)I,(long long)O); free(f); return 0; }
    quant_rows(f, w, I, O); free(f); return 1;
}

static int load_vec(int fd, GgufIndex *ix, long long fsz, const char *nm,
                    float **out, int64_t n, int req, char *err, size_t errcap) {
    const GgufTensorInfo *t = find_t(ix, nm);
    if (!t) { if (req) MERR("missing tensor %s", nm); return 0; }
    float *f=NULL; int64_t got = deq(fd,t,&f,fsz);
    if (!got) { MERR("cannot dequantize %s", nm); return 0; }
    if (got != n) { MERR("%s: %lld elements, expected %lld",nm,(long long)got,(long long)n); free(f); return 0; }
    *out=f; return 1;
}

/* ------------------------------------------------------------------- load */
coli_model *coli_load(const char *path, int max_ctx, int wq_int8, char *err, size_t errcap) {
    (void)wq_int8;   /* f32 mode not wired yet; see README "not built" */
    GgufIndex ix; char e[256];
    if (!gguf_index_open(path,&ix,e,sizeof e)) { MERR("gguf: %s", e); return NULL; }
    GgufMeta mt;
    if (!gguf_meta_open(path,&mt,e,sizeof e)) { MERR("meta: %s", e); return NULL; }

    coli_model *m = (coli_model*)calloc(1,sizeof *m);
    coli_cfg *c = &m->cfg;
    if (!gguf_meta_str(&mt,"general.architecture",c->arch,sizeof c->arch)) { MERR("no architecture"); goto fail; }
    if (strcmp(c->arch,"qwen2") && strcmp(c->arch,"llama")) {
        MERR("architecture '%s' is not validated here; refusing rather than guessing "
             "(a wrong arch produces fluent nonsense)", c->arch); goto fail; }

    long long v; float f;
    #define REQI(sfx,dst) do{ if(!gguf_meta_arch_i64(&mt,c->arch,sfx,&v)){MERR("missing %s.%s",c->arch,sfx);goto fail;} dst=(int)v; }while(0)
    REQI("block_count",c->n_layers); REQI("embedding_length",c->hidden);
    REQI("feed_forward_length",c->inter);
    REQI("attention.head_count",c->n_heads); REQI("attention.head_count_kv",c->n_kv_heads);
    if (gguf_meta_arch_i64(&mt,c->arch,"context_length",&v)) c->ctx_train=(int)v;
    c->rope_theta = gguf_meta_arch_f32(&mt,c->arch,"rope.freq_base",&f)?f:10000.f;
    c->eps = gguf_meta_arch_f32(&mt,c->arch,"attention.layer_norm_rms_epsilon",&f)?f:1e-5f;
    c->head_dim = gguf_meta_arch_i64(&mt,c->arch,"attention.key_length",&v)?(int)v:c->hidden/c->n_heads;
    if (gguf_meta_arch_i64(&mt,c->arch,"expert_count",&v)) c->n_expert=(int)v;
    if (gguf_meta_arch_i64(&mt,c->arch,"expert_used_count",&v)) c->n_expert_used=(int)v;
    if (gguf_meta_arch_i64(&mt,c->arch,"expert_feed_forward_length",&v)) c->expert_inter=(int)v;
    else c->expert_inter = c->inter;

    /* RoPE pairing is per-architecture. Getting this wrong costs ppl 639 vs 29;
     * see model.h. */
    c->rope = strcmp(c->arch,"qwen2")==0 ? COLI_ROPE_NEOX : COLI_ROPE_INTERLEAVED;

    const GgufTensorInfo *te = find_t(&ix,"token_embd.weight");
    if (!te) { MERR("no token_embd.weight"); goto fail; }
    c->vocab = (int)te->shape[1];
    c->qkv_bias = find_t(&ix,"blk.0.attn_q.bias") != NULL;
    if (c->n_heads % c->n_kv_heads) { MERR("n_heads %d not divisible by n_kv_heads %d",c->n_heads,c->n_kv_heads); goto fail; }

    int fd = open(path,O_RDONLY); if (fd<0) { MERR("open failed"); goto fail; }
    struct stat st; fstat(fd,&st); long long fsz=(long long)st.st_size;
    int D=c->hidden, hd=c->head_dim, qD=c->n_heads*hd, kvD=c->n_kv_heads*hd;

    if (!load_w(fd,&ix,fsz,"token_embd.weight",&m->tok_embd,D,c->vocab,1,err,errcap)) goto fail2;
    if (!load_w(fd,&ix,fsz,"output.weight",&m->out,D,c->vocab,0,err,errcap)) {
        /* tied embeddings: reuse token_embd as the head */
        m->out = m->tok_embd;
    }
    if (!load_vec(fd,&ix,fsz,"output_norm.weight",&m->out_norm,D,1,err,errcap)) goto fail2;
    if (find_t(&ix,"rope_freqs.weight"))
        load_vec(fd,&ix,fsz,"rope_freqs.weight",&m->rope_ff,hd/2,0,err,errcap);

    m->L = (coli_layer*)calloc((size_t)c->n_layers,sizeof(coli_layer));
    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l]; char nm[128];
        #define LW(field,suffix,II,OO,req) do{ snprintf(nm,sizeof nm,"blk.%d.%s",l,suffix); \
            if(!load_w(fd,&ix,fsz,nm,&L->field,II,OO,req,err,errcap) && req) goto fail2; }while(0)
        #define LV(field,suffix,N,req) do{ snprintf(nm,sizeof nm,"blk.%d.%s",l,suffix); \
            if(!load_vec(fd,&ix,fsz,nm,&L->field,N,req,err,errcap) && req) goto fail2; }while(0)
        LV(attn_norm,"attn_norm.weight",D,1);
        LV(ffn_norm ,"ffn_norm.weight" ,D,1);
        LW(wq,"attn_q.weight",D,qD,1);  LW(wk,"attn_k.weight",D,kvD,1);
        LW(wv,"attn_v.weight",D,kvD,1); LW(wo,"attn_output.weight",qD,D,1);
        if (c->qkv_bias) { LV(bq,"attn_q.bias",qD,1); LV(bk,"attn_k.bias",kvD,1); LV(bv,"attn_v.bias",kvD,1); }
        if (c->n_expert > 0) {
            /* GGUF stores the experts of one layer as ONE 3-D tensor,
             * [in, out, n_expert], contiguous in that order, so expert e's
             * matrix starts at e*in*out. Split into per-expert coli_w_i8 so the
             * grouped GEMM can address one expert's weights as a plain matrix. */
            int EI = c->expert_inter;
            snprintf(nm,sizeof nm,"blk.%d.ffn_gate_inp.weight",l);
            if(!load_w(fd,&ix,fsz,nm,&L->router,D,c->n_expert,1,err,errcap)) goto fail2;
            L->e_gate=(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            L->e_up  =(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            L->e_down=(coli_w_i8*)calloc((size_t)c->n_expert,sizeof(coli_w_i8));
            static const char *exs[3]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
            for (int t3=0;t3<3;t3++) {
                snprintf(nm,sizeof nm,"blk.%d.%s",l,exs[t3]);
                const GgufTensorInfo *ti=find_t(&ix,nm);
                if(!ti){ MERR("missing %s",nm); goto fail2; }
                int64_t II = (t3==2)?EI:D, OO = (t3==2)?D:EI;
                float *f=NULL; int64_t got=deq(fd,ti,&f,fsz);
                if(!got){ MERR("cannot dequantize %s",nm); goto fail2; }
                if(got != II*OO*(int64_t)c->n_expert){
                    MERR("%s: %lld elements, expected %lldx%lldx%d",nm,(long long)got,
                         (long long)II,(long long)OO,c->n_expert); free(f); goto fail2; }
                for (int e=0;e<c->n_expert;e++) {
                    coli_w_i8 *dst = (t3==0)?&L->e_gate[e]:(t3==1)?&L->e_up[e]:&L->e_down[e];
                    quant_rows(f + (int64_t)e*II*OO, dst, II, OO);
                }
                free(f);
            }
        } else {
        LW(gate,"ffn_gate.weight",D,c->inter,1);
        LW(up  ,"ffn_up.weight"  ,D,c->inter,1);
        LW(down,"ffn_down.weight",c->inter,D,1);
        }
        if (l==0 || l==c->n_layers-1) { fprintf(stderr,"  layer %d loaded\n", l); fflush(stderr); }
    }
    close(fd);

    m->max_ctx = max_ctx>0?max_ctx:(c->ctx_train?c->ctx_train:2048);
    m->K=(float**)xmal(sizeof(float*)*(size_t)c->n_layers);
    m->V=(float**)xmal(sizeof(float*)*(size_t)c->n_layers);
    for (int l=0;l<c->n_layers;l++){
        m->K[l]=fal((int64_t)c->n_kv_heads*m->max_ctx*hd);
        m->V[l]=fal((int64_t)c->n_kv_heads*m->max_ctx*hd); }
    m->n_past = 0;

    { Tok *T=(Tok*)calloc(1,sizeof(Tok)); int b,eo,ab;
      tok_load_gguf(T,path,&b,&eo,&ab);
      m->tok=T; c->bos=b; c->eos=eo; c->add_bos=ab; }

    gguf_meta_close(&mt);
    return m;
fail2: close(fd);
fail:  gguf_meta_close(&mt); free(m); return NULL;
}

void coli_free(coli_model *m) {
    if (!m) return;
    if (m->tok) { tok_free((Tok*)m->tok); free(m->tok); }
    free(m->out_norm); free(m->rope_ff);
    if (m->K) for (int l=0;l<m->cfg.n_layers;l++) free(m->K[l]);
    if (m->V) for (int l=0;l<m->cfg.n_layers;l++) free(m->V[l]);
    free(m->K); free(m->V); free(m->L); free(m);
}

int coli_encode(coli_model *m, const char *text, int *out, int max) {
    return tok_encode((Tok*)m->tok, text, (int)strlen(text), out, max); }
int coli_decode(coli_model *m, const int *ids, int n, char *out, int max) {
    return tok_decode((Tok*)m->tok, ids, n, out, max); }

/* -------------------------------------------------------------- primitives */
static void rmsnorm(float *o, const float *x, const float *g, int64_t n, float eps) {
    double ss=0; for (int64_t i=0;i<n;i++) ss += (double)x[i]*x[i];
    float inv = 1.f/sqrtf((float)(ss/(double)n)+eps);
    for (int64_t i=0;i<n;i++) o[i]=x[i]*inv*g[i];
}

/* `ff` is llama-3.1's rope_freqs.weight, applied by ggml as theta/ff
 * (ggml-cpu/ops.cpp:5619). NULL when the model has no such tensor. */
static void rope_head(float *v, int pos, int hd, float theta, const float *ff, int neox) {
    int half=hd/2;
    for (int i=0;i<half;i++) {
        float fr = powf(theta, -(float)(2*i)/(float)hd);
        if (ff) fr /= ff[i];
        float c=cosf(pos*fr), s=sinf(pos*fr);
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
    coli_a_i8 a; a_alloc(&a,n,w->I);
    coli_quantize_a(&a,x,n,w->I);
    coli_gemm_i8(y,&a,w);
    a_free(&a);
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

/* ---------------------------------------------------------------- forward */
float *coli_forward(coli_model *m, const int *ids, int S, int all_logits) {
    coli_cfg *c=&m->cfg;
    int D=c->hidden,H=c->n_heads,KVH=c->n_kv_heads,hd=c->head_dim;
    int qD=H*hd, kvD=KVH*hd, grp=H/KVH, pos0=m->n_past;
    if (pos0+S > m->max_ctx) { fprintf(stderr,"context overflow: %d+%d > %d\n",pos0,S,m->max_ctx); return NULL; }

    float *x=fal((int64_t)S*D);
    for (int s=0;s<S;s++) {
        int id=ids[s];
        if (id<0||id>=c->vocab){ fprintf(stderr,"token %d out of range [0,%d)\n",id,c->vocab); free(x); return NULL; }
        const uint8_t *r=m->tok_embd.qu+(int64_t)id*D; float sc=m->tok_embd.scale[id];
        for (int i=0;i<D;i++) x[(int64_t)s*D+i]=((int)r[i]-128)*sc;
    }
    float *xb=fal((int64_t)S*D);
    float *q=fal((int64_t)S*qD),*k=fal((int64_t)S*kvD),*vv=fal((int64_t)S*kvD);
    float *att=fal((int64_t)S*qD);
    float *g=fal((int64_t)S*c->inter),*u=fal((int64_t)S*c->inter);

    for (int l=0;l<c->n_layers;l++) {
        coli_layer *L=&m->L[l];
        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->attn_norm,D,c->eps);
        mm(q,xb,S,&L->wq); mm(k,xb,S,&L->wk); mm(vv,xb,S,&L->wv);
        if (L->bq) for (int s=0;s<S;s++) for (int i=0;i<qD;i++)  q[(int64_t)s*qD+i]+=L->bq[i];
        if (L->bk) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) k[(int64_t)s*kvD+i]+=L->bk[i];
        if (L->bv) for (int s=0;s<S;s++) for (int i=0;i<kvD;i++) vv[(int64_t)s*kvD+i]+=L->bv[i];

        for (int s=0;s<S;s++) {
            int pos=pos0+s;
            for (int h=0;h<H;h++)   rope_head(q+(int64_t)s*qD +h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
            for (int h=0;h<KVH;h++) rope_head(k+(int64_t)s*kvD+h*hd,pos,hd,c->rope_theta,m->rope_ff,c->rope);
        }
        for (int s=0;s<S;s++) for (int h=0;h<KVH;h++) {
            int64_t row=(int64_t)h*m->max_ctx+(pos0+s);
            memcpy(m->K[l]+row*hd,k +(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
            memcpy(m->V[l]+row*hd,vv+(int64_t)s*kvD+h*hd,(size_t)hd*sizeof(float));
        }
        float scale=1.f/sqrtf((float)hd);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h=0;h<H;h++) for (int s=0;s<S;s++) {
            int kvh=h/grp;                      /* GQA: many q heads share one kv head */
            const float *qv=q+(int64_t)s*qD+h*hd;
            int tmax=pos0+s;                    /* causal */
            float *sc=(float*)xmal((size_t)(tmax+1)*sizeof(float));
            float mx=-1e30f;
            for (int t=0;t<=tmax;t++){ const float *kv=m->K[l]+((int64_t)kvh*m->max_ctx+t)*hd;
                float d=0; for(int i=0;i<hd;i++) d+=qv[i]*kv[i];
                d*=scale; sc[t]=d; if(d>mx)mx=d; }
            float sum=0; for(int t=0;t<=tmax;t++){ sc[t]=expf(sc[t]-mx); sum+=sc[t]; }
            float inv=1.f/sum; float *o=att+(int64_t)s*qD+h*hd;
            for(int i=0;i<hd;i++) o[i]=0.f;
            for(int t=0;t<=tmax;t++){ float w=sc[t]*inv;
                const float *vp=m->V[l]+((int64_t)kvh*m->max_ctx+t)*hd;
                for(int i=0;i<hd;i++) o[i]+=w*vp[i]; }
            free(sc);
        }
        mm(xb,att,S,&L->wo);
        for (int s=0;s<S;s++) for (int i=0;i<D;i++) x[(int64_t)s*D+i]+=xb[(int64_t)s*D+i];

        for (int s=0;s<S;s++) rmsnorm(xb+(int64_t)s*D,x+(int64_t)s*D,L->ffn_norm,D,c->eps);
        if (c->n_expert > 0) {
            moe_ffn(m,L,x,xb,S);
        } else {
        mm(g,xb,S,&L->gate); mm(u,xb,S,&L->up);
        for (int64_t i=0;i<(int64_t)S*c->inter;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
        mm(xb,g,S,&L->down);
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
