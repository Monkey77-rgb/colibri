/* model.h — dense + MoE transformer, loaded from GGUF.
 *
 * SCOPE, stated so it cannot be overread: `qwen2` and `llama` dense, and MoE
 * models whose expert tensors follow the ffn_gate_exps/ffn_up_exps/ffn_down_exps
 * layout. Anything else is REFUSED at load, not guessed at. A loader that
 * silently accepts an architecture it has not been validated against produces
 * fluent nonsense, which is the hardest failure to notice.
 *
 * ⚠ VALIDATE EVERY ARCHITECTURE SEPARATELY. This is not a style note. The
 * predecessor of this file was validated on qwen2 alone (96.6% top-1 against
 * llama.cpp) and the llama path was assumed to follow because it loaded and
 * produced readable English. It did not: llama.cpp gives LLAMA and QWEN2
 * DIFFERENT RoPE pairings, and the llama path ran at perplexity 639 against
 * llama.cpp's 29 on the same file. Adding an architecture means adding a
 * measurement, every time.
 */
#ifndef COLI_MODEL_H
#define COLI_MODEL_H

#include <stdint.h>
#include "gemm_i8.h"

/* RoPE pairing. NOT a global constant -- see llama_model_rope_type() in
 * llama.cpp/src/llama-model.cpp: QWEN2 -> NEOX, LLAMA -> NORM. */
typedef enum { COLI_ROPE_NEOX = 1, COLI_ROPE_INTERLEAVED = 0 } coli_rope_kind;

typedef struct {
    char  arch[32];
    int   hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, ctx_train;
    float rope_theta, eps;
    int   qkv_bias;          /* qwen2 carries attn_{q,k,v}.bias; llama does not */
    coli_rope_kind rope;
    int   bos, eos, add_bos;
    /* MoE. n_expert == 0 means dense. */
    int   n_expert, n_expert_used, expert_inter;
} coli_cfg;

typedef struct {
    float *attn_norm, *ffn_norm;
    coli_w_i8 wq, wk, wv, wo;
    float *bq, *bk, *bv;          /* NULL when the arch has no qkv bias */
    /* dense FFN */
    coli_w_i8 gate, up, down;
    /* MoE: one coli_w_i8 per expert, plus the router */
    coli_w_i8 *e_gate, *e_up, *e_down;
    coli_w_i8  router;
} coli_layer;

typedef struct {
    coli_cfg     cfg;
    coli_layer  *L;
    coli_w_i8    tok_embd, out;
    float       *out_norm;
    float       *rope_ff;         /* rope_freqs.weight (llama-3.1); NULL if absent */
    /* KV cache, GQA-shaped: n_kv_heads rows per layer, never n_heads. Storing
     * n_heads rows would be 4-8x the memory for identical arithmetic. */
    float      **K, **V;
    int          max_ctx, n_past;
    void        *tok;             /* Tok*, opaque here to keep tok.h out of this header */
} coli_model;

/* Load. `err` gets a reason on failure. wq_int8=1 requantizes to int8 (fast,
 * lossy); 0 keeps f32 (slow, large, but isolates architecture bugs from
 * quantization loss -- do the f32 run first when validating a new arch). */
coli_model *coli_load(const char *gguf_path, int max_ctx, int wq_int8,
                      char *err, size_t errcap);
void        coli_free(coli_model *m);

/* Forward over `n` tokens starting at position m->n_past. Returns logits for
 * the LAST token only when all_logits==0, or for every token when 1 (the
 * teacher-forcing path used by the NLL test). Caller frees. */
float *coli_forward(coli_model *m, const int *ids, int n, int all_logits);

/* Tokenizer passthrough. Returns token count. */
int  coli_encode(coli_model *m, const char *text, int *out, int max);
int  coli_decode(coli_model *m, const int *ids, int n, char *out, int max);

/* --------------------------------------------------------------- sampling */
typedef struct {
    float temp;        /* <=0 => greedy */
    int   top_k;       /* 0 = off */
    float top_p;       /* >=1 = off */
    float min_p;       /* 0 = off */
    float repeat_penalty;
    int   repeat_last_n;
    uint64_t seed;
} coli_sampler;

void coli_sampler_default(coli_sampler *s);
/* prev/n_prev feed the repetition penalty; pass NULL/0 to disable. */
int  coli_sample(coli_sampler *s, float *logits, int n_vocab,
                 const int *prev, int n_prev);

#endif
