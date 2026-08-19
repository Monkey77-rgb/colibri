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

/* A weight that may exist in BOTH formats. Which one runs is decided per call by
 * batch size -- int4 for decode (2.3x, bound by bytes), int8 for prefill (1.5x,
 * bound by ALU). Holding both costs 1.5x the int8 memory; whether that trade is
 * worth it is the open question, so it is opt-in (`--w4`, the w4 argument to
 * coli_load) and measured rather
 * than assumed. */
typedef struct { coli_w_i8 i8; coli_w_i4 i4; int have4; } coli_wpair;

typedef struct {
    float *attn_norm, *ffn_norm;
    coli_w_i8 wq, wk, wv, wo;
    /* qwen3 / qwen3moe: RMSNorm applied to each attention head's q and k vector
     * after projection and BEFORE RoPE. Length head_dim, shared across heads.
     * NULL on qwen2 and llama, which do not have them. */
    float *q_norm, *k_norm;
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
    /* KV cache, GQA-shaped (n_kv_heads rows, never n_heads -- n_heads would be
     * 4-8x the memory for identical arithmetic) and SLOT-shaped: each concurrent
     * sequence owns a disjoint region, so K[l] is [n_slots][n_kv_heads][max_ctx][hd].
     * Slot 0 is what the single-sequence path uses, so nothing above had to change. */
    float      **K, **V;
    int          max_ctx, n_past;
    /* KV actually ALLOCATED per slot, which is <= max_ctx and grows on demand.
     * Reserving max_ctx up front made the cache the single largest allocation in
     * the process -- 2.25 GiB against 2.29 GiB of weights on qwen2.5-3b at
     * 32768 ctx -- and almost all of it was never touched. */
    int          kv_ctx;
    int          n_slots;
    /* PREFIX CACHE -- the token ids currently resident in each slot's KV, so a
     * prefill that shares a leading prefix with the last one can skip re-running
     * it. RoPE is position-dependent, but a shared PREFIX occupies the same
     * positions by definition, so its cached K/V stay valid unchanged. That is
     * the whole reason this works for prefixes and would NOT work for a shared
     * suffix or an insertion in the middle.
     * cache_ids is [n_slots][kv_ctx], grown alongside the KV itself. */
    int        **cache_ids;
    int         *cache_len;
    void        *tok;             /* Tok*, opaque here to keep tok.h out of this header */
} coli_model;

/* Load. `err` gets a reason on failure. wq_int8=1 requantizes to int8 (fast,
 * lossy); 0 keeps f32 (slow, large, but isolates architecture bugs from
 * quantization loss -- do the f32 run first when validating a new arch). */
/* n_slots: concurrent sequences the KV cache must hold. Was an environment
 * variable; that was a shortcut, and it broke the Windows build because setenv
 * is POSIX. A parameter is also simply correct -- the caller knows, and a
 * process-global is the wrong scope for a per-model property. */
/* w4: weight format. 0 = int8 only · 1 = both, chosen per batch · 2 = int4 only.
 * A PARAMETER, not an environment variable, for the reason written directly
 * above about n_slots -- and because the format is a property of one model, not
 * of the process. A server holding two models must be able to load one int4 and
 * one int8, which a global forecloses. */
coli_model *coli_load(const char *gguf_path, int max_ctx, int n_slots, int wq_int8,
                      int w4, char *err, size_t errcap);

/* Move this model's weight matrices onto the GPU and route mm() through the
 * Vulkan backend. Returns the number of matrices uploaded, or a negative value
 * with a reason in err.
 *
 * REQUIRES int4 weights (w4 == 2): the GPU path has one kernel, gemm_i4, and a
 * fused FFN built on it. Calling this on an int8 model is refused rather than
 * silently ignored.
 *
 * Separate from coli_load ON PURPOSE. Uploading is slow, needs a device that may
 * not exist, and can run out of VRAM -- three failure modes that a loader should
 * not silently absorb. The caller decides, and gets told what happened. */
/* Rebuild the int4 weights weighting the scale search by measured ACTIVATION
 * magnitude per input channel instead of weight magnitude -- the cheap end of
 * AWQ. Needs w4=1 on load; leaves the model int4-only. Returns matrices rebuilt. */
int coli_awq_calibrate(coli_model *m, const int *ids, int n, char *err, size_t errcap);

int coli_gpu_upload(coli_model *m, char *err, size_t errcap);
void coli_gpu_release(coli_model *m);
void        coli_free(coli_model *m);

/* ---------------------------------------------------- continuous batching ---
 * One decode step for `n` sequences at once. Each carries its own slot and its
 * own position, so sequences that started at different times step together --
 * that is what "continuous" means: a finishing sequence frees its slot and a
 * waiting one takes it mid-flight, without draining the batch.
 *
 * WHY THIS IS THE POINT OF THE WHOLE ENGINE. A single-slot server runs every
 * generated token at n=1, which is the regime measured to be DRAM-bound, where
 * the wide kernel is 17-22% SLOWER and there is nothing to be gained by a better
 * ISA. Batching k sequences turns each decode step into an n=k GEMM: past
 * COLI_GEMM_MIN_WIDE the wide kernel is selected and, more importantly, the
 * weights are read ONCE for all k sequences instead of k times. The weight
 * traffic per token falls by k.
 *
 * `logits` receives n rows of vocab. Positions are advanced by the caller. */
typedef struct { int slot; int pos; int token; } coli_seq;
int coli_decode_batch(coli_model *m, coli_seq *seq, int n, float *logits);

/* Prefill a prompt into `slot`, returning logits for the LAST token only. */
float *coli_prefill_slot(coli_model *m, int slot, const int *ids, int n);

/* Tokens the last coli_prefill_slot reused from cache rather than recomputing,
 * and total tokens it was asked for. Reset per call. For measurement -- a cache
 * whose hit rate you cannot see is a cache you cannot evaluate. */
int  coli_prefix_reused(const coli_model *m);
int  coli_prefix_asked(const coli_model *m);
/* Off by default? No -- ON by default, with this to disable it for A/B. A
 * correctness-neutral optimisation that ships off is one nobody measures. */
void coli_prefix_cache_enable(int on);

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
    /* RNG state, per sampler instance. A file-static RNG made "same seed, same
     * output" FALSE across requests in the server: the second request had the
     * same seed, so the reseed was skipped and it continued the previous
     * stream. Caught by issuing the same request twice and diffing. */
    uint64_t rng[4];
    int      rng_ready;
} coli_sampler;

void coli_sampler_default(coli_sampler *s);
/* prev/n_prev feed the repetition penalty; pass NULL/0 to disable. */
int  coli_sample(coli_sampler *s, float *logits, int n_vocab,
                 const int *prev, int n_prev);

#endif
