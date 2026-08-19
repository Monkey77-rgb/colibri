/* coli_api.h — the ABI boundary. C++ below, Go and Python above.
 *
 * WHY extern "C" AND NOT THE C++ HEADERS DIRECTLY. cgo can call C, not C++: it
 * has no way to express name mangling, exceptions, or non-trivial destructors
 * across the boundary. Python's ctypes has the same limitation. So the engine
 * keeps its C++ internals (RAII, templates, std::vector) and exposes a flat C
 * surface. This is the same shape MLX uses -- C++ core, thin bindings above --
 * and the reason is identical.
 *
 * RULES FOR ANYTHING ADDED HERE, because an ABI is a promise:
 *  - POD arguments only. No std::string, no std::vector, no references.
 *  - The caller owns every buffer it passes in; the engine never frees them.
 *  - Anything the engine allocates is freed by a matching coli_*_free.
 *  - Return 0 for success and negative for failure. Not bool -- callers in two
 *    other languages have to map it, and 0-is-success is the C convention both
 *    already expect from errno-style APIs.
 *  - Never remove or reorder a function. Add.
 *
 * THREADING. A coli_ctx is NOT thread-safe. It owns one KV cache and one set of
 * scratch buffers. The Go layer must serialise calls into a single context --
 * which it does naturally, because batching means one scheduler goroutine owns
 * the context and everything else talks to it over a channel. Concurrency lives
 * in the scheduler, not in the engine, and that is deliberate: two threads
 * stepping the same KV cache is a data race no amount of locking inside the
 * kernels would fix.
 */
#ifndef COLI_API_H
#define COLI_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct coli_ctx coli_ctx;

/* ---- lifecycle ---- */

/* n_slots is the maximum number of sequences that can decode together. Each slot
 * costs n_layers * 2 * n_kv_heads * ctx * head_dim * 4 bytes of KV cache, so it
 * is a memory decision, not a free knob. wq_int8=0 keeps f32 weights (~4x memory,
 * much slower) and exists to validate a new architecture without quantization
 * loss confusing the result. */
coli_ctx *coli_open(const char *gguf_path, int n_ctx, int n_slots, int wq_int8,
                    char *err, size_t errcap);

/* Same, plus the weight format: w4 = 0 int8 only (what coli_open uses) ·
 * 1 both formats chosen per batch · 2 int4 only.
 *
 * A SEPARATE FUNCTION rather than a widened coli_open, per the ABI rules above:
 * every Go and Python caller already compiled against the old signature keeps
 * working. w4=2 measured at 0.68x peak RSS and 1.42x decode against w4=0, for
 * 1.08x prefill and +3.87% NLL on qwen2.5-3b -- an accuracy trade the caller has
 * to make deliberately, which is the other reason it is not the default. */
coli_ctx *coli_open_w4(const char *gguf_path, int n_ctx, int n_slots, int wq_int8,
                       int w4, char *err, size_t errcap);
void      coli_close(coli_ctx *c);

/* ---- introspection, so the upper layers need not guess ---- */
int   coli_n_vocab(coli_ctx *c);
int   coli_n_ctx(coli_ctx *c);
int   coli_n_slots(coli_ctx *c);
int   coli_bos(coli_ctx *c);
int   coli_eos(coli_ctx *c);
int   coli_add_bos(coli_ctx *c);
const char *coli_arch(coli_ctx *c);
const char *coli_kernel(coli_ctx *c, int batch);   /* which kernel a batch would pick */

/* ---- tokenizer ---- */
int coli_tokenize(coli_ctx *c, const char *text, int32_t *out, int max);
/* Writes UTF-8 into out; returns bytes written, or negative. */
int coli_detokenize(coli_ctx *c, const int32_t *ids, int n, char *out, int max);

/* ---- inference ----
 * The batched step. `slots`, `positions` and `tokens` are n-element arrays; one
 * token is decoded for each sequence. `logits_out` receives n * n_vocab floats.
 * Sequences may sit at different positions and in any slot -- that is what makes
 * the batching continuous rather than lock-step. */
int coli_decode_batch(coli_ctx *c, const int32_t *slots, const int32_t *positions,
                      const int32_t *tokens, int n, float *logits_out);

/* Prefill `n` tokens into `slot`, leaving logits for the last one in logits_out
 * (n_vocab floats). */
int coli_prefill(coli_ctx *c, int slot, const int32_t *ids, int n, float *logits_out);

/* ---- prefix cache: control and visibility ----
 *
 * coli_prefix_cache_set turns KV prefix reuse on or off. Turn it OFF for any
 * benchmark that re-sends a prompt: with it on, the second and later sends
 * match the cached prefix and skip prefill, so the run measures the cache and
 * not the work. NOTE the flag is process-wide in the engine today, so the ctx
 * argument is accepted for future-proofing and currently ignored.
 *
 * coli_prefix_stats returns cumulative prompt tokens REUSED and ASKED across
 * every coli_prefill on this ctx; reused/asked is the hit rate. Either pointer
 * may be NULL.
 *
 * These existed in model.h from the day the cache shipped and were bound into
 * nothing, so the server ran the cache on every request with no way to disable
 * it and no way to see whether it hit. An optimisation you cannot observe is
 * an optimisation you cannot validate. */
void coli_prefix_cache_set(coli_ctx *c, int on);
void coli_prefix_stats(coli_ctx *c, long long *reused, long long *asked);

/* ---- sampling ----
 * Stateless apart from `rng_state`, which the CALLER owns and threads through.
 * A sampler holding its own static RNG is how "same seed, same output" became
 * false across requests in the first server; making the state an explicit
 * argument means the Go layer keeps one per request and cannot repeat it. */
typedef struct {
    float    temp;            /* <=0 => greedy */
    int32_t  top_k;           /* 0 = off */
    float    top_p;           /* >=1 = off */
    float    min_p;           /* 0 = off */
    float    repeat_penalty;  /* 1 = off */
    int32_t  repeat_last_n;
    uint64_t seed;
    uint64_t rng_state[4];    /* caller-owned; zero-initialise before first use */
} coli_sample_params;

int32_t coli_sample(coli_ctx *c, float *logits, coli_sample_params *p,
                    const int32_t *prev, int n_prev);

#ifdef __cplusplus
}
#endif
#endif
