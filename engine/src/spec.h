/* spec.h -- draftless n-gram speculative decoding for the CPU path.
 *
 * Decode is memory-bandwidth-bound: reading the weights dominates and costs the
 * same whether the batch is 1 token or k. So we PREDICT k tokens from an n-gram
 * count cache (no second model, no extra weight sweep), VERIFY all k+1 in ONE
 * coli_decode_batch, and keep the longest correct prefix. The first mismatch's
 * own sampled token is always kept, so a round nets >=1 real token and NEVER
 * regresses below plain decode -- only a few cheap hash lookups of overhead.
 *
 * Reproduces llama.cpp's lookup decoding (common/ngram-cache.cpp,
 * examples/lookup/lookup.cpp): the cache is n-gram -> {next-token -> count},
 * sizes 1..LLAMA_NGRAM_MAX, drafted longest-first, gated by sample-size and
 * dominance thresholds. Context cache only here (the live sequence) -- which the
 * upstream notes is where most of the win on repetitive/structured output comes
 * from, and it needs no on-disk corpus.
 *
 * ROLLBACK IS FREE in this engine: KV is a positioned array and attend_online
 * reads only t<=pos, so rejected drafts (written at higher positions) are never
 * read and are overwritten next round -- no truncation primitive needed. */
#ifndef COLI_SPEC_H
#define COLI_SPEC_H

#define COLI_NGRAM_MIN 1
#define COLI_NGRAM_MAX 4

typedef struct NgramCache NgramCache;

NgramCache *ngram_new(void);
void        ngram_free(NgramCache *);

/* Record n-grams from the token stream. Only the last `nnew` tokens' n-grams are
 * added (the stream is append-only), so this is O(nnew * NGRAM_MAX) per call. */
void ngram_update(NgramCache *, const int *toks, int n_tokens, int nnew);

/* Fill draft[1..] with up to max_draft predicted tokens that follow the sequence
 * toks[0..n-1] (draft[0] must already hold the seed = the pending token). Returns
 * the total draft length including the seed (>=1). Autoregressive: each drafted
 * token extends the window the next lookup sees. */
int ngram_draft(NgramCache *, const int *toks, int n, int *draft, int max_draft);

#endif
