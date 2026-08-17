/* sample.c — sampling.
 *
 * DETERMINISM. The RNG is an explicit xoshiro256** seeded from coli_sampler.seed,
 * not rand(). A shared global RNG makes two runs with the same seed disagree the
 * moment anything else in the process draws a number, and "same seed, same
 * output" is the only handle anyone has on reproducing a bad generation.
 *
 * ORDER OF FILTERS matters and is fixed here as: repetition penalty -> temp ->
 * top-k -> top-p -> min-p. Applying temperature after top-p, for instance, gives
 * a different distribution for the same parameters; llama.cpp exposes the order
 * as a `samplers` list precisely because it is not canonical. Ours is fixed and
 * documented rather than implicit.
 */
#define _GNU_SOURCE
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void coli_sampler_default(coli_sampler *s) {
    s->temp=0.f; s->top_k=0; s->top_p=1.f; s->min_p=0.f;
    s->repeat_penalty=1.f; s->repeat_last_n=0; s->seed=0;
}

/* xoshiro256** — small, fast, and good enough; seeded via splitmix64 so a seed
 * of 0 does not degenerate. */
static uint64_t sm64(uint64_t *x){ uint64_t z=(*x+=0x9E3779B97F4A7C15ull);
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31); }
static uint64_t rotl(uint64_t x,int k){ return (x<<k)|(x>>(64-k)); }
typedef struct { uint64_t s[4]; } rng_t;
static void rng_seed(rng_t *r, uint64_t seed){ uint64_t x=seed?seed:0x243F6A8885A308D3ull;
  for(int i=0;i<4;i++) r->s[i]=sm64(&x); }
static uint64_t rng_next(rng_t *r){
  uint64_t *s=r->s, res=rotl(s[1]*5,7)*9, t=s[1]<<17;
  s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45);
  return res; }
static float rng_f(rng_t *r){ return (float)((rng_next(r)>>11)*0x1.0p-53); }

typedef struct { int id; float p; } cand;
static int cmp_desc(const void *a,const void *b){
    float x=((const cand*)a)->p, y=((const cand*)b)->p;
    return x<y ? 1 : (x>y ? -1 : 0); }

int coli_sample(coli_sampler *s, float *logits, int n_vocab, const int *prev, int n_prev) {
    /* greedy: no allocation, no RNG draw, so a temp<=0 run is bit-reproducible
     * regardless of seed */
    if (s->temp <= 0.f && s->top_k == 0 && s->top_p >= 1.f && s->min_p <= 0.f
        && s->repeat_penalty == 1.f) {
        int b=0; for (int i=1;i<n_vocab;i++) if (logits[i]>logits[b]) b=i;
        return b;
    }

    /* 1. repetition penalty, on the last repeat_last_n tokens */
    if (s->repeat_penalty != 1.f && prev && n_prev > 0) {
        int start = n_prev - (s->repeat_last_n>0 ? s->repeat_last_n : n_prev);
        if (start < 0) start = 0;
        for (int i=start;i<n_prev;i++) {
            int t=prev[i]; if (t<0||t>=n_vocab) continue;
            /* divide when positive, multiply when negative -- penalising a
             * negative logit by dividing would INCREASE it */
            logits[t] = logits[t] > 0 ? logits[t]/s->repeat_penalty
                                      : logits[t]*s->repeat_penalty;
        }
    }

    /* 2. temperature, then softmax */
    float t = s->temp > 0.f ? s->temp : 1.f;
    float mx=-1e30f; for (int i=0;i<n_vocab;i++) if (logits[i]>mx) mx=logits[i];
    cand *cs = (cand*)malloc((size_t)n_vocab*sizeof(cand));
    double sum=0;
    for (int i=0;i<n_vocab;i++){ double p=exp(((double)logits[i]-mx)/t); cs[i].id=i; cs[i].p=(float)p; sum+=p; }
    for (int i=0;i<n_vocab;i++) cs[i].p=(float)(cs[i].p/sum);

    qsort(cs,(size_t)n_vocab,sizeof(cand),cmp_desc);
    int n = n_vocab;

    /* 3. top-k */
    if (s->top_k > 0 && s->top_k < n) n = s->top_k;

    /* 4. top-p (nucleus): keep the shortest prefix whose mass reaches top_p */
    if (s->top_p < 1.f) {
        double acc=0; int keep=n;
        for (int i=0;i<n;i++){ acc+=cs[i].p; if (acc >= s->top_p) { keep=i+1; break; } }
        n=keep;
    }

    /* 5. min-p: drop anything below min_p * p_max. Cheap and shape-aware --
     * on a peaked distribution it keeps 1-2 tokens, on a flat one many. */
    if (s->min_p > 0.f) {
        float thr = s->min_p * cs[0].p; int keep=n;
        for (int i=0;i<n;i++) if (cs[i].p < thr) { keep=i; break; }
        if (keep < 1) keep = 1;
        n = keep;
    }

    /* renormalize over the survivors and draw */
    double tot=0; for (int i=0;i<n;i++) tot+=cs[i].p;
    static rng_t rng; static int seeded=0; static uint64_t cur_seed=~0ull;
    if (!seeded || cur_seed != s->seed) { rng_seed(&rng,s->seed); seeded=1; cur_seed=s->seed; }
    double r = (double)rng_f(&rng)*tot, acc=0; int pick=cs[n-1].id;
    for (int i=0;i<n;i++){ acc+=cs[i].p; if (r<=acc) { pick=cs[i].id; break; } }
    free(cs);
    return pick;
}
