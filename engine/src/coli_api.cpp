/* coli_api.cpp — the extern "C" façade over the C++ engine. See coli_api.h. */
#include "coli_api.h"
#include "model.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

struct coli_ctx {
    coli_model *m = nullptr;
    ~coli_ctx(){ if (m) coli_free(m); }
};

coli_ctx *coli_open(const char *path, int n_ctx, int n_slots, int wq_int8,
                    char *err, size_t errcap) {
    if (n_slots < 1) n_slots = 1;
    coli_ctx *c = new coli_ctx();
    c->m = coli_load(path, n_ctx, n_slots, wq_int8, err, errcap);
    if (!c->m) { delete c; return nullptr; }
    return c;
}
void coli_close(coli_ctx *c){ delete c; }

int coli_n_vocab(coli_ctx *c){ return c->m->cfg.vocab; }
int coli_n_ctx  (coli_ctx *c){ return c->m->max_ctx; }
int coli_n_slots(coli_ctx *c){ return c->m->n_slots; }
int coli_bos    (coli_ctx *c){ return c->m->cfg.bos; }
int coli_eos    (coli_ctx *c){ return c->m->cfg.eos; }
int coli_add_bos(coli_ctx *c){ return c->m->cfg.add_bos; }
const char *coli_arch(coli_ctx *c){ return c->m->cfg.arch; }
const char *coli_kernel(coli_ctx *c, int batch){
    return coli_gemm_i8_kernel(batch, c->m->cfg.hidden, c->m->cfg.hidden); }

int coli_tokenize(coli_ctx *c, const char *text, int32_t *out, int max){
    static_assert(sizeof(int32_t)==sizeof(int), "int32_t must match int for the token ABI");
    return coli_encode(c->m, text, (int*)out, max); }
int coli_detokenize(coli_ctx *c, const int32_t *ids, int n, char *out, int max){
    return coli_decode(c->m, (const int*)ids, n, out, max); }

int coli_decode_batch(coli_ctx *c, const int32_t *slots, const int32_t *pos,
                      const int32_t *tok, int n, float *logits){
    if (n < 1) return -1;
    coli_seq *s = (coli_seq*)malloc(sizeof(coli_seq)*(size_t)n);
    if (!s) return -2;
    for (int i=0;i<n;i++){ s[i].slot=slots[i]; s[i].pos=pos[i]; s[i].token=tok[i]; }
    int r = coli_decode_batch(c->m, s, n, logits);
    free(s);
    return r;
}

int coli_prefill(coli_ctx *c, int slot, const int32_t *ids, int n, float *logits){
    float *lg = coli_prefill_slot(c->m, slot, (const int*)ids, n);
    if (!lg) return -1;
    memcpy(logits, lg, (size_t)c->m->cfg.vocab*sizeof(float));
    free(lg);
    return 0;
}

int32_t coli_sample(coli_ctx *c, float *logits, coli_sample_params *p,
                    const int32_t *prev, int n_prev){
    coli_sampler s;
    coli_sampler_default(&s);
    s.temp=p->temp; s.top_k=p->top_k; s.top_p=p->top_p; s.min_p=p->min_p;
    s.repeat_penalty=p->repeat_penalty; s.repeat_last_n=p->repeat_last_n;
    s.seed=p->seed;
    /* Thread the caller's RNG state in and back out, so the sequence continues
     * across calls without the engine holding any per-request state. */
    memcpy(s.rng, p->rng_state, sizeof s.rng);
    s.rng_ready = (p->rng_state[0]|p->rng_state[1]|p->rng_state[2]|p->rng_state[3]) ? 1 : 0;
    int32_t t = (int32_t)::coli_sample(&s, logits, c->m->cfg.vocab, (const int*)prev, n_prev);
    memcpy(p->rng_state, s.rng, sizeof s.rng);
    return t;
}
