/* spec.cpp -- see spec.h. n-gram count cache + draft, mirroring llama.cpp's
 * common_ngram_cache_update / common_ngram_cache_draft. Pure integer/hashmap
 * work; no SIMD, negligible cost next to a decode. */
#include "spec.h"
#include <unordered_map>
#include <cstdint>
#include <cstring>

/* An n-gram key: up to NGRAM_MAX token ids, unused slots = -1. Fixed size so a
 * 1..4-gram share one type, hashed and compared elementwise (as llama does). */
struct Ngram {
    int t[COLI_NGRAM_MAX];
    Ngram(){ for (int i=0;i<COLI_NGRAM_MAX;i++) t[i]=-1; }
    Ngram(const int *src, int len){
        for (int i=0;i<COLI_NGRAM_MAX;i++) t[i] = (i<len) ? src[i] : -1;
    }
    bool operator==(const Ngram &o) const {
        for (int i=0;i<COLI_NGRAM_MAX;i++) if (t[i]!=o.t[i]) return false;
        return true;
    }
};
struct NgramHash {
    size_t operator()(const Ngram &g) const {
        /* XOR of per-token Fibonacci hashes, as common_ngram_hash_function. */
        size_t h = 0;
        for (int i=0;i<COLI_NGRAM_MAX;i++)
            h ^= (size_t)(g.t[i]) * 11400714819323198485ULL + (h<<6) + (h>>2);
        return h;
    }
};

/* n-gram -> (next-token -> count). */
typedef std::unordered_map<int,int32_t> Part;
struct NgramCache { std::unordered_map<Ngram, Part, NgramHash> m; };

NgramCache *ngram_new(void){ return new NgramCache(); }
void ngram_free(NgramCache *c){ delete c; }

void ngram_update(NgramCache *c, const int *toks, int n, int nnew){
    for (int size = COLI_NGRAM_MIN; size <= COLI_NGRAM_MAX; ++size) {
        /* only the new tail: the n-gram ending just before i is followed by toks[i] */
        int i_start = n - nnew; if (i_start < size) i_start = size;
        for (int i = i_start; i < n; ++i) {
            Ngram key(&toks[i-size], size);
            c->m[key][toks[i]]++;
        }
    }
}

/* Lax thresholds from ngram-cache.cpp:60-63, indexed by n-gram size-1. */
static const int   MIN_SAMPLE[COLI_NGRAM_MAX] = {2,2,1,1};
static const int   MIN_PCT   [COLI_NGRAM_MAX] = {66,50,50,50};

/* Best next token for the longest n-gram (size 4..1) ending at seq[0..len-1]
 * that clears its sample-size and dominance gates. Returns -1 if none. */
static int draft_one(NgramCache *c, const int *seq, int len){
    for (int size = COLI_NGRAM_MAX; size >= COLI_NGRAM_MIN; --size) {
        if (len < size) continue;
        Ngram key(&seq[len-size], size);
        auto it = c->m.find(key);
        if (it == c->m.end()) continue;
        const Part &p = it->second;
        int best = -1; int32_t best_c = 0, sum = 0;
        for (const auto &kv : p) { sum += kv.second; if (kv.second > best_c) { best_c = kv.second; best = kv.first; } }
        if (sum < MIN_SAMPLE[size-1]) continue;              /* too few samples */
        if (100*best_c < MIN_PCT[size-1]*sum) continue;      /* not dominant enough */
        return best;
    }
    return -1;
}

int ngram_draft(NgramCache *c, const int *toks, int n, int *draft, int max_draft){
    /* Work in a small scratch = committed context tail + the seed + drafts, so a
     * drafted token feeds the next lookup (autoregressive, as get_token()). We
     * only need the last NGRAM_MAX-1 real tokens plus what we draft. */
    int keep = COLI_NGRAM_MAX - 1;
    int base = n - keep; if (base < 0) base = 0;
    int win[COLI_NGRAM_MAX + 64];
    int wlen = 0;
    for (int i = base; i < n; ++i) win[wlen++] = toks[i];   /* context tail */
    win[wlen++] = draft[0];                                  /* the seed */
    int nd = 1;
    while (nd <= max_draft) {
        int tok = draft_one(c, win, wlen);
        if (tok < 0) break;
        draft[nd++] = tok;
        if (wlen < (int)(sizeof(win)/sizeof(win[0]))) win[wlen++] = tok;
        else { for (int i=0;i<wlen-1;i++) win[i]=win[i+1]; win[wlen-1]=tok; }
    }
    return nd;
}
