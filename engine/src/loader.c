/* loader.c — C shim over ../c. See loader.h for why this file is C. */
#define _GNU_SOURCE
#include "platform.h"
#include "loader.h"
#include "gguf_reader.h"
#include "gguf_meta.h"
#include "ggml_dequant.h"
#include "tok.h"
#include "tok_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct coli_gguf { GgufIndex ix; GgufMeta mt; int fd; long long fsz; int open; };

coli_gguf *coli_gguf_open(const char *path, char *err, size_t errcap) {
    coli_gguf *g = calloc(1, sizeof *g);
    char e[256];
    if (!gguf_index_open(path,&g->ix,e,sizeof e)) {
        if (err&&errcap) snprintf(err,errcap,"gguf: %s",e); free(g); return NULL; }
    /* Model-level metadata (architecture, hyperparameters, tokenizer) lives
     * only in shard 0 of a split GGUF set -- `path` may name ANY shard (the
     * caller can point at shard 3 of 5 and still get a working load), so
     * gguf_meta_open() must read from ix.shard[0].path, not the literal
     * `path` argument. For a single-file (non-split) GGUF, ix.shard[0].path
     * IS `path` -- gguf_index_open() just parsed it -- so this is a no-op
     * change for the case that predates multi-shard support. */
    if (!gguf_meta_open(g->ix.shard[0].path,&g->mt,e,sizeof e)) {
        if (err&&errcap) snprintf(err,errcap,"meta: %s",e); gguf_index_free(&g->ix); free(g); return NULL; }
    /* g->fd/g->fsz stay exactly what they always were: the size of the LITERAL
     * file the caller passed as `path` (whichever shard that is), independent
     * of the multi-shard index above. This is deliberate, not an oversight --
     * coli_gguf_filesize() feeds model.cpp's w4snap staleness check, which
     * separately stat()s that same literal `path`; the two figures must keep
     * meaning the same file, or a split model would spuriously "go stale" the
     * moment coli_gguf_filesize() started reporting some other total. Shard
     * 2..N identity is covered instead inside coli_gguf_meta_hash() below. */
    g->fd = coli_open_ro(path);
    if (g->fd < 0) { if(err&&errcap) snprintf(err,errcap,"open failed"); gguf_meta_close(&g->mt); gguf_index_free(&g->ix); free(g); return NULL; }
    g->fsz = (long long)coli_fsize(g->fd);
    if (g->fsz <= 0) { if(err&&errcap) snprintf(err,errcap,"cannot size file"); coli_close(g->fd); gguf_meta_close(&g->mt); gguf_index_free(&g->ix); free(g); return NULL; }
    g->open = 1;
    return g;
}
/* gguf_index_free() now also closes every per-shard fd opened by
 * gguf_index_open() (see gguf_reader.h) -- necessary now that a split load
 * holds N fds, not the previous single one, and load/unload cycles are a real
 * production path (coli-server -sleep-idle-seconds tears the engine down and
 * reopens it repeatedly). Freeing idx->t here is also new but harmless: it was
 * never freed before (a one-shot small leak, invisible for one load per
 * process), and is exactly the array gguf_index_free() already frees for the
 * failure path inside gguf_index_open() itself. */
void coli_gguf_close(coli_gguf *g){ if(!g) return; if(g->open){ gguf_meta_close(&g->mt); coli_close(g->fd); gguf_index_free(&g->ix); } free(g); }

int coli_gguf_str(coli_gguf *g,const char*k,char*o,size_t n){ return gguf_meta_str(&g->mt,k,o,n); }
int coli_gguf_i64(coli_gguf *g,const char*k,long long*o){ return gguf_meta_i64(&g->mt,k,o); }
int coli_gguf_f32(coli_gguf *g,const char*k,float*o){ return gguf_meta_f32(&g->mt,k,o); }

static const GgufTensorInfo *ft(coli_gguf *g,const char*nm){
    for (size_t i=0;i<g->ix.n;i++) if(!strcmp(g->ix.t[i].name,nm)) return &g->ix.t[i];
    return NULL; }

int coli_gguf_has(coli_gguf *g,const char*nm){ return ft(g,nm)!=NULL; }
int64_t coli_gguf_shape(coli_gguf *g,const char*nm,int d){
    const GgufTensorInfo *t=ft(g,nm);
    if(!t||d<0||d>=t->rank) return -1;
    return (int64_t)t->shape[d]; }

int64_t coli_gguf_load_f32(coli_gguf *g,const char*nm,float**out){
    const GgufTensorInfo *t=ft(g,nm); if(!t) return 0;
    const GgmlType *gt = ggml_type(t->ttype);
    if(!gt||!gt->blck) return 0;
    int64_t ne=1; for(int d=0;d<t->rank;d++) ne*=(int64_t)t->shape[d];
    int64_t nblk = (gt->blck==1)?ne:ne/gt->blck;
    long long nb = (gt->blck==1)? ne*(long long)gt->bytes : nblk*(long long)gt->bytes;
    /* t->data_off is absolute WITHIN t->shard's own file, not within g->fd
     * (which is only ever the literal file the caller opened -- see the
     * comment in coli_gguf_open()). Every tensor, single-file or split, is
     * read from its own shard's fd and bounds-checked against that shard's
     * own size; for a single-file GGUF t->shard is always 0 and
     * g->ix.shard[0] names the same file g->fd does, so this is the same read
     * it always was. */
    if (t->shard < 0 || (size_t)t->shard >= g->ix.nshard) return 0;
    const GgufShard *sh = &g->ix.shard[t->shard];
    if ((long long)t->data_off + nb > sh->size) return 0;
    void *raw = malloc((size_t)nb); if(!raw) return 0;
    if (coli_pread(sh->fd,raw,(size_t)nb,(int64_t)t->data_off)!=(int64_t)nb){ free(raw); return 0; }
    float *dst = malloc((size_t)ne*sizeof(float));
    if(!dst){ free(raw); return 0; }
    switch(t->ttype){
        case 0:  gguf_dequant_f32  (raw,dst,ne);   break;
        case 1:  gguf_dequant_f16  (raw,dst,ne);   break;
        case 8:  gguf_dequant_q8_0 (raw,dst,nblk); break;
        case 11: gguf_dequant_q3_K (raw,dst,nblk); break;
        case 12: gguf_dequant_q4_K (raw,dst,nblk); break;
        case 13: gguf_dequant_q5_K (raw,dst,nblk); break;
        case 14: gguf_dequant_q6_K (raw,dst,nblk); break;
        case 30: gguf_dequant_bf16 (raw,dst,ne);   break;
        case 39: gguf_dequant_mxfp4(raw,dst,nblk); break;
        default: free(raw); free(dst); return 0;
    }
    free(raw); *out=dst; return ne;
}
void coli_gguf_free_f32(float *p){ free(p); }

int64_t coli_gguf_filesize(coli_gguf *g){ return g->fsz; }

/* See loader.h: hashes [0, shard 0's first tensor data_off) -- i.e. shard 0's
 * own metadata + tensor-directory region, since model-level metadata lives
 * only in shard 0 (see gguf_reader.h's MULTI-SHARD section). That region never
 * includes a single byte contributed by shards 1..N-1, so on its own it would
 * be blind to a stale/truncated/swapped later shard -- exactly the case a
 * w4snap identity check exists to catch. To cover that, every shard's own
 * (size, mtime) is folded into the same hash below: for the single-shard case
 * (nshard==1) that loop folds in shard 0's own (size, mtime), which is a NEW
 * input to the hash but a safe one -- it can only make a snapshot MORE likely
 * to be (rightly) treated as stale, never less, so an existing correct
 * snapshot is never wrongly accepted because of this change. It does mean a
 * .w4snap written before this change reads as stale once and is silently
 * regenerated (w4snap_try_open's whole design already treats any mismatch
 * this way -- see model.cpp) -- not a correctness break, just one paid-once
 * rebuild. */
int coli_gguf_meta_hash(coli_gguf *g, uint64_t *out_hash){
    if (!g->ix.nshard) return 0;
    const GgufShard *s0 = &g->ix.shard[0];
    long long off = s0->data_base;
    if (off <= 0 || off > s0->size) return 0;
    unsigned char *buf = (unsigned char*)malloc((size_t)off);
    if (!buf) return 0;
    if (coli_pread(s0->fd,buf,(size_t)off,0) != (int64_t)off) { free(buf); return 0; }
    uint64_t h = 1469598103934665603ull;
    for (long long i=0;i<off;i++){ h ^= buf[i]; h *= 1099511628211ull; }
    for (size_t i=0;i<g->ix.nshard;i++){
        uint64_t v[2]; v[0]=(uint64_t)g->ix.shard[i].size; v[1]=(uint64_t)g->ix.shard[i].mtime_ns;
        const unsigned char *bv = (const unsigned char*)v;
        for (size_t k=0;k<sizeof v;k++){ h ^= bv[k]; h *= 1099511628211ull; }
    }
    free(buf);
    *out_hash = h;
    return 1;
}

void *coli_tok_load(const char *path,int*bos,int*eos,int*add_bos){
    Tok *T=calloc(1,sizeof(Tok));
    tok_load_gguf(T,path,bos,eos,add_bos);
    return T; }
int coli_tok_encode(void*t,const char*s,int*o,int m){ return tok_encode((Tok*)t,s,(int)strlen(s),o,m); }
int coli_tok_decode(void*t,const int*i,int n,char*o,int m){ return tok_decode((Tok*)t,i,n,o,m); }
void coli_tok_free_(void*t){ if(t){ tok_free((Tok*)t); free(t);} }
