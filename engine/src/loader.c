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
    if (!gguf_meta_open(path,&g->mt,e,sizeof e)) {
        if (err&&errcap) snprintf(err,errcap,"meta: %s",e); free(g); return NULL; }
    g->fd = coli_open_ro(path);
    if (g->fd < 0) { if(err&&errcap) snprintf(err,errcap,"open failed"); free(g); return NULL; }
    g->fsz = (long long)coli_fsize(g->fd);
    if (g->fsz <= 0) { if(err&&errcap) snprintf(err,errcap,"cannot size file"); coli_close(g->fd); free(g); return NULL; }
    g->open = 1;
    return g;
}
void coli_gguf_close(coli_gguf *g){ if(!g) return; if(g->open){ gguf_meta_close(&g->mt); coli_close(g->fd); } free(g); }

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
    if ((long long)t->data_off + nb > g->fsz) return 0;
    void *raw = malloc((size_t)nb); if(!raw) return 0;
    if (coli_pread(g->fd,raw,(size_t)nb,(int64_t)t->data_off)!=(int64_t)nb){ free(raw); return 0; }
    float *dst = malloc((size_t)ne*sizeof(float));
    if(!dst){ free(raw); return 0; }
    switch(t->ttype){
        case 0:  gguf_dequant_f32 (raw,dst,ne);   break;
        case 1:  gguf_dequant_f16 (raw,dst,ne);   break;
        case 11: gguf_dequant_q3_K(raw,dst,nblk); break;
        case 12: gguf_dequant_q4_K(raw,dst,nblk); break;
        case 13: gguf_dequant_q5_K(raw,dst,nblk); break;
        case 14: gguf_dequant_q6_K(raw,dst,nblk); break;
        case 30: gguf_dequant_bf16(raw,dst,ne);   break;
        default: free(raw); free(dst); return 0;
    }
    free(raw); *out=dst; return ne;
}
void coli_gguf_free_f32(float *p){ free(p); }

void *coli_tok_load(const char *path,int*bos,int*eos,int*add_bos){
    Tok *T=calloc(1,sizeof(Tok));
    tok_load_gguf(T,path,bos,eos,add_bos);
    return T; }
int coli_tok_encode(void*t,const char*s,int*o,int m){ return tok_encode((Tok*)t,s,(int)strlen(s),o,m); }
int coli_tok_decode(void*t,const int*i,int n,char*o,int m){ return tok_decode((Tok*)t,i,n,o,m); }
void coli_tok_free_(void*t){ if(t){ tok_free((Tok*)t); free(t);} }
