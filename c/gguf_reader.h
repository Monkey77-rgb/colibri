/* gguf_reader.h — GGUF header/metadata/tensor-directory parsing (header-only, all static).
 *
 * WHY THIS EXISTS
 * ----------------
 * Stage 1 of the GGUF weight loader (see project handoff). The seam is OFFLINE
 * TRANSCODE: this file reads GGUF containers and builds a retaining tensor index
 * (name, type, shape, absolute file offset). It does not touch the runtime path --
 * nothing in st.h / colibri.c / quant.h / tensor.h changes because of this file.
 *
 * PROVENANCE
 * The scalar tables, bounds-checked KV/tensor-directory walk, and ggml type table
 * below are moved verbatim (behaviour-preserving) from modelprobe.c's private
 * `static` copies, where they were written and already hardened against malformed
 * GGUF input. modelprobe.c now includes this header instead of defining its own
 * copies (see the separate bisectable commit).
 *
 * WHAT IS NEW HERE (did not exist in modelprobe.c)
 * modelprobe.c's tensor-directory walk reads `toff` per tensor and folds sizes into
 * running totals -- it never retains per-tensor offsets or exposes them. This file
 * adds `GgufIndex` / `gguf_index_open()`, which retains one `GgufTensorInfo` per
 * tensor, including the tensor's ABSOLUTE file offset (`data_base + toff`).
 *
 * THE OFFSET RULE (verified against the upstream reference before relying on it)
 * GGUF's tensor data section begins at `GGML_PAD(tensor_info_end, alignment)`,
 * where `alignment` comes from the u32 KV key "general.alignment" (default 32 if
 * absent, and it MUST be a power of two -- llama.cpp itself refuses to load the
 * file otherwise). `GGML_PAD(x,n) = ((x)+(n)-1) & ~((n)-1)`.
 *   Verified against: $LLAMA_CPP/ggml/src/gguf.cpp
 *     - alignment key + power-of-two check:            lines 556-560
 *     - data section start = GGML_PAD(ftell, alignment): line 699 (`gguf_fseek(...,
 *       GGML_PAD(gguf_ftell(file), ctx->alignment), SEEK_SET)`), `ctx->offset` is
 *       then set to that seeked position (line ~702).
 *     - GGML_PAD macro:            ggml/include/ggml.h:267
 *     - GGUF_DEFAULT_ALIGNMENT=32: ggml/include/gguf.h:46
 * Upstream additionally requires each tensor's on-disk `offset` field to equal a
 * running packed-size counter starting at 0 (gguf.cpp ~line 705-712) -- i.e. tensor
 * data has no gaps and the file's own per-tensor `toff` values are already relative
 * to the data section start in that packed order. We do not assume that invariant:
 * `data_off` is computed as `data_base + toff` using the file's own `toff`, which is
 * correct whether or not upstream's packing invariant holds, and is simpler to prove.
 *
 * UNTRUSTED INPUT
 * A GGUF file is untrusted input from the internet. Every offset/length here is
 * bounds-checked against the actual file size before use, exactly as in
 * modelprobe.c's probe_gguf(): a malformed file must return an error, never
 * over-read. gguf_index_open() does not read tensor DATA, only the header/KV/
 * tensor-info sections plus the file size, so a truncated data section still opens
 * successfully (data reads happen later, out of scope for this stage) -- but a
 * tensor whose data would run past EOF is still something a caller can detect by
 * comparing data_off + tensor byte size against the file size before reading.
 *
 * CONVENTION
 * Header-only, all `static`, no Model/QT/shards dependency -- matches st.h and
 * quant.h's existing header-only pattern in this tree, so this header can be
 * included from multiple translation units (modelprobe.c, tests, later stages)
 * without linker collisions.
 */
#ifndef COLI_GGUF_READER_H
#define COLI_GGUF_READER_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* pread() -- same guard pattern as st.h */
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define GGUF_MAGIC   0x46554747u          /* "GGUF" little-endian */
#define MAX_KV       (1u << 20)           /* sanity caps on untrusted counts */
#define MAX_STRLEN   (1u << 20)
#define MAX_ARRLEN   (1u << 24)

/* GGUF metadata value types, per the upstream spec. */
enum { G_U8, G_I8, G_U16, G_I16, G_U32, G_I32, G_F32, G_BOOL,
       G_STR, G_ARR, G_U64, G_I64, G_F64 };

static int gguf_read_at(int fd, void *buf, size_t n, long long off) {
    ssize_t r = pread(fd, buf, n, (off_t)off);
    return r == (ssize_t)n;
}

static int gguf_scalar_size(uint32_t t, size_t *sz) {
    switch (t) {
        case G_U8: case G_I8: case G_BOOL: *sz = 1; return 1;
        case G_U16: case G_I16:            *sz = 2; return 1;
        case G_U32: case G_I32: case G_F32:*sz = 4; return 1;
        case G_U64: case G_I64: case G_F64:*sz = 8; return 1;
        default: return 0;                       /* string/array handled separately */
    }
}

/* Read a length-prefixed GGUF string. Advances *off. Bounds-checked against fsz. */
/* Strict variant: REFUSES a string that does not fit, instead of truncating it.
 *
 * gguf_str() below truncates to the caller's buffer and still reports success.
 * That is harmless for modelprobe, which only ever substring-matches names
 * (strstr(tname, "_exps")), and its behaviour is deliberately left unchanged so
 * modelprobe's output stays byte-identical.
 *
 * It is NOT harmless for a weight loader, which resolves tensors by EXACT name.
 * Two distinct tensors whose names differ only after byte `outn-1` truncate to
 * the SAME stored name; a by-name lookup then returns whichever record it meets
 * first and silently loads the wrong tensor, with the wrong shape and offset.
 * No crash, no sanitizer report, wrong numbers — and bit-exactness tests cannot
 * catch it, because both tensors decode perfectly; just not the ones requested.
 * Found by a 143-input fuzz corpus (Hardware session, 2026-08-13); not reachable
 * from any legitimate file, since real GGUF names are ~40 chars, but this parser
 * takes untrusted input and "real files don't do that" is exactly the assumption
 * the rest of it refuses to make.
 *
 * Using this for the tensor name makes the invariant "a stored name is the
 * complete name", which every by-name lookup depends on. */
static int gguf_str_exact(int fd, long long *off, long long fsz, char *out, size_t outn) {
    uint64_t len;
    if (*off + 8 > fsz || !gguf_read_at(fd, &len, 8, *off)) return 0;
    if (len > MAX_STRLEN || (long long)len > fsz - (*off + 8)) return 0;
    if (len >= outn) return 0;              /* would truncate -> refuse */
    *off += 8;
    if (len && !gguf_read_at(fd, out, (size_t)len, *off)) return 0;
    out[len] = 0;
    *off += (long long)len;
    return 1;
}

static int gguf_str(int fd, long long *off, long long fsz, char *out, size_t outn) {
    uint64_t len;
    if (*off + 8 > fsz || !gguf_read_at(fd, &len, 8, *off)) return 0;
    *off += 8;
    if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
    size_t take = (len < outn - 1) ? (size_t)len : outn - 1;
    if (take && !gguf_read_at(fd, out, take, *off)) return 0;
    out[take] = 0;
    *off += (long long)len;
    return 1;
}

/* Skip a value of type t. Returns 0 on malformed input. */
static int gguf_skip(int fd, long long *off, long long fsz, uint32_t t) {
    size_t sz;
    if (gguf_scalar_size(t, &sz)) {
        if (*off + (long long)sz > fsz) return 0;
        *off += (long long)sz;
        return 1;
    }
    if (t == G_STR) {
        uint64_t len;
        if (*off + 8 > fsz || !gguf_read_at(fd, &len, 8, *off)) return 0;
        *off += 8;
        if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
        *off += (long long)len;
        return 1;
    }
    if (t == G_ARR) {
        uint32_t et; uint64_t n;
        if (*off + 12 > fsz) return 0;
        if (!gguf_read_at(fd, &et, 4, *off) || !gguf_read_at(fd, &n, 8, *off + 4)) return 0;
        *off += 12;
        if (n > MAX_ARRLEN) return 0;
        if (gguf_scalar_size(et, &sz)) {
            long long need = (long long)n * (long long)sz;
            if (need < 0 || need > fsz - *off) return 0;
            *off += need;
            return 1;
        }
        if (et == G_STR) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len;
                if (*off + 8 > fsz || !gguf_read_at(fd, &len, 8, *off)) return 0;
                *off += 8;
                if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
                *off += (long long)len;
            }
            return 1;
        }
        return 0;                              /* nested arrays: not in practice */
    }
    return 0;
}

/* ---- ggml tensor types -------------------------------------------------------
 *
 * A ggml type is a block format: `blck` elements packed into `bytes` bytes. Values
 * below are the upstream block sizes; anything not listed returns NULL so the
 * caller reports "unknown type" rather than silently computing a wrong size. */
typedef struct { int blck; int bytes; const char *name; } GgmlType;

static const GgmlType *ggml_type(uint32_t t) {
    static const GgmlType T[] = {
        /*0*/{1,4,"F32"},   {1,2,"F16"},    {32,18,"Q4_0"},  {32,20,"Q4_1"},
        /*4*/{0,0,NULL},    {0,0,NULL},     {32,22,"Q5_0"},  {32,24,"Q5_1"},
        /*8*/{32,34,"Q8_0"},{32,36,"Q8_1"},
       /*10*/{256,84,"Q2_K"},{256,110,"Q3_K"},{256,144,"Q4_K"},{256,176,"Q5_K"},
       /*14*/{256,210,"Q6_K"},{256,292,"Q8_K"},
       /*16*/{256,66,"IQ2_XXS"},{256,74,"IQ2_XS"},{256,98,"IQ3_XXS"},{256,50,"IQ1_S"},
       /*20*/{32,18,"IQ4_NL"},{256,110,"IQ3_S"},{256,82,"IQ2_S"},{256,136,"IQ4_XS"},
       /*24*/{1,1,"I8"},   {1,2,"I16"},    {1,4,"I32"},     {1,8,"I64"},
       /*28*/{1,8,"F64"},  {256,56,"IQ1_M"},{1,2,"BF16"},
       /*31*/{0,0,NULL},   {0,0,NULL},     {0,0,NULL},
       /*34*/{256,54,"TQ1_0"},{256,66,"TQ2_0"},
    };
    if (t >= sizeof T / sizeof T[0]) return NULL;
    return T[t].blck ? &T[t] : NULL;
}

static long long gguf_read_int(int fd, long long off, uint32_t t) {
    uint64_t u = 0; int64_t s = 0;
    switch (t) {
        case G_U8:  { uint8_t v;  if (gguf_read_at(fd,&v,1,off)) u = v; break; }
        case G_I8:  { int8_t  v;  if (gguf_read_at(fd,&v,1,off)) s = v; return s; }
        case G_U16: { uint16_t v; if (gguf_read_at(fd,&v,2,off)) u = v; break; }
        case G_I16: { int16_t v;  if (gguf_read_at(fd,&v,2,off)) s = v; return s; }
        case G_U32: { uint32_t v; if (gguf_read_at(fd,&v,4,off)) u = v; break; }
        case G_I32: { int32_t v;  if (gguf_read_at(fd,&v,4,off)) s = v; return s; }
        case G_U64: { uint64_t v; if (gguf_read_at(fd,&v,8,off)) u = v; break; }
        case G_I64: { int64_t v;  if (gguf_read_at(fd,&v,8,off)) s = v; return s; }
        case G_BOOL:{ uint8_t v;  if (gguf_read_at(fd,&v,1,off)) u = v; break; }
        default: return -1;
    }
    return (long long)u;
}

/* ------------------------------------------------------------ retaining index */

#define GGUF_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

typedef struct {
    char     name[512];
    uint32_t ttype;
    int      rank;
    uint64_t shape[8];
    uint64_t data_off;     /* ABSOLUTE file offset: data_base + toff */
} GgufTensorInfo;

typedef struct {
    GgufTensorInfo *t;
    size_t          n, cap;
    uint32_t        alignment;
    long long       data_base;
} GgufIndex;

static void gguf_index_free(GgufIndex *idx) {
    if (!idx) return;
    free(idx->t);
    idx->t = NULL;
    idx->n = idx->cap = 0;
}

static int gguf_index_push(GgufIndex *idx, const GgufTensorInfo *e) {
    if (idx->n == idx->cap) {
        size_t ncap = idx->cap ? idx->cap * 2 : 64;
        /* Explicit cast: legal to omit in C, but this tree also builds .cu and
         * .mm translation units where an implicit void* conversion is an error. */
        GgufTensorInfo *nt = (GgufTensorInfo *)realloc(idx->t, ncap * sizeof *nt);
        if (!nt) return 0;
        idx->t = nt;
        idx->cap = ncap;
    }
    idx->t[idx->n++] = *e;
    return 1;
}

/* Open a GGUF file and build a retaining tensor index: name, type, shape, and the
 * ABSOLUTE file offset of every tensor's data. Read-only, does not read tensor
 * data. Returns 1 on success, 0 on any malformed/truncated input (err filled).
 * `idx` is zeroed by the caller's memset convention is NOT assumed here -- this
 * function initializes every field itself. On failure, any partially-built index
 * (tensors already parsed before the error was hit) is freed internally before
 * returning: idx is guaranteed empty (n=0, t=NULL) on failure, so the caller does
 * NOT need to call gguf_index_free() after a failed open, though doing so is also
 * safe (it is a no-op on an already-empty index). */
static int gguf_index_open(const char *path, GgufIndex *idx, char *err, size_t errcap) {
    memset(idx, 0, sizeof *idx);
    idx->alignment = 32; /* GGUF_DEFAULT_ALIGNMENT */

#define GIERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)

    int fd = open(path, O_RDONLY);
    if (fd < 0) { GIERR("cannot open '%s'", path); return 0; }

    struct stat stbuf;
    if (fstat(fd, &stbuf) != 0) { GIERR("fstat failed"); close(fd); return 0; }
    long long fsz = (long long)stbuf.st_size;

    uint32_t magic, ver; uint64_t ntensor, nkv;
    if (fsz < 24 || !gguf_read_at(fd,&magic,4,0) || magic != GGUF_MAGIC) {
        GIERR("not a GGUF file (bad magic or too small)"); close(fd); return 0;
    }
    if (!gguf_read_at(fd,&ver,4,4) || !gguf_read_at(fd,&ntensor,8,8) || !gguf_read_at(fd,&nkv,8,16)) {
        GIERR("truncated GGUF header"); close(fd); return 0;
    }
    if (nkv > MAX_KV) { GIERR("implausible kv count %llu",(unsigned long long)nkv); close(fd); return 0; }
    if (ntensor > (1u << 22)) { GIERR("implausible tensor count %llu",(unsigned long long)ntensor); close(fd); return 0; }

    long long off = 24;
    char key[256];
    uint32_t alignment = 32;

    for (uint64_t i = 0; i < nkv; i++) {
        if (!gguf_str(fd,&off,fsz,key,sizeof key)) { GIERR("malformed kv key at %llu",(unsigned long long)i); close(fd); return 0; }
        uint32_t t;
        if (off + 4 > fsz || !gguf_read_at(fd,&t,4,off)) { GIERR("truncated kv type at %llu",(unsigned long long)i); close(fd); return 0; }
        off += 4;
        long long vpos = off;

        if (!strcmp(key, "general.alignment") && t == G_U32) {
            long long v = gguf_read_int(fd, vpos, t);
            if (v <= 0 || (v & (v - 1)) != 0) {
                GIERR("general.alignment %lld is not a positive power of 2", v);
                close(fd); return 0;
            }
            alignment = (uint32_t)v;
        }

        if (!gguf_skip(fd,&off,fsz,t)) { GIERR("malformed kv value at %llu",(unsigned long long)i); close(fd); return 0; }
    }

    /* Tensor directory: name (str) | n_dims u32 | dims[n_dims] u64 | type u32 | offset u64 */
    for (uint64_t i = 0; i < ntensor; i++) {
        GgufTensorInfo info; memset(&info, 0, sizeof info);

        /* gguf_str_exact, NOT gguf_str: a truncated name would collide with any
         * other name sharing its first sizeof(info.name)-1 bytes, and this index
         * is looked up by exact name. See the comment on gguf_str_exact. */
        if (!gguf_str_exact(fd,&off,fsz,info.name,sizeof info.name)) {
            GIERR("malformed or over-long tensor name at %llu (max %zu bytes)",
                  (unsigned long long)i, sizeof info.name - 1);
            gguf_index_free(idx); close(fd); return 0;
        }
        uint32_t ndim;
        if (off + 4 > fsz || !gguf_read_at(fd,&ndim,4,off)) { GIERR("truncated tensor rank at %llu",(unsigned long long)i); gguf_index_free(idx); close(fd); return 0; }
        off += 4;
        if (ndim > 8) { GIERR("implausible tensor rank %u at %llu",ndim,(unsigned long long)i); gguf_index_free(idx); close(fd); return 0; }
        info.rank = (int)ndim;

        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (off + 8 > fsz || !gguf_read_at(fd,&dim,8,off)) { GIERR("truncated tensor dims at %llu",(unsigned long long)i); gguf_index_free(idx); close(fd); return 0; }
            off += 8;
            if (dim == 0 || dim > (1ULL<<40)) { GIERR("implausible dim at tensor %llu",(unsigned long long)i); gguf_index_free(idx); close(fd); return 0; }
            info.shape[d] = dim;
        }

        uint32_t ttype; uint64_t toff;
        if (off + 12 > fsz || !gguf_read_at(fd,&ttype,4,off) || !gguf_read_at(fd,&toff,8,off+4)) {
            GIERR("truncated tensor type/offset at %llu",(unsigned long long)i); gguf_index_free(idx); close(fd); return 0;
        }
        off += 12;
        info.ttype = ttype;

        /* data_off filled in below, once data_base is known (needs off to have
         * finished walking the whole tensor directory first). Stash toff in
         * data_off for now; it is converted to absolute below. */
        info.data_off = toff;

        if (!gguf_index_push(idx, &info)) { GIERR("out of memory"); gguf_index_free(idx); close(fd); return 0; }
    }

    /* Data section starts at GGML_PAD(tensor_info_end, alignment). `off` is exactly
     * the byte position right after the last tensor-info entry at this point. */
    long long data_base = GGUF_PAD(off, (long long)alignment);
    /* data_base > fsz is fine for a file with zero tensors (nothing references it);
     * for a file WITH tensors it means the file is truncated before any tensor data,
     * which the per-tensor offset check below catches. */

    for (size_t i = 0; i < idx->n; i++) {
        uint64_t toff = idx->t[i].data_off;
        /* absolute offset must not overflow and must land inside the file */
        if ((long long)toff < 0 || toff > (uint64_t)(fsz)) {
            GIERR("tensor '%s' has an implausible relative offset %llu", idx->t[i].name, (unsigned long long)toff);
            gguf_index_free(idx); close(fd); return 0;
        }
        long long abs_off = data_base + (long long)toff;
        if (abs_off < data_base || abs_off > fsz) {
            GIERR("tensor '%s' data offset %lld is outside the file (size %lld)", idx->t[i].name, abs_off, fsz);
            gguf_index_free(idx); close(fd); return 0;
        }
        idx->t[i].data_off = (uint64_t)abs_off;
    }

    idx->alignment = alignment;
    idx->data_base = data_base;
    close(fd);
    return 1;

#undef GIERR
}

#endif /* COLI_GGUF_READER_H */
