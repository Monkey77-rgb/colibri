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
 * MULTI-SHARD (SPLIT) GGUF
 * llama.cpp's `gguf-split` tool writes a large model as N files named
 * "<name>-00001-of-000NN.gguf" .. "<name>-000NN-of-000NN.gguf": every shard is
 * itself a complete, independently-parseable GGUF container (own header, own KV
 * section, own tensor directory covering only the tensors it carries), and the
 * FIRST shard additionally carries all of the model-level metadata (arch,
 * hyperparameters, tokenizer). Shards also carry three KV keys that exist only
 * for this purpose: `split.no` (u16, this shard's 0-based index), `split.count`
 * (u16, total shards) and `split.tensors.count` (i32, tensor count in THIS
 * shard) -- gguf_index_open() cross-checks these against what the filename
 * itself says, so a shard swapped for one from a different split, or renamed
 * out of sequence, is rejected rather than silently mis-assembled.
 *
 * gguf_index_open() detects a split filename (see gguf_split_name_parse()
 * below), locates all N sibling shards by reconstructing their names from the
 * shared prefix, and opens every one of them, folding all of their tensors into
 * ONE GgufIndex exactly as if they were one file. Each GgufTensorInfo now
 * additionally records which shard it lives in (`shard`, an index into
 * GgufIndex.shard[]); `data_off` stays what it always was -- the tensor's
 * absolute byte offset, but now absolute WITHIN THAT SHARD's own file, not a
 * global offset across the set (shards are independent files; there is no
 * single global address space to be absolute within). A single-file GGUF is
 * simply the nshard==1 case of the same code path, with idx->shard[0] the file
 * itself -- it takes the identical route through every check below and produces
 * byte-identical `GgufTensorInfo` records to before this feature existed.
 *
 * Per-shard file descriptors are opened once and kept open in idx->shard[i].fd
 * for the life of the index (closed together by gguf_index_free()) -- exactly
 * mirroring the single-file convention elsewhere in this tree (e.g. loader.c
 * keeping one fd open for the process lifetime of a loaded model): flat memory,
 * no re-opening per tensor read, no reading of any tensor payload at open time.
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
#include <ctype.h>   /* isdigit(), for split-filename parsing below */
#include <errno.h>   /* errno/strerror(), for naming a missing split shard */
#include <time.h>    /* clock_gettime(), for COLI_LOAD_PROF (io11) */

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
       /*36*/{0,0,NULL},   {0,0,NULL},     {0,0,NULL},
       /* 39: MXFP4 (gpt-oss). block_mxfp4 = 1-byte E8M0 exponent + 16 nibble
        * bytes / 32 values = 17 bytes; verified against ggml-common.h 2026-09-13
        * (see c/ggml_dequant.h PROVENANCE-2) and against the real
        * gpt-oss-120b-MXFP4.gguf tensor directory (ttype 39 on
        * blk.N.ffn_{gate,up,down}_exps.weight). */
       /*39*/{32,17,"MXFP4"},
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

/* ---- read-through buffer for the metadata/tensor-info walk -------------------
 *
 * io11 (Hardware session, 2026-09-17): strace of coli-gpu loading gpt-oss-120b
 * showed 5,797,198 read syscalls, one `pread` per scalar/string-length through
 * gguf_read_at() above -- the 201,088-entry vocab arrays, skipped whole via
 * gguf_skip()'s G_ARR/G_STR path (:204-213), are where the millions come from:
 * that loop still issues one 8-byte pread per element to learn each string's
 * length even though the string body itself is never read.
 *
 * This buffer is ONLY for the header/metadata/tensor-info region walked by
 * gguf_index_open_shard() below (KV section + tensor directory). Tensor DATA
 * reads (coli_gguf_load_*, slices, the expert store) stay on plain pread and
 * are untouched. One buffer per open shard fd, stack-allocated by the caller
 * and passed by pointer -- gguf_index_open_shard() opens shards strictly
 * sequentially (see MULTI-SHARD note above), so there is never more than one
 * live buffer per fd and no cross-shard aliasing risk.
 *
 * Identical parse, by construction: gguf_buf_read() returns exactly what
 * gguf_read_at(fd,buf,n,off) would have returned (1 iff `n` bytes were placed
 * at `buf`, 0 otherwise) for every (n, off) -- ranges that fit within the
 * cached window are served from it, everything else (including any n larger
 * than the buffer) falls through to the identical pread gguf_read_at already
 * does. No parsed value, struct, offset arithmetic, or string truncation rule
 * changes; gguf_str/gguf_str_exact/gguf_skip/gguf_read_int are each mirrored
 * below as an _buf variant whose body is the original with gguf_read_at(fd,...)
 * calls swapped for gguf_buf_read(fd,rb,...) -- nothing else differs. */
#define GGUF_RBUF_SZ (1u << 20)   /* 1 MiB, per the brief's "e.g. 1 MiB" */

typedef struct {
    long long     base;    /* file offset of data[0]; -1 == empty/invalid */
    size_t        len;     /* valid bytes in data[], starting at base */
    unsigned char data[GGUF_RBUF_SZ];
} GgufReadBuf;

static int gguf_buf_read(int fd, GgufReadBuf *rb, void *out, size_t n, long long off) {
    if (n > sizeof rb->data) return gguf_read_at(fd, out, n, off);   /* too big for the buffer: straight to pread, as today */
    if (rb->base < 0 || off < rb->base || (off - rb->base) + (long long)n > (long long)rb->len) {
        ssize_t r = pread(fd, rb->data, sizeof rb->data, (off_t)off);
        if (r < 0) { rb->base = -1; rb->len = 0; return 0; }
        rb->base = off;
        rb->len = (size_t)r;
    }
    if ((off - rb->base) + (long long)n > (long long)rb->len) return 0;   /* short read (EOF): same failure gguf_read_at would report */
    memcpy(out, rb->data + (off - rb->base), n);
    return 1;
}

/* _buf mirrors of gguf_str / gguf_str_exact / gguf_skip / gguf_read_int --
 * bodies copied verbatim from above with gguf_read_at(fd,...) -> gguf_buf_read(fd,rb,...). */
static int gguf_str_buf(int fd, GgufReadBuf *rb, long long *off, long long fsz, char *out, size_t outn) {
    uint64_t len;
    if (*off + 8 > fsz || !gguf_buf_read(fd, rb, &len, 8, *off)) return 0;
    *off += 8;
    if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
    size_t take = (len < outn - 1) ? (size_t)len : outn - 1;
    if (take && !gguf_buf_read(fd, rb, out, take, *off)) return 0;
    out[take] = 0;
    *off += (long long)len;
    return 1;
}

static int gguf_str_exact_buf(int fd, GgufReadBuf *rb, long long *off, long long fsz, char *out, size_t outn) {
    uint64_t len;
    if (*off + 8 > fsz || !gguf_buf_read(fd, rb, &len, 8, *off)) return 0;
    if (len > MAX_STRLEN || (long long)len > fsz - (*off + 8)) return 0;
    if (len >= outn) return 0;              /* would truncate -> refuse */
    *off += 8;
    if (len && !gguf_buf_read(fd, rb, out, (size_t)len, *off)) return 0;
    out[len] = 0;
    *off += (long long)len;
    return 1;
}

static int gguf_skip_buf(int fd, GgufReadBuf *rb, long long *off, long long fsz, uint32_t t) {
    size_t sz;
    if (gguf_scalar_size(t, &sz)) {
        if (*off + (long long)sz > fsz) return 0;
        *off += (long long)sz;
        return 1;
    }
    if (t == G_STR) {
        uint64_t len;
        if (*off + 8 > fsz || !gguf_buf_read(fd, rb, &len, 8, *off)) return 0;
        *off += 8;
        if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
        *off += (long long)len;
        return 1;
    }
    if (t == G_ARR) {
        uint32_t et; uint64_t n;
        if (*off + 12 > fsz) return 0;
        if (!gguf_buf_read(fd, rb, &et, 4, *off) || !gguf_buf_read(fd, rb, &n, 8, *off + 4)) return 0;
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
                if (*off + 8 > fsz || !gguf_buf_read(fd, rb, &len, 8, *off)) return 0;
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

static long long gguf_read_int_buf(int fd, GgufReadBuf *rb, long long off, uint32_t t) {
    uint64_t u = 0; int64_t s = 0;
    switch (t) {
        case G_U8:  { uint8_t v;  if (gguf_buf_read(fd,rb,&v,1,off)) u = v; break; }
        case G_I8:  { int8_t  v;  if (gguf_buf_read(fd,rb,&v,1,off)) s = v; return s; }
        case G_U16: { uint16_t v; if (gguf_buf_read(fd,rb,&v,2,off)) u = v; break; }
        case G_I16: { int16_t v;  if (gguf_buf_read(fd,rb,&v,2,off)) s = v; return s; }
        case G_U32: { uint32_t v; if (gguf_buf_read(fd,rb,&v,4,off)) u = v; break; }
        case G_I32: { int32_t v;  if (gguf_buf_read(fd,rb,&v,4,off)) s = v; return s; }
        case G_U64: { uint64_t v; if (gguf_buf_read(fd,rb,&v,8,off)) u = v; break; }
        case G_I64: { int64_t v;  if (gguf_buf_read(fd,rb,&v,8,off)) s = v; return s; }
        case G_BOOL:{ uint8_t v;  if (gguf_buf_read(fd,rb,&v,1,off)) u = v; break; }
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
    uint64_t data_off;     /* ABSOLUTE offset WITHIN ITS SHARD: data_base + toff.
                             * For a single-file GGUF "its shard" is the only
                             * file there is, so this is unchanged from before
                             * multi-shard support existed. */
    int      shard;        /* index into GgufIndex.shard[]; 0 for every tensor
                             * of a single-file (non-split) GGUF. */
} GgufTensorInfo;

/* One physical file backing a (possibly single-shard) GgufIndex. Kept open for
 * the life of the index -- see the MULTI-SHARD section of the file comment
 * above for why (flat memory, no re-opening per tensor read). */
typedef struct {
    char      path[4096];
    int       fd;          /* -1 if never opened (should not escape a failed open) */
    long long size;        /* st_size at open time */
    long long mtime_ns;    /* mtime at open time -- w4snap staleness (loader.c) */
    long long data_base;   /* GGML_PAD(tensor_info_end, alignment) for THIS shard:
                             * where its metadata+tensor-directory region ends and
                             * its tensor payload begins. shard[0]'s is also the
                             * region coli_gguf_meta_hash() (loader.c) hashes,
                             * since model-level metadata lives only in shard 0. */
} GgufShard;

typedef struct {
    GgufTensorInfo *t;
    size_t          n, cap;
    uint32_t        alignment;   /* shard[0]'s alignment (kept for callers/tests
                                   * written before multi-shard existed) */
    long long       data_base;   /* == shard[0].data_base, same reason */
    GgufShard      *shard;
    size_t          nshard;      /* 1 for a single-file (non-split) GGUF */
} GgufIndex;

static void gguf_index_free(GgufIndex *idx) {
    if (!idx) return;
    free(idx->t);
    idx->t = NULL;
    idx->n = idx->cap = 0;
    if (idx->shard) {
        for (size_t i = 0; i < idx->nshard; i++)
            if (idx->shard[i].fd >= 0) close(idx->shard[i].fd);
        free(idx->shard);
    }
    idx->shard = NULL;
    idx->nshard = 0;
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

/* Recognizes llama.cpp gguf-split's "<prefix>-NNNNN-of-MMMMM.gguf" naming
 * convention (NNNNN/MMMMM exactly 5 digits each, 1-based shard numbering). On
 * a match, fills `prefix` with everything before "-NNNNN-of-MMMMM.gguf" and
 * *count with MMMMM (the total shard count), and returns 1. Returns 0 for any
 * path that doesn't look like a split shard -- which is every ordinary GGUF,
 * and those must open exactly as they did before this feature existed.
 *
 * Deliberately does not need or return the shard's own NNNNN: gguf_index_open()
 * reconstructs all N sibling names itself from `prefix` and *count (0-based
 * loop, 1-based filenames), and cross-checks each opened shard's own split.no
 * KV against the slot it was opened FOR, in gguf_index_open_shard(). A whole
 * shard being open-able under the name the loop expected is itself already a
 * filename-level check; the KV cross-check catches the sneakier case of a
 * shard from a DIFFERENT split renamed to fit this one's sequence. */
static int gguf_split_name_parse(const char *path, char *prefix, size_t prefix_cap, int *count) {
    size_t len = strlen(path);
    static const char SUF[] = ".gguf";
    size_t suflen = sizeof(SUF) - 1;
    if (len <= suflen || strcmp(path + len - suflen, SUF) != 0) return 0;
    size_t body = len - suflen;                    /* length up to (not incl.) ".gguf" */
    static const size_t TAIL = 1 + 5 + 4 + 5;       /* "-NNNNN-of-MMMMM" = 15 chars */
    if (body < TAIL) return 0;
    const char *tail = path + (body - TAIL);
    if (tail[0] != '-') return 0;
    for (int i = 0; i < 5; i++) if (!isdigit((unsigned char)tail[1 + i])) return 0;
    if (strncmp(tail + 6, "-of-", 4) != 0) return 0;
    for (int i = 0; i < 5; i++) if (!isdigit((unsigned char)tail[10 + i])) return 0;
    char cnt[6]; memcpy(cnt, tail + 10, 5); cnt[5] = 0;
    int count_v = atoi(cnt);
    if (count_v < 1) return 0;
    size_t plen = body - TAIL;
    if (plen + 1 > prefix_cap) return 0;
    memcpy(prefix, path, plen);
    prefix[plen] = 0;
    *count = count_v;
    return 1;
}

/* Opens ONE shard file, parses its header/KV/tensor-directory, and appends its
 * tensors (tagged with shard index `shard_ix`) onto the shared `idx`. This is
 * the entire body of the pre-multi-shard gguf_index_open(): every
 * bounds-check below is unchanged from that version, just re-scoped to one
 * shard's own file size and re-targeted (via `start_n`) to only the tensor
 * entries THIS call pushed, since `idx` may already hold entries from earlier
 * shards. `*out` receives this shard's fd (left OPEN -- the caller now owns
 * closing it, via gguf_index_free()), size, mtime and data_base.
 *
 * `is_split`/`nshard_total` are only used to cross-check this shard's own
 * split.no/split.count KV (when present -- their absence is not itself an
 * error, since gguf_index_open() already established this is a split set from
 * the filename alone) against the slot the CALLER opened it for. A mismatch
 * means a shard from a different split, or one renamed out of sequence, and is
 * rejected with the shard's path and position named in `err` -- never mixed in
 * silently. */
static int gguf_index_open_shard(const char *path, int shard_ix, int is_split, int nshard_total,
                                  GgufIndex *idx, GgufShard *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    out->fd = -1;
    snprintf(out->path, sizeof out->path, "%s", path);

#define GIERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (is_split) GIERR("split shard %d/%d '%s': cannot open (%s)",
                             shard_ix + 1, nshard_total, path, strerror(errno));
        else          GIERR("cannot open '%s'", path);
        return 0;
    }

    struct stat stbuf;
    if (fstat(fd, &stbuf) != 0) { GIERR("fstat failed on '%s'", path); close(fd); return 0; }
    long long fsz = (long long)stbuf.st_size;
    out->size = fsz;
#if defined(__APPLE__)
    out->mtime_ns = (long long)stbuf.st_mtimespec.tv_sec * 1000000000ll + stbuf.st_mtimespec.tv_nsec;
#else
    out->mtime_ns = (long long)stbuf.st_mtim.tv_sec * 1000000000ll + stbuf.st_mtim.tv_nsec;
#endif

    /* io11: one read-through buffer for this shard's fd, covering the KV walk
     * and the tensor-info walk below. See the GgufReadBuf comment above
     * gguf_buf_read() for why this is correctness-neutral (identical parse). */
    GgufReadBuf rb; rb.base = -1; rb.len = 0;

    uint32_t magic, ver; uint64_t ntensor, nkv;
    if (fsz < 24 || !gguf_buf_read(fd,&rb,&magic,4,0) || magic != GGUF_MAGIC) {
        GIERR("'%s': not a GGUF file (bad magic or too small)", path); close(fd); return 0;
    }
    if (!gguf_buf_read(fd,&rb,&ver,4,4) || !gguf_buf_read(fd,&rb,&ntensor,8,8) || !gguf_buf_read(fd,&rb,&nkv,8,16)) {
        GIERR("'%s': truncated GGUF header", path); close(fd); return 0;
    }
    if (nkv > MAX_KV) { GIERR("'%s': implausible kv count %llu", path,(unsigned long long)nkv); close(fd); return 0; }
    if (ntensor > (1u << 22)) { GIERR("'%s': implausible tensor count %llu", path,(unsigned long long)ntensor); close(fd); return 0; }

    long long off = 24;
    char key[256];
    uint32_t alignment = 32;
    int have_split_no = 0, have_split_count = 0;
    long long split_no_v = -1, split_count_v = -1;

    for (uint64_t i = 0; i < nkv; i++) {
        if (!gguf_str_buf(fd,&rb,&off,fsz,key,sizeof key)) { GIERR("'%s': malformed kv key at %llu", path,(unsigned long long)i); close(fd); return 0; }
        uint32_t t;
        if (off + 4 > fsz || !gguf_buf_read(fd,&rb,&t,4,off)) { GIERR("'%s': truncated kv type at %llu", path,(unsigned long long)i); close(fd); return 0; }
        off += 4;
        long long vpos = off;

        if (!strcmp(key, "general.alignment") && t == G_U32) {
            long long v = gguf_read_int_buf(fd, &rb, vpos, t);
            if (v <= 0 || (v & (v - 1)) != 0) {
                GIERR("'%s': general.alignment %lld is not a positive power of 2", path, v);
                close(fd); return 0;
            }
            alignment = (uint32_t)v;
        } else if (!strcmp(key, "split.no")) {
            size_t sz; if (gguf_scalar_size(t, &sz)) { split_no_v = gguf_read_int_buf(fd, &rb, vpos, t); have_split_no = 1; }
        } else if (!strcmp(key, "split.count")) {
            size_t sz; if (gguf_scalar_size(t, &sz)) { split_count_v = gguf_read_int_buf(fd, &rb, vpos, t); have_split_count = 1; }
        }
        /* split.tensors.count is read only as a per-file sanity fact upstream
         * writers include; nothing here needs it (the tensor directory this
         * file actually carries is authoritative over any count claimed in a
         * KV), so it is left to gguf_skip() below like any other key. */

        if (!gguf_skip_buf(fd,&rb,&off,fsz,t)) { GIERR("'%s': malformed kv value at %llu", path,(unsigned long long)i); close(fd); return 0; }
    }

    if (is_split) {
        if (have_split_count && split_count_v != nshard_total) {
            GIERR("'%s': split.count=%lld does not match the %d-shard set implied by this file's "
                  "own filename -- looks like a shard from a different split", path, split_count_v, nshard_total);
            close(fd); return 0;
        }
        if (have_split_no && split_no_v != shard_ix) {
            GIERR("'%s': split.no=%lld but this file was opened as shard %d (0-based) of the set "
                  "-- renamed or mismatched shard", path, split_no_v, shard_ix);
            close(fd); return 0;
        }
    }

    size_t start_n = idx->n;

    /* Tensor directory: name (str) | n_dims u32 | dims[n_dims] u64 | type u32 | offset u64 */
    for (uint64_t i = 0; i < ntensor; i++) {
        GgufTensorInfo info; memset(&info, 0, sizeof info);
        info.shard = shard_ix;

        /* gguf_str_exact, NOT gguf_str: a truncated name would collide with any
         * other name sharing its first sizeof(info.name)-1 bytes, and this index
         * is looked up by exact name. See the comment on gguf_str_exact. */
        if (!gguf_str_exact_buf(fd,&rb,&off,fsz,info.name,sizeof info.name)) {
            GIERR("'%s': malformed or over-long tensor name at %llu (max %zu bytes)",
                  path, (unsigned long long)i, sizeof info.name - 1);
            close(fd); return 0;
        }
        uint32_t ndim;
        if (off + 4 > fsz || !gguf_buf_read(fd,&rb,&ndim,4,off)) { GIERR("'%s': truncated tensor rank at %llu", path,(unsigned long long)i); close(fd); return 0; }
        off += 4;
        if (ndim > 8) { GIERR("'%s': implausible tensor rank %u at %llu", path,ndim,(unsigned long long)i); close(fd); return 0; }
        info.rank = (int)ndim;

        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (off + 8 > fsz || !gguf_buf_read(fd,&rb,&dim,8,off)) { GIERR("'%s': truncated tensor dims at %llu", path,(unsigned long long)i); close(fd); return 0; }
            off += 8;
            if (dim == 0 || dim > (1ULL<<40)) { GIERR("'%s': implausible dim at tensor %llu", path,(unsigned long long)i); close(fd); return 0; }
            info.shape[d] = dim;
        }

        uint32_t ttype; uint64_t toff;
        if (off + 12 > fsz || !gguf_buf_read(fd,&rb,&ttype,4,off) || !gguf_buf_read(fd,&rb,&toff,8,off+4)) {
            GIERR("'%s': truncated tensor type/offset at %llu", path,(unsigned long long)i); close(fd); return 0;
        }
        off += 12;
        info.ttype = ttype;

        /* data_off filled in below, once data_base is known (needs off to have
         * finished walking the whole tensor directory first). Stash toff in
         * data_off for now; it is converted to absolute-within-this-shard below. */
        info.data_off = toff;

        if (!gguf_index_push(idx, &info)) { GIERR("out of memory"); close(fd); return 0; }
    }

    /* Data section starts at GGML_PAD(tensor_info_end, alignment). `off` is exactly
     * the byte position right after the last tensor-info entry at this point. */
    long long data_base = GGUF_PAD(off, (long long)alignment);
    /* data_base > fsz is fine for a file with zero tensors (nothing references it);
     * for a file WITH tensors it means the file is truncated before any tensor data,
     * which the per-tensor offset check below catches. */

    for (size_t i = start_n; i < idx->n; i++) {
        uint64_t toff = idx->t[i].data_off;
        /* absolute offset must not overflow and must land inside THIS shard's file */
        if ((long long)toff < 0 || toff > (uint64_t)(fsz)) {
            GIERR("'%s': tensor '%s' has an implausible relative offset %llu", path, idx->t[i].name, (unsigned long long)toff);
            close(fd); return 0;
        }
        long long abs_off = data_base + (long long)toff;
        if (abs_off < data_base || abs_off > fsz) {
            GIERR("'%s': tensor '%s' data offset %lld is outside the file (size %lld)", path, idx->t[i].name, abs_off, fsz);
            close(fd); return 0;
        }
        idx->t[i].data_off = (uint64_t)abs_off;
    }

    out->fd = fd;   /* kept open -- gguf_index_open()/gguf_index_free() own it now */
    out->data_base = data_base;
    if (shard_ix == 0) { idx->alignment = alignment; idx->data_base = data_base; }
    return 1;

#undef GIERR
}

/* Open a GGUF file -- or, for a llama.cpp split filename, the WHOLE set of
 * sibling shards -- and build ONE retaining tensor index covering every
 * tensor in the set: name, type, shape, shard, and the tensor's ABSOLUTE
 * (within-its-shard) file offset. Read-only, does not read tensor data.
 * Returns 1 on success, 0 on any malformed/truncated/missing-shard input (err
 * filled). `idx` is zeroed by the caller's memset convention is NOT assumed
 * here -- this function initializes every field itself. On failure, idx is
 * guaranteed empty (n=0, t=NULL, shard=NULL, nshard=0), so the caller does NOT
 * need to call gguf_index_free() after a failed open, though doing so is also
 * safe (a no-op on an already-empty index).
 *
 * `path` may be ANY shard of a split set, not just the first: all N sibling
 * filenames are reconstructed from the shared prefix once the split naming
 * convention is recognised (gguf_split_name_parse()), every shard is opened in
 * order and cross-checked (gguf_index_open_shard()), and their tensors are
 * folded into one index. A missing/unopenable shard fails loudly, naming which
 * shard and why, rather than silently returning a partial model. For an
 * ordinary (non-split) GGUF, nshard is 1 and this function takes exactly the
 * single-file code path it always did, producing byte-identical
 * `GgufTensorInfo` records to before this feature existed. */
/* io11 measurement: prints header-parse wall time to stderr under
 * COLI_LOAD_PROF=1, wrapping only this function's body (KV + tensor-info walk
 * for every shard) -- not tensor data reads, not process startup. CLOCK_MONOTONIC,
 * matches the "loaded in" line's own clock family so the two are comparable. */
static int gguf_index_open(const char *path, GgufIndex *idx, char *err, size_t errcap) {
    int prof = getenv("COLI_LOAD_PROF") != NULL;
    struct timespec t0, t1;
    if (prof) clock_gettime(CLOCK_MONOTONIC, &t0);

    memset(idx, 0, sizeof *idx);
    idx->alignment = 32; /* GGUF_DEFAULT_ALIGNMENT */

    char prefix[4096];
    int nshard_total = 1;
    int is_split = gguf_split_name_parse(path, prefix, sizeof prefix, &nshard_total);

    GgufShard *shard = (GgufShard *)calloc((size_t)nshard_total, sizeof *shard);
    if (!shard) { if (err && errcap) snprintf(err, errcap, "out of memory"); return 0; }
    for (int s = 0; s < nshard_total; s++) shard[s].fd = -1;

    for (int s = 0; s < nshard_total; s++) {
        char spath[4160];
        if (is_split) snprintf(spath, sizeof spath, "%s-%05d-of-%05d.gguf", prefix, s + 1, nshard_total);
        else          snprintf(spath, sizeof spath, "%s", path);

        if (!gguf_index_open_shard(spath, s, is_split, nshard_total, idx, &shard[s], err, errcap)) {
            for (int j = 0; j < s; j++) if (shard[j].fd >= 0) close(shard[j].fd);
            free(shard);
            gguf_index_free(idx);       /* frees idx->t; idx->shard is still NULL here */
            if (prof) { clock_gettime(CLOCK_MONOTONIC, &t1);
                fprintf(stderr, "[COLI_LOAD_PROF] gguf_index_open FAILED after %.3f ms\n",
                        (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6); }
            return 0;
        }
    }

    idx->shard = shard;
    idx->nshard = (size_t)nshard_total;
    if (prof) { clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "[COLI_LOAD_PROF] gguf_index_open (header parse, %d shard%s): %.3f ms\n",
                nshard_total, nshard_total == 1 ? "" : "s",
                (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6); }
    return 1;
}

#endif /* COLI_GGUF_READER_H */
