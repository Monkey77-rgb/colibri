/* gguf_meta.h — read GGUF key/value metadata (hyperparameters, vocab, merges).
 *
 * WHY THIS IS A SEPARATE HEADER. gguf_reader.h parses the KV section but keeps
 * only `general.alignment` and gguf_skip()s every other value -- deliberately, it
 * only ever needed the tensor directory. Its dequant path is proven bit-exact
 * against ggml over 473,956,352 elements of the live fleet models (2026-08-15),
 * so it is NOT edited here. This header adds a second, read-only pass over the
 * same KV section. gguf_reader.h must be included first; this reuses its
 * primitives (gguf_read_at / gguf_str / gguf_skip) and its bounds discipline.
 *
 * UNTRUSTED INPUT. Same rule as gguf_reader.h: a GGUF file is untrusted. Every
 * offset and length is bounds-checked against the real file size before use, and
 * a malformed file returns 0 rather than over-reading. Array element counts are
 * capped by MAX_ARRLEN, string lengths by MAX_STRLEN, both from gguf_reader.h.
 *
 * COST. Lookups re-scan the KV section (a few hundred entries) per call. That is
 * deliberate: it holds no state that could go stale, and loading a model does it
 * a few dozen times total. Do not put it in a loop over tensors.
 */
#ifndef COLI_GGUF_META_H
#define COLI_GGUF_META_H

#ifndef COLI_GGUF_READER_H
#error "include gguf_reader.h before gguf_meta.h"
#endif

typedef struct {
    int       fd;
    long long fsz;
    long long kv_off;    /* offset of the first KV entry */
    uint64_t  nkv;
} GgufMeta;

#define GMERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)

/* Opens for metadata reading. Takes its own fd so it is independent of any
 * GgufIndex lifetime. Close with gguf_meta_close(). */
static int gguf_meta_open(const char *path, GgufMeta *m, char *err, size_t errcap) {
    memset(m, 0, sizeof *m);
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { GMERR("open failed"); return 0; }
    struct stat st;
    if (fstat(m->fd, &st) != 0) { GMERR("fstat failed"); close(m->fd); m->fd = -1; return 0; }
    m->fsz = (long long)st.st_size;

    uint32_t magic, ver; uint64_t ntensor, nkv;
    if (!gguf_read_at(m->fd, &magic, 4, 0) || magic != GGUF_MAGIC) {
        GMERR("not a GGUF file"); close(m->fd); m->fd = -1; return 0;
    }
    if (!gguf_read_at(m->fd, &ver, 4, 4) || !gguf_read_at(m->fd, &ntensor, 8, 8) ||
        !gguf_read_at(m->fd, &nkv, 8, 16)) {
        GMERR("truncated header"); close(m->fd); m->fd = -1; return 0;
    }
    if (nkv > MAX_KV) { GMERR("implausible kv count %llu", (unsigned long long)nkv);
                        close(m->fd); m->fd = -1; return 0; }
    m->nkv = nkv;
    m->kv_off = 24;
    return 1;
}

static void gguf_meta_close(GgufMeta *m) {
    if (m && m->fd >= 0) { close(m->fd); m->fd = -1; }
}

/* Locate `key`. On success sets *type and *vpos (offset of the VALUE, i.e. just
 * past the type tag). Returns 0 if absent -- absence is a normal answer here,
 * not an error: llama and qwen2 legitimately carry different key sets. */
static int gguf_meta_find(GgufMeta *m, const char *key, uint32_t *type, long long *vpos) {
    long long off = m->kv_off;
    char k[256];
    for (uint64_t i = 0; i < m->nkv; i++) {
        if (!gguf_str(m->fd, &off, m->fsz, k, sizeof k)) return 0;
        uint32_t t;
        if (off + 4 > m->fsz || !gguf_read_at(m->fd, &t, 4, off)) return 0;
        off += 4;
        long long v = off;
        if (!strcmp(k, key)) { if (type) *type = t; if (vpos) *vpos = v; return 1; }
        if (!gguf_skip(m->fd, &off, m->fsz, t)) return 0;
    }
    return 0;
}

/* Integer read that accepts any GGUF integer/bool width and widens. A model's
 * head_count may legitimately be stored as u32 in one file and u64 in another;
 * demanding one exact type is how a loader rejects a valid model. */
static int gguf_meta_i64(GgufMeta *m, const char *key, long long *out) {
    uint32_t t; long long vp;
    if (!gguf_meta_find(m, key, &t, &vp)) return 0;
    size_t sz;
    if (!gguf_scalar_size(t, &sz)) return 0;
    if (vp + (long long)sz > m->fsz) return 0;
    unsigned char b[8];
    if (!gguf_read_at(m->fd, b, sz, vp)) return 0;
    switch (t) {
        case G_U8:  *out = (long long)*(uint8_t*)b;  return 1;
        case G_I8:  *out = (long long)*(int8_t*)b;   return 1;
        case G_U16: *out = (long long)*(uint16_t*)b; return 1;
        case G_I16: *out = (long long)*(int16_t*)b;  return 1;
        case G_U32: *out = (long long)*(uint32_t*)b; return 1;
        case G_I32: *out = (long long)*(int32_t*)b;  return 1;
        case G_U64: *out = (long long)*(uint64_t*)b; return 1;
        case G_I64: *out = (long long)*(int64_t*)b;  return 1;
        case G_BOOL:*out = (long long)*(uint8_t*)b;  return 1;
        default: return 0;
    }
}

/* Float read. Accepts f32/f64, and also integer types: rope.freq_base is f32 in
 * every file seen here, but a value like 1000000 stored as u32 is still valid. */
static int gguf_meta_f32(GgufMeta *m, const char *key, float *out) {
    uint32_t t; long long vp;
    if (!gguf_meta_find(m, key, &t, &vp)) return 0;
    if (t == G_F32) {
        float v; if (vp + 4 > m->fsz || !gguf_read_at(m->fd, &v, 4, vp)) return 0;
        *out = v; return 1;
    }
    if (t == G_F64) {
        double v; if (vp + 8 > m->fsz || !gguf_read_at(m->fd, &v, 8, vp)) return 0;
        *out = (float)v; return 1;
    }
    long long iv;
    if (gguf_meta_i64(m, key, &iv)) { *out = (float)iv; return 1; }
    return 0;
}

static int gguf_meta_str(GgufMeta *m, const char *key, char *out, size_t outn) {
    uint32_t t; long long vp;
    if (!gguf_meta_find(m, key, &t, &vp) || t != G_STR) return 0;
    long long off = vp;
    return gguf_str(m->fd, &off, m->fsz, out, outn);
}

/* ---- arrays ------------------------------------------------------------- */

typedef struct {
    uint32_t  etype;     /* element type */
    uint64_t  n;         /* element count */
    long long first;     /* offset of the first element */
} GgufArr;

static int gguf_meta_arr(GgufMeta *m, const char *key, GgufArr *a) {
    uint32_t t; long long vp;
    if (!gguf_meta_find(m, key, &t, &vp) || t != G_ARR) return 0;
    if (vp + 12 > m->fsz) return 0;
    uint32_t et; uint64_t n;
    if (!gguf_read_at(m->fd, &et, 4, vp) || !gguf_read_at(m->fd, &n, 8, vp + 4)) return 0;
    if (n > MAX_ARRLEN) return 0;
    a->etype = et; a->n = n; a->first = vp + 12;
    return 1;
}

/* Iterate a string array, calling fn(i, str, len, user) per element. Strings are
 * NOT null-terminated on disk; fn receives an explicit length. Returns the number
 * of elements delivered -- compare it against a->n at the call site. A short
 * return means the file is truncated, and reporting it as "iterated fine" is how
 * a half-read vocab becomes a silently wrong tokenizer. */
static uint64_t gguf_meta_arr_str_foreach(GgufMeta *m, const GgufArr *a,
                                          int (*fn)(uint64_t, const char*, uint32_t, void*),
                                          void *user) {
    if (a->etype != G_STR) return 0;
    long long off = a->first;
    char *buf = NULL; uint32_t bufcap = 0;
    uint64_t i = 0;
    for (; i < a->n; i++) {
        uint64_t len;
        if (off + 8 > m->fsz || !gguf_read_at(m->fd, &len, 8, off)) break;
        off += 8;
        if (len > MAX_STRLEN || off + (long long)len > m->fsz) break;
        if (len + 1 > bufcap) {
            uint32_t nc = (uint32_t)len + 1; if (nc < 64) nc = 64;
            char *nb = realloc(buf, nc);
            if (!nb) break;
            buf = nb; bufcap = nc;
        }
        if (len && !gguf_read_at(m->fd, buf, (size_t)len, off)) break;
        buf[len] = 0;
        off += (long long)len;
        if (fn && !fn(i, buf, (uint32_t)len, user)) { i++; break; }
    }
    free(buf);
    return i;
}

/* Read a numeric array into an int32 buffer (token_type is i32 in practice). */
static uint64_t gguf_meta_arr_i32(GgufMeta *m, const GgufArr *a, int32_t *out, uint64_t maxn) {
    size_t sz;
    if (!gguf_scalar_size(a->etype, &sz)) return 0;
    uint64_t n = a->n < maxn ? a->n : maxn;
    for (uint64_t i = 0; i < n; i++) {
        long long off = a->first + (long long)(i * sz);
        if (off + (long long)sz > m->fsz) return i;
        unsigned char b[8];
        if (!gguf_read_at(m->fd, b, sz, off)) return i;
        switch (a->etype) {
            case G_U8:  out[i] = *(uint8_t*)b;  break;
            case G_I8:  out[i] = *(int8_t*)b;   break;
            case G_U16: out[i] = *(uint16_t*)b; break;
            case G_I16: out[i] = *(int16_t*)b;  break;
            case G_U32: out[i] = (int32_t)*(uint32_t*)b; break;
            case G_I32: out[i] = *(int32_t*)b;  break;
            default: return i;
        }
    }
    return n;
}

/* Convenience: "<arch>.<suffix>", since every architectural key is namespaced by
 * general.architecture and hardcoding "llama." would silently fail on qwen2. */
static int gguf_meta_arch_i64(GgufMeta *m, const char *arch, const char *suffix, long long *out) {
    char k[256]; snprintf(k, sizeof k, "%s.%s", arch, suffix);
    return gguf_meta_i64(m, k, out);
}
static int gguf_meta_arch_f32(GgufMeta *m, const char *arch, const char *suffix, float *out) {
    char k[256]; snprintf(k, sizeof k, "%s.%s", arch, suffix);
    return gguf_meta_f32(m, k, out);
}

#undef GMERR
#endif /* COLI_GGUF_META_H */
