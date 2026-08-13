/* test_gguf_real_tensors.c — GgufIndex + ggml_dequant.h cross-checked against
 * REAL production GGUF files, using the REAL ggml reference decoder (linked from
 * libggml-base.so, no reimplementation) on a handful of real tensors.
 *
 * This does NOT dequantize whole files (multi-GB). It opens the file with
 * gguf_index_open(), picks a small number of named tensors, pread()s only THOSE
 * tensor's bytes at the index's data_off, and dequantizes with both our port and
 * the real ggml function, comparing bit-for-bit.
 *
 * Usage:
 *   ./test_gguf_real_tensors <path.gguf> [tensor_name_substring ...]
 * With no name filters, checks blk.0.* and blk.13.* (arbitrary early/mid layer)
 * plus token_embd.weight if present.
 *
 * Build (needs the same libggml-base.so as make_ggml_dequant_fixture.c):
 *   cc -O2 -std=c99 -o tests/test_gguf_real_tensors tests/test_gguf_real_tensors.c \
 *      -I. -L/tmp/ggml_build/bin -lggml-base -Wl,-rpath,/tmp/ggml_build/bin -lm
 */
#include "../ggml_dequant.h"   /* first: defines _GNU_SOURCE before any system header */
#include "../gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern void dequantize_row_q3_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q4_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q5_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q6_K(const void *x, float *y, int64_t k);

static int g_fail = 0;

static int64_t nelem_of(const GgufTensorInfo *t) {
    int64_t n = 1;
    for (int d = 0; d < t->rank; d++) n *= (int64_t)t->shape[d];
    return n;
}

static void check_tensor(const char *path, int fd, const GgufTensorInfo *t, long long fsz) {
    const GgmlType *g = ggml_type(t->ttype);
    if (!g) { printf("SKIP %-40s : unrecognized ggml type %u\n", t->name, t->ttype); return; }

    void (*mine)(const void*, float*, int64_t) = NULL;
    void (*ref)(const void*, float*, int64_t) = NULL;
    if      (!strcmp(g->name, "Q3_K")) { mine = gguf_dequant_q3_K; ref = dequantize_row_q3_K; }
    else if (!strcmp(g->name, "Q4_K")) { mine = gguf_dequant_q4_K; ref = dequantize_row_q4_K; }
    else if (!strcmp(g->name, "Q5_K")) { mine = gguf_dequant_q5_K; ref = dequantize_row_q5_K; }
    else if (!strcmp(g->name, "Q6_K")) { mine = gguf_dequant_q6_K; ref = dequantize_row_q6_K; }
    else if (!strcmp(g->name, "F32"))  { mine = gguf_dequant_f32; }
    else if (!strcmp(g->name, "F16"))  { mine = gguf_dequant_f16; }
    else if (!strcmp(g->name, "BF16")) { mine = gguf_dequant_bf16; }
    else { printf("SKIP %-40s : type %s not in the stage-1 decoder set\n", t->name, g->name); return; }

    int64_t nelem = nelem_of(t);
    if (nelem % g->blck != 0) {
        fprintf(stderr, "FAIL %-40s : nelem %lld not a multiple of block size %d (should be impossible for a valid GGUF)\n",
                t->name, (long long)nelem, g->blck);
        g_fail = 1; return;
    }
    int64_t nblk_or_nelem = (g->blck == 1) ? nelem : nelem / g->blck; /* K-quants take nblk; F32/F16/BF16 take nelem */

    /* Cap how much of a large tensor we actually pull+decode -- the point is to
     * cross-check the decoder on real production bytes, not to dequantize whole
     * multi-hundred-MB tensors. GGUF_TEST_MAX_BLOCKS (env, K-quants only) truncates
     * to a leading prefix of super-blocks; each super-block decodes independently,
     * so a prefix is exactly as valid a bit-exactness check as the whole tensor. */
    if (g->blck != 1) {
        const char *capenv = getenv("GGUF_TEST_MAX_BLOCKS");
        if (capenv) {
            int64_t cap = atoll(capenv);
            if (cap > 0 && cap < nblk_or_nelem) { nblk_or_nelem = cap; nelem = cap * g->blck; }
        }
    }
    /* bytes actually being fetched/decoded for THIS run (post-cap) */
    long long nbytes = (g->blck == 1) ? nelem * (long long)g->bytes
                                       : nblk_or_nelem * (long long)g->bytes;

    if ((long long)(t->data_off + (uint64_t)nbytes) > fsz) {
        fprintf(stderr, "FAIL %-40s : tensor data (%lld bytes at off %llu) runs past EOF (%lld) -- would over-read\n",
                t->name, nbytes, (unsigned long long)t->data_off, fsz);
        g_fail = 1; return;
    }

    void *raw = malloc((size_t)nbytes);
    ssize_t r = pread(fd, raw, (size_t)nbytes, (off_t)t->data_off);
    if (r != (ssize_t)nbytes) {
        fprintf(stderr, "FAIL %-40s : short read (%zd of %lld bytes) at offset %llu\n",
                t->name, r, nbytes, (unsigned long long)t->data_off);
        g_fail = 1; free(raw); return;
    }

    float *got = malloc((size_t)nelem * sizeof(float));
    mine(raw, got, nblk_or_nelem);

    if (ref) {
        float *want = malloc((size_t)nelem * sizeof(float));
        ref(raw, want, nelem);
        int64_t bad = 0, first_bad = -1;
        for (int64_t i = 0; i < nelem; i++) {
            uint32_t gb, wb; memcpy(&gb, &got[i], 4); memcpy(&wb, &want[i], 4);
            if (gb != wb) { bad++; if (first_bad < 0) first_bad = i; }
        }
        if (bad) {
            fprintf(stderr, "FAIL %-40s : %lld/%lld elements NOT bit-exact vs real ggml decoder (first at %lld, got=%.9g want=%.9g)\n",
                    t->name, (long long)bad, (long long)nelem, (long long)first_bad, got[first_bad], want[first_bad]);
            g_fail = 1;
        } else {
            printf("PASS %-40s : type=%-5s shape=[", t->name, g->name);
            for (int d = 0; d < t->rank; d++) printf("%s%llu", d ? "," : "", (unsigned long long)t->shape[d]);
            printf("] off=%llu bytes=%lld  %lld/%lld elements bit-exact vs real ggml decoder\n",
                   (unsigned long long)t->data_off, nbytes, (long long)nelem, (long long)nelem);
        }
        free(want);
    } else {
        /* F32/F16/BF16: no reference call needed here, the value semantics are
         * exhaustively covered by test_ggml_dequant.c; here we only prove the
         * index's offset/length let us read this exact tensor without an
         * over-read (already checked above) and that decode doesn't crash. */
        printf("PASS %-40s : type=%-5s shape=[", t->name, g->name);
        for (int d = 0; d < t->rank; d++) printf("%s%llu", d ? "," : "", (unsigned long long)t->shape[d]);
        printf("] off=%llu bytes=%lld  read+decoded OK (%s has no per-element reference call here; "
               "covered exhaustively in test_ggml_dequant.c)\n",
               (unsigned long long)t->data_off, nbytes, g->name);
    }
    free(raw); free(got);
}

/* Exact match, or a trailing '*' for a prefix match (e.g. "blk.0.*"). Deliberately
 * NOT a substring match: "output.weight" is a substring of EVERY
 * "blk.N.attn_output.weight", which would silently pull in dozens of unintended,
 * un-fetched (sparse-hole / all-zero) tensors and "pass" them meaninglessly. */
static int name_match(const char *name, const char *pat) {
    size_t pl = strlen(pat);
    if (pl && pat[pl-1] == '*') return strncmp(name, pat, pl-1) == 0;
    return strcmp(name, pat) == 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <path.gguf> [exact_name | prefix* ...]\n", argv[0]); return 2; }
    const char *path = argv[1];

    GgufIndex idx; char err[256];
    if (!gguf_index_open(path, &idx, err, sizeof err)) {
        fprintf(stderr, "gguf_index_open failed: %s\n", err);
        return 1;
    }
    printf("opened %s: %zu tensors, alignment=%u, data_base=%lld\n", path, idx.n, idx.alignment, idx.data_base);

    struct stat st; stat(path, &st);
    long long fsz = (long long)st.st_size;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    int nfilters = argc - 2;
    int checked = 0;
    for (size_t i = 0; i < idx.n; i++) {
        const GgufTensorInfo *t = &idx.t[i];
        int match;
        if (nfilters > 0) {
            match = 0;
            for (int f = 0; f < nfilters; f++) if (name_match(t->name, argv[2 + f])) { match = 1; break; }
        } else {
            match = name_match(t->name, "blk.0.*") || name_match(t->name, "blk.13.*") ||
                    name_match(t->name, "token_embd.weight");
        }
        if (!match) continue;
        checked++;
        check_tensor(path, fd, t, fsz);
    }
    printf("checked %d tensor(s)\n", checked);

    close(fd);
    gguf_index_free(&idx);
    return g_fail ? 1 : 0;
}
