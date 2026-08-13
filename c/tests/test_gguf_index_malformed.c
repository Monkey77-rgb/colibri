/* test_gguf_index_malformed.c — gguf_index_open() must refuse malformed/truncated
 * GGUF input rather than over-read, and must compute the same data_off convention
 * (data_base + toff, data_base = GGML_PAD(tensor_info_end, alignment)) that
 * gguf_reader.h documents and that test_gguf_real_tensors.c already validated
 * against two real production files. This test builds small synthetic GGUF
 * byte streams by hand (no llama.cpp dependency) to exercise:
 *   1. a valid minimal file with a non-default alignment -> data_off correctness
 *   2. bad magic
 *   3. truncated header (< 24 bytes)
 *   4. implausible kv count
 *   5. a kv value that runs past EOF
 *   6. a tensor whose data offset (once made absolute) runs past EOF
 *   7. a tensor name string whose declared length runs past EOF
 *
 * Every malformed case must return 0 from gguf_index_open AND must leave *idx*
 * safe to pass to gguf_index_free() (checked with ASan when built with
 * -fsanitize=address, but also with a plain build here as a baseline).
 *
 * Build: cc -O2 -std=c99 -Wall -Wextra -o tests/test_gguf_index_malformed \
 *           tests/test_gguf_index_malformed.c -I.
 */
#include "../gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int g_fail = 0;

static void wr(FILE *f, const void *p, size_t n) { fwrite(p, 1, n, f); }
static void wr_u32(FILE *f, uint32_t v) { wr(f, &v, 4); }
static void wr_u64(FILE *f, uint64_t v) { wr(f, &v, 8); }
static void wr_str(FILE *f, const char *s) { uint64_t n = strlen(s); wr_u64(f, n); wr(f, s, (size_t)n); }

/* Writes a minimal valid GGUF: optional "general.alignment" u32 KV, then ONE
 * tensor (F32, 4 elements = 16 bytes), then pads to alignment, then writes the
 * tensor's 16 bytes of real (non-zero, checkable) data. Returns the expected
 * absolute data offset so the caller can assert against it. */
static uint64_t write_minimal(const char *path, uint32_t alignment, int with_alignment_kv) {
    FILE *f = fopen(path, "wb");
    wr_u32(f, GGUF_MAGIC);
    wr_u32(f, 3);                                   /* version */
    wr_u64(f, 1);                                    /* n_tensors */
    wr_u64(f, with_alignment_kv ? 1 : 0);             /* n_kv */

    if (with_alignment_kv) {
        wr_str(f, "general.alignment");
        wr_u32(f, G_U32);
        wr_u32(f, alignment);
    }

    /* tensor info: name, n_dims=1, dims=[4], type=F32(0), offset=0 */
    wr_str(f, "t.weight");
    wr_u32(f, 1);
    wr_u64(f, 4);
    wr_u32(f, 0);     /* F32 */
    wr_u64(f, 0);     /* relative offset within data section */

    long pos = ftell(f);
    uint64_t data_base = ((uint64_t)pos + alignment - 1) & ~((uint64_t)alignment - 1);
    while ((uint64_t)ftell(f) < data_base) fputc(0, f);

    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    wr(f, vals, sizeof vals);
    fclose(f);
    return data_base;
}

static void expect_ok(const char *label, const char *path, uint64_t want_data_off) {
    GgufIndex idx; char err[256];
    int ok = gguf_index_open(path, &idx, err, sizeof err);
    if (!ok) {
        fprintf(stderr, "FAIL %s: expected success, got error: %s\n", label, err);
        g_fail = 1;
        return;
    }
    if (idx.n != 1 || idx.t[0].data_off != want_data_off) {
        fprintf(stderr, "FAIL %s: n=%zu data_off=%llu (want n=1 data_off=%llu)\n",
                label, idx.n, (unsigned long long)idx.t[0].data_off, (unsigned long long)want_data_off);
        g_fail = 1;
    } else {
        printf("PASS %s: data_off=%llu matches GGML_PAD(tensor_info_end, alignment)\n", label, (unsigned long long)want_data_off);
    }
    gguf_index_free(&idx);
}

static void expect_fail(const char *label, const char *path) {
    GgufIndex idx; char err[256];
    int ok = gguf_index_open(path, &idx, err, sizeof err);
    if (ok) {
        fprintf(stderr, "FAIL %s: gguf_index_open should have rejected this file but returned success (n=%zu)\n", label, idx.n);
        g_fail = 1;
        gguf_index_free(&idx);
    } else {
        printf("PASS %s: correctly rejected (%s)\n", label, err);
    }
}

int main(void) {
    const char *p = "/tmp/gguf_idx_test_case.gguf";

    /* 1. default alignment (32), no general.alignment key */
    uint64_t off1 = write_minimal(p, 32, 0);
    expect_ok("default_alignment_32", p, off1);

    /* 2. explicit non-default, non-32 power-of-two alignment (16) */
    uint64_t off2 = write_minimal(p, 16, 1);
    expect_ok("explicit_alignment_16", p, off2);

    /* 3. explicit alignment 1 (degenerate but a valid power of two: 2^0) */
    uint64_t off3 = write_minimal(p, 1, 1);
    expect_ok("explicit_alignment_1", p, off3);

    /* 4. bad magic */
    { FILE *f = fopen(p, "wb"); wr_u32(f, 0xdeadbeef); wr_u32(f, 3); wr_u64(f, 0); wr_u64(f, 0); fclose(f); }
    expect_fail("bad_magic", p);

    /* 5. truncated header (<24 bytes) */
    { FILE *f = fopen(p, "wb"); wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); fclose(f); }
    expect_fail("truncated_header", p);

    /* 6. implausible kv count */
    { FILE *f = fopen(p, "wb"); wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 0); wr_u64(f, (uint64_t)1 << 40); fclose(f); }
    expect_fail("implausible_kv_count", p);

    /* 7. kv value (a u64) whose declared bytes run past EOF: header claims 1 kv,
     * we write the key string but only 3 of the 8 promised value bytes */
    {
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 0); wr_u64(f, 1);
        wr_str(f, "some.key");
        wr_u32(f, G_U64);
        uint8_t partial[3] = {1,2,3};
        wr(f, partial, 3);   /* short by 5 bytes */
        fclose(f);
    }
    expect_fail("kv_value_past_eof", p);

    /* 8. tensor whose (absolute) data offset runs past EOF: valid header/KV, one
     * tensor claiming a huge relative offset */
    {
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 1); wr_u64(f, 0);
        wr_str(f, "t.weight");
        wr_u32(f, 1); wr_u64(f, 4); wr_u32(f, 0);   /* F32, 4 elements */
        wr_u64(f, (uint64_t)1 << 40);                /* relative offset -- absurdly past EOF */
        fclose(f);
    }
    expect_fail("tensor_offset_past_eof", p);

    /* 9. tensor name string length runs past EOF */
    {
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 1); wr_u64(f, 0);
        wr_u64(f, 1000000);   /* claims a 1,000,000-byte name */
        wr(f, "short", 5);    /* file actually ends 5 bytes later */
        fclose(f);
    }
    expect_fail("tensor_name_len_past_eof", p);

    /* 10. tensor rank > 8 (GGML_MAX_DIMS is 4 upstream; gguf_reader.h's own cap is 8) */
    {
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 1); wr_u64(f, 0);
        wr_str(f, "t.weight");
        wr_u32(f, 9);   /* rank 9 > our cap of 8 */
        fclose(f);
    }
    expect_fail("implausible_tensor_rank", p);

    /* 11. general.alignment present but NOT a power of two -- must be rejected,
     * matching upstream gguf.cpp's own refusal (line ~558-560). */
    {
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 0); wr_u64(f, 1);
        wr_str(f, "general.alignment");
        wr_u32(f, G_U32);
        wr_u32(f, 24);   /* NOT a power of two */
        fclose(f);
    }
    expect_fail("alignment_not_power_of_two", p);

    /* 12. REGRESSION: an over-long tensor name must be REJECTED, not truncated.
     *
     * GgufTensorInfo.name is char[512]. gguf_str() truncates to the buffer and
     * still reports success, so two DISTINCT tensors whose names share their
     * first 511 bytes used to be stored under the SAME name — the file was
     * accepted with two colliding records carrying different shapes and offsets.
     * A by-name lookup then silently returns the wrong tensor: no crash, no
     * sanitizer report, wrong numbers. Bit-exactness testing cannot catch it,
     * because both tensors decode perfectly; just not the ones asked for.
     *
     * Found by fuzzing (143-input corpus, Hardware session, 2026-08-13). Fixed by
     * reading tensor names with gguf_str_exact(), which refuses rather than
     * truncates. Two cases: a name exactly at the limit, and the collision pair
     * that made it a correctness bug rather than cosmetic truncation. */
    {
        char longname[600];
        memset(longname, 'a', sizeof longname);

        /* 12a. single name of 511 bytes (== sizeof(name)-1 is the last accepted
         * length, so 511 with the NUL is exactly at the boundary and must fail
         * only above it; use 512 to be unambiguously over). */
        FILE *f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 1); wr_u64(f, 0);
        wr_u64(f, 512); wr(f, longname, 512);      /* name length 512 -> too long */
        wr_u32(f, 1); wr_u64(f, 4); wr_u32(f, 0); wr_u64(f, 0);
        for (int i = 0; i < 64; i++) fputc(0, f);
        float v4[4] = {1,2,3,4}; wr(f, v4, sizeof v4);
        fclose(f);
        expect_fail("tensor_name_too_long", p);

        /* 12b. the collision pair: two names of 520 bytes differing only at byte
         * 515 — identical in their first 512 bytes. Under truncation both stored
         * as the same name with different shapes/offsets. Must be rejected. */
        char a[520], b[520];
        memset(a, 'x', sizeof a); memset(b, 'x', sizeof b);
        a[515] = 'A'; b[515] = 'B';
        f = fopen(p, "wb");
        wr_u32(f, GGUF_MAGIC); wr_u32(f, 3); wr_u64(f, 2); wr_u64(f, 0);
        wr_u64(f, sizeof a); wr(f, a, sizeof a);
        wr_u32(f, 1); wr_u64(f, 4);  wr_u32(f, 0); wr_u64(f, 0);
        wr_u64(f, sizeof b); wr(f, b, sizeof b);
        wr_u32(f, 1); wr_u64(f, 8);  wr_u32(f, 0); wr_u64(f, 64);
        for (int i = 0; i < 128; i++) fputc(0, f);
        float v12[12] = {1,2,3,4,5,6,7,8,9,10,11,12}; wr(f, v12, sizeof v12);
        fclose(f);
        expect_fail("tensor_name_collision_pair", p);
    }

    remove(p);
    if (g_fail) { fprintf(stderr, "\n*** ONE OR MORE TESTS FAILED ***\n"); return 1; }
    printf("\nall tests passed\n");
    return 0;
}
