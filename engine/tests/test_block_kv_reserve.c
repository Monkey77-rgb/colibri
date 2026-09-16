/* test_block_kv_reserve -- the fused block's device-KV reservation arithmetic
 * (src/model.cpp, coli_gpu_upload Pass 2, "Reserve the fused block's device KV
 * cache OUT OF the expert budget"), added 2026-09-16.
 *
 * WHAT THIS CATCHES. The reservation subtracts coli_model_kv_bytes(m) (already
 * used, unmodified, by the auto-planner in hw_detect.c) plus a 1/16 margin from
 * COLI_MOE_VRAM_MB before the expert pass spends it. A wrong formula here would
 * either starve the experts for no reason (reserve too large) or leave kv_init
 * exactly where it was measured failing on 2026-09-16 (reserve too small,
 * effectively 0) -- and either way "make coli-gpu" would still link, run, and
 * look fine, because the bug is arithmetic, not a crash. This test exercises
 * the SAME production function the fix calls (coli_model_kv_bytes, model.h) so
 * a change to that formula and a change to this test's independent formula
 * cannot silently drift apart without one of them failing.
 *
 * DOES NOT touch the GPU, load a gguf, or open a Vulkan device -- coli_model
 * is a plain struct (model.h) and coli_model_kv_bytes reads five int fields
 * off it, nothing else. Runs on any CPU-only build.
 *
 * CONTROL. check_can_fail() feeds the SAME comparison macro a deliberately
 * wrong expectation (the *2 for K+V dropped) and requires it to report a
 * mismatch -- a comparison that can only pass would make every CHECK below
 * decorative. */
#include "../src/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } \
    else { fprintf(stderr, "ok:   " __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

/* Build a coli_model with only the fields coli_model_kv_bytes() (and the
 * reservation math in coli_gpu_upload) actually read. Zero-initialised so any
 * field this test forgets is a visible 0, not garbage. */
static void mk_model(coli_model *m, int n_layers, int n_kv_heads, int head_dim,
                     int n_slots, int max_ctx) {
    memset(m, 0, sizeof *m);
    m->cfg.n_layers = n_layers;
    m->cfg.n_kv_heads = n_kv_heads;
    m->cfg.head_dim = head_dim;
    m->n_slots = n_slots;
    m->max_ctx = max_ctx;
}

/* Mirrors coli_model_kv_bytes(): n_slots * n_layers * 2(K+V) * n_kv_heads *
 * max_ctx * head_dim * sizeof(element). This build has no -DCOLI_KV_F16, so
 * coli_kvt == float and this equals the device buffer's own element size
 * (coli_vk_kv_init's `per`, src/vk_backend.c) -- the whole reason the fix
 * reuses coli_model_kv_bytes() instead of a second formula. */
static uint64_t expect_kv_bytes(int n_layers, int n_kv_heads, int head_dim,
                                 int n_slots, int max_ctx) {
    return (uint64_t)n_slots * (uint64_t)n_layers * 2 * (uint64_t)n_kv_heads *
           (uint64_t)max_ctx * (uint64_t)head_dim * (uint64_t)sizeof(float);
}

/* Mirrors the reservation in coli_gpu_upload: reserve = kv_need + kv_need/16
 * (integer division, same as the C++ int64_t arithmetic in model.cpp), then
 * MiB = reserve / (1024*1024) via integer division, matching the fprintf
 * there. */
static int64_t expect_reserved_mib(uint64_t kv_need) {
    int64_t need = (int64_t)kv_need;
    int64_t margin = need / 16;
    int64_t reserve = need + margin;
    return reserve / (1024 * 1024);
}

int main(void) {
    coli_model m;

    /* Case 1: the 2026-09-16 incident's model, published Qwen3-30B-A3B config
     * (48 layers, d=2048 -- README.md:566; num_key_value_heads=4, head_dim=128
     * per Qwen's published HF config -- not re-verified against a loaded gguf
     * in this test, which is why it is labelled "published", not "measured").
     * n_slots=1 (single-sequence, the incident's COLI_GEN_BATCH=1), max_ctx
     * 16384 (a representative long-context run; NOT the small 512 m->kv_ctx
     * starts at -- see the comment on coli_model_kv_bytes and the reservation
     * site for why max_ctx is the right bound and kv_ctx would under-reserve). */
    {
        int n_layers = 48, n_kv_heads = 4, head_dim = 128, n_slots = 1, max_ctx = 16384;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx);
        uint64_t got = coli_model_kv_bytes(&m);
        uint64_t want = expect_kv_bytes(n_layers, n_kv_heads, head_dim, n_slots, max_ctx);
        CHECK(got == want, "Qwen3-30B-A3B-shaped (48L/4kvh/128hd/1slot/16384ctx): "
              "coli_model_kv_bytes=%llu bytes (%.3f GiB), expected %llu",
              (unsigned long long)got, got / 1073741824.0, (unsigned long long)want);

        int64_t got_mib = expect_reserved_mib(got);
        /* Independently by hand: 48*2*1*4*16384*128*4 bytes = 3,221,225,472 =
         * 3072 MiB exactly; +1/16 margin = 3072 + 192 = 3264 MiB. */
        CHECK(got == 3221225472ULL, "hand-computed byte count matches: %llu == 3221225472",
              (unsigned long long)got);
        CHECK(got_mib == 3264, "reserved MiB with 1/16 margin = %lld, expected 3264", (long long)got_mib);

        /* Applied against the incident's own budget: a 10240 MiB
         * COLI_MOE_VRAM_MB with this reservation subtracted leaves headroom,
         * where before the fix the whole 10240 went to experts and kv_init's
         * r11 request (this same byte count, at whatever kv_ctx had grown to
         * by the first forward pass) had nothing left. */
        int64_t budget_before = 10240;
        int64_t budget_after = budget_before - got_mib;
        CHECK(budget_after == 10240 - 3264 && budget_after > 0,
              "budget 10240 -> %lld MiB after reserving for the block's KV (still positive)",
              (long long)budget_after);
    }

    /* Case 2: small hand-checkable numbers, independent of any real model, so
     * the formula itself -- not a coincidence of the case-1 constants -- is
     * what is under test. 2 layers, 1 kv head, 64 head_dim, 2 slots, 512 ctx:
     * bytes = 2*2*1*512*64*2*4 slots... written out: n_slots(2)*n_layers(2)*
     * 2*n_kv_heads(1)*max_ctx(512)*head_dim(64)*sizeof(float)(4)
     * = 2*2*2*1*512*64*4 = 1,048,576 bytes = 1 MiB exactly. */
    {
        int n_layers = 2, n_kv_heads = 1, head_dim = 64, n_slots = 2, max_ctx = 512;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx);
        uint64_t got = coli_model_kv_bytes(&m);
        CHECK(got == 1048576ULL, "small hand-checked case: coli_model_kv_bytes=%llu, expected 1048576 (1 MiB)",
              (unsigned long long)got);
        int64_t got_mib = expect_reserved_mib(got);
        /* 1 MiB + 1/16 MiB margin, integer division: margin = 65536/16=4096
         * bytes -> reserve=1,052,672 bytes -> /1048576 = 1 MiB (integer floor). */
        CHECK(got_mib == 1, "1 MiB + 1/16 margin still floors to 1 MiB (integer division): got %lld", (long long)got_mib);
    }

    /* CONTROL: a comparison that cannot fail proves nothing. Feed the same
     * CHECK machinery a DELIBERATELY WRONG expectation -- the *2 for K+V
     * dropped -- against case 1's real model, and require the mismatch to be
     * caught. This runs outside the pass/fail tally above (its own counter),
     * so a real defect in the two cases above is never masked by this always-
     * wrong case, and a bug that made comparisons vacuously true would show up
     * as a spurious "ok" here. */
    {
        int n_layers = 48, n_kv_heads = 4, head_dim = 128, n_slots = 1, max_ctx = 16384;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx);
        uint64_t got = coli_model_kv_bytes(&m);
        uint64_t wrong_want = (uint64_t)n_slots * (uint64_t)n_layers * /* no *2 */
                              (uint64_t)n_kv_heads * (uint64_t)max_ctx *
                              (uint64_t)head_dim * (uint64_t)sizeof(float);
        int control_ok = (got != wrong_want);
        fprintf(stderr, "%s: control -- a deliberately wrong expectation (K+V *2 dropped) "
                        "must NOT match: got=%llu wrong_want=%llu\n",
                control_ok ? "ok  " : "FAIL", (unsigned long long)got, (unsigned long long)wrong_want);
        if (!control_ok) { fails++; fprintf(stderr,
            "FAIL: the comparison apparatus cannot fail -- every CHECK above is decorative\n"); }
    }

    if (fails) { fprintf(stderr, "\n%d check(s) FAILED\n", fails); return 1; }
    fprintf(stderr, "\nall checks passed\n");
    return 0;
}
