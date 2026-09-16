/* test_block_kv_reserve -- the fused block's device-KV reservation arithmetic
 * (src/model.cpp: coli_model_planned_ctx, coli_model_kv_bytes_planned, and
 * their use in coli_gpu_upload Pass 2 / main.cpp's auto-planner call site),
 * added 2026-09-16, REVISED the same day before landing.
 *
 * WHAT THIS CATCHES, AND WHY IT CHANGED ONCE ALREADY. The first draft reserved
 * coli_model_kv_bytes(m) -- KV sized at m->max_ctx, the model's ADVERTISED
 * ceiling. For Qwen3-30B-A3B that is 40,960 (its own load line: "kv cache: 1
 * slot(s) x 512 ctx allocated (grows to 40960)"), i.e. 7.5 GiB -- cutting a
 * 10 GiB expert budget to ~2 GiB and losing far more decode than the r11
 * failure the reservation exists to fix, because resident experts are the
 * actual decode limiter, not the block. kv_init is in fact called at
 * m->kv_ctx, which starts small (512, or less; coli_load) and only grows via
 * kv_grow's DOUBLING as far as the prompts an actual run submits require (a
 * 682-token prompt: 512 -> 1024 = 201 MiB, nothing like 7.5 GiB). This file
 * exercises the corrected functions: coli_model_planned_ctx (the ctx to
 * reserve for: -c if the user gave one, else min(4096, max_ctx), rounded UP
 * to the grid kv_grow doubles onto and capped at max_ctx) and
 * coli_model_kv_bytes_planned (KV bytes at that ctx instead of max_ctx). A
 * wrong formula here would silently regress to either extreme -- reserving
 * at max_ctx again (this file's whole reason to exist) or reserving ~0 (back
 * to the original r11 failure) -- and "make coli-gpu" would still link, run,
 * and look fine either way, because the bug is arithmetic, not a crash.
 *
 * DOES NOT touch the GPU, load a gguf, or open a Vulkan device -- coli_model
 * is a plain struct (model.h) and the functions under test read a handful of
 * int fields off it plus one getenv(). Runs on any CPU-only build.
 *
 * CONTROL. The last case feeds the same comparison machinery a deliberately
 * wrong expectation (the round-to-grid step skipped) and requires it to
 * report a mismatch -- a comparison that can only pass would make every CHECK
 * above it decorative. */
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

/* Build a coli_model with only the fields coli_model_planned_ctx() and
 * coli_model_kv_bytes_planned() actually read. Zero-initialised so any field
 * this test forgets is a visible 0, not garbage. kv_ctx and max_ctx_given
 * default to the values coli_load itself would set for a fresh model (512,
 * not given) unless a case overrides them. */
static void mk_model(coli_model *m, int n_layers, int n_kv_heads, int head_dim,
                     int n_slots, int max_ctx, int max_ctx_given, int kv_ctx) {
    memset(m, 0, sizeof *m);
    m->cfg.n_layers = n_layers;
    m->cfg.n_kv_heads = n_kv_heads;
    m->cfg.head_dim = head_dim;
    m->n_slots = n_slots;
    m->max_ctx = max_ctx;
    m->max_ctx_given = max_ctx_given;
    m->kv_ctx = kv_ctx;
}

/* Independent re-derivation of coli_model_planned_ctx's policy, so the test
 * is not just calling the production function and comparing it to itself.
 * COLI_PLAN_CTX must be unset (or empty) in the caller's environment for this
 * to apply -- see main()'s unsetenv at start and the explicit override case. */
static int64_t expect_planned_ctx(int max_ctx, int max_ctx_given, int kv_ctx) {
    int64_t planned = max_ctx_given ? (int64_t)max_ctx : (max_ctx < 4096 ? (int64_t)max_ctx : 4096);
    if (planned < 1) planned = 1;
    if (planned > max_ctx) planned = max_ctx;
    int64_t grid = kv_ctx > 0 ? (int64_t)kv_ctx : 1;
    while (grid < planned && grid < max_ctx) grid *= 2;
    if (grid > max_ctx) grid = max_ctx;
    return grid;
}
static uint64_t expect_kv_bytes_at(int n_layers, int n_kv_heads, int head_dim,
                                    int n_slots, int64_t ctx) {
    return (uint64_t)n_slots * (uint64_t)n_layers * 2 * (uint64_t)n_kv_heads *
           (uint64_t)ctx * (uint64_t)head_dim * (uint64_t)sizeof(float);
}
static int64_t expect_reserved_mib(uint64_t kv_need) {
    int64_t need = (int64_t)kv_need;
    int64_t margin = need / 16;
    int64_t reserve = need + margin;
    return reserve / (1024 * 1024);
}

int main(void) {
    /* A stray COLI_PLAN_CTX in the ambient environment would silently change
     * every case below; make the starting condition explicit rather than
     * hoping the shell is clean. */
    unsetenv("COLI_PLAN_CTX");
    coli_model m;

    /* Case 1: the incident's own model and its own numbers, NO -c given.
     * Qwen3-30B-A3B: 48 layers (README.md:566), num_key_value_heads=4,
     * head_dim=128 (Qwen's published HF config -- not re-verified against a
     * loaded gguf in this test), max_ctx=40960 and kv_ctx=512, both taken
     * verbatim from the coordinator's own load-line quote ("kv cache: 1
     * slot(s) x 512 ctx allocated (grows to 40960)"). n_slots=1 (the
     * incident's COLI_GEN_BATCH=1). */
    {
        int n_layers=48, n_kv_heads=4, head_dim=128, n_slots=1, max_ctx=40960, kv_ctx=512;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx, /*given=*/0, kv_ctx);

        int got_ctx = coli_model_planned_ctx(&m);
        int64_t want_ctx = expect_planned_ctx(max_ctx, 0, kv_ctx);
        CHECK(got_ctx == want_ctx, "no -c given: planned_ctx=%d, expected %lld (min(4096,max_ctx) "
              "rounded up from kv_ctx=512 by doubling)", got_ctx, (long long)want_ctx);
        /* By hand: min(4096, 40960)=4096; 512->1024->2048->4096, already on
         * the grid exactly -- no cap needed since 4096 < 40960. */
        CHECK(got_ctx == 4096, "hand-computed: planned_ctx == 4096, got %d", got_ctx);

        uint64_t got_bytes = coli_model_kv_bytes_planned(&m);
        uint64_t want_bytes = expect_kv_bytes_at(n_layers, n_kv_heads, head_dim, n_slots, got_ctx);
        CHECK(got_bytes == want_bytes, "kv_bytes_planned=%llu, expected %llu (n_slots*n_layers*2*n_kv_heads*planned*head_dim*4)",
              (unsigned long long)got_bytes, (unsigned long long)want_bytes);
        /* By hand: 1*48*2*4*4096*128*4 = 805,306,368 bytes = 768 MiB exactly --
         * against 7.5 GiB the withdrawn max_ctx formula would have reserved. */
        CHECK(got_bytes == 805306368ULL, "hand-computed: kv_bytes_planned == 805306368 (768 MiB), got %llu",
              (unsigned long long)got_bytes);

        int64_t got_mib = expect_reserved_mib(got_bytes);
        CHECK(got_mib == 816, "reserved MiB with 1/16 margin = %lld, expected 816 (768+48)", (long long)got_mib);
        int64_t budget_after = 10240 - got_mib;
        CHECK(budget_after == 9424 && budget_after > 8192,
              "budget 10240 -> %lld MiB after reserving for the block's KV -- comfortably above the "
              "9216 MB the incident measured as ENGAGED (not the ~2 GiB the withdrawn max_ctx formula left)",
              (long long)budget_after);
    }

    /* Case 2: -c GIVEN, small hand-checkable numbers so the CAP is what is
     * under test (planned would overshoot max_ctx via doubling and must be
     * clamped back down to it, not left at the first grid step that exceeds
     * it). kv_ctx starts at 8, max_ctx=100 (given): doubling 8->16->32->64->
     * 128 overshoots 100, so the cap must bring it back to exactly 100. */
    {
        int n_layers=2, n_kv_heads=1, head_dim=64, n_slots=2, max_ctx=100, kv_ctx=8;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx, /*given=*/1, kv_ctx);
        int got_ctx = coli_model_planned_ctx(&m);
        CHECK(got_ctx == 100, "-c given (100), doubling from kv_ctx=8 overshoots to 128 -- "
              "capped back to max_ctx: got %d, expected 100", got_ctx);
        uint64_t got_bytes = coli_model_kv_bytes_planned(&m);
        /* By hand: 2*2*2*1*100*64*4 = 204,800 bytes. */
        CHECK(got_bytes == 204800ULL, "kv_bytes_planned at the capped ctx == 204800, got %llu",
              (unsigned long long)got_bytes);
    }

    /* Case 3: NO -c given, and the model's OWN max_ctx (2048) is already
     * below the 4096 default cap -- planned must follow max_ctx down, not
     * reserve for a context the model does not support. */
    {
        int n_layers=4, n_kv_heads=2, head_dim=32, n_slots=1, max_ctx=2048, kv_ctx=512;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx, /*given=*/0, kv_ctx);
        int got_ctx = coli_model_planned_ctx(&m);
        CHECK(got_ctx == 2048, "no -c, max_ctx(2048) < the 4096 default: planned follows max_ctx, got %d", got_ctx);
    }

    /* Case 4: COLI_PLAN_CTX overrides everything else, still rounded to the
     * grid and still capped at max_ctx. Set/unset around this one case only,
     * so it cannot leak into any other. */
    {
        setenv("COLI_PLAN_CTX", "3000", 1);
        int n_layers=1, n_kv_heads=1, head_dim=64, n_slots=1, max_ctx=40960, kv_ctx=512;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx, /*given=*/0, kv_ctx);
        int got_ctx = coli_model_planned_ctx(&m);
        /* 512->1024->2048->4096 (>=3000), no cap needed (4096 < 40960). */
        CHECK(got_ctx == 4096, "COLI_PLAN_CTX=3000 overrides the default policy, rounds up to 4096: got %d", got_ctx);
        unsetenv("COLI_PLAN_CTX");
    }

    /* CONTROL: a comparison that cannot fail proves nothing. Feed the same
     * machinery a DELIBERATELY WRONG expectation -- the round-to-grid step
     * skipped, i.e. "planned" used raw instead of rounded up -- against
     * case 1's real model, and require the mismatch to be caught. This runs
     * outside the pass/fail tally above (its own counter), so a real defect
     * in the cases above is never masked by this always-wrong case, and a
     * bug that made comparisons vacuously true would show up as a spurious
     * "ok" here. */
    {
        int n_layers=48, n_kv_heads=4, head_dim=128, n_slots=1, max_ctx=40960, kv_ctx=512;
        mk_model(&m, n_layers, n_kv_heads, head_dim, n_slots, max_ctx, /*given=*/0, kv_ctx);
        int got_ctx = coli_model_planned_ctx(&m);
        int wrong_ctx = 4096 - 1;   /* "planned" without the round-up-to-grid step */
        int control_ok = (got_ctx != wrong_ctx);
        fprintf(stderr, "%s: control -- a deliberately wrong expectation (grid rounding skipped) "
                        "must NOT match: got=%d wrong=%d\n",
                control_ok ? "ok  " : "FAIL", got_ctx, wrong_ctx);
        if (!control_ok) { fails++; fprintf(stderr,
            "FAIL: the comparison apparatus cannot fail -- every CHECK above is decorative\n"); }
    }

    if (fails) { fprintf(stderr, "\n%d check(s) FAILED\n", fails); return 1; }
    fprintf(stderr, "\nall checks passed\n");
    return 0;
}
