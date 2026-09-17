/* test_expert_store.c — CPU-only, no model load, no forward pass. Two things
 * under test:
 *
 *   (1) coli_gguf_tensor_slice()/coli_gguf_slice_pread() (loader.c/.h): does
 *       reading ONE expert's slice directly agree, byte for byte, with
 *       reading the WHOLE tensor via the pre-existing coli_gguf_load_raw()
 *       and indexing into it by hand -- on REAL data (a real expert tensor
 *       from the 30B GGUF on disk), for an expert at index 0, one in the
 *       middle, and the LAST index (the one most likely to be off-by-one).
 *       Plus two conditions the slice reader must REJECT: an out-of-range
 *       expert index, and a split count the tensor's byte size does not
 *       divide evenly.
 *
 *   (2) ColiEstore (expert_store.cpp): registered slices are readable,
 *       become resident on first use, and a tight byte budget forces real
 *       eviction+refill that still reproduces the correct bytes -- checked
 *       against the SAME whole-tensor reference used in (1), not just
 *       "some bytes came back".
 *
 * NEGATIVE CONTROL (per the task): a bit-exact comparison that can only ever
 * pass proves nothing (this file's own doctrine). test_slice_vs_whole's
 * final call perturbs one byte of a slice it just read and asserts the
 * comparison against the whole-tensor reference FAILS -- run inline, right
 * after the real positive comparisons, so one program demonstrates both that
 * the check finds the true positive AND that it can find a negative.
 *
 * Needs a real GGUF with MoE experts on disk (default: the 30B this task
 * measures against, override with argv[1] or $COLI_TEST_MODEL). Exits 77
 * (SKIP), not FAIL, if it is absent -- same convention test_mxfp4.c uses --
 * so `make test` stays runnable on a machine with no model on disk.
 */
#include "../src/loader.h"
#include "../src/expert_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail = 1; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
    else { fprintf(stderr, "ok:   " __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

/* --negctl-child <path> <tensor> <expert> <n_expert>: registers ONE expert
 * slice and does ONE coli_estore_get, then writes the resident bytes RAW to
 * stdout and exits -- nothing else. Exists so the parent (below, oracle case
 * 1c) can re-exec this same binary with COLI_BREAK_ESTORE=1 in the
 * environment and diff its output against the parent's own already-proven
 * bit-exact reference: COLI_BREAK_ESTORE flips a static, getenv-cached flag
 * the FIRST time anything in this process calls coli_estore_get, so testing
 * it from inside the already-running parent (which has long since read that
 * env as unset) is not possible without a fresh process. */
static int negctl_child(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "negctl_child: bad args\n"); return 2; }
    const char *path = argv[2]; const char *nm = argv[3];
    int64_t e = atoll(argv[4]); int64_t NE = atoll(argv[5]);
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { fprintf(stderr, "negctl_child: cannot open '%s': %s\n", path, err); return 2; }
    coli_gguf_slice sl;
    if (!coli_gguf_tensor_slice(g, nm, e, NE, &sl)) { fprintf(stderr, "negctl_child: slice failed\n"); return 2; }
    ColiEstore *st = coli_estore_create(/*budget=*/-1, /*direct_pref=*/0);
    int key = 0;
    if (!coli_estore_register(st, &key, &sl)) { fprintf(stderr, "negctl_child: register failed\n"); return 2; }
    const uint8_t *b = coli_estore_get(st, &key);
    if (!b) { fprintf(stderr, "negctl_child: get failed\n"); return 2; }
    fwrite(b, 1, (size_t)sl.nbytes, stdout);
    fflush(stdout);
    coli_estore_destroy(st);
    coli_gguf_close(g);
    return 0;
}

/* Oracle cases 1(a)/1(b), 2026-09-17 brief -- run against gpt-oss-120b's
 * REAL MXFP4 experts specifically, not the Q4_K model main()'s own default
 * argument selects: a Q4_K expert's (off, nbytes) at this model's shape
 * happens to already be a 4096-byte multiple (884736 = 216*4096), so it
 * cannot exercise the aligned-superset path change A exists for -- the
 * measured fact this whole brief rests on (4,406,400 %% 4096 = 3200) is
 * gpt-oss-specific. Independent of main()'s own COLI_TEST_MODEL (which
 * governs the Q4_K-hardcoded whole-tensor reference above, see the file
 * header) so as not to disturb that existing, still-valid check: this model
 * is MXFP4 (ttype 39), and the whole-tensor loader path above hardcodes
 * ttype==12 (Q4_K) -- pointing COLI_TEST_MODEL at gpt-oss breaks THAT check
 * for reasons unrelated to this brief, so it is not done. SKIPs (does not
 * FAIL) if the model file is not present, same convention as main(). */
static void test_gptoss_direct_and_prefetch(void) {
    const char *path = getenv("COLI_TEST_MODEL_GPTOSS");
    if (!path || !*path) path = "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf";
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { fprintf(stderr, "SKIP (gpt-oss cases a/b): cannot open '%s': %s\n", path, err); return; }
    const char *nm = "blk.0.ffn_gate_exps.weight";
    if (!coli_gguf_has(g, nm)) { fprintf(stderr, "SKIP (gpt-oss cases a/b): '%s' not found\n", nm); coli_gguf_close(g); return; }
    int64_t I = coli_gguf_shape(g, nm, 0), O = coli_gguf_shape(g, nm, 1), NE = coli_gguf_shape(g, nm, 2);
    if (I <= 0 || O <= 0 || NE < 16) { fprintf(stderr, "SKIP (gpt-oss cases a/b): unexpected shape\n"); coli_gguf_close(g); return; }
    fprintf(stderr, "gpt-oss model: %s\n  %s shape [I=%lld, O=%lld, n_expert=%lld] (MXFP4)\n",
            path, nm, (long long)I, (long long)O, (long long)NE);

    /* ---- case (a): a NOT-4096-aligned slice, byte-exact through the direct
     * path, with fills counted as direct (not silently falling back). ---- */
    coli_gguf_slice sl1;
    CHECK(coli_gguf_tensor_slice(g, nm, 1, NE, &sl1), "gpt-oss: slice resolves for expert 1 of %lld", (long long)NE);
    CHECK((sl1.off % 4096 != 0) || (sl1.nbytes % 4096 != 0),
          "gpt-oss expert 1 slice is NOT already 4096-aligned (off=%lld nbytes=%lld) -- the condition change A exists for",
          (long long)sl1.off, (long long)sl1.nbytes);
    unsigned char *ref1 = (unsigned char*)malloc((size_t)sl1.nbytes);
    CHECK(coli_gguf_slice_pread(&sl1, ref1), "gpt-oss: reference pread succeeds for expert 1");

    ColiEstore *sta = coli_estore_create(/*budget=*/-1, /*direct_pref=*/1);
    int keya = 0;
    CHECK(coli_estore_register(sta, &keya, &sl1), "gpt-oss: register expert 1 (direct_pref=1)");
    const uint8_t *gota = coli_estore_get(sta, &keya);
    CHECK(gota && memcmp(gota, ref1, (size_t)sl1.nbytes) == 0,
          "gpt-oss: unaligned slice is BIT-EXACT through the O_DIRECT aligned-superset path");
    ColiEstoreStats sta_stats; coli_estore_stats(sta, &sta_stats);
    CHECK(sta_stats.direct_reads == 1 && sta_stats.buffered_reads == 0,
          "gpt-oss: the fill actually went via O_DIRECT (direct_reads=%llu buffered_reads=%llu), not a silent buffered fallback",
          (unsigned long long)sta_stats.direct_reads, (unsigned long long)sta_stats.buffered_reads);
    coli_estore_destroy(sta);

    ColiEstore *stb = coli_estore_create(/*budget=*/-1, /*direct_pref=*/0);
    int keyb = 0;
    coli_estore_register(stb, &keyb, &sl1);
    const uint8_t *gotb = coli_estore_get(stb, &keyb);
    CHECK(gotb && memcmp(gotb, ref1, (size_t)sl1.nbytes) == 0,
          "gpt-oss: COLI_EXPERT_DIRECT=0 (direct_pref=0) equivalent still BIT-EXACT via buffered fallback");
    ColiEstoreStats stb_stats; coli_estore_stats(stb, &stb_stats);
    CHECK(stb_stats.direct_reads == 0 && stb_stats.buffered_reads == 1,
          "gpt-oss: with direct_pref=0 the fill is buffered only (direct_reads=%llu)",
          (unsigned long long)stb_stats.direct_reads);
    coli_estore_destroy(stb);
    free(ref1);

    /* ---- case (b): prefetch of 12 keys under a budget that fits exactly
     * 12 leaves all 12 resident+pinned; a 13th get() cannot evict any of
     * them (they're pinned) so it just grows past budget; unpinning then
     * lets a 14th get() evict one of the original 12. ---- */
    enum { NK = 14 };
    coli_gguf_slice sls[NK]; int dummy[NK];
    for (int e = 0; e < NK; e++) CHECK(coli_gguf_tensor_slice(g, nm, e, NE, &sls[e]), "gpt-oss: slice resolves for expert %d", e);
    int64_t budget_b = sls[0].nbytes * 12;   /* exact fit for 12 (MXFP4 experts are uniform size) */
    ColiEstore *stc = coli_estore_create(budget_b, /*direct_pref=*/0);
    for (int e = 0; e < NK; e++) CHECK(coli_estore_register(stc, &dummy[e], &sls[e]), "gpt-oss: register expert %d for prefetch/pin test", e);

    const void *batch12[12];
    for (int e = 0; e < 12; e++) batch12[e] = &dummy[e];
    CHECK(coli_estore_prefetch(stc, batch12, 12) == 1, "coli_estore_prefetch(12 keys) returns success");
    int all12_resident = 1, all12_pinned = 1;
    for (int e = 0; e < 12; e++) {
        if (!coli_estore_resident(stc, &dummy[e])) all12_resident = 0;
        if (!coli_estore_test_pinned(stc, &dummy[e])) all12_pinned = 0;
    }
    CHECK(all12_resident, "all 12 prefetched keys are resident");
    CHECK(all12_pinned, "all 12 prefetched keys are PINNED");
    {
        unsigned char *r3 = (unsigned char*)malloc((size_t)sls[3].nbytes);
        CHECK(coli_gguf_slice_pread(&sls[3], r3), "gpt-oss: reference pread succeeds for expert 3 (spot check)");
        const uint8_t *g3 = coli_estore_get(stc, &dummy[3]);
        CHECK(g3 && memcmp(g3, r3, (size_t)sls[3].nbytes) == 0, "gpt-oss: prefetched expert 3 is BIT-EXACT");
        free(r3);
    }

    ColiEstoreStats before13; coli_estore_stats(stc, &before13);
    const uint8_t *got13 = coli_estore_get(stc, &dummy[12]);   /* the 13th key: budget is full of PINNED entries */
    CHECK(got13 != NULL, "the 13th key still fills (over budget, since nothing pinned is evictable)");
    int still_all_resident = 1;
    for (int e = 0; e < 12; e++) if (!coli_estore_resident(stc, &dummy[e])) still_all_resident = 0;
    CHECK(still_all_resident, "none of the 12 PINNED entries were evicted by the 13th get() (only a non-pinned entry may ever be picked)");
    ColiEstoreStats after13; coli_estore_stats(stc, &after13);
    CHECK(after13.resident_bytes > budget_b,
          "resident_bytes (%lld) now exceeds the budget (%lld) -- expected: nothing evictable, so the store grows rather than corrupt a pinned entry",
          (long long)after13.resident_bytes, (long long)budget_b);

    coli_estore_unpin_all(stc);
    int any_unpinned_now = 0;
    for (int e = 0; e < 12; e++) if (!coli_estore_test_pinned(stc, &dummy[e])) any_unpinned_now = 1;
    CHECK(any_unpinned_now, "coli_estore_unpin_all cleared the pins on the original 12");

    const uint8_t *got14 = coli_estore_get(stc, &dummy[13]);   /* the 14th key: now eviction can proceed */
    CHECK(got14 != NULL, "the 14th key fills");
    int now_evicted_one = 0;
    for (int e = 0; e < 13; e++) if (!coli_estore_resident(stc, &dummy[e])) now_evicted_one = 1;
    CHECK(now_evicted_one, "UNPINNING THEN ALLOWS EVICTION: at least one previously-pinned entry was evicted by the 14th get()");
    {
        unsigned char *r13 = (unsigned char*)malloc((size_t)sls[13].nbytes);
        CHECK(coli_gguf_slice_pread(&sls[13], r13), "gpt-oss: reference pread succeeds for expert 13 (spot check)");
        CHECK(memcmp(got14, r13, (size_t)sls[13].nbytes) == 0, "gpt-oss: the 14th key's fill is itself BIT-EXACT");
        free(r13);
    }

    coli_estore_destroy(stc);
    coli_gguf_close(g);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--negctl-child")) return negctl_child(argc, argv);
    const char *path = argc > 1 ? argv[1]
                      : (getenv("COLI_TEST_MODEL") ? getenv("COLI_TEST_MODEL")
                      : "/home/monkey/Documents/Ai_Models/qwen3moe/Qwen3-30B-A3B-Q4_K_M.gguf");
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { fprintf(stderr, "SKIP: cannot open '%s': %s\n", path, err); return 77; }

    const char *nm = "blk.0.ffn_gate_exps.weight";
    if (!coli_gguf_has(g, nm)) { fprintf(stderr, "SKIP: '%s' not found in '%s'\n", nm, path); coli_gguf_close(g); return 77; }
    int64_t I = coli_gguf_shape(g, nm, 0);
    int64_t O = coli_gguf_shape(g, nm, 1);
    int64_t NE = coli_gguf_shape(g, nm, 2);
    if (I <= 0 || O <= 0 || NE <= 0) { fprintf(stderr, "SKIP: unexpected shape for '%s'\n", nm); coli_gguf_close(g); return 77; }
    fprintf(stderr, "model: %s\n  %s shape [I=%lld, O=%lld, n_expert=%lld]\n",
            path, nm, (long long)I, (long long)O, (long long)NE);

    /* ---- reference: the whole tensor, via the pre-existing loader path ---- */
    void *whole = NULL; int ttype = -1;
    int64_t ne = coli_gguf_load_raw(g, nm, &whole, &ttype);
    CHECK(ne == I*O*NE, "whole-tensor element count matches I*O*n_expert (%lld == %lld)", (long long)ne, (long long)(I*O*NE));
    CHECK(ttype == 12, "tensor type is Q4_K (ttype=12), got %d", ttype);
    if (!whole || ne != I*O*NE || ttype != 12) { fprintf(stderr, "cannot proceed without a valid reference\n"); return g_fail; }
    int64_t part_bytes = ne * 144 / 256 / NE;   /* Q4_K: 144 bytes / 256-weight superblock */

    /* ---- (1) slice reader vs whole-tensor reference, three indices ---- */
    int64_t idxs[3] = { 0, NE/2, NE-1 };
    for (int k = 0; k < 3; k++) {
        int64_t e = idxs[k];
        coli_gguf_slice sl;
        int ok = coli_gguf_tensor_slice(g, nm, e, NE, &sl);
        CHECK(ok, "coli_gguf_tensor_slice resolves expert %lld of %lld", (long long)e, (long long)NE);
        if (!ok) continue;
        CHECK(sl.nbytes == part_bytes, "expert %lld slice size == whole/n_expert (%lld == %lld)",
              (long long)e, (long long)sl.nbytes, (long long)part_bytes);
        unsigned char *buf = (unsigned char*)malloc((size_t)sl.nbytes);
        CHECK(coli_gguf_slice_pread(&sl, buf), "coli_gguf_slice_pread succeeds for expert %lld", (long long)e);
        const unsigned char *ref = (const unsigned char*)whole + (size_t)e*(size_t)part_bytes;
        CHECK(memcmp(buf, ref, (size_t)part_bytes) == 0,
              "expert %lld slice is BIT-EXACT with the whole-tensor reference", (long long)e);

        if (e == idxs[2]) {
            /* NEGATIVE CONTROL: perturb one byte of the slice we just proved
             * matches, then assert the SAME comparison now fails. If this
             * ever reports "ok", the check above cannot be trusted. */
            buf[sl.nbytes/2] ^= 0xFF;
            int diff = memcmp(buf, ref, (size_t)sl.nbytes) != 0;
            CHECK(diff, "NEGATIVE CONTROL: a one-byte perturbation is detected (must FAIL to be inert)");
            if (!diff) { fprintf(stderr, "CONTROL DID NOT FAIL -- the comparison proves nothing\n"); g_fail = 1; }

            /* Oracle case 1(c), 2026-09-17 brief: COLI_BREAK_ESTORE, the
             * store's OWN end-to-end negative control (expert_store.cpp),
             * must also disagree with this same bit-exact reference -- a
             * fresh child process, since the flag is a getenv() cached once
             * per process the first time anything calls coli_estore_get,
             * long since latched to "off" in this (the parent) process. */
            char cmd[1024];
            snprintf(cmd, sizeof cmd,
                     "COLI_BREAK_ESTORE=1 \"%s\" --negctl-child \"%s\" \"%s\" %lld %lld",
                     argv[0], path, nm, (long long)e, (long long)NE);
            FILE *cp = popen(cmd, "r");
            CHECK(cp != NULL, "COLI_BREAK_ESTORE child process launches");
            if (cp) {
                unsigned char *cbuf = (unsigned char*)malloc((size_t)sl.nbytes);
                size_t got = fread(cbuf, 1, (size_t)sl.nbytes, cp);
                pclose(cp);
                CHECK(got == (size_t)sl.nbytes, "COLI_BREAK_ESTORE child returned %lld bytes (got %zu)",
                      (long long)sl.nbytes, got);
                int cdiff = got != (size_t)sl.nbytes || memcmp(cbuf, ref, (size_t)sl.nbytes) != 0;
                CHECK(cdiff, "NEGATIVE CONTROL: COLI_BREAK_ESTORE=1 disagrees with the bit-exact reference (must FAIL to be inert)");
                if (!cdiff) fprintf(stderr, "CONTROL DID NOT FAIL -- COLI_BREAK_ESTORE proves nothing\n");
                free(cbuf);
            }
        }
        free(buf);
    }

    /* ---- bounds violations the slice reader must reject ---- */
    coli_gguf_slice sl;
    CHECK(!coli_gguf_tensor_slice(g, nm, NE, NE, &sl), "expert index == n_expert (out of range) is REJECTED");
    CHECK(!coli_gguf_tensor_slice(g, nm, -1, NE, &sl), "negative expert index is REJECTED");
    CHECK(!coli_gguf_tensor_slice(g, nm, 0, NE+1, &sl), "a split count the tensor size does not divide evenly is REJECTED");
    CHECK(!coli_gguf_tensor_slice(g, "blk.0.does_not_exist.weight", 0, NE, &sl), "an unknown tensor name is REJECTED");

    /* ---- (2) ColiEstore: register every expert as a disk-backed slice,
     * then force real eviction with a budget that holds only ~2.5 experts,
     * and confirm every fetch (hit or refill-after-eviction) still matches
     * the whole-tensor reference -- LRU correctness, not just "it read
     * something". Keys are just &dummy_keys[e], any stable distinct pointer
     * per expert (mirrors model.cpp using each expert's coli_w_i8* as its
     * store key). */
    int64_t budget = (int64_t)(part_bytes * 2.5);
    ColiEstore *st = coli_estore_create(budget, /*direct_pref=*/1);
    CHECK(st != NULL, "coli_estore_create succeeds with a %lld-byte budget (~2.5 experts)", (long long)budget);
    int dummy_keys[8];
    int nreg = NE < 8 ? (int)NE : 8;
    for (int e = 0; e < nreg; e++) {
        coli_gguf_slice s2;
        if (!coli_gguf_tensor_slice(g, nm, e, NE, &s2)) { CHECK(0, "register: slice %d resolves", e); continue; }
        CHECK(coli_estore_register(st, &dummy_keys[e], &s2), "register expert %d", e);
    }
    CHECK(coli_estore_register(st, &dummy_keys[0], NULL) == 0, "registering a NULL slice is REJECTED, not a crash");
    {
        coli_gguf_slice s0;
        coli_gguf_tensor_slice(g, nm, 0, NE, &s0);
        CHECK(coli_estore_register(st, &dummy_keys[0], &s0) == 0, "re-registering an already-registered key is REJECTED (dup)");
    }

    /* Access pattern designed to force eviction: 0,1,2,3,0,1 with a budget
     * of ~2.5 experts -- expert 0's first fetch and its LATER re-fetch (after
     * 1,2,3 have pushed it out) must both be bit-exact, i.e. the eviction+
     * refill path reproduces the same bytes as the first read, not just
     * "a" read. */
    int pattern[] = {0,1,2,3,0,1};
    int npat = (int)(sizeof pattern/sizeof pattern[0]);
    int all_match = 1;
    for (int i = 0; i < npat && pattern[i] < nreg; i++) {
        int e = pattern[i];
        const unsigned char *got = coli_estore_get(st, &dummy_keys[e]);
        if (!got) { CHECK(0, "coli_estore_get(%d) returned non-NULL", e); all_match = 0; continue; }
        const unsigned char *ref = (const unsigned char*)whole + (size_t)e*(size_t)part_bytes;
        if (memcmp(got, ref, (size_t)part_bytes) != 0) all_match = 0;
    }
    CHECK(all_match, "every fetch in the 0,1,2,3,0,1 pattern is bit-exact (proves eviction+refill correctness)");

    ColiEstoreStats stats; coli_estore_stats(st, &stats);
    fprintf(stderr, "estore stats: requests=%llu hits=%llu misses=%llu bytes_read=%llu resident=%lld budget=%lld\n",
            (unsigned long long)stats.requests, (unsigned long long)stats.hits, (unsigned long long)stats.misses,
            (unsigned long long)stats.bytes_read, (long long)stats.resident_bytes, (long long)stats.budget_bytes);
    CHECK(stats.misses >= (uint64_t)4, "at least one miss per distinct expert touched (misses=%llu >= 4 distinct)",
          (unsigned long long)stats.misses);
    CHECK(stats.misses > (uint64_t)4, "the tight budget forced a re-miss on expert 0/1's second visit (misses=%llu > 4)",
          (unsigned long long)stats.misses);
    CHECK(stats.resident_bytes <= budget, "resident bytes never exceeded the budget (%lld <= %lld)",
          (long long)stats.resident_bytes, (long long)budget);

    coli_estore_destroy(st);
    coli_gguf_free_raw(whole);
    coli_gguf_close(g);

    test_gptoss_direct_and_prefetch();

    if (g_fail) { fprintf(stderr, "\n=== test_expert_store: FAIL ===\n"); return 1; }
    fprintf(stderr, "\n=== test_expert_store: PASS ===\n");
    return 0;
}
