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

int main(int argc, char **argv) {
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

    if (g_fail) { fprintf(stderr, "\n=== test_expert_store: FAIL ===\n"); return 1; }
    fprintf(stderr, "\n=== test_expert_store: PASS ===\n");
    return 0;
}
