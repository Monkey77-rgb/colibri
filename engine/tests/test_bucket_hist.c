/* test_bucket_hist -- MoE per-expert row-bucket size histogram
 * (src/moe_bucket_hist.h), no GPU, no model. Feeds a known synthetic
 * assignment through the accumulation/print helper and checks the binned
 * counts and row shares against hand-computed values. Same shape as
 * tests/test_slot_cache.c: every property has a control, and the apparatus
 * control at the end proves a wrong expectation actually fails the test. */
#include <stdio.h>
#include "../src/moe_bucket_hist.h"
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    ColiBucketHist h;

    /* Two synthetic moe_ffn(S>1) calls ("layers"), NE=10 experts each.
     *
     * Layer A: 6 of 10 experts routed, bucket sizes {1,4,5,8,9,300}, 4 zero.
     * Layer B: 4 of 10 experts routed, bucket sizes {2,16,64,128}, 6 zero.
     *
     * Hand-computed bins (1-4,5-8,9-16,17-32,33-64,65-128,129-256,257+):
     *   bin0 1-4:    {1,4,2}      -> 3 buckets, 7 rows
     *   bin1 5-8:    {5,8}        -> 2 buckets, 13 rows
     *   bin2 9-16:   {9,16}       -> 2 buckets, 25 rows
     *   bin3 17-32:  {}           -> 0 buckets, 0 rows
     *   bin4 33-64:  {64}         -> 1 bucket, 64 rows
     *   bin5 65-128: {128}        -> 1 bucket, 128 rows
     *   bin6 129-256:{}           -> 0 buckets, 0 rows
     *   bin7 257+:   {300}        -> 1 bucket, 300 rows
     * total buckets = 10, total rows (pairs) = 1+4+5+8+9+300+2+16+64+128 = 537
     * sorted sizes: 1,2,4,5,8,9,16,64,128,300 -> median = (8+9)/2 = 8.5
     * zero experts: 4 + 6 = 10 over 2 calls -> mean 5.0
     *
     * Residency assigned per size to also exercise the GPU/CPU split, plus
     * ONE deliberately unknown (the ungrouped-debug-path case):
     *   1->cpu 4->gpu 5->cpu 8->gpu 9->cpu 300->gpu  (layer A)
     *   2->unknown 16->cpu 64->gpu 128->cpu           (layer B)
     * -> gpu rows = 4+8+64+300 = 376, cpu rows = 1+5+9+16+128 = 159,
     *    unknown rows = 2 (1 bucket). 376+159+2 = 537 == n_pairs.
     */
    int szA[6]  = {1,4,5,8,9,300};
    int resA[6] = {0,1,0,1,0,1};   /* 0=cpu 1=gpu */
    h.record_call(4);
    for (int i=0;i<6;i++) { h.record_size(szA[i]); h.record_residency(szA[i], resA[i]); }

    int szB[4]  = {2,16,64,128};
    int resB[4] = {-1,0,1,0};      /* -1=unknown */
    h.record_call(6);
    for (int i=0;i<4;i++) { h.record_size(szB[i]); h.record_residency(szB[i], resB[i]); }

    CHECK(h.n_calls == 2, "n_calls: got %ld want 2", h.n_calls);
    CHECK(h.n_pairs == 537, "n_pairs: got %ld want 537", h.n_pairs);
    CHECK(h.min_bucket == 1 && h.max_bucket == 300, "min/max: got %d/%d want 1/300", h.min_bucket, h.max_bucket);
    CHECK(h.median() == 8.5, "median: got %.3f want 8.5", h.median());
    CHECK(h.zero_expert_sum == 10, "zero_expert_sum: got %ld want 10", h.zero_expert_sum);

    long want_buckets[COLI_MBH_NBINS] = {3,2,2,0,1,1,0,1};
    long want_rows[COLI_MBH_NBINS]    = {7,13,25,0,64,128,0,300};
    for (int b=0;b<COLI_MBH_NBINS;b++) {
        CHECK(h.bin_buckets[b] == want_buckets[b], "bin_buckets[%d] (%s): got %ld want %ld",
              b, ColiBucketHist::bin_label(b), h.bin_buckets[b], want_buckets[b]);
        CHECK(h.bin_rows[b] == want_rows[b], "bin_rows[%d] (%s): got %ld want %ld",
              b, ColiBucketHist::bin_label(b), h.bin_rows[b], want_rows[b]);
    }

    long want_gpu[COLI_MBH_NBINS] = {4,8,0,0,64,0,0,300};
    long want_cpu[COLI_MBH_NBINS] = {1,5,25,0,0,128,0,0};
    for (int b=0;b<COLI_MBH_NBINS;b++) {
        CHECK(h.bin_rows_gpu[b] == want_gpu[b], "bin_rows_gpu[%d] (%s): got %ld want %ld",
              b, ColiBucketHist::bin_label(b), h.bin_rows_gpu[b], want_gpu[b]);
        CHECK(h.bin_rows_cpu[b] == want_cpu[b], "bin_rows_cpu[%d] (%s): got %ld want %ld",
              b, ColiBucketHist::bin_label(b), h.bin_rows_cpu[b], want_cpu[b]);
    }
    CHECK(h.residency_unknown_buckets == 1 && h.residency_unknown_rows == 2,
          "unknown residency: got %ld buckets / %ld rows, want 1 / 2",
          h.residency_unknown_buckets, h.residency_unknown_rows);

    /* bin() boundaries themselves, both sides of each edge */
    CHECK(ColiBucketHist::bin(4)==0 && ColiBucketHist::bin(5)==1, "bin edge 4/5");
    CHECK(ColiBucketHist::bin(256)==6 && ColiBucketHist::bin(257)==7, "bin edge 256/257");

    /* n<=0 must be a no-op (never invented by a real caller, but the guard is
     * part of the contract: a stray zero must not appear as a phantom bucket) */
    { ColiBucketHist z; z.record_size(0); z.record_residency(0,1); z.record_size(-1);
      CHECK(z.n_pairs == 0 && z.sizes.empty(), "n<=0 must be ignored, not recorded"); }

    /* dump() must run without crashing on both an empty and a populated histogram
     * (no assertion on text -- the numbers above are already checked structurally) */
    { ColiBucketHist e; e.dump(stdout); }
    h.dump(stdout);

    /* apparatus control: a deliberately wrong expectation must be CAUGHT */
    int before = fails;
    CHECK(h.bin_buckets[0] == 999, "(expected failure) bin0 buckets == 999");
    int caught = fails > before;
    fails = before;

    printf("%s (apparatus control %s)\n", fails ? "FAIL" : "PASS", caught ? "can fail" : "INERT");
    return (fails == 0 && caught) ? 0 : 1;
}
