/* moe_bucket_hist.h -- per-expert row-bucket size histogram for MoE prefill
 * (COLI_MOE_BUCKET_HIST=1, 2026-09-16). Pure bookkeeping over integers, no
 * coli/model dependency, so tests/test_bucket_hist.c can exercise it without
 * a model or a GPU -- same split as moe_slot_cache.h (policy here, IO and
 * compute in model.cpp).
 *
 * WHY: hybrid (GPU-resident + CPU) prefill spends 62-70% of MoE time in the
 * non-resident CPU experts even after the panel int4 GEMM speedup (see
 * README 2026-09-16). The next lever under consideration is batching each
 * expert's row group ("bucket") onto the GPU with a ragged shape, but whether
 * that pays off depends on the bucket-size DISTRIBUTION, which nobody had
 * measured. This module accumulates that distribution; it does not change
 * any computation.
 *
 * model.cpp's moe_ffn() calls, once per (layer, expert) bucket during an S>1
 * (prefill) call only -- decode (S==1) is a different shape and out of scope
 * here:
 *   record_call()      once per moe_ffn call, with the number of experts that
 *                       received zero routed rows this layer.
 *   record_size(n)      once per non-empty bucket, with its row count n.
 *   record_residency(n,r) once per non-empty bucket, r = 1 GPU-resident path,
 *                       0 CPU path, -1 not known at the hook (the ungrouped
 *                       debug path never resolves residency; reported as
 *                       "unknown", not guessed).
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define COLI_MBH_NBINS 8

struct ColiBucketHist {
    long n_calls = 0;
    long n_pairs = 0;
    long bin_buckets[COLI_MBH_NBINS]     = {0};
    long bin_rows[COLI_MBH_NBINS]        = {0};
    long bin_buckets_gpu[COLI_MBH_NBINS] = {0};
    long bin_rows_gpu[COLI_MBH_NBINS]    = {0};
    long bin_buckets_cpu[COLI_MBH_NBINS] = {0};
    long bin_rows_cpu[COLI_MBH_NBINS]    = {0};
    long residency_unknown_buckets = 0, residency_unknown_rows = 0;
    long zero_expert_sum = 0;        /* summed over calls; /n_calls == mean per layer */
    int  min_bucket = -1, max_bucket = 0;
    std::vector<int> sizes;          /* every non-zero bucket size seen, for the median */

    static int bin(int n) {
        if (n <= 4)   return 0;
        if (n <= 8)   return 1;
        if (n <= 16)  return 2;
        if (n <= 32)  return 3;
        if (n <= 64)  return 4;
        if (n <= 128) return 5;
        if (n <= 256) return 6;
        return 7;
    }
    static const char* bin_label(int b) {
        static const char *L[COLI_MBH_NBINS] =
            {"1-4","5-8","9-16","17-32","33-64","65-128","129-256","257+"};
        return L[b];
    }

    void record_call(int zero_experts) { n_calls++; zero_expert_sum += zero_experts; }

    void record_size(int n) {
        if (n <= 0) return;
        int b = bin(n);
        bin_buckets[b]++; bin_rows[b] += n; n_pairs += n;
        if (min_bucket < 0 || n < min_bucket) min_bucket = n;
        if (n > max_bucket) max_bucket = n;
        sizes.push_back(n);
    }

    /* resident: 1 = GPU-resident path, 0 = CPU path, -1 = unknown at the hook. */
    void record_residency(int n, int resident) {
        if (n <= 0) return;
        int b = bin(n);
        if (resident > 0)       { bin_buckets_gpu[b]++; bin_rows_gpu[b] += n; }
        else if (resident == 0) { bin_buckets_cpu[b]++; bin_rows_cpu[b] += n; }
        else                     { residency_unknown_buckets++; residency_unknown_rows += n; }
    }

    /* Median over every bucket size recorded. O(n log n); at a few thousand
     * (layer, expert) buckets per prefill call this is negligible next to the
     * GEMMs it is reporting on. */
    double median() const {
        if (sizes.empty()) return 0.0;
        std::vector<int> s = sizes;
        std::sort(s.begin(), s.end());
        size_t n = s.size();
        return (n & 1) ? (double)s[n/2] : 0.5*(s[n/2-1] + s[n/2]);
    }

    void dump(FILE *f) const {
        fprintf(f, "\n--- MoE bucket-size histogram (COLI_MOE_BUCKET_HIST, prefill S>1 only) ---\n");
        if (n_calls <= 0) {
            fprintf(f, "  no prefill (S>1) moe_ffn calls counted\n");
            return;
        }
        fprintf(f, "  %ld moe_ffn calls counted, %ld (token,expert) pairs total\n", n_calls, n_pairs);
        fprintf(f, "  bucket size: min %d  median %.1f  max %d\n", min_bucket, median(), max_bucket);
        fprintf(f, "  experts with zero routed rows: %.2f mean per layer (%ld total over %ld calls)\n",
                n_calls ? (double)zero_expert_sum / (double)n_calls : 0.0, zero_expert_sum, n_calls);
        fprintf(f, "  %-8s %10s %12s\n", "bin", "buckets", "row share");
        for (int b = 0; b < COLI_MBH_NBINS; b++) {
            double share = n_pairs > 0 ? 100.0 * bin_rows[b] / (double)n_pairs : 0.0;
            fprintf(f, "    %-8s %8ld  %6.2f%% of rows  (%ld rows)\n",
                    bin_label(b), bin_buckets[b], share, bin_rows[b]);
        }
        if (residency_unknown_rows == n_pairs) {
            fprintf(f, "  GPU/CPU split: NOT AVAILABLE at this hook -- every counted bucket went\n"
                       "                 through the ungrouped debug path (COLI_MOE_UNGROUPED=1),\n"
                       "                 which never resolves GPU vs CPU residency here.\n");
            return;
        }
        fprintf(f, "  GPU/CPU split by bucket-size bin (rows):\n");
        for (int b = 0; b < COLI_MBH_NBINS; b++) {
            long tot = bin_rows_gpu[b] + bin_rows_cpu[b];
            if (!tot) continue;
            double gshare = 100.0 * bin_rows_gpu[b] / (double)tot;
            fprintf(f, "    %-8s gpu %8ld rows (%5.1f%%)  cpu %8ld rows (%5.1f%%)\n",
                    bin_label(b), bin_rows_gpu[b], gshare, bin_rows_cpu[b], 100.0 - gshare);
        }
        if (residency_unknown_rows) {
            fprintf(f, "  NOTE: %ld of %ld rows (%.1f%%) had unknown residency at this hook (the\n"
                       "        ungrouped debug path) -- excluded from the split above, not guessed.\n",
                    residency_unknown_rows, n_pairs, 100.0 * residency_unknown_rows / (double)n_pairs);
        }
    }
};
