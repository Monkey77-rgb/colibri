/* test_cache_oom -- the prefix-cache id record must not outlive its allocation.
 *
 * WHAT THIS CATCHES. cache_ids_grow() reallocates one id row per slot. If any
 * of those allocations fails, the rows that did NOT grow are still sized to the
 * OLD kv_ctx -- while kv_grow() publishes the NEW, larger kv_ctx regardless.
 * The record memcpy in coli_prefill_slot was bounded by kv_ctx, so the very next
 * prompt that fits the new context but not the old row wrote past the end of a
 * heap buffer.
 *
 * WHY IT NEEDS A BREAKER. Under Linux overcommit malloc() essentially never
 * returns NULL, so this path is unreachable by ordinary means -- which is
 * exactly why it survived review. COLI_BREAK_CACHE_GROW forces the Nth
 * allocation to fail so the path is REACHED. The test prints the reach counter;
 * if it never moves, the run proves nothing and the test says so and fails.
 *
 * THE CONTROL. Built a second time as test_cache_oom_unsafe with
 * -DCOLI_CACHE_UNSAFE_GUARD, which restores the pre-fix behaviour. Under ASan
 * that build MUST die with a heap-buffer-overflow. `make check-cache` requires
 * it to fail; a control that cannot fail would make this whole file decorative.
 */
#define _GNU_SOURCE
#include "../src/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void coli_break_cache_grow_after(int n);   /* defined in model.cpp under COLI_BREAK_CACHE_GROW; C++ linkage, like the rest of this header */

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,"usage: %s <model.gguf>\n", argv[0]); return 2; }
    char err[512] = {0};
    /* Small starting ctx so a grow is guaranteed; 2 slots so there IS a row
     * after the failing one -- with a single slot the bug cannot show. */
    /* max_ctx 2048 => initial kv_ctx 512 (model.cpp:514), so a 800-token prompt
     * forces exactly one grow to 1024. max_ctx must stay ABOVE the prompt or
     * kv_grow refuses at model.cpp:191 and the path is never reached. */
    coli_model *m = coli_load(argv[1], /*max_ctx*/2048, /*slots*/2, /*wq_int8*/1, 0, err, sizeof err);
    if (!m) { fprintf(stderr,"load failed: %s\n", err); return 2; }

    const int cap0 = m->cache_cap;
    printf("start: kv_ctx=%d cache_cap=%d slots=%d\n", m->kv_ctx, cap0, m->n_slots);
    if (cap0 <= 0) { printf("FAIL: cache not allocated, nothing under test\n"); coli_free(m); return 1; }

    /* Fail the FIRST id allocation of the next grow. Slot 0 then keeps its old
     * row; without the fix, slot 0 is the overflow target. */
    coli_break_cache_grow_after(1);

    /* A prompt long enough to force kv_grow past the 128-token start. */
    int S = 800;
    int *ids = (int*)malloc(sizeof(int)*S);
    for (int i=0;i<S;i++) ids[i] = (i % 1000) + 1;

    float *lg = coli_prefill_slot(m, 0, ids, S);
    printf("after grow-failure prefill: kv_ctx=%d cache_cap=%d cache_len[0]=%d\n",
           m->kv_ctx, m->cache_cap, m->cache_len ? m->cache_len[0] : -1);

    int fail = 0;
    if (m->kv_ctx <= cap0) {
        printf("FAIL: kv_ctx never grew past %d -- the OOM path was NOT reached, "
               "this run proves nothing\n", cap0);
        fail = 1;                       /* reach not demonstrated => not a pass */
    } else {
        printf("reached: kv_ctx %d -> %d with a forced allocation failure\n", cap0, m->kv_ctx);
        /* The record must NOT claim a prefix it cannot describe. */
        if (m->cache_cap > 0 && m->cache_cap < m->kv_ctx) {
            printf("FAIL: cache_cap=%d < kv_ctx=%d -- a row is narrower than the bound\n",
                   m->cache_cap, m->kv_ctx);
            fail = 1;
        }
        if (m->cache_len && m->cache_len[0] > m->cache_cap) {
            printf("FAIL: cache_len[0]=%d exceeds cache_cap=%d -- claims uncached KV\n",
                   m->cache_len[0], m->cache_cap);
            fail = 1;
        }
    }
    if (!lg) { printf("FAIL: prefill returned NULL after a cache-only failure\n"); fail = 1; }

    /* Second prefill on the same slot: with the cache dropped this must simply
     * recompute, not match against a freed or short row. */
    free(lg);
    lg = coli_prefill_slot(m, 0, ids, S);
    if (!lg) { printf("FAIL: second prefill returned NULL\n"); fail = 1; }
    free(lg); free(ids);

    printf("RESULT: %s\n", fail ? "FAIL" : "PASS");
    coli_free(m);
    return fail;
}
