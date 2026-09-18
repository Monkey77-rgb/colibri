/* expert_store.h — a disk-resident, LRU-bounded cache of Q4_K expert blocks,
 * for MoE models whose expert bytes exceed available RAM (phase 1c of the
 * "any model" program; see loader.h's coli_gguf_tensor_slice, the primitive
 * this is built on).
 *
 * WHY THIS EXISTS. model.cpp's native-Q4_K expert loader (try_native_q4k /
 * the ffn_gate_exps.weight path in coli_load, see the struct Q4KSide comment)
 * reads a layer's WHOLE expert tensor into one malloc at load time and hands
 * out borrowed slices, one per expert -- correct and fast, but it requires
 * every expert byte in RAM before the first token. On a model whose experts
 * do not fit (e.g. Qwen3-235B-A22B on this box's ~20 GiB available RAM),
 * that load simply fails. This file replaces "read at load time, keep
 * forever" with "read on first use, evict the coldest resident expert to
 * stay under a byte budget" -- a plain LRU over (fixed-size) expert slices,
 * same idea as c/expert_store.h's ColiExpertStore for the OLD engine family,
 * reimplemented here against THIS engine's coli_w_q4k / coli_gguf_slice
 * primitives rather than ColiTensorView (different engine, same repo -- see
 * that header's own comment for the K3_EXPERT_GB-style design this borrows).
 *
 * THREADING / LIFETIME CONTRACT -- read this before calling coli_estore_get
 * from anywhere new. model.cpp's MoE forward visits experts of a layer ONE
 * AT A TIME through mm_a() -> q4k_find() (see model.cpp: the grouped/fused
 * multi-expert paths all require an int4 twin via w4_slot(), which a native-
 * Q4_K expert never has, so they decline and fall through to the serial
 * per-expert loop) -- there is never a second live coli_estore_get() call in
 * flight while a previous one's returned pointer is still being read by a
 * coli_gemm_q4k call. That is what makes "the returned pointer is valid
 * until some LATER coli_estore_get call evicts it" a safe contract here
 * instead of a use-after-free waiting to happen: nothing coli_estore_get's
 * own callers do interleaves two experts' resident lifetimes.
 *
 * PINNING / coli_estore_prefetch (change B, 2026-09-17 brief) is the "gathers
 * multiple experts' pointers BEFORE consuming any of them" case the previous
 * revision of this comment flagged as unsupported -- model.cpp calls it once
 * per MoE layer with that layer's selected (or, in prefill, batched) 3*K
 * keys, so all of them are resident and safe to read for the rest of the
 * layer even though nothing consumes them yet. Internally it: (1) unpins
 * whatever the PREVIOUS coli_estore_prefetch call pinned (or does nothing on
 * the first call) -- pins are a rolling one-batch-deep set, not stacked; (2)
 * resolves every not-yet-resident key's fd(s) and allocates its buffer on
 * the CALLING thread (map lookups, fd-cache lookups/inserts and mallocs are
 * NOT taken concurrently -- only the pread() itself is farmed out); (3) runs
 * the reads on a persistent pool of COLI_ESTORE_THREADS worker threads (0 or
 * 1: runs them inline instead, i.e. today's serial behaviour) that touch
 * nothing but the buffer they were handed and their own fd; (4) after every
 * worker has finished (a hard join, not fire-and-forget), the calling thread
 * -- and only the calling thread -- inserts the new entries into the map,
 * updates the LRU list and marks every key in the batch (resident already or
 * just filled) PINNED, so evict_one/evict_one_if skip them. A key can still
 * be read through the ordinary coli_estore_get() at any point while pinned
 * (a hit, since it is resident) -- pinning only ever prevents EVICTION, it
 * is not a lock against concurrent get() calls, and the existing "never two
 * live coli_estore_get() calls in flight" invariant above is unchanged and
 * still load-bearing for callers that do not use prefetch. A caller that
 * still wants a NEW kind of multi-expert-pointers-at-once access this does
 * not cover (e.g. concurrent GETS, not just concurrent disk reads) needs
 * more than pinning -- do not assume this contract extends that far.
 *
 * BUDGET. budget_bytes <= 0 means unbounded: every expert becomes resident
 * on its first use and is never evicted, i.e. COLI_EXPERT_STORE=1 with no
 * COLI_EXPERT_GB behaves like the old eager loader except that the read is
 * deferred to first use instead of happening for every expert at load time
 * (a strict improvement to cold-start time when not every expert is ever
 * selected in the run, and unbounded RAM growth is the tradeoff already
 * implied by "no budget"). budget_bytes > 0 evicts least-recently-used
 * resident experts (whole slices, not partial) to stay at or under it,
 * NEVER evicting the key currently being filled.
 *
 * O_DIRECT. direct_pref=1 opens a second, O_DIRECT fd per shard path (the
 * shard's own buffered fd, opened by coli_gguf_open, is never reused for
 * O_DIRECT -- mixing buffered and O_DIRECT on one fd is undefined territory
 * on Linux). ALIGNED-SUPERSET READ (change A, 2026-09-17): a slice's own
 * (off, nbytes) is essentially never already a multiple of COLI_ESTORE_ALIGN
 * (4096) itself -- Q4_K/MXFP4 block sizes mean a per-expert byte count is a
 * multiple of 4096 only for specific (I,O) shapes, so requiring the slice
 * itself to be pre-aligned (the previous behaviour here) meant O_DIRECT
 * never engaged for gpt-oss at all. Instead the store computes the smallest
 * 4096-aligned range that CONTAINS the slice (off0 = off & ~4095, len =
 * ((off+nbytes+4095) & ~4095) - off0 -- both aligned by construction
 * regardless of the slice's own alignment), reads THAT into a
 * posix_memalign(4096, len) buffer, and hands callers a pointer offset into
 * it (off-off0) to the slice's real first byte. The full aligned buffer,
 * not just the slice's nbytes, is what counts against the budget and what
 * evict_one/coli_estore_drop free() -- see Entry::buf_bytes/buf_off in the
 * .cpp. When the O_DIRECT open itself fails (filesystem/kernel support) or a
 * read comes up short, every read falls back to a plain buffered pread of
 * exactly the slice's own bytes on the shard's existing fd -- silently, not
 * an error, exactly as the task's own COLI_EXPERT_DIRECT spec requires. */
#ifndef COLI_EXPERT_STORE_H
#define COLI_EXPERT_STORE_H

#include <stdint.h>
#include "loader.h"   /* coli_gguf_slice */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiEstore ColiEstore;

ColiEstore *coli_estore_create(int64_t budget_bytes, int direct_pref);
void        coli_estore_destroy(ColiEstore *st);

/* Registers one expert's on-disk slice under `key` -- any stable pointer the
 * caller already uses to name this expert's matrix (model.cpp reuses the
 * same coli_w_i8* it keys g_q4ktab/g_w4tab with; this is not a new key
 * space). Reads nothing. Returns 1 on success, 0 on a duplicate key or an
 * invalid slice (nbytes<=0). */
int coli_estore_register(ColiEstore *st, const void *key, const coli_gguf_slice *slice);

/* Returns a pointer to `key`'s resident Q4_K bytes (coli_gguf_slice.nbytes
 * long), reading from disk on a miss. Evicts the store's own
 * least-recently-used OTHER resident expert(s) as needed to fit the budget;
 * never evicts `key` itself mid-fill. Returns NULL if `key` was never
 * registered (coli_estore_register was not called for it) or the disk read
 * failed -- a caller must treat NULL as fatal (a registered expert that
 * cannot be read is a corrupt/truncated model, not a normal "miss"). See the
 * file comment above for the pointer-lifetime contract. */
const uint8_t *coli_estore_get(ColiEstore *st, const void *key);
/* Release a resident expert's RAM without unregistering it (2026-09-14): the
 * GPU slot cache calls this after it has uploaded the bytes, so the store's
 * budget holds OTHER experts -- RAM and VRAM become two exclusive tiers instead
 * of the VRAM tier duplicating the hottest part of the RAM tier. The next
 * coli_estore_get() for the key simply re-reads it from disk. Returns 1 if a
 * buffer was freed, 0 if it was not resident or not registered. */
int coli_estore_drop(ColiEstore *st, const void *key);
/* 1 if the key's bytes are in RAM right now (no IO, no LRU touch). */
int coli_estore_resident(const ColiEstore *st, const void *key);

/* Change B (2026-09-17 brief) -- batched, concurrent fill for the `n` keys
 * in `keys` (duplicates and unregistered/NULL entries are silently
 * tolerated, not an error). Unpins whatever the PREVIOUS call to this
 * function pinned, fills every key in this batch that is not already
 * resident (concurrently, across a pool of COLI_ESTORE_THREADS threads --
 * default 4; 0 or 1 means serial, i.e. coli_estore_get's own behaviour one
 * key at a time), evicting other, non-pinned entries as needed but NEVER a
 * key that appears in THIS batch, then marks every key in the batch (newly
 * filled or already resident) PINNED so evict_one/evict_one_if skip them
 * until the NEXT coli_estore_prefetch call (or an explicit
 * coli_estore_unpin_all). See the file comment's PINNING section for the
 * full contract, in particular that a key can still be read with the
 * ordinary coli_estore_get() while pinned. Returns 0 for invalid arguments
 * (st/keys NULL, n<=0), 1 otherwise -- a per-key disk-read failure inside
 * the batch is logged to stderr and leaves that one key unresident (a
 * following coli_estore_get() for it will retry the read and report the
 * usual fatal error if it still fails); this function itself still returns
 * 1 in that case, matching coli_estore_get's own "NULL/failure is the
 * caller's problem to detect" convention rather than aborting a whole
 * layer's prefetch over one bad key. */
int coli_estore_prefetch(ColiEstore *st, const void *const *keys, int n);
/* Unpins every key currently pinned (by the most recent coli_estore_prefetch
 * call). A no-op if nothing is pinned. Does not evict anything itself --
 * unpinning only makes an entry ELIGIBLE for eviction again. */
void coli_estore_unpin_all(ColiEstore *st);

/* STICKY pin (COLI_MOE_RESID residual-hot policy, 2026-09-18): separate from
 * the rolling per-batch `pinned` flag coli_estore_prefetch/coli_estore_unpin_all
 * manage above -- a sticky key is NEVER unpinned by coli_estore_unpin_all, only
 * by an explicit coli_estore_unpin_sticky (never called in this build; it
 * exists so a test can undo one). Fetches the key first if not already
 * resident (a no-op read-through if it already is); returns 0 (and pins
 * nothing) for an unregistered key, a NULL store, or a fetch that fails.
 * Being sticky-pinned and being pinned by prefetch are independent: an
 * expert can be both, and losing the rolling pin at the next batch's
 * coli_estore_unpin_all never affects its sticky status. */
int coli_estore_pin_sticky(ColiEstore *st, const void *key);
void coli_estore_unpin_sticky(ColiEstore *st, const void *key);
/* Test-only, same rationale as coli_estore_test_pinned. */
int coli_estore_test_sticky(const ColiEstore *st, const void *key);

typedef struct {
    uint64_t requests, hits, misses;
    uint64_t bytes_read;        /* total bytes actually pread from disk across all fills */
    uint64_t direct_reads;      /* of which, via O_DIRECT (subset of fills, not of bytes_read) */
    uint64_t buffered_reads;
    uint64_t prefetch_fills;    /* change B2 (2026-09-17 brief 2, item 3): fills performed
                                  * by coli_estore_prefetch, counted HERE instead of in
                                  * requests/misses above. A key that coli_estore_prefetch
                                  * fills and a LATER coli_estore_get then finds resident
                                  * is exactly one real disk fill (this counter) and one
                                  * ordinary hit (requests/hits) -- not a miss-then-hit
                                  * pair, which would inflate `requests` with a bookkeeping
                                  * artefact of an access nothing ever repeated. requests/
                                  * hits/misses this way stay 1:1 with actual
                                  * coli_estore_get() calls from the compute path, so
                                  * hit-rate is comparable across the OFF/A2/A2+B2 arms
                                  * regardless of whether prefetch ran ahead of them. */
    int64_t  resident_bytes;    /* current sum of resident slice sizes */
    int64_t  budget_bytes;      /* <=0 means unbounded, as passed to create() */
} ColiEstoreStats;
void coli_estore_stats(const ColiEstore *st, ColiEstoreStats *out);
/* Test-only: exposes whether `key` is currently pinned (no such accessor was
 * needed before coli_estore_prefetch existed, and it is not part of the
 * contract model.cpp depends on -- do not add new production callers
 * without re-checking whether they should be reading ColiEstoreStats
 * instead). Returns 0 for an unregistered or NULL key/store too. */
int coli_estore_test_pinned(const ColiEstore *st, const void *key);

#ifdef __cplusplus
}
#endif
#endif
