/* expert_store.cpp — see expert_store.h for the contract and design. C++ so
 * this can use std::unordered_map/std::list the same way model.cpp's own
 * g_w4tab/g_q4ktab side tables do, exposed through the plain C ABI
 * expert_store.h declares (extern "C", same convention as coli_api.cpp). */
#define _GNU_SOURCE
#include "expert_store.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/stat.h>

#ifndef O_DIRECT
#define O_DIRECT 0   /* platforms without it: direct_pref silently never engages */
#endif

#define COLI_ESTORE_ALIGN 4096

namespace {

struct Entry {
    coli_gguf_slice slice;
    uint8_t        *buf = nullptr;   /* nullptr == not resident; base of the
                                       * allocation (may be an aligned superset
                                       * larger than slice.nbytes -- see buf_off) */
    int64_t         buf_bytes = 0;   /* malloc()'d size of buf, i.e. what
                                       * budget/resident_bytes accounts and what
                                       * evict_one/drop free()s -- NOT always
                                       * slice.nbytes (change A: an O_DIRECT
                                       * aligned-superset read allocates and
                                       * counts the aligned length, per the
                                       * 2026-09-17 brief) */
    int64_t         buf_off = 0;     /* add to buf to get the slice's first
                                       * byte; nonzero only when an O_DIRECT
                                       * read's aligned start preceded the
                                       * slice's real (possibly unaligned) off */
    std::list<const void*>::iterator lru_it;
    bool            in_lru = false;
    bool            pinned = false;  /* change B: set by coli_estore_prefetch,
                                       * evict_one must never pick this entry
                                       * while set; cleared by the next
                                       * coli_estore_prefetch call or an
                                       * explicit coli_estore_unpin_all */
    bool            sticky = false;  /* COLI_MOE_RESID (2026-09-18): set by
                                       * coli_estore_pin_sticky, independent of
                                       * `pinned` and never cleared by
                                       * coli_estore_unpin_all -- see
                                       * expert_store.h. */
};

/* Change B: a persistent fork-join pool of worker threads that ONLY pread()
 * into buffers the caller (coli_estore_prefetch) already allocated -- see
 * expert_store.h's updated threading contract. Workers never touch
 * ColiEstore's map, LRU list or fd caches; those are resolved on the caller
 * thread before a batch is dispatched and mutated on the caller thread again
 * after every worker has finished (the join below), so no lock is needed on
 * any of that state -- only this pool's own generation counter is
 * synchronized. Persistent (built once, reused every layer) rather than
 * spawned per call: over a 36-layer x tens-of-tokens run that is thousands
 * of prefetch calls, and thread creation is not free next to a handful of
 * ~ms pread()s. */
class FillPool {
public:
    void start(int nthreads) {
        for (int i = 0; i < nthreads; i++) workers.emplace_back([this]{ worker_loop(); });
    }
    ~FillPool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            gen++;
        }
        cv_go.notify_all();
        for (auto &t : workers) if (t.joinable()) t.join();
    }
    /* Runs every job in `jobvec` across the pool and returns once all have
     * completed. Safe to call with an empty pool (workers.empty()): runs the
     * jobs inline on the caller thread, i.e. today's serial behaviour --
     * COLI_ESTORE_THREADS=0 or 1 takes this path. */
    void run(std::vector<std::function<void()>> &jobvec) {
        if (workers.empty()) { for (auto &f : jobvec) f(); return; }
        std::unique_lock<std::mutex> lk(mu);
        jobs = &jobvec;
        next.store(0);
        remaining = (int)workers.size();
        gen++;
        cv_go.notify_all();
        cv_done.wait(lk, [this]{ return remaining == 0; });
        jobs = nullptr;
    }
private:
    void worker_loop() {
        int my_gen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_go.wait(lk, [&]{ return stop || gen != my_gen; });
            if (stop) return;
            my_gen = gen;
            std::vector<std::function<void()>> *j = jobs;
            lk.unlock();
            if (j) {
                size_t idx;
                while ((idx = next.fetch_add(1)) < j->size()) (*j)[idx]();
            }
            lk.lock();
            if (--remaining == 0) cv_done.notify_all();
        }
    }
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_go, cv_done;
    int gen = 0, remaining = 0;
    bool stop = false;
    std::vector<std::function<void()>> *jobs = nullptr;
    std::atomic<size_t> next{0};
};

} // namespace

struct ColiEstore {
    int64_t budget_bytes;
    int     direct_pref;
    std::unordered_map<const void*, Entry> map;
    std::list<const void*> lru;              /* front = most-recently-used */
    std::unordered_map<std::string,int> direct_fds;    /* shard path -> owned O_DIRECT fd, or -1 = tried+failed */
    std::unordered_map<std::string,int> buffered_fds;  /* shard path -> owned buffered fd, or -1 = tried+failed */
    int64_t resident_bytes = 0;
    ColiEstoreStats stat{};
    /* Change B */
    FillPool *pool = nullptr;                 /* lazily created on first prefetch, nullptr == not yet configured */
    int pool_threads = -1;                    /* -1 == COLI_ESTORE_THREADS not read yet */
    std::vector<const void*> pinned_keys;      /* currently-pinned keys, for O(1) unpin_all */
    /* Change A2 (2026-09-17 brief 2, item 1) -- per-length free-list of
     * already-faulted aligned buffers. io05 (bench_expert_store, round 1)
     * measured the WARM control (192 fills, nothing evicted or refilled from
     * disk) at 116 ms / 192 = 0.6 ms per fill with a fresh posix_memalign +
     * free every time: under mallopt(M_MMAP_THRESHOLD, 128 KiB) (see
     * coli_estore_create's comment) a ~4.4 MB allocation always comes from a
     * fresh mmap, and O_DIRECT's DMA target must be page-resident before the
     * read, so the FIRST touch of a freshly mmapped buffer faults in every
     * page of it (~1,080 4 KiB pages for gpt-oss's aligned-superset length) --
     * that fault storm, not the read itself, is the fixed cost. Returning a
     * same-length buffer to this list on eviction instead of free()'ing it,
     * and reusing it on the next fill of the same length instead of
     * allocating fresh, means the pages are already resident from the
     * buffer's PREVIOUS life: no fault storm, no new mmap. Keyed by aligned
     * length rather than one pool, because a non-gpt-oss model registered in
     * the same store (or gpt-oss's own down/gate/up if their aligned lengths
     * ever differ) must not be handed a buffer sized for a different slice.
     * Capped at budget_bytes total (see alloc_buf/release_buf) so a run with
     * COLI_EXPERT_GB set cannot grow unbounded RAM on top of the resident
     * budget it already accounts for. */
    std::unordered_map<int64_t, std::vector<uint8_t*>> free_bufs;
    int64_t freelist_bytes = 0;
};

/* alloc_buf/release_buf (change A2): the ONLY places a fill buffer is
 * allocated or given up, so every caller (coli_estore_get's direct attempt,
 * its buffered fallback, and coli_estore_prefetch's caller-thread resolve
 * loop) shares one reuse pool instead of three independent malloc/free
 * pairs. Both are caller-thread-only (matches expert_store.h's threading
 * contract: pool workers only pread into a buffer the caller already
 * allocated, never touch the map or any pool of buffers themselves). */
static uint8_t *alloc_buf(ColiEstore *st, int64_t len) {
    auto it = st->free_bufs.find(len);
    if (it != st->free_bufs.end() && !it->second.empty()) {
        uint8_t *b = it->second.back();
        it->second.pop_back();
        st->freelist_bytes -= len;
        return b;
    }
    uint8_t *b = nullptr;
    if (posix_memalign((void**)&b, COLI_ESTORE_ALIGN, (size_t)len) != 0 || !b) return nullptr;
    /* Deliberately NOT pre-faulted here (an earlier version of this function
     * did an unconditional memset -- reverted, see the commit message: it
     * measurably broke bench_expert_store's own warm-cache control by adding
     * a large anonymous-memory write on every FIRST-ever fill of a length,
     * which is exactly the fills a bench with an unbounded budget -- nothing
     * ever evicts, so the free-list above is never fed -- does on every
     * single one of its 192*5*3 gets; the resulting memory pressure evicted
     * page-cache pages an unrelated part of this filesystem's O_DIRECT path
     * was relying on, and the WARM CONTROL's disk-sector delta went from 0
     * to cold-arm-sized. A brand-new buffer's first touch is already paid
     * for by pread()'s own get_user_pages() fault-in during the read that
     * immediately follows this call -- doing it again first via memset does
     * not avoid that fault, it just moves an equal-sized one earlier and
     * ADDS the memset's own write pass on top. The real saving A2 exists for
     * is buffers that come off the free-list branch above: THEIR pages are
     * already resident from a previous life (release_buf below never
     * munmaps a buffer it keeps), so the win is "reuse, don't refault",
     * which needs nothing done here at all -- only not calling free() on
     * eviction, which release_buf already is. */
    return b;
}
static void release_buf(ColiEstore *st, uint8_t *buf, int64_t len) {
    if (!buf) return;
    int64_t cap = st->budget_bytes > 0 ? st->budget_bytes : INT64_MAX;
    if (len > 0 && st->freelist_bytes + len <= cap) {
        st->free_bufs[len].push_back(buf);
        st->freelist_bytes += len;
    } else {
        free(buf);
    }
}

ColiEstore *coli_estore_create(int64_t budget_bytes, int direct_pref) {
    /* Pin glibc's mmap threshold (2026-09-14). Every fill is a ~4.4 MB
     * posix_memalign and every eviction a free(); glibc's DYNAMIC threshold
     * rises to the size of the first freed mmapped chunk, after which fills
     * that size come off the brk heap and the holes evictions leave cannot be
     * trimmed. Measured on gpt-oss-120b (desktop, 6 GiB store, --gpu with 1656
     * expert matrices fetched at load): RSS 13.2 -> 18.5 GB and climbing over
     * 48 decode tokens with a FULL store; with the threshold pinned (this call,
     * or GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072) 9.1 GB flat, and the
     * 22 GiB-capped 10 GiB-store decode arm that was OOM-killed at 23.0 GB anon
     * fits. Pinning also disables the dynamic adjustment, so the store's
     * buffers always come from mmap and go back to the kernel on free. Process-
     * wide, deliberately: the store is one per process anyway. The per-token
     * scratch this engine mallocs is well under 128 KiB and unaffected. */
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
    ColiEstore *st = new ColiEstore();
    st->budget_bytes = budget_bytes;
    st->direct_pref = direct_pref;
    st->stat.budget_bytes = budget_bytes;
    return st;
}

void coli_estore_destroy(ColiEstore *st) {
    if (!st) return;
    for (auto &kv : st->map) if (kv.second.buf) free(kv.second.buf);
    for (auto &kv : st->free_bufs) for (auto b : kv.second) free(b);   /* change A2: free-list */
    for (auto &kv : st->direct_fds) if (kv.second >= 0) close(kv.second);
    for (auto &kv : st->buffered_fds) if (kv.second >= 0) close(kv.second);
    delete st->pool;
    delete st;
}

int coli_estore_register(ColiEstore *st, const void *key, const coli_gguf_slice *slice) {
    if (!st || !key || !slice || slice->nbytes <= 0) return 0;
    if (st->map.find(key) != st->map.end()) return 0;   /* duplicate key */
    Entry e;
    e.slice = *slice;
    st->map.emplace(key, e);
    return 1;
}

/* Opens (and caches) an O_DIRECT fd for `path`, or returns -1 if one is not
 * available -- cached either way so a path that fails O_DIRECT is not
 * retried on every subsequent miss.
 *
 * DELIBERATELY NEVER uses coli_gguf_slice.fd for anything beyond the moment
 * of registration. That fd is BORROWED from the coli_gguf the caller had
 * open at registration time (loader.c's coli_gguf_open/coli_gguf_close), and
 * model.cpp's coli_load() closes its coli_gguf (via the GgufOwner RAII
 * wrapper) before returning -- long before a lazily-filled expert's first
 * coli_estore_get() call during the forward pass. Reading through
 * e.slice.fd at that point is a read on a closed fd (measured: coli reported
 * "expert_store: disk read failed" on every registered expert, 2026-09-13,
 * first end-to-end run -- this comment records the defect the fix below is
 * for). The store opens and owns ITS OWN fd per shard path instead, entirely
 * independent of whatever coli_gguf the caller used to resolve the slice. */
static int owned_fd_for(ColiEstore *st, const char *path, bool direct) {
    auto &cache = direct ? st->direct_fds : st->buffered_fds;
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    int flags = O_RDONLY | (direct ? O_DIRECT : 0);
    int fd = (direct && !O_DIRECT) ? -1 : open(path, flags);
    cache.emplace(path, fd);
    return fd;
}

/* Change A (2026-09-17 brief): aligned-superset O_DIRECT read. A gpt-oss
 * expert slice's (off, nbytes) is essentially never itself 4096-aligned
 * (measured: 4,406,400 % 4096 = 3200), so requiring the SLICE to already be
 * aligned -- the previous behaviour here -- meant O_DIRECT never engaged at
 * all ("fills: 12759 buffered, 0 via O_DIRECT", the 09-14 a11_auto raw this
 * brief cites). Instead read the smallest 4096-aligned superset that
 * CONTAINS the slice: off0 = off & ~4095, end = (off+nbytes+4095) & ~4095,
 * len = end-off0. off0 and len are aligned by construction regardless of the
 * slice's own alignment, so this always qualifies for O_DIRECT when
 * direct_pref is set and the fd is available.
 *
 * Change A2 (2026-09-17 brief 2, item 1): no longer allocates -- `base` is a
 * buffer of exactly `alloc_len` bytes the CALLER already obtained from
 * alloc_buf() (fresh or reused, pre-faulted either way), so this function
 * only ever does the pread(). Returns *out_off = off-off0 (0 to add to get
 * the slice's first byte) and *out_len = alloc_len on success. On any
 * failure (fd unavailable, short read) frees nothing itself -- the caller
 * owns `base` before and after this call and decides (release_buf, back to
 * the reuse pool) whether to fall back to buffered_read with it or a fresh
 * allocation. */
static bool try_direct_read(ColiEstore *st, const coli_gguf_slice &slice,
                             uint8_t *base, int64_t alloc_len,
                             int64_t *out_off, int64_t *out_len) {
    if (!st->direct_pref) return false;
    int fd = owned_fd_for(st, slice.shard_path, /*direct=*/true);
    if (fd < 0) return false;
    int64_t off0 = slice.off & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
    ssize_t r = pread(fd, base, (size_t)alloc_len, (off_t)off0);
    if (r != (ssize_t)alloc_len) return false;
    *out_off = slice.off - off0; *out_len = alloc_len;
    return true;
}

/* Buffered fallback, via the store's OWN fd (see owned_fd_for's comment) --
 * NOT coli_gguf_slice_pread(&slice, buf), which would use slice.fd and
 * reintroduce the closed-fd bug above. coli_gguf_slice_pread remains correct
 * and useful for a caller whose coli_gguf is still open (this file's own
 * unit test uses it that way, legitimately); it is simply the wrong tool for
 * a cache that outlives the loader. Reads exactly slice.nbytes at slice.off
 * into buf (no alignment superset -- buffered I/O has no such requirement). */
static bool buffered_read(ColiEstore *st, const coli_gguf_slice &slice, uint8_t *buf) {
    int fd = owned_fd_for(st, slice.shard_path, /*direct=*/false);
    if (fd < 0) return false;
    ssize_t r = pread(fd, buf, (size_t)slice.nbytes, (off_t)slice.off);
    return r == (ssize_t)slice.nbytes;
}

static void lru_touch(ColiEstore *st, const void *key, Entry &e) {
    if (e.in_lru) st->lru.erase(e.lru_it);
    st->lru.push_front(key);
    e.lru_it = st->lru.begin();
    e.in_lru = true;
}

/* Evict from the back (least-recently-used) forward, skipping any entry
 * `protect` says to keep -- a PINNED entry (change B: set by
 * coli_estore_prefetch, see expert_store.h) is always protected on top of
 * whatever the caller's own predicate adds. Returns true if something was
 * freed. */
static bool evict_one_if(ColiEstore *st, const std::function<bool(const void*)> &protect) {
    for (auto rit = st->lru.rbegin(); rit != st->lru.rend(); ++rit) {
        const void *k = *rit;
        auto mit = st->map.find(k);
        if (mit == st->map.end() || !mit->second.buf) continue;
        Entry &victim = mit->second;
        if (victim.pinned || victim.sticky || protect(k)) continue;
        st->resident_bytes -= victim.buf_bytes;
        release_buf(st, victim.buf, victim.buf_bytes);   /* change A2: reuse pool, not free() */
        victim.buf = nullptr;
        victim.buf_bytes = 0;
        victim.buf_off = 0;
        st->lru.erase(victim.lru_it);
        victim.in_lru = false;
        return true;
    }
    return false;
}

static void evict_one(ColiEstore *st, const void *protect_key) {
    /* Single-key form used by coli_estore_get -- protect_key is not yet in
     * the LRU list at this point (lru_touch runs AFTER the fill below), so
     * in practice this only ever considers genuinely other, already-
     * resident entries; the check is defense in depth, not the only thing
     * enforcing it. */
    evict_one_if(st, [protect_key](const void *k){ return k == protect_key; });
}

int coli_estore_resident(const ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    return it != st->map.end() && it->second.buf != nullptr;
}

int coli_estore_test_pinned(const ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    return it != st->map.end() && it->second.pinned;
}

int coli_estore_drop(ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    if (it == st->map.end() || !it->second.buf) return 0;
    Entry &e = it->second;
    st->resident_bytes -= e.buf_bytes;
    release_buf(st, e.buf, e.buf_bytes);   /* change A2: reuse pool, not free() */
    e.buf = nullptr;
    e.buf_bytes = 0;
    e.buf_off = 0;
    if (e.in_lru) { st->lru.erase(e.lru_it); e.in_lru = false; }
    return 1;
}

const uint8_t *coli_estore_get(ColiEstore *st, const void *key) {
    if (!st || !key) return nullptr;
    st->stat.requests++;
    auto it = st->map.find(key);
    if (it == st->map.end()) return nullptr;   /* never registered */
    Entry &e = it->second;
    if (e.buf) {
        st->stat.hits++;
        lru_touch(st, key, e);
        return e.buf + e.buf_off;
    }
    st->stat.misses++;
    /* Predict how many bytes this fill will occupy BEFORE evicting or
     * allocating, so a tight budget never transiently exceeds it (matters on
     * a box where the budget is set close to available RAM, not just for the
     * accounting). Change A: when O_DIRECT will be attempted, that is the
     * 4096-aligned superset length, not slice.nbytes -- see try_direct_read's
     * comment. Falls back to the exact slice length below if the direct
     * attempt itself fails. */
    int64_t off0 = e.slice.off & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
    int64_t want_len = st->direct_pref
        ? ((e.slice.off + e.slice.nbytes + COLI_ESTORE_ALIGN - 1) & ~(int64_t)(COLI_ESTORE_ALIGN - 1)) - off0
        : e.slice.nbytes;
    if (st->budget_bytes > 0) {
        while (st->resident_bytes + want_len > st->budget_bytes && !st->lru.empty()) {
            int64_t before = st->resident_bytes;
            evict_one(st, key);
            if (st->resident_bytes == before) break;   /* nothing left to evict */
        }
    }
    /* Change A2: pull the fill buffer from the reuse pool instead of a
     * fresh posix_memalign -- see the struct comment / alloc_buf's own
     * comment for why this is the per-fill fixed cost the io05 warm control
     * measured. */
    uint8_t *base = nullptr; int64_t buf_off = 0; int64_t buf_len = e.slice.nbytes;
    bool ok = false;
    if (st->direct_pref) {
        base = alloc_buf(st, want_len);
        if (base) {
            int64_t dlen;
            ok = try_direct_read(st, e.slice, base, want_len, &buf_off, &dlen);
            if (ok) { buf_len = dlen; st->stat.direct_reads++; }
            else { release_buf(st, base, want_len); base = nullptr; }
        }
    }
    if (!ok) {
        /* Buffered fallback -- exact slice length, no offset. */
        buf_off = 0; buf_len = e.slice.nbytes;
        base = alloc_buf(st, buf_len);
        if (!base) {
            fprintf(stderr, "expert_store: out of memory allocating %lld bytes\n", (long long)buf_len);
            return nullptr;
        }
        ok = buffered_read(st, e.slice, base);
        if (ok) st->stat.buffered_reads++;
    }
    if (!ok) {
        fprintf(stderr, "expert_store: disk read failed for a registered expert (off=%lld n=%lld path=%s)\n",
                (long long)e.slice.off, (long long)e.slice.nbytes, e.slice.shard_path);
        release_buf(st, base, buf_len);
        return nullptr;
    }
    st->stat.bytes_read += (uint64_t)buf_len;
    /* COLI_BREAK_ESTORE=1: the end-to-end negative control (same convention as
     * COLI_BREAK_WIDE / COLI_BREAK_I4 elsewhere in this engine). Flips one byte
     * of every fill, at the SLICE's first byte (base+buf_off), not byte 0 of
     * the allocation -- with the change A superset, byte 0 of `base` can be
     * padding before the slice that no consumer ever reads, which would make
     * this control silently fail to perturb anything. Store-on nll1 matching
     * store-off proves nothing unless a store that hands back WRONG bytes
     * visibly moves nll1 -- this arm must disagree. Never set outside that
     * control. */
    static int brk = -1;
    if (brk < 0) { const char *b = getenv("COLI_BREAK_ESTORE"); brk = (b && *b == '1') ? 1 : 0; }
    if (brk) (base + buf_off)[0] ^= 0x55;
    e.buf = base;
    e.buf_bytes = buf_len;
    e.buf_off = buf_off;
    st->resident_bytes += buf_len;
    lru_touch(st, key, e);
    return e.buf + e.buf_off;
}

/* COLI_MOE_RESID (2026-09-18): fetch through the ordinary path (so a cold key
 * pays exactly one normal fill, counted in requests/hits/misses like any
 * other coli_estore_get) and then mark it sticky. Refuses (0) rather than
 * pinning a key that never became resident -- e.g. a disk read failure --
 * because a sticky pin on a null buffer would just be a permanently-wrong
 * no-op entry taking up a slot in nothing. */
int coli_estore_pin_sticky(ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    if (!coli_estore_get(st, key)) return 0;
    auto it = st->map.find(key);
    if (it == st->map.end() || !it->second.buf) return 0;
    it->second.sticky = true;
    return 1;
}
void coli_estore_unpin_sticky(ColiEstore *st, const void *key) {
    if (!st || !key) return;
    auto it = st->map.find(key);
    if (it != st->map.end()) it->second.sticky = false;
}
int coli_estore_test_sticky(const ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    return it != st->map.end() && it->second.sticky;
}

void coli_estore_stats(const ColiEstore *st, ColiEstoreStats *out) {
    if (!st || !out) return;
    *out = st->stat;
    out->resident_bytes = st->resident_bytes;
}

void coli_estore_unpin_all(ColiEstore *st) {
    if (!st) return;
    for (auto k : st->pinned_keys) {
        auto it = st->map.find(k);
        if (it != st->map.end()) it->second.pinned = false;
    }
    st->pinned_keys.clear();
}

namespace {
/* One key's fill, resolved on the caller thread (fds, buffer, predicted
 * length) and executed by a pool worker (only the pread()s). */
struct PrefetchJob {
    const void *key = nullptr;
    uint8_t *buf = nullptr; int64_t alloc_len = 0;
    int fd_direct = -1, fd_buffered = -1;
    bool attempt_direct = false;
    int64_t off0 = 0, off_direct = 0;
    int64_t slice_off = 0, slice_nbytes = 0;   /* copied out of the map on the
                                                 * caller thread so the worker
                                                 * never touches it (or any
                                                 * other map/Entry state) */
    bool ok = false, via_direct = false;
    int64_t final_len = 0, final_off = 0;
};
} // namespace

int coli_estore_prefetch(ColiEstore *st, const void *const *keys, int n) {
    if (!st || !keys || n <= 0) return 0;
    coli_estore_unpin_all(st);   /* release the previous batch's pins first, per the header contract */

    if (st->pool_threads < 0) {
        const char *e = getenv("COLI_ESTORE_THREADS");
        st->pool_threads = (!e || !*e) ? 4 : atoi(e);
        if (st->pool_threads < 0) st->pool_threads = 0;
        if (st->pool_threads > 1) { st->pool = new FillPool(); st->pool->start(st->pool_threads); }
    }

    /* Every valid, registered key in this batch (resident already or about
     * to be filled) is protected from eviction -- "never a key in this
     * batch", not just the one currently being filled. */
    std::unordered_set<const void*> batch;
    for (int i = 0; i < n; i++) if (keys[i] && st->map.count(keys[i])) batch.insert(keys[i]);

    std::vector<const void*> need;   /* not-yet-resident keys, de-duplicated, in order */
    need.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        const void *k = keys[i];
        if (!k) continue;
        auto it = st->map.find(k);
        if (it == st->map.end() || it->second.buf) continue;
        bool dup = false; for (auto q : need) if (q == k) { dup = true; break; }
        if (!dup) need.push_back(k);
    }

    /* Evict (excluding the whole batch) to make room for the predicted
     * footprint of every miss -- same aligned-superset predictor
     * coli_estore_get uses for a single fill. */
    if (st->budget_bytes > 0 && !need.empty()) {
        int64_t want_total = 0;
        for (auto k : need) {
            const coli_gguf_slice &sl = st->map.at(k).slice;
            int64_t off0 = sl.off & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
            int64_t len = st->direct_pref
                ? ((sl.off + sl.nbytes + COLI_ESTORE_ALIGN - 1) & ~(int64_t)(COLI_ESTORE_ALIGN - 1)) - off0
                : sl.nbytes;
            want_total += len;
        }
        while (st->resident_bytes + want_total > st->budget_bytes) {
            if (!evict_one_if(st, [&](const void *k){ return batch.count(k) != 0; })) break;
        }
    }

    /* Resolve fds and allocate every job's buffer here, on the caller
     * thread -- fd caches and mallocs are not taken concurrently below. */
    std::vector<PrefetchJob> jv(need.size());
    for (size_t i = 0; i < need.size(); i++) {
        const void *k = need[i];
        const coli_gguf_slice &sl = st->map.at(k).slice;
        PrefetchJob &j = jv[i];
        j.key = k;
        j.slice_off = sl.off; j.slice_nbytes = sl.nbytes;
        j.fd_buffered = owned_fd_for(st, sl.shard_path, /*direct=*/false);
        j.attempt_direct = (st->direct_pref != 0);
        if (j.attempt_direct) j.fd_direct = owned_fd_for(st, sl.shard_path, /*direct=*/true);
        if (j.attempt_direct && j.fd_direct >= 0) {
            j.off0 = sl.off & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
            int64_t end = (sl.off + sl.nbytes + COLI_ESTORE_ALIGN - 1) & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
            j.alloc_len = end - j.off0;
            j.off_direct = sl.off - j.off0;
        } else {
            j.attempt_direct = false;
            j.alloc_len = sl.nbytes;
        }
        /* Change A2: same reuse pool coli_estore_get uses, still resolved
         * entirely on the caller thread (the pool worker only preads into
         * whatever j.buf already points to). */
        j.buf = alloc_buf(st, j.alloc_len);
        if (!j.buf)
            fprintf(stderr, "expert_store: out of memory allocating %lld bytes (prefetch)\n", (long long)j.alloc_len);
    }

    /* The pool touches ONLY each job's own buf/fds -- no map, no LRU, no fd
     * cache, so no lock is needed there (see expert_store.h's PINNING
     * section). */
    std::vector<std::function<void()>> jobs;
    jobs.reserve(jv.size());
    for (size_t i = 0; i < jv.size(); i++) {
        PrefetchJob *j = &jv[i];
        jobs.emplace_back([j](){
            if (!j->buf) return;
            if (j->attempt_direct) {
                ssize_t r = pread(j->fd_direct, j->buf, (size_t)j->alloc_len, (off_t)j->off0);
                if (r == (ssize_t)j->alloc_len) {
                    j->ok = true; j->via_direct = true;
                    j->final_len = j->alloc_len; j->final_off = j->off_direct;
                    return;
                }
            }
            /* Buffered fallback: exactly slice_nbytes at slice_off, no
             * offset -- reusing the same allocation if it is already big
             * enough (the common case, since an attempted direct's aligned
             * superset is always >= nbytes), else reallocating (malloc is
             * thread-safe; this worker owns j->buf exclusively, nothing else
             * touches it). */
            if (j->alloc_len < j->slice_nbytes) {
                free(j->buf);
                if (posix_memalign((void**)&j->buf, COLI_ESTORE_ALIGN, (size_t)j->slice_nbytes) != 0 || !j->buf) {
                    j->buf = nullptr; return;
                }
                j->alloc_len = j->slice_nbytes;
            }
            if (j->fd_buffered < 0) return;
            ssize_t r = pread(j->fd_buffered, j->buf, (size_t)j->slice_nbytes, (off_t)j->slice_off);
            if (r == (ssize_t)j->slice_nbytes) {
                j->ok = true; j->via_direct = false;
                j->final_len = j->slice_nbytes; j->final_off = 0;
            }
        });
    }
    if (st->pool_threads > 1 && st->pool) st->pool->run(jobs); else for (auto &f : jobs) f();

    /* Change B2 (2026-09-17 brief 2, item 3): a prefetch fill is counted in
     * prefetch_fills, NOT in requests/misses -- see expert_store.h's
     * ColiEstoreStats comment. Nothing here increments requests/hits/misses;
     * that only happens in coli_estore_get, so a key this batch fills and a
     * later real coli_estore_get() call finds resident is exactly one
     * prefetch_fill (here) and one ordinary hit (there), not a miss+hit pair
     * that would inflate `requests` beyond the number of actual
     * coli_estore_get calls the compute path makes. */
    for (auto &j : jv) {
        if (!j.ok) {
            fprintf(stderr, "expert_store: disk read failed for a registered expert during prefetch\n");
            if (j.buf) release_buf(st, j.buf, j.alloc_len);   /* A2: don't leak a failed fill's buffer */
            continue;
        }
        static int brk = -1;
        if (brk < 0) { const char *b = getenv("COLI_BREAK_ESTORE"); brk = (b && *b == '1') ? 1 : 0; }
        if (brk) (j.buf + j.final_off)[0] ^= 0x55;
        Entry &e = st->map[j.key];
        e.buf = j.buf; e.buf_bytes = j.final_len; e.buf_off = j.final_off;
        st->resident_bytes += j.final_len;
        if (j.via_direct) st->stat.direct_reads++; else st->stat.buffered_reads++;
        st->stat.bytes_read += (uint64_t)j.final_len;
        st->stat.prefetch_fills++;
        lru_touch(st, j.key, e);
    }

    for (auto k : batch) {
        auto it = st->map.find(k);
        if (it == st->map.end() || !it->second.buf) continue;   /* a fill failure leaves it correctly unpinned */
        it->second.pinned = true;
        st->pinned_keys.push_back(k);
    }
    return 1;
}
