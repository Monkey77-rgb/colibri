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
};

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
    for (auto &kv : st->direct_fds) if (kv.second >= 0) close(kv.second);
    for (auto &kv : st->buffered_fds) if (kv.second >= 0) close(kv.second);
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
 * direct_pref is set and the fd is available. Returns the superset in
 * *out_base (a posix_memalign(4096, len) the caller now owns and must
 * eventually free()), *out_off = off-off0 (0 to add to get the slice's first
 * byte) and *out_len = len (what the caller must account in the budget and
 * free()). On any failure (fd unavailable, short read, OOM) frees anything
 * it allocated and returns false; the caller falls back to buffered_read. */
static bool try_direct_read(ColiEstore *st, const coli_gguf_slice &slice,
                             uint8_t **out_base, int64_t *out_off, int64_t *out_len) {
    if (!st->direct_pref) return false;
    int fd = owned_fd_for(st, slice.shard_path, /*direct=*/true);
    if (fd < 0) return false;
    int64_t off0 = slice.off & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
    int64_t end  = (slice.off + slice.nbytes + COLI_ESTORE_ALIGN - 1) & ~(int64_t)(COLI_ESTORE_ALIGN - 1);
    int64_t len  = end - off0;
    uint8_t *base = nullptr;
    if (posix_memalign((void**)&base, COLI_ESTORE_ALIGN, (size_t)len) != 0 || !base) return false;
    ssize_t r = pread(fd, base, (size_t)len, (off_t)off0);
    if (r != (ssize_t)len) { free(base); return false; }
    *out_base = base; *out_off = slice.off - off0; *out_len = len;
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

static void evict_one(ColiEstore *st, const void *protect_key) {
    /* Evict from the back (least-recently-used) forward, skipping the key
     * currently being filled -- it is not yet in the LRU list at this point
     * (lru_touch runs AFTER the fill below), so in practice this loop only
     * ever needs to consider genuinely other, already-resident entries; the
     * `protect_key` check is defense in depth, not the only thing enforcing
     * it. */
    for (auto rit = st->lru.rbegin(); rit != st->lru.rend(); ++rit) {
        const void *k = *rit;
        if (k == protect_key) continue;
        auto mit = st->map.find(k);
        if (mit == st->map.end() || !mit->second.buf) continue;
        Entry &victim = mit->second;
        st->resident_bytes -= victim.buf_bytes;
        free(victim.buf);
        victim.buf = nullptr;
        victim.buf_bytes = 0;
        victim.buf_off = 0;
        st->lru.erase(victim.lru_it);
        victim.in_lru = false;
        return;
    }
}

int coli_estore_resident(const ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    return it != st->map.end() && it->second.buf != nullptr;
}

int coli_estore_drop(ColiEstore *st, const void *key) {
    if (!st || !key) return 0;
    auto it = st->map.find(key);
    if (it == st->map.end() || !it->second.buf) return 0;
    Entry &e = it->second;
    st->resident_bytes -= e.buf_bytes;
    free(e.buf);
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
    uint8_t *base = nullptr; int64_t buf_off = 0; int64_t buf_len = e.slice.nbytes;
    bool ok = false;
    if (st->direct_pref) {
        int64_t dlen;
        ok = try_direct_read(st, e.slice, &base, &buf_off, &dlen);
        if (ok) { buf_len = dlen; st->stat.direct_reads++; }
    }
    if (!ok) {
        /* Buffered fallback -- exact slice length, no offset. */
        buf_off = 0; buf_len = e.slice.nbytes;
        if (posix_memalign((void**)&base, COLI_ESTORE_ALIGN, (size_t)buf_len) != 0 || !base) {
            fprintf(stderr, "expert_store: out of memory allocating %lld bytes\n", (long long)buf_len);
            return nullptr;
        }
        ok = buffered_read(st, e.slice, base);
        if (ok) st->stat.buffered_reads++;
    }
    if (!ok) {
        fprintf(stderr, "expert_store: disk read failed for a registered expert (off=%lld n=%lld path=%s)\n",
                (long long)e.slice.off, (long long)e.slice.nbytes, e.slice.shard_path);
        free(base);
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

void coli_estore_stats(const ColiEstore *st, ColiEstoreStats *out) {
    if (!st || !out) return;
    *out = st->stat;
    out->resident_bytes = st->resident_bytes;
}
