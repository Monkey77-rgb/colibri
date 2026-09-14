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
#include <unistd.h>
#include <sys/stat.h>

#ifndef O_DIRECT
#define O_DIRECT 0   /* platforms without it: direct_pref silently never engages */
#endif

#define COLI_ESTORE_ALIGN 4096

namespace {

struct Entry {
    coli_gguf_slice slice;
    uint8_t        *buf = nullptr;   /* nullptr == not resident */
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

static bool try_direct_read(ColiEstore *st, const Entry &e, uint8_t *buf) {
    if (!st->direct_pref) return false;
    if (e.slice.off % COLI_ESTORE_ALIGN != 0) return false;
    if (e.slice.nbytes % COLI_ESTORE_ALIGN != 0) return false;
    if ((reinterpret_cast<uintptr_t>(buf) % COLI_ESTORE_ALIGN) != 0) return false;
    int fd = owned_fd_for(st, e.slice.shard_path, /*direct=*/true);
    if (fd < 0) return false;
    ssize_t r = pread(fd, buf, (size_t)e.slice.nbytes, (off_t)e.slice.off);
    return r == (ssize_t)e.slice.nbytes;
}

/* Buffered fallback, via the store's OWN fd (see owned_fd_for's comment) --
 * NOT coli_gguf_slice_pread(&e.slice, buf), which would use e.slice.fd and
 * reintroduce the closed-fd bug above. coli_gguf_slice_pread remains correct
 * and useful for a caller whose coli_gguf is still open (this file's own
 * unit test uses it that way, legitimately); it is simply the wrong tool for
 * a cache that outlives the loader. */
static bool buffered_read(ColiEstore *st, const Entry &e, uint8_t *buf) {
    int fd = owned_fd_for(st, e.slice.shard_path, /*direct=*/false);
    if (fd < 0) return false;
    ssize_t r = pread(fd, buf, (size_t)e.slice.nbytes, (off_t)e.slice.off);
    return r == (ssize_t)e.slice.nbytes;
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
        st->resident_bytes -= victim.slice.nbytes;
        free(victim.buf);
        victim.buf = nullptr;
        st->lru.erase(victim.lru_it);
        victim.in_lru = false;
        return;
    }
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
        return e.buf;
    }
    st->stat.misses++;
    /* Evict BEFORE allocating, so a tight budget never transiently exceeds
     * it by one expert's worth (matters on a box where the budget is set
     * close to available RAM, not just for the accounting). */
    if (st->budget_bytes > 0) {
        while (st->resident_bytes + e.slice.nbytes > st->budget_bytes && !st->lru.empty()) {
            int64_t before = st->resident_bytes;
            evict_one(st, key);
            if (st->resident_bytes == before) break;   /* nothing left to evict */
        }
    }
    uint8_t *buf = nullptr;
    if (posix_memalign((void**)&buf, COLI_ESTORE_ALIGN, (size_t)e.slice.nbytes) != 0 || !buf) {
        fprintf(stderr, "expert_store: out of memory allocating %lld bytes\n", (long long)e.slice.nbytes);
        return nullptr;
    }
    bool ok = try_direct_read(st, e, buf);
    if (ok) st->stat.direct_reads++;
    else {
        ok = buffered_read(st, e, buf);
        if (ok) st->stat.buffered_reads++;
    }
    if (!ok) {
        fprintf(stderr, "expert_store: disk read failed for a registered expert (off=%lld n=%lld path=%s)\n",
                (long long)e.slice.off, (long long)e.slice.nbytes, e.slice.shard_path);
        free(buf);
        return nullptr;
    }
    st->stat.bytes_read += (uint64_t)e.slice.nbytes;
    /* COLI_BREAK_ESTORE=1: the end-to-end negative control (same convention as
     * COLI_BREAK_WIDE / COLI_BREAK_I4 elsewhere in this engine). Flips one byte
     * of every fill. Store-on nll1 matching store-off proves nothing unless a
     * store that hands back WRONG bytes visibly moves nll1 -- this arm must
     * disagree. Never set outside that control. */
    static int brk = -1;
    if (brk < 0) { const char *b = getenv("COLI_BREAK_ESTORE"); brk = (b && *b == '1') ? 1 : 0; }
    if (brk) buf[0] ^= 0x55;
    e.buf = buf;
    st->resident_bytes += e.slice.nbytes;
    lru_touch(st, key, e);
    return e.buf;
}

void coli_estore_stats(const ColiEstore *st, ColiEstoreStats *out) {
    if (!st || !out) return;
    *out = st->stat;
    out->resident_bytes = st->resident_bytes;
}
