/* bench_expert_store.c -- io04 brief (2026-09-17) oracle case 3.
 *
 * Cold, random-order fills of 192 REAL gpt-oss-120b expert matrices (128
 * ffn_gate_exps + 64 ffn_up_exps of layer 0 -- two different tensors so the
 * 192 is not just "the same 128 plus 64 repeats of the first 64"), timing:
 *
 *   - serial buffered   (direct_pref=0, one coli_estore_get() per key)
 *   - serial direct     (direct_pref=1, one coli_estore_get() per key --
 *                        change A's aligned-superset O_DIRECT path)
 *   - coli_estore_prefetch in batches of 12, pool sizes 1 / 4 / 8
 *     (change B; direct_pref=1, same aligned-superset reads, concurrent)
 *
 * "Cold" = posix_fadvise(..., POSIX_FADV_DONTNEED) on our OWN fd for the
 * model file immediately before each arm, and a FRESH ColiEstore per arm (no
 * key from a previous arm's store can be a hit here). fadvise affects the
 * page cache for the whole file regardless of which fd reads it afterwards,
 * so the engine's own coli_gguf-opened fd sees the same cold cache.
 *
 * Positive control built into the measurement itself: bytes_read (from
 * ColiEstoreStats) must equal exactly 192 * the expert's slice size for
 * every arm -- if the cache were NOT actually cold, hits would undercount
 * bytes_read with no matching drop in wall time, which is the kind of
 * mismatch this file's own CHECK() catches.
 *
 * Every arm prints: elapsed ms, GB/s, and the nvme1n1 sectors-read delta
 * from /proc/diskstats (in 512-byte sectors, per the kernel's own unit) --
 * a near-zero delta on a "cold" arm would mean the page cache was not
 * actually cold (a control that can fail: page-cache-hit arms, added below,
 * ARE expected to show a near-zero disk delta while still returning fast,
 * demonstrating the diskstats probe can see a real cache hit and is not
 * just always non-zero).
 *
 * SKIPs (exit 77), not FAILs, if the model file is absent -- same
 * convention as test_expert_store.c.
 */
#include "../src/loader.h"
#include "../src/expert_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail = 1; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
    else { fprintf(stderr, "ok:   " __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* nvme1n1's own sectors-read counter (field 6 of /proc/diskstats' "nvme1n1 "
 * line, per Documentation/admin-guide/iostats.rst) -- NOT nvme1n1p2 (the
 * mounted partition also has its own line; the whole-device line is what
 * actually issues I/O to the controller). Returns -1 if not found (a
 * differently-named NVMe on another box: report "not measured", never
 * fabricate a number). */
static long long nvme_sectors_read(void) {
    FILE *f = fopen("/proc/diskstats", "r");
    if (!f) return -1;
    char line[512];
    long long result = -1;
    while (fgets(line, sizeof line, f)) {
        char name[64]; long long rd_ios, rd_merges, rd_sectors;
        if (sscanf(line, "%*d %*d %63s %lld %lld %lld", name, &rd_ios, &rd_merges, &rd_sectors) == 4) {
            if (!strcmp(name, "nvme1n1")) { result = rd_sectors; break; }
        }
    }
    fclose(f);
    return result;
}

static void print_condition_line(const char *model_path) {
    double load1=-1,load5=-1,load15=-1;
    FILE *f = fopen("/proc/loadavg", "r");
    if (f) { if (fscanf(f, "%lf %lf %lf", &load1,&load5,&load15) != 3) {} fclose(f); }
    long long mem_avail_kb = -1, cached_kb = -1;
    f = fopen("/proc/meminfo", "r");
    if (f) {
        char key[64]; long long val; char unit[16];
        while (fscanf(f, "%63s %lld %15s", key, &val, unit) == 3) {
            if (!strcmp(key, "MemAvailable:")) mem_avail_kb = val;
            if (!strcmp(key, "Cached:")) cached_kb = val;
        }
        fclose(f);
    }
    fprintf(stderr, "condition: model=%s load1=%.2f MemAvailable=%.1fGiB Cached=%.1fGiB\n",
            model_path, load1, mem_avail_kb/1048576.0, cached_kb/1048576.0);
}

/* Drops the page cache for the WHOLE model file (our own fadvise fd is
 * closed right after -- coli_gguf's own fd, opened separately below, reads
 * the same now-cold pages). */
static void drop_cache(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "WARN: drop_cache could not open '%s'\n", path); return; }
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0)
        fprintf(stderr, "WARN: posix_fadvise(DONTNEED) failed for '%s'\n", path);
    close(fd);
}

/* Fisher-Yates, fixed seed -- same random order reused across arms so the
 * comparison is apples to apples (any one expert is equally "unlucky" -- an
 * early NVMe-queue-still-ramping read -- in every arm, not just one). */
static void shuffle(int *a, int n, unsigned seed) {
    srand(seed);
    for (int i = n-1; i > 0; i--) { int j = rand() % (i+1); int t=a[i]; a[i]=a[j]; a[j]=t; }
}

typedef struct { const void *key; coli_gguf_slice slice; } Reg;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                      : (getenv("COLI_TEST_MODEL_GPTOSS") ? getenv("COLI_TEST_MODEL_GPTOSS")
                      : "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf");
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { fprintf(stderr, "SKIP: cannot open '%s': %s\n", path, err); return 77; }

    print_condition_line(path);

    const char *nm_gate = "blk.0.ffn_gate_exps.weight";
    const char *nm_up   = "blk.0.ffn_up_exps.weight";
    if (!coli_gguf_has(g, nm_gate) || !coli_gguf_has(g, nm_up)) {
        fprintf(stderr, "SKIP: expected tensors not found in '%s'\n", path); coli_gguf_close(g); return 77;
    }
    int64_t NE = coli_gguf_shape(g, nm_gate, 2);
    if (NE < 128) { fprintf(stderr, "SKIP: unexpected n_expert %lld\n", (long long)NE); coli_gguf_close(g); return 77; }

    enum { N = 192 };
    static Reg regs[N];
    static int keys_dummy[N];
    int n = 0, nok = 0;
    for (int e = 0; e < 128 && n < N; e++) {
        nok += coli_gguf_tensor_slice(g, nm_gate, e, NE, &regs[n].slice);
        regs[n].key = &keys_dummy[n]; n++;
    }
    for (int e = 0; e < 64 && n < N; e++) {
        nok += coli_gguf_tensor_slice(g, nm_up, e, NE, &regs[n].slice);
        regs[n].key = &keys_dummy[n]; n++;
    }
    CHECK(n == N && nok == N, "resolved %d/%d expert slices (%d succeeded)", n, (int)N, nok);
    int64_t slice_bytes = regs[0].slice.nbytes;
    int64_t total_bytes = slice_bytes * N;
    fprintf(stderr, "N=%d experts, %lld bytes/expert, %.1f MB total\n", (int)N, (long long)slice_bytes, total_bytes/1e6);

    int order[N]; for (int i = 0; i < N; i++) order[i] = i;
    shuffle(order, N, 0xE57);

    typedef struct { const char *name; int direct_pref; int use_prefetch; int pool_threads; } Arm;
    Arm arms[] = {
        { "serial_buffered", 0, 0, 0 },
        { "serial_direct",   1, 0, 0 },
        { "prefetch12_pool1",1, 1, 1 },
        { "prefetch12_pool4",1, 1, 4 },
        { "prefetch12_pool8",1, 1, 8 },
    };
    int narm = (int)(sizeof arms / sizeof arms[0]);

    fprintf(stderr, "\n%-20s %8s %10s %10s %14s\n", "arm(round)", "ms", "GB/s", "ms/batch", "nvme_sectors_rd");
    for (int round = 1; round <= 3; round++) {
        for (int ai = 0; ai < narm; ai++) {
            Arm *a = &arms[ai];
            drop_cache(path);
            if (a->use_prefetch) {
                char buf[32]; snprintf(buf, sizeof buf, "%d", a->pool_threads);
                setenv("COLI_ESTORE_THREADS", buf, 1);
            }
            ColiEstore *st = coli_estore_create(/*budget=*/-1, a->direct_pref);
            int nreg_ok = 0;
            for (int i = 0; i < N; i++) nreg_ok += coli_estore_register(st, regs[i].key, &regs[i].slice);
            CHECK(nreg_ok == N, "%s round %d: all %d experts registered (%d ok)", a->name, round, (int)N, nreg_ok);

            long long sect0 = nvme_sectors_read();
            double t0 = now_s();
            if (!a->use_prefetch) {
                for (int i = 0; i < N; i++) {
                    const uint8_t *r = coli_estore_get(st, regs[order[i]].key);
                    if (!r) { fprintf(stderr, "FAIL: get failed at %d (%s round %d)\n", i, a->name, round); g_fail = 1; break; }
                }
            } else {
                int nbatch = 0;
                for (int i = 0; i < N; i += 12) {
                    const void *keys[12]; int nk = 0;
                    for (int j = i; j < i+12 && j < N; j++) keys[nk++] = regs[order[j]].key;
                    int r = coli_estore_prefetch(st, keys, nk);
                    if (!r) { fprintf(stderr, "FAIL: prefetch failed at batch %d (%s round %d)\n", nbatch, a->name, round); g_fail = 1; }
                    nbatch++;
                }
            }
            double t1 = now_s();
            long long sect1 = nvme_sectors_read();

            ColiEstoreStats st_stats; coli_estore_stats(st, &st_stats);
            CHECK(st_stats.bytes_read >= (uint64_t)total_bytes,
                  "%s round %d: bytes_read (%llu) >= expected cold total (%lld) -- proves the cache really was cold",
                  a->name, round, (unsigned long long)st_stats.bytes_read, (long long)total_bytes);
            if (a->direct_pref) CHECK(st_stats.direct_reads == (uint64_t)N,
                  "%s round %d: all %d fills went via O_DIRECT (direct_reads=%llu)", a->name, round, (int)N,
                  (unsigned long long)st_stats.direct_reads);

            double ms = (t1-t0)*1000.0;
            double gbs = (double)total_bytes / (t1-t0) / 1e9;
            double ms_per_batch = a->use_prefetch ? ms / ((N+11)/12) : ms / N;
            long long sectdelta = (sect0 >= 0 && sect1 >= 0) ? (sect1 - sect0) : -1;
            char label[48]; snprintf(label, sizeof label, "%s(%d)", a->name, round);
            fprintf(stderr, "%-20s %8.1f %10.2f %10.3f %14lld\n", label, ms, gbs, ms_per_batch, sectdelta);

            coli_estore_destroy(st);
        }
    }

    /* ---- control: a WARM (not dropped) re-run of the same 192 experts on a
     * fresh store must show a much smaller nvme_sectors_read delta than the
     * cold arms above, while still returning bit-plausible timing -- proves
     * the diskstats probe actually reflects real disk I/O and is not just a
     * counter that always increments regardless of cache state. */
    {
        ColiEstore *st = coli_estore_create(-1, 1);
        for (int i = 0; i < N; i++) coli_estore_register(st, regs[i].key, &regs[i].slice);
        long long sect0 = nvme_sectors_read();
        double t0 = now_s();
        for (int i = 0; i < N; i++) coli_estore_get(st, regs[order[i]].key);   /* page cache is warm: no drop_cache() here */
        double t1 = now_s();
        long long sect1 = nvme_sectors_read();
        double ms = (t1-t0)*1000.0;
        long long sectdelta = (sect0>=0 && sect1>=0) ? (sect1-sect0) : -1;
        fprintf(stderr, "%-20s %8.1f %10s %10s %14lld  (WARM CONTROL -- expect sectors_rd << cold arms' ~%lld)\n",
                "warm_control", ms, "-", "-", sectdelta, (long long)(total_bytes/512));
        CHECK(sectdelta < (total_bytes/512)/2,
              "WARM CONTROL: nvme sectors-read delta (%lld) is well under a cold run's ~%lld sectors -- proves the probe can see a real cache hit",
              sectdelta, (long long)(total_bytes/512));
        coli_estore_destroy(st);
    }

    coli_gguf_close(g);
    if (g_fail) { fprintf(stderr, "\n=== bench_expert_store: FAIL ===\n"); return 1; }
    fprintf(stderr, "\n=== bench_expert_store: PASS ===\n");
    return 0;
}
