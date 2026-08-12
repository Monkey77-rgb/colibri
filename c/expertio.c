/* expertio.c -- what expert streaming actually costs, at the granularity it happens.
 *
 * "NVMe is 5.45 GB/s, DRAM is 81.8 GB/s, therefore streaming is 15x worse" is wrong in a
 * way that matters, because it compares the wrong quantity. A streaming MoE engine does
 * not issue one 5 GB sequential read. It issues a few hundred SMALL RANDOM reads per
 * token -- one per activated expert per layer -- and if it issues them the naive way,
 * each one is a synchronous round trip.
 *
 * At that size the device is not bandwidth-limited at all. It is limited by how many
 * requests are in flight. A single outstanding 2.65 MB read gets a fraction of the
 * device's rated throughput, because the device spends most of the time idle waiting for
 * the next request to arrive. Rated bandwidth is a QUEUE-DEPTH-32 number quoted against a
 * QD1 access pattern.
 *
 * That distinction is the difference between "streaming is hopeless" and "streaming is an
 * engineering problem with a known fix", so this tool measures it rather than asserting
 * it. It sweeps queue depth at expert granularity and reports, for each depth:
 *
 *      achieved bandwidth, per-read latency, and the tok/s that implies
 *
 * The tok/s column is the point. It converts a storage microbenchmark directly into the
 * units the decision is made in, for a stated number of expert activations per token.
 *
 * O_DIRECT by default: buffered reads would be served from page cache on the second pass
 * and report memory bandwidth wearing a storage costume. --buffered measures the cached
 * path deliberately, which is its own useful number (it is what a warm engine sees).
 *
 * Build: cc -O2 -pthread -o expertio expertio.c
 * Usage: ./expertio FILE [--mb 2.65] [--acts 384] [--buffered] [--reads 256]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct {
    int      fd;
    size_t   blk;
    off_t   *offs;
    int      n;            /* total reads across all threads */
    int      tid, nthreads;
    long long bytes;
    double   lat_sum;      /* seconds, summed per read */
    int      done;
    int      err;
} Job;

static void *runner(void *arg) {
    Job *j = arg;
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, j->blk)) { j->err = 1; return NULL; }
    for (int i = j->tid; i < j->n; i += j->nthreads) {
        double t0 = now_s();
        ssize_t r = pread(j->fd, buf, j->blk, j->offs[i]);
        double dt = now_s() - t0;
        if (r <= 0) { j->err = errno ? errno : 1; break; }
        j->bytes   += r;
        j->lat_sum += dt;
        j->done++;
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    double blk_mb = 2.65;      /* one expert of a 30B-A3B at Q4_K_M, measured by modelprobe */
    int    acts    = 384;      /* activations per token: experts_active x layers */
    int    nreads  = 256;
    int    buffered = 0;
    /* --warm reads every offset once BEFORE timing, so the measurement is of the
     * page-cache-resident path rather than a mixture of cold and warm. Without this the
     * QD sweep contaminates itself: QD1 runs first and faults the blocks in, so every
     * later depth is reading from cache and the "queue depth helps" conclusion would be
     * an artifact of ordering rather than a property of the device. */
    int    warm = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mb")    && i+1 < argc) blk_mb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--acts")  && i+1 < argc) acts   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reads") && i+1 < argc) nreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--buffered")) buffered = 1;
        else if (!strcmp(argv[i], "--warm")) warm = 1;
        else if (argv[i][0] != '-') path = argv[i];
        else { puts("usage: expertio FILE [--mb 2.65] [--acts 384] [--reads 256] [--buffered]"); return 0; }
    }
    if (!path) { fprintf(stderr, "need a large file to read from\n"); return 2; }

    size_t blk = (size_t)(blk_mb * 1024 * 1024) & ~(size_t)4095;   /* O_DIRECT alignment */
    if (blk == 0) { fprintf(stderr, "block too small\n"); return 2; }

    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (!buffered) flags |= O_DIRECT;
#endif
    int fd = open(path, flags);
    if (fd < 0 && !buffered) {
        fprintf(stderr, "O_DIRECT unavailable (%s), falling back to buffered\n", strerror(errno));
        buffered = 1;
        fd = open(path, O_RDONLY);
    }
    if (fd < 0) { perror("open"); return 1; }

    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz < (off_t)blk * 4) { fprintf(stderr, "file too small for this block size\n"); return 1; }

    /* Fixed seed: every queue depth reads the SAME offsets in the same order, so the
     * comparison across depths is a comparison of concurrency and nothing else. */
    off_t *offs = malloc((size_t)nreads * sizeof *offs);
    srand(20260812);
    for (int i = 0; i < nreads; i++) {
        off_t r30 = ((off_t)rand() << 15) | rand();
        offs[i] = ((r30 * 4096) % (sz - (off_t)blk)) & ~(off_t)4095;
    }

    printf("expertio — random reads at expert granularity\n");
    printf("  file           : %s (%.2f GB)\n", path, (double)sz / 1e9);
    printf("  block          : %.2f MB   (one expert)\n", (double)blk / 1e6);
    printf("  mode           : %s\n", buffered ? "BUFFERED (page cache in play)"
                                               : "O_DIRECT (cold, device-only — the honest miss cost)");
    if (warm) {
        if (!buffered)
            puts("  WARNING        : --warm with O_DIRECT does nothing; O_DIRECT bypasses the cache.");
        void *wb = NULL;
        if (posix_memalign(&wb, 4096, blk) == 0) {
            for (int i = 0; i < nreads; i++)
                if (pread(fd, wb, blk, offs[i]) < 0) break;
            free(wb);
        }
        printf("  pre-warmed     : %d blocks (%.0f MB) faulted in before timing\n",
               nreads, (double)nreads * (double)blk / 1e6);
    }
    printf("  reads/config   : %d, identical offsets at every depth\n\n", nreads);

    printf("  %-6s %12s %14s %16s\n", "QD", "GB/s", "latency/read", "tok/s if ALL miss");
    printf("  %-6s %12s %14s %16s\n", "----", "-----", "------------", "----------------");

    int depths[] = {1, 2, 4, 8, 16, 32, 64};
    double best_gbs = 0; int best_qd = 1;
    for (size_t d = 0; d < sizeof depths / sizeof *depths; d++) {
        int qd = depths[d];
        if (qd > nreads) break;
        Job *jobs = calloc((size_t)qd, sizeof *jobs);
        pthread_t *th = calloc((size_t)qd, sizeof *th);
        for (int i = 0; i < qd; i++) {
            jobs[i] = (Job){ .fd = fd, .blk = blk, .offs = offs, .n = nreads,
                             .tid = i, .nthreads = qd };
        }
        double t0 = now_s();
        for (int i = 0; i < qd; i++) pthread_create(&th[i], NULL, runner, &jobs[i]);
        for (int i = 0; i < qd; i++) pthread_join(th[i], NULL);
        double wall = now_s() - t0;

        long long bytes = 0; double lat = 0; int done = 0, err = 0;
        for (int i = 0; i < qd; i++) {
            bytes += jobs[i].bytes; lat += jobs[i].lat_sum;
            done  += jobs[i].done;  if (jobs[i].err) err = jobs[i].err;
        }
        free(jobs); free(th);
        if (err) { fprintf(stderr, "  read error at QD%d: %s\n", qd, strerror(err)); continue; }

        double gbs   = (double)bytes / wall / 1e9;
        double lat_u = done ? lat / done * 1e6 : 0;
        /* If every activation misses, the token waits for `acts` reads served at this
         * aggregate rate. This is the pessimistic end of the two-tier model. */
        double tps   = (double)acts * (double)blk > 0
                     ? gbs * 1e9 / ((double)acts * (double)blk) : 0;
        if (gbs > best_gbs) { best_gbs = gbs; best_qd = qd; }
        printf("  %-6d %12.2f %11.0f us %16.1f\n", qd, gbs, lat_u, tps);
    }

    printf("\n  peak %.2f GB/s at QD%d", best_gbs, best_qd);
    printf(" — the QD1 row is what a synchronous demand-paged engine gets,\n");
    puts("  and the gap between them is what prefetch depth is worth. Rated device");
    puts("  bandwidth is a high-queue-depth figure; quoting it against QD1 access");
    puts("  overstates a naive streaming engine by exactly that ratio.");

    free(offs); close(fd);
    return 0;
}
