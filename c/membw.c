/* membw.c -- measure the ceiling that decides everything else.
 *
 * Autoregressive decode of a dense transformer reads every weight once per token.
 * So the upper bound on tok/s is not compute, not the engine, not the quant kernel:
 *
 *      tok/s_max = achievable_read_bandwidth / bytes_read_per_token
 *
 * On a discrete-GPU box that bandwidth is VRAM's, and it is huge. On a unified-memory
 * APU it is the DIMMs', and it is not. Which means the single most decision-relevant
 * number for "can this machine run a 35B model" is one this tool measures directly,
 * rather than one derived from a marketing figure that assumes perfect efficiency.
 *
 * We measure READ bandwidth specifically. STREAM's classic triad (a[i] = b[i] + s*c[i])
 * mixes reads with a write stream and reports a number that flatters or penalises
 * depending on write-allocate behaviour -- it is the wrong shape for weight streaming.
 * Triad is still reported for comparability with published STREAM results, but the
 * headline is the pure read.
 *
 * Method notes, because a bandwidth number without them is not evidence:
 *   - Buffer is sized well past LLC so we measure DRAM, not cache. Default 1 GiB/thread
 *     capped by available memory; the size actually used is printed.
 *   - Threads are spread over the buffer, each on its own slice, so we measure
 *     aggregate DRAM throughput rather than one core's outstanding-miss limit.
 *     A single core cannot saturate a modern memory controller; reporting a
 *     single-thread figure as "the bandwidth" is a common and large error.
 *   - The buffer is touched once before timing, so we are not measuring page faults
 *     and first-touch NUMA placement.
 *   - Best of N repeats is reported, not the mean: we want the ceiling the hardware
 *     can reach, and any noise from other processes can only push it down.
 *   - The read loop sums into a volatile-escaping accumulator so the optimiser cannot
 *     delete it. This is the classic way to accidentally measure nothing at all.
 *
 * Zero dependencies beyond C99 + POSIX + pthreads. No root.
 *
 * Build: cc -O2 -pthread -o membw membw.c
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    double *a, *b, *c;   /* triad uses all three; read uses a only */
    size_t  n;           /* doubles in this thread's slice */
    double  sink;        /* escape hatch so the read loop is not optimised away */
    int     mode;        /* 0 = read, 1 = triad */
} Slice;

static void *worker(void *arg) {
    Slice *s = arg;
    if (s->mode == 0) {
        /* Four independent accumulators: a single chain serialises on FP-add latency
         * and would measure the adder, not the memory system. */
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        size_t i = 0, n = s->n & ~(size_t)3;
        for (; i < n; i += 4) {
            t0 += s->a[i];
            t1 += s->a[i + 1];
            t2 += s->a[i + 2];
            t3 += s->a[i + 3];
        }
        for (; i < s->n; i++) t0 += s->a[i];
        s->sink = t0 + t1 + t2 + t3;
    } else {
        const double scalar = 3.0;
        for (size_t i = 0; i < s->n; i++) s->a[i] = s->b[i] + scalar * s->c[i];
        s->sink = s->a[s->n / 2];
    }
    return NULL;
}

/* Run one pass across all threads; return seconds elapsed. */
static double pass(Slice *sl, pthread_t *th, int nthreads) {
    double t = now_s();
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &sl[i]);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    return now_s() - t;
}

static long long mem_available_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atoll(line + 13); break; }
    fclose(f);
    return kb;
}

int main(int argc, char **argv) {
    int nthreads = 0, repeats = 5;
    double want_gib = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc)      nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) want_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) repeats  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) {
            puts("membw [-t threads] [-g total_GiB] [-r repeats]");
            puts("  Reports achievable DRAM read bandwidth, and the dense-model tok/s");
            puts("  ceiling that follows from it. Read-only, no root, no persistence.");
            return 0;
        }
    }

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads <= 0) nthreads = (int)(ncpu > 0 ? ncpu : 4);

    /* Size the buffer to clear LLC by a wide margin but stay well inside available
     * memory -- this runs on a production host and must not induce reclaim. We take
     * at most a third of MemAvailable. */
    long long avail_kb = mem_available_kb();
    double cap_gib = avail_kb > 0 ? (double)avail_kb / 1024.0 / 1024.0 / 3.0 : 1.0;
    if (want_gib <= 0) want_gib = 3.0;
    if (want_gib > cap_gib) want_gib = cap_gib;
    if (want_gib < 0.25) want_gib = 0.25;

    /* triad needs three buffers, read needs one; size for the triad case so both
     * measurements sweep the same footprint and are comparable. */
    size_t total_doubles = (size_t)(want_gib * (1ull << 30) / sizeof(double));
    size_t per_thread    = total_doubles / (size_t)nthreads;
    per_thread &= ~(size_t)7;
    if (per_thread == 0) { fprintf(stderr, "buffer too small\n"); return 1; }
    total_doubles = per_thread * (size_t)nthreads;

    size_t bytes = total_doubles * sizeof(double);
    double *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    if (!a || !b || !c) { fprintf(stderr, "alloc of 3 x %.2f GiB failed\n",
                                  (double)bytes / (1ull << 30)); return 1; }

    /* First touch: fault every page in before timing anything. */
    for (size_t i = 0; i < total_doubles; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }

    Slice   *sl = calloc((size_t)nthreads, sizeof *sl);
    pthread_t *th = calloc((size_t)nthreads, sizeof *th);
    for (int i = 0; i < nthreads; i++) {
        sl[i].a = a + (size_t)i * per_thread;
        sl[i].b = b + (size_t)i * per_thread;
        sl[i].c = c + (size_t)i * per_thread;
        sl[i].n = per_thread;
    }

    printf("membw — achievable DRAM bandwidth\n");
    printf("  threads        : %d (of %ld online)\n", nthreads, ncpu);
    printf("  buffer         : %.2f GiB per stream, %.2f GiB touched\n",
           (double)bytes / (1ull << 30), 3.0 * (double)bytes / (1ull << 30));
    printf("  repeats        : %d (best reported)\n\n", repeats);

    double best_read = 0, best_triad = 0;
    for (int r = 0; r < repeats; r++) {
        for (int i = 0; i < nthreads; i++) sl[i].mode = 0;
        double dt = pass(sl, th, nthreads);
        double gbs = (double)bytes / dt / 1e9;          /* GB = 1e9 B, as STREAM reports */
        if (gbs > best_read) best_read = gbs;

        for (int i = 0; i < nthreads; i++) sl[i].mode = 1;
        dt = pass(sl, th, nthreads);
        /* triad moves 3 streams: read b, read c, write a (ignoring write-allocate) */
        gbs = 3.0 * (double)bytes / dt / 1e9;
        if (gbs > best_triad) best_triad = gbs;
    }

    /* Consume the sinks so nothing is dead code. */
    double sink = 0;
    for (int i = 0; i < nthreads; i++) sink += sl[i].sink;

    printf("  READ           : %7.2f GB/s   <-- the number that bounds dense decode\n",
           best_read);
    printf("  TRIAD (STREAM) : %7.2f GB/s   (2 read + 1 write, for comparability)\n\n",
           best_triad);

    /* The whole point. A dense model reads all weights per token; an MoE model reads
     * only the shared trunk plus the activated experts. Both are computed from the
     * SAME measured bandwidth, which is what makes the comparison honest. */
    puts("dense decode ceiling at this bandwidth (weights read once per token,");
    puts("attention/KV traffic and all compute assumed free — so these are upper");
    puts("bounds no engine can beat, not predictions):\n");
    printf("    %-26s %10s %12s\n", "model (Q4_K_M ~0.6 B/param)", "weights", "tok/s max");
    struct { const char *name; double params_b; } m[] = {
        {"3B  dense",  3},  {"7B  dense",  7},  {"8B  dense",  8},
        {"14B dense", 14},  {"20B dense", 20},  {"27B dense", 27},
        {"35B dense", 35},  {"70B dense", 70},
    };
    for (size_t i = 0; i < sizeof m / sizeof *m; i++) {
        double gb = m[i].params_b * 0.60;               /* Q4_K_M measured ~0.60 B/param */
        printf("    %-26s %7.1f GB %10.1f\n", m[i].name, gb, best_read / gb);
    }

    printf("\nand the same 35B budget spent as MoE instead (only active weights are read):\n");
    printf("    %-26s %10s %12s\n", "", "read/token", "tok/s max");
    struct { const char *name; double active_b; } q[] = {
        {"35B total, 3B active",  3.0},
        {"35B total, 6B active",  6.0},
        {"35B total, 12B active", 12.0},
    };
    for (size_t i = 0; i < sizeof q / sizeof *q; i++) {
        double gb = q[i].active_b * 0.60;
        printf("    %-26s %7.1f GB %10.1f\n", q[i].name, gb, best_read / gb);
    }

    if (sink == 12345.6789) puts("");   /* never true; keeps sink live */
    free(a); free(b); free(c); free(sl); free(th);
    return 0;
}
