/* gpu_keepalive.cpp -- see gpu_keepalive.h. The loop is tests/gpu_keepalive.c's
 * heavy mode, measured 2026-09-15 (goss16/17), moved in-process. */
#include "gpu_keepalive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef COLI_HAVE_VK
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <atomic>
#include "vk_backend.h"
#include "gemm_i8.h"

static pthread_t g_thr; static std::atomic<int> g_run{0}, g_alive{0};
static int g_period_us = 2000; static long g_n = 0; static double g_busy = 0, g_wall = 0;
static const int64_t KA_I = 8192, KA_O = 8192, KA_REPS = 4;   /* 32 MB int4 per GEMM, x4 per submit */

static double mono(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static void *ka_main(void *) {
    /* SCHED_IDLE (2026-09-15): the first two in-process A/Bs (goss18/19) lost
     * 0.2 tok/s because the CPU expert GEMV slowed ~20% with this thread in the
     * process (57 s vs 45-49 s) while the same work as an external process did
     * not; hypothesis: a 9th runnable thread beside 8 active OpenMP threads on
     * 8 cores. Idle priority makes it yield to them whenever they are runnable. */
    { struct sched_param sp; sp.sched_priority = 0; pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp); }
    char err[256];
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { fprintf(stderr, "keepalive: not running (%s)\n", err); g_alive = 0; return NULL; }
    unsigned s = 7; auto frand = [&]{ s = s*1664525u+1013904223u; return ((s>>8)&0xffff)/65536.0f-0.5f; };
    float *W = (float*)malloc((size_t)KA_I*KA_O*4), *X = (float*)malloc((size_t)KA_I*4);
    for (int64_t i=0;i<KA_I*KA_O;i++) W[i]=frand(); for (int64_t i=0;i<KA_I;i++) X[i]=frand();
    coli_w_i4 w4; coli_quantize_w4(&w4, W, KA_I, KA_O); free(W);
    int64_t nb = KA_I/COLI_ABLK; coli_a_i8 a; memset(&a,0,sizeof a); a.I=KA_I; a.n=1;
    a.q=(int8_t*)aligned_alloc(64,(size_t)KA_I); a.scale=(float*)aligned_alloc(64,(size_t)nb*4); a.sum=(int32_t*)aligned_alloc(64,(size_t)nb*4);
    coli_quantize_a(&a, X, 1, KA_I); free(X);
    int h = coli_vk_upload_w4(v, &w4);
    float *Y = (float*)calloc((size_t)KA_O, 4);
    if (h < 0) { fprintf(stderr, "keepalive: not running (upload failed)\n"); coli_vk_free(v); g_alive = 0; return NULL; }
    double t0 = mono();
    while (g_run.load()) {
        double sec = coli_vk_bench_gemm4(v, h, &a, Y, (int)KA_REPS);
        if (sec > 0) g_busy += sec;
        g_n++;
        usleep((useconds_t)g_period_us);
    }
    g_wall = mono() - t0;
    coli_vk_free(v); free(Y); free(a.q); free(a.scale); free(a.sum);
    g_alive = 0; return NULL;
}

extern "C" int coli_gpu_keepalive_start(int period_us, char *err, size_t errcap) {
    if (g_run.load()) return 0;
    g_period_us = period_us > 0 ? period_us : 2000; g_n = 0; g_busy = 0; g_wall = 0;
    g_run = 1; g_alive = 1;
    if (pthread_create(&g_thr, NULL, ka_main, NULL) != 0) { g_run = 0; g_alive = 0; snprintf(err, errcap, "pthread_create failed"); return -1; }
    return 0;
}
extern "C" void coli_gpu_keepalive_stop(void) {
    if (!g_run.load()) return;
    g_run = 0; pthread_join(g_thr, NULL);
    if (g_n > 0) fprintf(stderr, "keepalive: %ld submits in %.1f s, GPU busy %.2f s (%.1f%%), period %d us, %ldx%ld int4 x%ld\n",
                         g_n, g_wall, g_busy, g_wall > 0 ? 100*g_busy/g_wall : 0.0, g_period_us, (long)KA_I, (long)KA_O, (long)KA_REPS);
}
extern "C" int coli_gpu_keepalive_running(void) { return g_alive.load(); }
#else
extern "C" int  coli_gpu_keepalive_start(int, char *err, size_t errcap) { snprintf(err, errcap, "keepalive needs a Vulkan build"); return -1; }
extern "C" void coli_gpu_keepalive_stop(void) {}
extern "C" int  coli_gpu_keepalive_running(void) { return 0; }
#endif
