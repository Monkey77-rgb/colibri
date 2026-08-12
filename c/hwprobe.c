/* hwprobe.c — hardware capability probe + engine planner.
 *
 * WHY THIS EXISTS
 * ---------------
 * An expert-streaming MoE engine is bound by four hardware facts, and every one of them
 * differs per machine: how much memory can hold experts, whether that memory is shared with
 * the GPU, how fast the disk refills the cache, and which compute backend exists at all.
 * Hard-coding any of these produces an engine that runs on one box. So we measure them.
 *
 * DESIGN RULES
 *  - Zero dependency: C99 + POSIX. No libs. Matches the engine's own posture.
 *  - Portable by degradation, not by assumption. Every probe has a fallback and every
 *    unknown is reported as "unknown", never as a default that looks like a measurement.
 *  - Measured beats declared. Disk bandwidth is timed here, not read off a spec sheet.
 *  - It plans, it does not act. Output is a recommendation; the operator applies it.
 *
 * UNIFIED MEMORY IS THE LOAD-BEARING DISTINCTION
 * On a discrete GPU the hierarchy is disk -> host RAM -> PCIe -> VRAM, so staging experts
 * in host RAM is a real win: it removes a disk read. On an integrated GPU, VRAM *is* host
 * RAM -- same DIMMs, same controller. Copying an expert "to VRAM" moves bytes for no
 * bandwidth gain, and a host-RAM staging tier buys nothing at all. The whole CUDA-era
 * offload literature optimizes away a PCIe hop that does not exist there. We detect which
 * world we are in and plan differently, because the correct answer inverts.
 *
 * BUILD:  cc -O2 -o hwprobe hwprobe.c            (add -fopenmp for threaded disk probe)
 * USAGE:  ./hwprobe [--json] [--probe-file PATH] [--quick]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#ifndef NAME_MAX
#  define NAME_MAX 255
#endif
#ifdef _OPENMP
#  include <omp.h>
#endif

#if defined(_POSIX_VERSION) || defined(__unix__) || defined(__APPLE__)
#  include <sys/stat.h>
#  include <sys/utsname.h>
#  define HW_POSIX 1
#endif
#ifdef __linux__
#  include <dirent.h>
#endif
#ifdef __APPLE__
#  include <sys/sysctl.h>
#endif

#define HW_UNKNOWN (-1)
#define MAXGPU 8

/* ------------------------------------------------------------------ helpers */

static long long read_ll(const char *path) {          /* -1 if unreadable */
    FILE *f = fopen(path, "r");
    if (!f) return HW_UNKNOWN;
    long long v = HW_UNKNOWN;
    if (fscanf(f, "%lld", &v) != 1) v = HW_UNKNOWN;
    fclose(f);
    return v;
}

static int read_str(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(out, (int)n, f)) { fclose(f); return 0; }
    fclose(f);
    out[strcspn(out, "\r\n")] = 0;
    return out[0] != 0;
}

static double now_s(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
    return (double)time(NULL);
}

static int file_exists(const char *p) { return access(p, F_OK) == 0; }

/* Is a shared library loadable-by-name present anywhere obvious? We deliberately only
 * check for the file: dlopen()ing a GPU driver has side effects (device init, power
 * state) that a read-only probe has no business causing. */
static int have_lib(const char *soname) {
    static const char *dirs[] = {
        "/usr/lib", "/usr/lib64", "/lib", "/lib64",
        "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
        "/opt/rocm/lib", "/usr/local/lib", NULL
    };
    char p[512];
    for (int i = 0; dirs[i]; i++) {
        snprintf(p, sizeof p, "%s/%s", dirs[i], soname);
        if (file_exists(p)) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------- cpu */

typedef struct {
    long  cores_online;
    char  isa[128];
    char  arch[96];        /* >= sizeof(struct utsname.machine) on every platform we target */
    long  l3_kb;
} HwCpu;

static void probe_cpu(HwCpu *c) {
    memset(c, 0, sizeof *c);
    c->l3_kb = HW_UNKNOWN;

#ifdef _SC_NPROCESSORS_ONLN
    c->cores_online = sysconf(_SC_NPROCESSORS_ONLN);
#else
    c->cores_online = HW_UNKNOWN;
#endif

#ifdef HW_POSIX
    struct utsname u;
    if (uname(&u) == 0) snprintf(c->arch, sizeof c->arch, "%s", u.machine);
    else                snprintf(c->arch, sizeof c->arch, "unknown");
#else
    snprintf(c->arch, sizeof c->arch, "unknown");
#endif

    /* ISA detection is inherently per-architecture. Report what we can prove and stay
     * silent otherwise -- an unlisted feature means "not detected", not "absent". */
    c->isa[0] = 0;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2"))    strcat(c->isa, "avx2 ");
    if (__builtin_cpu_supports("avx512f")) strcat(c->isa, "avx512f ");
    if (__builtin_cpu_supports("fma"))     strcat(c->isa, "fma ");
#elif defined(__aarch64__)
    strcat(c->isa, "neon ");   /* mandatory on aarch64 */
#  if defined(__ARM_FEATURE_SVE)
    strcat(c->isa, "sve ");
#  endif
#endif
    if (!c->isa[0]) snprintf(c->isa, sizeof c->isa, "unknown");

#ifdef __linux__
    long v = read_ll("/sys/devices/system/cpu/cpu0/cache/index3/size");
    if (v > 0) c->l3_kb = v;                       /* usually already KB */
    else {
        char s[64];
        if (read_str("/sys/devices/system/cpu/cpu0/cache/index3/size", s, sizeof s)) {
            long n = strtol(s, NULL, 10);
            if (strchr(s, 'M')) n *= 1024;
            if (n > 0) c->l3_kb = n;
        }
    }
#endif
}

/* ---------------------------------------------------------------- memory */

typedef struct {
    long long total_kb;
    long long avail_kb;      /* MemAvailable: what can actually be had without swapping */
    long long swap_total_kb;
    long long swap_used_kb;
    long      page_kb;
} HwMem;

static void probe_mem(HwMem *m) {
    memset(m, 0, sizeof *m);
    m->total_kb = m->avail_kb = m->swap_total_kb = m->swap_used_kb = HW_UNKNOWN;
#ifdef _SC_PAGESIZE
    m->page_kb = sysconf(_SC_PAGESIZE) / 1024;
#else
    m->page_kb = HW_UNKNOWN;
#endif

#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char k[64]; long long v; long long sfree = HW_UNKNOWN;
        while (fscanf(f, "%63[^:]: %lld kB\n", k, &v) == 2) {
            if      (!strcmp(k, "MemTotal"))     m->total_kb = v;
            else if (!strcmp(k, "MemAvailable")) m->avail_kb = v;
            else if (!strcmp(k, "SwapTotal"))    m->swap_total_kb = v;
            else if (!strcmp(k, "SwapFree"))     sfree = v;
        }
        fclose(f);
        if (m->swap_total_kb >= 0 && sfree >= 0) m->swap_used_kb = m->swap_total_kb - sfree;
        return;
    }
#endif
    /* POSIX fallback: total only. MemAvailable has no portable equivalent, and guessing
     * it from free+cache is exactly the kind of invented number this tool refuses to emit. */
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long long pages = sysconf(_SC_PHYS_PAGES);
    if (pages > 0) m->total_kb = pages * (sysconf(_SC_PAGESIZE) / 1024);
#endif
}

/* ------------------------------------------------------------------- gpu */

typedef struct {
    char  node[NAME_MAX + 1];   /* a DRM node name is a dirent name; size it as one */
    char  vendor[32];
    unsigned vid, did;
    long long vram_total_mb;   /* -1 unknown */
    long long vram_used_mb;
    long long gtt_total_mb;
    long long gtt_used_mb;
    int   integrated;          /* 1 = shares system RAM, 0 = discrete, -1 = unknown */
} HwGpu;

static const char *vendor_name(unsigned vid) {
    switch (vid) {
        case 0x1002: return "AMD";
        case 0x10de: return "NVIDIA";
        case 0x8086: return "Intel";
        case 0x13b5: return "ARM";
        case 0x5143: return "Qualcomm";
        default:     return "unknown";
    }
}

static int probe_gpus(HwGpu *g, int max) {
    int n = 0;
#ifdef __linux__
    DIR *d = opendir("/sys/class/drm");
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < max) {
        /* cardN only -- skip cardN-HDMI-A-1 connector nodes */
        if (strncmp(e->d_name, "card", 4) != 0) continue;
        const char *p = e->d_name + 4;
        if (!*p) continue;
        int digits = 1;
        for (const char *q = p; *q; q++) if (*q < '0' || *q > '9') { digits = 0; break; }
        if (!digits) continue;

        char base[512], path[768], buf[64];
        snprintf(base, sizeof base, "/sys/class/drm/%s/device", e->d_name);
        snprintf(path, sizeof path, "%s/vendor", base);
        if (!read_str(path, buf, sizeof buf)) continue;   /* no vendor => not a real card */

        HwGpu *x = &g[n];
        memset(x, 0, sizeof *x);
        x->vram_total_mb = x->vram_used_mb = x->gtt_total_mb = x->gtt_used_mb = HW_UNKNOWN;
        x->integrated = HW_UNKNOWN;
        snprintf(x->node, sizeof x->node, "%s", e->d_name);
        x->vid = (unsigned)strtoul(buf, NULL, 16);
        snprintf(path, sizeof path, "%s/device", base);
        if (read_str(path, buf, sizeof buf)) x->did = (unsigned)strtoul(buf, NULL, 16);
        snprintf(x->vendor, sizeof x->vendor, "%s", vendor_name(x->vid));

        /* amdgpu exposes real occupancy; other drivers mostly do not. */
        snprintf(path, sizeof path, "%s/mem_info_vram_total", base);
        long long v = read_ll(path); if (v > 0) x->vram_total_mb = v / (1024 * 1024);
        snprintf(path, sizeof path, "%s/mem_info_vram_used", base);
        v = read_ll(path);           if (v > 0) x->vram_used_mb  = v / (1024 * 1024);
        snprintf(path, sizeof path, "%s/mem_info_gtt_total", base);
        v = read_ll(path);           if (v > 0) x->gtt_total_mb  = v / (1024 * 1024);
        snprintf(path, sizeof path, "%s/mem_info_gtt_used", base);
        v = read_ll(path);           if (v > 0) x->gtt_used_mb   = v / (1024 * 1024);

        /* Integrated-vs-discrete. GTT is a system-RAM aperture, so a card reporting a GTT
         * pool is sharing host memory -- that is the strongest available signal and it is
         * driver-reported, not inferred.
         *
         * Absent GTT we fall back to vendor. NVIDIA ships no integrated GPU in any class
         * that exposes a DRM card node here, so NVIDIA-without-GTT is discrete. Note we
         * must NOT gate this on vram_total_mb: the proprietary nvidia driver does not
         * export amdgpu's mem_info_* sysfs files at all, so that field stays unknown on
         * exactly the cards we are trying to classify. (Found by running this probe on a
         * box with both an RTX 4070 and an AMD iGPU: it reported UNIFIED instead of MIXED.)
         *
         * Everything else stays unknown rather than being guessed from a device id. */
        if (x->gtt_total_mb > 0)   x->integrated = 1;
        else if (x->vid == 0x10de) x->integrated = 0;

        n++;
    }
    closedir(d);
#else
    (void)g; (void)max;
#endif
    return n;
}

/* --------------------------------------------------------------- backends */

typedef struct { int cuda, hip, vulkan, metal, opencl; } HwBackends;

static void probe_backends(HwBackends *b) {
    memset(b, 0, sizeof *b);
    b->cuda   = have_lib("libcuda.so.1")   || have_lib("libcuda.so");
    b->hip    = have_lib("libamdhip64.so") || file_exists("/opt/rocm");
    b->vulkan = have_lib("libvulkan.so.1") || have_lib("libvulkan.so");
    b->opencl = have_lib("libOpenCL.so.1") || have_lib("libOpenCL.so");
#ifdef __APPLE__
    b->metal = file_exists("/System/Library/Frameworks/Metal.framework");
#endif
}

/* ------------------------------------------------------------------ disk */

/* Measure what the storage actually delivers at expert-sized blocks. This is the number
 * that decides whether streaming is viable, and it is the one most often assumed.
 *
 * O_DIRECT where available, so we time the device and not the page cache. Random offsets
 * across the whole file, because expert access is scattered by routing, not sequential. */

typedef struct { double gbs_2mb_qd1, gbs_2mb_qdN, gbs_19mb_qd1, gbs_19mb_qdN; int qdN; int direct; } HwDisk;

static double time_reads(int fd, size_t blk, int iters, long long span, int nthreads, int *ok) {
    char *bufs[64];
    if (nthreads > 64) nthreads = 64;
    for (int i = 0; i < nthreads; i++) {
        if (posix_memalign((void **)&bufs[i], 4096, blk) != 0) {
            for (int j = 0; j < i; j++) free(bufs[j]);
            *ok = 0; return 0;
        }
    }
    long long maxoff = span - (long long)blk;
    if (maxoff <= 0) { for (int i = 0; i < nthreads; i++) free(bufs[i]); *ok = 0; return 0; }

    double t0 = now_s();
    long long total = 0;
    int failed = 0;
#ifdef _OPENMP
#   pragma omp parallel for num_threads(nthreads) reduction(+:total) reduction(|:failed) schedule(static)
#endif
    for (int i = 0; i < iters; i++) {
#ifdef _OPENMP
        int t = omp_get_thread_num() % nthreads;
#else
        int t = 0;
#endif
        /* deterministic scatter: cheap LCG, offsets 4K-aligned for O_DIRECT */
        unsigned long long h = (unsigned long long)i * 6364136223846793005ULL + 1442695040888963407ULL;
        long long off = (long long)((h >> 16) % (unsigned long long)maxoff) & ~4095LL;
        ssize_t r = pread(fd, bufs[t], blk, off);
        if (r > 0) total += r; else failed |= 1;
    }
    double dt = now_s() - t0;
    for (int i = 0; i < nthreads; i++) free(bufs[i]);
    *ok = !failed && dt > 0;
    return (*ok) ? (double)total / 1e9 / dt : 0.0;
}

static int probe_disk(HwDisk *d, const char *path, int quick) {
    memset(d, 0, sizeof *d);
    d->qdN = 8;
    if (!path) return 0;

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < (off_t)(64 * 1024 * 1024)) return 0;

    int fd = -1;
#ifdef O_DIRECT
    fd = open(path, O_RDONLY | O_DIRECT);
    d->direct = (fd >= 0);
#endif
    if (fd < 0) { fd = open(path, O_RDONLY); d->direct = 0; }
    if (fd < 0) return 0;

    int iters = quick ? 24 : 96, ok = 0;
    long long span = (long long)st.st_size;

    d->gbs_2mb_qd1  = time_reads(fd, 2u  * 1024 * 1024, iters, span, 1,       &ok);
    d->gbs_2mb_qdN  = time_reads(fd, 2u  * 1024 * 1024, iters, span, d->qdN,  &ok);
    d->gbs_19mb_qd1 = time_reads(fd, 19u * 1024 * 1024, iters / 2, span, 1,      &ok);
    d->gbs_19mb_qdN = time_reads(fd, 19u * 1024 * 1024, iters / 2, span, d->qdN, &ok);

    close(fd);
    return 1;
}

/* ------------------------------------------------------------------ plan */

/* Turn measurements into an engine configuration. Everything here is derived from a
 * number we actually took; where we lack the number we say so instead of defaulting. */

static void emit_plan(const HwCpu *c, const HwMem *m, const HwGpu *g, int ngpu,
                      const HwBackends *b, const HwDisk *dk, int have_disk) {
    puts("");
    puts("ENGINE PLAN  (derived from the measurements above)");
    puts("-------------------------------------------------");

    /* memory tier: how many bytes may the expert cache claim? */
    long long avail_mb = (m->avail_kb > 0) ? m->avail_kb / 1024 : HW_UNKNOWN;
    if (avail_mb == HW_UNKNOWN) {
        puts("  cache budget   : UNKNOWN — MemAvailable not readable on this platform.");
        puts("                   Set it explicitly; do not let the engine guess.");
    } else {
        /* Leave half of available memory to the rest of the system. An expert cache is
         * anonymous memory: under pressure it is OOM-killed, not evicted, so the headroom
         * is not optional. */
        long long budget = avail_mb / 2;
        printf("  cache budget   : %lld MB   (50%% of %lld MB MemAvailable)\n", budget, avail_mb);
        if (m->swap_used_kb > 0 && m->swap_total_kb > 0) {
            double sw = 100.0 * m->swap_used_kb / m->swap_total_kb;
            if (sw > 25.0)
                printf("  ** WARNING     : swap is %.0f%% used. The box is already over-committed;\n"
                       "                   an expert cache here will be swapped, not cached.\n", sw);
        }
        puts("  cgroup          : REQUIRED — MemoryMax=<budget> and MemorySwapMax=0.");
        puts("                    The cache is anon memory; without a cap the OOM killer");
        puts("                    is your eviction policy.");
    }

    /* memory topology: the decision that inverts */
    int unified = 0, discrete = 0;
    for (int i = 0; i < ngpu; i++) {
        if (g[i].integrated == 1) unified = 1;
        else if (g[i].integrated == 0) discrete = 1;
    }
    if (unified && !discrete) {
        puts("  topology       : UNIFIED memory (GPU shares system RAM).");
        puts("                   -> Do NOT stage experts in host RAM before the GPU:");
        puts("                      same DIMMs, same controller, zero bandwidth gained.");
        puts("                   -> Two tiers only: storage -> unified memory.");
        puts("                   -> Prefer routing COMPUTE to the idle unit over moving weights.");
    } else if (discrete && !unified) {
        puts("  topology       : DISCRETE GPU (VRAM behind PCIe).");
        puts("                   -> A host-RAM staging tier is worth it: it removes a disk read.");
        puts("                   -> Three tiers: storage -> host RAM -> VRAM.");
    } else if (unified && discrete) {
        puts("  topology       : MIXED (integrated + discrete present).");
        puts("                   -> Plan per device; the tiering differs for each.");
    } else {
        puts("  topology       : UNKNOWN — no GPU memory pools readable.");
    }

    /* compute backend */
    printf("  backend        : ");
    if (b->cuda)        puts("CUDA (libcuda present)");
    else if (b->hip)    puts("HIP/ROCm present — verify the specific gfx target is supported");
    else if (b->vulkan) puts("Vulkan (portable GPU path; the widest-compatibility option)");
    else if (b->metal)  puts("Metal");
    else                puts("CPU only — no GPU runtime detected");
    if (!b->cuda && b->vulkan) {
        puts("                   NOTE: most MoE-offload work upstream is CUDA-only. On this");
        puts("                   machine a Vulkan or CPU path is the compatible one.");
    }

    /* threads */
    if (c->cores_online > 0) {
        long t = c->cores_online > 4 ? c->cores_online - 2 : c->cores_online;
        printf("  threads        : %ld  (of %ld online; leave headroom — the cache path\n"
               "                   adds CPU work and co-tenant services need the cores)\n",
               t, c->cores_online);
    }

    /* disk-derived block/QD policy and the throughput ceiling */
    if (have_disk) {
        double best2  = dk->gbs_2mb_qdN  > dk->gbs_2mb_qd1  ? dk->gbs_2mb_qdN  : dk->gbs_2mb_qd1;
        double best19 = dk->gbs_19mb_qdN > dk->gbs_19mb_qd1 ? dk->gbs_19mb_qdN : dk->gbs_19mb_qd1;
        printf("  io policy      : read whole experts in ONE request; queue depth %d.\n", dk->qdN);
        if (dk->gbs_2mb_qd1 > 0.01) {
            double gain = dk->gbs_2mb_qdN / dk->gbs_2mb_qd1;
            printf("                   small-block parallelism gain: %.2fx (2MB QD1 -> QD%d)\n",
                   gain, dk->qdN);
            if (gain > 1.3)
                puts("                   -> concurrency matters here: issue expert loads in parallel.");
        }
        if (!dk->direct)
            puts("                   NOTE: O_DIRECT unavailable; figures include page cache and\n"
                 "                   therefore overstate cold-read performance.");
        puts("");
        puts("  THROUGHPUT CEILING from measured disk (tokens/s at 0% cache hit):");
        puts("    demand/token = activations x bytes_per_expert;  ceiling = measured_GBs / demand");
        printf("    %-34s %8s %8s\n", "workload (acts x expert size)", "@2MB", "@19MB");
        struct { const char *n; int acts; } w[] = {
            { "Qwen3.6-35B-A3B  (320 x 1.9MB)", 320 },
            { "DeepSeek-V4-Flash(252 x 15MB) ", 252 },
            { "GLM-5.2          (608 x 19MB) ", 608 },
        };
        double persz[3] = { 1.9e6, 15.2e6, 18.9e6 };
        for (int i = 0; i < 3; i++) {
            double demand = w[i].acts * persz[i];                 /* bytes per token */
            double c2 = best2  * 1e9 / demand, c19 = best19 * 1e9 / demand;
            printf("    %-34s %8.2f %8.2f\n", w[i].n, c2, c19);
        }
        puts("    (cold-cache floor. A 90% hit rate multiplies these by ~10.)");
        puts("    THIS IS AN I/O CEILING ONLY. Actual throughput is min(this, compute).");
        puts("    A model with many active parameters will bind on compute long before");
        puts("    the disk -- do not read these as predicted tokens/s.");
    } else {
        puts("  io policy      : UNKNOWN — no probe file given.");
        puts("                   Re-run with --probe-file <a model file >64MB> to measure it.");
        puts("                   Do not plan streaming against an unmeasured disk.");
    }
    puts("");
}

/* ------------------------------------------------------------------ main */

static void print_human(const HwCpu *c, const HwMem *m, const HwGpu *g, int ngpu,
                        const HwBackends *b, const HwDisk *dk, int have_disk) {
    puts("HARDWARE PROBE");
    puts("--------------");
    printf("  arch           : %s\n", c->arch);
    printf("  cores online   : %ld\n", c->cores_online);
    printf("  isa            : %s\n", c->isa);
    if (c->l3_kb > 0) printf("  l3 cache       : %ld KB\n", c->l3_kb);

    if (m->total_kb > 0) printf("  memory total   : %lld MB\n", m->total_kb / 1024);
    if (m->avail_kb > 0) printf("  memory avail   : %lld MB\n", m->avail_kb / 1024);
    else                 puts ("  memory avail   : unknown (no MemAvailable on this platform)");
    if (m->swap_total_kb > 0)
        printf("  swap           : %lld / %lld MB used\n",
               m->swap_used_kb / 1024, m->swap_total_kb / 1024);

    if (!ngpu) puts("  gpu            : none detected");
    for (int i = 0; i < ngpu; i++) {
        printf("  gpu[%d]         : %s %04x:%04x (%s)", i, g[i].vendor, g[i].vid, g[i].did, g[i].node);
        if (g[i].integrated == 1)      printf("  [integrated / unified memory]");
        else if (g[i].integrated == 0) printf("  [discrete]");
        puts("");
        if (g[i].vram_total_mb > 0)
            printf("                   vram %lld / %lld MB used\n", g[i].vram_used_mb, g[i].vram_total_mb);
        if (g[i].gtt_total_mb > 0)
            printf("                   gtt  %lld / %lld MB used\n", g[i].gtt_used_mb, g[i].gtt_total_mb);
    }
    printf("  backends       : %s%s%s%s%s\n",
           b->cuda ? "cuda " : "", b->hip ? "hip " : "", b->vulkan ? "vulkan " : "",
           b->metal ? "metal " : "", b->opencl ? "opencl " : "");

    if (have_disk) {
        printf("  disk (%s):\n", dk->direct ? "O_DIRECT, true device reads" : "buffered — includes page cache");
        printf("                   2MB  blocks: %6.2f GB/s QD1   %6.2f GB/s QD%d\n",
               dk->gbs_2mb_qd1, dk->gbs_2mb_qdN, dk->qdN);
        printf("                   19MB blocks: %6.2f GB/s QD1   %6.2f GB/s QD%d\n",
               dk->gbs_19mb_qd1, dk->gbs_19mb_qdN, dk->qdN);
    } else {
        puts("  disk           : not measured (pass --probe-file)");
    }
}

static void print_json(const HwCpu *c, const HwMem *m, const HwGpu *g, int ngpu,
                       const HwBackends *b, const HwDisk *dk, int have_disk) {
    printf("{\n");
    printf("  \"arch\": \"%s\",\n  \"cores_online\": %ld,\n  \"isa\": \"%s\",\n",
           c->arch, c->cores_online, c->isa);
    printf("  \"mem_total_mb\": %lld,\n  \"mem_avail_mb\": %lld,\n",
           m->total_kb > 0 ? m->total_kb / 1024 : -1,
           m->avail_kb > 0 ? m->avail_kb / 1024 : -1);
    printf("  \"swap_used_mb\": %lld,\n",
           m->swap_used_kb > 0 ? m->swap_used_kb / 1024 : -1);
    printf("  \"backends\": {\"cuda\": %d, \"hip\": %d, \"vulkan\": %d, \"metal\": %d, \"opencl\": %d},\n",
           b->cuda, b->hip, b->vulkan, b->metal, b->opencl);
    printf("  \"gpus\": [");
    for (int i = 0; i < ngpu; i++)
        printf("%s{\"vendor\": \"%s\", \"pci\": \"%04x:%04x\", \"integrated\": %d, "
               "\"vram_total_mb\": %lld, \"gtt_total_mb\": %lld}",
               i ? ", " : "", g[i].vendor, g[i].vid, g[i].did, g[i].integrated,
               g[i].vram_total_mb, g[i].gtt_total_mb);
    printf("],\n");
    if (have_disk)
        printf("  \"disk\": {\"direct\": %d, \"qd\": %d, \"gbs_2mb_qd1\": %.3f, \"gbs_2mb_qdn\": %.3f, "
               "\"gbs_19mb_qd1\": %.3f, \"gbs_19mb_qdn\": %.3f}\n",
               dk->direct, dk->qdN, dk->gbs_2mb_qd1, dk->gbs_2mb_qdN,
               dk->gbs_19mb_qd1, dk->gbs_19mb_qdN);
    else
        printf("  \"disk\": null\n");
    printf("}\n");
}

int main(int argc, char **argv) {
    const char *probe_file = NULL;
    int as_json = 0, quick = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--json"))  as_json = 1;
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--probe-file") && i + 1 < argc) probe_file = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [--json] [--quick] [--probe-file PATH]\n"
                   "  Probes CPU/memory/GPU/backends, optionally measures storage read\n"
                   "  bandwidth at expert-sized blocks, and prints an engine plan.\n"
                   "  Read-only: opens files for reading, changes nothing.\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s (try --help)\n", argv[i]);
            return 2;
        }
    }

    HwCpu cpu; HwMem mem; HwGpu gpu[MAXGPU]; HwBackends be; HwDisk disk;
    probe_cpu(&cpu);
    probe_mem(&mem);
    int ngpu = probe_gpus(gpu, MAXGPU);
    probe_backends(&be);
    int have_disk = probe_disk(&disk, probe_file, quick);

    if (as_json) print_json(&cpu, &mem, gpu, ngpu, &be, &disk, have_disk);
    else {
        print_human(&cpu, &mem, gpu, ngpu, &be, &disk, have_disk);
        emit_plan(&cpu, &mem, gpu, ngpu, &be, &disk, have_disk);
    }
    return 0;
}
