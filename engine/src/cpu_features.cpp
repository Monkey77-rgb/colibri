/* cpu_features.c — see cpu_features.h for why this is runtime and not #ifdef. */
#define _GNU_SOURCE
#include "cpu_features.h"
#include <stdio.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define COLI_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
static void coli_cpuid(int leaf, int sub, int r[4]) { __cpuidex(r, leaf, sub); }
static uint64_t coli_xgetbv(void) { return _xgetbv(0); }
#else
#include <cpuid.h>
static void coli_cpuid(int leaf, int sub, int r[4]) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static uint64_t coli_xgetbv(void) {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
#endif
#endif

#if defined(__aarch64__) || defined(__arm__)
#define COLI_ARM 1
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
static int coli_sysctl_yes(const char *name) {
    int v = 0; size_t n = sizeof v;
    return sysctlbyname(name, &v, &n, NULL, 0) == 0 && v != 0;
}
#endif
#endif

static uint32_t g_feat;
static int      g_done;
static char     g_name[128];

static void coli_detect(void) {
    uint32_t f = 0;
    snprintf(g_name, sizeof g_name, "unknown");

#if defined(COLI_X86)
    int r[4];
    coli_cpuid(0, 0, r);
    int maxleaf = r[0];

    /* Brand string, leaves 0x80000002..4 */
    coli_cpuid(0x80000000, 0, r);
    if ((unsigned)r[0] >= 0x80000004) {
        char b[49]; int *p = (int *)b;
        for (int i = 0; i < 3; i++) { coli_cpuid(0x80000002 + i, 0, r); memcpy(p + i*4, r, 16); }
        b[48] = 0;
        char *s = b; while (*s == ' ') s++;
        snprintf(g_name, sizeof g_name, "%s", s);
    }

    coli_cpuid(1, 0, r);
    if (r[3] & (1 << 26)) f |= COLI_CPU_SSE2;
    if (r[2] & (1 << 29)) f |= COLI_CPU_F16C;
    if (r[2] & (1 << 12)) f |= COLI_CPU_FMA;

    /* AVX state must be enabled by the OS, or the instructions fault. Checking
     * only the CPUID bit is how a binary SIGILLs on a machine that reports the
     * feature -- the xgetbv check is not optional. */
    int osxsave = (r[2] & (1 << 27)) != 0;
    int avx_os = 0, avx512_os = 0;
    if (osxsave) {
        uint64_t xcr0 = coli_xgetbv();
        avx_os    = (xcr0 & 0x6) == 0x6;           /* XMM + YMM */
        avx512_os = (xcr0 & 0xE6) == 0xE6;         /* + opmask + ZMM hi256 + hi16 */
    }

    if (maxleaf >= 7) {
        coli_cpuid(7, 0, r);
        if (avx_os    && (r[1] & (1 << 5)))  f |= COLI_CPU_AVX2;
        if (avx512_os && (r[1] & (1 << 16))) f |= COLI_CPU_AVX512F;
        if (avx512_os && (r[1] & (1 << 30))) f |= COLI_CPU_AVX512BW;
        if (avx512_os && (r[1] & (1u << 31)))f |= COLI_CPU_AVX512VL;
        if (avx512_os && (r[2] & (1 << 11))) f |= COLI_CPU_AVX512VNNI;

        coli_cpuid(7, 1, r);
        if (avx512_os && (r[0] & (1 << 5)))  f |= COLI_CPU_AVX512BF16;
        if (avx_os    && (r[0] & (1 << 4)))  f |= COLI_CPU_AVXVNNI;
    }
#elif defined(COLI_ARM)
    f |= COLI_CPU_NEON;                            /* baseline on aarch64 */
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
#if defined(HWCAP_ASIMDDP)
    if (hw & HWCAP_ASIMDDP) f |= COLI_CPU_DOTPROD;
#endif
#if defined(HWCAP_SVE)
    if (hw & HWCAP_SVE) f |= COLI_CPU_SVE;
#endif
    unsigned long hw2 = getauxval(AT_HWCAP2);
#if defined(HWCAP2_I8MM)
    if (hw2 & HWCAP2_I8MM) f |= COLI_CPU_I8MM;
#endif
    (void)hw2;
#elif defined(__APPLE__)
    /* Apple silicon: HWCAP does not exist, sysctl does. Every M-series has
     * dotprod; i8mm arrived with M2 (ARMv8.6), so it must be probed, not
     * assumed from "it's an M-chip". */
    if (coli_sysctl_yes("hw.optional.arm.FEAT_DotProd")) f |= COLI_CPU_DOTPROD;
    if (coli_sysctl_yes("hw.optional.arm.FEAT_I8MM"))    f |= COLI_CPU_I8MM;
    { size_t n = sizeof g_name; sysctlbyname("machdep.cpu.brand_string", g_name, &n, NULL, 0); }
#endif
#endif

    g_feat = f;
    g_done = 1;
}

uint32_t coli_cpu_features(void) { if (!g_done) coli_detect(); return g_feat; }
const char *coli_cpu_name(void) { if (!g_done) coli_detect(); return g_name; }

void coli_cpu_describe(char *buf, size_t cap) {
    uint32_t f = coli_cpu_features();
    static const struct { uint32_t bit; const char *nm; } t[] = {
        {COLI_CPU_AVX2,"avx2"},{COLI_CPU_FMA,"fma"},{COLI_CPU_F16C,"f16c"},
        {COLI_CPU_AVX512F,"avx512f"},{COLI_CPU_AVX512BW,"avx512bw"},
        {COLI_CPU_AVX512VL,"avx512vl"},{COLI_CPU_AVX512VNNI,"avx512vnni"},
        {COLI_CPU_AVX512BF16,"avx512bf16"},{COLI_CPU_AVXVNNI,"avxvnni"},
        {COLI_CPU_NEON,"neon"},{COLI_CPU_DOTPROD,"dotprod"},
        {COLI_CPU_I8MM,"i8mm"},{COLI_CPU_SVE,"sve"},
    };
    size_t o = 0;
    o += (size_t)snprintf(buf+o, o<cap?cap-o:0, "%s [", coli_cpu_name());
    int first = 1;
    for (unsigned i = 0; i < sizeof t/sizeof *t; i++)
        if (f & t[i].bit) { o += (size_t)snprintf(buf+o, o<cap?cap-o:0, "%s%s", first?"":" ", t[i].nm); first = 0; }
    o += (size_t)snprintf(buf+o, o<cap?cap-o:0, "]");
    if (cap) buf[cap-1] = 0;
}

uint64_t coli_cache_bytes(int level) {
#if defined(COLI_X86)
    /* Leaf 4 (Intel) / leaf 0x8000001D (AMD) share an encoding. Walk the
     * sub-leaves and take the first that reports this level as unified or data.
     * Returns 0 rather than a guess when the leaf is absent -- a wrong cache
     * size silently mistunes every tile in the engine. */
    int r[4];
    coli_cpuid(0, 0, r);
    int leaf = 4;
    coli_cpuid(0x80000000, 0, r);
    if ((unsigned)r[0] >= 0x8000001D) {
        coli_cpuid(0x80000000, 0, r);
        int amd = 0; coli_cpuid(0, 0, r);
        char v[13]; memcpy(v, &r[1], 4); memcpy(v+4, &r[3], 4); memcpy(v+8, &r[2], 4); v[12]=0;
        amd = (strcmp(v, "AuthenticAMD") == 0);
        if (amd) leaf = 0x8000001D;
    }
    for (int i = 0; i < 16; i++) {
        coli_cpuid(leaf, i, r);
        int type = r[0] & 0x1F;
        if (type == 0) break;                       /* no more caches */
        int lvl = (r[0] >> 5) & 0x7;
        if (lvl != level) continue;
        if (type != 1 && type != 3) continue;       /* data or unified only */
        uint64_t ways  = (uint64_t)((r[1] >> 22) & 0x3FF) + 1;
        uint64_t parts = (uint64_t)((r[1] >> 12) & 0x3FF) + 1;
        uint64_t line  = (uint64_t)(r[1] & 0xFFF) + 1;
        uint64_t sets  = (uint64_t)r[2] + 1;
        return ways * parts * line * sets;
    }
    return 0;
#elif defined(COLI_ARM) && defined(__APPLE__)
    const char *k = level==1 ? "hw.l1dcachesize" : level==2 ? "hw.l2cachesize" : "hw.l3cachesize";
    uint64_t v = 0; size_t n = sizeof v;
    if (sysctlbyname(k, &v, &n, NULL, 0) == 0) return v;
    return 0;
#else
    (void)level; return 0;
#endif
}
