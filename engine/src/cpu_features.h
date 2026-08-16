/* cpu_features.h — runtime ISA detection.
 *
 * WHY RUNTIME AND NOT -march=native. A binary built with -march=native runs only
 * on the machine that built it. This engine has to ship one binary that runs on
 * a Zen 5 desktop, a Zen 4 handheld, an Apple M-series laptop and an ARM phone,
 * so the ISA choice belongs at run time, in a dispatch table, not at compile
 * time in a #ifdef. Kernels are compiled for every ISA the target supports and
 * selected on first use.
 *
 * WHY NOT A MONOTONE "SCORE". The obvious design -- rank ISAs and always pick
 * the widest available -- is what ggml does (ggml-cpu/arch/x86/cpu-feats.cpp,
 * ggml_backend_cpu_x86_score(), a feature bitmask). It cannot express "AVX2 is
 * faster than AVX-512 VNNI for THIS shape", and we measured exactly that:
 *
 *   int8 GEMM, weights streaming from RAM, AMD 9800X3D, 2026-08-16
 *     n=1  VNNI 0.83x    n=2  0.78x    n=4  1.17x    n>=8  1.18-1.26x
 *
 * At n<4 the loop is bound by weight bytes (~70 GB/s DRAM ceiling), so the wider
 * ISA is strictly worse. A score that only goes up cannot choose it. Hence the
 * dispatch key here is (ISA, kernel shape), never ISA alone.
 */
#ifndef COLI_CPU_FEATURES_H
#define COLI_CPU_FEATURES_H

#include <stdint.h>
#include <stddef.h>

enum {
    COLI_CPU_SSE2        = 1u << 0,
    COLI_CPU_AVX2        = 1u << 1,
    COLI_CPU_F16C        = 1u << 2,
    COLI_CPU_FMA         = 1u << 3,
    COLI_CPU_AVX512F     = 1u << 4,
    COLI_CPU_AVX512BW    = 1u << 5,
    COLI_CPU_AVX512VL    = 1u << 6,
    COLI_CPU_AVX512VNNI  = 1u << 7,
    COLI_CPU_AVX512BF16  = 1u << 8,
    COLI_CPU_AVXVNNI     = 1u << 9,
    COLI_CPU_NEON        = 1u << 16,
    COLI_CPU_DOTPROD     = 1u << 17,   /* ARM sdot/udot */
    COLI_CPU_I8MM        = 1u << 18,   /* ARM smmla, the GEMM lever on aarch64 */
    COLI_CPU_SVE         = 1u << 19,
};

/* Bitmask of the above. Computed once, cached. Safe to call from any thread. */
uint32_t coli_cpu_features(void);

/* Human-readable, for the banner and for diagnostics that must record WHICH
 * kernel produced a number. A benchmark that does not say which ISA it ran is
 * not reproducible. */
const char *coli_cpu_name(void);
void coli_cpu_describe(char *buf, size_t cap);

/* Cache size in BYTES, 0 when unknown (never a guess -- a wrong cache size
 * silently mistunes every tile in the engine).
 *
 * ⚠ UNITS. This is the size ONE CORE SEES, which is what a per-thread tile must
 * fit in: per-core for L1/L2, shared for L3. It is NOT the socket total that
 * lscpu prints. On a 9800X3D this returns L1=48 KiB, L2=1 MiB, L3=96 MiB, where
 * lscpu says "384 KiB (8 instances)", "8 MiB (8 instances)", "96 MiB". Same
 * hardware, different question. Verified against lscpu on 2026-08-16.
 *
 * These figures decide whether a weight tile is cache-resident or streaming,
 * which is the largest single factor in kernel choice -- measured 1.28x
 * (resident) vs 0.78x (streaming) for the SAME VNNI kernel at n=2. A 9800X3D has
 * 96 MiB of L3 against a Z1 Extreme's 16 MiB, so a tile tuned on the desktop
 * will stream on the handheld. */
uint64_t coli_cache_bytes(int level);   /* level = 1, 2, 3 */

#endif
