/* gemm_q4k.cpp — see gemm_q4k.h for the numerics this kernel implements.
 *
 * Same -ffp-contract=off discipline as gemm_i8.cpp, and for the identical
 * reason: the reference and the dispatched kernel must do the SAME floating
 * point operations in the SAME order for a bit-exactness (mod summation
 * order across SIMD lanes) claim to mean anything. See gemm_i8.cpp's header
 * comment for the measured 76%-of-cells divergence this guards against.
 */
#define _GNU_SOURCE
#include "gemm_q4k.h"
#include <stdlib.h>
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include <string.h>
#include "q4k_shim.h"       /* coli_q4k_decode_scales/coli_q4k_qs -- the C
                             * shim over c/ggml_dequant.h's bit-exact Q4_K
                             * decode (gguf_scale_min_k4, f16_to_f32), reused
                             * rather than re-derived. See q4k_shim.h for why
                             * this file cannot include ggml_dequant.h itself
                             * (it drags in st.h/json.h, which are not
                             * C++-clean). */
#include "cpu_features.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define COLI_Q4K_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define COLI_Q4K_ARM 1
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

void coli_free_q4k(coli_w_q4k *w) {
    if (!w) return;
    if (w->owns && w->blocks) free((void*)w->blocks);
    w->blocks = nullptr; w->owns = 0;
}

/* ---- shared superblock decode --------------------------------------------
 * One Q4_K super-block (COLI_Q4K_BLOCK_BYTES=144 bytes, COLI_Q4K_SUPERBLOCK=256
 * weights) -> nib[256] (unsigned 4-bit codes, 0..15, NO offset -- Q4_K's own
 * scale/min pair, not this engine's q+8) plus db[8]/mb[8], the 8 effective
 * (d*sc, dmin*m) pairs, one per 32-weight sub-block (from q4k_shim.h's
 * coli_q4k_decode_scales). Element order in `nib` is the SAME order
 * dequantize_row_q4_K writes to `y` (see c/ggml_dequant.h's gguf_dequant_q4_K):
 * four 64-element groups, each low-nibbles-of-32-bytes then
 * high-nibbles-of-the-SAME-32-bytes -- which happens to require no interleave
 * step at all, unlike this engine's own int4 packing (coli_quantize_w4
 * interleaves nibble k=2j/2j+1 per byte j; Q4_K instead puts all 32 low
 * nibbles of a segment before all 32 high nibbles of it), which is exactly
 * why this kernel cannot reuse gemm_i8.cpp's i4_unpack_row -- the two on-disk
 * layouts are legitimately different, not just differently named. */
static inline void q4k_decode_block(const void *blk, uint8_t nib[256],
                                    float db[8], float mb[8]) {
    coli_q4k_decode_scales(blk, db, mb);
    const unsigned char *q = coli_q4k_qs(blk);
    for (int j = 0; j < 4; j++) {
        uint8_t *lo = nib + j*64, *hi = nib + j*64 + 32;
        for (int i = 0; i < 32; i++) { lo[i] = (uint8_t)(q[i] & 0x0F); hi[i] = (uint8_t)(q[i] >> 4); }
        q += 32;
    }
}

/* ------------------------------------------------------------ reference --- */
static inline int32_t dot16_ref(const int8_t *a, const uint8_t *nib) {
    int32_t s = 0;
    for (int i = 0; i < 16; i++) s += (int32_t)a[i] * (int32_t)nib[i];
    return s;
}

void coli_gemm_q4k_ref(float *y, const coli_a_i8 *a, const coli_w_q4k *w) {
    int64_t I = w->I, O = w->O, nsb = I / COLI_Q4K_SUPERBLOCK, anb = I / COLI_ABLK;
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *rowblk = w->blocks + (size_t)o*(size_t)nsb*COLI_Q4K_BLOCK_BYTES;
        for (int r = 0; r < a->n; r++) {
            const int8_t  *xr = a->q    + (int64_t)r*I;
            const float   *as = a->scale+ (int64_t)r*anb;
            const int32_t *su = a->sum  + (int64_t)r*anb;
            float acc = 0.f;
            for (int64_t sb = 0; sb < nsb; sb++) {
                const uint8_t *blk = rowblk + (size_t)sb*COLI_Q4K_BLOCK_BYTES;
                uint8_t nib[256]; float db[8], mb[8];
                q4k_decode_block(blk, nib, db, mb);
                int64_t base_ablk = sb * (COLI_Q4K_SUPERBLOCK/COLI_ABLK);
                for (int b = 0; b < 8; b++) {
                    int64_t elem = sb*COLI_Q4K_SUPERBLOCK + b*32;
                    int64_t ab0 = base_ablk + 2*b, ab1 = ab0 + 1;
                    int32_t d0 = dot16_ref(xr+elem,    nib+b*32);
                    int32_t d1 = dot16_ref(xr+elem+16, nib+b*32+16);
                    acc += as[ab0]*(db[b]*(float)d0 - mb[b]*(float)su[ab0]);
                    acc += as[ab1]*(db[b]*(float)d1 - mb[b]*(float)su[ab1]);
                }
            }
            y[(int64_t)r*O + o] = acc;
        }
    }
}

/* ------------------------------------------------------------------ AVX2 -- */
#if defined(COLI_Q4K_X86)
/* 16-wide int8(signed) . uint8(0..15, unsigned) dot -> int32. Same widen +
 * madd shape as gemm_i8.cpp's dot_blk_avx2, minus that kernel's -128 offset
 * undo: a Q4_K nibble is a native unsigned code, not an offset-encoded one,
 * so there is nothing to undo here. Max |product| = 127*15 = 1905, summed
 * over 16 by one _mm256_madd_epi16 (which reduces PAIRS into int32 lanes, so
 * the widest partial sum is 2*1905 before the final int32 adds) -- nowhere
 * near int32 overflow, no int16 accumulation anywhere (same overflow
 * discipline gemm_i8.h documents for its own AVX2 path). */
static inline int32_t dot16_u4_avx2(const int8_t *a, const uint8_t *nib) {
    __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)nib));
    __m256i p  = _mm256_madd_epi16(va, vb);
    __m128i s  = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    __m128i h  = _mm_unpackhi_epi64(s, s); s = _mm_add_epi32(s, h);
    h = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)); s = _mm_add_epi32(s, h);
    return _mm_cvtsi128_si32(s);
}
/* Decode one segment's 32 bytes into lo[32]/hi[32] with AVX2 mask+shift --
 * same "shift 16-bit lanes then mask 0x0F" trick gemm_i8.cpp's SSE int4
 * unpack uses, at 32 bytes/call instead of 16. Spillover bits from
 * _mm256_srli_epi16 crossing a byte boundary are removed by the AND, so this
 * is exact, not approximate -- the existing kernel already relies on this. */
static inline void q4k_unpack_seg_avx2(const uint8_t *seg32, uint8_t *lo, uint8_t *hi) {
    const __m256i m = _mm256_set1_epi8(0x0F);
    __m256i raw = _mm256_loadu_si256((const __m256i*)seg32);
    __m256i vlo = _mm256_and_si256(raw, m);
    __m256i vhi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m);
    _mm256_storeu_si256((__m256i*)lo, vlo);
    _mm256_storeu_si256((__m256i*)hi, vhi);
}
static inline void q4k_decode_block_avx2(const void *blk, uint8_t nib[256],
                                         float db[8], float mb[8]) {
    coli_q4k_decode_scales(blk, db, mb);
    const unsigned char *q = coli_q4k_qs(blk);
    for (int j = 0; j < 4; j++) {
        q4k_unpack_seg_avx2(q, nib + j*64, nib + j*64 + 32);
        q += 32;
    }
}
/* MUST accumulate in the SAME order as coli_gemm_q4k_ref for a given (r,o):
 * a fixed r's contribution is one continuously-growing `acc`, sb ascending,
 * b ascending within each sb -- never "sum this superblock, then add the
 * partial sums together". Those two groupings are NOT bit-identical in
 * floating point (pairwise-style summation rounds differently from strictly
 * sequential summation), and an EARLIER version of this kernel wrote a fresh
 * per-superblock partial sum into `y` and added the next superblock's partial
 * sum to it -- correct algebra, wrong associativity. Caught by
 * tests/test_gemm_q4k.c's kernel-vs-ref bound (1e-5), which is NOT an
 * arbitrary threshold: gemm_i8.cpp's own header explains why this codebase
 * treats "the dispatched kernel disagrees with its scalar reference by more
 * than float noise" as a dispatch bug, not a rounding footnote (the 76%-of-
 * cells FMA-contraction incident). The persistent `acc[]` below (one float
 * per activation row, held across the WHOLE sb loop) is what restores the
 * single continuous chain -- measured max relative diff 0.0 after this fix,
 * where the per-superblock-writeback version measured 7.238e-05 (I=2048)
 * and 1.261e-03 (I=768) on this box, 2026-09-13. */
static void gemm_q4k_avx2(float *y, const coli_a_i8 *a, const coli_w_q4k *w) {
    int64_t I = w->I, O = w->O, nsb = I / COLI_Q4K_SUPERBLOCK, anb = I / COLI_ABLK;
    enum { ACC_STACK = 64 };   /* covers every batch shape this codebase uses
                               * (decode n=1, COLI_GEMM_I4_MIN_WIDE-class n) */
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *rowblk = w->blocks + (size_t)o*(size_t)nsb*COLI_Q4K_BLOCK_BYTES;
        float acc_stack[ACC_STACK];
        float *acc = (a->n <= ACC_STACK) ? acc_stack : (float*)malloc(sizeof(float)*(size_t)a->n);
        for (int r = 0; r < a->n; r++) acc[r] = 0.f;
        /* Decode each superblock ONCE per row, reused across every activation
         * row (a->n) below -- same amortization idea as gemm_i4_wide's
         * i4_unpack_row, sized to fit on the stack (256 B) instead of needing
         * a per-thread pool, since it only needs to outlive one `o` iteration. */
        for (int64_t sb = 0; sb < nsb; sb++) {
            const uint8_t *blk = rowblk + (size_t)sb*COLI_Q4K_BLOCK_BYTES;
            uint8_t nib[256]; float db[8], mb[8];
            q4k_decode_block_avx2(blk, nib, db, mb);
            int64_t base_ablk = sb * (COLI_Q4K_SUPERBLOCK/COLI_ABLK);
            int64_t elem0 = sb*COLI_Q4K_SUPERBLOCK;
            for (int r = 0; r < a->n; r++) {
                const int8_t  *xr = a->q    + (int64_t)r*I;
                const float   *as = a->scale+ (int64_t)r*anb;
                const int32_t *su = a->sum  + (int64_t)r*anb;
                for (int b = 0; b < 8; b++) {
                    int64_t elem = elem0 + b*32;
                    int64_t ab0 = base_ablk + 2*b, ab1 = ab0 + 1;
                    int32_t d0 = dot16_u4_avx2(xr+elem,    nib+b*32);
                    int32_t d1 = dot16_u4_avx2(xr+elem+16, nib+b*32+16);
#if defined(COLI_BREAK_Q4K)
                    /* Negative control, build-time only, never shipped: drops
                     * the min-term correction on ONE of the two dispatched
                     * kernel's implementations only (coli_gemm_q4k_ref stays
                     * correct), so kernel-vs-ref is the arm that must catch
                     * it -- same shape as gemm_i8.cpp's COLI_BREAK_WIDE /
                     * COLI_BREAK_I4: perturb ONE implementation, because
                     * corrupting a shared INPUT (like a test poking the raw
                     * block bytes both kernels read) would leave kernel and
                     * ref agreeing with each other and disagreeing with
                     * `truth` by an amount easily lost in ordinary int8-
                     * activation quantization noise -- exactly what an
                     * earlier version of this control measured: 0 cells over
                     * the rtol+atol bound despite a real corrupted nibble. */
                    acc[r] += as[ab0]*(db[b]*(float)d0);
                    acc[r] += as[ab1]*(db[b]*(float)d1);
#else
                    acc[r] += as[ab0]*(db[b]*(float)d0 - mb[b]*(float)su[ab0]);
                    acc[r] += as[ab1]*(db[b]*(float)d1 - mb[b]*(float)su[ab1]);
#endif
                }
            }
        }
        for (int r = 0; r < a->n; r++) y[(int64_t)r*O + o] = acc[r];
        if (acc != acc_stack) free(acc);
    }
}
#endif /* COLI_Q4K_X86 */

/* ---------------------------------------------------------------- dispatch */
void coli_gemm_q4k(float *y, const coli_a_i8 *a, const coli_w_q4k *w) {
#if defined(COLI_Q4K_X86)
    if (coli_cpu_features() & COLI_CPU_AVX2) { gemm_q4k_avx2(y, a, w); return; }
#endif
    coli_gemm_q4k_ref(y, a, w);
}

const char *coli_gemm_q4k_kernel(int n) {
    (void)n;
#if defined(COLI_Q4K_X86)
    if (coli_cpu_features() & COLI_CPU_AVX2) return "avx2-q4k";
#endif
    return "scalar-q4k";
    /* No AVX-512 VNNI path in phase 1 -- NOT because this hardware lacks it.
     * CORRECTED 2026-09-13: an earlier version of this comment claimed the
     * 9800X3D "exposes AVX2 and AVX512F but NOT AVX512VNNI", from
     * `lscpu | grep -iE "avx512vnni|avx2|avx512f"` finding nothing -- a
     * pattern that cannot match lscpu's actual spelling, `avx512_vnni`
     * (underscore). Re-checked: `lscpu | grep Flags | tr ' ' '\n' | grep -i
     * vnni` returns both `avx_vnni` and `avx512_vnni`, and coli_cpu_features()
     * itself reports COLI_CPU_AVX512VNNI set on this box (see gemm_i8.cpp's
     * own `test` output: "avx512vnni-wide" / "avx512vnni-i4-wide" both
     * dispatch here). The real reason there is no VNNI kernel in this file is
     * time budget for this phase, not a hardware gap -- gemm_i4_wide's VNNI
     * path (gemm_i8.cpp) is the template to follow: unpack nibbles into an
     * unsigned scratch once per row, dot against int8 activations with
     * _mm512_dpbusd_epi32 the same way, fold in the two u8 nibble->weight
     * scale/min terms this kernel adds beyond int4's single scale. Tracked as
     * follow-up, not implemented here -- see this task's final report for
     * the corrected claim (a stale "hardware doesn't have it" would have
     * been carried forward as fact by the next person who read this file). */
}

void coli_gemm_q4k_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                         const coli_w_q4k *const *ws, int cnt) {
    /* One matrix at a time through the standard dispatch, each on a one-row
     * view of the shared activation buffer -- same fallback shape
     * coli_gemm_i4_multi uses when its wide path is unavailable. A fused
     * multi-matrix AVX2 region (like gemm_i4_wide's pooled unpack) is future
     * work once native Q4_K experts are the common case; today's callers
     * (phase 1) only exercise this through the differential test. */
    for (int j = 0; j < cnt; j++) {
        const int64_t I = ws[j]->I, anb = I / COLI_ABLK;
        const int r = arow[j];
        coli_a_i8 v;
        v.q = a->q + (int64_t)r*I; v.scale = a->scale + (int64_t)r*anb; v.sum = a->sum + (int64_t)r*anb;
        v.n = 1; v.I = I;
        coli_gemm_q4k(ys[j], &v, ws[j]);
    }
}
