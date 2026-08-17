/* gemm_i8.c — see gemm_i8.h for the measurements this dispatch is built on. */
#define _GNU_SOURCE
#include "gemm_i8.h"

/* REPRODUCIBILITY: no FMA contraction, engine-wide.
 * The kernels below and coli_gemm_i8_ref compute the same arithmetic, but the
 * compiler is free to contract a*b+c into an FMA in one and not the other --
 * different rounding, different result. Measured 2026-08-16: with contraction
 * on, the reference and every dispatched kernel disagreed on 76% of output
 * cells at n=1 (1557 of 2048) despite the INTEGER dots being identical; with it
 * off, 0 of 266,240. An engine whose own reference test cannot agree with its
 * kernels cannot make a bit-exactness claim about anything. The Makefile also
 * passes -ffp-contract=off; this pragma is here so a build that forgets the flag
 * still gets deterministic arithmetic. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define COLI_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define COLI_ARM 1
#endif

/* ---------------------------------------------------------------- quantize */
void coli_quantize_a(coli_a_i8 *out, const float *x, int n, int64_t I) {
    int64_t nb = I / COLI_ABLK;
    out->n = n; out->I = I;
    for (int r = 0; r < n; r++) {
        const float *xr = x + (int64_t)r * I;
        for (int64_t b = 0; b < nb; b++) {
            const float *xb = xr + b * COLI_ABLK;
            float am = 0.f;
            for (int i = 0; i < COLI_ABLK; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am / 127.f; if (s < 1e-12f) s = 1e-12f;
            out->scale[r*nb + b] = s;
            float inv = 1.f / s;
            int32_t sum = 0;
            for (int i = 0; i < COLI_ABLK; i++) {
                /* Same clamp as the weight quantizer: lrintf can return 128
                 * from a 127.5 tie and (int8_t)128 is -128, flipping the sign of
                 * the largest activation in the block. */
                int qi = (int)lrintf(xb[i] * inv);
                if (qi >  127) qi =  127;
                if (qi < -127) qi = -127;
                int8_t q = (int8_t)qi;
                out->q[(int64_t)r*I + b*COLI_ABLK + i] = q;
                sum += q;
            }
            out->sum[r*nb + b] = sum;
        }
    }
}

/* ------------------------------------------------------------- narrow path */
/* Chosen for n < COLI_GEMM_MIN_WIDE, i.e. decode, where the loop is bound by
 * weight bytes and the widest ISA measured SLOWER. */
#if defined(COLI_X86)
static inline int32_t dot_blk_avx2(const int8_t *a, const uint8_t *b) {
    __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)b));
    vb = _mm256_sub_epi16(vb, _mm256_set1_epi16(128));   /* undo the storage offset */
    __m256i p = _mm256_madd_epi16(va, vb);               /* -> int32, cannot overflow i16 */
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    __m128i h = _mm_unpackhi_epi64(s, s); s = _mm_add_epi32(s, h);
    h = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)); s = _mm_add_epi32(s, h);
    return _mm_cvtsi128_si32(s);
}
#elif defined(COLI_ARM)
static inline int32_t dot_blk_neon(const int8_t *a, const uint8_t *b) {
    int8x16_t va = vld1q_s8(a);
    int8x16_t vb = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(b), vdupq_n_u8(128)));
#if defined(__ARM_FEATURE_DOTPROD)
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), va, vb));
#else
    int32x4_t acc = vdupq_n_s32(0);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
    return vaddvq_s32(acc);
#endif
}
#endif

static inline int32_t dot_blk_ref(const int8_t *a, const uint8_t *b) {
    int32_t s = 0;
    for (int i = 0; i < COLI_ABLK; i++) s += (int32_t)a[i] * ((int32_t)b[i] - 128);
    return s;
}
static inline int32_t dot_blk(const int8_t *a, const uint8_t *b) {
#if defined(COLI_X86)
    return dot_blk_avx2(a, b);
#elif defined(COLI_ARM)
    return dot_blk_neon(a, b);
#else
    return dot_blk_ref(a, b);
#endif
}

static void gemm_narrow(float *y, const coli_a_i8 *a, const coli_w_i8 *w) {
    int64_t I = w->I, O = w->O, nb = I / COLI_ABLK;
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *wr = w->qu + o*I; float sc = w->scale[o];
        for (int r = 0; r < a->n; r++) {
            const int8_t *xr = a->q + (int64_t)r*I;
            const float  *sr = a->scale + (int64_t)r*nb;
            float acc = 0.f;
            for (int64_t b = 0; b < nb; b++) acc += sr[b]*(float)dot_blk(xr+b*COLI_ABLK, wr+b*COLI_ABLK);
            y[(int64_t)r*O + o] = acc*sc;
        }
    }
}

/* --------------------------------------------------------------- wide path */
#if defined(COLI_X86) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define COLI_HAVE_VNNI 1
static void gemm_wide_vnni(float *y, const coli_a_i8 *a, const coli_w_i8 *w) {
    int64_t I = w->I, O = w->O, nb = I / COLI_ABLK;
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *wr = w->qu + o*I; float sc = w->scale[o];
        for (int r = 0; r < a->n; r++) {
            const int8_t  *xr = a->q + (int64_t)r*I;
            const float   *sr = a->scale + (int64_t)r*nb;
            const int32_t *su = a->sum   + (int64_t)r*nb;
            float acc = 0.f; int64_t b = 0;
            for (; b + 4 <= nb; b += 4) {   /* 64 bytes = 4 activation blocks */
                __m512i vw = _mm512_loadu_si512((const void*)(wr + b*COLI_ABLK));
                __m512i vx = _mm512_loadu_si512((const void*)(xr + b*COLI_ABLK));
                __m512i p  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vw, vx);
                int32_t t[16]; _mm512_storeu_si512((void*)t, p);
                for (int k = 0; k < 4; k++)
#if defined(COLI_BREAK_WIDE)
                    /* Negative control, build-time only, never shipped: perturbs
                     * ONE implementation, which is the only kind of control that
                     * can fail a differential test. */
                    acc += sr[b+k]*(float)(t[k*4]+t[k*4+1]+t[k*4+2]+t[k*4+3] - 127*su[b+k]);
#else
                    acc += sr[b+k]*(float)(t[k*4]+t[k*4+1]+t[k*4+2]+t[k*4+3] - 128*su[b+k]);
#endif
            }
            /* tail in the narrow kernel, exactly as llama.cpp's repack path does
             * (GEMM over the largest multiple of 4, GEMV for the remainder). */
            for (; b < nb; b++) acc += sr[b]*(float)dot_blk(xr+b*COLI_ABLK, wr+b*COLI_ABLK);
            y[(int64_t)r*O + o] = acc*sc;
        }
    }
}
#endif

/* ---------------------------------------------------------------- dispatch */
const char *coli_gemm_i8_kernel(int n, int64_t I, int64_t O) {
    (void)I; (void)O;
#if defined(COLI_HAVE_VNNI)
    if (n >= COLI_GEMM_MIN_WIDE && (coli_cpu_features() & COLI_CPU_AVX512VNNI))
        return "avx512vnni-wide";
#endif
#if defined(COLI_X86)
    if (coli_cpu_features() & COLI_CPU_AVX2) return "avx2-narrow";
    return "scalar-narrow";
#elif defined(COLI_ARM)
    return (coli_cpu_features() & COLI_CPU_DOTPROD) ? "neon-dotprod-narrow" : "neon-narrow";
#else
    return "scalar-narrow";
#endif
}

void coli_gemm_i8(float *y, const coli_a_i8 *a, const coli_w_i8 *w) {
#if defined(COLI_HAVE_VNNI)
    if (a->n >= COLI_GEMM_MIN_WIDE && (coli_cpu_features() & COLI_CPU_AVX512VNNI)) {
        gemm_wide_vnni(y, a, w); return;
    }
#endif
    gemm_narrow(y, a, w);
}

/* Reference implementation, exported for tests only. Every dispatched kernel
 * must equal this exactly; it is the definition, not an approximation. */
void coli_gemm_i8_ref(float *y, const coli_a_i8 *a, const coli_w_i8 *w);
void coli_gemm_i8_ref(float *y, const coli_a_i8 *a, const coli_w_i8 *w) {
    int64_t I = w->I, O = w->O, nb = I / COLI_ABLK;
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *wr = w->qu + o*I; float sc = w->scale[o];
        for (int r = 0; r < a->n; r++) {
            const int8_t *xr = a->q + (int64_t)r*I;
            const float  *sr = a->scale + (int64_t)r*nb;
            float acc = 0.f;
            for (int64_t b = 0; b < nb; b++) acc += sr[b]*(float)dot_blk_ref(xr+b*COLI_ABLK, wr+b*COLI_ABLK);
            y[(int64_t)r*O + o] = acc*sc;
        }
    }
}
