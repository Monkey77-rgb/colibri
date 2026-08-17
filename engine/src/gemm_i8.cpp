/* gemm_i8.c — see gemm_i8.h for the measurements this dispatch is built on. */
#define _GNU_SOURCE
#include "gemm_i8.h"
#include <stdlib.h>

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
#include "platform.h"   /* coli_aligned_alloc/free -- the int4 wide scratch */
#ifdef _OPENMP
#include <omp.h>        /* thread id + count, for that scratch's per-thread slice */
#endif

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

/* f32 reference path. Deliberately the plainest possible loop: no SIMD pragma,
 * no reordering, so it is the definition the quantized paths are judged against
 * rather than another approximation with its own rounding. */
static void gemm_f32(float *y, const float *x, int n, const coli_w_i8 *w) {
    int64_t I=w->I, O=w->O;
    #pragma omp parallel for schedule(static)
    for (int64_t o=0;o<O;o++) {
        const float *wr = w->f + o*I;
        for (int r=0;r<n;r++) {
            const float *xr = x + (int64_t)r*I;
            float acc=0.f;
            for (int64_t i=0;i<I;i++) acc += xr[i]*wr[i];
            y[(int64_t)r*O+o] = acc;
        }
    }
}
void coli_gemm_f32(float *y, const float *x, int n, const coli_w_i8 *w) { gemm_f32(y,x,n,w); }

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

/* ------------------------------------------------------------------ int4 ---
 * See gemm_i8.h for why a second format exists and why the scales are
 * per-block rather than per-row. */
void coli_quantize_w4(coli_w_i4 *w, const float *f, int64_t I, int64_t O) {
    int64_t nb = I / COLI_W4BLK;
    w->I = I; w->O = O;
    w->q4     = (uint8_t*)malloc((size_t)I*O/2);
    w->bscale = (float*)  malloc((size_t)nb*O*sizeof(float));
    for (int64_t o = 0; o < O; o++) {
        const float *r = f + o*I;
        for (int64_t b = 0; b < nb; b++) {
            const float *rb = r + b*COLI_W4BLK;
            float am = 0.f;
            for (int i = 0; i < COLI_W4BLK; i++) { float a = fabsf(rb[i]); if (a > am) am = a; }
            /* 7, not 8: the range is [-8,7] and using 8 would let the positive
             * extreme round to 8, which does not exist. */
            float s = am/7.f; if (s < 1e-12f) s = 1e-12f;
            w->bscale[o*nb+b] = s;
            float inv = 1.f/s;
            for (int i = 0; i < COLI_W4BLK; i++) {
                int q = (int)lrintf(rb[i]*inv);
                if (q >  7) q =  7;
                if (q < -8) q = -8;
                int64_t k = b*COLI_W4BLK + i;
                uint8_t nib = (uint8_t)(q + 8);           /* [0,15] */
                uint8_t *dst = &w->q4[o*(I/2) + k/2];
                if (k & 1) *dst = (uint8_t)((*dst & 0x0F) | (nib << 4));
                else       *dst = (uint8_t)((*dst & 0xF0) | nib);
            }
        }
    }
}
void coli_free_w4(coli_w_i4 *w){ free(w->q4); free(w->bscale); w->q4=NULL; w->bscale=NULL; }

static void gemm_i4_narrow(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
    int64_t I=w->I, O=w->O, wnb=I/COLI_W4BLK, anb=I/COLI_ABLK, rowb=I/2;
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *wr = w->q4 + o*rowb;
        const float   *ws = w->bscale + o*wnb;
        for (int r = 0; r < a->n; r++) {
            const int8_t *xr = a->q + (int64_t)r*I;
            const float  *as = a->scale + (int64_t)r*anb;
            float acc = 0.f;
            for (int64_t b = 0; b < wnb; b++) {
                /* one 32-weight block = two 16-element activation blocks, each
                 * with its own activation scale */
                int8_t tmp[COLI_W4BLK];
#if defined(COLI_X86)
                const __m128i m  = _mm_set1_epi8(0x0F);
                const __m128i e8 = _mm_set1_epi8(8);
                __m128i raw = _mm_loadu_si128((const __m128i*)(wr + b*(COLI_W4BLK/2)));
                __m128i lo  = _mm_sub_epi8(_mm_and_si128(raw, m), e8);
                __m128i hi  = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw,4), m), e8);
                /* nibble k of byte j is element 2j+(k), so interleave back */
                _mm_storeu_si128((__m128i*)tmp,      _mm_unpacklo_epi8(lo,hi));
                _mm_storeu_si128((__m128i*)(tmp+16), _mm_unpackhi_epi8(lo,hi));
#else
                for (int i = 0; i < COLI_W4BLK; i++) {
                    int64_t k = b*COLI_W4BLK + i;
                    uint8_t byte = wr[k/2];
                    int q = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    tmp[i] = (int8_t)(q - 8);
                }
#endif
                int64_t ab = b*2;
                int32_t d0 = 0, d1 = 0;
                for (int i = 0; i < 16; i++) d0 += (int32_t)xr[ab*16 + i]      * tmp[i];
                for (int i = 0; i < 16; i++) d1 += (int32_t)xr[(ab+1)*16 + i]  * tmp[16+i];
                acc += ws[b] * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
            }
            y[(int64_t)r*O + o] = acc;
        }
    }
}

/* int4, wide. The SAME arithmetic as gemm_i4_narrow above, with the unpack moved
 * out of the r loop.
 *
 * WHY THIS EXISTS. The narrow kernel unpacks a weight block inside the loop over
 * activation rows, so an n-row call unpacks every weight n times. That is the
 * whole of the measured n=4 regression, not a property of int4: 3.46 ms at n=1
 * and 13.75 ms at n=4 is 3.97x for 4x the work, i.e. the dot product had become
 * free relative to the unpack. Unpacking a row ONCE into an int8 scratch and
 * reusing it across all n rows makes the unpack an O(I) cost amortized over n
 * instead of an O(I*n) cost -- which is what "int4 is bad at prefill" was really
 * measuring.
 *
 * BIT-EXACT WITH THE NARROW KERNEL, and it must stay that way: the per-block
 * int32 dots are integer (associativity is exact, so the VNNI lane grouping is
 * free to differ), and the float accumulation order over blocks is unchanged.
 *
 * The scratch holds the nibble as stored, u = q+8 in [0,15], NOT the signed
 * value. VPDPBUSD wants an unsigned first operand, and the correction term
 * 8*sum(x) comes from a->sum, which the activation quantizer already computes
 * per 16-block for exactly this reason. Subtracting 8 during the unpack instead
 * would force a sign-extending path and buy nothing. */
#if defined(COLI_X86) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
#define COLI_HAVE_VNNI_I4 1
static void gemm_i4_wide(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
    int64_t I=w->I, O=w->O, wnb=I/COLI_W4BLK, anb=I/COLI_ABLK, rowb=I/2;
    /* Scratch rows for the whole team, allocated ONCE outside the parallel
     * region -- one row per thread, so at 16 threads and I=11008 this is 176 KB.
     *
     * Allocating inside the region and guarding the loop with `if (u)` is what
     * this replaced, and it was wrong twice over: OpenMP requires every thread
     * in the team to encounter a worksharing construct, so an `omp for` inside a
     * conditional is undefined behaviour rather than a graceful degradation --
     * and on a failed allocation the skipped rows of `y` would simply never be
     * written, producing a wrong answer instead of an error. Allocating up front
     * makes the failure a single decision with a correct fallback. */
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();   /* the bound for the region below; we do not
                                   * set num_threads, so the team cannot exceed it */
#endif
    uint8_t *pool = (uint8_t*)coli_aligned_alloc(64, (size_t)I*(size_t)nt);
    if (!pool) { gemm_i4_narrow(y, a, w); return; }   /* correct, just slower */
    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint8_t *u = pool + (size_t)I*(size_t)tid;
        {
        #pragma omp for schedule(static)
        for (int64_t o = 0; o < O; o++) {
            const uint8_t *wr = w->q4 + o*rowb;
            const float   *ws = w->bscale + o*wnb;
            /* ---- unpack the row once ---- */
            const __m128i m = _mm_set1_epi8(0x0F);
            for (int64_t j = 0; j < rowb; j += 16) {
                __m128i raw = _mm_loadu_si128((const __m128i*)(wr + j));
                __m128i lo  = _mm_and_si128(raw, m);
                __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw,4), m);
                /* nibble k of byte j is element 2j+k, so interleave back */
                _mm_storeu_si128((__m128i*)(u + j*2),      _mm_unpacklo_epi8(lo,hi));
                _mm_storeu_si128((__m128i*)(u + j*2 + 16), _mm_unpackhi_epi8(lo,hi));
            }
            /* ---- then reuse it for every activation row ---- */
            for (int r = 0; r < a->n; r++) {
                const int8_t  *xr = a->q + (int64_t)r*I;
                const float   *as = a->scale + (int64_t)r*anb;
                const int32_t *su = a->sum   + (int64_t)r*anb;
                float acc = 0.f;
                for (int64_t b = 0; b < wnb; b++) {
                    __m256i vu = _mm256_loadu_si256((const __m256i*)(u  + b*COLI_W4BLK));
                    __m256i vx = _mm256_loadu_si256((const __m256i*)(xr + b*COLI_W4BLK));
                    __m256i p  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vu, vx);
                    int32_t t[8]; _mm256_storeu_si256((__m256i*)t, p);
                    int64_t ab = b*2;
                    /* lanes 0-3 are bytes 0-15 = activation block ab, lanes 4-7
                     * are bytes 16-31 = block ab+1. The -8*sum is the
                     * offset-to-unsigned correction, one per activation block
                     * because each has its own scale. */
#if defined(COLI_BREAK_I4)
                    /* Negative control, build-time only. Perturbs ONE of the two
                     * implementations -- corrupting a shared input would leave
                     * them agreeing and the differential would pass vacuously. */
                    int32_t d0 = t[0]+t[1]+t[2]+t[3] - 7*su[ab];
#else
                    int32_t d0 = t[0]+t[1]+t[2]+t[3] - 8*su[ab];
#endif
                    int32_t d1 = t[4]+t[5]+t[6]+t[7] - 8*su[ab+1];
                    acc += ws[b] * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
                }
                y[(int64_t)r*O + o] = acc;
            }
        }
        }
    }
    coli_aligned_free(pool);
}
#endif

void coli_gemm_i4(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
#if defined(COLI_HAVE_VNNI_I4)
    if (a->n >= COLI_GEMM_I4_MIN_WIDE && (coli_cpu_features() & COLI_CPU_AVX512VNNI)) {
        gemm_i4_wide(y, a, w); return;
    }
#endif
    gemm_i4_narrow(y, a, w);
}

/* Which int4 kernel a batch would pick. Same purpose as coli_gemm_i8_kernel:
 * a benchmark that silently ran the narrow kernel twice would report "no
 * speedup" and look like a result. */
const char *coli_gemm_i4_kernel(int n) {
#if defined(COLI_HAVE_VNNI_I4)
    if (n >= COLI_GEMM_I4_MIN_WIDE && (coli_cpu_features() & COLI_CPU_AVX512VNNI))
        return "avx512vnni-i4-wide";
#endif
    (void)n;
    return "i4-narrow";
}
