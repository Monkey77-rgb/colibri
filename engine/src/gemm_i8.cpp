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
    /* Rows are independent (every write is indexed by r), and at prefill n this loop was the
     * single-threaded part of every GEMM stage: 3 calls per layer at 683x4096 (2026-09-16 Plan
     * V2a). Parallel over rows, static schedule, only when there is enough work to pay for the
     * fork; the per-row arithmetic is untouched so the output is bit-identical to the serial loop. */
    #pragma omp parallel for schedule(static) if (n >= 8)
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
/* Least-squares scale search for one block, following llama.cpp's
 * make_qx_quants (ggml/src/ggml-quants.c:451, read at c96f608d9 -- not recalled).
 *
 * WHY A SEARCH AT ALL. Round-to-nearest against amax/7 picks the scale that makes
 * the largest weight representable, which is not the scale that minimises error
 * over the block: one outlier drags the step size up and every other weight in
 * the block pays for it. The search instead asks, for each candidate scale, what
 * the least-squares optimum would be after rounding, and keeps the best.
 *
 * The weighting w = x*x is ggml's rmse_type 1: error on a large weight costs more
 * than the same error on a small one. That is the cheap stand-in for the idea AWQ
 * makes properly -- which weights matter should be decided by their contribution,
 * not by their magnitude alone. AWQ decides it from ACTIVATIONS, which needs
 * calibration data we do not have at load time; this needs nothing.
 *
 * Cost is 19 passes over a 32-element block at load time and nothing at all at
 * inference: the output is the same q4 bytes and the same one scale per block, so
 * no kernel changes and no format change. Whether it is worth those load seconds
 * is measured, not assumed -- see the table in README.md.
 *
 * Returns the chosen scale; writes quantized values (range [-8,7]) into L. */
static float w4_block_scale_w(const float *x, int8_t *L, const float *imp) {
    const int n = COLI_W4BLK, nmax = 8;
    float amax = 0.f, max = 0.f;
    for (int i = 0; i < n; i++) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < 1e-30f) { for (int i = 0; i < n; i++) L[i] = 0; return 1e-12f; }

    float iscale = -(float)nmax / max;
    float sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < n; i++) {
        int l = (int)lrintf(iscale * x[i]);
        if (l < -nmax)   l = -nmax;
            if (l > nmax-1)  l = nmax-1;
        L[i] = (int8_t)l;
        float wt = imp ? imp[i] : x[i]*x[i];
        sumlx += wt*x[i]*(float)l;
        suml2 += wt*(float)l*(float)l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.f;
    float best  = scale*sumlx;
    /* 18 candidates either side of the amax-derived scale, exactly ggml's grid. */
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        float isc = -((float)nmax + 0.1f*(float)is) / max;
        float slx = 0.f, sl2 = 0.f;
        for (int i = 0; i < n; i++) {
            int l = (int)lrintf(isc * x[i]);
            if (l < -nmax)   l = -nmax;
            if (l > nmax-1)  l = nmax-1;
            float wt = imp ? imp[i] : x[i]*x[i];
            slx += wt*x[i]*(float)l;
            sl2 += wt*(float)l*(float)l;
        }
        if (sl2 > 0.f && slx*slx > best*sl2) {
            for (int i = 0; i < n; i++) {
                int l = (int)lrintf(isc * x[i]);
                if (l < -nmax)   l = -nmax;
            if (l > nmax-1)  l = nmax-1;
                L[i] = (int8_t)l;
            }
            scale = slx/sl2; best = scale*slx;
        }
    }
    /* A negative scale is legal here -- max carries the SIGN of the largest
     * magnitude, so iscale is negative when that weight is positive. The kernel
     * multiplies by it either way. Guard only against a zero step. */
    if (scale > -1e-12f && scale < 1e-12f) scale = 1e-12f;
    return scale;
}

static void quantize_w4_core(coli_w_i4 *w, const float *f, int64_t I, int64_t O,
                             int rmse, const float *imp);
void coli_quantize_w4_ex(coli_w_i4 *w, const float *f, int64_t I, int64_t O, int rmse) {
    quantize_w4_core(w, f, I, O, rmse, NULL);
}
void coli_quantize_w4_imp(coli_w_i4 *w, const float *f, int64_t I, int64_t O, const float *imp) {
    quantize_w4_core(w, f, I, O, 1, imp);
}
static void quantize_w4_core(coli_w_i4 *w, const float *f, int64_t I, int64_t O,
                             int rmse, const float *imp) {
    int64_t nb = I / COLI_W4BLK;
    w->I = I; w->O = O;
    w->q4     = (uint8_t*)malloc((size_t)I*O/2);
    w->bscale = (float*)  malloc((size_t)nb*O*sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        const float *r = f + o*I;
        for (int64_t b = 0; b < nb; b++) {
            const float *rb = r + b*COLI_W4BLK;
            int8_t L[COLI_W4BLK];
            if (rmse) {
                float s = w4_block_scale_w(rb, L, imp ? imp + b*COLI_W4BLK : NULL);
                w->bscale[o*nb+b] = s;
                for (int i = 0; i < COLI_W4BLK; i++) {
                    int q = L[i];
                    int64_t k = b*COLI_W4BLK + i;
                    uint8_t nib = (uint8_t)(q + 8);
                    uint8_t *dst = &w->q4[o*(I/2) + k/2];
                    if (k & 1) *dst = (uint8_t)((*dst & 0x0F) | (nib << 4));
                    else       *dst = (uint8_t)((*dst & 0xF0) | nib);
                }
                continue;
            }
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
/* The original signature, unchanged: plain round-to-nearest. Kept so the two
 * quantizers can be compared on the same build, which is how the default below
 * was chosen rather than assumed. */
void coli_quantize_w4(coli_w_i4 *w, const float *f, int64_t I, int64_t O) {
    coli_quantize_w4_ex(w, f, I, O, 0);
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
/* The two halves of one wide-kernel output row, shared by gemm_i4_wide and
 * coli_gemm_i4_multi so the two cannot drift apart: unpack a weight row's nibbles
 * into an int8 scratch (u = q+8, unsigned, see the note above), then dot that
 * scratch with ONE quantized activation row. */
static inline void i4_unpack_row(uint8_t *u, const uint8_t *wr, int64_t rowb) {
    const __m128i m = _mm_set1_epi8(0x0F);
    for (int64_t j = 0; j < rowb; j += 16) {
        __m128i raw = _mm_loadu_si128((const __m128i*)(wr + j));
        __m128i lo  = _mm_and_si128(raw, m);
        __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw,4), m);
        /* nibble k of byte j is element 2j+k, so interleave back */
        _mm_storeu_si128((__m128i*)(u + j*2),      _mm_unpacklo_epi8(lo,hi));
        _mm_storeu_si128((__m128i*)(u + j*2 + 16), _mm_unpackhi_epi8(lo,hi));
    }
}
static inline float i4_row_vnni(const uint8_t *u, const float *ws, int64_t wnb,
                                const int8_t *xr, const float *as, const int32_t *su) {
    float acc = 0.f;
    for (int64_t b = 0; b < wnb; b++) {
        __m256i vu = _mm256_loadu_si256((const __m256i*)(u  + b*COLI_W4BLK));
        __m256i vx = _mm256_loadu_si256((const __m256i*)(xr + b*COLI_W4BLK));
        __m256i p  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vu, vx);
        int32_t t[8]; _mm256_storeu_si256((__m256i*)t, p);
        int64_t ab = b*2;
        /* lanes 0-3 are bytes 0-15 = activation block ab, lanes 4-7 are bytes
         * 16-31 = block ab+1. The -8*sum is the offset-to-unsigned correction,
         * one per activation block because each has its own scale. */
#if defined(COLI_BREAK_I4)
        /* Negative control, build-time only. Perturbs ONE of the two
         * implementations -- corrupting a shared input would leave them
         * agreeing and the differential would pass vacuously. */
        int32_t d0 = t[0]+t[1]+t[2]+t[3] - 7*su[ab];
#else
        int32_t d0 = t[0]+t[1]+t[2]+t[3] - 8*su[ab];
#endif
        int32_t d1 = t[4]+t[5]+t[6]+t[7] - 8*su[ab+1];
        acc += ws[b] * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
    }
    return acc;
}
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
            i4_unpack_row(u, wr, rowb);
            /* ---- then reuse it for every activation row ---- */
            for (int r = 0; r < a->n; r++)
                y[(int64_t)r*O + o] = i4_row_vnni(u, ws, wnb,
                                                  a->q + (int64_t)r*I,
                                                  a->scale + (int64_t)r*anb,
                                                  a->sum   + (int64_t)r*anb);
        }
        }
    }
    coli_aligned_free(pool);
}

/* ------------------------------------------------------ int4, packed panel
 * ATTEMPT 2 (attempt 1 was a 2x4 register tile over separate (o,r) dots --
 * verified bit-identical, but 4-9% SLOWER across every shape/n measured. Root
 * cause, found by reading i4_row_vnni rather than by guessing: per 32-element
 * block it issues ONE dpbusd, then STORES the ymm to memory (int32_t t[8])
 * and runs ~20 scalar ops on it (8 lane reads, 2 int->float converts, 3
 * float mul/add). The VNNI unit is idle almost the whole time; tiling more
 * dots just multiplies that scalar epilogue, which is the actual cost. The
 * 2x4 tile is kept nowhere in this file -- this replaces it outright.)
 *
 * THE FIX: put OUTPUT ROWS in SIMD LANES instead of activation elements. Build
 * an 8-output-row PANEL where lane j (0..7) holds output row o0+j's unpacked
 * weights, laid out so that for global element index e (0, 4, 8, ... step 4)
 * the 32 bytes at P + e*8 are {row0[e..e+3], row1[e..e+3], ..., row7[e..e+3]}.
 * One VPDPBUSD then broadcasts 4 ACTIVATION bytes to every lane and dots them
 * against 8 DIFFERENT output rows' weight bytes in one instruction -- the
 * epilogue (int->float, scale multiply, accumulate) becomes ONE vector op
 * for 8 output rows instead of one scalar sequence per row, so it no longer
 * dominates.
 *
 * BIT-IDENTICAL BY CONSTRUCTION, same argument as attempt 1's but one level
 * deeper: VPDPBUSD's four-term dot-product-and-add is exact 32-bit integer
 * arithmetic (no rounding), and integer addition is associative and
 * commutative regardless of grouping. i4_row_vnni sums a block's 32 elements
 * as ONE dpbusd over 32 bytes (8 lanes of 4 elements each), then adds lanes
 * 0-3 for d0 and 4-7 for d1 -- four 4-element partial dot products per half,
 * summed. This kernel computes the SAME four 4-element partial dot products
 * per half (one dpbusd per 4-element chunk, accumulated into d0/d1 across 4
 * calls instead of extracted from one wider call), so d0 and d1 land on the
 * identical integer value either way -- not approximately, exactly, because
 * every partial sum involved is an exact integer. The float epilogue then
 * runs the reference's OWN expression, `ws[b]*(as[ab]*d0 + as[ab+1]*d1)`, as
 * explicit _mm256_mul_ps/_mm256_add_ps in that same left-to-right order, with
 * NO FMA intrinsics -- the tree already forbids FP contraction
 * (-ffp-contract=off, see this file's header) for exactly this reason,
 * verified rather than assumed: an FMA-fused mul-add rounds once instead of
 * twice and would silently disagree with the reference on some cells. Every
 * lane runs the identical sequence on identical inputs, so this is
 * bit-identical to i4_row_vnni cell by cell -- proved by
 * test_gemm_i4.c's run_panel_check() against the plain scalar definition,
 * not assumed from the algebra above.
 *
 * STILL 256-BIT (ymm) ONLY -- same reason as attempt 1 and the MXFP4
 * kernel's HISTORY comment above COLI_MX_RCH: zmm use measurably slows
 * subsequent 256-bit VNNI code in the same process on this Zen 5.
 *
 * TAIL. O is not always a multiple of 8 (test_gemm_i4.c's 2048x772 shape
 * exercises this deliberately): rows Ofull..O-1 fall through to the existing
 * per-row i4_unpack_row/i4_row_vnni path, unchanged. */
#define COLI_I4_PANEL_GRP 8

/* One activation row against one 8-row panel: same b loop, same d0/d1
 * per-block correction, same float expression as i4_row_vnni, just 8 output
 * rows wide. `y8` points at y[r*O + o0], eight contiguous floats to store. */
static inline void i4_panel_row(float *y8, const uint8_t *P, const float *WS,
                                 int64_t wnb, const int8_t *xr,
                                 const float *as, const int32_t *su) {
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < wnb; b++) {
        int64_t ab = b*2;
        int64_t base = b*COLI_W4BLK;
        __m256i d0 = _mm256_setzero_si256();
        __m256i d1 = _mm256_setzero_si256();
        for (int kk = 0; kk < 4; kk++) {
            int64_t e0 = base + kk*4;
            int32_t x0; memcpy(&x0, xr + e0, 4);
            __m256i vw0 = _mm256_loadu_si256((const __m256i*)(P + e0*COLI_I4_PANEL_GRP));
            d0 = _mm256_dpbusd_epi32(d0, vw0, _mm256_set1_epi32(x0));

            int64_t e1 = base + 16 + kk*4;
            int32_t x1; memcpy(&x1, xr + e1, 4);
            __m256i vw1 = _mm256_loadu_si256((const __m256i*)(P + e1*COLI_I4_PANEL_GRP));
            d1 = _mm256_dpbusd_epi32(d1, vw1, _mm256_set1_epi32(x1));
        }
#if defined(COLI_BREAK_I4_PANEL)
        /* Negative control, build-time only, never shipped: perturbs ONLY
         * the panel kernel's correction term, same discipline as
         * i4_row_vnni's COLI_BREAK_I4 above. */
        __m256i corr0 = _mm256_set1_epi32(7*su[ab]);
#else
        __m256i corr0 = _mm256_set1_epi32(8*su[ab]);
#endif
        __m256i corr1 = _mm256_set1_epi32(8*su[ab+1]);
        d0 = _mm256_sub_epi32(d0, corr0);
        d1 = _mm256_sub_epi32(d1, corr1);

        /* Explicit convert/mul/add -- NOT a*b+c in source, so there is
         * nothing for -ffp-contract to fuse even if it were on; each op maps
         * to one instruction (vcvtdq2ps/vmulps/vaddps), matching the
         * reference's separate multiply-then-add exactly. */
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        __m256 t0 = _mm256_mul_ps(_mm256_set1_ps(as[ab]),   f0);
        __m256 t1 = _mm256_mul_ps(_mm256_set1_ps(as[ab+1]), f1);
        __m256 inner = _mm256_add_ps(t0, t1);
        __m256 wsv = _mm256_loadu_ps(WS + b*COLI_I4_PANEL_GRP);
        __m256 prod = _mm256_mul_ps(wsv, inner);
        acc = _mm256_add_ps(acc, prod);
    }
    _mm256_storeu_ps(y8, acc);
}

static void gemm_i4_panel(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
    int64_t I=w->I, O=w->O, wnb=I/COLI_W4BLK, anb=I/COLI_ABLK, rowb=I/2;
    enum { GRP = COLI_I4_PANEL_GRP };
    int64_t Ofull = (O/GRP)*GRP;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    /* Per thread: P (panel, GRP*I bytes, L1/L2 resident -- 16 KiB at
     * I=2048, 112 KiB at I=14336) + u_row (I bytes, single-row unpack
     * scratch, reused for the panel build AND the tail path) + WS (wnb*GRP
     * floats, the group's scales transposed so WS+b*GRP is one ymm's worth).
     * One allocate-once-outside-the-region pool, same reasoning as
     * gemm_i4_wide's: an omp for inside a conditional allocation is UB, and
     * a failed allocation must be a single decision with a correct
     * fallback, not silently-skipped output rows. */
    size_t panel_bytes = (size_t)I*(size_t)GRP;
    size_t urow_bytes  = (size_t)I;
    size_t byte_stride  = panel_bytes + urow_bytes;
    uint8_t *pool = (uint8_t*)coli_aligned_alloc(64, byte_stride*(size_t)nt);
    float   *wsv_pool = (float*)coli_aligned_alloc(64, (size_t)wnb*(size_t)GRP*sizeof(float)*(size_t)nt);
    if (!pool || !wsv_pool) {
        if (pool) coli_aligned_free(pool);
        if (wsv_pool) coli_aligned_free(wsv_pool);
        gemm_i4_wide(y, a, w); return;   /* correct, just slower */
    }
    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint8_t *P     = pool + byte_stride*(size_t)tid;
        uint8_t *u_row = P + panel_bytes;
        float   *WS    = wsv_pool + (size_t)wnb*(size_t)GRP*(size_t)tid;

        #pragma omp for schedule(static)
        for (int64_t o0 = 0; o0 < Ofull; o0 += GRP) {
            for (int j = 0; j < GRP; j++) {
                i4_unpack_row(u_row, w->q4 + (o0+j)*rowb, rowb);
                for (int64_t e = 0; e < I; e += 4)
                    memcpy(P + e*GRP + j*4, u_row + e, 4);
            }
            for (int64_t b = 0; b < wnb; b++)
                for (int j = 0; j < GRP; j++)
                    WS[b*GRP+j] = w->bscale[(o0+j)*wnb + b];

            for (int r = 0; r < a->n; r++)
                i4_panel_row(y + (int64_t)r*O + o0, P, WS, wnb,
                             a->q + (int64_t)r*I,
                             a->scale + (int64_t)r*anb,
                             a->sum   + (int64_t)r*anb);
        }
        /* Tail: O % GRP rows that don't fill a panel, through the ordinary
         * per-row path. All threads encounter this second worksharing
         * construct too (see the OpenMP note above). */
        #pragma omp for schedule(static)
        for (int64_t o = Ofull; o < O; o++) {
            i4_unpack_row(u_row, w->q4 + o*rowb, rowb);
            const float *ws = w->bscale + o*wnb;
            for (int r = 0; r < a->n; r++)
                y[(int64_t)r*O + o] = i4_row_vnni(u_row, ws, wnb,
                                                   a->q + (int64_t)r*I,
                                                   a->scale + (int64_t)r*anb,
                                                   a->sum   + (int64_t)r*anb);
        }
    }
    coli_aligned_free(pool);
    coli_aligned_free(wsv_pool);
}

/* Rows at or above which the panel kernel is preferred over gemm_i4_wide.
 * Same starting value and same tunable convention as attempt 1's
 * COLI_GEMM_I4_MIN_TILE -- not yet swept, see the deliverable message. */
#ifndef COLI_GEMM_I4_MIN_TILE
#define COLI_GEMM_I4_MIN_TILE 4
#endif

/* COLI_I4_TILE=0 forces the panel kernel OFF regardless of n (name kept from
 * attempt 1 -- same knob, same purpose: compare against gemm_i4_wide on the
 * SAME build rather than a recompile, so a compiler-flag difference can't be
 * mistaken for the kernel doing the work). Read every call, deliberately NOT
 * cached in a static: this is a once-per-GEMM-call decision, so one getenv()
 * is noise next to the O(n*I) work it gates, and caching it would make the
 * env var un-settable mid-process -- exactly what the differential test
 * needs: flip it, call coli_gemm_i4 again, compare. */
static int i4_tile_pref(void) {
    const char *e = getenv("COLI_I4_TILE");
    return (e && e[0] == '0') ? 0 : 1;
}
#endif /* end of the COLI_HAVE_VNNI_I4 block opened above i4_unpack_row */

void coli_gemm_i4(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
#if defined(COLI_HAVE_VNNI_I4)
    if (i4_tile_pref() && a->n >= COLI_GEMM_I4_MIN_TILE && (coli_cpu_features() & COLI_CPU_AVX512VNNI)) {
        gemm_i4_panel(y, a, w); return;
    }
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
    if (i4_tile_pref() && n >= COLI_GEMM_I4_MIN_TILE && (coli_cpu_features() & COLI_CPU_AVX512VNNI))
        return "avx512vnni-i4-panel";
    if (n >= COLI_GEMM_I4_MIN_WIDE && (coli_cpu_features() & COLI_CPU_AVX512VNNI))
        return "avx512vnni-i4-wide";
#endif
    (void)n;
    return "i4-narrow";
}

/* See gemm_i8.h. One row-space, one team, one scratch slice per thread. */
void coli_gemm_i4_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                        const coli_w_i4 *const *ws, int cnt) {
    if (cnt <= 0) return;
#if defined(COLI_HAVE_VNNI_I4)
    enum { MAXM = 256 };
    if (cnt <= MAXM && (coli_cpu_features() & COLI_CPU_AVX512VNNI)) {
        int64_t off[MAXM+1]; int64_t Imax = 0;
        off[0] = 0;
        for (int j = 0; j < cnt; j++) { off[j+1] = off[j] + ws[j]->O; if (ws[j]->I > Imax) Imax = ws[j]->I; }
        const int64_t R = off[cnt];
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        uint8_t *pool = (uint8_t*)coli_aligned_alloc(64, (size_t)Imax*(size_t)nt);
        if (pool) {
            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                uint8_t *u = pool + (size_t)Imax*(size_t)tid;
                int j = 0;   /* rows are visited in increasing order per thread, so
                              * the matrix index only ever moves forward */
                #pragma omp for schedule(static)
                for (int64_t rr = 0; rr < R; rr++) {
                    while (rr >= off[j+1]) j++;
                    while (rr <  off[j])   j--;   /* first row of a static chunk may sit behind */
                    const coli_w_i4 *w = ws[j];
                    const int64_t o = rr - off[j];
                    const int64_t I = w->I, wnb = I/COLI_W4BLK, anb = I/COLI_ABLK, rowb = I/2;
                    const int r = arow[j];
                    i4_unpack_row(u, w->q4 + o*rowb, rowb);
                    ys[j][o] = i4_row_vnni(u, w->bscale + o*wnb, wnb,
                                           a->q + (int64_t)r*I,
                                           a->scale + (int64_t)r*anb,
                                           a->sum   + (int64_t)r*anb);
                }
            }
            coli_aligned_free(pool);
            return;
        }
    }
#endif
    /* Fallback: one matrix at a time through the standard dispatch, on a one-row
     * view of the activation. Same arithmetic, just cnt regions instead of one. */
    for (int j = 0; j < cnt; j++) {
        const int64_t I = ws[j]->I, anb = I/COLI_ABLK; const int r = arow[j];
        coli_a_i8 v;
        v.q = a->q + (int64_t)r*I; v.scale = a->scale + (int64_t)r*anb; v.sum = a->sum + (int64_t)r*anb;
        v.n = 1; v.I = I;
        coli_gemm_i4(ys[j], &v, ws[j]);
    }
}
