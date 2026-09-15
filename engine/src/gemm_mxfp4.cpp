/* gemm_mxfp4.cpp — see gemm_mxfp4.h for the layout, numerics and provenance
 * this kernel is built on.
 *
 * -ffp-contract=off is load-bearing here for the same reason gemm_i8.cpp pins
 * it (see that file's header comment): coli_gemm_mxfp4 and coli_gemm_mxfp4_ref
 * run the identical float accumulation `acc += wscale * (as0*(float)d0 +
 * as1*(float)d1)`, and an FMA-contracted build is free to round that
 * differently between two call sites even with nothing else different. */
#define _GNU_SOURCE
#include "gemm_mxfp4.h"
#include <stdlib.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cpu_features.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
/* SIMD path (2026-09-14): the design is FreeToken's CPU MXFP4 GEMV (FlashML-org,
 * cpu_moe_ext.cpp, Apache-2.0: nibble -> weight through a 16-entry byte LUT, one
 * scale multiply per 32-K block, weights read once per block and reused across
 * activation rows) carried into THIS engine's int4 VNNI structure (gemm_i8.cpp,
 * i4_row_vnni): VPDPBUSD wants an unsigned first operand, so the LUT emits
 * kvalues_mxfp4 + 12 (0..24) and the correction 12*sum(x) comes from a->sum,
 * which the activation quantizer already carries per 16-block. The float
 * accumulation `acc += wscale*(as0*d0 + as1*d1)` is the scalar reference's
 * expression to the operation, and d0/d1 are exact integers, so the SIMD kernel
 * is BIT-IDENTICAL to coli_gemm_mxfp4_ref (test_mxfp4 (b) asserts it cell by
 * cell; test_mxfp4_broken is the same build with one LUT entry perturbed and
 * must fail). Raw GGUF 17-byte blocks are read in place -- no repack, which is
 * what lets the expert store hand disk bytes straight to this kernel. */
#if defined(__x86_64__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__SSSE3__)
#define COLI_HAVE_VNNI_MX 1
#endif

/* e2m1 values, DOUBLED — identical table to c/ggml_dequant.h's
 * gguf_kvalues_mxfp4 (transcribed from the same upstream source, see that
 * file's PROVENANCE-2). Not #included from there: this is a small standalone
 * C++ TU, matching gemm_i8.cpp's convention of not depending on ../c headers. */
static const int8_t coli_mxfp4_lut[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* GGML_E8M0_TO_FP32_HALF, bit-identical to c/ggml_dequant.h's
 * gguf_e8m0_to_fp32_half — see that file's PROVENANCE-2 for the upstream
 * source. Kept as a second transcription rather than a shared include so this
 * kernel has no dependency on ../c, matching every other file in engine/src/. */
static inline float coli_mxfp4_e8m0_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = (uint32_t)0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

void coli_free_mxfp4(coli_w_mxfp4 *w) {
    if (!w) return;
    if (w->owned) free((void *)w->blocks);
    w->blocks = NULL;
}

/* One output row against every activation row. `nblk` = I/COLI_MXFP4_BLK,
 * `rowbytes` = nblk*COLI_MXFP4_BYTES -- both hoisted by the caller so this
 * function has no division in the hot loop. Shared, byte-identical, by
 * coli_gemm_mxfp4 and coli_gemm_mxfp4_ref: the ONLY difference between those
 * two entry points is whether the o-loop around this call carries an
 * `#pragma omp parallel for`, and each row's reduction is independent of every
 * other row's, so parallelizing the outer loop cannot change a single bit of
 * any one row's result. */
static inline void mxfp4_row_scalar(float *y, int64_t O, int64_t o,
                              const coli_a_i8 *a, int64_t I,
                              const uint8_t *wr, int64_t nblk) {
    const int64_t anb = I / COLI_ABLK;
    for (int r = 0; r < a->n; r++) {
        const int8_t *xr = a->q + (int64_t)r * I;
        const float  *as = a->scale + (int64_t)r * anb;
        float acc = 0.f;
        for (int64_t g = 0; g < nblk; g++) {
            const uint8_t *blk = wr + g * COLI_MXFP4_BYTES;
            const float wscale = coli_mxfp4_e8m0_half(blk[0]);
            const uint8_t *qs = blk + 1;
            const int64_t ab = g * 2;              /* two COLI_ABLK=16 act. blocks per group */
            int32_t d0 = 0, d1 = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t byte = qs[j];
                d0 += (int32_t)xr[ab*16       + j] * (int32_t)coli_mxfp4_lut[byte & 0x0F];
                d1 += (int32_t)xr[(ab+1)*16   + j] * (int32_t)coli_mxfp4_lut[byte >>   4];
            }
            acc += wscale * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
        }
        y[(int64_t)r * O + o] = acc;
    }
}

#if defined(COLI_HAVE_VNNI_MX)
/* kvalues_mxfp4 + 12, so every entry is a non-negative int8 for VPDPBUSD. */
#if defined(COLI_BREAK_MXFP4)
/* Negative control, build-time only (tests/test_mxfp4_broken): ONE LUT entry
 * off by one. Perturbs the SIMD kernel alone -- the reference is untouched, so
 * the differential must fail. Never define this in a shipping build. */
#define COLI_MX_LUT_E7 25
#else
#define COLI_MX_LUT_E7 24
#endif
static inline __m128i mx_lut12(void) {
    return _mm_setr_epi8(12,13,14,15,16,18,20,COLI_MX_LUT_E7, 12,11,10,9,8,6,4,0);
}
/* HISTORY -- measured losers, so nobody re-measures them (2026-09-15, commit
 * 23982c9; that commit's message said this note was here and it was not --
 * caught by a Codex read-only consultation the same day, which then proposed
 * attempt 2 afresh). Baseline, unmodified kernel, real gpt-oss-120b expert 3,
 * 2880x2880, 8 threads, 9800X3D quiet (load 0.9-2.0), best-of-90..1140:
 *   n=1 63.0 us (70 GB/s)   n=4 181 us (2.87x n=1)   n=8 338 us (5.36x)   n=16 689 us (10.95x)
 * So the RCH batching below amortises the block decode but the per-row cost
 * (32-byte q load, vpdpbusd, store+scalar lane sum, 12*sum correction, two
 * scale loads, float update) still scales ~linearly with n; prefill is ~n x
 * decode per expert. Two attempts to change that, both reverted:
 *   1. RCH 8->16 plus an in-register PHADDD fold instead of the store+scalar
 *      sum: WORSE at every n (n=1 77 us, n=4 3.13x, n=16 11.56x). PHADDD is
 *      slower than the store-forward path on Zen 5 for this pattern.
 *   2. Two activation rows packed into one 512-bit register, one
 *      _mm512_dpbusd_epi32 against the duplicated decoded weight (Zen 5 VNNI
 *      is native 512-bit), 256-bit path kept for odd rows and n=1. Bit-exact
 *      (test_mxfp4 (b) at n=1,2,4,5,8,16) but WORSE everywhere: n=1 90-162 us
 *      depending on how much zmm work preceded it in the process, n=16 890 us
 *      vs 689. Reproduced in a second same-window A/B with baseline stable at
 *      63/181/338/689 us. zmm use measurably slows subsequent 256-bit VNNI
 *      calls in the same process for a while (cpu MHz ~5.4 GHz and Tctl 50-59C
 *      throughout, so not a visible downclock) -- a cost any mixed decode/
 *      prefill workload would also pay.
 *   3. (18:55, proposed by a Codex consultation as distinct from 2) a strict
 *      TWO-row tile: o -> r0 += 2 -> g, decoded weight duplicated into both
 *      zmm halves, both rows' q in the other zmm, one _mm512_dpbusd_epi32,
 *      16-lane store + scalar sums, same float expression per row; n=1 and the
 *      odd trailing row on the 256-bit body. Bit-identical (test (b) at
 *      n=1,2,4,5,8,16), and SLOWER at every n in 3 alternating rounds against
 *      a clean build (same process order, best-of-N, quiet box load 0.8):
 *        n=2  105 vs  96 us (+9%)    n=4  210 vs 173 (+21%)
 *        n=8  419 vs 331 (+27%)      n=16 839 vs 667 (+26%)
 *      and n=1 61.8 vs 58.1 us although both binaries run the SAME 256-bit
 *      code there -- the zmm side effect of attempt 2 again (test (b) ran the
 *      zmm path earlier in the process). So the loss is not accumulator
 *      pressure from 8 rows; it is the zmm path itself on this Zen 5.
 * Still open: a design that cuts the per-row work itself (e.g. hoisting the
 * scale/sum loads and the lane reduction out of the per-block loop by
 * accumulating int32 across the two 16-blocks of an MX block before applying
 * scales) -- unmeasured; whatever is tried must keep the float accumulation
 * order per output element or re-justify the bit-identity test. */
/* One output row, up to RCH activation rows per pass: the weight block is
 * decoded ONCE per pass and dotted against every row of the chunk. */
#define COLI_MX_RCH 8
static inline void mxfp4_row_vnni(float *y, int64_t O, int64_t o,
                                  const coli_a_i8 *a, int64_t I,
                                  const uint8_t *wr, int64_t nblk) {
    const int64_t anb = I / COLI_ABLK;
    const __m128i lut = mx_lut12();
    const __m128i m4  = _mm_set1_epi8(0x0F);
    for (int r0 = 0; r0 < a->n; r0 += COLI_MX_RCH) {
        int rn = a->n - r0; if (rn > COLI_MX_RCH) rn = COLI_MX_RCH;
        float acc[COLI_MX_RCH];
        for (int r = 0; r < rn; r++) acc[r] = 0.f;
        for (int64_t g = 0; g < nblk; g++) {
            const uint8_t *blk = wr + g * COLI_MXFP4_BYTES;
            const float wscale = coli_mxfp4_e8m0_half(blk[0]);
            /* low nibble of byte j = element j (activation block ab), high = j+16 (ab+1) */
            __m128i raw = _mm_loadu_si128((const __m128i*)(blk + 1));
            __m128i lo  = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4));
            __m128i hi  = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
            __m256i vu  = _mm256_set_m128i(hi, lo);
            const int64_t ab = g * 2;
            for (int r = 0; r < rn; r++) {
                const int8_t  *xr = a->q + (int64_t)(r0 + r) * I;
                const float   *as = a->scale + (int64_t)(r0 + r) * anb;
                const int32_t *su = a->sum   + (int64_t)(r0 + r) * anb;
                __m256i vx = _mm256_loadu_si256((const __m256i*)(xr + ab * 16));
                __m256i p  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vu, vx);
                int32_t t[8]; _mm256_storeu_si256((__m256i*)t, p);
                int32_t d0 = t[0]+t[1]+t[2]+t[3] - 12*su[ab];
                int32_t d1 = t[4]+t[5]+t[6]+t[7] - 12*su[ab+1];
                acc[r] += wscale * (as[ab]*(float)d0 + as[ab+1]*(float)d1);
            }
        }
        for (int r = 0; r < rn; r++) y[(int64_t)(r0 + r) * O + o] = acc[r];
    }
}
#endif
static inline int mx_use_vnni(void) {
#if defined(COLI_HAVE_VNNI_MX)
    static int cached = -1;
    if (cached < 0) cached = (coli_cpu_features() & COLI_CPU_AVX512VNNI) ? 1 : 0;
    return cached;
#else
    return 0;
#endif
}
static inline void mxfp4_row(float *y, int64_t O, int64_t o,
                             const coli_a_i8 *a, int64_t I,
                             const uint8_t *wr, int64_t nblk) {
#if defined(COLI_HAVE_VNNI_MX)
    if (mx_use_vnni()) { mxfp4_row_vnni(y, O, o, a, I, wr, nblk); return; }
#endif
    mxfp4_row_scalar(y, O, o, a, I, wr, nblk);
}
const char *coli_gemm_mxfp4_kernel(void) {
    return mx_use_vnni() ? "avx512vnni-mxfp4" : "mxfp4-scalar";
}

void coli_gemm_mxfp4(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w) {
    const int64_t I = w->I, O = w->O;
    const int64_t nblk = I / COLI_MXFP4_BLK;
    const int64_t rowbytes = nblk * COLI_MXFP4_BYTES;
    #pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < O; o++) {
        mxfp4_row(y, O, o, a, I, w->blocks + o*rowbytes, nblk);
    }
}

void coli_gemm_mxfp4_ref(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w) {
    const int64_t I = w->I, O = w->O;
    const int64_t nblk = I / COLI_MXFP4_BLK;
    const int64_t rowbytes = nblk * COLI_MXFP4_BYTES;
    for (int64_t o = 0; o < O; o++) {
        mxfp4_row_scalar(y, O, o, a, I, w->blocks + o*rowbytes, nblk);   /* always scalar: the reference */
    }
}

/* See gemm_i8.h's coli_gemm_i4_multi for the pattern this mirrors: one
 * concatenated row-space, one parallel region, one thread-local scratch-free
 * pass (MXFP4 needs no unpack scratch -- each block is decoded and consumed
 * immediately -- so unlike the int4 multi kernel there is nothing to
 * pre-allocate per thread). */
void coli_gemm_mxfp4_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                            const coli_w_mxfp4 *const *ws, int cnt) {
    if (cnt <= 0) return;
    /* Fixed-size offset table, same MAXM convention as coli_gemm_i4_multi. */
    enum { MAXM = 256 };
    if (cnt <= MAXM) {
        int64_t off[MAXM+1];
        off[0] = 0;
        for (int j = 0; j < cnt; j++) off[j+1] = off[j] + ws[j]->O;
        const int64_t R = off[cnt];
        #pragma omp parallel
        {
            int j = 0;
            #pragma omp for schedule(static)
            for (int64_t rr = 0; rr < R; rr++) {
                while (rr >= off[j+1]) j++;
                while (rr <  off[j])   j--;   /* first row of a static chunk may sit behind */
                const coli_w_mxfp4 *w = ws[j];
                const int64_t o = rr - off[j];
                const int64_t I = w->I, nblk = I / COLI_MXFP4_BLK, rowbytes = nblk * COLI_MXFP4_BYTES;
                const int r = arow[j];
                coli_a_i8 v = *a;
                /* one-row view of activation row `r`, matching the fallback
                 * path in coli_gemm_i4_multi -- mxfp4_row indexes a->q /
                 * a->scale by [r][*] against a->I, so give it I=w->I and
                 * n=1 with q/scale advanced to row r's data. */
                const int64_t anb = I / COLI_ABLK;
                v.q     = a->q     + (int64_t)r * I;
                v.scale = a->scale + (int64_t)r * anb;
                v.n = 1; v.I = I;
                mxfp4_row(ys[j], w->O, o, &v, I, w->blocks + o*rowbytes, nblk);
            }
        }
        return;
    }
    /* Fallback for cnt > MAXM: one matrix at a time through the standard
     * dispatch, on a one-row view of the activation (same shape as
     * coli_gemm_i4_multi's fallback). */
    for (int j = 0; j < cnt; j++) {
        const int64_t I = ws[j]->I, anb = I / COLI_ABLK;
        const int r = arow[j];
        coli_a_i8 v = *a;
        v.q = a->q + (int64_t)r*I; v.scale = a->scale + (int64_t)r*anb;
        v.n = 1; v.I = I;
        coli_gemm_mxfp4(ys[j], &v, ws[j]);
    }
}

/* MXFP4 -> the int4 kernel's buffer shapes (2026-09-14). One block_mxfp4 is 17
 * bytes: E8M0 exponent, then 16 bytes whose LOW nibble is element j and HIGH
 * nibble element j+16 (c/ggml_dequant.h, PROVENANCE-2). The int4 buffers hold
 * element e in byte e/2, nibble e&1, with one float scale per 32 -- so this moves
 * every nibble to its int4 slot UNCHANGED (still the e2m1 code; the MXFP4_LUT
 * shader decodes it) and writes the E8M0 scale already halved, the same
 * gguf_e8m0_to_fp32_half the CPU kernel multiplies by. No value is rounded. */
/* ggml_e8m0_to_fp32_half, bit pattern verbatim (c/ggml_dequant.h PROVENANCE-2;
 * gemm_mxfp4.cpp carries the same static). 2^(x-128): the E8M0 scale halved. */
static inline float mxfp4_e8m0_half_rp(uint8_t x) {
    uint32_t bits = (x < 2) ? ((uint32_t)0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float r; memcpy(&r, &bits, sizeof r); return r;
}
int coli_mxfp4_repack_i4_ref(const uint8_t *blocks, int64_t I, int64_t O, coli_w_i4 *out) {
    if (I % COLI_MXFP4_BLK) return 0;
    int64_t nb = I / COLI_MXFP4_BLK;
    out->I = I; out->O = O;
    out->q4 = (uint8_t*)malloc((size_t)O * (size_t)I / 2);
    out->bscale = (float*)malloc((size_t)O * (size_t)nb * sizeof(float));
    if (!out->q4 || !out->bscale) { free(out->q4); free(out->bscale); out->q4 = NULL; out->bscale = NULL; return 0; }
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *row = blocks + (size_t)o * (size_t)nb * COLI_MXFP4_BYTES;
        uint8_t *dq = out->q4 + (size_t)o * (size_t)I / 2;
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * COLI_MXFP4_BYTES;
            out->bscale[o * nb + b] = mxfp4_e8m0_half_rp(blk[0]);
            uint8_t *d = dq + b * 16;                     /* 32 nibbles = 16 bytes */
            for (int j = 0; j < 16; j++) {
                unsigned lo = blk[1 + j] & 0x0F, hi = blk[1 + j] >> 4;   /* elements j, j+16 */
                /* element j -> byte j/2 nibble j&1; element j+16 -> byte 8+j/2, same nibble */
                if (j & 1) { d[j / 2]     |= (uint8_t)(lo << 4); d[8 + j / 2] |= (uint8_t)(hi << 4); }
                else       { d[j / 2]      = (uint8_t)lo;        d[8 + j / 2]  = (uint8_t)hi; }
            }
        }
    }
    return 1;
}

/* SIMD repack (2026-09-14): the same nibble move as coli_mxfp4_repack_i4_ref,
 * 16 bytes at a time. Measured before this existed: 1.12 ms per 2880x2880
 * matrix, 3.4 ms per expert -- at one fetch per layer per token that would be
 * 120 ms of a ~500 ms token, so the byte loop could not be the fetch path.
 * Per block: lo = raw & 0xF (elements 0..15), hi = raw >> 4 (16..31); gather
 * evens then odds of each with one PSHUFB, and out[k] = even[k] | odd[k] << 4
 * for the low 8 bytes (lo) and the high 8 bytes (hi). test_vk_oai checks it
 * against the reference byte for byte before using it. */
int coli_mxfp4_repack_i4(const uint8_t *blocks, int64_t I, int64_t O, coli_w_i4 *out) {
#if defined(__x86_64__) && defined(__SSSE3__)
    if (I % COLI_MXFP4_BLK) return 0;
    int64_t nb = I / COLI_MXFP4_BLK;
    out->I = I; out->O = O;
    out->q4 = (uint8_t*)malloc((size_t)O * (size_t)I / 2);
    out->bscale = (float*)malloc((size_t)O * (size_t)nb * sizeof(float));
    if (!out->q4 || !out->bscale) { free(out->q4); free(out->bscale); out->q4 = NULL; out->bscale = NULL; return 0; }
    const __m128i m4  = _mm_set1_epi8(0x0F);
    const __m128i idx = _mm_setr_epi8(0,2,4,6,8,10,12,14, 1,3,5,7,9,11,13,15);
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *row = blocks + (size_t)o * (size_t)nb * COLI_MXFP4_BYTES;
        uint8_t *dq = out->q4 + (size_t)o * (size_t)I / 2;
        float *bs = out->bscale + o * nb;
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * COLI_MXFP4_BYTES;
            bs[b] = mxfp4_e8m0_half_rp(blk[0]);
            __m128i raw = _mm_loadu_si128((const __m128i*)(blk + 1));
            __m128i lo  = _mm_shuffle_epi8(_mm_and_si128(raw, m4), idx);                       /* [even lo | odd lo] */
            __m128i hi  = _mm_shuffle_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), idx);    /* [even hi | odd hi] */
            __m128i plo = _mm_or_si128(lo, _mm_slli_epi16(_mm_srli_si128(lo, 8), 4));        /* low 8 bytes valid */
            __m128i phi = _mm_or_si128(hi, _mm_slli_epi16(_mm_srli_si128(hi, 8), 4));
            _mm_storeu_si128((__m128i*)(dq + b * 16), _mm_unpacklo_epi64(plo, phi));
        }
    }
    return 1;
#else
    return coli_mxfp4_repack_i4_ref(blocks, I, O, out);
#endif
}
