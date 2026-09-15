/* gemm_mxfp4.h — native CPU GEMV over raw GGUF MXFP4 (ttype 39) blocks.
 *
 * WHY THIS EXISTS
 * ----------------
 * First step toward gpt-oss-120b (MXFP4.gguf): a Stage-1 dequant path already
 * exists (c/ggml_dequant.h's gguf_dequant_mxfp4 -> f32 -> coli_quantize_w4/i8),
 * but that throws away the 4.25-bit/weight on-disk format and re-quantizes to
 * whatever this engine's OWN weight format is. This file is the OTHER path:
 * dot an int8-quantized activation row directly against the raw 17-byte
 * block_mxfp4 blocks, with no intermediate f32 or re-quantization step.
 *
 * NOT WIRED IN. Per the task this file was written under: model.cpp's dispatch
 * is being extended for a native Q4_K weight side-table on a different branch;
 * this file only delivers the kernel, the weight struct, and its own test so
 * that work can plug it in later. Nothing in model.cpp includes this header.
 *
 * LAYOUT — GGUF's block_mxfp4, NOT c/quant.h's matmul_mxfp4 layout. See
 * c/ggml_dequant.h's PROVENANCE-2 comment for where the block layout, the
 * doubled e2m1 table and the E8M0-half scale were transcribed from (ggml
 * upstream, fetched 2026-09-13). One row of a weight matrix is
 * ceil(I/32) contiguous 17-byte blocks:
 *     byte 0      : uint8_t e        (E8M0 shared exponent for this block's 32
 *                   weights; decoded scale = 2^(e-128), already halved to match
 *                   the doubled e2m1 table below — see gguf_e8m0_to_fp32_half)
 *     bytes 1..16 : uint8_t qs[16]   (2 nibbles/byte; for j in [0,16), the LOW
 *                   nibble of qs[j] is weight element j and the HIGH nibble is
 *                   element j+16 — NOT interleaved pairs)
 * A weight matrix of O rows requires I % 32 == 0 (true for every gpt-oss-120b
 * tensor measured: embedding_length = feed_forward_length = 2880 = 90*32).
 *
 * NUMERICS. Per output row, per MXFP4 group of 32 weights, the group splits
 * into the two COLI_ABLK=16 activation sub-blocks this engine already
 * quantizes activations into (see gemm_i8.h): each sub-block contributes an
 * exact int32 dot of quantized activation against the decoded (still-integer,
 * still-doubled) e2m1 codes, then the two sub-block partial sums are each
 * multiplied by (their own activation scale) and the block's single weight
 * scale, and accumulated in f32 across blocks in increasing block order. That
 * is the SAME arithmetic dequant-then-dot performs, reassociated only at the
 * per-sub-block float multiply-accumulate (dequant->f32 dot multiplies and
 * fully reduces one weight at a time; this reduces 16 weights in integer first)
 * — bounded, not bit-exact, against a naive per-element dequant dot; see
 * engine/tests/test_mxfp4.c for the measured bound. The kernel itself has no
 * SIMD variant yet (OpenMP over output rows only), so it IS its own scalar
 * reference — coli_gemm_mxfp4 and coli_gemm_mxfp4_ref below run identical
 * arithmetic in identical order and are bit-exact with each other by
 * construction, not by coincidence; the split exists only so a test can call
 * the unparallelized form directly. */
#ifndef COLI_GEMM_MXFP4_H
#define COLI_GEMM_MXFP4_H

#include <stddef.h>
#include <stdint.h>
#include "gemm_i8.h"   /* coli_a_i8 */

#define COLI_MXFP4_BLK   32   /* weights per block_mxfp4 */
#define COLI_MXFP4_BYTES 17   /* bytes per block_mxfp4 (1 exponent + 16 nibble) */

/* Raw MXFP4 weight matrix: O rows of ceil(I/COLI_MXFP4_BLK) contiguous
 * block_mxfp4 blocks each (I must be a multiple of COLI_MXFP4_BLK; the kernel
 * does not check this and reads a truncated final block if it is not).
 * `blocks` may alias directly into an mmap'd GGUF file — `owned` says whether
 * coli_free_mxfp4 should free it. */
typedef struct {
    const uint8_t *blocks;   /* [O][ (I/COLI_MXFP4_BLK) * COLI_MXFP4_BYTES ] */
    int64_t        I, O;
    int             owned;   /* if set, coli_free_mxfp4() frees `blocks` */
} coli_w_mxfp4;

void coli_free_mxfp4(coli_w_mxfp4 *w);

/* y[n][O] = a . W^T. Requires w->I % COLI_MXFP4_BLK == 0 and
 * w->I % COLI_ABLK == 0 (COLI_ABLK=16 divides COLI_MXFP4_BLK=32, so the second
 * is implied by the first). OpenMP over output rows; decode-oriented (no batch
 * kernel yet, matching this being the GEMV entry point for gpt-oss decode). */
void coli_gemm_mxfp4(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w);

/* Same arithmetic as coli_gemm_mxfp4, serial (no #pragma omp). Since the two
 * differ ONLY in whether the outer row loop is parallelized — every row is an
 * independent reduction — they are bit-exact with each other unconditionally;
 * this entry point exists so a test can assert that without depending on
 * thread-count or scheduling. */
void coli_gemm_mxfp4_ref(float *y, const coli_a_i8 *a, const coli_w_mxfp4 *w);
/* Which kernel coli_gemm_mxfp4 runs on this CPU ("avx512vnni-mxfp4" or
 * "mxfp4-scalar"): a benchmark that silently timed the scalar path would look
 * like a result. */
const char *coli_gemm_mxfp4_kernel(void);

/* GPU side (2026-09-14): repack raw MXFP4 blocks into a coli_w_i4 whose nibbles
 * are the UNCHANGED e2m1 codes in int4 element order and whose bscale is the
 * E8M0 scale already halved -- the buffer shapes shaders/gemm_i4_mx_dp.comp
 * reads. malloc'd q4/bscale, caller frees. 0 on OOM or I %% 32 != 0. */
int coli_mxfp4_repack_i4(const uint8_t *blocks, int64_t I, int64_t O, coli_w_i4 *out);
/* The byte-loop reference the SIMD repack is checked against (test_vk_oai). */
int coli_mxfp4_repack_i4_ref(const uint8_t *blocks, int64_t I, int64_t O, coli_w_i4 *out);

/* cnt independent MXFP4 GEMVs in one parallel region, same idea as
 * coli_gemm_i4_multi (see gemm_i8.h): the row spaces of `ws[0..cnt)` are
 * concatenated into one team, so a call over an MoE layer's K selected experts
 * is one fork/join instead of K. Bit-exact with cnt sequential coli_gemm_mxfp4
 * calls (each output row's arithmetic is unchanged, only the OpenMP chunking
 * differs) and it is a decode-only concern in gpt-oss-120b: 4-of-128
 * experts/layer routed at n=1. */
void coli_gemm_mxfp4_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                            const coli_w_mxfp4 *const *ws, int cnt);

#endif
