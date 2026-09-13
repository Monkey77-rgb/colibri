/* gemm_q4k.h — native GGUF Q4_K CPU GEMV: consume Q4_K blocks straight off the
 * on-disk layout, no dequant-to-f32 and no re-quantize-to-int4 at load time.
 *
 * WHY THIS EXISTS. The loader today does Q4_K -> f32 (c/ggml_dequant.h
 * gguf_dequant_q4_K) -> engine int4 (coli_quantize_w4_ex, gemm_i8.h/.cpp) --
 * two full passes over every weight, 188.7 s of a 207 s cold start on the
 * 30B MoE (model.cpp w4snap comment, 2026-09-13). The second quantization
 * also throws away 1.5 of Q4_K's 4.5 bits/weight for nothing: Q4_K already
 * has a per-32-weight scale+min, the same granularity engine int4 pays for
 * separately. This kernel reads the GGUF bytes AS THEY SIT ON DISK and does
 * the int8-activation dot product against the ORIGINAL nibbles, which is what
 * lets a future loader stream an expert's Q4_K blocks straight from disk per
 * token instead of holding a second, re-quantized copy in RAM at all.
 *
 * NUMERICS. Per c/ggml_dequant.h's dequantize_row_q4_K (transcribed from
 * ggml-quants.c, itself transcribed there from upstream -- see that file's
 * header for line numbers): a 256-weight super-block (GgufBlockQ4K, 144
 * bytes) decodes to
 *
 *     w_i = d*sc_b - dmin*m_b        (for i in sub-block b, 32 weights each)
 *
 * where d = f16(block.d), dmin = f16(block.dmin), and (sc_b, m_b) are one of
 * 8 packed 6-bit (scale, min) pairs unpacked by gguf_scale_min_k4. The 4-bit
 * code itself, q_i in [0,15], carries NO offset (unlike this engine's own
 * int4 format, which stores q+8) -- Q4_K's min term IS the zero point.
 *
 * An activation dot over one 32-weight sub-block is then
 *
 *     sum_i(a_i * w_i) = d*sc_b * sum_i(a_i*q_i)  -  dmin*m_b * sum_i(a_i)
 *
 * and with the engine's own int8 activation format (coli_a_i8: real a_i =
 * scale[blk] * q_i8, one scale AND one running sum per COLI_ABLK=16 codes),
 * splitting the 32-weight sub-block into its two COLI_ABLK halves h gives
 *
 *     contribution(h) = scale[h] * ( d*sc_b * dot16(q_i8, q4) - dmin*m_b * sum[h] )
 *
 * -- the exact "sum_q * d*sc - a_sum * dmin*m, then times the activation
 * scale" the task asked for. Summing the two halves of a sub-block, the four
 * sub-block-pairs of a super-block and every super-block of a row reproduces
 * dequant->f32->coli_gemm_i8-style int8-activation dot up to float summation
 * ORDER only (the reductions themselves are exact integer arithmetic, same
 * discipline as gemm_i8.cpp's -ffp-contract=off note) -- there is no
 * agreed-bit-exactness claim across summation orders, so tests compare with a
 * relative tolerance, not `==`, exactly as the model-level NLL comparisons in
 * this codebase already do.
 *
 * SCOPE, PHASE 1. I (the weight matrix's input dimension) MUST be a multiple
 * of GGUF_QK_K=256 -- a super-block cannot cross a row boundary, which is
 * also why coli_gguf_load_raw's block array is safely row-major-block-major:
 * with I%256==0, block index o*(I/256)+sb never straddles two output rows.
 * Every shape in the two models this codebase targets (qwen2.5-3b-instruct,
 * qwen3-30b-a3b) divides evenly; a matrix that does not is simply not
 * native-eligible and the loader falls back to the existing dequant path,
 * exactly like COLI_W4BLK's own I%32 guard in gemm_i8.h.
 */
#ifndef COLI_GEMM_Q4K_H
#define COLI_GEMM_Q4K_H

#include <stdint.h>
#include <stddef.h>
#include "gemm_i8.h"    /* coli_a_i8 -- the shared activation format */
#include "q4k_shim.h"   /* COLI_Q4K_BLOCK_BYTES / COLI_Q4K_SUPERBLOCK */

#ifdef __cplusplus
extern "C" {
#endif

/* One matrix's worth of native Q4_K blocks, row-major-then-block-major: block
 * (o, sb) -- output row o, super-block sb -- sits at
 * blocks[(o*(I/256) + sb) * 144].
 *
 * `blocks` is either OWNED (malloc'd, e.g. by coli_gguf_load_raw at load
 * time -- freed by coli_free_q4k) or BORROWED (pointing into a future
 * disk-resident mapping/cache -- the reason for the indirection this struct
 * exists to keep clean, per the task's own note. coli_free_q4k only frees
 * when `owns` is set, same discipline as W4Side's `borrowed` flag in
 * model.cpp for the w4snap mmap. */
typedef struct {
    const uint8_t *blocks;
    int            owns;
    int64_t        I, O;
} coli_w_q4k;

/* Release `blocks` iff w->owns. Always safe to call, including on a
 * zero-initialized struct. */
void coli_free_q4k(coli_w_q4k *w);

/* y[n][O] = a . W^T for a native Q4_K matrix. n is a->n; n=1 (decode) is the
 * hot case this kernel exists for, but any n is correct. Dispatches on ISA;
 * see coli_gemm_q4k_kernel to ask which one without running it. */
void coli_gemm_q4k(float *y, const coli_a_i8 *a, const coli_w_q4k *w);

/* Scalar reference. Deliberately the plainest possible loop -- no SIMD, no
 * reordering beyond what the algebra above requires -- so it is the
 * DEFINITION the dispatched kernel is judged against, same role
 * coli_gemm_i8_ref plays for gemm_i8.cpp. Exported (not static) so
 * tests/test_gemm_q4k.c can call it directly. */
void coli_gemm_q4k_ref(float *y, const coli_a_i8 *a, const coli_w_q4k *w);

/* cnt independent Q4_K GEMVs in one parallel region, same shape and purpose
 * as coli_gemm_i4_multi in gemm_i8.h: a layer's K selected CPU-side experts
 * become one row-space instead of K*3 separate fork/joins. BIT-EXACT with cnt
 * separate coli_gemm_q4k calls -- shares the same per-row routine. */
void coli_gemm_q4k_multi(float *const *ys, const coli_a_i8 *a, const int *arow,
                         const coli_w_q4k *const *ws, int cnt);

/* Which kernel a call would use, without running it -- same diagnostic role
 * as coli_gemm_i4_kernel. */
const char *coli_gemm_q4k_kernel(int n);

#ifdef __cplusplus
}
#endif
#endif
