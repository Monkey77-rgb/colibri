/* gemm_i8.h — int8 GEMM with dispatch on (ISA x batch size x residency).
 *
 * THE ONE IDEA THIS FILE EXISTS FOR. Every CPU inference engine picks its kernel
 * from the ISA. That is not enough, and we have the measurement:
 *
 *   AMD 9800X3D, int8 GEMM, weights 448 MiB streaming from RAM, min of 9 reps
 *     n=1   AVX-512 VNNI 0.83x     n=2  0.78x     n=4  1.17x    n>=8  1.18-1.26x
 *   same kernel, weights 16 MiB resident in the 96 MiB L3
 *     uniform 1.28x, every n from 1 to 64
 *
 * Below n=4 the loop is bound by weight BYTES -- both kernels sit on the same
 * ~70 GB/s DRAM ceiling -- so the wider ISA is strictly worse. An engine that
 * ranks ISAs monotonically and always takes the widest cannot express that.
 * ggml does exactly this (ggml-cpu/arch/x86/cpu-feats.cpp,
 * ggml_backend_cpu_x86_score() is a feature bitmask), so on a Zen 4/5 box it
 * always selects the AVX-512 variant and has no way to prefer AVX2 for decode.
 *
 * WE ARE NOT ALONE ON THE NUMBER 4. Read from a local llama.cpp checkout at
 * b8252, not recalled:
 *   ggml/src/ggml-cpu/repack.cpp:3104  "If there are more than three rows in
 *     src1, use gemm; otherwise, use gemv" -- and it runs GEMM over the largest
 *     multiple of 4, then GEMV for the remainder. We copy that tail handling.
 *   ggml/src/ggml-cpu/llamafile/sgemm.cpp:3689  "only enable sgemm for prompt
 *     processing", `if (n < 2) return false;`
 * MLX goes further and makes the limit a tuned function of (batch, K, N, GPU
 * generation), returning 12..33 (mlx/backend/metal/quantized.cpp,
 * get_qmv_batch_limit). So 4 is a floor for our shapes, not a universal law, and
 * COLI_GEMM_MIN_WIDE is therefore a tunable, not a constant.
 *
 * WEIGHT STORAGE IS UNSIGNED, ON PURPOSE. AVX-512 VPDPBUSD is u8 x s8. Offset
 * the ACTIVATIONS and you need a per-weight-block sum table: +12.5% of model
 * size. Offset the WEIGHTS and the correction is 128*sum(activation block),
 * which is n*I values against I*O weights, i.e. free:
 *
 *     dpbusd(u, x) = sum((q+128)*x) = sum(q*x) + 128*sum(x)
 *
 * exact in integer arithmetic. The AVX2 kernel reads the SAME bytes, widening
 * unsigned and subtracting 128 in int16 -- one extra instruction per 16 weights,
 * in the regime that is DRAM-bound anyway. One weight array, both kernels.
 * (Holding a signed and an unsigned copy side by side doubles weight memory:
 * measured 6.45 GiB vs 3.28 GiB on a 3B model. Do not do it.)
 *
 * THE i16 OVERFLOW TRAP, from ik_llama.cpp PR #141 (merged 2024-12-14), author's
 * own words: "I forgot for the 177'th time that the unsigned integers still need
 * to be within 0...127, else adding up two adjacent products may overflow the
 * int16_t range (and gets silently truncated if it does)". Their buggy version
 * scored perplexity 7.3725 against 7.3443 correct -- small enough to pass a
 * smoke test. Any kernel here that accumulates in int16 must state why it cannot
 * overflow. The AVX2 path below is safe because _mm256_madd_epi16 accumulates
 * into int32, not int16; the VNNI path is safe because dpbusd accumulates into
 * int32 directly.
 */
#ifndef COLI_GEMM_I8_H
#define COLI_GEMM_I8_H

#include <stdint.h>
#include <stddef.h>
#include "cpu_features.h"

/* Rows at or above which a wide-ISA kernel is allowed. Tunable per target; see
 * the measured table above for why the default is 4 and not 1. */
#ifndef COLI_GEMM_MIN_WIDE
#define COLI_GEMM_MIN_WIDE 4
#endif

/* Activation block length. One scale per this many activations. 16 keeps the
 * quantization error low and matches the block the AVX2 kernel reduces over. */
#define COLI_ABLK 16

/* Quantized weight matrix: O rows of I int8 values, stored offset-to-unsigned
 * (u = q + 128), one f32 scale per output row. */
typedef struct {
    uint8_t *qu;      /* [O][I], q+128 */
    float   *scale;   /* [O] */
    int64_t  I, O;
} coli_w_i8;

/* Quantized activations: n rows of I int8 values, one scale AND one sum per
 * COLI_ABLK block. The sum is what makes the unsigned-weight trick free. */
typedef struct {
    int8_t  *q;       /* [n][I] */
    float   *scale;   /* [n][I/COLI_ABLK] */
    int32_t *sum;     /* [n][I/COLI_ABLK]  sum of q over each block */
    int      n;
    int64_t  I;
} coli_a_i8;

/* Quantize n rows of f32 activations into `out`. Caller owns the buffers. */
void coli_quantize_a(coli_a_i8 *out, const float *x, int n, int64_t I);

/* y[n][O] = a . W^T, dispatching on (ISA, n, residency).
 * Every kernel it can select is bit-exact with every other -- the arithmetic is
 * integer and the reduction order per output element is fixed. If a future
 * kernel is not bit-exact it does NOT belong behind this call; give it its own
 * entry point so the difference is visible. */
void coli_gemm_i8(float *y, const coli_a_i8 *a, const coli_w_i8 *w);

/* Which kernel would be chosen, without running it. For diagnostics and for
 * tests that must assert the dispatch actually moved -- a benchmark comparing
 * "VNNI vs AVX2" that silently ran AVX2 twice is the failure this prevents. */
const char *coli_gemm_i8_kernel(int n, int64_t I, int64_t O);

#endif
