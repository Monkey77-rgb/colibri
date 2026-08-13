/* make_ggml_dequant_fixture.c — generate bit-exact fixtures for
 * tests/test_ggml_dequant.c from the REAL llama.cpp reference codec.
 *
 * This is the fixture-generation tool, following the same rule as
 * tools/make_e8_fixture.py: the fixture must come from the format's reference
 * implementation, never from the kernel under test. Here the reference is not a
 * reimplementation of any kind -- this program links directly against a locally
 * built libggml-base.so (from /home/monkey/.unsloth/llama.cpp) and calls its
 * actual quantize_row_*_ref / dequantize_row_* functions. It does not include any
 * ggml headers (to avoid a header/ABI version tangle); the six extern
 * declarations below are copied from ggml/src/ggml-quants.h and the block struct
 * layouts are treated as opaque by this file (nblk * block_size bytes, computed
 * the same way ggml_dequant.h computes it, and asserted equal at runtime).
 *
 * BUILD (not part of the normal colibri build -- one-off, requires the llama.cpp
 * checkout to be built first):
 *   cmake -S /home/monkey/.unsloth/llama.cpp -B /tmp/ggml_build \
 *         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 *         -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF \
 *         -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF
 *   cmake --build /tmp/ggml_build --target ggml -j
 *   cc -O2 -std=c99 -o make_ggml_dequant_fixture make_ggml_dequant_fixture.c \
 *      -I../.. -L/tmp/ggml_build/bin -lggml-base -Wl,-rpath,/tmp/ggml_build/bin -lm
 *   ./make_ggml_dequant_fixture ../tests/fixtures
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---- extern declarations, copied verbatim from ggml/src/ggml-quants.h -------
 * (lines 27-30 for the _ref quantizers, 53-56 for the dequantizers). Block
 * pointer types are declared void* here deliberately: this file never looks
 * inside the structs, only passes opaque buffers sized nblk*block_size, exactly
 * as gguf_reader.h's caller will. */
extern void quantize_row_q3_K_ref(const float *x, void *y, int64_t k);
extern void quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
extern void quantize_row_q5_K_ref(const float *x, void *y, int64_t k);
extern void quantize_row_q6_K_ref(const float *x, void *y, int64_t k);
extern void dequantize_row_q3_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q4_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q5_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q6_K(const void *x, float *y, int64_t k);
extern float ggml_fp16_to_fp32(uint16_t h);   /* used as-is for the F16/BF16 fixture */
extern float ggml_bf16_to_fp32(uint16_t h);

#define QK_K 256
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { const char *name; int block_bytes; void (*quant)(const float*, void*, int64_t); void (*dequant)(const void*, float*, int64_t); } Kind;

static void wr_i64(FILE *f, int64_t v) { fwrite(&v, sizeof v, 1, f); }

/* One fixture per K-quant type: NBLK super-blocks of random-ish weights (mimics
 * real model weight distribution: small magnitude, occasional outlier), quantized
 * and dequantized by the REAL reference codec. test_ggml_dequant.c re-dequantizes
 * the same quantized bytes with our port and diffs bit-for-bit against the f32
 * array stored here. */
static void gen_case(const char *outdir, Kind k, int64_t nblk, uint32_t seed, const char *tag) {
    int64_t nelem = nblk * QK_K;
    float *x = malloc(nelem * sizeof(float));
    srand(seed);
    for (int64_t i = 0; i < nelem; i++) {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double gauss = sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);      /* Box-Muller */
        double v = gauss * 0.02;
        if ((rand() % 97) == 0) v *= 12.0;                          /* rare outlier */
        x[i] = (float)v;
    }
    void *q = malloc((size_t)nblk * k.block_bytes);
    float *yref = malloc(nelem * sizeof(float));
    k.quant(x, q, nelem);
    k.dequant(q, yref, nelem);

    char path[1024];
    snprintf(path, sizeof path, "%s/dequant_%s_%s.bin", outdir, k.name, tag);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    wr_i64(f, nblk);
    wr_i64(f, nelem);
    fwrite(q, k.block_bytes, (size_t)nblk, f);
    fwrite(yref, sizeof(float), (size_t)nelem, f);
    fclose(f);
    printf("wrote %s: nblk=%lld nelem=%lld bytes/block=%d\n", path, (long long)nblk, (long long)nelem, k.block_bytes);
    free(x); free(q); free(yref);
}

/* F16/BF16 fixture: EXHAUSTIVE over all 65536 uint16 bit patterns (subnormals,
 * inf, nan, both signs, both zeros included), dequantized by ggml's own
 * ggml_fp16_to_fp32 / ggml_bf16_to_fp32. */
static void gen_half_case(const char *outdir, const char *name, float (*fn)(uint16_t)) {
    char path[1024];
    snprintf(path, sizeof path, "%s/dequant_%s.bin", outdir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int64_t n = 65536;
    wr_i64(f, n);
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = (uint16_t)i;
        fwrite(&h, sizeof h, 1, f);
    }
    for (int64_t i = 0; i < n; i++) {
        float v = fn((uint16_t)i);
        fwrite(&v, sizeof v, 1, f);
    }
    fclose(f);
    printf("wrote %s: n=%lld (exhaustive)\n", path, (long long)n);
}

int main(int argc, char **argv) {
    const char *outdir = argc > 1 ? argv[1] : "../tests/fixtures";

    Kind q3 = {"q3_K", 110, quantize_row_q3_K_ref, dequantize_row_q3_K};
    Kind q4 = {"q4_K", 144, quantize_row_q4_K_ref, dequantize_row_q4_K};
    Kind q5 = {"q5_K", 176, quantize_row_q5_K_ref, dequantize_row_q5_K};
    Kind q6 = {"q6_K", 210, quantize_row_q6_K_ref, dequantize_row_q6_K};

    /* main case: enough blocks to exercise every code path (multiple 128/64-elem
     * sub-loops), realistic weight-like distribution */
    gen_case(outdir, q3, 37, 0xC0FFEEu, "multi");
    gen_case(outdir, q4, 37, 0xC0FFEEu, "multi");
    gen_case(outdir, q5, 37, 0xC0FFEEu, "multi");
    gen_case(outdir, q6, 37, 0xC0FFEEu, "multi");

    /* single-block minimum case */
    gen_case(outdir, q3, 1, 0x1234u, "single");
    gen_case(outdir, q4, 1, 0x1234u, "single");
    gen_case(outdir, q5, 1, 0x1234u, "single");
    gen_case(outdir, q6, 1, 0x1234u, "single");

    gen_half_case(outdir, "f16", ggml_fp16_to_fp32);
    gen_half_case(outdir, "bf16", ggml_bf16_to_fp32);

    return 0;
}
