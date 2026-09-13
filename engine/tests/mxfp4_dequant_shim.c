/* mxfp4_dequant_shim.c — thin extern-linkage wrapper around
 * c/ggml_dequant.h's gguf_dequant_mxfp4 / gguf_dequant_q8_0, compiled as PLAIN
 * C (not -x c++).
 *
 * WHY THIS EXISTS. test_mxfp4.c is built the same way test_gemm_i8.c already
 * is (see engine/Makefile): compiled with `-x c++` so its C++ name mangling
 * matches the C++-compiled gemm_i8.cpp / gemm_mxfp4.cpp it links against
 * (neither header declares `extern "C"`, so consistent mangling on both sides
 * is what makes the link work at all). But c/ggml_dequant.h transitively
 * includes c/st.h and c/json.h, which are plain C89/C99 -- `void *` assigned
 * to a typed pointer without a cast, `malloc`'s return value used directly,
 * etc. -- and fail to compile as C++ (-fpermissive errors, verified: this
 * split was written after that exact build failure, not speculatively). This
 * shim isolates the two dequant functions the test actually needs behind a
 * C-linkage boundary compiled with a real C compiler, so the C++-compiled test
 * never has to include st.h/json.h at all. */
#include "../../c/ggml_dequant.h"

void coli_test_dequant_mxfp4(const void *src, float *dst, long long nblk) {
    gguf_dequant_mxfp4(src, dst, (int64_t)nblk);
}
void coli_test_dequant_q8_0(const void *src, float *dst, long long nblk) {
    gguf_dequant_q8_0(src, dst, (int64_t)nblk);
}
