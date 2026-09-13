/* q4k_shim.h — a C shim over c/ggml_dequant.h, for exactly the reason
 * loader.h gives for wrapping gguf_reader.h/tok.h instead of porting them:
 * c/ggml_dequant.h includes c/st.h, which includes c/json.h, both of which
 * use the implicit `void*` conversions C permits and C++ rejects (confirmed
 * by trying to compile gemm_q4k.cpp against ggml_dequant.h directly --
 * -fpermissive errors in json.h/st.h, not in ggml_dequant.h itself). Editing
 * st.h/json.h to be C++-clean is out of scope for a kernel change and risks
 * the same "proven bit-exact over the live fleet" property loader.h protects.
 * So: this file stays C, includes ggml_dequant.h unchanged, and exposes just
 * the two primitives gemm_q4k.cpp needs -- f16 conversion and the packed
 * 6-bit scale/min unpack -- as a small, boring C ABI. The nibble unpacking
 * itself (mask+shift, scalar and AVX2) stays in gemm_q4k.cpp: it needs
 * neither st.h nor json.h, only the raw `qs` byte pointer this shim hands
 * back.
 */
#ifndef COLI_Q4K_SHIM_H
#define COLI_Q4K_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror c/ggml_dequant.h's GGUF_K_SCALE_SIZE-derived block size (144 bytes)
 * and GGUF_QK_K (256 weights) so callers do not need to include a C-only
 * header to know them. q4k_shim.c static_asserts both against the real
 * values in ggml_dequant.h at build time, so a future change to that file
 * that disagrees with these two constants fails the BUILD, not a test run
 * three weeks later. */
#define COLI_Q4K_BLOCK_BYTES 144
#define COLI_Q4K_SUPERBLOCK  256

/* Decode one Q4_K super-block's 8 sub-block (d*sc, dmin*m) effective
 * scale/min pairs -- exactly the `d1,m1`/`d2,m2` gguf_dequant_q4_K computes,
 * indexed 0..7 in super-block order (see gemm_q4k.h's own header for how sub-
 * block index maps to element range). `blk_bytes` must point to
 * COLI_Q4K_BLOCK_BYTES readable bytes: one on-disk GgufBlockQ4K, unmodified. */
void coli_q4k_decode_scales(const void *blk_bytes, float d[8], float m[8]);

/* The block's 128-byte packed-nibble field (GgufBlockQ4K.qs), as a raw
 * pointer into `blk_bytes` -- no copy, no decode. Nibble unpacking is the
 * caller's job (gemm_q4k.cpp), since that part needs no C-only header. */
const unsigned char *coli_q4k_qs(const void *blk_bytes);

#ifdef __cplusplus
}
#endif
#endif
