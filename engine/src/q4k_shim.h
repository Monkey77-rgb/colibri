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

#include <stdint.h>

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

/* ---- Q6_K (added 2026-09-13) ------------------------------------------------
 * Why: Qwen3-235B-A22B Q4_K_M stores ffn_down_exps as Q6_K in 46 of 94 layers
 * (28.30 GiB, measured from all five shard headers). Without a native path those
 * experts take the eager dequant->f32->int4 load and cannot fit this box's RAM,
 * and the disk-resident expert store only accepted Q4_K. Same shim reason as
 * Q4_K above: gguf_dequant_q6_K lives in a C-only header.
 *
 * Block: 210 bytes (GgufBlockQ6K), 256 weights, 16 sub-blocks of 16 weights,
 * w = d * scales[k] * q with q in [-32, 31] and NO min term. The 16-weight
 * sub-block lines up exactly with the engine's COLI_ABLK=16 activation block. */
#define COLI_Q6K_BLOCK_BYTES 210

/* Decode one Q6_K super-block into q[256] (signed codes, element order identical
 * to gguf_dequant_q6_K's output) and ds[16] = d*scales[k], the effective scale of
 * weights [16k, 16k+16). ds[k]*(float)q[i] reproduces gguf_dequant_q6_K's value
 * bit for bit: same operands, same order. */
void coli_q6k_decode(const void *blk_bytes, int8_t q[256], float ds[16]);

/* The reference: c/ggml_dequant.h's gguf_dequant_q6_K (transcribed from
 * ggml-quants.c), exposed for tests only. */
void coli_q6k_dequant_ref(const void *blocks, float *dst, int64_t nblk);

#ifdef __cplusplus
}
#endif
#endif
