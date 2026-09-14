/* q4k_shim.c — see q4k_shim.h for why this is a separate C translation unit. */
#define _GNU_SOURCE
#include "q4k_shim.h"
#include "ggml_dequant.h"

/* Build-time proof the two constants in the header have not drifted from
 * ggml_dequant.h's own values -- same C99 negative-array-size trick that
 * file already uses for its own block-size assertions. */
typedef char q4k_shim_assert_block_bytes[(sizeof(GgufBlockQ4K) == COLI_Q4K_BLOCK_BYTES) ? 1 : -1];
typedef char q4k_shim_assert_superblock [(GGUF_QK_K == COLI_Q4K_SUPERBLOCK) ? 1 : -1];

void coli_q4k_decode_scales(const void *blk_bytes, float d[8], float m[8]) {
    const GgufBlockQ4K *blk = (const GgufBlockQ4K *)blk_bytes;
    const float fd = f16_to_f32(blk->d), fdmin = f16_to_f32(blk->dmin);
    for (int j = 0; j < 4; j++) {
        uint8_t sc, mn;
        gguf_scale_min_k4(2*j+0, blk->scales, &sc, &mn); d[2*j]   = fd*sc; m[2*j]   = fdmin*mn;
        gguf_scale_min_k4(2*j+1, blk->scales, &sc, &mn); d[2*j+1] = fd*sc; m[2*j+1] = fdmin*mn;
    }
}

const unsigned char *coli_q4k_qs(const void *blk_bytes) {
    return ((const GgufBlockQ4K *)blk_bytes)->qs;
}

/* ---- Q6_K -- see q4k_shim.h. A straight transcription of gguf_dequant_q6_K's
 * loop that writes the signed code instead of d*sc*q, so the two cannot drift
 * apart without test_gemm_q6k's decode check failing. */
typedef char q6k_shim_assert_block_bytes[(sizeof(GgufBlockQ6K) == COLI_Q6K_BLOCK_BYTES) ? 1 : -1];

void coli_q6k_decode(const void *blk_bytes, int8_t q[256], float ds[16]) {
    const GgufBlockQ6K *x = (const GgufBlockQ6K *)blk_bytes;
    const float d = f16_to_f32(x->d);
    for (int k = 0; k < 16; k++) ds[k] = d * x->scales[k];
    const uint8_t *ql = x->ql, *qh = x->qh;
    int8_t *y = q;
    for (int n = 0; n < GGUF_QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            y[l +  0] = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            y[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            y[l + 64] = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            y[l + 96] = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        y += 128; ql += 64; qh += 32;
    }
}

void coli_q6k_dequant_ref(const void *blocks, float *dst, int64_t nblk) {
    gguf_dequant_q6_K(blocks, dst, nblk);
}
