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
