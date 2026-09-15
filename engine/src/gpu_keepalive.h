/* gpu_keepalive.h -- engine-owned GPU memory-clock keepalive (2026-09-15).
 *
 * WHY. Measured 09-14/09-15 on the RTX 4070 (goss15/16/17): decode is bursty
 * (~3 ms kernels, ~100/s), and the driver drops the MEMORY clock to 1.7-1.8 GHz
 * (P5/P8) between them; the logit head and the batched experts are memory-bound,
 * so they ran 2-4x slower (head 877-3,600 ms vs 254-394 ms per 96 tokens) and
 * decode drifted 2.5 -> 2.1 tok/s with NO code change. An SM-only keepalive did
 * nothing; a memory-streaming one (32 MB int4 GEMM x4 every 2 ms, ~35 % util)
 * held P2 and gave +0.1-0.2 tok/s on both backends, A/B/A/B. PowerMizer cannot
 * be set without root, so this is the lever the engine owns.
 *
 * HOW. A second, independent Vulkan context on its own thread submits the
 * streaming GEMM every period. It shares nothing with the model's context, so
 * nothing in vk_backend.c needs to be thread-safe for it. Works beside the CUDA
 * backend too (the binary still has Vulkan). Costs GPU power and ~35 % of the
 * GPU's time; measured net positive here, so `--tune` turns it on for discrete
 * GPUs; COLI_GPU_KEEPALIVE=0/1 overrides, COLI_GPU_KEEPALIVE_US the period. */
#ifndef COLI_GPU_KEEPALIVE_H
#define COLI_GPU_KEEPALIVE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Start the thread; 0 on success, -1 with err filled (no Vulkan, no device).
 * Idempotent: a second call while running returns 0. */
int  coli_gpu_keepalive_start(int period_us, char *err, size_t errcap);
/* Stop and join; prints "keepalive: N submits, busy X s (Y %)" to stderr when it ran. */
void coli_gpu_keepalive_stop(void);
int  coli_gpu_keepalive_running(void);
#ifdef __cplusplus
}
#endif
#endif
