/* backend.h — the accelerator seam (2026-09-14).
 *
 * WHY. Until today model.cpp called coli_vk_* directly in 74 places, so "GPU"
 * meant "Vulkan" by construction and a second accelerator (CUDA on the 4070,
 * libtorch for the devices torch reaches -- ROCm, MPS, XPU) had nowhere to plug
 * in. The owner's direction (2026-09-14) is one engine that detects its hardware
 * and uses whatever is there. This header is the one table every backend fills.
 *
 * DESIGN. One struct of function pointers whose signatures are the coli_vk_*
 * signatures with `coli_vk *` replaced by `void *ctx`. The list lives in ONE
 * X-macro (COLI_BE_FUNCS) so the struct, the Vulkan adapter (backend_vk.c), the
 * default "declined" stubs (backend.c) and any new backend are generated from
 * the same source and cannot drift apart. A backend that does not implement an
 * entry leaves it NULL; coli_backend_open() fills every NULL with a stub that
 * returns the DECLINE value in the 5th column, which is exactly what model.cpp
 * already handles: has_*() -> 0 keeps the CPU path, gemm-style calls -> -1
 * keeps the CPU path for that call. The semantics of every entry are documented
 * once, in vk_backend.h, and are NOT repeated here -- a CUDA/torch implementer
 * reads that file as the contract.
 *
 * NOT IN THE TABLE: test-only entry points (coli_vk_attn_ref, rope_run,
 * kvwrite_run, bench_gemm4, probe_submit_ns, prof_dump). Tests call the Vulkan
 * backend directly, as before.
 *
 * SELECTION. coli_backend_open("auto"|"vulkan"|"cuda"|"torch", ...) -- "auto"
 * takes the first that opens in the order vulkan, cuda, torch, and says which
 * in `name`. The caller (main.cpp) decides the order from coli_hw_plan; this
 * file only knows how to open. Absence is a normal answer, never an error.
 */
#ifndef COLI_BACKEND_H
#define COLI_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "gemm_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

/* X(return_type, name, (params, void *ctx first), (args, CTX first -- the implementer defines CTX as its cast of ctx), decline_value) */
#define COLI_BE_FUNCS(X) \
  X(const char*, device_name, (void *ctx), (CTX), "") \
  X(const char*, memdesc,     (void *ctx), (CTX), "") \
  X(const char*, memdesc2,    (void *ctx), (CTX), "") \
  X(int, is_integrated, (void *ctx), (CTX), 0) \
  X(int, dot_used,      (void *ctx), (CTX), 0) \
  X(int, upload_w,  (void *ctx, const coli_w_i8 *w), (CTX, w), -1) \
  X(int, gemm,      (void *ctx, int wh, const coli_a_i8 *a, float *y), (CTX, wh, a, y), -1) \
  X(int, has_i4,    (void *ctx), (CTX), 0) \
  X(int, upload_w4, (void *ctx, const coli_w_i4 *w), (CTX, w), -1) \
  X(int, has_mx,    (void *ctx), (CTX), 0) \
  X(int, upload_w4_mx, (void *ctx, const coli_w_i4 *w), (CTX, w), -1) \
  X(int, slot_alloc_mx, (void *ctx, int64_t I, int64_t O), (CTX, I, O), -1) \
  X(int, slot_fill,     (void *ctx, int h, const coli_w_i4 *w), (CTX, h, w), -1) \
  X(int, slot_fill_async, (void *ctx, int h, const coli_w_i4 *w), (CTX, h, w), -1) \
  X(int, slot_fill_wait,  (void *ctx), (CTX), 0) \
  X(const char*, fill_mode, (void *ctx), (CTX), "synchronous (backend has no async fill)") \
  X(int, upload_begin,  (void *ctx), (CTX), 0) \
  X(int, upload_end,    (void *ctx), (CTX), 1) \
  X(int, gemm4,     (void *ctx, int wh, const coli_a_i8 *a, float *y), (CTX, wh, a, y), -1) \
  X(int, gemm4_qkv, (void *ctx, const int *wh, const coli_a_i8 *a, float **ys), (CTX, wh, a, ys), -1) \
  X(int, has_ffn,   (void *ctx), (CTX), 0) \
  X(int, ffn4,      (void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y), (CTX, hg, hu, hd, a, y), -1) \
  X(int, has_ffn_oai, (void *ctx), (CTX), 0) \
  X(int, ffn4_oai,  (void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y, const float *bg, const float *bu, float alpha, float limit), (CTX, hg, hu, hd, a, y, bg, bu, alpha, limit), -1) \
  X(int, moe4,      (void *ctx, const int *hg, const int *hu, const int *hd, int nexp, const coli_a_i8 *a, float *y), (CTX, hg, hu, hd, nexp, a, y), -1) \
  X(int, moe4_begin,(void *ctx, const int *hg, const int *hu, const int *hd, int nexp, const coli_a_i8 *a), (CTX, hg, hu, hd, nexp, a), -1) \
  X(int, moe4_end,  (void *ctx, float *y), (CTX, y), -1) \
  X(int, has_attn,  (void *ctx), (CTX), 0) \
  X(int, kv_init,   (void *ctx, int layers, int slots, int kv_heads, int kv_ctx, int hd), (CTX, layers, slots, kv_heads, kv_ctx, hd), -1) \
  X(int, kv_ready,  (void *ctx), (CTX), 0) \
  X(size_t, kv_bytes, (void *ctx), (CTX), 0) \
  /* VK_EXT_memory_budget (2026-09-16), the DEVICE_LOCAL heap's live heapBudget/heapUsage: \
   * so the r11 fused-block-KV decline (model.cpp) can print the real free-VRAM figure \
   * instead of "not reported". -1 declined (backend has no query, or the device did not \
   * advertise the extension) -- a NULL entry here gets backend.c's declining stub \
   * automatically, so cuda and torch need no changes to keep building. */ \
  X(int, mem_budget, (void *ctx, uint64_t *budget, uint64_t *usage), (CTX, budget, usage), -1) \
  X(int, kv_load,   (void *ctx, int layer, const float *K, const float *V), (CTX, layer, K, V), -1) \
  X(int, kv_get,    (void *ctx, int layer, float *K, float *V), (CTX, layer, K, V), -1) \
  X(int, kv_ctx,    (void *ctx), (CTX), 0) \
  X(int, kv_put,    (void *ctx, int slot, int kvh, int pos, int is_v, const float *row), (CTX, slot, kvh, pos, is_v, row), -1) \
  X(int, kv_write,  (void *ctx, int layer, int slot, int pos0, int count, const float *Khost, const float *Vhost), (CTX, layer, slot, pos0, count, Khost, Vhost), -1) \
  X(int, attn_sinks_upload, (void *ctx, const float *sinks, size_t nfloat), (CTX, sinks, nfloat), -1) \
  X(int, attn_ex,   (void *ctx, int layer, const float *q, float *out, const int *meta, int n, int H, float scale, int window, int sink_off), (CTX, layer, q, out, meta, n, H, scale, window, sink_off), -1) \
  X(int, has_qknorm,    (void *ctx), (CTX), 0) \
  X(int, qknorm_upload, (void *ctx, const float *w, size_t nfloat), (CTX, w, nfloat), -1) \
  X(int, rope_bias_upload, (void *ctx, const float *bias, size_t nfloat), (CTX, bias, nfloat), -1) \
  X(int, rope_cs_upload,   (void *ctx, const float *cs, size_t nfloat), (CTX, cs, nfloat), -1) \
  X(int, has_block, (void *ctx), (CTX), 0) \
  X(int, attn_block,(void *ctx, int layer, const int *wh, const coli_a_i8 *a, const int *meta, int n, int H, int KVH, int hd, int neox, int bias_off, int qk_off, float qk_eps, float scale, int stop_attn, float *y), (CTX, layer, wh, a, meta, n, H, KVH, hd, neox, bias_off, qk_off, qk_eps, scale, stop_attn, y), -1)

typedef struct coli_backend {
    const char *name;      /* "vulkan" | "cuda" | "torch" -- the one that opened */
    void       *ctx;       /* the backend's own state (coli_vk* for vulkan) */
    void      (*close)(void *ctx);
#define X(r, n, P, A, D) r (*n) P;
    COLI_BE_FUNCS(X)
#undef X
} coli_backend;

/* Open one backend by name, or the first available with "auto" (order: the
 * comma-separated COLI_BACKEND_ORDER env if set, else "vulkan,cuda,torch" -- the measured order, see hw_detect.c).
 * Returns NULL and fills err when none opens. Every NULL entry in the returned
 * table is replaced by a declining stub, so callers never test for NULL. */
coli_backend *coli_backend_open(const char *which, char *err, size_t errcap);
void          coli_backend_close(coli_backend *be);
/* Which backends this BINARY was built with, as "vulkan,cuda" etc. -- for the
 * hardware planner, which must never choose a backend the build lacks. */
const char   *coli_backend_built(void);
/* 1 = integrated (UMA), 0 = discrete, -1 = no usable device, for `name`
 * ("vulkan"/"cuda"). Probes without keeping a device open. */
int           coli_backend_probe_class(const char *name);
/* Startup calibration (backend_bench.cpp): open `name`, time one 2880x2880 int4
 * gemm4 n=1 (best of `reps` after warm-up), close. Microseconds, or -1 with err.
 * A proxy for the memory-bound GEMV share of a token, not the whole token. */
double        coli_backend_bench_gemv_us(const char *name, int reps, char *err, size_t errcap);

/* Constructors, one per backend; each returns a table with ctx set and the
 * entries it implements filled (the rest NULL), or NULL with err. Present only
 * in builds that have that backend. torch is a dlopen'ed plugin
 * (libcoli_torch.so exporting coli_backend_torch_open with this signature). */
coli_backend *coli_backend_vk_open  (char *err, size_t errcap);
coli_backend *coli_backend_cuda_open(char *err, size_t errcap);
int           coli_cuda_probe_class(void);   /* backend_cuda.cu: 0 discrete, -1 none */

#ifdef __cplusplus
}
#endif
#endif
