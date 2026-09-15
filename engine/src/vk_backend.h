/* vk_backend.h — Vulkan compute backend for the int8 GEMM.
 *
 * WHY VULKAN AND NOT A PORTABILITY LAYER. From the backend survey: llama.cpp
 * REMOVED Kompute in PR #14501 ("development for this backend has stopped"), and
 * its SYCL backend lost NVIDIA and AMD in 2026.02 because "the oneAPI plugin for
 * Nvidia & AMD GPU is unavailable" -- the cross-vendor promise died on toolchain
 * distribution, not performance. A portability layer you do not control is a
 * dependency that can strand you. Vulkan is the one API that reaches the Legion's
 * Radeon 780M, an Intel/NVIDIA desktop, an AMD laptop iGPU and Android from a
 * single shader.
 *
 * SCOPE. One kernel: the int8 GEMM. Not attention, not rope, not the sampler.
 *
 * BUFFERS ARE NOW PERSISTENT. The first version allocated, uploaded, downloaded
 * and destroyed four buffers per call. On a 2048x2048 matrix that put a ~1 ms
 * floor under every dispatch -- more than the arithmetic. Activation and output
 * buffers are now allocated once at the high-water mark and reused, and the
 * descriptor set is allocated once instead of per call. Measured effect below.
 *
 * DEVICE-LOCAL WEIGHTS on discrete GPUs. Weights used to live in HOST_VISIBLE
 * memory, which on a discrete card means system RAM and a PCIe crossing for
 * every read -- measured: the 4070 reported `heap1 22.8 GiB [HOST_VISIBLE
 * HOST_COHERENT]`, i.e. no DEVICE_LOCAL, against its 12 GiB of VRAM. Weights are
 * uploaded once, so paying a staging copy at load time to get them into VRAM is
 * obviously right THERE and pointless on a unified-memory APU. The choice is now
 * made from the device type rather than hardcoded either way.
 *
 * NOT BIT-EXACT WITH THE CPU, BY CONSTRUCTION. The shader reduces through shared
 * memory in a tree; the CPU accumulates sequentially. Different order, different
 * float rounding. Every other kernel in this engine is held to bit-exactness, so
 * this exception is called out loudly rather than quietly relaxed: the GPU path
 * is checked against a stated relative bound AND against a control that must
 * exceed it.
 */
#ifndef COLI_VK_BACKEND_H
#define COLI_VK_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "gemm_i8.h"

/* vk_backend.c is C (Vulkan's designated initialisers and void* conversions are
 * C idioms). Without this guard a C++ caller looks for mangled names and the
 * link fails with "undefined reference to coli_vk_init(char const*, ...)" --
 * note the argument list in the error, which is the tell. */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct coli_vk coli_vk;

/* Returns NULL and fills err when Vulkan, a device, or the shader is missing.
 * Absence is a normal answer -- a machine with no GPU is not an error state. */
coli_vk *coli_vk_init(const char *spv_path, char *err, size_t errcap);
void     coli_vk_free(coli_vk *v);

const char *coli_vk_device_name(coli_vk *v);
/* What memory the weights actually landed in. On a DISCRETE card a
 * HOST_VISIBLE-but-not-DEVICE_LOCAL type means every weight read crosses PCIe
 * instead of coming from VRAM, which is the difference between ~25 GB/s and
 * ~500 GB/s. Printed by the test so the limitation is visible, not inferred. */
const char *coli_vk_mem_desc(coli_vk *v);
int         coli_vk_is_integrated(coli_vk *v);
/* What memory the weights were GRANTED (not what was requested). memdesc is the
 * heap/type detail, memdesc2 is "DEVICE_LOCAL (staged)" or "HOST_VISIBLE". */
const char *coli_vk_memdesc (coli_vk *v);
const char *coli_vk_memdesc2(coli_vk *v);
/* 1 when the DP4a int4 kernel is the one actually in use, 0 for the scalar
 * fallback. Reported by --gpu so a run states which kernel produced it. */
int coli_vk_dot_used(coli_vk *v);
/* 1 = integrated (UMA), 0 = discrete, -1 = no usable device. Probes and frees;
 * safe to call before any upload. */
int coli_vk_probe_class(const char *spv_path);
int         coli_vk_wants_device_local(coli_vk *v);
/* Where the WEIGHTS ended up: "DEVICE_LOCAL (staged)" or "HOST_VISIBLE". On a
 * discrete card the difference is PCIe vs VRAM bandwidth. */
const char *coli_vk_weight_mem(coli_vk *v);

/* Upload a weight matrix once; returns an opaque handle index, or -1. Weights
 * are uploaded in the SAME offset-to-unsigned layout the CPU uses, so no
 * repacking happens on either side. */
int  coli_vk_upload_w(coli_vk *v, const coli_w_i8 *w);

/* y[n][O] = a . W^T on the GPU. Returns 0 on success. */
int  coli_vk_gemm(coli_vk *v, int wh, const coli_a_i8 *a, float *y);

/* ---- int4, the same two calls over the int4 weight format ----
 * Handles from coli_vk_upload_w4 index a SEPARATE table from coli_vk_upload_w
 * and are not interchangeable: handing an int4 handle to coli_vk_gemm would read
 * a half-length matrix and return plausible wrong numbers rather than fail.
 *
 * coli_vk_has_i4 reports whether shaders/gemm_i4.spv was found and compiled. It
 * is OPTIONAL -- a tree built without `make vk4` still initialises the backend
 * and serves int8 -- so a caller must check rather than assume, and both upload
 * and gemm return -1 when it is absent. */
int  coli_vk_has_i4(coli_vk *v);
int  coli_vk_upload_w4(coli_vk *v, const coli_w_i4 *w);
/* gpt-oss MXFP4 experts (2026-09-14): the SAME upload as coli_vk_upload_w4 over a
 * matrix repacked by the caller (nibble = raw e2m1 code in int4 element order,
 * bscale = E8M0 scale already halved), tagged so every GEMM over this handle
 * takes the MXFP4-decode pipeline (shaders/gemm_i4_mx_dp.spv). Returns -1 if
 * that pipeline is absent. The handle lives in the int4 table and is accepted by
 * coli_vk_ffn4_oai below and by coli_vk_gemm4 (which then runs the MXFP4
 * pipeline); handing it to coli_vk_ffn4 / coli_vk_moe4_begin is refused (-1). */
int  coli_vk_has_mx(coli_vk *v);
int  coli_vk_upload_w4_mx(coli_vk *v, const coli_w_i4 *w);
/* Expert SLOTS (2026-09-14, the design of FreeToken's GPU expert slot cache):
 * coli_vk_slot_alloc_mx allocates one MXFP4-tagged handle with device buffers
 * of the given shape and NO contents; coli_vk_slot_fill copies a repacked
 * matrix of exactly that shape into it (synchronous: staged copy + fence, the
 * same path the one-time upload uses). A slot is refilled with a different
 * expert as the cache evicts; the handle never changes. Returns -1 on any
 * failure, and a fill failure leaves the slot's contents UNDEFINED -- the caller
 * must mark it empty. */
int  coli_vk_slot_alloc_mx(coli_vk *v, int64_t I, int64_t O);
int  coli_vk_slot_fill(coli_vk *v, int h, const coli_w_i4 *w);

/* ASYNC SLOT FILL (2026-09-14). Measured 2026-09-14 on an RTX 4070: the
 * synchronous coli_vk_slot_fill above costs 8.7 ms per expert (three fills)
 * with the source RAM-resident, 6.6 ms at load time, for a ~13 MB copy that
 * PCIe 4.0 x16 moves in under 1 ms -- most of the cost is the staged
 * copy+submit+fence round trip ITSELF, not the bus. coli_vk_slot_fill_async
 * records the same staged copy into a command buffer submitted on a queue the
 * compute dispatch never uses (see vk_backend.c's queue-discovery comment for
 * how that queue is chosen, and coli_vk_fill_mode below for what a given
 * process actually got) and returns without waiting. coli_vk_slot_fill_wait
 * waits on every fill submitted since the last wait and is the ONLY place
 * that makes their writes visible to a subsequent compute dispatch -- no
 * other Vulkan call on v may be made on a slot between _async and _wait.
 * Falls back to calling coli_vk_slot_fill synchronously when the device
 * offered no second queue (coli_vk_fill_mode() says "synchronous" in that
 * case); same return convention, same undefined-slot-contents-on-failure
 * rule as the sync call. The async staging ring holds up to
 * COLI_VK_FILL_INFLIGHT (8) fills' worth of HOST_VISIBLE memory; a 9th async
 * fill before the next _wait reuses the oldest ring slot and blocks on ITS
 * fence first, rather than growing without bound. */
int  coli_vk_slot_fill_async(coli_vk *v, int h, const coli_w_i4 *w);
int  coli_vk_slot_fill_wait(coli_vk *v);
/* Which queue mode this process actually got: a dedicated transfer-only
 * family, a second queue in the compute family, or the synchronous fallback.
 * Printed by the test rather than inferred from the device name -- the same
 * physical GPU can expose different queue shapes under different drivers. */
const char *coli_vk_fill_mode(coli_vk *v);
/* Mean milliseconds per coli_vk_slot_fill / coli_vk_slot_fill_async call,
 * split the same way the task asked the code to be instrumented:
 * out[0]=memcpy into staging, out[1]=command recording + vkQueueSubmit,
 * out[2]=fence wait, out[3]=sum of the three. which=0 reads the sync-path
 * counters, which=1 the async-path counters (accumulated across both
 * coli_vk_slot_fill_async itself and any forced wait a full ring triggered).
 * Safe to call with zero recorded calls (all zero out). */
void coli_vk_fill_stats(coli_vk *v, int which, double out[4]);
/* Batch window for the one-time weight upload: between begin() and end() the
 * DEVICE_LOCAL uploads above are staged through one persistent ring and
 * submitted in bulk instead of one submit+fence per matrix half. end() flushes
 * and returns 0 if any batched copy failed to submit. COLI_VK_UPLOAD_BATCH_MB
 * (default 256; 0 disables = old path). See vk_backend.c for the measurement. */
int  coli_vk_upload_begin(coli_vk *v);
int  coli_vk_upload_end(coli_vk *v);
int  coli_vk_gemm4(coli_vk *v, int wh, const coli_a_i8 *a, float *y);

/* q, k and v as ONE submission over one shared activation upload. wh[3] are the
 * weight handles, ys[3] the three outputs. All three weights must share the same
 * input width. Returns 0 on success. */
int  coli_vk_gemm4_qkv(coli_vk *v, const int *wh, const coli_a_i8 *a, float **ys);

/* The dequantize-to-float variant of the SAME kernel, on the SAME uploaded
 * weights. Exists so "integer nibbles beat ggml's float dequant" is a
 * measurement rather than a claim -- see shaders/gemm_i4f.comp. */
int  coli_vk_has_i4f(coli_vk *v);
int  coli_vk_gemm4f(coli_vk *v, int wh, const coli_a_i8 *a, float *y);
/* reps dispatches in one submission; seconds for all of them (see vk_backend.c). */
double coli_vk_bench_gemm4(coli_vk *v, int wh, const coli_a_i8 *a, float *y, int reps);

/* ---- a whole SwiGLU FFN with ONE upload and ONE download ----
 * gate/up/down are int4 handles. Every intermediate -- both projections, the
 * nonlinearity, and the requantized activation the down-projection needs --
 * stays in device memory. The old way was three coli_vk_gemm4 calls: three
 * uploads, three downloads, and a CPU-side requantization in the middle, which
 * is what made residency impossible before shaders/silu_mul_q.comp existed. */
int  coli_vk_has_ffn(coli_vk *v);
int  coli_vk_ffn4(coli_vk *v, int hg, int hu, int hd, const coli_a_i8 *a, float *y);
/* gpt-oss expert FFN (2026-09-14): gate/up/down are coli_vk_upload_w4_mx handles;
 * bg/bu are this expert's gate and up biases (EI floats each, uploaded per call);
 * the middle op is shaders/swiglu_oai_q.spv (ggml's swiglu_oai with alpha/limit).
 * The DOWN bias is NOT applied here -- the caller adds it on the host, same as
 * the CPU expert path. Returns -1 (caller keeps the CPU path) when any handle is
 * not MXFP4-tagged or the pipeline is absent. */
int  coli_vk_has_ffn_oai(coli_vk *v);
int  coli_vk_ffn4_oai(coli_vk *v, int hg, int hu, int hd, const coli_a_i8 *a, float *y,
                      const float *bg, const float *bu, float alpha, float limit);

/* ---- N experts, ONE shared activation, ONE submission (grouped MoE decode) ----
 * hg/hu/hd are nexp-long arrays of resident int4 handles; a is the ONE activation
 * every expert consumes (the decode case, where all K selected experts see the
 * same token); y receives nexp*n*Dout floats, expert-major. Discrete GPU only --
 * on UMA the per-expert ffn4 path already has nothing to amortise. Returns 0, or
 * -1 (caller keeps the CPU/per-expert path). See coli_vk_moe4 in vk_backend.c. */
int  coli_vk_moe4(coli_vk *v, const int *hg, const int *hu, const int *hd,
                  int nexp, const coli_a_i8 *a, float *y);
/* Asynchronous form of coli_vk_moe4: _begin submits and returns at once, _end
 * waits and downloads y (nexp x a->n x Dout). No other Vulkan call may be made
 * on v between them. _begin returns -1 (nothing submitted) if one is pending. */
int  coli_vk_moe4_begin(coli_vk *v, const int *hg, const int *hu, const int *hd,
                        int nexp, const coli_a_i8 *a);
int  coli_vk_moe4_end(coli_vk *v, float *y);

/* Phase breakdown of every GPU call made so far. See the comment on the
 * definition; safe to call with no GPU calls recorded. */
void coli_vk_prof_dump(FILE *f);

/* Mean nanoseconds for one empty submit+fence round trip. Negative on failure. */
/* Empty submit+fence round trip. Returns the MEAN and, via out_min_ns (may be
 * NULL), the MIN over `reps`. A floor is a minimum; the mean-minus-min gap is
 * the contention on the machine at the time, which is a condition the caller
 * must state next to the number. 8 warm-up submits are discarded first. */
double coli_vk_probe_submit_ns(coli_vk *v, int reps, double *out_min_ns);
coli_vk *g_vk_handle(void);

/* ---------------------------------------------------------------- attention
 * Available only when shaders/attn_decode.spv was built. Absence is normal and
 * means the CPU path stays in use -- which is also the numerical reference this
 * kernel is validated against, so it is never removed. */
int coli_vk_has_attn(coli_vk *v);

/* The device's native subgroup width, and whether the attention pipeline had to
 * be pinned to 32 lanes to be correct. attn_decode.comp requires 32; a device
 * whose width is not 32 and cannot be pinned gets no attention pipeline at all,
 * so coli_vk_has_attn returns 0 and callers stay on the CPU. */
int coli_vk_subgroup_size(coli_vk *v);
int coli_vk_attn_pinned32(coli_vk *v);

/* Resident K/V. Allocate once per model load; kv_ctx must mirror the host
 * cache's stride, and coli_vk_kv_init must be called again if the host grows it
 * -- a stale kv_ctx indexes the wrong rows and returns confident nonsense. */
int    coli_vk_kv_init(coli_vk *v, int layers, int slots, int kv_heads, int kv_ctx, int hd);
int    coli_vk_kv_ready(coli_vk *v);
size_t coli_vk_kv_bytes(coli_vk *v);
/* Bulk-load one layer from the host cache. Init and GROWTH only -- growing
 * re-strides every row, so the device copy is rebuilt rather than patched. */
int coli_vk_kv_load(coli_vk *v, int layer, const float *K, const float *V);
/* Read the resident cache back. Needed when the fused block has written rows the
 * host cache does not have and the host cache is about to be re-strided. */
int coli_vk_kv_get(coli_vk *v, int layer, float *K, float *V);
/* The kv_ctx the device buffers were built for. Compare against the host's
 * before every use: a mismatch means the host grew and the device copy now
 * indexes the wrong rows. */
int coli_vk_kv_ctx(coli_vk *v);
/* Stage one row for the NEXT coli_vk_attn call on the same layer. A -1 return
 * means the row was NOT staged; the caller must fall back to the CPU path for
 * this token rather than continue, or every later position reads a hole. */
int coli_vk_kv_put(coli_vk *v, int slot, int kvh, int pos, int is_v, const float *row);

/* Bulk-write a contiguous run of positions for EVERY kv head of one layer, in
 * one submit. This is the PREFILL path: kv_put's 64-row staging ring caps a
 * batch at 32 tokens, and a 683-token prefill needs 10,928 rows per layer.
 * Khost/Vhost are the layer's full host caches, [kv_heads][kv_ctx][hd]. */
int coli_vk_kv_write(coli_vk *v, int layer, int slot, int pos0, int count,
                     const float *Khost, const float *Vhost);
/* Attention against the resident cache. Copies the staged rows and dispatches
 * in ONE command buffer, so writing the cache costs no extra fence. */
/* gpt-oss (2026-09-14): all layers' attention sinks in one buffer, [layers][H]
 * floats, uploaded once. coli_vk_attn_ex then takes this layer's offset (-1 =
 * none) and a sliding window (0 = none); coli_vk_attn is the (0, -1) case. */
int coli_vk_attn_sinks_upload(coli_vk *v, const float *sinks, size_t nfloat);
int coli_vk_attn_ex(coli_vk *v, int layer, const float *q, float *out,
                    const int *meta, int n, int H, float scale, int window, int sink_off);
int coli_vk_attn(coli_vk *v, int layer, const float *q, float *out,
                 const int *meta, int n, int H, float scale);

/* ------------------------------------------------------- split-K attention
 * attn_decode_split.comp + attn_decode_merge.comp (2026-09-13): the same
 * resident-cache attention as coli_vk_attn above, restructured so a workgroup
 * owns (row, kv-head, chunk) instead of (row, head) -- every query head
 * sharing a kv-head reuses one K/V read instead of repeating it, and decode
 * (n=1) gets many small workgroups instead of n*H. See the header of
 * attn_decode_split.comp for the measurement and the numerics/chunking notes.
 * Available only when BOTH pipelines built; absence means coli_vk_attn_block
 * keeps using coli_vk_attn's single-kernel path. */
int coli_vk_has_attn_split(coli_vk *v);
/* Shaped exactly like coli_vk_attn -- same resident cache, same q/meta/out
 * shapes -- so a test can time and correctness-check the two head to head
 * against identical inputs. Not called from the fused block, which records
 * the split path inline; this exists for tests/test_attn_split.c. */
int coli_vk_attn_split(coli_vk *v, int layer, const float *q, float *out,
                       const int *meta, int n, int H, float scale);

/* Correctness harness ONLY. Uploads the entire K/V cache per call, which is the
 * very cost moving attention to the device is meant to eliminate; its timing is
 * not a measurement of anything. See the comment on the definition. */
int coli_vk_attn_ref(coli_vk *v, const float *q, const float *K, const float *V,
                     float *out, const int *meta, int n, int H, int KVH, int hd,
                     int kv_ctx, int slots, float scale);

/* ---- RoPE + bias, and the KV scatter, on device ----------------------------
 *
 * These move a BARRIER, not a cost. RoPE is 0.2% of a decode token (measured
 * 2026-08-21); it matters because it sat on the CPU between two GPU ops and so
 * forced the layer to be three submissions instead of one.
 *
 * The (c,s) table is NOT computed on the GPU and must not be -- coli_sincos
 * exists because libm transcendentals differ by 1 ULP across platforms and that
 * flips a near-tie argmax. Build it on the host with rope_table() and upload it;
 * 512 bytes per row buys a determinism property that is hard to get back.
 *
 * Upload frequencies differ and matter: bias is per LAYER and constant, the
 * (c,s) table is per POSITION and shared by every layer.
 */
int coli_vk_has_rope(coli_vk *v);
/* qwen3 per-head q/k RMSNorm inside the fused block (qknorm.spv). Weights are
 * [layer][q hd | k hd] flat; pass the layer's float offset as qk_off, -1 = none. */
int coli_vk_has_qknorm(coli_vk *v);
int coli_vk_qknorm_upload(coli_vk *v, const float *w, size_t nfloat);
int coli_vk_rope_bias_upload(coli_vk *v, const float *bias, size_t nfloat);
int coli_vk_rope_cs_upload  (coli_vk *v, const float *cs,   size_t nfloat);

/* TEST-ONLY. Both wrap the kernel in an upload/submit/download round trip, which
 * is the exact cost the kernels exist to remove -- production uses the fused
 * block instead. They are the unit under test in tests/test_vk_rope. */
int coli_vk_rope_run(coli_vk *v, float *q, float *k,
                     int n, int H, int KVH, int hd, int neox, int bias_off);
int coli_vk_kvwrite_run(coli_vk *v, int layer, const float *k, const float *vv,
                        const int *slots, const int *poss,
                        int n, int KVH, int hd, int bv_off, int has_bias);


/* ---- the fused attention block ---------------------------------------------
 *
 * qkv -> rope+bias -> kv scatter -> attention -> quantize -> o_proj, as ONE
 * command buffer: one upload, eight dispatches, one download. Replaces three
 * submissions whose downloads and submits are 37.6% of a decode token (measured
 * 2026-08-21). What made it possible was moving RoPE off the CPU -- it sat in
 * the middle of the layer and split it.
 *
 * wh is { wq, wk, wv, wo }. bias_off indexes ONE buffer holding every layer's
 * qkv bias (see coli_vk_rope_bias_upload); negative means no bias. The (c,s)
 * table must already be uploaded for these rows.
 *
 * NOT USABLE FOR qwen3 -- it norms q and k between bias and rotation and no
 * kernel here does that. Check for the qk_norm tensors and stay on the CPU path.
 */
int coli_vk_has_block(coli_vk *v);
int coli_vk_attn_block(coli_vk *v, int layer, const int *wh, const coli_a_i8 *a,
                       const int *meta, int n, int H, int KVH, int hd,
                       int neox, int bias_off, int qk_off, float qk_eps,
                       float scale, int stop_attn, float *y);

#ifdef __cplusplus
}
#endif




#endif
