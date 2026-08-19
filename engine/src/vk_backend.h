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

/* ---- a whole SwiGLU FFN with ONE upload and ONE download ----
 * gate/up/down are int4 handles. Every intermediate -- both projections, the
 * nonlinearity, and the requantized activation the down-projection needs --
 * stays in device memory. The old way was three coli_vk_gemm4 calls: three
 * uploads, three downloads, and a CPU-side requantization in the middle, which
 * is what made residency impossible before shaders/silu_mul_q.comp existed. */
int  coli_vk_has_ffn(coli_vk *v);
int  coli_vk_ffn4(coli_vk *v, int hg, int hu, int hd, const coli_a_i8 *a, float *y);

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

#ifdef __cplusplus
}
#endif

#endif
