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
 * Those stay on the CPU and the tensors move per call, which is the WRONG shape
 * for a real engine and is stated here so nobody mistakes this for a finished
 * offload path. What it establishes is the device plumbing and a verified kernel;
 * keeping activations resident across layers is the next step, not this one.
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
#include "gemm_i8.h"

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

/* Upload a weight matrix once; returns an opaque handle index, or -1. Weights
 * are uploaded in the SAME offset-to-unsigned layout the CPU uses, so no
 * repacking happens on either side. */
int  coli_vk_upload_w(coli_vk *v, const coli_w_i8 *w);

/* y[n][O] = a . W^T on the GPU. Returns 0 on success. */
int  coli_vk_gemm(coli_vk *v, int wh, const coli_a_i8 *a, float *y);

#endif
