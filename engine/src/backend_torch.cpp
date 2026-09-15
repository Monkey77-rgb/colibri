/* backend_torch.cpp — libtorch (PyTorch C++ API) backend, the COVERAGE plugin
 * (2026-09-14).
 *
 * WHY THIS BACKEND EXISTS. Banana's hand-written backends (Vulkan, CUDA) reach
 * the hardware they were written for and nothing else. torch reaches every
 * device its own build supports -- ROCm, Apple MPS, Intel XPU, and whatever
 * the next libtorch release adds -- through the SAME C++ call, with no new
 * shader or kernel from us. It is NOT expected to beat the hand-written
 * Vulkan/CUDA kernels on the devices they already cover: this backend
 * dequantizes every weight to bf16 ONCE at upload and then runs plain
 * torch::matmul, paying 2x the bytes of this engine's int8 format and 4x the
 * bytes of its int4/MXFP4 formats for every weight, in exchange for a kernel
 * that is a five-line read instead of a hand-unrolled VNNI/DP4a routine. That
 * tradeoff -- memory and bandwidth for reach and auditability -- is the whole
 * point of a coverage backend and is stated here plainly, not discovered by
 * someone profiling it later.
 *
 * IT IS A PLUGIN, NEVER LINKED INTO THE ENGINE BINARY. Built as
 * libcoli_torch.so and dlopen'ed by src/backend.c's open_torch(), which looks
 * in $COLI_PLUGIN_DIR, next to the running binary, then the default dlopen
 * path, and resolves coli_backend_torch_open() by name. The `coli` / `coli-gpu`
 * binaries never see -ltorch on their link line, so a machine with no libtorch
 * installed (the 780M) builds and runs exactly as before. This file is a
 * single translation unit with NO other engine source compiled alongside it in
 * the .so (see the Makefile rule) -- it talks to the engine only through
 * backend.h's plain-C struct of function pointers.
 *
 * DEQUANTIZE-ONCE, MATMUL-FOREVER. Every upload_* entry converts the engine's
 * native quantized format (gemm_i8.h's coli_w_i8 / coli_w_i4) to a bf16
 * torch::Tensor on the target device exactly once; every gemm/ffn call below
 * reuses that resident tensor. Activations are dequantized fresh every call
 * (coli_a_i8 -> a float32 host buffer -> bf16 on device) because they change
 * every call by definition -- there is nothing to cache.
 *
 * NO INTEGER PATH, ON PURPOSE. The VK/CPU kernels carry an int8/int4 GEMM the
 * whole way, including a second activation-quantization step before the
 * down-projection inside the fused OAI FFN (ffn4_oai; see vk_backend.h and
 * shaders/silu_mul_q.comp -DSWIGLU_OAI). This backend has no integer kernel at
 * all -- once a weight is a bf16 tensor, every downstream op (gemm, gemm4,
 * ffn4, ffn4_oai) is torch::matmul/clamp/sigmoid end to end, and the hidden
 * vector between gate/up and down is NEVER requantized to int8. That is MORE
 * precise than the reference path, not a shortcut: tests/test_torch_backend.cpp
 * measures the resulting difference against the VK-style CPU reference (which
 * DOES quantize the hidden vector) and states the bound, exactly the way
 * tests/test_vk_oai.c already calls its own GPU-vs-CPU gap "a tolerance
 * comparison" rather than bit-exactness.
 *
 * EXCEPTION SAFETY. Every entry below is the one boundary between this engine's
 * C call sites and libtorch's C++ exceptions (c10::Error on a bad shape, OOM,
 * or a CUDA runtime error). None of that may cross into model.cpp, which is
 * not exception-aware. Every function here is a plain try/catch around the
 * torch work, printing one line to stderr and returning the X-macro's DECLINE
 * value (-1 for data calls, already the engine's "fall back to CPU" signal).
 *
 * HANDLE TABLES. Two handle spaces, matching the seam's own documented split
 * (backend.h / vk_backend.h): g_w8 for coli_vk_upload_w-style int8 handles,
 * and g_w4 for EVERYTHING that starts from a coli_w_i4 -- plain int4
 * (upload_w4), MXFP4-tagged (upload_w4_mx), and pre-allocated expert slots
 * (slot_alloc_mx + slot_fill). This mirrors the Vulkan backend's own design
 * ("the handle lives in the int4 table", vk_backend.h on upload_w4_mx): once
 * dequantized, a plain-int4 weight and an MXFP4 weight are both just an [O,I]
 * bf16 tensor, so there is no reason for this backend to track which path
 * produced it -- gemm4/ffn4/ffn4_oai treat every g_w4 entry identically.
 *
 * NOT PINNED. slot_fill's host-side dequant buffer is an ordinary (unpinned)
 * torch::Tensor; the copy into the resident device slot goes through the
 * normal non-pinned H2D path. Pinning would only pay for itself if slot
 * refills were copy-bound at the scale this engine's expert cache moves
 * through -- not measured here, so it is not claimed.
 */
#include "backend.h"

#include <torch/torch.h>
#if defined(COLI_TORCH_HAVE_CUDA_PROPS)
#include <ATen/cuda/CUDAContext.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

/* ---------------------------------------------------------------- context */

struct TorchCtx {
    torch::Device device{torch::kCPU};
    std::string    device_id;     /* "cuda:0" | "mps" | "xpu" | "cpu" */
    std::string    name;          /* full device_name() string */
    std::string    memdesc1;
    std::string    memdesc2;
    int            integrated = 0;

    /* g_w8[h]: [O,I] bf16, dequantized from coli_w_i8.
     * g_w4[h]: [O,I] bf16, dequantized from coli_w_i4 (plain int4, MXFP4, or a
     * slot-allocated/filled handle -- see file header). */
    std::vector<torch::Tensor> w8;
    std::vector<torch::Tensor> w4;

    /* Handle tables are not documented as thread-safe anywhere in the seam
     * (backend.h / vk_backend.h); model.cpp calls them from one thread per
     * coli_backend instance. This mutex only protects the vector `push_back`
     * (handle allocation) against a future caller that uploads from more than
     * one thread -- it does NOT make a gemm call re-entrant with an upload. */
    std::mutex handles_mu;
};

/* Print exactly one line per failure -- never the full torch backtrace twice,
 * never silent. Every public entry point below is wrapped so a c10::Error (bad
 * shape, OOM, a CUDA driver error) becomes a single stderr line and a -1/0
 * return, never an exception across the C boundary into model.cpp. */
void report(const char *fn, const char *what) {
    std::fprintf(stderr, "[coli_torch] %s: %s\n", fn, what);
}

#define TC_TRY(fn) try {
#define TC_CATCH(fn, declv) \
    } catch (const c10::Error &e) { report(fn, e.what()); return (declv); } \
      catch (const std::exception &e) { report(fn, e.what()); return (declv); } \
      catch (...) { report(fn, "unknown C++ exception"); return (declv); }

/* ------------------------------------------------------------- dequantize */

/* coli_w_i8 -> [O,I] float32 on CPU. Prefers w->f (full f32 weights) when the
 * caller supplied it -- same "debug the architecture before the quantizer"
 * escape hatch gemm_i8.h documents for the CPU path. Otherwise reconstructs
 * the offset-to-unsigned int8 weights exactly as coli_gemm_i8_ref does:
 * value = (qu - 128) * scale[row], ONE scale per output row. */
torch::Tensor dequant_i8_cpu(const coli_w_i8 *w) {
    const int64_t O = w->O, I = w->I;
    if (w->f) {
        return torch::from_blob((void *)w->f, {O, I}, torch::kFloat32).clone();
    }
    auto qu = torch::from_blob((void *)w->qu, {O, I}, torch::kUInt8).to(torch::kInt32);
    auto centered = qu - 128;
    auto scale = torch::from_blob((void *)w->scale, {O, 1}, torch::kFloat32).clone();
    return centered.to(torch::kFloat32) * scale;
}

/* coli_w_i4 nibble unpack, shared by the plain-int4 and MXFP4 decode below.
 * Byte layout (gemm_i8.cpp's quantize_w4_core / coli_mxfp4_repack_i4_ref,
 * BOTH use this convention): element k of a 32-wide block lives in byte k/2,
 * low nibble if k is even, high nibble if k is odd. Returns [O,I] int32 in
 * [0,15], raw nibble value -- caller decides plain-int4 (nib-8) or MXFP4
 * (lut[nib]) semantics. */
torch::Tensor unpack_nibbles_i32(const uint8_t *q4, int64_t O, int64_t I) {
    auto bytes = torch::from_blob((void *)q4, {O, I / 2}, torch::kUInt8).clone().to(torch::kInt32);
    auto lo = torch::bitwise_and(bytes, 0x0F);
    auto hi = torch::bitwise_and(torch::bitwise_right_shift(bytes, 4), 0x0F);
    /* stack on a new last dim then flatten: byte j -> elements 2j (lo), 2j+1
     * (hi), which is exactly the k/2,k&1 convention above. */
    return torch::stack({lo, hi}, /*dim=*/2).reshape({O, I});
}

/* Per-block scale, broadcast back out to [O,I] (block width 32 for BOTH
 * COLI_W4BLK and COLI_MXFP4_BLK -- they are the same constant, checked at
 * upload time below, not assumed). */
torch::Tensor expand_bscale(const float *bscale, int64_t O, int64_t nb, int64_t I) {
    auto s = torch::from_blob((void *)bscale, {O, nb}, torch::kFloat32).clone();
    return s.repeat_interleave(I / nb, /*dim=*/1);
}

/* Plain int4: value = (nibble - 8) * bscale[block]. Bit-for-bit the same
 * reconstruction gemm_i8.cpp's gemm_i4_narrow performs per element, just done
 * as one vectorized float op over the whole [O,I] matrix instead of per-row. */
torch::Tensor dequant_i4_plain_cpu(const coli_w_i4 *w) {
    const int64_t O = w->O, I = w->I, nb = I / 32 /* COLI_W4BLK */;
    auto nib = unpack_nibbles_i32(w->q4, O, I);
    auto scale = expand_bscale(w->bscale, O, nb, I);
    return (nib - 8).to(torch::kFloat32) * scale;
}

/* e2m1 LUT, DOUBLED -- transcribed from gemm_mxfp4.cpp's coli_mxfp4_lut, same
 * provenance note: ggml upstream's MXFP4 table. MXFP4 value = lut[nibble] *
 * bscale[block], where bscale is ALREADY the E8M0 scale halved (see
 * coli_mxfp4_e8m0_half / the "E8M0 scale already halved" note on
 * coli_vk_upload_w4_mx in vk_backend.h) -- the doubled table and the halved
 * scale cancel, giving the true weight magnitude with no extra factor. */
torch::Tensor dequant_i4_mx_cpu(const coli_w_i4 *w) {
    const int64_t O = w->O, I = w->I, nb = I / 32 /* COLI_MXFP4_BLK */;
    static const int32_t lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    auto lut_t = torch::from_blob((void *)lut, {16}, torch::kInt32).clone();
    auto nib = unpack_nibbles_i32(w->q4, O, I);
    auto decoded = lut_t.index_select(0, nib.reshape(-1).to(torch::kInt64)).reshape({O, I});
    auto scale = expand_bscale(w->bscale, O, nb, I);
    return decoded.to(torch::kFloat32) * scale;
}

/* coli_a_i8 -> [n,I] float32 on CPU. value = q * scale[block], one scale per
 * COLI_ABLK=16 block -- the exact reconstruction coli_gemm_i8_ref's dot_blk_ref
 * performs (there a->sum also feeds the unsigned-weight correction term; that
 * trick is an INTEGER-kernel optimization this backend has no use for, since
 * it never quantizes the weight side either). */
torch::Tensor dequant_a_cpu(const coli_a_i8 *a) {
    const int64_t n = a->n, I = a->I, nb = I / 16 /* COLI_ABLK */;
    auto q = torch::from_blob((void *)a->q, {n, I}, torch::kInt8).to(torch::kFloat32);
    auto scale = torch::from_blob((void *)a->scale, {n, nb}, torch::kFloat32).clone();
    auto scale_full = scale.repeat_interleave(I / nb, /*dim=*/1);
    return q * scale_full;
}

torch::Tensor to_device_bf16(TorchCtx *c, const torch::Tensor &cpu_f32) {
    return cpu_f32.to(torch::TensorOptions().device(c->device).dtype(torch::kBFloat16));
}

/* Copy a [n,O] float32 CPU-resident result tensor into the caller's flat
 * buffer. Always called after .to(kFloat32).cpu().contiguous(), so this is a
 * plain memcpy, not a second conversion. */
void copy_out(const torch::Tensor &t, float *y) {
    std::memcpy(y, t.data_ptr<float>(), (size_t)t.numel() * sizeof(float));
}

/* ------------------------------------------------------------- device pick
 *
 * COLI_TORCH_DEVICE, if set, is tried first and must actually allocate (a
 * string like "cuda:0" on a machine with no CUDA build is a normal miss, not
 * an error) -- falls through to auto order on failure, with one stderr line
 * saying so. Auto order is cuda, mps, xpu, cpu: the first that can allocate a
 * one-element tensor. mps/xpu have no availability query as stable as
 * torch::cuda::is_available() in this libtorch build, so they are PROBED by
 * trying the allocation directly inside a try/catch -- the probe itself is
 * therefore also the evidence for "this device is usable", not a separate
 * claim. */
torch::Device pick_device(std::string &chosen) {
    auto probe = [](torch::DeviceType t) -> bool {
        try {
            torch::Device d{t};
            auto x = torch::zeros({1}, torch::TensorOptions().device(d));
            return x.numel() == 1;
        } catch (...) { return false; }
    };

    const char *env = std::getenv("COLI_TORCH_DEVICE");
    if (env && *env) {
        try {
            std::string envs(env);
            torch::Device d{envs};
            auto x = torch::zeros({1}, torch::TensorOptions().device(d));
            if (x.numel() == 1) { chosen = env; return d; }
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                "[coli_torch] COLI_TORCH_DEVICE=%s unavailable (%s); falling back to auto order\n",
                env, e.what());
        }
    }
    if (torch::cuda::is_available()) { chosen = "cuda:0"; return torch::Device(torch::kCUDA, 0); }
    if (probe(torch::kMPS)) { chosen = "mps"; return torch::Device(torch::kMPS); }
    if (probe(torch::kXPU)) { chosen = "xpu"; return torch::Device(torch::kXPU); }
    chosen = "cpu";
    return torch::Device(torch::kCPU);
}

/* ----------------------------------------------------------------- entries
 *
 * Signatures copied verbatim from backend.h's COLI_BE_FUNCS (ctx first, same
 * as every other backend). Only the entries the header comment and the task
 * enumerate are implemented; everything else (moe4*, kv_*, attn_*, rope_*,
 * qknorm_*, attn_block) is left NULL below and gets backend.c's declining
 * stub, same as a Vulkan build missing an optional shader. */

const char *tc_device_name(void *ctx) { return ((TorchCtx *)ctx)->name.c_str(); }
const char *tc_memdesc(void *ctx)     { return ((TorchCtx *)ctx)->memdesc1.c_str(); }
const char *tc_memdesc2(void *ctx)    { return ((TorchCtx *)ctx)->memdesc2.c_str(); }
int tc_is_integrated(void *ctx)       { return ((TorchCtx *)ctx)->integrated; }
int tc_dot_used(void *ctx)            { (void)ctx; return 0; /* no hand-written dot kernel here */ }

int tc_upload_w(void *ctx, const coli_w_i8 *w) {
    TC_TRY("upload_w")
        TorchCtx *c = (TorchCtx *)ctx;
        c10::InferenceMode guard(true);
        torch::Tensor f = dequant_i8_cpu(w);
        torch::Tensor dev = to_device_bf16(c, f);
        std::lock_guard<std::mutex> lk(c->handles_mu);
        c->w8.push_back(dev);
        return (int)c->w8.size() - 1;
    TC_CATCH("upload_w", -1)
}

int tc_gemm(void *ctx, int wh, const coli_a_i8 *a, float *y) {
    TC_TRY("gemm")
        TorchCtx *c = (TorchCtx *)ctx;
        if (wh < 0 || (size_t)wh >= c->w8.size()) { report("gemm", "bad handle"); return -1; }
        c10::InferenceMode guard(true);
        torch::Tensor act = to_device_bf16(c, dequant_a_cpu(a));
        torch::Tensor W = c->w8[wh];
        torch::Tensor yb = torch::matmul(act, W.transpose(0, 1));
        torch::Tensor yf = yb.to(torch::kFloat32).to(torch::kCPU).contiguous();
        copy_out(yf, y);
        return 0;
    TC_CATCH("gemm", -1)
}

int tc_has_i4(void *ctx) { (void)ctx; return 1; }

int tc_upload_w4(void *ctx, const coli_w_i4 *w) {
    TC_TRY("upload_w4")
        TorchCtx *c = (TorchCtx *)ctx;
        c10::InferenceMode guard(true);
        torch::Tensor f = dequant_i4_plain_cpu(w);
        torch::Tensor dev = to_device_bf16(c, f);
        std::lock_guard<std::mutex> lk(c->handles_mu);
        c->w4.push_back(dev);
        return (int)c->w4.size() - 1;
    TC_CATCH("upload_w4", -1)
}

int tc_has_mx(void *ctx) { (void)ctx; return 1; }

int tc_upload_w4_mx(void *ctx, const coli_w_i4 *w) {
    TC_TRY("upload_w4_mx")
        TorchCtx *c = (TorchCtx *)ctx;
        c10::InferenceMode guard(true);
        torch::Tensor f = dequant_i4_mx_cpu(w);
        torch::Tensor dev = to_device_bf16(c, f);
        std::lock_guard<std::mutex> lk(c->handles_mu);
        c->w4.push_back(dev);
        return (int)c->w4.size() - 1;
    TC_CATCH("upload_w4_mx", -1)
}

int tc_slot_alloc_mx(void *ctx, int64_t I, int64_t O) {
    TC_TRY("slot_alloc_mx")
        TorchCtx *c = (TorchCtx *)ctx;
        c10::InferenceMode guard(true);
        torch::Tensor dev = torch::zeros({O, I}, torch::TensorOptions().device(c->device).dtype(torch::kBFloat16));
        std::lock_guard<std::mutex> lk(c->handles_mu);
        c->w4.push_back(dev);
        return (int)c->w4.size() - 1;
    TC_CATCH("slot_alloc_mx", -1)
}

int tc_slot_fill(void *ctx, int h, const coli_w_i4 *w) {
    TC_TRY("slot_fill")
        TorchCtx *c = (TorchCtx *)ctx;
        if (h < 0 || (size_t)h >= c->w4.size()) { report("slot_fill", "bad handle"); return -1; }
        if (c->w4[h].size(0) != w->O || c->w4[h].size(1) != w->I) {
            report("slot_fill", "shape mismatch between slot and fill weight");
            return -1;
        }
        c10::InferenceMode guard(true);
        /* slots are always MXFP4-tagged per the seam's contract (see
         * vk_backend.h on coli_vk_slot_alloc_mx / coli_vk_slot_fill). */
        torch::Tensor f = dequant_i4_mx_cpu(w);
        torch::Tensor dev = to_device_bf16(c, f);
        /* .copy_() under InferenceMode, NOT pinned host memory -- see file
         * header "NOT PINNED". */
        c->w4[h].copy_(dev);
        return 0;
    TC_CATCH("slot_fill", -1)
}

int tc_upload_begin(void *ctx) { (void)ctx; return 0; /* no batching window in this backend */ }
int tc_upload_end(void *ctx)   { (void)ctx; return 1; /* nothing batched to flush */ }

int tc_gemm4(void *ctx, int wh, const coli_a_i8 *a, float *y) {
    TC_TRY("gemm4")
        TorchCtx *c = (TorchCtx *)ctx;
        if (wh < 0 || (size_t)wh >= c->w4.size()) { report("gemm4", "bad handle"); return -1; }
        c10::InferenceMode guard(true);
        torch::Tensor act = to_device_bf16(c, dequant_a_cpu(a));
        torch::Tensor W = c->w4[wh];
        torch::Tensor yb = torch::matmul(act, W.transpose(0, 1));
        torch::Tensor yf = yb.to(torch::kFloat32).to(torch::kCPU).contiguous();
        copy_out(yf, y);
        return 0;
    TC_CATCH("gemm4", -1)
}

int tc_gemm4_qkv(void *ctx, const int *wh, const coli_a_i8 *a, float **ys) {
    TC_TRY("gemm4_qkv")
        TorchCtx *c = (TorchCtx *)ctx;
        for (int i = 0; i < 3; i++)
            if (wh[i] < 0 || (size_t)wh[i] >= c->w4.size()) { report("gemm4_qkv", "bad handle"); return -1; }
        c10::InferenceMode guard(true);
        /* ONE activation dequant/upload shared across q, k, v -- the same
         * amortization coli_vk_gemm4_qkv documents, done here at the torch
         * level instead of inside one Vulkan submission. */
        torch::Tensor act = to_device_bf16(c, dequant_a_cpu(a));
        for (int i = 0; i < 3; i++) {
            torch::Tensor yb = torch::matmul(act, c->w4[wh[i]].transpose(0, 1));
            torch::Tensor yf = yb.to(torch::kFloat32).to(torch::kCPU).contiguous();
            copy_out(yf, ys[i]);
        }
        return 0;
    TC_CATCH("gemm4_qkv", -1)
}

int tc_has_ffn(void *ctx) { (void)ctx; return 1; }

int tc_ffn4(void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y) {
    TC_TRY("ffn4")
        TorchCtx *c = (TorchCtx *)ctx;
        if (hg < 0 || (size_t)hg >= c->w4.size() || hu < 0 || (size_t)hu >= c->w4.size() ||
            hd < 0 || (size_t)hd >= c->w4.size()) { report("ffn4", "bad handle"); return -1; }
        c10::InferenceMode guard(true);
        torch::Tensor act = to_device_bf16(c, dequant_a_cpu(a));
        torch::Tensor g = torch::matmul(act, c->w4[hg].transpose(0, 1));
        torch::Tensor u = torch::matmul(act, c->w4[hu].transpose(0, 1));
        /* qwen-style plain SwiGLU, model.cpp's expert_act non-gptoss arm:
         * h = silu(gate) * up, no bias, no clamp. torch::silu is the same
         * x*sigmoid(x) expression, computed in bf16 on-device. */
        torch::Tensor h = torch::silu(g) * u;
        torch::Tensor yb = torch::matmul(h, c->w4[hd].transpose(0, 1));
        torch::Tensor yf = yb.to(torch::kFloat32).to(torch::kCPU).contiguous();
        copy_out(yf, y);
        return 0;
    TC_CATCH("ffn4", -1)
}

int tc_has_ffn_oai(void *ctx) { (void)ctx; return 1; }

int tc_ffn4_oai(void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y,
                const float *bg, const float *bu, float alpha, float limit) {
    TC_TRY("ffn4_oai")
        TorchCtx *c = (TorchCtx *)ctx;
        if (hg < 0 || (size_t)hg >= c->w4.size() || hu < 0 || (size_t)hu >= c->w4.size() ||
            hd < 0 || (size_t)hd >= c->w4.size()) { report("ffn4_oai", "bad handle"); return -1; }
        c10::InferenceMode guard(true);
        const int64_t EI = c->w4[hg].size(0);
        torch::Tensor act = to_device_bf16(c, dequant_a_cpu(a));
        torch::Tensor g = torch::matmul(act, c->w4[hg].transpose(0, 1)).to(torch::kFloat32);
        torch::Tensor u = torch::matmul(act, c->w4[hu].transpose(0, 1)).to(torch::kFloat32);
        torch::Tensor bg_t = to_device_bf16(c, torch::from_blob((void *)bg, {EI}, torch::kFloat32).clone())
                                 .to(torch::kFloat32);
        torch::Tensor bu_t = to_device_bf16(c, torch::from_blob((void *)bu, {EI}, torch::kFloat32).clone())
                                 .to(torch::kFloat32);
        /* ggml's swiglu_oai, model.cpp's expert_act gptoss arm, VERBATIM:
         *   x = gate+bg; x = min(x, limit)              (upper clamp ONLY)
         *   yv = up+bu;  yv = clamp(yv, -limit, limit)   (both sides)
         *   glu = x * sigmoid(alpha * x)                 (== x/(1+exp(-alpha*x)))
         *   h = glu * (yv + 1)
         * The DOWN bias is deliberately NOT added here -- the caller adds it
         * on the host, same as the CPU expert path (expert_down_bias in
         * model.cpp, and vk_backend.h's note on coli_vk_ffn4_oai). */
        torch::Tensor xg = torch::clamp_max(g + bg_t, limit);
        torch::Tensor yv = torch::clamp(u + bu_t, -limit, limit);
        torch::Tensor glu = xg * torch::sigmoid(alpha * xg);
        torch::Tensor h = glu * (yv + 1.0);
        torch::Tensor h_bf16 = h.to(torch::kBFloat16);
        torch::Tensor yb = torch::matmul(h_bf16, c->w4[hd].transpose(0, 1));
        torch::Tensor yf = yb.to(torch::kFloat32).to(torch::kCPU).contiguous();
        copy_out(yf, y);
        return 0;
    TC_CATCH("ffn4_oai", -1)
}

void tc_close(void *ctx) {
    delete (TorchCtx *)ctx;
}

} // namespace

extern "C" coli_backend *coli_backend_torch_open(char *err, size_t errcap) {
    try {
        std::unique_ptr<TorchCtx> c(new TorchCtx());
        c->device = pick_device(c->device_id);
        c->integrated = (c->device.is_cpu() || c->device.type() == torch::kMPS) ? 1 : 0;

        std::string dn = "torch:" + c->device_id;
        if (c->device.is_cuda()) {
#if defined(COLI_TORCH_HAVE_CUDA_PROPS)
            auto *props = at::cuda::getDeviceProperties(c->device.index());
            dn += std::string(" ") + props->name;
#else
            dn += " (CUDA device name query not compiled in this plugin build)";
#endif
        } else if (c->device.type() == torch::kMPS) {
            dn += " (Apple MPS)";
        } else if (c->device.type() == torch::kXPU) {
            dn += " (Intel XPU)";
        } else {
            dn += " (host CPU)";
        }
        c->name = dn;

        c->memdesc1 = std::string("torch bf16 resident weights on ") + c->device_id;
        c->memdesc2 = c->integrated
            ? "unified/host memory (integrated or CPU device)"
            : "device memory (discrete GPU)";

        coli_backend *be = (coli_backend *)calloc(1, sizeof *be);
        if (!be) { snprintf(err, errcap, "torch: out of memory"); return NULL; }
        be->ctx = c.release();
        be->close = tc_close;

        be->device_name   = tc_device_name;
        be->memdesc        = tc_memdesc;
        be->memdesc2       = tc_memdesc2;
        be->is_integrated  = tc_is_integrated;
        be->dot_used       = tc_dot_used;
        be->upload_w       = tc_upload_w;
        be->gemm           = tc_gemm;
        be->has_i4         = tc_has_i4;
        be->upload_w4      = tc_upload_w4;
        be->has_mx         = tc_has_mx;
        be->upload_w4_mx   = tc_upload_w4_mx;
        be->slot_alloc_mx  = tc_slot_alloc_mx;
        be->slot_fill      = tc_slot_fill;
        be->upload_begin   = tc_upload_begin;
        be->upload_end     = tc_upload_end;
        be->gemm4          = tc_gemm4;
        be->gemm4_qkv      = tc_gemm4_qkv;
        be->has_ffn        = tc_has_ffn;
        be->ffn4           = tc_ffn4;
        be->has_ffn_oai    = tc_has_ffn_oai;
        be->ffn4_oai       = tc_ffn4_oai;
        /* Everything else (moe4*, has_attn + kv_*, attn_sinks_upload, attn_ex,
         * has_qknorm + qknorm_upload, rope_bias_upload, rope_cs_upload,
         * has_block + attn_block) is left NULL -- backend.c's fill_defaults()
         * replaces each with the declining stub, so model.cpp sees has_*() ->
         * 0 and keeps the CPU path, same as a Vulkan build missing a shader. */
        return be;
    } catch (const std::exception &e) {
        snprintf(err, errcap, "torch: init failed: %s", e.what());
        return NULL;
    } catch (...) {
        snprintf(err, errcap, "torch: init failed: unknown C++ exception");
        return NULL;
    }
}
