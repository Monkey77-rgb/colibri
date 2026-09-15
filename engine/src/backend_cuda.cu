/* backend_cuda.cu — CUDA backend behind the coli_backend seam (2026-09-14).
 *
 * WHY A SEPARATE DESIGN FROM vk_backend.c. Vulkan must stay portable across an
 * iGPU-or-discrete, any-vendor future, so its GEMM kernels decode through a
 * 64-thread-workgroup shared-memory TREE (gemm_i8.comp) and are explicitly NOT
 * bit-exact with the CPU. This backend targets exactly one piece of hardware
 * (RTX 4070, sm_89) and the single-row-per-thread dp4a kernels below exist
 * because a stronger claim is available and cheap: every GEMM/GEMM4 kernel
 * here reduces each (weight-scale-block / activation-scale-block) pair as a
 * sequence of __dp4a 4-lane dot products inside ONE thread, in the SAME block
 * order the CPU reference (gemm_i8.cpp / gemm_i4_narrow) uses, and int32
 * addition is exact and associative below the block sizes in play (COLI_ABLK
 * =16, COLI_W4BLK=32 -- max |acc| ~ 127*127*16 ~ 2.6e5, nowhere near 2^31).
 * The only source of float nondeterminism is therefore the SAME one the CPU
 * has (compiler fp-contract), which the Makefile disables engine-wide, so the
 * test below checks EXACT float equality against the CPU int8/int4 references
 * rather than the bounded-relative-error Vulkan has to use. See
 * tests/test_cuda_gemm.c for the measurement. MXFP4 is checked against a
 * bounded tolerance instead (same TOL as test_vk_oai.c) because its CPU
 * reference (coli_gemm_mxfp4_ref) runs over the ORIGINAL 17-byte blocks, not
 * the repacked coli_w_i4 buffer this kernel reads -- a different regrouping
 * of the same 17-byte-block arithmetic, same as Vulkan's MXFP4 path.
 *
 * WEIGHT DECODE, NO CORRECTION TERM. The CPU's wide kernels use VPDPBUSD/VNNI,
 * which wants an UNSIGNED first operand, so they store weights offset to
 * unsigned (u = q+128 for int8, u = q+8 for int4) and add back
 * 128*sum(activation) / 8*sum(activation) after the dot (see gemm_i8.h and
 * gemm_i8.cpp's i4_row_vnni). CUDA's __dp4a takes two SIGNED int32 (four
 * packed signed int8 lanes each) -- there is no mixed u8xs8 builtin exposed at
 * this level. Rather than reach for inline PTX dp4a.u32.s32, the storage byte
 * is XORed back to its signed value first: for a 1-byte offset, u^0x80 is
 * EXACTLY u-128 reinterpreted as a signed int8 (two's complement identity),
 * no rounding, no correction term needed afterward. The int4 nibble uses the
 * same identity after deinterleaving (see dec_i4 below) -- the 4-bit version
 * needs a one-bit sign smear across the nibble because a 4-bit two's
 * complement value does not sit in the high bit of the *byte* it is stored
 * in; gemm_i4_dp.comp does the identical trick for the same reason. Both
 * decodes are EXACT integer recoveries of the signed weight, not approximations,
 * so dp4a(decoded_weight, activation) computes the identical int32 the CPU's
 * dpbusd-plus-correction does, just by a different (and here, simpler) route.
 *
 * UPLOAD SEMANTICS. upload_w / upload_w4 / upload_w4_mx / slot_alloc_mx copy
 * synchronously on the COMPUTE stream via a throwaway pinned staging buffer --
 * correct, not tuned; they run once at load time. slot_fill is different on
 * purpose (2026-09-14 instruction): it stages through a PERSISTENT pinned
 * buffer and issues the copy on a DEDICATED copy stream, then
 * cudaStreamSynchronize's that stream before returning -- synchronous, but
 * using the same stream and staging path slot_fill_async needs.
 *
 * slot_fill_async / slot_fill_wait (2026-09-15) are the async pair the
 * comment above was built ahead of: an N=8-deep ring of PERSISTENT pinned
 * staging buffers, each with its own cudaEvent, on c->copy_stream (created
 * cudaStreamNonBlocking so it is never implicitly serialized against the
 * legacy default stream -- moot today since nothing here uses stream 0, but
 * cheap insurance). slot_fill_async repacks into the next free ring slot
 * (waiting on THAT slot's event first if the ring is full -- the specified
 * backpressure), issues cudaMemcpyAsync H2D on copy_stream, records the
 * event, and returns without waiting. slot_fill_wait waits every outstanding
 * event (cudaEventSynchronize, host-blocking -- this IS the wait, and is
 * timed the same way vk_backend.c's fence wait is) and additionally calls
 * cudaStreamWaitEvent(c->stream, ...) so the compute stream itself carries a
 * dependency on the copy, not only the host's synchronous knowledge of it.
 * COLI_CUDA_NO_ASYNC_FILL=1 disables the ring and routes slot_fill_async
 * through the synchronous slot_fill instead; coli_cuda_fill_stats (declared
 * for tests only, backend.h has no fill_stats entry) reports which path ran
 * and its memcpy/issue/wait breakdown, mirroring coli_vk_fill_stats.
 *
 * RESIDENCY. Every upload_* lands in cudaMalloc'd (DEVICE_LOCAL) memory, never
 * re-copied. Per-call activations and outputs round-trip through pinned
 * (cudaHostAlloc) host buffers grown to the high-water mark seen so far --
 * the same lesson vk_backend.h documents at the top of its file: allocating
 * and freeing these every call put a ~1 ms floor under every dispatch on
 * Vulkan, and the fix generalizes verbatim to CUDA's pinned-memory path.
 *
 * FUSED PATHS (ffn4 / ffn4_oai / gemm4_qkv) never leave the device between
 * stages: the activation is uploaded once, every GEMM and the SwiGLU/quantize
 * kernel between them read and write cudaMalloc'd scratch, and only the final
 * result is downloaded. All of it -- uploads, every kernel launch, the
 * intermediate stages, the final download -- is enqueued on ONE compute
 * stream, so correctness comes from stream ordering and no entry point must
 * call cudaDeviceSynchronize except at its own return.
 *
 * ERROR HANDLING. No entry point aborts the process. A CUDA error is printed
 * to stderr once (via CUDA_CHECK below) and the call returns -1; callers keep
 * the CPU path, exactly as they do for an absent Vulkan pipeline.
 *
 * NOT IMPLEMENTED (left NULL, declines): moe4 / moe4_begin / moe4_end (grouped
 * multi-expert dispatch -- would need its own kernel and was out of scope for
 * this pass) and every rope_, qknorm_ and attn_block entry.
 *
 * KV CACHE + DECODE ATTENTION (2026-09-15) is implemented -- has_attn, kv_init,
 * kv_ready, kv_bytes, kv_load, kv_get, kv_ctx, kv_put, kv_write,
 * attn_sinks_upload, attn_ex -- in the section marked "KV cache + decode
 * attention" below, near the end of the file. See that section's own header
 * comment for the kernel design and how it differs from attn_decode.comp.
 */
#include "backend.h"
#include "gemm_i8.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

/* Only the two entry points backend.h declares (coli_backend_cuda_open,
 * coli_cuda_probe_class) need extern "C" linkage -- they are wrapped
 * individually at their definitions below. Everything above them (the
 * templated GEMV kernels in particular) must NOT sit inside an extern "C"
 * block: a template cannot have C linkage, and nvcc rejects it outright. */

/* ---------------------------------------------------------------- plumbing */

#define CUDA_MAX_W 16384

/* SLOT-FILL PHASE BREAKDOWN (2026-09-15) -- mirrors vk_backend.c's FSYNC/FASYNC
 * exactly (same field names, same meaning), so a caller comparing the two
 * backends' fill costs reads the same three-way split: memcpy into the
 * pinned staging buffer, command issue (cudaMemcpyAsync + cudaEventRecord),
 * and the wait (cudaEventSynchronize / cudaStreamSynchronize). One struct per
 * path because sync and async go through different code and a caller
 * comparing them needs the two kept apart, not summed. Process-wide, like
 * vk_backend.c's -- there is one CUDA backend instance per process in every
 * caller this engine has. */
static struct {
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;   /* n = number of e_slot_fill calls */
} CU_FSYNC;
static struct {
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;   /* n = number of e_slot_fill_async calls (incl. sync fallback) */
} CU_FASYNC;

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

static void cuda_warn_once(const char *what, cudaError_t e) {
    /* Printed, not fatal -- OWNER-07/kernel doctrine: a failing CUDA call is
     * evidence for the caller to fall back, not a reason to kill the process
     * that might also be running other backends. */
    fprintf(stderr, "coli-cuda: %s: %s\n", what, cudaGetErrorString(e));
}
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { cuda_warn_once(#call, _e); return -1; } \
} while (0)
#define CUDA_CHECK_NULL(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { cuda_warn_once(#call, _e); return NULL; } \
} while (0)

typedef struct { uint8_t *d_qu; float *d_scale; int64_t I, O; int used; } cu_w8;
typedef struct { uint8_t *d_q4; float *d_bscale; int64_t I, O; int used; int mx; } cu_w4;

/* ASYNC SLOT FILL ring entry (2026-09-15, mirrors vk_backend.c's fill_slot).
 * One pinned (cudaHostAlloc) staging buffer big enough for ONE slot's
 * weights+scale, plus its own cudaEvent. Fixed array of COLI_CUDA_FILL_INFLIGHT
 * of these, NOT a byte-addressed ring: reusing slot i means waiting on event i
 * first, which is exactly the backpressure the task specified ("when full,
 * wait") and needs no offset bookkeeping -- same design vk_backend.c's
 * COLI_VK_FILL_INFLIGHT ring uses, ported field-for-field. */
#define COLI_CUDA_FILL_INFLIGHT 8
typedef struct {
    void *h_buf; size_t cap;
    cudaEvent_t event;
    int pending;   /* 1 = cudaMemcpyAsync submitted on copy_stream, event not yet observed signaled */
} cu_fill_slot;

typedef struct coli_cuda {
    int dev;
    cudaStream_t stream;       /* everything runs here */
    cudaStream_t copy_stream;  /* slot_fill only, see the header comment */
    char devname[256];
    char memdesc[64];
    char memdesc2[64];
    int is_integrated;

    cu_w8 *W;  int nw;
    cu_w4 *W4; int nw4;

    /* Pinned host staging for one-time uploads (throwaway: freed right after
     * each upload_* call -- these run once at load time, not in the hot
     * path, so there is nothing to amortize). */

    /* Pinned staging for slot_fill specifically, kept resident and grown to
     * the high-water mark, per the async-plumbing instruction. */
    void   *h_fill; size_t fill_cap;

    /* ASYNC SLOT FILL (2026-09-15). fslots is the pinned staging ring;
     * fslot_next is the next slot to (re)use; fslot_cap is the per-slot
     * capacity every slot in the ring currently shares (grow-to-fit, like
     * fill_cap above, but sized once for the whole ring rather than per
     * call). async_disabled comes from COLI_CUDA_NO_ASYNC_FILL, read once at
     * open(). fill_mode_desc is what e_fill_mode() returns. */
    cu_fill_slot fslots[COLI_CUDA_FILL_INFLIGHT];
    int    fslot_next;
    size_t fslot_cap;
    int    async_disabled;
    char   fill_mode_desc[96];

    /* Activation staging, device side, grown to the high-water mark over
     * (n, I). Shared by every GEMM-family call. The host side (a->q,
     * a->scale) is the caller's own buffer -- copied directly by
     * cudaMemcpyAsync rather than staged through a second pinned copy,
     * since the caller already owns it and a GEMV's activation is small
     * next to its weights. */
    uint8_t *d_aq;  size_t d_aq_cap;
    float   *d_as;  size_t d_as_cap;

    /* Output staging, grown to the high-water mark over (n, O). */
    float *h_y; size_t y_cap; float *d_y; size_t d_y_cap;
    /* Second and third output buffers, for gemm4_qkv (q/k/v share one
     * activation upload but need three live outputs at once). */
    float *h_y2; size_t y2_cap; float *d_y2; size_t d_y2_cap;
    float *h_y3; size_t y3_cap; float *d_y3; size_t d_y3_cap;

    /* FFN intermediates, device-only, never downloaded. Sized to n*EI. */
    float   *d_fg; size_t fg_cap;
    float   *d_fu; size_t fu_cap;
    uint8_t *d_hq; size_t hq_cap;
    float   *d_hs; size_t hs_cap;
    int32_t *d_hm; size_t hm_cap;

    /* Per-call expert biases for ffn4_oai (gate then up, EI floats each). */
    float *h_bias; size_t bias_cap; float *d_bias; size_t d_bias_cap;

    /* ---- KV cache + decode attention (2026-09-15) ----
     * d_K[l]/d_V[l] are cudaMalloc'd [slots][kv_heads][kv_ctx][hd] float
     * buffers, one pair per layer -- the SAME layout vk_backend.c's kvK/kvV
     * use, so kv_load/kv_get/kv_write's host-side indexing arithmetic (owned
     * by model.cpp, not this file) needs no CUDA-specific variant. */
    float **d_K, **d_V;         /* [kv_layers] device pointers */
    int kv_layers, kv_slots, kv_heads, kv_ctx, kv_hd, kv_ok;
    float *d_sinks; size_t sinks_cap;   /* [layers][H] floats, uploaded once */

    /* attn_ex's own upload/download staging, grown to the high-water mark
     * over (n, H, hd) -- separate from d_aq/d_y above because those are sized
     * for GEMM's (n, I)/(n, O) and attention's shape does not share either. */
    float   *d_attn_q;  size_t d_attn_q_cap;
    float   *d_attn_o;  size_t d_attn_o_cap;  float *h_attn_o; size_t h_attn_o_cap;
    int32_t *d_attn_m;  size_t d_attn_m_cap;

    /* Pending kv_put rows: kv_put does not know which layer's buffer it is
     * headed for (the signature has no `layer`, see backend.h) -- only the
     * attn_ex call that follows does. Rows are copied into this host-side
     * ring immediately (the caller's row pointer is not guaranteed to
     * outlive the call) and applied to d_K[layer]/d_V[layer] at the start of
     * the next attn_ex, exactly mirroring vk_backend.c's kv_pend_off/
     * kv_pend_kv ring and coli_vk_kv_put's own comment about why a dropped
     * row is a silently wrong answer rather than a no-op. */
    int   pend_n;
    int   pend_slot[64], pend_kvh[64], pend_pos[64], pend_isv[64];
    float pend_row[64][256];   /* 256 = the hd<=256 ceiling attn_decode.comp shares */
} coli_cuda;

static int ensure_host_f(float **p, size_t *cap, size_t need_floats) {
    if (*cap >= need_floats) return 1;
    if (*p) cudaFreeHost(*p);
    if (cudaHostAlloc((void**)p, need_floats*sizeof(float), cudaHostAllocDefault) != cudaSuccess) { *p=NULL; *cap=0; return 0; }
    *cap = need_floats; return 1;
}
static int ensure_host_raw(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFreeHost(*p);
    if (cudaHostAlloc(p, need, cudaHostAllocDefault) != cudaSuccess) { *p=NULL; *cap=0; return 0; }
    *cap = need; return 1;
}
static int ensure_dev_u8(uint8_t **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFree(*p);
    if (cudaMalloc((void**)p, need) != cudaSuccess) { *p=NULL; *cap=0; return 0; }
    *cap = need; return 1;
}
static int ensure_dev_f(float **p, size_t *cap, size_t need_floats) {
    if (*cap >= need_floats) return 1;
    if (*p) cudaFree(*p);
    if (cudaMalloc((void**)p, need_floats*sizeof(float)) != cudaSuccess) { *p=NULL; *cap=0; return 0; }
    *cap = need_floats; return 1;
}
static int ensure_dev_i32(int32_t **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFree(*p);
    if (cudaMalloc((void**)p, need*sizeof(int32_t)) != cudaSuccess) { *p=NULL; *cap=0; return 0; }
    *cap = need; return 1;
}

/* ------------------------------------------------------------------ kernels
 *
 * All three GEMV kernels share one shape: ONE THREAD PER OUTPUT ROW, striding
 * over rows with the usual grid-stride loop, looping the n activation rows
 * inside. Parallelism comes from O (hundreds to low thousands of output
 * features), which is what a decode-time GEMV actually has to parallelize
 * over; a warp-cooperative reduction (the Vulkan shape) was considered and
 * set aside specifically BECAUSE it would reorder the block reduction and
 * give up the bit-exactness this file is built around for no measured
 * benefit at these row counts -- see test_cuda_gemm.c for the throughput
 * this choice actually gets.
 */

/* ---- int8: coli_w_i8, weights stored offset-to-unsigned (u = q+128) ----
 *
 * COALESCING (2026-09-15). `wpk` here is NOT the CPU's row-major [O][words]
 * layout -- e_upload_w transposes it to [words][O] on upload (see
 * transpose_i8_words) specifically so that consecutive THREADS (consecutive
 * `o`, still one thread per output row -- the parallelization axis is
 * unchanged) read ADJACENT device addresses at every step of the loop below,
 * instead of addresses `words` apart. Nothing about which thread computes
 * which row, which values it decodes, or the order it accumulates them in
 * changed -- only where each thread's next word comes from. `wsc[o]` was
 * already O-major (one scale per row, laid out contiguously by o) and is
 * untouched. */
__global__ void k_gemm_i8(const uint32_t * __restrict__ wpk, const float * __restrict__ wsc,
                           const uint32_t * __restrict__ xpk, const float * __restrict__ xsc,
                           float * __restrict__ y, int I, int O, int n) {
    int words = I >> 2;          /* COLI_ABLK=16 -> 4 words/block */
    int nb    = I >> 4;
    for (int o = blockIdx.x*blockDim.x + threadIdx.x; o < O; o += blockDim.x*gridDim.x) {
        float sc = wsc[o];
        for (int r = 0; r < n; r++) {
            const uint32_t *xr = xpk + (size_t)r*words;
            const float *xs = xsc + (size_t)r*nb;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                int32_t s = 0;
                int base = b*4;
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    /* wpk[words][O]: word index (base+w), row o -- adjacent o
                     * across the warp is adjacent in memory. */
                    uint32_t wv = wpk[(size_t)(base+w)*O + o] ^ 0x80808080u;   /* exact u-128, see header */
                    s = __dp4a((int)wv, (int)xr[base+w], s);
                }
                acc += xs[b]*(float)s;
            }
            y[(size_t)r*O + o] = acc*sc;
        }
    }
}

/* Deinterleave+decode one int4 weight block of 32 elements (coli_w_i4 layout:
 * element k lives in byte k/2, low nibble if k even, high if odd) into a
 * signed int8 scratch, exactly as the CPU scalar fallback in gemm_i8.cpp does
 * (NOT the AVX "interleave back" shuffle -- that produces the same values in
 * the same final positions, this is just the simpler form to write once). Two
 * decode tables share this shape and differ only in what `sel` returns. */
__device__ __forceinline__ int8_t dec_i4_biased(uint8_t nib) { return (int8_t)(nib - 8); }
__device__ __forceinline__ int8_t dec_i4_mx(uint8_t nib) {
    /* kvalues_mxfp4, DOUBLED e2m1 table -- same constant as gemm_i4_dp.comp's
     * mx_kv and gemm_mxfp4.cpp's CPU table. */
    static const int8_t kv[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    return kv[nib & 0xF];
}

/* COALESCING (2026-09-15), same fix as k_gemm_i8 above and for the same
 * measured reason (tests/test_cuda_gemm's 151936x2880 int4 head shape: 28-83
 * GB/s against a ~450-500 GB/s device roofline, see the CUDA-GEMV worklog).
 * `wq`/`wbs` here are the TRANSPOSED layout e_upload_w4 / e_slot_fill /
 * e_slot_fill_async build (see transpose_i4): bytes ordered [rowb][O] (byte
 * position within a row varies slowest, output row `o` fastest) and scales
 * [wnb][O]. `o` is still the parallelization axis -- one thread per output
 * row, same as before -- so this only changes which address a thread's next
 * byte/scale comes from, never which thread owns which row, which values it
 * decodes, or the order (b = 0..wnb-1, d0 before d1, same block-then-half
 * split) it accumulates them in. That is what keeps this EXACT against the
 * CPU reference rather than merely within-tolerance: int32 dp4a-4 groups are
 * identical, and the float `acc +=` sequence per row is untouched byte for
 * byte, only re-addressed. */
template<bool MX>
__global__ void k_gemm_i4(const uint8_t * __restrict__ wq, const float * __restrict__ wbs,
                           const int8_t * __restrict__ xq, const float * __restrict__ xsc,
                           float * __restrict__ y, int I, int O, int n) {
    int wnb  = I >> 5;    /* COLI_W4BLK=32 */
    int anb  = I >> 4;    /* COLI_ABLK=16 */
    for (int o = blockIdx.x*blockDim.x + threadIdx.x; o < O; o += blockDim.x*gridDim.x) {
        for (int r = 0; r < n; r++) {
            const int8_t *xr = xq  + (size_t)r*I;
            const float  *as = xsc + (size_t)r*anb;
            float acc = 0.f;
            for (int b = 0; b < wnb; b++) {
                int64_t k0 = (int64_t)b*32;
                int32_t d0 = 0, d1 = 0;
                /* sequential element order 0..31, matching gemm_i4_narrow's
                 * scalar fallback exactly -- see the header for why that is
                 * the order that keeps this bit-exact (non-MX case) with the
                 * CPU reference. Four elements at a time through __dp4a,
                 * which is an exact regrouping of the same int32 sum. */
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int32_t aw = 0, au = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int64_t k = k0 + q*4 + j;
                        /* wq[rowb][O]: byte index (k>>1), row o. */
                        uint8_t byte = wq[(size_t)(k >> 1)*O + o];
                        uint8_t nib = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                        int8_t wv = MX ? dec_i4_mx(nib) : dec_i4_biased(nib);
                        aw |= ((uint32_t)(uint8_t)wv) << (8*j);
                    }
                    au = *(const int32_t*)(xr + k0 + q*4);
                    d0 = __dp4a(aw, au, d0);
                }
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int32_t aw = 0, au = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int64_t k = k0 + 16 + q*4 + j;
                        uint8_t byte = wq[(size_t)(k >> 1)*O + o];
                        uint8_t nib = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                        int8_t wv = MX ? dec_i4_mx(nib) : dec_i4_biased(nib);
                        aw |= ((uint32_t)(uint8_t)wv) << (8*j);
                    }
                    au = *(const int32_t*)(xr + k0 + 16 + q*4);
                    d1 = __dp4a(aw, au, d1);
                }
                int64_t ab = k0 >> 4;   /* = b*2 */
                /* wbs[wnb][O]: block index b, row o. */
                acc += wbs[(size_t)b*O + o]*(as[ab]*(float)d0 + as[ab+1]*(float)d1);
            }
            y[(size_t)r*O + o] = acc;   /* no trailing row scale -- see header */
        }
    }
}

/* silu(g)*u, quantized to int8 COLI_ABLK=16 blocks -- the plain (non-OAI)
 * FFN middle. Mirrors coli_quantize_a's rounding (round-to-nearest via
 * __float2int_rn, matching lrintf under the default FE_TONEAREST both use)
 * and its [-127,127] clamp (128 would flip sign in int8 -- see gemm_i8.cpp). */
__global__ void k_silu_mul_q(const float * __restrict__ g, const float * __restrict__ u,
                              uint8_t * __restrict__ hq, float * __restrict__ hs, int32_t * __restrict__ hm,
                              int64_t total_blocks) {
    int64_t blk = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (blk >= total_blocks) return;
    int64_t base = blk*16;
    float h[16], amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float gv = g[base+i];
        float sv = gv/(1.f+expf(-gv));
        float hv = sv*u[base+i];
        h[i] = hv; amax = fmaxf(amax, fabsf(hv));
    }
    float sc = amax/127.f; if (sc < 1e-12f) sc = 1e-12f;
    hs[blk] = sc;
    float inv = 1.f/sc;
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int qi = __float2int_rn(h[i]*inv);
        if (qi > 127) qi = 127; if (qi < -127) qi = -127;
        sum += qi;
        hq[base+i] = (uint8_t)(int8_t)qi;
    }
    hm[blk] = sum;
}

/* ggml's swiglu_oai, biased, with alpha/limit, same quantize tail as above.
 * Mirrors silu_mul_q.comp's SWIGLU_OAI arm and model.cpp's expert_act CPU arm
 * (the ACT macro in test_vk_oai.c is the same arithmetic, transcribed). */
__global__ void k_swiglu_oai_q(const float * __restrict__ g, const float * __restrict__ u,
                                const float * __restrict__ bg, const float * __restrict__ bu,
                                uint8_t * __restrict__ hq, float * __restrict__ hs, int32_t * __restrict__ hm,
                                int64_t EI, int64_t total_blocks, float alpha, float limit) {
    int64_t blk = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (blk >= total_blocks) return;
    int64_t base = blk*16;
    int64_t col0 = base % EI;
    float h[16], amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float xg = g[base+i] + bg[col0+i]; if (xg > limit) xg = limit;
        float yu = u[base+i] + bu[col0+i]; if (yu > limit) yu = limit; if (yu < -limit) yu = -limit;
        float hv = (xg/(1.f+expf(alpha*(-xg))))*(yu+1.f);
        h[i] = hv; amax = fmaxf(amax, fabsf(hv));
    }
    float sc = amax/127.f; if (sc < 1e-12f) sc = 1e-12f;
    hs[blk] = sc;
    float inv = 1.f/sc;
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int qi = __float2int_rn(h[i]*inv);
        if (qi > 127) qi = 127; if (qi < -127) qi = -127;
        sum += qi;
        hq[base+i] = (uint8_t)(int8_t)qi;
    }
    hm[blk] = sum;
}

/* -------------------------------------------------------------- core calls
 * Operate entirely on device pointers already resident -- the shared inner
 * loop both the one-shot entry points (upload, dispatch, download) and the
 * fused FFN paths (upload once, dispatch three times, download once) funnel
 * through, so the two cannot drift apart. */
static int core_gemm8(coli_cuda *c, cu_w8 *w, const uint8_t *d_xq, const float *d_xs, int n, float *d_yout) {
    int threads = 128;
    int blocks = (int)((w->O + threads - 1)/threads); if (blocks < 1) blocks = 1;
    k_gemm_i8<<<blocks, threads, 0, c->stream>>>((const uint32_t*)w->d_qu, w->d_scale,
        (const uint32_t*)d_xq, d_xs, d_yout, (int)w->I, (int)w->O, n);
    CUDA_CHECK(cudaPeekAtLastError());
    return 0;
}
static int core_gemm4(coli_cuda *c, cu_w4 *w, const uint8_t *d_xq, const float *d_xs, int n, float *d_yout) {
    int threads = 128;
    int blocks = (int)((w->O + threads - 1)/threads); if (blocks < 1) blocks = 1;
    if (w->mx)
        k_gemm_i4<true><<<blocks, threads, 0, c->stream>>>(w->d_q4, w->d_bscale,
            (const int8_t*)d_xq, d_xs, d_yout, (int)w->I, (int)w->O, n);
    else
        k_gemm_i4<false><<<blocks, threads, 0, c->stream>>>(w->d_q4, w->d_bscale,
            (const int8_t*)d_xq, d_xs, d_yout, (int)w->I, (int)w->O, n);
    CUDA_CHECK(cudaPeekAtLastError());
    return 0;
}

/* ---------------------------------------------------------- upload repack
 *
 * CUDA-PRIVATE TRANSPOSE (2026-09-15). The CPU reference layout (gemm_i8.h)
 * is row-major -- [O][I] for int8, [O][I/2]+[O][wnb] for int4 -- which is the
 * right shape for the CPU's per-row VNNI/AVX2 kernels. k_gemm_i8/k_gemm_i4
 * parallelize over `o` with one thread per row (see their own header
 * comments for why that axis, not a warp-per-row split, is what keeps them
 * bit-exact), which means O *threads* need to advance through a row in
 * lockstep, and row-major storage puts consecutive threads `rowb` (or
 * `words`) bytes apart -- a fully strided, uncoalesced access on every load.
 * Transposing to [byte/word position][O] on upload puts consecutive threads
 * ADJACENT in memory instead, at zero cost to the kernel's correctness: it
 * is purely a change of WHERE a thread's next byte comes from, never which
 * thread owns which row, which values it decodes, or the order it
 * accumulates them in. This is the "change the upload layout, contained to
 * this file" case the task anticipated -- the CPU reference, coli_w_i4 and
 * the coli_backend seam are all untouched; only backend_cuda.cu's own device
 * buffers are reordered, and only backend_cuda.cu's own kernels read them. */
static void transpose_i8_words(uint32_t *dst, const uint8_t *src_qu, int64_t I, int64_t O) {
    int64_t words = I >> 2;
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *sr = src_qu + (size_t)o*I;
        for (int64_t w = 0; w < words; w++) {
            uint32_t v; memcpy(&v, sr + w*4, 4);
            dst[(size_t)w*O + o] = v;
        }
    }
}
static void transpose_i4(uint8_t *dst_q, float *dst_s, const uint8_t *src_q, const float *src_s,
                          int64_t I, int64_t O) {
    int64_t rowb = I >> 1, wnb = I/COLI_W4BLK;
    for (int64_t o = 0; o < O; o++) {
        const uint8_t *sr = src_q + (size_t)o*rowb;
        for (int64_t p = 0; p < rowb; p++) dst_q[(size_t)p*O + o] = sr[p];
        const float *ss = src_s + (size_t)o*wnb;
        for (int64_t b = 0; b < wnb; b++) dst_s[(size_t)b*O + o] = ss[b];
    }
}

/* Upload one coli_a_i8 to the shared device activation buffers. Returns the
 * device pointers through the three out-params (no aliasing: callers read
 * them immediately after). */
static int stage_activation(coli_cuda *c, const coli_a_i8 *a, uint8_t **out_q, float **out_s) {
    int64_t nb = a->I/COLI_ABLK;
    size_t qn = (size_t)a->n*a->I, sn = (size_t)a->n*nb;
    if (!ensure_dev_u8(&c->d_aq, &c->d_aq_cap, qn)) return -1;
    if (!ensure_dev_f(&c->d_as, &c->d_as_cap, sn)) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->d_aq, a->q, qn, cudaMemcpyHostToDevice, c->stream));
    CUDA_CHECK(cudaMemcpyAsync(c->d_as, a->scale, sn*sizeof(float), cudaMemcpyHostToDevice, c->stream));
    *out_q = c->d_aq; *out_s = c->d_as;
    return 0;
}

/* ------------------------------------------------------------------ entries */

static const char *e_device_name(void *ctx) { return ((coli_cuda*)ctx)->devname; }
static const char *e_memdesc(void *ctx)     { return ((coli_cuda*)ctx)->memdesc; }
static const char *e_memdesc2(void *ctx)    { return ((coli_cuda*)ctx)->memdesc2; }
static int e_is_integrated(void *ctx) { return ((coli_cuda*)ctx)->is_integrated; }
static int e_dot_used(void *ctx) { (void)ctx; return 1; }   /* __dp4a, every GEMM kernel above */

static int e_upload_w(void *ctx, const coli_w_i8 *w) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (c->nw >= CUDA_MAX_W) return -1;
    int h = c->nw;
    size_t wn = (size_t)w->I*w->O, sn = (size_t)w->O*sizeof(float);
    cu_w8 *s = &c->W[h];
    CUDA_CHECK(cudaMalloc((void**)&s->d_qu, wn));
    CUDA_CHECK(cudaMalloc((void**)&s->d_scale, sn));
    /* Transpose to [words][O] for k_gemm_i8's coalesced reads -- see the
     * "upload repack" section above stage_activation. One-time cost, this
     * runs at load time only, exactly like the plain memcpy it replaces. */
    uint32_t *tmp = (uint32_t*)malloc(wn);
    if (!tmp) { cudaFree(s->d_qu); cudaFree(s->d_scale); return -1; }
    transpose_i8_words(tmp, w->qu, w->I, w->O);
    CUDA_CHECK(cudaMemcpy(s->d_qu, tmp, wn, cudaMemcpyHostToDevice));
    free(tmp);
    CUDA_CHECK(cudaMemcpy(s->d_scale, w->scale, sn, cudaMemcpyHostToDevice));
    s->I = w->I; s->O = w->O; s->used = 1;
    c->nw++;
    return h;
}

static int e_gemm(void *ctx, int wh, const coli_a_i8 *a, float *y) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (wh < 0 || wh >= c->nw || !c->W[wh].used) return -1;
    if (a->I != c->W[wh].I) return -1;
    uint8_t *d_q; float *d_s;
    if (stage_activation(c, a, &d_q, &d_s) != 0) return -1;
    size_t yn = (size_t)a->n*c->W[wh].O;
    if (!ensure_dev_f(&c->d_y, &c->d_y_cap, yn)) return -1;
    if (!ensure_host_f(&c->h_y, &c->y_cap, yn)) return -1;
    if (core_gemm8(c, &c->W[wh], d_q, d_s, a->n, c->d_y) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->h_y, c->d_y, yn*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(y, c->h_y, yn*sizeof(float));
    return 0;
}

static int e_has_i4(void *ctx) { (void)ctx; return 1; }

static int e_upload_w4(void *ctx, const coli_w_i4 *w) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (c->nw4 >= CUDA_MAX_W) return -1;
    if (w->I % COLI_W4BLK) return -1;
    int h = c->nw4;
    size_t wn = (size_t)w->I*w->O/2, sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);
    cu_w4 *s = &c->W4[h];
    CUDA_CHECK(cudaMalloc((void**)&s->d_q4, wn));
    CUDA_CHECK(cudaMalloc((void**)&s->d_bscale, sn));
    /* Transpose to [rowb][O] / [wnb][O] for k_gemm_i4's coalesced reads --
     * see transpose_i4 above stage_activation. One-time cost (load time),
     * used by BOTH the plain and MX paths (e_upload_w4_mx calls through
     * here), matching the kernel's single MX-templated indexing. */
    uint8_t *tq = (uint8_t*)malloc(wn);
    float   *ts = (float*)malloc(sn);
    if (!tq || !ts) { free(tq); free(ts); cudaFree(s->d_q4); cudaFree(s->d_bscale); return -1; }
    transpose_i4(tq, ts, w->q4, w->bscale, w->I, w->O);
    CUDA_CHECK(cudaMemcpy(s->d_q4, tq, wn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_bscale, ts, sn, cudaMemcpyHostToDevice));
    free(tq); free(ts);
    s->I = w->I; s->O = w->O; s->used = 1; s->mx = 0;
    c->nw4++;
    return h;
}

static int e_has_mx(void *ctx) { (void)ctx; return 1; }

static int e_upload_w4_mx(void *ctx, const coli_w_i4 *w) {
    int h = e_upload_w4(ctx, w);
    if (h >= 0) ((coli_cuda*)ctx)->W4[h].mx = 1;
    return h;
}

static int e_slot_alloc_mx(void *ctx, int64_t I, int64_t O) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (c->nw4 >= CUDA_MAX_W || (I % COLI_W4BLK)) return -1;
    int h = c->nw4;
    size_t wn = (size_t)I*O/2, sn = (size_t)O*(I/COLI_W4BLK)*sizeof(float);
    cu_w4 *s = &c->W4[h];
    CUDA_CHECK(cudaMalloc((void**)&s->d_q4, wn));
    CUDA_CHECK(cudaMalloc((void**)&s->d_bscale, sn));
    s->I = I; s->O = O; s->used = 1; s->mx = 1;
    c->nw4++;
    return h;
}

/* Per the 2026-09-14 instruction: stage through a PERSISTENT pinned buffer and
 * copy on a DEDICATED stream, syncing that stream before returning. Timed the
 * same three ways as vk_backend.c's slot_fill_sync_timed, so CU_FSYNC and
 * CU_FASYNC (below) are directly comparable through coli_cuda_fill_stats. */
static int e_slot_fill(void *ctx, int h, const coli_w_i4 *w) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (h < 0 || h >= c->nw4 || !c->W4[h].used || !c->W4[h].mx) return -1;
    if (w->I != c->W4[h].I || w->O != c->W4[h].O) return -1;
    size_t wn = (size_t)w->I*w->O/2, sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);
    size_t need = wn + sn;
    if (!ensure_host_raw(&c->h_fill, &c->fill_cap, need)) return -1;
    uint64_t t0 = now_ns();
    /* Transpose straight into the pinned staging buffer -- same [rowb][O]/
     * [wnb][O] layout e_upload_w4 builds, see transpose_i4. Folded into the
     * existing "memcpy into staging" phase (CU_FSYNC.memcpy_ns): it is the
     * same asymptotic cost as the straight memcpy it replaces, just reordered
     * writes, and this phase is exactly what is supposed to carry that cost. */
    transpose_i4((uint8_t*)c->h_fill, (float*)((char*)c->h_fill + wn), w->q4, w->bscale, w->I, w->O);
    uint64_t t1 = now_ns(); CU_FSYNC.memcpy_ns += t1-t0;
    cu_w4 *s = &c->W4[h];
    CUDA_CHECK(cudaMemcpyAsync(s->d_q4, c->h_fill, wn, cudaMemcpyHostToDevice, c->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(s->d_bscale, (char*)c->h_fill + wn, sn, cudaMemcpyHostToDevice, c->copy_stream));
    uint64_t t2 = now_ns(); CU_FSYNC.recsub_ns += t2-t1;
    CUDA_CHECK(cudaStreamSynchronize(c->copy_stream));   /* synchronous for now, see header */
    uint64_t t3 = now_ns(); CU_FSYNC.wait_ns += t3-t2;
    CU_FSYNC.n++;
    return 0;
}

/* ---- async fill ring: grow-to-fit, lazily, like h_fill above ---- */
static int fill_slot_wait_one(coli_cuda *c, cu_fill_slot *s) {
    if (!s->pending) return 1;
    uint64_t t0 = now_ns();
    cudaError_t e = cudaEventSynchronize(s->event);
    CU_FASYNC.wait_ns += now_ns()-t0;
    s->pending = 0;
    if (e != cudaSuccess) { cuda_warn_once("cudaEventSynchronize(fill)", e); return 0; }
    return 1;
}

/* Ensure every ring slot has a pinned staging buffer of at least `need` bytes
 * and an event. Called on every e_slot_fill_async; cheap after the first call
 * to a given shape because `need` never changes for a fixed (I,O) expert
 * shape. Growing (a different, larger shape later) drains every pending fill
 * first -- the old, smaller staging buffers would otherwise be freed out from
 * under a copy still in flight. Mirrors vk_backend.c's fill_ring_ensure. */
static int fill_ring_ensure(coli_cuda *c, size_t need) {
    if (c->fslot_cap >= need && c->fslots[0].h_buf) return 1;
    for (int i = 0; i < COLI_CUDA_FILL_INFLIGHT; i++) fill_slot_wait_one(c, &c->fslots[i]);
    for (int i = 0; i < COLI_CUDA_FILL_INFLIGHT; i++) {
        cu_fill_slot *s = &c->fslots[i];
        if (s->h_buf) { cudaFreeHost(s->h_buf); s->h_buf = NULL; s->cap = 0; }
        if (cudaHostAlloc(&s->h_buf, need, cudaHostAllocDefault) != cudaSuccess) { s->h_buf = NULL; return 0; }
        s->cap = need;
        if (!s->event) {
            if (cudaEventCreate(&s->event) != cudaSuccess) return 0;
        }
    }
    c->fslot_cap = need;
    return 1;
}

/* ASYNC SLOT FILL (2026-09-15). Mirrors coli_vk_slot_fill_async's contract
 * exactly (see vk_backend.h's header comment on it): repacks into the next
 * free ring slot (waiting on THAT slot's event first if all
 * COLI_CUDA_FILL_INFLIGHT are still in flight -- the specified backpressure),
 * issues the H2D copy on c->copy_stream (never the compute stream), and
 * records an event. Returns 0 once the copy is QUEUED, not landed -- the
 * caller must not read the slot until e_slot_fill_wait() returns. Falls back
 * to the synchronous e_slot_fill when COLI_CUDA_NO_ASYNC_FILL is set (read
 * once at open()) or when this slot is not an MXFP4 slot to begin with (same
 * two guard checks e_slot_fill makes, kept in front so a caller cannot get a
 * different validation result from the two entry points). */
static int e_slot_fill_async(void *ctx, int h, const coli_w_i4 *w) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (h < 0 || h >= c->nw4 || !c->W4[h].used || !c->W4[h].mx) return -1;
    if (w->I != c->W4[h].I || w->O != c->W4[h].O) return -1;
    if (c->async_disabled) { int r = e_slot_fill(ctx, h, w); if (r == 0) CU_FASYNC.n++; return r; }

    size_t wn = (size_t)w->I*w->O/2, sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);
    size_t need = wn + sn;
    if (!fill_ring_ensure(c, need)) return -1;

    cu_fill_slot *s = &c->fslots[c->fslot_next];
    if (!fill_slot_wait_one(c, s)) return -1;   /* ring full: back-pressure, as specified */

    uint64_t t0 = now_ns();
    /* Same transpose as e_slot_fill, into this ring slot's pinned buffer. */
    transpose_i4((uint8_t*)s->h_buf, (float*)((char*)s->h_buf + wn), w->q4, w->bscale, w->I, w->O);
    uint64_t t1 = now_ns(); CU_FASYNC.memcpy_ns += t1-t0;

    cu_w4 *dst = &c->W4[h];
    CUDA_CHECK(cudaMemcpyAsync(dst->d_q4, s->h_buf, wn, cudaMemcpyHostToDevice, c->copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(dst->d_bscale, (char*)s->h_buf + wn, sn, cudaMemcpyHostToDevice, c->copy_stream));
    CUDA_CHECK(cudaEventRecord(s->event, c->copy_stream));
    uint64_t t2 = now_ns(); CU_FASYNC.recsub_ns += t2-t1;
    s->pending = 1;
    c->fslot_next = (c->fslot_next + 1) % COLI_CUDA_FILL_INFLIGHT;
    CU_FASYNC.n++;
    return 0;
}

/* The ONLY place that makes an async fill's writes visible to a subsequent
 * compute dispatch. cudaEventSynchronize blocks the calling (host) thread
 * until the copy stream's event has been observed signaled, which by CUDA's
 * stream-ordering guarantee means every byte the copy wrote is visible to
 * any kernel this process subsequently launches -- so the host wait alone is
 * already sufficient for correctness. cudaStreamWaitEvent on c->stream is
 * added on top per the 2026-09-14 instruction ("make the compute stream
 * depend on it") -- belt-and-suspenders against a future caller that stops
 * waiting on the host and instead only enqueues work on c->stream, which
 * would otherwise be reading a slot the runtime has no ordering guarantee
 * over. A caller that dispatches a GEMM over a slot without calling this
 * first is reading memory the device may not have finished writing, exactly
 * the bug the test's deliberate control is built to catch. */
static int e_slot_fill_wait(void *ctx) {
    coli_cuda *c = (coli_cuda*)ctx;
    int ok = 1;
    for (int i = 0; i < COLI_CUDA_FILL_INFLIGHT; i++) {
        cu_fill_slot *s = &c->fslots[i];
        int was_pending = s->pending;
        if (!fill_slot_wait_one(c, s)) ok = 0;
        if (was_pending) {
            cudaError_t e = cudaStreamWaitEvent(c->stream, s->event, 0);
            if (e != cudaSuccess) { cuda_warn_once("cudaStreamWaitEvent(fill)", e); ok = 0; }
        }
    }
    return ok ? 0 : -1;
}

static const char *e_fill_mode(void *ctx) {
    coli_cuda *c = (coli_cuda*)ctx;
    return (c && c->fill_mode_desc[0]) ? c->fill_mode_desc : "unknown";
}

/* Mean milliseconds per e_slot_fill / e_slot_fill_async call, same layout as
 * coli_vk_fill_stats: out[0]=memcpy into staging, out[1]=command issue
 * (cudaMemcpyAsync x2 [+cudaEventRecord]), out[2]=wait, out[3]=sum. which=0
 * reads the sync-path counters, which=1 the async-path counters. Not part of
 * the coli_backend seam (backend.h has no fill_stats entry, matching
 * vk_backend.h's own note that profiling entries are test-only) -- exported
 * so tests/test_cuda_fill.c can declare and call it directly, the same way
 * tests/test_cuda_gemm.c already declares coli_gemm_i8_ref itself. Safe to
 * call with zero recorded calls (all zero out). */
extern "C" void coli_cuda_fill_stats(void *ctx, int which, double out[4]) {
    (void)ctx;
    out[0]=out[1]=out[2]=out[3]=0.0;
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;
    if (which) { memcpy_ns=CU_FASYNC.memcpy_ns; recsub_ns=CU_FASYNC.recsub_ns; wait_ns=CU_FASYNC.wait_ns; n=CU_FASYNC.n; }
    else       { memcpy_ns=CU_FSYNC.memcpy_ns;  recsub_ns=CU_FSYNC.recsub_ns;  wait_ns=CU_FSYNC.wait_ns;  n=CU_FSYNC.n; }
    if (!n) return;
    out[0] = (double)memcpy_ns/1e6/(double)n;
    out[1] = (double)recsub_ns/1e6/(double)n;
    out[2] = (double)wait_ns/1e6/(double)n;
    out[3] = out[0]+out[1]+out[2];
}

static int e_upload_begin(void *ctx) { (void)ctx; return 0; }
static int e_upload_end(void *ctx)   { (void)ctx; return 1; }

static int e_gemm4(void *ctx, int wh, const coli_a_i8 *a, float *y) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (wh < 0 || wh >= c->nw4 || !c->W4[wh].used) return -1;
    if (a->I != c->W4[wh].I) return -1;
    uint8_t *d_q; float *d_s;
    if (stage_activation(c, a, &d_q, &d_s) != 0) return -1;
    size_t yn = (size_t)a->n*c->W4[wh].O;
    if (!ensure_dev_f(&c->d_y, &c->d_y_cap, yn)) return -1;
    if (!ensure_host_f(&c->h_y, &c->y_cap, yn)) return -1;
    if (core_gemm4(c, &c->W4[wh], d_q, d_s, a->n, c->d_y) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->h_y, c->d_y, yn*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(y, c->h_y, yn*sizeof(float));
    return 0;
}

static int e_gemm4_qkv(void *ctx, const int *wh, const coli_a_i8 *a, float **ys) {
    coli_cuda *c = (coli_cuda*)ctx;
    for (int i = 0; i < 3; i++) if (wh[i] < 0 || wh[i] >= c->nw4 || !c->W4[wh[i]].used || c->W4[wh[i]].I != a->I) return -1;
    uint8_t *d_q; float *d_s;
    if (stage_activation(c, a, &d_q, &d_s) != 0) return -1;
    size_t y0 = (size_t)a->n*c->W4[wh[0]].O, y1 = (size_t)a->n*c->W4[wh[1]].O, y2 = (size_t)a->n*c->W4[wh[2]].O;
    if (!ensure_dev_f(&c->d_y,  &c->d_y_cap,  y0)) return -1;
    if (!ensure_dev_f(&c->d_y2, &c->d_y2_cap, y1)) return -1;
    if (!ensure_dev_f(&c->d_y3, &c->d_y3_cap, y2)) return -1;
    if (!ensure_host_f(&c->h_y,  &c->y_cap,  y0)) return -1;
    if (!ensure_host_f(&c->h_y2, &c->y2_cap, y1)) return -1;
    if (!ensure_host_f(&c->h_y3, &c->y3_cap, y2)) return -1;
    if (core_gemm4(c, &c->W4[wh[0]], d_q, d_s, a->n, c->d_y)  != 0) return -1;
    if (core_gemm4(c, &c->W4[wh[1]], d_q, d_s, a->n, c->d_y2) != 0) return -1;
    if (core_gemm4(c, &c->W4[wh[2]], d_q, d_s, a->n, c->d_y3) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->h_y,  c->d_y,  y0*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaMemcpyAsync(c->h_y2, c->d_y2, y1*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaMemcpyAsync(c->h_y3, c->d_y3, y2*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(ys[0], c->h_y,  y0*sizeof(float));
    memcpy(ys[1], c->h_y2, y1*sizeof(float));
    memcpy(ys[2], c->h_y3, y2*sizeof(float));
    return 0;
}

static int e_has_ffn(void *ctx) { (void)ctx; return 1; }

static int e_ffn4(void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (hg<0||hg>=c->nw4||hu<0||hu>=c->nw4||hd<0||hd>=c->nw4) return -1;
    if (!c->W4[hg].used||!c->W4[hu].used||!c->W4[hd].used) return -1;
    if (c->W4[hg].mx||c->W4[hu].mx||c->W4[hd].mx) return -1;   /* plain int4 only; see ffn4_oai for MX */
    int64_t D = c->W4[hg].I, EI = c->W4[hg].O, Dout = c->W4[hd].O;
    if (a->I != D || c->W4[hu].I != D || c->W4[hu].O != EI || c->W4[hd].I != EI) return -1;
    int n = a->n; int64_t nbE = EI/COLI_ABLK;

    uint8_t *d_q; float *d_s;
    if (stage_activation(c, a, &d_q, &d_s) != 0) return -1;
    if (!ensure_dev_f(&c->d_fg, &c->fg_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_f(&c->d_fu, &c->fu_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_u8(&c->d_hq, &c->hq_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_f(&c->d_hs, &c->hs_cap, (size_t)n*nbE)) return -1;
    if (!ensure_dev_i32(&c->d_hm, &c->hm_cap, (size_t)n*nbE)) return -1;
    size_t yn = (size_t)n*Dout;
    if (!ensure_dev_f(&c->d_y, &c->d_y_cap, yn)) return -1;
    if (!ensure_host_f(&c->h_y, &c->y_cap, yn)) return -1;

    if (core_gemm4(c, &c->W4[hg], d_q, d_s, n, c->d_fg) != 0) return -1;
    if (core_gemm4(c, &c->W4[hu], d_q, d_s, n, c->d_fu) != 0) return -1;
    int64_t total_blocks = (int64_t)n*nbE;
    int threads = 64; int blocks = (int)((total_blocks+threads-1)/threads); if (blocks<1) blocks=1;
    k_silu_mul_q<<<blocks, threads, 0, c->stream>>>(c->d_fg, c->d_fu, c->d_hq, c->d_hs, c->d_hm, total_blocks);
    CUDA_CHECK(cudaPeekAtLastError());
    if (core_gemm4(c, &c->W4[hd], c->d_hq, c->d_hs, n, c->d_y) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->h_y, c->d_y, yn*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(y, c->h_y, yn*sizeof(float));
    return 0;
}

static int e_has_ffn_oai(void *ctx) { (void)ctx; return 1; }

static int e_ffn4_oai(void *ctx, int hg, int hu, int hd, const coli_a_i8 *a, float *y,
                       const float *bg, const float *bu, float alpha, float limit) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (hg<0||hg>=c->nw4||hu<0||hu>=c->nw4||hd<0||hd>=c->nw4) return -1;
    if (!c->W4[hg].used||!c->W4[hu].used||!c->W4[hd].used) return -1;
    if (!c->W4[hg].mx||!c->W4[hu].mx||!c->W4[hd].mx) return -1;
    int64_t D = c->W4[hg].I, EI = c->W4[hg].O, Dout = c->W4[hd].O;
    if (a->I != D || c->W4[hu].I != D || c->W4[hu].O != EI || c->W4[hd].I != EI) return -1;
    int n = a->n; int64_t nbE = EI/COLI_ABLK;

    uint8_t *d_q; float *d_s;
    if (stage_activation(c, a, &d_q, &d_s) != 0) return -1;
    if (!ensure_dev_f(&c->d_fg, &c->fg_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_f(&c->d_fu, &c->fu_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_u8(&c->d_hq, &c->hq_cap, (size_t)n*EI)) return -1;
    if (!ensure_dev_f(&c->d_hs, &c->hs_cap, (size_t)n*nbE)) return -1;
    if (!ensure_dev_i32(&c->d_hm, &c->hm_cap, (size_t)n*nbE)) return -1;
    if (!ensure_host_f(&c->h_bias, &c->bias_cap, (size_t)2*EI)) return -1;
    if (!ensure_dev_f(&c->d_bias, &c->d_bias_cap, (size_t)2*EI)) return -1;
    size_t yn = (size_t)n*Dout;
    if (!ensure_dev_f(&c->d_y, &c->d_y_cap, yn)) return -1;
    if (!ensure_host_f(&c->h_y, &c->y_cap, yn)) return -1;

    memcpy(c->h_bias, bg, (size_t)EI*sizeof(float));
    memcpy(c->h_bias + EI, bu, (size_t)EI*sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(c->d_bias, c->h_bias, (size_t)2*EI*sizeof(float), cudaMemcpyHostToDevice, c->stream));

    if (core_gemm4(c, &c->W4[hg], d_q, d_s, n, c->d_fg) != 0) return -1;
    if (core_gemm4(c, &c->W4[hu], d_q, d_s, n, c->d_fu) != 0) return -1;
    int64_t total_blocks = (int64_t)n*nbE;
    int threads = 64; int blocks = (int)((total_blocks+threads-1)/threads); if (blocks<1) blocks=1;
    k_swiglu_oai_q<<<blocks, threads, 0, c->stream>>>(c->d_fg, c->d_fu, c->d_bias, c->d_bias+EI,
        c->d_hq, c->d_hs, c->d_hm, EI, total_blocks, alpha, limit);
    CUDA_CHECK(cudaPeekAtLastError());
    if (core_gemm4(c, &c->W4[hd], c->d_hq, c->d_hs, n, c->d_y) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(c->h_y, c->d_y, yn*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(y, c->h_y, yn*sizeof(float));
    return 0;
}

/* ---- KV cache + decode attention (2026-09-15) ----
 *
 * WHY A SEPARATE KERNEL FROM attn_decode.comp RATHER THAN A PORT OF IT.
 * That shader's four-subgroup split exists to give AMD/Intel/NVIDIA alike
 * something to parallelize over at n=1 (32 workgroups is little concurrency
 * for a modern discrete part) and pays for it with a shared-memory merge at
 * the end. This kernel targets one device, so it keeps the same PARALLEL
 * DECOMPOSITION model.cpp's attend_online() already uses -- one thread group
 * per (row, head), walking t sequentially from t0 to tmax with a running
 * max/denominator -- and gets its parallelism from splitting head_dim across
 * one warp's 32 lanes (a __shfl_xor_sync tree reduces the dot product; no
 * shared memory, no barrier, no cross-subgroup merge to get subtly wrong).
 * Correct and simple first: the sequential-t loop means one warp's memory
 * traffic is 2*tmax*hd*4 bytes with no reuse across warps, which is the
 * obvious place a chunked/split-K version (mirroring attn_decode_split.comp)
 * would help if this ever profiles as the bottleneck. Not built here.
 *
 * MATCHES attend_online(), NOT attn_decode.comp's merge order. Sink handling
 * seeds (m, d) = (sink, 1) BEFORE the t loop, exactly as attend_online() does
 * (model.cpp) -- not folded in afterward the way the multi-subgroup shader
 * has to, because a single sequential accumulator has nothing to fold. Same
 * identity, and here it is also the SAME order of operations as the CPU
 * reference bar the dot-product reduction, so the two agree tighter than
 * Vulkan's bound (measured below, tests/test_cuda_attn.c).
 *
 * NOT BIT-EXACT with the CPU regardless: __shfl_xor_sync reduces the hd-wide
 * dot product in a tree, the CPU sums it sequentially (or via AVX2 8-wide
 * accumulators, itself not sequential) -- different order, different
 * rounding, same as every other float kernel in this file that touches a
 * reduction wider than one lane. Checked against a stated relative bound and
 * a control that must exceed it, same discipline as vk_backend.h documents.
 */
#define ATTN_MAXD 8   /* hd <= 32*ATTN_MAXD = 256, mirrors attn_decode.comp's MAXD */

__global__ void k_attn_decode(const float * __restrict__ q, const float * __restrict__ kc,
                               const float * __restrict__ vc, float * __restrict__ outp,
                               const int * __restrict__ meta, const float * __restrict__ sinks,
                               int H, int KVH, int hd, int kv_ctx, int n,
                               float scale, int window, int sink_off) {
    int wg = blockIdx.x;
    int r  = wg / H;
    int h  = wg % H;
    if (r >= n) return;

    int lane = threadIdx.x;              /* 0..31, one warp per block */
    int slot = meta[r*2 + 0];
    int tmax = meta[r*2 + 1];
    int grp  = H / KVH;
    int kvh  = h / grp;

    long long kvbase = ((long long)(slot*KVH + kvh) * kv_ctx) * hd;
    long long qbase  = (long long)r*H*hd + (long long)h*hd;

    float qv[ATTN_MAXD]; int nd = 0;
    for (int i = lane; i < hd; i += 32) { qv[nd] = q[qbase + i]; nd++; }

    float acc[ATTN_MAXD];
    #pragma unroll
    for (int j = 0; j < ATTN_MAXD; j++) acc[j] = 0.f;

    int has_sink = sink_off >= 0;
    float sk = has_sink ? sinks[sink_off + h] : 0.f;
    /* seeded exactly as attend_online(): m=sink,d=1 with a sink, else the
     * same -1e30f/0 pair (m starts below any real score so the first t is
     * never rescaled against a stale maximum -- see attend_online's comment
     * on why -1e30f rather than -INFINITY). */
    float m = has_sink ? sk : -1e30f;
    float d = has_sink ? 1.f  : 0.f;

    int t0 = 0;
    if (window > 0) { t0 = tmax - window + 1; if (t0 < 0) t0 = 0; }

    for (int t = t0; t <= tmax; t++) {
        long long kb = kvbase + (long long)t*hd;
        float part = 0.f; int j = 0;
        for (int i = lane; i < hd; i += 32) { part += qv[j]*kc[kb + i]; j++; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        float s = part*scale;             /* uniform across the warp */

        float mn   = fmaxf(m, s);
        float corr = expf(m - mn);
        float pw   = expf(s - mn);
        d = d*corr + pw;
        j = 0;
        for (int i = lane; i < hd; i += 32) { acc[j] = acc[j]*corr + pw*vc[kb + i]; j++; }
        m = mn;
    }

    int j = 0;
    for (int i = lane; i < hd; i += 32) { outp[qbase + i] = d > 0.f ? acc[j]/d : 0.f; j++; }
}

static int e_has_attn(void *ctx) { (void)ctx; return 1; }

static void kv_free_all_cuda(coli_cuda *c) {
    if (c->d_K) { for (int i = 0; i < c->kv_layers; i++) if (c->d_K[i]) cudaFree(c->d_K[i]); free(c->d_K); c->d_K = NULL; }
    if (c->d_V) { for (int i = 0; i < c->kv_layers; i++) if (c->d_V[i]) cudaFree(c->d_V[i]); free(c->d_V); c->d_V = NULL; }
    c->kv_ok = 0; c->pend_n = 0;
}

static int e_kv_init(void *ctx, int layers, int slots, int kv_heads, int kv_ctx, int hd) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (layers <= 0 || slots <= 0 || kv_heads <= 0 || kv_ctx <= 0) return -1;
    if (hd > 256 || (hd % 32)) return -1;

    kv_free_all_cuda(c);
    c->d_K = (float**)calloc((size_t)layers, sizeof(float*));
    c->d_V = (float**)calloc((size_t)layers, sizeof(float*));
    if (!c->d_K || !c->d_V) { kv_free_all_cuda(c); return -1; }
    c->kv_layers = layers;

    size_t per = (size_t)slots * kv_heads * kv_ctx * hd * sizeof(float);
    for (int i = 0; i < layers; i++) {
        /* Partial failure frees EVERYTHING -- see vk_backend.c's kv_init for
         * why a half-resident cache is worse than none: layer 3 on device and
         * layer 4 on host with no way to tell looks like a model bug. */
        if (cudaMalloc((void**)&c->d_K[i], per) != cudaSuccess ||
            cudaMalloc((void**)&c->d_V[i], per) != cudaSuccess) {
            cuda_warn_once("cudaMalloc(kv)", cudaGetLastError());
            kv_free_all_cuda(c); return -1;
        }
    }
    c->kv_slots = slots; c->kv_heads = kv_heads; c->kv_ctx = kv_ctx; c->kv_hd = hd;
    c->pend_n = 0; c->kv_ok = 1;
    return 0;
}

static int e_kv_ready(void *ctx) { return ((coli_cuda*)ctx)->kv_ok; }
static size_t e_kv_bytes(void *ctx) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok) return 0;
    return (size_t)c->kv_layers * 2 * c->kv_slots * c->kv_heads
         * (size_t)c->kv_ctx * c->kv_hd * sizeof(float);
}

static int e_kv_load(void *ctx, int layer, const float *K, const float *V) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok || layer < 0 || layer >= c->kv_layers) return -1;
    size_t n = (size_t)c->kv_slots * c->kv_heads * c->kv_ctx * c->kv_hd * sizeof(float);
    CUDA_CHECK(cudaMemcpy(c->d_K[layer], K, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_V[layer], V, n, cudaMemcpyHostToDevice));
    return 0;
}

static int e_kv_get(void *ctx, int layer, float *K, float *V) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok || layer < 0 || layer >= c->kv_layers) return -1;
    size_t n = (size_t)c->kv_slots * c->kv_heads * c->kv_ctx * c->kv_hd * sizeof(float);
    CUDA_CHECK(cudaMemcpy(K, c->d_K[layer], n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V, c->d_V[layer], n, cudaMemcpyDeviceToHost));
    return 0;
}

static int e_kv_ctx(void *ctx) { coli_cuda *c = (coli_cuda*)ctx; return c->kv_ok ? c->kv_ctx : 0; }

/* Stage one row; applied at the START of the next attn_ex (see the struct
 * comment on pend_* for why the layer is not known here). Returns -1 if the
 * ring is full -- callers MUST fall back to CPU for this token, not ignore
 * it, exactly as vk_backend.h's contract states. */
static int e_kv_put(void *ctx, int slot, int kvh, int pos, int is_v, const float *row) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok) return -1;
    if (c->pend_n >= 64) return -1;
    if (slot < 0 || slot >= c->kv_slots || kvh < 0 || kvh >= c->kv_heads) return -1;
    if (pos < 0 || pos >= c->kv_ctx) return -1;
    if (c->kv_hd > 256) return -1;
    memcpy(c->pend_row[c->pend_n], row, (size_t)c->kv_hd*sizeof(float));
    c->pend_slot[c->pend_n] = slot; c->pend_kvh[c->pend_n] = kvh;
    c->pend_pos[c->pend_n]  = pos;  c->pend_isv[c->pend_n]  = is_v ? 1 : 0;
    c->pend_n++;
    return 0;
}

/* Bulk-write a contiguous run of positions, every kv head of one layer --
 * the PREFILL path. Khost/Vhost are the layer's FULL host caches, shaped
 * [kv_heads][kv_ctx][hd]; this function does the (kvh, pos0) indexing on
 * both sides itself, mirroring coli_vk_kv_write's contract exactly (same
 * function, same reason: the caller must not get the stride right in one
 * place and wrong in the other). One cudaMemcpyAsync per kv head per buffer
 * on the shared compute stream, synced once at the end -- simple and
 * correct; a single fused copy would need the same restriding either on the
 * host side (an extra memcpy) or via cudaMemcpy2D (worth trying if this
 * profiles hot, not done here). */
static int e_kv_write(void *ctx, int layer, int slot, int pos0, int count,
                       const float *Khost, const float *Vhost) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok) return -1;
    if (layer < 0 || layer >= c->kv_layers) return -1;
    if (slot < 0 || slot >= c->kv_slots) return -1;
    if (count <= 0 || pos0 < 0 || pos0 + count > c->kv_ctx) return -1;
    if (!Khost || !Vhost) return -1;

    int KVH = c->kv_heads, hd = c->kv_hd;
    size_t run = (size_t)count * hd * sizeof(float);
    for (int h = 0; h < KVH; h++) {
        size_t doff = (((size_t)slot*KVH + h)*c->kv_ctx + pos0)*hd;
        const float *ksrc = Khost + ((size_t)h*c->kv_ctx + pos0)*hd;
        const float *vsrc = Vhost + ((size_t)h*c->kv_ctx + pos0)*hd;
        CUDA_CHECK(cudaMemcpyAsync(c->d_K[layer] + doff, ksrc, run, cudaMemcpyHostToDevice, c->stream));
        CUDA_CHECK(cudaMemcpyAsync(c->d_V[layer] + doff, vsrc, run, cudaMemcpyHostToDevice, c->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    return 0;
}

static int e_attn_sinks_upload(void *ctx, const float *sinks, size_t nfloat) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!sinks || !nfloat) return -1;
    if (!ensure_dev_f(&c->d_sinks, &c->sinks_cap, nfloat)) return -1;
    CUDA_CHECK(cudaMemcpy(c->d_sinks, sinks, nfloat*sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

/* PRODUCTION attention entry. Applies any rows staged by e_kv_put since the
 * last call (see pend_*), then dispatches k_attn_decode -- both on the same
 * stream, so the kernel is correctly ordered after the writes without an
 * explicit event: CUDA stream semantics guarantee in-order execution of
 * work enqueued on one stream, which is the same guarantee vk_backend.c gets
 * from recording both into one command buffer. */
static int e_attn_ex(void *ctx, int layer, const float *q, float *out,
                      const int *meta, int n, int H, float scale, int window, int sink_off) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c->kv_ok) return -1;
    if (sink_off >= 0 && !c->d_sinks) return -1;   /* sink offset with nothing uploaded: refuse, don't read garbage */
    if (layer < 0 || layer >= c->kv_layers) return -1;
    if (H % c->kv_heads) return -1;
    int hd = c->kv_hd;
    if (hd > 256 || (hd % 32)) return -1;

    size_t qn = (size_t)n*H*hd;
    size_t mn = (size_t)n*2;
    if (!ensure_dev_f(&c->d_attn_q, &c->d_attn_q_cap, qn)) return -1;
    if (!ensure_dev_f(&c->d_attn_o, &c->d_attn_o_cap, qn)) return -1;
    if (!ensure_dev_i32(&c->d_attn_m, &c->d_attn_m_cap, mn)) return -1;
    if (!ensure_host_f(&c->h_attn_o, &c->h_attn_o_cap, qn)) return -1;

    CUDA_CHECK(cudaMemcpyAsync(c->d_attn_q, q, qn*sizeof(float), cudaMemcpyHostToDevice, c->stream));
    /* `meta` is `int`, the device buffer is int32_t -- same width on every
     * platform this engine targets (LP64), so a straight byte copy is exact;
     * no per-element cast is needed the way there would be for a narrower
     * mismatch. */
    CUDA_CHECK(cudaMemcpyAsync(c->d_attn_m, meta, mn*sizeof(int32_t), cudaMemcpyHostToDevice, c->stream));

    for (int i = 0; i < c->pend_n; i++) {
        size_t off = (((size_t)c->pend_slot[i]*c->kv_heads + c->pend_kvh[i])*c->kv_ctx + c->pend_pos[i])*hd;
        float *dst = (c->pend_isv[i] ? c->d_V[layer] : c->d_K[layer]) + off;
        CUDA_CHECK(cudaMemcpyAsync(dst, c->pend_row[i], (size_t)hd*sizeof(float), cudaMemcpyHostToDevice, c->stream));
    }
    c->pend_n = 0;

    dim3 grid((unsigned)(n*H)), block(32);
    k_attn_decode<<<grid, block, 0, c->stream>>>(c->d_attn_q, c->d_K[layer], c->d_V[layer],
        c->d_attn_o, c->d_attn_m, c->d_sinks, H, c->kv_heads, hd, c->kv_ctx, n, scale, window, sink_off);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpyAsync(c->h_attn_o, c->d_attn_o, qn*sizeof(float), cudaMemcpyDeviceToHost, c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    memcpy(out, c->h_attn_o, qn*sizeof(float));
    return 0;
}
/* ---- end KV cache + decode attention ---- */

static void cuda_close(void *ctx) {
    coli_cuda *c = (coli_cuda*)ctx;
    if (!c) return;
    for (int i = 0; i < c->nw;  i++) { cudaFree(c->W[i].d_qu);  cudaFree(c->W[i].d_scale); }
    for (int i = 0; i < c->nw4; i++) { cudaFree(c->W4[i].d_q4); cudaFree(c->W4[i].d_bscale); }
    free(c->W); free(c->W4);
    if (c->h_fill) cudaFreeHost(c->h_fill);
    for (int i = 0; i < COLI_CUDA_FILL_INFLIGHT; i++) {
        if (c->fslots[i].h_buf) cudaFreeHost(c->fslots[i].h_buf);
        if (c->fslots[i].event) cudaEventDestroy(c->fslots[i].event);
    }
    if (c->d_aq) cudaFree(c->d_aq);
    if (c->d_as) cudaFree(c->d_as);
    if (c->h_y)  cudaFreeHost(c->h_y);  if (c->d_y)  cudaFree(c->d_y);
    if (c->h_y2) cudaFreeHost(c->h_y2); if (c->d_y2) cudaFree(c->d_y2);
    if (c->h_y3) cudaFreeHost(c->h_y3); if (c->d_y3) cudaFree(c->d_y3);
    if (c->d_fg) cudaFree(c->d_fg);
    if (c->d_fu) cudaFree(c->d_fu);
    if (c->d_hq) cudaFree(c->d_hq);
    if (c->d_hs) cudaFree(c->d_hs);
    if (c->d_hm) cudaFree(c->d_hm);
    if (c->h_bias) cudaFreeHost(c->h_bias); if (c->d_bias) cudaFree(c->d_bias);
    /* ---- KV cache + decode attention (2026-09-15) ---- */
    kv_free_all_cuda(c);
    if (c->d_sinks) cudaFree(c->d_sinks);
    if (c->d_attn_q) cudaFree(c->d_attn_q);
    if (c->d_attn_o) cudaFree(c->d_attn_o);
    if (c->d_attn_m) cudaFree(c->d_attn_m);
    if (c->h_attn_o) cudaFreeHost(c->h_attn_o);
    if (c->stream) cudaStreamDestroy(c->stream);
    if (c->copy_stream) cudaStreamDestroy(c->copy_stream);
    free(c);
}

extern "C" int coli_cuda_probe_class(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;
    return prop.integrated ? 1 : 0;
}

extern "C" coli_backend *coli_backend_cuda_open(char *err, size_t errcap) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        snprintf(err, errcap, "cuda: no device"); return NULL;
    }
    coli_cuda *c = (coli_cuda*)calloc(1, sizeof *c);
    if (!c) { snprintf(err, errcap, "cuda: out of memory"); return NULL; }
    c->W  = (cu_w8*)calloc(CUDA_MAX_W, sizeof(cu_w8));
    c->W4 = (cu_w4*)calloc(CUDA_MAX_W, sizeof(cu_w4));
    if (!c->W || !c->W4) { free(c->W); free(c->W4); free(c); snprintf(err, errcap, "cuda: out of memory"); return NULL; }
    c->dev = 0;
    if (cudaSetDevice(c->dev) != cudaSuccess) { free(c->W); free(c->W4); free(c); snprintf(err, errcap, "cuda: setDevice failed"); return NULL; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, c->dev) != cudaSuccess) { free(c->W); free(c->W4); free(c); snprintf(err, errcap, "cuda: getDeviceProperties failed"); return NULL; }
    snprintf(c->devname, sizeof c->devname, "%s", prop.name);
    snprintf(c->memdesc, sizeof c->memdesc, "DEVICE_LOCAL (cudaMalloc)");
    snprintf(c->memdesc2, sizeof c->memdesc2, "DEVICE_LOCAL (cudaMalloc)");
    c->is_integrated = prop.integrated ? 1 : 0;
    if (cudaStreamCreate(&c->stream) != cudaSuccess ||
        cudaStreamCreateWithFlags(&c->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        if (c->stream) cudaStreamDestroy(c->stream);
        free(c->W); free(c->W4); free(c);
        snprintf(err, errcap, "cuda: stream creation failed"); return NULL;
    }

    /* COLI_CUDA_NO_ASYNC_FILL=1 disables async fill (read once, per the task
     * instruction) -- e_slot_fill_async then always takes the synchronous
     * e_slot_fill path, and fill_mode says so, exactly like
     * COLI_VK_NO_ASYNC_FILL on the Vulkan backend. */
    { const char *e = getenv("COLI_CUDA_NO_ASYNC_FILL"); if (e && *e && *e != '0') c->async_disabled = 1; }
    if (c->async_disabled)
        snprintf(c->fill_mode_desc, sizeof c->fill_mode_desc, "synchronous (COLI_CUDA_NO_ASYNC_FILL set)");
    else
        snprintf(c->fill_mode_desc, sizeof c->fill_mode_desc, "async: dedicated copy stream, %d-deep pinned ring", COLI_CUDA_FILL_INFLIGHT);

    coli_backend *be = (coli_backend*)calloc(1, sizeof *be);
    if (!be) { cuda_close(c); snprintf(err, errcap, "cuda: out of memory"); return NULL; }
    be->ctx = c; be->close = cuda_close;
    be->device_name = e_device_name; be->memdesc = e_memdesc; be->memdesc2 = e_memdesc2;
    be->is_integrated = e_is_integrated; be->dot_used = e_dot_used;
    be->upload_w = e_upload_w; be->gemm = e_gemm;
    be->has_i4 = e_has_i4; be->upload_w4 = e_upload_w4;
    be->has_mx = e_has_mx; be->upload_w4_mx = e_upload_w4_mx;
    be->slot_alloc_mx = e_slot_alloc_mx; be->slot_fill = e_slot_fill;
    be->slot_fill_async = e_slot_fill_async; be->slot_fill_wait = e_slot_fill_wait;
    be->fill_mode = e_fill_mode;
    be->upload_begin = e_upload_begin; be->upload_end = e_upload_end;
    be->gemm4 = e_gemm4; be->gemm4_qkv = e_gemm4_qkv;
    be->has_ffn = e_has_ffn; be->ffn4 = e_ffn4;
    be->has_ffn_oai = e_has_ffn_oai; be->ffn4_oai = e_ffn4_oai;
    /* ---- KV cache + decode attention (2026-09-15) ---- */
    be->has_attn = e_has_attn;
    be->kv_init = e_kv_init; be->kv_ready = e_kv_ready; be->kv_bytes = e_kv_bytes;
    be->kv_load = e_kv_load; be->kv_get = e_kv_get; be->kv_ctx = e_kv_ctx;
    be->kv_put = e_kv_put; be->kv_write = e_kv_write;
    be->attn_sinks_upload = e_attn_sinks_upload; be->attn_ex = e_attn_ex;
    /* moe4, moe4_begin, moe4_end and every rope_, qknorm_ and attn_block
     * entry is left NULL -- see the header comment. coli_backend_open's
     * fill_defaults() replaces them with declining stubs. */
    return be;
}
