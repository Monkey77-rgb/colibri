/* vk_backend.c — see vk_backend.h for scope and the bit-exactness exception. */
#define _GNU_SOURCE
#include "vk_backend.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Rows per workgroup. Must be <= MAXTILE in the shaders (8). Chosen by
 * MEASUREMENT: same-binary sweep at 3584x3584 n=512 on an RTX 4070 gave
 * tile=1 31.50 ms, tile=2 24.14 ms, tile=4 22.52 ms, tile=8 23.41 ms. 8 was
 * the initial guess and it is not the best -- 4 is, and 8 is measurably worse,
 * which is why this is a measured constant and not a round number. */
#define COLI_VK_TILE_R 4

/* ------------------------------------------------------------------ profiling
 * Where does a GPU forward pass actually spend its time? The end-to-end number
 * on the 780M (86.9s GPU vs 45.6s CPU) contradicts the kernel microbenchmark
 * (0.78ms GPU vs 3.79ms CPU at n=1), and no amount of reading the code settles
 * which of upload / record / submit-and-wait / download eats the difference.
 *
 * So it gets counted. clock_gettime(MONOTONIC) is ~20ns against microsecond-
 * scale sections, so this is left always-on rather than compiled out -- an
 * instrument you have to remember to enable is an instrument you will not have
 * when you need it. `coli_vk_prof_dump` prints it. */
static struct {
    uint64_t up_ns, dl_ns, rec_ns, sub_ns;   /* wall time per phase */
    /* The phase split said submit+fence was 77.8%, but not WHICH operation was
     * inside it. Three call sites reach the queue in the hot path -- a single
     * gemm, the fused ffn4, and the batched qkv -- and they have different
     * shapes and different fixes, so an aggregate cannot choose between them. */
    uint64_t sub_ns_op[3], dl_ns_op[3];
    uint64_t sub_n_op[3],  dl_n_op[3];
    int      cur_op;                          /* which of the three is running */
    /* COLI_VK_TS=1 (2026-09-10): GPU timestamps inside the fused attention block,
     * one per stage boundary, so the 139 us/layer the block costs on the 4070
     * can be split into its stages. Device time, not wall; read back after the
     * fence. Off by default: query writes are not free. */
    uint64_t ts_ns[9]; uint64_t ts_n;
    uint64_t up_bytes, dl_bytes;             /* transfer volume */
    uint64_t up_n, dl_n, sub_n, gemm_n, ffn_n;
    /* One-time weight staging, counted apart from per-token activation traffic.
     * Folding the two together made a 1.7 GiB model load look like 142 MiB/token
     * of activations -- a positive-looking number that is entirely the wrong
     * quantity. Split at the source rather than subtracted afterwards. */
    uint64_t w_ns, w_bytes, w_n;
    int in_weight_upload;
    /* WHERE THE OUTPUTS ACTUALLY LIVE, and how many bytes went through the
     * staging copy. Without this the profile prints the same shape whether the
     * DEVICE_LOCAL output path ran or silently fell back to host-visible -- and
     * a fallback that reads as a result is the failure this project keeps
     * hitting. out_state: 0 = off, 1 = on, 2 = requested but the allocation
     * fell back. */
    int      out_state;
    char     out_desc[64];
    uint64_t copy_bytes, copy_n;
    /* Which coopmat epilogue actually dispatched. Same reason as out_state: a
     * kernel switch that silently did not switch reads as "the change did
     * nothing", which is the wrong conclusion from the right number. */
    uint64_t coop_n, coop_ds_n;
    uint64_t moe_fused_n;      /* grouped-expert calls that went through the multi-matrix kernel */
} P;

/* SLOT-FILL PHASE BREAKDOWN (2026-09-14). Separate from P above on purpose:
 * upload_device_local's existing counters fold the memcpy into P.w_ns but
 * never time the submit or the fence wait at all -- the 8.7 ms/expert the
 * task measured has nowhere to land in the existing profile. One struct per
 * path (sync / async) because they go through different code and a caller
 * comparing them needs the two kept apart, not summed. */
static struct {
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;   /* n = number of coli_vk_slot_fill calls */
} FSYNC;
static struct {
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;   /* n = number of coli_vk_slot_fill_async calls */
} FASYNC;

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

#define VKERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)
/* One handle per weight matrix in the model, not per benchmark. A dense 36-layer
 * qwen2.5-3b has 36 x 7 + 1 = 253; 64 was sized for the test harness and would
 * have failed at layer 9 of a real model.
 *
 * MoE selective residency (2026-09-01): coli_gpu_upload can now pin the hottest
 * subset of experts in VRAM. A 48-layer x 128-expert model has 18,432 expert
 * matrices, but only the budgeted resident subset gets a handle -- ~69/layer x 48
 * x 3 ~= 9,936 experts plus ~250 dense at a ~10 GiB VRAM budget. 16384 covers that
 * with headroom (struct grows ~2.4 MiB, heap-allocated). The 4070 has no
 * maxMemoryAllocationCount limit (measured 4.3e9); this cap is purely the engine's.
 * Table stays a fixed array rather than a realloc to keep every W4[i] access site
 * untouched. */
#define MAX_W 16384

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; } vkbuf;

/* ASYNC SLOT FILL ring entry (2026-09-14). One HOST_VISIBLE staging buffer big
 * enough for ONE slot's weights+scale, its own command buffer and fence. Fixed
 * array of COLI_VK_FILL_INFLIGHT of these, NOT a byte-addressed ring: reusing
 * slot i means waiting on fence i first, which is exactly the backpressure the
 * task asked for ("when full, wait") and needs no offset bookkeeping. */
#define COLI_VK_FILL_INFLIGHT 8
typedef struct {
    vkbuf buf; void *map; VkDeviceSize cap;
    VkCommandBuffer cmd; VkFence fence;
    int pending;   /* 1 = submitted, fence not yet waited */
} fill_slot;

struct coli_vk {
    VkInstance inst;
    VkPhysicalDevice pdev;
    VkDevice dev;
    VkQueue q;
    uint32_t qfam;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkPipeline pipe4;      /* int4 kernel; NULL when shaders/gemm_i4.spv is absent */
    VkPipeline pipe4s;     /* short-row int4 kernel (LANES=8, OUTS=8), for I <= 1024 at small n; NULL if absent */
    int short_rows_off;    /* COLI_VK_NO_SHORT=1: never use pipe4s (the A/B control) */
    /* multi-matrix int4 kernel (gemm_i4_moe.comp): one dispatch per stage of the
     * grouped-expert call, matrices addressed through a device-address table.
     * Needs VK_KHR_buffer_device_address (has_bda). COLI_MOE_FUSED=1 engages it;
     * 2026-09-08. */
    VkPipeline pipe4m, pipe4ms;
    /* gpt-oss (2026-09-14): gemm_i4_dp with the MXFP4 nibble decode (+ short-row
     * sibling), and silu_mul_q with the swiglu_oai epilogue + biases. NULL when
     * the .spv is absent; coli_vk_has_mx / coli_vk_has_ffn_oai report it. */
    VkPipeline pipe4mx, pipe4mxs, pipe_swq;
    vkbuf asink;           /* [layers][H] attention sinks, uploaded once */
    vkbuf fbias;           /* per-call gate+up expert biases for ffn4_oai */
    int  has_bda;
    PFN_vkGetBufferDeviceAddressKHR pfn_bda;
    vkbuf moe_tab;         /* host-visible address table, two stages at 0 and 4096 */
    VkPipeline pipe4f;     /* int4 dequant-to-float variant, for the comparison */
    VkPipeline pipe_smq;   /* silu*mul + quantize, the op that makes residency possible */
    /* Attention. Present only if shaders/attn_decode.spv was built, exactly like
     * pipe4: absence means the caller keeps using the CPU path, which is also
     * the numerical reference. */
    VkPipeline pipe_attn;
    /* Split-K / flash-decoding-style decode attention (2026-09-13): two
     * pipelines, attn_decode_split.comp + attn_decode_merge.comp, sharing dsl
     * (six storage buffers, same as everything else) but NOT v->pl -- they
     * need a 7th push-constant field (nchunk) that does not fit v->pl's fixed
     * 24-byte range, so they get their own pipeline layout, pl2, built from
     * the SAME dsl. Absence of either pipeline (old shaders, no glslc at build
     * time) means coli_vk_attn_block keeps using pipe_attn alone -- see
     * coli_vk_has_attn_split. */
    VkPipeline pipe_attn_split, pipe_attn_merge;
    VkPipelineLayout pl2;
    /* Partial online-softmax state (m, d, acc[hd]) written by pipe_attn_split
     * and consumed by pipe_attn_merge: [n][H][nchunk][hd+2] floats. Pure
     * device intermediate, like fg/fu/hq/hs/hm -- see mkbuf_dev. Sized to the
     * high-water mark of n*H*nchunk*(hd+2), grown lazily. */
    vkbuf attn_scratch;
    VkDescriptorSet ds_asplit; int ds_asplit_ok;
    /* RoPE+bias and the KV scatter. Same reason as pipe_attn: they exist to
     * remove a SUBMISSION, not because their arithmetic is expensive. */
    VkPipeline pipe_rope, pipe_kvw, pipe_quant;
    VkPipeline pipe_qkn;          /* qknorm.spv: qwen3 per-head q/k RMSNorm inside the block (2026-09-07) */
    vkbuf      qkw;               /* its weights, [layer][q hd | k hd], uploaded once */
    /* The BATCHED int4 GEMM. A second kernel, not a mode of pipe4: pipe4 is a
     * decode kernel at 81% of spec bandwidth and must not be disturbed. See
     * shaders/gemm_i4_tile.comp. NULL when dp4a is absent -- it is dp4a-only,
     * and the scalar path keeps using pipe4 at every n. */
    VkPipeline pipe4t;
    VkPipeline pipe4c;           /* cooperative-matrix (tensor core) batched GEMM */
    /* Same kernel, direct-store epilogue. Only usable because outputs are
     * DEVICE_LOCAL; kept alongside pipe4c so COLI_VK_COOP_DS switches kernels
     * inside ONE binary. */
    VkPipeline pipe4cd;
    int coop_ds;                 /* 1 = prefer pipe4cd when it exists */
    int has_coop;
    int coop_min_n;              /* batch size at or above which pipe4c is used; 0 = off */
    int tile_min_n;              /* batch size at or above which pipe4t is used */
    vkbuf aq, ak, av, ao, am;
    vkbuf rbias, rcs;              /* ALL layers' qkv bias, and the host (c,s) table */
    vkbuf batt, bq8, bs8, bm8;     /* fused block: attention out, then its int8 form */
    VkDescriptorSet ds_blk[3]; int ds_blk_ok;   /* attn, quant, o_proj */
    VkDescriptorSet ds_attn; int ds_attn_ok;
    VkDescriptorSet ds_rope, ds_kvw, ds_qkn; int ds_rk_ok;   /* ds_qkn: qknorm's own set -- a set bound in the open cmd buffer must not be re-written */
    /* RESIDENT KV. One K and one V buffer per layer, DEVICE_LOCAL on a discrete
     * card. kv_ctx mirrors the host cache's stride and must be re-mirrored when
     * the host grows it -- see coli_vk_kv_grow. */
    vkbuf *kvK, *kvV;
    vkbuf  kvstage;                 /* HOST_VISIBLE; rows land here, then copy */
    vkbuf  kvwstage;                /* HOST_VISIBLE; BULK contiguous runs, see coli_vk_kv_write */
    int    kv_layers, kv_slots, kv_heads, kv_ctx, kv_hd, kv_ok;
    int    kv_pend;                 /* rows staged and not yet copied */
    int    kv_pend_off[64];         /* destination row offsets, in floats */
    int    kv_pend_kv[64];          /* 0 = K, 1 = V */
    /* FFN intermediates, resident on the device between the three GEMMs. Grown
     * to the high-water mark like the scratch above and never downloaded. */
    vkbuf fg, fu, hq, hs, hm;
    VkDescriptorSet ds2; int ds2_ok;   /* second set: the smq pipeline's bindings */
    VkDescriptorSet dsf[4]; int dsf_ok; /* the FFN's four dispatches, one set each */
    /* q, k and v: three independent GEMMs over the SAME activation, so they are
     * one command buffer and one fence rather than three of each. Own output
     * buffers because all three results are live at once. */
    vkbuf yq[3];
    VkDescriptorSet dsq[3]; int dsq_ok;
    /* MoE grouped decode (coli_vk_moe4): N experts share ONE activation upload and
     * ONE submit+fence. Own output buffer (all N results live at once, expert-major)
     * and own descriptor pool so the up-to-N*4 sets do not contend with the 64-set
     * shared dpool. Allocated lazily on first moe4 call. */
    vkbuf ymoe;
    VkDescriptorPool moe_pool;
    VkDescriptorSet *moe_ds; int moe_ds_cap;
    VkDescriptorPool dpool;
    VkFence fence;
    VkQueryPool tsq; float ts_period; int ts_on;   /* COLI_VK_TS stage timestamps (attn_block) */
    char devname[256];
    char memdesc[160];
    char memdesc2[64];
    int  integrated;
    /* Opt-in: try DEVICE_LOCAL for weights even on an integrated GPU. Default
     * off, because on a UMA part it is NOT self-evidently a win and the old
     * code asserted it was self-evidently a loss -- both are claims. See
     * coli_vk_upload_w4 for what the 780M heap table actually says. */
    int  want_device_local;
    /* The other half of the switch, and the reason it exists: with an opt-in
     * flag alone, the HOST_VISIBLE fallback below is unreachable on a discrete
     * GPU, so the memdesc2 instrumentation could only ever print one of its two
     * values -- a control that cannot produce the opposite result. =0 forces the
     * fallback on ANY device, which is what makes memdesc2 falsifiable and lets
     * both arms of an A/B run from one binary. */
    int  force_host_visible;
    int  has_dot;          /* VK_KHR_shader_integer_dot_product available AND enabled */
    /* SUBGROUP WIDTH. attn_decode.comp is written for 32-lane subgroups and
     * says so in a #define; nothing used to check it. NVIDIA reports 32 and the
     * kernel was validated there, so the assumption held everywhere it was
     * tested and nowhere else. RADV on gfx1103 reports 64. */
    uint32_t sg_size;      /* driver's native subgroupSize */
    uint32_t sg_min, sg_max;
    int      sg_ctl;       /* VK_EXT_subgroup_size_control enabled: size is pinnable */
    int      sg_pinned;    /* the attn pipeline was actually created pinned to 32 */
    int  dot_used;         /* the DP4a spv actually loaded and built a pipeline */
    int  tile;             /* rows per workgroup, resolved once at init */
    VkPhysicalDeviceMemoryProperties memprops;
    struct { vkbuf w, ws; int64_t I, O; int used; } W[MAX_W];
    int nw;
    /* int4 weights, in their own table. Kept separate rather than tagged inside
     * W[] so a caller cannot hand an int4 handle to the int8 GEMM and have it
     * read half a matrix -- the type confusion would produce plausible numbers,
     * which is the failure mode worth designing out. */
    struct { vkbuf w, ws; int64_t I, O; int used; int mx; int dl; } W4[MAX_W];   /* dl: DEVICE_LOCAL buffers (slot refill path) */   /* mx: MXFP4-tagged (gpt-oss), see coli_vk_upload_w4_mx */
    int nw4;
    /* Persistent scratch, grown to the high-water mark and reused. Allocating
     * and destroying these per call cost more than the kernel ran for. */
    vkbuf xb, xs, xm, yb;
    VkDescriptorSet ds;
    int ds_ok;
    int has_transfer;      /* device-local weights need a copy queue */
    /* GEMM OUTPUTS IN VRAM, read back through one staging buffer. See mkbuf_out.
     * dlstage is HOST_CACHED and TRANSFER_DST only; out_dev says whether the
     * mechanism is active at all, so every download site has one branch and the
     * old path stays reachable from the same binary. */
    vkbuf dlstage;
    int   out_dev;
    char  memdesc3[64];

    /* BATCHED WEIGHT UPLOAD (2026-09-13). Measured on the 4070 the same day:
     * the 30B's 11,115 int4 matrices (10.7 GiB) took 14.2-16.3 s to upload =
     * ~0.7 GB/s, an order of magnitude under PCIe, because upload_device_local
     * did, PER MATRIX HALF (weights, then scales): create+allocate a staging
     * buffer, map+memcpy, record one copy, submit, wait on the fence, free the
     * staging buffer -- 22,230 submit/fence round trips. Between
     * coli_vk_upload_begin() and coli_vk_upload_end() the same function instead
     * memcpy's into ONE persistent host-visible ring and records the copy into
     * the open command buffer; the batch is submitted and waited on only when
     * the ring is full or at end(). Nothing about the destination buffers or
     * the dispatch path changes, so the oracle (greedy text md5 + nll1) must be
     * identical. Outside a begin/end window every caller sees the old path.
     * COLI_VK_UPLOAD_BATCH_MB sets the ring (default 256; 0 = old path, the
     * control). A matrix half larger than the ring falls back to the old path
     * after flushing what is pending. */
    vkbuf        upring;
    void        *upring_map;
    VkDeviceSize upring_cap, upring_off;
    int          up_batch, up_pending, up_failed, up_flushes;

    /* ASYNC SLOT FILL (2026-09-14). See coli_vk_slot_fill_async/_wait in
     * vk_backend.h for the contract and the comment at queue discovery above
     * for how xfam/fq/fill_mode get set. fpool is a SEPARATE command pool only
     * when fq's family differs from v->qfam's (fill_mode==1) -- a command pool
     * is bound to one queue family at creation and v->pool was created for
     * qfam; reusing it for xfam's buffers would be invalid. */
    uint32_t        xfam;            /* transfer-only family, UINT32_MAX = none */
    VkQueue         fq;              /* queue fills submit on; == v->q when fill_mode==0 */
    int             fill_mode;       /* 0 sync-fallback, 1 dedicated transfer family, 2 2nd queue/same family */
    char            fill_mode_desc[80];
    VkCommandPool   fpool;           /* == v->pool when fill_mode!=1 */
    fill_slot       fslots[COLI_VK_FILL_INFLIGHT];
    int             fslot_next;
    VkDeviceSize    fslot_cap;        /* current per-slot staging capacity */
};

static uint32_t find_mem(coli_vk *v, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < v->memprops.memoryTypeCount; i++)
        if ((bits & (1u<<i)) && (v->memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

/* HOST_VISIBLE|HOST_COHERENT for everything. On a discrete card that means the
 * GPU reads over PCIe instead of from VRAM, which is slower -- but on the actual
 * target (Legion 780M, laptop Vega, any APU) memory is UNIFIED and there is no
 * device-local copy to make. Staging buffers would be pure overhead there.
 * A discrete-GPU path would add a device-local copy for the weights; not done. */
/* Reports what the chosen memory type actually IS, so "the weights are in system
 * RAM" is a measurement rather than an inference from the allocation flags. */
static void describe_mem(coli_vk *v, uint32_t mt, char *out, size_t cap) {
    VkMemoryPropertyFlags f = v->memprops.memoryTypes[mt].propertyFlags;
    uint32_t heap = v->memprops.memoryTypes[mt].heapIndex;
    snprintf(out,cap,"type%u heap%u %.1f GiB [%s%s%s%s]", mt, heap,
        (double)v->memprops.memoryHeaps[heap].size/1073741824.0,
        (f&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)?"DEVICE_LOCAL ":"",
        (f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)?"HOST_VISIBLE ":"",
        (f&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)?"HOST_COHERENT ":"",
        (f&VK_MEMORY_PROPERTY_HOST_CACHED_BIT)?"HOST_CACHED":"");
}

/* EXCLUSIVE-vs-CONCURRENT, and why a slot buffer needs the choice at all
 * (2026-09-14). A buffer written by a copy on one queue and read by a
 * dispatch on another crosses a QUEUE FAMILY boundary whenever fill_mode==1
 * (the dedicated transfer family is a different family index from qfam).
 * Vulkan's EXCLUSIVE sharing mode requires an explicit ownership-transfer
 * barrier (srcQueueFamilyIndex/dstQueueFamilyIndex) before such a resource is
 * used by the new family, REGARDLESS of host-side fence synchronization --
 * ownership and memory-visibility are separate guarantees, and a fence wait
 * only buys the second one. CONCURRENT sharing mode removes the requirement
 * entirely (the spec allows any queue in the listed families to use the
 * resource with no transfer), at a cost the driver is free to charge only on
 * the actual cross-family access pattern this project uses rarely (one
 * expert-slot refill, not a hot per-token buffer). That is the one chosen
 * here rather than a manual barrier: fewer failure modes for a fixed, small
 * set of buffers, and it is the ALTERNATIVE the Vulkan spec itself names for
 * exactly this situation.
 *
 * fill_mode==2 (second queue, SAME family as compute) needs neither: ownership
 * transfer is a cross-FAMILY concept only, and coli_vk_slot_fill_wait's fence
 * wait is exactly the host-observed-completion step that Vulkan guarantees
 * makes prior writes available to whatever the host submits afterward -- see
 * the comment on coli_vk_slot_fill_wait below. mkbuf_flags (no sharing args)
 * keeps every existing EXCLUSIVE caller untouched; mkbuf_flags_ex is the one
 * new entry point slot_alloc_mx calls when it needs to choose. */
static int mkbuf_flags_ex(coli_vk *v, VkDeviceSize sz, vkbuf *b,
                          VkMemoryPropertyFlags want, VkBufferUsageFlags usage,
                          VkSharingMode sharing, const uint32_t *fams, uint32_t nfam) {
    if (sz == 0) sz = 4;
    VkBufferCreateInfo bi = { .sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=sz, .usage=usage | (v->has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
        .sharingMode=sharing,
        .queueFamilyIndexCount = (sharing==VK_SHARING_MODE_CONCURRENT) ? nfam : 0,
        .pQueueFamilyIndices   = (sharing==VK_SHARING_MODE_CONCURRENT) ? fams : NULL };
    if (vkCreateBuffer(v->dev,&bi,NULL,&b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(v->dev,b->buf,&mr);
    uint32_t mt = find_mem(v, mr.memoryTypeBits, want);
    if (mt == UINT32_MAX) { vkDestroyBuffer(v->dev,b->buf,NULL); b->buf=0; return 0; }
    /* every buffer gets a device address when the feature is on: the multi-matrix
     * kernel reads expert weights AND scratch slices through the table */
    VkMemoryAllocateFlagsInfo afi = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
    VkMemoryAllocateInfo ai = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = v->has_bda ? &afi : NULL, .allocationSize=mr.size, .memoryTypeIndex=mt };
    if (vkAllocateMemory(v->dev,&ai,NULL,&b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev,b->buf,NULL); b->buf=0; return 0; }
    vkBindBufferMemory(v->dev,b->buf,b->mem,0);
    b->size = sz;
    return 1;
}

static int mkbuf_flags(coli_vk *v, VkDeviceSize sz, vkbuf *b,
                       VkMemoryPropertyFlags want, VkBufferUsageFlags usage) {
    return mkbuf_flags_ex(v,sz,b,want,usage,VK_SHARING_MODE_EXCLUSIVE,NULL,0);
}

/* DOWNLOAD TARGETS WANT HOST_CACHED. mkbuf below asks for HOST_VISIBLE |
 * HOST_COHERENT, which on a discrete NVIDIA card is write-combined: excellent to
 * write, terrible to READ, because CPU reads of WC memory bypass the cache and
 * come back one uncached transaction at a time.
 *
 * Measured 2026-08-23, 683-token prefill of an 8B model on an RTX 4070: the
 * download phase moved 1195.7 MiB in 3519 ms = ~340 MB/s, and was 39.4% of the
 * whole prefill. The upload phase moved 384 MiB in 13.6 ms = ~28 GB/s over the
 * same bus. Two orders of magnitude apart, same PCIe link, same buffers -- the
 * asymmetry IS the write-combining.
 *
 * Asking for HOST_CACHED as well gets a cached mapping where reads go through
 * the cache hierarchy. It is a preference, not a requirement: if no such type
 * exists (common on UMA parts, where it does not matter because there is no bus
 * to cross) this falls back to exactly what mkbuf would have done. */
static int mkbuf(coli_vk *v, VkDeviceSize sz, vkbuf *b);
static void freebuf(coli_vk *v, vkbuf *b);

static int mkbuf_dl(coli_vk *v, VkDeviceSize sz, vkbuf *b) {
    /* COLI_VK_NO_CACHED_DL=1 forces the old uncoherent-read path, so the A/B has
     * a control that can actually fail. Without it "downloads got faster" is a
     * cross-time claim against a rebuilt binary. */
    static int off = -1;
    if (off < 0) { const char *e = getenv("COLI_VK_NO_CACHED_DL"); off = (e && *e && *e!='0') ? 1 : 0; }
    if (off) return mkbuf(v, sz, b);
    if (mkbuf_flags(v, sz, b,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return 1;
    return mkbuf(v, sz, b);
}

/* The coopmat kernel writes whole 16x16 blocks, so the output buffer is padded
 * up to a multiple of its 64-row tile. Costs at most 63 rows of float per
 * matrix; buys the removal of a per-subgroup shared staging array AND the
 * divergent-barrier edge path that a partially-outside block would otherwise
 * need. Measured with the edge path in place: 7.76 s against 2.72 s without. */
/* THE COOPMAT TILE, IN ONE PLACE. These MUST equal TSR and TSC in
 * shaders/gemm_i4_coop.comp. They were written out longhand at two dispatch
 * sites and in the row padding, which is three chances to change the shader and
 * not the host -- and that exact mismatch (host 128, shader 256) was caught by
 * luck once already. A wrong grid computes the wrong outputs silently. */
#define COOP_TSR 32
#define COOP_TSC 128
/* 32x128 measured, not chosen. Interleaved rebuild-and-run x3, RTX 4070, quiet
 * GPU, 683-token prefill of ARIAofWebsec v6, submit+fence:
 *   32x64  822.2 / 788.5 / 820.0 ms      (the shipped shape before this)
 *   32x128 682.8 / 646.1 / 680.8 ms      1.20x, non-overlapping
 *   64x64  756.3 ms      64x128 778.4 ms      32x256 786.1 ms
 * TF-NLL 2.7433 in every one, so the shape is a pure throughput change.
 * 64x128 needs 58,368 bytes of shared against this device's REPORTED 49,152 and
 * created a pipeline anyway -- NVIDIA does not enforce the limit it advertises.
 * It is therefore out of spec and not shippable, and note what that means for
 * the pipeline-failure print added on 2026-08-24: it cannot catch this. */

#define COOP_ROW_PAD(n) (((n) + (COOP_TSR-1)) & ~(COOP_TSR-1))

static int ensure_dl(coli_vk *v, vkbuf *b, VkDeviceSize need) {
    if (b->buf && b->size >= need) return 1;
    freebuf(v,b);
    return mkbuf_dl(v,need,b);
}

/* PURE DEVICE INTERMEDIATES BELONG IN VRAM. mkbuf hands out HOST_VISIBLE memory,
 * which on a discrete card is SYSTEM RAM reached over PCIe -- the 4070 reports
 * its HOST_VISIBLE|HOST_COHERENT type on heap1 (22.8 GiB), not on the 12 GiB
 * DEVICE_LOCAL heap. Weights were already given the staged DEVICE_LOCAL
 * treatment; the FFN's intermediates were not, so fg/fu/hq/hs/hm -- which the
 * CPU never touches -- were being written and re-read across the bus.
 *
 * How this was found, 2026-08-23: ffn4 took 145.1 ms per call at n=683 with the
 * decode kernel and 147.9 ms with a completely different tiled kernel. Two
 * kernels with unrelated memory access patterns landing within 2% of each other
 * is not a coincidence about the kernels; it says neither one is the limit.
 * Meanwhile the GPU sat at 100% utilisation, 2790-2805 MHz, drawing 95 W of a
 * 200 W budget -- full occupancy, half the power, which is stalling, not
 * computing.
 *
 * Falls back to mkbuf when no DEVICE_LOCAL-only type exists, which is the normal
 * case on UMA parts where the distinction does not exist. */
static int mkbuf_dev(coli_vk *v, VkDeviceSize sz, vkbuf *b) {
    static int off = -1;
    if (off < 0) { const char *e = getenv("COLI_VK_NO_DEV_SCRATCH"); off = (e && *e && *e!='0') ? 1 : 0; }
    if (off || v->integrated) return mkbuf(v, sz, b);
    if (mkbuf_flags(v, sz, b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return 1;
    return mkbuf(v, sz, b);
}

static int ensure_dev(coli_vk *v, vkbuf *b, VkDeviceSize need) {
    if (b->buf && b->size >= need) return 1;
    freebuf(v,b);
    return mkbuf_dev(v,need,b);
}

/* GEMM OUTPUTS ARE STILL HOST-VISIBLE, AND THAT IS NOW THE CONSTRAINT.
 *
 * mkbuf_dl put the output in HOST_CACHED system memory, which fixed the 340 MB/s
 * readback. It did not change WHERE the shader writes: a kernel on a discrete
 * card still pushes every result across PCIe as it produces it, and the
 * cooperative-matrix kernel cannot write there directly at all -- coopMatStore
 * scatters by lane, and measured 2026-08-24 that was 9.62 s against 2.59 s, so
 * the kernel stages through a shared array instead. That array is 8 KiB of the
 * 48 KiB budget and is what caps the tile at 32x64.
 *
 * So: allocate the output DEVICE_LOCAL, and read it back with a vkCmdCopyBuffer
 * into one HOST_CACHED staging buffer recorded into the SAME command buffer as
 * the GEMM. No extra submission, no extra fence -- the copy is a transfer the
 * GPU does at VRAM speed and the CPU reads from cached system memory.
 *
 * On UMA this is a pure loss (one extra full copy of the result through the same
 * memory, to reach memory the CPU could already read), so it is off on
 * integrated parts, exactly like mkbuf_dev. COLI_VK_NO_DEV_OUT=1 forces it off
 * on a discrete card too, so the A/B has an arm that can fail. */
static int mkbuf_out(coli_vk *v, VkDeviceSize sz, vkbuf *b) {
    if (!v->out_dev) { P.out_state = 0; return mkbuf_dl(v, sz, b); }
    if (mkbuf_flags(v, sz, b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) { if (P.out_state != 2) P.out_state = 1; return 1; }
    P.out_state = 2;
    return mkbuf_dl(v, sz, b);
}

static int ensure_out(coli_vk *v, vkbuf *b, VkDeviceSize need) {
    if (b->buf && b->size >= need) return 1;
    freebuf(v,b);
    return mkbuf_out(v,need,b);
}

/* The one staging buffer every readback lands in. Grown to the high-water mark
 * like the rest of the scratch. TRANSFER_DST only -- no shader ever binds it. */
static int ensure_stage(coli_vk *v, VkDeviceSize need) {
    if (!v->out_dev) return 1;
    if (v->dlstage.buf && v->dlstage.size >= need) return 1;
    freebuf(v,&v->dlstage);
    if (!mkbuf_flags(v, need, &v->dlstage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        /* No cached host type: fall back to the plain one rather than failing,
         * and record it, because "the staging buffer is write-combined" would
         * otherwise present as an unexplained slow readback. */
        if (!mkbuf_flags(v, need, &v->dlstage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return 0;
        snprintf(v->memdesc3,sizeof v->memdesc3,"stage: HOST_COHERENT (uncached)");
    } else if (!v->memdesc3[0]) {
        snprintf(v->memdesc3,sizeof v->memdesc3,"stage: HOST_CACHED");
    }
    snprintf(P.out_desc,sizeof P.out_desc,"%s",v->memdesc3);
    return 1;
}

/* Recorded into the GEMM's own command buffer, after the last dispatch. The
 * barrier is SHADER_WRITE -> TRANSFER_READ, not the SHADER_READ that
 * record_barrier issues: a compute-to-compute barrier does not order a
 * subsequent transfer, and the result would be a race that reads correct data
 * most of the time. */
static void record_copy_out(coli_vk *v, vkbuf *src, VkDeviceSize dstoff, VkDeviceSize bytes) {
    VkMemoryBarrier mb = { .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT };
    vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkBufferCopy bc = { .srcOffset=0, .dstOffset=dstoff, .size=bytes };
    vkCmdCopyBuffer(v->cmd, src->buf, v->dlstage.buf, 1, &bc);
    P.copy_n++; P.copy_bytes += bytes;
}

static int mkbuf(coli_vk *v, VkDeviceSize sz, vkbuf *b) {
    if (sz == 0) sz = 4;
    VkBufferCreateInfo bi = { .sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=sz, .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (v->has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
        .sharingMode=VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(v->dev,&bi,NULL,&b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(v->dev,b->buf,&mr);
    uint32_t mt = find_mem(v, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { vkDestroyBuffer(v->dev,b->buf,NULL); return 0; }
    if (!v->memdesc[0]) describe_mem(v, mt, v->memdesc, sizeof v->memdesc);
    VkMemoryAllocateFlagsInfo afi = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
    VkMemoryAllocateInfo ai = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = v->has_bda ? &afi : NULL, .allocationSize=mr.size, .memoryTypeIndex=mt };
    if (vkAllocateMemory(v->dev,&ai,NULL,&b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev,b->buf,NULL); return 0; }
    vkBindBufferMemory(v->dev,b->buf,b->mem,0);
    b->size = sz;
    return 1;
}
static void freebuf(coli_vk *v, vkbuf *b) {
    if (b->buf) vkDestroyBuffer(v->dev,b->buf,NULL);
    if (b->mem) vkFreeMemory(v->dev,b->mem,NULL);
    memset(b,0,sizeof *b);
}
/* Reallocate only when the request outgrows what is already there. Steady state
 * for a decoding server is a fixed n, so after the first call this never
 * allocates again. */
static int ensure(coli_vk *v, vkbuf *b, VkDeviceSize need) {
    if (b->buf && b->size >= need) return 1;
    freebuf(v,b);
    return mkbuf(v,need,b);
}

static int upload(coli_vk *v, vkbuf *b, const void *src, size_t n) {
    uint64_t t0 = now_ns();
    void *p; if (vkMapMemory(v->dev,b->mem,0,n,0,&p) != VK_SUCCESS) return 0;
    memcpy(p,src,n); vkUnmapMemory(v->dev,b->mem);
    if (P.in_weight_upload) { P.w_ns += now_ns()-t0; P.w_bytes += n; P.w_n++; }
    else                     { P.up_ns += now_ns()-t0; P.up_bytes += n; P.up_n++; }
    return 1;
}
static int download(coli_vk *v, vkbuf *b, void *dst, size_t n) {
    uint64_t t0 = now_ns();
    void *p; if (vkMapMemory(v->dev,b->mem,0,n,0,&p) != VK_SUCCESS) return 0;
    memcpy(dst,p,n); vkUnmapMemory(v->dev,b->mem);
    { uint64_t d = now_ns()-t0; P.dl_ns += d; P.dl_bytes += n; P.dl_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.dl_ns_op[oi]+=d; P.dl_n_op[oi]++; } }
    return 1;
}

/* Read back what record_copy_out already moved into the staging buffer, or fall
 * through to reading the output buffer directly when the mechanism is off.
 * Timed on the same counters as download() so the two arms are comparable -- an
 * A/B where one arm is not instrumented measures the instrumentation. */
/* Read n bytes of a host-visible buffer starting at srcoff. For the one reader
 * (moe4 without out_dev) whose slices live at offsets in the SOURCE buffer.
 * UNVERIFIED as of 2026-09-07 00:55 -- written after the bisect, not yet run. */
static int download_at(coli_vk *v, vkbuf *src, VkDeviceSize srcoff, void *dst, size_t n) {
    uint64_t t0 = now_ns();
    void *p; if (vkMapMemory(v->dev,src->mem,0,VK_WHOLE_SIZE,0,&p) != VK_SUCCESS) return 0;
    memcpy(dst,(const char*)p + srcoff,n); vkUnmapMemory(v->dev,src->mem);
    { uint64_t d = now_ns()-t0; P.dl_ns += d; P.dl_bytes += n; P.dl_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.dl_ns_op[oi]+=d; P.dl_n_op[oi]++; } }
    return 1;
}

static int download_out(coli_vk *v, vkbuf *src, VkDeviceSize off, void *dst, size_t n) {
    /* `off` is the offset in the STAGING buffer, not in src: gemm4_qkv reads three
     * separate source buffers (yq[j], each from 0) that record_copy_out packed at
     * qoff[j]. Honoring `off` here for the host-visible arm therefore returns the
     * wrong bytes -- done briefly 2026-09-07 00:02, bisected the same night
     * (X1/X3 garbage, X2 with COLI_VK_DEV_OUT=1 correct). Readers whose SOURCE is
     * sliced use download_at() below. */
    if (!v->out_dev) return download(v, src, dst, n);
    uint64_t t0 = now_ns();
    /* Mapped at 0 and indexed, not mapped at `off`: vkMapMemory only guarantees
     * minMemoryMapAlignment on the pointer it returns for offset 0, and a
     * mid-buffer map is one more thing that would work on this driver and not
     * the next. */
    void *p; if (vkMapMemory(v->dev,v->dlstage.mem,0,VK_WHOLE_SIZE,0,&p) != VK_SUCCESS) return 0;
    memcpy(dst,(const char*)p + off,n); vkUnmapMemory(v->dev,v->dlstage.mem);
    { uint64_t d = now_ns()-t0; P.dl_ns += d; P.dl_bytes += n; P.dl_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.dl_ns_op[oi]+=d; P.dl_n_op[oi]++; } }
    return 1;
}

/* Read a .spv file and make a shader module. Returns 0 and leaves *out
 * untouched when the file is missing -- absence is a normal answer here, because
 * the int4 shader is optional and a build that never ran `make vk` still has to
 * work with the int8 one. */
static int load_module(coli_vk *v, const char *path, VkShaderModule *out) {
    FILE *f = fopen(path,"rb");
    if (!f) return 0;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    if (sz<=0 || sz%4) { fclose(f); return 0; }
    uint32_t *code = malloc((size_t)sz);
    if (!code) { fclose(f); return 0; }
    if (fread(code,1,(size_t)sz,f)!=(size_t)sz) { fclose(f); free(code); return 0; }
    fclose(f);
    VkShaderModuleCreateInfo smi = { .sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=(size_t)sz, .pCode=code };
    int ok = vkCreateShaderModule(v->dev,&smi,NULL,out)==VK_SUCCESS;
    free(code);
    return ok;
}

coli_vk *coli_vk_init(const char *spv_path, char *err, size_t errcap) {
    coli_vk *v = (coli_vk*)calloc(1,sizeof *v);
    VkApplicationInfo app = { .sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="colibri", .apiVersion=VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app };
    if (vkCreateInstance(&ici,NULL,&v->inst) != VK_SUCCESS) { VKERR("no Vulkan instance (driver missing?)"); free(v); return NULL; }

    uint32_t nd=0; vkEnumeratePhysicalDevices(v->inst,&nd,NULL);
    if (!nd) { VKERR("no Vulkan device"); vkDestroyInstance(v->inst,NULL); free(v); return NULL; }
    VkPhysicalDevice *devs = malloc(nd*sizeof *devs);
    vkEnumeratePhysicalDevices(v->inst,&nd,devs);
    /* Prefer an integrated GPU: on this project's targets that is the ONLY GPU
     * and it shares memory with the CPU. Falls back to whatever exists. */
    int pick=-1;
    for (uint32_t i=0;i<nd;i++){ VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(devs[i],&pr);
        if (pr.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { pick=(int)i; break; } }
    if (pick<0) pick=0;
    v->pdev = devs[pick];
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(v->pdev,&pr);
    v->ts_period = pr.limits.timestampPeriod;
    { const char *e = getenv("COLI_VK_TS"); v->ts_on = (e && *e && *e!='0') ? 1 : 0; }
    snprintf(v->devname,sizeof v->devname,"%s",pr.deviceName);

    /* Native subgroup width, and whether it can be pinned. Core 1.1 for the
     * width; the min/max come from VK_EXT_subgroup_size_control when present. */
    { VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgp = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT };
      VkPhysicalDeviceSubgroupProperties sgprops = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext=&sgp };
      VkPhysicalDeviceProperties2 p2 = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext=&sgprops };
      vkGetPhysicalDeviceProperties2(v->pdev,&p2);
      v->sg_size = sgprops.subgroupSize;
      v->sg_min  = sgp.minSubgroupSize;
      v->sg_max  = sgp.maxSubgroupSize; }
    v->integrated = (pr.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    /* Tri-state: unset = per-device default, 1 = force DEVICE_LOCAL, 0 = force
     * the HOST_VISIBLE fallback. Unset must keep the old behaviour exactly. */
    { const char *e = getenv("COLI_VK_DEVICE_LOCAL");
      v->want_device_local  = (e && *e && *e!='0');
      v->force_host_visible = (e && *e && *e=='0'); }
    free(devs);
    vkGetPhysicalDeviceMemoryProperties(v->pdev,&v->memprops);

    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&nq,NULL);
    VkQueueFamilyProperties *qs = malloc(nq*sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&nq,qs);
    v->qfam = UINT32_MAX;
    for (uint32_t i=0;i<nq;i++) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { v->qfam=i; break; }
    if (v->qfam==UINT32_MAX) { free(qs); VKERR("no compute queue"); goto fail; }

    /* ASYNC SLOT FILL (2026-09-14) -- queue discovery only, nothing submitted
     * yet. A queue the compute dispatch never touches lets a staged copy into
     * an expert slot run WHILE the compute queue is busy with something else,
     * which is the whole point: coli_vk_slot_fill's 8.7 ms/expert is a
     * synchronous stall on THIS queue, not a bandwidth cost (13 MB over PCIe
     * 4.0 x16 is <1 ms). Preference order, decided ONCE here because a
     * physical device's queue families are fixed for its lifetime:
     *   1. a TRANSFER-only family -- queueFlags has TRANSFER and NEITHER
     *      GRAPHICS NOR COMPUTE. The 4070 exposes exactly one of these (2
     *      queues, TRANSFER|SPARSE_BINDING only, measured 2026-09-14 via this
     *      same enumeration) -- a dedicated DMA engine, genuine hardware
     *      overlap with the compute queue.
     *   2. a second queue inside v->qfam itself (queueCount > 1). Still two
     *      independent queues the driver can schedule concurrently, for a
     *      device with no separate transfer engine but more than one compute
     *      queue. Untested on the Legion's 780M as of this writing -- RADV
     *      may expose either shape or neither.
     *   3. neither: coli_vk_slot_fill_async falls back to calling the
     *      synchronous coli_vk_slot_fill directly. coli_vk_fill_mode() reports
     *      which of the three a running process actually got, so a caller
     *      never has to guess from the device name. */
    v->xfam = UINT32_MAX;
    for (uint32_t i=0;i<nq;i++) {
        VkQueueFlags f = qs[i].queueFlags;
        if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT) && !(f & VK_QUEUE_COMPUTE_BIT)) {
            v->xfam = i; break;
        }
    }
    uint32_t qfam_queue_count = qs[v->qfam].queueCount;
    free(qs);
    if (v->xfam != UINT32_MAX)      v->fill_mode = 1;
    else if (qfam_queue_count > 1)  v->fill_mode = 2;
    else                             v->fill_mode = 0;
    { const char *e = getenv("COLI_VK_NO_ASYNC_FILL"); if (e && *e && *e!='0') v->fill_mode = 0; }

    /* VK_KHR_shader_integer_dot_product = DP4a: four int8 products and an
     * accumulate in ONE instruction. This kernel is instruction-bound, not
     * bandwidth-bound -- measured 2026-08-20 on an RTX 4070, ~6.58 G MAC in
     * 20.1 ms at n=512 is ~2.0 T instr/s against ~14.7 T/s of int32 issue,
     * i.e. ~13% of throughput, with ~6 scalar ops per MAC. Widening loads to
     * uvec4 changed nothing (22.08 -> 22.03 ms), which is what ruled the
     * memory system out.
     *
     * Enabled ONLY if the device advertises it, and the shader that uses it is
     * a SEPARATE spv with the scalar one kept as fallback: this has to stay
     * loadable on the Legion's gfx1103/RADV, and "the fast path exists" is not
     * the same claim as "the fast path is supported here". */
    uint32_t nx = 0; v->has_dot = 0;
    vkEnumerateDeviceExtensionProperties(v->pdev, NULL, &nx, NULL);
    if (nx) {
        VkExtensionProperties *xp = malloc(nx * sizeof *xp);
        if (xp) {
            vkEnumerateDeviceExtensionProperties(v->pdev, NULL, &nx, xp);
            for (uint32_t i = 0; i < nx; i++)
                if (!strcmp(xp[i].extensionName, "VK_KHR_shader_integer_dot_product")) v->has_dot = 1;
            for (uint32_t i = 0; i < nx; i++)
                if (!strcmp(xp[i].extensionName, "VK_KHR_buffer_device_address")) v->has_bda = 1;
            free(xp);
        }
    }
    { const char *e = getenv("COLI_VK_NO_DOT"); if (e && *e && *e!='0') v->has_dot = 0; }
    { const char *e = getenv("COLI_VK_NO_BDA"); if (e && *e && *e!='0') v->has_bda = 0; }
    v->tile = COLI_VK_TILE_R;
    /* OFF BY DEFAULT, because it is currently SLOWER. Set to INT_MAX so no batch
     * size selects it; COLI_VK_TILE_MIN_N=<n> turns it on for measurement.
     *
     * Measured 2026-08-23, ARIAofWebsec v6, 682-token prefill, RTX 4070:
     * decode kernel 11.6 s, this kernel 40.3 s. Shipping it on would be a 3.5x
     * prefill regression, so it ships off until it wins. It is kept because the
     * work is real and the remaining limit is now quantified rather than
     * guessed -- see the shader header. */
    /* ON at 64, which is one full row tile (TSR). Below that the 64-row tile is
     * mostly padding and the decode kernel -- which is at 81% of spec bandwidth
     * at n=1 and must not be disturbed -- wins outright.
     *
     * Measured 2026-08-24 on a quiet GPU, 683-token prefill, ARIAofWebsec v6,
     * RTX 4070: decode kernel 4.31/4.30 s, this 3.62/3.60 s, and ffn4 55.8 ms
     * -> 33.4 ms per call. Note it LOST until the memory-placement fixes landed:
     * on the same box the day before it was at parity, because both kernels were
     * stalled on PCIe and the kernel was not what either was waiting for. */
    v->tile_min_n = 64;
    /* Tensor-core path, ON at 32 rows (one TSR tile). Measured 2026-08-24,
     * 683-token prefill, ARIAofWebsec v6, RTX 4070, quiet GPU: dp4a-tiled
     * 182 tok/s, this 264 tok/s. Correct: TF-NLL 2.7433 against the dp4a
     * kernel's 2.7458 and the CPU path's 2.7441 -- the fp16 folding rounds where
     * the integer path does not, which was predicted rather than discovered.
     * COLI_VK_COOP_MIN_N=0 turns it off for A/B on the same binary. */
    v->coop_min_n = COOP_TSR;
    { const char *e = getenv("COLI_VK_COOP_MIN_N");
      if (e && *e) { int t = atoi(e); if (t >= 0) v->coop_min_n = t; } }
    { const char *e = getenv("COLI_VK_TILE_MIN_N");
      if (e && *e) { int t = atoi(e); if (t >= 1) v->tile_min_n = t; } }
    { const char *e = getenv("COLI_VK_TILE_R");
      if (e && *e) { int t = atoi(e); if (t >= 1 && t <= 4) v->tile = t; } }  /* <= MAXTILE */

    /* GEMM outputs in VRAM with a staged readback. See mkbuf_out.
     *
     * OFF BY DEFAULT, BECAUSE IT MEASURED AS A LOSS. RTX 4070, 683-token
     * prefill of ARIAofWebsec v6, quiet GPU, interleaved 2026-08-25:
     * submit+fence 852.0/871.5 ms with it on against 773.0/787.1 ms with it
     * off. The readback it was meant to fix was already fast -- 1529.4 MiB in
     * ~56 ms either way, ~27 GB/s, because mkbuf_dl had made it HOST_CACHED --
     * so the VRAM->system copy is added cost the shader write-side does not buy
     * back. It stays in the tree because it is the only way to run the
     * direct-store epilogue at all, and because a refuted idea deleted is an
     * idea someone re-proposes next month.
     *
     * COLI_VK_DEV_OUT=1 turns it on. Never on an integrated part: there is no
     * bus to cross, so the copy would be pure cost. */
    { const char *e = getenv("COLI_VK_DEV_OUT");
      v->out_dev = (!v->integrated && e && *e && *e!='0') ? 1 : 0; }
    /* The direct-store coopmat epilogue. Requires out_dev, and ALSO measured as
     * a loss once it had one: 882.3/875.7 ms against 846.0/836.3 ms for the
     * shared-staging epilogue on the same DEVICE_LOCAL outputs, same binary,
     * interleaved, identical NLL. So coopMatStore's scatter is worse than
     * shared-then-coalesced even in VRAM -- the 9.62 s vs 2.72 s measured on
     * host-visible memory was PCIe, and the inference "therefore VRAM will make
     * the direct store win" did not survive being tested. */
    { const char *e = getenv("COLI_VK_COOP_DS");
      v->coop_ds = (v->out_dev && e && *e && *e!='0') ? 1 : 0; }

    const char *devexts[8]; uint32_t nexts = 0;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dotf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
        .shaderIntegerDotProduct=VK_TRUE };
    if (v->has_dot) devexts[nexts++] = "VK_KHR_shader_integer_dot_product";

    /* COOPERATIVE MATRIX -- the tensor cores. Needs three things enabled
     * together, and the shader fails to compile into a pipeline if any is
     * missing: the extension itself, the Vulkan memory model (coopmat's
     * GL_KHR_memory_scope_semantics is defined in its terms) and shaderFloat16
     * (this kernel stages fp16 into shared memory). All three are queried, not
     * assumed -- an absent one leaves has_coop 0 and the caller keeps the dp4a
     * kernel, which is the same fallback discipline as has_dot. */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix=VK_TRUE };
    VkPhysicalDeviceVulkanMemoryModelFeatures memf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
        .vulkanMemoryModel=VK_TRUE, .vulkanMemoryModelDeviceScope=VK_TRUE };
    VkPhysicalDeviceShaderFloat16Int8Features f16f = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16=VK_TRUE };
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bdaf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
        .bufferDeviceAddress=VK_TRUE };
    if (v->has_bda) devexts[nexts++] = "VK_KHR_buffer_device_address";
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgcf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
        .subgroupSizeControl=VK_TRUE };
    {
        uint32_t ne=0; vkEnumerateDeviceExtensionProperties(v->pdev,NULL,&ne,NULL);
        VkExtensionProperties *ep = (VkExtensionProperties*)calloc(ne?ne:1,sizeof *ep);
        vkEnumerateDeviceExtensionProperties(v->pdev,NULL,&ne,ep);
        int have_coop=0, have_mm=0, have_f16=0, have_sgc=0;
        for (uint32_t i=0;i<ne;i++) {
            if (!strcmp(ep[i].extensionName,"VK_KHR_cooperative_matrix")) have_coop=1;
            if (!strcmp(ep[i].extensionName,"VK_KHR_vulkan_memory_model")) have_mm=1;
            if (!strcmp(ep[i].extensionName,"VK_KHR_shader_float16_int8")) have_f16=1;
            /* SUBGROUP SIZE CONTROL. attn_decode.comp needs 32-lane subgroups;
             * RADV's native width on gfx1103 is 64. This extension lets the
             * pipeline REQUIRE 32, which turns "silently wrong on AMD" into
             * "correct on AMD". Without it the attention pipeline is not created
             * at all -- see the guard at its creation site. */
            if (!strcmp(ep[i].extensionName,"VK_EXT_subgroup_size_control")) have_sgc=1;
        }
        free(ep);
        v->sg_ctl = (have_sgc && v->sg_min <= 32 && v->sg_max >= 32 && nexts < 6) ? 1 : 0;
        if (v->sg_ctl) devexts[nexts++] = "VK_EXT_subgroup_size_control";

        const char *e = getenv("COLI_VK_NO_COOP");
        if (e && *e && *e!='0') have_coop = 0;
        v->has_coop = (have_coop && nexts < 5) ? 1 : 0;
        if (v->has_coop) {
            devexts[nexts++] = "VK_KHR_cooperative_matrix";
            if (have_mm)  devexts[nexts++] = "VK_KHR_vulkan_memory_model";
            if (have_f16) devexts[nexts++] = "VK_KHR_shader_float16_int8";
            coopf.pNext = have_mm ? (void*)&memf : NULL;
            memf.pNext  = have_f16 ? (void*)&f16f : NULL;
            dotf.pNext  = (void*)&coopf;
        }
    }

    /* Two priorities, never more than two queues requested: qci[0].queueCount
     * is 2 only in fill_mode==2 (second queue, same family as compute); a
     * dedicated transfer family (fill_mode==1) is a SEPARATE create-info entry
     * at queueCount 1. Both arms land in the same qcis[]/nqci pair so the one
     * vkCreateDevice call below (and its bare retry) cover every fill_mode
     * without duplicating the call. */
    float prios[2] = {1.f, 1.f};
    VkDeviceQueueCreateInfo qcis[2];
    uint32_t nqci = 0;
    qcis[nqci++] = (VkDeviceQueueCreateInfo){ .sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=v->qfam, .queueCount=(v->fill_mode==2)?2:1, .pQueuePriorities=prios };
    if (v->fill_mode==1)
        qcis[nqci++] = (VkDeviceQueueCreateInfo){ .sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex=v->xfam, .queueCount=1, .pQueuePriorities=prios };
    const void *featchain = v->has_dot ? (const void*)&dotf
                          : (v->has_coop ? (const void*)&coopf : NULL);
    if (v->sg_ctl) { sgcf.pNext = (void*)featchain; featchain = (const void*)&sgcf; }
    if (v->has_bda) { bdaf.pNext = (void*)featchain; featchain = (const void*)&bdaf; }
    VkDeviceCreateInfo dci = { .sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=nqci, .pQueueCreateInfos=qcis,
        .enabledExtensionCount = nexts,
        .ppEnabledExtensionNames = nexts ? devexts : NULL,
        .pNext = featchain };
    if (vkCreateDevice(v->pdev,&dci,NULL,&v->dev) != VK_SUCCESS) {
        /* Retry bare: an advertised extension whose feature the driver refuses
         * must not cost us the GPU entirely. The queue request itself is kept
         * (qcis/nqci) -- a refused feature and an unavailable queue family are
         * unrelated failure modes and must not be conflated. */
        v->has_dot = 0; v->has_coop = 0; v->sg_ctl = 0; v->has_bda = 0;
        VkDeviceCreateInfo bare = { .sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount=nqci, .pQueueCreateInfos=qcis };
        if (vkCreateDevice(v->pdev,&bare,NULL,&v->dev) != VK_SUCCESS) { VKERR("vkCreateDevice failed"); goto fail; }
    }
    vkGetDeviceQueue(v->dev,v->qfam,0,&v->q);
    if      (v->fill_mode==1) vkGetDeviceQueue(v->dev,v->xfam,0,&v->fq);
    else if (v->fill_mode==2) vkGetDeviceQueue(v->dev,v->qfam,1,&v->fq);
    else                       v->fq = v->q;   /* sync fallback: fill just reuses the compute queue */
    snprintf(v->fill_mode_desc,sizeof v->fill_mode_desc,
        v->fill_mode==1 ? "async: dedicated transfer-only family %u" :
        v->fill_mode==2 ? "async: second queue, main family %u"      :
                           "synchronous (no second queue available, family %u)",
        v->fill_mode==1 ? v->xfam : v->qfam);

    /* shader */
    VkShaderModule sm;
    if (!load_module(v, spv_path, &sm)) { VKERR("cannot load SPIR-V from %s", spv_path); goto fail; }

    VkDescriptorSetLayoutBinding bind[6];
    for (int i=0;i<6;i++) bind[i]=(VkDescriptorSetLayoutBinding){ .binding=(uint32_t)i,
        .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount=1,
        .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=6, .pBindings=bind };
    vkCreateDescriptorSetLayout(v->dev,&dlci,NULL,&v->dsl);
    /* 32, not 24, since 2026-09-14: attn_decode.comp and swiglu_oai_q.comp declare
     * 8 words (window/sink_off, bias offsets + alpha/limit). Shaders that declare
     * fewer still receive their 24 unchanged. */
    VkPushConstantRange pc = { .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, .offset=0, .size=32 };
    VkPipelineLayoutCreateInfo plci = { .sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1, .pSetLayouts=&v->dsl, .pushConstantRangeCount=1, .pPushConstantRanges=&pc };
    vkCreatePipelineLayout(v->dev,&plci,NULL,&v->pl);
    VkComputePipelineCreateInfo cpci = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm, .pName="main" },
        .layout=v->pl };
    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cpci,NULL,&v->pipe)!=VK_SUCCESS) {
        VKERR("pipeline creation failed"); vkDestroyShaderModule(v->dev,sm,NULL); goto fail; }
    vkDestroyShaderModule(v->dev,sm,NULL);

    /* The int4 pipeline, from gemm_i4.spv beside the int8 one. OPTIONAL: if it
     * is not there, coli_vk_upload_w4 refuses and the caller stays on int8
     * rather than the whole backend failing to initialise. Both pipelines share
     * dsl/pl -- the two shaders declare the same six bindings and the same push
     * constants deliberately, so one descriptor set serves either. */
    {
        char p4[512];
        size_t L = strlen(spv_path);
        const char *base = "gemm_i8.spv";
        size_t bl = strlen(base);
        if (L > bl && !strcmp(spv_path + L - bl, base)) {
            snprintf(p4, sizeof p4, "%.*sgemm_i4.spv", (int)(L - bl), spv_path);
        } else {
            snprintf(p4, sizeof p4, "%s", "shaders/gemm_i4.spv");
        }
        /* COLI_VK_I4_SPV=<path> swaps the int4 kernel for an A/B on the SAME binary
         * (2026-09-08, batch-1 GEMV work). The sibling f/dp/attn paths are still
         * derived from the default name above, so only pipe4 changes. */
        { const char *alt = getenv("COLI_VK_I4_SPV"); if (alt && *alt) snprintf(p4, sizeof p4, "%s", alt); }
        /* The float-dequant variant, same construction. Optional in exactly the
         * same way -- it exists to be measured against pipe4, not to ship. */
        {
            char pf[512]; size_t n4 = strlen(p4);
            if (n4 > 4) { snprintf(pf, sizeof pf, "%.*sf.spv", (int)(n4-4), p4); }
            else pf[0] = 0;
            VkShaderModule smf;
            if (pf[0] && load_module(v, pf, &smf)) {
                VkComputePipelineCreateInfo cf = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                             .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=smf, .pName="main" },
                    .layout=v->pl };
                if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cf,NULL,&v->pipe4f)!=VK_SUCCESS)
                    v->pipe4f = VK_NULL_HANDLE;
                vkDestroyShaderModule(v->dev,smf,NULL);
            }
        }
        {   /* short-row int4 kernel beside gemm_i4.spv: gemm_i4_short.spv, with the
             * _dp sibling preferred exactly as for pipe4. Optional. */
            char ps[512]; size_t n4 = strlen(p4);
            v->pipe4s = VK_NULL_HANDLE;
            { const char *e = getenv("COLI_VK_NO_SHORT"); v->short_rows_off = (e && *e && *e!='0') ? 1 : 0; }
            if (n4 > 4) {
                const char *cands[2]; char pa_[512], pb_[512];
                snprintf(pa_, sizeof pa_, "%.*s_short_dp.spv", (int)(n4-4), p4);
                snprintf(pb_, sizeof pb_, "%.*s_short.spv",    (int)(n4-4), p4);
                cands[0] = v->has_dot ? pa_ : pb_; cands[1] = v->has_dot ? pb_ : pa_;
                for (int ci=0; ci<2 && !v->pipe4s; ci++) {
                    VkShaderModule sm2;
                    if (!load_module(v, cands[ci], &sm2)) continue;
                    VkComputePipelineCreateInfo c2 = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm2, .pName="main" },
                        .layout=v->pl };
                    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&c2,NULL,&v->pipe4s)!=VK_SUCCESS) v->pipe4s = VK_NULL_HANDLE;
                    vkDestroyShaderModule(v->dev,sm2,NULL);
                }
            }
        }
        {   /* multi-matrix int4 kernel: gemm_i4_moe_dp.spv (+ _moe_short_dp for
             * I <= 1024) beside gemm_i4.spv. DP4a-only and needs device addresses;
             * absent or disabled means coli_vk_moe4 keeps per-matrix dispatches. */
            v->pipe4m = VK_NULL_HANDLE; v->pipe4ms = VK_NULL_HANDLE; v->pfn_bda = NULL;
            size_t n4 = strlen(p4);
            if (v->has_dot && v->has_bda && n4 > 4) {
                v->pfn_bda = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(v->dev, "vkGetBufferDeviceAddressKHR");
                if (!v->pfn_bda) v->pfn_bda = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(v->dev, "vkGetBufferDeviceAddress");
                const char *suf[2] = { "_moe_dp.spv", "_moe_short_dp.spv" }; VkPipeline *dst[2] = { &v->pipe4m, &v->pipe4ms };
                for (int ci=0; ci<2 && v->pfn_bda; ci++) {
                    char pm_[512]; snprintf(pm_, sizeof pm_, "%.*s%s", (int)(n4-4), p4, suf[ci]);
                    VkShaderModule sm3;
                    if (!load_module(v, pm_, &sm3)) continue;
                    VkComputePipelineCreateInfo c3 = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm3, .pName="main" },
                        .layout=v->pl };
                    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&c3,NULL,dst[ci])!=VK_SUCCESS) *dst[ci] = VK_NULL_HANDLE;
                    vkDestroyShaderModule(v->dev,sm3,NULL);
                }
            }
        }
        {   /* gpt-oss (2026-09-14): MXFP4-decode int4 kernels beside gemm_i4.spv
             * (gemm_i4_mx_dp.spv, gemm_i4_mx_short_dp.spv) and the swiglu_oai
             * epilogue (swiglu_oai_q.spv). DP4a-only, like the _moe kernels. */
            v->pipe4mx = VK_NULL_HANDLE; v->pipe4mxs = VK_NULL_HANDLE; v->pipe_swq = VK_NULL_HANDLE;
            size_t n4 = strlen(p4);
            if (v->has_dot && n4 > 4) {
                const char *suf[2] = { "_mx_dp.spv", "_mx_short_dp.spv" }; VkPipeline *dst[2] = { &v->pipe4mx, &v->pipe4mxs };
                for (int ci=0; ci<2; ci++) {
                    char pm_[512]; snprintf(pm_, sizeof pm_, "%.*s%s", (int)(n4-4), p4, suf[ci]);
                    VkShaderModule sm3;
                    if (!load_module(v, pm_, &sm3)) continue;
                    VkComputePipelineCreateInfo c3 = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm3, .pName="main" },
                        .layout=v->pl };
                    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&c3,NULL,dst[ci])!=VK_SUCCESS) *dst[ci] = VK_NULL_HANDLE;
                    vkDestroyShaderModule(v->dev,sm3,NULL);
                }
            }
            { char ps_[512]; const char *slash = strrchr(p4, '/');
              if (slash) snprintf(ps_, sizeof ps_, "%.*s/swiglu_oai_q.spv", (int)(slash-p4), p4);
              else       snprintf(ps_, sizeof ps_, "swiglu_oai_q.spv");
              VkShaderModule sm4;
              if (load_module(v, ps_, &sm4)) {
                  VkComputePipelineCreateInfo c4 = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                      .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                               .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm4, .pName="main" },
                      .layout=v->pl };
                  if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&c4,NULL,&v->pipe_swq)!=VK_SUCCESS) v->pipe_swq = VK_NULL_HANDLE;
                  vkDestroyShaderModule(v->dev,sm4,NULL);
              } }
        }
        {   /* attention, from shaders/attn_decode.spv beside the rest. Shares
             * dsl/pl: it declares the same six bindings (the sixth unused) and
             * the same 24 bytes of push constants, so no second layout is
             * needed. See the note on gemm_i4.spv above. */
            char pa[512]; const char *slash = strrchr(p4, '/');
            if (slash) snprintf(pa, sizeof pa, "%.*s/attn_decode.spv", (int)(slash-p4), p4);
            else       snprintf(pa, sizeof pa, "%s", "shaders/attn_decode.spv");
            /* THE SUBGROUP WIDTH GUARD, which used to exist only as a comment.
             *
             * attn_decode.comp is written for 32-lane subgroups: a lane owns
             * dims lane, lane+32, ... and the score reduction is a subgroupAdd
             * across exactly those 32 lanes. On a 64-lane device both of those
             * are wrong, and wrong QUIETLY -- the dispatch succeeds and returns
             * numbers.
             *
             * The shader's own header said the width was "asserted by the host
             * before dispatch". It was not. Nothing in this file mentioned
             * subgroups. NVIDIA reports 32, the kernel was validated there, and
             * so the missing check cost nothing until it was run on RADV
             * (gfx1103, native width 64), where tests/test_vk_attn reported
             * rel=2.166e+01 against a tolerance of 1e-4 -- measured 2026-08-27.
             * That is also what the decode path (COLI_GPU_ATTN) would have done
             * on that machine, and it predates the prefill work entirely.
             *
             * Preferred fix is to PIN the width to 32 where the device allows
             * it. Failing that the pipeline is not created, has_attn stays 0,
             * and every caller falls back to the CPU -- slower, and right. */
            VkShaderModule sma;
            if (load_module(v, pa, &sma)) {
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rq = {
                    .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
                    .requiredSubgroupSize=32 };
                int pin = (v->sg_size != 32) && v->sg_ctl;
                VkComputePipelineCreateInfo ca = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                             .pNext = pin ? (const void*)&rq : NULL,
                             .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sma, .pName="main" },
                    .layout=v->pl };
                if (v->sg_size != 32 && !pin) {
                    /* 64-lane device with no way to ask for 32. Refuse. */
                    v->pipe_attn = VK_NULL_HANDLE;
                } else if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&ca,NULL,&v->pipe_attn)!=VK_SUCCESS) {
                    v->pipe_attn = VK_NULL_HANDLE;
                } else {
                    v->sg_pinned = pin;
                }
                vkDestroyShaderModule(v->dev,sma,NULL);
            }
        }
        {   /* Split-K decode attention: attn_decode_split.spv + attn_decode_merge.spv,
             * beside attn_decode.spv. Added 2026-09-13 -- see the header of
             * attn_decode_split.comp for why (measured: attn_decode.comp's
             * dispatch is 64.7 us of a 119.8 us fused attention block on an
             * RTX 4070, Qwen3-30B-A3B, decode n=1, kv_ctx~266).
             *
             * NEEDS ITS OWN PIPELINE LAYOUT (pl2), built from the SAME dsl:
             * both shaders take a 7th push-constant field (nchunk) that does
             * not fit v->pl's fixed 24-byte range, which every OTHER pipeline
             * in this file shares deliberately (see record_dispatch's push6
             * comment on that range). A second pipeline layout over the same
             * descriptor set layout is ordinary Vulkan -- descriptor sets are
             * compatible with any pipeline layout built from the same set
             * layout, only the push-constant range differs -- and it is
             * smaller in scope than growing v->pl's range for every kernel
             * that never uses the extra bytes. */
            v->pipe_attn_split = VK_NULL_HANDLE; v->pipe_attn_merge = VK_NULL_HANDLE; v->pl2 = VK_NULL_HANDLE;
            VkPushConstantRange pc2 = { .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, .offset=0, .size=28 };
            VkPipelineLayoutCreateInfo pl2ci = { .sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount=1, .pSetLayouts=&v->dsl, .pushConstantRangeCount=1, .pPushConstantRanges=&pc2 };
            if (vkCreatePipelineLayout(v->dev,&pl2ci,NULL,&v->pl2)==VK_SUCCESS) {
                const char *slash2 = strrchr(p4, '/');
                char psplit[512], pmerge[512];
                if (slash2) { snprintf(psplit,sizeof psplit,"%.*s/attn_decode_split.spv",(int)(slash2-p4),p4);
                              snprintf(pmerge,sizeof pmerge,"%.*s/attn_decode_merge.spv",(int)(slash2-p4),p4); }
                else        { snprintf(psplit,sizeof psplit,"%s","shaders/attn_decode_split.spv");
                              snprintf(pmerge,sizeof pmerge,"%s","shaders/attn_decode_merge.spv"); }

                /* Same 32-lane requirement as attn_decode.comp: attn_decode_split.comp
                 * uses subgroupAdd across the same 32-lane dim split. Refuse under
                 * the identical condition pipe_attn refuses under -- a kernel that
                 * silently ran at the wrong width is the exact defect the pipe_attn
                 * guard above was written to stop, and this is the same kernel
                 * family. attn_decode_merge.comp does no subgroup ops and needs no
                 * pin. */
                VkShaderModule smsplit;
                if (v->pipe_attn && load_module(v, psplit, &smsplit)) {
                    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rq2 = {
                        .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
                        .requiredSubgroupSize=32 };
                    int pin2 = (v->sg_size != 32) && v->sg_ctl;
                    VkComputePipelineCreateInfo cs = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .pNext = pin2 ? (const void*)&rq2 : NULL,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=smsplit, .pName="main" },
                        .layout=v->pl2 };
                    if (v->sg_size == 32 || pin2) {
                        if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cs,NULL,&v->pipe_attn_split)!=VK_SUCCESS)
                            v->pipe_attn_split = VK_NULL_HANDLE;
                    }
                    vkDestroyShaderModule(v->dev,smsplit,NULL);
                }
                VkShaderModule smmerge;
                if (v->pipe_attn_split && load_module(v, pmerge, &smmerge)) {
                    VkComputePipelineCreateInfo cm = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=smmerge, .pName="main" },
                        .layout=v->pl2 };
                    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cm,NULL,&v->pipe_attn_merge)!=VK_SUCCESS)
                        v->pipe_attn_merge = VK_NULL_HANDLE;
                    vkDestroyShaderModule(v->dev,smmerge,NULL);
                }
                /* Half-loaded is not usable: a split pipeline with no merge
                 * pipeline (or vice versa) can dispatch garbage into a real
                 * decode. coli_vk_has_attn_split and the block below both
                 * require BOTH, so tear down an orphan half explicitly rather
                 * than relying on every caller to check both fields. */
                if (!v->pipe_attn_split || !v->pipe_attn_merge) {
                    if (v->pipe_attn_split) { vkDestroyPipeline(v->dev,v->pipe_attn_split,NULL); v->pipe_attn_split=VK_NULL_HANDLE; }
                    if (v->pipe_attn_merge) { vkDestroyPipeline(v->dev,v->pipe_attn_merge,NULL); v->pipe_attn_merge=VK_NULL_HANDLE; }
                }
            }
        }
        {   /* silu*mul+quantize, from shaders/silu_mul_q.spv beside the rest */
            char ps[512]; size_t n4 = strlen(p4);
            const char *slash = strrchr(p4, '/');
            if (slash) snprintf(ps, sizeof ps, "%.*s/silu_mul_q.spv", (int)(slash-p4), p4);
            else       snprintf(ps, sizeof ps, "silu_mul_q.spv");
            (void)n4;
            VkShaderModule sms;
            if (load_module(v, ps, &sms)) {
                VkComputePipelineCreateInfo cs = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                             .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sms, .pName="main" },
                    .layout=v->pl };
                if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cs,NULL,&v->pipe_smq)!=VK_SUCCESS)
                    v->pipe_smq = VK_NULL_HANDLE;
                vkDestroyShaderModule(v->dev,sms,NULL);
            }
        }
        {   /* rope_bias.spv and kvwrite.spv. Both use the same 6 storage buffers
             * and the same 24 bytes of push constants as everything else, so
             * neither needs a second descriptor layout -- see the note on
             * attn_decode.spv above. Absence of either is a normal answer: the
             * caller keeps the CPU path, which is also the numerical reference. */
            const char *names[7] = { "rope_bias.spv", "kvwrite.spv", "quant.spv", "gemm_i4_tile.spv", "gemm_i4_coop.spv", "gemm_i4_coop_ds.spv", "qknorm.spv" };
            VkPipeline *dst[7]   = { &v->pipe_rope, &v->pipe_kvw, &v->pipe_quant, &v->pipe4t, &v->pipe4c, &v->pipe4cd, &v->pipe_qkn };
            const char *slash = strrchr(p4, '/');
            for (int i=0;i<7;i++) {
                /* the coopmat pipelines are only attempted when the device gave us
                 * the extension; loading them otherwise guarantees a create failure. */
                if (i>=4 && !v->has_coop) { *dst[i] = VK_NULL_HANDLE; continue; }
                char pn[512];
                if (slash) snprintf(pn,sizeof pn,"%.*s/%s",(int)(slash-p4),p4,names[i]);
                else       snprintf(pn,sizeof pn,"%s",names[i]);
                VkShaderModule sm;
                if (!load_module(v,pn,&sm)) { *dst[i]=VK_NULL_HANDLE; continue; }
                VkComputePipelineCreateInfo ci = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                             .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm, .pName="main" },
                    .layout=v->pl };
                if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&ci,NULL,dst[i])!=VK_SUCCESS) {
                    *dst[i] = VK_NULL_HANDLE;
                    /* SAY SO. A pipeline that fails to create makes the caller
                     * fall back silently, and a fallback is indistinguishable
                     * from a slow kernel in a timing. gemm_i4_tile.spv is the
                     * live case: at KC=256 it needs 95,488 bytes of shared
                     * memory against this device's 49,152, glslang compiles it
                     * happily because it does not check device limits, and the
                     * only symptom would have been a benchmark quietly measuring
                     * the OTHER kernel. */
                    fprintf(stderr,"vk: pipeline %s failed to create -- falling back "
                                   "(shared memory or feature limit?)\n", names[i]);
                }
                vkDestroyShaderModule(v->dev,sm,NULL);
            }
        }
        /* Prefer the DP4a build of the same kernel when the device supports the
         * instruction. Same maths, same bindings, same push constants -- only
         * the inner product changes -- so a fallback to the scalar spv is a
         * performance difference and never a correctness one. v->dot_used
         * records which one actually loaded, because "supported" and "in use"
         * are different claims and only the second is worth reporting. */
        v->dot_used = 0;
        if (v->has_dot) {
            char pdp[512]; size_t n4 = strlen(p4);
            if (n4 > 4) {
                snprintf(pdp, sizeof pdp, "%.*s_dp.spv", (int)(n4-4), p4);
                VkShaderModule smd;
                if (load_module(v, pdp, &smd)) {
                    VkComputePipelineCreateInfo cd = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=smd, .pName="main" },
                        .layout=v->pl };
                    if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cd,NULL,&v->pipe4)==VK_SUCCESS)
                        v->dot_used = 1;
                    vkDestroyShaderModule(v->dev,smd,NULL);
                }
            }
        }
        VkShaderModule sm4;
        if (!v->dot_used && load_module(v, p4, &sm4)) {
            VkComputePipelineCreateInfo c4 = { .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage={ .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                         .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm4, .pName="main" },
                .layout=v->pl };
            if (vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&c4,NULL,&v->pipe4)!=VK_SUCCESS)
                v->pipe4 = VK_NULL_HANDLE;
            vkDestroyShaderModule(v->dev,sm4,NULL);
        }
    }

    VkDescriptorPoolSize ps = { .type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount=6*64 };
    VkDescriptorPoolCreateInfo dpci = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets=64, .poolSizeCount=1, .pPoolSizes=&ps,
        .flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT };
    vkCreateDescriptorPool(v->dev,&dpci,NULL,&v->dpool);

    VkCommandPoolCreateInfo cpi = { .sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex=v->qfam, .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    vkCreateCommandPool(v->dev,&cpi,NULL,&v->pool);
    VkCommandBufferAllocateInfo cbi = { .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool, .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount=1 };
    vkAllocateCommandBuffers(v->dev,&cbi,&v->cmd);
    VkFenceCreateInfo fci = { .sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(v->dev,&fci,NULL,&v->fence);

    /* fpool: a command pool is bound to ONE queue family at creation. v->pool
     * above was created for qfam; fslots' command buffers run on fq, which is
     * a DIFFERENT family only when fill_mode==1 -- reuse v->pool otherwise
     * (fill_mode==2's second queue is still qfam; fill_mode==0 never submits
     * through fpool at all). */
    if (v->fill_mode == 1) {
        VkCommandPoolCreateInfo fcpi = { .sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex=v->xfam, .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
        if (vkCreateCommandPool(v->dev,&fcpi,NULL,&v->fpool) != VK_SUCCESS) {
            /* can't get a pool for xfam -- fall all the way back rather than
             * hand out command buffers from the wrong family's pool */
            v->fill_mode = 0; v->fq = v->q; v->fpool = v->pool;
            snprintf(v->fill_mode_desc,sizeof v->fill_mode_desc,
                "synchronous (xfam command pool creation failed)");
        }
    } else {
        v->fpool = v->pool;
    }
    return v;
fail:
    if (v->dev) vkDestroyDevice(v->dev,NULL);
    vkDestroyInstance(v->inst,NULL); free(v); return NULL;
}

const char *coli_vk_device_name(coli_vk *v){ return v?v->devname:"(none)"; }
/* Subgroup state, exposed so a test can PRINT which arm it exercised. A kernel
 * that silently ran at the wrong width is exactly what went unnoticed here, and
 * the defence is an instrument that makes the width visible in the output. */
int coli_vk_subgroup_size(coli_vk *v){ return v ? (int)v->sg_size : 0; }
int coli_vk_attn_pinned32(coli_vk *v){ return v ? v->sg_pinned : 0; }
const char *coli_vk_mem_desc(coli_vk *v){ return (v&&v->memdesc[0])?v->memdesc:"(no allocation yet)"; }
const char *coli_vk_weight_mem(coli_vk *v){ return (v&&v->memdesc2[0])?v->memdesc2:"(none)"; }
int coli_vk_is_integrated(coli_vk *v){ return v?v->integrated:0; }
/* Which memory the WEIGHTS actually landed in. Reported rather than inferred:
 * the allocation flags say what was asked for, these say what was granted. */
const char *coli_vk_memdesc (coli_vk *v){ return (v&&v->memdesc[0]) ?v->memdesc :"unknown"; }
const char *coli_vk_memdesc2(coli_vk *v){ return (v&&v->memdesc2[0])?v->memdesc2:"unknown"; }
int coli_vk_dot_used(coli_vk *v){ return v?v->dot_used:0; }

/* Device class WITHOUT uploading anything: init the device, read its type, tear
 * it down. Exists so a backend can be CHOSEN before paying the weight upload --
 * asking "is this integrated?" after uploading 4.5 GiB defeats the purpose.
 * Returns 1 integrated, 0 discrete, -1 no usable Vulkan device. */
int coli_vk_probe_class(const char *spv_path) {
    char err[256] = {0};
    coli_vk *v = coli_vk_init(spv_path, err, sizeof err);
    if (!v) return -1;
    int integ = v->integrated;
    coli_vk_free(v);
    return integ ? 1 : 0;
}
int coli_vk_wants_device_local(coli_vk *v){ return v?v->want_device_local:0; }

/* Copy through a staging buffer into DEVICE_LOCAL memory. Only worth doing for
 * weights, which are written once and read every step; doing it for activations
 * would add a copy per call to save nothing. */
/* Submit whatever copies are recorded in the batch window and wait for them.
 * Returns 0 if a submit or wait failed; that is sticky (up_failed) so
 * coli_vk_upload_end() reports it even if a later flush succeeds. */
static int flush_uploads(coli_vk *v) {
    if (!v->up_pending) return !v->up_failed;
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    int ok = (vkQueueSubmit(v->q,1,&si,v->fence)==VK_SUCCESS) &&
             (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)==VK_SUCCESS);
    v->up_pending = 0; v->upring_off = 0; v->up_flushes++;
    if (!ok) v->up_failed = 1;
    return ok;
}

/* Batched path of upload_device_local. 1 = copy recorded (lands at the next
 * flush), 0 = a flush failed, -1 = cannot batch this one (no ring, or larger
 * than the ring) -- the caller falls back to the per-matrix path. */
static int upload_batched(coli_vk *v, vkbuf *dst, const void *src, size_t n) {
    if (!v->upring.buf) {
        if (!mkbuf_flags(v, v->upring_cap, &v->upring,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return -1;
        if (vkMapMemory(v->dev,v->upring.mem,0,v->upring_cap,0,&v->upring_map) != VK_SUCCESS) {
            freebuf(v,&v->upring); v->upring_map = NULL; return -1; }
    }
    if ((VkDeviceSize)n > v->upring_cap) return -1;
    VkDeviceSize need = ((VkDeviceSize)n + 63) & ~(VkDeviceSize)63;
    if (v->upring_off + need > v->upring_cap && !flush_uploads(v)) return 0;
    uint64_t t0 = now_ns();
    memcpy((char*)v->upring_map + v->upring_off, src, n);
    P.w_ns += now_ns()-t0; P.w_bytes += n; P.w_n++;
    if (!v->up_pending) {
        VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkResetCommandBuffer(v->cmd,0);
        vkBeginCommandBuffer(v->cmd,&bi);
        v->up_pending = 1;
    }
    VkBufferCopy cp = { .srcOffset=v->upring_off, .dstOffset=0, .size=n };
    vkCmdCopyBuffer(v->cmd, v->upring.buf, dst->buf, 1, &cp);
    v->upring_off += need;
    return 1;
}

int coli_vk_upload_begin(coli_vk *v) {
    if (!v) return 0;
    const char *e = getenv("COLI_VK_UPLOAD_BATCH_MB");
    unsigned long mb = (e && *e) ? strtoul(e,NULL,10) : 256;
    v->up_batch = 0; v->up_failed = 0; v->up_flushes = 0;
    if (!mb) return 0;
    if (v->upring.buf && v->upring_cap != (VkDeviceSize)mb<<20) {
        if (v->upring_map) vkUnmapMemory(v->dev,v->upring.mem);
        freebuf(v,&v->upring); v->upring_map = NULL;
    }
    v->upring_cap = (VkDeviceSize)mb << 20;
    v->up_batch = 1;
    return 1;
}

int coli_vk_upload_end(coli_vk *v) {
    if (!v) return 0;
    if (!v->up_batch) return 1;
    int ok = flush_uploads(v);
    v->up_batch = 0;
    fprintf(stderr, "gpu upload: batched staging ring %llu MiB, %d flushes%s\n",
            (unsigned long long)(v->upring_cap>>20), v->up_flushes,
            v->up_failed ? " -- A FLUSH FAILED" : "");
    return ok && !v->up_failed;
}

static int upload_device_local(coli_vk *v, vkbuf *dst, const void *src, size_t n) {
    P.in_weight_upload = 1;
    if (v->up_batch) {
        int r = upload_batched(v, dst, src, n);
        if (r >= 0) { P.in_weight_upload = 0; return r; }
        if (!flush_uploads(v)) { P.in_weight_upload = 0; return 0; }   /* then the old path below */
    }
    vkbuf stage = {0};
    if (!mkbuf_flags(v, n, &stage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) { P.in_weight_upload = 0; return 0; }
    if (!upload(v,&stage,src,n)) { freebuf(v,&stage); P.in_weight_upload = 0; return 0; }
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    VkBufferCopy cp = { .srcOffset=0, .dstOffset=0, .size=n };
    vkCmdCopyBuffer(v->cmd, stage.buf, dst->buf, 1, &cp);
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    int ok = (vkQueueSubmit(v->q,1,&si,v->fence)==VK_SUCCESS) &&
             (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)==VK_SUCCESS);
    freebuf(v,&stage);
    P.in_weight_upload = 0;
    return ok;
}

int coli_vk_upload_w(coli_vk *v, const coli_w_i8 *w) {
    if (v->nw >= MAX_W) return -1;
    int h = v->nw;
    size_t wn = (size_t)w->I*w->O, sn = (size_t)w->O*sizeof(float);

    /* On a DISCRETE card, put the weights in VRAM. They are written once and
     * read on every step, so a one-time staging copy trades load time for the
     * difference between PCIe (~25 GB/s) and VRAM (hundreds).
     *
     * ON AN INTEGRATED GPU this used to be skipped unconditionally, on the
     * stated grounds that "the memory is already shared and the copy would be
     * pure waste". That is a claim, and on the Radeon 780M the heap table
     * contradicts its premise: RADV exposes TWO heaps, and the HOST_VISIBLE|
     * HOST_COHERENT type this falls back to is heap 0, which is NOT
     * DEVICE_LOCAL, while heap 1 IS DEVICE_LOCAL and larger. HOST_COHERENT on
     * AMD is uncached/write-combined, so "shared memory" does not imply "same
     * read path for the GPU". Whether heap 1 is actually faster is an empirical
     * question, so this is OPT-IN (COLI_VK_DEVICE_LOCAL=1) and off by default
     * until measured on the target. */
    if ((!v->integrated || v->want_device_local) && !v->force_host_visible) {
        int okw = mkbuf_flags(v, wn, &v->W[h].w, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        int oks = okw && mkbuf_flags(v, sn, &v->W[h].ws, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (oks && upload_device_local(v,&v->W[h].w,w->qu,wn)
                && upload_device_local(v,&v->W[h].ws,w->scale,sn)) {
            if (!v->memdesc2[0]) snprintf(v->memdesc2,sizeof v->memdesc2,"DEVICE_LOCAL (staged)");
            if (!v->memdesc[0]) {
                uint32_t mt = find_mem(v, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (mt != UINT32_MAX) describe_mem(v, mt, v->memdesc, sizeof v->memdesc);
            }
            v->W[h].I=w->I; v->W[h].O=w->O; v->W[h].used=1; v->nw++;
            return h;
        }
        /* Fall through to host-visible: a device with no suitable DEVICE_LOCAL
         * type must still work, just slower. */
        freebuf(v,&v->W[h].w); freebuf(v,&v->W[h].ws);
    }
    if (!mkbuf(v,wn,&v->W[h].w))  return -1;
    if (!mkbuf(v,sn,&v->W[h].ws)) { freebuf(v,&v->W[h].w); return -1; }
    if (!upload(v,&v->W[h].w, w->qu, wn))   return -1;
    if (!upload(v,&v->W[h].ws,w->scale,sn)) return -1;
    if (!v->memdesc2[0]) snprintf(v->memdesc2,sizeof v->memdesc2,"HOST_VISIBLE");
    v->W[h].I=w->I; v->W[h].O=w->O; v->W[h].used=1;
    v->nw++;
    return h;
}

/* The dispatch, shared by both formats. Only three things differ between the
 * int8 and int4 calls -- which pipeline, which two weight buffers, and where the
 * shape comes from -- so they are arguments rather than a duplicated 60-line
 * function that would drift the moment one of them was fixed. */
/* The dispatch, over activation buffers that are ALREADY on the device and into
 * an output buffer that is NOT downloaded. This is the form residency needs:
 * gemm_dispatch below is now just this plus an upload and a download.
 *
 * Splitting it is the whole enabling change. While every GEMM uploaded its
 * inputs and downloaded its result, an FFN paid three round trips and a CPU
 * requantization in the middle, and no amount of kernel tuning could remove
 * them. */
/* Write a descriptor set, without touching the command buffer. Split out so the
 * FFN can prepare all four of its sets before recording, which it must: a
 * descriptor set cannot be updated while a command buffer that uses it is being
 * built with a different binding. */
static void write_set(coli_vk *v, VkDescriptorSet ds, VkBuffer b0, VkBuffer b1,
                      VkBuffer b2, VkBuffer b3, VkBuffer b4, VkBuffer b5) {
    VkBuffer bufs[6] = { b0,b1,b2,b3,b4,b5 };
    VkDescriptorBufferInfo dbi[6]; VkWriteDescriptorSet wr[6];
    for (int i=0;i<6;i++) {
        dbi[i]=(VkDescriptorBufferInfo){ .buffer=bufs[i], .offset=0, .range=VK_WHOLE_SIZE };
        wr[i]=(VkWriteDescriptorSet){ .sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=ds, .dstBinding=(uint32_t)i, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[i] };
    }
    vkUpdateDescriptorSets(v->dev,6,wr,0,NULL);
}

/* Like write_set but binding 5 (the output) starts at byte offset off5, so one
 * buffer can hold every expert's result and each dispatch writes its own slice.
 * off5 must be minStorageBufferOffsetAlignment-aligned (16 B on the 4070); the
 * expert-major COOP_ROW_PAD(n)*Dout*4 stride always is. Used by coli_vk_moe4. */
/* Per-binding byte offsets into six buffers. Every offset must be
 * minStorageBufferOffsetAlignment-aligned (16 B on the 4070; callers pad slices
 * to 256 B so any device is safe). Added 2026-09-08 for the de-serialized
 * grouped-expert call, which binds each expert its own slice of the shared
 * fg/fu/hq/hs/hm scratch. */
static void write_set_offs(coli_vk *v, VkDescriptorSet ds, const VkBuffer bufs[6], const VkDeviceSize offs[6]) {
    VkDescriptorBufferInfo dbi[6]; VkWriteDescriptorSet wr[6];
    for (int i=0;i<6;i++) {
        dbi[i]=(VkDescriptorBufferInfo){ .buffer=bufs[i], .offset=offs[i], .range=VK_WHOLE_SIZE };
        wr[i]=(VkWriteDescriptorSet){ .sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=ds, .dstBinding=(uint32_t)i, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[i] };
    }
    vkUpdateDescriptorSets(v->dev,6,wr,0,NULL);
}

/* RECORD one dispatch into an already-open command buffer. No submit, no fence.
 *
 * This is the half of ggml's design that memory residency alone does not give
 * you: ggml_backend_vk_graph_compute records ~100 nodes (or ~100 MB of matmul)
 * into ONE command buffer before submitting, explicitly "to overlap CPU cmdbuffer
 * generation with GPU execution". Submitting per operator pays a queue round trip
 * and a fence wait each time, and measurement said that -- not the transfers --
 * was what a "fused" FFN with four submits was still paying. */
static void record_dispatch(coli_vk *v, VkPipeline pipe, VkDescriptorSet ds,
                            int64_t I, int64_t O, int n, uint32_t groups) {
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    /* ALL 24 BYTES, ALWAYS. gemm_i4.comp declares six ints -- I, O, n, nblk,
     * tile, outs -- and this used to push only the first four. The remaining two
     * then held whatever the last push into this command buffer left behind,
     * which for ffn4 and gemm4_qkv was nothing, so they read zero, clamped to
     * tile=1/outs=1, and matched the groups count the callers pass. Correct by
     * luck, and the luck ran out the moment a command buffer pushed 24 bytes
     * first: the fused block dispatches attention (24 bytes) before o_proj, so
     * o_proj inherited attention's n as `tile` and the bit pattern of its float
     * scale as `outs`. Measured 2026-08-23: layer 0 exact at 4.9e-10, layer 1
     * wrong at 1.3e-2, whole-model TF-NLL 15.98 against a true 2.68 -- and the
     * layer-dependence was the tell, because leftover state is what varies by
     * position in a command buffer while arithmetic does not.
     *
     * tile=1/outs=1 is stated rather than inherited. It is what every caller's
     * group count already assumes; making it explicit is the fix. (outs=1 does
     * leave three quarters of each 64-thread workgroup idle on the int4 kernel,
     * where gemm_on_device passes outs=4 -- that is a real and separate
     * performance question, not to be conflated with this correctness one.) */
    int32_t push[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), 1, 1 };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,push);
    vkCmdDispatch(v->cmd,groups,1,1);
}

/* WHICH coopmat kernel. Two call sites choose it and they used to name pipe4c
 * directly -- exactly the shape that let o_proj keep the old kernel for a day
 * while the profile read as "coopmat did not help". One function, both sites. */
static VkPipeline coop_pipe(coli_vk *v) {
    if (v->coop_ds && v->pipe4cd) return v->pipe4cd;
    return v->pipe4c;
}

/* RECORD a GEMM with the SAME geometry gemm_on_device submits: tile from the
 * device, and outs=4 on the int4 kernel because it runs one 16-lane cluster per
 * output, so a 64-thread workgroup covers four. The group count is derived here
 * rather than passed in, because it is a function of tile and outs and the two
 * must not be allowed to disagree.
 *
 * This exists because outs is the difference between using a whole workgroup and
 * a quarter of one. When record_dispatch pushed only 16 bytes, the recorded
 * paths inherited outs from whatever a previous submission had left -- which on
 * this driver was gemm_on_device's 4, so ffn4 and gemm4_qkv were getting the
 * FAST geometry by accident. Pinning them honestly to outs=1 was correct and
 * measured 2x slower; this is the fix that is both. */
static void record_gemm(coli_vk *v, VkPipeline pipe, VkDescriptorSet ds,
                        int64_t I, int64_t O, int n) {
    /* BATCHED PATH. Same bindings, same 24-byte push, different geometry: one
     * workgroup covers a 64x64 output tile instead of `outs` outputs. Only for
     * the int4 kernel, only when dp4a gave us pipe4t, and only above the
     * measured crossover -- decode (n=1) must keep the kernel it is tuned for. */
    /* TENSOR CORES first when available and worth it. Same bindings, same push
     * constants; only the grid differs (32x64 tile, 8 subgroups of one 16x16
     * accumulator each). COLI_VK_COOP_MIN_N=0 disables so the dp4a kernel can be
     * measured against it on the same binary. */
    if (pipe == v->pipe4 && coop_pipe(v) && v->coop_min_n > 0 && n >= v->coop_min_n) {
        vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,coop_pipe(v));
        vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
        int32_t pc[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), 0, 0 };
        vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,pc);
        uint32_t rt = (uint32_t)((n + COOP_TSR - 1) / COOP_TSR);
        uint32_t ot = (uint32_t)((O + COOP_TSC - 1) / COOP_TSC);
        vkCmdDispatch(v->cmd, rt*ot, 1, 1);
        P.coop_n++; if (coop_pipe(v) == v->pipe4cd) P.coop_ds_n++;
        return;
    }
    if (pipe == v->pipe4 && v->pipe4t && n >= v->tile_min_n) {
        vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe4t);
        vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
        int32_t pt[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), 0, 0 };
        vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,pt);
        /* MUST match TSR/TSC in shaders/gemm_i4_tile.comp. They are read from
         * the env so a sweep does not need a host rebuild, and they default to
         * the shader's compiled-in values -- a mismatch computes the wrong tiles
         * silently rather than failing, which is why they are named here. */
        int tsr = 64, tsc = 256;   /* MUST match TSR/TSC in gemm_i4_tile.comp */
        { const char *e = getenv("COLI_VK_TSR"); if (e && *e) tsr = atoi(e);
          const char *f = getenv("COLI_VK_TSC"); if (f && *f) tsc = atoi(f); }
        uint32_t rt = (uint32_t)((n + tsr - 1) / tsr);
        uint32_t ot = (uint32_t)((O + tsc - 1) / tsc);
        vkCmdDispatch(v->cmd, rt*ot, 1, 1);
        return;
    }
    /* SHORT ROWS (I <= 1024, e.g. an MoE expert's down_proj) at small n: 8 lanes per
     * row and 8 rows per workgroup instead of 16x4. Measured 2026-09-08 (bench_gemv,
     * 4070, 200 dispatches per fence): 768-wide rows 5.2-6.3 -> 2.98 us; 2048-wide
     * rows get slower with 8 lanes, so the default stays for them. */
    int outs_override = 0;
    if (pipe == v->pipe4 && v->pipe4s && !v->short_rows_off && I <= 1024) { pipe = v->pipe4s; outs_override = 8; }
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    int tile = v->tile; if (tile < 1) tile = 1;
    /* OUTS must equal the int4 shader's compile-time OUTS (default 4). COLI_VK_I4_OUTS
     * pairs with COLI_VK_I4_SPV for the batch-1 sweep; a mismatch computes the
     * wrong rows silently, which is why the bench checks every result. */
    static int i4_outs = -1;
    if (i4_outs < 0) { const char *e = getenv("COLI_VK_I4_OUTS"); i4_outs = (e && atoi(e) > 0) ? atoi(e) : 4; }
    int outs = outs_override ? outs_override : ((pipe == v->pipe4) ? i4_outs : 1);
    int32_t push[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), tile, outs };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,push);
    int64_t rtiles = ((int64_t)n + tile - 1) / tile;
    int64_t otiles = ((int64_t)O + outs - 1) / outs;
    vkCmdDispatch(v->cmd,(uint32_t)(rtiles*otiles),1,1);
}

/* Each stage reads what the previous wrote, so they must not overlap. One global
 * barrier is the blunt version of what ggml does per buffer; it is correct, and
 * with four stages the precision would buy nothing measurable. */
static void record_barrier(coli_vk *v) {
    VkMemoryBarrier mb = { .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}


/* dlbytes > 0 records a copy of yb into the staging buffer as the last thing in
 * this command buffer, so the readback costs no extra submission. 0 means the
 * caller reads yb directly (out_dev off, or an integrated part). */
static int gemm_on_device(coli_vk *v, VkPipeline pipe, vkbuf wbuf, vkbuf wsbuf,
                          int64_t I, int64_t O, int n,
                          vkbuf xb, vkbuf xs, vkbuf xm, vkbuf yb,
                          VkDeviceSize dlbytes) {
    if (!v->ds_ok) {
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
        if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds)!=VK_SUCCESS) return -1;
        v->ds_ok = 1;
    }
    VkDescriptorSet ds = v->ds;
    VkBuffer bufs[6] = { wbuf.buf, wsbuf.buf, xb.buf, xs.buf, xm.buf, yb.buf };
    VkDescriptorBufferInfo dbi[6]; VkWriteDescriptorSet wr[6];
    for (int i=0;i<6;i++) {
        dbi[i]=(VkDescriptorBufferInfo){ .buffer=bufs[i], .offset=0, .range=VK_WHOLE_SIZE };
        wr[i]=(VkWriteDescriptorSet){ .sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=ds, .dstBinding=(uint32_t)i, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[i] };
    }
    vkUpdateDescriptorSets(v->dev,6,wr,0,NULL);

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    /* BATCH TILE. One workgroup now covers COLI_VK_TILE_R rows of one output
     * instead of a single (row, output) pair, so a weight word is read once and
     * reused across the tile. Before this the dispatch was n*O workgroups and
     * each re-read the whole weight row: measured 0.07 ms at n=1 rising to
     * 28.79 ms at n=512 on an RTX 4070 (3584x3584) -- 0.056 ms/row flat from
     * n=8 up, i.e. no amortization whatever.
     *
     * The tile width is PUSHED, not compiled into the shader, so host and
     * shader cannot drift apart; the shader clamps it to its own MAXTILE, which
     * is the only compile-time quantity (it sizes the shared array). All three
     * pipelines -- i8, i4, i4f -- share this dispatch and MUST agree on the
     * (row-tile, o) meaning of gl_WorkGroupID.x. */
    /* Settable at runtime so the tile can be MEASURED rather than argued about,
     * and so tile=1 reproduces the pre-tile kernel exactly from the same binary
     * -- a cross-build comparison would confound the tile with everything else
     * that changed. Clamped to the shaders' MAXTILE. */
    /* Resolved ONCE at init, not here. This block used to call getenv+atoi on
     * every dispatch -- ~197 weight matrices x 681 tokens = ~134,000 scans of
     * environ per run -- which is a cost paid in the hot path to support a
     * debug knob. Measured 2026-08-20: see the A/B in the commit message. */
    /* This single-dispatch path is o_proj's, and it did NOT get the tensor cores
     * when record_gemm did -- it has its own dispatch and never went through it.
     * Measured 2026-08-24, once coopmat was live everywhere else: qkv submit fell
     * 463 -> 87 ms and ffn4 1192 -> 588 ms while THIS stayed at 288 ms and rose
     * from 14% of prefill to 28%. A path that quietly keeps the old kernel looks
     * exactly like a kernel that did not help. */
    if (pipe == v->pipe4 && coop_pipe(v) && v->coop_min_n > 0 && n >= v->coop_min_n) {
        vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,coop_pipe(v));
        vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
        int32_t pc[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), 0, 0 };
        vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,pc);
        vkCmdDispatch(v->cmd,(uint32_t)(((n + COOP_TSR - 1)/COOP_TSR) *
                                        ((O + COOP_TSC - 1)/COOP_TSC)),1,1);
        P.coop_n++; if (coop_pipe(v) == v->pipe4cd) P.coop_ds_n++;
    } else {
    int tile = v->tile;
    /* The int4 shaders run one 16-lane CLUSTER per output, so a 64-thread
     * workgroup covers 64/16 = 4 outputs. i8 and i4f are unchanged and take
     * outs=1; the geometry is keyed off the bound pipeline because a mismatch
     * computes the wrong outputs silently rather than failing. */
    int outs = (pipe == v->pipe4) ? 4 : 1;
    int32_t push[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), tile, outs };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,push);
    int64_t rtiles = ((int64_t)n + tile - 1) / tile;
    int64_t otiles = ((int64_t)O + outs - 1) / outs;
    vkCmdDispatch(v->cmd,(uint32_t)(rtiles*otiles),1,1);
    }
    if (dlbytes) record_copy_out(v,&yb,0,dlbytes);
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.gemm_n++; P.cur_op = 0;
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    uint64_t tsub = now_ns();
    if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    { uint64_t d = now_ns()-tsub; P.sub_ns += d; P.sub_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.sub_ns_op[oi]+=d; P.sub_n_op[oi]++; } }
    return 0;
}

static int gemm_dispatch(coli_vk *v, VkPipeline pipe, vkbuf wbuf, vkbuf wsbuf,
                         int64_t I, int64_t O, const coli_a_i8 *a, float *y) {
    int64_t nb=I/COLI_ABLK;
    int n=a->n;
    if (a->I != I) return -1;

    /* Persistent, grown to the high-water mark. Was: allocate 4 buffers, upload,
     * dispatch, download, destroy 4 buffers -- every call. */
    size_t ybytes = (size_t)n*O*4;
    if (!ensure(v,&v->xb,(size_t)n*I) || !ensure(v,&v->xs,(size_t)n*nb*4) ||
        !ensure(v,&v->xm,(size_t)n*nb*4) || !ensure_out(v,&v->yb,(size_t)COOP_ROW_PAD(n)*O*4)) return -1;
    if (!ensure_stage(v,ybytes)) return -1;
    vkbuf xb=v->xb, xs=v->xs, xm=v->xm, yb=v->yb;
    if (!upload(v,&xb,a->q,(size_t)n*I)) return -1;
    if (!upload(v,&xs,a->scale,(size_t)n*nb*4)) return -1;
    if (!upload(v,&xm,a->sum,(size_t)n*nb*4)) return -1;
    if (gemm_on_device(v,pipe,wbuf,wsbuf,I,O,n,xb,xs,xm,yb,
                       v->out_dev ? ybytes : 0)!=0) return -1;
    download_out(v,&yb,0,y,ybytes);
    return 0;
}

int coli_vk_gemm(coli_vk *v, int wh, const coli_a_i8 *a, float *y) {
    if (wh<0 || wh>=v->nw || !v->W[wh].used) return -1;
    return gemm_dispatch(v, v->pipe, v->W[wh].w, v->W[wh].ws,
                         v->W[wh].I, v->W[wh].O, a, y);
}

/* int4 upload. Same bytes coli_quantize_w4 produced, no repacking -- and the
 * scale array is per BLOCK here (O * I/32 floats) where int8's is per row (O),
 * which is the only shape difference between the two uploads. */
int coli_vk_has_mx(coli_vk *v){ return v && v->pipe4mx != VK_NULL_HANDLE; }
int coli_vk_slot_alloc_mx(coli_vk *v, int64_t I, int64_t O) {
    if (!coli_vk_has_mx(v) || v->nw4 >= MAX_W || (I % COLI_W4BLK)) return -1;
    int h = v->nw4;
    size_t wn = (size_t)I*O/2, sn = (size_t)O*(I/COLI_W4BLK)*sizeof(float);
    int dl = (!v->integrated || v->want_device_local) && !v->force_host_visible;
    VkMemoryPropertyFlags mp = dl ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                  : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    /* CONCURRENT sharing only when a fill can actually land from a DIFFERENT
     * queue family (fill_mode==1) -- see the long comment on mkbuf_flags_ex.
     * fill_mode==2 and fill_mode==0 both stay EXCLUSIVE: same family either
     * way, so there is no ownership boundary to cross. */
    VkSharingMode sm = (v->fill_mode==1) ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    uint32_t fams[2] = { v->qfam, v->xfam }, nfam = (v->fill_mode==1) ? 2 : 0;
    if (!mkbuf_flags_ex(v, wn, &v->W4[h].w, mp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, sm, fams, nfam)) return -1;
    if (!mkbuf_flags_ex(v, sn, &v->W4[h].ws, mp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, sm, fams, nfam)) { freebuf(v,&v->W4[h].w); return -1; }
    v->W4[h].I = I; v->W4[h].O = O; v->W4[h].used = 1; v->W4[h].mx = 1; v->W4[h].dl = dl;
    v->nw4++;
    return h;
}

/* Staged copy through its OWN staging buffer + its own record/submit/wait,
 * timed in three pieces -- this is what coli_vk_slot_fill calls now instead of
 * upload_device_local, which never timed the submit or the wait at all. Same
 * Vulkan calls upload_device_local made (mkbuf_flags for staging, one
 * vkCmdCopyBuffer, submit, wait, free), so the SYNC path's effect is
 * unchanged; only the instrumentation is new. HOST_VISIBLE (non-DL) slots
 * keep calling plain upload() below, unmodified -- there is no submit/fence
 * to split there, so the breakdown would be all memcpy and nothing else. */
static int slot_fill_sync_timed(coli_vk *v, vkbuf *dst, const void *src, size_t n) {
    vkbuf stage = {0};
    if (!mkbuf_flags(v, n, &stage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return 0;
    uint64_t t0 = now_ns();
    void *p; if (vkMapMemory(v->dev,stage.mem,0,n,0,&p) != VK_SUCCESS) { freebuf(v,&stage); return 0; }
    memcpy(p,src,n); vkUnmapMemory(v->dev,stage.mem);
    uint64_t t1 = now_ns(); FSYNC.memcpy_ns += t1-t0;

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    VkBufferCopy cp = { .srcOffset=0, .dstOffset=0, .size=n };
    vkCmdCopyBuffer(v->cmd, stage.buf, dst->buf, 1, &cp);
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    uint64_t t2 = now_ns();
    int subok = vkQueueSubmit(v->q,1,&si,v->fence)==VK_SUCCESS;
    uint64_t t3 = now_ns(); FSYNC.recsub_ns += t3-t2;
    int waitok = subok && vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)==VK_SUCCESS;
    uint64_t t4 = now_ns(); FSYNC.wait_ns += t4-t3;
    freebuf(v,&stage);
    return subok && waitok;
}

int coli_vk_slot_fill(coli_vk *v, int h, const coli_w_i4 *w) {
    if (!v || h < 0 || h >= v->nw4 || !v->W4[h].used || !v->W4[h].mx) return -1;
    if (w->I != v->W4[h].I || w->O != v->W4[h].O) return -1;
    size_t wn = (size_t)w->I*w->O/2, sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);
    int ok;
    if (v->W4[h].dl) ok = slot_fill_sync_timed(v,&v->W4[h].w,w->q4,wn) && slot_fill_sync_timed(v,&v->W4[h].ws,w->bscale,sn);
    else { P.in_weight_upload = 1; ok = upload(v,&v->W4[h].w,w->q4,wn) && upload(v,&v->W4[h].ws,w->bscale,sn); P.in_weight_upload = 0; }
    FSYNC.n++;
    return ok ? 0 : -1;
}

/* ---- async fill ring: grow-to-fit, lazily, like the upload ring above ---- */
static int fill_slot_wait_one(coli_vk *v, fill_slot *s) {
    if (!s->pending) return 1;
    uint64_t t0 = now_ns();
    int ok = vkWaitForFences(v->dev,1,&s->fence,VK_TRUE,60000000000ull)==VK_SUCCESS;
    FASYNC.wait_ns += now_ns()-t0;
    s->pending = 0;
    return ok;
}

/* Ensure every ring slot has a staging buffer of at least `need` bytes, a
 * command buffer and a fence. Called on every coli_vk_slot_fill_async; cheap
 * after the first call to a given shape because `need` never changes for a
 * fixed (I,O) expert shape, so only the FIRST call per process actually
 * allocates. Growing (a different, larger shape later) drains every pending
 * fill first -- the old, smaller staging buffers are about to be freed out
 * from under any fence still in flight otherwise. */
static int fill_ring_ensure(coli_vk *v, VkDeviceSize need) {
    if (v->fslot_cap >= need && v->fslots[0].buf.buf) return 1;
    for (int i=0;i<COLI_VK_FILL_INFLIGHT;i++) fill_slot_wait_one(v,&v->fslots[i]);
    for (int i=0;i<COLI_VK_FILL_INFLIGHT;i++) {
        fill_slot *s = &v->fslots[i];
        if (s->buf.buf) { if (s->map) vkUnmapMemory(v->dev,s->buf.mem); freebuf(v,&s->buf); s->map=NULL; }
        if (!mkbuf_flags(v, need, &s->buf,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return 0;
        if (vkMapMemory(v->dev,s->buf.mem,0,need,0,&s->map) != VK_SUCCESS) { freebuf(v,&s->buf); return 0; }
        if (!s->cmd) {
            VkCommandBufferAllocateInfo cbi = { .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool=v->fpool, .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount=1 };
            if (vkAllocateCommandBuffers(v->dev,&cbi,&s->cmd) != VK_SUCCESS) return 0;
        }
        if (!s->fence) {
            VkFenceCreateInfo fci = { .sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            if (vkCreateFence(v->dev,&fci,NULL,&s->fence) != VK_SUCCESS) return 0;
        }
    }
    v->fslot_cap = need;
    return 1;
}

int coli_vk_slot_fill_async(coli_vk *v, int h, const coli_w_i4 *w) {
    if (!v || h < 0 || h >= v->nw4 || !v->W4[h].used || !v->W4[h].mx) return -1;
    if (w->I != v->W4[h].I || w->O != v->W4[h].O) return -1;
    if (v->fill_mode == 0) { int r = coli_vk_slot_fill(v,h,w); if (r==0) FASYNC.n++; return r; }
    if (!v->W4[h].dl) { int r = coli_vk_slot_fill(v,h,w); if (r==0) FASYNC.n++; return r; }  /* no PCIe crossing to hide on a HOST_VISIBLE slot */

    size_t wn = (size_t)w->I*w->O/2, sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);
    VkDeviceSize need = (VkDeviceSize)(wn+sn);
    if (!fill_ring_ensure(v, need)) return -1;

    fill_slot *s = &v->fslots[v->fslot_next];
    if (!fill_slot_wait_one(v,s)) return -1;   /* ring full: back-pressure, as specified */

    uint64_t t0 = now_ns();
    memcpy(s->map, w->q4, wn);
    memcpy((char*)s->map + wn, w->bscale, sn);
    uint64_t t1 = now_ns(); FASYNC.memcpy_ns += t1-t0;

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(s->cmd,0);
    vkBeginCommandBuffer(s->cmd,&bi);
    VkBufferCopy cpw = { .srcOffset=0,  .dstOffset=0, .size=wn };
    vkCmdCopyBuffer(s->cmd, s->buf.buf, v->W4[h].w.buf,  1, &cpw);
    VkBufferCopy cps = { .srcOffset=wn, .dstOffset=0, .size=sn };
    vkCmdCopyBuffer(s->cmd, s->buf.buf, v->W4[h].ws.buf, 1, &cps);
    vkEndCommandBuffer(s->cmd);
    vkResetFences(v->dev,1,&s->fence);
    VkSubmitInfo si = { .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&s->cmd };
    int ok = vkQueueSubmit(v->fq,1,&si,s->fence)==VK_SUCCESS;
    uint64_t t2 = now_ns(); FASYNC.recsub_ns += t2-t1;
    s->pending = ok;
    v->fslot_next = (v->fslot_next + 1) % COLI_VK_FILL_INFLIGHT;
    if (ok) FASYNC.n++;
    return ok ? 0 : -1;
}

/* The ONLY place that makes an async fill's writes visible to the compute
 * queue. Waiting for a fence and observing it signaled guarantees (Vulkan
 * 7.x, host write/fence ordering) that the copy's writes are available to any
 * work the HOST subsequently submits, on any queue -- so no pipeline barrier
 * is needed here on top of the CONCURRENT sharing mode chosen in
 * coli_vk_slot_alloc_mx: CONCURRENT removes the ownership-transfer
 * requirement, this wait removes the visibility-timing one, and together
 * they are the two separate guarantees a cross-queue write needs. A caller
 * that dispatches a GEMM over a slot without calling this first is reading
 * memory the device may not have finished writing, which is exactly the bug
 * the test's deliberate control below is built to catch. */
int coli_vk_slot_fill_wait(coli_vk *v) {
    if (!v) return -1;
    int ok = 1;
    for (int i=0;i<COLI_VK_FILL_INFLIGHT;i++) if (!fill_slot_wait_one(v,&v->fslots[i])) ok = 0;
    return ok ? 0 : -1;
}

const char *coli_vk_fill_mode(coli_vk *v) { return (v && v->fill_mode_desc[0]) ? v->fill_mode_desc : "unknown"; }

void coli_vk_fill_stats(coli_vk *v, int which, double out[4]) {
    (void)v;
    out[0]=out[1]=out[2]=out[3]=0.0;
    uint64_t memcpy_ns, recsub_ns, wait_ns, n;
    if (which) { memcpy_ns=FASYNC.memcpy_ns; recsub_ns=FASYNC.recsub_ns; wait_ns=FASYNC.wait_ns; n=FASYNC.n; }
    else       { memcpy_ns=FSYNC.memcpy_ns;  recsub_ns=FSYNC.recsub_ns;  wait_ns=FSYNC.wait_ns;  n=FSYNC.n; }
    if (!n) return;
    out[0] = (double)memcpy_ns/1e6/(double)n;
    out[1] = (double)recsub_ns/1e6/(double)n;
    out[2] = (double)wait_ns/1e6/(double)n;
    out[3] = out[0]+out[1]+out[2];
}
int coli_vk_upload_w4_mx(coli_vk *v, const coli_w_i4 *w) {
    if (!coli_vk_has_mx(v)) return -1;
    int h = coli_vk_upload_w4(v, w);
    if (h >= 0) v->W4[h].mx = 1;
    return h;
}
int coli_vk_upload_w4(coli_vk *v, const coli_w_i4 *w) {
    if (!v->pipe4) return -1;             /* no int4 shader built; caller stays on int8 */
    v->W4[v->nw4].mx = 0;                 /* plain int4 unless coli_vk_upload_w4_mx tags it */
    if (v->nw4 >= MAX_W) return -1;
    if (w->I % COLI_W4BLK) return -1;     /* the shader indexes whole blocks */
    int h = v->nw4;
    size_t wn = (size_t)w->I*w->O/2;
    size_t sn = (size_t)w->O*(w->I/COLI_W4BLK)*sizeof(float);

    /* Same opt-in as the int8 path above. This is the path `--gpu --w4 2`
     * actually takes, and before this it had NO memdesc2 at all -- there was no
     * way to tell from the outside which heap the int4 weights landed in. */
    if ((!v->integrated || v->want_device_local) && !v->force_host_visible) {
        int okw = mkbuf_flags(v, wn, &v->W4[h].w, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        int oks = okw && mkbuf_flags(v, sn, &v->W4[h].ws, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (oks && upload_device_local(v,&v->W4[h].w,w->q4,wn)
                && upload_device_local(v,&v->W4[h].ws,w->bscale,sn)) {
            if (!v->memdesc2[0]) {
                snprintf(v->memdesc2,sizeof v->memdesc2,"DEVICE_LOCAL (staged)");
                uint32_t mt = find_mem(v, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (mt != UINT32_MAX) describe_mem(v, mt, v->memdesc, sizeof v->memdesc);
            }
            v->W4[h].I=w->I; v->W4[h].O=w->O; v->W4[h].used=1; v->nw4++;
            return h;
        }
        freebuf(v,&v->W4[h].w); freebuf(v,&v->W4[h].ws);
    }
    if (!mkbuf(v,wn,&v->W4[h].w))  return -1;
    if (!mkbuf(v,sn,&v->W4[h].ws)) { freebuf(v,&v->W4[h].w); return -1; }
    /* Flag this as WEIGHT traffic. Only upload_device_local used to set it, so
     * on the HOST_VISIBLE path -- i.e. every integrated GPU -- the one-time
     * weight load landed in the PER-TOKEN upload bucket instead of the one-time
     * staging line. Measured 2026-08-20: the Legion's upload bucket read
     * 4510.1 MiB against the desktop's 296.8 MiB for the SAME 174,049 calls,
     * because ~4.2 GiB of one-time weights were counted as per-token traffic.
     * The two hosts' profiles were not comparable and nothing said so. */
    P.in_weight_upload = 1;
    int okq = upload(v,&v->W4[h].w, w->q4,     wn)
           && upload(v,&v->W4[h].ws,w->bscale, sn);
    P.in_weight_upload = 0;
    if (!okq) return -1;
    if (!v->memdesc2[0]) snprintf(v->memdesc2,sizeof v->memdesc2,"HOST_VISIBLE");
    v->W4[h].I=w->I; v->W4[h].O=w->O; v->W4[h].used=1;
    v->nw4++;
    return h;
}

int coli_vk_gemm4(coli_vk *v, int wh, const coli_a_i8 *a, float *y) {
    if (!v->pipe4) return -1;
    if (wh<0 || wh>=v->nw4 || !v->W4[wh].used) return -1;
    if (v->W4[wh].mx) {            /* MXFP4 handle: the int4 decode would read e2m1 codes as biased ints */
        if (!v->pipe4mx) return -1;
        return gemm_dispatch(v, v->pipe4mx, v->W4[wh].w, v->W4[wh].ws, v->W4[wh].I, v->W4[wh].O, a, y);
    }
    return gemm_dispatch(v, v->pipe4, v->W4[wh].w, v->W4[wh].ws,
                         v->W4[wh].I, v->W4[wh].O, a, y);
}

int coli_vk_has_i4(coli_vk *v) { return v && v->pipe4 != VK_NULL_HANDLE; }
int coli_vk_has_i4f(coli_vk *v){ return v && v->pipe4f != VK_NULL_HANDLE; }
int coli_vk_has_ffn(coli_vk *v){ return v && v->pipe_smq != VK_NULL_HANDLE && v->pipe4; }

/* A whole SwiGLU FFN with ONE upload and ONE download.
 *
 * x -> [gate] -> g          three GEMMs and the nonlinearity between them, with
 * x -> [up]   -> u          every intermediate staying in device memory. The
 * silu(g)*u -> quantize     old path was three coli_vk_gemm4 calls: three
 *          -> [down] -> y   uploads, three downloads, and a CPU requantization
 *                           of the intermediate in the middle.
 *
 * Handles are int4 handles (coli_vk_upload_w4). n is the batch. */
int coli_vk_ffn4(coli_vk *v, int hg, int hu, int hd, const coli_a_i8 *a, float *y) {
    if (!coli_vk_has_ffn(v)) return -1;
    if (hg<0||hg>=v->nw4||hu<0||hu>=v->nw4||hd<0||hd>=v->nw4) return -1;
    if (!v->W4[hg].used||!v->W4[hu].used||!v->W4[hd].used) return -1;
    if (v->W4[hg].mx||v->W4[hu].mx||v->W4[hd].mx) return -1;   /* MXFP4 handles: coli_vk_ffn4_oai only */
    int64_t D = v->W4[hg].I, EI = v->W4[hg].O, Dout = v->W4[hd].O;
    if (a->I != D || v->W4[hu].I != D || v->W4[hu].O != EI || v->W4[hd].I != EI) return -1;
    if (EI % COLI_ABLK) return -1;
    int n = a->n;
    int64_t nbD = D/COLI_ABLK, nbE = EI/COLI_ABLK;

    size_t ybytes = (size_t)n*Dout*4;
    if (!ensure(v,&v->xb,(size_t)n*D) || !ensure(v,&v->xs,(size_t)n*nbD*4) ||
        !ensure(v,&v->xm,(size_t)n*nbD*4) || !ensure_out(v,&v->yb,(size_t)COOP_ROW_PAD(n)*Dout*4)) return -1;
    if (!ensure_stage(v,ybytes)) return -1;
    /* DEVICE_LOCAL: the CPU never reads or writes these. See mkbuf_dev. */
    if (!ensure_dev(v,&v->fg,(size_t)n*EI*4) || !ensure_dev(v,&v->fu,(size_t)n*EI*4) ||
        !ensure_dev(v,&v->hq,(size_t)n*EI)   || !ensure_dev(v,&v->hs,(size_t)n*nbE*4) ||
        !ensure_dev(v,&v->hm,(size_t)n*nbE*4)) return -1;

    /* the ONE upload */
    if (!upload(v,&v->xb,a->q,(size_t)n*D)) return -1;
    if (!upload(v,&v->xs,a->scale,(size_t)n*nbD*4)) return -1;
    if (!upload(v,&v->xm,a->sum,(size_t)n*nbD*4)) return -1;

    /* Four dispatches, ONE submission, ONE fence wait. */
    if (!v->dsf_ok) {
        VkDescriptorSetLayout ls[4] = { v->dsl, v->dsl, v->dsl, v->dsl };
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=4, .pSetLayouts=ls };
        if (vkAllocateDescriptorSets(v->dev,&dsai,v->dsf)!=VK_SUCCESS) return -1;
        v->dsf_ok = 1;
    }
    write_set(v,v->dsf[0], v->W4[hg].w.buf, v->W4[hg].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fg.buf);
    write_set(v,v->dsf[1], v->W4[hu].w.buf, v->W4[hu].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fu.buf);
    write_set(v,v->dsf[2], v->fg.buf, v->fu.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->fg.buf);
    write_set(v,v->dsf[3], v->W4[hd].w.buf, v->W4[hd].ws.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->yb.buf);

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    record_gemm(v,v->pipe4,v->dsf[0],D,EI,n);
    record_gemm(v,v->pipe4,v->dsf[1],D,EI,n);
    record_barrier(v);
    {   /* the nonlinearity: one invocation per 16-element block */
        uint32_t blocks = (uint32_t)((int64_t)n*nbE);
        record_dispatch(v,v->pipe_smq,v->dsf[2],EI,0,n,(blocks+63)/64);
    }
    record_barrier(v);
    record_gemm(v,v->pipe4,v->dsf[3],EI,Dout,n);
    if (v->out_dev) record_copy_out(v,&v->yb,0,ybytes);
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.ffn_n++; P.cur_op = 1;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    { uint64_t d = now_ns()-tsub; P.sub_ns += d; P.sub_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.sub_ns_op[oi]+=d; P.sub_n_op[oi]++; } }

    /* the ONE download */
    download_out(v,&v->yb,0,y,ybytes);
    return 0;
}

/* GROUPED-EXPERT decode (#2b): nexp experts, ONE shared activation, ONE submission.
 *
 * #2a's fused ffn4 already runs a whole expert in one submit, but at decode the K
 * selected experts all consume the SAME token, so K separate ffn4 calls upload
 * that token K times and pay K submit+fence round trips (~14us each x 48 layers).
 * Here the token uploads ONCE and all nexp experts' four-dispatch FFNs record into
 * ONE command buffer -- serialized on the shared fg/fu/hq scratch with barriers,
 * which is fine at n=1 where each dispatch is tiny and the win is the eliminated
 * per-expert fence, not intra-expert parallelism. Each expert's down-projection
 * writes its own COOP_ROW_PAD(n)*Dout slice of ymoe via a descriptor offset; y
 * receives nexp*n*Dout floats, expert-major. All experts must share D/EI/Dout
 * (always true within an MoE layer).
 *
 * Discrete GPU only (out_dev): the per-expert readback offset indexes the staging
 * buffer, which only exists when out_dev. On UMA the caller keeps #2a per-expert,
 * where there is no staging cost to amortise anyway. Returns -1 on any not-ready
 * condition so the caller falls back; never a partial result. */
/* Split submission so the CPU can do its own experts while the GPU runs these.
 * _begin records and SUBMITS (no wait); _end WAITS and downloads. Between the two
 * the caller must issue no other Vulkan call on this context -- there is one
 * command buffer and one fence, and a second op would clobber both. _begin
 * refuses (-1) if a submission is already pending. coli_vk_moe4 is begin+end and
 * behaves exactly as before. Added 2026-09-06 after the h2h found the hybrid
 * decode (19.0 tok/s) SLOWER than CPU-only (26.6): resident experts and CPU
 * experts ran back to back, never at the same time. */
static struct { int active, nexp, n; int64_t ystride, Dout; uint64_t tsub; } g_moe_pend;

/* One dispatch over `nmat` same-shape matrices with the multi-matrix kernel; the
 * set's binding 4 holds the address table. Geometry per matrix is record_gemm's
 * decode geometry (tile rows x outs outputs), short-row variant for I <= 1024
 * exactly as record_gemm chooses it, so results match the per-matrix path bit
 * for bit. */
static void record_gemm_multi(coli_vk *v, VkDescriptorSet ds, int nmat, int64_t I, int64_t O, int n) {
    VkPipeline pipe = v->pipe4m; int outs = 4;
    if (v->pipe4ms && !v->short_rows_off && I <= 1024) { pipe = v->pipe4ms; outs = 8; }
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    int tile = v->tile; if (tile < 1) tile = 1;
    int32_t push[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), tile, outs };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,push);
    int64_t rtiles = ((int64_t)n + tile - 1) / tile;
    int64_t otiles = ((int64_t)O + outs - 1) / outs;
    vkCmdDispatch(v->cmd,(uint32_t)((int64_t)nmat*rtiles*otiles),1,1);
}

int coli_vk_has_ffn_oai(coli_vk *v){ return v && v->pipe4mx && v->pipe_swq && coli_vk_has_ffn(v); }
/* Bind + push + dispatch one MXFP4 GEMV (decode geometry only: no coop/tile
 * paths -- those are batch kernels and this is the n=1 expert path). Mirrors the
 * tail of record_gemm with pipe4mx / pipe4mxs in place of pipe4 / pipe4s. */
static void record_gemm_mx(coli_vk *v, VkDescriptorSet ds, int64_t I, int64_t O, int n) {
    VkPipeline pipe = v->pipe4mx; int outs = 4;
    static int i4_outs = -1;
    if (i4_outs < 0) { const char *e = getenv("COLI_VK_I4_OUTS"); i4_outs = (e && atoi(e) > 0) ? atoi(e) : 4; }
    outs = i4_outs;
    if (v->pipe4mxs && !v->short_rows_off && I <= 1024) { pipe = v->pipe4mxs; outs = 8; }
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    int tile = v->tile; if (tile < 1) tile = 1;
    int32_t push[6] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), tile, outs };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,push);
    int64_t rtiles = ((int64_t)n + tile - 1) / tile;
    int64_t otiles = ((int64_t)O + outs - 1) / outs;
    vkCmdDispatch(v->cmd,(uint32_t)(rtiles*otiles),1,1);
}
int coli_vk_ffn4_oai(coli_vk *v, int hg, int hu, int hd, const coli_a_i8 *a, float *y,
                     const float *bg, const float *bu, float alpha, float limit) {
    if (!coli_vk_has_ffn_oai(v)) return -1;
    if (hg<0||hg>=v->nw4||hu<0||hu>=v->nw4||hd<0||hd>=v->nw4) return -1;
    if (!v->W4[hg].used||!v->W4[hu].used||!v->W4[hd].used) return -1;
    if (!v->W4[hg].mx||!v->W4[hu].mx||!v->W4[hd].mx) return -1;
    int64_t D = v->W4[hg].I, EI = v->W4[hg].O, Dout = v->W4[hd].O;
    if (a->I != D || v->W4[hu].I != D || v->W4[hu].O != EI || v->W4[hd].I != EI) return -1;
    if (EI % COLI_ABLK) return -1;
    int n = a->n;
    int64_t nbD = D/COLI_ABLK, nbE = EI/COLI_ABLK;

    size_t ybytes = (size_t)n*Dout*4;
    if (!ensure(v,&v->xb,(size_t)n*D) || !ensure(v,&v->xs,(size_t)n*nbD*4) ||
        !ensure(v,&v->xm,(size_t)n*nbD*4) || !ensure_out(v,&v->yb,(size_t)COOP_ROW_PAD(n)*Dout*4)) return -1;
    if (!ensure_stage(v,ybytes)) return -1;
    if (!ensure_dev(v,&v->fg,(size_t)n*EI*4) || !ensure_dev(v,&v->fu,(size_t)n*EI*4) ||
        !ensure_dev(v,&v->hq,(size_t)n*EI)   || !ensure_dev(v,&v->hs,(size_t)n*nbE*4) ||
        !ensure_dev(v,&v->hm,(size_t)n*nbE*4)) return -1;
    if (!ensure(v,&v->fbias,(size_t)2*EI*4)) return -1;

    if (!upload(v,&v->xb,a->q,(size_t)n*D)) return -1;
    if (!upload(v,&v->xs,a->scale,(size_t)n*nbD*4)) return -1;
    if (!upload(v,&v->xm,a->sum,(size_t)n*nbD*4)) return -1;
    { /* gate bias then up bias, one small upload; offsets go in the push constants */
      void *pm; if (vkMapMemory(v->dev,v->fbias.mem,0,(size_t)2*EI*4,0,&pm)!=VK_SUCCESS) return -1;
      memcpy(pm, bg, (size_t)EI*4); memcpy((char*)pm+(size_t)EI*4, bu, (size_t)EI*4);
      vkUnmapMemory(v->dev,v->fbias.mem); }

    if (!v->dsf_ok) {
        VkDescriptorSetLayout ls[4] = { v->dsl, v->dsl, v->dsl, v->dsl };
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=4, .pSetLayouts=ls };
        if (vkAllocateDescriptorSets(v->dev,&dsai,v->dsf)!=VK_SUCCESS) return -1;
        v->dsf_ok = 1;
    }
    write_set(v,v->dsf[0], v->W4[hg].w.buf, v->W4[hg].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fg.buf);
    write_set(v,v->dsf[1], v->W4[hu].w.buf, v->W4[hu].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fu.buf);
    write_set(v,v->dsf[2], v->fg.buf, v->fu.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->fbias.buf);
    write_set(v,v->dsf[3], v->W4[hd].w.buf, v->W4[hd].ws.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->yb.buf);

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    record_gemm_mx(v,v->dsf[0],D,EI,n);
    record_gemm_mx(v,v->dsf[1],D,EI,n);
    record_barrier(v);
    {   /* swiglu_oai + quantize: 8 push words -- I,O,n,nblk, bg_off, bu_off, alpha, limit */
        uint32_t blocks = (uint32_t)((int64_t)n*nbE);
        vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_swq);
        vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->dsf[2],0,NULL);
        struct { int32_t I,O,n,nblk,bg,bu; float alpha,limit; } pc = { (int32_t)EI, 0, n, (int32_t)nbE, 0, (int32_t)EI, alpha, limit };
        vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&pc);
        vkCmdDispatch(v->cmd,(blocks+63)/64,1,1);
    }
    record_barrier(v);
    record_gemm_mx(v,v->dsf[3],EI,Dout,n);
    if (v->out_dev) record_copy_out(v,&v->yb,0,ybytes);
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.ffn_n++; P.cur_op = 1;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    { uint64_t d = now_ns()-tsub; P.sub_ns += d; P.sub_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.sub_ns_op[oi]+=d; P.sub_n_op[oi]++; } }
    download_out(v,&v->yb,0,y,ybytes);
    return 0;
}

int coli_vk_moe4_begin(coli_vk *v, const int *hg, const int *hu, const int *hd,
                       int nexp, const coli_a_i8 *a) {
    if (g_moe_pend.active) return -1;
    for (int e=0;e<nexp;e++) if (hg[e]>=0 && hg[e]<v->nw4 && v->W4[hg[e]].mx) return -1;   /* MXFP4: not this path */
    /* out_dev is no longer required: the host-visible readback honors the slice
     * offset (download_out). With the default env the old gate made this whole
     * function decline silently -- measured 2026-09-07: 0 layers overlapped. */
    if (!coli_vk_has_ffn(v) || nexp <= 0) return -1;
    for (int e=0;e<nexp;e++) {
        if (hg[e]<0||hg[e]>=v->nw4||hu[e]<0||hu[e]>=v->nw4||hd[e]<0||hd[e]>=v->nw4) return -1;
        if (!v->W4[hg[e]].used||!v->W4[hu[e]].used||!v->W4[hd[e]].used) return -1;
    }
    int64_t D = v->W4[hg[0]].I, EI = v->W4[hg[0]].O, Dout = v->W4[hd[0]].O;
    for (int e=0;e<nexp;e++)
        if (v->W4[hg[e]].I!=D||v->W4[hg[e]].O!=EI||v->W4[hu[e]].I!=D||v->W4[hu[e]].O!=EI||
            v->W4[hd[e]].I!=EI||v->W4[hd[e]].O!=Dout) return -1;
    if (a->I != D || (EI % COLI_ABLK)) return -1;
    int n = a->n;
    int64_t nbD = D/COLI_ABLK, nbE = EI/COLI_ABLK;
    int64_t ystride = COOP_ROW_PAD(n)*Dout;             /* floats per expert slice */

    if (!ensure(v,&v->xb,(size_t)n*D) || !ensure(v,&v->xs,(size_t)n*nbD*4) ||
        !ensure(v,&v->xm,(size_t)n*nbD*4)) return -1;
    /* PER-EXPERT SCRATCH SLICES (2026-09-08). Before this every expert reused one
     * fg/fu/hq/hs/hm, which forced a barrier after each expert's down-projection:
     * 8 experts became 32 dispatches with 24 barriers, each stage a ~0.8 MB GEMV
     * that cannot fill the GPU on its own (bench_gemv: 3.7 us for gate/up, ~2.4 us
     * of it a fixed per-dispatch floor). With a slice per expert the call is THREE
     * stages -- all gate+up, barrier, all swiglu, barrier, all down -- and the
     * 16 gate/up dispatches run concurrently. COLI_MOE_SERIAL=1 restores the old
     * order on the same binary as the A/B control. Slices are padded to 256 B so
     * the descriptor offsets satisfy any minStorageBufferOffsetAlignment. */
    const VkDeviceSize sfg = ((VkDeviceSize)n*EI*4   + 255) & ~(VkDeviceSize)255;
    const VkDeviceSize shq = ((VkDeviceSize)n*EI     + 255) & ~(VkDeviceSize)255;
    const VkDeviceSize shs = ((VkDeviceSize)n*nbE*4  + 255) & ~(VkDeviceSize)255;
    /* DEFAULT IS THE SERIAL ORDER. Measured 2026-09-08 (hyb9/hyb13, 4070, rank-major
     * 240 greedy, ABAB): staged 41.0/41.4/41.4/41.6 vs serial 43.1/43.5 tok/s -- the
     * per-call GPU saving (150.7 -> 116-138 us, only visible layer-major) is hidden
     * behind the CPU experts in the default mode and the staged order costs 4-5%
     * end to end there, cause not isolated. COLI_MOE_STAGED=1 opts in (needed by
     * COLI_MOE_FUSED); COLI_MOE_SERIAL=1 is honoured as the explicit control. */
    static int moe_serial = -1;
    if (moe_serial < 0) { const char *e = getenv("COLI_MOE_SERIAL"), *g = getenv("COLI_MOE_STAGED");
                          moe_serial = (e && atoi(e)) ? 1 : ((g && atoi(g)) ? 0 : 1); }
    const int nsl = moe_serial ? 1 : nexp;      /* slices actually distinct */
    if (!ensure_dev(v,&v->fg,(size_t)nsl*sfg) || !ensure_dev(v,&v->fu,(size_t)nsl*sfg) ||
        !ensure_dev(v,&v->hq,(size_t)nsl*shq) || !ensure_dev(v,&v->hs,(size_t)nsl*shs) ||
        !ensure_dev(v,&v->hm,(size_t)nsl*shs)) return -1;
    if (!ensure_out(v,&v->ymoe,(size_t)nexp*ystride*4)) return -1;
    if (!ensure_stage(v,(size_t)nexp*ystride*4)) return -1;

    static int moe_fused = -1;
    if (moe_fused < 0) { const char *e = getenv("COLI_MOE_FUSED"); moe_fused = (e && atoi(e)) ? 1 : 0; }
    const int fused = moe_fused && !moe_serial && v->pipe4m && v->pfn_bda;
    int need = nexp*4 + 2;                 /* +2: the two multi-matrix stage sets */
    if (v->moe_ds_cap < need) {
        if (v->moe_pool) { vkDestroyDescriptorPool(v->dev,v->moe_pool,NULL); v->moe_pool=0; }
        free(v->moe_ds); v->moe_ds=NULL; v->moe_ds_cap=0;
        VkDescriptorPoolSize ps = { .type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount=(uint32_t)(6*need) };
        VkDescriptorPoolCreateInfo dpci = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets=(uint32_t)need, .poolSizeCount=1, .pPoolSizes=&ps };
        if (vkCreateDescriptorPool(v->dev,&dpci,NULL,&v->moe_pool)!=VK_SUCCESS) return -1;
        v->moe_ds = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet)*(size_t)need);
        VkDescriptorSetLayout *ls = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout)*(size_t)need);
        if (!v->moe_ds || !ls) { free(v->moe_ds); v->moe_ds=NULL; free(ls); return -1; }
        for (int i=0;i<need;i++) ls[i]=v->dsl;
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->moe_pool, .descriptorSetCount=(uint32_t)need, .pSetLayouts=ls };
        VkResult r = vkAllocateDescriptorSets(v->dev,&dsai,v->moe_ds);
        free(ls);
        if (r!=VK_SUCCESS) { free(v->moe_ds); v->moe_ds=NULL; return -1; }
        v->moe_ds_cap = need;
    }

    if (!upload(v,&v->xb,a->q,(size_t)n*D)) return -1;
    if (!upload(v,&v->xs,a->scale,(size_t)n*nbD*4)) return -1;
    if (!upload(v,&v->xm,a->sum,(size_t)n*nbD*4)) return -1;

    /* prepare every set BEFORE recording -- a set in use by the open command
     * buffer must not be re-written (the rule ffn4's four-set split obeys). */
    for (int e=0;e<nexp;e++) {
        VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
        const VkDeviceSize k = moe_serial ? 0 : (VkDeviceSize)e;   /* slice index */
        const VkDeviceSize ofg = k*sfg, ohq = k*shq, ohs = k*shs, oy = (VkDeviceSize)e*ystride*4;
        { VkBuffer b[6]={ v->W4[hg[e]].w.buf, v->W4[hg[e]].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fg.buf };
          VkDeviceSize o[6]={0,0,0,0,0,ofg}; write_set_offs(v,S[0],b,o); }
        { VkBuffer b[6]={ v->W4[hu[e]].w.buf, v->W4[hu[e]].ws.buf, v->xb.buf, v->xs.buf, v->xm.buf, v->fu.buf };
          VkDeviceSize o[6]={0,0,0,0,0,ofg}; write_set_offs(v,S[1],b,o); }
        { VkBuffer b[6]={ v->fg.buf, v->fu.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->fg.buf };
          VkDeviceSize o[6]={ofg,ofg,ohq,ohs,ohs,ofg}; write_set_offs(v,S[2],b,o); }
        { VkBuffer b[6]={ v->W4[hd[e]].w.buf, v->W4[hd[e]].ws.buf, v->hq.buf, v->hs.buf, v->hm.buf, v->ymoe.buf };
          VkDeviceSize o[6]={0,0,ohq,ohs,ohs,oy}; write_set_offs(v,S[3],b,o); }
    }

    /* MULTI-MATRIX STAGES: one dispatch for all gate+up (2*nexp matrices), one for
     * all down (nexp), addressed through a table of device addresses. Table A at
     * byte 0, table B at 4096 (a 256 B-aligned descriptor offset). */
    VkDescriptorSet SA = v->moe_ds[nexp*4], SB = v->moe_ds[nexp*4+1];
    if (fused) {
        if (!ensure(v,&v->moe_tab,8192)) return -1;
        uint64_t tabA[2*16*5], tabB[16*5];
        if (nexp > 16) return -1;
        VkBufferDeviceAddressInfo ai = { .sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        #define BDA(bufv) (ai.buffer=(bufv), v->pfn_bda(v->dev,&ai))
        uint64_t axb=BDA(v->xb.buf), axs=BDA(v->xs.buf), afg=BDA(v->fg.buf), afu=BDA(v->fu.buf),
                 ahq=BDA(v->hq.buf), ahs=BDA(v->hs.buf), ay=BDA(v->ymoe.buf);
        for (int e=0;e<nexp;e++) {
            const uint64_t k = moe_serial ? 0 : (uint64_t)e;
            uint64_t *g = tabA + (size_t)(2*e)*5, *u = tabA + (size_t)(2*e+1)*5, *d = tabB + (size_t)e*5;
            g[0]=BDA(v->W4[hg[e]].w.buf); g[1]=BDA(v->W4[hg[e]].ws.buf); g[2]=axb; g[3]=axs; g[4]=afg + k*sfg;
            u[0]=BDA(v->W4[hu[e]].w.buf); u[1]=BDA(v->W4[hu[e]].ws.buf); u[2]=axb; u[3]=axs; u[4]=afu + k*sfg;
            d[0]=BDA(v->W4[hd[e]].w.buf); d[1]=BDA(v->W4[hd[e]].ws.buf); d[2]=ahq + k*shq; d[3]=ahs + k*shs;
            d[4]=ay + (uint64_t)e*ystride*4;
        }
        #undef BDA
        { void *pm; if (vkMapMemory(v->dev,v->moe_tab.mem,0,8192,0,&pm)!=VK_SUCCESS) return -1;
          memcpy(pm, tabA, sizeof(uint64_t)*(size_t)(2*nexp)*5);
          memcpy((char*)pm+4096, tabB, sizeof(uint64_t)*(size_t)nexp*5);
          vkUnmapMemory(v->dev,v->moe_tab.mem); }
        { VkBuffer b[6]={ v->xb.buf, v->xs.buf, v->xb.buf, v->xs.buf, v->moe_tab.buf, v->fg.buf };
          VkDeviceSize o[6]={0,0,0,0,0,0};    write_set_offs(v,SA,b,o);
          VkDeviceSize o2[6]={0,0,0,0,4096,0}; write_set_offs(v,SB,b,o2); }
    }

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    if (fused) {
        record_gemm_multi(v, SA, 2*nexp, D, EI, n);
        record_barrier(v);
        for (int e=0;e<nexp;e++) { VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
            uint32_t blocks=(uint32_t)((int64_t)n*nbE); record_dispatch(v,v->pipe_smq,S[2],EI,0,n,(blocks+63)/64); }
        record_barrier(v);
        record_gemm_multi(v, SB, nexp, EI, Dout, n);
        record_barrier(v);
        P.moe_fused_n++;
    } else if (moe_serial) {
        for (int e=0;e<nexp;e++) {          /* pre-2026-09-08 order: the A/B control */
            VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
            record_gemm(v,v->pipe4,S[0],D,EI,n);
            record_gemm(v,v->pipe4,S[1],D,EI,n);
            record_barrier(v);
            { uint32_t blocks=(uint32_t)((int64_t)n*nbE); record_dispatch(v,v->pipe_smq,S[2],EI,0,n,(blocks+63)/64); }
            record_barrier(v);
            record_gemm(v,v->pipe4,S[3],EI,Dout,n);
            record_barrier(v);    /* next expert reuses fg/fu/hq */
        }
    } else {
        /* three stages, two barriers: every expert has its own scratch slice */
        for (int e=0;e<nexp;e++) { VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
            record_gemm(v,v->pipe4,S[0],D,EI,n); record_gemm(v,v->pipe4,S[1],D,EI,n); }
        record_barrier(v);
        for (int e=0;e<nexp;e++) { VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
            uint32_t blocks=(uint32_t)((int64_t)n*nbE); record_dispatch(v,v->pipe_smq,S[2],EI,0,n,(blocks+63)/64); }
        record_barrier(v);
        for (int e=0;e<nexp;e++) { VkDescriptorSet *S = v->moe_ds + (size_t)e*4;
            record_gemm(v,v->pipe4,S[3],EI,Dout,n); }
        record_barrier(v);        /* before the copy-out / host read */
    }
    if (v->out_dev) record_copy_out(v,&v->ymoe,0,(size_t)nexp*ystride*4);   /* no staging buffer otherwise */
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.ffn_n++; P.cur_op = 1;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    g_moe_pend.active=1; g_moe_pend.nexp=nexp; g_moe_pend.n=n;
    g_moe_pend.ystride=ystride; g_moe_pend.Dout=Dout; g_moe_pend.tsub=tsub;
    return 0;
}

int coli_vk_moe4_end(coli_vk *v, float *y) {
    if (!g_moe_pend.active) return -1;
    g_moe_pend.active = 0;
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    { uint64_t d=now_ns()-g_moe_pend.tsub; P.sub_ns+=d; P.sub_n++; int oi=P.cur_op; if(oi>=0&&oi<3){P.sub_ns_op[oi]+=d;P.sub_n_op[oi]++;} }
    const int nexp=g_moe_pend.nexp, n=g_moe_pend.n; const int64_t ystride=g_moe_pend.ystride, Dout=g_moe_pend.Dout;
    /* ONE staging copy above; scatter each expert's first n rows to y. */
    for (int e=0;e<nexp;e++) {
        if (v->out_dev) download_out(v,&v->ymoe,(size_t)e*ystride*4, y+(int64_t)e*n*Dout, (size_t)n*Dout*4);
        else            download_at (v,&v->ymoe,(size_t)e*ystride*4, y+(int64_t)e*n*Dout, (size_t)n*Dout*4);
    }
    return 0;
}

int coli_vk_moe4(coli_vk *v, const int *hg, const int *hu, const int *hd,
                 int nexp, const coli_a_i8 *a, float *y) {
    if (coli_vk_moe4_begin(v,hg,hu,hd,nexp,a) != 0) return -1;
    return coli_vk_moe4_end(v,y);
}

/* BENCH: `reps` back-to-back dispatches of the int4 GEMM on one weight handle in
 * ONE command buffer with a barrier between each, one submit, one fence. Returns
 * seconds for the whole submission, so per-dispatch cost is measured WITHOUT the
 * ~22-30 us submit+fence floor that makes a single 0.8 MB GEMV read as "0.03 ms"
 * in test_vk_gemm. y receives the last dispatch's result (for a correctness
 * check). Added 2026-09-08 for the batch-1 GEMV shader work. */
double coli_vk_bench_gemm4(coli_vk *v, int wh, const coli_a_i8 *a, float *y, int reps) {
    if (!v || !v->pipe4 || wh<0 || wh>=v->nw4 || !v->W4[wh].used || reps<1) return -1.0;
    int64_t I = v->W4[wh].I, O = v->W4[wh].O; int n = a->n;
    int64_t nb = I/COLI_ABLK;
    size_t ybytes = (size_t)n*O*4;
    if (!ensure(v,&v->xb,(size_t)n*I) || !ensure(v,&v->xs,(size_t)n*nb*4) ||
        !ensure(v,&v->xm,(size_t)n*nb*4) || !ensure_out(v,&v->yb,(size_t)COOP_ROW_PAD(n)*O*4)) return -1.0;
    if (!upload(v,&v->xb,a->q,(size_t)n*I) || !upload(v,&v->xs,a->scale,(size_t)n*nb*4) ||
        !upload(v,&v->xm,a->sum,(size_t)n*nb*4)) return -1.0;
    if (!v->ds_ok) {
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
        if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds)!=VK_SUCCESS) return -1.0;
        v->ds_ok = 1;
    }
    write_set(v,v->ds,v->W4[wh].w.buf,v->W4[wh].ws.buf,v->xb.buf,v->xs.buf,v->xm.buf,v->yb.buf);
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    for (int r=0;r<reps;r++) { record_gemm(v,v->pipe4,v->ds,I,O,n); record_barrier(v); }
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    uint64_t t0 = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1.0; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1.0;
    double secs = (double)(now_ns()-t0)*1e-9;
    if (y && !download(v,&v->yb,y,ybytes)) return -1.0;
    return secs;
}

/* Same weights, same buffers, same access pattern -- only the arithmetic differs.
 * See shaders/gemm_i4f.comp. */
int coli_vk_gemm4f(coli_vk *v, int wh, const coli_a_i8 *a, float *y) {
    if (!v->pipe4f) return -1;
    if (wh<0 || wh>=v->nw4 || !v->W4[wh].used) return -1;
    return gemm_dispatch(v, v->pipe4f, v->W4[wh].w, v->W4[wh].ws,
                         v->W4[wh].I, v->W4[wh].O, a, y);
}

/* ------------------------------------------------------- resident KV cache
 *
 * The point of the whole exercise. attn_decode.comp reads K and V straight out
 * of device memory; what makes that worth doing is that they GET there one
 * 2 KB row at a time as tokens are produced, instead of the whole cache being
 * uploaded per call. At a 681-token context the difference is 2 KB against
 * ~24 MB, per layer, per token.
 *
 * DEVICE_LOCAL on a discrete card, for the same reason the weights are: a
 * HOST_VISIBLE buffer on a discrete GPU lives in system RAM, so every one of the
 * up-to-kv_ctx reads per head would cross PCIe -- and attention reads the whole
 * cache every token, which is exactly the access pattern that punishes it most.
 * On an integrated GPU there is no staging to pay for and the flag is skipped;
 * measured 2026-08-20, DEVICE_LOCAL on the 780M is 2.2% SLOWER and lost 3 of 3.
 *
 * Rows are written into a HOST_VISIBLE staging buffer and copied by
 * vkCmdCopyBuffer recorded into the SAME command buffer as the attention
 * dispatch. That is deliberate: a separate submit per row write would add a
 * fence per layer and give back what this is meant to save. */
static void kv_free_all(coli_vk *v) {
    if (v->kvK) { for (int i=0;i<v->kv_layers;i++) freebuf(v,&v->kvK[i]); free(v->kvK); v->kvK=NULL; }
    if (v->kvV) { for (int i=0;i<v->kv_layers;i++) freebuf(v,&v->kvV[i]); free(v->kvV); v->kvV=NULL; }
    freebuf(v,&v->kvstage);
    freebuf(v,&v->kvwstage);
    v->kv_ok = 0; v->kv_pend = 0;
}

int coli_vk_kv_init(coli_vk *v, int layers, int slots, int kv_heads,
                    int kv_ctx, int hd) {
    if (!v || !v->pipe_attn) return -1;
    if (layers <= 0 || slots <= 0 || kv_heads <= 0 || kv_ctx <= 0) return -1;
    if (hd > 256 || (hd % 32)) return -1;

    kv_free_all(v);
    v->kvK = (vkbuf*)calloc((size_t)layers, sizeof(vkbuf));
    v->kvV = (vkbuf*)calloc((size_t)layers, sizeof(vkbuf));
    if (!v->kvK || !v->kvV) { kv_free_all(v); return -1; }
    v->kv_layers = layers;

    size_t per = (size_t)slots * kv_heads * kv_ctx * hd * sizeof(float);
    int devlocal = (!v->integrated || v->want_device_local) && !v->force_host_visible;
    for (int i = 0; i < layers; i++) {
        int ok = devlocal
            ? mkbuf_flags(v, per, &v->kvK[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT)
            : mkbuf(v, per, &v->kvK[i]);
        ok = ok && (devlocal
            ? mkbuf_flags(v, per, &v->kvV[i], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT)
            : mkbuf(v, per, &v->kvV[i]));
        /* Partial failure frees EVERYTHING rather than leaving some layers
         * allocated: a half-resident cache would have the caller reading device
         * memory for layer 3 and host memory for layer 4 with no way to tell,
         * and the wrong answer would look like a model problem. */
        if (!ok) { kv_free_all(v); return -1; }
    }
    /* Staging holds at most 64 rows (the pending-copy arrays are that size). */
    if (!mkbuf(v, (size_t)64 * hd * sizeof(float), &v->kvstage)) { kv_free_all(v); return -1; }

    v->kv_slots = slots; v->kv_heads = kv_heads; v->kv_ctx = kv_ctx; v->kv_hd = hd;
    v->kv_pend = 0; v->kv_ok = 1;
    return 0;
}

int coli_vk_kv_ready(coli_vk *v){ return v && v->kv_ok; }
size_t coli_vk_kv_bytes(coli_vk *v) {
    if (!v || !v->kv_ok) return 0;
    return (size_t)v->kv_layers * 2 * v->kv_slots * v->kv_heads
         * (size_t)v->kv_ctx * v->kv_hd * sizeof(float);
}

/* Stage one K or V row. Copied to the device by the next coli_vk_attn, in that
 * call's command buffer. Returns -1 if the staging ring is full, which the
 * caller must treat as "fall back to CPU for this token" rather than ignore:
 * a dropped row is a silently wrong answer for every later position. */
int coli_vk_kv_put(coli_vk *v, int slot, int kvh, int pos, int is_v, const float *row) {
    if (!v || !v->kv_ok) return -1;
    if (v->kv_pend >= 64) return -1;
    if (slot < 0 || slot >= v->kv_slots || kvh < 0 || kvh >= v->kv_heads) return -1;
    if (pos  < 0 || pos  >= v->kv_ctx) return -1;

    int hd = v->kv_hd;
    void *p;
    if (vkMapMemory(v->dev, v->kvstage.mem,
                    (VkDeviceSize)v->kv_pend*hd*sizeof(float),
                    (VkDeviceSize)hd*sizeof(float), 0, &p) != VK_SUCCESS) return -1;
    memcpy(p, row, (size_t)hd*sizeof(float));
    vkUnmapMemory(v->dev, v->kvstage.mem);

    v->kv_pend_off[v->kv_pend] = (((slot*v->kv_heads + kvh)*v->kv_ctx) + pos) * hd;
    v->kv_pend_kv [v->kv_pend] = is_v ? 1 : 0;
    v->kv_pend++;
    return 0;
}

/* Bulk-write a CONTIGUOUS run of positions, for EVERY kv head of one layer.
 *
 * WHY THIS EXISTS. coli_vk_kv_put stages one row at a time into a 64-row ring,
 * and that ring is the entire reason gpu_attn_ready refuses n > 32: 32 rows x
 * KVH x 2 buffers is already 64 at KVH=1, and the check is n*KVH*2 > 64. A
 * 683-token prefill needs 683 x 8 x 2 = 10,928 rows PER LAYER, which the ring
 * cannot express and should not have to.
 *
 * It does not have to, because prefill writes positions pos0..pos0+count-1 and
 * those are CONTIGUOUS on both sides -- the host cache is [kvh][kv_ctx][hd] and
 * the device cache is [slot][kvh][kv_ctx][hd], so for a fixed (slot, kvh) the
 * run is one unbroken extent in each. The whole layer is therefore 2*KVH
 * vkCmdCopyBuffer regions in ONE command buffer and ONE submit, not 10,928
 * staged rows.
 *
 * NOT coli_vk_kv_load: that uploads slots*KVH*kv_ctx*hd floats, i.e. every
 * position including the ones nothing has written yet. At kv_ctx 16384 that is
 * 64 MiB per buffer per layer to move 2.8 MiB of new data.
 *
 * Khost/Vhost point at the layer's FULL host cache, not at the run -- the
 * function does the (kvh, pos0) indexing itself, so the caller cannot get the
 * stride wrong in one place and right in the other. */
int coli_vk_kv_write(coli_vk *v, int layer, int slot, int pos0, int count,
                     const float *Khost, const float *Vhost) {
    if (!v || !v->kv_ok) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    if (slot < 0 || slot >= v->kv_slots) return -1;
    if (count <= 0 || pos0 < 0 || pos0 + count > v->kv_ctx) return -1;
    if (!Khost || !Vhost) return -1;

    int KVH = v->kv_heads, hd = v->kv_hd;
    size_t run   = (size_t)count * hd;                 /* floats per (kvh, buffer) */
    size_t need  = (size_t)KVH * run * 2 * sizeof(float);

    if (v->kvwstage.size < need) {
        freebuf(v, &v->kvwstage);
        if (!mkbuf_flags(v, need, &v->kvwstage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) { v->kvwstage.size = 0; return -1; }
    }

    /* One map for the whole staging area. K for every head first, then V, so the
     * two destination buffers each get one ascending sweep of source offsets. */
    void *p;
    { uint64_t t0 = now_ns();
      if (vkMapMemory(v->dev, v->kvwstage.mem, 0, need, 0, &p) != VK_SUCCESS) return -1;
      float *dst = (float*)p;
      for (int h = 0; h < KVH; h++)
          memcpy(dst + (size_t)h*run,
                 Khost + ((size_t)h*v->kv_ctx + pos0)*hd, run*sizeof(float));
      for (int h = 0; h < KVH; h++)
          memcpy(dst + ((size_t)KVH + h)*run,
                 Vhost + ((size_t)h*v->kv_ctx + pos0)*hd, run*sizeof(float));
      vkUnmapMemory(v->dev, v->kvwstage.mem);
      P.up_ns += now_ns()-t0; P.up_bytes += need; P.up_n++; }

    VkBufferCopy ck[64], cv[64];
    if (KVH > 64) return -1;                    /* ck/cv are sized for this */
    for (int h = 0; h < KVH; h++) {
        size_t doff = (((size_t)slot*KVH + h)*v->kv_ctx + pos0)*hd;
        ck[h] = (VkBufferCopy){ .srcOffset=(VkDeviceSize)((size_t)h*run*sizeof(float)),
                                .dstOffset=(VkDeviceSize)(doff*sizeof(float)),
                                .size=(VkDeviceSize)(run*sizeof(float)) };
        cv[h] = (VkBufferCopy){ .srcOffset=(VkDeviceSize)(((size_t)KVH+h)*run*sizeof(float)),
                                .dstOffset=(VkDeviceSize)(doff*sizeof(float)),
                                .size=(VkDeviceSize)(run*sizeof(float)) };
    }

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    if (vkBeginCommandBuffer(v->cmd,&bi)!=VK_SUCCESS) return -1;
    vkCmdCopyBuffer(v->cmd, v->kvwstage.buf, v->kvK[layer].buf, (uint32_t)KVH, ck);
    vkCmdCopyBuffer(v->cmd, v->kvwstage.buf, v->kvV[layer].buf, (uint32_t)KVH, cv);
    /* Same barrier argument as coli_vk_attn: the dispatch that reads these rows
     * is in a LATER submission here, so queue submission order covers it -- but
     * the barrier costs nothing and removes the dependency on that staying true
     * if the two are ever fused into one command buffer. */
    { VkMemoryBarrier mb = { .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT };
      vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); }
    if (vkEndCommandBuffer(v->cmd)!=VK_SUCCESS) return -1;

    vkResetFences(v->dev,1,&v->fence);
    { uint64_t t0 = now_ns();
      VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
      if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
      P.sub_ns += now_ns()-t0; P.sub_n++; }
    return 0;
}

int coli_vk_has_attn(coli_vk *v){ return v && v->pipe_attn != VK_NULL_HANDLE; }
/* Both pipelines or neither -- init() already tears down an orphan half, so
 * this only needs to check one, but checking both costs nothing and does not
 * depend on that invariant holding forever. */
int coli_vk_has_attn_split(coli_vk *v){
    return v && v->pipe_attn_split != VK_NULL_HANDLE && v->pipe_attn_merge != VK_NULL_HANDLE;
}

/* NCHUNK: how many pieces a row's [0,tmax] range is split into for the split-K
 * kernel. Chosen from n and KVH ALONE -- independent of tmax -- so it can be
 * computed before the descriptor set and scratch buffer are sized, and so it
 * does not change from one token to the next while tmax grows through a
 * session (a size that depended on tmax would force a re-size and a
 * redundant descriptor rewrite on every single decode step).
 *
 * TARGET 128 workgroups at n*KVH: measured 2026-09-13 on the RTX 4070, the
 * un-split kernel dispatches only n*H = 32 workgroups at n=1, H=32 -- far
 * short of the concurrency the device can use. n*KVH*NCHUNK=128 was chosen to
 * quadruple that (KVH=4 -> NCHUNK=32, matching the worked example in the
 * kernel's design note) without a dedicated sweep of NCHUNK itself, which is
 * a real gap: see the "shape/limit not covered" note in the test report.
 * Clamped to [1,64] so pathological shapes (huge n, or n*KVH already >= 128)
 * neither divide to zero nor explode the scratch buffer and the merge
 * kernel's per-chunk loop. */
/* COLI_VK_ATTN_SPLIT=0 is the CONTROL: it forces coli_vk_attn_block back onto
 * the single attn_decode.comp dispatch even when both split-K pipelines
 * loaded, so the A/B has an arm that can fail and a stale "it's faster" claim
 * from a rebuilt binary is falsifiable against the same one. Default ON
 * (anything else, or unset) once the pipelines exist -- see coli_vk_has_attn_split. */
static int attn_split_env(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_ATTN_SPLIT"); on = (e && e[0]=='0' && e[1]=='\0') ? 0 : 1; }
    return on;
}

static int split_nchunk(int n, int KVH) {
    int denom = n * KVH; if (denom < 1) denom = 1;
    int nc = 128 / denom;
    if (nc < 1) nc = 1;
    if (nc > 64) nc = 64;
    return nc;
}

/* Push constants shared by both split-K pipelines. Declared once so the host
 * struct and its two vkCmdPushConstants calls cannot drift apart -- see the
 * push6 comment above on why an inconsistent push is a silent wrong-answer
 * bug and not a crash. */
struct coli_attn_split_pc { int32_t H, KVH, hd, kv_ctx, n, nchunk; float scale; };

/* RECORD both split-K dispatches (with the barrier between them) into the
 * already-open command buffer. Does not touch the descriptor set contents --
 * the caller writes it once, with bindings {Q,K,V,SCRATCH,META,OUT}, and both
 * pipelines read the SAME set: pipe_attn_split writes bindings 3 (scratch),
 * pipe_attn_merge reads binding 3 and writes binding 5 (out). One write_set
 * call therefore serves both dispatches, exactly like ds_blk[0] already
 * serves multiple stages elsewhere in this file. */
static void record_attn_split(coli_vk *v, VkDescriptorSet ds,
                              int H, int KVH, int hd, int kv_ctx, int n,
                              int nchunk, float scale) {
    struct coli_attn_split_pc pc = { H, KVH, hd, kv_ctx, n, nchunk, scale };

    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_attn_split);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl2,0,1,&ds,0,NULL);
    vkCmdPushConstants(v->cmd,v->pl2,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof pc,&pc);
    vkCmdDispatch(v->cmd,(uint32_t)(n*KVH),(uint32_t)nchunk,1);

    /* pipe_attn_merge reads exactly what pipe_attn_split just wrote into
     * SCRATCH -- same compute-to-compute barrier record_barrier uses
     * everywhere else in the fused block. */
    record_barrier(v);

    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_attn_merge);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl2,0,1,&ds,0,NULL);
    vkCmdPushConstants(v->cmd,v->pl2,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof pc,&pc);
    vkCmdDispatch(v->cmd,(uint32_t)(n*H),1,1);
}

/* Bulk-load one layer's K and V from the host cache.
 *
 * Only for init and for growth. The host doubles kv_ctx and RE-STRIDES every
 * row when the context outgrows the cache, which changes the address of data
 * that has not moved -- so the device copy cannot be patched, it has to be
 * rebuilt from the authoritative host copy. That happens O(log) times over a
 * session, which is why paying a full upload for it is fine and why doing it
 * per token would not be.
 *
 * The host cache stays authoritative on purpose. It costs 16.4 ms across a
 * 681-token run (1.1%) to keep writing it, and it buys a CPU fallback that is
 * always correct and always available as the numerical reference. */
/* Read the resident cache back to the host. The inverse of coli_vk_kv_load, and
 * it exists for exactly one reason: with the fused block running, the device
 * cache is AHEAD of the host one, so when the host cache doubles its stride the
 * old contents must come back before the re-stride, or every row the block wrote
 * is lost. Without this, growth mid-sequence silently produced a cache with
 * holes -- measured 2026-08-23 as TF-NLL 13.6551 against a true 2.6768.
 *
 * DEVICE_LOCAL memory cannot be mapped, so that path stages through a temporary
 * HOST_VISIBLE buffer, mirroring upload_device_local. Growth is logarithmic in
 * context length, so this runs a handful of times per sequence, not per token. */
static int download_device_local(coli_vk *v, vkbuf *src, void *dst, size_t n) {
    vkbuf stage = {0};
    if (!mkbuf_flags(v, n, &stage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return 0;
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    VkBufferCopy cp = { .srcOffset=0, .dstOffset=0, .size=n };
    vkCmdCopyBuffer(v->cmd, src->buf, stage.buf, 1, &cp);
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    int ok = (vkQueueSubmit(v->q,1,&si,v->fence)==VK_SUCCESS) &&
             (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)==VK_SUCCESS) &&
             download(v,&stage,dst,n);
    freebuf(v,&stage);
    return ok;
}

int coli_vk_kv_get(coli_vk *v, int layer, float *K, float *V) {
    if (!v || !v->kv_ok) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    size_t n = (size_t)v->kv_slots * v->kv_heads * v->kv_ctx * v->kv_hd * sizeof(float);
    if (v->kvK[layer].size < n || v->kvV[layer].size < n) return -1;
    int devlocal = (!v->integrated || v->want_device_local) && !v->force_host_visible;
    int ok = devlocal ? (download_device_local(v,&v->kvK[layer],K,n) &&
                         download_device_local(v,&v->kvV[layer],V,n))
                      : (download(v,&v->kvK[layer],K,n) && download(v,&v->kvV[layer],V,n));
    return ok ? 0 : -1;
}

int coli_vk_kv_load(coli_vk *v, int layer, const float *K, const float *V) {
    if (!v || !v->kv_ok) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    size_t n = (size_t)v->kv_slots * v->kv_heads * v->kv_ctx * v->kv_hd * sizeof(float);
    /* upload() maps and memcpys; for a DEVICE_LOCAL buffer it cannot, so the
     * staged path is used there. Both are one-time. */
    if (v->kvK[layer].size < n || v->kvV[layer].size < n) return -1;
    int devlocal = (!v->integrated || v->want_device_local) && !v->force_host_visible;
    if (devlocal) {
        P.in_weight_upload = 1;
        int ok = upload_device_local(v,&v->kvK[layer],K,n)
              && upload_device_local(v,&v->kvV[layer],V,n);
        P.in_weight_upload = 0;
        return ok ? 0 : -1;
    }
    P.in_weight_upload = 1;
    int ok = upload(v,&v->kvK[layer],K,n) && upload(v,&v->kvV[layer],V,n);
    P.in_weight_upload = 0;
    return ok ? 0 : -1;
}

int coli_vk_kv_ctx(coli_vk *v){ return (v && v->kv_ok) ? v->kv_ctx : 0; }

/* PRODUCTION attention. One submit and one fence for the whole thing:
 * the staged K/V rows are copied and the kernel is dispatched in the SAME
 * command buffer, so writing the cache costs no extra synchronisation.
 *
 * The pipeline barrier between them is not optional. vkCmdCopyBuffer and
 * vkCmdDispatch recorded into one command buffer have NO ordering guarantee
 * without it -- the dispatch may legally read the destination before the copy
 * lands, and the result would be this token's attention over last token's
 * cache: plausible text, subtly wrong.
 *
 * ⚠️ NO TEST HERE PROVES THIS IS NEEDED. Measured 2026-08-22: the barrier was
 * removed and tests/test_vk_attn -- which deliberately stages the final row and
 * reads it in the SAME submission, the exact case at risk -- still passed at
 * rel=1.595e-05, identical to the barrier build. This driver (NVIDIA 610.57.04)
 * happens to order transfer-then-compute anyway. That is a driver behaviour, not
 * a guarantee, and RADV on the Legion's gfx1103 is a different implementation.
 * The barrier is kept because the specification requires it. Do not delete it on
 * the strength of a green test: the test cannot see this defect. */
int coli_vk_attn_sinks_upload(coli_vk *v, const float *sinks, size_t nfloat) {
    if (!v || !sinks || !nfloat) return -1;
    if (!ensure(v,&v->asink,nfloat*sizeof(float))) return -1;
    return upload(v,&v->asink,sinks,nfloat*sizeof(float)) ? 0 : -1;
}
int coli_vk_attn(coli_vk *v, int layer, const float *q, float *out,
                 const int *meta, int n, int H, float scale) {
    return coli_vk_attn_ex(v, layer, q, out, meta, n, H, scale, 0, -1);
}
int coli_vk_attn_ex(coli_vk *v, int layer, const float *q, float *out,
                    const int *meta, int n, int H, float scale, int window, int sink_off) {
    if (!v || !v->pipe_attn || !v->kv_ok) return -1;
    if (sink_off >= 0 && !v->asink.buf) return -1;   /* a sink offset with no sinks uploaded: refuse, do not read garbage */
    if (layer < 0 || layer >= v->kv_layers) return -1;
    if (H % v->kv_heads) return -1;

    int hd = v->kv_hd;
    size_t qn = (size_t)n * H * hd * sizeof(float);
    size_t mn = (size_t)n * 2 * sizeof(int);

    if (v->aq.size < qn && (freebuf(v,&v->aq), !mkbuf(v,qn,&v->aq))) return -1;
    if (v->ao.size < qn && (freebuf(v,&v->ao), !mkbuf_dl(v,qn,&v->ao))) return -1;
    if (v->am.size < mn && (freebuf(v,&v->am), !mkbuf(v,mn,&v->am))) return -1;
    if (!upload(v,&v->aq,q,qn))    return -1;
    if (!upload(v,&v->am,meta,mn)) return -1;

    if (!v->ds_attn_ok) {
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
        if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds_attn)!=VK_SUCCESS) return -1;
        v->ds_attn_ok = 1;
    }
    write_set(v, v->ds_attn, v->aq.buf, v->kvK[layer].buf, v->kvV[layer].buf,
              v->ao.buf, v->am.buf, v->asink.buf ? v->asink.buf : v->ao.buf);

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    if (vkBeginCommandBuffer(v->cmd,&bi)!=VK_SUCCESS) return -1;

    if (v->kv_pend > 0) {
        VkBufferCopy ck[64], cv[64]; int nk=0, nv=0;
        for (int i = 0; i < v->kv_pend; i++) {
            VkBufferCopy r = { .srcOffset=(VkDeviceSize)i*hd*sizeof(float),
                               .dstOffset=(VkDeviceSize)v->kv_pend_off[i]*sizeof(float),
                               .size=(VkDeviceSize)hd*sizeof(float) };
            if (v->kv_pend_kv[i]) cv[nv++] = r; else ck[nk++] = r;
        }
        if (nk) vkCmdCopyBuffer(v->cmd, v->kvstage.buf, v->kvK[layer].buf, (uint32_t)nk, ck);
        if (nv) vkCmdCopyBuffer(v->cmd, v->kvstage.buf, v->kvV[layer].buf, (uint32_t)nv, cv);
        VkMemoryBarrier mb = { .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask=VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        v->kv_pend = 0;
    }

    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_attn);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_attn,0,NULL);
    struct { int H,KVH,hd,kv_ctx,n; float scale; int window, sink_off; } pcv =
        { H, v->kv_heads, hd, v->kv_ctx, n, scale, window, sink_off };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&pcv);
    vkCmdDispatch(v->cmd,(uint32_t)(n*H),1,1);
    if (vkEndCommandBuffer(v->cmd)!=VK_SUCCESS) return -1;

    vkResetFences(v->dev,1,&v->fence);
    { uint64_t t0 = now_ns();
      VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
      if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
      P.sub_ns += now_ns()-t0; P.sub_n++; }

    return download(v,&v->ao,out,qn) ? 0 : -1;
}

/* TEST-ONLY, shaped exactly like coli_vk_attn above so the two can be timed
 * and correctness-checked apples-to-apples against the SAME resident cache:
 * same q/meta/out buffers, same kv_pend staging, same fence-wait shape. The
 * production caller is coli_vk_attn_block, which records the split-K path
 * inline (see record_attn_split there) rather than through this wrapper --
 * this exists so tests/test_attn_split.c does not have to reimplement the
 * resident-KV setup coli_vk_attn already has. */
int coli_vk_attn_split(coli_vk *v, int layer, const float *q, float *out,
                       const int *meta, int n, int H, float scale) {
    if (!v || !coli_vk_has_attn_split(v) || !v->kv_ok) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    if (H % v->kv_heads) return -1;

    int hd = v->kv_hd, KVH = v->kv_heads;
    size_t qn = (size_t)n * H * hd * sizeof(float);
    size_t mn = (size_t)n * 2 * sizeof(int);
    int nchunk = split_nchunk(n, KVH);
    size_t sn = (size_t)n * H * nchunk * (hd+2) * sizeof(float);

    if (v->aq.size < qn && (freebuf(v,&v->aq), !mkbuf(v,qn,&v->aq))) return -1;
    if (v->ao.size < qn && (freebuf(v,&v->ao), !mkbuf_dl(v,qn,&v->ao))) return -1;
    if (v->am.size < mn && (freebuf(v,&v->am), !mkbuf(v,mn,&v->am))) return -1;
    if (!ensure_dev(v,&v->attn_scratch,sn)) return -1;
    if (!upload(v,&v->aq,q,qn))    return -1;
    if (!upload(v,&v->am,meta,mn)) return -1;

    if (!v->ds_asplit_ok) {
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
        if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds_asplit)!=VK_SUCCESS) return -1;
        v->ds_asplit_ok = 1;
    }
    write_set(v, v->ds_asplit, v->aq.buf, v->kvK[layer].buf, v->kvV[layer].buf,
              v->attn_scratch.buf, v->am.buf, v->ao.buf);

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    if (vkBeginCommandBuffer(v->cmd,&bi)!=VK_SUCCESS) return -1;

    if (v->kv_pend > 0) {
        VkBufferCopy ck[64], cv[64]; int nk=0, nv=0;
        for (int i = 0; i < v->kv_pend; i++) {
            VkBufferCopy r = { .srcOffset=(VkDeviceSize)i*hd*sizeof(float),
                               .dstOffset=(VkDeviceSize)v->kv_pend_off[i]*sizeof(float),
                               .size=(VkDeviceSize)hd*sizeof(float) };
            if (v->kv_pend_kv[i]) cv[nv++] = r; else ck[nk++] = r;
        }
        if (nk) vkCmdCopyBuffer(v->cmd, v->kvstage.buf, v->kvK[layer].buf, (uint32_t)nk, ck);
        if (nv) vkCmdCopyBuffer(v->cmd, v->kvstage.buf, v->kvV[layer].buf, (uint32_t)nv, cv);
        VkMemoryBarrier mb = { .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask=VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(v->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        v->kv_pend = 0;
    }

    record_attn_split(v, v->ds_asplit, H, KVH, hd, v->kv_ctx, n, nchunk, scale);
    if (vkEndCommandBuffer(v->cmd)!=VK_SUCCESS) return -1;

    vkResetFences(v->dev,1,&v->fence);
    { uint64_t t0 = now_ns();
      VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
      if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
      P.sub_ns += now_ns()-t0; P.sub_n++; }

    return download(v,&v->ao,out,qn) ? 0 : -1;
}

/* VALIDATION ENTRY POINT -- NOT THE PRODUCTION PATH.
 *
 * This uploads the WHOLE K and V cache on every call, which is exactly the cost
 * the GPU attention work exists to remove: at a 681-token context that is ~24 MB
 * per layer per token. It is here to answer one question and only one -- does
 * attn_decode.comp compute the same thing as attend_online() -- and its timing
 * is meaningless. The production path keeps K and V resident and writes one row
 * per token; that is the next piece of work, and it is gated on this being
 * correct first.
 *
 * Named _ref so nobody wires it into the decode loop by accident. If you are
 * reading this from a profile that shows enormous upload traffic, this function
 * is why, and the answer is not to optimise it but to stop calling it. */
int coli_vk_attn_ref(coli_vk *v, const float *q, const float *K, const float *V,
                     float *out, const int *meta, int n, int H, int KVH, int hd,
                     int kv_ctx, int slots, float scale) {
    if (!v || !v->pipe_attn) return -1;
    if (hd > 256 || (hd % 32)) return -1;     /* the shader strides dims by 32 */
    if (H % KVH) return -1;                   /* GQA fan-out must be integral */

    size_t qn  = (size_t)n * H * hd * sizeof(float);
    size_t kvn = (size_t)slots * KVH * kv_ctx * hd * sizeof(float);
    size_t mn  = (size_t)n * 2 * sizeof(int);

    if (v->aq.size < qn  && (freebuf(v,&v->aq), !mkbuf(v,qn,&v->aq)))  return -1;
    if (v->ao.size < qn  && (freebuf(v,&v->ao), !mkbuf_dl(v,qn,&v->ao)))  return -1;
    if (v->ak.size < kvn && (freebuf(v,&v->ak), !mkbuf(v,kvn,&v->ak))) return -1;
    if (v->av.size < kvn && (freebuf(v,&v->av), !mkbuf(v,kvn,&v->av))) return -1;
    if (v->am.size < mn  && (freebuf(v,&v->am), !mkbuf(v,mn,&v->am)))  return -1;

    if (!upload(v,&v->aq,q,qn))    return -1;
    if (!upload(v,&v->ak,K,kvn))   return -1;
    if (!upload(v,&v->av,V,kvn))   return -1;
    if (!upload(v,&v->am,meta,mn)) return -1;

    if (!v->ds_attn_ok) {
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
        if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds_attn)!=VK_SUCCESS) return -1;
        v->ds_attn_ok = 1;
    }
    /* binding 5 is unused by the shader; the layout still demands a valid
     * descriptor there, so it gets the output buffer rather than a null handle. */
    write_set(v, v->ds_attn, v->aq.buf, v->ak.buf, v->av.buf, v->ao.buf,
              v->am.buf, v->ao.buf);

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    if (vkBeginCommandBuffer(v->cmd,&bi)!=VK_SUCCESS) return -1;
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_attn);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_attn,0,NULL);
    /* 32 bytes since 2026-09-14: attn_decode.comp reads window/sink_off after
     * scale, and push-constant bytes not written are UNDEFINED -- pushing 24
     * here left the harness reading garbage (test_vk_attn 2.76e-3 vs 1.6e-5). */
    struct { int H,KVH,hd,kv_ctx,n; float scale; int window, sink_off; } pcv = { H,KVH,hd,kv_ctx,n,scale, 0, -1 };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&pcv);
    vkCmdDispatch(v->cmd,(uint32_t)(n*H),1,1);
    if (vkEndCommandBuffer(v->cmd)!=VK_SUCCESS) return -1;

    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;

    return download(v,&v->ao,out,qn) ? 0 : -1;
}


/* ------------------------------------------------------- rope + kv scatter
 *
 * These are split into RECORD and RUN halves on purpose. The record halves take
 * an already-open command buffer and add one dispatch; the run halves wrap them
 * in upload/submit/download so a test can exercise the kernel alone. Production
 * calls only the record halves, from inside the fused attention block, where the
 * whole point is that nothing is uploaded or downloaded between stages -- a
 * "fused" path built out of the run halves would fuse nothing.
 */

/* record_dispatch pushes 16 bytes (I,O,n,nblk), which is the GEMM's shape. These
 * kernels need six ints, so they push their own 24 -- the same 24 the layout was
 * created with, and the same amount attention pushes. */
static void push6(coli_vk *v, const int32_t *six) {
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,24,six);
}

static int alloc_rk_sets(coli_vk *v) {
    if (v->ds_rk_ok) return 1;
    VkDescriptorSetLayout ls[3] = { v->dsl, v->dsl, v->dsl };
    VkDescriptorSet out[3];
    VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=v->dpool, .descriptorSetCount=3, .pSetLayouts=ls };
    if (vkAllocateDescriptorSets(v->dev,&dsai,out)!=VK_SUCCESS) return 0;
    v->ds_rope=out[0]; v->ds_kvw=out[1]; v->ds_qkn=out[2]; v->ds_rk_ok=1;
    return 1;
}

int coli_vk_has_rope(coli_vk *v){ return v && v->pipe_rope && v->pipe_kvw ? 1 : 0; }
int coli_vk_has_qknorm(coli_vk *v){ return v && v->pipe_qkn ? 1 : 0; }
/* qwen3 per-head norm weights, [layer][q hd | k hd] flat; uploaded once like the rope bias. */
int coli_vk_qknorm_upload(coli_vk *v, const float *w, size_t nfloat) {
    if (!v || !v->pipe_qkn) return -1;
    if (!ensure(v,&v->qkw,nfloat*4)) return -1;
    return upload(v,&v->qkw,w,nfloat*4) ? 0 : -1;
}
static void record_qknorm(coli_vk *v, vkbuf *q, vkbuf *k, int n, int H, int KVH, int hd, int off, float eps) {
    if (!alloc_rk_sets(v)) return;
    write_set(v,v->ds_qkn,q->buf,k->buf,k->buf,v->qkw.buf,v->qkw.buf,k->buf);
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_qkn);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_qkn,0,NULL);
    int32_t eb; memcpy(&eb,&eps,4);
    int32_t push[6] = { n, H, KVH, hd, off, eb };
    push6(v,push);
    vkCmdDispatch(v->cmd,(uint32_t)(n*(H+KVH)),1,1);
}

/* Bias is per LAYER and never changes, so it is uploaded by the caller once per
 * layer rather than per token; the (c,s) table is per POSITION and shared by
 * every layer, so it is uploaded once per forward call. Getting those two
 * frequencies backwards is 28x the traffic for no result. */
int coli_vk_rope_bias_upload(coli_vk *v, const float *bias, size_t nfloat) {
    if (!v) return -1;
    if (!ensure(v,&v->rbias,nfloat*4)) return -1;
    return upload(v,&v->rbias,bias,nfloat*4) ? 0 : -1;
}
int coli_vk_rope_cs_upload(coli_vk *v, const float *cs, size_t nfloat) {
    if (!v) return -1;
    if (!ensure(v,&v->rcs,nfloat*4)) return -1;
    return upload(v,&v->rcs,cs,nfloat*4) ? 0 : -1;
}

/* q and k are DEVICE buffers already holding the QKV matmul's output. */
static void record_rope(coli_vk *v, vkbuf *q, vkbuf *k, vkbuf *vv,
                        int n, int H, int KVH, int hd, int neox, int bias_off) {
    write_set(v,v->ds_rope,q->buf,k->buf,vv->buf,v->rbias.buf,v->rcs.buf,vv->buf);
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_rope);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_rope,0,NULL);
    int32_t push[6] = { n, H, KVH, hd, neox, bias_off };
    push6(v,push);
    uint32_t work = (uint32_t)n * (uint32_t)(H+KVH) * (uint32_t)(hd/2);
    vkCmdDispatch(v->cmd,(work+63)/64,1,1);
}

static void record_kvw(coli_vk *v, vkbuf *k, vkbuf *vv, int layer,
                       int n, int KVH, int hd, int kv_ctx, int bv_off, int has_bias) {
    write_set(v,v->ds_kvw,k->buf,vv->buf,v->kvK[layer].buf,v->kvV[layer].buf,
              v->am.buf,v->rbias.buf);
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_kvw);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_kvw,0,NULL);
    int32_t push[6] = { n, KVH, hd, kv_ctx, bv_off, has_bias };
    push6(v,push);
    uint32_t work = (uint32_t)n * (uint32_t)KVH * (uint32_t)hd;
    vkCmdDispatch(v->cmd,(work+63)/64,1,1);
}

/* TEST-ONLY wrapper: upload q/k, rotate, download. Production must not call this
 * -- the round trip it pays is the exact cost the kernel exists to remove. */
int coli_vk_rope_run(coli_vk *v, float *q, float *k,
                     int n, int H, int KVH, int hd, int neox, int bias_off) {
    if (!coli_vk_has_rope(v)) return -1;
    if (hd % 2) return -1;
    size_t qn=(size_t)n*H*hd*4, kn=(size_t)n*KVH*hd*4;
    if (!ensure(v,&v->aq,qn) || !ensure(v,&v->ak,kn) || !ensure(v,&v->av,kn)) return -1;
    if (!alloc_rk_sets(v)) return -1;
    if (!upload(v,&v->aq,q,qn) || !upload(v,&v->ak,k,kn)) return -1;

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    record_rope(v,&v->aq,&v->ak,&v->av,n,H,KVH,hd,neox,bias_off);
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    download(v,&v->aq,q,qn); download(v,&v->ak,k,kn);
    return 0;
}

/* TEST-ONLY wrapper, same caveat. Writes into the resident KV cache, so
 * coli_vk_kv_init must have run and the layer must be in range. */
int coli_vk_kvwrite_run(coli_vk *v, int layer, const float *k, const float *vv,
                        const int *slots, const int *poss,
                        int n, int KVH, int hd, int bv_off, int has_bias) {
    if (!coli_vk_has_rope(v) || !coli_vk_kv_ready(v)) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    size_t kn=(size_t)n*KVH*hd*4, mn=(size_t)n*2*4;
    if (!ensure(v,&v->ak,kn) || !ensure(v,&v->av,kn) || !ensure(v,&v->am,mn)) return -1;
    if (!alloc_rk_sets(v)) return -1;
    int32_t *meta=(int32_t*)malloc(mn); if(!meta) return -1;
    for (int r=0;r<n;r++){ meta[r*2]=slots[r]; meta[r*2+1]=poss[r]; }
    int ok = upload(v,&v->ak,k,kn) && upload(v,&v->av,vv,kn) && upload(v,&v->am,meta,mn);
    free(meta);
    if (!ok) return -1;

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    record_kvw(v,&v->ak,&v->av,layer,n,KVH,hd,v->kv_ctx,bv_off,has_bias);
    vkEndCommandBuffer(v->cmd);
    vkResetFences(v->dev,1,&v->fence);
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    return 0;
}


/* ------------------------------------------------- the fused attention block
 *
 * WHAT THIS REPLACES. Three submissions per layer, each with its own fence wait:
 *
 *     qkv gemm   -> DOWNLOAD q,k,v -> CPU bias+rope+kv copy -> UPLOAD q
 *     attention  -> DOWNLOAD att   -> UPLOAD att
 *     o_proj     -> DOWNLOAD xb
 *
 * Measured on this desktop 2026-08-21, qkv submit 8.1% + qkv download 6.9% +
 * gemm submit 11.1% + gemm download 11.5% is 37.6% of a decode token, and none
 * of it is arithmetic. The reason the layer could not be one submission was that
 * RoPE ran on the CPU in the middle of it. With rope_bias.comp and kvwrite.comp
 * that reason is gone, so the whole block becomes:
 *
 *     ONE upload (the normed activation) -> 8 dispatches -> ONE download
 *
 * WHY THE QUANTIZE STEP IS IN HERE. o_proj is an int8 GEMM and attention emits
 * float. Until quant.comp existed the requantization could only be done on the
 * CPU, which forced the download this function exists to remove -- the same
 * dependency silu_mul_q.comp broke for the FFN.
 *
 * THE HOST CACHE IS STILL WRITTEN by the caller, in parallel. See kvwrite.comp:
 * it costs 1.1% of decode and it is what keeps the CPU path a correct fallback
 * and the numerical reference.
 *
 * NOT USABLE FOR qwen3. That architecture RMSNorms q and k between the bias and
 * the rotation, and no kernel here does that. The caller must check for the
 * qk_norm tensors and stay on the CPU path when they are present; getting this
 * wrong is not a rounding difference, it is a different function.
 */
/* Deliberately does NOT test kv_ok. The resident cache is created by the
 * caller's readiness check, which runs AFTER this one -- requiring it here makes
 * the answer permanently no, which is how the first wiring of this block ran a
 * whole 681-token decode on the CPU path while reporting success. Whether the
 * cache exists is checked in coli_vk_attn_block, where it is actually needed. */
int coli_vk_has_block(coli_vk *v) {
    return v && v->pipe4 && v->pipe_rope && v->pipe_kvw && v->pipe_quant
             && v->pipe_attn ? 1 : 0;
}

int coli_vk_attn_block(coli_vk *v, int layer, const int *wh, const coli_a_i8 *a,
                       const int *meta, int n, int H, int KVH, int hd,
                       int neox, int bias_off, int qk_off, float qk_eps,
                       float scale, int stop_attn, float *y) {
    if (qk_off >= 0 && (!v || !v->pipe_qkn || !v->qkw.buf)) return -1;
    if (!coli_vk_has_block(v) || !wh || !a || !meta || !y) return -1;
    if (layer < 0 || layer >= v->kv_layers) return -1;
    if (KVH != v->kv_heads || hd != v->kv_hd) return -1;
    if (H % KVH) return -1;
    for (int j=0;j<4;j++) if (wh[j]<0 || wh[j]>=v->nw4 || !v->W4[wh[j]].used) return -1;

    int64_t I = v->W4[wh[0]].I, D = v->W4[wh[3]].O;
    int64_t qD = (int64_t)H*hd, kvD = (int64_t)KVH*hd;
    if (a->I != I || a->n != n) return -1;
    for (int j=1;j<3;j++) if (v->W4[wh[j]].I != I) return -1;
    if (v->W4[wh[0]].O != qD || v->W4[wh[1]].O != kvD || v->W4[wh[2]].O != kvD) return -1;
    if (v->W4[wh[3]].I != qD) return -1;
    if (qD % COLI_ABLK) return -1;

    int64_t nb = I/COLI_ABLK, nbq = qD/COLI_ABLK;
    if (!ensure(v,&v->xb,(size_t)n*I) || !ensure(v,&v->xs,(size_t)n*nb*4) ||
        !ensure(v,&v->xm,(size_t)n*nb*4)) return -1;
    /* yq and yb are SHARED with the qkv and ffn paths, so this one has to agree
     * with them on the allocation: a buffer left HOST_CACHED here would be kept
     * by ensure_out over there (it only reallocates on growth) and the staged
     * readback would quietly stop being staged. In this path yq is never
     * downloaded at all -- rope and kvwrite consume it on the device -- so VRAM
     * costs nothing here and pays elsewhere. */
    for (int j=0;j<3;j++) if (!ensure_out(v,&v->yq[j],(size_t)COOP_ROW_PAD(n)*v->W4[wh[j]].O*4)) return -1;
    size_t ybytes = stop_attn ? (size_t)n*qD*4 : (size_t)n*D*4;
    if (!ensure_dl(v,&v->batt,(size_t)n*qD*4) || !ensure(v,&v->bq8,(size_t)n*qD) ||
        !ensure(v,&v->bs8,(size_t)n*nbq*4)  || !ensure(v,&v->bm8,(size_t)n*nbq*4) ||
        !ensure_out(v,&v->yb,(size_t)COOP_ROW_PAD(n)*D*4)  || !ensure(v,&v->am,(size_t)n*2*4)) return -1;
    if (!ensure_stage(v,ybytes)) return -1;

    /* THE ONE UPLOAD. Everything after this stays in device memory until the
     * single download at the bottom. */
    if (!upload(v,&v->xb,a->q,(size_t)n*I))       return -1;
    if (!upload(v,&v->xs,a->scale,(size_t)n*nb*4)) return -1;
    if (!upload(v,&v->xm,a->sum,(size_t)n*nb*4))  return -1;
    if (!upload(v,&v->am,meta,(size_t)n*2*4))     return -1;

    if (!v->dsq_ok) {
        VkDescriptorSetLayout ls[3] = { v->dsl, v->dsl, v->dsl };
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=3, .pSetLayouts=ls };
        if (vkAllocateDescriptorSets(v->dev,&dsai,v->dsq)!=VK_SUCCESS) return -1;
        v->dsq_ok = 1;
    }
    if (!alloc_rk_sets(v)) return -1;
    if (!v->ds_blk_ok) {
        VkDescriptorSetLayout ls[3] = { v->dsl, v->dsl, v->dsl };
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=3, .pSetLayouts=ls };
        if (vkAllocateDescriptorSets(v->dev,&dsai,v->ds_blk)!=VK_SUCCESS) return -1;
        v->ds_blk_ok = 1;
    }
    for (int j=0;j<3;j++)
        write_set(v, v->dsq[j], v->W4[wh[j]].w.buf, v->W4[wh[j]].ws.buf,
                  v->xb.buf, v->xs.buf, v->xm.buf, v->yq[j].buf);
    write_set(v, v->ds_blk[0], v->yq[0].buf, v->kvK[layer].buf, v->kvV[layer].buf,
              v->batt.buf, v->am.buf, v->batt.buf);                      /* attention */
    write_set(v, v->ds_blk[1], v->batt.buf, v->batt.buf, v->bq8.buf,
              v->bs8.buf, v->bm8.buf, v->batt.buf);                      /* quantize  */
    write_set(v, v->ds_blk[2], v->W4[wh[3]].w.buf, v->W4[wh[3]].ws.buf,
              v->bq8.buf, v->bs8.buf, v->bm8.buf, v->yb.buf);            /* o_proj    */

    /* SPLIT-K DECODE ATTENTION (2026-09-13). See attn_decode_split.comp's
     * header for the measurement that motivates this: attn_decode.comp's
     * dispatch was 64.7 us of a 119.8 us fused block (RTX 4070, Qwen3-30B-A3B,
     * decode n=1, kv_ctx~266). use_split falls back to the untouched single
     * dispatch below whenever the split pipelines did not load (old shaders on
     * disk, or a build predating them) or COLI_VK_ATTN_SPLIT=0 -- the control
     * that makes this an A/B rather than a one-way door. */
    int use_split = attn_split_env() && coli_vk_has_attn_split(v);
    int a_nchunk = 1;
    if (use_split) {
        a_nchunk = split_nchunk(n, KVH);
        if (!ensure_dev(v,&v->attn_scratch,(size_t)n*H*a_nchunk*(hd+2)*4)) return -1;
        if (!v->ds_asplit_ok) {
            VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool=v->dpool, .descriptorSetCount=1, .pSetLayouts=&v->dsl };
            if (vkAllocateDescriptorSets(v->dev,&dsai,&v->ds_asplit)!=VK_SUCCESS) return -1;
            v->ds_asplit_ok = 1;
        }
        /* Same six bindings ds_blk[0] uses for the un-split kernel -- Q, K, V,
         * (scratch instead of the attention-output alias), META, out (batt) --
         * so the two paths are a drop-in swap of which descriptor set and
         * which pipeline(s) get bound below, nothing else in this function
         * changes. */
        write_set(v, v->ds_asplit, v->yq[0].buf, v->kvK[layer].buf, v->kvV[layer].buf,
                  v->attn_scratch.buf, v->am.buf, v->batt.buf);
    }

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    if (vkBeginCommandBuffer(v->cmd,&bi)!=VK_SUCCESS) return -1;
    /* COLI_VK_TS: a timestamp at each stage boundary (TOP at 0, BOTTOM_OF_PIPE after). */
    if (v->ts_on && !v->tsq) {
        VkQueryPoolCreateInfo qi = { .sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType=VK_QUERY_TYPE_TIMESTAMP, .queryCount=9 };
        if (vkCreateQueryPool(v->dev,&qi,NULL,&v->tsq)!=VK_SUCCESS) { v->tsq=VK_NULL_HANDLE; v->ts_on=0; }
    }
#define TS(i) do { if (v->tsq) vkCmdWriteTimestamp(v->cmd, (i)==0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT \
                                     : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, v->tsq, (i)); } while (0)
    if (v->tsq) vkCmdResetQueryPool(v->cmd,v->tsq,0,9);
    TS(0);
    for (int j=0;j<3;j++)                                   /* wq, wk, wv */
        record_gemm(v,v->pipe4,v->dsq[j],I,v->W4[wh[j]].O,n);
    record_barrier(v); TS(1);
    if (qk_off >= 0) {   /* qwen3: RMSNorm q and k per head BEFORE the rotation (CPU order: bias, norm, rope) */
        record_qknorm(v,&v->yq[0],&v->yq[1],n,H,KVH,hd,qk_off,qk_eps);
        record_barrier(v);
    }
    TS(2);
    record_rope(v,&v->yq[0],&v->yq[1],&v->yq[2],n,H,KVH,hd,neox,bias_off);
    record_barrier(v); TS(3);
    record_kvw(v,&v->yq[1],&v->yq[2],layer,n,KVH,hd,v->kv_ctx,
               bias_off>=0 ? bias_off+(int)qD+(int)kvD : -1, bias_off>=0);
    record_barrier(v); TS(4);
    if (use_split) {
        record_attn_split(v, v->ds_asplit, H, KVH, hd, v->kv_ctx, n, a_nchunk, scale);
    } else {
        vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pipe_attn);
        vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&v->ds_blk[0],0,NULL);
        /* 32 bytes, window 0 / no sink: see coli_vk_attn_ref. The fused block is
         * qwen-only; gpt-oss never reaches it (gpu_block_ready). */
        struct { int H,KVH,hd,kv_ctx,n; float scale; int window, sink_off; } pcv = { H, KVH, hd, v->kv_ctx, n, scale, 0, -1 };
        vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&pcv);
        vkCmdDispatch(v->cmd,(uint32_t)(n*H),1,1);
    }
    TS(5);
    /* stop_attn ends the block after attention and returns its raw output
     * instead of o_proj's, so the caller can run the requantization and o_proj
     * on the CPU. It bisects this function -- the stages before attention are
     * each unit-tested, the two after it are not -- and running the block twice
     * is safe: kvwrite is idempotent and attention is a pure read. */
    if (!stop_attn) {
        record_barrier(v);
        record_dispatch(v,v->pipe_quant,v->ds_blk[1],qD,0,n,(uint32_t)(((int64_t)n*nbq+63)/64));
        record_barrier(v); TS(6);
        record_gemm(v,v->pipe4,v->ds_blk[2],qD,D,n);
    }
    TS(7);
    /* batt is still HOST_CACHED, so the stop_attn arm reads it directly and only
     * the o_proj arm needs the copy. */
    if (v->out_dev && !stop_attn) record_copy_out(v,&v->yb,0,ybytes);
    TS(8);
#undef TS

    if (vkEndCommandBuffer(v->cmd)!=VK_SUCCESS) return -1;
    P.rec_ns += now_ns()-trec; P.cur_op = 2;

    vkResetFences(v->dev,1,&v->fence);
    { uint64_t t0 = now_ns();
      VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
      if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
      uint64_t d = now_ns()-t0; P.sub_ns += d; P.sub_n++;
      if (P.cur_op>=0 && P.cur_op<3) { P.sub_ns_op[P.cur_op]+=d; P.sub_n_op[P.cur_op]++; } }
    if (v->tsq) {
        uint64_t q[9];
        if (vkGetQueryPoolResults(v->dev,v->tsq,0,9,sizeof q,q,8,
                VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)==VK_SUCCESS) {
            for (int i=1;i<9;i++) P.ts_ns[i] += (uint64_t)((double)(q[i]-q[i-1]) * v->ts_period);
            P.ts_ns[0] += (uint64_t)((double)(q[8]-q[0]) * v->ts_period);   /* whole block, device time */
            P.ts_n++;
        }
    }

    /* THE ONE DOWNLOAD. */
    if (stop_attn) return download(v,&v->batt,y,ybytes) ? 0 : -1;
    return download_out(v,&v->yb,0,y,ybytes) ? 0 : -1;
}

void coli_vk_free(coli_vk *v) {
    if (!v) return;
    if (v->tsq) vkDestroyQueryPool(v->dev,v->tsq,NULL);
    for (int i=0;i<v->nw;i++)  if (v->W[i].used)  { freebuf(v,&v->W[i].w);  freebuf(v,&v->W[i].ws); }
    for (int i=0;i<v->nw4;i++) if (v->W4[i].used) { freebuf(v,&v->W4[i].w); freebuf(v,&v->W4[i].ws); }
    freebuf(v,&v->xb); freebuf(v,&v->xs); freebuf(v,&v->xm); freebuf(v,&v->yb);
    freebuf(v,&v->dlstage);
    if (v->upring_map) vkUnmapMemory(v->dev,v->upring.mem);
    freebuf(v,&v->upring);
    freebuf(v,&v->ymoe);
    if (v->moe_pool) vkDestroyDescriptorPool(v->dev,v->moe_pool,NULL);
    free(v->moe_ds);
    freebuf(v,&v->fg); freebuf(v,&v->fu); freebuf(v,&v->hq); freebuf(v,&v->hs); freebuf(v,&v->hm);
    freebuf(v,&v->aq); freebuf(v,&v->ak); freebuf(v,&v->av); freebuf(v,&v->ao); freebuf(v,&v->am);
    freebuf(v,&v->rbias); freebuf(v,&v->rcs);
    freebuf(v,&v->batt); freebuf(v,&v->bq8); freebuf(v,&v->bs8); freebuf(v,&v->bm8);
    kv_free_all(v);
    /* Async fill ring: wait out anything still in flight before freeing its
     * buffers out from under the GPU, then free the ring's own resources.
     * fpool is destroyed separately from v->pool ONLY when it is a distinct
     * pool (fill_mode==1); otherwise it IS v->pool and the generic destroy
     * below covers it. */
    for (int i=0;i<COLI_VK_FILL_INFLIGHT;i++) {
        fill_slot *s = &v->fslots[i];
        if (s->fence) { fill_slot_wait_one(v,s); vkDestroyFence(v->dev,s->fence,NULL); }
        if (s->map) vkUnmapMemory(v->dev,s->buf.mem);
        freebuf(v,&s->buf);
    }
    if (v->fpool && v->fpool != v->pool) vkDestroyCommandPool(v->dev,v->fpool,NULL);
    if (v->fence) vkDestroyFence(v->dev,v->fence,NULL);
    if (v->pool)  vkDestroyCommandPool(v->dev,v->pool,NULL);
    for (int j=0;j<3;j++) freebuf(v,&v->yq[j]);
    if (v->dpool) vkDestroyDescriptorPool(v->dev,v->dpool,NULL);
    if (v->pipe_attn) vkDestroyPipeline(v->dev,v->pipe_attn,NULL);
    if (v->pipe_attn_split) vkDestroyPipeline(v->dev,v->pipe_attn_split,NULL);
    if (v->pipe_attn_merge) vkDestroyPipeline(v->dev,v->pipe_attn_merge,NULL);
    if (v->pl2) vkDestroyPipelineLayout(v->dev,v->pl2,NULL);
    freebuf(v,&v->attn_scratch);
    if (v->pipe4s)    vkDestroyPipeline(v->dev,v->pipe4s,NULL);
    if (v->pipe4m)    vkDestroyPipeline(v->dev,v->pipe4m,NULL);
    if (v->pipe4ms)   vkDestroyPipeline(v->dev,v->pipe4ms,NULL);
    freebuf(v,&v->moe_tab);
    if (v->pipe_rope) vkDestroyPipeline(v->dev,v->pipe_rope,NULL);
    if (v->pipe_qkn)  vkDestroyPipeline(v->dev,v->pipe_qkn,NULL);
    freebuf(v,&v->qkw);
    if (v->pipe_kvw)  vkDestroyPipeline(v->dev,v->pipe_kvw,NULL);
    if (v->pipe_quant) vkDestroyPipeline(v->dev,v->pipe_quant,NULL);
    if (v->pipe4t) vkDestroyPipeline(v->dev,v->pipe4t,NULL);
    if (v->pipe4c) vkDestroyPipeline(v->dev,v->pipe4c,NULL);
    if (v->pipe4cd) vkDestroyPipeline(v->dev,v->pipe4cd,NULL);
    if (v->pipe)  vkDestroyPipeline(v->dev,v->pipe,NULL);
    if (v->pipe4) vkDestroyPipeline(v->dev,v->pipe4,NULL);
    if (v->pipe4f) vkDestroyPipeline(v->dev,v->pipe4f,NULL);
    if (v->pipe_smq) vkDestroyPipeline(v->dev,v->pipe_smq,NULL);
    if (v->pl)    vkDestroyPipelineLayout(v->dev,v->pl,NULL);
    if (v->dsl)   vkDestroyDescriptorSetLayout(v->dev,v->dsl,NULL);
    if (v->dev)   vkDestroyDevice(v->dev,NULL);
    if (v->inst)  vkDestroyInstance(v->inst,NULL);
    free(v);
}

/* Print the phase breakdown. Percentages are of the SUM of the four measured
 * phases, not of wall time -- everything the CPU does between GPU calls
 * (quantize, attention, norms, sampling) is outside this and is deliberately
 * not attributed here.
 *
 * Read it as: if sub_ns dominates and the transfer volume is small, the cost is
 * the round trip, not the bus. If up/dl dominate, it is the bus. Those two
 * point at different fixes, which is the whole reason for splitting them. */
void coli_vk_prof_dump(FILE *f) {
    double up=(double)P.up_ns/1e6, dl=(double)P.dl_ns/1e6;
    double rc=(double)P.rec_ns/1e6, sb=(double)P.sub_ns/1e6;
    double tot = up+dl+rc+sb; if (tot <= 0) { fprintf(f,"vk-prof: no GPU calls\n"); return; }
    /* The header used to print "gemm + ffn = submissions", an equality that is
     * FALSE: measured 2026-08-20, 76953 + 19068 = 96021 against 57885 submits.
     * They count different things -- a fused FFN records several dispatches into
     * ONE submission -- so the "=" claimed a relationship the numbers do not
     * have. Print them as the separate counters they are. */
    fprintf(f,"\n--- vk phase breakdown (%llu gemm, %llu ffn, %llu submissions) ---\n",
            (unsigned long long)P.gemm_n,(unsigned long long)P.ffn_n,(unsigned long long)P.sub_n);
    fprintf(f,"  [one-time weight staging: %.1f ms, %llu calls, %.1f MiB -- NOT in the total below]\n",
            (double)P.w_ns/1e6, (unsigned long long)P.w_n, (double)P.w_bytes/1048576.0);
    fprintf(f,"  upload        %9.1f ms  %5.1f%%   %llu calls, %.1f MiB\n",
            up, 100*up/tot, (unsigned long long)P.up_n, (double)P.up_bytes/1048576.0);
    fprintf(f,"  record cmdbuf %9.1f ms  %5.1f%%\n", rc, 100*rc/tot);
    fprintf(f,"  submit+fence  %9.1f ms  %5.1f%%   %llu submits, %.1f us each\n",
            sb, 100*sb/tot, (unsigned long long)P.sub_n,
            P.sub_n ? (double)P.sub_ns/1000.0/(double)P.sub_n : 0.0);
    fprintf(f,"  download      %9.1f ms  %5.1f%%   %llu calls, %.1f MiB\n",
            dl, 100*dl/tot, (unsigned long long)P.dl_n, (double)P.dl_bytes/1048576.0);
    fprintf(f,"  measured tot  %9.1f ms\n", tot);
    /* The positive control for mkbuf_out. `copies` MOVING is what proves the
     * staged path ran; a line that says "on" while copies stays 0 is the bug it
     * exists to catch. */
    { static const char *st[3] = { "OFF (host-visible outputs)", "ON", "REQUESTED, FELL BACK to host-visible" };
      fprintf(f,"  coop dispatch %llu total, %llu direct-store\n",
              (unsigned long long)P.coop_n,(unsigned long long)P.coop_ds_n);
      fprintf(f,"  moe fused-dispatch calls %llu of %llu grouped\n",
              (unsigned long long)P.moe_fused_n,(unsigned long long)P.ffn_n);
      fprintf(f,"  gemm outputs  %s%s%s -- %llu copies, %.1f MiB staged\n",
              st[P.out_state<0||P.out_state>2 ? 0 : P.out_state],
              P.out_desc[0] ? ", " : "", P.out_desc,
              (unsigned long long)P.copy_n, (double)P.copy_bytes/1048576.0); }
    /* Per-operation split of the two buckets that dominate. Attention is NOT
     * here: nothing in this backend runs it, so it is CPU-side and shows up as
     * the gap between `measured tot` and the engine's wall time. Saying that
     * explicitly matters -- an operation missing from a profile reads as free. */
    { static const char *nm[3] = { "gemm (single)", "ffn4 (fused)", "qkv (batched)" };
      if (P.ts_n) {
        static const char *st[9] = { "whole block (device)", "qkv gemms", "qknorm", "rope", "kv write",
                                     "attention", "requantize", "o_proj gemm", "copy out" };
        fprintf(f,"  --- fused attention block, GPU timestamps (%llu blocks, us each) ---\n",(unsigned long long)P.ts_n);
        for (int i=0;i<9;i++) fprintf(f,"    %-22s %8.1f\n", st[i], P.ts_ns[i]/1e3/(double)P.ts_n);
      }
      fprintf(f,"  --- submit+fence and download, by operation ---\n");
      for (int i=0;i<3;i++) {
        double sb_i=(double)P.sub_ns_op[i]/1e6, dl_i=(double)P.dl_ns_op[i]/1e6;
        if (P.sub_n_op[i]==0 && P.dl_n_op[i]==0) continue;
        fprintf(f,"  %-14s submit %8.1f ms (%4.1f%%, %llu x %6.1f us)  download %8.1f ms (%4.1f%%, %llu)\n",
                nm[i], sb_i, 100*sb_i/tot, (unsigned long long)P.sub_n_op[i],
                P.sub_n_op[i] ? (double)P.sub_ns_op[i]/1000.0/(double)P.sub_n_op[i] : 0.0,
                dl_i, 100*dl_i/tot, (unsigned long long)P.dl_n_op[i]);
      } }
    /* Expert-slot fill breakdown (2026-09-14). NOT folded into `tot` above --
     * slot fills are one-time-per-refill weight traffic, the same reason
     * P.w_ns is split from P.up_ns, and summing them here would make a
     * per-token percentage include a per-refill cost it cannot be divided by
     * fairly. coli_vk_fill_stats gives the same numbers without parsing this. */
    if (FSYNC.n || FASYNC.n) {
        fprintf(f,"  --- expert-slot fill (coli_vk_slot_fill[_async]), ms/call ---\n");
        if (FSYNC.n) fprintf(f,"  sync  memcpy %6.3f  rec+submit %6.3f  wait %6.3f  total %6.3f  (%llu calls)\n",
                (double)FSYNC.memcpy_ns/1e6/FSYNC.n, (double)FSYNC.recsub_ns/1e6/FSYNC.n,
                (double)FSYNC.wait_ns/1e6/FSYNC.n,
                (double)(FSYNC.memcpy_ns+FSYNC.recsub_ns+FSYNC.wait_ns)/1e6/FSYNC.n,
                (unsigned long long)FSYNC.n);
        if (FASYNC.n) fprintf(f,"  async memcpy %6.3f  rec+submit %6.3f  wait %6.3f  total %6.3f  (%llu calls)\n",
                (double)FASYNC.memcpy_ns/1e6/FASYNC.n, (double)FASYNC.recsub_ns/1e6/FASYNC.n,
                (double)FASYNC.wait_ns/1e6/FASYNC.n,
                (double)(FASYNC.memcpy_ns+FASYNC.recsub_ns+FASYNC.wait_ns)/1e6/FASYNC.n,
                (unsigned long long)FASYNC.n);
    }
}


/* q, k, v in ONE submission.
 *
 * They read the same normalised input and write three different outputs, with no
 * dependency between them -- so issuing them as three submit-and-block calls
 * bought nothing and cost two extra fence waits plus two redundant re-uploads of
 * an activation that had not changed.
 *
 * MEASURED: 1.115x on an RTX 4070 (8 reps) and 1.104x on a Radeon 780M (6 reps),
 * distributions non-overlapping on both. ~10% either way.
 *
 * That equality is the interesting part, because it refuted the prediction that
 * justified writing this. The 780M's EMPTY submit+fence costs 72-349us against
 * the 4070's 22us, so removing 68 round trips per token was forecast to be worth
 * ~19 ms/token there versus ~1.5 ms here -- a 13x-larger win. It was the same
 * ~10%. Batching removes submissions but not work; what it recovers is only the
 * fixed part of each one, and that part is a similar FRACTION on both devices
 * rather than the dominant term an idle-GPU probe suggested. In a real Legion
 * run submit+fence averaged 546us, higher than even the cold idle floor.
 *
 * An idle GPU's round trip and a loaded GPU's submission are different
 * quantities. Do not size the next optimisation off the empty-submit number. */
/* No barrier between the three: they touch disjoint outputs. */
int coli_vk_gemm4_qkv(coli_vk *v, const int *wh, const coli_a_i8 *a, float **ys) {
    if (!v || !v->pipe4 || !wh || !a || !ys) return -1;
    for (int j=0;j<3;j++)
        if (wh[j]<0 || wh[j]>=v->nw4 || !v->W4[wh[j]].used) return -1;

    int64_t I = v->W4[wh[0]].I;
    if (a->I != I) return -1;
    for (int j=1;j<3;j++) if (v->W4[wh[j]].I != I) return -1;  /* same input width */

    int n = a->n;
    int64_t nb = I/COLI_ABLK;
    if (!ensure(v,&v->xb,(size_t)n*I) || !ensure(v,&v->xs,(size_t)n*nb*4) ||
        !ensure(v,&v->xm,(size_t)n*nb*4)) return -1;
    for (int j=0;j<3;j++)
        if (!ensure_out(v,&v->yq[j],(size_t)COOP_ROW_PAD(n)*v->W4[wh[j]].O*4)) return -1;
    /* Three results live at once, so they share one staging buffer at three
     * offsets. 256-byte aligned so no copy starts mid-cacheline. */
    size_t qb[3], qoff[3], total = 0;
    for (int j=0;j<3;j++) {
        qb[j] = (size_t)n*v->W4[wh[j]].O*4;
        qoff[j] = total;
        total += (qb[j] + 255) & ~(size_t)255;
    }
    if (!ensure_stage(v,total)) return -1;

    /* ONE upload, not three. */
    if (!upload(v,&v->xb,a->q,(size_t)n*I)) return -1;
    if (!upload(v,&v->xs,a->scale,(size_t)n*nb*4)) return -1;
    if (!upload(v,&v->xm,a->sum,(size_t)n*nb*4)) return -1;

    if (!v->dsq_ok) {
        VkDescriptorSetLayout ls[3] = { v->dsl, v->dsl, v->dsl };
        VkDescriptorSetAllocateInfo dsai = { .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool=v->dpool, .descriptorSetCount=3, .pSetLayouts=ls };
        if (vkAllocateDescriptorSets(v->dev,&dsai,v->dsq)!=VK_SUCCESS) return -1;
        v->dsq_ok = 1;
    }
    for (int j=0;j<3;j++)
        write_set(v, v->dsq[j], v->W4[wh[j]].w.buf, v->W4[wh[j]].ws.buf,
                  v->xb.buf, v->xs.buf, v->xm.buf, v->yq[j].buf);

    uint64_t trec = now_ns();
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    for (int j=0;j<3;j++)
        record_gemm(v,v->pipe4,v->dsq[j],I,v->W4[wh[j]].O,n);
    if (v->out_dev) for (int j=0;j<3;j++) record_copy_out(v,&v->yq[j],qoff[j],qb[j]);
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.gemm_n += 3; P.cur_op = 2;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    { uint64_t d = now_ns()-tsub; P.sub_ns += d; P.sub_n++;
      int oi = P.cur_op; if (oi>=0 && oi<3) { P.sub_ns_op[oi]+=d; P.sub_n_op[oi]++; } }

    for (int j=0;j<3;j++)
        if (!download_out(v,&v->yq[j],qoff[j],ys[j],qb[j])) return -1;
    return 0;
}

/* The FLOOR: submit an EMPTY command buffer and wait on the fence, nothing else.
 * No dispatch, no descriptor write, no transfer -- so whatever this costs is
 * pure queue round trip plus fence signal, and every real submission pays it on
 * top of its actual work.
 *
 * This exists because two data points looked linear in n and I was one step away
 * from calling the per-submit cost "fixed overhead" on the strength of a
 * two-point fit. A constant you inferred is not a constant you measured. */
double coli_vk_probe_submit_ns(coli_vk *v, int reps, double *out_min_ns) {
    if (!v || reps <= 0) return -1.0;
    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    /* warm the queue: the first submission after init pays one-off driver cost */
    for (int i=0;i<8;i++) {
        vkResetCommandBuffer(v->cmd,0); vkBeginCommandBuffer(v->cmd,&bi); vkEndCommandBuffer(v->cmd);
        vkResetFences(v->dev,1,&v->fence);
        if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1.0;
        if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1.0;
    }
    /* MIN and MEAN, not mean alone. A floor is a minimum: one scheduler stall
     * amortised across `reps` raises a mean and hides the real best case, and
     * every other timing harness in this tree (test_gemm_i8, test_vk_gemm)
     * already uses min-of-N for exactly that reason -- this probe was the
     * exception. The gap between the two IS the contention signal, so both are
     * returned rather than one being chosen here on the caller's behalf. */
    uint64_t best = ~0ull;
    uint64_t t0 = now_ns();
    for (int i=0;i<reps;i++) {
        uint64_t a = now_ns();
        vkResetCommandBuffer(v->cmd,0); vkBeginCommandBuffer(v->cmd,&bi); vkEndCommandBuffer(v->cmd);
        vkResetFences(v->dev,1,&v->fence);
        if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1.0;
        if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1.0;
        uint64_t d = now_ns()-a;
        if (d < best) best = d;
    }
    double mean = (double)(now_ns()-t0)/(double)reps;
    if (out_min_ns) *out_min_ns = (double)best;
    return mean;
}
