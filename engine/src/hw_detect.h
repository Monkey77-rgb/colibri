/* hw_detect.h — one probe of "what is this machine", and a thin planner on
 * top of it. Plain C99, no Vulkan/CUDA headers required: callers that never
 * compile -DCOLI_HAVE_VK still link and run coli_hw_probe, they just get
 * n_vk==0. CUDA is never hard-linked either way -- see hw_detect.c's comment
 * on cuInit et al -- so a box with no CUDA driver still builds and runs this
 * file, it only reports cuda.present==0.
 *
 * WHY ONE STRUCT AND NOT A HANDFUL OF GETTERS. Every existing probe in this
 * engine (coli_cpu_features, coli_vk_is_integrated, coli_vk_probe_class) is
 * queried piecemeal, on demand, from whichever call site needs it -- fine
 * when there is one GPU and one caller. A startup-time hardware plan needs
 * CPU + every Vulkan device + CUDA + the torch plugin all answered from ONE
 * coherent snapshot, so a decision ("put 3.2 GB of experts on the GPU") is
 * never made from two probes taken milliseconds apart that could in
 * principle disagree (a GPU going away between calls, a second probe
 * re-creating a VkInstance unnecessarily). coli_hw_probe fills one struct
 * once; everything downstream reads fields, not re-probes.
 *
 * WHY coli_hw_plan IS SEPARATE FROM coli_hw. The probe states what exists;
 * the plan states what to DO given what exists, a model's stated byte costs,
 * and a preference the caller (CLI flag, env var) already resolved. Keeping
 * them apart means a test can hand coli_hw_plan_make a hand-built coli_hw
 * (e.g. "build_backends=0") without needing a real GPU to exercise the
 * decision logic -- see tests/test_hw_detect.c's apparatus control.
 */
#ifndef COLI_HW_DETECT_H
#define COLI_HW_DETECT_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_HW_MAX_VK 4

/* One Vulkan-visible physical device. Fields mirror what vk_backend.c's
 * coli_vk_init already reads off VkPhysicalDeviceProperties/
 * VkPhysicalDeviceMemoryProperties for its OWN device choice; this just
 * records it for every device instead of picking one. */
typedef struct coli_hw_gpu {
    char     name[128];
    int      vendor_id;
    int      device_id;
    int      is_integrated;          /* VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU */
    uint64_t vram_bytes;             /* largest heap flagged DEVICE_LOCAL */
    uint64_t host_visible_bytes;     /* largest heap NOT flagged DEVICE_LOCAL */
    int      subgroup_size;          /* VkPhysicalDeviceSubgroupProperties */
    int      has_dedicated_transfer_queue; /* a queue family with TRANSFER and
                                             * neither GRAPHICS nor COMPUTE */
    int      api_major, api_minor;   /* VkPhysicalDeviceProperties.apiVersion */
} coli_hw_gpu;

/* CUDA, probed through dlopen("libcuda.so.1") ONLY -- see hw_detect.c. A
 * machine with the Vulkan device above and no NVIDIA driver leaves every
 * field here zeroed, which is a normal answer, not an error. */
typedef struct coli_hw_cuda {
    int      present;
    int      device_count;
    char     name[128];              /* device 0 only; a plan never needs more */
    int      cc_major, cc_minor;     /* device 0's compute capability */
    uint64_t total_mem;              /* device 0's VRAM, bytes */
    int      driver_version;         /* cuDriverGetVersion's raw int, e.g. 12040 */
} coli_hw_cuda;

typedef struct coli_hw {
    uint32_t      cpu_features;      /* coli_cpu_features(), see cpu_features.h */
    int           cpu_physical_cores;
    int           cpu_logical_cores;
    char          cpu_name[128];
    uint64_t      ram_total_bytes;
    uint64_t      ram_available_bytes;
    int           n_vk;
    coli_hw_gpu   vk[COLI_HW_MAX_VK];
    coli_hw_cuda  cuda;
    int           torch_plugin_present; /* libcoli_torch.so found, see hw_detect.c */
} coli_hw;

/* Fills every field of *out. Missing subsystems (no Vulkan build, no CUDA
 * driver, no torch plugin, an unreadable /proc file) are left zeroed -- NEVER
 * an error return -- matching this project's "absence is a normal answer"
 * convention (see vk_backend.h's coli_vk_init comment). Always returns 1. */
int coli_hw_probe(coli_hw *out);

/* One line per subsystem, human readable. States explicitly what was NOT
 * found (a blank line for a missing GPU, not an omitted one) -- see the
 * lesson in gemm_q6k's own header: a silently-absent thing and a correctly-
 * absent thing must not look the same on the page. */
void coli_hw_print(const coli_hw *hw, FILE *f);

/* Build-time backend availability, for coli_hw_plan_make_ex's build_backends
 * argument. A probe can see an NVIDIA driver on a binary that was never
 * linked against CUDA; the plan must not pick a backend this binary cannot
 * actually run. */
enum {
    COLI_BE_VULKAN = 1u << 0,
    COLI_BE_CUDA   = 1u << 1,
    COLI_BE_TORCH  = 1u << 2,
};

typedef struct coli_hw_plan {
    char backend[16];      /* "cuda" | "vulkan" | "cpu" -- never "torch": no
                             * torch backend exists in this engine yet, so a
                             * prefer="torch" with the plugin present still
                             * has nowhere to route and falls back, see .c */
    int  threads;
    int  moe_vram_mb;
    int  expert_store_gb;
    int  gpu_attn;
    char reason[256];
} coli_hw_plan;

/* Convenience wrapper: calls coli_hw_plan_make_ex with every backend bit set,
 * i.e. "this binary could use any backend it finds" -- the common case for a
 * fully-built `coli-gpu`/`banana-gpu`. A caller that KNOWS its own binary was
 * built CPU-only (plain `coli`/`banana`) should call _ex directly with the
 * narrower mask, or this function will plan for a backend the binary cannot
 * actually dispatch to. */
int coli_hw_plan_make(const coli_hw *hw, const char *prefer,
                       uint64_t model_dense_bytes, uint64_t model_kv_bytes,
                       coli_hw_plan *out);

int coli_hw_plan_make_ex(const coli_hw *hw, const char *prefer,
                          uint64_t model_dense_bytes, uint64_t model_kv_bytes,
                          unsigned build_backends, coli_hw_plan *out);

#ifdef __cplusplus
}
#endif
#endif /* COLI_HW_DETECT_H */
