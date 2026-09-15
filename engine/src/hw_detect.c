/* hw_detect.c — see hw_detect.h for the "one snapshot, one planner" rationale.
 *
 * CPU FEATURE BITS COME FROM cpu_features.h, NEVER REDETECTED. coli_cpu_features()
 * already does the cpuid/getauxval work cpu_features.cpp's own header explains at
 * length (runtime dispatch, not -march=native); duplicating that here would be a
 * second ISA detector that can silently disagree with the one the kernels actually
 * dispatch on. This file calls it once and copies the bitmask.
 *
 * VULKAN IS OPTIONAL AT COMPILE TIME. Everything under #ifdef COLI_HAVE_VK is the
 * only code in this file that touches <vulkan/vulkan.h>; without that macro the
 * whole block compiles to "n_vk = 0" and the file still links with no libvulkan
 * dependency at all -- same contract vk_backend.h documents for coli_vk_init.
 *
 * CUDA IS NEVER A COMPILE-TIME DEPENDENCY, WITH OR WITHOUT A MACRO. There is no
 * -DCOLI_HAVE_CUDA: cuda.h is never included and libcuda is never linked. Instead
 * this dlopen()s libcuda.so.1 at RUNTIME and dlsym()s the handful of driver-API
 * entry points it needs, with the prototypes declared by hand below. That is what
 * lets a machine with no CUDA toolchain installed -- this engine's normal case,
 * see engine/Makefile's own "a fresh clone must build without a Vulkan SDK"
 * reasoning for coli-gpu -- still compile this file and report cuda.present=0
 * instead of failing to build. If dlopen fails for any reason (no driver, no
 * permission, wrong arch) the whole struct is left zeroed; that is the correct
 * answer, not an error path to report.
 */
#include "hw_detect.h"
#include "cpu_features.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <glob.h>

#ifdef COLI_HAVE_VK
#include <vulkan/vulkan.h>
#endif

/* ===================================================================== CPU */

static void probe_cpu(coli_hw *out) {
    out->cpu_features = coli_cpu_features();

    out->cpu_logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (out->cpu_logical_cores < 0) out->cpu_logical_cores = 0;

    /* Physical cores: unique core_id under each cpuN's topology dir in sysfs.
     * sysconf has no "physical cores" query at all -- _SC_NPROCESSORS_ONLN counts
     * hyperthreads as separate processors, which is the logical count, not this
     * one. Falls back to the logical count when sysfs is unreadable (containers,
     * some VMs) rather than reporting 0, which would make coli_hw_plan_make's
     * threads field zero and every caller has to cope with that; sysconf's
     * logical figure is the closest honest answer available in that case. */
    {
        glob_t g;
        int seen[1024]; int nseen = 0;
        if (glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id", 0, NULL, &g) == 0) {
            for (size_t i = 0; i < g.gl_pathc; i++) {
                FILE *f = fopen(g.gl_pathv[i], "r");
                if (!f) continue;
                int id = -1;
                if (fscanf(f, "%d", &id) == 1 && id >= 0) {
                    int dup = 0;
                    for (int k = 0; k < nseen; k++) if (seen[k] == id) { dup = 1; break; }
                    if (!dup && nseen < (int)(sizeof seen / sizeof *seen)) seen[nseen++] = id;
                }
                fclose(f);
            }
            globfree(&g);
        }
        out->cpu_physical_cores = nseen > 0 ? nseen : out->cpu_logical_cores;
    }

    /* Name: /proc/cpuinfo's "model name", first match. This is the marketing
     * string ("AMD Ryzen 7 9800X3D 8-Core Processor"), a different thing from
     * coli_cpu_name()'s brand string used for the ISA banner -- kept separate on
     * purpose so a change to one never silently changes the other's wording. */
    snprintf(out->cpu_name, sizeof out->cpu_name, "unknown");
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof line, f)) {
                if (!strncmp(line, "model name", 10)) {
                    char *c = strchr(line, ':');
                    if (c) {
                        c++;
                        while (*c == ' ' || *c == '\t') c++;
                        size_t n = strlen(c);
                        while (n && (c[n-1] == '\n' || c[n-1] == '\r')) c[--n] = 0;
                        snprintf(out->cpu_name, sizeof out->cpu_name, "%s", c);
                    }
                    break;
                }
            }
            fclose(f);
        }
    }
}

/* =============================================================== /proc/meminfo */

static void probe_ram(coli_hw *out) {
    out->ram_total_bytes = 0;
    out->ram_available_bytes = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        unsigned long long kb;
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) out->ram_total_bytes = kb * 1024ull;
        else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) out->ram_available_bytes = kb * 1024ull;
    }
    /* cgroup v2 memory cap (2026-09-14): the engine is routinely run under
     * `systemd-run -p MemoryMax=22G` on this homelab, and a plan sized from
     * MemAvailable alone (25.9 GiB measured) would size the expert store past
     * the cap and be OOM-killed by the cgroup, not the kernel. Walk
     * /proc/self/cgroup -> /sys/fs/cgroup/<path>/memory.max up the hierarchy
     * and take the smallest limit; "max" means none. The available figure the
     * planner uses is then min(MemAvailable, limit - memory.current). */
    out->ram_cgroup_limit_bytes = 0;
    {
        FILE *cg = fopen("/proc/self/cgroup", "r"); char cl[512];
        if (cg) {
            while (fgets(cl, sizeof cl, cg)) {
                char *p = strrchr(cl, ':'); if (!p) continue; p++; p[strcspn(p, "\n")] = 0;
                char path[1024]; unsigned long long best = 0, cur = 0;
                snprintf(path, sizeof path, "%s", p);
                for (;;) {
                    char fn[1200]; snprintf(fn, sizeof fn, "/sys/fs/cgroup%s/memory.max", path);
                    FILE *f = fopen(fn, "r"); unsigned long long v = 0;
                    if (f) { if (fscanf(f, "%llu", &v) == 1 && v > 0 && (!best || v < best)) best = v; fclose(f); }
                    if (!cur) { snprintf(fn, sizeof fn, "/sys/fs/cgroup%s/memory.current", path);
                        f = fopen(fn, "r"); if (f) { if (fscanf(f, "%llu", &cur) != 1) cur = 0; fclose(f); } }
                    char *sl = strrchr(path, '/'); if (!sl || sl == path) break; *sl = 0;
                }
                if (best) { out->ram_cgroup_limit_bytes = best;
                    unsigned long long left = best > cur ? best - cur : 0;
                    if (left < out->ram_available_bytes) out->ram_available_bytes = left; }
                break;
            }
            fclose(cg);
        }
    }
    fclose(f);
}

/* ================================================================== Vulkan */

#ifdef COLI_HAVE_VK
/* Same style as coli_vk_init (vk_backend.c): a throwaway 1.1 instance, no
 * validation layers, destroyed before returning. Unlike coli_vk_init this
 * does NOT pick one device and does NOT create a logical device or queue at
 * all -- a hardware probe has no shader to run and must not pay for, or risk
 * failing on, device creation just to read properties that vkEnumeratePhysicalDevices
 * and vkGetPhysicalDeviceProperties2 already expose at the PHYSICAL device level. */
static void probe_vulkan(coli_hw *out) {
    out->n_vk = 0;

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "coli_hw_detect", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return; /* no driver -- normal */

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    if (!nd) { vkDestroyInstance(inst, NULL); return; }
    VkPhysicalDevice *devs = (VkPhysicalDevice*)malloc(nd * sizeof *devs);
    if (!devs) { vkDestroyInstance(inst, NULL); return; }
    vkEnumeratePhysicalDevices(inst, &nd, devs);

    int lim = (int)nd < COLI_HW_MAX_VK ? (int)nd : COLI_HW_MAX_VK;
    for (int i = 0; i < lim; i++) {
        coli_hw_gpu *g = &out->vk[out->n_vk];
        memset(g, 0, sizeof *g);

        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(devs[i], &pr);
        snprintf(g->name, sizeof g->name, "%s", pr.deviceName);
        g->vendor_id = (int)pr.vendorID;
        g->device_id = (int)pr.deviceID;
        g->is_integrated = (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        g->api_major = (int)VK_VERSION_MAJOR(pr.apiVersion);
        g->api_minor = (int)VK_VERSION_MINOR(pr.apiVersion);

        /* Subgroup width via core-1.1 Properties2, same struct vk_backend.c's
         * coli_vk_init chains (VkPhysicalDeviceSubgroupProperties). No
         * VK_EXT_subgroup_size_control probe here -- that extension's min/max
         * pinning range matters to the attention kernel, not to this report. */
        VkPhysicalDeviceSubgroupProperties sgprops = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgprops };
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        g->subgroup_size = (int)sgprops.subgroupSize;

        /* Heaps: largest DEVICE_LOCAL heap is vram_bytes, largest heap that is
         * NOT DEVICE_LOCAL is host_visible_bytes -- a heap-flag question, not a
         * memory-TYPE question (HOST_VISIBLE is a type property that points at
         * a heap index; the task's own framing of these two fields is by heap,
         * so that is what is measured here). */
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
            uint64_t sz = mp.memoryHeaps[h].size;
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                if (sz > g->vram_bytes) g->vram_bytes = sz;
            } else {
                if (sz > g->host_visible_bytes) g->host_visible_bytes = sz;
            }
        }

        /* Dedicated transfer queue: TRANSFER set, GRAPHICS and COMPUTE both
         * clear. On the 4070 there is such a family (2 queues) alongside the
         * graphics+compute family; an iGPU commonly has exactly one family that
         * advertises all three bits and this correctly reports 0 there -- see
         * the task's own framing, report either honestly. */
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        if (nq) {
            VkQueueFamilyProperties *qs = (VkQueueFamilyProperties*)malloc(nq * sizeof *qs);
            if (qs) {
                vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qs);
                for (uint32_t q = 0; q < nq; q++) {
                    VkQueueFlags fl = qs[q].queueFlags;
                    if ((fl & VK_QUEUE_TRANSFER_BIT) &&
                        !(fl & VK_QUEUE_GRAPHICS_BIT) && !(fl & VK_QUEUE_COMPUTE_BIT)) {
                        g->has_dedicated_transfer_queue = 1;
                        break;
                    }
                }
                free(qs);
            }
        }

        out->n_vk++;
    }

    free(devs);
    vkDestroyInstance(inst, NULL);
}
#else
static void probe_vulkan(coli_hw *out) { out->n_vk = 0; }
#endif

/* ==================================================================== CUDA */

/* Driver-API prototypes, declared by hand -- see the file header for why
 * cuda.h is never included. Types match the real driver API's width exactly
 * (CUdevice is a 32-bit int handle, CUresult a 32-bit enum encoded as int).
 * CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR} = 75/76 are the real
 * enumerator values from cuda.h (stable across CUDA versions; the driver API
 * is backward compatible by contract), used here as plain ints so no header
 * defining them is needed either. */
typedef int CUdevice;
typedef int CUresult;
#define COLI_CU_ATTR_CC_MAJOR 75
#define COLI_CU_ATTR_CC_MINOR 76

typedef CUresult (*coli_cuInit_fn)(unsigned int);
typedef CUresult (*coli_cuDeviceGetCount_fn)(int *);
typedef CUresult (*coli_cuDeviceGetName_fn)(char *, int, CUdevice);
typedef CUresult (*coli_cuDeviceGetAttribute_fn)(int *, int, CUdevice);
typedef CUresult (*coli_cuDeviceTotalMem_fn)(size_t *, CUdevice);
typedef CUresult (*coli_cuDriverGetVersion_fn)(int *);

static void probe_cuda(coli_hw *out) {
    memset(&out->cuda, 0, sizeof out->cuda);

    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return; /* no driver installed -- normal, not an error */

    coli_cuInit_fn               cuInit               = (coli_cuInit_fn)dlsym(h, "cuInit");
    coli_cuDeviceGetCount_fn     cuDeviceGetCount     = (coli_cuDeviceGetCount_fn)dlsym(h, "cuDeviceGetCount");
    coli_cuDeviceGetName_fn      cuDeviceGetName      = (coli_cuDeviceGetName_fn)dlsym(h, "cuDeviceGetName");
    coli_cuDeviceGetAttribute_fn cuDeviceGetAttribute = (coli_cuDeviceGetAttribute_fn)dlsym(h, "cuDeviceGetAttribute");
    coli_cuDeviceTotalMem_fn     cuDeviceTotalMem     = (coli_cuDeviceTotalMem_fn)dlsym(h, "cuDeviceTotalMem_v2");
    coli_cuDriverGetVersion_fn   cuDriverGetVersion   = (coli_cuDriverGetVersion_fn)dlsym(h, "cuDriverGetVersion");

    if (!cuInit || !cuDeviceGetCount || !cuDeviceGetName || !cuDeviceGetAttribute ||
        !cuDeviceTotalMem || !cuDriverGetVersion) {
        dlclose(h);
        return; /* driver present but API surface unexpected -- treat as absent */
    }

    if (cuInit(0) != 0) { dlclose(h); return; }

    int count = 0;
    if (cuDeviceGetCount(&count) != 0 || count <= 0) { dlclose(h); return; }

    out->cuda.present = 1;
    out->cuda.device_count = count;

    char name[128] = {0};
    cuDeviceGetName(name, (int)sizeof name, 0);
    snprintf(out->cuda.name, sizeof out->cuda.name, "%s", name);

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, COLI_CU_ATTR_CC_MAJOR, 0);
    cuDeviceGetAttribute(&minor, COLI_CU_ATTR_CC_MINOR, 0);
    out->cuda.cc_major = major;
    out->cuda.cc_minor = minor;

    size_t total = 0;
    cuDeviceTotalMem(&total, 0);
    out->cuda.total_mem = (uint64_t)total;

    int drv = 0;
    cuDriverGetVersion(&drv);
    out->cuda.driver_version = drv;

    dlclose(h);
}

/* ============================================================== torch plugin */

static int file_exists(const char *path) {
    return path && path[0] && access(path, F_OK) == 0;
}

static void probe_torch(coli_hw *out) {
    out->torch_plugin_present = 0;
    char path[1024];

    const char *dir = getenv("COLI_PLUGIN_DIR");
    if (dir && *dir) {
        snprintf(path, sizeof path, "%s/libcoli_torch.so", dir);
        if (file_exists(path)) { out->torch_plugin_present = 1; return; }
    }

    /* Directory of the running binary -- readlink, not argv[0], because
     * argv[0] can be relative or a shell-chosen alias and /proc/self/exe
     * always resolves to the real executable path on Linux. */
    {
        char exe[1024];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            exe[n] = 0;
            char *slash = strrchr(exe, '/');
            if (slash) {
                *slash = 0;
                snprintf(path, sizeof path, "%s/libcoli_torch.so", exe);
                if (file_exists(path)) { out->torch_plugin_present = 1; return; }
            }
        }
    }

    if (file_exists("libcoli_torch.so")) out->torch_plugin_present = 1;
}

/* ================================================================== public */

int coli_hw_probe(coli_hw *out) {
    memset(out, 0, sizeof *out);
    probe_cpu(out);
    probe_ram(out);
    probe_vulkan(out);
    probe_cuda(out);
    probe_torch(out);
    return 1;
}

void coli_hw_print(const coli_hw *hw, FILE *f) {
    fprintf(f, "cpu: %s | %d physical / %d logical cores | features 0x%08x\n",
            hw->cpu_name, hw->cpu_physical_cores, hw->cpu_logical_cores, hw->cpu_features);
    fprintf(f, "ram: %.2f GiB total, %.2f GiB available (cgroup cap: %s)\n",
            (double)hw->ram_total_bytes / (1024.0*1024.0*1024.0),
            (double)hw->ram_available_bytes / (1024.0*1024.0*1024.0),
            hw->ram_cgroup_limit_bytes ? "yes" : "none");
    if (hw->ram_cgroup_limit_bytes)
        fprintf(f, "ram: cgroup memory.max %.2f GiB applies to this process\n",
                (double)hw->ram_cgroup_limit_bytes / (1024.0*1024.0*1024.0));

    if (hw->n_vk == 0) {
        fprintf(f, "vulkan: NOT FOUND (no VkInstance, no build, or no device)\n");
    } else {
        for (int i = 0; i < hw->n_vk; i++) {
            const coli_hw_gpu *g = &hw->vk[i];
            fprintf(f, "vulkan[%d]: %s | vendor 0x%04x device 0x%04x | %s | api %d.%d | "
                       "vram %.2f GiB | host-visible %.2f GiB | subgroup %d | "
                       "dedicated transfer queue: %s\n",
                    i, g->name, g->vendor_id, g->device_id,
                    g->is_integrated ? "integrated" : "discrete",
                    g->api_major, g->api_minor,
                    (double)g->vram_bytes / (1024.0*1024.0*1024.0),
                    (double)g->host_visible_bytes / (1024.0*1024.0*1024.0),
                    g->subgroup_size,
                    g->has_dedicated_transfer_queue ? "yes" : "no");
        }
    }

    if (!hw->cuda.present) {
        fprintf(f, "cuda: NOT FOUND (no libcuda.so.1, no driver, or cuInit failed)\n");
    } else {
        fprintf(f, "cuda: %s | %d device(s) | cc %d.%d | %.2f GiB | driver %d\n",
                hw->cuda.name, hw->cuda.device_count, hw->cuda.cc_major, hw->cuda.cc_minor,
                (double)hw->cuda.total_mem / (1024.0*1024.0*1024.0), hw->cuda.driver_version);
    }

    fprintf(f, "torch plugin: %s\n", hw->torch_plugin_present ? "found" : "NOT FOUND");
}

/* ================================================================== planning
 *
 * See hw_detect.h's struct comments for WHAT each field means; this is the
 * HOW. Every constant below (1 GiB headroom, 6 GiB headroom, 256 MiB
 * rounding, /4 UMA cap, [0,64] GiB clamp) is a caller-overridable DEFAULT,
 * not a measured fact about any one model -- the caller (CLI flag, env var)
 * owns overriding it; this function just has to pick something defensible
 * with no other information than the byte costs it was handed.
 */

#define GIB ((uint64_t)1 << 30)
#define MIB ((uint64_t)1 << 20)

static int64_t clampi(int64_t v, int64_t lo, int64_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int coli_hw_plan_make_ex(const coli_hw *hw, const char *prefer,
                          uint64_t model_dense_bytes, uint64_t model_kv_bytes,
                          unsigned build_backends, coli_hw_plan *out) {
    memset(out, 0, sizeof *out);

    int want_cuda   = (prefer && !strcmp(prefer, "cuda"));
    int want_vulkan = (prefer && !strcmp(prefer, "vulkan"));
    int want_cpu    = (prefer && !strcmp(prefer, "cpu"));
    int want_torch  = (prefer && !strcmp(prefer, "torch"));
    int want_auto   = (!prefer || !*prefer || !strcmp(prefer, "auto"));

    int have_cuda   = hw->cuda.present && (build_backends & COLI_BE_CUDA);
    int have_vulkan = hw->n_vk > 0     && (build_backends & COLI_BE_VULKAN);

    char reason[256] = {0};
    const char *backend = "cpu";
    int use_vk_idx = -1; /* which hw->vk[] entry backs "vulkan"/gpu_attn decisions */

    if (want_cpu) {
        backend = "cpu";
        snprintf(reason, sizeof reason, "prefer=cpu requested explicitly");
    } else if (want_torch) {
        /* coli_hw_plan.backend only ever names an engine that can actually run
         * a forward pass; there is no torch execution path in this engine
         * today (see hw_detect.h), so a torch preference always falls back to
         * the auto order below regardless of torch_plugin_present. */
        snprintf(reason, sizeof reason,
                 "prefer=torch has no execution path in this engine (plugin %s); falling back to auto order",
                 hw->torch_plugin_present ? "present" : "absent");
        want_auto = 1;
    } else if (want_cuda) {
        if (have_cuda) {
            backend = "cuda";
            snprintf(reason, sizeof reason, "prefer=cuda honored: driver present, %d device(s), build has CUDA",
                      hw->cuda.device_count);
        } else if (have_vulkan) {
            backend = "vulkan"; use_vk_idx = 0;
            snprintf(reason, sizeof reason,
                     "prefer=cuda unavailable (cuda.present=%d, build_backends&CUDA=%d) -- falling back to vulkan (%d device(s))",
                     hw->cuda.present, (build_backends & COLI_BE_CUDA) != 0, hw->n_vk);
        } else {
            backend = "cpu";
            snprintf(reason, sizeof reason,
                     "prefer=cuda unavailable and no usable vulkan device (n_vk=%d) -- falling back to cpu",
                     hw->n_vk);
        }
    } else if (want_vulkan) {
        if (have_vulkan) {
            backend = "vulkan"; use_vk_idx = 0;
            snprintf(reason, sizeof reason, "prefer=vulkan honored: %d device(s) visible, build has Vulkan", hw->n_vk);
        } else {
            backend = "cpu";
            snprintf(reason, sizeof reason,
                     "prefer=vulkan unavailable (n_vk=%d, build_backends&VULKAN=%d) -- falling back to cpu",
                     hw->n_vk, (build_backends & COLI_BE_VULKAN) != 0);
        }
    }

    if (want_auto) {
        /* auto order: vulkan > cuda > cpu, never picking a backend not present
         * AND not in this binary's build_backends mask. MEASURED, not assumed
         * (2026-09-14, RTX 4070, gpt-oss-120b, 96 greedy tokens, same knobs):
         * Vulkan static 2.5-2.6 tok/s with device attention; CUDA backend 1.9
         * (it declines attention -> CPU attend, and its thread-per-row kernels
         * are ~25 % slower per call at 2880x2880). The first draft of this
         * planner put cuda first on the assumption that native beats portable;
         * the numbers said otherwise on this engine. COLI_BACKEND_ORDER still
         * overrides for a machine where the measurement differs. */
        if (have_vulkan) {
            backend = "vulkan"; use_vk_idx = 0;
            snprintf(reason, sizeof reason,
                     "auto: %d vulkan device(s) and built in -- preferred over cuda (present=%d): measured faster on this engine, 09-14",
                     hw->n_vk, hw->cuda.present);
        } else if (have_cuda) {
            backend = "cuda";
            snprintf(reason, sizeof reason, "auto: no usable vulkan (n_vk=%d, build&VULKAN=%d) -- cuda present (%d device(s)) and built in",
                      hw->n_vk, (build_backends & COLI_BE_VULKAN) != 0, hw->cuda.device_count);
        } else {
            backend = "cpu";
            snprintf(reason, sizeof reason,
                     "auto: no usable cuda or vulkan (cuda.present=%d, n_vk=%d, build_backends=0x%x) -- cpu only",
                     hw->cuda.present, hw->n_vk, build_backends);
        }
    }

    snprintf(out->backend, sizeof out->backend, "%s", backend);

    /* threads: physical cores, falling back to logical if the probe could not
     * determine physical count (see probe_cpu's own fallback). */
    out->threads = hw->cpu_physical_cores > 0 ? hw->cpu_physical_cores : hw->cpu_logical_cores;

    int is_integrated_vk = (use_vk_idx >= 0 && use_vk_idx < hw->n_vk) ? hw->vk[use_vk_idx].is_integrated : 0;

    if (!strcmp(backend, "cpu")) {
        out->moe_vram_mb = 0;
        out->gpu_attn = 0;
        out->gpu_keepalive = 0;
    } else {
        uint64_t vram = 0;
        int integrated = 0;
        if (!strcmp(backend, "cuda")) {
            vram = hw->cuda.total_mem;
            integrated = 0; /* no integrated NVIDIA part in this fleet; discrete by construction */
        } else { /* vulkan */
            vram = (use_vk_idx >= 0) ? hw->vk[use_vk_idx].vram_bytes : 0;
            integrated = is_integrated_vk;
        }

        int64_t headroom = (int64_t)vram - (int64_t)model_dense_bytes - (int64_t)model_kv_bytes - (int64_t)(1 * GIB);
        headroom = clampi(headroom, 0, INT64_MAX);
        int64_t mb = headroom / (int64_t)MIB;
        mb = (mb / 256) * 256; /* round DOWN to 256 MiB */

        if (integrated) {
            int64_t uma_cap = (int64_t)(hw->ram_available_bytes / 4) / (int64_t)MIB;
            if (mb > uma_cap) mb = uma_cap;
        }
        out->moe_vram_mb = (int)clampi(mb, 0, INT32_MAX);
        out->gpu_attn = integrated ? 0 : 1;
        /* ON by default for a discrete GPU (09-15, goss22: 3 rotated rounds,
         * keepalive 2.2-2.5 tok/s vs bare 2.2/2.2/2.4, logit head 1,450 -> 390
         * ms/96 tokens, expert GEMV unchanged, 4070 + 9800X3D with the desktop's
         * services running). The earlier in-process loss (goss18/19/21) was the
         * keepalive thread's own OpenMP team spinning, fixed in gpu_keepalive.cpp,
         * not a cost of the keepalive itself. COLI_GPU_KEEPALIVE=0 turns it off.
         * Integrated parts: not measured, and a UMA part shares the memory clock
         * with the CPU anyway -- off. */
        out->gpu_keepalive = integrated ? 0 : 1;
    }

    /* expert_store_gb: RAM left after the dense weights and a 6 GiB headroom,
     * clamped to [0, 64] GiB -- independent of which backend was picked, since
     * the expert store (expert_store.h) is a HOST-side LRU regardless of where
     * the active expert's matmul actually runs. */
    {
        int64_t headroom = (int64_t)hw->ram_available_bytes - (int64_t)model_dense_bytes - (int64_t)(6 * GIB);
        headroom = clampi(headroom, 0, INT64_MAX);
        int64_t gb = headroom / (int64_t)GIB;
        out->expert_store_gb = (int)clampi(gb, 0, 64);
    }

    snprintf(out->reason, sizeof out->reason, "%s", reason);
    return 1;
}

int coli_hw_plan_make(const coli_hw *hw, const char *prefer,
                       uint64_t model_dense_bytes, uint64_t model_kv_bytes,
                       coli_hw_plan *out) {
    return coli_hw_plan_make_ex(hw, prefer, model_dense_bytes, model_kv_bytes,
                                 COLI_BE_VULKAN | COLI_BE_CUDA | COLI_BE_TORCH, out);
}
