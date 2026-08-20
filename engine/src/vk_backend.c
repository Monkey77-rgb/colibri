/* vk_backend.c — see vk_backend.h for scope and the bit-exactness exception. */
#define _GNU_SOURCE
#include "vk_backend.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    uint64_t up_bytes, dl_bytes;             /* transfer volume */
    uint64_t up_n, dl_n, sub_n, gemm_n, ffn_n;
    /* One-time weight staging, counted apart from per-token activation traffic.
     * Folding the two together made a 1.7 GiB model load look like 142 MiB/token
     * of activations -- a positive-looking number that is entirely the wrong
     * quantity. Split at the source rather than subtracted afterwards. */
    uint64_t w_ns, w_bytes, w_n;
    int in_weight_upload;
} P;

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

#define VKERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)
/* One handle per weight matrix in the model, not per benchmark. A dense 36-layer
 * qwen2.5-3b has 36 x 7 + 1 = 253; 64 was sized for the test harness and would
 * have failed at layer 9 of a real model. MoE is far beyond this (48 layers x
 * 128 experts x 3) and is deliberately NOT uploaded -- see coli_gpu_upload. */
#define MAX_W 512

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; } vkbuf;

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
    VkPipeline pipe4f;     /* int4 dequant-to-float variant, for the comparison */
    VkPipeline pipe_smq;   /* silu*mul + quantize, the op that makes residency possible */
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
    VkDescriptorPool dpool;
    VkFence fence;
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
    int  dot_used;         /* the DP4a spv actually loaded and built a pipeline */
    VkPhysicalDeviceMemoryProperties memprops;
    struct { vkbuf w, ws; int64_t I, O; int used; } W[MAX_W];
    int nw;
    /* int4 weights, in their own table. Kept separate rather than tagged inside
     * W[] so a caller cannot hand an int4 handle to the int8 GEMM and have it
     * read half a matrix -- the type confusion would produce plausible numbers,
     * which is the failure mode worth designing out. */
    struct { vkbuf w, ws; int64_t I, O; int used; } W4[MAX_W];
    int nw4;
    /* Persistent scratch, grown to the high-water mark and reused. Allocating
     * and destroying these per call cost more than the kernel ran for. */
    vkbuf xb, xs, xm, yb;
    VkDescriptorSet ds;
    int ds_ok;
    int has_transfer;      /* device-local weights need a copy queue */
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

static int mkbuf_flags(coli_vk *v, VkDeviceSize sz, vkbuf *b,
                       VkMemoryPropertyFlags want, VkBufferUsageFlags usage) {
    if (sz == 0) sz = 4;
    VkBufferCreateInfo bi = { .sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=sz, .usage=usage, .sharingMode=VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(v->dev,&bi,NULL,&b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(v->dev,b->buf,&mr);
    uint32_t mt = find_mem(v, mr.memoryTypeBits, want);
    if (mt == UINT32_MAX) { vkDestroyBuffer(v->dev,b->buf,NULL); b->buf=0; return 0; }
    VkMemoryAllocateInfo ai = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=mr.size, .memoryTypeIndex=mt };
    if (vkAllocateMemory(v->dev,&ai,NULL,&b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(v->dev,b->buf,NULL); b->buf=0; return 0; }
    vkBindBufferMemory(v->dev,b->buf,b->mem,0);
    b->size = sz;
    return 1;
}

static int mkbuf(coli_vk *v, VkDeviceSize sz, vkbuf *b) {
    if (sz == 0) sz = 4;
    VkBufferCreateInfo bi = { .sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=sz, .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode=VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(v->dev,&bi,NULL,&b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(v->dev,b->buf,&mr);
    uint32_t mt = find_mem(v, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { vkDestroyBuffer(v->dev,b->buf,NULL); return 0; }
    if (!v->memdesc[0]) describe_mem(v, mt, v->memdesc, sizeof v->memdesc);
    VkMemoryAllocateInfo ai = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=mr.size, .memoryTypeIndex=mt };
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
    P.dl_ns += now_ns()-t0; P.dl_bytes += n; P.dl_n++;
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
    snprintf(v->devname,sizeof v->devname,"%s",pr.deviceName);
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
    free(qs);
    if (v->qfam==UINT32_MAX) { VKERR("no compute queue"); goto fail; }

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
                if (!strcmp(xp[i].extensionName, "VK_KHR_shader_integer_dot_product")) { v->has_dot = 1; break; }
            free(xp);
        }
    }
    { const char *e = getenv("COLI_VK_NO_DOT"); if (e && *e && *e!='0') v->has_dot = 0; }

    const char *devexts[1] = { "VK_KHR_shader_integer_dot_product" };
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dotf = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
        .shaderIntegerDotProduct=VK_TRUE };

    float prio=1.f;
    VkDeviceQueueCreateInfo qci = { .sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=v->qfam, .queueCount=1, .pQueuePriorities=&prio };
    VkDeviceCreateInfo dci = { .sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1, .pQueueCreateInfos=&qci,
        .enabledExtensionCount = v->has_dot ? 1u : 0u,
        .ppEnabledExtensionNames = v->has_dot ? devexts : NULL,
        .pNext = v->has_dot ? (const void*)&dotf : NULL };
    if (vkCreateDevice(v->pdev,&dci,NULL,&v->dev) != VK_SUCCESS) {
        /* Retry bare: an advertised extension whose feature the driver refuses
         * must not cost us the GPU entirely. */
        v->has_dot = 0;
        VkDeviceCreateInfo bare = { .sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount=1, .pQueueCreateInfos=&qci };
        if (vkCreateDevice(v->pdev,&bare,NULL,&v->dev) != VK_SUCCESS) { VKERR("vkCreateDevice failed"); goto fail; }
    }
    vkGetDeviceQueue(v->dev,v->qfam,0,&v->q);

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
    VkPushConstantRange pc = { .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, .offset=0, .size=20 };
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
    return v;
fail:
    if (v->dev) vkDestroyDevice(v->dev,NULL);
    vkDestroyInstance(v->inst,NULL); free(v); return NULL;
}

const char *coli_vk_device_name(coli_vk *v){ return v?v->devname:"(none)"; }
const char *coli_vk_mem_desc(coli_vk *v){ return (v&&v->memdesc[0])?v->memdesc:"(no allocation yet)"; }
const char *coli_vk_weight_mem(coli_vk *v){ return (v&&v->memdesc2[0])?v->memdesc2:"(none)"; }
int coli_vk_is_integrated(coli_vk *v){ return v?v->integrated:0; }
/* Which memory the WEIGHTS actually landed in. Reported rather than inferred:
 * the allocation flags say what was asked for, these say what was granted. */
const char *coli_vk_memdesc (coli_vk *v){ return (v&&v->memdesc[0]) ?v->memdesc :"unknown"; }
const char *coli_vk_memdesc2(coli_vk *v){ return (v&&v->memdesc2[0])?v->memdesc2:"unknown"; }
int coli_vk_dot_used(coli_vk *v){ return v?v->dot_used:0; }
int coli_vk_wants_device_local(coli_vk *v){ return v?v->want_device_local:0; }

/* Copy through a staging buffer into DEVICE_LOCAL memory. Only worth doing for
 * weights, which are written once and read every step; doing it for activations
 * would add a copy per call to save nothing. */
static int upload_device_local(coli_vk *v, vkbuf *dst, const void *src, size_t n) {
    P.in_weight_upload = 1;
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
    int32_t push[4] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK) };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,push);
    vkCmdDispatch(v->cmd,groups,1,1);
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

/* Rows per workgroup. Must be <= MAXTILE in the shaders (8). Chosen by
 * MEASUREMENT: same-binary sweep at 3584x3584 n=512 on an RTX 4070 gave
 * tile=1 31.50 ms, tile=2 24.14 ms, tile=4 22.52 ms, tile=8 23.41 ms. 8 was
 * the initial guess and it is not the best -- 4 is, and 8 is measurably worse,
 * which is why this is a measured constant and not a round number. */
#define COLI_VK_TILE_R 4

static int gemm_on_device(coli_vk *v, VkPipeline pipe, vkbuf wbuf, vkbuf wsbuf,
                          int64_t I, int64_t O, int n,
                          vkbuf xb, vkbuf xs, vkbuf xm, vkbuf yb) {
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
    int tile = COLI_VK_TILE_R;
    { const char *e = getenv("COLI_VK_TILE_R");
      if (e && *e) { int t = atoi(e); if (t >= 1 && t <= 4) tile = t; } }  /* <= MAXTILE */
    int32_t push[5] = { (int32_t)I, (int32_t)O, n, (int32_t)(I/COLI_ABLK), tile };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,20,push);
    int64_t rtiles = ((int64_t)n + tile - 1) / tile;
    vkCmdDispatch(v->cmd,(uint32_t)(rtiles*O),1,1);
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.gemm_n++;
    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    uint64_t tsub = now_ns();
    if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1;
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    P.sub_ns += now_ns()-tsub; P.sub_n++;
    return 0;
}

static int gemm_dispatch(coli_vk *v, VkPipeline pipe, vkbuf wbuf, vkbuf wsbuf,
                         int64_t I, int64_t O, const coli_a_i8 *a, float *y) {
    int64_t nb=I/COLI_ABLK;
    int n=a->n;
    if (a->I != I) return -1;

    /* Persistent, grown to the high-water mark. Was: allocate 4 buffers, upload,
     * dispatch, download, destroy 4 buffers -- every call. */
    if (!ensure(v,&v->xb,(size_t)n*I) || !ensure(v,&v->xs,(size_t)n*nb*4) ||
        !ensure(v,&v->xm,(size_t)n*nb*4) || !ensure(v,&v->yb,(size_t)n*O*4)) return -1;
    vkbuf xb=v->xb, xs=v->xs, xm=v->xm, yb=v->yb;
    if (!upload(v,&xb,a->q,(size_t)n*I)) return -1;
    if (!upload(v,&xs,a->scale,(size_t)n*nb*4)) return -1;
    if (!upload(v,&xm,a->sum,(size_t)n*nb*4)) return -1;
    if (gemm_on_device(v,pipe,wbuf,wsbuf,I,O,n,xb,xs,xm,yb)!=0) return -1;
    download(v,&yb,y,(size_t)n*O*4);
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
int coli_vk_upload_w4(coli_vk *v, const coli_w_i4 *w) {
    if (!v->pipe4) return -1;             /* no int4 shader built; caller stays on int8 */
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
    if (!upload(v,&v->W4[h].w, w->q4,     wn)) return -1;
    if (!upload(v,&v->W4[h].ws,w->bscale, sn)) return -1;
    if (!v->memdesc2[0]) snprintf(v->memdesc2,sizeof v->memdesc2,"HOST_VISIBLE");
    v->W4[h].I=w->I; v->W4[h].O=w->O; v->W4[h].used=1;
    v->nw4++;
    return h;
}

int coli_vk_gemm4(coli_vk *v, int wh, const coli_a_i8 *a, float *y) {
    if (!v->pipe4) return -1;
    if (wh<0 || wh>=v->nw4 || !v->W4[wh].used) return -1;
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
    int64_t D = v->W4[hg].I, EI = v->W4[hg].O, Dout = v->W4[hd].O;
    if (a->I != D || v->W4[hu].I != D || v->W4[hu].O != EI || v->W4[hd].I != EI) return -1;
    if (EI % COLI_ABLK) return -1;
    int n = a->n;
    int64_t nbD = D/COLI_ABLK, nbE = EI/COLI_ABLK;

    if (!ensure(v,&v->xb,(size_t)n*D) || !ensure(v,&v->xs,(size_t)n*nbD*4) ||
        !ensure(v,&v->xm,(size_t)n*nbD*4) || !ensure(v,&v->yb,(size_t)n*Dout*4)) return -1;
    if (!ensure(v,&v->fg,(size_t)n*EI*4) || !ensure(v,&v->fu,(size_t)n*EI*4) ||
        !ensure(v,&v->hq,(size_t)n*EI)   || !ensure(v,&v->hs,(size_t)n*nbE*4) ||
        !ensure(v,&v->hm,(size_t)n*nbE*4)) return -1;

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
    record_dispatch(v,v->pipe4,v->dsf[0],D,EI,n,(uint32_t)((int64_t)n*EI));
    record_dispatch(v,v->pipe4,v->dsf[1],D,EI,n,(uint32_t)((int64_t)n*EI));
    record_barrier(v);
    {   /* the nonlinearity: one invocation per 16-element block */
        uint32_t blocks = (uint32_t)((int64_t)n*nbE);
        record_dispatch(v,v->pipe_smq,v->dsf[2],EI,0,n,(blocks+63)/64);
    }
    record_barrier(v);
    record_dispatch(v,v->pipe4,v->dsf[3],EI,Dout,n,(uint32_t)((int64_t)n*Dout));
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.ffn_n++;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    P.sub_ns += now_ns()-tsub; P.sub_n++;

    /* the ONE download */
    download(v,&v->yb,y,(size_t)n*Dout*4);
    return 0;
}

/* Same weights, same buffers, same access pattern -- only the arithmetic differs.
 * See shaders/gemm_i4f.comp. */
int coli_vk_gemm4f(coli_vk *v, int wh, const coli_a_i8 *a, float *y) {
    if (!v->pipe4f) return -1;
    if (wh<0 || wh>=v->nw4 || !v->W4[wh].used) return -1;
    return gemm_dispatch(v, v->pipe4f, v->W4[wh].w, v->W4[wh].ws,
                         v->W4[wh].I, v->W4[wh].O, a, y);
}

void coli_vk_free(coli_vk *v) {
    if (!v) return;
    for (int i=0;i<v->nw;i++)  if (v->W[i].used)  { freebuf(v,&v->W[i].w);  freebuf(v,&v->W[i].ws); }
    for (int i=0;i<v->nw4;i++) if (v->W4[i].used) { freebuf(v,&v->W4[i].w); freebuf(v,&v->W4[i].ws); }
    freebuf(v,&v->xb); freebuf(v,&v->xs); freebuf(v,&v->xm); freebuf(v,&v->yb);
    freebuf(v,&v->fg); freebuf(v,&v->fu); freebuf(v,&v->hq); freebuf(v,&v->hs); freebuf(v,&v->hm);
    if (v->fence) vkDestroyFence(v->dev,v->fence,NULL);
    if (v->pool)  vkDestroyCommandPool(v->dev,v->pool,NULL);
    for (int j=0;j<3;j++) freebuf(v,&v->yq[j]);
    if (v->dpool) vkDestroyDescriptorPool(v->dev,v->dpool,NULL);
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
    fprintf(f,"\n--- vk phase breakdown (%llu gemm + %llu ffn = %llu submissions) ---\n",
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
        if (!ensure(v,&v->yq[j],(size_t)n*v->W4[wh[j]].O*4)) return -1;

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
    for (int j=0;j<3;j++) {
        int64_t O = v->W4[wh[j]].O;
        record_dispatch(v,v->pipe4,v->dsq[j],I,O,n,(uint32_t)((int64_t)n*O));
    }
    vkEndCommandBuffer(v->cmd);
    P.rec_ns += now_ns()-trec; P.gemm_n += 3;

    vkResetFences(v->dev,1,&v->fence);
    uint64_t tsub = now_ns();
    { VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
      if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) return -1; }
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) return -1;
    P.sub_ns += now_ns()-tsub; P.sub_n++;

    for (int j=0;j<3;j++)
        if (!download(v,&v->yq[j],ys[j],(size_t)n*v->W4[wh[j]].O*4)) return -1;
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
