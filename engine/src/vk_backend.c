/* vk_backend.c — see vk_backend.h for scope and the bit-exactness exception. */
#define _GNU_SOURCE
#include "vk_backend.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VKERR(...) do { if (err && errcap) snprintf(err, errcap, __VA_ARGS__); } while (0)
#define MAX_W 64

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
    VkDescriptorPool dpool;
    VkFence fence;
    char devname[256];
    char memdesc[160];
    char memdesc2[64];
    int  integrated;
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
    void *p; if (vkMapMemory(v->dev,b->mem,0,n,0,&p) != VK_SUCCESS) return 0;
    memcpy(p,src,n); vkUnmapMemory(v->dev,b->mem); return 1;
}
static int download(coli_vk *v, vkbuf *b, void *dst, size_t n) {
    void *p; if (vkMapMemory(v->dev,b->mem,0,n,0,&p) != VK_SUCCESS) return 0;
    memcpy(dst,p,n); vkUnmapMemory(v->dev,b->mem); return 1;
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
    free(devs);
    vkGetPhysicalDeviceMemoryProperties(v->pdev,&v->memprops);

    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&nq,NULL);
    VkQueueFamilyProperties *qs = malloc(nq*sizeof *qs);
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&nq,qs);
    v->qfam = UINT32_MAX;
    for (uint32_t i=0;i<nq;i++) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { v->qfam=i; break; }
    free(qs);
    if (v->qfam==UINT32_MAX) { VKERR("no compute queue"); goto fail; }

    float prio=1.f;
    VkDeviceQueueCreateInfo qci = { .sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=v->qfam, .queueCount=1, .pQueuePriorities=&prio };
    VkDeviceCreateInfo dci = { .sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1, .pQueueCreateInfos=&qci };
    if (vkCreateDevice(v->pdev,&dci,NULL,&v->dev) != VK_SUCCESS) { VKERR("vkCreateDevice failed"); goto fail; }
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
    VkPushConstantRange pc = { .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, .offset=0, .size=16 };
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
        VkShaderModule sm4;
        if (load_module(v, p4, &sm4)) {
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

/* Copy through a staging buffer into DEVICE_LOCAL memory. Only worth doing for
 * weights, which are written once and read every step; doing it for activations
 * would add a copy per call to save nothing. */
static int upload_device_local(coli_vk *v, vkbuf *dst, const void *src, size_t n) {
    vkbuf stage = {0};
    if (!mkbuf_flags(v, n, &stage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return 0;
    if (!upload(v,&stage,src,n)) { freebuf(v,&stage); return 0; }
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
    return ok;
}

int coli_vk_upload_w(coli_vk *v, const coli_w_i8 *w) {
    if (v->nw >= MAX_W) return -1;
    int h = v->nw;
    size_t wn = (size_t)w->I*w->O, sn = (size_t)w->O*sizeof(float);

    /* On a DISCRETE card, put the weights in VRAM. They are written once and
     * read on every step, so a one-time staging copy trades load time for the
     * difference between PCIe (~25 GB/s) and VRAM (hundreds). On an integrated
     * GPU the memory is already shared and the copy would be pure waste, so the
     * device type decides -- not a hardcoded preference. */
    if (!v->integrated) {
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

    VkCommandBufferBeginInfo bi={ .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkResetCommandBuffer(v->cmd,0);
    vkBeginCommandBuffer(v->cmd,&bi);
    vkCmdBindPipeline(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(v->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,v->pl,0,1,&ds,0,NULL);
    int32_t push[4] = { (int32_t)I, (int32_t)O, n, (int32_t)nb };
    vkCmdPushConstants(v->cmd,v->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,push);
    vkCmdDispatch(v->cmd,(uint32_t)((int64_t)n*O),1,1);
    vkEndCommandBuffer(v->cmd);

    vkResetFences(v->dev,1,&v->fence);
    VkSubmitInfo si={ .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&v->cmd };
    if (vkQueueSubmit(v->q,1,&si,v->fence)!=VK_SUCCESS) goto cleanup2;
    /* 60 s is a hang, not slowness -- a wedged queue must fail loudly rather
     * than block the caller forever. */
    if (vkWaitForFences(v->dev,1,&v->fence,VK_TRUE,60000000000ull)!=VK_SUCCESS) goto cleanup2;
    download(v,&yb,y,(size_t)n*O*4);
    return 0;
cleanup2:
    return -1;
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

    if (!v->integrated) {
        int okw = mkbuf_flags(v, wn, &v->W4[h].w, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        int oks = okw && mkbuf_flags(v, sn, &v->W4[h].ws, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (oks && upload_device_local(v,&v->W4[h].w,w->q4,wn)
                && upload_device_local(v,&v->W4[h].ws,w->bscale,sn)) {
            v->W4[h].I=w->I; v->W4[h].O=w->O; v->W4[h].used=1; v->nw4++;
            return h;
        }
        freebuf(v,&v->W4[h].w); freebuf(v,&v->W4[h].ws);
    }
    if (!mkbuf(v,wn,&v->W4[h].w))  return -1;
    if (!mkbuf(v,sn,&v->W4[h].ws)) { freebuf(v,&v->W4[h].w); return -1; }
    if (!upload(v,&v->W4[h].w, w->q4,     wn)) return -1;
    if (!upload(v,&v->W4[h].ws,w->bscale, sn)) return -1;
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

void coli_vk_free(coli_vk *v) {
    if (!v) return;
    for (int i=0;i<v->nw;i++)  if (v->W[i].used)  { freebuf(v,&v->W[i].w);  freebuf(v,&v->W[i].ws); }
    for (int i=0;i<v->nw4;i++) if (v->W4[i].used) { freebuf(v,&v->W4[i].w); freebuf(v,&v->W4[i].ws); }
    freebuf(v,&v->xb); freebuf(v,&v->xs); freebuf(v,&v->xm); freebuf(v,&v->yb);
    if (v->fence) vkDestroyFence(v->dev,v->fence,NULL);
    if (v->pool)  vkDestroyCommandPool(v->dev,v->pool,NULL);
    if (v->dpool) vkDestroyDescriptorPool(v->dev,v->dpool,NULL);
    if (v->pipe)  vkDestroyPipeline(v->dev,v->pipe,NULL);
    if (v->pipe4) vkDestroyPipeline(v->dev,v->pipe4,NULL);
    if (v->pl)    vkDestroyPipelineLayout(v->dev,v->pl,NULL);
    if (v->dsl)   vkDestroyDescriptorSetLayout(v->dev,v->dsl,NULL);
    if (v->dev)   vkDestroyDevice(v->dev,NULL);
    if (v->inst)  vkDestroyInstance(v->inst,NULL);
    free(v);
}
