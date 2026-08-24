/* coopmat_probe — what cooperative-matrix shapes does this device actually support?
 *
 * WHY THIS EXISTS. Tier 2 of closing the prefill gap is a cooperative-matrix
 * (tensor core) GEMM, and that kernel cannot be written without knowing the
 * exact (M,N,K) and type combinations the driver offers -- they are not a
 * property of the API, they are enumerated per device. vulkaninfo on this box
 * does not print them, so guessing would mean writing a shader that fails
 * pipeline creation and, given how this backend used to behave, fall back
 * silently to the kernel it was meant to replace.
 *
 * Build: cc -O2 tools/coopmat_probe.c -lvulkan -o /tmp/coopmat_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const char *scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:      return "Device";
        case VK_SCOPE_WORKGROUP_KHR:   return "Workgroup";
        case VK_SCOPE_SUBGROUP_KHR:    return "Subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR:return "QueueFamily";
        default: return "?";
    }
}
static const char *ct_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "fp16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "fp32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "fp64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "sint8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "sint16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "sint32";
        case VK_COMPONENT_TYPE_SINT64_KHR:  return "sint64";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "uint8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "uint16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "uint32";
        case VK_COMPONENT_TYPE_UINT64_KHR:  return "uint64";
        default: return "?";
    }
}

int main(void) {
    VkApplicationInfo app = { .sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="coopmat_probe", .apiVersion=VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app };
    VkInstance inst;
    if (vkCreateInstance(&ici,NULL,&inst) != VK_SUCCESS) { puts("no vulkan instance"); return 1; }

    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,NULL);
    VkPhysicalDevice *pd = malloc(nd*sizeof *pd);
    vkEnumeratePhysicalDevices(inst,&nd,pd);

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst,"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fn) { puts("VK_KHR_cooperative_matrix entry point absent"); return 2; }

    for (uint32_t d=0; d<nd; d++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[d],&p);
        printf("device %u: %s\n", d, p.deviceName);
        uint32_t n=0;
        if (fn(pd[d],&n,NULL) != VK_SUCCESS || n==0) { puts("  (no cooperative matrix configs)"); continue; }
        VkCooperativeMatrixPropertiesKHR *props = calloc(n,sizeof *props);
        for (uint32_t i=0;i<n;i++) props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        fn(pd[d],&n,props);
        printf("  %u configurations\n", n);
        printf("  %-4s %-4s %-4s  %-7s %-7s %-7s %-7s  %-10s %s\n",
               "M","N","K","A","B","C","Result","scope","sat");
        for (uint32_t i=0;i<n;i++) {
            VkCooperativeMatrixPropertiesKHR *q=&props[i];
            printf("  %-4u %-4u %-4u  %-7s %-7s %-7s %-7s  %-10s %s\n",
                   q->MSize,q->NSize,q->KSize,
                   ct_name(q->AType),ct_name(q->BType),ct_name(q->CType),ct_name(q->ResultType),
                   scope_name(q->scope), q->saturatingAccumulation?"yes":"no");
        }
        free(props);
    }
    return 0;
}
