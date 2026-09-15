/* test_hw_detect.c — CPU-only assertions always apply; the GPU assertions are
 * gated on COLI_TEST_EXPECT_GPU=1 so `make tests/test_hw_detect` stays
 * runnable (and a clean PASS) on a box with no Vulkan device and no NVIDIA
 * driver. This machine (the desktop, 9800X3D + RTX 4070) sets that env var
 * when run via the `test-hw-detect` Makefile target below.
 *
 * APPARATUS CONTROL (per this project's own doctrine: a comparison that
 * cannot fail proves nothing). Two independent checks that the planner can
 * actually reach its "unavailable" branches:
 *   (a) build_backends=0 must force backend="cpu" with a reason that says so,
 *       even when the probed hw has a real GPU -- if this ever reported
 *       "cuda"/"vulkan" the mask would be decorative, not load-bearing.
 *   (b) a dense-model size larger than the GPU's own VRAM must force
 *       moe_vram_mb == 0 -- if this ever returned a positive number the
 *       headroom arithmetic would be silently wrapping or the clamp missing.
 */
#include "../src/hw_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } \
    else { fprintf(stderr, "ok:   " __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

static void print_plan(const char *label, const coli_hw_plan *p) {
    fprintf(stderr, "plan[%s]: backend=%s threads=%d moe_vram_mb=%d expert_store_gb=%d gpu_attn=%d\n"
                     "          reason: %s\n",
            label, p->backend, p->threads, p->moe_vram_mb, p->expert_store_gb, p->gpu_attn, p->reason);
}

int main(void) {
    coli_hw hw;
    CHECK(coli_hw_probe(&hw) == 1, "coli_hw_probe returns 1 (absence is never an error return)");

    fprintf(stderr, "--- coli_hw_print ---\n");
    coli_hw_print(&hw, stderr);
    fprintf(stderr, "--- end coli_hw_print ---\n");

    /* Always true on any machine this builds on. */
    CHECK(hw.cpu_logical_cores > 0, "cpu_logical_cores > 0 (got %d)", hw.cpu_logical_cores);
    CHECK(hw.cpu_physical_cores > 0, "cpu_physical_cores > 0 (got %d)", hw.cpu_physical_cores);
    CHECK(hw.ram_total_bytes > 0, "ram_total_bytes > 0 (got %llu)", (unsigned long long)hw.ram_total_bytes);

    /* Machine-specific: this repo's desktop is a 9800X3D + RTX 4070 with the
     * NVIDIA proprietary driver and libvulkan.so.1 installed. A different
     * machine (no GPU) must still PASS -- hence the env gate, not a bare
     * assert -- see the file header. */
    if (getenv("COLI_TEST_EXPECT_GPU") && !strcmp(getenv("COLI_TEST_EXPECT_GPU"), "1")) {
#ifdef COLI_HAVE_VK
        CHECK(hw.n_vk >= 1, "COLI_TEST_EXPECT_GPU=1: n_vk >= 1 (got %d)", hw.n_vk);
#else
        /* A build without Vulkan cannot see a Vulkan device; asserting one
         * here would fail for the build, not the hardware (2026-09-14). */
        CHECK(hw.n_vk == 0, "no-VK build reports n_vk == 0 (got %d)", hw.n_vk);
#endif
        CHECK(hw.cuda.present == 1, "COLI_TEST_EXPECT_GPU=1: cuda.present == 1 (got %d)", hw.cuda.present);
        CHECK(hw.cuda.device_count >= 1, "COLI_TEST_EXPECT_GPU=1: cuda.device_count >= 1 (got %d)", hw.cuda.device_count);
    } else {
        fprintf(stderr, "skip: GPU-specific assertions (set COLI_TEST_EXPECT_GPU=1 on a box known to have one)\n");
    }

    /* Plans under every `prefer`, on whatever hw this box actually has. */
    coli_hw_plan p;
    coli_hw_plan_make(&hw, NULL, 0, 0, &p);       print_plan("prefer=NULL", &p);
    coli_hw_plan_make(&hw, "cuda", 0, 0, &p);     print_plan("prefer=cuda", &p);
    coli_hw_plan_make(&hw, "vulkan", 0, 0, &p);   print_plan("prefer=vulkan", &p);
    coli_hw_plan_make(&hw, "cpu", 0, 0, &p);      print_plan("prefer=cpu", &p);
    CHECK(!strcmp(p.backend, "cpu"), "prefer=cpu always yields backend=cpu (got %s)", p.backend);

    /* --- apparatus control (a): build_backends=0 must force cpu --- */
    coli_hw_plan pa;
    coli_hw_plan_make_ex(&hw, "auto", 0, 0, 0u, &pa);
    print_plan("build_backends=0", &pa);
    CHECK(!strcmp(pa.backend, "cpu"), "build_backends=0 forces backend=cpu even with a real GPU probed (got %s)", pa.backend);
    CHECK(strstr(pa.reason, "build") != NULL || strstr(pa.reason, "cpu only") != NULL,
          "build_backends=0 reason names the missing build (got: %s)", pa.reason);

    /* --- apparatus control (b): dense bytes > VRAM must force moe_vram_mb==0 --- */
    uint64_t huge = (uint64_t)1 << 40; /* 1 TiB -- larger than any VRAM this box has */
    coli_hw_plan pb;
    coli_hw_plan_make_ex(&hw, "auto", huge, 0, COLI_BE_VULKAN | COLI_BE_CUDA, &pb);
    print_plan("dense_bytes=1TiB", &pb);
    CHECK(pb.moe_vram_mb == 0, "dense_bytes larger than VRAM forces moe_vram_mb==0 (got %d, backend=%s)",
          pb.moe_vram_mb, pb.backend);

    fprintf(stderr, "%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
