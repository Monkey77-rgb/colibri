/* The submit floor, with NO model loaded.
 *
 * Deliberately standalone: measuring this through the CLI would upload ~1.7 GiB
 * of weights first, and on the Legion the GTT carve-out is already ~88% consumed
 * by live services. The number we want has nothing to do with the weights, so
 * paying that cost -- and risking eviction on a production host -- to obtain it
 * would be a self-inflicted wound. Init the device, submit nothing, measure. */
#include "vk_backend.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 2000;
    char err[256] = {0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { fprintf(stderr, "no Vulkan device: %s\n", err); return 1; }
    printf("device : %s (%s)\n", coli_vk_device_name(v),
           coli_vk_is_integrated(v) ? "integrated" : "discrete");
    printf("memory : %s\n", coli_vk_mem_desc(v));
    /* Repeat the whole measurement: a single sample cannot separate the floor
     * from a scheduler hiccup, and this workspace has published noise before. */
    for (int r = 0; r < 5; r++) {
        double ns = coli_vk_probe_submit_ns(v, reps);
        if (ns < 0) { fprintf(stderr, "probe failed\n"); coli_vk_free(v); return 1; }
        printf("run %d: %8.2f us per empty submit+fence  (%d reps)\n", r + 1, ns / 1000.0, reps);
    }
    coli_vk_free(v);
    return 0;
}
