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
    /* The 5 runs used to be printed with nothing aggregated, leaving the reader
     * to pick which one was "the floor". Aggregate here and say which is which:
     * the FLOOR is the best min across runs; the mean-vs-min gap is contention. */
    double best_min = 1e30, best_mean = 1e30, worst_mean = 0;
    for (int r = 0; r < 5; r++) {
        double mn = -1;
        double ns = coli_vk_probe_submit_ns(v, reps, &mn);
        if (ns < 0) { fprintf(stderr, "probe failed\n"); coli_vk_free(v); return 1; }
        printf("run %d: mean %8.2f us   min %8.2f us  (%d reps)\n",
               r + 1, ns / 1000.0, mn / 1000.0, reps);
        if (mn < best_min)  best_min  = mn;
        if (ns < best_mean) best_mean = ns;
        if (ns > worst_mean) worst_mean = ns;
    }
    printf("\nFLOOR  : %.2f us  (best min over 5 x %d reps)\n", best_min/1000.0, reps);
    printf("mean   : %.2f - %.2f us across runs\n", best_mean/1000.0, worst_mean/1000.0);
    printf("spread : mean/min = %.2fx -- this is CONTENTION on this machine at this\n"
           "         moment, not a property of the device. State the machine's load\n"
           "         next to the floor or the number is not reproducible.\n",
           best_mean / best_min);
    coli_vk_free(v);
    return 0;
}
