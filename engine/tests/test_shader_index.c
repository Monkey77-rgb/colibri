/* test_shader_index.c — does gemm_i4.comp INDEX the same bytes the CPU kernel
 * does? Runs on the CPU, needs no GPU and no Vulkan headers.
 *
 * WHY THIS EXISTS AS A SEPARATE TEST. The GPU test (tests/test_vk_gemm.c) is the
 * real check, but it cannot be built here: the Vulkan headers lived in /tmp and
 * did not survive a reboot, and installing them is a system change that is not
 * mine to make. Leaving the shader entirely unverified until that is resolved
 * would mean shipping a kernel nobody has ever executed, so this covers the part
 * that can be covered without a device.
 *
 * WHAT IT PROVES, exactly: that the shader's ADDRESSING is right -- nibble
 * order, which activation word a weight word pairs with, which activation scale
 * and which weight-block scale apply. Those are where a new quantized kernel
 * actually goes wrong, and they are pure integer arithmetic, so they transliterate
 * from GLSL to C exactly.
 *
 * WHAT IT DOES NOT PROVE, and must not be read as proving: that the shader
 * compiles for a device, that Vulkan binds the buffers in this order, that the
 * workgroup reduction is correct, or that the GPU produces this answer. It is a
 * transliteration of the shader body, not the shader. If the two ever diverge,
 * this test goes on passing while the GPU is wrong -- so it is a floor, not a
 * substitute, and tests/test_vk_gemm.c remains the check that matters.
 *
 * The transliteration is deliberately literal, including the loop bounds and the
 * variable names, so a reviewer can diff it against the .comp by eye.
 */
#include "../src/gemm_i8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* bitfieldExtract(value, offset, bits) for unsigned, as GLSL defines it. */
static inline unsigned bfe(unsigned v, int off, int bits) {
    return (v >> off) & ((1u << bits) - 1u);
}

/* The body of gemm_i4.comp's main(), with the 64-lane workgroup collapsed into
 * a sequential loop. The tree reduction is replaced by a plain sum: this test is
 * about addressing, and float ordering is covered by the tolerance below. */
static void shader_emul(float *y, const coli_a_i8 *a, const coli_w_i4 *w) {
    int I = (int)w->I, O = (int)w->O, n = a->n, nblk = I / COLI_ABLK;
    const unsigned *w_packed = (const unsigned*)w->q4;
    const float    *w_bscale = w->bscale;
    const unsigned *x_packed = (const unsigned*)a->q;
    const float    *x_scale  = a->scale;

    for (int r = 0; r < n; r++)
    for (int o = 0; o < O; o++) {
        int wwords = I / 8;
        int wbase  = o * wwords;
        int xwords = I / 4;
        int xbase  = r * xwords;
        int wnb    = I / 32;

        double acc = 0.0;            /* double here so the sum order is not the thing under test */
        for (int wi = 0; wi < wwords; wi++) {
            unsigned wv = w_packed[wbase + wi];
            int e = wi * 8;
            unsigned xw0 = x_packed[xbase + (e / 4)];
            unsigned xw1 = x_packed[xbase + (e / 4) + 1];
            int s = 0;
            for (int b = 0; b < 8; b++) {
                int q = (int)bfe(wv, b * 4, 4) - 8;
                unsigned xw = (b < 4) ? xw0 : xw1;
                int xq = (int)bfe(xw, (b & 3) * 8, 8);
                xq = (xq > 127) ? xq - 256 : xq;
                s += q * xq;
            }
            int ab = e / 16;
            int wb = e / 32;
            acc += (double)w_bscale[o * wnb + wb] * (double)x_scale[r * nblk + ab] * (double)s;
        }
        y[(long)r * O + o] = (float)acc;
    }
}

int main(int argc, char **argv) {
    int I = argc > 1 ? atoi(argv[1]) : 512, O = argc > 2 ? atoi(argv[2]) : 256;
    int n = 4;
    if (I % 32) { printf("I must be a multiple of 32\n"); return 2; }

    float *F = (float*)malloc((size_t)I*O*4);
    srand(7);
    for (long i = 0; i < (long)I*O; i++) { float u=(float)(rand()%20001-10000)/10000.f; F[i]=u*u*u*0.1f; }
    coli_w_i4 w4; coli_quantize_w4(&w4, F, I, O);

    int nb = I/COLI_ABLK;
    coli_a_i8 a = {0}; a.I = I;
    a.q     = (int8_t*) malloc((size_t)I*n);
    a.scale = (float*)  malloc((size_t)nb*n*4);
    a.sum   = (int32_t*)malloc((size_t)nb*n*4);
    float *X = (float*)malloc((size_t)I*n*4);
    for (long i = 0; i < (long)I*n; i++) X[i] = (float)((rand()%2001)-1000)/500.0f;
    coli_quantize_a(&a, X, n, I);

    float *Ycpu = (float*)malloc((size_t)O*n*4);
    float *Yshd = (float*)malloc((size_t)O*n*4);
    coli_gemm_i4(Ycpu, &a, &w4);
    shader_emul(Yshd, &a, &w4);

    /* An indexing bug moves a whole term, so it shows up as a LARGE relative
     * error; float reassociation shows up as a tiny one. 1e-6 separates them
     * with room to spare, and the control below confirms the comparison can
     * actually fail. */
    const double TOL = 1e-6;
    double ymax = 0, maxd = 0;
    for (long i = 0; i < (long)n*O; i++) { double c = fabs(Ycpu[i]); if (c > ymax) ymax = c; }
    for (long i = 0; i < (long)n*O; i++) { double d = fabs((double)Ycpu[i]-(double)Yshd[i]); if (d > maxd) maxd = d; }
    double rel = maxd/(ymax > 0 ? ymax : 1);
    printf("shader addressing vs coli_gemm_i4: rel=%.3e  %s\n", rel, rel < TOL ? "ok" : "MISMATCH");

    /* CONTROL: swap the nibble order in the emulation only -- the single most
     * likely way to get this wrong. It must be detected, or the comparison is
     * not testing addressing at all. */
    for (long i = 0; i < (long)n*O; i++) Yshd[i] = 0;
    {
        int wwords = I/8, wnb = I/32;
        const unsigned *w_packed=(const unsigned*)w4.q4, *x_packed=(const unsigned*)a.q;
        for (int r = 0; r < n; r++) for (int o = 0; o < O; o++) {
            double acc = 0;
            for (int wi = 0; wi < wwords; wi++) {
                unsigned wv = w_packed[o*wwords + wi]; int e = wi*8;
                unsigned xw0=x_packed[r*(I/4)+e/4], xw1=x_packed[r*(I/4)+e/4+1];
                int s = 0;
                for (int b = 0; b < 8; b++) {
                    int q = (int)bfe(wv, (b ^ 1) * 4, 4) - 8;      /* <-- nibbles swapped */
                    unsigned xw = (b<4)?xw0:xw1;
                    int xq=(int)bfe(xw,(b&3)*8,8); xq=(xq>127)?xq-256:xq;
                    s += q*xq;
                }
                acc += (double)w4.bscale[o*wnb + e/32] * (double)a.scale[r*nb + e/16] * (double)s;
            }
            Yshd[(long)r*O+o] = (float)acc;
        }
    }
    double cd = 0;
    for (long i = 0; i < (long)n*O; i++) { double d = fabs((double)Ycpu[i]-(double)Yshd[i]); if (d > cd) cd = d; }
    double crel = cd/(ymax > 0 ? ymax : 1);
    printf("control (nibbles swapped):          rel=%.3e  %s\n", crel,
           crel >= TOL ? "detected, as required" : "NOT DETECTED -- this test proves nothing");

    coli_free_w4(&w4);
    int fail = (rel >= TOL) || (crel < TOL);
    printf("%s\n", fail ? "FAIL" : "PASS (addressing only -- the GPU itself is NOT exercised here)");
    return fail;
}
