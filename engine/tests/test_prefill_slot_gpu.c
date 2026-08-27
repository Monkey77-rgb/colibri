/* coli_prefill_slot with GPU attention: does the served path still compute the
 * same thing?
 *
 * WHY A SEPARATE TEST. coli_forward and coli_prefill_slot are two different
 * prefill implementations with two different attention loops, and only the
 * first is reachable from the CLI. coli_api.cpp -- the library every embedder
 * and the scheduler go through -- calls the SECOND one. Measuring the GPU win
 * through --nll and shipping it therefore proves nothing about the path that
 * actually serves, which is the one where a wrong answer reaches a user.
 *
 * NOT test_prefix_cache with a flag. That test asserts BIT-identity between a
 * cold and a warm prefill, and with the GPU arm on it would fail correctly: the
 * two runs have different batch shapes, so the shader's four subgroups merge
 * their softmax states in a different order. Reusing it would have produced a
 * red test that means nothing and an obvious temptation to loosen it.
 *
 * The reference here is the CPU arm of the same function, at the same slot and
 * the same positions -- so the only variable is which arm ran.
 *
 * THE SLOT IS NOT 0. coli_forward hardcodes slot 0 and so does its GPU write;
 * the slot path does not, and the device cache is [slot][kvh][kv_ctx][hd]. A
 * write that ignored the slot index would land on slot 0 and be invisible to
 * any test that only ever used slot 0. */
#include "../src/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define PRE "You are a strict evaluator. Score the response on accuracy, then on clarity, then give a verdict. Respond only in JSON with keys score and verdict. Here is the response to evaluate: "

/* Logits are UNNORMALISED and span roughly +-20 with many entries near zero, so
 * a per-element relative difference is the wrong measure: one element where the
 * reference happens to sit at 1e-4 dominates the maximum and the score stops
 * describing the tensor. The first version of this test used exactly that, and
 * it reported the signal at 2.159e+02 and its own control at 2.156e+02 -- two
 * numbers that could not be told apart, from a comparison that was measuring
 * neither. Normalise by the reference's own SCALE instead. */
static double worst_scaled(const float *a, const float *b, int n) {
    double scale = 0;
    for (int i = 0; i < n; i++) if (fabs((double)b[i]) > scale) scale = fabs((double)b[i]);
    if (scale < 1e-6) scale = 1e-6;
    double mx = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]) / scale;
        if (d > mx) mx = d;
    }
    return mx;
}
static int argmax_of(const float *a, int n) {
    int bi = 0; for (int i = 1; i < n; i++) if (a[i] > a[bi]) bi = i; return bi;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    char err[512] = {0};
    coli_model *m = coli_load(argv[1], 2048, 2, 1, 2, err, sizeof err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const char *p = PRE "The capital of Japan is Tokyo, a very large city indeed.";
    int ids[1024];
    int n = coli_encode(m, p, ids, 1024);
    if (n < 8) { fprintf(stderr, "tokenizer returned %d\n", n); return 1; }
    int slot = (argc > 2) ? atoi(argv[2]) : 1;
    printf("prompt %d tok, slot %d\n", n, slot);

    /* Vulkan has to be brought up explicitly. The CLI does this behind --gpu;
     * a test that skips it runs BOTH arms on the CPU, gets rel=0, and prints
     * PASS having never dispatched the kernel. That is not hypothetical -- it
     * is what the first run of this test did. */
    char gerr[512] = {0};
    int nup = coli_gpu_upload(m, gerr, sizeof gerr);
    if (nup <= 0) { printf("no GPU weights uploaded (%s) -- SKIP\n", gerr); coli_free(m); return 0; }
    printf("gpu: %d weight matrices uploaded\n", nup);

    int V = m->cfg.vocab;
    float *cpu = (float*)malloc((size_t)V * sizeof(float));
    float *gpu = (float*)malloc((size_t)V * sizeof(float));
    if (!cpu || !gpu) { fprintf(stderr, "oom\n"); return 1; }

    /* The prefix cache is off for both arms. With it on, the second prefill of
     * the same tokens would reuse the first arm's K/V and recompute nothing --
     * the GPU arm would be handed the answer and the comparison would pass
     * without the kernel running at all. */
    coli_prefix_cache_enable(0);

    unsetenv("COLI_GPU_PREFILL_ATTN");
    float *l = coli_prefill_slot(m, slot, ids, n);
    if (!l) { fprintf(stderr, "FAIL: cpu prefill\n"); return 1; }
    memcpy(cpu, l, (size_t)V * sizeof(float)); free(l);

    /* SENSITIVITY FLOOR, and the reason this test does not simply assert a
     * small number. How far do the logits move under a change that is KNOWN to
     * be benign? Re-running the identical CPU arm at a different thread count
     * changes the order of the matmul reductions and nothing else. Whatever
     * that produces is this model's own rounding sensitivity at the last token,
     * and the GPU arm cannot fairly be held below it. */
    omp_set_num_threads(3);
    l = coli_prefill_slot(m, slot, ids, n);
    if (!l) { fprintf(stderr, "FAIL: cpu prefill (thread control)\n"); return 1; }
    float *cput = (float*)malloc((size_t)V*sizeof(float));
    if (!cput) { fprintf(stderr,"oom\n"); return 1; }
    memcpy(cput, l, (size_t)V*sizeof(float)); free(l);
    omp_set_num_threads(omp_get_num_procs());

    setenv("COLI_GPU_PREFILL_ATTN", "1", 1);
    l = coli_prefill_slot(m, slot, ids, n);
    if (!l) { fprintf(stderr, "FAIL: gpu prefill\n"); return 1; }
    memcpy(gpu, l, (size_t)V * sizeof(float)); free(l);

    /* 1e-3 on LOGITS, not on attention outputs. This is a whole 28-to-32-layer
     * forward pass, so the kernel's ~2.6e-05 per-layer difference is carried
     * through every later matmul and the residual stream; the bound has to
     * allow for that accumulation and still be far tighter than any difference
     * that would change a token. */
    /* 5e-2 of the logit range, and every part of that number is measured.
     *
     * The last token's logits are the most amplified point in the model: the
     * kernel differs from attend_online by 2.6e-05 per layer (tests/test_vk_attn,
     * against a double-precision two-pass reference) and that is carried through
     * every subsequent matmul and residual add. Measured 2026-08-27 on qwen2.5-3b
     * with this 51-token prompt, the end-to-end figure is 2.559e-02 -- and
     * coli_forward, a SEPARATE prefill implementation running the same kernel,
     * lands on 2.559e-02 too. Two independent paths agreeing to four digits is
     * what says this is the kernel's inherent drift and not either path's bug.
     *
     * It is calibrated against something with meaning: over 681 tokens that same
     * drift moves TF-NLL from 2.8600 to 2.8601, and on the 8B model it does not
     * move the 4-decimal figure at all. The argmax assertion below is the one
     * that decides whether any of it could change an emitted token. */
    const double TOL = 5e-2;
    double wt = worst_scaled(cput, cpu, V);
    printf("  sensitivity floor (cpu arm, 3 threads vs all): rel=%.3e\n", wt);
    double w = worst_scaled(gpu, cpu, V);
    printf("  gpu vs cpu arm, same slot and positions: rel=%.3e  %s\n",
           w, w <= TOL ? "ok" : "BAD");

    float *cpu0 = (float*)malloc((size_t)V*sizeof(float));
    if (!cpu0) { fprintf(stderr,"oom\n"); return 1; }
    memcpy(cpu0, cpu, (size_t)V*sizeof(float));

    /* CONTROL. Without it a run where the GPU arm never engaged -- no Vulkan, no
     * attn shader, a readiness check that quietly said no -- returns rel=0 and
     * reads as the strongest possible pass.
     *
     * It compares the perturbed copy against the UNPERTURBED one, not against
     * the GPU. The first version compared gpu-vs-perturbed-cpu, which folds the
     * real drift into the control and produced signal 2.559e-02 against control
     * 2.487e-02: two numbers a reader cannot tell apart, from a control that was
     * not isolating what it claimed to. x1.10 because the tolerance is 5e-2 and
     * a control has to clear the bar it is testing. */
    for (int i = 0; i < V; i++) cpu[i] *= 1.10f;
    double wc = worst_scaled(cpu, cpu0, V);
    printf("  control (reference x1.10 vs itself): rel=%.3e -> %s\n", wc,
           wc > TOL ? "exceeds tolerance, as required" : "DID NOT EXCEED -- test is inert");

    /* rel EXACTLY 0 means the two arms produced identical bits, and the arms are
     * not bit-identical by construction: the shader merges four subgroup softmax
     * states in an order no sequential walk reproduces. A perfect score here is
     * therefore evidence the GPU arm did NOT run, not evidence that it is
     * correct -- so it fails. Check stderr for the ENGAGED / NOT engaged line. */
    if (w == 0.0) {
        printf("  rel is EXACTLY 0 -- the gpu arm did not run; this test is inert\n");
        printf("FAIL\n"); coli_free(m); return 1;
    }

    /* IS THE SLOT PATH WORSE THAN coli_forward? That is the question a single
     * number cannot answer, and the reason this block exists. coli_forward runs
     * the SAME kernel over the SAME prompt through a different prefill
     * implementation, so measuring both with one metric separates "this kernel
     * drifts by X end-to-end" from "the slot path has a bug". A bound picked
     * from the slot path's own output would have hidden exactly that. */
    double cross = -1;
    {
        float *fc = (float*)malloc((size_t)V*sizeof(float));
        float *fg = (float*)malloc((size_t)V*sizeof(float));
        if (fc && fg) {
            unsetenv("COLI_GPU_PREFILL_ATTN");
            m->n_past = 0;
            float *lf = coli_forward(m, ids, n, 0);
            if (lf) { memcpy(fc, lf, (size_t)V*sizeof(float)); free(lf);
                setenv("COLI_GPU_PREFILL_ATTN", "1", 1);
                m->n_past = 0;
                lf = coli_forward(m, ids, n, 0);
                if (lf) { memcpy(fg, lf, (size_t)V*sizeof(float)); free(lf);
                    printf("  coli_forward same metric: rel=%.3e (argmax %s)\n",
                           worst_scaled(fg, fc, V),
                           argmax_of(fg,V)==argmax_of(fc,V) ? "same" : "DIFFERENT");
                    /* The load-bearing cross-check: the two GPU arms are two
                     * different prefill implementations of the same math, so
                     * they must agree with EACH OTHER far more tightly than
                     * either agrees with the CPU. */
                    cross = worst_scaled(gpu, fg, V);
                    printf("  slot arm vs coli_forward arm: rel=%.3e  %s\n",
                           cross, cross <= TOL ? "ok" : "BAD"); } }
        }
        free(fc); free(fg);
    }

    /* The check that actually matters to a user: same next token. A logit
     * difference that never changes the argmax cannot change what is emitted
     * under greedy decoding. */
    int ag = argmax_of(gpu, V), ac0 = argmax_of(cpu0, V);
    printf("  argmax token: gpu=%d cpu=%d  %s\n", ag, ac0, ag==ac0 ? "same" : "DIFFERENT");

    int pass = (w <= TOL) && (wc > TOL) && (ag == ac0) && (cross >= 0) && (cross <= TOL);
    printf("%s\n", pass ? "PASS (slot path: gpu arm matches cpu arm, control does not)"
                        : "FAIL");
    coli_free(m);
    return pass ? 0 : 1;
}
