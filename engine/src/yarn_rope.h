/* yarn_rope.h -- the YaRN cos/sin table as plain arrays, shared by arch_ops.cpp
 * (tests) and model.cpp (the forward pass).
 *
 * WHY A SEPARATE HEADER. model.cpp and arch_ops.h each define their own
 * coli_rope_tab / COLI_ROPE_MAXHALF (512 vs 256), so model.cpp cannot include
 * arch_ops.h. This declares only the array form, which both can call.
 *
 * truncate=0: OpenAI's reference (gpt-oss torch model.py, and HF config.json
 *   "rope_scaling": {"truncate": false} for openai/gpt-oss-120b, fetched
 *   2026-09-14) -- ramp bounds low/high used as real numbers.
 * truncate=1: ggml's form -- ggml_rope_yarn_corr_dims() floors low and ceils
 *   high (llama.cpp b9766 ggml/src/ggml.c:4338-4341), clamped to [0, hd-1]. On
 *   gpt-oss-120b that is low 8.09 -> 8 and high 17.4 -> 18, so frequencies 9..17
 *   differ slightly. Exists so an oracle against llama.cpp can tell this known
 *   difference apart from a bug. */
#ifndef COLI_YARN_ROPE_H
#define COLI_YARN_ROPE_H
#ifdef __cplusplus
extern "C" {
#endif
void coli_yarn_rope_cs(float *c, float *s, int half, int pos, int hd, float base,
                       float factor, float beta_fast, float beta_slow, int orig_ctx,
                       int truncate);
#ifdef __cplusplus
}
#endif
#endif
