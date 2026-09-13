/* arch_ops.h -- CPU primitives the qwen2/llama/qwen3/qwen3moe path in
 * model.cpp does not need, and gpt-oss does. STANDALONE and NOT wired into
 * model.cpp yet (that is step 3) -- each function here is independently
 * testable against its own scalar oracle (see tests/test_arch_ops.cpp) before
 * anything touches the forward pass coli_load()'s callers depend on.
 *
 * What is deliberately NOT reimplemented here, and why: RoPE's rotation
 * itself (rotate the pair (i, i+half) by (cos,sin)) is UNCHANGED -- gpt-oss
 * uses the identical half-split ("neox") pairing model.cpp's rope_head_tab()
 * already implements (verified against gptoss_model.py's
 * _apply_rotary_emb(): x1,x2 = chunk(x,2); o1=x1*cos-x2*sin; o2=x2*cos+x1*sin
 * -- exactly rope_head_tab's ia=i/ib=i+half form). Only the FREQUENCY TABLE
 * feeding that rotation changes under YaRN, which is what
 * coli_yarn_rope_table() below builds. */
#ifndef COLI_ARCH_OPS_H
#define COLI_ARCH_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_ROPE_MAXHALF 256

/* Same shape as model.cpp's file-local coli_rope_tab (c[]/s[]/half) --
 * duplicated rather than shared because model.cpp's struct is not exposed in
 * any header; the two are BY CONSTRUCTION consumed the same way (a
 * rope_head_tab-style loop: v[i]=v[i]*c[i]-v[i+half]*s[i] etc), which is
 * exactly the property the differential test in test_arch_ops.cpp checks. */
typedef struct { float c[COLI_ROPE_MAXHALF], s[COLI_ROPE_MAXHALF]; int half; } coli_rope_tab;

/* Builds the position-`pos` cos/sin table for a YaRN-scaled RoPE, per the
 * NTK-by-parts formula in OpenAI's own gpt-oss reference implementation
 * (scratchpad/gptoss_model.py, RotaryEmbedding._compute_concentration_and_
 * inv_freq / _compute_cos_sin, read this session) -- which is the formula the
 * gguf's gpt-oss.rope.scaling.* keys were generated to reproduce:
 *
 *   freq[i]      = base^(2i/hd),                      i in [0, hd/2)
 *   if factor > 1 (YaRN active):
 *     concentration = 0.1*ln(factor) + 1
 *     d_half = hd/2
 *     low  = d_half * ln(orig_ctx / (beta_fast*2*pi)) / ln(base)
 *     high = d_half * ln(orig_ctx / (beta_slow*2*pi)) / ln(base)
 *     ramp[i] = clamp((i - low) / (high - low), 0, 1);  mask[i] = 1 - ramp[i]
 *     inv_freq[i] = (1/(factor*freq[i]))*(1-mask[i]) + (1/freq[i])*mask[i]
 *   else: concentration = 1, inv_freq[i] = 1/freq[i]
 *   c[i] = cos(pos*inv_freq[i]) * concentration
 *   s[i] = sin(pos*inv_freq[i]) * concentration
 *
 * factor<=1 reproduces plain (non-YaRN) RoPE exactly (concentration=1,
 * inv_freq=1/freq), so this function is also correct to call for a
 * non-YaRN model -- test_arch_ops.cpp's test_yarn_factor1_matches_plain_rope
 * is the control that checks that rather than assuming it. */
void coli_yarn_rope_table(coli_rope_tab *t, int pos, int hd, float base,
                           float factor, float beta_fast, float beta_slow, int orig_ctx);

/* out[i] = clamp(gate[i], -inf, limit) * sigmoid(alpha*clamp(gate[i],-inf,limit))
 *          * (clamp(up[i], -limit, limit) + 1)
 * Transcribed from gptoss_model.py's swiglu() (read this session): gate gets
 * an UPPER clamp only, up gets clamped BOTH sides and has +1 added -- getting
 * either wrong (e.g. clamping gate on both sides, the natural typo) changes
 * output only for |gate| or |up| > limit=7, which is why the test below
 * exercises inputs on both sides of +-7, not just small values near 0. */
void coli_swiglu_oai(float *out, const float *gate, const float *up, int n,
                      float alpha, float limit);

/* Router top-k-then-softmax-over-selected, WITH an optional per-expert bias
 * added before selection (gpt-oss's router carries one; qwen's does not,
 * pass bias=NULL and this is bit-identical to model.cpp's existing inline
 * moe_ffn() gating -- see arch.h's COLI_GATE_TOPK_SOFTMAX_BIASED comment for
 * the llama.cpp source this was checked against). logits[n_expert] is READ,
 * not modified. sel[k] gets the chosen expert indices (descending by
 * bias-adjusted logit, ties broken by lowest index -- same rule model.cpp's
 * inline loop already uses), wgt[k] the softmax weight over just those k. */
void coli_moe_gate_topk_softmax(const float *logits, const float *bias,
                                 int n_expert, int k, int *sel, float *wgt);

/* attn_sinks: appends `sink` as one extra logit BEFORE the softmax over
 * scores[0..n), then DROPS that column from the output -- i.e. the n output
 * weights do NOT sum to 1 (the remainder is the probability mass the sink
 * absorbed). Transcribed from gptoss_model.py's sdpa(): QK = cat([QK, S]);
 * W = softmax(QK); W = W[..., :-1]. `scores` is read and OVERWRITTEN in
 * place with the post-softmax weights (same convention as an in-place
 * softmax would use). Numerically stable (max-subtracted over all n+1
 * values including the sink). */
void coli_softmax_with_sink(float *scores, int n, float sink);

/* True iff key position `kpos` must be masked out of query position `qpos`'s
 * attention under a sliding window of `window` tokens (window<=0 means "no
 * window, causal only"). Transcribed from gptoss_model.py's sdpa(): causal
 * mask (kpos>qpos) PLUS `mask += tril(fill(-inf), diagonal=-window)`, i.e. an
 * additional mask wherever kpos <= qpos-window. Combined: masked iff
 * kpos>qpos OR qpos-kpos>=window. */
int coli_swa_masked(int qpos, int kpos, int window);

#ifdef __cplusplus
}
#endif
#endif
