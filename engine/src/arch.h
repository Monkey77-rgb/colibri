/* arch.h -- architecture descriptor read from GGUF metadata, additive to the
 * qwen2/llama/qwen3/qwen3moe path that already exists in model.cpp.
 *
 * WHY A SEPARATE FILE INSTEAD OF EXTENDING coli_cfg IN model.h.
 * coli_load()'s hardwired refusal list (model.cpp ~line 906) and its
 * arithmetic are the thing 96.6%-vs-llama.cpp and NLL 2.8582 were measured
 * against. Adding a struct alongside it that ONLY READS METADATA -- never
 * touches coli_load, coli_cfg or the forward pass -- means the existing path
 * is provably unaffected by construction, not by care. coli_arch_from_gguf()
 * opens its OWN coli_gguf handle (header/KV/tensor-index only, see loader.h --
 * no weight bytes read) and coli_load() is never called by it.
 *
 * SCOPE. This descriptor is populated for the five architectures this engine
 * has GGUF metadata evidence for: qwen2, llama, qwen3, qwen3moe (unchanged
 * behaviour, verified against the values coli_load already derives) and
 * gpt-oss (new fields, verified against a real gpt-oss-120b-MXFP4.gguf header
 * and against llama.cpp's src/models/openai-moe.cpp, fetched 2026-09-13 --
 * see engine/tests/test_arch.cpp for the citations next to each assertion).
 * Any other `general.architecture` string is refused, same discipline as
 * coli_load: a wrong arch produces fluent nonsense, not a crash, so guessing
 * is worse than refusing.
 */
#ifndef COLI_ARCH_H
#define COLI_ARCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct coli_gguf coli_gguf;   /* opaque, from loader.h */

/* How the router's top-k logits become per-expert weights.
 *
 * TOPK_SOFTMAX  -- qwen2moe/qwen3moe/mixtral family, per model.cpp's existing
 *                  moe_ffn(): top-k SELECTED by raw logit value, THEN softmax
 *                  over just those k values. No bias added first.
 * TOPK_SOFTMAX_BIASED -- gpt-oss (llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_
 *                  SOFTMAX_WEIGHT, src/llama-graph.cpp build_moe_ffn(),
 *                  fetched 2026-09-13): logits = router_matmul + gate_inp_b
 *                  (the router carries a bias, qwen's does not); top-k select
 *                  on those BIASED logits; softmax over the k selected biased
 *                  values; norm_w=false and w_scale=0 so nothing further is
 *                  applied. Read build_moe_ffn's switch on
 *                  LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT (weights =
 *                  gathered logits, "probs = logits" i.e. NO softmax before
 *                  top-k) -- the only arithmetic difference from
 *                  TOPK_SOFTMAX is the bias add before selection. Once that
 *                  bias is folded in, TOPK_SOFTMAX's existing top-k-then-
 *                  softmax-over-selected code computes the identical result;
 *                  this enum value exists so that fact is asserted by a test
 *                  rather than assumed. */
typedef enum { COLI_GATE_TOPK_SOFTMAX = 0, COLI_GATE_TOPK_SOFTMAX_BIASED = 1 } coli_gate_func;

typedef enum { COLI_ACT_SILU = 0, COLI_ACT_SWIGLU_OAI = 1 } coli_activation;

typedef struct {
    char arch[32];
    int  n_layers, hidden, n_heads, n_kv_heads, head_dim, ffn_len, vocab, ctx_train;
    float eps;

    /* MoE. n_expert == 0 means dense (llama/qwen2). */
    int n_expert, n_expert_used, expert_ffn_len;
    coli_gate_func gate_func;
    int router_has_bias;   /* blk.0.ffn_gate_inp.bias present (gpt-oss: yes, qwen3moe: no) */
    int expert_has_bias;   /* blk.0.ffn_{gate,up,down}_exps.bias present */

    /* Attention extras. */
    int attn_qkv_bias;     /* blk.0.attn_{q,k,v}.bias present */
    int attn_o_bias;       /* blk.0.attn_output.bias present (gpt-oss only, of the
                             * archs handled here) */
    int attn_has_sinks;    /* blk.0.attn_sinks.weight present (gpt-oss) --
                             * length n_heads, one extra softmax logit per head,
                             * added before the row max/exp/sum and DROPPED
                             * before the value weighted-sum (see arch_ops.h) */
    int two_norms_per_layer; /* gpt-oss: attn_norm (pre-attn) AND
                               * post_attention_norm (pre-ffn) are BOTH RMSNorm,
                               * same shape/role as qwen's attn_norm/ffn_norm --
                               * named differently in the GGUF, not a different
                               * op. Kept as a flag rather than silently aliasing
                               * so a load path can assert the tensor it expects
                               * actually exists under gpt-oss's tensor names. */

    /* Sliding-window attention. window==0 means none. Layer il is SWA iff
     * coli_arch_is_swa_layer() below, itself derived from swa_period and
     * swa_dense_first exactly as llama.cpp's llama_hparams::set_swa_pattern()
     * (fetched 2026-09-13, src/llama-hparams.cpp) computes is_swa_impl[il]. */
    int swa_window;
    int swa_period;
    int swa_dense_first;

    /* RoPE. yarn==0: plain theta, coli_pow(theta,-2i/hd)-based table, i.e.
     * EXACTLY what model.cpp's rope_table()/rope_head_tab() already compute
     * (same neox half-split pairing -- see arch.h's file comment for why that
     * needed no new rotation code). yarn==1: scale that table's frequencies
     * and its cos/sin outputs per coli_yarn_rope_table() in arch_ops.h. */
    float rope_theta;
    int   yarn;
    float yarn_factor, yarn_beta_fast, yarn_beta_slow;
    int   yarn_orig_ctx;

    /* FFN activation. SILU: existing dense/MoE path, unchanged. SWIGLU_OAI:
     * gpt-oss, see arch_ops.h's coli_swiglu_oai(). */
    coli_activation activation;
    float swiglu_alpha, swiglu_limit;
} coli_arch;

/* Reads `path`'s GGUF header (metadata + tensor index only -- see loader.h's
 * coli_gguf_open, which does not read tensor data) and fills `out`. Returns 1
 * on success, 0 with a reason in err on failure (unreadable file, missing
 * required key, or an architecture string this file has no evidence for). */
int coli_arch_from_path(const char *path, coli_arch *out, char *err, size_t errcap);

/* Same, from an already-open coli_gguf (so a caller that also needs
 * coli_gguf_load_f32 on the same handle doesn't open the file twice). Does
 * NOT close `g`. */
int coli_arch_from_gguf(coli_gguf *g, coli_arch *out, char *err, size_t errcap);

/* llama_hparams::set_swa_pattern(), transcribed (src/llama-hparams.cpp,
 * fetched 2026-09-13): with dense_first=false (gpt-oss's default -- openai-moe.cpp
 * never sets it true), layer il is SWA iff (il % period) < (period - 1). For
 * gpt-oss period=2 that is il%2==0, i.e. EVEN layers -- matches the verified
 * fact this task was started with ("SWA on even layers") and the real
 * gpt-oss-120b-MXFP4.gguf header (attention.sliding_window=128, no explicit
 * pattern key => period defaults to 2 per openai-moe.cpp). period==0 means
 * "every layer is SWA" (not used by gpt-oss; kept for fidelity to the source). */
int coli_arch_is_swa_layer(const coli_arch *a, int il);

#ifdef __cplusplus
}
#endif
#endif
