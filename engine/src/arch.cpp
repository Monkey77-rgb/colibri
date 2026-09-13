/* arch.cpp -- see arch.h. Metadata-only: opens its own coli_gguf handle via
 * loader.h and never calls coli_load(), so it cannot change the behaviour of
 * the qwen2/llama/qwen3/qwen3moe path model.cpp already implements. */
#include "arch.h"
#include "loader.h"
#include <cstdio>
#include <cstring>
#include <cstdarg>

static void set_err(char *err, size_t errcap, const char *fmt, ...) {
    if (!err || errcap == 0) return;
    va_list ap; va_start(ap, fmt);
    vsnprintf(err, errcap, fmt, ap);
    va_end(ap);
}

/* "<arch>.<suffix>" key builder -- same reason model.cpp's arch_i64/arch_f32
 * lambdas exist: hardcoding "llama." silently fails on every other arch. */
static int akey_i64(coli_gguf *g, const char *arch, const char *sfx, long long *o) {
    char k[256]; snprintf(k, sizeof k, "%s.%s", arch, sfx);
    return coli_gguf_i64(g, k, o);
}
static int akey_f32(coli_gguf *g, const char *arch, const char *sfx, float *o) {
    char k[256]; snprintf(k, sizeof k, "%s.%s", arch, sfx);
    return coli_gguf_f32(g, k, o);
}
int coli_arch_is_swa_layer(const coli_arch *a, int il) {
    if (a->swa_window <= 0) return 0;
    int p = a->swa_period;
    if (p == 0) return 1;
    if (a->swa_dense_first) return (il % p) != 0;
    return (il % p) < (p - 1);
}

int coli_arch_from_gguf(coli_gguf *g, coli_arch *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    if (!coli_gguf_str(g, "general.architecture", out->arch, sizeof out->arch)) {
        set_err(err, errcap, "no general.architecture key"); return 0;
    }
    const char *a = out->arch;
    int is_qwenish = !strcmp(a,"qwen2") || !strcmp(a,"llama") || !strcmp(a,"qwen3") || !strcmp(a,"qwen3moe");
    int is_gptoss  = !strcmp(a,"gpt-oss");
    if (!is_qwenish && !is_gptoss) {
        set_err(err, errcap,
            "architecture '%s' has no coli_arch evidence; refusing rather than "
            "guessing (a wrong arch produces fluent nonsense)", a);
        return 0;
    }

    long long v; float f;
    #define REQI(sfx,dst) do{ if(!akey_i64(g,a,sfx,&v)){ set_err(err,errcap,"missing %s.%s",a,sfx); return 0;} dst=(int)v; }while(0)
    REQI("block_count", out->n_layers);
    REQI("embedding_length", out->hidden);
    REQI("attention.head_count", out->n_heads);
    if (akey_i64(g,a,"attention.head_count_kv",&v)) out->n_kv_heads=(int)v; else out->n_kv_heads=out->n_heads;
    if (akey_i64(g,a,"context_length",&v)) out->ctx_train=(int)v;
    if (akey_i64(g,a,"feed_forward_length",&v)) out->ffn_len=(int)v;
    out->head_dim = akey_i64(g,a,"attention.key_length",&v) ? (int)v : out->hidden/out->n_heads;
    out->eps = akey_f32(g,a,"attention.layer_norm_rms_epsilon",&f) ? f : 1e-5f;
    out->rope_theta = akey_f32(g,a,"rope.freq_base",&f) ? f : 10000.f;

    if (akey_i64(g,a,"expert_count",&v)) out->n_expert=(int)v;
    if (akey_i64(g,a,"expert_used_count",&v)) out->n_expert_used=(int)v;
    if (akey_i64(g,a,"expert_feed_forward_length",&v)) out->expert_ffn_len=(int)v;

    char rk[64];
    snprintf(rk,sizeof rk,"%s.rope.scaling.type",a);
    char stype[32] = {0};
    if (coli_gguf_str(g, rk, stype, sizeof stype) && !strcmp(stype,"yarn")) {
        out->yarn = 1;
        out->yarn_factor     = akey_f32(g,a,"rope.scaling.factor",&f) ? f : 1.f;
        out->yarn_beta_fast  = akey_f32(g,a,"rope.scaling.yarn_beta_fast",&f) ? f : 32.f;
        out->yarn_beta_slow  = akey_f32(g,a,"rope.scaling.yarn_beta_slow",&f) ? f : 1.f;
        out->yarn_orig_ctx   = akey_i64(g,a,"rope.scaling.original_context_length",&v) ? (int)v : 4096;
    }

    out->attn_qkv_bias = coli_gguf_has(g,"blk.0.attn_q.bias");
    out->attn_o_bias   = coli_gguf_has(g,"blk.0.attn_output.bias");
    out->attn_has_sinks = coli_gguf_has(g,"blk.0.attn_sinks.weight");
    out->router_has_bias = coli_gguf_has(g,"blk.0.ffn_gate_inp.bias");
    out->expert_has_bias = coli_gguf_has(g,"blk.0.ffn_down_exps.bias");
    out->two_norms_per_layer = coli_gguf_has(g,"blk.0.post_attention_norm.weight");

    if (is_gptoss) {
        out->gate_func = COLI_GATE_TOPK_SOFTMAX_BIASED;
        out->activation = COLI_ACT_SWIGLU_OAI;
        out->swiglu_alpha = 1.702f;
        out->swiglu_limit = 7.0f;
        int window = 0;
        if (akey_i64(g,a,"attention.sliding_window",&v)) window=(int)v;
        out->swa_window = window;
        long long period = 2; /* openai-moe.cpp's default when the pattern key is absent */
        char pk[80]; snprintf(pk,sizeof pk,"%s.attention.sliding_window_pattern",a);
        if (coli_gguf_i64(g, pk, &period)) { /* explicit override, not present in the file we measured */ }
        out->swa_period = (int)period;
        out->swa_dense_first = 0;
        if (!out->attn_has_sinks) { set_err(err,errcap,"gpt-oss file missing blk.0.attn_sinks.weight"); return 0; }
        if (!out->two_norms_per_layer) { set_err(err,errcap,"gpt-oss file missing blk.0.post_attention_norm.weight"); return 0; }
    } else {
        out->gate_func = COLI_GATE_TOPK_SOFTMAX;
        out->activation = COLI_ACT_SILU;
        out->swa_window = 0;
        out->swa_period = 0;
    }
    return 1;
}

int coli_arch_from_path(const char *path, coli_arch *out, char *err, size_t errcap) {
    coli_gguf *g = coli_gguf_open(path, err, errcap);
    if (!g) return 0;
    int ok = coli_arch_from_gguf(g, out, err, errcap);
    coli_gguf_close(g);
    return ok;
}
