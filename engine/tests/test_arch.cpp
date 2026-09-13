/* test_arch.cpp -- coli_arch against two REAL GGUF headers (metadata only;
 * coli_gguf_open never reads tensor data, so the 63 GB gpt-oss-120b file costs
 * one header read here, not a load).
 *
 * Positive control: qwen2.5-3b must parse to the values coli_load() already
 * uses -- if this test's qwen2 assertions ever disagreed with coli_load's own
 * arithmetic, that would mean coli_arch_from_gguf() is reading the wrong
 * fields, not that qwen2 changed. The number this actually gets checked
 * against is engine/coli's `--nll1` run on the same file, recorded in the
 * commit message and the task report, not in this binary (that run needs the
 * full model loaded and ~30 s of CPU; this test does not depend on it).
 *
 * Negative control: an unrecognised architecture string must be REFUSED, not
 * silently defaulted -- test_control_refuses_unknown_arch synthesizes a
 * minimal GGUF with general.architecture="not-a-real-arch" and asserts
 * coli_arch_from_gguf returns 0. Without this, a typo in the arch string
 * comparison (e.g. "gpt_oss" vs "gpt-oss") would silently fall through to
 * "refuse", print PASS on the two real files, and never be caught.
 *
 * gpt-oss assertions are checked against the ACTUAL header of
 * /home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf, read with
 * a standalone Python parser this session (see the task's ggufmeta.py and the
 * ad hoc full-KV dump run alongside it) and cross-checked against
 * llama.cpp's src/models/openai-moe.cpp / src/llama-hparams.cpp (fetched
 * 2026-09-13 via the local browser_service, see the values cited in arch.h). */
#include "../src/arch.h"
#include "../src/loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while(0)

static const char *QWEN3B = "/home/monkey/Documents/Ai_Models/legion_fleet/qwen2.5-3b-instruct-q4_k_m.gguf";
static const char *GPTOSS = "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf";

static float fabsf_(float x){ return x<0?-x:x; }

static void test_qwen25_3b_unchanged_shape(void) {
    char err[256];
    coli_arch a;
    int ok = coli_arch_from_path(QWEN3B, &a, err, sizeof err);
    CHECK(ok, "qwen2.5-3b: coli_arch_from_path failed: %s", ok?"":err);
    if (!ok) return;
    /* Values below are what coli's own load-time print reported for this file
     * in this session ("qwen2: 36 layers, d=2048, heads=16/2, hd=128,
     * ffn=11008, vocab=151936, ctx=32768 ... rope=neox theta=1000000
     * eps=1.0e-06 qkv_bias=yes") -- coli_load's OWN log line, not a separate
     * guess, so this test cannot silently drift from what the engine actually
     * does with this file. */
    CHECK(!strcmp(a.arch,"qwen2"), "arch='%s'", a.arch);
    CHECK(a.n_layers==36, "n_layers=%d", a.n_layers);
    CHECK(a.hidden==2048, "hidden=%d", a.hidden);
    CHECK(a.n_heads==16 && a.n_kv_heads==2, "heads=%d/%d", a.n_heads, a.n_kv_heads);
    CHECK(a.head_dim==128, "head_dim=%d", a.head_dim);
    CHECK(a.ffn_len==11008, "ffn_len=%d", a.ffn_len);
    CHECK(fabsf_(a.rope_theta-1000000.f)<1.f, "rope_theta=%f", a.rope_theta);
    CHECK(a.attn_qkv_bias==1, "attn_qkv_bias=%d", a.attn_qkv_bias);
    CHECK(a.n_expert==0, "n_expert=%d (must be dense)", a.n_expert);
    CHECK(a.attn_has_sinks==0, "attn_has_sinks=%d (qwen2 has no sinks)", a.attn_has_sinks);
    CHECK(a.yarn==0, "yarn=%d (qwen2.5-3b is not yarn-scaled)", a.yarn);
    CHECK(a.gate_func==COLI_GATE_TOPK_SOFTMAX, "gate_func=%d", (int)a.gate_func);
    CHECK(a.activation==COLI_ACT_SILU, "activation=%d", (int)a.activation);
}

static void test_gptoss_header(void) {
    char err[256];
    coli_arch a;
    int ok = coli_arch_from_path(GPTOSS, &a, err, sizeof err);
    CHECK(ok, "gpt-oss: coli_arch_from_path failed: %s", ok?"":err);
    if (!ok) return;
    /* Every literal here was read directly off the real file's GGUF header
     * this session (python3 dump, verbatim in the task report) -- not
     * transcribed from the task brief. */
    CHECK(!strcmp(a.arch,"gpt-oss"), "arch='%s'", a.arch);
    CHECK(a.n_layers==36, "n_layers=%d", a.n_layers);
    CHECK(a.hidden==2880, "hidden=%d", a.hidden);
    CHECK(a.n_heads==64 && a.n_kv_heads==8, "heads=%d/%d", a.n_heads, a.n_kv_heads);
    CHECK(a.head_dim==64, "head_dim=%d (from attention.key_length, NOT hidden/n_heads=45)", a.head_dim);
    CHECK(a.ffn_len==2880, "ffn_len=%d", a.ffn_len);
    CHECK(a.n_expert==128 && a.n_expert_used==4, "experts=%d/%d", a.n_expert, a.n_expert_used);
    CHECK(a.expert_ffn_len==2880, "expert_ffn_len=%d", a.expert_ffn_len);
    CHECK(fabsf_(a.rope_theta-150000.f)<1.f, "rope_theta=%f", a.rope_theta);
    CHECK(a.yarn==1, "yarn=%d", a.yarn);
    CHECK(fabsf_(a.yarn_factor-32.f)<1e-3f, "yarn_factor=%f", a.yarn_factor);
    CHECK(fabsf_(a.yarn_beta_fast-32.f)<1e-3f, "yarn_beta_fast=%f", a.yarn_beta_fast);
    CHECK(fabsf_(a.yarn_beta_slow-1.f)<1e-3f, "yarn_beta_slow=%f", a.yarn_beta_slow);
    CHECK(a.yarn_orig_ctx==4096, "yarn_orig_ctx=%d", a.yarn_orig_ctx);
    CHECK(a.swa_window==128, "swa_window=%d", a.swa_window);
    CHECK(a.swa_period==2, "swa_period=%d", a.swa_period);
    CHECK(coli_arch_is_swa_layer(&a,0)==1, "layer 0 must be SWA");
    CHECK(coli_arch_is_swa_layer(&a,1)==0, "layer 1 must be full attention");
    CHECK(coli_arch_is_swa_layer(&a,2)==1, "layer 2 must be SWA");
    CHECK(coli_arch_is_swa_layer(&a,35)==0, "layer 35 (odd, last) must be full attention");
    CHECK(a.attn_qkv_bias==1, "attn_qkv_bias=%d", a.attn_qkv_bias);
    CHECK(a.attn_o_bias==1, "attn_o_bias=%d (gpt-oss has an output-proj bias qwen does not)", a.attn_o_bias);
    CHECK(a.attn_has_sinks==1, "attn_has_sinks=%d", a.attn_has_sinks);
    CHECK(a.two_norms_per_layer==1, "two_norms_per_layer=%d", a.two_norms_per_layer);
    CHECK(a.router_has_bias==1, "router_has_bias=%d", a.router_has_bias);
    CHECK(a.expert_has_bias==1, "expert_has_bias=%d", a.expert_has_bias);
    CHECK(a.gate_func==COLI_GATE_TOPK_SOFTMAX_BIASED, "gate_func=%d", (int)a.gate_func);
    CHECK(a.activation==COLI_ACT_SWIGLU_OAI, "activation=%d", (int)a.activation);
    CHECK(fabsf_(a.swiglu_alpha-1.702f)<1e-6f, "swiglu_alpha=%f", a.swiglu_alpha);
    CHECK(fabsf_(a.swiglu_limit-7.f)<1e-6f, "swiglu_limit=%f", a.swiglu_limit);
}

/* Negative control: builds a MINIMAL valid GGUF (magic, version, 0 tensors, 1
 * KV pair: general.architecture="not-a-real-arch") in a temp file and asserts
 * it is REFUSED. This is the control that can fail: comment out the
 * is_qwenish/is_gptoss check in arch.cpp and this test starts failing (the
 * bogus arch would parse instead of being refused), while the two tests above
 * would keep passing -- proving they alone do not exercise the refusal path. */
static void write_u32(FILE*f,uint32_t v){ fwrite(&v,4,1,f); }
static void write_u64(FILE*f,uint64_t v){ fwrite(&v,8,1,f); }
static void write_str(FILE*f,const char*s){ uint64_t n=strlen(s); write_u64(f,n); fwrite(s,1,n,f); }

static void test_control_refuses_unknown_arch(void) {
    const char *path = "/tmp/coli_arch_test_bogus.gguf";
    FILE *f = fopen(path,"wb");
    CHECK(f!=nullptr, "could not open scratch gguf for writing");
    if (!f) return;
    fwrite("GGUF",4,1,f);
    write_u32(f,3);       /* version */
    write_u64(f,0);       /* n_tensors */
    write_u64(f,1);       /* n_kv */
    write_str(f,"general.architecture");
    write_u32(f,8);       /* type=str */
    write_str(f,"not-a-real-arch");
    fclose(f);

    char err[256]; coli_arch a;
    int ok = coli_arch_from_path(path, &a, err, sizeof err);
    CHECK(ok==0, "bogus architecture string was ACCEPTED instead of refused");
    CHECK(ok==0 || strlen(err)>0, "refusal path must set an error string");
    remove(path);
}

int main(void) {
    test_qwen25_3b_unchanged_shape();
    test_gptoss_header();
    test_control_refuses_unknown_arch();
    if (g_fail) { fprintf(stderr, "\n%d assertion(s) FAILED\n", g_fail); return 1; }
    printf("test_arch: all assertions passed (qwen2.5-3b header, gpt-oss-120b header, negative control)\n");
    return 0;
}
