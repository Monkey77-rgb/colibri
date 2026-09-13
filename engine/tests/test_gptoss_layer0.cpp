/* test_gptoss_layer0.cpp -- gpt-oss-120b-MXFP4.gguf, LAYER 0's DENSE attention
 * block only, on 4 fixed tokens, CPU. Does NOT touch any expert tensor (the
 * 108 MXFP4 tensors are ~60 GB and the disk-resident expert store is being
 * built by a different agent in parallel this session) and does NOT run more
 * than one layer. This is a bounded correctness probe, not a forward pass:
 * the oracle for it is tests/gptoss_layer0_ref.py, an independent
 * NumPy/PyTorch-free Python re-implementation reading the SAME tensor bytes
 * via its own from-scratch GGUF/Q8_0 parser -- not this program's output
 * copied elsewhere.
 *
 * What this program dequantizes (coli_gguf_load_f32, via loader.h -- the
 * SAME dequant path model.cpp's coli_load uses, proven bit-exact on
 * 473,956,352 live-fleet elements per loader.h's own header comment):
 *   blk.0.attn_norm.weight     [2880]        f32
 *   blk.0.attn_q.{weight,bias} [2880,4096],[4096]   Q8_0 / f32
 *   blk.0.attn_k.{weight,bias} [2880,512],[512]     Q8_0 / f32
 *   blk.0.attn_v.{weight,bias} [2880,512],[512]     Q8_0 / f32
 *   blk.0.attn_output.{weight,bias} [4096,2880],[2880] Q8_0 / f32
 *   blk.0.attn_sinks.weight    [64]          f32
 *   token_embd.weight          [2880,201088] Q8_0  (only 4 rows are USED,
 *                                            but coli_gguf_load_f32 dequants
 *                                            the whole tensor -- ~2.3 GB f32,
 *                                            measured to fit this host's
 *                                            23 GiB free comfortably; freed
 *                                            before exit)
 *
 * Math (RMSNorm -> Wq/Wk/Wv+bias -> YaRN RoPE (arch_ops.h) -> GQA attention
 * with an attn_sinks logit (arch_ops.h) -> Wo+bias), no residual add: the
 * oracle compares the ATTENTION BLOCK's own output (llama.cpp's "cur" in
 * openai-moe.cpp, before `ggml_add(cur, inpSA)`), one vector per token,
 * length 2880. Layer 0 is SWA (window=128, see coli_arch_is_swa_layer) but
 * with only 4 tokens the window never binds -- causal masking alone decides
 * every pair here, which this program's own printed mask table (stderr)
 * makes checkable rather than assumed. */
#include "../src/loader.h"
#include "../src/arch.h"
#include "../src/arch_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static float *must_load(coli_gguf *g, const char *name, int64_t want) {
    float *p; int64_t n = coli_gguf_load_f32(g, name, &p);
    if (n != want) { fprintf(stderr, "load '%s' failed: got %lld want %lld\n", name, (long long)n, (long long)want); exit(1); }
    return p;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss=0; for (int i=0;i<n;i++) ss += (double)x[i]*x[i];
    float scale = 1.0f/sqrtf((float)(ss/n) + eps);
    for (int i=0;i<n;i++) out[i] = x[i]*scale*w[i];
}

/* y[O] = x[I] . W[O][I]^T + b[O]. W stored row-major, O rows of I contiguous
 * floats (coli_gguf_load_f32's dequant order for a [I,O]-shaped GGUF tensor
 * -- ne[0]=I is the fast/contiguous dim, ne[1]=O is the row count, same
 * convention model.cpp's coli_w_i8 already assumes for every arch it
 * supports). */
static void matvec(float *y, const float *x, const float *W, const float *b, int I, int O) {
    for (int o=0;o<O;o++) {
        const float *row = W + (int64_t)o*I;
        double acc = b ? (double)b[o] : 0.0;
        for (int i=0;i<I;i++) acc += (double)x[i]*row[i];
        y[o] = (float)acc;
    }
}

int main(void) {
    const char *path = "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf";
    char err[256];
    coli_gguf *g = coli_gguf_open(path, err, sizeof err);
    if (!g) { fprintf(stderr, "open failed: %s\n", err); return 1; }

    coli_arch a;
    if (!coli_arch_from_gguf(g, &a, err, sizeof err)) { fprintf(stderr, "arch parse failed: %s\n", err); return 1; }
    fprintf(stderr, "arch: %s layers=%d hidden=%d heads=%d/%d hd=%d sinks=%d yarn=%d swa_window=%d\n",
            a.arch, a.n_layers, a.hidden, a.n_heads, a.n_kv_heads, a.head_dim, a.attn_has_sinks, a.yarn, a.swa_window);

    int D = a.hidden, H = a.n_heads, KVH = a.n_kv_heads, hd = a.head_dim;
    int qD = H*hd, kvD = KVH*hd, mult = H/KVH;

    float *attn_norm = must_load(g, "blk.0.attn_norm.weight", D);
    float *wq = must_load(g, "blk.0.attn_q.weight", (int64_t)D*qD);
    float *bq = must_load(g, "blk.0.attn_q.bias", qD);
    float *wk = must_load(g, "blk.0.attn_k.weight", (int64_t)D*kvD);
    float *bk = must_load(g, "blk.0.attn_k.bias", kvD);
    float *wv = must_load(g, "blk.0.attn_v.weight", (int64_t)D*kvD);
    float *bv = must_load(g, "blk.0.attn_v.bias", kvD);
    float *wo = must_load(g, "blk.0.attn_output.weight", (int64_t)qD*D);
    float *bo = must_load(g, "blk.0.attn_output.bias", D);
    float *sinks = must_load(g, "blk.0.attn_sinks.weight", H);
    fprintf(stderr, "dense layer-0 tensors loaded\n");

    float *embd; int64_t vocab_elems = coli_gguf_load_f32(g, "token_embd.weight", &embd);
    if (vocab_elems <= 0) { fprintf(stderr, "token_embd load failed\n"); return 1; }
    fprintf(stderr, "token_embd loaded: %lld elements (%.2f GiB)\n", (long long)vocab_elems, vocab_elems*4.0/(1<<30));

    /* 4 fixed token ids -- arbitrary but in-range and reproducible, not a
     * real prompt (this is a numerics probe, not a coherence test). */
    const int N = 4;
    int ids[4] = { 1000, 2000, 3000, 4000 };

    std::vector<std::vector<float>> x(N, std::vector<float>(D));
    for (int t=0;t<N;t++) memcpy(x[t].data(), embd + (int64_t)ids[t]*D, D*sizeof(float));

    std::vector<std::vector<float>> xn(N, std::vector<float>(D));
    for (int t=0;t<N;t++) rmsnorm(xn[t].data(), x[t].data(), attn_norm, D, a.eps);

    std::vector<std::vector<float>> Q(N, std::vector<float>(qD)), K(N, std::vector<float>(kvD)), V(N, std::vector<float>(kvD));
    for (int t=0;t<N;t++) {
        matvec(Q[t].data(), xn[t].data(), wq, bq, D, qD);
        matvec(K[t].data(), xn[t].data(), wk, bk, D, kvD);
        matvec(V[t].data(), xn[t].data(), wv, bv, D, kvD);
    }

    /* YaRN RoPE, half-split (neox) pairing -- same rotation model.cpp's
     * rope_head_tab() uses (see arch_ops.h's file comment), applied here
     * inline since rope_head_tab itself is file-static in model.cpp.
     *
     * COLI_L0_BREAK_ROPE: perturbation control for compare_layer0.py --
     * skips RoPE entirely so `make test-gptoss-layer0-control` can show the
     * comparison against gptoss_layer0_ref.py actually FAILS when the CPU
     * side is wrong, instead of just happening to agree. */
#ifndef COLI_L0_BREAK_ROPE
    for (int t=0;t<N;t++) {
        coli_rope_tab rt;
        coli_yarn_rope_table(&rt, t, hd, a.rope_theta, a.yarn?a.yarn_factor:1.f, a.yarn_beta_fast, a.yarn_beta_slow, a.yarn_orig_ctx);
        int half = rt.half;
        for (int h=0; h<H; h++) {
            float *v = Q[t].data() + h*hd;
            for (int i=0;i<half;i++) { float c=rt.c[i], s=rt.s[i]; float A=v[i], B=v[i+half]; v[i]=A*c-B*s; v[i+half]=A*s+B*c; }
        }
        for (int h=0; h<KVH; h++) {
            float *v = K[t].data() + h*hd;
            for (int i=0;i<half;i++) { float c=rt.c[i], s=rt.s[i]; float A=v[i], B=v[i+half]; v[i]=A*c-B*s; v[i+half]=A*s+B*c; }
        }
    }
#endif

    /* Attention with per-head sink logit, causal (+ SWA, inert at N=4). */
    fprintf(stderr, "causal/SWA mask (window=%d), q=row k=col, X=masked:\n", a.swa_window);
    for (int qp=0; qp<N; qp++) { fprintf(stderr, "  "); for (int kp=0; kp<N; kp++) fprintf(stderr, "%c", coli_swa_masked(qp,kp,coli_arch_is_swa_layer(&a,0)?a.swa_window:0)?'X':'.'); fprintf(stderr, "\n"); }

    std::vector<std::vector<float>> attn_out(N, std::vector<float>(qD, 0.f));
    float scale = 1.0f/sqrtf((float)hd);
    int swa_window = coli_arch_is_swa_layer(&a,0) ? a.swa_window : 0;
    for (int h=0; h<H; h++) {
        int kvh = h / mult;
        for (int qp=0; qp<N; qp++) {
            float scores[4];
            for (int kp=0; kp<N; kp++) {
                if (coli_swa_masked(qp,kp,swa_window)) { scores[kp] = -1e30f; continue; }
                const float *qv = Q[qp].data()+h*hd, *kv = K[kp].data()+kvh*hd;
                double dot=0; for (int i=0;i<hd;i++) dot += (double)qv[i]*kv[i];
                scores[kp] = (float)(dot*scale);
            }
            /* masked entries (-1e30) survive coli_softmax_with_sink's max-sub
             * unharmed (exp(-huge)=0), same convention softmax always uses. */
            coli_softmax_with_sink(scores, N, sinks[h]);
            float *out = attn_out[qp].data() + h*hd;
            for (int kp=0; kp<N; kp++) {
                const float *vv = V[kp].data()+kvh*hd;
                float w = scores[kp];
                for (int i=0;i<hd;i++) out[i] += w*vv[i];
            }
        }
    }

    std::vector<std::vector<float>> cur(N, std::vector<float>(D));
    for (int t=0;t<N;t++) matvec(cur[t].data(), attn_out[t].data(), wo, bo, qD, D);

    FILE *outf = fopen("/tmp/coli_gptoss_layer0_cpp.txt", "w");
    for (int t=0;t<N;t++) {
        fprintf(outf, "token %d id=%d\n", t, ids[t]);
        for (int i=0;i<D;i++) fprintf(outf, "%.9g\n", cur[t][i]);
    }
    fclose(outf);
    fprintf(stderr, "wrote /tmp/coli_gptoss_layer0_cpp.txt (%d tokens x %d dims)\n", N, D);

    coli_gguf_free_f32(embd);
    coli_gguf_free_f32(attn_norm); coli_gguf_free_f32(wq); coli_gguf_free_f32(bq);
    coli_gguf_free_f32(wk); coli_gguf_free_f32(bk); coli_gguf_free_f32(wv); coli_gguf_free_f32(bv);
    coli_gguf_free_f32(wo); coli_gguf_free_f32(bo); coli_gguf_free_f32(sinks);
    coli_gguf_close(g);
    return 0;
}
