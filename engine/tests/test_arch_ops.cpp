/* test_arch_ops.cpp -- each coli_arch_ops function against an INDEPENDENT
 * oracle (a different algorithm/code path, not a copy-paste of the
 * implementation), plus at least one control that can fail. Per the
 * project's own discipline (see gemm_mxfp4.h's ref/parallel split and every
 * *_broken test target in this Makefile): a test that cannot produce a
 * negative result proves nothing. */
#include "../src/arch_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while(0)
#define NEAR(a,b,tol) (fabs((double)(a)-(double)(b)) <= (tol))

/* ---------------------------------------------------------- YaRN RoPE --- */
/* factor<=1 must reproduce plain (unscaled) RoPE exactly: inv_freq=1/freq,
 * concentration=1. Oracle computed directly from the definition, not by
 * calling anything in arch_ops.cpp. */
static void test_yarn_factor1_matches_plain_rope(void) {
    int hd = 64; float base = 1000000.f;
    coli_rope_tab t;
    int pos = 37;
    coli_yarn_rope_table(&t, pos, hd, base, /*factor=*/1.0f, 32.f, 1.f, 4096);
    CHECK(t.half == hd/2, "half=%d", t.half);
    for (int i=0;i<t.half;i++) {
        double freq = pow((double)base, (2.0*i)/(double)hd);
        double ang = pos / freq;
        double want_c = cos(ang), want_s = sin(ang);
        CHECK(NEAR(t.c[i], want_c, 1e-5), "i=%d c=%.8f want=%.8f", i, t.c[i], want_c);
        CHECK(NEAR(t.s[i], want_s, 1e-5), "i=%d s=%.8f want=%.8f", i, t.s[i], want_s);
    }
}

/* Real gpt-oss params (theta=150000, factor=32, beta_fast=32, beta_slow=1,
 * orig_ctx=4096, hd=64 -- verified against the actual GGUF header in step 1).
 * Checks the NTK-by-parts BOUNDARY BEHAVIOUR, an invariant of the formula
 * rather than a re-derivation of it: below `low` the ramp must clamp to pure
 * extrapolation (inv_freq == 1/freq exactly, concentration aside), above
 * `high` it must clamp to pure interpolation (inv_freq == 1/(factor*freq)).
 * low/high computed independently here from gptoss_model.py's own formula
 * (d_half*ln(orig_ctx/(beta*2*pi))/ln(base)) to know WHICH indices to check --
 * i=0 is provably below low (low>0 whenever orig_ctx/(beta_fast*2*pi) < 1
 * is false here; low~=8.10 computed below) and i=31 provably above high
 * (~=17.40). This is a property test, not a value transcription: getting
 * low/high swapped in the implementation (the natural bug -- interpolation
 * and extrapolation reversed) flips which end is which and this catches it. */
static void test_yarn_gptoss_ntk_boundary(void) {
    float base=150000.f, factor=32.f, beta_fast=32.f, beta_slow=1.f; int orig_ctx=4096, hd=64;
    double d_half = hd/2.0, lnbase = log((double)base);
    double low  = d_half * log((double)orig_ctx/((double)beta_fast*2.0*M_PI)) / lnbase;
    double high = d_half * log((double)orig_ctx/((double)beta_slow*2.0*M_PI)) / lnbase;
    CHECK(low > 0 && low < high && high < d_half, "low=%.3f high=%.3f d_half=%.1f (expected 0<low<high<d_half)", low, high, d_half);

    coli_rope_tab t; int pos = 100;
    coli_yarn_rope_table(&t, pos, hd, base, factor, beta_fast, beta_slow, orig_ctx);
    double concentration = 0.1*log((double)factor) + 1.0;

    int i_below = (int)floor(low);        /* provably ramp<0 -> pure extrapolation */
    if ((double)i_below >= low) i_below--;
    CHECK(i_below >= 0, "test setup: low=%.3f too small for a below-low index", low);
    double freq_b = pow((double)base, (2.0*i_below)/(double)hd);
    double want_c = cos(pos/freq_b)*concentration, want_s = sin(pos/freq_b)*concentration;
    CHECK(NEAR(t.c[i_below], want_c, 1e-4), "below-low i=%d c=%.6f want=%.6f (extrapolation)", i_below, t.c[i_below], want_c);
    CHECK(NEAR(t.s[i_below], want_s, 1e-4), "below-low i=%d s=%.6f want=%.6f (extrapolation)", i_below, t.s[i_below], want_s);

    int i_above = (int)ceil(high);         /* provably ramp>=1 -> pure interpolation */
    if ((double)i_above <= high) i_above++;
    CHECK(i_above < hd/2, "test setup: high=%.3f too large for an above-high index in [0,%d)", high, hd/2);
    double freq_a = pow((double)base, (2.0*i_above)/(double)hd);
    double want_c2 = cos(pos/(factor*freq_a))*concentration, want_s2 = sin(pos/(factor*freq_a))*concentration;
    CHECK(NEAR(t.c[i_above], want_c2, 1e-4), "above-high i=%d c=%.6f want=%.6f (interpolation)", i_above, t.c[i_above], want_c2);
    CHECK(NEAR(t.s[i_above], want_s2, 1e-4), "above-high i=%d s=%.6f want=%.6f (interpolation)", i_above, t.s[i_above], want_s2);
}

/* PERTURBATION CONTROL: a version with low/high swapped -- the natural bug --
 * must FAIL the boundary test above. Reimplemented inline (not by flipping a
 * macro in the shipped source) so the control cannot accidentally ship live. */
static void test_control_yarn_boundary_can_fail(void) {
    float base=150000.f, factor=32.f, beta_fast=32.f, beta_slow=1.f; int orig_ctx=4096, hd=64, pos=100;
    int half = hd/2;
    double d_half = hd/2.0, lnbase = log((double)base);
    double low  = d_half * log((double)orig_ctx/((double)beta_fast*2.0*M_PI)) / lnbase;
    double high = d_half * log((double)orig_ctx/((double)beta_slow*2.0*M_PI)) / lnbase;
    double swap_low = high, swap_high = low; /* the bug */
    double concentration = 0.1*log((double)factor)+1.0;
    std::vector<double> broken_c(half);
    for (int i=0;i<half;i++) {
        double freq = pow((double)base,(2.0*i)/(double)hd);
        double interp = 1.0/((double)factor*freq), extrap = 1.0/freq;
        double ramp = (swap_high!=swap_low) ? (i-swap_low)/(swap_high-swap_low) : 0.0;
        if (ramp<0) ramp=0;
        if (ramp>1) ramp=1;
        double mask = 1.0-ramp;
        double inv_freq = interp*(1.0-mask)+extrap*mask;
        broken_c[i] = cos(pos*inv_freq)*concentration;
    }
    coli_rope_tab t;
    coli_yarn_rope_table(&t, pos, hd, base, factor, beta_fast, beta_slow, orig_ctx);
    int i_below = (int)floor(low); if ((double)i_below>=low) i_below--;
    int mismatches=0;
    for (int i=0;i<half;i++) if (!NEAR(t.c[i], broken_c[i], 1e-9)) mismatches++;
    CHECK(mismatches > 0, "CONTROL DID NOT FAIL: swapped-low/high produced identical output to the real implementation -- the boundary test above cannot distinguish correct from broken and proves nothing");
}

/* ------------------------------------------------------------- swiglu --- */
/* Oracle: sigmoid via the tanh identity sigmoid(x)=0.5*(1+tanh(x/2)) -- a
 * genuinely different libm call path than arch_ops.cpp's 1/(1+exp(-x)),
 * so a sign/formula bug in either is unlikely to agree by coincidence. */
static double swiglu_oracle(double gate, double up, double alpha, double limit) {
    double g = std::min(gate, limit);
    double u = std::max(-limit, std::min(up, limit));
    double glu = g * (0.5*(1.0+tanh(alpha*g/2.0)));
    return glu * (u + 1.0);
}
static void test_swiglu_oai(void) {
    float alpha=1.702f, limit=7.0f;
    float gate[7] = {0.f, 1.f, -1.f, 7.f, 100.f, -100.f, 3.3f};
    float up[7]   = {0.f, -1.f, 1.f, 7.f, -100.f, 100.f, -8.f};
    float out[7];
    coli_swiglu_oai(out, gate, up, 7, alpha, limit);
    for (int i=0;i<7;i++) {
        double want = swiglu_oracle(gate[i], up[i], alpha, limit);
        CHECK(NEAR(out[i], want, 1e-4), "i=%d gate=%.3f up=%.3f got=%.6f want=%.6f", i, gate[i], up[i], out[i], want);
    }
    /* gate has NO lower clamp (reference: x_glu.clamp(min=None, max=limit)) --
     * a very negative gate must NOT be clamped, only sigmoid-suppressed. */
    CHECK(NEAR(out[5]/1.0, swiglu_oracle(-100.f,100.f,alpha,limit), 1e-3), "gate must be unclamped below zero");
}
/* PERTURBATION CONTROL: omitting gate's upper clamp entirely (the natural
 * bug -- "forgot the clamp on the gate path") must change the output for a
 * large positive gate, proving the oracle comparison is actually sensitive
 * to it. (A double-lower-clamp on very-negative gate was tried first and
 * measured NOT to separate -- both saturate sigmoid to ~0 in that regime, so
 * that would have been a control that could not fail; large-positive gate,
 * where sigmoid saturates to ~1 instead and glu~=gate, is where a missing
 * clamp is actually visible.) */
static void test_control_swiglu_can_fail(void) {
    float alpha=1.702f, limit=7.0f;
    float gate=100.f, up=0.f, out_correct[1];
    coli_swiglu_oai(out_correct, &gate, &up, 1, alpha, limit);
    /* the bug: no upper clamp on gate at all */
    double g_wrong = gate;
    double glu_wrong = g_wrong * (1.0/(1.0+exp(-alpha*g_wrong)));
    double out_wrong = glu_wrong * (up + 1.0);
    CHECK(fabs((double)out_correct[0]-out_wrong) > 1.0, "CONTROL DID NOT FAIL: an unclamped gate path produced ~the same output as the real (clamped) path for gate=100 -- the swiglu test cannot see a missing clamp (correct=%.6f wrong=%.6f)", out_correct[0], out_wrong);
}

/* --------------------------------------------------------- MoE gating --- */
/* Oracle: full sort of (logit+bias) descending, take first k -- a completely
 * different algorithm than the incremental "scan for remaining max" the
 * implementation uses. */
static void moe_gate_oracle(const float *logits, const float *bias, int n, int k,
                             std::vector<int> &sel, std::vector<float> &wgt) {
    std::vector<std::pair<float,int>> v(n);
    for (int e=0;e<n;e++) v[e] = { logits[e] + (bias?bias[e]:0.f), e };
    std::stable_sort(v.begin(), v.end(), [](auto&a,auto&b){ return a.first > b.first; });
    sel.assign(k,0); wgt.assign(k,0.f);
    double mx=-1e300; for (int j=0;j<k;j++) { sel[j]=v[j].second; wgt[j]=v[j].first; if (wgt[j]>mx) mx=wgt[j]; }
    double sum=0; for (int j=0;j<k;j++) { wgt[j]=(float)exp((double)wgt[j]-mx); sum+=wgt[j]; }
    for (int j=0;j<k;j++) wgt[j]/=(float)sum;
}
static void test_moe_gate_topk_softmax(void) {
    /* Chosen so the bias FLIPS which experts are in the top-4 vs raw logits
     * alone -- proves the bias is actually applied before selection, not
     * just added to already-selected weights. */
    int n=8, k=4;
    float logits[8] = { 5.0f, 4.9f, 4.8f, 4.7f, 1.0f, 0.9f, 0.8f, 0.7f };
    float bias[8]   = { 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f };
    /* without bias, top-4 by raw logit = {0,1,2,3}; with bias, expert 4's
     * biased value (11.0) beats all of them, so top-4 must include 4 and
     * exclude one of {0,1,2,3} (the weakest, index 3). */
    int sel_raw[4]; float wgt_raw[4];
    coli_moe_gate_topk_softmax(logits, nullptr, n, k, sel_raw, wgt_raw);
    bool raw_has4=false; for (int j=0;j<k;j++) if (sel_raw[j]==4) raw_has4=true;
    CHECK(!raw_has4, "sanity: without bias expert 4 must NOT be selected");

    int sel[4]; float wgt[4];
    coli_moe_gate_topk_softmax(logits, bias, n, k, sel, wgt);
    std::vector<int> sel_o; std::vector<float> wgt_o;
    moe_gate_oracle(logits, bias, n, k, sel_o, wgt_o);
    for (int j=0;j<k;j++) CHECK(sel[j]==sel_o[j], "sel[%d]=%d oracle=%d", j, sel[j], sel_o[j]);
    for (int j=0;j<k;j++) CHECK(NEAR(wgt[j],wgt_o[j],1e-6), "wgt[%d]=%.8f oracle=%.8f", j, wgt[j], wgt_o[j]);
    bool has4=false; float sum=0; for (int j=0;j<k;j++){ if(sel[j]==4) has4=true; sum+=wgt[j]; }
    CHECK(has4, "biased router must select expert 4");
    CHECK(NEAR(sum,1.0,1e-5), "weights must sum to 1 (softmax over the selected k only): sum=%.8f", sum);
}
/* PERTURBATION CONTROL: bias ignored entirely must fail the "has4" property
 * above -- proves the oracle comparison is sensitive to the bias-vs-no-bias
 * bug rather than passing regardless. */
static void test_control_moe_gate_can_fail(void) {
    int n=8, k=4;
    float logits[8] = { 5.0f, 4.9f, 4.8f, 4.7f, 1.0f, 0.9f, 0.8f, 0.7f };
    int sel_ignoring_bias[4]; float wgt_tmp[4];
    coli_moe_gate_topk_softmax(logits, nullptr, n, k, sel_ignoring_bias, wgt_tmp); /* bug: bias dropped */
    bool has4=false; for (int j=0;j<k;j++) if (sel_ignoring_bias[j]==4) has4=true;
    CHECK(!has4, "CONTROL DID NOT FAIL: a router that ignores its bias entirely still selected expert 4 -- test_moe_gate_topk_softmax's bias-materiality check cannot see this bug");
}

/* ------------------------------------------------------- sink softmax --- */
static void test_softmax_with_sink(void) {
    /* n=1, score=0, sink=0 -> softmax([0,0]) = [0.5,0.5]; dropping the sink
     * column leaves exactly [0.5]. Hand-computable, not a re-derivation. */
    float s1[1] = {0.f};
    coli_softmax_with_sink(s1, 1, 0.f);
    CHECK(NEAR(s1[0], 0.5, 1e-6), "s1[0]=%.8f want 0.5", s1[0]);

    /* sink -> -inf must reproduce ordinary softmax exactly (sink absorbs zero
     * probability). Oracle: plain stable softmax computed independently. */
    float scores[4] = { 1.0f, 2.0f, -1.0f, 0.5f };
    float ref[4]; memcpy(ref, scores, sizeof ref);
    double mx=-1e300; for (int i=0;i<4;i++) if (ref[i]>mx) mx=ref[i];
    double sum=0; for (int i=0;i<4;i++) { ref[i]=(float)exp((double)ref[i]-mx); sum+=ref[i]; }
    for (int i=0;i<4;i++) ref[i]/=(float)sum;
    float got[4]; memcpy(got, scores, sizeof got);
    coli_softmax_with_sink(got, 4, -1e30f);
    for (int i=0;i<4;i++) CHECK(NEAR(got[i], ref[i], 1e-5), "i=%d got=%.8f ref=%.8f", i, got[i], ref[i]);

    /* A REAL (non -inf) sink must make the output sum to LESS than 1 -- the
     * defining property of "the sink absorbed some probability mass". */
    float s2[4] = {1.f,2.f,-1.f,0.5f};
    coli_softmax_with_sink(s2, 4, 3.0f /* bigger than any real score */);
    double s=0; for (int i=0;i<4;i++) s+=s2[i];
    CHECK(s < 0.999, "a real sink must leave sum<1 (mass absorbed by the dropped column): sum=%.6f", s);
}
/* PERTURBATION CONTROL: a version that does NOT drop the sink from the
 * normalizer denominator (i.e. omits the sink from softmax's exp-sum
 * entirely, silently ignoring it) must fail the sum<1 property. */
static void test_control_sink_can_fail(void) {
    float scores[4] = {1.f,2.f,-1.f,0.5f};
    float ref[4]; memcpy(ref, scores, sizeof ref);
    double mx=-1e300; for (int i=0;i<4;i++) if (ref[i]>mx) mx=ref[i];
    double sum=0; for (int i=0;i<4;i++) { ref[i]=(float)exp((double)ref[i]-mx); sum+=ref[i]; }
    for (int i=0;i<4;i++) ref[i]/=(float)sum; /* ordinary softmax, sink ignored entirely -- the bug */
    double s=0; for (int i=0;i<4;i++) s+=ref[i];
    CHECK(s > 0.999, "CONTROL DID NOT FAIL: a sink-ignoring softmax should sum to 1, proving the sum<1 check is sensitive to this bug class (got sum=%.6f)", s);
}

/* --------------------------------------------------- sliding-window --- */
/* Oracle: builds the SAME triu/tril mask llama.cpp/gptoss_model.py's sdpa()
 * builds, as an n x n boolean matrix, using ITS matrix construction
 * (triu(fill(-inf),diag=1) + tril(fill(-inf),diag=-window)) rather than
 * coli_swa_masked's per-pair formula -- genuinely different code shape. */
static void test_swa_masked(void) {
    int n=6, window=3;
    std::vector<std::vector<int>> masked(n, std::vector<int>(n,0));
    for (int q=0;q<n;q++) for (int k=0;k<n;k++) {
        int triu = (k > q) ? 1 : 0;                 /* diagonal=1: strictly above main diag */
        int tril = (window>0 && k <= q-window) ? 1 : 0; /* diagonal=-window */
        masked[q][k] = triu || tril;
    }
    for (int q=0;q<n;q++) for (int k=0;k<n;k++) {
        int got = coli_swa_masked(q,k,window);
        CHECK(got == masked[q][k], "q=%d k=%d got=%d want=%d", q, k, got, masked[q][k]);
    }
    /* window<=0 must be causal-only. */
    for (int q=0;q<n;q++) for (int k=0;k<n;k++) {
        int got = coli_swa_masked(q,k,0);
        int want = (k>q)?1:0;
        CHECK(got==want, "window=0 q=%d k=%d got=%d want=%d", q,k,got,want);
    }
}
static void test_control_swa_can_fail(void) {
    /* The natural off-by-one bug: window boundary at k<q-window (strict)
     * instead of k<=q-window. Must disagree with the real function somewhere
     * in a 6x3 grid, or this test proves nothing about the boundary. */
    int n=6, window=3; int mismatches=0;
    for (int q=0;q<n;q++) for (int k=0;k<n;k++) {
        int broken = (k>q) || (k < q-window);
        int got = coli_swa_masked(q,k,window);
        if (broken != got) mismatches++;
    }
    CHECK(mismatches>0, "CONTROL DID NOT FAIL: strict vs inclusive window boundary produced identical masks -- the swa test cannot see an off-by-one at the window edge");
}

int main(void) {
    test_yarn_factor1_matches_plain_rope();
    test_yarn_gptoss_ntk_boundary();
    test_control_yarn_boundary_can_fail();
    test_swiglu_oai();
    test_control_swiglu_can_fail();
    test_moe_gate_topk_softmax();
    test_control_moe_gate_can_fail();
    test_softmax_with_sink();
    test_control_sink_can_fail();
    test_swa_masked();
    test_control_swa_can_fail();
    if (g_fail) { fprintf(stderr, "\n%d assertion(s) FAILED\n", g_fail); return 1; }
    printf("test_arch_ops: all assertions passed (yarn rope, swiglu_oai, moe gate+bias, sink softmax, SWA mask -- each with a control that can fail)\n");
    return 0;
}
