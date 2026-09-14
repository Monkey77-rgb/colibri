/* arch_ops.cpp -- see arch_ops.h. */
#include "arch_ops.h"
#include "yarn_rope.h"
#include <math.h>
#include <string.h>

void coli_yarn_rope_cs(float *c, float *s, int half, int pos, int hd, float base,
                       float factor, float beta_fast, float beta_slow, int orig_ctx,
                       int truncate) {
    double concentration = 1.0;
    double low = 0, high = 1; /* only used when factor>1 */
    double lnbase = log((double)base);
    if (factor > 1.0) {
        concentration = 0.1 * log((double)factor) + 1.0;
        double d_half = hd / 2.0;
        low  = d_half * log((double)orig_ctx / ((double)beta_fast * 2.0 * M_PI)) / lnbase;
        high = d_half * log((double)orig_ctx / ((double)beta_slow * 2.0 * M_PI)) / lnbase;
        if (truncate) {   /* ggml_rope_yarn_corr_dims: floor/ceil, clamp to [0, hd-1] */
            low = floor(low); high = ceil(high);
            if (low < 0) low = 0;
            if (high > hd - 1) high = hd - 1;
        }
    }
    for (int i=0;i<half;i++) {
        double freq = pow((double)base, (2.0*i)/(double)hd);
        double inv_freq;
        if (factor > 1.0) {
            double interp = 1.0/((double)factor*freq);
            double extrap = 1.0/freq;
            double ramp = (high != low) ? ((double)i - low) / (high - low) : 0.0;
            if (ramp < 0.0) ramp = 0.0;
            if (ramp > 1.0) ramp = 1.0;
            double mask = 1.0 - ramp;
            inv_freq = interp*(1.0-mask) + extrap*mask;
        } else {
            inv_freq = 1.0/freq;
        }
        double ang = (double)pos * inv_freq;
        c[i] = (float)(cos(ang) * concentration);
        s[i] = (float)(sin(ang) * concentration);
    }
}

void coli_yarn_rope_table(coli_rope_tab *t, int pos, int hd, float base,
                           float factor, float beta_fast, float beta_slow, int orig_ctx) {
    int half = hd/2; if (half > COLI_ROPE_MAXHALF) half = COLI_ROPE_MAXHALF;
    t->half = half;
    /* truncate=0: the reference form this function has always computed. */
    coli_yarn_rope_cs(t->c, t->s, half, pos, hd, base, factor, beta_fast, beta_slow, orig_ctx, 0);
}

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

void coli_swiglu_oai(float *out, const float *gate, const float *up, int n,
                      float alpha, float limit) {
    for (int i=0;i<n;i++) {
        float g = gate[i]; if (g > limit) g = limit;
        float u = up[i];
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
        float glu = g * sigmoidf_(alpha * g);
        out[i] = glu * (u + 1.0f);
    }
}

void coli_moe_gate_topk_softmax(const float *logits, const float *bias,
                                 int n_expert, int k, int *sel, float *wgt) {
    for (int j=0;j<k;j++) {
        int best=-1; float bv=-1e30f;
        for (int e=0;e<n_expert;e++) {
            int taken=0; for (int p=0;p<j;p++) if (sel[p]==e) { taken=1; break; }
            if (taken) continue;
            float v = logits[e] + (bias ? bias[e] : 0.0f);
            if (v > bv) { bv=v; best=e; }
        }
        sel[j]=best; wgt[j]=bv;
    }
    float mx=-1e30f; for (int j=0;j<k;j++) if (wgt[j]>mx) mx=wgt[j];
    float sum=0.0f; for (int j=0;j<k;j++) { wgt[j]=expf(wgt[j]-mx); sum+=wgt[j]; }
    for (int j=0;j<k;j++) wgt[j]/=sum;
}

void coli_softmax_with_sink(float *scores, int n, float sink) {
    float mx = sink; for (int i=0;i<n;i++) if (scores[i]>mx) mx=scores[i];
    float sum = expf(sink-mx);
    for (int i=0;i<n;i++) { scores[i]=expf(scores[i]-mx); sum+=scores[i]; }
    for (int i=0;i<n;i++) scores[i]/=sum;
    /* sink's own exp(sink-mx)/sum term is intentionally left out of `scores` --
     * that is the probability mass "spent" on the sink and dropped. */
}

int coli_swa_masked(int qpos, int kpos, int window) {
    if (kpos > qpos) return 1;
    if (window > 0 && (qpos - kpos) >= window) return 1;
    return 0;
}
