/* test_vk_oai -- the three gpt-oss GPU pieces (2026-09-14), each against a
 * reference that is NOT the kernel's algorithm, each with a control that must
 * fail:
 *
 *  1. attn_decode.comp with a sliding window and an attention sink, vs a
 *     two-pass CPU softmax that includes the sink as one extra logit with no
 *     value row (ggml's form: max over {scores, sink}, den += exp(sink-max)).
 *     Controls: the reference WITHOUT the sink and the reference WITHOUT the
 *     window must both exceed tolerance against the same GPU output; the GPU
 *     with window=0/sink=-1 must match the plain reference (so the flags are
 *     read, not ignored, in both directions).
 *  2. gemm_i4_mx_dp.comp over a repacked MXFP4 matrix vs coli_gemm_mxfp4_ref
 *     over the ORIGINAL 17-byte blocks (the CPU kernel the 09-14 oracle used).
 *     Control: one nibble flipped in the blocks after upload -> the reference
 *     moves, the comparison must fail.
 *  3. coli_vk_ffn4_oai (gate/up MXFP4, biased SwiGLU-OAI, down MXFP4) vs the
 *     CPU chain coli_gemm_mxfp4_ref + expert_act arithmetic + coli_quantize_a +
 *     coli_gemm_mxfp4_ref. Controls: alpha x1.1 and limit halved in the
 *     reference must both fail.
 *
 * Tolerances are stated next to the measured value they were set from. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../src/vk_backend.h"
#include "../src/gemm_i8.h"
#include "../src/gemm_mxfp4.h"
/* the Makefile compiles this as C++ (-x c++), so malloc needs the cast */
#define FM(n) ((float*)malloc((size_t)(n)*4))
#define UM(n) ((uint8_t*)malloc((size_t)(n)))

static unsigned long rs = 424242;
static float frnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (float)((long)((rs>>33)%20001) - 10000) / 10000.f; }
static unsigned urnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (unsigned)(rs>>33); }

static double worst_rel(const float *a, const float *b, size_t n, double floor_) {
    double w=0; for (size_t i=0;i<n;i++){ double d=fabs((double)a[i]-b[i])/(fabs((double)b[i])+floor_); if(d>w) w=d; } return w; }

/* two-pass softmax attention, optional window and sink, double accumulation */
static void attn_ref(float *o, const float *qv, const float *Kb, const float *Vb, int tmax, int hd,
                     float scale, int t0, int has_sink, float sink) {
    int nt = tmax - t0 + 1;
    float *sc = FM(nt);
    float mx = has_sink ? sink : -1e30f;
    for (int t=t0;t<=tmax;t++) { double d=0; for (int i=0;i<hd;i++) d += (double)qv[i]*Kb[(size_t)t*hd+i];
        sc[t-t0]=(float)(d*scale); if (sc[t-t0]>mx) mx=sc[t-t0]; }
    double den = has_sink ? exp((double)sink-mx) : 0.0;
    for (int t=0;t<nt;t++){ sc[t]=expf(sc[t]-mx); den+=sc[t]; }
    for (int i=0;i<hd;i++){ double a=0; for (int t=0;t<nt;t++) a += (double)sc[t]*Vb[(size_t)(t+t0)*hd+i]; o[i]=(float)(a/den); }
    free(sc);
}

static void fill_mx(uint8_t *blocks, int64_t I, int64_t O) {
    int64_t nb=I/COLI_MXFP4_BLK;
    for (int64_t i=0;i<O*nb;i++) { uint8_t *b=blocks+i*COLI_MXFP4_BYTES;
        b[0] = (uint8_t)(120 + urnd()%9);            /* 2^-7 .. 2^1 scales, like real experts */
        for (int j=1;j<17;j++) b[j]=(uint8_t)(urnd()&0xFF); }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    char err[256]={0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { printf("no Vulkan device (%s) -- SKIP\n", err); return 0; }
    printf("device: %s  mx=%d ffn_oai=%d attn=%d\n", coli_vk_device_name(v), coli_vk_has_mx(v), coli_vk_has_ffn_oai(v), coli_vk_has_attn(v));
    if (!coli_vk_has_mx(v) || !coli_vk_has_ffn_oai(v) || !coli_vk_has_attn(v)) { printf("FAIL: a gpt-oss pipeline is missing (make vk)\n"); return 1; }
    int pass = 1;

    /* ------------------------------------------------ 1. attention window + sink */
    {
        int H=8, KVH=2, hd=64, n=2, slots=2, kv_ctx=300, window=128, L=3;
        size_t qn=(size_t)n*H*hd, kvn=(size_t)slots*KVH*kv_ctx*hd;
        float *q=FM(qn), *K=FM(kvn), *V=FM(kvn), *og=FM(qn), *oc=FM(qn), *oc_nosink=FM(qn), *oc_nowin=FM(qn), *oc_plain=FM(qn);
        for (size_t i=0;i<qn;i++) q[i]=frnd()*2.f;
        for (size_t i=0;i<kvn;i++){ K[i]=frnd()*2.f; V[i]=frnd(); }
        float *sinks=FM((size_t)L*H); for (int i=0;i<L*H;i++) sinks[i]=frnd()*3.f;
        int meta[4]={0,kv_ctx-1, 1,kv_ctx/2+1};
        float scale=1.f/sqrtf((float)hd); int grp=H/KVH; int layer=2;
        for (int r=0;r<n;r++) for (int h=0;h<H;h++) {
            int slot=meta[r*2], tmax=meta[r*2+1], kvh=h/grp;
            const float *qv=q+(size_t)r*H*hd+(size_t)h*hd;
            const float *Kb=K+(((size_t)slot*KVH+kvh)*kv_ctx)*hd, *Vb=V+(((size_t)slot*KVH+kvh)*kv_ctx)*hd;
            int t0=tmax-window+1; if (t0<0) t0=0;
            float sk=sinks[layer*H+h];
            attn_ref(oc       +(size_t)r*H*hd+(size_t)h*hd, qv,Kb,Vb,tmax,hd,scale,t0,1,sk);
            attn_ref(oc_nosink+(size_t)r*H*hd+(size_t)h*hd, qv,Kb,Vb,tmax,hd,scale,t0,0,0.f);
            attn_ref(oc_nowin +(size_t)r*H*hd+(size_t)h*hd, qv,Kb,Vb,tmax,hd,scale,0,1,sk);
            attn_ref(oc_plain +(size_t)r*H*hd+(size_t)h*hd, qv,Kb,Vb,tmax,hd,scale,0,0,0.f);
        }
        if (coli_vk_kv_init(v,L,slots,KVH,kv_ctx,hd)!=0) { printf("FAIL kv_init\n"); return 1; }
        for (int l=0;l<L;l++) if (coli_vk_kv_load(v,l,K,V)!=0) { printf("FAIL kv_load\n"); return 1; }
        if (coli_vk_attn_sinks_upload(v,sinks,(size_t)L*H)!=0) { printf("FAIL sinks upload\n"); return 1; }
        if (coli_vk_attn_ex(v,layer,q,og,meta,n,H,scale,window,layer*H)!=0) { printf("FAIL attn_ex\n"); return 1; }
        /* 1e-4: test_vk_attn's tolerance (measured 1.4-1.6e-5 across 128x..16384 ctx on the 4070) */
        const double TOL=1e-4;
        double w=worst_rel(og,oc,qn,1e-3), wns=worst_rel(og,oc_nosink,qn,1e-3), wnw=worst_rel(og,oc_nowin,qn,1e-3);
        printf("  attn window=%d + sink: rel=%.3e %s | controls: no-sink ref %.3e %s, no-window ref %.3e %s\n",
               window, w, w<=TOL?"ok":"BAD", wns, wns>TOL?"fails as required":"INERT", wnw, wnw>TOL?"fails as required":"INERT");
        pass &= (w<=TOL) && (wns>TOL) && (wnw>TOL);
        if (coli_vk_attn_ex(v,layer,q,og,meta,n,H,scale,0,-1)!=0) { printf("FAIL attn_ex plain\n"); return 1; }
        double wp=worst_rel(og,oc_plain,qn,1e-3);
        printf("  attn window=0 sink=-1 vs plain ref: rel=%.3e %s (the flags are read in both directions)\n", wp, wp<=TOL?"ok":"BAD");
        pass &= (wp<=TOL);
        /* the window must also be visible at the CPU-reference level, else the control above is trivially loud */
        double wcw=worst_rel(oc,oc_nowin,qn,1e-3); printf("  (ref: windowed vs unwindowed differ by %.3e)\n", wcw);
        free(q);free(K);free(V);free(og);free(oc);free(oc_nosink);free(oc_nowin);free(oc_plain);free(sinks);
    }

    /* ------------------------------------------------ 2. MXFP4 gemm, real block layout */
    int64_t I=2880, O=2880;   /* gpt-oss-120b expert shape; I>1024 so the full (not short-row) kernel runs */
    {
        int64_t nb=I/COLI_MXFP4_BLK; size_t bytes=(size_t)O*nb*COLI_MXFP4_BYTES;
        uint8_t *blocks=UM(bytes); fill_mx(blocks,I,O);
        coli_w_mxfp4 mw={blocks,I,O,0};
        coli_w_i4 w4; if (!coli_mxfp4_repack_i4(blocks,I,O,&w4)) { printf("FAIL repack\n"); return 1; }
        int h=coli_vk_upload_w4_mx(v,&w4); free(w4.q4); free(w4.bscale);
        if (h<0) { printf("FAIL upload_w4_mx\n"); return 1; }
        for (int n=1;n<=4;n*=4) {
            float *x=FM((size_t)n*I); for (int64_t i=0;i<n*I;i++) x[i]=frnd();
            coli_a_i8 a; a.n=n; a.I=I; a.q=(int8_t*)malloc((size_t)n*I); a.scale=(float*)malloc((size_t)n*(I/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(I/COLI_ABLK)*4);
            coli_quantize_a(&a,x,n,I);
            float *yr=FM((size_t)n*O), *yg=FM((size_t)n*O);
            coli_gemm_mxfp4_ref(yr,&a,&mw);
            if (coli_vk_gemm4(v,h,&a,yg)!=0) { printf("FAIL gemm4 (mx) n=%d\n", n); return 1; }
            /* Metric and tolerance are test_vk_gemm's: max|gpu-ref| over max|ref|
             * (a per-element ratio with a small floor magnifies near-zero outputs
             * and measured 1.3e-4 / 6.5e-4 here for what is summation order --
             * see the exact-double arbiter printed alongside). 2e-5 is float32
             * tree-vs-sequential over I terms; a wrong nibble map, scale or LUT
             * sign is O(1) and the control shows one nibble already reads 2e-3. */
            const double TOL=2e-5;
            double ymax=0, md=0; for (int64_t i=0;i<n*O;i++){ if (fabs(yr[i])>ymax) ymax=fabs(yr[i]); double d=fabs((double)yg[i]-yr[i]); if (d>md) md=d; }
            double w=md/ymax;
            /* exact arbiter: dequantize both operands to double, dot in double */
            double mdg=0, mdc=0;
            for (int r=0;r<n;r++) for (int64_t o=0;o<O;o++) {
                double acc=0; const uint8_t *row=blocks+o*nb*COLI_MXFP4_BYTES;
                for (int64_t g=0;g<nb;g++){ const uint8_t *blk=row+g*COLI_MXFP4_BYTES; uint32_t bits=(blk[0]<2)?(0x00200000u<<blk[0]):((uint32_t)(blk[0]-1)<<23); float ws; memcpy(&ws,&bits,4);
                    static const int kv[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
                    for (int j=0;j<16;j++){ int64_t e0=g*32+j, e1=g*32+16+j;
                        acc += (double)ws*kv[blk[1+j]&15]*(double)a.q[r*I+e0]*a.scale[r*(I/COLI_ABLK)+e0/COLI_ABLK];
                        acc += (double)ws*kv[blk[1+j]>>4]*(double)a.q[r*I+e1]*a.scale[r*(I/COLI_ABLK)+e1/COLI_ABLK]; } }
                double dg=fabs(yg[r*O+o]-acc), dc=fabs(yr[r*O+o]-acc); if (dg>mdg) mdg=dg; if (dc>mdc) mdc=dc; }
            /* control: flip one nibble in row 7 block 3 of the ORIGINAL blocks and re-reference */
            blocks[(7*nb+3)*COLI_MXFP4_BYTES+5] ^= 0x7;
            float *yc=FM((size_t)n*O); coli_gemm_mxfp4_ref(yc,&a,&mw);
            blocks[(7*nb+3)*COLI_MXFP4_BYTES+5] ^= 0x7;
            double mc=0; for (int64_t i=0;i<n*O;i++){ double d=fabs((double)yg[i]-yc[i]); if (d>mc) mc=d; }
            double wc=mc/ymax;
            printf("  mx gemm n=%d I=%lld O=%lld: max|gpu-cpu|/max|y|=%.3e %s [vs exact double: gpu %.3e, cpu %.3e] | control (one nibble flipped in ref) %.3e %s\n",
                   n,(long long)I,(long long)O,w,w<=TOL?"ok":"BAD",mdg/ymax,mdc/ymax,wc,wc>TOL?"fails as required":"INERT");
            pass &= (w<=TOL) && (wc>TOL);
            free(x);free(a.q);free(a.scale);free(a.sum);free(yr);free(yg);free(yc);
        }
        free(blocks);
    }

    /* ------------------------------------------------ 3. fused expert FFN, SwiGLU-OAI + biases */
    {
        int64_t D=I, EI=O; int64_t nbD=D/COLI_MXFP4_BLK, nbE=EI/COLI_MXFP4_BLK;
        uint8_t *bg_=UM((size_t)EI*nbD*COLI_MXFP4_BYTES), *bu_=UM((size_t)EI*nbD*COLI_MXFP4_BYTES), *bd_=UM((size_t)D*nbE*COLI_MXFP4_BYTES);
        fill_mx(bg_,D,EI); fill_mx(bu_,D,EI); fill_mx(bd_,EI,D);
        coli_w_mxfp4 mg={bg_,D,EI,0}, mu={bu_,D,EI,0}, md={bd_,EI,D,0};
        int hs[3]; const uint8_t *bl[3]={bg_,bu_,bd_}; int64_t Is[3]={D,D,EI}, Os[3]={EI,EI,D};
        for (int t=0;t<3;t++){ coli_w_i4 w4; if(!coli_mxfp4_repack_i4(bl[t],Is[t],Os[t],&w4)){printf("FAIL repack\n");return 1;}
            hs[t]=coli_vk_upload_w4_mx(v,&w4); free(w4.q4); free(w4.bscale); if(hs[t]<0){printf("FAIL upload\n");return 1;} }
        float *bg=FM((size_t)EI), *bu=FM((size_t)EI); for (int64_t i=0;i<EI;i++){ bg[i]=frnd()*0.5f; bu[i]=frnd()*0.5f; }
        float alpha=1.702f, limit=7.0f;
        for (int n=1;n<=3;n+=2) {
            float *x=FM((size_t)n*D); for (int64_t i=0;i<n*D;i++) x[i]=frnd()*4.f;   /* wide enough to hit the clamp */
            coli_a_i8 a; a.n=n; a.I=D; a.q=(int8_t*)malloc((size_t)n*D); a.scale=(float*)malloc((size_t)n*(D/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(D/COLI_ABLK)*4);
            coli_quantize_a(&a,x,n,D);
            float *G=FM((size_t)n*EI), *U=FM((size_t)n*EI), *yr=FM((size_t)n*D), *yg=FM((size_t)n*D);
            coli_gemm_mxfp4_ref(G,&a,&mg); coli_gemm_mxfp4_ref(U,&a,&mu);
            int nclamp=0;
            /* reference activation: expert_act's gpt-oss arm verbatim (model.cpp) */
            float *Hh=FM((size_t)n*EI);
            #define ACT(al,lim,dst) do { for (int r=0;r<n;r++) for (int64_t i=0;i<EI;i++){ \
                float xg=G[r*EI+i]+bg[i]; if (xg>(lim)) { xg=(lim); nclamp++; } \
                float yu=U[r*EI+i]+bu[i]; if (yu>(lim)) yu=(lim); if (yu<-(lim)) yu=-(lim); \
                float glu=xg/(1.f+expf((al)*(-xg))); (dst)[r*EI+i]=glu*(yu+1.f);} } while(0)
            ACT(alpha,limit,Hh);
            coli_a_i8 ah; ah.n=n; ah.I=EI; ah.q=(int8_t*)malloc((size_t)n*EI); ah.scale=(float*)malloc((size_t)n*(EI/COLI_ABLK)*4); ah.sum=(int32_t*)malloc((size_t)n*(EI/COLI_ABLK)*4);
            coli_quantize_a(&ah,Hh,n,EI); coli_gemm_mxfp4_ref(yr,&ah,&md);
            if (coli_vk_ffn4_oai(v,hs[0],hs[1],hs[2],&a,yg,bg,bu,alpha,limit)!=0) { printf("FAIL ffn4_oai n=%d\n",n); return 1; }
            /* The device quantizes the hidden vector with silu_mul_q's own block
             * quantizer (not coli_quantize_a), so this is a tolerance comparison;
             * the controls show what a real defect looks like at the same scale. */
            const double TOL=3e-2;
            double w=worst_rel(yg,yr,(size_t)n*D,1e-1);
            int nc0=nclamp; nclamp=0;
            float *Hc=FM((size_t)n*EI), *yc1=FM((size_t)n*D), *yc2=FM((size_t)n*D);
            ACT(alpha*1.1f,limit,Hc); coli_quantize_a(&ah,Hc,n,EI); coli_gemm_mxfp4_ref(yc1,&ah,&md);
            ACT(alpha,limit*0.5f,Hc); coli_quantize_a(&ah,Hc,n,EI); coli_gemm_mxfp4_ref(yc2,&ah,&md);
            double wc1=worst_rel(yg,yc1,(size_t)n*D,1e-1), wc2=worst_rel(yg,yc2,(size_t)n*D,1e-1);
            printf("  ffn4_oai n=%d D=%lld EI=%lld (%d gate clamps hit): rel=%.3e %s | controls: alpha x1.1 %.3e %s, limit/2 %.3e %s\n",
                   n,(long long)D,(long long)EI,nc0/ (n? 1:1),w,w<=TOL?"ok":"BAD",wc1,wc1>TOL?"fails as required":"INERT",wc2,wc2>TOL?"fails as required":"INERT");
            pass &= (w<=TOL) && (wc1>TOL) && (wc2>TOL) && (nc0>0);
            free(x);free(a.q);free(a.scale);free(a.sum);free(G);free(U);free(yr);free(yg);free(Hh);free(ah.q);free(ah.scale);free(ah.sum);free(Hc);free(yc1);free(yc2);
        }
        free(bg_);free(bu_);free(bd_);free(bg);free(bu);
    }
    coli_vk_free(v);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
