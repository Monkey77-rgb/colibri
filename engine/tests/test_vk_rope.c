/* test_vk_rope -- do rope_bias.comp and kvwrite.comp agree with the CPU?
 *
 * TWO KERNELS, TWO DIFFERENT KINDS OF CLAIM.
 *
 * rope_bias is checked for BIT-EXACTNESS, not a tolerance, and that is a real
 * claim rather than an optimistic one. The (c,s) table is host-computed and
 * handed to both sides, so the only arithmetic in question is `a*c - b*s` and
 * `a*s + b*c` in fp32 -- IEEE-754 requires each of those roundings to be
 * correctly rounded, so the two must agree exactly unless the SPIR-V compiler
 * contracts a multiply-add into an FMA, which rounds once where the CPU rounds
 * twice. If this test starts reporting a 1-ULP difference, that contraction is
 * the first thing to look at; do NOT paper over it with a tolerance, because
 * the whole reason the table stays on the host is cross-platform determinism.
 *
 * kvwrite is checked END-TO-END through attention rather than by reading the
 * cache back. What can actually go wrong is not the copy, it is the ADDRESS:
 * the kernel recomputes KVOFF and if its stride disagrees with the one attention
 * reads, every value is individually correct and in the wrong place. A getter
 * would have to duplicate the same index arithmetic to check it, so it would
 * agree with a wrong kernel. Running attention over the cache the kernel wrote
 * and comparing against a CPU attention over the rows we MEANT to write catches
 * a stride error; a getter does not.
 *
 * EVERY PHASE HAS A CONTROL THAT MUST FAIL. This file exists partly because of
 * an earlier lesson in this engine: deleting a pipeline barrier did not fail
 * test_vk_attn, because the driver happened to order the work anyway. A check
 * that cannot report the opposite result is not evidence. Each control below
 * perturbs exactly one thing and must be reported as MISMATCH.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../src/vk_backend.h"
#include "../src/trig.h"

static unsigned long rs = 987654321;
/* (long) cast is load-bearing -- see the same note in test_vk_attn.c: without it
 * the subtraction happens in unsigned arithmetic and every input becomes ~9.2e14,
 * which both sides then consume identically and partly agree on. */
static float frnd(void){ rs = rs*6364136223846793005ULL + 1442695040888963407ULL;
                         return (float)((long)((rs>>33)%20001) - 10000) / 10000.0f; }

/* The host table, byte-for-byte the arithmetic of rope_table() in model.cpp. */
static void cpu_rope_table(float *cs, int pos, int hd, float theta) {
    int half = hd/2;
    for (int i=0;i<half;i++) {
        double fr = coli_pow((double)theta, -(double)(2*i)/(double)hd);
        double sd, cd; coli_sincos((double)pos*fr, &sd, &cd);
        cs[i*2] = (float)cd; cs[i*2+1] = (float)sd;
    }
}
/* rope_head_tab() from model.cpp, same order of operations. */
static void cpu_rope_head(float *v, const float *cs, int hd, int neox) {
    int half = hd/2;
    for (int i=0;i<half;i++) {
        float c = cs[i*2], s = cs[i*2+1];
        int ia = neox ? i : 2*i, ib = neox ? i+half : 2*i+1;
        float a=v[ia], b=v[ib];
        v[ia]=a*c-b*s; v[ib]=a*s+b*c;
    }
}

static void cpu_attend(float *out, const float *q, const float *K, const float *V,
                       int hd, int tmax, float scale) {
    double m = -1e30;
    for (int t=0;t<=tmax;t++){ double d=0; for(int i=0;i<hd;i++) d+=(double)q[i]*K[(size_t)t*hd+i];
                               d*=scale; if(d>m) m=d; }
    double den=0; for (int i=0;i<hd;i++) out[i]=0;
    for (int t=0;t<=tmax;t++){ double d=0; for(int i=0;i<hd;i++) d+=(double)q[i]*K[(size_t)t*hd+i];
        double w=exp(d*scale-m); den+=w;
        for(int i=0;i<hd;i++) out[i]+=(float)(w*V[(size_t)t*hd+i]); }
    for (int i=0;i<hd;i++) out[i]=(float)(out[i]/den);
}

int main(void) {
    const int H=28, KVH=4, hd=128, n=3, slots=2, kv_ctx=64, layers=2, neox=1;
    const float theta=1000000.f;
    const int qD=H*hd, kvD=KVH*hd, half=hd/2;

    char err[256]={0};
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, sizeof err);
    if (!v) { printf("no Vulkan device (%s) -- SKIP\n", err); return 0; }
    if (!coli_vk_has_rope(v)) { printf("rope_bias.spv/kvwrite.spv absent -- SKIP\n"); coli_vk_free(v); return 0; }
    printf("device: %s\n", coli_vk_device_name(v));

    int pos[3] = { 0, 17, 63 };          /* 0 exercises the c=1,s=0 identity */
    int slot[3] = { 0, 1, 1 };

    float *q  = (float*)malloc((size_t)n*qD*4),  *q2 = (float*)malloc((size_t)n*qD*4);
    float *k  = (float*)malloc((size_t)n*kvD*4), *k2 = (float*)malloc((size_t)n*kvD*4);
    float *vv = (float*)malloc((size_t)n*kvD*4);
    float *bias = (float*)malloc((size_t)(qD+2*kvD)*4);
    float *cs = (float*)malloc((size_t)n*half*2*4);
    for (int i=0;i<n*qD;i++)  q[i]=frnd();
    for (int i=0;i<n*kvD;i++) k[i]=frnd();
    for (int i=0;i<n*kvD;i++) vv[i]=frnd();
    for (int i=0;i<qD+2*kvD;i++) bias[i]=frnd()*0.25f;
    for (int r=0;r<n;r++) cpu_rope_table(cs+(size_t)r*half*2, pos[r], hd, theta);
    memcpy(q2,q,(size_t)n*qD*4); memcpy(k2,k,(size_t)n*kvD*4);

    /* ---------------- phase 1: bias + rotation, expected BIT-EXACT ---------- */
    for (int r=0;r<n;r++) {
        for (int i=0;i<qD;i++)  q2[(size_t)r*qD+i]  += bias[i];
        for (int i=0;i<kvD;i++) k2[(size_t)r*kvD+i] += bias[qD+i];
        for (int h=0;h<H;h++)   cpu_rope_head(q2+(size_t)r*qD +h*hd, cs+(size_t)r*half*2, hd, neox);
        for (int h=0;h<KVH;h++) cpu_rope_head(k2+(size_t)r*kvD+h*hd, cs+(size_t)r*half*2, hd, neox);
    }
    if (coli_vk_rope_bias_upload(v,bias,(size_t)(qD+2*kvD))!=0) { printf("bias upload FAILED\n"); return 1; }
    if (coli_vk_rope_cs_upload(v,cs,(size_t)n*half*2)!=0)       { printf("cs upload FAILED\n");   return 1; }
    if (coli_vk_rope_run(v,q,k,n,H,KVH,hd,neox,1)!=0)           { printf("rope_run FAILED\n");    return 1; }

    long qbad=0, kbad=0; float qmax=0;
    for (int i=0;i<n*qD;i++)  { if (q[i]!=q2[i]) { qbad++; float d=fabsf(q[i]-q2[i]); if(d>qmax)qmax=d; } }
    for (int i=0;i<n*kvD;i++) { if (k[i]!=k2[i]) kbad++; }
    printf("phase1 rope+bias: q mismatches %ld/%d, k mismatches %ld/%d, max abs diff %.3e -- %s\n",
           qbad, n*qD, kbad, n*kvD, qmax, (qbad==0&&kbad==0)?"BIT-EXACT":"DIFFERS");

    /* CONTROL: the comparison must be able to see a one-element change. If this
     * prints CONTROL BROKEN the loop above proves nothing. */
    { float save=q[7]; q[7]=save+1e-4f;
      long c=0; for (int i=0;i<n*qD;i++) if (q[i]!=q2[i]) c++;
      printf("  control (perturb q[7] by 1e-4): %ld mismatches -- %s\n",
             c, c>qbad ? "OK, comparison is live" : "CONTROL BROKEN");
      q[7]=save; }

    /* ---------------- phase 2: KV scatter, checked THROUGH attention -------- */
    if (coli_vk_kv_init(v,layers,slots,KVH,kv_ctx,hd)!=0) { printf("kv_init FAILED\n"); return 1; }
    const int layer=1, bv_off=qD+kvD;

    /* Fill both slots with a history so attention has something to attend over,
     * using the load path (already covered by test_vk_attn), then let kvwrite
     * place the three new rows on top of it. */
    size_t per = (size_t)slots*KVH*kv_ctx*hd;
    float *hK = (float*)calloc(per,4), *hV = (float*)calloc(per,4);
    for (size_t i=0;i<per;i++){ hK[i]=frnd(); hV[i]=frnd(); }
    if (coli_vk_kv_load(v,layer,hK,hV)!=0) { printf("kv_load FAILED\n"); return 1; }
    /* The control below re-runs the scatter with a wrong position. It must start
     * from the PRE-scatter cache: the first (correct) scatter already wrote row 0
     * at pos 0, and row 0 has tmax=0, so attention would read that still-correct
     * value and the control would pass no matter what -- which is exactly what it
     * did before this copy existed. */
    float *hK0=(float*)malloc(per*4), *hV0=(float*)malloc(per*4);
    memcpy(hK0,hK,per*4); memcpy(hV0,hV,per*4);

    /* what we MEAN for the cache to hold after the scatter */
    for (int r=0;r<n;r++) for (int h=0;h<KVH;h++) {
        size_t dst = (((size_t)slot[r]*KVH + h)*kv_ctx + pos[r])*hd;
        for (int d=0;d<hd;d++) {
            hK[dst+d] = k[(size_t)r*kvD + h*hd + d];
            hV[dst+d] = vv[(size_t)r*kvD + h*hd + d] + bias[bv_off + h*hd + d];
        }
    }
    if (coli_vk_kvwrite_run(v,layer,k,vv,slot,pos,n,KVH,hd,bv_off,1)!=0) { printf("kvwrite FAILED\n"); return 1; }

    float *ga = (float*)malloc((size_t)n*qD*4), *ca = (float*)malloc((size_t)n*qD*4);
    int meta[6]; for (int r=0;r<n;r++){ meta[r*2]=slot[r]; meta[r*2+1]=pos[r]; }
    if (coli_vk_attn(v,layer,q,ga,meta,n,H,1.f/sqrtf((float)hd))!=0) {
        printf("attn FAILED\n"); return 1; }
    for (int r=0;r<n;r++) for (int h=0;h<H;h++) {
        int kvh=h/(H/KVH);
        cpu_attend(ca+(size_t)r*qD+h*hd, q+(size_t)r*qD+h*hd,
                   hK+(((size_t)slot[r]*KVH+kvh)*kv_ctx)*hd,
                   hV+(((size_t)slot[r]*KVH+kvh)*kv_ctx)*hd,
                   hd, pos[r], 1.f/sqrtf((float)hd));
    }
    double num=0,den=0;
    for (int i=0;i<n*qD;i++){ double d=ga[i]-ca[i]; num+=d*d; den+=(double)ca[i]*ca[i]; }
    double rel = sqrt(num/(den>0?den:1));
    /* Same 1e-4 bound as test_vk_attn, and for the same reason: the kernel's
     * four-subgroup softmax merge is not the reference's sequential walk. */
    printf("phase2 kvwrite->attn: rel=%.3e -- %s\n", rel, rel<1e-4 ? "PASS" : "FAIL");

    /* CONTROL: place row 0 one position late. If the comparison is live this
     * must break -- and it is the exact failure a wrong KVOFF stride produces. */
    { int bad[3]={pos[0]+1,pos[1],pos[2]};
      if (coli_vk_kv_load(v,layer,hK0,hV0)!=0) { printf("control reload FAILED\n"); return 1; }
      if (coli_vk_kvwrite_run(v,layer,k,vv,slot,bad,n,KVH,hd,bv_off,1)!=0) { printf("control write FAILED\n"); return 1; }
      if (coli_vk_attn(v,layer,q,ga,meta,n,H,1.f/sqrtf((float)hd))!=0) { printf("control attn FAILED\n"); return 1; }
      double n2=0; for (int i=0;i<n*qD;i++){ double d=ga[i]-ca[i]; n2+=d*d; }
      double r2 = sqrt(n2/(den>0?den:1));
      printf("  control (row 0 written to pos+1): rel=%.3e -- %s\n",
             r2, r2>1e-4 ? "OK, comparison is live" : "CONTROL BROKEN"); }

    coli_vk_free(v);
    free(q);free(q2);free(k);free(k2);free(vv);free(bias);free(cs);free(hK);free(hV);free(hK0);free(hV0);free(ga);free(ca);
    return 0;
}
