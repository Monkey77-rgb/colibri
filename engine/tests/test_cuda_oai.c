/* test_cuda_oai -- mirrors tests/test_vk_oai.c's arms 2 and 3 (MXFP4 GEMM and
 * the fused ffn4_oai expert), through the coli_backend seam. Arm 1 (attention
 * window+sink) is NOT mirrored: backend_cuda.cu leaves every attn_ and kv_ entry
 * NULL by design (see its header) -- this engine's CUDA backend does GEMM
 * only, and the CPU attention path stays in use, exactly like a Vulkan build
 * without shaders/attn_decode.spv.
 *
 * Also exercises slot_alloc_mx + slot_fill directly (allocate empty, fill,
 * then gemm4 on the filled slot) per the task's instruction, rather than only
 * through upload_w4_mx.
 */
#include "../src/backend.h"
#include "../src/gemm_i8.h"
#include "../src/gemm_mxfp4.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define FM(n) ((float*)malloc((size_t)(n)*4))
#define UM(n) ((uint8_t*)malloc((size_t)(n)))
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static unsigned long rs = 424242;
static float frnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (float)((long)((rs>>33)%20001) - 10000) / 10000.f; }
static unsigned urnd(void) { rs = rs*6364136223846793005ul + 1442695040888963407ul; return (unsigned)(rs>>33); }

static void fill_mx(uint8_t *blocks, int64_t I, int64_t O) {
    int64_t nb=I/COLI_MXFP4_BLK;
    for (int64_t i=0;i<O*nb;i++) { uint8_t *b=blocks+i*COLI_MXFP4_BYTES;
        b[0] = (uint8_t)(120 + urnd()%9);
        for (int j=1;j<17;j++) b[j]=(uint8_t)(urnd()&0xFF); }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    char err[256]={0};
    coli_backend *be = coli_backend_cuda_open(err, sizeof err);
    if (!be) { printf("no CUDA device (%s) -- SKIP\n", err); return 0; }
    printf("device: %s  mx=%d ffn_oai=%d\n", be->device_name(be->ctx), be->has_mx(be->ctx), be->has_ffn_oai(be->ctx));
    if (!be->has_mx(be->ctx) || !be->has_ffn_oai(be->ctx)) { printf("FAIL: MXFP4 pipeline missing\n"); return 1; }
    int pass = 1;

    /* ------------------------------------------------ 1. MXFP4 gemm, via slot_alloc_mx + slot_fill */
    int64_t I=2880, O=2880;
    {
        int64_t nb=I/COLI_MXFP4_BLK; size_t bytes=(size_t)O*nb*COLI_MXFP4_BYTES;
        uint8_t *blocks=UM(bytes); fill_mx(blocks,I,O);
        coli_w_mxfp4 mw={blocks,I,O,0};
        coli_w_i4 w4; if (!coli_mxfp4_repack_i4(blocks,I,O,&w4)) { printf("FAIL repack\n"); return 1; }

        /* exercise slot_alloc_mx + slot_fill directly, rather than upload_w4_mx */
        int h = be->slot_alloc_mx(be->ctx, I, O);
        if (h < 0) { printf("FAIL slot_alloc_mx\n"); return 1; }
        if (be->slot_fill(be->ctx, h, &w4) != 0) { printf("FAIL slot_fill\n"); return 1; }
        free(w4.q4); free(w4.bscale);

        for (int n=1;n<=4;n*=4) {
            float *x=FM((size_t)n*I); for (int64_t i=0;i<n*I;i++) x[i]=frnd();
            coli_a_i8 a; a.n=n; a.I=I; a.q=(int8_t*)malloc((size_t)n*I); a.scale=(float*)malloc((size_t)n*(I/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(I/COLI_ABLK)*4);
            coli_quantize_a(&a,x,n,I);
            float *yr=FM((size_t)n*O), *yg=FM((size_t)n*O);
            coli_gemm_mxfp4_ref(yr,&a,&mw);
            if (be->gemm4(be->ctx,h,&a,yg)!=0) { printf("FAIL gemm4 (mx) n=%d\n", n); return 1; }
            const double TOL=2e-5;
            double ymax=0, md=0; for (int64_t i=0;i<n*O;i++){ if (fabs(yr[i])>ymax) ymax=fabs(yr[i]); double d=fabs((double)yg[i]-yr[i]); if (d>md) md=d; }
            double w=md/ymax;
            /* control: flip one nibble in row 7 block 3 of the ORIGINAL blocks and re-reference */
            blocks[(7*nb+3)*COLI_MXFP4_BYTES+5] ^= 0x7;
            float *yc=FM((size_t)n*O); coli_gemm_mxfp4_ref(yc,&a,&mw);
            blocks[(7*nb+3)*COLI_MXFP4_BYTES+5] ^= 0x7;
            double mc=0; for (int64_t i=0;i<n*O;i++){ double d=fabs((double)yg[i]-yc[i]); if (d>mc) mc=d; }
            double wc=mc/ymax;
            double gt=1e30; for (int rep=0;rep<7;rep++){ double t0=now(); be->gemm4(be->ctx,h,&a,yg); double d=now()-t0; if(d<gt)gt=d; }
            double gbps=(double)(I*O/2)/gt/1e9;
            printf("  mx gemm n=%d I=%lld O=%lld: max|gpu-cpu|/max|y|=%.3e %s | control (nibble flip) %.3e %s | %.3f us/call %.2f GB/s\n",
                   n,(long long)I,(long long)O,w,w<=TOL?"ok":"BAD",wc,wc>TOL?"fails as required":"INERT",gt*1e6,gbps);
            pass &= (w<=TOL) && (wc>TOL);
            free(x);free(a.q);free(a.scale);free(a.sum);free(yr);free(yg);free(yc);
        }
        free(blocks);
    }

    /* ------------------------------------------------ 2. fused expert FFN, SwiGLU-OAI + biases */
    {
        int64_t D=I, EI=O; int64_t nbD=D/COLI_MXFP4_BLK, nbE=EI/COLI_MXFP4_BLK;
        uint8_t *bg_=UM((size_t)EI*nbD*COLI_MXFP4_BYTES), *bu_=UM((size_t)EI*nbD*COLI_MXFP4_BYTES), *bd_=UM((size_t)D*nbE*COLI_MXFP4_BYTES);
        fill_mx(bg_,D,EI); fill_mx(bu_,D,EI); fill_mx(bd_,EI,D);
        coli_w_mxfp4 mg={bg_,D,EI,0}, mu={bu_,D,EI,0}, md={bd_,EI,D,0};
        int hs[3]; const uint8_t *bl[3]={bg_,bu_,bd_}; int64_t Is[3]={D,D,EI}, Os[3]={EI,EI,D};
        for (int t=0;t<3;t++){ coli_w_i4 w4; if(!coli_mxfp4_repack_i4(bl[t],Is[t],Os[t],&w4)){printf("FAIL repack\n");return 1;}
            hs[t]=be->upload_w4_mx(be->ctx,&w4); free(w4.q4); free(w4.bscale); if(hs[t]<0){printf("FAIL upload\n");return 1;} }
        float *bg=FM((size_t)EI), *bu=FM((size_t)EI); for (int64_t i=0;i<EI;i++){ bg[i]=frnd()*0.5f; bu[i]=frnd()*0.5f; }
        float alpha=1.702f, limit=7.0f;
        for (int n=1;n<=3;n+=2) {
            float *x=FM((size_t)n*D); for (int64_t i=0;i<n*D;i++) x[i]=frnd()*4.f;
            coli_a_i8 a; a.n=n; a.I=D; a.q=(int8_t*)malloc((size_t)n*D); a.scale=(float*)malloc((size_t)n*(D/COLI_ABLK)*4); a.sum=(int32_t*)malloc((size_t)n*(D/COLI_ABLK)*4);
            coli_quantize_a(&a,x,n,D);
            float *G=FM((size_t)n*EI), *U=FM((size_t)n*EI), *yr=FM((size_t)n*D), *yg=FM((size_t)n*D);
            coli_gemm_mxfp4_ref(G,&a,&mg); coli_gemm_mxfp4_ref(U,&a,&mu);
            int nclamp=0;
            float *Hh=FM((size_t)n*EI);
            #define ACT(al,lim,dst) do { for (int r=0;r<n;r++) for (int64_t i=0;i<EI;i++){ \
                float xg=G[r*EI+i]+bg[i]; if (xg>(lim)) { xg=(lim); nclamp++; } \
                float yu=U[r*EI+i]+bu[i]; if (yu>(lim)) yu=(lim); if (yu<-(lim)) yu=-(lim); \
                float glu=xg/(1.f+expf((al)*(-xg))); (dst)[r*EI+i]=glu*(yu+1.f);} } while(0)
            ACT(alpha,limit,Hh);
            coli_a_i8 ah; ah.n=n; ah.I=EI; ah.q=(int8_t*)malloc((size_t)n*EI); ah.scale=(float*)malloc((size_t)n*(EI/COLI_ABLK)*4); ah.sum=(int32_t*)malloc((size_t)n*(EI/COLI_ABLK)*4);
            coli_quantize_a(&ah,Hh,n,EI); coli_gemm_mxfp4_ref(yr,&ah,&md);
            if (be->ffn4_oai(be->ctx,hs[0],hs[1],hs[2],&a,yg,bg,bu,alpha,limit)!=0) { printf("FAIL ffn4_oai n=%d\n",n); return 1; }
            /* 1e-4, not test_vk_oai's 3e-2: this path measures EXACT (0.00e+00)
             * agreement with the CPU reference chain (unlike Vulkan, which uses
             * its own GPU exp() and a tree-reduced GEMM), so the tolerance is
             * tightened to what the alpha x1.1 control can actually be measured
             * against -- see the header for why 3e-2 would leave that control
             * INERT here. */
            const double TOL=1e-4;
            double w=0; { double ym=0,md2=0; for(int64_t i=0;i<n*D;i++){ double c=fabs((double)yr[i]); if(c>ym)ym=c; double d=fabs((double)yg[i]-yr[i]); if(d>md2)md2=d; } w=md2/(ym+1e-1); }
            int nc0=nclamp; nclamp=0;
            float *Hc=FM((size_t)n*EI), *yc1=FM((size_t)n*D), *yc2=FM((size_t)n*D);
            ACT(alpha*1.1f,limit,Hc); coli_quantize_a(&ah,Hc,n,EI); coli_gemm_mxfp4_ref(yc1,&ah,&md);
            ACT(alpha,limit*0.5f,Hc); coli_quantize_a(&ah,Hc,n,EI); coli_gemm_mxfp4_ref(yc2,&ah,&md);
            double wc1=0,wc2=0; { double ym=0; for(int64_t i=0;i<n*D;i++){ double c=fabs((double)yr[i]); if(c>ym)ym=c;}
                double d1=0,d2=0; for(int64_t i=0;i<n*D;i++){ double a1=fabs((double)yg[i]-yc1[i]); if(a1>d1)d1=a1; double a2=fabs((double)yg[i]-yc2[i]); if(a2>d2)d2=a2; }
                wc1=d1/(ym+1e-1); wc2=d2/(ym+1e-1); }
            double gt=1e30; for (int rep=0;rep<5;rep++){ double t0=now(); be->ffn4_oai(be->ctx,hs[0],hs[1],hs[2],&a,yg,bg,bu,alpha,limit); double d=now()-t0; if(d<gt)gt=d; }
            double gbps=(double)(D*EI/2*2 + EI*D/2)/gt/1e9;
            printf("  ffn4_oai n=%d D=%lld EI=%lld (%d gate clamps hit): rel=%.3e %s | controls: alpha x1.1 %.3e %s, limit/2 %.3e %s | %.3f us/call %.2f GB/s\n",
                   n,(long long)D,(long long)EI,nc0,w,w<=TOL?"ok":"BAD",wc1,wc1>TOL?"fails as required":"INERT",wc2,wc2>TOL?"fails as required":"INERT",gt*1e6,gbps);
            pass &= (w<=TOL) && (wc1>TOL) && (wc2>TOL) && (nc0>0);
            free(x);free(a.q);free(a.scale);free(a.sum);free(G);free(U);free(yr);free(yg);free(Hh);free(ah.q);free(ah.scale);free(ah.sum);free(Hc);free(yc1);free(yc2);
        }
        free(bg_);free(bu_);free(bd_);free(bg);free(bu);
    }

    coli_backend_close(be);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
