/* Real-GGUF placement oracle, no routing or fused activation. Run from engine/.
 * Same quantized activation goes to native CPU and repacked Vulkan MXFP4.
 * Metrics normalized by max|CPU| match bench_gemm_n; elementwise ratios use
 * a stated 1e-3*max|CPU| floor to expose cancellation without dividing by zero.
 * Synthetic activations are deterministic, not captured model activations.
 */
#include "../src/vk_backend.h"
#include "../src/gemm_mxfp4.h"
#include "../../c/gguf_reader.h"
#include <vector>
#include <algorithm>
#include <cmath>

static double exact(const uint8_t *row, const coli_a_i8 &a, int r, int64_t I) {
    static const int lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    double acc=0;
    for (int64_t g=0; g<I/32; ++g) {
        const uint8_t *b=row+g*17;
        uint32_t bits=b[0]<2 ? 0x00200000u<<b[0] : uint32_t(b[0]-1)<<23;
        float ws; memcpy(&ws,&bits,4);
        for(int j=0;j<32;++j) {
            int k=g*32+j, code=j<16 ? b[j+1]&15 : b[j-15]>>4;
            acc += double(ws)*lut[code]*a.q[r*I+k]*a.scale[r*(I/16)+k/16];
        }
    }
    return acc;
}

int main(int argc, char **argv) {
    if(argc!=2) { fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 2; }
    char err[512]={}; GgufIndex idx={};
    if(!gguf_index_open(argv[1],&idx,err,sizeof err)) { puts(err); return 1; }
    coli_vk *v=coli_vk_init("shaders/gemm_i8.spv",err,sizeof err);
    if(!v) { printf("SKIP: %s\n",err); gguf_index_free(&idx); return 77; }
    if(!coli_vk_has_mx(v)) { puts("FAIL: MX pipeline missing"); return 1; }
    printf("device=%s CPU=%s tolerance=3.1e-4 normalized max error\n",
           coli_vk_device_name(v),coli_gemm_mxfp4_kernel());
    bool pass=true;
    const int layers[]={0,17,35}, experts[]={3,67,127};
    const char *kind[]={"gate","up","down"};
    unsigned seed=7;
    for(int s=0;s<3;++s) for(int t=0;t<3;++t) {
        char name[128]; snprintf(name,sizeof name,"blk.%d.ffn_%s_exps.weight",layers[s],kind[t]);
        const GgufTensorInfo *ti=nullptr;
        for(size_t j=0;j<idx.n;++j) if(!strcmp(idx.t[j].name,name)) ti=&idx.t[j];
        if(!ti || ti->ttype!=39 || ti->rank!=3 || ti->shape[2]<=unsigned(experts[s])) {
            printf("FAIL: missing/unsupported %s\n",name); return 1;
        }
        int64_t I=ti->shape[0], O=ti->shape[1];
        if(I%32 || I>65536 || O>65536) return 1;
        size_t bytes=O*(I/32)*17;
        std::vector<uint8_t> blocks(bytes);
        if(!gguf_read_at(idx.shard[ti->shard].fd,blocks.data(),bytes,ti->data_off+bytes*experts[s])) return 1;
        coli_w_mxfp4 mw={blocks.data(),I,O,0}; coli_w_i4 w4={};
        if(!coli_mxfp4_repack_i4(blocks.data(),I,O,&w4)) return 1;
        int h=coli_vk_upload_w4_mx(v,&w4); free(w4.q4); free(w4.bscale);
        if(h<0) return 1;
        for(int n: {1,8}) {
            std::vector<float> x(n*I), sc(n*I/16), cpu(n*O), ref(n*O), gpu(n*O), ctl(n*O);
            std::vector<int8_t> q(n*I); std::vector<int32_t> sum(n*I/16);
            for(auto &z:x) { seed=seed*1664525u+1013904223u; z=float((seed>>8)&65535)/32768.f-1.f; }
            coli_a_i8 a={}; a.n=n; a.I=I; a.q=q.data(); a.scale=sc.data(); a.sum=sum.data();
            coli_quantize_a(&a,x.data(),n,I);
            coli_gemm_mxfp4(cpu.data(),&a,&mw);
            coli_gemm_mxfp4_ref(ref.data(),&a,&mw);
            bool identical=!memcmp(cpu.data(),ref.data(),n*O*sizeof(float));
            if(coli_vk_gemm4(v,h,&a,gpu.data())) return 1;
            double ymax=0, md=0, mean=0, elemmax=0, elemmean=0, dg=0, dc=0;
            for(float z:cpu) ymax=std::max(ymax,std::abs(double(z)));
            if(!(ymax>0) || !std::isfinite(ymax)) return 1;
            for(int r=0;r<n;++r) for(int64_t o=0;o<O;++o) {
                size_t i=r*O+o; double d=std::abs(double(gpu[i])-cpu[i]);
                if(!std::isfinite(d)) return 1;
                md=std::max(md,d); mean+=d;
                double e=d/std::max(std::abs(double(cpu[i])),ymax*1e-3);
                elemmax=std::max(elemmax,e); elemmean+=e;
                double ex=exact(blocks.data()+o*(I/32)*17,a,r,I);
                dg=std::max(dg,std::abs(gpu[i]-ex)); dc=std::max(dc,std::abs(cpu[i]-ex));
            }
            // Change one activation operand's scale by 10%; GPU output stays fixed.
            for(float &z:sc) z*=1.1f;
            coli_gemm_mxfp4(ctl.data(),&a,&mw);
            double cm=0; for(size_t i=0;i<ctl.size();++i) cm=std::max(cm,std::abs(double(ctl[i])-gpu[i]));
            bool ok=identical && md/ymax<=3.1e-4 && cm/ymax>3.1e-4;
            printf("%s expert=%d I=%lld O=%lld n=%d max_rel=%.9g mean_rel=%.9g element_max=%.9g element_mean=%.9g double_gpu=%.9g double_cpu=%.9g CPU_ref_bytes=%s control=%.9g %s\n",
                name,experts[s],(long long)I,(long long)O,n,md/ymax,mean/(n*O*ymax),elemmax,elemmean/(n*O),dg/ymax,dc/ymax,identical?"SAME":"DIFFER",cm/ymax,ok?"PASS":"FAIL");
            pass &= ok;
        }
    }
    coli_vk_free(v); gguf_index_free(&idx);
    puts(pass?"PASS":"FAIL"); return pass?0:1;
}
