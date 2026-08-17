/* Definitive comparison: LUT-int4 vs unpack-int4 vs int8+VNNI, both cache
 * regimes, n=1..16. All parallel over output rows. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static void build_planes_T(const uint8_t*W4,int64_t I,int64_t O,uint8_t*PLT){
  int64_t ng=I/4; memset(PLT,0,(size_t)4*ng*(O/2));
  #pragma omp parallel for schedule(static)
  for(int64_t o=0;o<O;o++){ const uint8_t*w=W4+o*(I/2);
    for(int64_t g=0;g<ng;g++) for(int p=0;p<4;p++){ int idx=0;
      for(int j=0;j<4;j++){ int64_t k=g*4+j; int v=(w[k>>1]>>((k&1)?4:0))&0xF; if((v>>p)&1) idx|=1<<j; }
      uint8_t*pl=PLT+((int64_t)p*ng+g)*(O/2);
      if(o&1) pl[o>>1]|=(uint8_t)(idx<<4); else pl[o>>1]|=(uint8_t)idx; } } }

/* tables for ALL groups of one activation row: ng * 32 int16 (16 used, dup'd) */
static void build_tables(const int8_t*x,int64_t I,int16_t*T){
  int64_t ng=I/4;
  for(int64_t g=0;g<ng;g++) for(int m=0;m<16;m++){ int s=0;
    for(int j=0;j<4;j++) if((m>>j)&1) s+=x[g*4+j];
    T[g*32+m]=(int16_t)s; T[g*32+m+16]=(int16_t)s; } }

static void lut_gemm(float*y,const int8_t*X,const float*XS,int n,const int16_t*TT,
                     const uint8_t*PLT,const float*sc,int64_t I,int64_t O){
  int64_t ng=I/4, nb=I/16, ostride=O/2;
  #pragma omp parallel
  {
    int32_t *acc=aligned_alloc(64,(size_t)512*4);
    #pragma omp for schedule(static)
    for(int64_t o0=0;o0<O;o0+=512){
      int64_t oe=o0+512<O?o0+512:O;
      for(int r=0;r<n;r++){
        const int16_t *T0=TT+(int64_t)r*ng*32;
        const int8_t  *xr=X+(int64_t)r*I;
        const float   *sr=XS+(int64_t)r*nb;
        for(int64_t o=o0;o<oe;o++) y[(int64_t)r*O+o]=0.f;
        for(int64_t b=0;b<nb;b++){
          memset(acc,0,(size_t)(oe-o0)*4);
          for(int64_t g=b*4;g<b*4+4;g++){
            __m512i T=_mm512_loadu_si512((const void*)(T0+g*32));
            for(int p=0;p<4;p++){
              const uint8_t*pl=PLT+((int64_t)p*ng+g)*ostride;
              for(int64_t o=o0;o<oe;o+=32){
                __m128i raw=_mm_loadu_si128((const __m128i*)(pl+(o>>1)));
                __m256i w16=_mm256_cvtepu8_epi16(raw);
                __m256i lo=_mm256_and_si256(w16,_mm256_set1_epi16(0x0F));
                __m256i hi=_mm256_and_si256(_mm256_srli_epi16(w16,4),_mm256_set1_epi16(0x0F));
                __m256i i0=_mm256_unpacklo_epi16(lo,hi), i1=_mm256_unpackhi_epi16(lo,hi);
                __m512i idx=_mm512_inserti64x4(_mm512_castsi256_si512(
                              _mm256_permute2x128_si256(i0,i1,0x20)),
                              _mm256_permute2x128_si256(i0,i1,0x31),1);
                __m512i v=_mm512_permutexvar_epi16(idx,T);
                __m512i v0=_mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));
                __m512i v1=_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v,1));
                int64_t a=o-o0;
                _mm512_storeu_si512((void*)(acc+a),   _mm512_add_epi32(_mm512_loadu_si512((const void*)(acc+a)),   _mm512_slli_epi32(v0,p)));
                _mm512_storeu_si512((void*)(acc+a+16),_mm512_add_epi32(_mm512_loadu_si512((const void*)(acc+a+16)),_mm512_slli_epi32(v1,p)));
              } }
            int32_t off=0; for(int j=0;j<4;j++) off+=8*xr[g*4+j];
            __m512i vo=_mm512_set1_epi32(off);
            for(int64_t a=0;a<oe-o0;a+=16)
              _mm512_storeu_si512((void*)(acc+a),_mm512_sub_epi32(_mm512_loadu_si512((const void*)(acc+a)),vo));
          }
          float s=sr[b];
          for(int64_t o=o0;o<oe;o++) y[(int64_t)r*O+o]+=s*(float)acc[o-o0];
        }
        for(int64_t o=o0;o<oe;o++) y[(int64_t)r*O+o]*=sc[o];
      } } free(acc); } }

/* int8 + VNNI, weights offset-to-unsigned */
static void i8_vnni(float*y,const int8_t*X,const float*XS,const int32_t*XSUM,int n,
                    const uint8_t*W,const float*sc,int64_t I,int64_t O){
  int64_t nb=I/16;
  #pragma omp parallel for schedule(static)
  for(int64_t o=0;o<O;o++){ const uint8_t*w=W+o*I; float s=sc[o];
    for(int r=0;r<n;r++){ const int8_t*xr=X+(int64_t)r*I; const float*sr=XS+(int64_t)r*nb;
      const int32_t*su=XSUM+(int64_t)r*nb; float a=0;
      for(int64_t b=0;b+4<=nb;b+=4){
        __m512i vw=_mm512_loadu_si512((const void*)(w+b*16));
        __m512i vx=_mm512_loadu_si512((const void*)(xr+b*16));
        __m512i p=_mm512_dpbusd_epi32(_mm512_setzero_si512(),vw,vx);
        int32_t t[16]; _mm512_storeu_si512((void*)t,p);
        for(int k=0;k<4;k++) a+=sr[b+k]*(float)(t[k*4]+t[k*4+1]+t[k*4+2]+t[k*4+3]-128*su[b+k]); }
      y[(int64_t)r*O+o]=a*s; } } }

static void unpack_i4(float*y,const int8_t*X,const float*XS,int n,const uint8_t*W4,
                      const float*sc,int64_t I,int64_t O){
  int64_t nb=I/16;
  #pragma omp parallel for schedule(static)
  for(int64_t o=0;o<O;o++){ const uint8_t*w=W4+o*(I/2);
    for(int r=0;r<n;r++){ const int8_t*xr=X+(int64_t)r*I; const float*sr=XS+(int64_t)r*nb; float acc=0;
      for(int64_t b=0;b<nb;b++){ int32_t s=0;
        for(int i=0;i<16;i++){ int64_t k=b*16+i; int v=(w[k>>1]>>((k&1)?4:0))&0xF; s+=(int32_t)xr[k]*(v-8); }
        acc+=sr[b]*(float)s; }
      y[(int64_t)r*O+o]=acc*sc[o]; } } }

/* SIMD int4 unpack -- the SAME kernel that measured 3.35 ms / 70 GB/s in the
 * earlier bytes-vs-bytes test. The scalar unpack_i4 above is NOT a fair
 * baseline for a vector LUT and must not be quoted as one. */
static inline int32_t dot16_avx2s(const int8_t*a,const int8_t*b){
  __m256i va=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
  __m256i vb=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
  __m256i p=_mm256_madd_epi16(va,vb);
  __m128i s=_mm_add_epi32(_mm256_castsi256_si128(p),_mm256_extracti128_si256(p,1));
  __m128i h=_mm_unpackhi_epi64(s,s); s=_mm_add_epi32(s,h);
  h=_mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1)); s=_mm_add_epi32(s,h);
  return _mm_cvtsi128_si32(s); }
static void unpack_i4_simd(float*y,const int8_t*X,const float*XS,int n,const uint8_t*W4,
                           const float*sc,int64_t I,int64_t O){
  int64_t nb=I/16;
  const __m128i m=_mm_set1_epi8(0x0F), e=_mm_set1_epi8(8);
  #pragma omp parallel for schedule(static)
  for(int64_t o=0;o<O;o++){ const uint8_t*w=W4+o*(I/2);
    for(int r=0;r<n;r++){ const int8_t*xr=X+(int64_t)r*I; const float*sr=XS+(int64_t)r*nb; float acc=0;
      for(int64_t b=0;b<nb;b+=2){
        __m128i raw=_mm_loadu_si128((const __m128i*)(w+b*8));
        __m128i lo=_mm_sub_epi8(_mm_and_si128(raw,m),e);
        __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw,4),m),e);
        int8_t t[32];
        _mm_storeu_si128((__m128i*)t,_mm_unpacklo_epi8(lo,hi));
        _mm_storeu_si128((__m128i*)(t+16),_mm_unpackhi_epi8(lo,hi));
        acc+=sr[b]*(float)dot16_avx2s(xr+b*16,t);
        if(b+1<nb) acc+=sr[b+1]*(float)dot16_avx2s(xr+(b+1)*16,t+16); }
      y[(int64_t)r*O+o]=acc*sc[o]; } } }

int main(int argc,char**argv){
  int64_t I=argc>1?atoll(argv[1]):4096, O=argc>2?atoll(argv[2]):16384;
  int reps=argc>3?atoi(argv[3]):5; int NM=16;
  int64_t nb=I/16, ng=I/4;
  uint8_t*W4=aligned_alloc(64,(size_t)I*O/2), *W8=aligned_alloc(64,(size_t)I*O);
  uint8_t*PLT=aligned_alloc(64,(size_t)4*ng*(O/2));
  int8_t*X=aligned_alloc(64,(size_t)I*NM);
  float*XS=aligned_alloc(64,(size_t)nb*NM*4),*sc=aligned_alloc(64,(size_t)O*4);
  int32_t*XSUM=aligned_alloc(64,(size_t)nb*NM*4);
  int16_t*TT=aligned_alloc(64,(size_t)ng*32*NM*2);
  float*Y=aligned_alloc(64,(size_t)O*NM*4),*Y2=aligned_alloc(64,(size_t)O*NM*4);
  srand(5);
  for(int64_t i=0;i<I*O/2;i++) W4[i]=(uint8_t)(rand()&0xFF);
  for(int64_t i=0;i<I*O;i++) W8[i]=(uint8_t)(rand()&0xFF);
  for(int64_t i=0;i<I*NM;i++) X[i]=(int8_t)((rand()%255)-127);
  for(int r=0;r<NM;r++){ for(int64_t b=0;b<nb;b++){ XS[r*nb+b]=0.001f; int32_t s=0;
      for(int i=0;i<16;i++) s+=X[(int64_t)r*I+b*16+i]; XSUM[r*nb+b]=s; }
    build_tables(X+(int64_t)r*I,I,TT+(int64_t)r*ng*32); }
  for(int64_t o=0;o<O;o++) sc[o]=0.002f;
  build_planes_T(W4,I,O,PLT);
  printf("I=%lld O=%lld   int4/bitplane=%.0f MiB   int8=%.0f MiB   (L3=96 MiB)\n",
    (long long)I,(long long)O,(double)I*O/2/1048576.0,(double)I*O/1048576.0);
  /* correctness: LUT vs unpack at n=1 */
  unpack_i4(Y,X,XS,1,W4,sc,I,O); lut_gemm(Y2,X,XS,1,TT,PLT,sc,I,O);
  double ma=0,ym=0; for(int64_t o=0;o<O;o++){ double d=fabs((double)Y[o]-Y2[o]); if(d>ma)ma=d;
    if(fabs((double)Y[o])>ym) ym=fabs((double)Y[o]); }
  printf("LUT vs unpack: max|diff|/max|y| = %.2e %s\n",ma/ym, ma/ym<1e-4?"(equivalent)":"<-- WRONG");
  if(ma/ym>=1e-4) return 1;
  printf("%4s %11s %11s %11s %11s %9s %9s\n","n","scalar-i4","SIMD-i4","LUT-i4","int8vnni","LUT/SIMD","LUT/i8");
  int ns[]={1,2,4,8,16};
  for(unsigned q=0;q<sizeof ns/sizeof*ns;q++){ int n=ns[q];
    double b1=1e30,b2=1e30,b3=1e30,b4=1e30;
    unpack_i4(Y,X,XS,n,W4,sc,I,O); unpack_i4_simd(Y,X,XS,n,W4,sc,I,O);
    lut_gemm(Y2,X,XS,n,TT,PLT,sc,I,O); i8_vnni(Y,X,XS,XSUM,n,W8,sc,I,O);
    for(int r=0;r<reps;r++){ double t=now(); unpack_i4(Y,X,XS,n,W4,sc,I,O); t=now()-t; if(t<b1)b1=t; }
    for(int r=0;r<reps;r++){ double t=now(); unpack_i4_simd(Y,X,XS,n,W4,sc,I,O); t=now()-t; if(t<b4)b4=t; }
    for(int r=0;r<reps;r++){ double t=now(); lut_gemm(Y2,X,XS,n,TT,PLT,sc,I,O); t=now()-t; if(t<b2)b2=t; }
    for(int r=0;r<reps;r++){ double t=now(); i8_vnni(Y,X,XS,XSUM,n,W8,sc,I,O); t=now()-t; if(t<b3)b3=t; }
    printf("%4d %11.2f %11.2f %11.2f %11.2f %8.2fx %8.2fx\n",n,b1*1e3,b4*1e3,b2*1e3,b3*1e3,b4/b2,b3/b2); }
  return 0; }
