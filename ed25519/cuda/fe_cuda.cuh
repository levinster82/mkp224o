#pragma once

#include <stdint.h>
#include <string.h>

// 10-limb 26/25-bit alternating representation of GF(2^255-19) elements.
// Same layout as ref10 fe[10]. All computation in registers; no shared memory.
typedef int32_t fe_cuda[10];

static __device__ __forceinline__ void
fe_add_cuda(fe_cuda h, const fe_cuda f, const fe_cuda g)
{
    h[0]=f[0]+g[0]; h[1]=f[1]+g[1]; h[2]=f[2]+g[2]; h[3]=f[3]+g[3]; h[4]=f[4]+g[4];
    h[5]=f[5]+g[5]; h[6]=f[6]+g[6]; h[7]=f[7]+g[7]; h[8]=f[8]+g[8]; h[9]=f[9]+g[9];
}

static __device__ __forceinline__ void
fe_sub_cuda(fe_cuda h, const fe_cuda f, const fe_cuda g)
{
    h[0]=f[0]-g[0]; h[1]=f[1]-g[1]; h[2]=f[2]-g[2]; h[3]=f[3]-g[3]; h[4]=f[4]-g[4];
    h[5]=f[5]-g[5]; h[6]=f[6]-g[6]; h[7]=f[7]-g[7]; h[8]=f[8]-g[8]; h[9]=f[9]-g[9];
}

static __device__ __forceinline__ void
fe_neg_cuda(fe_cuda h, const fe_cuda f)
{
    h[0]=-f[0]; h[1]=-f[1]; h[2]=-f[2]; h[3]=-f[3]; h[4]=-f[4];
    h[5]=-f[5]; h[6]=-f[6]; h[7]=-f[7]; h[8]=-f[8]; h[9]=-f[9];
}

static __device__ __forceinline__ void
fe_copy_cuda(fe_cuda h, const fe_cuda f)
{
    h[0]=f[0]; h[1]=f[1]; h[2]=f[2]; h[3]=f[3]; h[4]=f[4];
    h[5]=f[5]; h[6]=f[6]; h[7]=f[7]; h[8]=f[8]; h[9]=f[9];
}

static __device__ __forceinline__ void fe_0_cuda(fe_cuda h)
{
    h[0]=h[1]=h[2]=h[3]=h[4]=h[5]=h[6]=h[7]=h[8]=h[9]=0;
}

static __device__ __forceinline__ void fe_1_cuda(fe_cuda h)
{
    h[0]=1; h[1]=h[2]=h[3]=h[4]=h[5]=h[6]=h[7]=h[8]=h[9]=0;
}

static __device__ __forceinline__ void
fe_cmov_cuda(fe_cuda f, const fe_cuda g, unsigned int b)
{
    int32_t bm = -(int32_t)b;
    f[0]^=(f[0]^g[0])&bm; f[1]^=(f[1]^g[1])&bm;
    f[2]^=(f[2]^g[2])&bm; f[3]^=(f[3]^g[3])&bm;
    f[4]^=(f[4]^g[4])&bm; f[5]^=(f[5]^g[5])&bm;
    f[6]^=(f[6]^g[6])&bm; f[7]^=(f[7]^g[7])&bm;
    f[8]^=(f[8]^g[8])&bm; f[9]^=(f[9]^g[9])&bm;
}

static __device__ __forceinline__ int
fe_isnegative_cuda(const fe_cuda f)
{
    // sign bit is the LSB of the canonical byte representation
    // which equals f[0] & 1 after reduction (f[0] is the low limb)
    return f[0] & 1;
}

static __device__ __forceinline__ void
fe_mul_cuda(fe_cuda h, const fe_cuda f, const fe_cuda g)
{
    int32_t f0=f[0],f1=f[1],f2=f[2],f3=f[3],f4=f[4];
    int32_t f5=f[5],f6=f[6],f7=f[7],f8=f[8],f9=f[9];
    int32_t g0=g[0],g1=g[1],g2=g[2],g3=g[3],g4=g[4];
    int32_t g5=g[5],g6=g[6],g7=g[7],g8=g[8],g9=g[9];
    int32_t g1_19=19*g1,g2_19=19*g2,g3_19=19*g3,g4_19=19*g4;
    int32_t g5_19=19*g5,g6_19=19*g6,g7_19=19*g7,g8_19=19*g8,g9_19=19*g9;
    int32_t f1_2=2*f1,f3_2=2*f3,f5_2=2*f5,f7_2=2*f7,f9_2=2*f9;
    long long h0=f0*(long long)g0+f1_2*(long long)g9_19+f2*(long long)g8_19
               +f3_2*(long long)g7_19+f4*(long long)g6_19+f5_2*(long long)g5_19
               +f6*(long long)g4_19+f7_2*(long long)g3_19+f8*(long long)g2_19
               +f9_2*(long long)g1_19;
    long long h1=f0*(long long)g1+f1*(long long)g0+f2*(long long)g9_19
               +f3*(long long)g8_19+f4*(long long)g7_19+f5*(long long)g6_19
               +f6*(long long)g5_19+f7*(long long)g4_19+f8*(long long)g3_19
               +f9*(long long)g2_19;
    long long h2=f0*(long long)g2+f1_2*(long long)g1+f2*(long long)g0
               +f3_2*(long long)g9_19+f4*(long long)g8_19+f5_2*(long long)g7_19
               +f6*(long long)g6_19+f7_2*(long long)g5_19+f8*(long long)g4_19
               +f9_2*(long long)g3_19;
    long long h3=f0*(long long)g3+f1*(long long)g2+f2*(long long)g1
               +f3*(long long)g0+f4*(long long)g9_19+f5*(long long)g8_19
               +f6*(long long)g7_19+f7*(long long)g6_19+f8*(long long)g5_19
               +f9*(long long)g4_19;
    long long h4=f0*(long long)g4+f1_2*(long long)g3+f2*(long long)g2
               +f3_2*(long long)g1+f4*(long long)g0+f5_2*(long long)g9_19
               +f6*(long long)g8_19+f7_2*(long long)g7_19+f8*(long long)g6_19
               +f9_2*(long long)g5_19;
    long long h5=f0*(long long)g5+f1*(long long)g4+f2*(long long)g3
               +f3*(long long)g2+f4*(long long)g1+f5*(long long)g0
               +f6*(long long)g9_19+f7*(long long)g8_19+f8*(long long)g7_19
               +f9*(long long)g6_19;
    long long h6=f0*(long long)g6+f1_2*(long long)g5+f2*(long long)g4
               +f3_2*(long long)g3+f4*(long long)g2+f5_2*(long long)g1
               +f6*(long long)g0+f7_2*(long long)g9_19+f8*(long long)g8_19
               +f9_2*(long long)g7_19;
    long long h7=f0*(long long)g7+f1*(long long)g6+f2*(long long)g5
               +f3*(long long)g4+f4*(long long)g3+f5*(long long)g2
               +f6*(long long)g1+f7*(long long)g0+f8*(long long)g9_19
               +f9*(long long)g8_19;
    long long h8=f0*(long long)g8+f1_2*(long long)g7+f2*(long long)g6
               +f3_2*(long long)g5+f4*(long long)g4+f5_2*(long long)g3
               +f6*(long long)g2+f7_2*(long long)g1+f8*(long long)g0
               +f9_2*(long long)g9_19;
    long long h9=f0*(long long)g9+f1*(long long)g8+f2*(long long)g7
               +f3*(long long)g6+f4*(long long)g5+f5*(long long)g4
               +f6*(long long)g3+f7*(long long)g2+f8*(long long)g1
               +f9*(long long)g0;
    long long carry0,carry1,carry2,carry3,carry4,carry5,carry6,carry7,carry8,carry9;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    carry1=(h1+(long long)(1<<24))>>25; h2+=carry1; h1-=carry1<<25;
    carry5=(h5+(long long)(1<<24))>>25; h6+=carry5; h5-=carry5<<25;
    carry2=(h2+(long long)(1<<25))>>26; h3+=carry2; h2-=carry2<<26;
    carry6=(h6+(long long)(1<<25))>>26; h7+=carry6; h6-=carry6<<26;
    carry3=(h3+(long long)(1<<24))>>25; h4+=carry3; h3-=carry3<<25;
    carry7=(h7+(long long)(1<<24))>>25; h8+=carry7; h7-=carry7<<25;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    carry8=(h8+(long long)(1<<25))>>26; h9+=carry8; h8-=carry8<<26;
    carry9=(h9+(long long)(1<<24))>>25; h0+=carry9*19; h9-=carry9<<25;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    h[0]=(int32_t)h0; h[1]=(int32_t)h1; h[2]=(int32_t)h2; h[3]=(int32_t)h3;
    h[4]=(int32_t)h4; h[5]=(int32_t)h5; h[6]=(int32_t)h6; h[7]=(int32_t)h7;
    h[8]=(int32_t)h8; h[9]=(int32_t)h9;
}

static __device__ __forceinline__ void
fe_sq_cuda(fe_cuda h, const fe_cuda f)
{
    int32_t f0=f[0],f1=f[1],f2=f[2],f3=f[3],f4=f[4];
    int32_t f5=f[5],f6=f[6],f7=f[7],f8=f[8],f9=f[9];
    int32_t f0_2=2*f0,f1_2=2*f1,f2_2=2*f2,f3_2=2*f3,f4_2=2*f4;
    int32_t f5_2=2*f5,f6_2=2*f6,f7_2=2*f7;
    int32_t f5_38=38*f5,f6_19=19*f6,f7_38=38*f7,f8_19=19*f8,f9_38=38*f9;
    long long h0=f0*(long long)f0+f1_2*(long long)f9_38+f2_2*(long long)f8_19
               +f3_2*(long long)f7_38+f4_2*(long long)f6_19+f5*(long long)f5_38;
    long long h1=f0_2*(long long)f1+f2*(long long)f9_38+f3_2*(long long)f8_19
               +f4*(long long)f7_38+f5_2*(long long)f6_19;
    long long h2=f0_2*(long long)f2+f1_2*(long long)f1+f3_2*(long long)f9_38
               +f4_2*(long long)f8_19+f5_2*(long long)f7_38+f6*(long long)f6_19;
    long long h3=f0_2*(long long)f3+f1_2*(long long)f2+f4*(long long)f9_38
               +f5_2*(long long)f8_19+f6*(long long)f7_38;
    long long h4=f0_2*(long long)f4+f1_2*(long long)f3_2+f2*(long long)f2
               +f5_2*(long long)f9_38+f6_2*(long long)f8_19+f7*(long long)f7_38;
    long long h5=f0_2*(long long)f5+f1_2*(long long)f4+f2_2*(long long)f3
               +f6*(long long)f9_38+f7_2*(long long)f8_19;
    long long h6=f0_2*(long long)f6+f1_2*(long long)f5_2+f2_2*(long long)f4
               +f3_2*(long long)f3+f7_2*(long long)f9_38+f8*(long long)f8_19;
    long long h7=f0_2*(long long)f7+f1_2*(long long)f6+f2_2*(long long)f5
               +f3_2*(long long)f4+f8*(long long)f9_38;
    long long h8=f0_2*(long long)f8+f1_2*(long long)f7_2+f2_2*(long long)f6
               +f3_2*(long long)f5_2+f4*(long long)f4+f9*(long long)f9_38;
    long long h9=f0_2*(long long)f9+f1_2*(long long)f8+f2_2*(long long)f7
               +f3_2*(long long)f6+f4_2*(long long)f5;
    long long carry0,carry1,carry2,carry3,carry4,carry5,carry6,carry7,carry8,carry9;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    carry1=(h1+(long long)(1<<24))>>25; h2+=carry1; h1-=carry1<<25;
    carry5=(h5+(long long)(1<<24))>>25; h6+=carry5; h5-=carry5<<25;
    carry2=(h2+(long long)(1<<25))>>26; h3+=carry2; h2-=carry2<<26;
    carry6=(h6+(long long)(1<<25))>>26; h7+=carry6; h6-=carry6<<26;
    carry3=(h3+(long long)(1<<24))>>25; h4+=carry3; h3-=carry3<<25;
    carry7=(h7+(long long)(1<<24))>>25; h8+=carry7; h7-=carry7<<25;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    carry8=(h8+(long long)(1<<25))>>26; h9+=carry8; h8-=carry8<<26;
    carry9=(h9+(long long)(1<<24))>>25; h0+=carry9*19; h9-=carry9<<25;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    h[0]=(int32_t)h0; h[1]=(int32_t)h1; h[2]=(int32_t)h2; h[3]=(int32_t)h3;
    h[4]=(int32_t)h4; h[5]=(int32_t)h5; h[6]=(int32_t)h6; h[7]=(int32_t)h7;
    h[8]=(int32_t)h8; h[9]=(int32_t)h9;
}

// h = 2 * f^2
static __device__ __forceinline__ void
fe_sq2_cuda(fe_cuda h, const fe_cuda f)
{
    fe_sq_cuda(h, f);
    for (int i = 0; i < 10; i++) h[i] *= 2;
    // Re-normalize carries after doubling
    long long h0=h[0],h1=h[1],h2=h[2],h3=h[3],h4=h[4];
    long long h5=h[5],h6=h[6],h7=h[7],h8=h[8],h9=h[9];
    long long carry0,carry4;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    long long carry1=(h1+(long long)(1<<24))>>25; h2+=carry1; h1-=carry1<<25;
    long long carry5=(h5+(long long)(1<<24))>>25; h6+=carry5; h5-=carry5<<25;
    long long carry2=(h2+(long long)(1<<25))>>26; h3+=carry2; h2-=carry2<<26;
    long long carry6=(h6+(long long)(1<<25))>>26; h7+=carry6; h6-=carry6<<26;
    long long carry3=(h3+(long long)(1<<24))>>25; h4+=carry3; h3-=carry3<<25;
    long long carry7=(h7+(long long)(1<<24))>>25; h8+=carry7; h7-=carry7<<25;
    carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    long long carry8=(h8+(long long)(1<<25))>>26; h9+=carry8; h8-=carry8<<26;
    long long carry9=(h9+(long long)(1<<24))>>25; h0+=carry9*19; h9-=carry9<<25;
    carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    h[0]=(int32_t)h0; h[1]=(int32_t)h1; h[2]=(int32_t)h2; h[3]=(int32_t)h3;
    h[4]=(int32_t)h4; h[5]=(int32_t)h5; h[6]=(int32_t)h6; h[7]=(int32_t)h7;
    h[8]=(int32_t)h8; h[9]=(int32_t)h9;
}

static __device__ __forceinline__ void
fe_tobytes_cuda(uint8_t *s, const fe_cuda h)
{
    int32_t h0=h[0],h1=h[1],h2=h[2],h3=h[3],h4=h[4];
    int32_t h5=h[5],h6=h[6],h7=h[7],h8=h[8],h9=h[9];
    int32_t q=(19*h9+((int32_t)1<<24))>>25;
    q=(h0+q)>>26; q=(h1+q)>>25; q=(h2+q)>>26; q=(h3+q)>>25;
    q=(h4+q)>>26; q=(h5+q)>>25; q=(h6+q)>>26; q=(h7+q)>>25;
    q=(h8+q)>>26; q=(h9+q)>>25;
    h0+=19*q;
    int32_t carry0=h0>>26; h1+=carry0; h0-=carry0<<26;
    int32_t carry1=h1>>25; h2+=carry1; h1-=carry1<<25;
    int32_t carry2=h2>>26; h3+=carry2; h2-=carry2<<26;
    int32_t carry3=h3>>25; h4+=carry3; h3-=carry3<<25;
    int32_t carry4=h4>>26; h5+=carry4; h4-=carry4<<26;
    int32_t carry5=h5>>25; h6+=carry5; h5-=carry5<<25;
    int32_t carry6=h6>>26; h7+=carry6; h6-=carry6<<26;
    int32_t carry7=h7>>25; h8+=carry7; h7-=carry7<<25;
    int32_t carry8=h8>>26; h9+=carry8; h8-=carry8<<26;
    int32_t carry9=h9>>25;              h9-=carry9<<25;
    s[0]=(uint8_t)(h0>>0); s[1]=(uint8_t)(h0>>8); s[2]=(uint8_t)(h0>>16);
    s[3]=(uint8_t)((h0>>24)|(h1<<2));
    s[4]=(uint8_t)(h1>>6); s[5]=(uint8_t)(h1>>14);
    s[6]=(uint8_t)((h1>>22)|(h2<<3));
    s[7]=(uint8_t)(h2>>5); s[8]=(uint8_t)(h2>>13);
    s[9]=(uint8_t)((h2>>21)|(h3<<5));
    s[10]=(uint8_t)(h3>>3); s[11]=(uint8_t)(h3>>11);
    s[12]=(uint8_t)((h3>>19)|(h4<<6));
    s[13]=(uint8_t)(h4>>2); s[14]=(uint8_t)(h4>>10); s[15]=(uint8_t)(h4>>18);
    s[16]=(uint8_t)(h5>>0); s[17]=(uint8_t)(h5>>8); s[18]=(uint8_t)(h5>>16);
    s[19]=(uint8_t)((h5>>24)|(h6<<1));
    s[20]=(uint8_t)(h6>>7); s[21]=(uint8_t)(h6>>15);
    s[22]=(uint8_t)((h6>>23)|(h7<<3));
    s[23]=(uint8_t)(h7>>5); s[24]=(uint8_t)(h7>>13);
    s[25]=(uint8_t)((h7>>21)|(h8<<4));
    s[26]=(uint8_t)(h8>>4); s[27]=(uint8_t)(h8>>12);
    s[28]=(uint8_t)((h8>>20)|(h9<<6));
    s[29]=(uint8_t)(h9>>2); s[30]=(uint8_t)(h9>>10); s[31]=(uint8_t)(h9>>18);
}

static __device__ __forceinline__ void
fe_frombytes_cuda(fe_cuda h, const uint8_t *s)
{
    typedef unsigned long long u64;
    auto load3 = [](const uint8_t *in) -> u64 {
        return (u64)in[0] | ((u64)in[1]<<8) | ((u64)in[2]<<16);
    };
    auto load4 = [](const uint8_t *in) -> u64 {
        return (u64)in[0] | ((u64)in[1]<<8) | ((u64)in[2]<<16) | ((u64)in[3]<<24);
    };
    long long h0=load4(s);
    long long h1=load3(s+4)<<6;
    long long h2=load3(s+7)<<5;
    long long h3=load3(s+10)<<3;
    long long h4=load3(s+13)<<2;
    long long h5=load4(s+16);
    long long h6=load3(s+20)<<7;
    long long h7=load3(s+23)<<5;
    long long h8=load3(s+26)<<4;
    long long h9=(load3(s+29)&8388607)<<2;
    long long carry9=(h9+(long long)(1<<24))>>25; h0+=carry9*19; h9-=carry9<<25;
    long long carry1=(h1+(long long)(1<<24))>>25; h2+=carry1; h1-=carry1<<25;
    long long carry3=(h3+(long long)(1<<24))>>25; h4+=carry3; h3-=carry3<<25;
    long long carry5=(h5+(long long)(1<<24))>>25; h6+=carry5; h5-=carry5<<25;
    long long carry7=(h7+(long long)(1<<24))>>25; h8+=carry7; h7-=carry7<<25;
    long long carry0=(h0+(long long)(1<<25))>>26; h1+=carry0; h0-=carry0<<26;
    long long carry2=(h2+(long long)(1<<25))>>26; h3+=carry2; h2-=carry2<<26;
    long long carry4=(h4+(long long)(1<<25))>>26; h5+=carry4; h4-=carry4<<26;
    long long carry6=(h6+(long long)(1<<25))>>26; h7+=carry6; h6-=carry6<<26;
    long long carry8=(h8+(long long)(1<<25))>>26; h9+=carry8; h8-=carry8<<26;
    h[0]=(int32_t)h0; h[1]=(int32_t)h1; h[2]=(int32_t)h2; h[3]=(int32_t)h3;
    h[4]=(int32_t)h4; h[5]=(int32_t)h5; h[6]=(int32_t)h6; h[7]=(int32_t)h7;
    h[8]=(int32_t)h8; h[9]=(int32_t)h9;
}

// fe^(2^255-21) = fe^-1 mod p, using the same addition chain as ref10/pow225521.h
static __device__ __forceinline__ void
fe_invert_cuda(fe_cuda out, const fe_cuda z)
{
    fe_cuda t0, t1, t2, t3;
    int i;
    fe_sq_cuda(t0, z);
    fe_sq_cuda(t1, t0); for (i=1;i<1;i++) fe_sq_cuda(t1,t1);
    fe_mul_cuda(t1, z, t1);
    fe_mul_cuda(t0, t0, t1);
    fe_sq_cuda(t2, t0); for (i=1;i<1;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t1, t1, t2);
    fe_sq_cuda(t2, t1); for (i=1;i<4;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t1, t2, t1);
    fe_sq_cuda(t2, t1); for (i=1;i<9;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t2, t2, t1);
    fe_sq_cuda(t3, t2); for (i=1;i<19;i++) fe_sq_cuda(t3,t3);
    fe_mul_cuda(t2, t3, t2);
    fe_sq_cuda(t2, t2); for (i=1;i<9;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t1, t2, t1);
    fe_sq_cuda(t2, t1); for (i=1;i<49;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t2, t2, t1);
    fe_sq_cuda(t3, t2); for (i=1;i<99;i++) fe_sq_cuda(t3,t3);
    fe_mul_cuda(t2, t3, t2);
    fe_sq_cuda(t2, t2); for (i=1;i<49;i++) fe_sq_cuda(t2,t2);
    fe_mul_cuda(t1, t2, t1);
    fe_sq_cuda(t1, t1); for (i=1;i<4;i++) fe_sq_cuda(t1,t1);
    fe_mul_cuda(out, t1, t0);
}

// Montgomery batch inversion on strided global memory.
// zbuf layout:    int32_t[n * zstep * stride + zoff * stride]
//   slot b, limb li → zbuf[(b * zstep + zoff + li) * stride + tid]
// tmpbuf layout:  int32_t[n * 10 * stride]
//   slot b, limb li → tmpbuf[(b * 10 + li) * stride + tid]
// After return, zbuf's Z fields contain the inverses of the original Z values.
//
// For a standalone Z buffer: zstep=10, zoff=0
// For an XYZ buffer (X=0..9, Y=10..19, Z=20..29 within each 30-element slot):
//   zstep=30, zoff=20
static __device__ void
fe_batchinvert_cuda(int32_t * __restrict__ zbuf,
                    int32_t * __restrict__ tmpbuf,
                    int n, int zstep, int zoff, int stride, int tid)
{
    fe_cuda acc, z, tmp;
    fe_1_cuda(acc);

    // Forward pass: store prefix products in tmpbuf
    for (int b = 0; b < n; b++) {
        // save acc → tmpbuf[b]
        #pragma unroll
        for (int li = 0; li < 10; li++)
            tmpbuf[(b * 10 + li) * stride + tid] = acc[li];
        // load Z[b] from strided position
        #pragma unroll
        for (int li = 0; li < 10; li++)
            z[li] = zbuf[(b * zstep + zoff + li) * stride + tid];
        fe_mul_cuda(acc, acc, z);
    }

    fe_invert_cuda(acc, acc);

    // Backward pass: compute per-element inverses
    for (int b = n - 1; b >= 0; b--) {
        // load original Z[b]
        #pragma unroll
        for (int li = 0; li < 10; li++)
            z[li] = zbuf[(b * zstep + zoff + li) * stride + tid];
        // load prefix product
        #pragma unroll
        for (int li = 0; li < 10; li++)
            tmp[li] = tmpbuf[(b * 10 + li) * stride + tid];
        // Z[b]^-1 = acc * prefix
        fe_cuda zinv;
        fe_mul_cuda(zinv, acc, tmp);
        // store inverse back into Z field
        #pragma unroll
        for (int li = 0; li < 10; li++)
            zbuf[(b * zstep + zoff + li) * stride + tid] = zinv[li];
        // advance: acc = acc * original_Z[b]
        fe_mul_cuda(tmp, acc, z);
        fe_copy_cuda(acc, tmp);
    }
}
