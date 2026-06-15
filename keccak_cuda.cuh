#pragma once
// SHA3-256 (FIPS 202) as a CUDA device function.
// Pure register/local-array computation; no shared or global memory.
// Ported from keccak.c; uses uint64_t state for register-friendly access.

#include <stdint.h>

#define KC_ROL(a,o) (((uint64_t)(a)<<(o))^((uint64_t)(a)>>(64-(o))))

static __device__ __forceinline__ void
keccakF1600_cuda(uint64_t *s)
{
    uint64_t C[5], D, tmp;
    uint8_t R = 0x01;
    int r, x, y, j, Y;
    for (int i = 0; i < 24; i++) {
        // theta
        for (x = 0; x < 5; x++) C[x] = s[x]^s[x+5]^s[x+10]^s[x+15]^s[x+20];
        for (x = 0; x < 5; x++) {
            D = C[(x+4)%5] ^ KC_ROL(C[(x+1)%5], 1);
            for (y = 0; y < 5; y++) s[x+5*y] ^= D;
        }
        // rho pi
        x=1; y=r=0; D=s[x+5*y];
        for (j = 0; j < 24; j++) {
            r += j+1;
            Y = (2*x+3*y)%5; x=y; y=Y;
            tmp = s[x+5*y];
            s[x+5*y] = KC_ROL(D, r%64);
            D = tmp;
        }
        // chi
        for (y = 0; y < 5; y++) {
            uint64_t t0=s[0+5*y],t1=s[1+5*y],t2=s[2+5*y],t3=s[3+5*y],t4=s[4+5*y];
            s[0+5*y]=t0^((~t1)&t2); s[1+5*y]=t1^((~t2)&t3);
            s[2+5*y]=t2^((~t3)&t4); s[3+5*y]=t3^((~t4)&t0);
            s[4+5*y]=t4^((~t0)&t1);
        }
        // iota (LFSR86540)
        for (j = 0; j < 7; j++) {
            R = (uint8_t)((R<<1) ^ ((R&0x80) ? 0x71 : 0));
            if (R & 2) s[0] ^= (uint64_t)1 << ((1<<j)-1);
        }
    }
}
#undef KC_ROL

// Store 8 bytes little-endian from uint64 into byte buffer
static __device__ __forceinline__ void
kc_store64(uint8_t *x, uint64_t u)
{
    x[0]=(uint8_t)u; x[1]=(uint8_t)(u>>8); x[2]=(uint8_t)(u>>16); x[3]=(uint8_t)(u>>24);
    x[4]=(uint8_t)(u>>32); x[5]=(uint8_t)(u>>40); x[6]=(uint8_t)(u>>48); x[7]=(uint8_t)(u>>56);
}

// XOR byte `v` at byte-offset `pos` within the uint64_t state array
static __device__ __forceinline__ void
kc_xor_byte(uint64_t *s, uint32_t pos, uint8_t v)
{
    s[pos >> 3] ^= (uint64_t)v << ((pos & 7) << 3);
}

// SHA3-256: rate=136 bytes, suffix=0x06, output=32 bytes
// Handles arbitrary input length.
static __device__ __forceinline__ void
sha3_256_cuda(uint8_t *out, const uint8_t *in, uint32_t inLen)
{
    const uint32_t RATE = 136; // 1088 bits / 8
    uint64_t s[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = 0;

    // absorb
    uint32_t pos = 0;
    while (inLen > 0) {
        uint32_t chunk = inLen < (RATE - pos) ? inLen : (RATE - pos);
        for (uint32_t i = 0; i < chunk; i++)
            kc_xor_byte(s, pos + i, in[i]);
        in += chunk; inLen -= chunk; pos += chunk;
        if (pos == RATE) { keccakF1600_cuda(s); pos = 0; }
    }

    // pad: domain suffix 0x06, final 0x80
    kc_xor_byte(s, pos, 0x06);
    kc_xor_byte(s, RATE - 1, 0x80);
    keccakF1600_cuda(s);

    // squeeze 32 bytes (4 × uint64)
    for (int i = 0; i < 4; i++)
        kc_store64(out + i * 8, s[i]);
}
