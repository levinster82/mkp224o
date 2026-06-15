#pragma once

#include "fe_cuda.cuh"
#include <stdint.h>

// Group element types matching ref10 ge.h
typedef struct { fe_cuda X, Y, Z, T; } ge_p3_cuda;
typedef struct { fe_cuda X, Y, Z, T; } ge_p1p1_cuda;
typedef struct { fe_cuda yplusx, yminusx, xy2d; } ge_precomp_cuda;
typedef struct { fe_cuda YplusX, YminusX, Z, T2d; } ge_cached_cuda;

// Eightpoint precomputed in constant memory, initialized at GPU startup.
// Defined once in worker_cuda.cu; declared extern here so both translation
// units share the same constant-memory slot (requires -dc + -dlink).
extern __constant__ ge_precomp_cuda cuda_ge_eightpoint;

// r = p + q  (p in extended coords, q precomputed)
// ge_p1p1 result; caller converts with ge_p1p1_to_p3_cuda
static __device__ __forceinline__ void
ge_madd_cuda(ge_p1p1_cuda *r, const ge_p3_cuda *p, const ge_precomp_cuda *q)
{
    fe_cuda t0;
    fe_add_cuda(r->X, p->Y, p->X);
    fe_sub_cuda(r->Y, p->Y, p->X);
    fe_mul_cuda(r->Z, r->X, q->yplusx);
    fe_mul_cuda(r->Y, r->Y, q->yminusx);
    fe_mul_cuda(r->T, q->xy2d, p->T);
    fe_add_cuda(t0, p->Z, p->Z);
    fe_sub_cuda(r->X, r->Z, r->Y);
    fe_add_cuda(r->Y, r->Z, r->Y);
    fe_add_cuda(r->Z, t0, r->T);
    fe_sub_cuda(r->T, t0, r->T);
}

// r = p  (completed → extended)
static __device__ __forceinline__ void
ge_p1p1_to_p3_cuda(ge_p3_cuda *r, const ge_p1p1_cuda *p)
{
    fe_mul_cuda(r->X, p->X, p->T);
    fe_mul_cuda(r->Y, p->Y, p->Z);
    fe_mul_cuda(r->Z, p->Z, p->T);
    fe_mul_cuda(r->T, p->X, p->Y);
}

// Add constant-memory ge_eightpoint to p, result in p.
// This is the hot inner loop: one ge_madd + one p1p1_to_p3.
static __device__ __forceinline__ void
ge_add_eightpoint_cuda(ge_p3_cuda *p)
{
    ge_p1p1_cuda sum;
    ge_madd_cuda(&sum, p, &cuda_ge_eightpoint);
    ge_p1p1_to_p3_cuda(p, &sum);
}

// Convert ge_p3 to 32-byte canonical representation.
// Computes x = X/Z, y = Y/Z, packs y and sets sign bit from x.
// Note: destroys Z (replaces with Z_inv) and computes x, y in registers.
static __device__ __forceinline__ void
ge_p3_tobytes_cuda(uint8_t *s, const ge_p3_cuda *p)
{
    fe_cuda recip, x, y;
    fe_invert_cuda(recip, p->Z);
    fe_mul_cuda(x, p->X, recip);
    fe_mul_cuda(y, p->Y, recip);
    fe_tobytes_cuda(s, y);
    s[31] ^= (uint8_t)(fe_isnegative_cuda(x) << 7);
}

// Batch version: use pre-computed Z inverse to convert to bytes without inverting each point.
// z_inv: the inverse of this point's Z coordinate
static __device__ __forceinline__ void
ge_p3_tobytes_batched_cuda(uint8_t *s, const fe_cuda X, const fe_cuda Y, const fe_cuda z_inv)
{
    fe_cuda x, y;
    fe_mul_cuda(x, X, z_inv);
    fe_mul_cuda(y, Y, z_inv);
    fe_tobytes_cuda(s, y);
    s[31] ^= (uint8_t)(fe_isnegative_cuda(x) << 7);
}

// Helpers to store/load X,Y,Z limbs in strided global memory.
// Buffer layout: int32_t[BATCHNUM * 30 * stride], coordinates packed as:
//   slot b, field f (0=X,1=Y,2=Z), limb li → index (b*30 + f*10 + li)*stride + tid
static __device__ __forceinline__ void
ge_store_xyz(int32_t *buf, int b, int stride, int tid, const ge_p3_cuda *p)
{
    #pragma unroll
    for (int li = 0; li < 10; li++) {
        buf[(b * 30 + 0 * 10 + li) * stride + tid] = p->X[li];
        buf[(b * 30 + 1 * 10 + li) * stride + tid] = p->Y[li];
        buf[(b * 30 + 2 * 10 + li) * stride + tid] = p->Z[li];
    }
}

static __device__ __forceinline__ void
ge_load_xy(fe_cuda X_out, fe_cuda Y_out, const int32_t *buf, int b, int stride, int tid)
{
    #pragma unroll
    for (int li = 0; li < 10; li++) {
        X_out[li] = buf[(b * 30 + 0 * 10 + li) * stride + tid];
        Y_out[li] = buf[(b * 30 + 1 * 10 + li) * stride + tid];
    }
}

// Extract just Z coordinate from batch buffer
static __device__ __forceinline__ void
ge_load_z(fe_cuda Z_out, const int32_t *buf, int b, int stride, int tid)
{
    #pragma unroll
    for (int li = 0; li < 10; li++)
        Z_out[li] = buf[(b * 30 + 2 * 10 + li) * stride + tid];
}

// Store Z coordinate back (used to write Z_inv after batch inversion)
static __device__ __forceinline__ void
ge_store_z(int32_t *buf, int b, int stride, int tid, const fe_cuda Z)
{
    #pragma unroll
    for (int li = 0; li < 10; li++)
        buf[(b * 30 + 2 * 10 + li) * stride + tid] = Z[li];
}
