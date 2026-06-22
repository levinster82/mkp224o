#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sodium/randombytes.h>
#include <sodium/utils.h>

extern "C" {
#include "types.h"
#include "common.h"
#include "vec.h"
#include "worker.h"
}
#include "worker_cuda.h"
#include "ed25519/cuda/ge_cuda.cuh"

// Ref10 types: fe is int32_t[10], ge_p3/ge_precomp are plain structs.
// Layout-identical to ge_p3_cuda / ge_precomp_cuda (same int32_t[10] fields).
typedef int32_t fe_ref10[10];
typedef struct { fe_ref10 X, Y, Z, T; } ge_p3_ref10;
typedef struct { fe_ref10 yplusx, yminusx, xy2d; } ge_precomp_ref10;

// Use CRYPTO_NAMESPACE so symbol names match however the ed25519 impl was compiled.
extern "C" {
void CRYPTO_NAMESPACE(ge_scalarmult_base)(ge_p3_ref10 *, const unsigned char *);
void CRYPTO_NAMESPACE(ge_get_base_precomp)(ge_precomp_ref10 *);
void CRYPTO_NAMESPACE(ge_get_eightpoint_precomp)(ge_precomp_ref10 *);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e)); \
            return -1; \
        } \
    } while (0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e)); \
        } \
    } while (0)

// Clamp n to nearest power of two in [lo, hi]
static int clamp_pow2(int n, int lo, int hi)
{
    if (n < lo) n = lo;
    if (n > hi) n = hi;
    int p = 1;
    while (p < n) p <<= 1;
    // round to nearest (either p/2 or p)
    if (p > lo && (p - n) > (n - p/2)) p >>= 1;
    if (p < lo) p = lo;
    if (p > hi) p = hi;
    return p;
}

extern "C" int gpu_autoconf(struct gpu_config *cfg, int quiet)
{
    // Check for at least one CUDA device before doing anything else.
    // Return -1 silently so main() falls back to CPU without a scary error.
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0)
        return -1;

    // Enable mapped pinned memory (needed for zero-copy result ring buffer)
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (!quiet) {
        fprintf(stderr, "GPU: %s (sm_%d%d, %d SMs, %zu MB)\n",
                prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                (size_t)(prop.totalGlobalMem >> 20));
    }

    // 256 threads/block on sm_8x+; 128 on older — stay warp-aligned
    int tpb = (prop.major >= 8) ? 256 : 128;
    tpb = (tpb / prop.warpSize) * prop.warpSize;
    if (tpb < 32) tpb = 32;

    // 4 persistent blocks per SM (2 on Hopper where blocks are bigger)
    int blocks_per_sm = (prop.major >= 9) ? 2 : 4;
    int num_blocks = prop.multiProcessorCount * blocks_per_sm;
    int total_threads = num_blocks * tpb;

    // 60% of VRAM budget for batch buffers
    size_t vram_budget = (size_t)((double)prop.totalGlobalMem * 0.60);

    // Each thread×slot needs 40 int32 (30 for xyz + 10 for tmp prefix product)
    size_t bytes_per_thread_per_slot = 40 * sizeof(int32_t);
    size_t max_batchnum = vram_budget / ((size_t)total_threads * bytes_per_thread_per_slot);
    if (max_batchnum < 1) max_batchnum = 1;

    int batchnum = clamp_pow2((int)max_batchnum, 64, 1024);

    cfg->num_blocks       = num_blocks;
    cfg->threads_per_block = tpb;
    cfg->batchnum         = batchnum;
    cfg->vram_budget      = vram_budget;

    if (!quiet) {
        fprintf(stderr, "GPU config: %d blocks × %d threads = %d, batchnum=%d\n",
                num_blocks, tpb, total_threads, batchnum);
    }

    return 0;
}

extern "C" int gpu_init(struct gpu_state *st, int quiet)
{
    (void)quiet;
    struct gpu_config *cfg = &st->cfg;
    int total_threads = cfg->num_blocks * cfg->threads_per_block;
    int B = cfg->batchnum;

    // Copy 8*B into __constant__ memory. The kernel steps by +8*B per inner
    // loop iteration; the sk offset adds 8 per step so clamped scalars stay
    // clamped (sk[0]&7==0 is preserved). ge_precomp and ge_precomp_cuda are
    // layout-identical (same int32_t[10] fields in the same order).
    ge_precomp_ref10 eightpt_ref;
    CRYPTO_NAMESPACE(ge_get_eightpoint_precomp)(&eightpt_ref);
    ge_precomp_cuda eightpt_host;
    memcpy(eightpt_host.yplusx,  eightpt_ref.yplusx,  10 * sizeof(int32_t));
    memcpy(eightpt_host.yminusx, eightpt_ref.yminusx, 10 * sizeof(int32_t));
    memcpy(eightpt_host.xy2d,    eightpt_ref.xy2d,    10 * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpyToSymbol(cuda_ge_eightpoint, &eightpt_host, sizeof(ge_precomp_cuda)));

    // Batch buffers: xyz [B×30×stride], tmp [B×10×stride]
    size_t xyz_bytes = (size_t)B * 30 * total_threads * sizeof(int32_t);
    size_t tmp_bytes = (size_t)B * 10 * total_threads * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&st->d_batch_xyz, xyz_bytes));
    CUDA_CHECK(cudaMalloc(&st->d_tmp,       tmp_bytes));

    // Starting points: ge_p3 = 40 int32 per thread; sk = 64 bytes per thread
    size_t pts_bytes = (size_t)total_threads * 40 * sizeof(int32_t);
    size_t sk_bytes  = (size_t)total_threads * 64;
    CUDA_CHECK(cudaMalloc(&st->d_start_pts, pts_bytes));
    CUDA_CHECK(cudaMalloc(&st->d_start_sk,  sk_bytes));

    int32_t *h_pts = (int32_t *)malloc(pts_bytes);
    uint8_t *h_sk  = (uint8_t *)malloc(sk_bytes);
    if (!h_pts || !h_sk) { free(h_pts); free(h_sk); return -1; }

    for (int t = 0; t < total_threads; t++) {
        uint8_t *sk_t = h_sk + t * 64;

        // Generate 64 random bytes; use first 32 as the ed25519 scalar.
        // Clamping replaces the need for SHA-512 expansion: the result is a
        // valid group element and a well-distributed starting point.
        randombytes(sk_t, 64);
        sk_t[0]  &= 248;  // clear bottom 3 bits
        sk_t[31] &= 63;   // clear top 2 bits
        sk_t[31] |= 64;   // set bit 254

        ge_p3_ref10 pt;
        CRYPTO_NAMESPACE(ge_scalarmult_base)(&pt, sk_t);

        // Flatten ge_p3 {X,Y,Z,T} into h_pts as [f*10+li] per thread
        const int32_t *coords[4] = {pt.X, pt.Y, pt.Z, pt.T};
        for (int f = 0; f < 4; f++)
            for (int li = 0; li < 10; li++)
                h_pts[t * 40 + f * 10 + li] = coords[f][li];
    }

    CUDA_CHECK(cudaMemcpy(st->d_start_pts, h_pts, pts_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_start_sk,  h_sk,  sk_bytes,  cudaMemcpyHostToDevice));
    sodium_memzero(h_sk, sk_bytes);
    free(h_pts);
    free(h_sk);

    // Result ring buffer — mapped pinned memory shared between GPU and CPU.
    // GPU writes results + done flags directly; CPU reads without cudaMemcpy.
    // Requires cudaDeviceMapHost (enabled by gpu_autoconf via cudaSetDeviceFlags).
    st->result_ring_size = 4096;
    size_t ring_bytes  = (size_t)st->result_ring_size * sizeof(struct gpu_result);
    size_t done_bytes  = (size_t)st->result_ring_size * sizeof(int32_t);

    CUDA_CHECK(cudaHostAlloc(&st->h_results, ring_bytes,  cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&st->h_done, done_bytes, cudaHostAllocMapped));
    memset((void *)st->h_done, 0, done_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&st->d_results, st->h_results, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&st->d_done,    (void*)st->h_done,  0));

    CUDA_CHECK(cudaHostAlloc((void**)&st->h_endwork, sizeof(int), cudaHostAllocMapped));
    *st->h_endwork = 0;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&st->d_endwork, (void*)st->h_endwork, 0));

    CUDA_CHECK(cudaHostAlloc((void**)&st->h_numcalc, sizeof(unsigned long long), cudaHostAllocMapped));
    *st->h_numcalc = 0;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&st->d_numcalc, (void*)st->h_numcalc, 0));

    CUDA_CHECK(cudaMalloc(&st->d_result_head, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(st->d_result_head, 0, sizeof(int32_t)));

    return 0;
}

extern "C" void gpu_cleanup(struct gpu_state *st)
{
    CUDA_CHECK_VOID(cudaFree(st->d_batch_xyz));
    CUDA_CHECK_VOID(cudaFree(st->d_tmp));
    CUDA_CHECK_VOID(cudaFree(st->d_start_pts));
    CUDA_CHECK_VOID(cudaFree(st->d_start_sk));
    if (st->h_results)
        sodium_memzero(st->h_results,
                       (size_t)st->result_ring_size * sizeof(struct gpu_result));
    CUDA_CHECK_VOID(cudaFreeHost(st->h_results));  // mapped: free host ptr only
    CUDA_CHECK_VOID(cudaFreeHost((void *)st->h_done));
    CUDA_CHECK_VOID(cudaFree(st->d_result_head));
    CUDA_CHECK_VOID(cudaFreeHost((void *)st->h_endwork));
    CUDA_CHECK_VOID(cudaFreeHost((void *)st->h_numcalc));
    memset(st, 0, sizeof(*st));
}
