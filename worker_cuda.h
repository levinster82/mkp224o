#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runtime GPU configuration computed by gpu_autoconf()
struct gpu_config {
    int num_blocks;
    int threads_per_block;
    int batchnum;          // BATCHNUM_GPU, must be in {64,128,256,512,1024}
    size_t vram_budget;    // bytes allocated for batch buffers
};

// Populated by gpu_init(); used by worker and cleanup
struct gpu_state {
    struct gpu_config cfg;

    // Device buffers
    int32_t *d_batch_xyz;  // [batchnum * 20 * total_threads] int32 — Y,Z per slot
                           // (X is not stored: it only affects the pubkey sign
                           // bit, which never lands in a vanity prefix, so the
                           // GPU filters on the sign-less key and the CPU drain
                           // thread recomputes the exact key on a hit.)
    int32_t *d_tmp;        // [batchnum * 10 * total_threads] int32 — prefix products

    // Starting points: one ge_p3 (40 int32) + secret key (64 bytes) per thread
    int32_t *d_start_pts;  // [total_threads * 40] int32  (ge_p3: X,Y,Z,T each 10 limbs)
    uint8_t *d_start_sk;   // [total_threads * 64] uint8

    // Result ring buffer — mapped pinned memory so GPU writes are directly visible
    // to CPU after __threadfence_system(), with no cudaMemcpy in the hot path.
    int32_t *d_result_head;       // atomic total write count (device allocation)
    struct gpu_result *h_results; // pinned host pointer to result slots
    struct gpu_result *d_results; // device pointer to same mapped pinned allocation
    volatile int32_t *h_done;     // pinned host pointer to per-slot done flags
    int32_t          *d_done;     // device pointer to same mapped pinned allocation
    int      result_ring_size;

    // Stop flag — mapped pinned so CPU can write directly without CUDA stream ops.
    // Using cudaMemcpyToSymbol() would serialize behind the persistent kernel.
    volatile int *h_endwork;     // CPU writes 1 here to stop the kernel
    int          *d_endwork;     // GPU reads from this device pointer

    // Candidate counter — mapped pinned; one block-level atomicAdd per outer loop.
    volatile unsigned long long *h_numcalc; // CPU reads total candidates tested
    unsigned long long          *d_numcalc; // GPU writes via atomicAdd
};

// One matched candidate returned to CPU drain thread. The GPU emits only the
// secret scalar; the drain thread recomputes the exact public key (correct
// sign), checksum and onion address from it. PUBONION_LEN is the size of that
// CPU-side buffer: prefix(32) + pk(32) + checksum(2) + version(1).
#define GPU_RESULT_PUBONION_LEN (32 + 32 + 3)
#define GPU_RESULT_SECRET_LEN   (32 + 64)        // prefix(32) + sk(64)

struct gpu_result {
    uint8_t secret[GPU_RESULT_SECRET_LEN];
};

// Number of available CUDA devices (0 if none / on error).
int gpu_device_count(void);

// Query GPU `device_idx` and compute its configuration. Also binds the calling
// thread to that device (cudaSetDevice) and enables mapped pinned memory.
int gpu_autoconf(int device_idx, struct gpu_config *cfg, int quiet);

// Allocate device memory, copy ge_eightpoint to __constant__, fill start_pts
int gpu_init(struct gpu_state *st, int quiet);

// Free all device allocations
void gpu_cleanup(struct gpu_state *st);

// Launch GPU kernel and block until endwork=1.
// Results are drained on a CPU thread and forwarded to onionready().
// reportdelay: statistics print interval in microseconds (0 = disabled).
// realtimestats: 1 = rolling window (reset each period), 0 = cumulative.
// Returns 0 on success, -1 on CUDA error.
int gpu_worker_launch(int quiet, u64 reportdelay, int realtimestats);

#ifdef __cplusplus
}
#endif
