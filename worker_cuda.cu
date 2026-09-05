// GPU worker kernel for mkp224o — ed25519 onion vanity address generator.
// Each thread runs an independent copy of the worker_batch loop, using
// Montgomery batch inversion to amortize the dominant fe_invert cost.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>

extern "C" {
#include "types.h"
#include "common.h"
#include "vec.h"
#include "worker.h"
#ifndef _WIN32
#include "ioutil.h"
#include "base32.h"
#include "keccak.h"
#include "yaml.h"
#endif
#include "filters.h"
}
#include "worker_cuda.h"
#include "statline.h"

// ── CUDA device headers ────────────────────────────────────────────────────
#include "ed25519/cuda/fe_cuda.cuh"
#include "ed25519/cuda/ge_cuda.cuh"
#include "keccak_cuda.cuh"

// Definition of the shared constant-memory eightpoint (extern-declared in ge_cuda.cuh)
__constant__ ge_precomp_cuda cuda_ge_eightpoint;

// Host-side ref10 ed25519, used by the CPU drain thread to recompute the exact
// public key (with correct x sign) from an emitted scalar. ge_p3_ref10 is
// layout-identical to ref10 ge_p3 (four int32[10] fields). Namespaced to match
// however the ed25519 impl was compiled (the ref10 objects are always linked
// for the GPU build).
typedef int32_t fe_ref10[10];
typedef struct { fe_ref10 X, Y, Z, T; } ge_p3_ref10;
extern "C" {
void CRYPTO_NAMESPACE(ge_scalarmult_base)(ge_p3_ref10 *, const unsigned char *);
void CRYPTO_NAMESPACE(ge_p3_tobytes)(unsigned char *, const ge_p3_ref10 *);
}

// ── Filter state in constant memory ───────────────────────────────────────
// We support INTFILTER (fast uint64) and BINFILTER (byte array) on GPU.
// PCRE2FILTER is CPU-only.

#define GPU_MAX_FILTERS 256
#define GPU_BINFILTER_LEN 32

struct gpu_filter_entry {
    uint64_t ifast;           // first 8 bytes as uint64 (for fast pre-filter)
    uint64_t imask;           // mask for ifast comparison
    uint8_t  f[GPU_BINFILTER_LEN]; // full binary filter
    uint8_t  fmask[GPU_BINFILTER_LEN]; // mask per byte
    int      flen;            // number of full bytes to compare
    uint8_t  final_mask;      // mask for byte at flen
    int      filter_bits;     // number of base32 characters this filter covers
};

__constant__ struct gpu_filter_entry cuda_filter_table[GPU_MAX_FILTERS];
__constant__ int cuda_filter_count;
// skprefix is prepended to the emitted secret scalar by the kernel.
__constant__ uint8_t cuda_skprefix[32];

// ── Kernel argument block ──────────────────────────────────────────────────
struct kernel_args {
    int32_t *batch_xyz;        // [B * 20 * stride] — Y,Z per batch slot (no X)
    int32_t *tmp;              // [B * 10 * stride] — scratch for batchinvert
    int32_t *start_pts;        // [stride * 40]     — starting ge_p3 per thread
    uint8_t *start_sk;         // [stride * 64]     — base secret key per thread
    int32_t *result_head;      // atomic total write counter (device allocation)
    struct gpu_result *results; // mapped pinned result slots (device pointer)
    volatile int32_t *done;    // mapped pinned per-slot done flags (device pointer)
    volatile int          *endwork_flag; // mapped pinned stop flag (CPU writes 1 to stop)
    unsigned long long    *numcalc;      // mapped pinned candidate counter (block-level atomicAdd)
    int result_ring_size;
    int stride;                // total_threads = gridDim.x * blockDim.x
    int numwords;
    unsigned long long max_iters; // 0 = run forever; >0 = stop after this many
                                  // candidates/thread (profiling only, set via
                                  // MKP_PROFILE_ITERS so ncu sees a finite kernel)
};

// ── Secret key multi-byte addition ────────────────────────────────────────
// Adds a 64-bit scalar to the little-endian sk[0..31] array (same as addsztoscalar32)
static __device__ __forceinline__ void
addsztoscalar32_cuda(uint8_t *sk, unsigned long long v)
{
    uint32_t c = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        c = (uint32_t)sk[i] + (uint32_t)(v & 0xFF) + c;
        sk[i] = (uint8_t)(c & 0xFF);
        c >>= 8;
        v >>= 8;
    }
}

// ── Public key bit-shift (for multi-word pattern chaining) ────────────────
// Equivalent to worker.c shiftpk(): shifts src left by sbits bits into dst.
// Safe for in-place use (dst == src) because reads are always ahead of writes.
static __device__ __forceinline__ void
shiftpk_cuda(uint8_t *dst, const uint8_t *src, int sbits)
{
    int sbytes = sbits / 8;
    int srem   = sbits % 8;
    int i;
    for (i = 0; i + sbytes < 32; i++) {
        uint8_t hi = src[i + sbytes];
        uint8_t lo = (i + sbytes + 1 < 32) ? src[i + sbytes + 1] : 0;
        dst[i] = srem ? (uint8_t)((hi << srem) | (lo >> (8 - srem))) : hi;
    }
    for (; i < 32; i++)
        dst[i] = 0;
}

// ── GPU filter check ───────────────────────────────────────────────────────
// Returns filter index (0-based) if match found, else -1.
static __device__ __forceinline__ int
gpu_check_filter(const uint8_t *pk)
{
    int n = cuda_filter_count;
#ifdef INTFILTER
    // Fast uint64 comparison (INTFILTER build)
    uint64_t pkval;
    // Byte-swapping not needed; IFT is read as native endian same as CPU
    pkval  = (uint64_t)pk[0]       | ((uint64_t)pk[1]<<8)
           | ((uint64_t)pk[2]<<16) | ((uint64_t)pk[3]<<24)
           | ((uint64_t)pk[4]<<32) | ((uint64_t)pk[5]<<40)
           | ((uint64_t)pk[6]<<48) | ((uint64_t)pk[7]<<56);
    for (int i = 0; i < n; i++) {
        if ((pkval & cuda_filter_table[i].imask) == cuda_filter_table[i].ifast)
            return i;
    }
#else
    // BINFILTER: byte-by-byte comparison
    for (int i = 0; i < n; i++) {
        const struct gpu_filter_entry *fe = &cuda_filter_table[i];
        int match = 1;
        for (int j = 0; j < fe->flen && match; j++) {
            if (pk[j] != fe->f[j]) match = 0;
        }
        if (match && ((pk[fe->flen] & fe->final_mask) == fe->f[fe->flen]))
            return i;
    }
#endif
    return -1;
}

// ── Core persistent kernel ─────────────────────────────────────────────────
template<int B>
__global__ void worker_cuda_kernel(struct kernel_args args)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = args.stride;

    // Load starting ge_p3 point from device memory
    // Layout in start_pts: [X(10), Y(10), Z(10), T(10)] per thread
    ge_p3_cuda ge_pub;
    #pragma unroll
    for (int li = 0; li < 10; li++) {
        ge_pub.X[li] = args.start_pts[tid * 40 + 0*10 + li];
        ge_pub.Y[li] = args.start_pts[tid * 40 + 1*10 + li];
        ge_pub.Z[li] = args.start_pts[tid * 40 + 2*10 + li];
        ge_pub.T[li] = args.start_pts[tid * 40 + 3*10 + li];
    }

    // Load base secret key
    uint8_t base_sk[64];
    #pragma unroll
    for (int i = 0; i < 64; i++)
        base_sk[i] = args.start_sk[tid * 64 + i];

    // Prefix for checksum computation: ".onion checksum" (15 bytes) + pk(32) + 0x03
    // hashsrc is built at match time

    unsigned long long counter = 0;

    while (!*args.endwork_flag) {
        // ── Inner loop: accumulate B ge_p3 candidates ──────────────────
        for (int b = 0; b < B; b++) {
            // Store Y, Z in batch buffer (X not needed — only affects the sign
            // bit; T not needed for tobytes). Slot layout: Y=0..9, Z=10..19.
            #pragma unroll
            for (int li = 0; li < 10; li++) {
                args.batch_xyz[(b * 20 + 0*10 + li) * stride + tid] = ge_pub.Y[li];
                args.batch_xyz[(b * 20 + 1*10 + li) * stride + tid] = ge_pub.Z[li];
            }
            // Advance: ge_pub += ge_eightpoint
            ge_add_eightpoint_cuda(&ge_pub);
        }

        // ── Batch inversion forward pass ───────────────────────────────
        // Prefix products into tmp; acc = (product of all Z)^-1.
        // Z field at slot b: offset (b*20 + 10 + li)*stride + tid.
        fe_cuda acc;
        fe_batch_prefix_invert_cuda(args.batch_xyz, args.tmp, acc, B, 20, 10, stride, tid);

        // ── Fused backward pass: derive each Z^-1 and check inline ──────
        // Iterates b high→low. acc is advanced (acc *= Z[b]) every iteration
        // before any early-out, so the inverted Z is consumed in-register and
        // never written back to global memory.
        for (int b = B - 1; b >= 0; b--) {
            fe_cuda Y, Z, prefix, Zinv;
            #pragma unroll
            for (int li = 0; li < 10; li++) {
                Z[li]      = args.batch_xyz[(b * 20 + 1*10 + li) * stride + tid];
                prefix[li] = args.tmp[(b * 10 + li) * stride + tid];
            }
            fe_mul_cuda(Zinv, acc, prefix); // Z[b]^-1 = acc * prefix[b]
            fe_mul_cuda(acc, acc, Z);       // advance acc for the next slot
            #pragma unroll
            for (int li = 0; li < 10; li++)
                Y[li] = args.batch_xyz[(b * 20 + 0*10 + li) * stride + tid];

            // Compute the sign-less public key y = Y*Z_inv. The x-sign bit is
            // left clear; it never affects a vanity prefix and the CPU drain
            // thread recomputes the exact key (with sign) on a match.
            uint8_t pk[32];
            ge_y_tobytes_batched_cuda(pk, Y, Zinv);

            // ── Filter check ─────────────────────────────────────────────
            int fi = gpu_check_filter(pk);
            if (fi < 0)
                continue;

            // ── Multi-word check (numwords > 1) ──────────────────────────
            // For each additional word, shift the pk left by the matched
            // filter's bit-width and check again — same as CPU shiftpk loop.
            if (args.numwords > 1) {
                uint8_t wpk[32];
                const uint8_t *src = pk;
                int fi2 = fi;
                bool multiword_ok = true;
                for (int w = 1; w < args.numwords; w++) {
                    shiftpk_cuda(wpk, src, cuda_filter_table[fi2].filter_bits);
                    fi2 = gpu_check_filter(wpk);
                    if (fi2 < 0) { multiword_ok = false; break; }
                    src = wpk; // in-place on next iteration
                }
                if (!multiword_ok) continue;
            }

            // ── Found a candidate ─────────────────────────────────────────
            // Emit only the secret scalar. The CPU drain thread recomputes the
            // exact public key (correct x sign), checksum and onion address —
            // cheap because matches are astronomically rare. This keeps SHA3
            // and key formatting out of the hot kernel (fewer registers).

            // Build secret: skprefix + (base_sk + counter_offset)
            uint8_t secret[GPU_RESULT_SECRET_LEN];
            #pragma unroll
            for (int i = 0; i < 32; i++)
                secret[i] = cuda_skprefix[i];
            #pragma unroll
            for (int i = 0; i < 64; i++)
                secret[32 + i] = base_sk[i];
            // Step is 8*B: add 8 per inner-loop step so the scalar stays a
            // multiple of 8 (clamping condition sk[0]&7==0 is preserved).
            unsigned long long offset = 8ULL * (counter + (unsigned long long)b);
            addsztoscalar32_cuda(&secret[32], offset);

            // Sanity check (matches CPU's check)
            uint8_t s0 = secret[32]; // sk[0]
            uint8_t s31 = secret[63]; // sk[31]
            if ((s0 & 248) != s0 || ((s31 & 63) | 64) != s31)
                continue; // bad scalar after addition, skip

            // Reserve a ring slot, then spin until the drain thread has
            // consumed it from the previous cycle (backpressure).
            // Also exit on *args.endwork_flag to avoid deadlock if a stop signal
            // arrives while the ring is full.
            int slot = atomicAdd(args.result_head, 1) % args.result_ring_size;
            while (args.done[slot] && !*args.endwork_flag) { /* spin */ }
            if (*args.endwork_flag) return;
            struct gpu_result *res = &args.results[slot];
            #pragma unroll
            for (int i = 0; i < GPU_RESULT_SECRET_LEN; i++)
                res->secret[i] = secret[i];
            __threadfence_system(); // flush writes to mapped pinned memory
            args.done[slot] = 1;   // signal CPU that this slot is ready
        }

        // One atomic per block per outer loop — avoids per-thread contention.
        if (threadIdx.x == 0)
            atomicAdd(args.numcalc, (unsigned long long)blockDim.x * B);

        counter += (unsigned long long)B;

        // Profiling cap: when MKP_PROFILE_ITERS is set the kernel becomes a
        // finite workload so Nsight Compute can replay it. 0 = normal (forever).
        if (args.max_iters && counter >= args.max_iters)
            break;
    }
}

// Explicit instantiations for the five supported BATCHNUM values
template __global__ void worker_cuda_kernel< 64>(struct kernel_args);
template __global__ void worker_cuda_kernel<128>(struct kernel_args);
template __global__ void worker_cuda_kernel<256>(struct kernel_args);
template __global__ void worker_cuda_kernel<512>(struct kernel_args);
template __global__ void worker_cuda_kernel<1024>(struct kernel_args);

// ── Multi-GPU shared state ────────────────────────────────────────────────
// Array of per-device gpu_states, populated by gpu_worker_launch(). The
// primary drain thread reads all devices' h_numcalc to report an aggregated
// throughput number; every other CUDA-side call always operates on its own
// gpu_state via drain_args / gpu_worker_ctx.
static struct gpu_state *g_gpu_states = NULL;
static int               g_gpu_state_count = 0;

// ── CPU-side drain thread ──────────────────────────────────────────────────
// Polls the result ring buffer and calls onionready() for each found key.

struct drain_args {
    struct gpu_state *st;
    int quiet;
    int is_primary;      // 1 = this thread also reports aggregated stats
#ifdef STATISTICS
    u64 reportdelay;
    int realtimestats;
#endif
};

static void *drain_thread(void *arg)
{
    struct drain_args *da = (struct drain_args *)arg;
    struct gpu_state  *st = da->st;
    int ring_size = st->result_ring_size;
    int read_idx  = 0;

    struct timespec ts = { 0, 1000000 }; // 1ms poll interval

#ifdef STATISTICS
    u64 istarttime, inowtime, ireporttime = 0, elapsedoffset = 0;
    u64 sumcalc = 0;
    u64 last_numcalc = 0;
    u64 local_success = 0; // independent of keysgenerated (only incremented under -n)
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        istarttime = (u64)now.tv_sec * 1000000ULL + (u64)now.tv_nsec / 1000;
    }
#endif

    while (!endwork) {
        // Each slot sets done[slot]=1 (after __threadfence_system) when ready.
        // No cudaMemcpy needed: h_results and h_done are mapped pinned memory.
        while (st->h_done[read_idx]) {
            struct gpu_result *hres = &st->h_results[read_idx];

            // The GPU emitted only the secret scalar. Recompute the exact
            // public key (with correct x sign) from it, then build the formatted
            // pubonion: pkprefix(32) + pk(32) + checksum(2) + version(1).
            static const char checksumstr[] = ".onion checksum"; // 15 bytes
            const uint8_t *sk = hres->secret + 32; // expanded sk; scalar in [0..31]
            uint8_t pubonion[GPU_RESULT_PUBONION_LEN];
            memcpy(pubonion, pkprefix, 32);

            ge_p3_ref10 pt;
            CRYPTO_NAMESPACE(ge_scalarmult_base)(&pt, sk);
            CRYPTO_NAMESPACE(ge_p3_tobytes)(&pubonion[32], &pt);

            uint8_t hashsrc[15 + 32 + 1];
            memcpy(hashsrc, checksumstr, 15);
            memcpy(&hashsrc[15], &pubonion[32], 32);
            hashsrc[47] = 0x03; // version
            uint8_t chk[FIPS202_SHA3_256_LEN];
            FIPS202_SHA3_256(hashsrc, sizeof(hashsrc), chk);
            pubonion[64] = chk[0];
            pubonion[65] = chk[1];
            pubonion[66] = 0x03; // version

            char *sname = makesname();
            if (sname) {
                strcpy(base32_to(&sname[direndpos], pubonion + 32, 35), ".onion");
                onionready(sname, hres->secret, pubonion, 0);
                free(sname);
            }
#ifdef STATISTICS
            local_success++;
#endif
            // Reset done flag so this slot can be reused
            st->h_done[read_idx] = 0;
            read_idx = (read_idx + 1) % ring_size;
        }

        nanosleep(&ts, 0);

#ifdef STATISTICS
        // Only the primary drain thread reports stats (aggregated across all
        // devices), otherwise multi-GPU would print one line per device.
        if (da->is_primary && da->reportdelay) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            inowtime = (u64)now.tv_sec * 1000000ULL + (u64)now.tv_nsec / 1000;

            u64 cur_calc = 0;
            for (int i = 0; i < g_gpu_state_count; i++) {
                if (g_gpu_states[i].h_numcalc)
                    cur_calc += (u64)*g_gpu_states[i].h_numcalc;
            }
            if (!ireporttime) {
                // First tick: align to whatever kernels have already produced
                // during their startup so the initial reporting window covers
                // real work instead of blowing up on a tiny elapsed window.
                last_numcalc = cur_calc;
                istarttime   = inowtime;
            } else {
                sumcalc     += cur_calc - last_numcalc;
                last_numcalc = cur_calc;
            }

            if (!ireporttime || (i64)(inowtime - ireporttime) >= (i64)da->reportdelay) {
                if (ireporttime)
                    ireporttime += da->reportdelay;
                else
                    ireporttime = inowtime;
                if (!ireporttime) ireporttime = 1;

                // Use window duration for rates; total elapsed for display.
                u64 window  = inowtime - istarttime;
                u64 elapsed = window + elapsedoffset;
                double calcpersec = window ? 1000000.0 * (double)sumcalc / window : 0.0;
                print_stats_line(stderr, calcpersec, elapsed, (u64)keysgenerated);

                if (da->realtimestats) {
                    sumcalc    = 0;
                    local_success = 0;
                    elapsedoffset += window;
                    istarttime = inowtime;
                }
            }
        }
#endif
    }

#ifdef STATISTICS
    if (da->is_primary) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        inowtime = (u64)now.tv_sec * 1000000ULL + (u64)now.tv_nsec / 1000;
        u64 elapsed = inowtime - istarttime + elapsedoffset;
        u64 total_calc = 0;
        for (int i = 0; i < g_gpu_state_count; i++) {
            if (g_gpu_states[i].h_numcalc)
                total_calc += (u64)*g_gpu_states[i].h_numcalc;
        }
        double calcpersec = elapsed ? 1000000.0 * (double)total_calc / elapsed : 0.0;
        print_stats_line(stderr, calcpersec, elapsed, (u64)keysgenerated);
    }
#endif

    // Write stop flag directly to mapped pinned memory — no CUDA stream call needed.
    // cudaMemcpyToSymbol() would serialize behind the persistent kernel (deadlock).
    *st->h_endwork = 1;

    return 0;
}

// ── Main GPU launch function ───────────────────────────────────────────────
// Called from main.c instead of the pthread loop when CUDA is enabled.

#define CUDA_CHECK_LAUNCH(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e)); \
            return -1; \
        } \
    } while (0)

// Serialize CPU filter table to GPU constant memory
static int upload_filters(void)
{
    struct gpu_filter_entry htable[GPU_MAX_FILTERS];
    int count = 0;

#ifdef INTFILTER
    // In OMITMASK mode all filters share a single global mask; in normal mode
    // each filter carries its own mask in .m.
    uint64_t global_imask = 0;
#ifdef OMITMASK
    memcpy(&global_imask, &ifiltermask, sizeof(global_imask));
#endif
    for (size_t i = 0; i < VEC_LENGTH(filters) && count < GPU_MAX_FILTERS; i++, count++) {
        struct gpu_filter_entry *e = &htable[count];
        memset(e, 0, sizeof(*e));
        uint64_t f, m;
        memcpy(&f, &VEC_BUF(filters, i).f, sizeof(uint64_t));
#ifdef OMITMASK
        m = global_imask;
#else
        memcpy(&m, &VEC_BUF(filters, i).m, sizeof(uint64_t));
#endif
        e->ifast = f;
        e->imask = m;
        // filter_bits = number of set bits in the prefix mask (= N_chars * 5)
        e->filter_bits = __builtin_popcountll(m);
    }
#elif defined(BINFILTER)
    for (size_t i = 0; i < VEC_LENGTH(filters) && count < GPU_MAX_FILTERS; i++, count++) {
        struct gpu_filter_entry *e = &htable[count];
        memset(e, 0, sizeof(*e));
        const struct binfilter *bf = &VEC_BUF(filters, i);
        int flen = (int)bf->len;
        if (flen > GPU_BINFILTER_LEN - 1) flen = GPU_BINFILTER_LEN - 1;
        memcpy(e->f, bf->f, flen + 1);
        e->flen = flen;
        e->final_mask = bf->mask;
        // Count set bits in partial-byte mask (leading 1-bits from MSB)
        int mbits = 0;
        uint8_t mv = bf->mask;
        while (mbits < 8 && (mv & 0x80)) { mbits++; mv = (uint8_t)(mv << 1); }
        e->filter_bits = flen * 8 + mbits;
        // Fast uint64 pre-filter for the first 8 bytes
        uint64_t fast = 0, fast_mask = 0;
        for (int j = 0; j < 8 && j <= flen; j++) {
            fast      |= (uint64_t)bf->f[j] << (j * 8);
            fast_mask |= (uint64_t)0xFF       << (j * 8);
        }
        if (flen < 8) fast_mask &= ((uint64_t)1 << (flen * 8)) - 1;
        e->ifast = fast;
        e->imask = fast_mask;
    }
#endif

    CUDA_CHECK_LAUNCH(cudaMemcpyToSymbol(cuda_filter_table, htable,
                                          count * sizeof(struct gpu_filter_entry)));
    CUDA_CHECK_LAUNCH(cudaMemcpyToSymbol(cuda_filter_count, &count, sizeof(int)));
    return 0;
}

// Per-device worker context passed to gpu_worker_thread().
struct gpu_worker_ctx {
    int device_idx;
    int is_primary;
    int quiet;
#ifdef STATISTICS
    u64 reportdelay;
    int realtimestats;
#endif
    int result;   // 0 on success, -1 on error
};

// Per-device worker thread: binds to its device, initialises it, launches
// the persistent kernel, drains results, cleans up. Every CUDA call in this
// thread implicitly uses the device set by cudaSetDevice(device_idx).
static void *gpu_worker_thread(void *arg)
{
    struct gpu_worker_ctx *ctx = (struct gpu_worker_ctx *)arg;
    struct gpu_state *st = &g_gpu_states[ctx->device_idx];
    memset(st, 0, sizeof(*st));
    ctx->result = -1;

    // gpu_autoconf() calls cudaSetDevice internally, but we also do it here
    // so a mid-init failure that returns early still leaves the thread
    // pointing at the intended device (helps if we later free per-device
    // allocations from a shared teardown path).
    if (cudaSetDevice(ctx->device_idx) != cudaSuccess) return NULL;

    if (gpu_autoconf(ctx->device_idx, &st->cfg, ctx->quiet) < 0) return NULL;

    if (gpu_init(st, ctx->quiet) < 0) return NULL;

    // Constant-memory symbols (filter table, skprefix) live in the current
    // device's context — must be uploaded per device, after cudaSetDevice.
    if (upload_filters() < 0) { gpu_cleanup(st); return NULL; }
    cudaError_t e = cudaMemcpyToSymbol(cuda_skprefix, skprefix, 32);
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error uploading skprefix on device %d: %s\n",
                ctx->device_idx, cudaGetErrorString(e));
        gpu_cleanup(st); return NULL;
    }

    // Build kernel_args
    struct kernel_args kargs;
    memset(&kargs, 0, sizeof(kargs));
    kargs.batch_xyz        = st->d_batch_xyz;
    kargs.tmp              = st->d_tmp;
    kargs.start_pts        = st->d_start_pts;
    kargs.start_sk         = st->d_start_sk;
    kargs.result_head      = st->d_result_head;
    kargs.results          = st->d_results;
    kargs.done             = st->d_done;
    kargs.result_ring_size = st->result_ring_size;
    kargs.stride           = st->cfg.num_blocks * st->cfg.threads_per_block;
    kargs.numwords         = numwords;
    kargs.endwork_flag     = st->d_endwork;
    kargs.numcalc          = st->d_numcalc;

    // Profiling hook: MKP_PROFILE_ITERS=N makes the persistent kernel exit
    // after ~N candidates/thread so tools that replay/await the kernel
    // (e.g. Nsight Compute) get a finite workload. Unset = run forever.
    kargs.max_iters = 0;
    {
        const char *pi = getenv("MKP_PROFILE_ITERS");
        if (pi && *pi) {
            kargs.max_iters = strtoull(pi, NULL, 10);
            if (!ctx->quiet && kargs.max_iters && ctx->is_primary)
                fprintf(stderr, "PROFILE: kernel will stop after %llu candidates/thread\n",
                        kargs.max_iters);
        }
    }

    // Start CPU drain thread
    struct drain_args da = { st, ctx->quiet, ctx->is_primary
#ifdef STATISTICS
        , ctx->reportdelay, ctx->realtimestats
#endif
    };
    pthread_t drain;
    if (pthread_create(&drain, NULL, drain_thread, &da) != 0) {
        fprintf(stderr, "failed to create GPU drain thread for device %d\n",
                ctx->device_idx);
        gpu_cleanup(st); return NULL;
    }

    // Launch kernel with correct BATCHNUM template
    int G = st->cfg.num_blocks, T = st->cfg.threads_per_block;
    switch (st->cfg.batchnum) {
        case   64: worker_cuda_kernel<  64><<<G, T>>>(kargs); break;
        case  128: worker_cuda_kernel< 128><<<G, T>>>(kargs); break;
        case  256: worker_cuda_kernel< 256><<<G, T>>>(kargs); break;
        case  512: worker_cuda_kernel< 512><<<G, T>>>(kargs); break;
        case 1024: worker_cuda_kernel<1024><<<G, T>>>(kargs); break;
        default:
            fprintf(stderr, "invalid batchnum %d on device %d\n",
                    st->cfg.batchnum, ctx->device_idx);
            endwork = 1;  // wake drain, which will exit
            pthread_join(drain, 0);
            gpu_cleanup(st);
            return NULL;
    }

    // Wait for kernel to finish (endwork=1 causes kernel to return)
    cudaDeviceSynchronize();

    // If a profiling cap made the kernel self-terminate, the drain thread is
    // still spinning on the global endwork flag — set it so it can exit.
    if (kargs.max_iters)
        endwork = 1;

    pthread_join(drain, 0);
    gpu_cleanup(st);
    ctx->result = 0;
    return NULL;
}

// Orchestrator: detects all CUDA devices, spawns one worker thread per
// device, and joins them. Every device runs the same persistent kernel with
// its own state; the primary device's drain thread reports aggregated stats.
// The global `endwork` (set by main.c's signal handler or shutdown) stops
// every device at once.
extern "C" int gpu_worker_launch(int quiet, u64 reportdelay, int realtimestats)
{
#ifndef STATISTICS
    (void)reportdelay; (void)realtimestats;
#endif

    int dev_count = gpu_device_count();
    if (dev_count <= 0) return -1;

    // MKP_GPUS=N caps the number of devices used (useful when one card is
    // busy with something else). Silently clamps to [1, dev_count].
    {
        const char *g = getenv("MKP_GPUS");
        if (g && *g) {
            int v = atoi(g);
            if (v >= 1 && v < dev_count) dev_count = v;
        }
    }

    if (!quiet)
        fprintf(stderr, "using GPU acceleration on %d device%s\n",
                dev_count, dev_count == 1 ? "" : "s");

    g_gpu_states = (struct gpu_state *)calloc((size_t)dev_count, sizeof(struct gpu_state));
    if (!g_gpu_states) { fprintf(stderr, "gpu_worker_launch: OOM\n"); return -1; }
    g_gpu_state_count = dev_count;

    struct gpu_worker_ctx *ctxs =
        (struct gpu_worker_ctx *)calloc((size_t)dev_count, sizeof(*ctxs));
    pthread_t *threads = (pthread_t *)calloc((size_t)dev_count, sizeof(pthread_t));
    if (!ctxs || !threads) {
        free(ctxs); free(threads);
        free(g_gpu_states); g_gpu_states = NULL; g_gpu_state_count = 0;
        return -1;
    }

    int started = 0;
    for (int i = 0; i < dev_count; i++) {
        ctxs[i].device_idx    = i;
        ctxs[i].is_primary    = (i == 0);
        ctxs[i].quiet         = quiet;
#ifdef STATISTICS
        ctxs[i].reportdelay   = reportdelay;
        ctxs[i].realtimestats = realtimestats;
#endif
        if (pthread_create(&threads[i], NULL, gpu_worker_thread, &ctxs[i]) != 0) {
            fprintf(stderr, "failed to create GPU worker thread for device %d\n", i);
            endwork = 1;  // signal already-started threads to stop
            break;
        }
        started++;
    }

    int overall = 0;
    for (int i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
        if (ctxs[i].result < 0) overall = -1;
    }
    if (started < dev_count) overall = -1;

    free(ctxs);
    free(threads);
    free(g_gpu_states);
    g_gpu_states = NULL;
    g_gpu_state_count = 0;
    return overall;
}
