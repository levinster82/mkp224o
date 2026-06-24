# GPU profiling kit (gpu-optimization branch)

Goal: **find the actual bottleneck of the CUDA search kernel before changing any
arithmetic.** We optimize what the profile says is limiting — not what we guess.

The kernel (`worker_cuda_kernel` in `worker_cuda.cu`) is a *persistent* kernel:
one launch that loops `while (!*endwork_flag)` until the CPU stops it. That
matters for tooling:

- **Nsight Systems (`nsys`)** samples a running kernel — works on the normal
  binary, no special build. Use it for the quick "compute- vs memory- vs
  latency-bound + occupancy" read.
- **Nsight Compute (`ncu`)** *replays* a kernel and waits for it to finish, so a
  forever-kernel hangs it. Set `MKP_PROFILE_ITERS=N` to make the kernel exit
  after ~N candidates/thread (and the app exit cleanly). Use it for the detailed
  per-section analysis (registers, occupancy limiter, warp stalls, pipe mix).

Both tools should run against a **hard prefix that won't be found** during the
short profiling window (e.g. `b32profiling`). That keeps the rare match/result
path out of the profile so we measure the real steady-state hot loop.

---

## Phase 1 — nsys (no rebuild needed, ~30s)

```bash
nsys profile \
  --gpu-metrics-devices=all \
  --duration=10 \
  --force-overwrite=true \
  -o mkp_nsys \
  ./mkp224o -S 5 b32profiling
```

> Older nsys used `--gpu-metrics-device` (singular); recent versions want
> `--gpu-metrics-devices`. If you hit `ERR_NVGPUCTRPERM` / "Insufficient
> privilege", see the permissions note at the bottom.

Then summarize:

```bash
nsys stats --report gpukernsum,gpummasum mkp_nsys.nsys-rep
```

Read back:
- **SM (compute) throughput %** vs **Memory throughput %** (from GPU metrics).
- **Achieved occupancy** (% of theoretical).
- Whether one is clearly pinned near 100% while the other is low.

This alone usually tells us the broad category. ncu confirms and localizes it.

---

## Phase 2 — ncu (needs MKP_PROFILE_ITERS; ~1–3 min)

`batchnum` on the 3070 is 512, and the cap is in candidates/thread, so
`MKP_PROFILE_ITERS=8192` ≈ 16 outer loops — a finite ~0.5s kernel.

Use **application replay** (`--replay-mode application`): the binary self-exits
under the cap, and re-running the whole app per pass avoids any state issues from
the kernel's mapped-pinned-memory writes.

```bash
MKP_PROFILE_ITERS=8192 ncu \
  --set full \
  --replay-mode application \
  --launch-count 1 \
  -f -o mkp_ncu \
  ./mkp224o -S 5 b32profiling
```

Quick targeted version (faster, the four sections that decide the strategy):

```bash
MKP_PROFILE_ITERS=8192 ncu \
  --replay-mode application \
  --section SpeedOfLight \
  --section Occupancy \
  --section LaunchStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  ./mkp224o -S 5 b32profiling
```

Read back (paste these numbers):
1. **SpeedOfLight**: `SM [%]` vs `Memory [%]` (Compute vs Memory bound).
2. **Launch Stats**: `Registers Per Thread`, `Achieved Occupancy`,
   `Theoretical Occupancy`, and the **occupancy limiter** (registers? block size?).
3. **Warp State Stats**: the top **stall reasons** (this is the tiebreaker).
4. **Compute Workload Analysis**: which **pipe** is hottest (FMA / ALU / LSU).

---

## Decision tree (what the numbers mean → what we do)

- **Memory % ≫ SM %**, stalls dominated by *Long Scoreboard* →
  **memory/latency bound.** Lever: batch-buffer layout & coalescing, cut traffic
  (e.g. keep more of the point in registers instead of round-tripping `batch_xyz`),
  raise occupancy. *Field-mul micro-opt would NOT help here.*

- **SM % high**, a math pipe (ALU/FMA/IMAD) near saturation, stalls on
  *Math Pipe Throttle* / *MIO Throttle* → **compute bound.**
  Lever: the int64→32-bit PTX field core (`fe_mul_cuda`/`fe_sq_cuda`) is the win.

- **Both %s moderate, achieved occupancy ≪ theoretical**, high registers/thread,
  stalls on *Not Selected* / *Wait* with few eligible warps →
  **occupancy/latency bound.** Lever: cut register pressure (fewer live `fe`
  temporaries, recompute-vs-store, `__restrict__`), tune block size. *Usually the
  biggest single win; do this before touching arithmetic.*

---

## Notes

- `MKP_PROFILE_ITERS` is **profiling-only**. Unset (the default) the kernel runs
  forever exactly as before — verified by the `max_iters == 0` guard in the
  kernel loop and launch setup. A run with it set prints a `PROFILE:` line.
- Use a real **hard** prefix; do not profile with an easy prefix or the result
  path will fire and pollute the measurement.
- **GPU counter permissions (`ERR_NVGPUCTRPERM`).** By default only root can
  read GPU performance counters. Two fixes:
  - Per-run: prefix with `sudo` (for `ncu`, pass the env var through with
    `sudo env MKP_PROFILE_ITERS=8192 ncu ...`).
  - Persistent (recommended for repeated runs):
    ```bash
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
      | sudo tee /etc/modprobe.d/nvidia-profiler.conf
    sudo dracut --force   # rebuild initramfs (Fedora)
    sudo reboot
    ```
    After reboot, run the commands above as your normal user.
