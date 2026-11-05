# The host (CPU) backend

`WOOF_BACKEND=host` runs woof's GPU code on any CPU. As on the MLX backend,
the CUDA sources are not ported by hand: each kernel unit is the exact text
NVRTC would compile, built by the host C++ compiler (clang++ or g++) into a
shared library the first time it loads. `import cupy` resolves to a
CuPy-compatible module over NumPy, so the Python that launches the kernels
runs unchanged.

```
WOOF_BACKEND=host pixi run -e default pytest tests/test_kessler_wrf461_parity.py
WOOF_BACKEND=host CUDA_VISIBLE_DEVICES=-1 python -m woof ...   # nothing touches a GPU
```

The backend is chosen only by name. `cpu` (the default on a machine with no
CuPy and no MLX) still installs no facade, so `import cupy` stays an
ImportError there and the CPU-only test tier skips GPU work as before. The
pixi `default` environment works with `WOOF_BACKEND=host`. A test process that
imports cupy before woof needs `PYTHONPATH=woof/backend/startup`, as the mlx
environment sets it. No pixi environment selects it yet.

## Layout

| Path | What it is |
|---|---|
| `woof/backend/host/rt.h` | CUDA's language on the host: the keywords, `threadIdx`/`blockIdx`, `__syncthreads`, warp shuffles and votes, atomics, the intrinsics, and the launcher template. It is force-included before every unit. |
| `woof/backend/host/rt_pool.cpp` | The runtime: one pool of worker threads that runs a launch's blocks, plus the fibers and context switch for a block's threads. Built once and shared by every unit library. |
| `woof/backend/host/loader.py` | Builds a unit (preprocess, rewrite, find kernels, append launchers, compile, cache) and launches its kernels with CuPy's signature. |
| `woof/backend/host/include/` | Empty `cuda.h`/`cuda_runtime.h` for units that include them. |
| `woof/backend/hostfacade/` | CuPy's API over NumPy: arrays (`__init__.py`), `cupy.cuda` (`cuda.py`), `ElementwiseKernel`/`ReductionKernel` (`_elementwise.py`). |

## How a unit is built

1. `extern __shared__ T name[];` becomes a pointer to the launch's dynamic
   shared buffer.
2. The text is preprocessed after `rt.h`, with the unit's own `-D` and `-I`
   options, as NVRTC preprocesses it (`__CUDA_ARCH__` is 860, the value the
   Metal translator also uses). Inline PTX is rewritten by the translator's
   rules (`woof.backend.translate.rewrite_ptx_asm`). The unit's own
   `uint64_t`/`int64_t` typedefs are dropped, because the C library already
   declares them (with `long` where NVRTC says `long long`).
3. Kernels are read off the preprocessed text. For each one, a launcher is
   appended that unpacks the arguments by the kernel's own parameter types
   (pointer, integer, floating or by-value struct, read from the C++ types)
   and runs the grid.
4. The unit is compiled with `-O2 -ffp-contract=off -fno-fast-math -fwrapv
   -fno-strict-aliasing` and cached under `~/.cache/woof/host-kernels`
   (`WOOF_HOST_CACHE`). The cache key is the source, the options, every
   header in the include directories, `rt.h`, and the compiler and its flags.
   `WOOF_HOST_CXX` (or `CXX`) picks the compiler, and `WOOF_HOST_OPT=-O1`
   trades run time for a faster cold compile.

## How a launch runs

A launch's blocks are shared out among `WOOF_HOST_THREADS` worker threads
(default: every CPU this process may use). Each worker takes the next block
index from an atomic counter. Blocks are independent in CUDA, so any order is
a valid schedule. Launches are serialised, as on one CUDA stream, and the call
returns when the kernel has finished.

Within a block there are two modes:

* **A unit that never synchronises** runs each block's threads one after
  another on the worker. This is the fast path, and most WRF ports (one
  thread per column) take it.
* **A unit that names `__syncthreads`, `__syncwarp`, a warp shuffle or vote,
  or `__syncthreads_count/and/or`** runs each block's threads as fibers on
  the worker that owns the block. A barrier switches to the next fiber, and
  a fiber's stack is used again once its thread finishes. `__syncthreads`
  releases when every live thread of the block has arrived (a thread that
  has returned does not hold it, as on the hardware). A warp collective
  releases when every live lane of the warp has arrived. Shuffles and votes
  pass values through a per-warp exchange. If the threads of one block wait
  at different barriers, the launch fails with "barrier deadlock" instead of
  hanging.

`__shared__` variables are `static thread_local`: one copy per worker, and a
block runs on one worker. Atomics are the host's atomic read-modify-writes,
sequentially consistent, because blocks of one launch run at once on several
workers. A floating `atomicAdd` is a compare-and-swap loop, as CUDA's
fallback is, so its order (and the rounding of a contended sum) varies from
run to run, as it does on a GPU. A failed device `assert` or `__trap()` stops
that thread and raises `LaunchError` from the launch.

Stacks: workers get 64 MiB (`WOOF_HOST_THREAD_STACK_KB`) and fibers 4 MiB
(`WOOF_HOST_FIBER_STACK_KB`). Both are reserved, not committed, and have a
guard page below them. A kernel whose locals overflow its stack ends the
process with SIGSEGV.

## Numerics: what each CUDA arithmetic rule becomes

| CUDA | host |
|---|---|
| binary32/binary64 `+ - * /`, `sqrt`, `fma` | the host's IEEE operations (SSE2/AVX on x86-64), round to nearest |
| `--fmad=true` / `false` | never contracted (`-ffp-contract=off`): CUDA's `--fmad=false` words in every unit |
| `-ftz=true` (CuPy's default) | the SSE flush-to-zero and denormals-are-zero modes are set for the launch |
| `--ftz=false` units | subnormals kept |
| inline PTX without `.ftz` (`add.rn.f32` …) | the operation with the flush modes cleared around it, so subnormals are kept in any unit |
| `fminf`/`fmaxf`/`min`/`max` | CUDA's: −0 orders below +0, and a NaN operand yields the other operand |
| `__float2int_*` and similar | saturating, with NaN giving 0 (`cvt.*i`) |
| libdevice binary32/binary64 (`expf`, `powf` …) | the C library's (glibc): a vendor-libm difference, as on Metal. Units built on woof's glibc ports are exact. |
| `__expf`, `__powf`, `__fdividef`, `rsqrtf` … | the accurate functions, where CUDA's hardware approximates |
| NaN results | x86's: the default NaN is `0xffc00000` and payloads propagate, where CUDA gives `0x7fffffff` |

## Measured against CUDA

Measured on 2026-11-02 against CUDA 13.4 (NVRTC 13.4.92, CuPy 14.2.0) on an
RTX A6000 (sm_86). The pixi `default` environment was used throughout. The
host build used clang 22.1.8 with 6 workers per test process.

**WRF parity tests.** The test set is every `tests/*wrf4*{parity,exact}*`
file plus the Kessler, WSM6, YSU, deformation and advection families
(102 files, 2,362 tests), run test by test on both backends:

| numerics | CUDA (pass / fail / skip) | host (pass / fail / skip) | same outcome | differ |
|---|---|---|---|---|
| default (`fast`) | 2271 / 21 / 70 | 2234 / 59 / 69 | 2324 | 38 |
| `GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1` | 1960 / 341 / 61 | 1957 / 344 / 61 | 2359 | 3 |

(Errors are counted as failures. The 341 strict failures common to both
backends are failures on CUDA too, such as stale receipts.)

Each of the 38 default-mode differences has a known cause:

* **26 are pins of CUDA's contracted words**: bigstep momentum (15), the RK
  tendency (6), and small-step horizontal (2) and vertical (3). They fail
  the same way on CUDA with `WOOF_FMAD_OFF=1`, and the host is closer to
  WRF there. For example, momentum's `rw_t` is 0 ULP from WRF against the
  pinned 50.
* **8 are libm** (phy_prep's `t_phy`/`pi_phy`). The Exner power is CuPy's
  `powf` on CUDA and glibc's (through NumPy) on the host: 0 ULP from WRF on
  the host, against CUDA's pinned 2.
* **1 asserts that CUDA's `powf` differs from WRF**
  (`test_milbrandt2…::test_the_comparison_sees_cudas_powf`). On the host it
  does not differ.
* **1 is NaN payloads.** In Noah mosaic column 24, x86's NaN words are
  WRF's, and CUDA's are canonical.
* **2 ask NVRTC to identify itself** (Shin-Hong's compiler-keyed rows).

Under strict numerics, where the units use woof's glibc ports, only the last
three classes remain.

**End to end.** `woof.verify.cases.moist_bubble` (WK82 moist bubble with
Kessler, PD advection, km_opt=4 and 6th-order diffusion, on a 128×1×40
channel at dt = 6 s) was run for one hour (600 steps) on both backends.

Under `GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1`, **every word of every
field after the hour is identical** on the two backends (u, v, w, θ′, p,
qv, qc, qr, RAINNC: 0 differing words of 46,616).

With default numerics, the backends part at the first launch (`calc_p_alpha`:
CUDA's `powf` against glibc's), and the difference grows as the convection
develops. After the hour, the two CUDA builds differ from each other about
as much as the host differs from either:

| field (max \|Δ\| / rms) | CUDA vs CUDA `--fmad=false` | CUDA vs host | CUDA `--fmad=false` vs host |
|---|---|---|---|
| u (m/s) | 0.28 / 0.019 | 0.41 / 0.028 | 0.36 / 0.028 |
| w (m/s) | 0.19 / 0.014 | 0.24 / 0.022 | 0.35 / 0.026 |
| θ′ (K) | 0.31 / 0.0095 | 0.41 / 0.014 | 0.42 / 0.015 |
| p (Pa) | 5.4 / 0.25 | 9.1 / 0.45 | 8.7 / 0.47 |
| qc (kg/kg) | 1.9e-5 / 9.9e-7 | 3.9e-5 / 1.8e-6 | 4.1e-5 / 1.9e-6 |
| qr (kg/kg) | 7.8e-5 / 4.4e-6 | 1.3e-4 / 7.8e-6 | 1.0e-4 / 6.4e-6 |
| RAINNC (mm, max 37.7) | 0.016 / 0.0037 | 0.029 / 0.0053 | 0.015 / 0.0024 |

Peak updraft over the hour: 28.708 m/s on CUDA, 28.707 with
`--fmad=false`, and 28.704 on the host.

To find where the two backends first part, record launches on both with
`tools/launch_trace.py` (it records the host backend's launches) and compare
the traces. CUDA's compiles can be forced to `--fmad=false` with
`WOOF_FMAD_OFF=1`.

## Speed

One thread per element, run serially within a block, is not how a CPU wants
to be used, so the backend is for correctness and for machines without a
GPU, not for throughput. These figures are from the 32-core build host while
other jobs were running, so treat them as approximate:

* **128×1×40 moist bubble, 600 steps:** host 44 s (8 workers), CUDA 68 s.
  CUDA is launch-bound on a domain this small.
* **The same case on 128×128×40:** about 1.8 s per step with 24 workers,
  against 0.017 s on the A6000, so roughly 100× slower.
* **Cold builds:** the first run of the test set above built every unit
  it used and took 154 s; the cached run took 130 s. Once a unit is built,
  loading it is a `dlopen`.

## Not supported

* Cooperative launches and grid-wide synchronisation, dynamic parallelism,
  textures and surfaces, and `__half` arithmetic (`half` is `_Float16`).
* Inline PTX outside the idioms the Metal translator reads. Such a unit
  fails to build, naming the line.
* Template kernels, except those named by `name_expressions`. Kernels in
  namespaces, and overloaded C++-linkage kernels.
* Partial warp masks. A warp collective waits for every live lane of the
  warp, whatever the mask says.
* The ternary max/min idiom on signed zeros (`a > b ? a : b`). NVRTC may
  compile it to `max.f32`, and the host keeps C++'s answer; the Metal
  translator spells that idiom out.
* The flush modes also flush binary64 subnormals in an `-ftz=true` unit
  (CUDA flushes binary32 only), and `fabsf`/negation of a subnormal are not
  flushed.
* Facade array operations (`cp.exp`, reductions, `cupyx.scipy`) are NumPy's
  and SciPy's, not CuPy's kernels. `ReductionKernel` folds serially in index
  order. The memory pool reports 0 bytes.
* The hex (MPAS) lane: its kernel cache asks CUDA questions (runtime
  version, NVRTC build) that the facade answers with zeros, and it is not
  measured on this backend.
