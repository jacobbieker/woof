# MLX (Apple GPU) backend for the WRF and MPAS ports, with four-way parity

## Context

woof's forecast runs only on CUDA. The WRF ARW port lives in `woof/core` (209 `.cu` files in `woof/core/kernels/`) and the MPAS-A v8.4.2 port in `woof/hex` (139 Python files carry inline `__global__` source). Together that is about 247k lines of CUDA C++. Kernels compile with NVRTC through CuPy, and 583 Python files import cupy.

On macOS nothing runs, not even on the CPU. There is no `xp` layer and no full-model CPU path (`woof/ingest/preprocess_backend.py:3`: "The forecast model itself remains CUDA-only").

The goal is for an Apple-silicon Mac to run both models on its GPU through MLX, as close to bit-for-bit as possible against:
- the CUDA build,
- a new real CPU backend,
- the source WRF/MPAS Fortran (the gfortran column oracles in `tools/*_oracle/` and `tests/data/oracles/`).

**Decisions already made:**
- Plan every phase of the port: all of WRF, all of MPAS, every subsystem.
- The `.cu` sources and hex inline strings stay the **single source of truth**. Metal Shading Language (MSL) is generated from them automatically.
- fp64:
  - strict mode uses **bit-exact soft-float binary64** (integer arithmetic on Metal's 64-bit `ulong`);
  - fast mode uses **df64** (float-float).

  This mirrors the existing `GPUWM_NUMERICS=strict|fast` split (`woof/numerics.py`).
- "CPU" means a real runtime backend: the same kernel source compiled as host C++. This generalises `tools/thompson_real_column_parity/host_backend.py` and `cuda_host_shim.h`.

**Parity target, stated honestly:**
- **Bit-identical across all four** wherever a kernel's arithmetic is pinned: no FMA contraction, IEEE operations, and libm through the in-repo word-exact ports (`glibc_flt32*.cuh`, `glibc_flt64.cuh`, `portable_libm64.cuh`, `cr_cbrt_flt64.cuh`, `f64cr.cuh`).
- **Recorded known deltas** wherever a kernel calls a vendor libm (CUDA libdevice `expf`, which Metal cannot reproduce) or uses fast mode.
- Phase 6 moves those vendor-libm call sites onto portable headers, so the delta set shrinks over time.

## Architecture (chosen approach)

```
woof/backend/                    NEW: backend selection + runtimes
  __init__.py                    active() -> "cuda" | "mlx" | "cpu"; install() before any woof.core import
  facade/                        a cupy-compatible module installed as sys.modules["cupy"] for mlx/cpu
    ndarray.py                   array wrapper (mutable handle over storage + view offset/strides)
    cuda_ns.py                   cp.cuda.{Stream,Event,Device,runtime,memory pool,...} equivalents
    kernels.py                   cp.RawModule / RawKernel / ElementwiseKernel / ReductionKernel -> translator
  translate/                     CUDA C++ -> MSL source-to-source (and -> host C++)
    lexer.py, rewrite.py, census.py
  metal/
    cuda_metal_shim.h            intrinsics: __fadd_rn..., min/max, bitcasts, atomics, barriers
    f64_soft.h                   bit-exact IEEE binary64 on ulong (add/sub/mul/div/sqrt/fma/cvt/cmp, RNE)
    f64_df.h                     df64 fast type, same API
    loader.py                    compile + cache + launch with CuPy call signature
    ext/                         small MLX C++ extension (custom Primitive) for in-place dispatch
  host/
    cuda_host_shim.h             promoted + generalised from tools/thompson_real_column_parity/
    loader.py                    builds .dylib/.so, ctypes launch, block-parallel thread pool
```

**Why a cupy facade rather than an `xp` refactor.** The repo already installs a NumPy-backed fake `cupy` for the host tier (`host_backend.install()`). The cupy API actually used is only **197 distinct attributes**; the top 10 are `float32`, `asarray`, `empty`, `ndarray`, `zeros`, `ascontiguousarray`, `asnumpy`, `int32`, `cuda.Device` and `float64`. Implementing that surface once keeps all 583 call-site files and every CUDA code path untouched, and the CUDA build keeps using real CuPy with zero overhead. The facade raises a named `BackendUnsupported` error on any attribute it doesn't implement, so it never degrades silently.

**Why soft-float binary64 is right for interop too.** `f64` is a `ulong` wrapper, so `double*` buffers keep binary64 layout in memory. Facade float64 arrays, `.npz` fixtures and NumPy all see real doubles; there is no repacking. df64 (fast mode) converts at load and store.

**In-place semantics risk.** CUDA kernels mutate their argument buffers, but MLX is functional: `mx.fast.metal_kernel` allocates outputs. The plan is a small MLX C++ extension (`woof/backend/metal/ext`) defining one generic `LaunchTranslated` Primitive. It encodes the compiled pipeline itself and aliases the outputs onto the input buffers (`copy_shared_buffer`), so non-`const` pointer parameters are written in place. The facade ndarray is a mutable handle whose underlying `mx.array` is rebound after each launch; views write through to their base. Phase 0 decides whether plain `mx.fast.metal_kernel` is good enough instead.

## Phases

### Phase 0: Receipts and spikes (go/no-go before any port work)

1. **Metal arithmetic receipt**, in the style of `tools/ftz_receipt/probe.py` and `tools/cuda_rounding_check.py`: `tools/metal_arith_receipt.py`. It measures on an M-series GPU:
   - correctly rounded fp32 `+ - * /` and `sqrt`;
   - whether `#pragma METAL fp contract(off)` and `fp math_mode(safe)` (MSL 3.2) actually stop fusing;
   - subnormal retention vs flush;
   - float↔int conversion rounding;
   - `-0.0` and NaN payload propagation.

   It records `woof/backend/metal/arith_receipt.json`. Anything not correctly rounded gets a shim helper (e.g. `__fdiv_rn` becomes a correctly rounded software divide).
2. **MLX dispatch spike.** Check whether `mx.fast.metal_kernel` can alias inputs to outputs, accept compile pragmas, and take a grid/threadgroup the way CuPy does. Prototype `LaunchTranslated` and measure launch overhead against `metal_kernel`. Pick one route.
3. **Soft-float spike.** Prototype `f64_soft.h` add/mul/div/sqrt/fma. Grade it against NumPy on 10^8 random plus edge-case operands (subnormals, overflow, ties, signed zero, NaN), and measure throughput against fp32.
4. **Translator spike.** Translate `kessler`, `wsm6`, `ysu_default` and `acoustic` end-to-end, compile them with `xcrun metal`, and run them against `tests/data/oracles`.

**Exit criteria:** Kessler and WSM6 are bit-identical to the WRF oracle on MLX, and soft-float passes its grade.

### Phase 1: Platform, selection, gating

- `pyproject.toml`:
  - add `osx-arm64` to pixi `platforms` (:146);
  - add features `mlx` (`mlx`, Xcode CLT check) and `cpu`;
  - add environments `mlx` and `cpu` with no cu12/cu13 feature;
  - add extras `mlx` and `all-mlx`;
  - add classifiers.
- **Backend selection:**
  - env var `WOOF_BACKEND=cuda|mlx|cpu` plus config key `[runtime] backend` (`woof/config_keys.py`, `woof/config.py`);
  - default is `cuda` if cupy is importable, else `mlx` on darwin-arm64, else `cpu`;
  - `woof/backend.install()` is called at CLI entry (`woof/cli.py`, `woof/__main__.py`) before `woof.core` imports, refusing (as `host_backend.install()` does) if real cupy is already loaded.
- **Capabilities and doctor:**
  - `woof/capabilities.py:112-134` gains `MLX_RUNTIME` and `CPU_RUNTIME` rows;
  - `woof/doctor.py` gains a Metal smoke compile;
  - `woof/doctor.py:515` stops treating darwin as "no GPU".
- **Device plumbing:**
  - `woof/core/device_cache.py`'s `cuda_cache` keys on the backend plus a device/stream token;
  - `woof/core/device_probe.py` and `device_inventory.py` price unified memory (`mx.metal.device_info()`);
  - `woof/core/devices.py` refuses `[devices] count>1` on mlx/cpu with a named message.
- **Tests:**
  - `tests/conftest.py` gains markers `mlx` and `cpu_backend`, plus a `--backend` option that parametrises the parity tests;
  - `tools/battery/run_stage1.py` gains an mlx leg;
  - `must_run_gates.txt` and `no_silent_skip.py` cover it.

### Phase 2: Facade and translator

- **Facade** (`woof/backend/facade/`):
  - Implement the 197 used attributes. The list comes from a generated census, `tools/cupy_surface_census.py`, which also becomes a CI gate that fails when new code uses an unimplemented attribute.
  - Array ops map to `mlx.core` (mlx) or `numpy` (cpu).
  - `cp.float64` arrays on mlx are stored as raw 8-byte words: soft-float on the GPU, NumPy views on the CPU.
  - `cp.cuda.Stream` / `Event` / `get_elapsed_time` map to MLX streams and timing, and `memGetInfo` maps to Metal device info.
  - CUDA graphs (`WOOF_HEX_CUDA_GRAPHS`) become a no-op capture: MLX already batches lazily.
- **Translator** (`woof/backend/translate/`): token-level rewrite of `module_source(name)` output. That function is the exact string NVRTC receives (`woof/core/kernels/__init__.py:411`), and hex inline strings go through it too. Rules:
  - `extern "C" __global__ void f(T* a, const T* b, int n)` becomes `[[kernel]] void f(device T* a [[buffer(0)]], const device T* b [[buffer(1)]], constant int& n [[buffer(2)]], uint3 tid [[thread_position_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]], ...)`. `threadIdx`, `blockIdx`, `blockDim` and `gridDim` are bound from these.
  - `__device__` functions taking pointers get the `device` / `thread` / `threadgroup` address space, inferred per call site with a small dataflow pass, or failing that by templating on address space.
  - `__shared__` becomes `threadgroup`; `extern __shared__` (9 files) becomes a `threadgroup` buffer argument sized from `shared_mem=`.
  - `__syncthreads()` becomes `threadgroup_barrier(mem_flags::mem_threadgroup)`.
  - `__shfl*` (1 file) becomes a `simd_shuffle`.
  - Float `atomicAdd` (12 files) becomes `atomic_fetch_add_explicit` on `atomic<float>`. Other atomics (33 files) map to `atomic_uint` / `atomic_int` CAS loops. Double atomics use a CAS loop on `ulong`.
  - `double` becomes `f64` (soft or df). Bitcasts (100 files) become `as_type<>`. `__dadd_rn` and similar (69 files) become `f64` ops.
  - `__constant__` becomes `constant`.
  - Templates (51 files) pass through, since MSL is C++14.
  - **`goto` (13 files) and recursion are not legal MSL.** The census refuses these units by name. They are restructured by hand in the `.cu` source itself (the CUDA build keeps passing its existing oracle gates), so there is still one source.
  - `printf` is stripped.
- **`module_options` mapping** (`kernels/__init__.py:440-524`):

  | CUDA option | Metal equivalent |
  |---|---|
  | `--fmad=false` | `#pragma METAL fp contract(off)` |
  | `--ftz=false` | the subnormal policy from the Phase 0 receipt |
  | `--prec-div` / `--prec-sqrt` | shim correctly rounded helpers |
  | fast mode | `contract(fast)` plus df64 |

- **`tools/metal_translation_census.py`** lists every unit across `woof/core/kernels`, the `woof/hex` inline strings, `ocean`, `hydro` and scattered `RawModule`s, each marked translated / compiles / refused-with-reason. It is a CI gate; its counts only go up.

### Phase 3: Loader integration

- `woof/core/kernels/__init__.py`: `get_kernel`, `get_kernel_int_defines`, `load_module*` and the unflushed and diffusion routes dispatch on `woof.backend.active()`. On mlx/cpu they call `woof/backend/{metal,host}/loader.py`, which returns an object with CuPy's call signature `kernel(grid, block, args, shared_mem=)`. `kernel_shape.specialize_source` and integer defines are reused unchanged.
- `woof/hex/cuda_backend/runtime.py`: `KernelCache` (~:294, :443) and `launch_checked` / `launch_timed` (:593/:621) take the same dispatch.
- The scattered `cp.RawModule` / `RawKernel` / `ElementwiseKernel` sites (~40 files) need no edits: the facade's classes route through the same translator.
- **Caching:**
  - the translated MSL and `.metallib` are keyed by source hash plus options plus the receipt hash, following `woof/nvrtc_ptx_cache.py`;
  - the AOT metallib packs mirror `woof/kernel_aot.py` and `tools/build_kernel_aot.py`.

### Phase 4: CPU backend

- Promote `host_backend.py` and `cuda_host_shim.h` into `woof/backend/host/`.
- Generalise the shim for `__shared__` and `__syncthreads`: a block executes as N cooperative threads on a `std::barrier`, and blocks run in parallel on a pool. Atomics use `std::atomic_ref`. Build flags are kept as-is: `-ffp-contract=off -fno-unsafe-math-optimizations -fno-tree-vectorize`.
- **libm on the host.** On macOS the shim's `expf` / `powf` etc. resolve to the in-repo glibc word-exact ports where they exist, not Apple libm. Otherwise they resolve to the musl bridge in `woof/core/portable_math.py`. This is recorded in the parity ledger.
- The same backend becomes the reference tier for the soft-float and df64 graders.

### Phase 5: Parity harness

- `woof/verify/backend_parity.py` plus `tools/verify_backend_parity.py`:
  - run each unit's oracle columns (`tests/data/oracles/*.npz`, existing `woof/verify/*_oracle.py` drivers in replay mode) through cpu, mlx and cuda (when present);
  - score words with `woof/core/fp32_ulp.py` (`bitwise_identical`, `max_ulp`, `assert_bit_exact`);
  - signed-ULP whole-state comparisons use `woof/verify/state_equiv.py`.
- **CUDA receipts.** Macs have no CUDA, so a Linux job records the CUDA output words per unit to `tests/data/backend_receipts/<unit>.npz` (with sha256 and a toolchain stamp). MLX and CPU then grade against CUDA anywhere.
- **Parity ledger** `woof/verify/backend_parity_ledger.json`. Each unit carries a class:
  - `bit-identical-4way`
  - `bit-identical-vs-wrf, delta-vs-cuda(libdevice)`
  - `fast-mode-delta`

  Each delta links a known-delta doc, following the `docs/wdm6_oracle_known_deltas.md` style. Ledger entries may only be promoted or demoted on test evidence.
- **Whole model:**
  - `tools/wrfout_bit_identity.py` across backends for an ideal case and an ERA5 case;
  - `woof verify-exact --backend mlx`;
  - hex runs use `woof/hex/step_oracle.py` and the `cuda_dualrun.py` SHA capsules against MLX.

### Phase 6: Port rollout (each step lands with its ledger rows green)

| Step | Scope | Key paths |
|---|---|---|
| 6a | WRF dycore + state | `woof/core/state.py` (lift CUDA-only guard for facade arrays), `dycore.py`, `acoustic.py`, `acoustic_fusion.py`, `advection.py`, `diffusion.py`, `moist.py`, boundaries, `clock.py` |
| 6b | WRF default suite | `morrison.py`, `ysu.py`, `sfclay.py`, `noah.py`, `kf.py` (unflushed route), `rrtmgp.py`, `physics.py` `PhysicsDriver`. First full forecast on MLX |
| 6c | Remaining WRF physics by family | microphysics (Thompson, WSM*, WDM6, NSSL, P3, fast_sbm, TEMPO), PBL (MYNN, MYJ, Shin-Hong, uwpbl, satmedmf), cumulus (GF, Tiedtke, SASE), LSM (Noah-MP, RUC, CLM lake, urban), radiation (RRTMG) |
| 6d | fp64 schemes on soft-float | `satmedmf.cu`, CLM lake (`f64cr.cuh`), `sase.cu`, `nextgen_cfe.cu`, GF real(8), fp64-libm users. Each is graded strict=bit-exact vs CUDA/WRF; fast=df64 delta recorded |
| 6e | MPAS hex | `woof/hex/cuda_backend/runtime.py`, `cuda_driver.py` and the `cuda_*_v841/v842/rrfs` drivers, `woof/hex/mpasphys/*`. Add `Backend.HEX_MLX` to `driver_v842.py:155` and rows in `physics_backend_admission.py:75`. `NATIVE_CPU` (Fortran ctypes) stays as the source-MPAS oracle. Also the `mpas_cycle_bridge` worker env, and the seam `woof/core/mpas_column_batch.py` |
| 6f | Peripheral GPU subsystems | `woof/ocean`, `woof/hydro`, chem, `woof/da`, `woof/ensemble` (batch_physics), ingest CUDA preprocess path, `woof/verify/*_oracle.py` RawModules |
| 6g | libm unification | Move vendor-libm call sites to the portable headers on all backends, re-grade against WRF, and promote ledger rows to 4-way |
| 6h | Performance | Threadgroup sizing, command-buffer batching, `mx.compile` for facade array-op chains, metallib AOT packs, a benchmark via `tools/benchmark_seeded_step.py --backend` |

Torch/Earth2 paths are out of scope. They can use `device="mps"` separately and are noted in the docs.

### Phase 7: Docs and release

- New `docs/dev/mlx-backend.md`: architecture, the numerics mapping, the receipt, and how to add a unit.
- Update `docs/community-support-matrix.md` (a macOS arm64 row), `docs/install.md`, `docs/platform-wheels-design-note.md` (osx-arm64 wheel), `CHANGELOG.md` and `README.md`.

## Critical files

- **Modify:**
  - `woof/core/kernels/__init__.py`, `woof/hex/cuda_backend/runtime.py`
  - `woof/core/device_cache.py`, `woof/core/state.py`
  - `woof/capabilities.py`, `woof/doctor.py`, `woof/numerics.py`, `woof/wrf_exact.py`
  - `pyproject.toml`, `tests/conftest.py`, `woof/hex/mpasphys/driver_v842.py`
  - the 13 `goto` units (hand-restructured, CUDA gates re-run)
- **Reuse:**
  - `tools/thompson_real_column_parity/host_backend.py` and `cuda_host_shim.h` (facade + host pattern)
  - `woof/core/fp32_ulp.py`, `woof/verify/state_equiv.py`, `woof/nvrtc_ptx_cache.py`, `woof/kernel_aot.py`, `woof/core/kernel_shape.py`
  - `tools/wrfout_bit_identity.py`, `woof/hex/step_oracle.py`, `woof/hex/physics_backend_admission.py`
  - the oracle fixtures in `tests/data/oracles/`

## Verification

1. **Phase 0:** `python tools/metal_arith_receipt.py` writes the receipt. The soft-float grader passes 10^8 random and edge operands bit-for-bit against NumPy.
2. **Unit parity** on the Mac: `pixi run -e mlx pytest -m "mlx" tests/test_backend_parity.py --backend mlx`, then `--backend cpu`. Every ledger row's class is checked. On Linux, a CUDA job regenerates `tests/data/backend_receipts`, and the same tests pass with `--backend cuda`.
3. **Existing oracle suites** (e.g. `tests/test_kessler_wrf461_parity.py`, `test_wsm6_wrf461_column_exact.py`, `test_ysu_wrf461_parity.py`) run parametrised over backends and stay bit-exact where the ledger says 4-way.
4. **Gates:**
   - `tools/metal_translation_census.py` and `tools/cupy_surface_census.py` report no regressions;
   - `pytest -m "not gpu"` stays green on Linux and macOS;
   - the existing CUDA suite on Linux is unchanged. The facade isn't installed when backend=cuda, so a CUDA job confirms zero diff in `wrfout_bit_identity` against main.
5. **End-to-end:**
   - `WOOF_BACKEND=mlx woof run <ideal case>` and an ERA5 real case, comparing with `tools/wrfout_bit_identity.py` against the CPU backend (strict) and the CUDA receipt;
   - `woof verify-exact --backend mlx`;
   - a hex RK3 trajectory against `woof/hex/step_oracle.py`;
   - `woof doctor` on the Mac reports the MLX runtime.

## Phase 0 status (2026-10-29)

**Done, with evidence:**
- **pixi:** `osx-arm64` (macOS ≥ 14) plus the `mlx` and `cpu` environments. The Linux CUDA lock entries are byte-identical.
- **Dispatch:** `mx.fast.metal_kernel` cannot write in place (unwritten words come back undefined). It is replaced by the `LaunchTranslated` MLX primitive (`woof/backend/metal/ext/`):
  - in-place via `copy_shared_buffer`, byte-offset binding, a barrier for write-after-read;
  - about 44 µs per launch, against 71 µs for `metal_kernel`;
  - pending lazy readers are encoded before each launch (0 hazards in 50 trials).
- **Arithmetic receipt** (`tools/metal_arith_receipt.py` → `woof/backend/metal/arith_receipt.json`, M3 Max):
  - fp32 `+ - * fma` and `/` (in `safe` mode) are IEEE and exactly CUDA's `-ftz=true` model;
  - plain `sqrt` is ≤ 2 ULP off, while `precise::sqrt` is exact;
  - `a*b+c` fuses even in `safe` mode; `#pragma METAL fp contract(off)` unfuses it;
  - subnormals always flush on arithmetic but survive copies;
  - Metal has no `cbrt`.
- **Soft-float binary64** (`woof/backend/metal/f64_soft.h`): bit-exact on 10^8 host and 16.8M GPU samples (`tools/f64_soft_grade.py`). Throughput relative to fp32:
  - `mul+add` and `fma`: 7.5–8× slower;
  - `div`: 127× slower; `sqrt`: 331× slower (bit-serial for now; the fp32-seeded Newton step is a perf item).
- **Translator** (`woof/backend/translate/`, libclang): Kessler runs through the production `launch_kessler`. In strict mode:
  - t/qv/qc/rainnc/rainncv are **bit-identical to WRF** on every word;
  - qr has 112 subnormal residues, identical to CUDA's documented default-build figure.

**Design changes forced by the spikes:**
- The translator is AST-based, not regex. CUDA evaluates unsuffixed literals and mixed float/double expressions in binary64, and only a typed AST finds them.
- Strict Metal never contracts, in any unit. NVRTC's contraction choices can't be reproduced, so strict Metal matches the host tier and gfortran -O0 instead.
- **Subnormal retention** (`GPUWM_WRF_EXACT=1`, `_NO_FTZ` units) needs an `ieee32` value type: a hardware fast path plus a soft-float32 slow path when an operand or result is tiny. Without it, the strict build equals CUDA's ftz build, not WRF, on subnormal words.
- **Facade array ops** route through MLX's own kernels, which are not graded yet. sqrt and the transcendentals need woof Metal kernels or a known-delta entry (facade op receipt).

**Phase 0 exit: met (2026-10-29).** The repo's own WRF 4.6.1 parity tests run unchanged on MLX (`pixi run -e mlx pytest tests/test_{kessler_wrf461_parity,wsm6_wrf461_column_exact,ysu_wrf461_parity}.py`):
- **24 passed in each build:**
  - `GPUWM_WRF_EXACT=1`: every word, subnormals included;
  - default: CUDA's `-ftz=true` behaviour, residue for residue.
- **Translator mechanisms these needed, all general:**
  - address-space inference, with per-combination overloads and nullptr casts;
  - `const`-pointer localisation for cross-space swaps;
  - thread builtins threaded into device functions;
  - packed scalar structs, plus argument tables for more than 29 pointers (needs MLX residency sets: `set_wired_limit` + a guard);
  - exact constant folding;
  - binary64 libm (`f64cr` correctly rounded, plus the Rust-libm `atan2`/`hypot`/`log2`/`expm1`/`tanh` in `woof/backend/translate/vendor_libm64.cuh`, from `tools/vendor_libm64.cuh.in`);
  - **`f32s`**, the subnormal-retaining binary32 for `--ftz=false` units: hardware fast path, exact binary64 slow path, round-to-odd fma.
- **Facade fixes:**
  - storage is always dense (a stride-0 `cp.full` buffer was the WSM6 58k-word bug);
  - Apple GPU device attributes (`MultiProcessorCount` = GPU cores).
- **Process plumbing:**
  - `import woof` installs `WOOF_BACKEND=mlx|cpu`;
  - `tests/conftest.py` installs it before probing cupy;
  - pixi `mlx` activation sets `WOOF_BACKEND=mlx` plus a `sitecustomize` hook on `PYTHONPATH`, and wheels ship `woof-backend.pth`. Tools and test subprocesses that import cupy first therefore get the facade too.

**Census facts (drive Phase 2):** 1,894 kernels. 117 have more than 30 parameters; 66 have more than 30 pointers (max 93). The cupy API surface is 197 attributes.
