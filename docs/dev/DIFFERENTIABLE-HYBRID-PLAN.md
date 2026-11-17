# Gradients against observations and hybrid physics/NN: feasibility and plan

Status: design study, 2026-11-21. Nothing in this document is implemented.
It answers two requests: (1) get gradients of a woof forecast with respect
to the observations that are available, and (2) work towards a
differentiable WRF/MPAS/hex in which physics schemes can be swapped for
neural-network components. Both live inside metal-woof; no new repository.

Numbers marked *measured* come from the repo's own docs or from counting
files in this checkout (cited by path). Numbers marked *estimate* are
arithmetic on those, and the first phase of the plan replaces them with
measurements.

## Verdict, in short

- **Gradients of an observation cost with respect to a small control
  vector are feasible now**, without any adjoint code: finite differences
  and ensemble sensitivity on top of the observation operators and the
  batched ensembles that already exist (phase P0). This is the cheapest
  useful answer to "how does the forecast's fit to these observations
  depend on X".
- **A column-physics NN plug-in (inference only, opt-in) is feasible and
  cheap.** woof already has the seats for it: the held physics tendencies
  that SPPT/SKEBS modify on both routes, the hex `PhysicsBackend` protocol,
  and the `woof.hex.rows` entry-point group (P2).
- **Adjoints of single column schemes are feasible** as hand-written (or
  Tapenade-generated) adjoint kernels wrapped in `torch.autograd.Function`
  and `mx.custom_function` (P1).
- **An adjoint of one dynamical core is feasible but large.** The WRF dry
  core is the right first target, because WRFPLUS (WRF's own Tapenade-made
  tangent-linear and adjoint) is a public-domain oracle for it; MPAS has no
  such oracle (P3).
- **End-to-end differentiable windows** (4D-Var-style gradients and online
  hybrid training) follow from P3 plus checkpointing. They are practical for
  1-6 h windows at 9-15 km, not for 24 h at 3 km (P4; the cost table in
  section 6 says why).
- **Not worth doing:** rewriting woof in PyTorch or JAX (a second model that
  would drift from the WRF-identical one), adjoints of every physics scheme,
  Enzyme on Metal (there is no route), and differentiating through the
  strict soft-float binary64 path on Metal.

None of this may move a default word. Every piece below is a separate
module, a separate set of kernels, or a hook whose absent state launches
nothing; section 8 lists the tests that prove it.

## 1. What woof is, as it bears on differentiability

### 1.1 Kernels and backends

- *Measured:* 478 `.cu` files with 327,978 lines in the repo; 433 files and
  313,881 lines under `woof/` (`woof/core/kernels/`, `woof/hex/`, ...).
  The hex kernel pack inventory
  (`woof.hex.cuda_backend.aot_build.pack_inventory`) has 361 units, of which
  353 translate to Metal ([mlx-backend.md](mlx-backend.md)).
- CUDA kernels are CuPy `RawModule`/`RawKernel`/`ElementwiseKernel` objects
  compiled by NVRTC at runtime (`woof/nvrtc_ptx_cache.py`), with the
  numerics options rewritten at the final NVRTC boundary
  (`woof/numerics.py`).
- On a Mac, `import cupy` resolves to a facade over `mlx.core`
  (`woof/backend/facade/`). Each `.cu` unit is translated to MSL with
  libclang (`woof/backend/translate/__init__.py`, 4,889 lines) and
  launched **in place** on its argument buffers by woof's own MLX
  primitive (`woof/backend/metal/ext/launch.cpp`); `mx.fast.metal_kernel`
  is not used for production launches because it allocates every output
  fresh.
- **Every kernel mutates its arguments in place.** There is no functional
  graph, no tape, and no autodiff anywhere: a grep for `jax`, `mx.grad`,
  `mx.vjp`, `autograd`, `enzyme`, `tapenade` finds nothing, and "adjoint"
  appears only in unrelated senses. `torch` is imported only by the Earth2
  AI-weather door (`woof/earth2_*.py`), and there only as a seeded RNG.
- An earlier gap review
  ([gaps-and-roadmap.md](../comparisons/clima-numericalearth/gaps-and-roadmap.md),
  section 5) rated a differentiable single-scheme column "M" and a full-model
  adjoint "L, not recommended". This plan refines that: the full-model
  adjoint stays out of scope, but a dry-core adjoint with an existing oracle
  and NN plug-ins are worth doing.

### 1.2 Numerics modes

- `GPUWM_NUMERICS` is `fast` (default) or `strict` (`woof/numerics.py`).
  `GPUWM_WRF_EXACT=1` implies strict and adds WRF operation ordering
  ([wrf-exact-arithmetic.md](wrf-exact-arithmetic.md)). Strict compiles
  with `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
- Prognostic state is binary32 on both backends (`DTYPE = cp.float32`,
  `woof/core/state.py:71`). Metal has no binary64, so `double` in a kernel
  becomes a 64-bit word: bit-exact soft-float in strict
  (`woof/backend/metal/f64_soft.h`), df64 estimates in fast
  (`f64_df64.h`; 52-242 ulp measured, 512 ulp limit).
- **Consequence for gradients.** The soft-float and df64 paths are integer
  bit manipulation; no AD tool can trace them, and none should. Gradient
  code is a different program from the primal: it runs in native binary32
  (both backends) or binary64 (CUDA only, at 1/32 rate on an A6000), and it
  is never part of an identity proof. Identity is a property of the primal
  words only.
- Tangent-linear/adjoint *tests* (section 5.7) need binary64 to reach the
  usual 1e-12 dot-product agreement; in binary32 the achievable agreement is
  about 1e-5 to 1e-6. So adjoint kernels are written templated on the real
  type, and the binary64 instantiation is the test build.

### 1.3 State layout and size

- **WRF route.** `DomainState` (`woof/core/state.py:443`): float32 CuPy
  arrays, `(nz, ny, nx)` with C-grid staggering (`u (nz,ny,nx+1)`,
  `v (nz,ny+1,nx)`, `w`/`php (nz+1,ny,nx)`, `thp`, `p`, `al`, `alt`,
  `mup (ny,nx)`), moist species with time-t copies, `h_diabatic`, and
  optional held physics tendencies. The base state is in `BaseState`
  (`woof/core/grid.py:575`).
- **hex route.** `PrognosticState` (`woof/hex/state.py:15`): `(level,
  entity)` layout; `rho`, `rho_theta (nz,nCells)`, `rho_u (nz,nEdges)`,
  `rho_w (nz+1,nCells)`, `scalars (nScalars,nz,nCells)`; device copies are
  float32 (`woof/hex/cuda_acoustic.py`). nEdges is about 3 x nCells.
- *Measured* whole-run device peaks (they include physics scratch and the
  CUDA context, not just the state):
  WRF route 448x360x49 with full physics, 8.49 GiB; 200x200-class grids
  3.6-4.1 GiB
  ([vram-sizing-calibration-20260801.md](../vram-sizing-calibration-20260801.md)).
  hex: 49-84 KB per cell (`woof/hex/card_fit.py`); a 128,942-cell globe
  peaked at 13,036 MiB.
- *Estimate*, trajectory checkpoint (what an adjoint must keep per saved
  step: prognostics, moist species, and the carried physics state such as
  `h_diabatic`, held radiation tendencies, PBL TKE and soil):
  - WRF 200x200x50 (2.0 M cells, 8 MB per 3-D field): about 25 fields,
    **~0.2 GB per checkpoint**.
  - hex 100k cells x 55 levels (22 MB per cell field, 66 MB for `rho_u`):
    about 3 + 8 scalars + 6 carried cell fields plus `rho_u`,
    **~0.45 GB per checkpoint**.
  - P0 replaces these with the size of the existing restart payloads
    (`woof/io/restart.py`, `woof/hex/checkpoint_io.py`), which are by
    construction the full trajectory state.

### 1.4 Time stepping

- **WRF route** (`woof/core/dycore.py`, `step` at :4951): RK3 with
  Wicker-Skamarock acoustic substeps. `_rk_acoustic_stages` (:4941) gives
  `(1, dt/3), (ns//2, dt/ns), (ns, dt/ns)`; with `time_step_sound = 4` that
  is **3 stages and 7 acoustic substeps per step**. Physics
  (`PhysicsDriver.compute`, `woof/core/physics.py:7398`: radiation on its
  cadence, surface layer, LSM, PBL, cumulus) runs before the RK loop, and
  its tendencies are **held constant across all three stages**
  (`PhysicsTendencies.add_to_slow`). Microphysics is time-split after the
  loop and hands `h_diabatic` to the next step.
- **hex route** (`woof/hex/cuda_driver.py`, `_step_device_v841` at :4870):
  MPAS split-explicit. Defaults (`woof/hex/mpas_defaults.py`):
  `config_dynamics_split_steps = 3`, `config_number_of_sub_steps = 4`, so
  **3 x (1+2+4) = 21 acoustic substeps per physics step**, then scalar
  transport once over the whole step. Physics is two-phase: phase 1
  (radiation, surface/PBL, cumulus) before the dynamics with held
  tendencies; phase 2 (microphysics) after transport
  (`woof/hex/cuda_arwen_physics_v841.py:1771`, `:2117`).
- *Measured* step costs:
  - WRF route, RTX A6000, CUPTI
    ([perf-launch-overhead-20261015.md](perf-launch-overhead-20261015.md)):
    an ordinary full-physics step is **390-409 kernel launches** at every
    grid size, 250 of them dynamics, **93 for the dry core**; radiation steps
    issue 697-7,646. Wall time per ordinary step: 49.6 ms (40x40x30),
    100.3 ms (100x100x49), **238.6 ms (200x200x49)**; the dry core alone is
    6.1, 18.2 and 34.3 ms.
  - hex: the first step on a 1,607-cell, 55-level mesh issues 1,709
    launches; Maui 3 km (7,042 cells, dt 18 s) runs **0.15 s/step on the
    A6000 and 0.60 s/step on the M3 Max**
    ([mlx-backend.md](mlx-backend.md), [ADAPTIVE-TIMESTEP.md](../ADAPTIVE-TIMESTEP.md)).
- Steps for a 24 h run (dt about 6 s per km; `mpas_defaults.default_dt_seconds`
  on hex): 3 km, dt 15-18 s, **4,800-5,760 steps**; 9 km, dt 54 s, 1,600;
  12 km, dt 60-72 s, 1,200-1,440; hex 15 km, dt 90 s, 960.

### 1.5 The column-physics seams

- **WRF route.** `initialize_physics` (`woof/core/physics.py:8318`) picks
  scheme callables from RunConfig IDs (`mp_physics`, `bl_pbl_physics`,
  `cu_physics`, `ra_lw/sw_physics`, `sf_*`). Radiation and cumulus can be
  supplied as Python callables returning `RadiationResult` /
  `CumulusResult` (`physics.py:1228`, `:1289`), with a strict-JSON
  `restart_identity` (`implementation: "custom"`, `woof/io/restart.py:2772`).
  `_compose_tendencies` (:4141) sums PBL, radiation and cumulus rates into
  `PhysicsTendencies` (:1505).
- **Stochastic physics is the template for an NN tendency.** On the WRF
  route `StochasticTimestepHook` (`woof/ensemble/stochastic.py:447`) is
  called as `after_nonmicrophysics(state, cfg, physics_tendencies)`
  (`dycore.py:5192`) and returns a modified `PhysicsTendencies` (SKEBS then
  SPPT); on hex `StochasticPhysics.apply` modifies the held sources
  (`woof/hex/mpasphys/runtime_v842.py:317`), followed by IAU and nudging in
  the same seat. SPP (`woof/hex/spp.py`, #426) shows the other pattern:
  feeding a field *into* a scheme (`PhysicsDriver.bind_spp_patterns`).
- **The generic column seam** (`woof/core/mpas_column_suite.py`,
  `MpasColumnPhysics` :1467; contract in [mpas-seam.md](../mpas-seam.md))
  runs the WRF `PhysicsDriver` on mesh columns: `[level, column]` float32
  device arrays in, held A-grid tendencies (`du, dv, dtheta, dq*,
  h_diabatic`) out of phase 1, in-place microphysics in phase 2.
- **The hex physics backend protocol** (`woof/hex/physics_seam.py`):
  `ColumnBatch` (:628), `ColumnTendencies` (:1153) and
  `PhysicsBackend.run(batch, selection, dt) -> ColumnTendencies` (:1211),
  with a CUDA-free test double (`DeterministicNumpyBackend`, :1496).
  Third-party backends can be installed through the `woof.hex.rows`
  entry-point group (`woof/hex/physics_backend_admission.py:357`), and
  `register_backend_row` refuses to rebind an existing name, so a plug-in
  cannot displace the default row.
- The physics registry (`woof/physics_registry.py`,
  `physics_registry_v2.json`) has a maturity ladder (`planned` ...
  `wrf-matched-run`) that an NN option would enter at
  `experimental-runtime`.

### 1.6 Observations, operators and grading

- **Operators** (`woof/da/`; all pure functions of a member state, CuPy or
  NumPy):
  - `obs_conventional.py`: `pressure_level` (bilinear horizontally, linear
    in ln p; T, u, v, Td from aircraft, sondes, AMVs) and `surface` (t2, q2,
    rotated u10/v10).
  - `obs_point.py`: the same plus GNSS radio-occultation refractivity
    (`woof/obs/column_operators.py:refractivity_n`, Smith-Weintraub).
    `brightness_temperature_k` is refused (`requires_column_radiance_operator`).
  - `obs_surface.py`: ASOS T2, wind speed, opt-in dewpoint.
  - `obsop.py`: radar reflectivity (over `woof/core/refl.py`, per
    `mp_physics`) and radial velocity (4/3-earth beam geometry).
  - `obsop_cwp.py`: GOES cloud water path, linear in the hydrometeors.
  - No GNSS PW/ZTD operator; no CRTM/RTTOV. `woof/radtran/`
    ([radtran.md](radtran.md)) is a GPU spectral RT used only for output
    diagnostics.
  - All operators are **3-D at an analysis time** (window +/-30 min); none
    is a 4-D (time-interpolated) operator yet.
- **Smoothness of H.** T, wind, RO refractivity and CWP are linear or
  smooth and have trivial adjoints. Reflectivity (log, -35 dBZ floor, per
  scheme) and radial velocity are piecewise; superobbing and nearest-point
  placement are index operations (no gradient with respect to location).
- **DA present.** A batched GPU LETKF (`woof/da/letkf.py`, Hunt et al. 2007;
  "EXPERIMENTAL. Nothing in the forecast path calls this module") with
  device kernels (`letkf_device.py`), an EnOI-style static sample
  (`static_covariance.py`), IAU (`iau.py`), radar latent-heat nudging, WRF
  obs nudging (`woof/fddaobs/`, a translation of `module_fddaobs_rtfdda.F`)
  and hex obs nudging (`woof/hex/obs_nudging.py`). No tangent-linear model,
  adjoint, B matrix, variational cost or minimizer exists.
- **Observation data.** Icechunk on Source Cooperative: `bkr/obs`
  (`igra_v2_16_levels`, `wis2_upper_air`, `wis2_land`; read by
  `woof/hex/sonde_profile_referee.py` and `station_wind_referee.py`);
  Dynamical ASOS Parquet (`woof/obs/dynamical_asos.py`); MRMS
  (`woof/obs/mrms_fetch.py`); Rust front doors for ASOS, MADIS, IGRA,
  prepbufr, AMV, GNSS-RO, MRMS, Stage IV, OPERA, NDBC
  (`tools/rustwx/crates/rw-obs`). The `bkr/aoml/v2`, `bkr/nsrdb`,
  `bkr/mrms` and `bkr/precipradar` stores have no loader in this tree yet;
  P0 adds thin readers where a cost term needs them (dropsondes and SFMR,
  irradiance, radar QPE).
- **Grading.** `woof obs scorecard` (RMSE, MAE, bias, FSS against ASOS and
  MRMS; [obs-scorecard.md](../obs-scorecard.md)), the hex referees
  (Stage IV, MRMS, ASOS, sondes, WIS2 wind) and `tools/da_*score*.py`.
  These score finished runs; P0's cost function reuses their pairing logic
  but must be device-resident and per-step.

## 2. How "gradients against observations" is posed

The standard form is a 4D-Var cost on a control vector `x0` (initial state,
or parameters, or NN weights):

```
J(x0) = 1/2 (x0 - xb)^T B^-1 (x0 - xb)
      + 1/2 sum_t (H_t(M_0->t(x0)) - y_t)^T R_t^-1 (H_t(M_0->t(x0)) - y_t)
```

and the gradient is
`grad J = B^-1 (x0 - xb) + sum_t M^T H_t^T R_t^-1 d_t`, i.e. one backward
(adjoint) integration of the model `M` per gradient. Choices for woof:

- **Observations first used.** Surface T2/Td/wind (ASOS, WIS2 land),
  radiosondes (IGRA/WIS2 on 16 levels), aircraft (MADIS), GNSS-RO
  refractivity, GOES CWP. These have smooth H. Radar reflectivity and
  radial velocity come later: their H is non-smooth and their information
  lives in microphysics, which is exactly the non-differentiable part.
  Satellite radiances need an RTM in the observation operator; `radtran`
  is a candidate but needs a Jacobian (its own adjoint), so it is out of
  scope until P4.
- **Window.** 1-6 h for anything adjoint-based (cost, nonlinearity and
  tangent-linear validity at convective scales all argue short); 6-24 h
  only for finite-difference and ensemble sensitivity, which do not
  linearise.
- **Control variables.** P0: a handful of parameters (the entries of
  `physics_params_registry_v1.json`, SPP-style multipliers) or a few
  low-dimensional initial-state perturbation patterns. P4: the full
  initial state (needs a B model; the LETKF ensemble gives a sampled B as in
  4DEnVar) or NN weights.
- **4-D operator.** The present operators read one state. A cost over a
  window needs First-Guess-at-Appropriate-Time (apply H at the model step
  nearest each obs time) as the first step, and time interpolation between
  steps later.

## 3. Gradient options

### (a) Hand-written discrete adjoints (WRFPLUS / MPAS-JEDI style)

- **What exists elsewhere.** WRFPLUS is WRF's tangent-linear and adjoint,
  regenerated with Tapenade from the WRF model and integrated into WRFDA
  4D-Var (Zhang, Huang and Pan 2013; Zhang et al. 2014). Since WRF v4 it
  ships inside the WRF source tree (`wrftladj/`, built with
  `./configure wrfplus`), so it is covered by WRF's public-domain notice
  (`licenses/LICENSE-WRF-public-domain.txt`). It supports a single regional
  ARW domain, no nests and no adaptive time step, and only simplified
  physics: surface drag, large-scale condensation (`mp_physics = 98`) and a
  simplified cumulus (`cu_physics = 98`).
- MPAS-JEDI has **no** tangent-linear or adjoint model; its variational
  options are 3DVar, 3D/4DEnVar and hybrids (Liu et al. 2022). So hex has no
  external oracle.
- **Use in woof.** Port WRFPLUS's TL/AD of the dry dynamics (and later its
  simplified physics) to `.cu` kernels, kernel by kernel, against WRFPLUS
  built on pb as the Fortran oracle, in the same way woof's primal kernels
  were ported against WRF. Licence: compatible. Tapenade itself is
  MIT-licensed since 3.16 (Oct 2020), so it may be used as a generator.
- **Cost.** *Estimate:* an adjoint step costs 2-4 x the forward step
  (it reruns the forward arithmetic for its linearisation state and adds
  the transposed operations); memory is dominated by checkpoints
  (option e).
- **Effort.** L: the WRF dry core is 93 launches per step; the full
  dynamics 250. Moist transport with positive-definite/monotonic limiters
  needs the treatment in (f).
- **Exactness.** Exact adjoint of the discrete tangent-linear code; tested
  by the dot-product test to 1e-12 in binary64.
- **Identity.** No primal kernel is touched; adjoint kernels are new files.

### (b) Source-to-source or compiler AD of the `.cu` kernels

- **Enzyme** differentiates LLVM IR, including CUDA kernels
  (Moses et al. 2021, SC21), within an order of magnitude of the primal
  cost on their benchmarks. Constraints for woof:
  - Enzyme needs clang-CUDA (offline) compilation to LLVM IR; woof compiles
    with NVRTC at runtime. Enzyme output would be PTX loaded through
    `RawModule`, which bypasses woof's numerics-option rewriting and the
    rounding audit (`tools/cuda_rounding_check.py`) for those kernels.
  - **Metal: no route.** Apple's Metal toolchain is not an LLVM pipeline
    Enzyme can plug into, and woof's Metal path translates CUDA *source*
    with libclang, so a PTX artefact has nothing to translate. An
    Enzyme-only adjoint would be CUDA-only.
  - Use: as a generator and cross-check for hand-written adjoints on pb, not
    as a runtime dependency.
- **Tapenade on the kernels' C.** Tapenade handles C (and its maintainers
  describe CUDA support). A Tapenade-generated adjoint emitted as C/CUDA
  *source* can go through the same libclang translator to Metal, which is
  the property Enzyme lacks. Expect manual cleanup (WRFPLUS-Chem reported
  substantial manual intervention). Effort M per scheme, less for small
  kernels.

### (c) Re-expressing the step in PyTorch/JAX with custom autograd

- **Full rewrite.** Not worth it for metal-woof. 314k lines of kernels, and
  the result would be a second model whose words drift from the
  WRF-identical one. Two published efforts show the scale:
  - PyWRF (Chu et al. 2026) rebuilt the WRF dynamical core and WSM6 in
    PyTorch and trained a column correction online, using truncated
    backpropagation and local gradient stopping.
  - WRF-JAX (Yuan et al. 2026, arXiv 2610.35095) reimplemented WRF 4.5's
    dynamics and the CONUS suite. It reports RMSE ratios of 1.05-1.07
    against WRF's own numerical-difference baseline at 24 h (statistical
    consistency, not identity), 175 s per model hour for 700x700 at 3 km on
    one A800, and directional-derivative errors of 2.6-10 % against finite
    differences through 100 full-physics steps. That last figure is the
    non-differentiable physics showing through (f).
- **Wrapping woof's kernels instead.** Each woof operation `y = f(x)` gets a
  `torch.autograd.Function` (CUDA) and an `mx.custom_function` with a
  registered `vjp` (Metal) whose backward launches a hand-written adjoint
  kernel. This keeps the forward words woof's own. Constraints:
  - woof's kernels write in place; autograd needs out-of-place semantics or
    saved copies. Wrappers allocate outputs (or save inputs) only on the
    gradient path.
  - CUDA: CuPy and PyTorch exchange device arrays zero-copy through DLPack
    (`torch.from_dlpack`), with an explicit stream handoff.
  - Metal: woof arrays under the facade already are `mx.array`s, so MLX
    autograd is the natural host. MLX documents the `custom_function` plus
    `metal_kernel` pattern for forward and backward kernels; an open issue
    (ml-explore/mlx #4580) reports that custom `jvp` and `vmap` rules are
    ignored when the body calls a Metal kernel, while `vjp` works. woof's
    in-place launcher is its own primitive (`launch.cpp`), so the wrapper
    either calls the adjoint through that primitive inside a `vjp` rule or
    implements `vjp` on a C++ primitive. torch-MPS cannot share MLX buffers
    without a copy; PyTorch on a Mac is for host-side training only.
  - `torch.compile`/Triton add nothing for woof's own kernels (they are
    already hand-written CUDA); they matter only for the NN components.
- **Recommendation.** Use wrappers for the column scheme(s) and the NN
  components (P1, P2); never re-express the dycore as framework ops.

### (d) Finite differences, tangent-linear and ensemble sensitivities

- **Finite differences.** `dJ/dp ~ (J(p+e) - J(p-e)) / 2e` per control
  direction: 2 forecasts each. woof's restart is byte-exact (a resumed run
  matches an uninterrupted one on CUDA and Metal), and `woof branch` starts
  from a checkpoint, so FD runs share their spin-up and differ only by the
  perturbation. Cost: 2 runs per direction, so only for few parameters.
  Noise floor: strict numerics gives deterministic words, so the FD
  difference is model sensitivity, not run-to-run noise; still, at
  convective scales `e` must be chosen against chaotic growth (measure it
  in P0).
- **Ensemble sensitivity** (Ancell and Hakim 2007): `dJ/dx0 ~ cov(J, x0) /
  var(x0)` from an ensemble, with no adjoint. woof has batched ensembles
  (`woof/ensemble/batch_*`), per-member H(x) inside the LETKF (`Yb`), and
  GPU covariance machinery. **Ensemble forecast sensitivity to
  observations** (EFSO; Kalnay et al. 2012) uses the LETKF's own analysis
  weights to attribute a forecast-error reduction to each observation,
  again with no adjoint; it is the direct answer to "which observations
  helped this forecast".
- Cost: an N-member forecast (woof batches members per GPU). Exactness:
  linear-regression estimate, sampling noise ~1/sqrt(N), localisation
  needed.
- This is P0. It is the only option that answers the user's question within
  weeks.

### (e) Checkpointing schedules for long windows

Storing every intermediate of a 24 h run is impossible (section 1.3: a
full-physics step's scratch is GBs; 4,800 steps). Binomial checkpointing
(revolve; Griewank and Walther 2000) reverses `N` steps with `c` stored
states and at most `r` recomputation sweeps when `N <= C(c+r, r)`.

*Estimates* (checkpoint sizes from 1.3; a per-step tape of about 4 x the
checkpoint while one step is reversed):

| case | steps (24 h) | checkpoint | c, r | snapshot memory | cost per gradient |
|---|---|---|---|---|---|
| WRF 200x200x50, 3 km, dt 18 s | 4,800 | 0.2 GB | 20, 4 (C=10,626) | 4 GB | ~4 forward + 1 adjoint sweep: ~7-9 x forward |
| same | 4,800 | 0.2 GB | 60, 3 (C=39,711) | 12 GB | ~6-8 x forward |
| WRF 200x200x50, 12 km, dt 72 s | 1,200 | 0.2 GB | 20, 3 (C=1,771) | 4 GB | ~6-8 x forward |
| hex 100k cells x 55, 15 km, dt 90 s | 960 | 0.45 GB | 10, 4 (C=1,001) | 4.5 GB | ~7-9 x forward |

Wall-clock, *estimate*: the 3 km WRF row's forward run is about 4,800 x
0.24 s = 19 min on the A6000 without radiation steps, so one gradient is
about 2.5 h, and a 30-iteration minimisation about 3 days. That is not a
usable 4D-Var. A 6 h window at 12 km (300 steps) is about 1.2 min forward
and 10 min per gradient: usable. Hence the windows in section 2.

Snapshots can sit in device memory, pinned host memory (pb 125 GB; the
Mac's 128 GB is unified) or NVMe, as multi-level checkpointing. woof's
restart serialisation (`restart_state`/`restore_restart_state`, byte-exact)
defines what a snapshot must contain; the scheduler itself is new
(`woof/grad/checkpoint.py`).

### (f) Non-differentiable physics

Sources of non-differentiability in woof's schemes: `max/min` clipping and
positive-definite/monotonic limiters in transport, saturation adjustment,
thresholds in microphysics (autoconversion onsets, `qc0`-type cut-offs),
convective triggers (KF, GF, Tiedtke), PBL regime switches (stable/unstable
branches in YSU/MYNN surface layers), lookup tables, and `if`-branches on
time (radiation cadence). Options:

1. **Differentiate the branch taken** (what Tapenade/Enzyme produce). The
   adjoint is exact for the realised branch and wrong across switches;
   fine for dot-product tests, misleading for long windows. WRF-JAX's
   2.6-10 % directional-derivative error is this effect.
2. **Simplified linearised physics**, WRFPLUS's route: TL/AD exist only for
   surface drag, large-scale condensation and a simplified cumulus; the
   nonlinear trajectory still uses full physics. woof should port exactly
   that set first (it has an oracle).
3. **Smoothing** for the gradient only: replace `max(x,0)` by a softplus
   with a scale tied to the variable's noise, step triggers by sigmoids,
   in the adjoint kernels only. The forward stays the true scheme.
4. **Surrogates.** Train a smooth NN emulator of a scheme on woof's own
   tendencies (P2 data) and use its autograd Jacobian as the scheme's
   adjoint. This is the bridge between the two halves of this plan.
5. **Freeze.** Treat a scheme's tendency as a fixed forcing in the adjoint
   (zero Jacobian): the simplest correct-by-construction choice for
   radiation on its cadence.

### (g) Verification of gradients

- **TL test:** `||M(x + a dx) - M(x)|| / ||a TL(dx)|| -> 1` as `a -> 0`,
  over at least three decades of `a`.
- **Dot-product (adjoint) test:** `<TL dx, dy> = <dx, AD dy>` for random
  `dx, dy`, to about 1e-12 relative in binary64 builds (1e-5 in binary32),
  per kernel, per step, and per window. WRFDA's guidance applies: "the
  tangent linear model might be imperfect; the adjoint test must be
  perfect".
- **Gradient (Taylor) check:** `(J(x + a h) - J(x)) / (a <grad J, h>) -> 1`.
- **Cross-backend:** Metal and CUDA adjoint agree within the float tolerance
  for each kernel (not bitwise).
- **Oracle:** WRFPLUS on pb for the WRF-route kernels, at the same inputs.

## 4. The hybrid physics/NN path

### 4.1 Where NN components go

- **Replace a scheme's tendency** (e.g. a learned PBL or cumulus) or **add a
  learned residual** to the held physics tendencies (a correction model,
  as PyWRF and NeuralGCM do). Both are column operations.
- Seats, already on both routes:
  - WRF route: the `after_nonmicrophysics` seat where SPPT/SKEBS modify
    `PhysicsTendencies` (`dycore.py:5192`); for replacing a whole scheme,
    the callable-injection path (`initialize_physics(radiation=...,
    cumulus=...)`) extended to PBL.
  - hex: the held-source seat where `StochasticPhysics.apply`, IAU and
    nudging act (`runtime_v842.py:317`); for a whole backend, the
    `PhysicsBackend` protocol and the `woof.hex.rows` entry points.
  - Both: the generic column seam's phase-1 output, which already is the
    `[level, column]` device layout an NN wants.
- Microphysics is the hardest to replace (in-place phase 2, positivity,
  sedimentation, `h_diabatic` coupling) and the last to try.

### 4.2 Interfaces

A small `woof/nn/` package (no import on the default path):

```python
class ColumnModel(Protocol):
    inputs: tuple[str, ...]          # names in the seam's vocabulary, with units
    outputs: tuple[str, ...]         # du, dv, dtheta, dqv, ... (per second)
    levels: int
    def __call__(self, x: Array) -> Array: ...   # [features, column] -> [outputs, column]
```

- **PyTorch on CUDA:** a TorchScript or `torch.export` artefact, fed by
  DLPack views of the seam's CuPy arrays on woof's stream; no host copies.
- **MLX on Metal:** an `mlx.nn.Module` with saved weights, fed the facade's
  `mx.array`s directly.
- **ONNX** as the portable interchange (one artefact, both runtimes) is
  worth a look but is not required.
- Normalisation statistics, input bounds, variable list, level count and
  a SHA-256 of the weights travel with the artefact; the scheme's
  `restart_identity` carries the hash, so `woof/checkpoint_identity.py`
  refuses a resume onto different weights.

### 4.3 Training: offline, online, differentiable online

- **Offline (tendency learning).** Train on (column state -> scheme
  tendency) pairs. woof can produce these itself: an opt-in "tendency tap"
  that writes the seam's phase-1 inputs and outputs per scheme (the
  ClimSim pattern; `woof ml-export` today writes history fields on
  pressure levels, not tendencies). It can also produce coarse-grained
  truth: run a 1 km or LES nest, coarse-grain to 9 km, and learn the
  subgrid tendency, which is how ClimSim and Yuval and O'Gorman obtained
  targets. Cheap and well-trodden; the known failure is online instability
  of models that score well offline (Rasp et al. 2018; Yuval and O'Gorman
  2020, 2021; ClimSim-Online's MLP baseline was unstable while its U-Net
  ran stably).
- **Online, derivative-free.** Run the hybrid model, score against
  observations with P0's cost, and tune a few NN or physics parameters by
  ensemble Kalman inversion: no adjoint, works now.
- **Differentiable online (end-to-end).** Backpropagate the obs cost through
  the dycore into the NN weights, as NeuralGCM does (whole model in JAX)
  and PyWRF does (truncated backprop through a PyTorch WRF). In woof this
  needs P3's dycore adjoint, the column NN's autograd, the adjoint of the
  coupling (how held tendencies enter the RK stages, which is linear), and
  P4's checkpointing. Short windows only.

### 4.4 Stability and conservation

- Predict **fluxes** at interfaces rather than tendencies where possible, so
  the column integral of water and energy closes by construction; or apply
  a column fixer (scale the moisture tendency so precipitation plus
  column-integrated `dq` matches the surface fluxes).
- Positivity: limit the NN's moisture tendency so `q + dt * dq >= 0`, as
  woof's moist transport does for its own fields.
- Bounds: refuse (or fall back to the physics scheme for) columns outside
  the training input range; record the fallback count per step.
- Hu et al. (2025) and ClimSim-Online report that physical constraints
  materially improve hybrid stability; measure it here with the same
  gates woof uses for any new option: multi-day hex runs on uniform **and**
  variable-resolution meshes (`tools/hex_varres.py`), the mass-budget
  accumulator (`MassFluxAccumulator`), and the obs scorecard against the
  physics-only baseline.

### 4.5 Coexisting with the identity machinery

- NN options are **new selectors outside WRF's ranges** (as
  `woof/microphysics_schemes.py` does with IDs >= 900) or a separate
  `[nn]` config table; default absent. A configuration that does not name
  one never imports torch or `mlx.nn` and launches nothing extra.
- Registry rows enter at `experimental-runtime` and are labelled
  non-identical in the registry's evidence axes and in `physics_mode`'s
  divergence ledger, like other fork forms.
- Identity proofs (`GPUWM_WRF_EXACT=1`, `physics.mode = exact`) refuse NN
  options by name.
- No pinned file changes: the seams are read, not edited, except for a
  dormant hook call whose absence is proven by the identity tests in
  section 8.

### 4.6 Inference cost, *estimates*

A column MLP with about 600 inputs (10 variables x 60 levels) and four
512-wide hidden layers is about 1.5 M parameters and 3 MFLOP per column.

| case | columns | FLOP per call | A6000 (38 TFLOP/s fp32 peak, ~30 % achieved) | M3 Max GPU (~14 TFLOP/s fp32, ~30 %) |
|---|---|---|---|---|
| WRF 200x200 | 40,000 | 0.12 TFLOP | ~10 ms (vs 239 ms per step) | ~30 ms |
| hex 100k cells | 100,000 | 0.3 TFLOP | ~25 ms | ~70 ms |

Called at the PBL/cumulus cadence (every step, or every few), this is a few
per cent to ~15 % of a step on CUDA and more on Metal, where woof's own
step is about 4x slower. Measuring this is a P2 gate.

### 4.7 Related work (verified references)

- **NeuralGCM** (Kochkov et al. 2024, Nature 632, 1060-1066,
  [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)):
  differentiable spectral dycore (Dinosaur,
  [github.com/neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur))
  with learned column physics, trained end to end in JAX.
- **JCM / JAX-GCM** (GMD 19, 6451-6466, 2026;
  [github.com/climate-analytics-lab/jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)):
  physics coupled to Dinosaur through a swappable interface; jit/grad/vmap.
- **SpeedyWeather.jl with Enzyme**, in DJ4Earth (Moses et al. 2026, JAMES,
  [doi:10.1029/2025MS005615](https://doi.org/10.1029/2025MS005615)):
  Enzyme plus Checkpointing.jl through a spectral GCM after minor changes to
  state and scratch handling.
- **Enzyme for GPU kernels** (Moses et al. 2021, SC21,
  [doi:10.1145/3458817.3476165](https://doi.org/10.1145/3458817.3476165)).
- **PyWRF** (Chu, Cao and Wang 2026, J. Meteor. Res. 40, 653-668,
  [doi:10.1007/s13351-026-5154-1](https://doi.org/10.1007/s13351-026-5154-1)):
  WRF dycore and WSM6 in PyTorch with an online-trained column correction.
- **WRF-JAX** (Yuan et al. 2026,
  [arXiv:2610.35095](https://arxiv.org/abs/2610.35095)): WRF 4.5 dynamics
  and CONUS physics in JAX; figures quoted in 3(c).
- **WRFPLUS** (Zhang, Huang and Pan 2013, J. Atmos. Oceanic Technol.;
  Zhang et al. 2014, BAMS); WRFDA user's guide for the simplified physics.
- **MPAS-JEDI** EnVar (Liu et al. 2022, GMD 15, 7859-7878,
  [doi:10.5194/gmd-15-7859-2022](https://doi.org/10.5194/gmd-15-7859-2022)).
- **ClimSim** (Yu et al., NeurIPS 2023 Datasets and Benchmarks,
  [arXiv:2306.08754](https://arxiv.org/abs/2306.08754)), ClimSim-Online,
  and Hu et al. (2025, JAMES,
  [doi:10.1029/2024MS004618](https://doi.org/10.1029/2024MS004618)):
  stable multi-year E3SM-MMF hybrids coupled through FTorch.
- **FTorch** (Atkinson et al. 2025, JOSS,
  [doi:10.21105/joss.07602](https://doi.org/10.21105/joss.07602)).
- **Stable ML parameterisation:** Rasp, Pritchard and Gentine 2018 (PNAS,
  [doi:10.1073/pnas.1810286115](https://doi.org/10.1073/pnas.1810286115));
  Yuval and O'Gorman 2020 (Nat. Commun. 11, 3295, random forest,
  [doi:10.1038/s41467-020-17142-3](https://doi.org/10.1038/s41467-020-17142-3));
  Yuval, O'Gorman and Hill 2021 (GRL, NN,
  [doi:10.1029/2020GL091363](https://doi.org/10.1029/2020GL091363)).
- **Emulators / surrogates:** FourCastNet (Pathak et al. 2022,
  [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)); Pangu-Weather (Bi
  et al. 2023, Nature 619, 533-538,
  [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3));
  GraphCast (Lam et al. 2023, Science 382, 1416-1421,
  [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336));
  ACE (Watt-Meyer et al. 2023, [arXiv:2310.02074](https://arxiv.org/abs/2310.02074))
  and ACE2 ([arXiv:2411.11268](https://arxiv.org/abs/2411.11268)). These are
  whole-model emulators, differentiable for free; as surrogates they give
  cheap but approximate sensitivities (an emulator's Jacobian is not the
  model's) and are out of scope except as a comparison.
- **Sensitivity without adjoints:** Ancell and Hakim 2007 (MWR 135,
  4117-4134, [doi:10.1175/2007MWR1904.1](https://doi.org/10.1175/2007MWR1904.1));
  Kalnay et al. 2012 (Tellus 64A, 18462,
  [doi:10.3402/tellusa.v64i0.18462](https://doi.org/10.3402/tellusa.v64i0.18462)).
- **Checkpointing:** Griewank and Walther 2000, revolve (ACM TOMS 26, 19-45,
  [doi:10.1145/347837.347846](https://doi.org/10.1145/347837.347846)).
- **Tools:** Tapenade (MIT since 3.16,
  [gitlab.inria.fr/tapenade/tapenade](https://gitlab.inria.fr/tapenade/tapenade));
  MLX `custom_function`
  ([docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html),
  [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
  [issue #4580](https://github.com/ml-explore/mlx/issues/4580)).

A "torch-wrf" project as such was not found; PyWRF is the PyTorch
counterpart found.

## 5. Phased plan in metal-woof

New code lives under `woof/grad/` (costs, sensitivities, adjoint wrappers,
checkpointing) and `woof/nn/` (column models). Adjoint kernels live beside
them (`woof/grad/kernels/*.cu`), go through the same Metal translator,
and are registered in the rounding-audit inventory like any new kernel.
Effort is in focused engineer-weeks (w).

### P0. Observation cost and adjoint-free sensitivities (4-6 w)

| PR | content | effort |
|---|---|---|
| P0.1 | `woof/grad/obs_cost.py`: `J_o` over a window from the existing operators (pressure-level, surface, RO refractivity, CWP), FGAT pairing at model steps, diagonal R from the type table; device-resident. Readers for the `bkr/obs` stores it needs. Tests on hand-built states. | 1-1.5 w |
| P0.2 | Trajectory sizes: record restart-payload bytes per route and grid (replaces 1.3's estimates). | 0.5 w |
| P0.3 | `woof grad fd`: central-difference sensitivities of `J_o` to named parameters and to a few initial-state patterns, via `woof branch` from a shared checkpoint; strict-numerics noise floor measured. | 1 w |
| P0.4 | `woof grad ens`: ensemble sensitivity `cov(J, x0)/var(x0)` with localisation, on batched members; EFSO from the LETKF weights. | 1.5-2 w |

**Gate G0:** FD and ensemble sensitivities agree in sign and pattern for at
least three directions on one regional case, on both routes, uniform and
variable-resolution meshes; `J_o` reproduces the scorecard's RMSE for the
same pairs. If G0 shows sensitivities dominated by noise at 3 km, stop
gradient work at convective scales and keep P1-P4 at 9-15 km.

### P1. Tensor bridge and one differentiable column scheme (4-6 w)

| PR | content | effort |
|---|---|---|
| P1.1 | `woof/grad/bridge.py`: CuPy<->torch via DLPack with stream handoff; facade `mx.array` passthrough on Metal; documented copy path for torch-MPS. Zero-copy tests. | 0.5-1 w |
| P1.2 | Kessler (`mp_physics = 1`) TL and AD kernels, templated on real type, with `torch.autograd.Function` and `mx.custom_function` wrappers. Kessler first: small, oracle-exact already (`test_kessler_wrf461_parity`), and it has the threshold and saturation-adjustment cases of 3(f). | 2-3 w |
| P1.3 | Gradient test kit: TL convergence, dot-product, Taylor check, CUDA vs Metal tolerance; binary64 test builds. | 1 w |
| P1.4 | Second scheme with a smooth core: a surface layer (revised MM5) or WRFPLUS's large-scale condensation (has an oracle). | 1-2 w |

**Gate G1:** dot-product to 1e-12 (binary64) and 1e-5 (binary32) on both
backends; adjoint kernel at most 3x the forward kernel's time; identity
tests unchanged. If the Metal wrapper cannot run the adjoint through
woof's in-place primitive, the Metal leg uses out-of-place copies and the
gate records the overhead.

### P2. NN column physics, inference only, opt-in (4-6 w; parallel with P1)

| PR | content | effort |
|---|---|---|
| P2.1 | `woof/nn/` protocol and loaders (TorchScript/`torch.export` on CUDA, `mlx.nn` on Metal), artefact manifest with weight hash, input bounds and fallback. | 1 w |
| P2.2 | Tendency tap: opt-in writer of the seam's per-scheme inputs and tendencies (Zarr/Icechunk, like `ml-export`). | 1 w |
| P2.3 | WRF-route seat: an `[nn]` residual or replacement applied at the `after_nonmicrophysics` seat; registry row, restart identity, refusal under exact mode. | 1-1.5 w |
| P2.4 | hex seat: the same on held sources, and a `woof.hex.rows` example backend. | 1 w |
| P2.5 | Reference model: an offline-trained PBL residual on woof's own tendencies (and a coarse-grained LES/nest target), with the stability gates. | 1-2 w |

**Gate G2:** with the option absent, identity tests unchanged (section 8);
with it present, 5-day hex runs stable on uniform and variable-resolution
meshes, mass budget closed to the physics-only run's level, obs scorecard
not worse than the physics baseline at the 95 % bootstrap level, inference
cost measured on both backends.

### P3. A dynamical-core adjoint (12-20 w)

- **P3a, WRF dry core** (93 launches per step): TL/AD kernels ported from
  WRFPLUS (`wrftladj/`) built on pb, kernel by kernel, with WRFPLUS as the
  Fortran oracle at `-O0 -ffp-contract=off`. Acoustic substeps, RK stages,
  big-step tendencies, boundary relaxation. No limiters (`moist_adv_opt =
  0`) at first. About 8-12 w.
- **P3b, moist transport and WRFPLUS's simplified physics** (surface drag,
  `mp_physics = 98`, `cu_physics = 98`), and the coupling of held physics
  tendencies into the stages. About 4-6 w.
- **P3c, hex dry core:** only after P3a works, and without an external
  oracle (MPAS-JEDI has none); dot-product and Taylor tests are the only
  proof. Defer unless hex is the route the user wants to assimilate on.

**Gate G3:** per-kernel and per-step dot-product tests pass in binary64;
TL against nonlinear perturbation growth over 1-6 h matches as WRFPLUS's
checks do; adjoint step at most 4x the forward step; WRFPLUS oracle
agreement per kernel. A no at G3 (adjoint too slow, or TL invalid beyond
an hour at the target resolution) ends the plan at P2 plus P0's
sensitivities.

### P4. Differentiable windows, 4D-Var-lite and online hybrid training (8-12 w)

| PR | content | effort |
|---|---|---|
| P4.1 | `woof/grad/checkpoint.py`: revolve-style binomial schedule over device, pinned host and NVMe snapshots, reusing the restart payload definition. | 2 w |
| P4.2 | Window gradient `grad J_o` with respect to the initial state, then an incremental minimiser (L-BFGS or CG) with a sampled B from the LETKF ensemble (4DEnVar-like preconditioning). 1-6 h windows at 9-15 km. | 3-4 w |
| P4.3 | Online hybrid training: backprop the obs cost through P3's adjoint into a P2 column model; truncated windows as PyWRF does. | 3-4 w |
| P4.4 | Optional: radiance cost term with `radtran` once it has a Jacobian. | later |

**Gate G4:** a 6 h, 12 km window gradient passes the Taylor test; a short
4D-Var-lite reduces `J_o` on independent observations relative to the
LETKF analysis on the same case; online-trained NN not worse than its
offline-trained parent on the scorecard.

### What is not feasible, or not worth it

- **A PyTorch/JAX woof.** Re-expressing 314k lines of kernels as framework
  ops would create a second model; WRF-JAX and PyWRF show it is possible and
  also that the result is statistically, not bitwise, WRF. metal-woof's
  value is the identical model, so it keeps its kernels and adds adjoints.
- **Adjoints of every physics scheme.** Hundreds of schemes, most with
  switches; follow WRFPLUS and use simplified linearised physics, smoothing
  or NN surrogates.
- **Enzyme on Metal.** No toolchain route; and on CUDA it bypasses NVRTC,
  the numerics rewrite and the Metal translator. Generator/oracle use only.
- **24 h, 3 km 4D-Var.** About 2.5 h per gradient on the A6000 (section
  3e); tangent-linear validity at convective scales is also far shorter
  than 24 h. Use ensemble sensitivity for 24 h questions.
- **Differentiating the soft-float/df64 binary64 path on Metal.** It is
  integer code by design; gradients run in native float.
- **A hex adjoint before the WRF one.** No oracle exists for it.

## 6. Risks

| risk | effect | mitigation |
|---|---|---|
| Adjoint drift from primal as kernels change | wrong gradients silently | dot-product and Taylor tests per kernel in CI; adjoint kernels hash-bound to their primal source (as `engine-helper-sets.v1.json` binds helper sources) so a primal edit fails the adjoint's test until re-derived |
| Tangent-linear validity at km scales | gradients meaningless beyond ~1-3 h | measure in G0/G3; keep windows short; ensemble sensitivity for longer |
| Metal autograd limits (MLX #4580, in-place primitive) | Metal adjoint slower or CUDA-only at first | out-of-place fallback; record the overhead in G1 |
| NN online instability | crashes or drift | constraints, bounds-with-fallback, variable-resolution stability gates |
| Memory at large domains | checkpoint pressure | multi-level snapshots; coarser outer-loop grids |
| Licence | none expected: WRFPLUS public domain, Tapenade MIT, PyTorch BSD, MLX MIT | still record each in `licenses/` and `PROVENANCE.md` when code lands; nothing LGPL or NVIDIA-derived |

## 7. Effort summary

| phase | effort | depends on | stop condition |
|---|---|---|---|
| P0 | 4-6 w | none | G0 |
| P1 | 4-6 w | P0.1 for the cost | G1 |
| P2 | 4-6 w | none (parallel) | G2 |
| P3 | 12-20 w | P1 test kit | G3 |
| P4 | 8-12 w | P2, P3 | G4 |

P0 + P1 + P2 (about 3-4 months of one person, partly parallel) deliver
observation gradients for parameters and patterns, one differentiable
scheme on both backends, and opt-in NN physics. P3 + P4 (about 5-8 more
months) deliver short-window variational gradients and online hybrid
training on the WRF route.

## 8. The byte-identical default path

Every PR in this plan must show that a configuration not naming a
gradient or NN option produces the same words as main:

- identity tests run in child processes under **both** `fast` and `strict`,
  and `GPUWM_WRF_EXACT=0/1`, with WRF-like weights and poisoned workspaces;
- the Metal gates (Kessler, WSM6, YSU parity; `tools/f64_soft_grade.py`;
  `tests/test_metal_launch_ordering.py`; `tests/test_backend_parity_ledger.py`;
  `tests/test_uwpbl_driver_seam.py`) and the CUDA rounding audit
  (`tests/test_cuda_rounding_check.py`, rebinding rows for new kernels);
- no import of `torch`, `mlx.nn` or `woof.grad` on the default path (an
  import-graph test);
- no extra kernel launch on the default path (launch counts compared with
  CUPTI on pb and `WOOF_METAL_TRACE` on the Mac).
