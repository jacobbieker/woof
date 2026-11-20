# P3a: the WRF dry-core tangent-linear and adjoint (design note)

Part of [DIFFERENTIABLE-HYBRID-PLAN.md](DIFFERENTIABLE-HYBRID-PLAN.md),
section 3(a), 3(e), 3(g), P3 and gate G3. 4D-Var in
[WRFDA-3DVAR-PLAN.md](WRFDA-3DVAR-PLAN.md) (section 1.12, `var4d`) needs
this same tangent-linear (TL) and adjoint (AD) model, so the interfaces
in section 6 serve both. This note adds no code. Each later PR ports one
kernel or a few.

## 1. What is being linearised

- **Route:** the WRF route, dry, with `moist_adv_opt = 0`, specified or
  periodic lateral boundaries, one domain, no nests and a fixed `dt`.
  These are WRFPLUS's own limits.
- **Routine boundaries:** woof's strict sequence (`GPUWM_WRF_EXACT=1`).
  It keeps WRF's routine boundaries (`small_step_prep`, `advance_uv`,
  `advance_mu_t`, `advance_w`, `small_step_finish`, ...), so each TL/AD
  kernel maps to one WRFPLUS routine and can be checked against it alone.
  The fast mode fuses some of these steps. Its trajectory still serves
  as the linearisation point (section 3), because the two modes differ
  only by roundoff.
- **Vertical coordinate.** WRFPLUS 4.7.1's dynamics TL/AD has no hybrid
  coordinate. `g_advance_uv`, `g_advance_mu_t`, `g_rhs_ph` and the rest
  use `muu`/`mu`, not `c1h*muu+c2h`; `c1h/c2h` only pass through
  `solve_em_tl` and `first_rk_step_part*`.
  - woof's TL/AD follows woof's primal, so it carries `c1h/c2h`.
  - It is checked against WRFPLUS at `hybrid_opt = 0` (`c1h = 1`,
    `c2h = 0`), where the two coincide.
  - At `hybrid_opt = 2`, only the dot-product, Taylor and TL-vs-NL tests
    cover it. WRFPLUS gives no evidence there.

## 2. Kernel inventory

**Measured on pb (CUDA).** This is one step of the `dry` suite in
`tools/profile_launch_overhead.py` (periodic, flat, 40x40x30,
`time_step_sound = 4`, so 1 + 2 + 4 acoustic substeps), traced with
`tools/launch_trace.py`:

- **77 raw-kernel launches in fast mode and 78 in strict mode.**
- CUPTI's 93 (plan 1.4) also counts CuPy elementwise kernels and copies.

The per-step strict sequence:

```
rk_copy_words
per stage (x3):
  rk_zero_words  calc_p_alpha[_carried_muts]
  flux_div_scalar flux_div_u flux_div_v flux_div_w
  slow_pgf slow_buoyancy slow_geopotential
  small_step_init_uv small_step_init_column  calc_coefs x2
  per substep (1, 2, 4):  advance_uv advance_mu_th advance_w_phi x2
  small_step_finish_uv small_step_finish_column
calc_p_alpha_carried_muts set_surface_w health_partial health_final
```

A real leaf adds the following:

| woof kernel | file |
|---|---|
| `coriolis_curvature` | `coriolis_map.cu` |
| `w_damp` | `openbc.cu` |
| `specified_relaxation` | `spec_bdy.cu` |
| `state_specified_relaxation`, `state_relaxation_multi`, `finalize_state_*` | `lbc_state.cu` |
| `evaluate_linear_boundary` | `lbc_time.cu` |
| `zero_gradient_w`, `flow_dependent_batch` | `lbc_flow.cu` |
| diffusion and diff6, when chosen | |

The map below runs from WRF routine (`dyn_em`), to WRFPLUS TL/AD routine
(`wrftladj/`), to woof primal kernel(s), to the P3a PR group. The TL/AD
routines are named `g_*` (TL) and `a_*` (AD).

**A: acoustic substep**

| WRF routine | WRFPLUS TL/AD (file) | woof primal |
|---|---|---|
| `advance_uv` | `g_/a_advance_uv` (small_step) | `advance_uv` |
| `advance_mu_t` | `g_/a_advance_mu_t` | `advance_mu_th` (strict: `wrf_advance_mu_theta`) |
| `advance_w` | `g_/a_advance_w` | `advance_w_phi` |
| `calc_p_rho` | `g_/a_calc_p_rho` | inside `advance_w_phi` / `calc_p_alpha` |
| `calc_coef_w` | `g_/a_calc_coef_w` | `calc_coefs` |
| `small_step_prep` | `g_/a_small_step_prep` | `small_step_init_uv`, `small_step_init_column` |
| `small_step_finish` | `g_/a_small_step_finish` | `small_step_finish_uv`, `small_step_finish_column` |
| `calc_mu_uv_1` | `g_/a_calc_mu_uv_1` (big_step) | `average_mass_faces` |
| `spec_bdyupdate(_ph)`, `zero_grad_bdy` | `g_/a_spec_bdyupdate` (bc), `g_/a_spec_bdyupdate_ph` (bc_em), `g_/a_zero_grad_bdy` (bc) | `advance_specified_phi_w`, `zero_gradient_w` |

**B: RK stage**

| WRF routine | WRFPLUS TL/AD (file) | woof primal |
|---|---|---|
| `rk_step_prep`: `calc_mu_uv`, `couple_momentum`, `calc_ww_cp`, `calc_cq` | `g_/a_rk_step_prep` (em), big_step utilities | `average_mass_faces`, `glue_total_theta`, `rk_copy_words` |
| `calc_p_rho_phi` | `g_/a_calc_p_rho_phi` | `calc_p_alpha[_carried_muts]` |
| `rk_addtend_dry` | `g_/a_rk_addtend_dry` (em) | strict add-fixed/slow path |
| `set_w_surface` | `g_/a_set_w_surface` (bc_em) | `set_surface_w` |
| `set_physical_bc2d/3d` | `g_/a_set_physical_bc*` (bc) | periodic aliasing (`close_periodic_alias`) |

**C: big-step tendencies (`rk_tendency`)**

| WRF routine | WRFPLUS TL/AD (file) | woof primal |
|---|---|---|
| `advect_u/v/w`, `advect_scalar` (non-PD, `h/v_mom_adv_order` 2-6) | `G_/a_ADVECT_U/V/W/SCALAR` (advect) | `flux_div_u/v/w/scalar` |
| `horizontal_pressure_gradient` | `G_/a_HORIZONTAL_PRESSURE_GRADIENT` | `slow_pgf` |
| `pg_buoy_w` | `g_/a_pg_buoy_w` | `slow_buoyancy` |
| `rhs_ph` | `g_/a_rhs_ph` | `slow_geopotential[_faces/_open/_vertical]` |
| `coriolis`, `curvature` | `G_/a_CORIOLIS`, `G_/a_CURVATURE` | `coriolis_curvature` |
| `w_damp` | `g_/a_w_damp` | `w_damp` |
| `sixth_order_diffusion`, `horizontal_diffusion` (`diff_opt = 1`, `km_opt = 1`) | `g_/a_sixth_order_diffusion`, `G_/a_HORIZONTAL_DIFFUSION` | diff6, diffusion kernels (optional, last) |

**D: lateral boundaries**

| WRF routine | WRFPLUS TL/AD (file) | woof primal |
|---|---|---|
| `relax_bdy_dry`, `relax_bdytend` | `g_/a_relax_bdy_dry`, `G_/a_RELAX_BDYTEND_CORE` | `specified_relaxation`, `state_relaxation_multi` |
| `spec_bdy_dry`, `spec_bdytend` | `g_/a_spec_bdy_dry`, `g_/a_spec_bdytend` | `state_specified_relaxation` |
| `spec_bdy_final`, `zero_bdytend` | `g_/a_SPEC_BDY_FINAL`, `g_/a_zero_bdytend` | `finalize_state_*`, `evaluate_linear_boundary` |

Tables A-D have 24 rows. Counted one WRF routine at a time (the
`rk_step_prep` parts, the advection and diffusion variants, each boundary
routine), they cover about 37 WRF routines, against about 25 distinct woof
kernels. Advection is the largest by far: WRFPLUS's
`module_advect_em_ad.F` is 35.5 k lines, `small_step` AD 5 k, `bc` AD
3.5 k.

**Out of P3a:** scalar transport with PD/monotonic limiters,
`moist_adv_opt > 0`, moist `cq`, physics and `h_diabatic` (all P3b), and
`sumflux`, which only the scalars need.

## 3. Checkpointing

Two levels. Neither adds a hook to `dycore.step`.

1. **Across steps.** The forward model is woof as it runs, in float32,
   fast or strict.
   - The driver loop calls `dycore.step` and copies the step-start dry
     prognostics: `u, v, w, ph', t', mu', p', al'`, plus the static base
     state, captured once.
   - Size: about 8 fields, 64 MB per checkpoint at 200x200x50.
   - A 6 h, 12 km window (300 steps) is 19 GB. It fits in pb's pinned
     host memory, or the Mac's unified memory, with every step kept.
     Longer windows use the binomial schedule of plan 3(e), which P4.1
     owns as `woof/grad/checkpoint.py`. P3a needs only "give me the
     state at the start of step n".
   - With physics (P3b), the restart manifest's held tendencies join the
     checkpoint, as plan 1.3 says.
2. **Within a step: recompute, then tape.** This is WRFPLUS's own
   pattern: `solve_em_ad` reruns the forward routine and pushes to
   `adBuffer`.
   - From the step-start checkpoint, the TL/AD driver reruns the step's
     dry arithmetic in **binary64 (NL64)**, inside the same `g_*`
     kernels. Like Tapenade's output, those kernels advance the base
     state and the perturbation together.
   - It keeps the per-stage and per-substep arrays that the AD kernels
     read: about 80 fields per step at 7 substeps, 1.3 GB in binary64 at
     200x200x50.
   - The adjoint sweep then runs the `a_*` kernels in reverse.
   - Per step, the adjoint costs one NL64 recompute plus the transposed
     kernels.
   - The linearisation point is woof's float32 trajectory at step starts
     and NL64 inside a step. The gap between the two is roundoff, the
     same gap that WRFPLUS's TL model has against WRF's float NL run.

## 4. Precision, backends and the oracle harness

- **Kernels.** The new `.cu` units go under `woof/grad/wrfplus/kernels/`,
  one per WRF routine group. Each defines its own `typedef` real type,
  `double` by default.
  - The primal `common.cuh` is not included where it would pin
    `real = float`. Shared index macros are copied with a pointer to
    their source, so primal files are not edited.
  - Each unit holds `<routine>_nl` (NL64), `<routine>_tl` (joint base
    plus perturbation) and `<routine>_ad`.
- **Backends.**
  - **CUDA:** binary64. This is the G3 lane; the A6000 runs fp64 at
    1/32 rate, which goes into the cost ratio. A binary32 instantiation
    is also built for the 1e-5 lane.
  - **Mac:** the host backend (`WOOF_BACKEND=host`, the same `.cu` built
    as C++ at `-ffp-contract=off`) gives binary64 on the CPU. Dot-product,
    Taylor and WRFPLUS tests run there, and in hosted CI.
  - **Metal:** no fp64. It runs the binary32 instantiation. The gate
    there is a dot product to about 1e-5 and agreement with CUDA's
    binary32 within float tolerance (plan 3(g)). The soft-float df64 path
    is not differentiated (plan, "not feasible").
- **WRFPLUS oracle.**
  - **Build.** Built on pb under `/mnt/NVME2/woof-work/scratch/p3a`:
    - source: the pristine `v4.7.1.tar.gz` (sha256 `11186188...`);
    - toolchain: inside `wrf-oracle:u2404`, `./configure wrfplus`, option
      17 (serial GNU);
    - flags: `RWORDSIZE 8` (`-fdefault-real-8`) and
      `FCOPTIM = -O0 -ffp-contract=off -fno-fast-math`;
    - `./compile wrfplus` takes 6.5 min and gives `main/libwrflib.a` and
      `wrfplus.exe`.
  - **Drivers.** Per routine, a small Fortran driver (`tools/wrfplus_oracle/`)
    is linked against `libwrflib.a`. It reads a raw binary64 input record
    written by Python, which carries the arguments, the `ims:ime`
    memory-order dimensions and the config flags. It calls `g_<routine>`
    and `a_<routine>`, then writes the outputs.
  - **Inputs.** The Python side makes seeded, physically scaled synthetic
    states on small grids (for example 9x8x7, `spec_zone = 1`). It
    converts woof's `(k, j, i)` C-grid to WRF's `(i, k, j)`, and runs the
    driver in the container.
  - **Fixtures.** The fixture is `tests/data/wrfplus/<routine>.npz`. It
    holds the inputs and WRFPLUS's TL and AD outputs, is small, and holds
    no observation data. Each fixture records the WRFPLUS source digest,
    the driver digest and the flags.
  - **The test.** The woof TL/AD kernel must reproduce the fixture to
    1e-13 relative in binary64; summation order may differ from the
    Fortran. It runs on host, CUDA and Metal (Metal at binary32
    tolerance), so after the fixture is made no Fortran is needed.
    Regenerating a fixture needs pb.
- **Tests per kernel**, shared with P1's shapes:
  - TL vs NL64 finite difference, ratio to 1 over at least three
    decades;
  - dot product `<TL dx, dy> = <dx, AD dy>` to 1e-12 (binary64) and
    1e-5 (binary32);
  - WRFPLUS fixture agreement;
  - a binding test: NL64 against woof's float32 primal kernel on the
    same input, within float tolerance. This is the plan's drift
    guard: a primal edit fails this test until the TL/AD is
    re-derived.
- **Helpers.** P1's `woof/grad/checks.py` is torch-autograd based (agreed
  with the P1 worker). P3a keeps array helpers in
  `woof/grad/wrfplus/_checks.py` with P1's argument shapes: sequences of
  arrays, inner products in float64. A thin wrapper can unify them later.
- **Default path.** No primal file changes. `woof.grad.wrfplus` is
  imported only by its own tests and CLI. Each PR shows:
  - identity in child processes under fast and strict, with
    `GPUWM_WRF_EXACT=0/1`;
  - the Metal gates;
  - the rounding audit, with the new kernels registered as opt-in rows;
  - no import of `woof.grad` from the default path;
  - unchanged launch counts on the dry trace above.

## 5. Order of work

Each step is one small PR, taken to merged before the next widens.

1. `advance_uv` TL/AD, the `_checks` helpers, the oracle driver
   framework and fixture writer, and the `<routine>.npz` format. This is
   the pattern PR.
2. `advance_mu_t`, `advance_w` (with `calc_p_rho`), `calc_coef_w`.
3. `small_step_prep`, `small_step_finish`, `calc_mu_uv_1`, plus a
   **per-acoustic-substep** chained dot-product test
   (`acoustic_substep_tl/_ad`: advance_uv, advance_mu_t, advance_w in
   woof's strict order, two substeps, with a finite-difference check of
   the chain).
4. RK-stage glue: `rk_step_prep` (`couple_momentum`, `calc_ww_cp`),
   `calc_p_rho_phi`, `rk_addtend_dry`, `set_w_surface`, periodic
   physical BCs, plus a per-stage test with tendencies frozen.
   - Done in `woof/grad/wrfplus/stage.py`, with the chained
     `rk_stage_tl`/`rk_stage_ad`.
   - woof's arrays carry no halo, so the periodic physical BCs are the
     kernels' wrapped face and terrain indices. There is no separate
     operator; the oracle's halo columns carry the wrapped neighbours.
   - `rk_addtend_dry` follows woof's decomposition: the held tendf are
     already divided by the map factor, and the heating carries the
     hybrid c1h/c2h. The oracle maps WRF's tendf to woof's tendf times
     msf, at rk_step 2. The rk_step 1 save fold belongs to item 7
     (lateral boundaries).
   - Not covered yet, all for item 7: on rk_step 1 with a non-periodic
     axis, the strict `stage_fluxes(first_stage=True)` takes the
     boundary-face mu from `carried_wrf_mu_halo`, where these kernels
     clamp; `_capture_wrf_muts` has a boundary-zone branch the chain
     does not take (it forms muts = mub + mup + mu'' everywhere).
   - The chained stage runs calc_p_rho_phi, rk_step_prep,
     rk_addtend_dry, small_step_init, calc_coef_w, the substeps and
     small_step_finish, and no others. It leaves out `set_w_surface` (checked on its own),
     the lateral boundary update and `w_damp` (item 5).
5. Big-step tendencies: `horizontal_pressure_gradient`, `pg_buoy_w`,
   `rhs_ph`, `coriolis`, `curvature`, `w_damp`.
   - `horizontal_pressure_gradient` and `pg_buoy_w` are done in
     `woof/grad/wrfplus/bigstep.py`. They cover woof's strict `slow_pgf`
     and `slow_buoyancy`, dry, on periodic, open and specified axes, with
     the lid extrapolation, map factors and hybrid levels. p is the strict
     diagnosis's p'.
   - The pgf AD runs in two passes. One thread per face writes the
     face's four local adjoints (bracket, layer mass, half-level
     geopotential difference, vertical term) to scratch, and three gathers
     (cells, w levels, columns) sum them. That is 2x faster than
     recomputing the faces in the gathers.
   - Checks: dot-product test about 1e-16 and FD for 11 configurations.
     The float base state is the strict primal's words (0 differing, host
     and CUDA). WRFPLUS 4.7.1 (`tools/wrfplus_oracle/bigstep.F90`) agrees
     to about 6e-16 on 6 cases.
   - WRFPLUS's versions are not hybrid (no c1/c2), so the oracle cases
     use c1 = 1, c2 = 0. The driver forms WRF's precomputed face masses
     (calc_mu_uv) and half-level geopotential (calc_php) in the fixture,
     with their transposes.
   - WRFPLUS bug, found here: with `top_lid`, the reverse sweep of
     `A_HORIZONTAL_PRESSURE_GRADIENT` still applies the pre-2016 lid
     extrapolation (cf1-cf3 on kde-1..kde-3). Its forward sweep and the TL
     use cfn/cfn1, so its adjoint of p at the top three levels is not the
     transpose of its TL. The oracle leaves those levels out there. The
     dot-product test holds woof's AD to woof's TL, which matches WRFPLUS's
     TL.
   - Not covered: the moist cqu/cqv/cqw factors, the hydrostatic option
     and a 1-D base state.
   - `rhs_ph` is done in the same module (`geo_tl`, `geo_ad`). It covers
     woof's strict `slow_geopotential_open` (3-D base, vertical term on,
     no supplied face masses) for `h_sca_adv_order` 2-6 and 0, on periodic,
     open and specified axes, with map factors and hybrid levels. For
     order 5 on periodic or specified axes that kernel gives the fused
     `slow_geopotential`'s words, and the float base state is checked
     against both. The outer upwind faces take WRFPLUS's derivative of
     the clip: min(s, 0) passes the tangent only for s < 0, max(s, 0) for
     s >= 0.
   - The AD writes each cell's face and stencil adjoints to y and x
     scratch, then four gathers (w levels with php, u faces, v faces,
     columns) sum them, one writer per word. The top level is overwritten
     by the primal, so its incoming adjoint is zeroed.
   - Checks: dot-product test about 3e-15 (binary64) and 1.3e-8
     (binary32) and FD for 11 configurations. The float base state is the
     strict primal's words (0 differing, host and CUDA). WRFPLUS 4.7.1
     (`tools/wrfplus_oracle/rhsph.F90`) agrees to about 4e-16 on 10 cases,
     orders 2-6 (order 0 has the dot-product, FD and primal checks only). WRFPLUS's `rhs_ph` is not hybrid, so the cases use
     c1 = 1, c2 = 0.
   - WRFPLUS bugs, found here. (a) `a_rhs_ph` declares its order-3/4
     trajectory arrays from jts+2 and its+2, but stores below those
     bounds on a periodic y axis or a specified x axis, which corrupts the
     heap (SIGSEGV in free; -fcheck=bounds names tmpv4012 and tmpv4016).
     Those three cases are TL only. (b) For orders 2-4 WRFPLUS ends a
     bounded axis's interior loop at min(ite, ide-1)-2 (or -3), where
     native WRF uses ite-2 (-3), so it drops the term one point inside
     the end (for orders 3/4 in x only with open_xe). The fixture masks
     those TL points and zeroes the seed there.
   - Binary32: the u and v adjoints take the across-face difference of
     phb + php (phb to 3e4, differences about 50-100), so they agree with
     WRFPLUS to 6.5e-6 to 1.6e-5 on host and CUDA. Removing each level's
     mean phb brings that under 2e-6. The Metal oracle limit is 1e-4 for
     those keys, as for the pressure gradient's mu' adjoint.
   - `coriolis` and `curvature` are done in the same module (`cc_tl`,
     `cc_ad`). They cover woof's strict `coriolis_curvature`
     (GPUWM_WRF_EXACT_C_BIGSTEP: coriolis, then curvature, each with its
     own add) on periodic, open and specified axes, with a rotated grid
     (sina/cosa), map factors and hybrid levels. The active inputs are
     ru, rv, u, v, w and the total column mass mut, from which the primal
     forms rw = (c1f mut + c2f) w / msft.
   - The AD writes each face's adjoints of its local averages to scratch
     (rv4, rw4, vxgm and the own wind on u faces; the same on v faces; the
     four half->full averages on w levels). A vxgm gather then sums the
     mass-point vxgm adjoints. Gathers for u and ru, v and rv, w (which
     keeps the rw adjoint) and the column mass collect the rest, one
     writer per word.
   - Checks: dot-product test about 2e-15 (binary64) and 6e-8
     (binary32) and FD for 7 configurations. The float base state is the
     strict primal's words (0 differing, host and CUDA). WRFPLUS 4.7.1
     G_/A_CORIOLIS then G_/A_CURVATURE (`tools/wrfplus_oracle/coriolis.F90`)
     agree to about 5e-16 on 5 cases, hybrid ones included: WRFPLUS takes
     the coupled rw, which the fixture forms from w and mut with its
     tangent and adjoint. A periodic axis's end face repeats face 0, which
     WRF reads through its halo, so the two faces' adjoints are compared
     as a sum. Binary32 agrees to 2.3e-6 or better.
   - Not covered: the polar / map_proj 6 branch of curvature (tan(lat)),
     which woof's primal does not port either.
6. Advection `advect_u/v/w/scalar` (non-PD, h orders 3-6, v orders 2-6). This is the
   largest group; once the pattern is set, the mechanical ports go to
   Sonnet sub-agents.
   - Done in `woof/grad/wrfplus/advect.py`: the TL/AD of woof's strict
     `flux_div_{scalar,u,v,w}` (WRF `advect_scalar/u/v/w`) for periodic,
     open and specified lateral boundaries, map factors,
     h_*_adv_order 3-6 and v_*_adv_order 2-6. The TL is one kernel per
     field kind. The AD is a gather (one writer per word, no atomics) in
     six kernels: the advected field's x, y and z adjoints, then ru, rv
     and rw. That is 16% faster than one kernel per output, because the
     fused u/v field gather needed about 148 registers.
   - Checks: dot-product test about 1e-15 and FD about 1e-10 for 4 kinds x
     9 configurations. The float base state is the strict primal's words
     (0 differing, host and CUDA). WRFPLUS 4.7.1 `g_/a_advect_*`
     (`tools/wrfplus_oracle/advect.F90`) agrees to about 1e-16 on 16
     cases.
   - The oracle compares every point except three groups. (a) Open
     non-specified u faces i = 0, nx and v faces j = 0, ny: WRF adds its
     cb terms there, and woof adds them in the separate
     `apply_open_radiative_bc` launch. (b) The periodic duplicate end point
     (u i = nx, v j = ny): WRF reads its halo column, woof wraps, so the
     fixture makes the two ru/rv end faces equal and compares the summed
     adjoint of columns 0 and nx. (c) The open non-specified w lid's
     boundary columns: see the primal finding below.
   - Primal finding, not fixed here: woof's strict `flux_div_w` lid
     (woof/core/kernels/advection.cu) forms the open non-cb x/y fluxes at
     k = nz but applies them only under `GPUWM_NO_HADV`. WRF 4.7.1's
     `advect_w` applies them at k = ktf+1. Fixing it changes production
     output for open non-specified runs, so it needs its own PR. The TL/AD
     follow woof's primal.
   - Not covered: h_*_adv_order 2 (the legacy idealized path), WENO and
     the monotonic or positive-definite limiters.
7. Lateral boundaries: `relax_bdy_dry`, `spec_bdy_dry`,
   `spec_bdyupdate(_ph)`, `zero_grad_bdy`, `spec_bdy_final`, with LBC
   perturbations as an AD output (4D-Var's `var4d_lbc`).
8. Step and window driver (`woof/grad/wrfplus/model.py`):
   - per-step dot-product and Taylor tests;
   - TL vs nonlinear perturbation growth over 1-6 h on a 12 km dry
     case, measured the way WRFPLUS's checks do (the ratio of TL to NL
     difference norms by field and lead time);
   - the adjoint/forward cost ratio on CUDA and Metal.

   This is where G3 is decided.
9. Optional, if asked for: `sixth_order_diffusion`, `horizontal_diffusion`
   (`diff_opt = 1`).

Cost is measured from item 2 on. Kernel-level AD/forward ratios are
reported at every merge, so a G3 cost miss shows up before item 8.

## 6. Interfaces for P4 and 4D-Var

To be agreed with the 3D-Var owner. `woof/da` is not edited.

- `woof.grad.wrfplus.DryState`: a named tuple of device arrays in
  woof's layout. `u, v, w, ph, t, mu` are the control fields; the
  diagnostics `p, al` are derived.
- `Trajectory.record(state, cfg, nsteps)` runs `dycore.step` and keeps
  the step-start states (section 3).
- `tl(traj, dx0, steps) -> [dx_n]` and `ad(traj, dy_by_step) -> dx0`.
  The adjoint takes forcing at any step, as FGAT and 4D-Var need, and
  returns `dx0`. With LBCs on, it also returns `dlbc`.
- Arrays are CuPy (or facade) device arrays, taken without a host copy,
  so 4D-Var's device-resident `U`/`U^T` (the planned `woof/da/var`) feeds them
  directly.
- `DryState` stays a named tuple in woof's `(k, j, i)` layout. The
  control-variable transforms (psi/chi and so on) and the mapping from
  `xa` to the model state stay in `woof/da`.
- Moist fields are an optional passthrough: `None` (or zeros) while the
  model is dry, so the signature does not change at P3b.
- Inner products and norms are binary64. They take a `dot` hook whose
  default has the signature of the planned `woof.da.var.control.dot(x, y,
  reduction=...)` (`"wrfda-serial"`, `"fixed-tree"`; WRFDA-3DVAR-PLAN
  4.7).
- Agreed with the 3D-Var owner on 2026-11-19.

## 7. Effort, against the plan's 8-12 w

| item | estimate |
|---|---|
| 1 pattern PR (kernel, oracle framework, helpers) | 1.5 w |
| 2-3 rest of the acoustic substep | 1.5-2 w |
| 4 RK glue | 1 w |
| 5 big-step tendencies | 1.5-2 w |
| 6 advection | 2-3 w |
| 7 lateral boundaries | 1-1.5 w |
| 8 step/window driver, G3 measurements | 1.5-2 w |
| total | **10-13 w** |

That is 1-2 w over the plan's range. The overrun comes from three things:

- the hybrid-coordinate gap (section 1), which needs woof-derived
  linearisations beyond WRFPLUS's;
- the binding tests against woof's float32 primal;
- the three-backend lanes.

Diffusion (item 9) is extra: 1-1.5 w.

## 8. Risks for G3, watched from the start

G3's cost limit is about 5x the forward step. It was 4x until the user's
decision of 2026-11-20.

- **Cost.** The binary64 adjoint on the A6000 runs at 1/32 fp64
  throughput. The adjoint step may exceed the limit over the float32
  forward step even if it is about 3x in operation count.
  - Fallback: a binary32 adjoint for production gradients, checked
    against the binary64 tests.
  - The kernel ratios are measured from item 2, and reported early.
- **TL validity at km scales.** Measured at item 8 on a 12 km dry case
  first, then 3 km. Dry TL validity is an upper bound on the moist one.
- **WRFPLUS has no hybrid coordinate.** `hybrid_opt = 2` gets no
  oracle evidence (section 1).

### Measured cost and the step projection

These kernel ratios are AD and TL time over the binary32 primal. They were
measured on the pb A6000 at 200x200x49, as interleaved A/B in one process.
Other jobs share that card; the advect and big-step rows were timed with
it idle (under 10% utilisation for 20 s before the run).

| kernel (woof primal) | PR | AD32 | TL32 | AD64 |
|---|---|---|---|---|
| advance_uv | #652, #655 | 3.4-4.0x | ~3x | 12-15x |
| advance_w + calc_p_rho (advance_w_phi) | #669 | 3.5-3.7x | 2.3x | 9.3-9.8x |
| calc_coef_w (calc_coefs) | #675 | 4.8x | 3.2x | 21-23x |
| advance_mu_t + spec_bdyupdate (advance_mu_th + advance_exact_frame_mu_t) | advance_mu_t PR | 3.35x (3.2-3.5) | 2.0x | 8.2x |
| small_step_prep + calc_p_rho, small_step_finish (small_step_{init,finish}_{uv,column}) | #685 | 4.8x (4.7-5.0) | 2.1x | 7.7-8.0x |
| calc_p_rho_phi (calc_p_alpha_carried_muts) | #691 | 3.2x (3.1-3.5) | 1.0x | 4.9x |
| rk_step_prep: couple_momentum + calc_ww_cp (stage_fluxes, CuPy) | #691 | 0.85x (0.82-0.86) | 0.32x | 1.2x |
| rk_addtend_dry (dry tendency add + add_held_heating) | #691 | 1.0x | 1.7x | 2.2x |
| set_w_surface (set_surface_w) | #691 | 2.1x (2.0-2.3) | 1.2x | 4.0x |
| advect_scalar/u/v/w (flux_div_{scalar,u,v,w}) | #700 | 3.9x (3.3-4.5) | 1.5-1.8x | 19-27x |
| horizontal_pressure_gradient (slow_pgf) | #701 | 4.0x (3.9-4.1) | 1.9x | 14-15x |
| pg_buoy_w (slow_buoyancy) | #701 | 1.3x | 1.9x | 2.1x |
| rhs_ph (slow_geopotential, fused or open) | #702 | 4.2x (4.1-4.4) | 1.2-1.3x | 13-15x |
| coriolis + curvature (coriolis_curvature) | this PR | 3.4x | 2.0x | 13x |

advance_uv's 3.7x in the projection is the midpoint of its range. The
advance_mu_t ratio includes the frame kernel (advance_exact_frame_mu_t),
whose forward share sits in "other woof kernels", so its 11.4% slightly
understates the weight. For spec_zone >= 2 the TL/AD follow woof's
primal, which treats the whole spec_zone ring as frame; WRFPLUS
integrates from ids+1 whatever spec_zone is, and the oracle runs
spec_zone = 1 only.

The RK-stage glue (item 4) is cheap next to the acoustic kernels. Its
four operators together run at 1.8x AD32 over their primal (0.95x TL32).
rk_step_prep's primal is CuPy (stage_fluxes' face masses and
elementwise couplings), so its fused TL and AD kernels run faster than
the forward. In a dry step only calc_p_alpha_carried_muts has a share
that matters: 5.8% of the strict step (0.93 ms for its four calls,
CUPTI on the same card, 2026-11-20). It was part of "other woof
kernels" below. set_surface_w is 0.03%. add_held_heating does not run
without microphysics, and the held-tendency add without a mixing
package. stage_fluxes sits inside the 6.8% CuPy share, which is left
unmeasured below; at 0.85x it can only lower the projection.

small_step's ratio is over all four primal kernels (init and finish).
Most of its adjoint time is the init column adjoint (3.3 ms of about
5.8 ms at this size, against 1.16 ms for the init column TL). Taking
reciprocals in its forward recompute saved only about 6%, so the kernel
looks bound by memory traffic rather than division. The finish pass runs first in
the adjoint, init second, as WRFPLUS does. calc_mu_uv_1 has no woof
kernel of its own in the strict path (small_step_finish's face masses
are means of muts), so the finish TL/AD and its oracle cover it.

The shares below are each kernel's part of the forward step's GPU kernel
time. They come from a dry step at 200x200x49, using CUPTI kernel
activity tagged by woof kernel and the minimum over 10 steps
(`tools/launch_profile`).

| kernel | strict | fast |
|---|---|---|
| advance_w_phi | 34.5% | 32.9% |
| advance_mu_th | 11.4% | 12.2% |
| advance_uv | 8.6% | 10.5% |
| flux_div_{u,v,w,scalar} | 14.4% | 15.9% |
| small_step_{init,finish}_* | 8.2% | 10.5% |
| slow_pgf, slow_geopotential | 5.2% | 6.0% |
| calc_coefs | 2.3% | 3.0% |
| other woof kernels | 8.5% | 6.1% |
| CuPy kernels and copies | 6.8% | 3.0% |

flux_div is now measured on the in-tree port (item 6):
TL and AD over woof's production float32 `flux_div_{scalar,u,v,w}`
under GPUWM_WRF_EXACT=1, summed over the four kinds at 200x200x49 on pb,
for periodic and for open+specified boundaries. On an idle card
(2026-11-20) AD32 was 4.5x with open+specified boundaries and 3.3x
periodic (3.9x is used below), TL32 1.5-1.8x and AD64 19-27x. A card
another job held at 73-80% gave AD32 4.0-5.2x. AD64 is compute bound on the A6000's 1/32-rate float64; the glue
and acoustic kernels are memory bound and do not show this. The earlier
out-of-tree prototype (periodic flux_div_scalar only) gave 2.56x; the
port is dearer because the u and v field gathers average their
advecting mass fluxes and visit both wrapped end faces, and the
boundary paths add branches.

slow_pgf and slow_buoyancy run once per RK stage. Their idle-card
primals (0.154 ms and 0.038 ms) give 2.9% and 0.7% of the strict step's
16 ms for three calls. slow_pgf is part of the 5.2% row above (the
other 2.3% is slow_geopotential); slow_buoyancy sits in "other woof
kernels". slow_geopotential's idle-card primal is 0.20-0.24 ms (fused
for specified and periodic axes, the open kernel for open ones); its
row above is over the kernel the strict path dispatches.
coriolis_curvature runs once per RK stage on rotational grids. Its
idle-card primal (0.20 ms) gives 3.8% of the strict step for three
calls, taken out of "other woof kernels".

The projection is the per-step adjoint cost over the forward step:
1 (forward recompute) + Σ share x AD32 ratio. For the kernels measured so
far (94.9% of the strict step: the 65.0% above, calc_p_rho_phi's 5.8%,
set_w_surface's 0.03%, flux_div's 14.4%, slow_pgf's 2.9%,
slow_geopotential's 2.3%, coriolis_curvature's 3.8% and slow_buoyancy's
0.7%), Σ share x AD32 = 3.56, on top of the forward recompute's 1. The
rest is projected at a range of ratios:

| ratio for the unmeasured 5.1% | projection |
|---|---|
| 2.2x (memory-traffic floor) | 4.67x |
| 3.0x | 4.71x |
| 3.75x (the measured share-weighted mean) | 4.75x |

The 5x limit holds while the rest averages at most about 8.6x. This is a
projection from kernel GPU time, not a measured step ratio: launch gaps
are not counted, most ratios come from a shared card, and 5.1% of the
step is not yet ported. calc_coefs and small_step are each about 4.8x;
at 2.3% and 8.2% of the step they add about 0.11 and 0.39. Each later kernel PR replaces a
projected share with a measured one.
