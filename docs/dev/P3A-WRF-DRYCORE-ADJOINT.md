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
   **per-acoustic-substep** chained dot-product test.
4. RK-stage glue: `rk_step_prep` (`couple_momentum`, `calc_ww_cp`),
   `calc_p_rho_phi`, `rk_addtend_dry`, `set_w_surface`, periodic
   physical BCs, plus a per-stage test with tendencies frozen.
5. Big-step tendencies: `horizontal_pressure_gradient`, `pg_buoy_w`,
   `rhs_ph`, `coriolis`, `curvature`, `w_damp`.
6. Advection `advect_u/v/w/scalar` (non-PD, orders 2-6). This is the
   largest group; once the pattern is set, the mechanical ports go to
   Sonnet sub-agents.
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
Other jobs share that card.

| kernel (woof primal) | PR | AD32 | TL32 | AD64 |
|---|---|---|---|---|
| advance_uv | #652, #655 | 3.4-4.0x | ~3x | 12-15x |
| advance_w + calc_p_rho (advance_w_phi) | #669 | 3.5-3.7x | 2.3x | 9.3-9.8x |
| calc_coef_w (calc_coefs) | this PR | 4.8x | 3.2x | 21-23x |

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

The projection is the per-step adjoint cost over the forward step:
1 (forward recompute) + Σ share x AD32 ratio. For the kernels measured so
far (45.4% of the strict step), Σ share x AD32 = 1.67, on top of the
forward recompute's 1. The rest is projected at a range of ratios:

| ratio for the unmeasured 54.6% | projection |
|---|---|
| 2.2x (memory-traffic floor) | 3.9x |
| 3.0x | 4.3x |
| 3.7x (as measured so far) | 4.7x |

The 5x limit holds while the rest averages at most about 4.3x. This is a
projection from kernel GPU time, not a measured step ratio: launch gaps
are not counted, the ratios come from a shared card, and 54.6% of the
step is not yet ported. calc_coefs alone is 4.8x, above the limit, but at
2.3% of the step it adds about 0.11. Each later kernel PR replaces a
projected share with a measured one.
