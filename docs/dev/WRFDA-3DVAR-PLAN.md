# Variational data assimilation (WRFDA, 3D-Var first): scope and plan

Status: design study (requested 2026-11-21). Nothing in this document is
implemented. It scopes adding WRFDA-style variational data assimilation to
woof, starting with 3D-Var, and says how to build and verify it. A second
part (section 6) scopes gradient-based and neural-network options inside the
DA itself.

It is the companion of
[DIFFERENTIABLE-HYBRID-PLAN.md](DIFFERENTIABLE-HYBRID-PLAN.md) (merged as
#579), and is written so the two do not overlap:

- **3D-Var needs no model adjoint.** Its gradient is
  `grad J = v + U^T H^T R^-1 (H U v - d)`: the adjoints it needs are of the
  control-variable transform `U` and of the observation operator `H` only.
  Those are this plan's work (phases V0-V4).
- **4D-Var needs the model's tangent-linear and adjoint**, which is the
  differentiable plan's P3 (a WRF dry-core adjoint checked against WRFPLUS)
  and P4 (checkpointed windows). This plan stops at the hooks for it (V5)
  and reuses everything V0-V4 build.
- **One observation cost, not two.** The differentiable plan's P0.1
  (`woof/grad/obs_cost.py`, in progress) defines the observation table, the
  error table and FGAT pairing. 3D-Var's `J_o` takes the same inputs and
  adds the tangent-linear and adjoint of the same operators (section 4.10).

Numbers marked *measured* come from the WRFDA source, from the oracle run in
section 8, or from the repo's own docs (cited by path). Numbers marked
*estimate* are arithmetic, and the phase that first runs the code replaces
them with measurements.

## Verdict, in short

- **3D-Var on the WRF route is feasible and medium-sized**: about 15-21
  engineer-weeks for conventional observations, WRFDA's own `be.dat`, the
  CG minimiser and a WRFDA oracle agreement gate (V0-V1), and about 8 more
  to make woof's own background-error statistics (V2), then 3-4 more for
  hybrid 3DEnVar with the LETKF ensemble (V3).
- **Everything except the minimiser's inner products is local or
  line-parallel**, and fits the GPU well: the recursive filters are one
  thread per grid line, the vertical EOFs a batched small matrix product
  per column, the balance a per-column matrix-vector product, and the
  observation operators a gather (and, transposed, a fixed-order scatter).
  *Estimate:* well under a second per 50-iteration analysis on a
  200x200x50 grid on the A6000 (section 4.12).
- **WRFDA is a usable oracle.** It is public domain, ships inside the WRF
  4.7.1 tree (`var/`), builds serially in the existing `wrf-oracle:u2404`
  container on pb, and prints its cost function to 16 significant digits
  per iteration. Section 8 records the build and a tutorial run.
- **It runs in binary64.** WRFDA with gfortran is built with
  `-fdefault-real-8`; woof's state is binary32 and Metal has no binary64.
  So the plan has an oracle tier (CUDA, binary64, WRFDA's operation order)
  that agrees with WRFDA to a stated tolerance, and a production tier
  (binary32 storage, binary64 or double-float inner products, fixed-order
  reductions) on which CUDA and Metal agree bit for bit under strict
  numerics (section 4.7).
- **Radiances are opt-in at most.** The CRTM 2.3.0 copy bundled with
  WRFDA 4.7.1 has 15 files with GPL-2.0-or-later headers; CRTM v2.4.0 from
  JCSDA is dedicated to the public domain (CC0). RTTOV is free but only
  under an NWP SAF licence agreement. Neither may enter the default wheel;
  section 4.11 says how an opt-in links one.
- **Default forecasts do not change.** DA is a separate command and a
  separate package; its result reaches a forecast only through the existing
  analysis bundle and `--analysis` door, as an LETKF analysis does today.

## 1. WRFDA 4.7.1, as it is built

Source: the WRF v4.7.1 release tarball
(`https://github.com/wrf-model/WRF/releases/download/v4.7.1/v4.7.1.tar.gz`,
SHA-256 `11186188b033d26332e31769c1f7aff9349406920a7f72eeb256d5881f7223f4`),
directory `var/`. WRFDA is part of WRF and carries WRF's public-domain
notice (`LICENSE.txt` at the tree's root, already in this repo as
`licenses/LICENSE-WRF-public-domain.txt`): UCAR "make no proprietary claims
... and consider WRF to be in the public domain", and requests that the
notice travel with partial or full copies. Any WRFDA-derived code in woof
records that notice in `PROVENANCE.md` as the WRF ports do. Section 4.11
covers the bundled third-party libraries, which do not all share it.

All paths in this section are relative to `var/da/` unless they say
otherwise.

### 1.1 Driver

`da_main/da_wrfvar_main.f90` calls `da_wrfvar_init1/2`, `da_wrfvar_run`
and `da_wrfvar_finalize`; the analysis is `da_main/da_solve.inc`. In
order:

1. Checks of the namelist combinations (e.g. `cv_options = 3` refuses
   `ensdim_alpha > 0`; radar reflectivity options against
   `cloud_cv_options`).
2. `da_setup_firstguess`: the background `x_b` from `fg` (a `wrfinput` or
   `wrfout`), with the derived fields the operators need (pressure, height,
   `q_s`, grid-box area).
3. `da_setup_obs_structures`: read, window, QC and thin the observations
   (section 1.8).
4. `da_setup_background_errors`: read `be.dat` (section 1.5) into the `be`
   structure; if `ensdim_alpha > 0`, `da_setup_flow_predictors` reads the
   ensemble perturbations (section 1.10).
5. The outer loop `do it = 1, max_ext_its`:
   - `da_get_innov_vector`: `d = y - H(x_b)` with the full nonlinear `H`,
     and the gross-error check (`check_max_iv`).
   - Optionally `da_check_gradient` and `da_check` (the adjoint tests,
     section 1.13), then exit.
   - Minimise: `da_minimise_cg` (default) or `da_minimise_lz`
     (`use_lanczos`).
   - `da_transform_vtox`: `x_a' = U v` (section 1.4); for the hybrid,
     `da_transform_vpatox` adds the ensemble part.
   - `da_transform_xtoxa` (derived increments), `da_write_diagnostics`
     (O-B, O-A, `J` per type), `da_varbc_update` if VarBC is on.
   - `da_transfer_xatoanalysis`: add the increment to the analysis, which
     is the next outer loop's linearisation state.
6. `da_update_firstguess` writes `wrfvar_output` (a `wrfinput`-format
   analysis).

The lateral boundary update is a separate program, `da_update_bc.exe`
(`da_update_bc/`), run after the analysis (section 1.12).

### 1.2 Cost function

WRFDA minimises the incremental cost in control space `v`, with
`x_a' = U v` and `B = U U^T`:

```
J(v) = Jb + Jo + Je + Jp + ...
Jb   = 1/2 v_b^T v_b                       (static part, jb_factor)
Je   = 1/2 v_alpha^T v_alpha               (ensemble alpha part, je_factor)
Jo   = 1/2 (d - H' U v)^T R^-1 (d - H' U v)
Jp   = VarBC coefficient penalty            (section 1.9)
```

`da_calculate_j` returns the parts (`J, Jb, Jo, Jc, Je, Jd, Jp, Js, Jl,
Jm, Jt`; the others are TAMDAR VarBC, rain, satellite control variables,
divergence constraints and others that 3D-Var does not need first), and
`da_calculate_gradj` the gradient. `R` is diagonal: each observation's own
error, from OBSPROC's `obserr.txt` or the reader's table, optionally scaled
by `use_obs_errfac`.

### 1.3 Minimisation

`da_minimisation/da_minimise_cg.inc` is a preconditioned conjugate gradient
on the quadratic `J(v)`:

- `ghat` is the gradient at the start, `phat = -precon * ghat`; per
  iteration one `da_calculate_gradj` evaluates `fhat = (I + U^T H^T R^-1
  H U) phat` (the Hessian-vector product), `step = rrmold / <fhat, phat>`,
  `ghat += step fhat`, `xhat += step phat`.
- Optional modified Gram-Schmidt re-orthonormalisation of the gradient
  against all previous ones (`orthonorm_gradient`, which the tutorial
  namelist sets), and an optional diagonal preconditioner by control
  variable block (`precondition_cg`).
- Stops when `|g| < eps(it) * |g_0|` (`eps` default 0.01) or at `ntmax(it)`
  iterations (default 75).
- `J` per iteration is either recomputed (`calculate_cg_cost_fn`) or taken
  from the quadratic identity `J = J0 + 1/2 <g0, xhat>`.
- Each iteration prints one line, ` minimize_cg  it iter  J  |g|  step`,
  in `1PD22.15` format: 16 significant digits of `J`, `|g|` and the step.
  This is the oracle's per-iteration record (section 5).

`da_minimise_lz.inc` is a Lanczos version that can save and reuse the
Lanczos vectors (`write_lanczos`, `use_lanczos`); Lanczos-CG for
variational DA follows Fisher (1998).

Inner products are `da_dot` (a serial sum unrolled by five: the `mod(n,5)`
leading terms first, then groups of five) inside `da_dot_cv`, which sums
the per-process parts with `wrf_dm_sum_real` (an MPI all-reduce, so the
result depends on the decomposition), or, with `test_dm_exact`, gathers
the whole vector to one task and sums it serially. A serial build therefore
has one fixed summation order; woof's oracle tier reproduces it
(section 4.7).

### 1.4 Control variables and the transform `U`

`U = U_p U_v U_h`: a horizontal correlation (`U_h`, recursive filters), a
vertical correlation (`U_v`, EOFs) and a physical transform (`U_p`,
balance and change of variable). In code, `da_transform_vtox` calls
`da_transform_vtovv` (`U_h`), `da_vertical_transform` (`U_v`,
`vert_corr = 2`), then `da_transform_vptox` / `da_transform_bal` (`U_p`).

**Control variables** by `cv_options` (regional, `da_setup_be_regional`):

| `cv_options` | v1 | v2 | v3 | v4 | v5 | balance |
|---|---|---|---|---|---|---|
| 5 (default) | psi (stream function) | chi_u (unbalanced velocity potential) | T_u | pseudo-RH `q/q_s(x_b)` | ps_u | `chi = chi_u + c psi`, `T = T_u + G psi`, `ps = ps_u + W psi` |
| 6 | psi | chi_u | T_u | RH_u | ps_u | cv5 plus the named regressions in `be.dat` (`regcoeff_psi_rh`, `regcoeff_chi_u_t`, `regcoeff_chi_u_ps`, `regcoeff_chi_u_rh`, `regcoeff_t_u_rh`, `regcoeff_ps_u_rh`) |
| 7 | u | v | T | pseudo-RH | ps | none: `B` is univariate in u, v, T, q, ps |
| 3 | global NCEP-style `be.dat.cv3` (`var/run/`), a different reader (`da_setup_be_ncep_gfs`) | | | | | |

`cloud_cv_options >= 2` adds hydrometeor control variables v6-v10 (`qcw,
qrn, qci, qsn, qgr`), and `use_cv_w` adds `w` (v11); both are later
options, not 3D-Var's first target.

**`U_h`, recursive filters** (`da_recursive_filter/`).
`da_transform_through_rf`:

1. Divide the field by `sqrt(grid-box area)`.
2. Apply `rf_passes / 2` passes of `da_recursive_filter_1d` along x, for
   every (y, mode), then the same along y. `rf_passes` defaults to 6.
3. Multiply by `sqrt(area)` again; with `vert_corr = 1`, also scale by the
   standard deviation.

`da_recursive_filter_1d` is a first-order forward-backward recursion with
a coefficient `alpha` per vertical mode. The forward sweep is
`b(j) = alpha b(j-1) + (1-alpha) a(j)`, the backward sweep
`c(j) = alpha c(j+1) + (1-alpha) b(j)`. The boundary treatment depends on
the pass number: pass 1, pass 2, and later passes each have their own
closed form, the "infinite-domain" starts. `alpha` comes from the length
scale per mode in `be.dat`, scaled by `len_scaling1..5`
(`da_calculate_rf_factors`). The adjoint (`da_recursive_filter_1d_adj`) is
the reversed recursion. Background: Purser et al. (2003a, b); Wu et al.
(2002) for the related GSI filters.

**`U_v`, vertical EOFs.** `da_transform_vvtovp`:
`vp(k) = sum_m E(k,m) sqrt(lambda_m) vv(m)` per column, with the
eigenvectors `E` and eigenvalues `lambda` of the vertical covariance per
latitude bin (`be%v1%evec(j,k,m)`, `be%v1%val(j,m)`), truncated to `mz`
modes by `max_vert_var1..5` (default 99 % of the variance). `ps_u` (v5) is
2-D and has one mode.

**`U_p`, balance** (`da_vtox_transforms/da_transform_bal.inc`): T is
`T_u` plus `agvz(i,j,k,kk) psi(kk)`, summed over levels; `chi` is `chi_u`
plus `bvz(i,j,k) psi(k)`; `ps` is `ps_u` plus a sum of `wgvz psi`, which is an
increment of `ln p_s`, then multiplied by `p_s(x_b)` to give `p_s`). Then
`da_psichi_to_uv` turns psi and chi into u and v by centred differences
with map-factor coefficients. Humidity is `q = pseudo-RH * q_s(x_b)`
(`cv_options_hum = 1`). The regression coefficients come from `be.dat`
per bin and are interpolated to the model's levels and latitudes.

### 1.5 `be.dat`

`da_setup_structures/da_setup_be_regional.inc` reads it. The format,
*measured* on the tutorial `be.dat`:

- Fortran unformatted sequential, **big-endian** (WRFDA builds with
  `-fconvert=big-endian`), **4-byte record markers**
  (`-frecord-marker=4`), and **8-byte reals** (gen_be is built with
  `-fdefault-real-8`). Integers are 4 bytes.
- Records, in order:
  1. `ni, nj, nk` (the tutorial file: 89, 59, 40).
  2. `bin_type` (5: domain-averaged statistics).
  3. Optionally `eta_be(nk)`. Older files, the v3.9 tutorial's among them,
     do not have it; the reader then falls back to the namelist's
     `be_eta`.
  4. `lat_min, lat_max, binwidth_lat` (-90, 90, 10).
  5. `hgt_min, hgt_max, binwidth_hgt`.
  6. `num_bins, num_bins2d` (40, 1), then `bin(ni,nj,nk)` and
     `bin2d(ni,nj)`.
  7. For cv5, `regcoeff_psi_chi`, `regcoeff_psi_ps`, `regcoeff_psi_t`. For
     cv6, a named list (`regcoeff_chi_u_t`, ..., `regcoeff_ps_u_rh`). cv7
     has none.
  8. Per control variable: `variable` (a name, e.g. `psi`, `chi_u`, `t_u`,
     `rh_u`, `ps_u`, or `u`, `v`, `t`, `rh`, `ps` for cv7), `nk,
     num_bins2d`, the global `evec`/`eval`, then the local
     (per-bin) `evec`/`eval`.
  9. Per control variable, `variable` and its RF length scales per mode
     (`rfls`).
- The reader checks the variable names against the `cv_options` it
  expects, and refuses a mismatch. woof's reader does the same and also
  records the file's SHA-256.

A woof `be.dat` reader that reads these words exactly (big-endian, real*8
to binary64, no conversion until the GPU upload) is the first PR (V0), so
that woof and WRFDA can be fed the same `B`.

### 1.6 gen_be

`var/gen_be/` (and the driver `var/scripts/gen_be/gen_be.ksh`) estimates
`be.dat` from forecast differences:

- **NMC method** (Parrish and Derber 1992): differences of 24 h and 12 h
  forecasts valid at the same time.
- **Ensemble method**: member minus ensemble mean.

The stages:

- **Stage 0** (`gen_be_stage0_wrf`): psi, chi, T, RH, ps perturbations
  from `wrfout` pairs.
- **Stage 1**: remove the mean and bin.
- **Stage 2/2a**: the balance regressions (psi to chi, T, ps) and the
  unbalanced fields.
- **Stage 3**: vertical covariances and EOFs per bin.
- **Stage 4** (`gen_be_stage4_regional`): horizontal length scales per
  mode, by fitting a Gaussian correlation to the binned correlations
  against distance.

`gen_be_diags` writes `be.dat`. `var/gen_be_v3/gen_be_v3.F90` is a
user-contributed alternative for `bin_type = 5` only, which also writes
the ensemble perturbation files for the hybrid. GEN_BE v2.0 (Descombes et
al. 2015) is the generalised successor with more control variables and
binning options; it is a separate download, not in the WRF tree. The
WRFDA download page also offers 11 one-day forecasts (2008-02-03 to
02-08) for practising gen_be.

### 1.7 Observation operators

There are 33 operators, each `da_<type>/da_transform_xtoy_<type>.inc`
with an `_adj` twin, plus the innovation (nonlinear) path
`da_get_innov_vector_<type>`:

- **Conventional:** `sound`, `sonde_sfc`, `synop`, `metar`, `ships`,
  `buoy`, `airep`, `pilot`, `profiler`, `geoamv`, `polaramv`, `qscat`,
  `tamdar` (and `tamdar_sfc`), `bogus`, `satem`, `mtgirs`, `airsr`.
- **GNSS:** `gpspw`, `gpsztd`, `gpsref` (local refractivity), `gpseph`
  (excess phase).
- **Radar:** radial velocity and reflectivity (`da_radar/`; direct
  reflectivity on `qr`, or indirect via retrieved hydrometeors and
  in-cloud humidity: Xiao et al. 2005, 2007; Wang et al. 2013).
- **Precipitation:** `rain` (4D-Var only).
- **Lightning.**
- **Radiances:** `crtm` and `rttov` (section 1.11), and SSM/I
  (`ssmi_rv`, `ssmi_tb`).
- **Chemistry:** surface chemistry (`chem_sfc`, WRF-Chem builds only).

Most are linear in the analysis increment: `da_interp_lin_3d` is bilinear
in the horizontal and linear in the model-level coordinate `zk`, which the
innovation step computes once from the observation's pressure or height.
Wind speed and direction (`da_uv_to_sd_lin`), surface corrections
(`sfc_assi_options`), GNSS PW and refractivity, and reflectivity are
nonlinear and have true tangent-linears.

### 1.8 Observation preprocessing, QC and thinning

- **OBSPROC** (`var/obsproc/`, a separate program): reads `little_r`
  reports, merges duplicates, assigns errors from `obserr.txt`, checks
  domain, time window and vertical consistency, and writes the ASCII
  `obs_gts_<date>.3DVAR` that `ob_format = 2` reads (the tutorial's
  `ob.ascii`).
- `ob_format = 1` reads NCEP PREPBUFR directly through the bundled BUFR
  library (`da_read_obs_bufr`). GPS RO BUFR, AMV BUFR, MADIS, radar and
  rain each have their own reader in `da_obs_io/`.
- **QC in WRFDA:** windowing (`time_window_min/max`), the gross-error check
  `|O - B| > max_error_* x sigma` (`check_max_iv`, per type), optional
  rejection of both wind components together (`qc_rej_both`), buddy
  checks (`max_buddy_*` thresholds in `da_control`), and limits such as
  `max_omb_spd`.
- **Thinning:** `thin_conv` with per-type mesh sizes `thin_mesh_conv` (km),
  keeping the best report per thinning box; radiances have their own
  thinning.

### 1.9 VarBC

`da_varbc/` (Auligné et al. 2007; Derber and Wu 1998 before it) corrects
radiance biases with a linear model `b = sum_k beta_k p_k` of predictors
(scan position, layer thicknesses, skin temperature, ...). The
coefficients `beta` are extra control variables, with a background penalty
`Jp`. They are read from and written to `VARBC.in`/`VARBC.out`, warm or
cold started, and preconditioned (`da_varbc_precond`). There is a TAMDAR
variant (`da_varbc_tamdar/`). VarBC is a radiance-era component; woof needs
it only with V4's radiances.

### 1.10 Hybrid ensemble-variational (alpha control variable)

With `ensdim_alpha = N > 0`, `da_transform_vpatox` /
`da_calc_flow_dependence_xa` add an ensemble increment
`x_ens' = sum_n alpha_n o x'_n` (an elementwise product per grid point,
for u, v, T, q and ps, and optionally the hydrometeors) to the static
`U v`. Here:

- `x'_n` are the ensemble perturbations (`ep_format` 1/2/3 files; e.g. from
  `gen_be_ep2` or `gen_be_v3`), scaled by `1/sqrt(N-1)`;
- the `alpha_n` fields are control variables with their own correlation:
  a recursive filter of scale `alpha_corr_scale` (default 200 km) and a
  vertical localisation (`alpha_vertloc_opt`; 2 = computed in log p);
- the weights: `je_factor` (ensemble) is a namelist key, and WRFDA derives
  the static weight `jb_factor = je_factor / (je_factor - 1)`
  (`da_setup_background_errors`), so `1/jb_factor + 1/je_factor = 1`.

This is Lorenc (2003)'s alpha control variable as implemented for WRF by
Wang et al. (2008). `anal_type_hybrid_dual_res` runs the ensemble part on a
coarser grid. This is the link to woof's LETKF ensemble: the members
already exist on the device, so `x'_n` needs no file round trip (V3).

### 1.11 Radiances: CRTM and RTTOV

- `rtm_option = 1` is RTTOV, `2` CRTM (`da_radiance/`, `da_crtm_*`,
  `da_rttov_*`, with TL, AD and K-matrix calls).
- **CRTM 2.3.0 is bundled** (`var/external/crtm_2.3.0/`), and the default
  configure compiles it in (`-DCRTM`). Its coefficients are a separate
  download (`var/run/crtm_coeffs` is a symlink to an NCAR path).
- **CRTM licences.** *Measured:* 15 of the bundled 2.3.0 source files carry
  GPL-2.0-or-later headers (`Endian_Utility.f90`, `File_Utility.f90` and 13
  `NESDIS_*` emissivity modules). The JCSDA CRTM repository's v2.4.0
  release (master at 2023-02-27, `8bf10076`) carries a single CC0 public
  domain dedication in its `LICENSE`, and none of its files has a GPL
  header. So any CRTM woof links is v2.4 or later, never the 2.3.0 copy in
  the WRF tree.
- **RTTOV** is free of charge but distributed by the NWP SAF only to
  registered users under a licence agreement
  (`https://nwp-saf.eumetsat.int/site/software/rttov/download/`). It can
  only be a user-supplied, opt-in library.
- Other bundled libraries: NCEP's BUFR library (`var/external/bufr`),
  BLAS/LAPACK, and a wavelet package (for `use_rf = .false.`). woof needs
  none of them: it already decodes PREPBUFR in Rust (`rw_prepbufr`), and
  its eigensolvers are its own ([da_jacobi_eigensolver.md](../da_jacobi_eigensolver.md)).

### 1.12 Outer loops, FGAT, 4D-Var, `da_update_bc`

- **Outer loops** (`max_ext_its`; arrays sized by `max_outer_iterations = 100`): relinearise `H` about the
  updated state; `eps`, `ntmax`, `len_scaling`, `var_scaling` can differ per
  outer loop.
- **FGAT** (`num_fgat_time > 1`): 3D-Var with innovations computed against
  first guesses at the observation times (`fg01`, `fg02`, ...), with the
  increment still 3-D at the analysis time.
- **4D-Var** (`var4d = .true.`) needs WRFPLUS linked in
  (`./configure 4dvar` with `WRFPLUS_DIR`): Huang et al. (2009).
- **`multi_inc`**: a multi-resolution incremental scheme. It is out of
  scope here.
- **`da_update_bc`**: after the analysis replaces the start, re-derive the
  first lateral boundary record's tendency so that the end of the first
  interval is unchanged. woof already does this
  (`woof/da/analysis_boundary.py`, `rebase_first_record`), and applies it
  automatically at the `--analysis` door.

### 1.13 Built-in tests

- `test_transforms`: `da_check_vtox_adjoint` (is `U^T` the adjoint of
  `U`?), `da_check_xtoy_adjoint` and its per-type variants (is `H'^T` the
  adjoint of `H'`?), and the `cv <-> vv` round trips.
- `test_gradient`: `da_check_gradient`, a Taylor test of
  `(J(v + a h) - J(v)) / (a <grad J, h>)` over decreasing `a`.

woof's V0/V1 tests are written to the same definitions, so both programs'
numbers can sit side by side.

### 1.14 Namelist (`&wrfvar1` ... `&wrfvar22`, `Registry/registry.var`)

The keys 3D-Var needs first, with their defaults:

| group | key | default | meaning |
|---|---|---|---|
| wrfvar1 | `var4d` | false | 4D-Var |
| wrfvar1 | `use_background_errors` | true | Jb on |
| wrfvar3 | `ob_format` | 2 | 1 PREPBUFR, 2 ASCII (OBSPROC) |
| wrfvar3 | `num_fgat_time` | 1 | FGAT slots |
| wrfvar4 | `use_synopobs`, `use_soundobs`, ... | true | observation types |
| wrfvar4 | `thin_conv`, `thin_mesh_conv` | true, per type | thinning |
| wrfvar4 | `use_obs_errfac` | false | error factors |
| wrfvar5 | `check_max_iv` | true | gross-error check |
| wrfvar6 | `max_ext_its` | 1 | outer loops |
| wrfvar6 | `ntmax`, `eps` | 75, 0.01 | inner iterations, gradient reduction |
| wrfvar6 | `orthonorm_gradient`, `precondition_cg`, `use_lanczos` | false | minimiser options |
| wrfvar7 | `cv_options` | 5 | control variables |
| wrfvar7 | `use_rf`, `rf_passes` | true, 6 | recursive filter |
| wrfvar7 | `len_scaling1..5`, `var_scaling1..5` | 1.0 | B tuning per variable |
| wrfvar7 | `je_factor` | 1.0 | ensemble weight (`jb_factor` is derived from it) |
| wrfvar10 | `test_transforms`, `test_gradient` | false | adjoint and gradient tests |
| wrfvar11 | `cv_options_hum`, `sfc_assi_options`, `calculate_cg_cost_fn` | 1, 1, false | |
| wrfvar13 | `vert_corr`, `vertical_ip`, `max_vert_var1..5` | 2, 0, 99 | vertical EOFs |
| wrfvar14 | `rtm_option`, `use_varbc` | 1, true | radiances |
| wrfvar16 | `ensdim_alpha`, `alpha_corr_scale`, `alpha_vertloc_opt`, `ep_format` | 0, 200 km, 2, 1 | hybrid |
| wrfvar17 | `analysis_type` | `3D-VAR` | also `VERIFY`, `RANDOMCV` |
| wrfvar18 | `analysis_date` | | |
| wrfvar21/22 | `time_window_min/max` | | |

## 2. What woof has, against what 3D-Var needs

| need | woof today | gap |
|---|---|---|
| Observation ingest | Rust front doors for MADIS, IGRA, PREPBUFR, AMV, GNSS-RO, ASOS, NDBC, MRMS, radar (`tools/rustwx/crates/rw-obs`), neutral tables (`woof.obs.neutral_table`), GSI's error table read as GSI reads it (`rw-obs/src/errtable.rs`), WRF `OBS_DOMAIN` records (`rw-obs/src/fddaobs.rs`); P0's flat `ObsSet` | A reader for WRFDA's `obs_gts_*.3DVAR` ASCII (OBSPROC output), so that woof and the oracle read the same observations. `little_r` itself only if OBSPROC is ported (not planned: run OBSPROC as a tool). Small |
| Nonlinear `H` | `obs_conventional` (pressure level, linear in ln p; surface), `obs_point` (adds RO refractivity), `obs_surface`, `obsop` (reflectivity, radial velocity), `obsop_cwp`; P0's `simulate` over `ModelFrame` columns | woof's operators interpolate in ln p in each column, P0 uses the nearest column; WRFDA is bilinear in the horizontal and linear in `zk`. The oracle lane needs WRFDA's own interpolation as a selectable stencil. Medium |
| `H'` and `H'^T` | none (the LETKF needs only `H(x_k)`) | All. Most are a sparse linear stencil (weights fixed at the innovation step), so TL is a gather and AD a scatter; the nonlinear ones (wind speed, GNSS, reflectivity) need true linearisations. Medium |
| QC | type-table gross checks, background checks and localisation in `obs_conventional`; GSI-style error inflation | WRFDA's `check_max_iv` per-type thresholds and `thin_conv` boxes, to match the oracle's accepted set. Small |
| `B` | none. `static_covariance.py` is an EnOI sample through the LETKF, and `perturb.py` draws correlated noise | All of section 1.4: RF, EOFs, balance, psi/chi to u/v, pseudo-RH. Medium-large |
| `be.dat` | none | Reader (V0) and writer (V2). Small |
| gen_be statistics | ensembles and cycling exist: `woof.ensemble`, `woof/da/member_*`, `cycle-spine.md`, forecast history | NMC pairs or ensemble differences to `be.dat` (V2). Medium |
| Minimiser | none. The LETKF solves in ensemble space with a Jacobi eigensolver | CG with re-orthonormalisation, later Lanczos. Small, but determinism matters (section 4.7) |
| Hybrid | GPU LETKF (`letkf.py`, `letkf_device.py`), device-resident members, `moments.py` | The alpha control variable and its RF; members to `x'_n` on device. Medium |
| Analysis output | analysis bundle (`woof/io/analysis_bundle.py`, [analysis-bundle-wrf.md](../analysis-bundle-wrf.md)) with increment, `--analysis` door, IAU (`iau.py`), `da_update_bc` (`analysis_boundary.py`) | None: 3D-Var writes a bundle like any producer |
| Cycling | `cycle_admission`, `cadence`, `hotstart`, [cycle-spine.md](../cycle-spine.md), [local-da-rapid-cycling.md](../local-da-rapid-cycling.md) | A 3D-Var analysis step as one more analysis kind. Small |
| Radiances | `woof/radtran/` ([radtran.md](radtran.md)) is a spectral RT for diagnostics, with no Jacobian; `obs_point` refuses brightness temperatures by name | An RTM with TL/AD (CRTM v2.4+, opt-in), VarBC. Large |
| Deterministic reductions | `fixed_order_gemm.py` (one summation order on every card, written for the LETKF) | Extend to long dot products and the AD scatter (section 4.7). Small |
| Oracle | WRF 4.7.1 and MPAS 8.4.2 oracles in `wrf-oracle:u2404` on pb | WRFDA build: done in this study (section 8). Gate scripts: V1 |

## 3. Scope

**In scope (this plan):**

- WRF route, regional, one domain.
- 3D-Var with `cv_options = 5` and 7 (6 later), `vert_corr = 2`, the RF
  `U_h`.
- Conventional observations first.
- CG minimisation, outer loops.
- woof's own gen_be.
- Hybrid 3DEnVar with the LETKF ensemble.
- Radar and GOES CWP operators with TL/AD.
- Radiances through an opt-in CRTM.
- FGAT.
- The 4D-Var hooks that P3/P4 plug into.

**Out of scope:**

- Global WRFDA (`cv_options = 3`, spectral `da_spectral/`).
- Wavelet `B` (`use_rf = .false.`).
- WRF-Chem DA (`chem_cv_options`).
- NMM.
- KMA formats.
- Dual-resolution hybrid.
- `multi_inc`.
- Lightning and rain operators.
- The hex route, for now: MPAS's variational counterpart is MPAS-JEDI
  (whose EnVar the differentiable plan cites), and on an unstructured mesh
  the natural `B` is ensemble-only (3DEnVar with localisation) rather than
  line recursive filters. A hex 3DEnVar is a follow-on once V3 works on the
  WRF route, reusing its alpha machinery with a mesh correlation operator.

## 4. Design for woof

### 4.1 Formulation

Incremental 3D-Var in WRFDA's control space, exactly as section 1.2:
`x_a = x_b + U v`, `J(v) = 1/2 v^T v + 1/2 (d - H' U v)^T R^-1 (d - H' U v)`
(+ `Je`, `Jp` later), minimised by CG from `v = 0`, with outer loops that
recompute `d = y - H(x_b + U v)` with the nonlinear operators. WRFDA's
variable conventions are kept (psi/chi, pseudo-RH, ps as a fraction of
`p_s(x_b)`), so that `be.dat` means the same thing to both programs.

The increment is formed on WRFDA's analysis variables (u, v on the C grid,
T, q, p_s; then `da_transfer_xatoanalysis` turns them into WRF's
prognostics: `theta`, `mu`, `ph` by hydrostatic integration, moisture).
woof already has the forward side of that last step: the analysis bundle
accepts a WRF-format analysis, and `woof/da/hydrostatic.py` makes an
increment's geopotential hydrostatic at the column's own dry mass. The V1 PR ports `da_transfer_xatowrf` for the oracle lane.

### 4.2 Module layout

New package `woof/da/var/`, nothing imported on any forecast path:

| module | content |
|---|---|
| `__init__.py` | experimental banner; the import-graph test's anchor |
| `config.py` | `VarConfig` dataclass; WRFDA namelist import (`&wrfvar*` groups) and woof TOML table (section 4.3) |
| `be_dat.py` | `be.dat` reader/writer (section 4.9); `BackgroundError` with per-bin EOFs, regressions, length scales; SHA-256 |
| `control.py` | control-vector layout (`cv_mz`, sizes per variable, alpha block, VarBC block), device allocation, `dot` and `axpy` with the fixed-order reductions (section 4.7) |
| `transforms.py` | `U = U_p U_v U_h` and `U^T`: RF, EOF, balance, psi/chi to u/v, humidity, as kernel launches; one `ControlTransform` protocol (`apply`, `adjoint`, `size`) |
| `kernels/*.cu` | RF line sweeps (forward and adjoint, the three boundary forms), EOF batched products, balance, psi/chi stencil, interpolation gather/scatter, alpha products; through the existing Metal translator |
| `obsop/` | `ObsStencil` (fixed weights and indices from the innovation step) with `tl` (gather) and `ad` (fixed-order scatter); per-type operators wrapping the existing nonlinear ones (`obs_conventional`, `obs_point`, `obsop`, `obsop_cwp`) and adding their TL/AD; a WRFDA-interpolation mode for the oracle |
| `obs_wrfda.py` | reader for OBSPROC's `obs_gts_*.3DVAR` (and later the radar ASCII), producing P0's `ObsSet` |
| `qc.py` | `check_max_iv`-style gross check, `thin_conv` boxes; decisions recorded per observation |
| `cost.py` | `VarCost`: `J`, its parts and `grad J` in `v`, the Hessian-vector product for CG; built on P0's pairing and error table |
| `minimise.py` | CG with WRFDA's options (re-orthonormalisation, block preconditioner, `eps`, `ntmax`); later Lanczos; per-iteration log in WRFDA's format |
| `analysis.py` | outer loops, `x_a`, the WRFDA-variable to WRF-prognostic transfer, analysis bundle write |
| `hybrid.py` | alpha control variable from on-device LETKF members (V3) |
| `gen_be.py` | NMC and ensemble statistics to `be.dat` (V2) |
| `varbc.py`, `rtm/` | V4 radiances, opt-in |
| `oracle.py` | runs and parses the WRFDA oracle, compares (section 5) |

Tests: `tests/test_var_*.py`. Oracle fixtures under `tests/data/wrfda471/`
(test data, not package data, as for the advection oracle).

### 4.3 Configuration and command line

- **TOML:** a `[da.var]` table whose keys are WRFDA's names where a WRFDA
  key exists (`cv_options`, `ntmax`, `eps`, `max_ext_its`,
  `orthonorm_gradient`, `rf_passes`, `len_scaling1..5`,
  `var_scaling1..5`, `max_vert_var1..5`, `check_max_iv`, `thin_conv`,
  `thin_mesh_conv`, `ensdim_alpha`, `je_factor`,
  `alpha_corr_scale`, `alpha_vertloc_opt`, `num_fgat_time`,
  `test_transforms`, `test_gradient`). Defaults equal WRFDA's. woof-only
  keys get a `woof_` prefix: `woof_precision` (`oracle` | `production`),
  `woof_reduction` (`wrfda-serial` | `fixed-tree`),
  `woof_interpolation` (`wrfda` | `woof`).
- **WRFDA namelist import:** `config.py` reads a WRFDA `namelist.input`'s
  `&wrfvar*` groups and maps every key it knows. It refuses, by name, any
  key set to a non-default value that woof does not implement, rather than
  ignoring it. A WRFDA user's namelist then either runs with the same
  meaning or fails loudly.
- **Command:** a new `woof var` door (proposed name; the doc test for
  commands will hold the user docs to it once it exists) with `analyse`,
  `check` (the adjoint and gradient tests), `gen-be` and `oracle`
  subcommands. It reads a background (`wrfinput`/`wrfout` or a woof
  checkpoint), observations, `be.dat` and optional members, and writes an
  analysis bundle that the existing `woof run --analysis` door consumes,
  with `da_update_bc` applied there as today.
- **Install:** pure Python plus `.cu` kernels in the existing wheel layout,
  so no new extra is needed for 3D-Var. The CRTM build (section 4.11) and
  any NN option (section 6) are separate opt-in extras.

### 4.4 Device layout

- Control vector: one contiguous device buffer per variable block (v1..v5,
  alpha, VarBC). Each 3-D block is laid out `(mode, y, x)` after `U_v^T`,
  so an x-line is contiguous for the x sweep. The y sweep reads strided
  (or goes through one transpose kernel; measure both). WRFDA's MPI
  transposes (`da_transpose_z2x`, `x2y`, `y2z`) move data only; on one GPU
  they are layout choices, not arithmetic, so they cannot change a result.
- Increment and background on WRFDA's A-grid analysis variables plus C-grid
  winds, as WRFDA's `xa`/`xb`.
- Observations: structure-of-arrays, sorted by type then by (j, i) of the
  lower-left stencil corner. The sort is what makes the adjoint scatter
  deterministic (section 4.7).

### 4.5 Kernels

| kernel | parallelism | notes |
|---|---|---|
| RF x/y sweeps, forward and adjoint | one thread per line (`ny x mz` lines for x, `nx x mz` for y) | Sequential recursion along the line, exactly WRFDA's order, so the oracle tier can match WRFDA's words. A parallel-scan formulation would be faster on very long lines but reassociates the sums, so it can only be a fast-mode option. The pass-dependent boundary forms are three code paths. The adjoint is the reversed recursion with the boundary forms transposed |
| `sqrt(area)` scaling, `var_scaling` | elementwise | |
| EOF `U_v`, `U_v^T` | one thread block per column batch; `nk x mz` per latitude bin | `fixed_order_gemm`-style fixed summation order. nk is about 40-60, so this is tiny |
| balance `U_p`, `U_p^T` | per column: `agvz` is `nk x nk` per bin; `bvz`, `wgvz` per level | the AD of the `ps` sum is a per-column broadcast, no atomics |
| psi/chi to u/v and AD | 2-D stencil, elementwise | the AD is a stencil transpose written as a gather (each output point reads its four neighbours' contributions), so no atomics |
| pseudo-RH, `ps` fraction | elementwise | |
| `H'` (TL) | one thread per observation: gather 8 corners | |
| `H'^T` (AD) | one thread per grid point over its sorted observation list (CSR), or segmented reduction over the sorted list | not `atomicAdd`: float atomics are order-nondeterministic, and Metal's float atomics are not a match for CUDA's |
| alpha `sum_n alpha_n o x'_n` and AD | elementwise over members; the AD is a per-member product | |
| dot products | block partials in index order, then a fixed-order tree | section 4.7 |

Every kernel is a new `.cu` file. It goes through the Metal translator and
into the rounding-audit inventory (`tools/cuda_rounding_check.py`) like any
kernel, and is templated on the real type: a `double` instance for the
oracle tier and the binary64 adjoint tests, a `float` instance for
production.

### 4.6 Observation operators: TL and AD

- At the innovation step the nonlinear operator computes, per observation,
  its stencil: (indices, weights) for the horizontal corners and the
  vertical pair. For a linear operator `H' = H` restricted to that
  stencil; TL is a gather, AD the transpose.
- For a nonlinear operator (wind speed/direction, GNSS PW and refractivity,
  reflectivity) the innovation step also saves the background values the
  linearisation needs, as WRFDA's `_lin` routines do with `xb`.
- WRFDA's operators are the reference for the TL/AD formulas: each
  `da_transform_xtoy_<type>` and `_adj` pair is short (the `sound` TL is a
  few `da_interp_lin_3d` calls and, for speed/direction winds, a
  `da_uv_to_sd_lin`).
- woof's existing nonlinear operators stay the innovation path for woof's
  own mode; the `wrfda` interpolation mode reproduces `da_interp_lin_3d`
  and `zk` for the oracle lane.
- **Shared with P0.** The stencil is computed from the same `ObsSet` and
  FGAT pairing as P0's `J_o` (section 4.10), so that
  `VarCost.jo(v = 0) == woof.grad.obs_cost.evaluate(...).J` for the same
  observations, operators and errors, which is a test.

### 4.7 Deterministic inner products, and the precision tiers

CG is sensitive to its inner products: a different summation order changes
`step` in the last bit, and that grows over 50 iterations. Both of woof's
backends must agree, and the oracle must be matchable. Two reduction
modes:

- **`wrfda-serial`** (oracle tier): reproduces `da_dot`'s order exactly,
  i.e. the `mod(n,5)` head then the five-term groups, each group summed
  left to right and added to the running total. It is inherently serial;
  it runs as one thread (or a short chain of chunk carries that preserve
  the left-to-right association), and it is used only for oracle
  comparison and tests. *Estimate:* 10 M terms in about 10-30 ms on one GPU
  thread, acceptable for a test lane.
- **`fixed-tree`** (production): fixed chunk size (e.g. 4,096), each
  chunk summed in index order by one thread block with a fixed intra-block
  tree, partials combined in a fixed binary tree; no atomics and no
  vendor reductions (`cupy.sum`, cuBLAS `dot` and MLX `sum` all choose
  their own orders). Accumulation in binary64 on CUDA and in double-float
  (`f64_df64.h` arithmetic) or the strict soft binary64
  (`f64_soft.h`) on Metal. With strict numerics, the same chunking and
  the same binary64 words, CUDA and Metal give the same bits; this is a
  test (`test_var_reduction_backend_identity`).

The precision tiers:

| tier | storage | arithmetic | reductions | purpose |
|---|---|---|---|---|
| `oracle` | binary64 | binary64, `--fmad=false`, WRFDA's operation order | `wrfda-serial` | agree with WRFDA (section 5); CUDA only (Metal has no native binary64; its soft binary64 can run it slowly for tests) |
| `production` | binary32 fields and control vector | binary32 kernels | `fixed-tree`, binary64/df64 accumulation | the analysis; CUDA and Metal bit-identical under strict numerics, within a stated tolerance under fast |

A binary32 CG loses conjugacy faster than WRFDA's binary64 one. The
tutorial's `orthonorm_gradient = true` (full re-orthonormalisation against
all previous gradients) counters that at a cost of `O(iter^2)` dot
products. A V1 measurement decides between binary32 vectors with
re-orthonormalisation and binary64 control vectors (memory is small: a
200x200x50 cv5 vector is about 8 M values).

### 4.8 Verification against WRFDA (summary; section 5 is the gate list)

The oracle is WRFDA 4.7.1 built serially on pb (section 8). Each test feeds
both programs the same `fg`, `be.dat` and observation file, and compares:

- the per-iteration `J` and `|g|` lines;
- the analysis increment fields;
- the O-B and O-A per observation;
- the adjoint-test and gradient-test outputs.

### 4.9 `be.dat` compatibility

- **Read** WRFDA's `be.dat` byte for byte (section 1.5): detect endianness
  from the first record marker (it must be 12: three 4-byte integers) and
  the real size from the `lat_min` record (24 bytes means real*8). Handle
  the optional `eta_be` record. Refuse a variable-name mismatch, trailing
  bytes, or a record whose marker pair disagrees.
- **Interpolate** to the model's levels and latitudes exactly as
  `da_setup_be_regional` does (bins, `be_eta`/`eta_be` interpolation when
  the levels differ), in binary64. That code is in the oracle's comparison
  scope: V0 compares woof's interpolated `evec`, `val`, regression and
  `rf_alpha` arrays with the oracle's, dumped from a debug build of the
  same sources.
- **Write** (V2) the same format, so a woof-made `be.dat` runs in WRFDA:
  a cross-check that woof's gen_be means what WRFDA expects.
- Agreement targets: arrays derived from `be.dat` are equal to the last
  bit in the oracle tier (same binary64 arithmetic, same order). Transforms
  `U v` for a given `v` agree to 1e-13 relative (WRFDA's own transcendental
  calls, e.g. in `da_calculate_rf_factors`, use glibc libm; woof's use its
  own libm, so equality is not guaranteed).

### 4.10 Interfaces shared with the differentiable plan

Coordinated with P0 (`woof/grad/`, branch `grad/p0-obs-cost`, in progress)
and P2 (`woof/nn/`, #584, open):

- **Observations:** P0's `woof.grad.obs_cost.ObsSet` (flat table: platform,
  kind, station, time, lat, lon, pressure, height, elevation, value,
  error) is the input to 3D-Var too. `obs_wrfda.py` produces one;
  variational-only fields (the WRFDA QC flag, the thinning box) travel in a
  side table keyed by row index, not as new `ObsSet` columns, so P0's
  schema and digest stay its own.
- **Errors:** P0's error table (`woof/data/grad/obs-cost-types.v1.json`,
  `ErrorTable.sigma`) supplies `R` where the reader gives no error; the
  WRFDA lane uses the errors in OBSPROC's file, as WRFDA does.
- **Pairing:** P0's `pair` (FGAT nearest frame) decides which state an
  observation is compared with; V5's FGAT reuses it unchanged.
- **Cost:** P0's `evaluate` is the nonlinear `J_o`; `VarCost` adds `H'`,
  `H'^T` and `Jb`. One identity test binds the two.
- **Gradient kit:** the TL-convergence, dot-product and Taylor tests of the
  differentiable plan's P1.3 are the same tests V0/V1 need. Whichever lands
  first owns `woof/grad/checks.py` (proposed), and the other imports it.
- **NN artefacts:** any learned component in section 6 is a P2 artefact
  (`woof.nn.manifest`, weight SHA-256, input bounds) loaded through
  `woof.nn.models`, never a second loader.

### 4.11 Opt-in and default policy

- `woof.da.var` is imported only by the `woof var` door and its tests; an
  import-graph test proves no forecast path imports it.
- No kernel launches and no default words change: the forecast reads an
  analysis only through `--analysis BUNDLE`, which exists today.
- The identity tests the differentiable plan lists (section 8 there) run
  unchanged for every V PR.
- **Radiances:** a `crtm` opt-in builds JCSDA CRTM v2.4 or later (CC0)
  from a user-supplied source tree or a pinned download into a woof-owned
  directory, and links it through a small C shim. Coefficients are a
  user-supplied path. Neither enters the published wheels. The 2.3.0 copy
  inside WRF is never compiled or vendored (GPL-headed files). RTTOV, if
  ever, is a user-supplied library path behind its own opt-in, with its
  licence the user's own.
- `licenses/` gets the CRTM CC0 text when the opt-in lands; `PROVENANCE.md`
  records the WRFDA-derived files with the WRF notice.

### 4.12 Performance targets

*Estimates* for a 200x200x50 grid, cv5, `rf_passes = 6`, 10^5 observations,
A6000:

- **RF:** 5 variables x 3 passes x 2 directions x 2 sweeps, each about one
  read and one write of an 8 MB field: roughly 1 GB of traffic per `U`,
  about 2 ms at the card's bandwidth. The serial line length (200) is
  latency, not bandwidth; with 40 modes x 200 lines there are 8,000 lines,
  enough to fill the card.
- **EOF and balance:** under 1 GFLOP per `U`, so below 1 ms.
- **`H'`/`H'^T`:** 10^5 x 16 gathers, well below 1 ms.
- **Per CG iteration:** one `U`, one `U^T`, one `H'`, one `H'^T`, and a few
  dot products: about 5-10 ms.
- **Per analysis:** 50 iterations x 1 outer loop is **under 1 s** on the
  A6000; on the M3 Max, *estimate* 3-4x that.

Targets, gated in V1:

- a 50-iteration conventional 3D-Var on 200x200x50 in at most 5 s on the
  A6000 and 20 s on the M3 Max, excluding I/O;
- in the oracle tier, at most 10x the production tier's time.

The WRFDA serial wall time on the same case (section 8 has the tutorial
case's) is recorded as the reference, not as a gate.

## 5. Verification: the oracle lanes and gates

All on pb, with the build in section 8.

| lane | what | agreement |
|---|---|---|
| O1 `be.dat` | woof's reader against a debug dump of `be%v*%evec/val`, regressions, `rf_alpha` after `da_setup_be_regional` | bitwise (binary64) |
| O2 `U` and `U^T` | `U v` and `U^T x` for the same `v`, `x`, dumped from a debug build of WRFDA's own transforms (`analysis_type = 'RANDOMCV'` is the existing driver that applies `U` to a random `v`) | 1e-13 relative per field, oracle tier |
| O3 adjoint tests | woof's `<U v, x> = <v, U^T x>` and `<H' dx, dy> = <dx, H'^T dy>`; WRFDA's `test_transforms` output beside it | woof: 1e-13 binary64, 1e-6 binary32; WRFDA's printed ratios recorded |
| O4 gradient test | Taylor ratio over 8 decades, as `da_check_gradient` | ratio to 1 within 1e-6 at the best `a` |
| O5 innovations | `d` per observation for the same `obs_gts` file, after the same QC | same accepted set; `d` to 1e-10 relative |
| O6 minimisation | the `minimize_cg` lines: `J_0`, gradient norm `g_0`, `J` and the gradient norm at every iteration | `J_0` to 1e-12 relative; first 10 iterations to 1e-8; final `J` to 1e-6 (CG amplifies ulp differences) |
| O7 analysis | `wrfvar_output` against woof's analysis, per field | max abs increment difference under 1 % of the increment's RMS, and the increments' correlation at least 0.9999 |
| O8 backends | production tier CUDA vs Metal | bitwise under strict numerics; under fast, `J` per iteration to 1e-5 |

**Cases:**

1. The WRFDA tutorial (2008-02-05 12 UTC, 90x60x41, 60 km, cv5 `be.dat`,
   conventional obs; section 8). Public test data, not committed: the
   fixture records its URL and SHA-256 and a fetch step.
2. A woof regional case at 3 km with a woof-made `be.dat` (V2), and WRFDA
   run on the same files.
3. Single-observation tests (WRFDA's `pseudo` obs, `num_pseudo = 1`): the
   increment's shape shows `B` directly, the standard check of a `B` model.

WRFDA at `-O2` (its configure default) is the first oracle; an
`-O0 -ffp-contract=off` build is the tie-breaker when O2/O6 disagree,
as for the WRF oracles ([verify-exact.md](verify-exact.md)).

## 6. Gradient-based and neural-network options in the DA

Requested 2026-11-21 as an addition to the 3D-Var scope. Every option below
is opt-in, enters through an interface the classical path already has, and
is verified against the classical path, against observations, and with the
gradient tests. Default forecast words do not change under any of them.
The table at the end of the section sets them against each other.

Shared machinery from the differentiable plan:

- the P1 tensor bridge (CuPy/torch via DLPack, MLX arrays on Metal);
- P2.1's artefact manifest and runners (`woof/nn/`, #584), installed with
  that PR's torch extra (`nn-torch`, proposed there);
- P0's cost and observation table;
- P3's adjoints and P4's checkpointing.

Nothing here imports torch unless the option is named.

### 6.1 Gradient-based options

**G1. Automatic TL/AD of observation operators (and a checker for the
hand-written ones).**

- *What:* express a nonlinear operator's per-observation formula (not the
  stencil gather, which is already trivially linear) in torch or MLX. Take
  its Jacobian-vector and vector-Jacobian products with autograd, and use
  them (a) to prototype a new operator's TL/AD before a kernel exists and
  (b) to check every hand-written `.cu` TL/AD against.
- *Value:* the TL/AD of wind speed/direction, GNSS refractivity, PW and
  reflectivity are where hand-written adjoints go wrong; an autograd
  reference catches it in a unit test.
- *Interfaces:* the P1 bridge; `ObsOperator` (section 4.6) gains an
  optional `reference_vjp`.
- *Verification:*
  - dot-product test of the hand-written pair;
  - the hand-written VJP against autograd's to 1e-12 (binary64) on random
    profiles;
  - both against WRFDA's `_adj` output in O3.
- *Effort:* 1-1.5 w after P1.1.
- *Risks:* torch on the test path only, so none for the product.
- *Worth it:* yes, as the checker. Not as the production path: it would
  put torch in the analysis and lose the fixed-order CUDA/Metal identity
  that section 4.7 needs.

**G2. A differentiable cost for the minimiser.**

- *What:* `VarCost` exposes `J(v)`, `grad J(v)` and the Hessian-vector
  product as one object, built on P0's `evaluate` for the nonlinear `J_o`
  (section 4.10). CG uses the Hessian-vector product; outer loops and any
  quasi-Newton minimiser use `J` and `grad J`.
- *Value:* this is how 3D-Var is built anyway; stating it as the shared
  interface lets the same object serve P4's 4D-Var-lite and any learned
  minimiser (N4).
- *Verification:* O4's Taylor test, and `VarCost.jo(0) == P0 J_o`.
- *Effort:* inside V1.
- *Risks:* none beyond V1's.
- *Worth it:* yes.

**G3. FGAT and 4D-Var through P3/P4.**

- *What:* FGAT (`num_fgat_time`) uses P0's pairing over the forecast's
  frames; it needs no adjoint. 4D-Var replaces `H'` by `H'_t M'_{0->t}` and
  its adjoint by `M'^T H'^T`, using P3's adjoint and P4's checkpointing.
  The control vector, `U`, `R`, the minimiser and the output stay V0-V4's.
- *Value:* 4D-Var's dynamical consistency over a window.
- *Interfaces:* the `VarCost` `H'` slot takes a `TimeOperator`.
- *Verification:*
  - FGAT against WRFDA's FGAT on the tutorial's hourly files;
  - 4D-Var against WRFDA 4D-Var (built with WRFPLUS, the differentiable
    plan's oracle) on the tutorial's 4D-Var case (the test data has
    hourly obs 12-18 UTC).
- *Effort:* FGAT 1-2 w (V5); 4D-Var 3-4 w after P3/P4 (it is P4.2 there,
  counted once).
- *Risks:* P3's gate G3.
- *Worth it:* FGAT yes. 4D-Var only if P3 passes.

**G4. Gradient-based tuning of `B`, `R` and VarBC.**

- *What:*
  - (a) The classical, adjoint-free diagnostics first: Desroziers et al.
    (2005) consistency of O-B, O-A and A-B statistics, for which WRFDA
    ships `da_tune_obs_desroziers` (*measured*: it is built by
    `all_wrfvar`), to tune `R` factors and `var_scaling`.
  - (b) Then gradients of a verification score with respect to the
    hyperparameters (`len_scaling`, `var_scaling`, `je_factor`, error
    factors), by differentiating the analysis through the converged
    minimiser. Implicit differentiation of the optimality condition needs
    one extra linear solve with the same Hessian, no unrolling.
  - (c) VarBC coefficients are already control variables, so their
    gradient is part of `grad J` (V4).
- *Value:* `B` tuning is currently by hand in WRFDA practice; (a) and (b)
  make it measurable.
- *Interfaces:* `VarConfig` hyperparameters as a vector; the minimiser's
  Hessian-vector product.
- *Verification:*
  - (a) against WRFDA's tool on the same diagnostics;
  - (b) by finite differences of the score (P0.3's machinery).
- *Effort:* (a) 1 w, (b) 2-3 w.
- *Risks:* overfitting to one case, so tune on a training period and
  verify on another.
- *Worth it:* (a) yes, cheap. (b) after V2, when woof makes its own `B`.

### 6.2 Neural-network options (all opt-in)

**N1. Learned observation operators.**

- *What:* an NN in place of a physical operator, with autograd giving its
  TL/AD:
  - a radiance emulator in place of CRTM (Stegmann et al. 2022 emulate
    fast radiative transfer with deep learning);
  - a visible-reflectance operator (Scheck 2021, an NN forward operator
    for visible satellite images, with its adjoint);
  - a radar reflectivity operator (Stefanelli et al., arXiv 2512.18289,
    a preprint, uses a CNN operator's Jacobian in 3D-Var).
- *Value:*
  - GOES visible and near-infrared reflectances have no operator in woof
    or WRFDA today;
  - a radiance emulator runs on Metal, where CRTM (Fortran) does not;
  - an NN reflectivity operator is smooth where the scheme-specific one
    is not (the differentiable plan, section 3(f)).
- *Interfaces:* an `ObsOperator` whose `nonlinear` is the P2 runner and
  whose TL/AD are autograd's JVP/VJP (MLX: VJP only, mlx #4580); the
  artefact via `woof.nn.manifest`; input bounds and fallback via
  `woof.nn.safety`.
- *Verification:*
  - against the classical operator (CRTM v2.4 opt-in, woof's
    `refl.py`) on held-out model columns: bias and RMS per channel or per
    dBZ bin;
  - O-B statistics against real observations, which must be no worse
    than the classical operator's;
  - the dot-product test on the autograd pair;
  - a single-observation increment, which must look physical.
- *Effort:* 3-5 w per operator after V4, including training data from
  woof's own runs.
- *Risks:* extrapolation outside the training columns (the bounds check
  must fall back or reject the observation); emulator error entering `R`.
- *Worth it:* visible reflectance, yes (no alternative). Radiance
  emulator, after the CRTM path exists to train and check against. Radar,
  research.

**N2. Learned or NN-parameterised `B`.**

- *What:* replace `U_p` (balance) or the whole `U` by a learned transform:
  - an autoencoder latent space (Melinc and Zaplotnik 2024 run 3D-Var in
    a variational autoencoder's latent space; Melinc, Perkan and
    Zaplotnik 2026 learn a background-error model that captures
    flow-dependent balances);
  - an NN emulator of gen_be's binned statistics conditioned on the flow.
- *Value:* flow-dependent balance without an ensemble.
- *Interfaces:* the `ControlTransform` protocol (`apply`, `adjoint`,
  `size`) of section 4.2; autograd gives `adjoint`.
- *Verification:*
  - O3's adjoint test;
  - single-observation increments against gen_be's `B` and the ensemble's
    sample covariance;
  - an OSSE (`woof/da/osse.py`) comparing analysis error with the
    classical `B`.
- *Effort:* 4-6 w after V2.
- *Risks:*
  - a learned `B` whose `U U^T` is not a valid covariance at all
    resolutions;
  - training on woof's own forecasts at one resolution.
- *Worth it:* research option, after V3 shows what the ensemble already
  buys.

**N3. Learned bias correction.**

- *What:* NN-derived predictors inside VarBC's linear model
  (`b = sum beta_k p_k`, with `p_k` from an NN of the column and scan
  geometry), keeping VarBC's linear-in-coefficients form and its `Jp`.
  Model-bias correction in the forecast is a different thing (Laloyaux et
  al. 2022 estimate model bias with deep learning inside ECMWF's
  weak-constraint 4D-Var) and belongs with P2's NN physics.
- *Value:* bias predictors are the weakest part of radiance DA.
- *Interfaces:* `varbc.py`'s predictor list; a P2 artefact as a predictor.
- *Verification:* O-B bias and standard deviation per channel with and
  without; coefficients' stability over cycles.
- *Effort:* 2 w after V4's VarBC.
- *Risks:* the NN absorbs real signal (the classic VarBC aliasing risk),
  so keep anchor observations unbiased.
- *Worth it:* only once radiances are in.

**N4. NN-accelerated minimisation.**

- *What:*
  - a learned preconditioner for CG (Trappler and Vidard, arXiv
    2501.04369, a preprint, build a state-dependent preconditioner for
    the inner loop with ML);
  - a learned first guess for `v` (Frerix et al. 2021 learn an inverse
    observation operator to initialise and constrain variational DA).
- *Value:* fewer iterations. With the per-iteration cost of section 4.12
  (milliseconds), the gain on woof's GPUs is small.
- *Interfaces:* the minimiser's `precon` and `v_0` arguments.
- *Verification:* the converged `J` and analysis must equal the classical
  run's within O6/O7 tolerances (a preconditioner may change the path, not
  the minimum); count iterations.
- *Effort:* 2-3 w.
- *Risks:* a non-SPD preconditioner breaks CG.
- *Worth it:* low priority.

**N5. ML-based QC and thinning.**

- *What:*
  - proactive QC: Honda et al. (2024) use an ML predictor of EFSO to
    reject observations that would degrade the forecast, before it runs.
    P0.4 already plans EFSO from the LETKF weights, which supplies the
    labels;
  - a learned gross-error classifier beside `check_max_iv`.
- *Value:* fewer harmful observations at high cadence (radar, mesonets).
- *Interfaces:* `qc.py` decisions as a mask with a recorded reason per
  observation.
- *Verification:*
  - decisions against `check_max_iv` on the same set (agreement matrix);
  - impact by observation-denial experiments on held-out cases.
- *Effort:* 2-3 w after P0.4.
- *Risks:* rejecting the rare observations that matter most (extremes).
- *Worth it:* EFSO-based proactive QC, yes; a black-box gross-error
  classifier, no.

**N6. DA cycling a forecast model with P2's NN physics.**

- *What:* nothing new in the DA. A woof forecast with an NN column scheme
  (P2) provides the background and the ensemble; gen_be (V2) and the
  hybrid (V3) take their statistics from that model.
- *Value:* evaluates P2's hybrid model where it matters operationally (in
  cycling), and lets VarBC and O-B statistics expose its biases.
- *Verification:* cycled O-B and forecast scores against the physics-only
  model's cycle.
- *Effort:* about 1 w of configuration and scoring once V3 and P2.3
  exist.
- *Risks:* the NN model's `B` differs, so gen_be must be rerun, not
  reused.
- *Worth it:* yes, cheap.

**N7. End-to-end learned DA.**

- *What exists:*
  - diffusion-model analysis (DiffDA: Huang et al. 2024);
  - learned assimilation of satellite observations into an ML forecast
    model (FuXi-DA: Xu et al. 2025);
  - 4D-Var coupled to a data-driven forecast model (FengWu-4DVar: Xiao et
    al., arXiv 2312.12455);
  - observation-to-forecast pipelines with no classical DA (Aardvark
    Weather: Allen et al. 2025; GraphDOP: Alexe et al., arXiv
    2412.15687);
  - jointly learned variational models and solvers (Fablet et al. 2021).
- *Assessment:* these are global, about 0.25-1 degree, trained on
  ERA5-scale archives, and produce a state for an ML model rather than for
  WRF. For metal-woof they would be a second model beside the WRF-identical
  one, which the differentiable plan already declines. They also give no
  regional, convection-allowing analysis that a woof forecast can start
  from with known error properties.
- *Use instead:* as baselines through the existing Earth2 door
  (`woof/earth2_*.py`), scored against the same observations by P0's
  `J_o`, and the ML-DA literature as a source of components (N1, N2, N4).
  Brajard et al. (2020) and Bocquet (2023) review the DA-plus-ML surrogate
  line of work; Hatfield et al. (2021) show NN tangent-linear and adjoint
  models for DA, which is G1's idea applied to a whole model.
- *Effort:* a comparison harness, 1-2 w, if wanted.
- *Worth it:* no as a woof component; yes as a scored baseline.

**Summary of section 6:**

| option | depends on | effort | worth it |
|---|---|---|---|
| G1 autograd TL/AD checker | P1.1, V1 | 1-1.5 w | yes (checker) |
| G2 differentiable `VarCost` | P0.1 | in V1 | yes |
| G3 FGAT / 4D-Var | P0.1 / P3, P4 | 1-2 w / in P4.2 | FGAT yes; 4D-Var if G3 passes |
| G4 B/R/VarBC tuning | V1 (a), V2 (b), V4 (c) | 1 w + 2-3 w | (a) yes, (b) later |
| N1 learned obs operators | V4, P2.1 | 3-5 w each | visible yes; others later |
| N2 learned `B` | V2, V3, P2.1 | 4-6 w | research |
| N3 learned VarBC predictors | V4 | 2 w | after radiances |
| N4 learned preconditioner / first guess | V1 | 2-3 w | low |
| N5 EFSO-based proactive QC | P0.4, V1 | 2-3 w | yes |
| N6 cycling with NN physics | V3, P2.3 | about 1 w | yes |
| N7 end-to-end learned DA | none | 1-2 w (baseline harness) | baseline only |

## 7. Phased plan

Effort in focused engineer-weeks (w). Each row is one PR unless it says
otherwise; each PR keeps the default path byte-identical (section 4.11) and
is under the 300-file limit.

### V0. `be.dat`, the control-variable transforms and their adjoints (4-5 w)

| PR | content | effort |
|---|---|---|
| V0.1 | `be_dat.py` reader (both endians, real*4/*8, optional `eta_be`, cv5/6/7 name lists), interpolation to model levels and latitudes as `da_setup_be_regional`; O1 against the oracle's debug dump | 1-1.5 w |
| V0.2 | RF kernels (three boundary forms, forward and adjoint, binary64 and binary32), `da_calculate_rf_factors`; adjoint test | 1 w |
| V0.3 | EOF and balance transforms, psi/chi to u/v, pseudo-RH, `ps` fraction; `U`, `U^T`; adjoint tests; O2 against `RANDOMCV` | 1.5-2 w |
| V0.4 | `control.py` reductions (`wrfda-serial`, `fixed-tree`), backend identity test (O8 for dot products) | 0.5 w |

**Gate GV0:** O1 bitwise, O2 to 1e-13, adjoint tests to 1e-13 (binary64)
and 1e-6 (binary32) on CUDA and Metal, reductions bitwise across backends
under strict numerics. If O2 cannot reach 1e-13, find the operation-order
difference before V1. Every later gate inherits it.

### V1. Conventional 3D-Var against the WRFDA oracle (6-8 w)

| PR | content | effort |
|---|---|---|
| V1.1 | `obs_wrfda.py` (OBSPROC `obs_gts` ASCII to `ObsSet` plus side table); WRFDA QC (`check_max_iv`, `thin_conv`) | 1-1.5 w |
| V1.2 | Linear operators and their TL/AD: sound, sonde_sfc, synop, metar, ships, buoy, airep, pilot, profiler, geoamv, polaramv; `wrfda` interpolation mode; O5 | 1.5-2 w |
| V1.3 | Nonlinear: wind speed/direction, `sfc_assi_options`, gpspw, gpsref; TL tests; G1 checker if P1.1 has landed | 1-1.5 w |
| V1.4 | `VarCost`, `minimise.py` (CG, re-orthonormalisation, block preconditioner), outer loops, WRFDA-format iteration log; O4, O6 | 1-1.5 w |
| V1.5 | `analysis.py`: increment to WRF prognostics (`da_transfer_xatowrf`), analysis bundle, `woof var` door; O7; Desroziers diagnostics (G4a) | 1-1.5 w |
| V1.6 | Performance measurement on CUDA and Metal at 90x60x41 and 200x200x50; tier decision of section 4.7 | 0.5 w |

**Gate GV1:** O3-O7 pass on the tutorial case and on single-observation
cases. Production tier: CUDA and Metal bitwise under strict numerics. The
performance targets of section 4.12 are met. A forecast started from the
woof analysis matches, within the forecast's own CUDA/Metal tolerance, one
started from WRFDA's `wrfvar_output` through the same `--analysis` door.
**No-go:** if O6/O7 cannot be reached because WRFDA's operation order in
some transform cannot be reproduced on the GPU, record the residual and
keep V1 as "WRFDA-compatible to tolerance" rather than oracle-exact; this
does not stop V2-V3.

### V2. gen_be from woof forecasts (3-4 w)

| PR | content | effort |
|---|---|---|
| V2.1 | NMC pairs (24 h minus 12 h valid together) and ensemble differences from woof history or device members; psi/chi by a Poisson solve (as stage 0) | 1-1.5 w |
| V2.2 | Stages 1-4: binning (`bin_type` 5 first), regressions, vertical EOFs (woof's own eigensolver), length-scale fits; `be.dat` writer | 1.5-2 w |
| V2.3 | Cross-checks: woof's gen_be on WRFDA's own 11 sample forecasts against WRFDA's gen_be (same `be.dat` to tolerance); the woof-made `be.dat` runs in WRFDA | 0.5 w |

**Gate GV2:** woof's gen_be reproduces WRFDA's `be.dat` from the same
forecasts. Eigenvalues and regressions agree to 1e-6 relative, and length
scales to 1 %; the fit is iterative, so the tolerance is stated rather than
bitwise. Single-observation increments with the woof `be.dat` are
physically shaped. Over a 10-day cycle on one woof domain, the analysis
fits independent observations no worse than with WRFDA's tutorial-style
default `B`.

### V3. Hybrid 3DEnVar with the LETKF ensemble (3-4 w)

| PR | content | effort |
|---|---|---|
| V3.1 | `hybrid.py`: alpha control variables, `alpha_corr_scale` RF (reusing V0's kernels), vertical localisation (`alpha_vertloc_opt = 2`), `je_factor` (and the derived `jb_factor`); `x'_n` straight from device-resident LETKF members (no `ep` files); `ep_format` reader for the oracle lane | 2-2.5 w |
| V3.2 | Oracle: the same members written as WRFDA `ep` files, WRFDA hybrid against woof (O6/O7); cycling: 3DEnVar mean analysis plus LETKF perturbation update (the usual hybrid EnKF-Var pairing) | 1-1.5 w |

**Gate GV3:**

- WRFDA hybrid agreement (O6/O7 tolerances).
- On an OSSE (`woof/da/osse.py`), 3DEnVar's analysis error is no worse
  than both pure 3D-Var and pure LETKF at 95 % bootstrap confidence on at
  least one regime. If it is not, keep hybrid as an option and say so.

### V4. Radar, GOES, radiances (8-14 w)

| PR | content | effort |
|---|---|---|
| V4.1 | Radar radial velocity (TL/AD; geometry shared with `obsop.py`) and indirect reflectivity (Wang et al. 2013: retrieved `qr`/`qs`/`qg` and in-cloud humidity), against WRFDA's radar ASCII path | 2-3 w |
| V4.2 | GOES CWP (`obsop_cwp`, linear in the hydrometeors) as a variational operator; needs `cloud_cv_options >= 2` or a hydrometeor alpha (V3) | 1 w |
| V4.3 | `crtm` opt-in: CRTM v2.4+ build and C shim, coefficients path, TL/AD/K calls, radiance readers (AMSU-A/MHS BUFR first; the tutorial has AMSU), thinning; against WRFDA's CRTM path built with the same CRTM version | 3-6 w |
| V4.4 | VarBC (predictors, `Jp`, `VARBC.in/out`), cold and warm start | 2-3 w |

**Gate GV4:**

- O-B statistics per channel and per radar type match WRFDA's within
  stated tolerances on the same inputs.
- The opt-in builds leave the default wheel unchanged: a wheel-content
  test checks that no CRTM symbol or file is in it.

### V5. FGAT and 4D-Var hooks (2-4 w here; 4D-Var itself is P4.2)

| PR | content | effort |
|---|---|---|
| V5.1 | FGAT (`num_fgat_time`) with P0's pairing over forecast frames; against WRFDA FGAT on the tutorial's hourly files | 1-2 w |
| V5.2 | `TimeOperator` slot in `VarCost` for P3's `M'` and `M'^T`; a 4D-Var test case staged (WRFDA 4D-Var tutorial with WRFPLUS on pb) | 1-2 w |

**Gate GV5:** FGAT agrees with WRFDA's FGAT (O6/O7). 4D-Var waits for the
differentiable plan's G3.

### Effort summary

| phase | effort | depends on | gate |
|---|---|---|---|
| V0 | 4-5 w | none | GV0 |
| V1 | 6-8 w | V0; P0.1 interfaces | GV1 |
| V2 | 3-4 w | V0 | GV2 |
| V3 | 3-4 w | V1 | GV3 |
| V4 | 8-14 w | V1 (V3 for hydrometeor alpha) | GV4 |
| V5 | 2-4 w | V1, P0.1; 4D-Var needs P3/P4 | GV5 |
| section 6 options | see its table | as listed | per option |

V0 + V1 (10-13 w) give a conventional-observation 3D-Var that runs on CUDA
and Metal and is checked against WRFDA. Adding V2 + V3 (16-21 w in all)
gives woof-made `B` and hybrid 3DEnVar with the LETKF members, i.e. a
complete regional variational system for conventional data in about four to
five months of one person. Radiances (V4) are the largest single block, and
the one with external licensing.

## 8. The oracle build on pb (done in this study)

Done on 2026-11-17, in scratch on pb, kept there for V0/V1
(`/mnt/NVME2/woof-work/scratch/wrfda/`; nothing from it is committed). It
uses the existing `wrf-oracle:u2404` container that the WRF 4.7.1 and MPAS
8.4.2 oracles are built in, with no new dependencies.

**Build** (`scripts/build.sh` there):

```
# inside wrf-oracle:u2404, /d = /mnt/NVME2/woof-work
export NETCDF=/usr
cd /d/scratch/wrfda/src/WRFV4.7.1         # the release tarball, unmodified
printf "32\n" | ./configure wrfda         # 32 = GNU gfortran/gcc, serial
./compile all_wrfvar
```

- *Measured:* configure selects `-fdefault-real-8`, `-O2 -ftree-vectorize
  -funroll-loops`, `-fconvert=big-endian -frecord-marker=4`, and compiles
  the bundled CRTM 2.3.0 in (`-DCRTM`).
- *Measured:* 19 min wall at `J=6` under `nice` with the machine's load at
  about 30-40, 3.7 GB peak memory, exit status 0.
- Outputs: `var/build/da_wrfvar.exe` (21.5 MB), `da_update_bc.exe`, the
  `gen_be*` stages, `da_tune_obs_desroziers.exe`, and
  `var/obsproc/obsproc.exe`.
- A serial build is chosen on purpose: it has one summation order
  (section 1.3), so the oracle is deterministic.

**Tutorial case** (`scripts/run_tutorial.sh`):

- Data: the WRFDA v3.9 test data
  (`https://www2.mmm.ucar.edu/wrf/users/wrfda/download/V39/WRFDAV3.9-testdata.tar.gz`,
  SHA-256 `c6467c17a0beb9c41b3660c0e6bcf5e6c671b002015daf576573ae18900ef9a0`).
- Setup: 2008-02-05 12 UTC, 90x60x41 staggered (89x59x40 mass points),
  60 km. `fg` is `rc/2008020512/wrfinput_d01`, `ob.ascii` is OBSPROC
  output, and the namelist is `var/test/tutorial/namelist.input` (cv5,
  `ntmax = 50`, `orthonorm_gradient = true`, one outer loop).
- The first guess was made by WRF v3.9's real.exe, so WRFDA 4.7.1 refuses
  it unless `force_use_old_data = .true.` is added to `&time_control`. That
  is the only namelist change. A woof-made case (section 5, case 2) will
  not need it.

*Measured* results, serial, one thread:

| quantity | value |
|---|---|
| observations used | 41,857 |
| `J_0`, gradient norm `g_0` | `2.971045778742540D+04`, `4.231702456802100D+02` |
| iterations to gradient norm below 0.01 of its start | 29 |
| final `J` (`Jb`, `Jo`) | `2.166324942823930D+04` (2,689.669, 18,973.580) |
| wall time, whole run including I/O | 8.4 s and 8.7 s (two runs); 0.49 GB peak memory |
| repeatability | the two runs' 30 `minimize_cg` lines are byte-identical |
| `test_transforms` (0.8 s) | `<x, x>` against `<v_adj, v>` for `U`: `2.28357076546071E+07` and `...080E+07` (4e-14 relative); `H'`: equal to 15 digits (`4.47121256415551E+04`) |
| `test_gradient` (1.7 s) | Taylor ratio `T1` 0.99999039, 0.99999884, 0.99999943, 0.99999725, 0.99997282, 0.99972827 for `a` = 1e-10 ... 1e-5 |

These are the reference words for O3, O4 and O6 on case 1. The per-iteration
lines and the run outputs (`wrfvar_output`, `gts_omb_oma_01`, `jo`,
`statistics`, `cost_fn`, `grad_fn`) stay in scratch. V1 adds a script that
regenerates them from the pinned tarballs, and the test fixture records
their SHA-256 rather than committing the data.

Not done here: the `-O0 -ffp-contract=off` tie-breaker build, a debug
build with array dumps (V0.1 adds the dump calls as a patch file applied
to a copy, never to the pinned tree), WRFPLUS and 4D-Var, and an
OBSPROC run on the tutorial's `little_r` file.

## 9. Risks

| risk | effect | mitigation |
|---|---|---|
| WRFDA's operation order not reproducible on the GPU in a transform (e.g. a fused libm call in `da_calculate_rf_factors`) | O2/O6 stop at a tolerance instead of exact | `-O0 -ffp-contract=off` oracle as tie-breaker; record the residual and its cause; GV1's no-go keeps going on tolerance |
| binary32 CG loses conjugacy | slower convergence or a different minimum | re-orthonormalisation; binary64 control vector (small); measured in V1.6 |
| Nondeterministic adjoint scatter | CUDA and Metal disagree; run-to-run drift | sorted observations and CSR gather, no atomics (section 4.5); a backend identity test |
| `be.dat` binning and interpolation subtleties | woof's `B` differs from WRFDA's silently | O1 bitwise against a debug dump; single-observation tests |
| Background error statistics from woof forecasts differ by resolution or regime | poor analyses on new domains | V2 per domain; `len_scaling`/`var_scaling` tuning with Desroziers diagnostics (G4a) |
| Operator differences between woof's own and WRFDA's interpolation | O5 fails or two "correct" answers | the `wrfda` interpolation mode is the oracle lane; woof's mode is a separate, documented choice |
| Radiance licensing | a GPL or registered-licence file in a wheel | CRTM v2.4+ only, built outside the wheel; RTTOV user-supplied; wheel-content test (GV4) |
| Scope creep into WRFDA's long tail (33 operators, chem, global, wavelets) | never finished | section 3's out-of-scope list; one operator per PR |
| Overlap with the differentiable plan | duplicated observation or gradient code | section 4.10's shared interfaces; one owner per module |
| NN options entering the default path | changed default words | everything in section 6 is a named opt-in; the import-graph test refuses torch imports from `woof.da.var` |

## 10. References (each checked against its DOI, arXiv or publisher page)

WRFDA and variational DA:

- Barker, D. M., et al., 2004: A three-dimensional variational data
  assimilation system for MM5: implementation and initial results. *Mon.
  Wea. Rev.*, 132, 897-914,
  [doi:10.1175/1520-0493(2004)132<0897:ATVDAS>2.0.CO;2](https://doi.org/10.1175/1520-0493(2004)132%3C0897:ATVDAS%3E2.0.CO;2).
- Barker, D., et al., 2012: The Weather Research and Forecasting Model's
  Community Variational/Ensemble Data Assimilation System: WRFDA. *Bull.
  Amer. Meteor. Soc.*, 93, 831-843,
  [doi:10.1175/BAMS-D-11-00167.1](https://doi.org/10.1175/BAMS-D-11-00167.1).
- Huang, X.-Y., et al., 2009: Four-dimensional variational data
  assimilation for WRF: formulation and preliminary results. *Mon. Wea.
  Rev.*, 137, 299-314,
  [doi:10.1175/2008MWR2577.1](https://doi.org/10.1175/2008MWR2577.1).
- Descombes, G., T. Auligné, F. Vandenberghe, D. M. Barker and J. Barré,
  2015: Generalized background error covariance matrix model (GEN_BE
  v2.0). *Geosci. Model Dev.*, 8, 669-696,
  [doi:10.5194/gmd-8-669-2015](https://doi.org/10.5194/gmd-8-669-2015).
- Parrish, D. F., and J. C. Derber, 1992: The National Meteorological
  Center's spectral statistical-interpolation analysis system. *Mon. Wea.
  Rev.*, 120, 1747-1763,
  [doi:10.1175/1520-0493(1992)120<1747:TNMCSS>2.0.CO;2](https://doi.org/10.1175/1520-0493(1992)120%3C1747:TNMCSS%3E2.0.CO;2).
- Purser, R. J., W.-S. Wu, D. F. Parrish and N. M. Roberts, 2003a, b:
  Numerical aspects of the application of recursive filters to variational
  statistical analysis. Part I: spatially homogeneous and isotropic
  Gaussian covariances; Part II: spatially inhomogeneous and anisotropic
  general covariances. *Mon. Wea. Rev.*, 131, 1524-1535 and 1536-1548,
  [doi:10.1175//1520-0493(2003)131<1524:NAOTAO>2.0.CO;2](https://doi.org/10.1175//1520-0493(2003)131%3C1524:NAOTAO%3E2.0.CO;2),
  [doi:10.1175//2543.1](https://doi.org/10.1175//2543.1).
- Wu, W.-S., R. J. Purser and D. F. Parrish, 2002: Three-dimensional
  variational analysis with spatially inhomogeneous covariances. *Mon.
  Wea. Rev.*, 130, 2905-2916,
  [doi:10.1175/1520-0493(2002)130<2905:TDVAWS>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130%3C2905:TDVAWS%3E2.0.CO;2).
- Courtier, P., J.-N. Thépaut and A. Hollingsworth, 1994: A strategy for
  operational implementation of 4D-Var, using an incremental approach.
  *Quart. J. Roy. Meteor. Soc.*, 120, 1367-1387,
  [doi:10.1002/qj.49712051912](https://doi.org/10.1002/qj.49712051912).
- Fisher, M., 1998: Minimization algorithms for variational data
  assimilation. *ECMWF Seminar on Recent Developments in Numerical Methods
  for Atmospheric Modelling*, 364-385,
  [ECMWF eLibrary](https://www.ecmwf.int/en/elibrary/74480-minimization-algorithms-variational-data-assimilation).
- Desroziers, G., L. Berre, B. Chapnik and P. Poli, 2005: Diagnosis of
  observation, background and analysis-error statistics in observation
  space. *Quart. J. Roy. Meteor. Soc.*, 131, 3385-3396,
  [doi:10.1256/qj.05.108](https://doi.org/10.1256/qj.05.108).
- Lorenc, A. C., 2003: The potential of the ensemble Kalman filter for
  NWP: a comparison with 4D-Var. *Quart. J. Roy. Meteor. Soc.*, 129,
  3183-3203, [doi:10.1256/qj.02.132](https://doi.org/10.1256/qj.02.132).
- Wang, X., D. M. Barker, C. Snyder and T. M. Hamill, 2008: A hybrid
  ETKF-3DVAR data assimilation scheme for the WRF model. Part I: observing
  system simulation experiment. *Mon. Wea. Rev.*, 136, 5116-5131,
  [doi:10.1175/2008MWR2444.1](https://doi.org/10.1175/2008MWR2444.1).
- Hunt, B. R., E. J. Kostelich and I. Szunyogh, 2007: Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230, 112-126,
  [doi:10.1016/j.physd.2006.11.008](https://doi.org/10.1016/j.physd.2006.11.008).
- Auligné, T., A. P. McNally and D. P. Dee, 2007: Adaptive bias correction
  for satellite data in a numerical weather prediction system. *Quart. J.
  Roy. Meteor. Soc.*, 133, 631-642,
  [doi:10.1002/qj.56](https://doi.org/10.1002/qj.56).
- Derber, J. C., and W.-S. Wu, 1998: The use of TOVS cloud-cleared
  radiances in the NCEP SSI analysis system. *Mon. Wea. Rev.*, 126,
  2287-2299,
  [doi:10.1175/1520-0493(1998)126<2287:TUOTCC>2.0.CO;2](https://doi.org/10.1175/1520-0493(1998)126%3C2287:TUOTCC%3E2.0.CO;2).
- Xiao, Q., et al., 2005: Assimilation of Doppler radar observations with
  a regional 3DVAR system: impact of Doppler velocities on forecasts of a
  heavy rainfall case. *J. Appl. Meteor.*, 44, 768-788,
  [doi:10.1175/JAM2248.1](https://doi.org/10.1175/JAM2248.1).
- Xiao, Q., et al., 2007: An approach of radar reflectivity data
  assimilation and its assessment with the inland QPF of Typhoon Rusa
  (2002) at landfall. *J. Appl. Meteor. Climatol.*, 46, 14-22,
  [doi:10.1175/JAM2439.1](https://doi.org/10.1175/JAM2439.1).
- Wang, H., J. Sun, S. Fan and X.-Y. Huang, 2013: Indirect assimilation of
  radar reflectivity with WRF 3D-Var and its impact on prediction of four
  summertime convective events. *J. Appl. Meteor. Climatol.*, 52, 889-902,
  [doi:10.1175/JAMC-D-12-0120.1](https://doi.org/10.1175/JAMC-D-12-0120.1).
- Johnson, B. T., et al., 2023: The Community Radiative Transfer Model
  (CRTM): community-focused collaborative model development accelerating
  research to operations. *Bull. Amer. Meteor. Soc.*, 104, E1817-E1830,
  [doi:10.1175/BAMS-D-22-0015.1](https://doi.org/10.1175/BAMS-D-22-0015.1).
  Code: [github.com/JCSDA/crtm](https://github.com/JCSDA/crtm) (CC0 from
  v2.4.0).
- Saunders, R., et al., 2018: An update on the RTTOV fast radiative
  transfer model (currently at version 12). *Geosci. Model Dev.*, 11,
  2717-2737,
  [doi:10.5194/gmd-11-2717-2018](https://doi.org/10.5194/gmd-11-2717-2018).
- WRF Users' Guide, Chapter 6: WRF Data Assimilation (WRFDA),
  [v4.4 edition](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.4/users_guide_chap6.html);
  WRFDA test data,
  [download page](https://www2.mmm.ucar.edu/wrf/users/wrfda/download/testdata.html).

Machine learning and DA (section 6):

- Hatfield, S., et al., 2021: Building tangent-linear and adjoint models
  for data assimilation with neural networks. *J. Adv. Model. Earth
  Syst.*, 13, e2021MS002521,
  [doi:10.1029/2021MS002521](https://doi.org/10.1029/2021MS002521).
- Frerix, T., D. Kochkov, J. Smith, D. Cremers, M. Brenner and S. Hoyer,
  2021: Variational data assimilation with a learned inverse observation
  operator. *Proc. ICML*, PMLR 139,
  [proceedings.mlr.press/v139/frerix21a](https://proceedings.mlr.press/v139/frerix21a.html).
- Fablet, R., et al., 2021: Learning variational data assimilation models
  and solvers. *J. Adv. Model. Earth Syst.*, 13, e2021MS002572,
  [doi:10.1029/2021MS002572](https://doi.org/10.1029/2021MS002572).
- Brajard, J., A. Carrassi, M. Bocquet and L. Bertino, 2020: Combining
  data assimilation and machine learning to emulate a dynamical model from
  sparse and noisy observations: a case study with the Lorenz 96 model.
  *J. Comput. Sci.*, 44, 101171,
  [doi:10.1016/j.jocs.2020.101171](https://doi.org/10.1016/j.jocs.2020.101171).
- Bocquet, M., 2023: Surrogate modeling for the climate sciences dynamics
  with machine learning and data assimilation. *Front. Appl. Math.
  Stat.*, 9, 1133226,
  [doi:10.3389/fams.2023.1133226](https://doi.org/10.3389/fams.2023.1133226).
- Huang, L., L. Gianinazzi, Y. Yu, P. D. Dueben and T. Hoefler, 2024:
  DiffDA: a diffusion model for weather-scale data assimilation. *Proc.
  ICML*, PMLR 235, 19798-19815,
  [proceedings.mlr.press/v235/huang24h](https://proceedings.mlr.press/v235/huang24h.html).
- Xu, X., et al., 2025: FuXi-DA: a generalized deep learning data
  assimilation framework for assimilating satellite observations. *npj
  Clim. Atmos. Sci.*, 8, 156,
  [doi:10.1038/s41612-025-01039-3](https://doi.org/10.1038/s41612-025-01039-3).
- Xiao, Y., et al., 2023: FengWu-4DVar: coupling the data-driven weather
  forecasting model with 4D variational assimilation. Preprint,
  [arXiv:2312.12455](https://arxiv.org/abs/2312.12455).
- Allen, A., et al., 2025: End-to-end data-driven weather prediction.
  *Nature*, 641, 1172-1179,
  [doi:10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0).
- Alexe, M., et al., 2024: GraphDOP: towards skilful data-driven
  medium-range weather forecasts learnt and initialised directly from
  observations. Preprint,
  [arXiv:2412.15687](https://arxiv.org/abs/2412.15687).
- Laloyaux, P., et al., 2022: Deep learning to estimate model biases in an
  operational NWP assimilation system. *J. Adv. Model. Earth Syst.*, 14,
  e2022MS003016,
  [doi:10.1029/2022MS003016](https://doi.org/10.1029/2022MS003016).
- Melinc, B., and Ž. Zaplotnik, 2024: 3D-Var data assimilation using a
  variational autoencoder. *Quart. J. Roy. Meteor. Soc.*, 150, 2273-2295,
  [doi:10.1002/qj.4708](https://doi.org/10.1002/qj.4708).
- Melinc, B., U. Perkan and Ž. Zaplotnik, 2026: A unified neural
  background-error covariance model for midlatitude and tropical
  atmospheric data assimilation. *J. Adv. Model. Earth Syst.*, 18,
  e2025MS005360,
  [doi:10.1029/2025MS005360](https://doi.org/10.1029/2025MS005360).
- Stegmann, P. G., et al., 2022: A deep learning approach to fast
  radiative transfer. *J. Quant. Spectrosc. Radiat. Transfer*, 280,
  108088,
  [doi:10.1016/j.jqsrt.2022.108088](https://doi.org/10.1016/j.jqsrt.2022.108088).
- Scheck, L., 2021: A neural network based forward operator for visible
  satellite images and its adjoint. *J. Quant. Spectrosc. Radiat.
  Transfer*, 274, 107841,
  [doi:10.1016/j.jqsrt.2021.107841](https://doi.org/10.1016/j.jqsrt.2021.107841).
- Stefanelli, M., et al., 2025: A neural-network model-measurement-based
  observation operator for weather radar reflectivity assimilation.
  Preprint, [arXiv:2512.18289](https://arxiv.org/abs/2512.18289).
- Trappler, V., and A. Vidard, 2025: State-dependent preconditioning for
  the inner-loop in variational data assimilation using machine learning.
  Preprint, [arXiv:2501.04369](https://arxiv.org/abs/2501.04369).
- Honda, T., et al., 2024: Machine learning enables real-time proactive
  quality control: a proof-of-concept study. *Geophys. Res. Lett.*, 51,
  e2023GL107938,
  [doi:10.1029/2023GL107938](https://doi.org/10.1029/2023GL107938).
