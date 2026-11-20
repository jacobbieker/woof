# Variational data assimilation (WRFDA, 3D-Var first): scope and plan

Status: design study (the coordinator's brief is dated 2026-11-21; the
oracle build and measurements in section 8 are dated as run, 2026-11-17).
Nothing in this document is
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

- **3D-Var on the WRF route is feasible and medium-sized**: about 10-13
  engineer-weeks for conventional observations, WRFDA's own `be.dat`, the
  CG minimiser and a WRFDA oracle agreement gate (V0-V1), 3-4 more to make
  woof's own background-error statistics (V2), and 3-4 more for hybrid
  3DEnVar with the LETKF ensemble (V3): 16-21 weeks in all (section 7).
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
  the static weight in `da_setup_background_errors`. With `ensdim_alpha > 0`
  and `je_factor > 1` it sets `jb_factor = je_factor / (je_factor - 1)`, so
  `1/jb_factor + 1/je_factor = 1`. With `ensdim_alpha > 0` and
  `je_factor <= 1` it sets `jb_factor = -999` ("full ensemble mode"),
  zeroes every `max_vert_var*` and so drops the static `Jb` control
  variables. With `ensdim_alpha = 0`, `jb_factor = 1`.

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
| `varbc.py` | V4 radiances, opt-in; the CRTM builder, shim and operator are `woof/rtm/` (section 11.9), shared with the LETKF path |
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
  i.e. the `mod(n,5)` head then the five-term groups. Each product is added
  to the running total in turn: the Fortran statement
  `dtemp1 = dtemp1 + p1 + p2 + p3 + p4 + p5` parses as
  `((((dtemp1 + p1) + p2) + p3) + p4) + p5`, so the order is one long
  left-to-right sum, not per-group partial sums. It is inherently serial;
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
  `U v` for a given `v` agree to 1e-13 relative. `da_calculate_rf_factors`
  uses only arithmetic and `sqrt` (correctly rounded on both sides), so the
  RF coefficients themselves can be bitwise; the residual comes from
  operation order and contraction (FMA) differences in the longer
  transforms, which the `-O0 -ffp-contract=off` oracle build isolates.

### 4.10 Interfaces shared with the differentiable plan

Coordinated with P0 (`woof/grad/`, branch `grad/p0-obs-cost`, in progress)
and P2 (`woof/nn/`, #584, merged; `docs/nn-physics.md`):

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
  about 2 ms at the card's bandwidth. The sweeps are latency-bound,
  not bandwidth-bound: each thread walks a 200-point line serially, and one
  sweep of one variable has only 200 x 50 = 10,000 lines (about 1 per
  CUDA core), so a single sweep cannot hide memory latency. The fix is to
  batch: all five variables (and, where the boundary form allows, both
  directions of independent lines) in one launch, which gives about 50,000
  lines; in fast mode a parallel first-order linear-recurrence scan along
  the line is the alternative. *Estimate*, widened for this: 2-10 ms per
  `U`, measured in V1.6.
- **EOF and balance:** under 1 GFLOP per `U`, so below 1 ms.
- **`H'`/`H'^T`:** 10^5 x 16 gathers, well below 1 ms.
- **Per CG iteration:** one `U`, one `U^T`, one `H'`, one `H'^T`, and a few
  dot products: about 5-25 ms.
- **Re-orthonormalisation:** iteration `k` costs `k` extra dot products and
  `k` axpys on the 8 M-value control vector; over 50 iterations about 1,275
  of each, i.e. roughly 80 GB of traffic in binary64, about 0.1-0.15 s on
  the A6000 with `fixed-tree` reductions. It also needs all previous
  gradients stored (50 x 64 MB in binary64).
- **Per analysis:** 50 iterations x 1 outer loop is **about 1-1.5 s** on
  the A6000; on the M3 Max, *estimate* 3-4x that.

Targets, gated in V1:

- a 50-iteration conventional 3D-Var on 200x200x50 in at most 5 s on the
  A6000 and 20 s on the M3 Max, excluding I/O;
- the oracle tier has no time target: `wrfda-serial` reductions run on one
  thread, and with re-orthonormalisation that is `O(iter^2)` serial dot
  products (about 1,300 of 8 M terms each, minutes rather than seconds).
  It runs only on the oracle cases (90x60x41), where its time is recorded,
  not gated.

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

Added to the 3D-Var scope by the same brief (dated 2026-11-21). Every option below
is opt-in, enters through an interface the classical path already has, and
is verified against the classical path, against observations, and with the
gradient tests. Default forecast words do not change under any of them.
The table at the end of the section sets them against each other.

Shared machinery from the differentiable plan:

- the P1 tensor bridge (CuPy/torch via DLPack, MLX arrays on Metal);
- P2.1's artefact manifest and runners (`woof/nn/`, #584), installed with
  the `nn-torch` extra that PR added;
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
production-tier performance targets of section 4.12 are met (the oracle
tier is recorded, not gated). A forecast started from the
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
| V3.1 | `hybrid.py`: alpha control variables, `alpha_corr_scale` RF (reusing V0's kernels), vertical localisation (`alpha_vertloc_opt = 2`), `je_factor` and the derived `jb_factor` (all three branches: hybrid, full-ensemble `-999` with the static CVs dropped, and `ensdim_alpha = 0`); `x'_n` straight from device-resident LETKF members (no `ep` files); `ep_format` reader for the oracle lane | 2-2.5 w |
| V3.2 | Oracle: the same members written as WRFDA `ep` files, WRFDA hybrid against woof (O6/O7); cycling: 3DEnVar mean analysis plus LETKF perturbation update (the usual hybrid EnKF-Var pairing) | 1-1.5 w |

**Gate GV3:**

- WRFDA hybrid agreement at O6/O7 tolerances, or, if GV1 ended on its
  no-go branch, at the residual GV1 recorded ("WRFDA-compatible to
  tolerance"); GV3 inherits V1's status rather than asking for more.
- On an OSSE (`woof/da/osse.py`), 3DEnVar's analysis error is no worse
  than both pure 3D-Var and pure LETKF at 95 % bootstrap confidence on at
  least one regime. If it is not, keep hybrid as an option and say so.

Status (2026-11-20): V3.1 and the oracle half of V3.2 landed
(`woof/da/var/hybrid.py`, `tests/test_var_hybrid.py`,
`tests/test_var_hybrid_o6.py`, `tools/wrfda_oracle/run_hybrid.sh`). On the
tutorial case with ten `gen_be` forecasts as members (`je_factor = 2`,
`alpha_corr_scale = 200` km, `alpha_vertloc_opt = 2`) the reference tier
matches all 51 of WRFDA's `minimize_cg` lines to 3.5e-14 relative, also
with woof's own log-p vertical localisation in place of `be.vertloc.dat`.
The oracle tier equals the reference word for word on the device; the
production tier meets the final `J` to 7e-8 with the same 50 iterations and
one digest in fast and strict. The full-ensemble branch (`je_factor <= 1`,
`jb_factor = -999`, static control variables dropped) builds its cost, but
WRFDA's `precondition_cg` divides by zero there, so woof skips the empty blocks
(as for `max_vert_var5 = 0`). O6's namelist leaves `precondition_cg` off; its
diagonal is checked against a transcription of `da_minimise_cg`'s loop. Still open: the LETKF-cycling
pairing and the OSSE half of GV3.

### V4. Hydrometeor CVs, radar, GOES, radiances (9.5-16 w)

| PR | content | effort |
|---|---|---|
| V4.0 | Hydrometeor control variables (`cloud_cv_options = 2` first: `qcw`, `qrn`, `qci`, `qsn`, `qgr` as CVs with WRFDA's `be.dat` hydrometeor records, their RF/EOF transforms and adjoints; `cloud_cv_options = 3` later); O1/O2 extended to the hydrometeor records; needed by V4.1's indirect reflectivity and V4.2 | 1.5-2 w |
| V4.1 | Radar radial velocity (TL/AD; geometry shared with `obsop.py`) and indirect reflectivity (Wang et al. 2013: retrieved `qr`/`qs`/`qg` and in-cloud humidity), against WRFDA's radar ASCII path | 2-3 w |
| V4.2 | GOES CWP (`obsop_cwp`, linear in the hydrometeors) as a variational operator; needs V4.0's hydrometeor CVs (or a hydrometeor alpha from V3) | 1 w |
| V4.3 | `crtm` opt-in: TL/AD/K calls through `woof/rtm/` (its CRTM v2.4+ build, C shim and coefficients path are section 11's R4, one builder for both paths), radiance readers (AMSU-A/MHS BUFR first; the tutorial has AMSU), thinning; against WRFDA's CRTM path built with the same CRTM version | 3-6 w |
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

**Gate GV5:** FGAT agrees with WRFDA's FGAT (O6/O7, or GV1's recorded
residual, as for GV3). 4D-Var waits for the
differentiable plan's G3.

### Effort summary

| phase | effort | depends on | gate |
|---|---|---|---|
| V0 | 4-5 w | none | GV0 |
| V1 | 6-8 w | V0; P0.1 interfaces | GV1 |
| V2 | 3-4 w | V0 | GV2 |
| V3 | 3-4 w | V1 | GV3 |
| V4 | 9.5-16 w (V4.0 hydrometeor CVs 1.5-2 w, then 8-14 w) | V1 (V3 for hydrometeor alpha) | GV4 |
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
| WRFDA's operation order not reproducible on the GPU in a transform (e.g. the compiler's FMA contraction or reassociation in the EOF or balance loops) | O2/O6 stop at a tolerance instead of exact | `-O0 -ffp-contract=off` oracle as tie-breaker; record the residual and its cause; GV1's no-go keeps going on tolerance |
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

## 11. Satellite imagers: SatWeave and native imager radiances

Status: design (2026-11-19). The user asked for SatWeave's radiances and
brightness temperatures, and those of other imagers, to reach woof's initial
conditions. This section states how. The pieces land as small PRs in the
order of section 11.10. The forward operator feeds the LETKF first (path A),
which needs no adjoint. The tangent-linear and adjoint feed 3D-Var at V4
(path B). Everything in this section is opt-in. No default forecast word
changes, and the default wheel carries no CRTM, no cloud-storage client and
no coefficient file.

### 11.1 SatWeave as published

[SatWeave](https://developers.google.com/weathernext/guides/satweave) is a
global geostationary mosaic made by Google DeepMind and Google Research. It
blends GOES-East, GOES-West, Meteosat 0°, Meteosat IODC and Himawari
onto one grid.

- **Grid:** 0.1° equirectangular (EPSG:4326), 1801 x 3600. Latitude runs
  90 to -90 (north first) and longitude 0 to 359.9.
- **Bands:** 17 float32 bands.
  - Seven reflectances in %: `mosaic_470_nm`, `509`, `640`, `863`, `1370`,
    `1610` and `2250`.
  - Ten brightness temperatures in K: `mosaic_3830_nm`, `6200`, `6900`,
    `7330`, `8590`, `9620`, `10400`, `11200`, `12400` and `13300`.
  - Each band's attributes name the source channels, e.g. for 6200 nm:
    "GOES-East (ABI B08), GOES-West (ABI B08), Himawari (AHI B08),
    Meteosat Prime & IODC (SEVIRI WV6.2)". Each band also carries a
    `valid_range` (150-350 K for the infrared).
- **Stores:**
  - `gs://satweave_1/zarr/2016_to_2025/mosaic.zarr` has 30-minute steps
    (175 344 of them). Its time is in `minutes since 2016-01-01`.
  - From 2026, `gs://satweave_1/zarr/2026_to_present/<YYYYMMDDTHHMMSS>/mosaic.zarr`
    holds one store per hour, available about 1 h after the hour. Its time
    is in `days since <that hour>`, so a reader has to decode the CF units
    of each store.
- **Format (measured 2026-11-19):**
  - Zarr v3 with consolidated metadata. The chunk shape is
    `(1, 1801, 3600)`, i.e. one whole-globe chunk per band per step.
  - Each chunk is the `bytes` codec (little-endian) followed by `zstd`.
    One 6.2 µm chunk is 8.2 MB on the wire.
  - Anonymous HTTPS reads work
    (`https://storage.googleapis.com/satweave_1/...`), so the reader needs
    no Google account and no GCS client.
  - A bounding box still costs a whole chunk per band per step, because the
    store has no spatial chunking.
- **Licence:** CC BY 4.0. The third-party credits are EUMETSAT (Meteosat,
  modified by Google), NOAA (GOES-R, via Google Cloud Public Datasets) and
  JMA (Himawari 8 and 9, via NOAA NODD, modified by Google). woof's
  provenance records carry that attribution. woof commits and ships no
  SatWeave data.
- **NaNs:**
  - beyond the geostationary disc (the 2026-11-18T12 6.2 µm chunk is 69 %
    finite: 73 % of the 70° rows, 10 % of the 80° rows);
  - for bands a sector's satellite does not carry;
  - for reflectances at night;
  - in 2016-2017, at the 30-minute steps between the legacy imagers'
    3-hourly scans over the Americas.
- **Not published:** a per-pixel satellite id, the view geometry and the
  scan time.
- **Known artefacts:** limb artefacts in the MSG infrared, glint, missing
  scan lines, seams along overlaps and outlier pixels.
- **Sensor eras:**
  - 2016 to July 2017: the five-band legacy GOES-13/15 imagers.
  - From July 2017: GOES-16 ABI in the East.
  - From November 2018: GOES-17, later GOES-18, ABI in the West.

### 11.2 Recovering the satellite and the view geometry per pixel

The operator needs the view zenith angle and the sensor. Both follow from
which satellite saw the pixel. SatWeave does not publish that, so woof
reconstructs it in three steps.

1. **Band availability.** Some bands are carried by only some satellites,
   so the NaN pattern names the sector.
   - 1370 nm: only GOES.
   - 509 nm: only Himawari.
   - 470, 2250, 6900 and 10400 nm: GOES and Himawari, never Meteosat.
   - So a pixel with a finite 6200 nm value and a NaN 6900 nm value is in
     a Meteosat sector. That test works at night, from the infrared alone.
     It holds only from 2018-11-14, when every sector is ABI, AHI or
     SEVIRI. Before that, the legacy GOES-13/15 imagers had a water-vapour
     channel but no 6.9 µm band, and would be misread as Meteosat.
2. **Nominal sub-satellite longitudes, per era.** woof keeps a dated table:
   - GOES-East at 75.2° W: GOES-16, then GOES-19 from 2025-04-04 (the
     data transition; GOES-19 was declared operational on 2025-04-07).
   - GOES-West at 137.2° W: GOES-17, then GOES-18 from 2023-01-04.
     GOES-17 drifted to 137.2° W in November 2018 and was declared
     operational on 2019-02-12; from 2018-11-14 woof treats it as the
     West sector at its final position.
   - Himawari at 140.7° E: Himawari-8, then Himawari-9 from 2022-12-13.
   - Meteosat 0°: Meteosat-11 (SEVIRI) from 2018-02-20.
   - Meteosat IODC: Meteosat-8 at 41.5° E, then Meteosat-9 at 45.5° E from
     2022-06-01.

   PR R2 cites each row against the operator's notice. It also confirms
   whether the 0° service has moved to Meteosat-12 (FCI), and which
   instrument SatWeave uses after that date; the band attributes still name
   SEVIRI. The table starts on 2018-11-14. The archive's earlier years
   (2016 to 2018) are refused for reconstruction by name, never guessed:
   they need GOES-13/15 (no 6.9 µm), GOES-16 at its 89.5° W checkout
   position (July to December 2017), Meteosat-10 at 0° (until 2018-02-20)
   and Meteosat-7 IODC at 57.5° E (until 2017), each with its own rows and
   band test. Within the band-availability
   class, a pixel's satellite is the candidate with the smallest view
   zenith angle, which is the usual mosaic rule.
3. **Geometry.** For a geostationary satellite at height 35 786 km above
   the equator at longitude `lambda_s`, the view zenith, view azimuth and
   instrument scan (nadir) angle of each pixel are closed-form. The
   reconstruction stores them with the satellite id. The solar zenith comes
   from the nominal time.

**Overlaps.** SatWeave blends overlapping sectors, so in an overlap a value
is a mixture of two instruments at two view angles. The reconstruction flags
every pixel where a second satellite sees it with a view zenith under 70°.
Path A rejects flagged pixels at first. Later they are used with an error
inflated by a representativeness term (11.7), estimated from the O-B
difference between flagged and unflagged pixels of the same sector.

**Time.** A full-disc scan takes about 10 min (ABI, AHI) or 12-15 min
(SEVIRI), so a pixel's true time is up to about 15 min from the nominal one.
Measured against native ABI (11.12): SatWeave's step at `T` is the scan
that *starts* at `T`. Over the central US at 2025-03-01 18:00 UTC, the
GOES-16 scan of 18:00-18:10 matches SatWeave with a 6.2 µm difference of
0.20 K std. The 17:50 scan gives 0.54 K. So a pixel's time is `T` plus 0
to 10 min (ABI), never before `T`. FGAT pairing takes `T` plus the row's
offset in the scan. The timing error enters as a
representativeness term that scales with the band's local tendency. That
term is measured as the O-B difference between consecutive 30-minute archive
steps, not assumed.

### 11.3 RTM coefficients for a cross-calibrated mosaic

SatWeave cross-calibrates its sources to one another, and does not publish
how. woof simulates each pixel with its own sector's instrument coefficients
in CRTM, and treats the cross-calibration as a bias:

| SatWeave band | ABI (GOES-16/18/19) | AHI (Himawari-8/9) | SEVIRI (Meteosat) |
|---|---|---|---|
| 3830 nm | ch 7 | band 7 | IR3.9 (ch 4) |
| 6200 nm | ch 8 | band 8 | WV6.2 (ch 5) |
| 6900 nm | ch 9 | band 9 | not carried |
| 7330 nm | ch 10 | band 10 | WV7.3 (ch 6) |
| 8590 nm | ch 11 | band 11 | IR8.7 (ch 7) |
| 9620 nm | ch 12 | band 12 | IR9.7 (ch 8) |
| 10400 nm | ch 13 | band 13 | not carried |
| 11200 nm | ch 14 | band 14 | IR10.8 (ch 9) |
| 12400 nm | ch 15 | band 15 | IR12.0 (ch 10) |
| 13300 nm | ch 16 | band 16 | IR13.4 (ch 11) |

- The bias model is `b(band, sector) = b0 + sum_k beta_k p_k`. The
  predictors are the usual ones: view zenith, the band's own simulated
  brightness temperature, and layer thicknesses for the sounding bands.
- Path A estimates `b0` and the `beta` as a static correction, by least
  squares on clear-sky O-B over a training window. It is stored per
  `(band, sector, era)` with the window's digest.
- Path B makes the `beta` VarBC control variables (section 1.9).
- SEVIRI's 10.8 µm channel standing in for 11.2 µm, and 13.4 µm for
  13.3 µm, is where the sector bias is largest. That is why the bias is
  per sector and not per band.
- **Measured, SatWeave against native ABI (2026-11-19).** GOES-16,
  2025-03-01 18:00 UTC, box 105-80 W, 25-45 N. Native ABI L1b from the
  bkr/geo virtual store (11.12) was averaged into the mosaic's 0.1° cells
  (at least 9 pixels per cell). Pixels in overlaps or with view zenith
  over 65° were excluded, leaving 6140 cells:

  | band | SatWeave minus ABI, mean | std | 5-95 % | correlation |
  |---|---|---|---|---|
  | 6200 nm | 0.00 K | 0.20 K | -0.17 to 0.19 K | 0.997 |
  | 6900 nm | 0.00 K | 0.33 K | -0.26 to 0.28 K | 0.994 |
  | 7330 nm | 0.00 K | 0.62 K | -0.78 to 0.82 K | 0.992 |
  | 11200 nm | 0.00 K | 2.19 K | -3.4 to 3.6 K | 0.990 |

  GOES-East in SatWeave is therefore not adjusted. The spread is
  resampling, since SatWeave's own regridding is not a cell mean, and it is
  largest in the window band, where sub-cell contrast is largest. The other
  sectors are measured the same way in R4 (11.12 has the sources): AHI from
  the Himawari virtual stores, SEVIRI from `msg_3000m`/`odegree_3000m`.
  That gives the per-sector cross-calibration bias that 11.3 treats as a
  static term.
- **Measured (2026-11-19):**
  - CRTM 2.4.1's coefficient package (`crtm_coefficients_2.4.1_skylab_4.0`)
    has `abi_g16`, `abi_g17`, `abi_g18`, `ahi_himawari8`, `ahi_himawari9`
    and `seviri_m08`-`m11`, but no GOES-19.
  - The `abi_g19` files that CRTM's maintainers posted separately do not
    work with v2.4.1. The 2024 binary gives infinite brightness
    temperatures, and the 2026 SpcCoeff fails its byte-order check.
  - So until a v2.4-readable GOES-19 file exists, the GOES-East sector from
    2025-04-07 uses `abi_g16`. The two ABIs are built to the same
    specification, and the residual goes into the sector bias. The
    provenance records the substitution.

### 11.4 Superobbing and thinning onto the model grid

- **The model grid is coarser than 0.1° (dx > 11 km).**
  - Average the clear pixels inside each model column's cell, which gives
    a superob.
  - Require a minimum clear fraction, 2/3 at first.
  - Keep the pixel standard deviation as a representativeness indicator.
- **The grid is finer (dx < 11 km).**
  - Interpolate the model to the pixel instead, by taking the model column
    nearest the pixel centre.
  - Thin to one observation per box of `thin_km`, 30 km at first. Imager
    errors are horizontally correlated over tens of kilometres, and a
    diagonal `R` with every pixel would overweight them.
  - Within each box keep the pixel nearest the box centre, not the
    "best" one, so the choice does not depend on the innovation.
- **Every grid.**
  - Thin in time to one step per analysis window, the step nearest the
    analysis time.
  - Columns in the regional boundary zone (the relaxation rows) are never
    used. P0's `ObsSet` enforces the same rule.

### 11.5 Clear-sky first, all-sky later

Path A is clear-sky. Cloud screening is symmetric: an observation is used
only if both the observation and the background are clear. That avoids the
one-sided bias of observation-only screening (Geer et al. 2018).

- **Observation-side tests**, all on superob boxes:
  - The 11.2 µm window O-B is below `-3 K`. Cloud is colder than the clear
    simulation.
  - The 11.2 µm spatial standard deviation inside the box exceeds 1 K,
    which marks cloud edges.
  - The split window, 11.2 minus 12.4 µm, flags thin cirrus.
  - At night only, 3.83 minus 11.2 µm flags low cloud and fog.
- **Background-side test:** the member-mean condensate path *above the
  band's Jacobian-peak pressure minus 100 hPa* is above a threshold
  (`qc + qi + qs` > 5 g m-2, the `obsop_cwp` operator). The window bands
  use the whole column. Integrating only above the weighting function lets
  the water-vapour exception below keep a column whose background has the
  same low cloud as the observation.
- **Water-vapour bands above cloud.** A cloud below a band's weighting
  function does not affect it.
  1. Estimate the cloud-top pressure by matching the 11.2 µm brightness
     temperature to the background temperature profile.
  2. Keep the 6.2 µm band if that pressure is more than 100 hPa below
     (higher pressure than) the band's Jacobian peak.

  This keeps the band over low cloud, where most of its information is.
- **All-sky later.**
  - CRTM's cloud scattering needs the CloudCoeff file and per-species
    effective radii from the microphysics.
  - The error model is symmetric and cloud-dependent: Okamoto et al. 2014
    for the infrared, and Minamide and Zhang 2017 for adaptive inflation.
  - Honda et al. 2018 assimilated all-sky Himawari-8 infrared radiances
    with an LETKF, the same kind of filter as woof's. That is the
    reference for the all-sky step.

### 11.6 Which bands first

1. **Water vapour 6.2, 7.3 and 6.9 µm, clear-sky.** 6.2 and 7.3 come from
   all five satellites, 6.9 from three. Over the US standard-like test
   profile, CRTM 2.4.1 puts their Jacobian peaks at about 315, 560 and
   380 hPa (ABI, measured in section 11.8). The surface sensitivity of 6.2
   and 6.9 is nil, so their only surface concern is high terrain for 7.3.
   They constrain the humidity the conventional network misses.
2. **Infrared windows 10.4, 11.2 and 12.4 µm, clear-sky over sea.** These
   are skin-temperature and low-level-humidity information, and the sea
   emissivity is known (CRTM's Nalli model). Over land they wait for an
   emissivity atlas (11.7).
3. **13.3 µm (CO2), 8.59 µm, 9.62 µm (ozone; needs a model ozone profile),
   and 3.83 µm at night.** The 3.83 µm band is solar-contaminated by day.
4. **Reflectances, last.** They need a visible RT or a neural observation
   operator (11.9).

### 11.7 Error model and quality control

- **Initial errors (standard deviations, clear-sky):**
  - water vapour: 1.5 K;
  - windows: 2.0 K;
  - others: 2.5 K.

  Desroziers et al. (2005) diagnostics then tune each `(band, sector)`
  from the first month of O-B and O-A.
- **Inflation:**
  - by the superob's pixel standard deviation, added in quadrature;
  - by the timing term (11.2);
  - by the overlap term once overlaps are used.
- **Gross check:** after bias correction, `|O - B| > 3 sigma` is rejected.
  The rejection is recorded per observation, as P0's frozen QC does.
- **Limb:**
  - View zenith above 65° is rejected.
  - From 55° to 65° the error is inflated linearly to twice its value.
  - Meteosat sectors stop at 55° because of the published MSG infrared
    limb artefacts.
- **Glint:** none is needed at first. 3.83 µm is used only at night
  (11.6), and the other infrared bands are unaffected by glint. A daytime
  3.83 µm would add a glint-angle cut (under 25°).
- **Surface:**
  - Window bands over land wait for an emissivity atlas. The CRTM IR land
    model is per surface class, which is too coarse over deserts.
  - Any band whose skin-temperature Jacobian `dBT/dT_skin` exceeds 0.1 in
    the background is rejected over land and sea ice. CRTM's K-matrix
    gives that Jacobian directly.
  - The test drops 7.3 µm over the Tibetan plateau and the Andes and keeps
    it elsewhere.
- **Model top:** a band whose Jacobian peaks above the model's top level
  is rejected. CRTM adds climatological layers above the model top, but
  the analysis cannot correct them.
- **Inter-channel correlation:** the water-vapour bands' errors are
  correlated. Path A keeps `R` diagonal and gives each band 1.3 times the
  error above, then measures the correlation (Desroziers) before any
  non-diagonal `R`.
- **Artefacts:** the published 150-350 K `valid_range` is applied. A pixel
  outside it is dropped and counted, not clipped.

### 11.8 The forward operator (CRTM, opt-in)

The operator is JCSDA CRTM v2.4 or later, which is CC0 (section 1.11).
woof links it through a small Fortran shim with C bindings (11.10, R4).

- **Measured (2026-11-19, M3 Max):**
  - The licence was checked on the tag built, not assumed from v2.4.0:
    `v2.4.1-jedi.2` (`e477c08`) carries the CC0 dedication and no
    GPL-headed file under `src/`. woof's builder repeats that scan on
    every tree it builds and refuses a tree that fails it.
  - CRTM v2.4.1 (tag `v2.4.1-jedi.2`, 203 sources) compiles with gfortran
    16 and netCDF-Fortran in under a minute. No ecbuild is needed: woof's
    builder reads CRTM's own source list from `src/CMakeLists.txt` and
    orders it by module dependency.
  - The clear-sky forward and K-matrix calls run for `abi_g16`, `abi_g18`,
    `ahi_himawari9` and `seviri_m11`.
  - On a test profile, the ABI water-vapour Jacobian peaks are at 315 hPa
    (ch 8), 381 hPa (ch 9) and 563 hPa (ch 10).
  - Drying the column by 70 % warms ch 8 by 9.1 K.
- **Profiles from woof's state:**
  - Layer pressure, temperature from `theta` and `p`, water vapour as a
    mass mixing ratio in g/kg, and level pressures at the eta interfaces.
  - Ozone from the WRF ozone climatology woof's RRTMG uses
    (`wrf_ozone.py`) when the run has it; otherwise CRTM's US standard
    climatology.
  - Skin temperature from `TSK`. Land, water, snow and ice fractions from
    `LANDMASK`, `SNOWC` and `XICE`. Wind from `U10`/`V10`.
- **Coefficients:**
  - The user downloads them: CRTM's coefficient release, or the
    per-sensor files. woof never ships them.
  - woof records the SHA-256 of every file it loaded. The path is
    `WOOF_CRTM_COEFFS` or an explicit argument.
- **Calls:**
  - Forward returns the brightness temperature per `(profile, channel)`.
  - K-matrix returns `dBT/dT`, `dBT/dq` and `dBT/dO3` per layer, and
    `dBT/dT_skin`.
  - Path A uses K for screening, for the vertical placement of the
    observation (11.9) and for diagnostics.
  - Path B uses K as the operator's tangent-linear and adjoint: for a
    column operator, `H' dx = K dx` and `H'^T dy = K^T dy`.

### 11.9 Two paths into the analysis

**A. Forward-only, into the existing LETKF (this work).**

- The LETKF needs `H(x_k)` for each member and nothing else.
- The adapter (`woof/da/obs_radiance.py`, like `obs_goes.py`) first
  builds the observation column list (superobs, thinning, QC that needs
  only the observation), then runs the forward operator for every member
  on those columns only. A 30 km thinning keeps a few per cent of a
  regional grid's columns, so this is what keeps the CPU CRTM affordable.
- It places each superob as a column observation at one level, the level
  of the member-mean Jacobian peak, as `obs_goes.py` places CWP.
  Localisation does the rest. The vertical radius has to span the
  weighting function's width, and the adapter records the ratio. Honda et
  al. (2018) used the same convention.
- It returns `GriddedObs` batches, one per `(band, sector)`, each carrying
  its own localisation.
- The analysis goes through `woof.da.letkf` to member checkpoints and an
  analysis bundle, and reaches a forecast through the existing
  `woof run --analysis` door. No forecast code changes.

**B. Tangent-linear and adjoint, into 3D-Var at V4 (V4.3 and V4.4).**

- The 3D-Var worker owns `woof/da/var/`. Its `obsop/` wraps woof's
  radiance operator as an `ObsStencil`-like column operator:
  - `tl` is `K dx` over the column's layers;
  - `ad` is `K^T dy`, scattered in fixed order;
  - K is computed once at the innovation step, from the background.
- VarBC (V4.4) takes the static bias of 11.3 as its cold start.
- The 3D-Var worker's GV4 gate compares O-B per channel with WRFDA's CRTM
  path built on the same CRTM version. WRFDA 4.7.1 reads imagers directly:
  `da_read_obs_ncgoesabi` (GOES ABI netCDF), `da_read_obs_hdf5ahi` and
  the two `netcdf4ahi` readers (Himawari AHI), `da_read_obs_bufrseviri`
  (SEVIRI BUFR), each with its own QC (`da_qc_goesabi`, `da_qc_ahi`,
  `da_qc_seviri`). So woof's imager O-B is checked against the oracle on
  the same ABI or AHI granule, not only against path A's own numbers.

**Interfaces between the two owners** (recorded in the
WRFDA-3DVAR-HANDOFF.md working notes):

| module | owner | what the other side calls |
|---|---|---|
| `woof/obs/satweave.py`, `woof/obs/geo_icechunk.py`, the imager readers | this work | `read_window(...) -> ImagerScene` (11.10) |
| `woof/rtm/` | this work | `ClearSkyOperator(sensors, coeffs).simulate(profiles, geometry) -> bt` and `.jacobian(...) -> K` (layers top-down, numpy) |
| `woof/rtm/profiles.py` | this work | `profiles_from_wrf_state(state, setup, columns)` |
| `woof/da/obs_radiance.py` | this work | the LETKF adapter (path A) |
| `woof/da/var/` and `be.dat` | the 3D-Var worker | V4.3 wraps `woof.rtm` as TL/AD; V4.4 VarBC |

### 11.10 Readers, the scene schema, and the PR sequence

Every reader returns one in-memory **`ImagerScene`**:

- `band_um` and the source band name;
- `time` (the nominal time, plus the per-pixel scan time where known);
- `lat` and `lon`;
- `value` (K or %);
- `satellite` and `sensor` (CRTM sensor id), per pixel;
- `view_zenith`, `view_azimuth`, `scan_angle`, `solar_zenith`;
- `flags` (overlap, limb, NaN cause);
- `provenance` (source, licence, attribution text, byte digests).

SatWeave fills the geometry by reconstruction (11.2), and the native
readers fill it exactly.

**Native imagers, alongside SatWeave:**

| imager | source | licence and access |
|---|---|---|
| GOES ABI, AHI, AMI, SEVIRI, FCI as Icechunk stores | Source Cooperative `bkr/geo` (`us-west-2.opendata.source.coop`, anonymous); section 11.12 | the underlying producers' terms (rows below); the GOES virtual stores reference NOAA's own files |
| GOES ABI (16-19) L1b `RadF`/`RadC`, L2 `CMIPF` | NOAA NODD: `s3://noaa-goes{16,17,18,19}` (AWS), `gs://gcp-public-data-goes-{16,17,18,19}` (GCP); netCDF-4 | open, no account; NOAA requests attribution of unaltered data and no implied endorsement (AWS Open Data registry entry `noaa-goes`) |
| Himawari AHI (8/9) L1b full disc | NOAA NODD: `s3://noaa-himawari{8,9}`, `AHI-L1b-FLDK`; Himawari Standard Data segments, bzip2 | JMA data distributed by NOAA; NOAA and JMA request attribution of unaltered data and no implied endorsement (registry entry `noaa-himawari`) |
| MSG SEVIRI L1.5, MTG FCI L1C | EUMETSAT Data Store (a free EUMETSAT account and API key) | EUMETSAT Data Policy (amended 27 June 2024): Core data, including hourly Level 1 SEVIRI and FCI, free and unrestricted under CC BY 4.0, attribution "[Contains modified] EUMETSAT Meteosat data [year]"; higher-frequency Level 1 data (for example 15-minute SEVIRI) are "Recommended" data, free for non-commercial use but with conditions on commercial use and redistribution; woof caches them locally and never redistributes them |

- **Polar sounders** (ATMS, CrIS, IASI, AMSU-A/MHS) are a follow-on. They
  share the operator, but they need BUFR ingest, scan-dependent bias and
  channel selection. WRFDA's V4.3 starts with AMSU-A, which is that
  follow-on.
- **CRTM coefficients for FCI** are not in the v2.4.1 package (measured),
  so FCI waits for v2.4-readable coefficients.

**PR sequence:**

| PR | content |
|---|---|
| R1 | this section |
| R2 | `woof/obs/satweave.py`: time, bounding box and band subsetting over anonymous HTTPS; both store layouts; the NaN rules; the 11.2 reconstruction; a local cache under the obs cache with a provenance record (CC BY 4.0 attribution, chunk digests); `woof fetch satweave` |
| R3 | native readers from the `bkr/geo` Icechunk stores first (11.12): `woof/obs/geo_icechunk.py` with GOES ABI L1b from the virtual stores, then AHI and AMI from their virtual stores, and SEVIRI and FCI from the Satpy stores. The raw-file route is the fallback where the stores do not cover a case, reusing what woof already has: `rw_goes bt` (ABI L1b to brightness temperature, DQF-gated, ACM mask and CMIP cross-check) and `rw_goes geo-fetch`/`geo-bt` (Himawari AHI HSD and GK-2A AMI from NODD, per-row scan time) already write `gpuwm-obs.goes-bt.v1` packs. R3 is the Python reader of those packs into the same `ImagerScene`, with true geometry from the pack's navigation; no second decoder |
| R4 | `woof/rtm/`: the CRTM opt-in builder and shim, `ClearSkyOperator`, profiles from woof's state, coefficient digests; O-B per band per sector on a real case. `rw_goes forward` (ABI clear-sky, a fast model trained on CRTM optical depths, table `gpuwm-da.abi-fast-model.v1`) stays the fast path; CRTM is its reference, and GR4 reports both where a table exists |
| R5 | `woof/da/obs_radiance.py`: superobs, thinning, QC, cloud screening, bias; radiance batches through the LETKF into an analysis bundle; one regional case showing a smaller first-guess departure in the water-vapour bands and no worse conventional scorecard |
| R6 (plan only) | 3D-Var TL/AD with VarBC (V4.3, V4.4); all-sky; reflectances through a visible RT or a neural operator (section 6.2: Scheck 2021); the hex route through the same column operator on hex columns |

### 11.11 Gates

- **GR4 (operator).** Clear-sky O-B per `(band, sector)` on a real case,
  after the static bias. The bias and standard deviation are reported, and
  each standard deviation is checked against the expected range for a
  regional model: 1-2 K for the water-vapour bands and 1.5-3 K for the
  windows. A band outside its range blocks that band, not the PR.
- **GR5 (impact).** On one regional case, the analysis reduces the
  water-vapour bands' RMS first-guess departure at the next analysis time.
  The next SatWeave step is not independent: it is the same instrument,
  with the same bias correction and time-correlated errors. So the
  departure is also checked against an independent humidity source
  (radiosonde and GNSS precipitable water from the bkr/obs stores) and
  against a different imager where one overlaps (a native AHI or ABI scan
  not assimilated). The conventional-observation scorecard
  (P0's `J_o` against the bkr/obs stores, and the surface and upper-air
  referees) is no worse beyond its member-to-member noise.
- **Default identity.** Nothing on a forecast path imports `woof.rtm`,
  `woof.obs.satweave` or `woof.da.obs_radiance`, and an import-graph test
  checks it. A wheel-content test checks that no CRTM file or symbol is in
  the wheel (GV4's test, shared).

### 11.12 The `bkr/geo` Icechunk stores (surveyed 2026-11-19)

Source Cooperative's `bkr/geo` (bucket `us-west-2.opendata.source.coop`,
prefix `bkr/geo/`) holds geostationary imagery as Icechunk repositories,
readable anonymously. Every store was opened and its time coordinate read.
Coverage is uneven, so a reader checks the store's own times for the case
and never assumes coverage from a store's name. `*_test` stores are
ignored.

**GOES: `virtualized/` (the GOES source, per the user).**

- **Layout.**
  - `goes{16,17,18,19}_radf_C{01..16}` holds one ABI channel's full-disk
    L1b radiance (`ABI-L1b-RadF`).
  - Its chunks are *virtual references* into NOAA's own netCDF files. The
    repository config names `s3://noaa-goes{16..19}/` as virtual-chunk
    containers, anonymous, `us-east-1`. So a read returns NOAA's L1b bytes
    with their DQF, Planck constants (`planck_fk1/fk2/bc1/bc2`), `kappa0`,
    fixed-grid `x`/`y`, `time_bounds` and the actual sub-point.
  - woof authorises only `s3://noaa-*` containers.
- **Time split.**
  - A channel's archive is split into segments: `_<date>` stores hold
    scans up to that date, and the undated store holds the newest segment.
  - GOES-16: `_2018-01-03` from 2017-02-28 (15-min cadence), then
    `_2024-12-28`, then undated from 2024-12-29 to 2025-04-07 (10-min
    cadence).
  - GOES-17: `_2021-12-31`, then undated.
  - GOES-18: `_2023-03-26`, `_2026-07-08`, `_2026-11-18`, then undated.
  - GOES-19: `_2026-07-08` from 2024-10-10, `_2026-11-18`, then undated.
  - Spans from the 2026-11-19 survey, by reading each head's time
    coordinate: undated `goes17_radf_C08` holds 1245 scans,
    2023-01-02 to 2023-01-10, at 10-min cadence. The GOES-17/18/19 dated
    segments timed out from here, so their spans are not listed; woof reads
    the span from the store at run time.
  - woof lists the segments, opens the one whose end date first reaches
    the case, and searches its time coordinate. That coordinate is chunked
    27 scans per chunk, so woof bisects rather than reading thousands of
    chunks. If the time is not there, woof tries the other segments, then
    refuses with each segment's span.
- **Being written.**
  - Some repositories are mid-ingest. On 2026-11-19 the undated
    `goes19_radf_*` and `goes18_radf_*` heads name manifests that are not
    in the bucket, and the `_2026-11-18` stores had no repository yet.
  - So woof falls back up to 8 snapshots from `main`'s head, and records
    which snapshot it read.
  - The stores were renamed after ingest: `Rad` became `abi_l1b_radiances`,
    and `t` became `time`. Every array keeps its NOAA name in
    `source_name`, and older snapshots still carry the original names. woof
    addresses arrays by `source_name`.
- **Measured.**
  - One 4-channel, 849 x 1201-pixel box (105-80 W, 25-45 N) takes 39 s
    cold from the UK, with channels read in parallel, and 2.5 s from
    woof's cache.
  - Chunks are 226 x 226 pixels, so a regional box reads a few dozen
    chunks per channel, not the disc.
- **Also there.** `goes1x_l2/` and `goes1x_l2_meso/` hold L2 products as
  virtual stores: ACHA and ACHA2km, ACHP2km, ACHT, ACM, ACTP, per sector,
  split the same way. ACM, the clear-sky mask, is the observation-side
  cloud screen's natural input (11.5).
- **The top-level GOES stores** (`goes-east_*`, `goes-west_*`,
  `goes{16..19}_radf_C*`, `goes1x_l2_points`) are Satpy or partial copies.
  Per the user, GOES is read from `virtualized/` only.

**Other imagers (top level, and `virtualized/` for AHI and AMI).**

| store family | instrument, CRTM sensor | grid | scans held | first, last |
|---|---|---|---|---|
| `virtualized/himawari{8,9}_isatss_C{01..16}[_<date>]` | AHI, `ahi_himawari8`/`9` | native L1b (virtual) | H8: 2020-07 to 2022-12-12 by year, and late 2025; H9: 2022-12-13 to 2026-10-21; 10-min | H8 `_2019-12-31` and `_2020-07-02` have no root group |
| `virtualized/gk2a_ami_fd_{vi004..ir133,nr013,nr016,sw038}[_<date>]` | GK-2A AMI (CRTM `ami_gk2a`, presence in v2.4.1 to be checked) | native L1b (virtual) | about 12 days per year segment (early January 2024, 2025 and 2026; 16-27 February 2023) | sparse |
| `himawari_{500,1000,2000}m` plus monthly `_backfill_YYYYMM` | AHI | Satpy | 1000m/500m: 2015-07 to 2025-01, 2-4 % of 10-min scans; backfills 2020-02 to 2025-12 (about 240 k scans at 1 km) | |
| `msg_{1000,3000}m` | SEVIRI 0° (Meteosat-11 era), `seviri_m11` | 3 km (11 channels) or 1 km HRV | 2025-01-01 to 2026-10-24, 56 % (3 km) of 15-min scans | |
| `rss_{1000,3000}m` | SEVIRI rapid scan, 5-min | 3 km and 1 km | 2025-01-02 to 2026-10-24, 29-41 % | |
| `odegree_3000m` | SEVIRI 0°, Meteosat-10/11 | 3 km | 2018-01-01 to 2022-04-20, 27 % | |
| `odegree-12*`, `mtg_{500,1000,2000}m`, `mtg_highres_1000m` | MTG-I1 FCI (Meteosat-12) | 0.5-2 km | `mtg_2000m`: 2024-09-24 to 2026-11-19, 97 %; `odegree-12*` to 2026-01, 13-18 % | |
| `iodc_1000m`, `gk2a_*`, `himawari_2000m` (top level) | | | one day (2026-10-06) only | |
| `mtg_irs_l1` | MTG-S1 IRS, hyperspectral infrared sounder | L1 spectra | 2026-07-31 to 2026-11-19, 11 s step | `mtg_irs_l1_lac4_hourly` has no root group |
| `sentinel4_l2`, `tempo` | Sentinel-4 UVN, TEMPO (composition) | | | out of scope |

- **CRTM coefficients.**
  - ABI (`abi_g16/17/18`, with GOES-19 on `abi_g16`, see 11.3), AHI
    (`ahi_himawari8/9`) and SEVIRI (`seviri_m08`-`m11`) are in the v2.4.1
    package.
  - FCI is not: the package has no `fci_*` file. Whether it has an AMI
    file (`ami_gk2a`): checked 2026-11-20 in `fix_REL-2.4.1_20221109`'s TauCoeff
    listing, it has none either (CRTM 3 lists `ami_gk2`); the skylab inventory in 11.3 names none. FCI and
    AMI wait for v2.4-readable files, or a proxy-plus-bias like GOES-19's.
    woof refuses `--satellite gk2a` by name.
- **MTG-IRS** is the natural later hyperspectral-sounder target. It needs
  the same operator with an IRS coefficient file, channel selection, and
  the polar-sounder machinery of 11.10.
- **Sentinel-4 and TEMPO** are atmospheric composition, so they are out of
  scope here.
- **Preference order per case.**
  1. The `bkr/geo` virtual store, for native L1b and true geometry.
  2. The raw NOAA, JMA or EUMETSAT files through `rw_goes bt` and
     `geo-bt`, where the stores have no scan in tolerance.
  3. SatWeave, where neither covers the case. Its geometry is
     reconstructed.

References for this section:

- Schmit, T. J., et al., 2017: A closer look at the ABI on the GOES-R
  series. *Bull. Amer. Meteor. Soc.*, 98, 681-698,
  [doi:10.1175/BAMS-D-15-00230.1](https://doi.org/10.1175/BAMS-D-15-00230.1).
- Bessho, K., et al., 2016: An introduction to Himawari-8/9: Japan's
  new-generation geostationary meteorological satellites. *J. Meteor. Soc.
  Japan*, 94, 151-183,
  [doi:10.2151/jmsj.2016-009](https://doi.org/10.2151/jmsj.2016-009).
- Schmetz, J., et al., 2002: An introduction to Meteosat Second Generation
  (MSG). *Bull. Amer. Meteor. Soc.*, 83, 977-992,
  [doi:10.1175/1520-0477(2002)083<0977:AITMSG>2.3.CO;2](https://doi.org/10.1175/1520-0477(2002)083%3C0977:AITMSG%3E2.3.CO;2).
- Geer, A. J., et al., 2018: All-sky satellite data assimilation at
  operational weather forecasting centres. *Quart. J. Roy. Meteor. Soc.*,
  144, 1191-1217, [doi:10.1002/qj.3202](https://doi.org/10.1002/qj.3202).
- Okamoto, K., A. P. McNally and W. Bell, 2014: Progress towards the
  assimilation of all-sky infrared radiances: an evaluation of cloud
  effects. *Quart. J. Roy. Meteor. Soc.*, 140, 1603-1614,
  [doi:10.1002/qj.2242](https://doi.org/10.1002/qj.2242).
- Honda, T., et al., 2018: Assimilating all-sky Himawari-8 satellite
  infrared radiances: a case of Typhoon Soudelor (2015). *Mon. Wea. Rev.*,
  146, 213-229,
  [doi:10.1175/MWR-D-16-0357.1](https://doi.org/10.1175/MWR-D-16-0357.1).
- Minamide, M., and F. Zhang, 2017: Adaptive observation error inflation
  for assimilating all-sky satellite radiance. *Mon. Wea. Rev.*, 145,
  1063-1081,
  [doi:10.1175/MWR-D-16-0257.1](https://doi.org/10.1175/MWR-D-16-0257.1).
- Desroziers, G., L. Berre, B. Chapnik and P. Poli, 2005: Diagnosis of
  observation, background and analysis-error statistics in observation
  space. *Quart. J. Roy. Meteor. Soc.*, 131, 3385-3396,
  [doi:10.1256/qj.05.108](https://doi.org/10.1256/qj.05.108).
- Harris, B. A., and G. Kelly, 2001: A satellite radiance-bias correction
  scheme for data assimilation. *Quart. J. Roy. Meteor. Soc.*, 127,
  1453-1468,
  [doi:10.1002/qj.49712757418](https://doi.org/10.1002/qj.49712757418).
- Han, Y., P. van Delst, Q. Liu, F. Weng, B. Yan, R. Treadon and J. Derber,
  2006: JCSDA Community Radiative Transfer Model (CRTM), version 1. NOAA
  Technical Report NESDIS 122.

## 12. WIS2 observations from `bkr/obs/wis2.icechunk` (surveyed 2026-11-20)

The user asked (2026-11-20) for data assimilation from the WIS2 store,
<https://source.coop/bkr/obs/wis2.icechunk>. It holds many sensors,
including radiances. This section covers:

- what the store holds;
- how each family maps onto woof's observation types, with WRFDA's errors
  and QC;
- which families reuse an existing operator and which need a new one;
- the order of work.

The numbers were *measured* on 2026-11-20 by opening the repository
anonymously (bucket `us-west-2.opendata.source.coop`, prefix
`bkr/obs/wis2.icechunk`) and reading its attributes and arrays. The
observation data stays in the store. Nothing from it is committed, and the
tests use synthetic fixtures.

### 12.1 What `wis2.icechunk` is

- **A newer, combined store, not a different dataset.**
  - Its root attributes say it holds "Observations from the WMO WIS2 Global
    Cache", with "one group per kind of observation".
  - Its groups reuse the per-category stores' schemas unchanged:
    - `wis2_land` root becomes `land/reports`, and its `synop_tac`,
      `hydrometric` and `radar` groups become `land/*`;
    - `wis2_upper_air` root becomes `upper_air/soundings`, and its
      `wind_profiler` becomes `upper_air/wind_profiler`;
    - `wis2_aircraft` becomes `aircraft`;
    - `wis2_ocean/{surface,profiles,ship_tac}` becomes `ocean/*`;
    - `wis2_satellite/*` becomes `satellite/*`.
- **The per-category stores are frozen.**
  - Their newest snapshots were written on 2026-11-09, with the
    `wis2_20261109T1200Z` partition.
  - `wis2.icechunk`'s history starts on 2026-11-10. It back-filled the
    archive from `wis2_20261027T1200Z`, and on the survey day its head was
    `wis2_20261120T0000Z` (65,723 snapshots, newest 2026-11-20 03:41Z).
  - It is therefore a superset in time of the per-category stores, and the
    only one still being written. woof reads `wis2.icechunk`. The
    per-category stores stay readable for `woof grad obs`
    (`woof/grad/obs_stores.py`), which predates it.
- **Partitions.** Each group's `partitions_written` lists 12-hour archive
  windows (`wis2_YYYYMMDDTHHZ`), 48 per group from 2026-10-27T12Z to
  2026-11-20T00Z. `satellite/mws` starts at 2026-10-28T12Z, and
  `satellite/swim_nadir` at 2026-11-02T12Z.
- **Provenance per row.**
  - `source_key` (the object in `s3://wis2globalcache`), `source_centre`
    or `centre` (the WIS2 centre it came through), `originating_centre`.
  - `arrival_time` and `first_arrival_time`, as the Global Cache arrival
    times. woof maps `first_arrival_time` to the neutral table's
    `received_time`, so `obs_point`'s causal cut-off applies.
  - Correction bookkeeping: `update_sequence_number` on land reports, and
    `report_id` or `observation_identity` hashes elsewhere.

### 12.2 QC flags in the store

Every observation group carries the same flags, and the store never alters
or masks a value.

- **`quality_flag`** (uint16) has eleven bits:
  - 1 `outside_physical_range`;
  - 2 `outside_plausible_range`;
  - 4 `internally_inconsistent`;
  - 8 `vertically_inconsistent`;
  - 16 `position_invalid`;
  - 32 `position_suspect`;
  - 64 `time_invalid`;
  - 128 `time_in_future`;
  - 256 `time_too_old`;
  - 512 `duplicate`;
  - 1024 `provider_flagged`.

  Profiles carry the same flags per level as `level_quality_flag`.
- **`<variable>_quality`** (uint8) holds the first four bits for one
  variable.

woof's rule:

- A row is refused when any bit other than 256 is set, either in its own
  `quality_flag` (or `level_quality_flag`) or in the `<variable>_quality`
  of the variable used.
- Bit 256 is ignored, because woof selects its own time window.
- The counts are recorded per bit, so a receipt shows what the store's QC
  removed.

The store's flags are necessary, not sufficient (12.6). woof's own checks
follow:

- OBSPROC's gross checks;
- WRFDA's `check_max_iv` (3D-Var), or `obs_point`'s background check
  (LETKF);
- the Part D rule in 12.6.

### 12.3 Inventory

Rows per hour = rows / (12 h x partitions), an average over the archive.
Radiance rows are fields of view, each carrying every channel.

| group | rows | per hour | what one row is | key variables |
|---|---:|---:|---|---|
| `land/reports` | 8,319,511 | 14,400 | SYNOP/SYNOP-mobile BUFR report | `2m_temperature`, `2m_dewpoint_temperature`, `2m_relative_humidity`, `10m_wind_speed/direction`, `surface_pressure`, `mean_sea_level_pressure`, sensor heights, `station_elevation`, `update_sequence_number` |
| `land/synop_tac` | 377,144 | 655 | SYNOP received only as text | as above, plus `report` |
| `land/hydrometric` | 37,584,445 | 66,600 | Canadian gauge line | `discharge`, `water_level` (not for the atmosphere) |
| `land/radar/<node>_<qty>` | per site | - | ODIM PVOL (BE, CA sites) | out of scope here (radar has its own door) |
| `ocean/surface` | 2,651,960 | 4,600 | ship, buoy or drifter report | `air_temperature`, `dewpoint_temperature`, `surface_air_pressure`, `wind_speed/direction`, `sea_surface_temperature`, `platform_kind`, `data_buoy_type` |
| `ocean/ship_tac` | 69,663 | 121 | ship text report | as above |
| `ocean/profiles` | 220,874 profiles | 383 | Argo, XBT, CTD | sea-water T/S (ocean, not used) |
| `upper_air/soundings` | 96,106 soundings; 66.6 M levels | 167 | one TEMP part or one BUFR sounding | `pressure`, `temperature`, `dewpoint_temperature`, `geopotential_height`, `wind_speed/direction`, `time_since_launch`, drift, `vertical_sounding_significance`, `radiosonde_type` |
| `upper_air/wind_profiler` | 256,016 profiles; 6.0 M gates | 444 | profiler or radar wind profile | `height`, `eastward_wind`, `northward_wind`, `wind_quality_significance` |
| `aircraft` | 2,236,130 | 3,900 | AMDAR report | `flight_level`, `temperature`, `wind_speed/direction`, `humidity_mixing_ratio`, `phase_of_flight`, `report_id` |
| `satellite/amsua` | 27.5 M | 47,800 | AMSU-A FOV, 15 ch | `brightness_temperature`, angles, `satellite_identifier` 3/5 (Metop-B/C) |
| `satellite/mhs` | 178.9 M | 311,000 | MHS FOV, 5 ch | Metop-B/C |
| `satellite/atms` | 229.8 M | 399,000 | ATMS FOV, 22 ch | 225/226 (NOAA-20/21) |
| `satellite/mws` | 76.4 M | 138,000 | MWS FOV, 24 ch | 24 (Metop-SG-A1) |
| `satellite/mwhs2`, `mwts2` | 29.0 M, 6.9 M | 50,000, 12,000 | FY-3 MW FOV | 523/524 (FY-3D/E) |
| `satellite/iasi` | 89.7 M | 156,000 | IASI FOV, 8,700 scaled radiances | Metop-C |
| `satellite/cris` | 130.8 M | 227,000 | CrIS FOV, 2,211 radiances | NOAA-21 |
| `satellite/amv` | 622.8 M | 1,080,000 | AMV (EUMETSAT, NOAA, KMA AMI) | `pressure`, `wind_speed`, `wind_direction`, `percent_confidence` (QI with and without forecast), `amv_quality_flag` |
| `satellite/ahi_amv` | 32.7 M | 56,800 | Himawari AMV | as above, JMA layout |
| `satellite/scatterometer` | 25.3 M | 43,900 | wind-vector cell (ASCAT on Metop-B/C, HY-2) | `wind_speed_at10_m`/`wind_direction_at10_m` (4 ambiguities), `index_of_selected_wind_vector`, `wind_vector_cell_quality` |
| `satellite/gnss_ro` | 231,812 occultations | 402 | one occultation | `bending_angle` vs `impact_parameter`, `atmospheric_refractivity` vs `height`, `percent_confidence`, `radio_occultation_data_quality_flags` |
| `satellite/seviri_asr`, `fci_asr`, `ami_csr` | 43.5 M, 302.7 M, 89.2 M | 75,600 to 525,000 | geostationary clear-sky or all-sky radiance segments | `brightness_temperature` per channel, cloud fraction in segment |
| `satellite/altimetry`, `swim_nadir`, `sar_ocean`, `olci_wv` | | | altimeter, SWIM, SAR wave spectra, OLCI TCWV | waves and sea level (not atmospheric) or a retrieved product |

Notes:

- Sounding rows are *parts*, not launches.
  - A US station's 12Z sounding appears as up to five rows: TTAA, TTBB,
    TTCC, TTDD and a merged significant-level row.
  - A reader merges them per (station, launch).
  - `obs_stores.wis2_upper_air` keeps only the part with the most levels,
    which is how it met the Part D fault (12.6).
- The satellite rows are flattened BUFR, so several fields carry a
  replication axis.
  - `amv.pressure` is (n, 15) and `amv.percent_confidence` is (n, 4).
  - The scatterometer keeps four wind ambiguities and the index of the
    selected one.
  - The readers pick the documented replication explicitly, never by
    position alone.

### 12.4 Mapping to woof types, errors and QC

Errors are WRFDA's, from OBSPROC's `obserr.txt` (WRF 4.7.1,
`var/obsproc/obserr.txt`). They are interpolated linearly in pressure
between its levels (OBSPROC `intplin`, as `obs_err_afwa` does), and clamped
at the end levels. "LETKF" means a `gpuwm-obs.table.v2` row for
`woof.da.obs_point`. "3D-Var" means the interface agreed with the 3D-Var
work (12.7).

| WIS2 family | WRFDA type (FM) | woof variables (measurement) | WRFDA error (obserr.txt) | QC beyond 12.2 |
|---|---|---|---|---|
| `land/reports`, `synop_tac` | synop (12) | `temperature_k` (`screen_temperature_2m`), `wind_u/v_m_s` (`anemometer_wind_10m`), `dewpoint_k`, `surface_pressure_pa` (`station_pressure`) | T 2.0 K, wind 2.0-2.5 m/s (SURFACE LAND), RH 10 %, p 100 Pa | sensor heights 1-3 m (T) and 8-12 m (wind) as `obs_stores`; corrections by highest `update_sequence_number`; elevation mismatch over 200 m refused (`obs_point`) |
| `ocean/surface` ship | ships (13) | as synop, `platform_wind` | T 2.0 K, p 160 Pa (SURFACE SHIP) | position bits 16 and 32; a duplicate in `ship_tac` is dropped by `observation_identity` |
| `ocean/surface` buoy or drifter | buoy (18) | as ships, 5 m wind `anemometer_wind_5m_reduced_to_10m` where the height is stated | T 2.0 K, wind 2.5-2.7 m/s, p 10-20 Pa (BUOY) | as ships |
| `upper_air/soundings` | sound (35) | `temperature_k`, `dewpoint_k`, `wind_u/v_m_s` (`sonde_level`) at reported levels, each at its launch-plus-elapsed time and drifted position | T 1.0 K, wind 2.1-3.0 m/s, RH 10-15 % (RAOBS) | the Part D rule (12.6); monotone pressure per part; superadiabatic and lapse-rate checks as OBSPROC `check_obs`; duplicate parts merged |
| `upper_air/wind_profiler` | profiler (132) | `wind_u/v_m_s` (`profiler_level`) | 2.8 m/s (PROFL) | `wind_quality_significance`; height to pressure through the background column |
| `aircraft` | airep or AMDAR (42) | `temperature_k`, `wind_u/v_m_s` (`aircraft_level`), pressure from flight level (ICAO) | T 1.0 K, wind 3.6 m/s (AIREPS) | one row per `report_id`; WRFDA's aircraft thinning is a later option |
| `satellite/amv`, `ahi_amv` | geoamv or polaramv (88) | `wind_u/v_m_s` (`amv_assigned_pressure`) | WRFDA's satob error by pressure (confirmed against `da_read_obs_ascii` in PR C) | QI (without forecast) at least 80; pressure 100-1000 hPa; WRFDA's AMV level and land rules |
| `satellite/scatterometer` | qscat (281) | `wind_u/v_m_s` at 10 m over sea (`scatterometer_wind_10m`, new label) | WRFDA's qscat error floor (in `obs_wrfda.read_obs_ascii`) | selected ambiguity only; `wind_vector_cell_quality` clear; land and ice cells refused |
| `satellite/gnss_ro` | gpsref (116) | `refractivity_n` (`ro_refractivity_tangent_point`) | WRFDA's gpsref error profile (`da_read_obs_bufrgpsro` `err_opt = 1`, ported in PR E as `obserr_wrfda.gpsref_error`) | profile `percent_confidence` 100 and the RO quality flags; tangent height 0 to 30 km; `da_qc_gpsref`'s observed-gradient checks (super-refraction) |
| MW sounders (`amsua`, `mhs`, `atms`, `mwhs2`, `mwts2`, `mws`) | radiance | `brightness_temperature_k` with the stream's own operator | WRFDA `radiance_info` per channel | clear-sky over sea first: scattering and cloud-liquid indices, channel and surface screens; VarBC later |
| IR sounders (`iasi`, `cris`) | radiance | per selected channel | WRFDA `radiance_info` | clear-sky channel screen (McNally-Watts), channel selection (later) |
| geo radiances (`seviri_asr`, `fci_asr`, `ami_csr`) | radiance | as the imagers of section 11 | section 11.7 | CRTM has no FCI or AMI coefficients (section 11.12); SEVIRI only |

### 12.5 Operators: what exists and what is new

**Existing operators (reader only).** Every in-situ family uses
`obs_point`'s operators:

- log-p interpolation for `sonde_level`, `amv_assigned_pressure` and
  pressure-level rows;
- `t2`, `u10_earth` and `v10_earth` for surface rows.

`obs_point` lacks two things:

- the `aircraft_level` and `profiler_level` labels in its pressure-level
  branch. PR B writes `aircraft_level` rows; the label is enabled only by
  the opt-in `assimilate_aircraft` request field (`woof local-da
  --assimilate-aircraft`), because the ACARS and PrepBUFR doors already
  write it and enabling it by default would change existing runs;
- surface dewpoint and station-pressure operators. These rows are written,
  and refused by name until those operators exist.

3D-Var already has TL/AD for all 15 conventional WRFDA types (#659), so
synop, ships, buoy, sound, airep, profiler, geoamv, polaramv, qscat and
gpsref need only the hand-off.

**GNSS-RO refractivity.** `obs_point` already has a local refractivity
operator at the tangent height (`refractivity_n` via
`woof.obs.column_operators.refractivity_n`), as does 3D-Var's gpsref.

- The store carries the retrieved `atmospheric_refractivity` against
  `height`, so the local 1-D operator needs only a reader.
- Bending angle (a non-local ray-path operator) waits on evidence that
  refractivity is not enough.

**Microwave radiances.** These need CRTM's microwave clear-sky path in
`woof/rtm`:

- the MW sensor IDs;
- FASTEM-6 over water (its coefficient file is already staged);
- MW land emissivity, refused at first;
- a geometry adapter for scanning sounders (the FOV's own zenith and
  azimuth, from the store).

CRTM v2.4.1's fix set must hold SpcCoeff and TauCoeff for each
sensor/platform, and woof checks the file before use, as for GK-2A in
section 11.12. The candidates are:

- `amsua_metop-b`, `amsua_metop-c`;
- `mhs_metop-b`, `mhs_metop-c`;
- `atms_n20`, `atms_n21`;
- `mwhs2_fy3d`, `mwts2_fy3d`.

A sensor without a file (expected: MWS on Metop-SG-A1, the FY-3E
instruments) is refused by name, never mapped to a neighbour's
coefficients.

**AMVs and scatterometer winds.** These are point winds: the AMV at its
assigned pressure, and the scatterometer at 10 m over sea. They use the
existing operators.

**Hyperspectral IR (IASI, CrIS).** CRTM IR is already there (#651). The new
work is a channel selection, a cloud screen and data volume (8,700 IASI
channels per FOV), so they come after MW.

### 12.6 Known data fault: TTDD pressures ten times too large

A reported case: on 2026-10-28 12Z, stations 72261, 72265, 72327 and 72340
reported 200-214 K at 850-500 hPa, where the model gave 260-291 K. The
cause is in the data, and it is the same in `wis2.icechunk` and in
`wis2_upper_air`.

- **What is wrong.**
  - The bad levels are TEMP Part D objects:
    `data/us-noaa-nws/.../temp/TEMP_DD_<station>_<uuid>.bufr4`, BUFR
    template 309052, relayed by `us-noaa-nws`.
  - Part D holds the significant levels above 100 hPa, yet every one of
    these rows has pressures from about 98,400 Pa down to 6,700 Pa.
  - The pressures are ten times too large: 98.4 hPa is stored as 984 hPa
    (T 208 K), and 6.7 hPa as 67 hPa.
  - Its wind levels run the same way, 981 to 67 "hPa".
- **Extent.**
  - All 9,694 `us-noaa-nws` TEMP_DD rows in the store have a level above
    101 hPa.
  - 8,895 of them have T < 225 K at a "pressure" of at least 600 hPa.
  - The other parts (AA 10,733, BB 11,016 and CC 9,502 rows) and every
    other centre have no such rows.
- **Not flagged.**
  - These levels carry `level_quality_flag = 0` and
    `temperature_quality = 0`.
  - The first-level pressures of some merged parts are not monotone: for
    example, 400 Pa followed by 97,300 Pa.
- **How the readers met it.**
  - `obs_stores.wis2_upper_air` keeps, per station and launch, the part
    with the most levels. At these stations that was the TTDD part, and
    its "500 hPa" level is really 50 hPa.
- **woof's rule (PR B, `woof.obs.wis2.pressure_scale_fault`).**
  - The label rule: a Part C or Part D (TTCC or TTDD) holds levels above
    100 hPa by definition, so a part with a level at more than 101 hPa is
    refused whole and counted as `sonde_part_part_above_100hpa`. Any part
    with a level above 1100 hPa is refused as `pressure_out_of_range`,
    which catches a whole sounding at ten times its pressure.
  - The detection, for parts with no label (BUFR soundings) or a label
    that passes: each part is compared, in ln p, with the union of the
    same launch's other parts, only between reference levels at most 15%
    apart in pressure. If the median temperature difference exceeds
    10 K over at least 3 levels, but is within 3 K once the part's
    pressures are divided by ten, the part is refused as
    `sonde_part_pressure_scale_x10`. If rescaling does not help, the
    worst part is refused as `inconsistent_part` (both, when only two
    parts disagree).
  - woof does not rescale, because the store says values are as reported.
  - The background check (`check_max_iv` or `obs_point`'s) is the last
    net.
  - On 2026-10-28 11-13Z over (25-50N, 110-85W) the 19 TTDD parts are
    refused, and 72261, 72265, 72327 and 72340 have 260-291 K at
    850-500 hPa.
- **For the source.**
  - Either decode NWS Part D pressures with the right scale (tenths of hPa
    in the TAC; the BUFR reports them at ten times the value), or set
    `vertically_inconsistent` (bit 8) on those levels.

### 12.7 Interfaces

- **LETKF.**
  - `woof obs wis2 fetch --start --end --bbox S N W E --families ...
    --table table.csv` (opt-in) writes `gpuwm-obs.table.v2`;
    `--obs-gts ob.ascii` writes the 3D-Var file and `--receipt` a JSON
    receipt (snapshot id, counts per QC rule).
  - `obs_point.read_tables` already reads that format, and `point_batches`
    applies the receipt cut-off from `received_time`.
  - Nothing changes when the command is not run.
- **3D-Var (agreed with the da.var owner on 2026-11-20).**
  - The reader also writes OBSPROC's `obs_gts` ASCII. That file is read by
    `woof.da.var.obs_wrfda.read_obs_ascii`, by `woof var --obs` and by
    WRFDA's oracle lane. No Python objects cross the interface, and
    `woof/da/var` gets no WIS2 code.
  - Header. OBSPROC's header block is copied verbatim: the INFO, SRFC and
    EACH format lines and the MISS and TOTAL lines. Every F field is
    written with its decimal point, because a blank reads as 0 and a
    value without a point gets implied decimals.
  - INFO line. The platform is written as `FM-nn NAME`, with FM codes 12
    synop, 13 ships, 18 buoy, 35 TEMP, 42 AMDAR and 132 profiler (not 32,
    which is PILOT). The date is `YYYY-MM-DD_HH:MM:SS` inside the window
    [t0, t1). Longitude is in [-180, 180). The level count equals the
    number of EACH lines.
  - Winds are written as speed and direction, never as u and v.
  - Moisture is written as rh, because WRFDA derives q from rh, t and p;
    a dewpoint alone gives no q.
  - Missing is -888888.000 with qc -88.
  - The errors are OBSPROC's `obserr.txt` interpolated with `intplin`,
    and are used exactly as written (apart from WRFDA's qscat floor), so
    their source is recorded in the file's provenance.
  - Sondes include the surface level, with height exactly equal to elv.
    Levels are written in decreasing pressure.
  - GNSS-RO is FM-116, in a separate optional file. The refractivity goes
    in the td slot and the impact height in the height slot.
- **Radiances.**
  - Each MW FOV becomes the same `ImagerScene`-like record that
    `woof.da.obs_radiance` consumes: brightness temperatures per channel,
    the FOV's own geometry and `crtm_sensor()`.
  - For 3D-Var, the radiance operator plugs in at the ObsTerm seam
    (`woof/da/var/cost.py`), as section 11.9 describes.

### 12.8 Order of work

Each step is one small PR, opt-in, and leaves default runs unchanged.

1. **PR A: this section.**
2. **PR B: in-situ reader.** `woof obs wis2 fetch` covers land/reports,
   synop_tac, ocean/surface, ship_tac, upper_air/soundings,
   wind_profiler and aircraft. It applies 12.2's flags, OBSPROC's errors
   and gross checks, and the Part D rule. It writes neutral table v2 and
   `obs_gts`. Aircraft rows carry `aircraft_level`, which `obs_point`
   uses only when `assimilate_aircraft` is set (12.5); profilers
   (heights, no pressure) go to `obs_gts` only, and ocean platform rows
   carry labels `obs_point` has no operator for yet. Tests use synthetic
   fixtures.
3. **PR C: point winds.** AMVs (`amv`, `ahi_amv`) and scatterometer winds
   enter as point winds with WRFDA's AMV and qscat errors. As built
   (opt-in families `amv` and `scat`):
   - The satellite groups are found by bisection on their nearly sorted
     `time` (milliseconds), with a 6 h margin for late arrivals.
   - AMVs: template 310077 only. The QI is the `percent_confidence` whose
     `standard_generating_application` (0-01-044) in the same column is 5
     (QI without forecast), at least 80. Pressure 100-1000 hPa. Geostationary
     by satellite id; polar by imager (AVHRR, MODIS, VIIRS). Copies relayed
     by several centres are one AMV (first arrival). VIIRS AMVs stay out of
     `obs_gts`, because WRFDA keys polaramv on AVHRR or MODIS.
   - `ahi_amv` (JMA's template) decodes no quality index here, so it is
     skipped and counted.
   - Scatterometer: ASCAT (template 312061), the selected ambiguity.
     `wind_vector_cell_quality` (0-21-155) bits 1-11 and 16-18 refuse a
     cell; an empirical check on 2026-10-28 matched bit 9 to land, bits
     10, 11 and 18 to ice and bit 13 to speeds of 3 m/s or less. Land and ice
     fractions are checked directly as well.
   - `--thin-km` (default 0) keeps one AMV per box and 50 hPa layer (the
     highest QI) and one scatterometer wind per box.
   - On 2026-10-28 11-13Z globally: 1,276,760 AMVs and 13,733 ASCAT winds
     pass. WIS2 has no GOES AMVs, so a CONUS box gets none.
4. **PR D: MW radiances.** CRTM v2.4 MW clear-sky for AMSU-A, MHS and
   ATMS (FY-3 where coefficients exist), over sea first, through
   `obs_radiance`. IASI and CrIS come after, with a channel selection.
5. **PR E: GNSS-RO refractivity.** The local 1-D operator is the existing
   one; the reader and the WRFDA gpsref error are new. Bending angle comes
   later, if warranted. As built (opt-in family `ro`):
   - Template 310026 lists the bending-angle levels (each with its tangent
     point), then the refractivity levels; the counts are the row's
     `extended_delayed_descriptor_replication_factor`, and
     `percent_confidence` is the profile's, then one per bending-angle
     level, then one per refractivity level. As `da_read_obs_bufrgpsro`
     does, refractivity level k takes bending-angle level k's location when
     the counts agree. WRFDA skips a profile whose counts differ; here it
     is kept at the occultation point, and counted.
   - A profile is refused under 100 % confidence (WRFDA's rule) or with
     0-33-039 bits 1, 4, 5, 6 or 15 set (non-nominal, excess phase, bending
     angle or refractivity processing non-nominal, background profile).
     A level is dropped at 0 % confidence, outside 0-30 km, or where the
     observed dN/dz is below -50 N/km or |d2N/dz2| above 100 N/km^2 (the
     observed half of `da_qc_gpsref`'s gradient checks).
   - Copies (satellite, transmitter, time) are one profile, first arrival.
   - Errors: `gpsref_error`, 1 % of N scaled by height and latitude
     (`err_opt = 1`).
   - The table rows carry the height in `elevation_m` and its
     standard-atmosphere pressure in `level_pa`, which only keeps the
     levels' identities apart. `obs_gts` has one FM-116 report per profile
     at the occultation point (the ascii reader has one location per
     report), refractivity in the dew point slot, its error not below
     0.01 N (the F7.2 field; near 30 km WRFDA's error is smaller).
   - On 2026-10-28 09-15Z globally: 2,568 profiles in the window, 1,502
     used (207,599 levels); 1,001 are EUMETSAT bending-angle-only
     products, which need a bending-angle operator.
6. **PR F: impact.** One real case, as in #657: O-B and O-A per family,
   and a radiosonde and surface scorecard showing no degradation.
