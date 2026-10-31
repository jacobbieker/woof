# Spectral radiation, WRF-Solar and height-dependent resolution, on CUDA and MLX

## Context

The MLX (Apple GPU) backend is on branch `mlx-backend` (PR jacobbieker/woof#139).

- **WRF parity on Metal:** 12 failures, and the RTX A6000 fails the same 12.
- **Goddard:** now translates.

The user has asked for four more features. Each must run on both CUDA and MLX from the same `.cu` source, and be checked against oracles:

1. **RRTMG band output.** RRTMG's per-band and per-g-point fluxes, as an output option.
2. **WRF-Solar, complete, in both WRF and MPAS.**
3. **A libRadtran-class spectral solver on the GPU.** Irradiance and radiance at chosen wavelengths, checked against libRadtran's uvspec.
4. **Height-dependent horizontal resolution.** For example 500 m below 5 km and 5 km above 10 km, where each finer region is no larger horizontally than the coarser one above it.

**Decisions (user, 2026-10-29):**
- Order: bands, then WRF-Solar, then the spectral solver, then height-dependent resolution.
- Spectral solver scope: DISORT (pseudo-spherical) and two-stream, REPTRAN, Rayleigh, water and ice cloud optics, and OPAC aerosols.
- Licence: woof is Apache-2.0 and libRadtran is GPL-2. The solver is therefore clean-room, written from the published papers and DISORT's distribution terms. uvspec is used only as a black-box oracle.
- Height-dependent resolution: a partial-depth two-way nest. Each interface is a fixed-pressure level in WRF (the parent's `etac` must reach it) and a flat height level in MPAS.
- Meaning of "matches the oracles" for height-dependent resolution:
  - bit-identical to WRF/MPAS with a single band;
  - every coupling operator held to compiled WRF;
  - MPAS feedback held to an exact float64 reference;
  - the whole system scored against all-fine WRF/MPAS runs, within a recorded envelope.

**Conventions that apply throughout:**
- New behaviour goes in new kernel entry points, so existing kernels and their launch receipts stay unchanged.
- Oracles are gfortran/gcc builds at `-O0 -ffp-contract=off`, with receipts of source sha256s and flags.
- CUDA and MLX parity is checked with `tools/launch_trace.py` and `tools/backend_receipts.py`.
- CUDA runs on `ssh pb` (RTX A6000), in the `~/Development/woof-mlx` worktree.
- Commit and push to PR #139 after each verified step.

## Phase A: RRTMG per-band and per-g-point fluxes (WRF and MPAS)

**Config.** `rrtmg_spectral_output = none|bands|gpoints`, `rrtmg_spectral_levels = sfc_toa|all`, and `rrtmg_spectral_quantities`.
- They live in `woof/config.py`, in the WRF namelist import, and in MPAS `&radiation`.
- `gpoints` at all levels is refused above a memory budget.

**SW.** A new kernel `rsw_spc_band_accum` in `woof/core/kernels/rrtmg_sw.cu`.
- It reads the transient `zfd/zfu/zcd/zcu/ztdbt` buffers that `rsw_spc_accum_body` (:1964) already uses.
- Band sums use ascending g-point order, as the broadband loop does. g-point outputs are the weighted products themselves.
- It is launched from `CudaSW` (`woof/core/rrtmg_sw.py:2971`), and the MPAS `rrtmg_sw_v842` subclass inherits it.

**LW.** A new kernel `rlw_rtrn_accum_bands`, plus a rows variant. It keeps the per-band `urad/drad` that `rlw_rtrn_accum` (`rrtmg_lw_chain.cu:503`) already computes, scaled in Fortran `rtrnmc`'s factor order.

**RRTMGP.** A band-reduce kernel beside `rrtmgp_*_flux_reduce` (`rrtmgp_rte.cu:293/502`), in RTE's `sum_byband` order.

**Outputs.**
- WRF: `SW*/LW*_BAND` and `*_GPT` in `woof/io/wrf_output_schema.py` (beside :661), with `history_layout.py` entries and a `spectral_radiation` preset. Band-edge wavenumbers and the g-point-to-band map go in as attributes.
- MPAS: `streams_output_v842.py`, the `resident_output_diagnostics_v842.py` PRODUCED set, and a `radiation_spectral` stream.

**Oracle and gates.**
- `tools/rrtmg_wrf461_oracle/spectral_dump.F90` patches dump hooks into `spcvmc_sw` and `rtrnmc`. Fixtures go to `tests/data/oracles/rrtmg_spectral/`; the MPAS fixtures are added to `components/hex/tests/mpas_rrtmg_sw_oracle`.
- Gates:
  - broadband words byte-identical with the option on and off;
  - band and g-point words equal to the oracle's;
  - CUDA and MLX receipts equal.

## Phase B: WRF-Solar in WRF and MPAS (WRF v4.7.1 sources)

The first task is to confirm each source file below in the WRF 4.7.1 tree. MAD-WRF's location in particular is unverified.

| Order | Component | WRF source | WRF port (lift the refusal) | MPAS |
|---|---|---|---|---|
| B1 | `icloud=3` | `cal_cldfra3`, `module_radiation_driver.F` | `kernels/cldfra3.cu` (`namelist_import.py:5258`) | `cld_fraction_cldfra3`, after diffing against `cloud_thompson` in `radshared_v842.py` |
| B2 | `solar_diagnostics=1` | `module_diag_solar.F` | `kernels/diag_solar.cu`, plus schema accumulators (:4957) | new `solar_diagnostics` stream |
| B3 | FARMS (`swint_opt=2`, farms SW) | `module_ra_farms.F` | `woof/core/farms.py` + `kernels/farms.cu` (`config.py:3791`, `_RA_SW_MAP` :845) | per-step SW post-processing slot after `radiation_sw` |
| B4 | Deng shallow Cu (`shcu_physics=5`) | `module_shcu_deng.F`, `module_shallowcu_driver.F` | `woof/core/shcu_deng.py` + kernel (:1368) | new `shallow_convection` slot (`config_shcu_scheme`) |
| B5 | `cu_rad_feedback` (incl. Deng) | radiation and cumulus drivers | radiation pre-pass (:4820) | radiation reads GF `qc_cu/qi_cu` and Deng cloud |
| B6 | EPS `pert_farms/deng/thom/cld3` | per-scheme perturbation arguments, `module_stoch.F` | `woof/namelist_stochastic.py` + per-scheme multipliers | same multipliers through the suite table |
| B7 | MAD-WRF (`madwrf_opt`, mp 96) | `module_madwrf*` and dyn_em hooks | `woof/core/madwrf.py` (`config.py:3496`) | cloud init/nudging on the hex state, passive-hydrometeor microphysics slot |

**MPAS wiring.**
- Slots and schemes go in `woof/hex/data/profiles/mpas-physics-suites.v1.json`.
- Each scheme module follows the `bind_native_pools` / `*_slot` / `register_hex_cuda` pattern (e.g. `rrtmg_sw_v842.py`).

**Oracle.**
- `tools/wrfsolar_wrf471_oracle/` compiles the modules above against stubs, following `tools/rrtmg_wrf461_oracle/build.sh`.
- It is driven by WRF column fixtures and, through `driver_extract.py`, by MPAS columns. Receipts go to `tests/data/oracles/wrfsolar/` and `components/hex/tests/wrfsolar_v842/`.
- Perturbation fields are oracle inputs. Pattern generation is checked separately against `module_stoch`.

## Phase C: clean-room spectral solver (`woof/radtran/`)

**Host side.**
- `options.py`: a uvspec-style option subset.
- `reptran.py`: reads the netCDF into flat device tables.
- `cloud_tables.py` (Hu & Stamnes, Fu, Baum) and `opac.py`.

**Kernels** (`woof/core/kernels/radtran_*.cu`, translated to Metal automatically):
- `optprop`: Rayleigh after Bodhaine 1999, REPTRAN gases, cloud and aerosol mixing, δ-M scaling.
- `disort`: ASYMTX eigensolver, banded LU, pseudo-spherical Chapman path, all in binary64.
- `twostr`.
- `reduce`: REPTRAN-weighted sums in a fixed order.
- libm calls go through `glibc_flt64.cuh` / `f64cr.cuh`, so CUDA and Metal give the same words.

**Model columns.**
- From the model: z, p, T and air density; H₂O from qv; O₃ and CO₂.
- Above the model top, an AFGL standard atmosphere (choice recorded).
- Clouds: LWC/IWC with effective radii.
- Aerosol: from `aer_opt` or Thompson `nwfa/nifa`, mapped to OPAC.
- Albedo and the model's solar geometry.

**Outputs.** `RT_EDIR/EDN/EUP/RAD` at the surface and TOA, by wavelength.
- Controlled by `radtran_wavelengths`, `radtran_umu/phi` and `radtran_diag_interval`.
- WRF: schema entries. MPAS: a `radtran` stream.

**Oracle.**
- A `libradtran` pixi feature and environment (Python 3.11 plus conda-forge `rubin-libradtran` 2.0.6, linux-64 and osx-arm64), and a `tools/uvspec_oracle/` driver.
- Cases: Rayleigh, REPTRAN gases, water cloud, ice cloud (Fu and Baum), OPAC aerosol, SZA 85° and 89° pseudo-spherical, two-stream vs DISORT, 16-stream radiances.
- Receipts record the conda build.
- Tolerance (uvspec is built at -O2): 1e-6 relative for irradiance and 1e-5 for radiance, with an absolute floor. Deliberate input differences are listed in the receipt.
- Word-exact against an -O0 build of the original DISORT Fortran, if its distribution terms allow building it as an oracle.

## Phase D: height-dependent horizontal resolution (partial-depth two-way nest)

**D0. Config and validation.**
- Bands are listed coarse to fine.
- Each band's interface must be:
  - a parent level at or above `etac` (fixed pressure p_i) in WRF, or a flat `zgrid` level in MPAS;
  - nested inside the parent's footprint, with dx no larger than the parent's and an interface no higher than the parent band's.
- Deep cumulus is refused while an interface is below ~12 km.
- A single band falls back to the existing code path, which makes it bit-identical. Kernel-hash receipts prove the existing kernels are unchanged.

**D1. Child ladder** (`woof/core/grid.py`).
- The child takes the parent's levels up to k_i: c3' = c3, c4' = c4 + c3(p_i − p_top) + p_top − p_i, and p_top' = p_i.
- So mu_c = p_s − p_i, a constant offset from the parent's mu. Lateral boundaries use that offset (`lateral_bc.py` `couple_nest_field`).

**D2. Mass flux through the top.**
- `ww_top(y,x)` is the parent's `ww` at k_i, interpolated with `sint` and mass-scaled, then interpolated in time on each small step.
- New entry points:
  - mu tendency and `calc_ww` with a nonzero top `ww`;
  - `advance_mu_t`;
  - an acoustic w/phi solve with w and ph specified at the top (`acoustic.cu`);
  - upwinded top-face scalar and momentum fluxes (`advection.cu`).
- New file: `kernels/nest_top.cu`.

**D3. Top zone and feedback.**
- Top zone: one specified level and about five relaxation levels, reusing `spec_bdy.cu`'s `relax_bdytend` arithmetic on a top strip. mu is not relaxed.
- Feedback below k_i uses `copy_fcn/m/i` and the sm121/smdsm smoother as they are (`nest_interp.py`). Feedback of mu adds back the constant offset. Interface w and ph are not fed back.

**D4. Physics, radiation and multiple bands.**
- Radiation uses an extended column: the parent's levels above k_i, interpolated with `sint`, refreshed each radiation call. Heating is applied below k_i only.
- Multiple bands schedule through `clock.execute_schedule`: coarsest first, feedback last, `parent_time_step_ratio`.
- STATUS (branch partial-nest-d4): done.
  - The child steps through `dycore.step(partial_top=...)`. Its hooks are the
    stage forcing, the interface Omega, the face fluxes and top zone, the
    specified-top acoustic substep, and the scalars' interface flux. So
    moist scalars, physics and microphysics run on the child as on any
    domain.
  - The interface face takes the parent's second-order value, not D3's
    upwind value; that fixed the gravity-wave ratchet. Feedback defaults to
    theta plus the moist scalars. mu, u, v and w are opt-in. phi is refused,
    because its feedback is a measured instability.
  - Chains: band n's parent is band n-1, and each band's snapshot uses its
    partial parent's own interface-closed Omega.
  - Radiation runs on the extended column (`partial_nest_physics.py`).
  - The model wiring is `DomainConfig.partial_nest` (the loader's
    `partial_nest` domain key) plus `model.realize_partial_nests` /
    `partial_nest_steppers`, on the idealized tree. The real-data route
    refuses partial nests by name.
  - Receipts are in `tests/data/receipts/partial-nest-d4`.

**D5. MPAS.**
- The fine band is a regional variable-resolution mesh truncated at z_i (`density.py`, `rw_mpas_mesh`).
- Top face: a new specified-w kernel entry point, upwinded top fluxes, and a top zone using `regional_v841.py`'s relaxation and divergence damping with a per-level mask.
- Lateral zones are filled online through the `lbc.py` machinery.
- Coarse to fine is barycentric, held to MPAS init_atmosphere's own interpolation where it applies.
- Fine to coarse is a conservative overlap remap (`moving_remap.py` tables), held to an exact float64 reference that conserves to round-off.

**Oracles.**
- (a) The existing WRF and MPAS suites with a single band, plus kernel hashes.
- (b) `tools/partial_nest_wrf471_oracle/`, modelled on `tools/vert_nest_wrf471_oracle/`. It compiles `interp_fcn.F` (`sint`, `bdy_interp1`, `copy_fcn*`, `smoother`, `blend_terrain`) and `relax_bdytend_core`/`spec_bdytend`, the latter fed a transposed top strip.
- `ww_top` and `calc_ww` changes: a full-column child driven by its own level-k_i `ww` must reproduce itself bit-for-bit below k_i.
- (c) Envelope runs. These are recorded as JSON receipts and enforced as regression tests.
  - References: stock WRF 4.7.1 full-column fine nests and all-fine runs (same `etac`), and all-fine MPAS.
  - Cases: a resting atmosphere, a Skamarock–Klemp gravity wave, a warm bubble crossing the interface, a 2-D hill wave, and three real 24 h cases.
  - Metrics: near-surface RMSE and bias, precipitation FSS, kinetic-energy spectra, top-zone w variance and reflection, dry-mass and energy drift, and speedup.

**Optional D6.** A time-varying p_i(x,y,t), only if the `etac` constraint proves unacceptable.

## Remaining MLX-backend items (carried over)

- hex `test_phase1_leaves_every_input_byte_identical` crashed a worker under load. It passes alone; recheck it in the next full run.
- satmedmf driver: "humidity reached unit specific humidity" on Metal. This predates the current changes.
- TEMPO: `tempo_column.cuh` is not on the translator's include path.
- Vendor-libm unification, df64 fast mode, and the CPU backend.

## Verification (every phase)

1. Each oracle in its build environment, `-O0 -ffp-contract=off`, with receipts.
2. `pixi run -e mlx pytest <phase tests>` on the Mac, and `pixi run -e default python -m pytest <phase tests>` on `pb`. Then `tools/backend_receipts.py` over traces from both.
3. Regression:
   - the WRF parity set, which must show no new failures against the A6000;
   - the hex suite;
   - the WRF gates in both `GPUWM_WRF_EXACT` modes;
   - `tools/f64_soft_grade.py`.
   - For phase D, kernel-hash receipts proving the existing kernels are unchanged.
4. Update `docs/dev/mlx-backend.md` and the user docs for each new option. Commit and push to PR #139.
