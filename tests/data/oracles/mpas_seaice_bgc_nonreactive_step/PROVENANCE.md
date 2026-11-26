# MPAS-Seaice v8.4.2 column-package oracle: the nonreactive tracer through the column step

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases nonreactive_step` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_nonreactive`
  and `../mpas_seaice_bgc_nonreactive_carry`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_nonreactive`, every other
  biogeochemistry flag off. `nBioLayers` is 7. The oracle gives
  `totalChlorophyll` an array that is not compared (the step runs with the
  opt-in correction `bgc_inactive_diagnostics`, as in
  `../mpas_seaice_bgc_brine_carry`).
- Driver: `run_column.F90`, stages `column_step` (the column package's part
  of `seaice_timestep`, several steps) and `seaice_step` (`seaice_timestep`
  with the velocity solver on a culled hexagonal torus): the pre-dynamics end
  with `column_separate_snow_ice_tracers`, and each dynamics subcycle's
  `seaice_column_dynamics_time_integration` starts with
  `column_combine_snow_ice_tracers` before the ridging. No advection (the
  column oracle's coupled cases run without it).
- Cases: `BGC_NONREACTIVE_STEP_CASES`: three column steps and
  `seaice_nonreactive_bl99_dedd_rothrock`, the coupled step (12x6 torus,
  3 owned columns). `_nonreactive_carry_state` adds the
  tracer profiles, mobile fractions, ocean concentrations,
  `oceanBioFluxes` and the brine grids to the step state.
  - `step_nonreactive_bl99_dedd_subcycles_gregorian`: two dynamics
    subcycles, so the second combine overwrites the first subcycle's ridging
    of the tracer with the pre-dynamics `*Snow`/`*Ice` (v8.4.2's order; the
    opt-in correction `subcycle_snow_ice_tracers` is not held to the oracle).
  - `step_nonreactive_mushy_level_ponds_no_vertical_thermodynamics`: without
    the vertical thermodynamics `iceAreaCategoryInitial` is never refreshed,
    so in the second step `update_snow_bgc` divides 0/0 for the category the
    first step's ITD opened in cell 1, and `oceanBioFluxes(1, 2)` is NaN
    (kept word for word; the opt-in correction `bgc_snow_new_category` is not
    held to the oracle).
  Every case changes `verticalNonreactiveConc`, `verticalNonreactiveSnow`,
  `verticalNonreactiveIce`, their cell aggregates and `oceanBioFluxes`;
  `coupling_prep` zeroes the per-species ocean fluxes (`oceanNitrateFlux`
  and the rest), since the nonreactive tracer has no ocean species. None
  aborts.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: the steps' delta-Eddington shortwave and BL99 call glibc's `exp`,
  `atan` and the rest, compared under the opt-in LGPL transcription
  (`WOOF_GLIBC239_LIBM64=1`).
