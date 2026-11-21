# MPAS-Seaice v8.4.2 column-package oracle, patched: the solved vertical salinity (`zsal_solve`) through the ITD, the ridging and the steps

Written by `WOOF_ORACLE_PATCH=zsal_solve bash tools/mpas_seaice_oracle/build.sh`
and `make_column_fixture.py --cases zsal_solve_carry` on pb (Linux 6.12
x86_64, glibc 2.44), 2026-11-21.

This is not v8.4.2's behaviour. It is the reference for the opt-in
correction `zsal_solve` (`woof.hex.mpas_seaice.column_v842.CORRECTIONS`).
The default words (v8.4.2's inert vertical salinity) are those of
`../mpas_seaice_bgc_zsal_carry/PROVENANCE.md`.

- Source and patch: the same as `../mpas_seaice_bgc_zsal_solve/PROVENANCE.md`
  (MPAS-Model v8.4.2 `f34984b2`, `tools/mpas_seaice_oracle/zsal_solve.patch`,
  gfortran 16.2.0, `-O0 -ffp-contract=off`, RKIND 8).
- Driver: `run_column.F90`, stages `itd_thermodynamics`, `ridging`,
  `column_step` and `seaice_step`.
  - With the vertical salinity on, the `column_step` and `seaice_step`
    stages call `colpkg_init_zsalinity`, as
    `seaice_init_column_physics_package_variables` does before any step.
    This sets `zsalinity`'s sub-step `dts_b`. Before this change those
    stages left `dts_b` unset; that is a driver omission, not v8.4.2
    behaviour.
  - `dts_b` is read only when `solve_zsal` is set, so the unpatched
    fixtures are unchanged. `../mpas_seaice_bgc_zsal_carry` regenerates
    byte for byte with this driver.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`
  and `config_use_vertical_zsalinity` with BL99, `nBioLayers` 7,
  `woof_v842_corrections = zsal_solve`. The steps also declare
  `bgc_inactive_diagnostics` (see `../mpas_seaice_bgc_brine_carry/PROVENANCE.md`).
  The stream carries the two names as one string, which the oracle does not
  read.
- Cases: `BGC_ZSAL_SOLVE_CARRY_CASES` and `BGC_ZSAL_SOLVE_CARRY_COUPLED_CASES`.
  `_zsal_solve_carry_state` gives a positive vertical salinity, a salt flux
  to add to and, for the ITD and the ridging, `colpkg_init_hbrine`'s grids.
  - ITD:
    - with the ocean fluxes updated;
    - the delta function with four layers and four categories;
    - one category;
    - `itd_zsal_solve_abort_regrid`: cell 2's first category has a slightly
      negative brine fraction under the spread frazil. `regrid_stationary`'s
      remap stops on it, and `add_new_ice_bgc` returns the ITD abort
      `ice: remap_layers_bgc error` at `indexToCellID` 3.
  - Ridging: exponential; Thorndike/Hibler with two subcycles.
  - Two column steps: dEdd; three categories on the Gregorian calendar.
  - One coupled step (Rothrock).
- What `solve_zsal` adds there:
  - `add_new_ice`'s salt loss with the frazil into `zSalinityFlux`, and no
    mixing of the new ice's `sice`;
  - `add_new_ice_bgc`'s `adjust_tracer_profile`, either `regrid_stationary`
    or the frazil mixed throughout, then `remap_zbgc` back into `sice`;
  - the first category's surface temperature from that `sice`;
  - `lateral_melt_bgc`'s and `zap_small_areas`' salt into `zSalinityFlux`;
  - `zsalinity` inside the steps' `column_biogeochemistry`.
- Not covered: the time-step cases run without advection
  (`config_use_advection = .false.`), so the incremental remap of
  `verticalSalinity` is not exercised.
- Files: `oracle-r8.npz` holds, per case, every output field whose words
  differ from the input's. `column-logs.json` holds the tracer object and
  the log lines. `inputs.sha256` holds the input-stream digests.
- libm: as in `../mpas_seaice_bgc_zsal_carry/PROVENANCE.md`. The steps need
  the LGPL opt-in `WOOF_GLIBC239_LIBM64=1`.
