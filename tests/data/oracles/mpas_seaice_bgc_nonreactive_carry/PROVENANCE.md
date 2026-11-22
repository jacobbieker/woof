# MPAS-Seaice v8.4.2 column-package oracle: the nonreactive tracer through the ITD and the ridging

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases nonreactive_carry` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-21.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_nonreactive`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_nonreactive`, every other
  biogeochemistry flag off. `nBioLayers` is 7. Mushy unless the case name
  says BL99.
- Driver: `run_column.F90`, stages `itd_thermodynamics` and `ridging`.
  With the column package's `z_tracers` these run:
  - `add_new_ice_bgc`: the frazil's tracer taken from `oceanBioFluxes`
    (`vi0_init`, `add_new_ice`'s first frazil volume), and
    `adjust_tracer_profile` (`regrid_stationary` on the bio grid under a
    thickness surplus, its flux discarded; else the frazil mixed through
    the profile);
  - `lateral_melt`'s snow tracers and `lateral_melt_bgc`'s ice layers;
  - `cleanup_itd`'s `zap_small_areas` (`zap_small_bgc`, `zap_snow`) and
    `zap_snow_temperature`, into `oceanBioFluxes`.
  The ridging moves the tracers through the tracer object and runs the
  same cleanup.
- Cases: `BGC_NONREACTIVE_CARRY_CASES`. `_nonreactive_carry_state` adds to
  the ITD and ridging states the tracer profiles, mobile fractions, ocean
  concentrations, `oceanBioFluxes` and the brine grids. Every case changes
  `oceanBioFluxes`, `brineFraction`, `mobileFraction` and
  `verticalNonreactiveConc`.
  - Abort: `itd_nonreactive_abort_regrid`, a negative brine fraction in
    cell 2 stops the tracer's `regrid_stationary` with `remap_layers_bgc
    error` at `indexToCellID` 3.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: none of these cases needs an opt-in.
