# MPAS-Seaice v8.4.2 column-package oracle: the z-aerosols through the ITD and the ridging

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases zaero_carry` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_zaero`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_zaerosols`, alone or with
  `config_use_nonreactive`; every other biogeochemistry flag off.
  `nzAerosols` is 3 and `nBioLayers` 7. Mushy unless the case name says BL99.
- Driver: `run_column.F90`, stages `itd_thermodynamics` and `ridging`: the
  branches listed in `../mpas_seaice_bgc_nonreactive_carry/PROVENANCE.md`,
  each looping over the 3 or 4 bio tracers with each tracer's own
  `bio_index`, ocean concentration (`bio_index_o` 17 and 23-25),
  `zbgc_init_frac` and `oceanBioFluxes` entry.
- Cases: `BGC_ZAERO_CARRY_CASES`. `_nonreactive_carry_state` draws the
  nonreactive profiles (kept only when that tracer is on), mobile
  fractions, ocean concentrations, `oceanBioFluxes` and the brine grids,
  then the aerosols' profiles (positive, nBioLayersP3 per aerosol).
  - `itd_zaero_nonreactive_types_frazil_three_layers`: a transport type per
    aerosol with the dust and the nonreactive tracer not mobile, so the
    tracers' `zbgc_init_frac` differ (0.7 and 0.6).
  - Abort: `itd_zaero_nonreactive_abort_regrid`, a negative brine fraction
    in cell 2 stops the first tracer's `regrid_stationary` with
    `remap_layers_bgc error` at `indexToCellID` 3.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: none of these cases needs an opt-in.
