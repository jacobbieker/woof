# MPAS-Seaice v8.4.2 column-package oracle: the z-aerosols through the time steps

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases zaero_step` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build as `../mpas_seaice_bgc_zaero`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_zaerosols`, alone or with
  `config_use_nonreactive`; every other biogeochemistry flag off, and
  `woof_v842_corrections = ('bgc_inactive_diagnostics',)` as in
  `../mpas_seaice_bgc_nonreactive_step`. `nzAerosols` is 3 and `nBioLayers` 7.
- Driver: `run_column.F90`, stages `column_step` (the column package's part
  of `seaice_timestep`) and the coupled step with the velocity solver, 3
  steps each. With the z-aerosols:
  - `column_separate_snow_ice_tracers` after the pre-dynamics and
    `column_combine_snow_ice_tracers` before each subcycle's ridging move
    `verticalAerosolsConc` to and from `verticalAerosolsSnow` and
    `verticalAerosolsIce`;
  - `seaice_column_coupling_prep` maps the first dust's ocean flux
    (`bio_index_o` 25) into `oceanDustIronFlux` and skips the black
    carbons' (23-24) and the nonreactive tracer's (17).
- Cases: `BGC_ZAERO_STEP_CASES`: BL99 with dEdd; with the nonreactive
  tracer, a transport type per aerosol, two subcycles, four layers, three
  categories and the Gregorian calendar; mushy with level ponds and no
  vertical thermodynamics; and the coupled step with Rothrock ridging, with
  and without the nonreactive tracer. `_nonreactive_carry_state` gives
  their tracer inputs. None of them meets `bgc_snow_new_category`'s NaN.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: the delta-Eddington shortwave's `sin`, `cos`, `atan`, `asin`, `exp`
  and `log` are the host's glibc 2.44; on the GPU they need the LGPL opt-in
  `WOOF_GLIBC239_LIBM64=1`.
