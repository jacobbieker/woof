# MPAS-Seaice v8.4.2 column-package oracle: the vertical tracers with the nonreactive tracer

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases nonreactive` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-21.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8).
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_nonreactive`, every other
  biogeochemistry flag off. `nBioLayers` is 7 (the Registry fixes it).
  Mushy unless the case name says BL99.
- Driver: `run_column.F90`, stages `init` and `biogeochemistry`. With the
  vertical tracers on, the tracer object adds the row `colpkg_zbgc`:
  `[nBioTracers, nt_zbgc_frac, bio_index(1:n), bio_index_o(1:n)]`. This row is
  written only with the vertical tracers, so the other fixtures do not
  change: `../mpas_seaice_bgc_brine`, `../mpas_seaice_bgc_brine_carry`,
  `../mpas_seaice_bgc_zsal` and `../mpas_seaice_bgc_zsal_carry` regenerate
  byte for byte with this driver.
- Cases: `BGC_NONREACTIVE_CASES`. `_nonreactive_state` adds to
  `_bgc_state`:
  - positive ice and snow profiles, one value below puny;
  - mobile fractions, ocean concentrations and snowfall (none in every third
    cell);
  - the snow before the step;
  - the shortwave through the ice layers;
  - the fluxes and integrals the stage accumulates into.
  - Init: the default; BL99 with a mobile nitrate type and a frazil
    fraction; a bgc restart (the profiles keep their words); three layers
    and four categories.
  - `column_biogeochemistry`:
    - the default;
    - BL99 retained;
    - rapid;
    - released over half an hour;
    - stationary with four layers and three categories;
    - BL99 retained over two hours.
  - Aborts:
    - `bgc_nonreactive_abort_init`: a negative tracer under old ice stops
      `z_biogeochemistry` with `zbgc initialization error` at
      `indexToCellID` 7;
    - `bgc_nonreactive_abort_negative`: a negative ocean concentration under
      thin old ice stops it with `negative bgc` at `indexToCellID` 4.
- Files: `oracle-r8.npz` holds, per case, every output field whose words
  differ from the input's. `column-logs.json` holds the tracer object and
  the log lines. `inputs.sha256` holds the input-stream digests.
- libm: `exp` (`update_hbrine`'s drainage, and the mobile phase's release
  and retention) is the host's glibc 2.44. On the GPU it is the
  repository's default transcription in `glibc_flt64.cuh` (MIT), so no LGPL
  opt-in is needed.
