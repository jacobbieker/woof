# MPAS-Seaice v8.4.2 column-package oracle: the vertical tracers with the z-aerosols

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases zaero` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-22.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8). Same build and driver as
  `../mpas_seaice_bgc_nonreactive`.
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`,
  `config_use_vertical_tracers` and `config_use_zaerosols`, alone or with
  `config_use_nonreactive`; every other biogeochemistry flag off. The
  Registry fixes `nzAerosols` at 3 (black carbon 1 and 2, the first dust) and
  `nBioLayers` at 7. Mushy unless the case name says BL99.
- Driver: `run_column.F90`, stages `init` and `biogeochemistry`. The tracer
  object's row `colpkg_zbgc` is
  `[nBioTracers, nt_zbgc_frac, bio_index(1:n), bio_index_o(1:n)]`: the
  aerosols' `bio_index_o` are 23-25, after the nonreactive tracer's 17.
- Cases: `BGC_ZAERO_CASES`. `_nonreactive_state` adds to `_bgc_state` the
  vertical tracers' inputs (the nonreactive profiles only when that tracer
  is on, drawn either way), then `_zaero_state` the aerosols' positive ice
  and snow profiles, one value below puny.
  - Init: the aerosols alone; with the nonreactive tracer, a transport type
    per band (dust not mobile) and the frazil and new-ice fractions; three
    layers and four categories.
  - `column_biogeochemistry`:
    - the aerosols alone; with the nonreactive tracer;
    - BL99 with a transport type per band;
    - released over half an hour (dust not mobile);
    - four layers and three categories.
  - Aborts:
    - `bgc_zaero_nonreactive_abort_negative`: negative ocean concentrations
      under cell 3's thin old ice stop `z_biogeochemistry` with
      `negative bgc` at `indexToCellID` 4;
    - `bgc_zaero_nonreactive_abort_order`: the same, and the second
      aerosol negative in that cell's first category. Each of
      `z_biogeochemistry`'s phases runs over every tracer before the next,
      so it stops with `zbgc initialization error` before the nonreactive
      tracer's thin-ice flux makes it negative.
- Every non-aborting case's `atmosBioFluxes` holds only the last owned
  cell's deposition: `column_biogeochemistry` zeroes the whole array on
  each cell's turn (`mpas_seaice_column.F:3131`).
- Files: `oracle-r8.npz` holds, per case, every output field whose words
  differ from the input's. `column-logs.json` holds the tracer object and
  the log lines. `inputs.sha256` holds the input-stream digests.
- libm: `exp` (`update_hbrine`'s drainage, and the mobile phase's release
  and retention) is the host's glibc 2.44. On the GPU it is the
  repository's default transcription in `glibc_flt64.cuh` (MIT), so no LGPL
  opt-in is needed.
