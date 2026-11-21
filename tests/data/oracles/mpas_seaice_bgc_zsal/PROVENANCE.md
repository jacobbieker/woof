# MPAS-Seaice v8.4.2 column-package oracle: the vertical salinity as v8.4.2 runs it (inert)

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases zsal` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim as in
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8).
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`
  and `config_use_vertical_zsalinity` with BL99 (`linear_salt`), every other
  biogeochemistry flag off; `nBioLayers` 7.
- Driver: `run_column.F90`, stages `init`, `biogeochemistry` (which, with
  the vertical salinity, first runs `colpkg_init_thermo` for `l_brine` and
  `colpkg_init_zsalinity` with `restart_zsal` for its `dts_b`, as MPAS's
  init would have) and `vertical_thermodynamics`. The tracer object adds
  `colpkg_zsal` (`nt_bgc_S`).
- What v8.4.2 does with it (the reason this fixture exists):
  - `colpkg_init_parameters` takes `solve_zsal_in` and never assigns the
    column package's `solve_zsal` (`ice_colpkg.F90:4325-4345`; the MPAS
    assignment is commented out, `mpas_seaice_column.F:9608`), so it stays
    `.false.`: `colpkg_biogeochemistry` runs the brine height's
    `compute_microS_mushy` and BL99's `dTemp` stays 0.02;
  - `set/get_cice_biogeochemistry_tracer_array_category` and their cell
    versions copy `verticalSalinity(Cell)` only inside `elseif
    (config_use_vertical_tracers)` (`mpas_seaice_column.F:6292, 6664, 7075,
    7562`), so `colpkg_init_zsalinity`'s `sss*salt_loss` never reaches the
    pool; the init only zeroes `rayleighCriteriaReal`.
  The tracer count still includes the `nBioLayers` zsal tracers.
- Cases: `woof.verify.mpas_seaice_column_oracle.BGC_ZSAL_CASES` (the init,
  the stage with `_bgc_state` and `_zsal_state`, the remap abort, BL99's
  vertical thermodynamics with cell 5 inside the `dTemp` band). The
  stage's remap abort is the brine height's: cell 8's last category,
  `indexToCellID` 8.
- `column_biogeochemistry`'s `abortMessage` is `strKIND` long, never
  initialised, and written through the column package's `char_len` (80)
  stop labels: past 80 characters the logged line is stack bytes, so
  `parse_log` keeps the label only.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object, the log lines and the abort
  case's `crash`. `inputs.sha256`: the input-stream digests.
- libm: `exp` is glibc 2.44's (the GPU's MIT transcription); the vertical
  thermodynamics' `atan` is glibc's, compared under the opt-in LGPL
  transcription (`WOOF_GLIBC239_LIBM64=1`).
