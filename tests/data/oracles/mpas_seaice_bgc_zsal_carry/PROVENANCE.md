# MPAS-Seaice v8.4.2 column-package oracle: the inert vertical salinity through the ITD, the ridging and the steps

Written by `make_column_fixture.py --cases zsal_carry` on pb (Linux 6.12
x86_64, glibc 2.44), 2026-11-21, from the same unpatched build as
`../mpas_seaice_bgc_zsal/PROVENANCE.md` (`tools/mpas_seaice_oracle/build.sh`
without `WOOF_ORACLE_PATCH`).

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`, compiled verbatim (gfortran
  16.2.0, `-O0 -ffp-contract=off`, RKIND 8).
- Configuration: `config_use_column_biogeochemistry`, `config_use_brine`
  and `config_use_vertical_zsalinity` with BL99 (`linear_salt`), every other
  biogeochemistry flag off; `nBioLayers` 7. The steps declare the opt-in
  correction `bgc_inactive_diagnostics` (see
  `../mpas_seaice_bgc_brine_carry/PROVENANCE.md`): `totalChlorophyll` gets an
  array that is not compared.
- Driver: `run_column.F90`, stages `itd_thermodynamics`, `ridging`,
  `column_step` and `seaice_step` (the time step with the velocity solver on
  a culled hexagonal torus).
- What v8.4.2 does with the vertical salinity here: its `solve_zsal` is never
  set (`ice_colpkg.F90:4325-4345`) and `verticalSalinity` is copied to and
  from the tracer arrays only with the vertical tracers
  (`mpas_seaice_column.F:6292, 6664`), so the tracers are counted and never
  read back: `cleanup_itd` adds a zero `dfzsal` to `zSalinityFlux`,
  `seaice_column_reinitialize_diagnostics_bgc` zeroes `zSalinityFlux` and
  `zSalinityGDFlux`, and the pool's `verticalSalinity` keeps its words. The
  time step runs without advection (`config_use_advection = .false.`, as
  every column-package time-step case), so the incremental remap's
  `verticalSalinity` (`mpas_seaice_advection_incremental_remap_tracers.F:463-464`)
  is not exercised here.
- Cases: `woof.verify.mpas_seaice_column_oracle.BGC_ZSAL_CARRY_CASES` and
  `BGC_ZSAL_CARRY_COUPLED_CASES`; none aborts.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: the steps' delta-Eddington shortwave and BL99 call glibc's `exp`,
  `atan` and the rest, compared under the opt-in LGPL transcription
  (`WOOF_GLIBC239_LIBM64=1`).
