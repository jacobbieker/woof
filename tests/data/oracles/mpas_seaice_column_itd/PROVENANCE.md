# MPAS-Seaice v8.4.2 column-package oracle: ITD thermodynamics

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases itd` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-15.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, stage `itd_thermodynamics`. It runs
  `seaice_init_column_physics_package_parameters`, then `colpkg_init_thermo`
  (`heat_capacity`) and `init_column_itd` (`colpkg_init_itd`:
  `categoryThicknessLimits` and `hi_min`), then
  `column_itd_thermodynamics`. The abort status is logged as `TOBJ IERR`,
  and the column package's warnings as `WARN` lines.
- Cases: `woof.verify.mpas_seaice_column_oracle.ITD_CASES`:
  - mushy with level ponds (the Registry default), with three layers and
    `config_update_ocean_fluxes`;
  - BL99 with `config_update_ocean_fluxes`, with CESM ponds;
  - mushy with topo ponds and no age or first-year tracers;
  - the delta function on the `new` bounds (four categories), and one
    category (`reduce_area`);
  - three aborts: `shift_ice: negative daice` in linear_itd (cell 6),
    `aggregate ice area out of bounds` (cell 1) and `Zap ice: negative ice
    area` (cell 7), each with a later aborting cell MPAS never reaches.

  The owned cells cover the branches (`woof.verify.mpas_seaice_column_oracle._itd_state`):
  plain growth and lateral melt, new ice in open water, frazil thicker than
  hi0max spread over every category, sub-puny categories zapped, snow too
  warm and too cold (zap_snow_temperature), growth past the bounds (rebin),
  category 1 thinned below hi_min, a total area in (1, 1 + puny), and melt.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `column_itd_thermodynamics` calls none. `colpkg_init_itd`'s `tanh`
  (the `original` bounds) is the host's glibc 2.44; on the GPU it is the
  correctly rounded `cr_tanh` of stage 1, and the two agree on every bound
  here. No LGPL code and no opt-in is involved.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/column_itd_v842.cu` is a modified derivative of the
column package's `ice_therm_itd.F90`, `ice_itd.F90`, `ice_colpkg.F90` and
`ice_colpkg_tracers.F90` and of `mpas_seaice_column.F`, under the same
licence (`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
