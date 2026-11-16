# MPAS-Seaice v8.4.2 column-package oracle: ridging

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases ridge` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-15.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, stage `ridging`. It runs
  `seaice_init_column_physics_package_parameters`, then `colpkg_init_thermo`
  (`heat_capacity`, which `cleanup_itd` reads) and `init_column_itd`
  (`categoryThicknessLimits`), then `column_ridging`. The abort status is
  logged as `TOBJ IERR`, and the column package's warnings as `WARN` lines.
- Cases: `woof.verify.mpas_seaice_column_oracle.RIDGE_CASES`:
  - the exponential participation and redistribution with mushy and level
    ponds (the Registry default);
  - Thorndike75 with Hibler80 (BL99); Thorndike75 with the exponential
    redistribution and CESM ponds; the exponential participation with
    Hibler80 and topo ponds;
  - three dynamics subcycles (`dynamicsTimeStep` = `config_dt` / 3), an
    e-folding scale of 4.5, topo ponds and three ice layers;
  - the delta function on the `new` bounds (four categories) without level
    ice, ponds or age, and one category;
  - `ridge_negative_opening_dies`: a negative convergence with no open
    water. `ridge_shift` stops with `Ridging error: aice0 < 0`, then writes
    that label (512 characters) list-directed into its 256-character
    warning, and v8.4.2 dies: `At line 1257 of file ice_mechred.F90 /
    Fortran runtime error: End of record`. The log keeps that error as
    `crash`, and the case has no outputs;
  - two aborts: `ridge_ice: Exceeded max number of ridging iterations`
    (cell 6: 10 km ice whose ridges are barely thicker) and
    `ice: ITD cleanup error in colpkg_step_ridge` (cell 8: a negative
    category that zap_small_areas stops on), each with a later aborting
    cell MPAS never reaches.

  The owned cells cover the branches (`woof.verify.mpas_seaice_column_oracle._ridge_state`):
  plain closing, open water only, transport convergence with the closing
  cut back (`Repeat ridging`), divergence (opening), a sub-puny category and
  snow too warm (zap_snow_temperature), thick ice above the bounds, strong
  convergence of thin ice, no forcing, and ice without volume (under
  Hibler80 its `ratioRidgeThicknessToIce` is `hrmean/0 = +inf`).
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object, the log lines and, for a
  case v8.4.2 dies on, `crash`. `inputs.sha256`: the input-stream digests.
- libm: the participation and redistribution functions call `exp`, the host
  glibc 2.44's. On the GPU that is `glibc_exp` (`glibc_flt64.cuh`), the MIT
  transcription of the same routine. gfortran folds the constant
  `c1/(c1 - exp(-astari))` with MPFR; its `exp(-20)` is the correctly rounded
  word, and glibc gives the same. `colpkg_init_itd`'s `tanh` is stage 1's.
  No LGPL code and no opt-in is involved.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/column_ridge_v842.cu` is a modified derivative of the
column package's `ice_mechred.F90` and `ice_colpkg.F90` and of
`mpas_seaice_column.F`, under the same licence
(`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
