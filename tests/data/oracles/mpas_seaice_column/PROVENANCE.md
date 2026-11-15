# MPAS-Seaice v8.4.2 column-package oracle: initialisation and aggregation

Written by `tools/mpas_seaice_oracle/build.sh` and `make_column_fixture.py`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The column package
  (`src/core_seaice/column`, with `constants/cice`) and `mpas_seaice_column.F`
  (`-Dprivate=public`, visibility only) are compiled verbatim; the
  `mpas_seaice_initialize.F` and `mpas_seaice_forcing.F` routines the init
  runs are cut verbatim (`tools/mpas_seaice_oracle/README.md`).
- Compiler and flags: gfortran 16.2.0, `-O0 -ffp-contract=off -cpp
  -ffree-form -ffree-line-length-none`, RKIND 8 only: the column package is
  `dbl_kind` and `mpas_seaice_column.F` passes it RKIND arrays, so the
  wrappers do not compile with `-DSINGLE_PRECISION`. No FMA in any object
  (objdump). `cos`, `pow` and `tanh` are the host's glibc 2.44.
- Drivers: `run_column.F90` (`seaice_init_column_physics_package_parameters`,
  then `init`, `aggregate` or `update_state`), `run_init_thermo.F90`
  (`colpkg_init_thermo` for 1 to 64 ice layers, BL99).
- Cases: `woof.verify.mpas_seaice_column_oracle`: every config at its Registry
  default unless the case overrides it; the field table is the Registry's,
  with the fields of packages that are off absent.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object (`TOBJ`) and any log lines.
  `init-thermo-r8.npy`: the profiles `sprofile(1:n+1)`, n = 1..64, one after
  the other. `inputs.sha256`: the input-stream digests.
- Grading: the GPU's `uw_cos` (CORE-MATH, correctly rounded) equals these
  profiles for every count except 11, 31, 33 and 55 ice layers, where glibc's
  cos is 1 ULP off on one level; `column_v842.COS_REFUSED_LAYERS` refuses
  them. The inputs draw `random_sample` and scale it in numpy, because
  `RandomState.uniform` fuses an FMA on arm64; the digests are host-independent.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR). Its
`src/core_seaice/column/documentation/README` (dated June 30, 2015, marked
"CHANGE FOR RELEASE") says the column package was available to approved
collaborators and "may not be redistributed beyond the approved
collaboration". The project owner decided to treat that pre-release notice as
stale, since the column package ships publicly inside the BSD-3 MPAS-Model
release, and to port it under MPAS's BSD-3 licence
(`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
