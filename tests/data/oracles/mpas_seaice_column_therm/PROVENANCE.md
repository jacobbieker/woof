# MPAS-Seaice v8.4.2 column-package oracle: BL99 vertical thermodynamics

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases therm` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-15.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, stage `vertical_thermodynamics`. It runs
  `seaice_init_column_physics_package_parameters`, then `colpkg_init_thermo`
  (which sets `l_brine` and `heat_capacity`), then
  `seaice_column_vertical_thermodynamics`. The abort status is logged as
  `TOBJ IERR`.
- Cases: `woof.verify.mpas_seaice_column_oracle.VT_CASES`:
  - bubbly and MU71 conductivity (MU71 with highfreq and three atmosphere
    iterations);
  - the coupled surface temperature (`config_use_surface_temperature`
    false);
  - melting;
  - first-year ice reset in each hemisphere;
  - one layer and one category;
  - an abort on low starting salinity (cell 6, `init_vertical_profile`).

  Every cell's state is consistent with BL99: the snow-layer shortwave is
  zero where hsn <= 1e-4, and the interior shortwave is the sum of the layer
  terms. Some cells are ice-free, sub-puny or have no snow, and one cell
  forms snow ice.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `atan` (`psimhu`, `psixhu`), `exp` and `log` are the host's
  glibc 2.44. glibc's binary64 atan is not correctly rounded on these
  arguments. `tools/mpas_seaice_oracle/grade_atmo_atan.cpp` compared it with
  the repository's correctly rounded `wcr_atan` on 75,497,472 arguments in
  [1, 3.5625), the range of psimhu's `xqq`. They differ in 43,325 words
  (0.057%). The GPU port therefore calls `g239::g239_atan`
  (`tools/glibc239_libm_gpu`, an LGPL transcription of glibc's atan, graded
  word for word against glibc 2.44 in `verification.log`). It does so only
  with the existing opt-in `WOOF_GLIBC239_LIBM64=1`. Without the opt-in the
  port refuses BL99 by name.
- The arguments of `log` are folded on the host: `rdn_ice` is
  `vonkar/log(zref/iceruf)` and `al2` is `log(zref/zTrf)`. These are the
  host's words.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
glibc is LGPL-2.1. The atan transcription stays behind the opt-in, and
neither the default build nor the wheels carry it.
