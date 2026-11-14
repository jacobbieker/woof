# MPAS-Seaice v8.4.2 incremental-remap oracle: the conservation and monotonicity checks

Written by `tools/mpas_seaice_oracle/build.sh` and `make_remap_fixture.py --checks`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source, compiler, flags, driver and stand-ins: as
  `tests/data/oracles/mpas_seaice_remap` (v8.4.2, `f34984b2`, gfortran 16.2.0
  `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`; `run_incremental_remap.F90`, stage 4). The framework stand-in
  `mpas_log_write` records MPAS_LOG_ERR messages (`stub_log_err_unit`) and
  carries on, as MPAS does; only MPAS_LOG_CRIT stops.
- Cases: `woof.verify.mpas_seaice_remap_checks_oracle`, with
  `config_conservation_check` and/or `config_monotonicity_check` on: planar
  hexagon and quadrilateral blocks (`woof.verify.mpas_seaice_planar_mesh`) and
  spherical ones (`woof.verify.mpas_seaice_sphere_mesh`), two of them
  variable-resolution. The `closed` cases keep every vertex of a halo cell
  still, so nothing crosses the owned cells' rim.
- `oracle-r{4,8}.npz`: per case, every output field whose words differ from the
  input's (the sums `<tracer>Cons`/`Cons@2` and local extremes
  `<tracer>LocalMin`/`LocalMax` included). `remap-logs.json`: the ordered tracer
  list, the warnings, the error messages (`errors`) and the error code.
