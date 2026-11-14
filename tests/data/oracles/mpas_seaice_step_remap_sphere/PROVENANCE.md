# MPAS-Seaice v8.4.2 time-step oracle with incremental-remap advection on the sphere

Written by `tools/mpas_seaice_oracle/build.sh` and `make_step_remap_fixture.py --sphere`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source, compiler, flags and driver: as
  `tests/data/oracles/mpas_seaice_step_remap` (v8.4.2, `f34984b2`, gfortran
  16.2.0 `-O0 -ffp-contract=off`, RKIND 4 and 8; `run_seaice_step.F90` with
  `config_use_advection`).
- Cases: `woof.verify.mpas_seaice_step_remap_sphere_oracle`. Each joins a step
  case and a spherical remap case on one `woof.verify.mpas_seaice_sphere_mesh`
  block: hexagons about the North Pole with `config_rotate_cartesian_grid` over
  3 steps, and stretched (variable-resolution) quadrilaterals at 53 degrees
  north without it, with level ice and ponds, two dynamics subcycles, revised
  EVP and the mean-recovery check, over 2 steps.
- `oracle-r{4,8}.npz`: per case, every output field whose words changed. The
  remap's triangle values (scratch) are left out; the fluxes built from them
  are kept.
