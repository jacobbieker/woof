# MPAS-Seaice v8.4.2 time-step oracle with incremental-remap advection

Written by `tools/mpas_seaice_oracle/build.sh` and `make_step_remap_fixture.py`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source, compiler and flags: as `tests/data/oracles/mpas_seaice_step`
  (v8.4.2, `f34984b2`, gfortran 16.2.0 `-O0 -ffp-contract=off`, RKIND 4 and 8).
- Driver: `run_seaice_step.F90` with `config_use_advection`. The init runs
  `seaice_init_advection` (the remap's init, compiled verbatim; its RBF
  coefficients are the input's). Each dynamics subcycle runs
  `seaice_run_advection` (`mpas_seaice_advection.F` and the remap, compiled
  verbatim) after the velocity solver.
- Cases: `woof.verify.mpas_seaice_step_remap_oracle`. Each joins a step case
  and a remap case on one planar block. One case is hexagons over 3 steps; the
  other is quadrilaterals with level ice, ponds and ice age, two dynamics
  subcycles, revised EVP and the mean-recovery check, over 2 steps.
- `oracle-r{4,8}.npz`: per case, every output field whose words changed. The
  remap's triangle values (scratch) are left out; the fluxes built from them
  are kept.
