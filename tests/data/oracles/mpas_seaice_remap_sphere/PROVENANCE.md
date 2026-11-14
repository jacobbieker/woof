# MPAS-Seaice v8.4.2 incremental-remap oracle on the sphere

Written by `tools/mpas_seaice_oracle/build.sh` and `make_remap_fixture.py --sphere`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source, compiler, flags, driver and stand-ins: as
  `tests/data/oracles/mpas_seaice_remap` (v8.4.2, `f34984b2`, gfortran 16.2.0
  `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`; `run_incremental_remap.F90`, stages 2, 3 and 4).
- Cases: `woof.verify.mpas_seaice_remap_sphere_oracle`, on
  `woof.verify.mpas_seaice_sphere_mesh` blocks (`on_a_sphere`): hexagons and
  quadrilaterals about the North and South Poles with
  `config_rotate_cartesian_grid`, and at 53 degrees north without it; each kind
  also stretched to variable resolution. The init
  (`seaice_init_advection_incremental_remap`, verbatim) rotates the global
  vectors, builds the local frames and the tangent-plane geometry; those are
  the GPU's input. Inputs are built from correctly rounded operations only, so
  `inputs.sha256` holds on every host.
- `oracle-r{4,8}.npz`: per case, every output field whose words differ from the
  input's (the init's geometry included). `remap-logs.json`: the ordered tracer
  list, the logged warnings (the mean-recovery check) and stage 4's error code.
