# MPAS-Seaice v8.4.2 incremental-remap oracle

Written by `tools/mpas_seaice_oracle/build.sh` and `make_remap_fixture.py` on pb
(Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256`, which now pins
  `mpas_seaice_advection_incremental_remap.F` and its tracers module.
- Compiled verbatim:
  - `mpas_seaice_advection_incremental_remap_tracers.F`;
  - `mpas_seaice_advection_incremental_remap.F`, with `-Dprivate=public`
    (visibility only).
- Stand-ins (`remap_standins.F90`): the RBF reconstruction setup (its
  `coeffs_reconstruct` is the input's), `seaice_error`'s codes and abort
  write, and upwind advection (stops by name: v8.4.2 cannot run it).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`. No FMA instruction in any object.
- Driver: `run_incremental_remap.F90`. It runs
  `seaice_init_advection_incremental_remap`, then the stage the case names
  (`oracle_stage`).
- Cases: `woof.verify.mpas_seaice_remap_oracle`, on planar hexagon and
  quadrilateral blocks (`woof.verify.mpas_seaice_planar_mesh`, with land and
  halo layers). The cases are built from correctly rounded operations only,
  so `inputs.sha256` holds on every host. The arm64 (macOS) build of the
  oracle writes the same words.
- `oracle-r{4,8}.npz`: per case, every output field whose words differ from
  the input's (the init's geometry included). `remap-logs.json`: the ordered
  tracer list and the logged warnings, per case and precision.
