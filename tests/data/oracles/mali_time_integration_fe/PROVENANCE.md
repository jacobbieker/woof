# MALI v8.4.2 forward-Euler time-step oracle

Written by `tools/mali_oracle/build.sh` (`run_time_integration_fe_r4`, `_r8`) and
`make_time_integration_fe_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-11.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified
  (`tools/mali_oracle/SOURCES.sha256`). Beyond the other MALI oracles' files it
  compiles `mpas_li_time_integration_fe.F`, `mpas_li_velocity.F`,
  `mpas_li_velocity_simple.F`, `mpas_li_calving.F` and
  `mpas_li_subglacial_hydro.F` whole, and `mpas_reconstruct_1d`/`_2d` cut
  verbatim out of `mpas_vector_reconstruction.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, with `-DSINGLE_PRECISION` for RKIND 4 and without for
  RKIND 8; no object contains an FMA instruction. The libm calls are glibc
  2.44's `powf`, `expf`, `cosf`, `sinf`, `atanf` and `tanhf` (RKIND 4), and
  `pow`, `exp`, `tanh`, `atan` (RKIND 8).
- Driver: `run_time_integration_fe.F90` reads a self-describing stream (the
  clock, every config, every field by pool), builds a one-block domain, and
  calls `li_thermal_init(domain, err)` as on a restart, then
  `li_time_integrator_forwardeuler(domain, err)`.
- The stand-ins (`stubs.F90`): halo exchanges and one-rank reductions are the
  identity; the clock is whole seconds (only `set_timestep` reads it); the
  external (Albany) velocity solver stops if reached. Scratch fields resolve
  to arrays the driver fills with random words. The model leaves them
  uninitialised, so no output may depend on them.
- The fixture is x86-64 gfortran's. Its `MAX`/`MIN` are `maxss`/`maxsd`/`minss`,
  which keep an operand on a tie of signed zeros: in
  `calculate_layerThicknessEdge`, `max(+0, -0)` is `+0` and `max(-0, +0)` is
  `-0`. arm64 gfortran (`fmaxnm`) orders `-0` below `+0`, so an oracle rebuilt
  on Apple silicon differs in those words only.

## Coverage

| case | nCells | levels | velocity | thermal | advection (thickness/tracer) | floating melt | time step |
|---|---|---|---|---|---|---|---|
| `sia_temperature_fo` | 96 | 8 | sia | temperature | fo/fo | constant | fixed, 30 days |
| `sia_enthalpy_adaptive` | 96 | 10 | sia (kite areas, PB1982) | enthalpy | fo/fo | mismip | adaptive with the diffusive CFL; the forcing interval binds |
| `none_temperature_no_tracers` | 80 | 6 | none | temperature | fo/none | seroussi (r4), none (r8) | adaptive; the maximum step binds and is floored |
| `sia_no_thermal_sphere` | 72 | 5 | sia (normal slope, constant A) | none | fo/fo | none | fixed; on a sphere at RKIND 4, planar at RKIND 8 |
| `sia_no_advection` | 64 | 6 | sia | enthalpy | none/fo | file | fixed |

`nCellsSolve = nCells`: `li_calculate_apparent_diffusivity` reads the
reconstructed slope on every cell, and `mpas_reconstruct` sets it only on the
owned cells. In a halo the model would read its uninitialised scratch.

At RKIND 8 the fixture leaves out `eTheta`. glibc's binary64 `atan` is LGPL and
not correctly rounded, so the port refuses it. Every other output of the step
is compared at both precisions:

- the masks;
- thickness, layer thickness and edge layer thickness;
- surfaces;
- mass balances;
- temperature, enthalpy and water fraction;
- the pressure melting point;
- velocities, reconstructions, speeds and strain rates;
- `apparentDiffusivity`;
- `dHdt`, `thicknessOld` and `calvingThickness`;
- the time step and the CFL limits.
