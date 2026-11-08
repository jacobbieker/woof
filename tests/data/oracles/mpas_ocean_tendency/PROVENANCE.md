# MPAS-Ocean v8.4.2 full-tendency oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece tendency` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-08.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. This oracle compiles
  `mpas_ocn_tendency.F` and every term `ocn_tend_vel` and `ocn_tend_thick`
  call: `mpas_ocn_{vel_coriolis,vel_vadv,vel_pressure_grad,vel_hmix,vel_hmix_del2,vel_hmix_leith,vel_hmix_del4,surface_bulk_forcing,forcing,vel_forcing,vel_forcing_surface_stress,vel_forcing_explicit_bottom_drag,vel_forcing_rayleigh,thick_hadv,thick_vadv,thick_surface_flux,constants}.F`,
  `framework/mpas_constants.F` and `framework/mpas_kind_types.F`.
- Stand-ins (`stubs.F90`, `stubs_tendency.F90`), no arithmetic:
  - the framework pools;
  - the tracer, diagnostics, vertical-mixing and equation-of-state modules
    `mpas_ocn_tendency.F` USEs. Their routines are reached only from
    `ocn_tend_tracer` and stop by name.
  - land-ice fluxes and frazil, which return when their switch is off (as the
    real routines do) and stop by name when it is on.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Reference of record: `-O0 -ffp-contract=off`, with `-DSINGLE_PRECISION`
  (RKIND = 4) for `oracle-r4.npz` and RKIND = 8 for `oracle-r8.npz`. build.sh
  fails if an ocean object contains an FMA instruction.
- libm: gfortran calls glibc 2.44's `expf` (`ocn_forcing_transmission`) and
  `cosf`/`sinf` (the bulk wind stress) at RKIND 4, and `exp` at RKIND 8. `sqrt`
  is the instruction and `x**2` is `x*x`.

  The GPU port uses the repository's MIT transcriptions of the same
  functions (`woof/core/kernels/glibc_flt32.cuh`, `glibc_trig_flt32.cuh`,
  `glibc_flt64.cuh`). On a host whose libm is not glibc, build.sh links them
  in place of the host libm (`tools/oracle_libm/glibc_host.cpp`).

  glibc's double `cos`/`sin` have no permissive equal, so the bulk wind stress
  case is recorded at RKIND 4 only. The port refuses that switch at RKIND 8.
- Driver: `run_tendency.F90`.
  1. It fills the mesh, state (with the `tracers` subpool), diagnostics,
     forcing, tendency and scratch pools, and the configs.
  2. It calls `ocn_constants_init`, `ocn_tendency_init` and the `_init` of
     every term.
  3. It calls `ocn_tend_vel`, then `ocn_tend_thick`.
  4. It then calls `ocn_vel_forcing_tend` with a nonzero `surfaceStressIn`
     into `tendVelForcing`, and `ocn_thick_surface_flux_tend` with nonzero
     `fluxIn`/`runoffIn` into `tendThickSflux`. These grade the forcing terms
     at both precisions whatever the bulk switches are.

  Every output starts from the case's sentinel words.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<field>` for `tendNormalVelocity`, `surfaceStress`, `surfaceStressMagnitude`, `viscosity`, `tendLayerThickness`, `surfaceThicknessFlux`, `surfaceThicknessFluxRunoff`, `tendVelForcing`, `tendThickSflux` |
| `inputs.sha256` | the input-stream digest of each case at each of its precisions (`woof.verify.mpas_ocean_tendency_oracle.input_bytes`) |

## Coverage

| case | precisions | what it switches on |
|---|---|---|
| `zmid_del2_split` | r4, r8 | split-explicit, pressure_and_zmid, del2, bulk thickness flux, surface stress (zero in the composition, nonzero standalone), surface flux; dry edges |
| `jts_rk4_bulk_drag` | r4 | RK4, Jacobian_from_TS, del4 + Leith, bulk wind stress and thickness flux, explicit bottom drag, Rayleigh friction and bottom friction; every edge wet |
| `jdens_drag_rayleigh` | r4, r8 | Jacobian_from_density, del2 + del4, bulk thickness flux, bottom drag, Rayleigh and bottom friction, rho_sw 1027.5; every edge wet |
| `all_tend_off` | r4, r8 | `config_disable_vel_all_tend` and `_thick_all_tend` (the composed outputs are the Fortran's zeros); standalone forcing with bottom drag; every edge wet |
| `mont_forcing_off` | r4, r8 | RK4, pgrad and hmix disabled, surface stress and surface flux disabled, Rayleigh friction |

Every case also reaches the transmission's `exp(-100)` floor and has
absorbed-fraction column sums on both sides of 1.

## Grading

`tests/test_mpas_ocean_tendency_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.forcing_v842` (`tend_vel`, `tend_thick`,
`vel_forcing_tend`, `thick_surface_flux_tend`), and every word must be
bit-identical. On 2026-11-08 the result was 0 differing words in every field,
case and precision on CUDA (RTX A6000).
