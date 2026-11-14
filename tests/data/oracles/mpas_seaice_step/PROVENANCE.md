# MPAS-Seaice v8.4.2 time-step oracle (dynamics only)

Written by `tools/mpas_seaice_oracle/build.sh` and `make_step_fixture.py` on pb
(Linux 6.12 x86_64, glibc 2.44), 2026-11-14.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256`, which now pins
  `mpas_seaice_diagnostics.F`, `mpas_seaice_column.F` and
  `mpas_seaice_time_integration.F`.
- Cut verbatim:
  - `seaice_timestep`, `store_timestep_counter`, `seaice_timestep_finalize`;
  - the column package's time-integration wrappers and re-initialisations
    (no-ops without the column package, except the atmospheric fluxes'
    re-initialisation);
  - `seaice_initialize_time_diagnostics`, `seaice_set_time_diagnostics`,
    `seaice_check_state`, `seaice_load_balance_timers`.
- Compiled verbatim: `mpas_seaice_velocity_solver.F`,
  `mpas_seaice_velocity_solver_variational.F`, the constitutive relation and
  `seaice_constants`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`. No FMA instruction in any object. libm: the host glibc
  (`expf`/`exp` in the ice strength).
- Driver: `run_seaice_step.F90`.
  - Init: `dynamicsTimeStep` and `elasticTimeStep` as
    `seaice_init_velocity_solver` sets them, `seaice_init_evp`, and
    `seaice_initialize_time_diagnostics`.
  - Then `oracle_n_steps` times: the clock advanced, `seaice_timestep` and
    `seaice_timestep_finalize`, as `seaice_core_run` calls them.
  - The time manager is the shared stubs' (whole seconds, proleptic
    Gregorian).

## Cases

| case | torus (owned columns) | steps | dt | what it covers |
|---|---|---|---|---|
| `evp_three_steps` | 12 x 6 (3) | 3 | 1800 s | EVP, 30 elastic subcycles, air stress, ocean stress, geostrophic tilt |
| `revised_two_dynamics_subcycles` | 12 x 6 (3) | 2 | 3600 s | revised EVP, two dynamics subcycles of 12 elastic |
| `no_air_no_tilt` | 10 x 5 (3) | 2 | 1800 s | no air stress, no tilt, no ocean stress; a start across a leap day (its time diagnostics restart from the start time) |

The meshes are `woof.verify.mpas_ocean_hexmesh.hex_mesh`'s culled hexagonal
tori as one block of a decomposition, so `nCellsSolve` < `nCells`; MPAS's
missing-cell element is carried. `interiorVertex` and `cellVerticesAtVertex`
are derived as MPAS derives them. The variational basis is random, with a 10 km
mesh's magnitudes; the ice state has five categories and an ice-free run of
cells.

## Grading

`tests/test_mpas_seaice_timestep_parity.py` and `woof hex seaice verify` run
`woof.hex.mpas_seaice.timestep_v842` (`initialize`, `advance`), and every word
must be bit-identical. The restart test runs one step, writes a restart, reads
it into fresh pools and runs the rest.

2026-11-14: the cases now set `config_advection_type = 'incremental_remap'`
(MPAS's default), which `seaice_init_advection` reads. The time-step oracle
now compiles `mpas_seaice_advection.F` and the incremental remap verbatim.
`inputs.sha256` was regenerated; every output word is unchanged.
