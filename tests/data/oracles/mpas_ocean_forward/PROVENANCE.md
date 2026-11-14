# MPAS-Ocean v8.4.2 forward-mode oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece forward` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`, which now pins
  `mpas_ocn_time_integration.F`.
- Cut verbatim:
  - `ocn_timestep` and `ocn_timestep_init` (`timestep_cut_head.F90`);
  - `ocn_time_integrator_split` and `ocn_time_integration_split_init` (`split_step_cut_head.F90`).
- Every module the step reaches is compiled or cut verbatim, as for the
  split-step oracle. GM and RK4 have stand-ins that stop by name.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. libm: the host glibc 2.44 (`expf`/`exp` only; no case is on the sphere).
- Driver: `run_forward.F90`. It follows `ocn_forward_mode_init` and
  `ocn_forward_mode_run` for what changes the state:
  - every module's `_init`, `ocn_timestep_init` and
    `ocn_time_integration_split_init`;
  - the state part of `ocn_init_routines_block` (transcribed in the driver);
  - `oracle_n_steps` times: the clock advanced, the absorbed fractions,
    `ocn_timestep`, and `mpas_pool_shift_time_levels`.

  The time manager is the stub's: whole seconds on the proleptic Gregorian
  calendar. Every array is written back by the allocation it was given to, so
  after an odd number of steps `NAME` holds what MPAS then calls time level 2.

## Cases

| case | torus | nVertLevels | steps | dt / btr dt | what it covers |
|---|---|---|---|---|---|
| `split_three_steps` | 7 x 6 | 8 | 3 | 120 s / 50 s (3 subcycles, a remainder) | split-explicit cold start, an odd step count |
| `split_jm_rich_bulk` | 8 x 6 | 10 | 4 | 180 s / 60 s | JM, Jacobian_from_TS, Richardson mixing, bulk forcing with Jerlov, budgets, tracer del2, the coupler's running means over four steps |
| `unsplit_two_steps` | 6 x 5 | 6 | 2 | 60 s / 20 s | unsplit-explicit cold start, constant vertical mixing |
| `split_filter_btr_mode` | 6 x 6 | 7 | 2 | 120 s / 40 s | `config_filter_btr_mode`'s cold start; the clock across a leap day (2000-02-28 23:58 to 2000-02-29), days since a start in January |

The meshes are `woof.verify.mpas_ocean_hexmesh.hex_mesh`'s: hexagonal cells on
a doubly periodic torus, with a sixth culled as land the way MPAS culls
meshes. Their extents, masks, signs and kite indices are derived as v8.4.2's
init routines derive them, and their geometry is random.

## Grading

`tests/test_mpas_ocean_forward_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.forward_v842` (`cold_start`, `advance`), and every word
must be bit-identical. The restart test runs one step, writes a restart, reads
it into fresh pools and runs the rest; every word must again match the oracle.
