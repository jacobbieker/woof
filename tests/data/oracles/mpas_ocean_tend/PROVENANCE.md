# MPAS-Ocean v8.4.2 tendency-term oracle

Written by `tools/mpas_ocean_oracle/build.sh` and `make_fixture.py` on pb
(Linux 6.12 x86_64, glibc 2.44), 2026-11-07.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`, which covers
  `src/core_ocean/shared/mpas_ocn_{constants,vel_coriolis,vel_vadv,thick_hadv,thick_vadv}.F`
  and `src/framework/mpas_kind_types.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Reference of record: `-O0 -ffp-contract=off`, built twice:
  - `-DSINGLE_PRECISION` (RKIND = 4, MPAS's default build) gives `oracle-r4.npz`;
  - RKIND = 8, the precision MPAS-Ocean is run at, gives `oracle-r8.npz`.

  build.sh fails if an ocean object contains an FMA instruction. No libm call
  is made.
- Driver: `run_oracle.F90`.
  1. It puts the case's fields in the mesh pool and the configs in `ocnConfigs`:
     the four `config_disable_*` switches false, `config_time_integrator`
     'RK4' or 'split_explicit', and `config_vert_coord_movement`
     'uniform_stretching'.
  2. It calls the four `_init` routines.
  3. It calls each `_tend` routine into its own tendency array, which starts
     from the case's sentinel words.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<field>` for `tendVelCoriolis`, `tendVelVadv`, `tendThickHadv`, `tendThickVadv` |
| `inputs.sha256` | the input-stream digest of each case at each precision (`woof.verify.mpas_ocean_tend_oracle.input_bytes`) |

The inputs are regenerated, not committed. The method is the same as the
sea-ice fixture's: legacy `RandomState`, `randint` and `random_sample` only.
Reals are drawn in float64 and rounded to float32 for r4.

## Coverage

| case | nCells | nEdges | nVertLevels | integrator |
|---|---|---|---|---|
| `split` | 64 | 192 | 24 | split_explicit (RK4On 0) |
| `rk4_deep` | 40 | 120 | 60 | RK4 (RK4On 1) |

Both cases are random-connectivity meshes with the magnitudes of a 10-30 km
ocean mesh. Each contains:

- edges with `maxLevelEdgeTop` 0;
- `maxLevelEdgeBot` below `maxLevelEdgeTop`;
- short `maxLevelCell`;
- `edgeMask` zeros inside the active column;
- both edge signs;
- never-read padding.

## Grading

`tests/test_mpas_ocean_tend_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.tend_v842`, and every word must be bit-identical. On
2026-11-07 the result was 0 differing words in every field, case and precision
on CUDA (RTX A6000) and on Metal (Apple silicon, the mlx facade; r8 through
woof's soft-f64).
