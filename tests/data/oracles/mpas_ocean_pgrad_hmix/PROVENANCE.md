# MPAS-Ocean v8.4.2 pressure-gradient and hmix oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece pgrad_hmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-08.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_{constants,vel_pressure_grad,vel_hmix,vel_hmix_del2,vel_hmix_leith,vel_hmix_del4}.F`
  (and the tendency oracle's files), `src/framework/mpas_constants.F` and
  `src/framework/mpas_kind_types.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Reference of record: `-O0 -ffp-contract=off`, built twice:
  - `-DSINGLE_PRECISION` (RKIND = 4, MPAS's default build) gives `oracle-r4.npz`;
  - RKIND = 8 gives `oracle-r8.npz`.

  build.sh fails if an ocean object contains an FMA instruction. No libm call
  is made. Leith's `x**3` compiles to libgcc's `__powisf2`/`__powidf2(x, 3)`,
  which multiplies `x*(x*x)`. gfortran folds `sqrt(3.0_RKIND)` to the correctly
  rounded constant.
- Driver: `run_pgrad_hmix.F90`.
  1. It puts the case's fields in the mesh pool, the del4 scratch fields in a
     scratch pool and the configs in `ocnConfigs`. Every halo extent
     (`nCellsArray`, `nEdgesArray`, `nVerticesArray`) is the owned count: one
     block, nothing exchanged.
  2. It calls `ocn_constants_init` (rho_sw = `config_density0`),
     `ocn_vel_pressure_grad_init` and `ocn_vel_hmix_init`.
  3. It calls `ocn_vel_pressure_grad_tend` into `tendPgrad`, passing the
     in-situ expansion coefficients whatever the type, as `ocn_tend_vel`
     does.
  4. It calls `ocn_vel_hmix_tend` into `tendHmix` and `viscosity`. The
     tensor forms are off.
  `tendPgrad`, `tendHmix` and `viscosity` start from the case's sentinel
  words.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<field>` for `tendPgrad`, `tendHmix`, `viscosity` |
| `inputs.sha256` | the input-stream digest of each case at each precision (`woof.verify.mpas_ocean_pgrad_hmix_oracle.input_bytes`) |

The inputs are regenerated, not committed. They use legacy `RandomState`,
`randint` and `random_sample` only. Reals are drawn in float64 and rounded to
float32 for r4.

## Coverage

| case | nCells | nVertLevels | pressure gradient | hmix |
|---|---|---|---|---|
| `zmid_del2` | 40 | 20 | pressure_and_zmid | del2 |
| `mont_leith` | 36 | 24 | MontgomeryPotential | Leith |
| `montd_del4` | 40 | 18 | MontgomeryPotential_and_density | del4, div factor 0.3 |
| `jdens_all` | 48 | 30 | Jacobian_from_density, weight 0.5, rho_sw 1027.5 | del2 + Leith + del4 |
| `jts_gamma1` | 40 | 26 | Jacobian_from_TS, weight 1.0, T and S at tracer 2, 3 | del2 + del4 |
| `jts_gamma0_hmix_off` | 32 | 40 | Jacobian_from_TS, weight 0.0, T and S at tracer 3, 1 | disabled (viscosity untouched) |
| `pgrad_off_leith` | 32 | 16 | disabled | Leith; del4 asked for with `config_mom_del4` 0 (stays off) |

Each case is a random-connectivity mesh (nEdges = 3 nCells, nVertices =
2 nCells, vertexDegree 3) with the magnitudes of a 10-30 km ocean mesh. Each
contains:

- land edges (`maxLevelEdgeTop` 0), half of them with `edgeMask(1)` 1, so the
  Jacobian branches' unconditional layer-1 term shows;
- `edgeMask` zeros inside active columns;
- short `maxLevelCell` and `maxLevelVertexTop`;
- both edge signs on cells and vertices;
- `dvEdge` below `0.25*dcEdge` (del4's `max`);
- Leith viscosities on both sides of `config_Leith_visc2_max`.

## Grading

`tests/test_mpas_ocean_pgrad_hmix_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.pgrad_hmix_v842`, and every word must be bit-identical.
On 2026-11-08 the result was 0 differing words in every field, case and
precision on CUDA (RTX A6000).
