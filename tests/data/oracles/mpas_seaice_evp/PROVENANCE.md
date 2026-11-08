# MPAS-Seaice v8.4.2 variational EVP internal-stress oracle

Written by `tools/mpas_seaice_oracle/build.sh` and `make_fixture.py` on pb
(Linux 6.12 x86_64, glibc 2.44): `oracle-r4.npz` on 2026-11-07, and
`oracle-r8.npz` on 2026-11-08. The 2026-11-08 run rewrote `oracle-r4.npz` with
the same words.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified (build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256` and refuses a differing file):
  - `src/core_seaice/shared/mpas_seaice_velocity_solver_constitutive_relation.F`
    sha256 `2de0313a166a1124c46ea49e1993551cc6d04bbec9fc8e5c73b361596c7fde00`
  - `src/core_seaice/shared/mpas_seaice_velocity_solver_variational.F`
    sha256 `ed7f647e90c22a126431991ebbfd44c4efbc9be16bf6f473a67512357f7401e6`
  - `src/framework/mpas_kind_types.F`
    sha256 `13db2ffce7d70f0c725913eb8c58c2e0b3fbdffb7bfe0c2ac77798e554127c51`
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Builds of record: `-O0 -ffp-contract=off`, at `-DSINGLE_PRECISION`
  (RKIND = 4, MPAS's default precision) and without it (RKIND = 8; the column
  package is double, so production MPAS-Seaice is built double). build.sh fails if either seaice object contains
  an FMA instruction. Only scalar SSE (`addss`/`sd` ... `sqrtss`/`sd`, `maxss`/`sd`,
  `minss`/`sd`) arithmetic is used; no libm call is made.
- Driver: `run_oracle.F90` puts the case's fields in the pools the routines
  read. It calls `seaice_init_evp(domain)` once, then
  `seaice_internal_stress_variational(domain)` nIter times, then
  `seaice_final_divergence_shear_variational(block)`, which is the order the
  model calls them in. The framework stand-ins (`stubs.F90`) only look fields
  up by name; they do no arithmetic.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | every output word of every case at that precision, `<case>/<field>`: `numericalInertiaCoefficient`, `strain11/22/12`, `stress11/22/12`, `replacementPressure`, `stressDivergenceU/V`, `divergence`, `shear`, `ridgeConvergence`, `ridgeShear` |
| `inputs.sha256` | the sha256 of each case's input stream at each precision (`<case>/<precision>`), as `woof.verify.mpas_seaice_evp_oracle.input_bytes` writes it |

The inputs are not committed. `woof.verify.mpas_seaice_evp_oracle.cases()`
regenerates them from numpy's legacy `RandomState`, using only `randint` and
`random_sample` with float64 scaling, so the bytes do not depend on the
platform's libm. The r8 run takes the float64 draws as they are; the r4 run
rounds them once to float32. The test refuses a regenerated case whose digest differs.

## Coverage

There are three cases on random-connectivity meshes, with magnitudes of a
10 km MPAS-Seaice mesh:

| case | nCells | maxEdges | revised EVP | column package | nIter | dynamics dt |
|---|---|---|---|---|---|---|
| `evp` | 160 | 7 | no | yes | 12 | 3600 s |
| `revised` | 160 | 7 | yes | yes | 12 | 1800 s |
| `evp_nocolumn` | 96 | 6 | no | no | 1 | 900 s |

The elastic dt is the dynamics dt over 120, the model default
(`config_elastic_subcycle_number`). Each case contains:

- pentagons, hexagons and, for maxEdges 7, heptagons;
- unsolved cells and vertices (solveStress/solveVelocity 0), whose outputs
  keep their prior sentinel words;
- cells with every vertex at rest (Delta below `puny`);
- zero ice strength;
- padding past `nEdgesOnCell` that holds an out-of-range vertex index.

## Grading

`tests/test_mpas_seaice_evp_parity.py` and `woof hex seaice verify` run
`woof.hex.mpas_seaice.evp_v842` on each case. Every word must be
bit-identical. Results: 0 differing words in every field of every case on
CUDA (RTX A6000) and on Metal (Apple silicon, the mlx facade), at RKIND 4
(2026-11-07) and RKIND 8 (2026-11-08; on Metal through soft-f64).
