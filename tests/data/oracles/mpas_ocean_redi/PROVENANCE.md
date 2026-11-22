# MPAS-Ocean v8.4.2 Redi isoneutral mixing oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece redi` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_tracer_hmix{,_del2,_del4,_redi}.F` and
  `mpas_ocn_constants.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. No libm call.
- Driver: `run_tracer_hmix.F90` (the horizontal mixing's; its stream is
  self-describing). It calls `ocn_constants_init`, `ocn_tracer_hmix_init` and
  `ocn_tracer_hmix_redi_init`, then `ocn_tracer_hmix_tend` with
  `config_use_standardGM` on. GM's slopes and tapering carry their Registry's
  `nVertLevels+1` levels, and Redi's scratch fields are declared (they start
  at `huge(1.0_RKIND)`).

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `redi` | 31 | 12 | Redi, kappa 400 |
| `redi_del2` | 32 | 16 | del2 then Redi, kappa 900 |
| `term1_off` | 27 | 10 | `config_disable_redi_horizontal_term1` |
| `term2_off` | 26 | 10 | `config_disable_redi_horizontal_term2` |
| `term3_off` | 26 | 8 | `config_disable_redi_horizontal_term3`, one-level columns |
| `nested` | 54 | 9 | one block of a decomposition: `nCellsArray` 23/33/44/54, `nEdgesArray` 82/116/154/187 |
| `disabled` | 21 | 6 | `config_disable_tr_hmix` with GM on: nothing runs |

The meshes are `hex_mesh` tori with a sixth of the cells land. Boundary
edges point at the missing cell and have `maxLevelEdgeTop` 0. The slopes,
tapering and tendency start from noise.

## Grading

`tests/test_mpas_ocean_redi_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.tracer_hmix_v842`, and every word must be bit-identical.
