# MPAS-Ocean v8.4.2 Gent-McWilliams bolus velocity oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece gm` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_gm.F` and `mpas_ocn_constants.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. At RKIND 4, `mpas_ocn_gm.F` alone also takes `-fno-range-check`:
  it assigns `huge(0D0)` to its RKIND scratch fields, which gfortran otherwise
  rejects as an overflow (with the flag the fill is +Inf). The only libm call is
  `sqrt`, which is correctly rounded.
- Driver: `run_gm.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_constants_init` and `ocn_gm_init`, as `ocn_forward_mode_init` does,
  then `ocn_gm_compute_Bolus_velocity`, and writes every field back. The
  scratch fields start at `huge(1.0_RKIND)`, as the stub scratch allocation
  leaves them.

## Cases

| case | nCells | nVertLevels | what it exercises |
|---|---|---|---|
| `default` | 34 | 14 | GM kappa 600, max slope 0.01: the max-slope taper |
| `tapering` | 32 | 18 | surface-layer tapering (extent 30 m, `boundaryLayerDepth`) and bottom-layer tapering (60 m), max slope 0.3 |
| `steep_no_k33` | 29 | 10 | `config_disable_redi_k33`, wave speed 1.2, kappa 1800, steep horizontal gradients and unstable interfaces: slopes clipped to +-1 |
| `shallow` | 25 | 6 | columns of one level, edges too shallow for the stream function's solve, unstable interfaces |

The meshes are `hex_mesh` tori with a sixth of the cells land, on one block
(every halo extent the block's count). Boundary edges point at the missing
cell and have `maxLevelEdgeTop` 0. Every output starts from noise, so the
words the routine does not write are graded too.

## Grading

`tests/test_mpas_ocean_gm_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.gm_v842`, and every word must be bit-identical.
