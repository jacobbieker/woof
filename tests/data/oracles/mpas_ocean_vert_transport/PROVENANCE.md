# MPAS-Ocean v8.4.2 vertical transport oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece vert_transport` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- `ocn_vert_transport_velocity_top` is cut verbatim out of
  `mpas_ocn_diagnostics.F` (`diagnostics_cut_head.F90`), and
  `mpas_ocn_thick_ale.F` is compiled verbatim.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object; no libm call.
- Driver: `run_vert_transport.F90` (stream: `oracle_stream.F90`; the
  `thicknessFilterActive` package is off). It calls
  `ocn_vert_transport_velocity_top(mesh, verticalMesh, scratch, layerThickness,
  layerThicknessEdge, normalVelocity, ssh, dt, vertAleTransportTop, err)` as the
  split-explicit step does, and writes every field back.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `zstar` | 48 | 20 | z-star over the whole column |
| `zstar_min_max` | 44 | 24 | thin resting layers and large SSH swings: the min/max thickness limits act both ways |
| `fixed_bottom_weights` | 40 | 16 | `vertCoordMovementWeights` zero below mid-depth |
| `impermeable_interfaces` | 24 | 8 | the transport zeroed |

The meshes are random connectivity on one block. `maxLevelCell` is at least 1.

## Grading

`tests/test_mpas_ocean_vert_transport_parity.py` and `woof hex ocean verify`
run `woof.hex.mpas_ocean.vert_transport_v842`, and every word must be
bit-identical.
