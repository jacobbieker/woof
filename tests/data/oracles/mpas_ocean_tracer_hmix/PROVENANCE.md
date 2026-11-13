# MPAS-Ocean v8.4.2 horizontal tracer mixing oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece tracer_hmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_tracer_hmix{,_del2,_del4,_redi}.F` and
  `mpas_ocn_constants.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. No libm call.
- Driver: `run_tracer_hmix.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_constants_init`, `ocn_tracer_hmix_init` and `ocn_tracer_hmix_redi_init`,
  as `ocn_forward_mode_init` does, then `ocn_tracer_hmix_tend` on
  `activeTracers` into `activeTracersTend`, and writes every field back.

## Cases

| case | nCells | nVertLevels | mixing |
|---|---|---|---|
| `del2` | 48 | 20 | del2, 250 |
| `del4` | 44 | 16 | del4, 3e9 |
| `del2_del4` | 40 | 24 | both |
| `off_by_coefficient` | 32 | 8 | del4 asked for with coefficient 0, del2's coefficient set with its switch off: nothing runs |
| `disabled` | 24 | 8 | `config_disable_tr_hmix` |

The meshes are random connectivity on one block (every halo extent the
block's count). Boundary edges point at the missing cell and have
`maxLevelEdgeTop` 0, and `edgeMask` has zeros inside active columns.

## Grading

`tests/test_mpas_ocean_tracer_hmix_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.tracer_hmix_v842`, and every word must be bit-identical.
Redi (`config_use_standardGM`) is refused by name.
