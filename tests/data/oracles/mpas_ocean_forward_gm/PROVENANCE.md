# MPAS-Ocean v8.4.2 forward mode with GM and Redi oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece forward_gm` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- The binary is the forward oracle's (`run_forward.F90`; see
  `tests/data/oracles/mpas_ocean_forward/PROVENANCE.md`) with GM linked in
  (`tests/data/oracles/mpas_ocean_split_step_gm/PROVENANCE.md`). Under
  `config_use_standardGM` its cold start sets `boundaryLayerDepth` to half the
  top layer (`mpas_ocn_init_routines.F:615-619`; only these streams carry the
  field), computes GM's bolus velocity, adds it to the transport velocity and
  reconstructs GM's vectors (`mpas_ocn_init_routines.F:656-664,685-687`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`.

## Cases

The forward oracle's culled hexagonal tori, single blocks (v8.4.2's slopes read
GM's `huge` scratch fill on the outermost computed halo edges), with
`config_use_standardGM` on (GM kappa 600, Redi kappa 400):

| case | torus | nVertLevels | steps | what it covers |
|---|---|---|---|---|
| `gm_split_three_steps` | 7 x 6 | 8 | 3 | the Registry GM and Redi defaults, an odd number of shifts |
| `gm_split_jm_rich_tapering` | 8 x 6 | 10 | 2 | JM, Richardson mixing, the budgets, tracer del2; the surface tapering on the cold start's boundary layer depth, the bottom tapering |
| `gm_unsplit_no_k33` | 6 x 5 | 6 | 2 | 'unsplit_explicit', constant mixing, `config_disable_redi_k33` |

## Grading

`tests/test_mpas_ocean_forward_gm_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.forward_v842`, and every word must be bit-identical.
