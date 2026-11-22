# MPAS-Ocean v8.4.2 split-explicit step with GM and Redi oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece split_step_gm` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- The binary is the split step oracle's (`run_split_step.F90`; see
  `tests/data/oracles/mpas_ocean_split_step/PROVENANCE.md`), with GM
  (`mpas_ocn_gm.F` verbatim, at RKIND 4 with `-fno-range-check`) and
  `ocn_reconstruct_gm_vectors` (cut verbatim out of `mpas_ocn_diagnostics.F`)
  linked in place of the stand-ins that used to stop. The same build (GM
  linked in, off in their cases) reproduces `mpas_ocean_split_step`,
  `mpas_ocean_forward`, `mpas_ocean_forward_frazil` and
  `mpas_ocean_forward_land_ice` byte for byte.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`.

## Cases

The split step oracle's random single-block meshes with `config_use_standardGM`
on (GM kappa 600, Redi kappa 400), and the slopes, tapering and bolus velocity
of a previous step's GM as input:

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `gm_split_defaults` | 40 | 8 | the Registry GM and Redi defaults; stage 2's transport with a bolus velocity, Redi in the tracer tendency, `k33` in the implicit mixing |
| `gm_split_jm_rich_tapering_budgets` | 44 | 10 | JM, Richardson mixing, the budgets, tracer del2, three iterations; the surface and bottom tapering, a steep slope cap, a fast wave-speed truncation |
| `gm_unsplit_const_no_k33_term1_off` | 32 | 6 | 'unsplit_explicit', constant mixing, `config_disable_redi_k33`, Redi's first term off |

## Grading

`tests/test_mpas_ocean_split_step_gm_parity.py` and `woof hex ocean verify`
run `woof.hex.mpas_ocean.split_step_v842`, and every word must be bit-identical.
