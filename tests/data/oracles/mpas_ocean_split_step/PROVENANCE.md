# MPAS-Ocean v8.4.2 split-explicit step oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece split_step` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- `ocn_time_integrator_split` is cut verbatim out of
  `mpas_ocn_time_integration_split.F` (`split_step_cut_head.F90`).
- Everything the step calls is real v8.4.2 code:
  - compiled verbatim: the tendency module and every term it reaches, the
    equation of state, the ALE thickness, tracer advection and mixing,
    implicit vertical mixing (`stubs_vmix.F90` stands in for CVMix),
    `mpas_ocn_time_average_coupled.F` and
    `mpas_ocn_effective_density_in_land_ice.F`;
  - cut verbatim: the diagnostics module's routines (including `ocn_fuperp`)
    and `mpas_reconstruct_1d`/`_2d`.
  - GM is not driven: `stubs_split.F90` stops by name if it is reached.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. libm: the host glibc 2.44.
  - The sphere's `cos`/`sin` take latitudes in [-1.4, 1.4] and longitudes in
    [0, 6.28]. There glibc 2.44's binary64 words equal glibc 2.39's, which
    the opt-in transcription reproduces; the two differ only for arguments of
    10^8 and above.
- Driver: `run_split_step.F90` (stream: `oracle_stream.F90`; a field's time
  level 2 is the pool entry `NAME@2`). It runs every module's `_init`, computes
  the starting state's diagnostics (`ocn_diagnostic_solve` at time level 1,
  as MPAS's init does), sets `nBtrSubcycles` from the config of that name,
  takes one step, and writes every field back.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `split_defaults` | 40 | 8 | split-explicit with v8.4.2's iteration defaults, del2 momentum mixing, the linear EOS |
| `split_jm_rich_bulk_budgets` | 44 | 10 | three outer iterations, JM EOS with the Jacobian_from_TS pressure gradient, Richardson mixing, bulk forcing with Jerlov short wave, tracer del2, the budgets per unit thickness, the SSH corrector, near-fresh columns (the salinity floor acts) |
| `unsplit_const_vmix_prescribed` | 32 | 6 | unsplit-explicit, constant vertical mixing, no velocity correction, the prescribed thickness |
| `split_sphere` | 36 | 8 | the sphere's reconstruction (RKIND 8 under the glibc opt-in), APVM |

The meshes are random connectivity on one block with every halo extent the
block's count. There is no flow through land edges, and `maxLevelCell` is at
least 2.

## Grading

`tests/test_mpas_ocean_split_step_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.split_step_v842` after the same starting diagnostics, and
every word must be bit-identical.
