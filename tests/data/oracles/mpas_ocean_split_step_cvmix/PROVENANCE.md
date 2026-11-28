# MPAS-Ocean v8.4.2 split-explicit step with CVMix oracle (RKIND 8)

Written by `tools/mpas_ocean_oracle/build.sh` (with `CVMIX_SOURCE`) and
`make_fixture.py --piece split_step_cvmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-22.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified
  (`tools/mpas_ocean_oracle/SOURCES.sha256`).
- CVMix source: github.com/CVMix/CVMix-src tag v0.84-beta, commit
  `d83f582714e7f0f98d20efd8fac8fab01fa3bfe6`, the ten `src/shared` modules
  byte-unmodified (`tools/mpas_ocean_oracle/CVMIX_SOURCES.sha256`). Nothing
  from CVMix is committed; licence history in
  `tests/data/oracles/mpas_ocean_cvmix/PROVENANCE.md`.
- The binary, `run_split_step_cvmix_r8`, is the split step oracle's
  (`run_split_step.F90`; `tests/data/oracles/mpas_ocean_split_step/PROVENANCE.md`)
  with every object the same except the vertical mixing: CVMix and
  `mpas_ocn_vmix_cvmix.F` (compiled in the build's `cvmix/` directory) in
  place of `eos/`'s vertical mixing and `stubs_vmix.F90`. The driver now also
  calls `ocn_tracer_nonlocalflux_init`; the same build reproduces
  `mpas_ocean_split_step`, `mpas_ocean_split_step_gm` and the forward
  fixtures byte for byte.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, RKIND 8 (CVMix does not compile at
  RKIND 4). No object contains an FMA instruction.

## Cases

The split step oracle's random single-block meshes with `config_use_cvmix` on
and CVMix's Registry defaults, a previous step's boundary layer and
`vertNonLocalFlux` (Registry shape: one member) as input:

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `kpp_split_defaults` | 40 | 12 | background, PP shear, convection, KPP; the non-local flux in the tracer tendency; no wind (zero friction velocity) |
| `kpp_split_jm_bulk_implicit_budgets` | 44 | 14 | JM, bulk forcing with the bulk thickness flux, Jerlov, the budgets, three iterations; the non-local flux in the implicit solve |
| `kpp_split_match_both_ice_attenuated` | 40 | 12 | MatchBoth, cubic interpolation, sea ice (minimum OBL 20 m), attenuation coefficients 10 and 4 m |
| `cvmix_unsplit_no_kpp` | 32 | 10 | 'unsplit_explicit' with background, shear and convection, no KPP |
| `kpp_split_wind_match_both` | 40 | 12 | the bulk wind stress (KPP's friction velocity), MatchBoth, budgets; at RKIND 8 behind `WOOF_GLIBC239_LIBM64=1` |

## Grading

`tests/test_mpas_ocean_split_step_cvmix_parity.py` runs
`woof.hex.mpas_ocean.split_step_v842`, and every word must be bit-identical
(the wind case where the opt-in is on).
