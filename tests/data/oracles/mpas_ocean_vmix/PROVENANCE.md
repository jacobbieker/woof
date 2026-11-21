# MPAS-Ocean v8.4.2 implicit vertical mixing oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece vmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_vmix{,_coefs_const,_coefs_tanh,_coefs_rich,_coefs_redi}.F`
  and the equation of state.
- These are compiled verbatim in the build's `eos/` directory, with the real
  equation of state. `stubs_vmix.F90` stands in for CVMix (`mpas_ocn_vmix_cvmix.F`
  links the external CVMix library); like the real routines it does nothing
  when `config_use_cvmix` is off.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. JM's `exp` is the host glibc's.
- Driver: `run_vmix.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_constants_init`, `ocn_equation_of_state_init`, `ocn_vmix_init` and
  `ocn_vmix_implicit(timeLevelIn=1)`, then writes every field back. GM's `k33`
  is unassociated (its package is off) except in the `redi_*` cases, where
  `config_use_standardGM` is on and `ocn_tracer_vmix_coefs_redi` adds it.
  The `redi_*` cases were added on 2026-11-21 (same toolchain); the earlier
  cases' words did not change.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `const` | 40 | 16 | constant viscosity and diffusivity, the bottom drag |
| `rich_budgets` | 48 | 24 | Richardson viscosity and diffusivity (JM), the budget |
| `const_rich_linear_no_drag` | 44 | 20 | both, linear EOS, no bottom drag |
| `rich_diff_only_one_level_columns` | 36 | 12 | a third of the columns one level deep: one-level edges |
| `vel_disabled` | 24 | 8 | `config_disable_vel_vmix`, the budget |
| `tracers_disabled` | 24 | 8 | `config_disable_tr_vmix` |
| `redi_const` | 40 | 14 | Redi's vertical diffusion (`k33`, v8.4.2: without `config_Redi_kappa`) on the constant coefficients |
| `redi_rich_budgets` | 36 | 12 | Redi's `k33` on the Richardson coefficients, the budget |

The meshes are random connectivity on one block (every halo extent the
block's count). `maxLevelCell` is at least 1: for an empty column v8.4.2
writes `C(0)`, outside its array. Velocities are zero below `maxLevelEdgeTop`.

## Grading

`tests/test_mpas_ocean_vmix_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.vmix_v842`, and every word must be bit-identical.
The opt-in `woof_redi_k33 = 'kappa'` (not v8.4.2) is checked against
`config_Redi_kappa * k33` added to the diffusivity without it.
