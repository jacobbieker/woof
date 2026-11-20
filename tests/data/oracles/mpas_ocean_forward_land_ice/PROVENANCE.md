# MPAS-Ocean v8.4.2 forward mode with the land-ice fluxes on

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece forward_land_ice` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- The forward oracle's binary (`run_forward.F90`), which calls
  `ocn_surface_land_ice_fluxes_init` and, per step before the frazil build,
  `ocn_surface_land_ice_fluxes_build_arrays` as `ocn_forward_mode_run` does.
  Every module is compiled verbatim, `mpas_ocn_surface_land_ice_fluxes.F`
  and `mpas_ocn_effective_density_in_land_ice.F` included, so the
  diagnostics compute the land-ice input fields, the tendencies add the
  land-ice forcing and the split step updates the effective density and the
  coupler's means in 'coupled'.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`; no FMA instruction in any object. libm: the host glibc 2.44.

## Cases

| case | torus | nVertLevels | steps | what it covers |
|---|---|---|---|---|
| `standalone_jenkins_three_steps` | 7 x 6 | 8 | 3 | 'standalone', Jenkins, neighbour weight 0.3 |
| `standalone_isomip_nested_halos` | 12 x 6 | 8 | 3 | 'standalone', ISOMIP, a 30 m boundary layer, nested halos |
| `coupled_jenkins_two_steps` | 6 x 5 | 6 | 2 | 'coupled': the effective density, the coupler's means, attenuation 25 |

About half the cells lie under land ice. The land-ice mask starts as integer
noise (the diagnostics rewrite it from the fraction).

## Grading

`tests/test_mpas_ocean_forward_land_ice_parity.py` runs
`woof.hex.mpas_ocean.forward_v842`, and every word must be bit-identical.
