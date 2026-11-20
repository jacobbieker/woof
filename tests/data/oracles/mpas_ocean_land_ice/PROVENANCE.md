# MPAS-Ocean v8.4.2 land-ice fluxes and effective density oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece land_ice` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- `mpas_ocn_surface_land_ice_fluxes.F`, `mpas_ocn_effective_density_in_land_ice.F`
  and `mpas_ocn_equation_of_state.F` (its `ocn_freezing_temperature`, the
  land-ice cavity coefficients) are compiled verbatim.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object; `sqrt` compiles to the IEEE square-root instruction. libm: the
  host glibc 2.44.
- Driver: `run_land_ice.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_surface_land_ice_fluxes_init`, `ocn_surface_land_ice_fluxes_build_arrays`
  with the stream's dt, `ocn_surface_land_ice_fluxes_thick` on
  `surfaceThicknessFlux`, `ocn_surface_land_ice_fluxes_vel` on `surfaceStress`
  and `surfaceStressMagnitude`, `ocn_surface_land_ice_fluxes_tracers` on
  `activeTracersSurfaceFlux`, then `ocn_effective_density_in_land_ice_update`,
  and writes every field back.

## Cases

| case | nCells (solved) | what it covers |
|---|---|---|
| `standalone_isomip` | 40 (40) | 'standalone', ISOMIP |
| `standalone_jenkins` | 44 (44) | 'standalone', Jenkins; 3 tracers; the 0.001 interface-salinity floor |
| `standalone_jenkins_halo_configured` | 48 (31) | Jenkins with every coefficient configured, swapped var_array indices, a 1e6 Pa cavity reference pressure, 17 halo cells |
| `standalone_isomip_halo_configured` | 36 (25) | ISOMIP configured, swapped indices, 11 halo cells |
| `coupled` | 42 (42) | 'coupled': the coupler's fluxes kept, the effective density updated |
| `pressure_only` | 30 (30) | 'pressure_only': nothing written |

Three-fifths of the cells lie under land ice (fraction at least 0.5); an
eighth of the boundary layers are fresh (salinity from -0.5 to 5e-4), so the
Jenkins interface salinity reaches its floor. The neighbour lists hold a
tenth missing cells and the cell mask a fifth of zeros. The fields the
routines write start from noise.

## Grading

`tests/test_mpas_ocean_land_ice_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.land_ice_v842`, and every word must be bit-identical.
