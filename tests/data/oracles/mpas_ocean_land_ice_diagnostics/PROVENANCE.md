# MPAS-Ocean v8.4.2 diagnostics oracle with the land-ice fluxes on

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece land_ice_diagnostics` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- The diagnostics oracle's binary (`run_diagnostics.F90`): `ocn_diagnostic_solve`
  and `ocn_compute_land_ice_flux_input_fields` cut verbatim out of
  `mpas_ocn_diagnostics.F`, with the real equation of state. The field table
  is `woof.verify.mpas_ocean_diagnostics_oracle`'s plus the land-ice flux
  package's diagnostics, `cellsOnCell`, `cellMask` and the boundary layer's two
  scratch fields (`woof.verify.mpas_ocean_land_ice_diagnostics_oracle`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`; no FMA instruction in any object. libm: the host glibc 2.44.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `standalone_jenkins_neighbours` | 48 | 12 | Jenkins, the boundary layer weighted with its neighbours (0.5; masked and missing neighbours), nested halos |
| `standalone_isomip_thick_layer` | 40 | 10 | ISOMIP (transfer velocities left alone), 'jm', a 150 m boundary layer, attenuation 25, swapped indices |
| `coupled_jenkins_ke_vertex_apvm` | 52 | 14 | 'coupled', the vertex kinetic energy and APVM, neighbour weight 1.25, frazil's surface pressure, nested halos |

The kinetic energy the loops do not reach is drawn non-negative (it feeds
the top drag's square root).

## Grading

`tests/test_mpas_ocean_land_ice_parity.py` runs
`woof.hex.mpas_ocean.diagnostics_v842`, and every word must be bit-identical.
