# MPAS-Ocean v8.4.2 forward mode with CVMix oracle (RKIND 8)

Written by `tools/mpas_ocean_oracle/build.sh` (with `CVMIX_SOURCE`) and
`make_fixture.py --piece forward_cvmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-22.

- Sources and compiler as `tests/data/oracles/mpas_ocean_split_step_cvmix/PROVENANCE.md`.
- The binary, `run_forward_cvmix_r8`, is the forward oracle's (`run_forward.F90`;
  `tests/data/oracles/mpas_ocean_forward/PROVENANCE.md`) with CVMix and
  `mpas_ocn_vmix_cvmix.F` linked in place of `stubs_vmix.F90`. The cold start
  sets the boundary layer depth to half the top layer (as
  `ocn_init_routines_block` does) in the GM and CVMix cases, whose streams
  carry it.

## Cases

`woof.verify.mpas_ocean_hexmesh.hex_mesh` tori (a sixth culled as land) with a
hexagon's geometry at a uniform 20 km spacing, or a spacing varying from 20 km
to 80 km by graph distance from the first ocean cell (`_resolution`
'variable'; edge, cell, triangle and kite areas and mesh scaling following it):

| case | torus | nVertLevels | steps | what it covers |
|---|---|---|---|---|
| `kpp_forward_uniform` | 7x6 | 12 | 3 | KPP, uniform mesh |
| `kpp_forward_variable_jm_bulk` | 8x6 | 14 | 3 | variable resolution, JM, bulk forcing, Jerlov, budgets |
| `kpp_forward_variable_implicit_ice` | 7x6 | 12 | 3 | variable resolution, non-local flux in the implicit solve, MatchBoth, sea ice |
| `kpp_forward_nested_halos_variable` | 12x6 | 12 | 3 | one block of a decomposition (three owned columns, three halo layers): KPP's BLD smoothing over the halo |
| `cvmix_unsplit_uniform` | 6x5 | 10 | 2 | 'unsplit_explicit', background, convection, KPP-scheme shear, no KPP |
| `kpp_forward_wind_uniform` | 7x6 | 12 | 3 | the bulk wind stress (behind `WOOF_GLIBC239_LIBM64=1` at RKIND 8), uniform |
| `kpp_forward_wind_variable_halos` | 12x6 | 12 | 3 | the bulk wind stress, variable resolution, nested halos, budgets |

## Grading

`tests/test_mpas_ocean_forward_cvmix_parity.py` runs
`woof.hex.mpas_ocean.forward_v842`, and every word must be bit-identical.
