# MPAS-Ocean v8.4.2 frazil ice formation oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece frazil` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- `mpas_ocn_frazil_forcing.F` and `mpas_ocn_equation_of_state.F` (its
  `ocn_freezing_temperature`) are compiled verbatim.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object; no libm call.
- Driver: `run_frazil.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_frazil_forcing_init`, `ocn_frazil_forcing_build_arrays(domain, mesh,
  forcing, diagnostics, state, err)` with dt from the clock (`config_start_time`,
  `config_dt`), `ocn_frazil_forcing_layer_thickness` on `tend % layerThickness`
  and `ocn_frazil_forcing_tracers` on `activeTracersTend`, in the order the
  forward mode and the tendencies call them, and writes every field back.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `open_ocean` | 40 | 12 | the Registry defaults, open ocean only (no land-ice mask) |
| `land_ice_cavity` | 44 | 10 | a land-ice mask: the cavity's freezing coefficients, the land-ice frazil mass |
| `land_ice_only` | 36 | 9 | `config_frazil_in_open_ocean` off; tracer indices 2, 3 of 3 |
| `open_only_no_pressure_halo` | 48 | 14 | `config_frazil_under_land_ice` and the surface pressure off; 18 halo cells beyond `nCellsArray(2)` |
| `configured_limits` | 42 | 16 | every coefficient configured: a binding 0.002 thickness limit, a 40 m depth cutoff, a -1.2 C maximum freezing temperature; salinity index first |

Water is drawn from -2.3 to 0.6 C (a sixth of the columns warm), so frazil
forms and melts back in a column. The level-2 accumulated fields, the
tendencies and the surface pressure start from noise, so a word the routines
must leave alone is checked.

## Grading

`tests/test_mpas_ocean_frazil_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.frazil_v842`, and every word must be bit-identical.
