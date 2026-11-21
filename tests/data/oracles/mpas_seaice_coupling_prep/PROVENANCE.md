# MPAS-Seaice v8.4.2 coupling-preparation oracle

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_coupling_prep_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256`.
- Cut verbatim out of `mpas_seaice_column.F`: `seaice_column_coupling_prep`
  (lines 3634-4041) and `seaice_column_scale_fluxes` (lines 4055-4306), into
  module `seaice_column_coupling` behind `coupling_cut_head.F90`. That head
  holds a `ciceTracerObject` stand-in and a `seaice_column_ocean_mixed_layer`
  stand-in that stops 6. The cases run with `config_use_ocean_mixed_layer`
  off, so the stand-in is never called; the port refuses the option by name.
- Compiled verbatim: `seaice_constants` and the colpkg constants.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`, `-DSINGLE_PRECISION` for `oracle-r4.npz`, RKIND 8 for
  `oracle-r8.npz`. No FMA instruction in any object. No libm call.
- Driver: `run_coupling_prep.F90`. It builds one block whose 12 structs are
  filled from the stream, calls `seaice_column_coupling_prep(domain)` and
  writes every real field back.
- `inputs.sha256` pins the stream bytes of every case at both precisions.

## Cases

| case | nCells (nCellsSolve) | categories | dt | what it covers |
|---|---|---|---|---|
| `five_categories` | 40 (40) | 5 | 3600 s | ice and open water, sun up and down (coszen exactly 0, exactly seaicePuny and between them included), no pond feedback |
| `pond_feedback_halo` | 48 (33) | 5 | 1800 s | the pond freshwater feedback; halo cells beyond nCellsSolve left alone |
| `one_category_low_sun` | 30 (30) | 1 | 600 s | one category, the sun mostly down, the pond feedback |
| `bgc_brine_aggregate_past_solve` | 48 (33; aggregate 40) | 5 | 1800 s | the column biogeochemistry with the brine height alone: the 14 ocean biogeochemistry fluxes zeroed on the owned cells, scaled on cells 33-39 |
| `bgc_brine_aggregate_short_of_solve` | 30 (30; aggregate 24) | 3 | 600 s | the same, with the pond feedback; cells 24-29 zeroed and not scaled |

About 25% of the cells are open water and 10% have `solarZenithAngleCosine`
exactly 0, 5% exactly seaicePuny and 5% half of it. In the first three cases the column biogeochemistry is off, so
its fields are absent (null pointers in the driver). The two `bgc_brine_*` cases (added 2026-11-21; the first
three cases' bytes and words are unchanged) turn on `config_use_column_biogeochemistry` and `config_use_brine`,
carry the biogeochemistry's fields at the Registry's extents (`nZBGCTracers` 29, the size of coupling_prep's
`oceanBioFluxesAll`, of which it reads 29 words) with nonzero noise, and give the aggregate pool its own
`nCellsSolve`, past and short of the tracers pool's: coupling_prep runs over the one, scale_fluxes over the
other. MPAS gives both pools the block's `nCellsSolve`; the cases hold the two routines to their own loops.
