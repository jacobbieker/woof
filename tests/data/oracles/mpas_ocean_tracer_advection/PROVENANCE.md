# MPAS-Ocean v8.4.2 tracer advection oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece tracer_advection` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`. That covers
  `src/core_ocean/shared/mpas_ocn_tracer_advection{,_std,_mono}.F`,
  `src/operators/mpas_tracer_advection_helpers.F`, `mpas_ocn_constants.F` and
  `src/framework/mpas_kind_types.F`.
- The three advection modules are compiled verbatim. `mpas_tracer_advection_vflux4`
  and `_vflux3` are cut verbatim (awk ranges) out of
  `mpas_tracer_advection_helpers.F` into `tracer_advection_helpers_head.F90`'s
  module. The module's other routines build the advection coefficients at
  init from the mesh, and need `mpas_sort`, `mpas_hash` and the geometry
  utilities.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. No libm call.
- Driver: `run_tracer_advection.F90` (stream: `oracle_stream.F90`). It calls
  `ocn_constants_init`, `ocn_tracer_advection_init` (orders,
  `coef_3rd_order`, `config_monotonic`) and
  `ocn_tracer_advection_tend(activeTracers, normalThicknessFlux,
  vertAleTransportTop, layerThickness, dt, mesh, scratch, diagnostics,
  activeTracersTend, 'activeTracers')`, and writes every field back.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | `<case>/<pool>/<field>` for every real and integer field, after the call |
| `inputs.sha256` | the input-stream digest of each case at each precision |

## Cases

| case | nCells | nVertLevels | orders (horizontal/vertical) | other |
|---|---|---|---|---|
| `defaults_3rd_order` | 48 | 20 | 3/3, coef 0.25 | the Registry defaults |
| `vert4_horiz4_budgets` | 40 | 24 | 4/4 | the active-tracer budgets, the monotonicity check |
| `vert2_horiz2` | 44 | 16 | 2/2 | |
| `three_levels_coef1` | 36 | 3 | 3/3, coef 1.0 | the fewest levels the scheme stays inside |
| `disabled` | 24 | 8 | | `config_disable_tr_adv` |

The meshes are random connectivity on one block (every halo extent the
block's count). Random connectivity has no halo locality, so nested extents
would read words outside the computed halos. `maxLevelCell` is at least 2 and
`nVertLevels` at least 3: below that v8.4.2 reads outside the column, and the
port refuses it. Boundary edges point at the missing cell and have
`maxLevelEdgeTop` 0. Some fluxes are signed zeros.

## Grading

`tests/test_mpas_ocean_tracer_advection_parity.py` and `woof hex ocean verify`
run `woof.hex.mpas_ocean.tracer_advection_v842`, and every word must be
bit-identical. The scheme's MAX/MIN ties and SIGN of a signed-zero flux only
decide the sign of a zero term added to others, so no output depends on them.
