# MPAS-Ocean v8.4.2 ocn_tend_tracer oracle

Written by `tools/mpas_ocean_oracle/build.sh` and
`make_fixture.py --piece tend_tracer` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-13.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_ocean_oracle/SOURCES.sha256`.
- `mpas_ocn_tendency.F` is compiled verbatim in the build's `eos/` directory
  against the real modules it reaches:
  - tracer advection (with the vertical stencils cut out of the operators'
    helpers), horizontal mixing, surface flux, short-wave absorption (Jerlov)
    and the non-local flux;
  - the bulk forcing, recompiled there against the real equation of state;
  - `ocn_diagnostics`, cut out of `mpas_ocn_diagnostics.F` with
    `ocn_compute_KPP_input_fields`.

  `stubs_unported.F90` stands in for what no oracle compiles: land-ice
  fluxes and frazil, which return when switched off as the real routines do,
  restoring, decay, ideal age, TTD, the ecosystem groups, vertical mixing and
  the ohlmann00 short-wave absorption.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, `-DSINGLE_PRECISION` for
  `oracle-r4.npz`, RKIND 8 for `oracle-r8.npz`; no FMA instruction in any
  object. Jerlov's `exp` is the host glibc's `expf`/`exp`.
- Driver: `run_tend_tracer.F90` (stream: `oracle_stream.F90`). It calls the
  `_init` routines `ocn_forward_mode_init` calls for these modules, then
  `ocn_tend_tracer(..., dt, activeTracersOnlyIn=.true., timeLevelIn=1)`. The
  stubs iterate the tracers pool in the order its members were put; it holds
  `activeTracers` alone.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `advection_only` | 40 | 16 | advection, the surface flux and its remainder |
| `bulk_jerlov_del2_budgets` | 48 | 20 | bulk forcing with the bulk thickness flux and freezing temperature, Jerlov (type 2, buoyancy depth 12 m), del2, the budgets |
| `bulk_salt_limit_del4` | 44 | 24 | bulk forcing with the sea-ice salt flux limited, Jerlov (type 5, buoyancy depth 400 m: the search runs past the column), del4 |
| `sflux_disabled_runoff` | 32 | 8 | `config_disable_tr_sflux`, 2nd-order vertical advection |
| `active_tracers_off` | 24 | 6 | `config_use_activeTracers` off |
| `all_tend_disabled` | 24 | 6 | `config_disable_tr_all_tend` |

## Grading

`tests/test_mpas_ocean_tend_tracer_parity.py` and `woof hex ocean verify` run
`woof.hex.mpas_ocean.tend_tracer_v842`, and every word must be bit-identical.
Terms orders of magnitude below the sums they join (the frozen-flux heat,
the fresh-water freezing heat, a short-wave increment against the advective
tendency) cannot show a rounding change in those sums. The mutation tests
target the terms that do.
