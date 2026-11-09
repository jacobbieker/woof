# MALI v8.4.2 thickness and tracer advection oracle

Written by `tools/mali_oracle/build.sh` (`run_advection_r4`) and
`make_advection_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-09.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. It is the same
  12 files as the other MALI oracles (`tools/mali_oracle/SOURCES.sha256`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0. Reference of record:
  `-O0 -ffp-contract=off -DSINGLE_PRECISION` (RKIND = 4), and no object may
  contain an FMA instruction. The routines make no libm call.
- Driver: `run_advection.F90` builds a one-block domain and calls:
  1. `li_thermal_init(domain, err)` as on a restart (`config_do_restart`, the
     air temperature and heat flux from file). It then only copies
     `config_ice_density`/`config_ocean_density` into the module and builds
     its vertical-grid terms.
  2. `li_advection_thickness_tracers(dt, ...)` with `advectTracersIn` true or
     false, as `li_time_integration_fe` calls it.
- The stand-ins (`stubs.F90`) resolve the five scratch fields the routine
  allocates (`workLevelCell`, `workTracerCell[2]`, `workTracerLevelCell[2]`)
  to arrays the driver fills with the case's random words. The model leaves
  them uninitialised, so no output may depend on them.

## Coverage

| case | nCells | levels | velocity solver (masks) | thermal solver | tracers | thickness advection |
|---|---|---|---|---|---|---|
| `fo_temperature` | 120 | 10 | sia | temperature | yes | fo |
| `fo_enthalpy` | 120 | 10 | sia | enthalpy | yes | fo |
| `fo_thickness_only` | 96 | 6 | none | temperature | no | fo |
| `fo_no_thermal` | 80 | 8 | sia | none | yes | fo |
| `none` | 64 | 5 | sia | temperature | yes | none |

The time step is 30 days. Each case covers:

- halo cells past `nCellsSolve`;
- ice-free cells;
- cells thinner than a metre, where ablation melts whole layers;
- surface and basal mass balance of both signs;
- layer velocities of +0 and -0;
- upwind layer thicknesses that empty layers;
- air temperatures on both sides of 273.15 K.

## Files

| file | content |
|---|---|
| `oracle-r4.npz` | per case, `<case>/<field>`: `errInit`, `errAdvect`, `logErrors` (all 0), `cellMask`, `vertexMask`, `edgeMask`, `thickness`, `sfcMassBal`, `basalMassBal`, `dynamicThickening`, `layerThickness`, `temperature`, `waterfrac`, `enthalpy` |
| `inputs.sha256` | the sha256 of each case's input stream (`woof.verify.mali_advection_oracle.input_bytes`) |

## Grading

`tests/test_mali_advection_parity.py` and `woof hex ice verify` run
`woof.hex.mpas_ice.advection_v842` on each case. Every word must be
bit-identical.
