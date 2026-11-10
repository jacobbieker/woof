# MALI v8.4.2 thermal-initialisation oracle

Written by `tools/mali_oracle/build.sh` (`run_thermal_r4`) and
`make_thermal_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-08.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. It is the same
  12 files as the geometry oracle (`tools/mali_oracle/SOURCES.sha256`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0. Reference of record:
  `-O0 -ffp-contract=off -DSINGLE_PRECISION` (RKIND = 4), and no object may
  contain an FMA instruction. No libm call is made.
- Driver: `run_thermal.F90` builds a one-block domain with mesh, geometry and
  thermal subpools. It then calls:
  1. `li_thermal_init(domain, err)`, the model's init call, which also sets
     the module's `rhoi`;
  2. `li_compute_pressure_melting_point_fields` for all cells;
  3. `li_temperature_to_enthalpy` for each cell, on the case's temperature and
     water-fraction columns;
  4. `li_enthalpy_to_temperature` for each cell, on the case's enthalpy
     columns, with the surface and basal enthalpy.

## Coverage

| case | nCells | levels | temperature init | air temperature |
|---|---|---|---|---|
| `sfc_constant` | 90 | 10 | sfc_air_temperature | constant 268 K |
| `linear_lapse` | 70 | 12 | linear | lapse from 276 K, 0.0065 K/m |
| `file_lapse` | 40 | 8 | file | lapse from 276 K |

The basal heat flux is constant. Each case contains:

- unsolved cells past `nCellsSolve`, which keep their prior temperature;
- ice-free cells;
- air temperatures above freezing and above the pressure melting point;
- enthalpies either side of each layer's, the surface's and the bed's
  pressure-melting enthalpy.

## RKIND 8

`oracle-r8.npz` holds the same routines compiled without `-DSINGLE_PRECISION`
(RKIND = 8, the same flags otherwise, no FMA instruction), on pb, 2026-11-10.
Each case is drawn from the same seed in float64 instead of rounded to
float32. Its input digests are the `<case>/r8` lines of `inputs.sha256`.
The routines make no libm call.
The GPU port runs the `MPAS_RKIND8` unit and must match every word.

## Grading

`tests/test_mali_thermal_init_parity.py` and `woof hex ice verify` require
every word to be bit-identical. On 2026-11-08 the result was 0 differing words
on CUDA (RTX A6000) and on Metal (Apple silicon, the mlx facade).
