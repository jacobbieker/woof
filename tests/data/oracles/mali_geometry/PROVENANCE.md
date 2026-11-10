# MALI v8.4.2 geometry-diagnostics oracle

Written by `tools/mali_oracle/build.sh` (`run_geometry_r4`) and
`make_geometry_fixture.py` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-08.

- MPAS source: MPAS-Model v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. These are the
  SIA oracle's 11 files plus `src/core_landice/mode_forward/mpas_li_advection.F`
  (`tools/mali_oracle/SOURCES.sha256`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0. Reference of record:
  `-O0 -ffp-contract=off -DSINGLE_PRECISION` (RKIND = 4), and no object may
  contain an FMA instruction. These routines call no libm function.
- Driver: `run_geometry.F90` calls, in the model's order:
  1. `li_calculate_mask(meshPool, velocityPool, geometryPool, err)`;
  2. `li_update_geometry(geometryPool)`;
  3. `li_calculate_layerThickness(meshPool, thickness, layerThickness)`;
  4. `li_layer_normal_velocity(meshPool, normalVelocity, layerNormalVelocity, minOfMaxAllowableDt, err)`.

  It writes both error flags, the error-log count, the CFL step and every
  output.

## RKIND 8

`oracle-r8.npz` holds the same routines compiled without `-DSINGLE_PRECISION`
(RKIND = 8, the same flags otherwise, no FMA instruction), on pb, 2026-11-10.
Each case is drawn from the same seed in float64 instead of rounded to
float32. Its input digests are the `<case>/r8` lines of `inputs.sha256`.
The routines make no libm call.
The GPU port runs the `MPAS_RKIND8` unit and must match every word.

## Files

| file | content |
|---|---|
| `oracle-r4.npz` | `<case>/<field>`: `cellMask`, `vertexMask`, `edgeMask`, `upperSurface`, `lowerSurface`, `layerThickness`, `layerNormalVelocity`, `minOfMaxAllowableDt`, `err` |
| `oracle-r8.npz` | the same fields at RKIND 8 |
| `inputs.sha256` | each case's input-stream digest, at each precision (`woof.verify.mali_geometry_oracle.input_bytes`) |

## Coverage

| case | nCells | levels | velocity solver | sea level | dynamic thickness |
|---|---|---|---|---|---|
| `sia_shelf` | 120 | 10 | sia | 0 m | 10 m |
| `none_thin` | 80 | 6 | none | 50 m | 100 m |

Each case contains:

- every mask bit the SIA/none paths set;
- thick floating margin cells next to open ocean, whose dynamic bit the
  calving-front rule clears;
- ice thinner than the dynamic threshold;
- initial masks with the initial-extent bit and stale bits;
- layer velocities that average to zero (the CFL `bigNumber` branch);
- unsolved edges.

## Grading

`tests/test_mali_geometry_parity.py` and `woof hex ice verify` require every
word to be bit-identical. On 2026-11-08 the result was 0 differing words on
CUDA (RTX A6000) and on Metal (Apple silicon, the mlx facade).
