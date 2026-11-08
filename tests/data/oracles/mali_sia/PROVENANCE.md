# MALI v8.4.2 shallow-ice (SIA) velocity-solver oracle

Written by `tools/mali_oracle/build.sh` and `make_fixture.py` on pb
(Linux 6.12 x86_64, glibc 2.44, CPU with FMA), 2026-11-08.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified (build.sh checks
  `tools/mali_oracle/SOURCES.sha256`, which covers 11 files):
  - `core_landice`: `mpas_li_constants.F`, `mpas_li_setup.F`, `mpas_li_mask.F`,
    `mpas_li_thermal.F`, `mpas_li_diagnostic_vars.F`, `mpas_li_sia.F`;
  - framework: `mpas_kind_types.F`, `mpas_constants.F`,
    `mpas_vector_operations.F`, `mpas_matrix_operations.F`,
    `mpas_geometry_utils.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Reference of record: `-O0 -ffp-contract=off -DSINGLE_PRECISION` (RKIND = 4).
  build.sh fails if an object contains an FMA instruction. The objects call
  the host libm's `powf` (the real powers of `li_sia_solve`) and `expf`
  (`li_calculate_flowParamA`). The outputs were unchanged when the oracle was
  re-run with `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA`.
- Driver: `run_oracle.F90` puts the case in the mesh, geometry, thermal and
  velocity pools and the configs in `liConfigs`. It makes one
  `li_sia_solve(meshPool, geometryPool, thermalPool, velocityPool, err)` call
  and writes `err`, the count of error-level log lines, and every output. The
  framework stand-ins (`stubs.F90`) look entries up by name and carry no
  arithmetic. The routines the call never reaches (scratch fields,
  reconstruction, unit tests) stop with their name if reached.

## Files

| file | content |
|---|---|
| `oracle-r4.npz` | `<case>/<field>`: `normalVelocity`, `flowParamA`, `slopeEdge`, `tangentSlopeEdge`, `upperSurfaceVertex`, and `err` (`err`, log errors; both 0) |
| `inputs.sha256` | the input-stream digest of each case (`woof.verify.mali_sia_oracle.input_bytes`) |

The inputs are regenerated, not committed, with the sea-ice fixture's
RandomState method.

## Coverage

| case | nCells | levels | tangent slope | flowParamA | n | enhancement |
|---|---|---|---|---|---|---|
| `bary_pb1982` | 80 | 10 | from_vertex_barycentric | PB1982 | 3.0 | 1.0 |
| `kite_cp2010` | 64 | 12 | from_vertex_barycentric_kiteareas | CP2010 | 3.0 | 2.5 |
| `normal_constant_n35` | 48 | 8 | from_normal_slope | constant | 3.5 | 1.0 |

Each case contains:

- ice-free cells;
- non-dynamic edges;
- dynamic edges with one dynamic neighbour;
- temperatures either side of the 263.15 K branch and beyond both clamps;
- unsolved edges past `nEdgesSolve` (they keep their prior words);
- mask bits other than the dynamic-ice bit.

## Grading

`tests/test_mali_sia_parity.py` and `woof hex ice verify` run
`woof.hex.mpas_ice.sia_v842`, and every word must be bit-identical. On
2026-11-08 the result was 0 differing words in every field of every case on
CUDA (RTX A6000) and on Metal (Apple silicon, the mlx facade).
