# MPAS-Seaice v8.4.2 velocity-solver pre-subcycle oracle

Written by `tools/mpas_seaice_oracle/build.sh` and `make_presubcycle_fixture.py`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-09.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256` and refuses a differing file. This
  oracle adds one file to the velocity-solve oracle's set:
  - `src/core_seaice/shared/mpas_seaice_mesh.F`
    sha256 `e85777e3d8c4b07a73e43b2559377e2d61c23645e24519c86b7ab438e3ca30bd`.
    build.sh cuts `seaice_interpolate_cell_to_vertex` and
    `seaice_interpolate_vertex_to_cell` out of it byte for byte (from the
    first routine's `subroutine` line to the second's `end subroutine` line)
    into a module named `seaice_mesh`; the rest of the file needs the stream
    manager and is not built.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Builds of record: `-O0 -ffp-contract=off`, at `-DSINGLE_PRECISION`
  (RKIND = 4) and without it (RKIND = 8).
  - `mpas_seaice_velocity_solver.F` is compiled with `-Dprivate=public`, a
    visibility change only.
  - build.sh fails if a seaice object contains an FMA instruction.
  - Scalar SSE arithmetic, plus one libm call: `ice_strength`'s `exp`, which
    gfortran emits as `expf` (RKIND 4) and `exp` (RKIND 8). They are glibc
    2.44's on pb. On a host whose libm is not glibc, build.sh links
    `tools/oracle_libm` (the repository's transcriptions of those glibc
    routines) instead.
- Driver: `run_presubcycle.F90`. Without the column package's vertical
  thermodynamics it calls `velocity_solver_pre_subcycle` itself. With it, it
  makes the same calls in the same order without `ice_strength`, which would
  call the column package's `colpkg_ice_strength`; the port refuses that by
  name. The framework stand-ins (`stubs.F90`) look fields up by name and do no
  arithmetic.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | every output word of every case at that precision, `<case>/<field>`: the int32 `solveStress`, `solveVelocity`, `solveVelocityPrevious`, and `iceAreaCell`, `iceVolumeCell`, `snowVolumeCell`, `totalMassCell`, `iceAreaCellInitial`, `icePressure`, `airStressCellU/V`, `iceAreaVertex`, `totalMassVertex`, `u/vOceanVelocityVertex`, `u/vVelocity`, `u/vVelocityInitial`, `stressDivergenceU/V`, `oceanStressU/V`, `oceanStressCoeff`, `totalMassVertexfVertex`, `airStressVertexU/V`, `seaSurfaceTiltVertexU/V`, `surfaceTiltForceU/V`, `strain11/22/12`, `stress11/22/12` |
| `inputs.sha256` | the sha256 of each case's input stream at each precision, as `woof.verify.mpas_seaice_presubcycle_oracle.input_bytes` writes it |

The inputs are not committed. `woof.verify.mpas_seaice_presubcycle_oracle.cases()`
regenerates them, and the test refuses a regenerated case whose digest differs.

## Coverage

| case | column package | vertical thermodynamics | air stress | ocean stress | surface tilt |
|---|---|---|---|---|---|
| `standalone_geostrophic` | no | no | yes | yes | geostrophic |
| `column_thermo_ssh_tilt` | yes | yes (no `ice_strength`) | yes | yes | sea-surface-height gradient |
| `column_no_thermo_no_air` | yes | no | no | no | none |
| `standalone_ssh_no_air` | no | no | no | yes | sea-surface-height gradient |

Each case runs at r4 and r8. Fields are drawn in float64 and rounded once to
float32 for r4. Each case also covers:

- halo vertices and cells (nVerticesSolve and nCellsSolve short of the
  totals), which keep their prior words;
- both values of both masks, cells solved only through a neighbour, and
  vertices newly solved and solved before;
- ice-free cells, and cells whose area or mass is just under the minima;
  where the column supplies the initial area, areas of exactly 0.001 on lone
  thin-ice cells, so the strict comparison decides the mask;
- land ice on cells and vertices, and boundary vertices;
- fVertex of +0.0 and -0.0.

## Grading

`tests/test_mpas_seaice_presubcycle_parity.py` and `woof hex seaice verify` run
`woof.hex.mpas_seaice.presubcycle_v842` on each case. Every word must be
bit-identical.
