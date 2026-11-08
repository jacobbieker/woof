# MPAS-Seaice v8.4.2 EVP velocity-solve oracle

Written by `tools/mpas_seaice_oracle/build.sh` and `make_velocity_fixture.py`
on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-08.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified. build.sh checks
  `tools/mpas_seaice_oracle/SOURCES.sha256` and refuses a differing file. The
  four files this oracle adds were also checked against GitHub's copy at that
  commit:
  - `src/core_seaice/shared/mpas_seaice_velocity_solver.F`
    sha256 `21622b1c616031453d319d02ec739ebd9c371dcab93738c47597ae6268a0991c`
  - `src/core_seaice/shared/mpas_seaice_constants.F`
    sha256 `b0605800bb3e5b0e3660ca635d61f184f076efd638214a1e129c1a79fe83e576`
  - `src/core_seaice/column/constants/cice/ice_constants_colpkg.F90` (the
    stand-alone build's copy, as `column/Makefile` picks it without ESM)
    sha256 `a25de356f0dbe2a43881c38ce7693e6887d6be602b018b896563311e42cb3c47`
  - `src/core_seaice/column/ice_kinds_mod.F90`
    sha256 `5d4cbed440df4fe786ad1af1203776f3ca78d2228e29babfe7c3323a2d39442d`
  - and, as for the internal-stress oracle,
    `mpas_seaice_velocity_solver_{constitutive_relation,variational}.F` and
    `src/framework/mpas_kind_types.F`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, from the
  `mpas-oracle` pixi environment.
- Builds of record: `-O0 -ffp-contract=off`, at `-DSINGLE_PRECISION`
  (RKIND = 4) and without it (RKIND = 8).
  - `mpas_seaice_velocity_solver.F` is compiled with `-Dprivate=public`. That
    changes only the visibility of its routines, so the driver can call the
    private ones; no arithmetic changes.
  - build.sh fails if a seaice object contains an FMA instruction.
  - Only scalar SSE arithmetic is used (`addss`/`sd` ... `sqrtss`/`sd`;
    `sign()` is `andps`/`orps`). No libm call is made.
- Driver: `run_subcycle.F90`. Its header lists the calls of the two modes:
  `full` (whole elastic subcycles) and pieces (velocity solves against a
  fixed stress-divergence forcing). The framework stand-ins (`stubs.F90`)
  look fields up by name and do no arithmetic. The halo exchanges, mesh
  interpolation, weak-form solver and column-package routines stop by name if
  reached.

## Files

| file | content |
|---|---|
| `oracle-r4.npz`, `oracle-r8.npz` | every output word of every case at that precision, `<case>/<field>`: `numericalInertiaCoefficient`, `uVelocity`, `vVelocity`, `totalMassVertexfVertex`, `oceanStressU/V`, `oceanStressCoeff`, `strain11/22/12`, `stress11/22/12`, `replacementPressure`, `stressDivergenceU/V`, `divergence`, `shear`, `ridgeConvergence`, `ridgeShear`, `principalStress1/2` |
| `inputs.sha256` | the sha256 of each case's input stream at each precision, as `woof.verify.mpas_seaice_velocity_oracle.input_bytes` writes it |

The inputs are not committed; `woof.verify.mpas_seaice_velocity_oracle.cases()`
regenerates them, and the test refuses a regenerated case whose digest differs.

## Coverage

| case | mode | revised EVP | ocean stress | column package | nSub | dynamics dt | precisions |
|---|---|---|---|---|---|---|---|
| `subcycle` | full | no | yes | yes | 120 | 3600 s | r4 |
| `subcycle_revised` | full | yes | yes | yes | 120 | 1800 s | r4 |
| `solve` | pieces | no | yes | yes | 8 | 3600 s | r4, r8 |
| `solve_revised` | pieces | yes | yes | no | 8 | 1800 s | r4, r8 |
| `solve_no_ocean_stress` | pieces | no | no | yes | 4 | 900 s | r4, r8 |

The full subcycle is RKIND 4 only because the internal stress is ported at
RKIND 4 for now. The meshes and EVP state are the internal-stress oracle's.
Each case also covers:

- halo vertices and cells (nVerticesSolve and nCellsSolve short of the
  totals), which keep their prior words;
- unsolved vertices, some with zero mass;
- fVertex of +0.0 and -0.0, so `sign(1, f)` takes both values;
- principal stresses on both sides of the `puny` test.

## Grading

`tests/test_mpas_seaice_velocity_parity.py` and `woof hex seaice verify` run
`woof.hex.mpas_seaice.velocity_v842` on each case. Every word must be
bit-identical.
