# MPAS-Seaice v8.4.2 oracles: variational EVP internal stress and the EVP velocity solve

`run_oracle.F90` drives two MPAS-Model v8.4.2 modules
(`f34984b2c89353cf31c235b08d3b4acd96b37bb3`), byte-unmodified:

- `mpas_seaice_velocity_solver_constitutive_relation.F`;
- `mpas_seaice_velocity_solver_variational.F`.

It calls them in the model's order: `seaice_init_evp`, then
`seaice_internal_stress_variational` nIter times, then
`seaice_final_divergence_shear_variational`. `stubs.F90` stands in for the
framework (pools, blocks, timers, the one-rank `mpas_dmpar_min_real`). It
looks fields up by name and carries no arithmetic.

`run_subcycle.F90` drives `mpas_seaice_velocity_solver.F` itself, also
byte-unmodified. It is compiled with `-Dprivate=public`, a visibility change
only, because the routines it calls are private to the module. It calls them
in `seaice_run_velocity_solver`'s order:

1. the pre-subcycle Coriolis coefficient, ocean stress and subcycle
   initialisation;
2. then either whole elastic subcycles (internal stress, ocean stress
   coefficient, `solve_velocity` or `solve_velocity_revised`), or the velocity
   solve alone against a fixed stress-divergence forcing;
3. then `velocity_solver_post_subcycle` (the final divergence/shear, the
   principal stresses and `ocean_stress_final`), or in the velocity-solve
   mode the principal stresses and `ocean_stress_final`.

`run_presubcycle.F90` drives the same module's `velocity_solver_pre_subcycle`
(aggregates, masks, new-ice velocities, Hibler ice strength, air stress,
Coriolis coefficient, ocean stress, surface tilt, subcycle initialisation).
Under the column package's vertical thermodynamics `ice_strength` calls
`colpkg_ice_strength`, which is not built; the driver then calls the same
routines without it (the GPU port refuses that ice strength by name).
`seaice_mesh`'s two interpolation routines are cut verbatim out of
`mpas_seaice_mesh.F` by build.sh into a module of that name; the rest of that
file needs the stream manager. `ice_strength` calls `exp`: the host glibc's
`expf`/`exp`, or `tools/oracle_libm`'s transcriptions of them elsewhere.

It reads `seaice_constants`, so the stand-alone column constants
(`column/constants/cice`) and `ice_kinds_mod` are compiled too. Both oracles
are built at RKIND 4 and at RKIND 8. The column package is double, so
production MPAS-Seaice is built double.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mpas_seaice_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mpas_seaice_oracle/make_fixture.py "$BUILD_DIR"           # internal stress
pixi run -e default python tools/mpas_seaice_oracle/make_velocity_fixture.py "$BUILD_DIR"  # velocity solve
pixi run -e default python tools/mpas_seaice_oracle/make_presubcycle_fixture.py "$BUILD_DIR"  # pre-subcycle
```

- build.sh checks `SOURCES.sha256` and compiles at `-O0 -ffp-contract=off`,
  once with `-DSINGLE_PRECISION` (RKIND 4) and once without (RKIND 8). It
  fails if an object contains an FMA instruction.
- make_fixture.py writes `tests/data/oracles/mpas_seaice_evp/oracle-r{4,8}.npz`
  and `inputs.sha256` from the cases in `woof.verify.mpas_seaice_evp_oracle`.
- make_velocity_fixture.py does the same for
  `tests/data/oracles/mpas_seaice_velocity` (`woof.verify.mpas_seaice_velocity_oracle`).
- make_presubcycle_fixture.py does the same for
  `tests/data/oracles/mpas_seaice_presubcycle` (`woof.verify.mpas_seaice_presubcycle_oracle`).

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mpas_seaice_evp_parity.py tests/test_mpas_seaice_velocity_parity.py \
    tests/test_mpas_seaice_presubcycle_parity.py
pixi run -e default woof hex seaice verify
```

To rebuild the oracle inside the test and compare it with the committed
fixture, set `GFORTRAN=.pixi/envs/mpas-oracle/bin/gfortran` and
`WOOF_MPAS_V842_SOURCE=<tree>`. Without them, that test skips by name.
