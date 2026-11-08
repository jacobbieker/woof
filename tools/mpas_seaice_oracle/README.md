# MPAS-Seaice v8.4.2 oracle: variational EVP internal stress

`run_oracle.F90` drives two MPAS-Model v8.4.2 modules
(`f34984b2c89353cf31c235b08d3b4acd96b37bb3`), byte-unmodified:

- `mpas_seaice_velocity_solver_constitutive_relation.F`;
- `mpas_seaice_velocity_solver_variational.F`.

It calls them in the model's order: `seaice_init_evp`, then
`seaice_internal_stress_variational` nIter times, then
`seaice_final_divergence_shear_variational`. `stubs.F90` stands in for the
framework (pools, blocks, timers, the one-rank `mpas_dmpar_min_real`). It
looks fields up by name and carries no arithmetic.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mpas_seaice_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mpas_seaice_oracle/make_fixture.py "$BUILD_DIR"
```

- build.sh checks `SOURCES.sha256` and compiles at `-O0 -ffp-contract=off
  -DSINGLE_PRECISION`, MPAS's default precision. It fails if an object
  contains an FMA instruction.
- make_fixture.py writes `tests/data/oracles/mpas_seaice_evp/oracle-r4.npz`
  and `inputs.sha256` from the cases in `woof.verify.mpas_seaice_evp_oracle`.

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mpas_seaice_evp_parity.py
pixi run -e default woof hex seaice verify
```

To rebuild the oracle inside the test and compare it with the committed
fixture, set `GFORTRAN=.pixi/envs/mpas-oracle/bin/gfortran` and
`WOOF_MPAS_V842_SOURCE=<tree>`. Without them, that test skips by name.
