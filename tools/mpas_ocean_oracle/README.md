# MPAS-Ocean v8.4.2 oracle: momentum and thickness tendency terms

`run_oracle.F90` drives `ocn_vel_coriolis_tend`, `ocn_vel_vadv_tend`,
`ocn_thick_hadv_tend` and `ocn_thick_vadv_tend`, with their `_init`
routines, from MPAS-Model v8.4.2 (`f34984b2c89353cf31c235b08d3b4acd96b37bb3`),
byte-unmodified. Each routine adds into its own tendency array. `stubs.F90`
stands in for the framework. It uses the name-lookup pool of
`tools/mpas_seaice_oracle/stubs.F90`, extended with character and real
configs and the 1-D dimension arrays, and it carries no arithmetic.
`mpas_ocn_constants.F` is compiled verbatim for its `ocnConfigs` pointer.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mpas_ocean_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR"
```

build.sh checks `SOURCES.sha256` and builds `run_oracle_r4`
(`-DSINGLE_PRECISION`, MPAS's default) and `run_oracle_r8` (RKIND 8), both at
`-O0 -ffp-contract=off`. It fails if an object contains an FMA instruction.

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mpas_ocean_tend_parity.py
pixi run -e default woof hex ocean verify
```
