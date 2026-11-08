# MALI v8.4.2 oracle: the shallow-ice velocity solver

`run_oracle.F90` makes one `li_sia_solve` call from MPAS-Model v8.4.2
(`f34984b2c89353cf31c235b08d3b4acd96b37bb3`), byte-unmodified. That call
reaches:

- `li_calculate_flowParamA`;
- the barycentric and kite-area vertex interpolations;
- `mpas_tangential_vector_1d`;
- the SIA edge loop.

So build.sh compiles every land-ice module and framework operator on that
path verbatim: `li_constants`, `li_setup`, `li_mask`, `li_thermal`,
`li_diagnostic_vars`, `li_sia`, `mpas_constants`, `mpas_vector_operations`,
`mpas_matrix_operations` and `mpas_geometry_utils`.

`stubs.F90` stands in for the rest of the framework:

- the name-lookup pool, extended with integer configs, field types and
  `mpas_pool_add_config`;
- `mpas_log`, `mpas_abort`, `mpas_timekeeping` and
  `mpas_vector_reconstruction`.

It carries no arithmetic. Routines the call never reaches stop with their
name if reached.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mali_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mali_oracle/make_fixture.py "$BUILD_DIR"
```

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mali_sia_parity.py
pixi run -e default woof hex ice verify
```
