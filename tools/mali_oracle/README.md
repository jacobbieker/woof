# MALI v8.4.2 oracles: the shallow-ice velocity solver and the geometry diagnostics

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

`run_geometry.F90` makes the calls the model makes before thickness
advection, in this order: `li_calculate_mask`, `li_update_geometry`,
`li_calculate_layerThickness` and `li_layer_normal_velocity`. They come from
the same verbatim objects plus `mpas_li_advection.F`.

`run_thermal.F90` calls `li_thermal_init` and then
`li_compute_pressure_melting_point_fields`, `li_temperature_to_enthalpy` and
`li_enthalpy_to_temperature`.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mali_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mali_oracle/make_fixture.py "$BUILD_DIR"           # SIA
pixi run -e default python tools/mali_oracle/make_geometry_fixture.py "$BUILD_DIR"  # geometry
pixi run -e default python tools/mali_oracle/make_thermal_fixture.py "$BUILD_DIR"   # thermal init
```

The fixtures were written on x86-64 Linux, where gfortran's `x**y` and
`exp` call glibc 2.39's `powf` and `expf`. Another libm returns different
words for some arguments: built on macOS against Apple's libm, the SIA
oracle differs in 10 words of `bary_pb1982/normalVelocity` (the PB1982
flow-law exponential and the real powers). So on a host without glibc,
build.sh links `glibc_flt32_host.c` into the oracles. That file is the
repository's MIT glibc-float32 transcription (`woof/core/kernels/glibc_flt32.cuh`)
compiled for the host. It equals pb's glibc 2.44 `expf` on all 2^32 inputs
and its `powf` on 4e8 random pairs, and with it the macOS rebuild writes the
committed fixtures word for word (gfortran 16.2.0 from Homebrew, arm64).
`TOOLCHAIN.txt` records which `powf`/`expf` the build used.

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mali_sia_parity.py tests/test_mali_geometry_parity.py \
    tests/test_mali_thermal_init_parity.py
pixi run -e default woof hex ice verify
```
