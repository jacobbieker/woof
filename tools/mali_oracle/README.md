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

`run_advection.F90` calls `li_thermal_init` as on a restart, where it only
copies the densities into the module, and then
`li_advection_thickness_tracers` as `li_time_integration_fe` does (tracers on
or off). The 2-D and 3-D scratch fields it allocates resolve, in the
stand-ins, to arrays the driver fills with random words: the model leaves
them uninitialised, so no output may depend on them.

`run_thermal_solver.F90` calls `li_thermal_init` as on a restart, which
copies the densities into the module and builds `dsigmaTerm`, and then
`li_basal_melt_floating_ice` and `li_thermal_solver` as
`li_time_integration_fe` does. The scratch fields they allocate
(`iceCellMask`, `workLevelEdge`) resolve to driver arrays filled with random
words.

## Build and run (Linux, gfortran, CPU only)

build.sh compiles everything twice: at RKIND 4 (`-DSINGLE_PRECISION`,
`BUILD_DIR/*_r4`) and at RKIND 8 (`BUILD_DIR/*_r8`). Each `make_*` script
runs both and writes `oracle-r4.npz` and `oracle-r8.npz`.

```bash
pixi run -e mpas-oracle bash tools/mali_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mali_oracle/make_fixture.py "$BUILD_DIR"           # SIA
pixi run -e default python tools/mali_oracle/make_geometry_fixture.py "$BUILD_DIR"  # geometry
pixi run -e default python tools/mali_oracle/make_thermal_fixture.py "$BUILD_DIR"   # thermal init
pixi run -e default python tools/mali_oracle/make_advection_fixture.py "$BUILD_DIR" # advection
pixi run -e default python tools/mali_oracle/make_thermal_solver_fixture.py "$BUILD_DIR" # thermal solver
```

The fixtures were written on x86-64 Linux, where gfortran's `x**y`, `exp`,
`sin` and `tanh` call glibc's `powf`, `expf`, `sinf` and `tanhf` at RKIND 4,
and `x**y` and `exp` call `pow` and `exp` at RKIND 8. Another libm returns different
words for some arguments: built on macOS against Apple's libm, the SIA
oracle differs in 10 words of `bary_pb1982/normalVelocity` (the PB1982
flow-law exponential and the real powers). So on a host without glibc,
build.sh links `tools/oracle_libm/glibc_host.cpp` into the oracles (the
host libm shim every MPAS oracle shares). That file is the repository's MIT
glibc transcriptions (`glibc_flt32.cuh`, `glibc_trig_flt32.cuh`,
`sfire_libm.cuh` and, at RKIND 8, `glibc_flt64.cuh` in `woof/core/kernels`)
compiled for the host. It equals pb's glibc 2.44 `expf` and `tanhf` on all
2^32 inputs and its `powf` on 4e8 random pairs, and with it the macOS rebuild writes the
committed fixtures word for word (gfortran 16.2.0 from Homebrew, arm64).
`TOOLCHAIN.txt` records which `powf`/`expf` the build used.

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mali_sia_parity.py tests/test_mali_geometry_parity.py \
    tests/test_mali_thermal_init_parity.py tests/test_mali_advection_parity.py \
    tests/test_mali_thermal_solver_parity.py
pixi run -e default woof hex ice verify
```
