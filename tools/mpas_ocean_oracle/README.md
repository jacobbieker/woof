# MPAS-Ocean v8.4.2 oracles: momentum and thickness tendency terms

`run_oracle.F90` drives `ocn_vel_coriolis_tend`, `ocn_vel_vadv_tend`,
`ocn_thick_hadv_tend` and `ocn_thick_vadv_tend`, with their `_init`
routines, from MPAS-Model v8.4.2 (`f34984b2c89353cf31c235b08d3b4acd96b37bb3`),
byte-unmodified. Each routine adds into its own tendency array. `stubs.F90`
stands in for the framework. It uses the name-lookup pool of
`tools/mpas_seaice_oracle/stubs.F90`, extended with character and real
configs and the 1-D dimension arrays, and it carries no arithmetic.
`mpas_ocn_constants.F` is compiled verbatim for its `ocnConfigs` pointer.

`run_pgrad_hmix.F90` drives `ocn_vel_pressure_grad_tend` (every
`config_pressure_gradient_type`) and `ocn_vel_hmix_tend` (del2, Leith and del4,
through `ocn_vel_hmix_del2`, `_leith` and `_del4`), byte-unmodified, with
`mpas_constants.F` (gravity, pii). The stubs add the del4 scratch fields
(`field2DReal`, `mpas_allocate_scratch_field`), `mpas_threading` and
`mpas_log`, which drops informational lines and stops on an error or critical
one. The tensor hmix forms are not driven: the operator modules they call
(`mpas_vector/matrix/tensor_operations`) are stand-ins that stop by name.

`run_tendency.F90` drives `ocn_tend_vel` and `ocn_tend_thick` from
`mpas_ocn_tendency.F` (byte-unmodified), with every term they call compiled
verbatim (bulk forcing, `ocn_forcing`, the velocity forcing modules and the
thickness surface flux added to the above). It also drives
`ocn_vel_forcing_tend` and `ocn_thick_surface_flux_tend` on their own.
`stubs_tendency.F90` stands in for the modules `mpas_ocn_tendency.F` USEs
but `ocn_tend_vel`/`ocn_tend_thick` never reach: tracers, diagnostics, vmix
and the equation of state. Their dummy declarations are copied from v8.4.2
and their bodies stop by name. Land-ice fluxes and frazil return when
switched off and stop by name when on.

`run_eos.F90` drives `ocn_equation_of_state_density` ('linear' and 'jm'),
byte-unmodified, as `ocn_diagnostic_solve` and `ocn_vmix_coefs_rich` call it
(the five displacement/coefficient combinations, and an 'absolute' one at the
deepest level), and `ocn_freezing_temperature` with its salinity derivative.
build.sh compiles the three equation-of-state modules in a directory of their
own (`eos/`), so the real `ocn_equation_of_state` replaces
`stubs_tendency.F90`'s stand-in there; `stubs.F90` gains the `dm_info` type
they declare.

`run_diagnostics.F90` drives `ocn_diagnostic_solve`, cut verbatim with
`ocn_diagnostics_init` and `ocn_compute_land_ice_flux_input_fields` out of
`mpas_ocn_diagnostics.F` (awk ranges, in `eos/`) into
`diagnostics_cut_head.F90`'s module, with the real equation of state and
`mpas_ocn_diagnostics_routines.F`. It is self-describing (configs and fields
by pool and name; an inactive package's fields are unassociated pointers),
so later pieces can reuse its pattern.

`run_tracer_advection.F90` drives `ocn_tracer_advection_init` and
`ocn_tracer_advection_tend` (the three advection modules verbatim, the two
vertical stencils cut out of `src/operators/mpas_tracer_advection_helpers.F`).
`run_diagnostics.F90` and `run_tracer_advection.F90` read and write
`oracle_stream.F90`'s self-describing streams.

gfortran calls `expf`/`cosf`/`sinf` (RKIND 4) and `exp` (RKIND 8). On glibc
these are the host's. Elsewhere build.sh links `tools/oracle_libm/glibc_host.cpp`,
the repository's MIT glibc transcriptions compiled for the host, which the
MALI oracles share.

## Build and run (Linux, gfortran, CPU only)

```bash
pixi run -e mpas-oracle bash tools/mpas_ocean_oracle/build.sh "$MPAS_V842" "$BUILD_DIR"
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR"
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR" --piece pgrad_hmix
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR" --piece tendency
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR" --piece eos
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR" --piece diagnostics
pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py "$BUILD_DIR" --piece tracer_advection
```

build.sh checks `SOURCES.sha256` and builds `run_oracle_r4` and
`run_pgrad_hmix_r4` (`-DSINGLE_PRECISION`, MPAS's default) and their `_r8`
twins (RKIND 8), all at
`-O0 -ffp-contract=off`. It fails if an object contains an FMA instruction.

## Grade the GPU port

```bash
pixi run -e default python -m pytest tests/test_mpas_ocean_tend_parity.py tests/test_mpas_ocean_pgrad_hmix_parity.py \
    tests/test_mpas_ocean_tendency_parity.py tests/test_mpas_ocean_eos_parity.py \
    tests/test_mpas_ocean_diagnostics_parity.py tests/test_mpas_ocean_tracer_advection_parity.py
pixi run -e default woof hex ocean verify
```
