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

## The time-step oracle (`run_seaice_step.F90`)

`run_seaice_step.F90` drives the forward model over N steps with the column
package, advection and forcing off:
- init: the velocity solver's time steps, `seaice_init_evp`, and
  `seaice_initialize_time_diagnostics`;
- each step: the clock, `seaice_timestep`, and `seaice_timestep_finalize`.

Its inputs come in three kinds:
- **Cut verbatim by build.sh (awk ranges):**
  - `seaice_timestep`, `store_timestep_counter` and `seaice_timestep_finalize`
    (`timestep_cut_head.F90`);
  - the column wrappers and re-initialisations (`column_cut_head.F90`;
    `column_cut_tail.F90` stands in for the column routines they call only
    under the column package);
  - the time diagnostics and `seaice_check_state` (`diagnostics_cut_head.F90`).
- **Compiled verbatim:** `mpas_seaice_velocity_solver.F`, without
  `-Dprivate=public` (the driver calls its public routine).
- **Stand-ins:**
  - `step_standins.F90` for the basis setup, the weak solver, colpkg, the unit
    tests and advection;
  - the framework from `tools/mpas_ocean_oracle/stubs.F90` and
    `oracle_stream.F90`, shared with the ocean oracles (the time manager on
    whole seconds and the self-describing stream with MPAS-Seaice's pools).

`make_step_fixture.py` writes `tests/data/oracles/mpas_seaice_step`.

## The incremental-remap oracle (`run_incremental_remap.F90`)

`run_incremental_remap.F90` drives MPAS-Seaice v8.4.2's incremental remap on
one block:
- `mpas_seaice_advection_incremental_remap_tracers.F` is compiled verbatim;
- `mpas_seaice_advection_incremental_remap.F` is compiled verbatim with
  `-Dprivate=public` (visibility only), so the driver can call its stages.

Every run starts with `seaice_init_advection_incremental_remap` itself: the
tracer list from the packages, the halo check and the remap geometry. In
`run_incremental_remap_{r4,r8}` the RBF reconstruction coefficients
(`coeffs_reconstruct`) are the input's: `rbf_standins.F90` makes
`mpas_rbf_interp_initialize` and `mpas_init_reconstruct` no-ops. In
`run_incremental_remap_init_{r4,r8}` they are MPAS's own: build.sh cuts
`mpas_initialize_vectors` (with `mpas_unit_vec_in_r3`,
`mpas_cross_product_in_r3` and `mpas_fix_periodicity`),
`mpas_rbf_interp_initialize`,
`mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs`, `evaluate_rbf`,
`mpas_set_up_vector_dirichlet_rbf_matrix_and_rhs`, `mpas_legs`, `elgs` and
`mpas_init_reconstruct` verbatim out of `src/operators` (`init/rbf_cut.F`);
the mesh pool carries `is_periodic`, `x_period` and `y_period` (`mesh:` real
and logical configs). `mpas_legs`/`elgs` are Tao Pang's partial-pivoting
elimination ("Copyright (c) Tao Pang 2001"), as distributed in MPAS's
BSD-licensed tree. Then, by `oracle_stage`:
- 0: nothing more (the tracer tree and the geometry);
- 1: the volume-to-thickness loop of `seaice_run_advection_incremental_remap`
  (and back, under `oracle_round_trip`);
- 2: `find_departure_points`, `find_departure_triangles` and
  `get_triangle_quadrature_points`;
- 3: `make_masks` and `construct_linear_tracer_fields`;
- 4: `seaice_run_advection_incremental_remap`, `oracle_n_steps` times;
- 5: nothing more (the init binary's cases: the geometry and RBF set-up).

The third argument is a log file. It gets the ordered tracer list, every
warning the routines log (the mean-recovery check, with its real arguments
to 17 digits), every MPAS_LOG_ERR message (the conservation and monotonicity
checks; the stubs' `stub_log_err_unit` records them and carries on, as MPAS
does, where by default they stop) and stage 4's error code.

`remap_standins.F90` also supplies:
- `seaice_error`'s codes;
- a no-op abort write;
- an upwind module that stops by name. v8.4.2's upwind advection reads
  configs the sea-ice Registry does not declare and mesh fields it does not
  carry, so it cannot run.

The meshes (`woof.verify.mpas_seaice_planar_mesh`) are planar hexagon and
quadrilateral blocks with land and halo layers. The spherical cases
(`woof.verify.mpas_seaice_remap_sphere_oracle`) lay the same blocks on the
sphere (`woof.verify.mpas_seaice_sphere_mesh`: the inverse gnomonic
projection, turned to either pole or 53 degrees north, optionally stretched to
variable resolution), with and without `config_rotate_cartesian_grid`; the
init derives the local frames and the tangent-plane geometry from them. Cases
are built with correctly rounded operations only (no libm trigonometry, BLAS
or LAPACK), so their input words are the same on every host.

`make_remap_fixture.py` writes `tests/data/oracles/mpas_seaice_remap` (with
`--sphere`, `tests/data/oracles/mpas_seaice_remap_sphere`; with `--checks`,
the conservation and monotonicity checks' cases,
`woof.verify.mpas_seaice_remap_checks_oracle`, into
`tests/data/oracles/mpas_seaice_remap_checks`). For each case it keeps the
output fields whose words changed, and writes `remap-logs.json` (the tracer
lists, warnings and error messages).

The time-step oracle (`run_seaice_step.F90`) runs `seaice_init_advection` and,
under `config_use_advection`, `seaice_run_advection` (both
`mpas_seaice_advection.F` and the remap compiled verbatim).
`make_step_remap_fixture.py` writes `tests/data/oracles/mpas_seaice_step_remap`
(with `--sphere`, the spherical cases into
`tests/data/oracles/mpas_seaice_step_remap_sphere`).

## The column-package oracle (`run_column.F90`, `run_init_thermo.F90`)

`run_column.F90` drives MPAS-Seaice v8.4.2's column package (colpkg) on one
block, at RKIND 8 only: the column package is `dbl_kind`, and
`mpas_seaice_column.F` passes it RKIND arrays, so the wrappers do not compile
with `-DSINGLE_PRECISION`. build.sh compiles, in a separate `column`
directory:
- the column package (`column/ice_*.F90`, `column/constants/cice`) verbatim;
- `mpas_seaice_column.F` verbatim with `-Dprivate=public` (visibility only:
  the driver logs the tracer object);
- `init_ice_state`, `init_ice_state_uniform_ice`, `init_ice_cice_default`,
  `initial_category_areas_and_volumes`, `init_ice_velocity_uniform` and
  `initialize_coupler_fields` cut verbatim from `mpas_seaice_initialize.F`
  (`initialize_cut_head.F90`/`initialize_cut_tail.F90`: the initial
  conditions and velocities the port does not carry stop by name), and
  `seaice_reset_coupler_fluxes` with its two helpers from
  `mpas_seaice_forcing.F`.

Every run starts with `seaice_init_column_physics_package_parameters`, then
by `oracle_stage`: `init` (`initialize_coupler_fields`,
`seaice_init_column_physics_package_variables`, `init_ice_state`,
`seaice_reset_coupler_fluxes`), `aggregate` (`seaice_column_aggregate`) or
`update_state` (`seaice_column_update_state`) or `vertical_thermodynamics`
(`colpkg_init_thermo`, which sets `l_brine` and `heat_capacity`, then
`seaice_column_vertical_thermodynamics`, whose abort status is logged as
`TOBJ IERR`). The log carries the tracer
object (`TOBJ` lines). The field table is the Registry's
(`woof/hex/mpas_seaice/data/registry-fields.json`, written by
`tools/mpas_coupled_options.py --fields-out`), with the fields of packages
that are off absent.

`run_init_thermo.F90` writes `colpkg_init_thermo`'s salinity profile for
every ice-layer count from 1 to N: its `cos` and `pow` are the host libm's,
and the GPU port accepts only the counts whose words it reproduces.

`make_column_fixture.py` writes `tests/data/oracles/mpas_seaice_column`
(the changed fields, `column-logs.json`, `init-thermo-r8.npy`,
`inputs.sha256`). Generate it on the x86-64 glibc oracle host.
`--cases therm` writes the vertical-thermodynamics cases to
`tests/data/oracles/mpas_seaice_column_therm` instead. Their `atan`
(`psimhu`/`psixhu`), `exp`, `log` and `pow` are the host glibc 2.44's.
