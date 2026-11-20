# MPAS-Seaice v8.4.2 oracle: the time step with the column package and the velocity solver

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases coupled` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-20.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The column package,
  `mpas_seaice_column.F`, the velocity solver (variational), the advection
  and `mpas_seaice_time_integration.F`'s cut are compiled verbatim with
  gfortran 16.2.0 at `-O0 -ffp-contract=off`, RKIND 8 only (the column
  package does not compile at RKIND 4). None of the 51 objects nor
  `run_column_r8` contains an FMA (checked with objdump).
- Driver: `run_column.F90`, stage `seaice_step`. Setup is
  `colpkg_init_thermo`, `init_column_itd` and, under the column dEdd
  shortwave, `colpkg_init_orbit`; then the time steps, `seaice_init_evp`,
  `seaice_init_advection` and the time diagnostics. Then, for each of
  `oracle_n_steps` steps, `mpas_advance_clock`, the whole
  `seaice_timestep` (`mpas_seaice_time_integration.F:42-174`: the time
  diagnostics and step counter, the re-initialised thermodynamic and bgc
  diagnostics, `seaice_column_predynamics_time_integration`, per dynamics
  subcycle the re-initialised dynamic diagnostics,
  `seaice_run_velocity_solver` and
  `seaice_column_dynamics_time_integration`, then
  `seaice_column_postdynamics_time_integration`), stopping on `ierr > 0`,
  and `seaice_timestep_finalize` after it unless it is the last step and
  `oracle_finalize_last` is false (a coupler reads the last step's fluxes).
  The abort status is logged as `TOBJ IERR`, the step it stopped at as
  `TOBJ STEP`.
- The velocity solver calls `colpkg_ice_strength` (`ice_colpkg`); the
  step oracle's stand-in for that module moved to `colpkg_standin.F90` so
  that the column build links the real one.
- Mesh: a culled hexagonal torus (`hex_mesh`, (nq, nr, owned columns) per
  case), so the velocity solver's stencils are those of a real mesh; the
  solver's own state is built as the time-step oracle's
  (`mpas_seaice_step_oracle`) and the column state as the column step's.
- Cases: `woof.verify.mpas_seaice_column_oracle.COUPLED_CASES`, dt 1800 s:
  - BL99, dEdd, Rothrock75 strength, linear participation and
    redistribution, 2001-06-01 from 10:00, three steps;
  - two dynamics subcycles, revised EVP, exponential participation and
    redistribution, four layers and three categories, on `gregorian` from
    2000-07-10 06:00, two steps;
  - Hibler79 strength, the CCSM3 shortwave with constant albedos, no
    surface tilt, in January, three steps;
  - the Registry defaults (mushy, level ponds, dEdd) without the vertical
    thermodynamics, two steps;
  - the last step not finalised, two steps;
  - an abort on low starting salinity in step 1 (`init_vertical_profile`,
    cell 6, ierr 10).
- v8.4.2's own behaviour, kept word for word:
  - `colpkg_ice_strength`'s `Cf` is `ice_constants_colpkg`'s 17;
    `config_ratio_ridging_work_to_PE` is not read under the column package
    (the port refuses any other value by name);
  - `calculation_masks` and `air_stress` leave their work to the column
    package under its vertical thermodynamics, and
    `final_divergence_shear` writes `ridgeConvergence` and `ridgeShear`.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object, the abort status, the
  step and the log lines. `inputs.sha256`: the input-stream digests.
- libm: the vertical thermodynamics needs glibc's `atan`, and the dEdd
  shortwave needs its `sin`, `cos`, `atan` and `asin`. These are not
  correctly rounded. The GPU matches them only through the LGPL
  transcriptions behind `WOOF_GLIBC239_LIBM64=1`
  (`tools/glibc239_libm_gpu/`). `exp` (Hibler79) and `log` are
  `glibc_flt64.cuh`'s MIT transcriptions.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/timestep_v842.py` and `column_ridge_v842`'s
`ice_strength` follow `mpas_seaice_time_integration.F`,
`mpas_seaice_velocity_solver.F` and `ice_mechred.F90`, under the same
licence (`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
