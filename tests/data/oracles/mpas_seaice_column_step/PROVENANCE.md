# MPAS-Seaice v8.4.2 column-package oracle: the column part of the time step

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases step` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-20.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, stage `column_step`. Setup is
  `colpkg_init_thermo`, `init_column_itd` and, under the column dEdd
  shortwave, `colpkg_init_orbit`. Then, for each of `oracle_n_steps` steps,
  in `seaice_timestep`'s order (`mpas_seaice_time_integration.F:42-174`)
  without the velocity solver and the advection:
  - from the second step on, `seaice_column_reinitialize_fluxes` (as
    `seaice_timestep_finalize` runs it after the previous step);
  - `mpas_advance_clock`;
  - `seaice_column_reinitialize_diagnostics_thermodynamics` and `_bgc`;
  - `seaice_column_predynamics_time_integration`;
  - per dynamics subcycle, `seaice_column_reinitialize_diagnostics_dynamics`
    and `seaice_column_dynamics_time_integration`;
  - `seaice_column_postdynamics_time_integration`.

  The step stops on `ierr > 0`, as `seaice_timestep` returns. The
  abort status is logged as `TOBJ IERR`. Before the clock it calls
  `mpas_timekeeping_init` with `config_calendar_type`, and the stub time
  manager (`tools/mpas_ocean_oracle/stubs.F90`) counts the 365-day year for
  `gregorian_noleap`. `get_day_of_year` and `get_seconds_into_day` read the
  clock's current time, so each step's day and time are the advanced
  clock's, not the start's.
- Pools: the vertical thermodynamics' pools plus `velocity_variational`.
  The routines look its fields up by the Registry's `name_in_code`
  (`principalStress1`, not `principalStress1Var`), and the stream writes
  them under those names (`NAME_IN_CODE`). `seaice_column_coupling_prep`
  reads `nCellsSolve` and `nCategories` from the `tracers` pool, so the
  stream carries them there.
- Cases: `woof.verify.mpas_seaice_column_oracle.STEP_CASES`. All but one
  are BL99 without ponds and without the ocean mixed layer (refused by
  name, `coupling_prep_v842`):
  - the dEdd shortwave on 2001-06-01 from 10:00, three steps;
  - two dynamics subcycles, four layers and three categories, on
    `gregorian` from 2000-07-10 06:00, three steps;
  - the CCSM3 shortwave with constant albedos in January, four steps;
  - the coupled surface temperature, three steps;
  - the Registry defaults (mushy, level ponds, dEdd) without the vertical
    thermodynamics, two steps;
  - melting in July, two steps;
  - an abort on low starting salinity in step 1 (`init_vertical_profile`,
    cell 6, ierr 10).

  The state is built as the vertical thermodynamics' (`_thermo_state`),
  plus open water, ridging rates, the shortwave and rain forcing, the
  albedos, and a physical initial salinity profile (`_step_state`).
- v8.4.2's own behaviour, kept word for word:
  - `oceanDragCoefficient` is re-initialised to `dragio`, 0.00536 in the
    stand-alone column constants (0.00962 only under `RASM_MODS`), and
    `airDragCoefficient` to `seaice_column_initial_air_drag_coefficient`;
  - the re-initialisations are whole-array assignments, the missing-cell
    element included;
  - `seaice_column_vertical_thermodynamics` returns at the first aborting
    cell. The owned cells after it keep their input words, and the step
    stops there.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object, the abort status and the
  log lines. `inputs.sha256`: the input-stream digests.
- libm: the vertical thermodynamics needs glibc's `atan`, and the dEdd
  shortwave needs its `sin`, `cos`, `atan` and `asin`. These are not
  correctly rounded. The GPU matches them only through the LGPL
  transcriptions behind `WOOF_GLIBC239_LIBM64=1`
  (`tools/glibc239_libm_gpu/`); without the opt-in those stages are refused
  by name. `exp` and `log` are `glibc_flt64.cuh`'s MIT transcriptions.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
`woof/hex/mpas_seaice/column_step_v842.py` follows
`mpas_seaice_time_integration.F` and `mpas_seaice_column.F`, under the same
licence (`licenses/LICENSE-MPAS-BSD-3-Clause.txt`).
