# MPAS-Seaice v8.4.2 column-package oracle: BL99 vertical thermodynamics

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases therm` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-15; regenerated 2026-11-20 with the calendar-aware clock
(`freezeOnset`/`meltOnset` of the 2000 cases moved a day; the two
`vt_bl99_first_year_north_*` cases added).

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only. No object contains an FMA (checked
  with objdump).
- Driver: `run_column.F90`, stage `vertical_thermodynamics`. It runs
  `seaice_init_column_physics_package_parameters`, then `colpkg_init_thermo`
  (which sets `l_brine` and `heat_capacity`), then
  `seaice_column_vertical_thermodynamics`. The abort status is logged as
  `TOBJ IERR`. Before the clock it calls `mpas_timekeeping_init` with
  `config_calendar_type`, as MPAS's framework does
  (`mpas_framework.F:111-116`); the stub time manager
  (`tools/mpas_ocean_oracle/stubs.F90`) counts the 365-day year for
  `gregorian_noleap` and stops on a date the calendar lacks.
- Cases: `woof.verify.mpas_seaice_column_oracle.VT_CASES`:
  - bubbly and MU71 conductivity (MU71 with highfreq and three atmosphere
    iterations);
  - the coupled surface temperature (`config_use_surface_temperature`
    false);
  - melting;
  - first-year ice reset in each hemisphere. The north's is at day 259 on
    `config_calendar_type`'s calendar: 2000-09-16 on `gregorian_noleap` (the
    Registry default; 260 on a Gregorian count), 2000-09-15 on `gregorian`,
    while 2000-09-15 on `gregorian_noleap` is day 258 and does not reset;
  - one layer and one category;
  - an abort on low starting salinity (cell 6, `init_vertical_profile`).

  Every cell's state is consistent with BL99: the snow-layer shortwave is
  zero where hsn <= 1e-4, and the interior shortwave is the sum of the layer
  terms. Some cells are ice-free, sub-puny or have no snow, and one cell
  forms snow ice.
- v8.4.2's own bugs, kept word for word by default, each with an opt-in
  correction (`woof_v842_corrections`; not in the oracle, tested in
  `tests/test_mpas_seaice_column_corrections.py`):
  - `colpkg_step_therm1` passes `set_sfcflux` the coupled conductive flux
    as the surface flux and the reverse (`ice_colpkg.F90:2095-2101` against
    `ice_flux_colpkg.F90:169-177`). `set_sfcflux_order` reads each as what
    it is; the test gives the corrected run the two fluxes under their own
    names and gets v8.4.2's run word for word;
  - `colpkg_atm_boundary` gets `uVelocity`, a vertex field, at the cell's
    index (`mpas_seaice_column.F:1511-1512`) and passes `uvel` for `vvel`
    (`ice_colpkg.F90:3474-3479`), so high-frequency coupling's stress is
    against another point's u and no v. `ice_velocity_at_cell` passes both
    components of the cell's ice velocity
    (`seaice_interpolate_vertex_to_cell`); the tests check that ice
    drifting with the wind feels no stress and that the stress follows the
    wind relative to the areaTriangle-weighted vertex mean.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: `atan` (`psimhu`, `psixhu`), `exp`, `log` and `pow` are the host's
  glibc 2.44. glibc's binary64 atan is not correctly rounded on these
  arguments. `tools/mpas_seaice_oracle/grade_atmo_atan.cpp` compared it with
  the repository's correctly rounded `wcr_atan` on 75,497,472 arguments in
  [1, 3.5625), the range of psimhu's `xqq`. They differ in 43,325 words
  (0.057%). The GPU port therefore calls `g239::g239_atan`
  (`tools/glibc239_libm_gpu`, an LGPL transcription of glibc's atan, graded
  word for word against glibc 2.44 in `verification.log`). It does so only
  with the existing opt-in `WOOF_GLIBC239_LIBM64=1`. Without the opt-in the
  port refuses BL99 by name.
- On the GPU, `exp`, `log` and `pow` (`deltaT**1.36`, the lateral melt) are the
  repository's default transcriptions in `glibc_flt64.cuh`, the same as in
  stage 1. gfortran folds two `log` constant expressions, `rdn_ice =
  vonkar/log(zref/iceruf)` and `al2 = log(zref/zTrf)`. The port computes them
  on the host, with each operation correctly rounded as gfortran's MPFR
  folding does.

## Licence

MPAS-Model v8.4.2 is BSD-3-Clause (LANS and UCAR); see
`../mpas_seaice_column/PROVENANCE.md` for the column package's licence note.
glibc is LGPL-2.1. The atan transcription stays behind the opt-in, and
neither the default build nor the wheels carry it.
