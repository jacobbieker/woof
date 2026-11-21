# MPAS-Seaice v8.4.2 column-package oracle: the biogeochemistry framework and brine height

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases bgc` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-20.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_column/PROVENANCE.md`: the column package and
  `mpas_seaice_column.F` are compiled verbatim, with gfortran 16.2.0 at
  `-O0 -ffp-contract=off`, RKIND 8 only.
- Configuration: `config_use_column_biogeochemistry` and `config_use_brine`,
  every other biogeochemistry flag off (the Registry's defaults), so
  `colpkg_init_zbgc` adds only the brine fraction tracer (`nt_fbri`, parent
  the ice volume, no ancestors) and `nBioLayers` is the Registry's 7.
- Driver: `run_column.F90`.
  - Stage `init` (see `../mpas_seaice_column/PROVENANCE.md`): with the
    biogeochemistry, `initialize_coupler_fields` zeroes `atmosBioFluxes`,
    sets the uncoupled build's black carbon and dust deposition and
    `colpkg_init_ocean_conc`'s ocean concentrations on cells 1 to nCells and
    the carbon to nitrogen ratios; `seaice_init_column_physics_package_variables`
    runs `colpkg_init_hbrine`'s grids and `init_column_biogeochemistry_profiles`
    (brine fraction 1 and new ice on every owned cell unless
    `config_do_restart_hbrine`); `seaice_reset_coupler_fluxes` zeroes the
    deposition again with the biogeochemistry's ocean fluxes.
  - Stage `biogeochemistry` calls `column_biogeochemistry(domain, ierr)`
    (`mpas_seaice_column.F:2781-3272`) and logs ierr as `TOBJ IERR`.
- Cases: `woof.verify.mpas_seaice_column_oracle.BGC_CASES`:
  - the init: mushy; BL99 with distinct carbon to nitrogen ratios; a
    brine-height restart; the biogeochemistry without the brine height;
    three ice layers with CESM ponds;
  - the stage (`_bgc_state`): mushy; BL99 at half an hour; a 0.5 m gravity
    drainage length; four ice layers and three categories. The cells carry
    runoff (brine above sea level under little snow), flooding (below it
    under heavy snow), no ice, ice below `thinS`, new ice in the first
    category, non-positive brine fractions, a sub-puny initial area and
    thickness, a layer above the liquidus break salinity, a nearly melted
    category and a warm ocean;
  - `bgc_brine_remap_stop_cleared`: a negative thickness in a middle
    category of cell 5. `remap_zbgc` stops, `colpkg_biogeochemistry` does
    not check it (`ice_colpkg.F90:5127-5139`), the category's profiles keep
    their words, `update_hbrine` runs with `kavg = bphi_o = 0`, and the next
    category's `preflushing_changes` clears the stop (`ice_brine.F90:100`);
  - `bgc_brine_abort_remap`: that, and a negative thickness in the last
    category of cells 8 and 9. The stop of cell 8's last category reaches
    the wrapper, which logs the message at `MPAS_LOG_ERR` and the cell at
    `MPAS_LOG_CRIT` (`mpas_seaice_column.F:3229-3231`): MPAS terminates
    there. The stub (`tools/mpas_ocean_oracle/stubs.F90`) records the
    critical line and its integers before it stops, so this case has no
    outputs; its log carries `crash` (the error, the critical message and
    the cell), which the port's `CriticalAbort` must match.
- The wrapper's `abortMessage` is longer than the 80-character stop label
  `colpkg_biogeochemistry` writes into it and uninitialised past it; the
  stub cuts a logged message at its first NUL, so the log is the same on
  every run.
- v8.4.2's own behaviour, kept word for word:
  - `colpkg_init_zbgc` sets `R_C2N(1) = ratio_C2N_proteins` after
    `R_C2N(1) = ratio_C2N_diatoms` (`ice_colpkg.F90:1108,1121`) and never
    sets `R_C2N_DON`, so the diatoms take the proteins' ratio and DON zero
    (`init_bgc_brine_bl99_c2n`: 5.5, 7.25, 7.75 and 0). The opt-in
    correction `zbgc_c2n_proteins` (`woof_v842_corrections`; not in the
    oracle) gives each its own;
  - `compute_microS_mushy` adds the gravity drainage where the brine height
    `hbr_old >= Ra_c`, the critical Rayleigh number: a length against a
    dimensionless number;
  - `calculate_drho`'s `mm` loop sets `rho_2a(k)` to the same expression
    twice or not at all, and `remap_zbgc` would read `trdr(0)`, which the
    brine height's remap (`hinS = hice`) never does;
  - `colpkg_init_hbrine` rewrites `phi_snow`, which the brine path does not
    read.
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines (and the
  abort case's `crash`). `inputs.sha256`: the input-stream digests.
- libm: `exp` (`update_hbrine`'s drainage) is the host's glibc 2.44; on the
  GPU it is the repository's default transcription in `glibc_flt64.cuh`
  (MIT), so no LGPL opt-in is needed. `x**2` and `x**3` are libgcc's
  `__powidf2` at -O0, transcribed as `x*x` and `x*(x*x)`.
