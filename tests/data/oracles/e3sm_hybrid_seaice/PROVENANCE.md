# E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration: sea-ice oracle

Written on pb (Linux x86_64, glibc 2.44) on 2026-11-21, in three steps:
1. `tools/mpas_seaice_oracle/build.sh`;
2. `tools/e3sm_coupler_oracle/build.sh`, with `MPAS_V842_SOURCE_ROOT` and
   `MPAS_SEAICE_ORACLE_BUILD` set;
3. `tools/e3sm_coupler_oracle/make_fixture.py --oracle hybrid_seaice`.

- MPAS-Model v8.4.2 (`f34984b2`), byte-unmodified, checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. This is the time step with the
  column package and the velocity solver, as `run_column`'s `seaice_step`
  stage builds it, at RKIND 8. It uses the stand-alone build's
  `column/constants/cice` over `tools/mpas_ocean_oracle/stubs.F90`.
  - `post_atmospheric_coupling` and `post_oceanic_coupling`
    (`mpas_seaice_forcing.F`) are cut byte for byte, as is
    `seaice_init_post_clock_advance` (`mpas_seaice_initialize.F`).
  - The rotation routines of `mpas_seaice_mesh.F` are cut byte for byte too.
- E3SM v3.2.0 (`648b167b7f95267a159a9fdee25faf604c2bfc4f`), byte-unmodified.
  - `ice_import_mct`, `ice_export_mct`, `basal_pressure` and `frazil_mass`
    are cut verbatim out of `components/mpas-seaice/driver/ice_comp_mct.F`.
  - The host's `use` lines for `seaice_constants` and `seaice_mesh` are
    pointed at `hybrid_seaice_constants` and `hybrid_seaice_mesh`
    (renames only).
  - These are compiled against v8.4.2's modules. In
    `hybrid_seaice_shims.F90`:
    - `seaiceReferenceSalinity`, `seaiceFrazilSalinityReduction` and
      `seaiceFrazilIcePorosity` are the column package's
      `ice_ref_salinity`, `dSin0_frazil` and `phi_init`;
    - Icepack's `sea_freezing_temperature`, `liquidus_temperature` and
      `enthalpy_mush` are colpkg's.
  - MCT is at `82b0071e`.
- Compiler: GNU Fortran (conda-forge) 16.2.0, `-O0 -ffp-contract=off`.
  `build.sh` checks the objects for fused multiply-adds.
- Driver, `run_hybrid_seaice.F90`:
  - At start (`ice_init_mct`, runtype 'initial'): `nextsw_cday`, the core's
    init, `seaice_column_coupling_prep`, the export, then the import of a
    zero x2i.
  - Each window (`ice_run_mct`): `seaice_column_reinitialize_fluxes`, the
    import, `nextsw_cday`, the post-couplings, then `ice_cpl_dt / config_dt`
    steps. Each step advances the clock, runs
    `seaice_init_post_clock_advance` on the run's first step only, then
    `seaice_timestep` with no finalize. Then the export.
  - It writes i2x after the init and after each window, then the rotation's
    words (glibc's sin, cos, asin and atan2), then every field.

## Cases

All real(8). The starts are the column oracle's `COUPLED_CASES`
(`woof.verify.mpas_seaice_column_oracle`). Each window's x2i is drawn
within 2.5% of the start's own forcing; the sea-ice freezing potential and
the frazil flux are drawn outright. `sphere_radius` is 1 (the torus's cells
are on the unit sphere).

| case | start | steps x windows | what it covers |
|---|---|---|---|
| `hybrid_bl99_dedd` | `seaice_bl99_dedd_rothrock` | 1 x 3 | BL99, dEdd (its init on the first step), rotated grid |
| `hybrid_bl99_two_steps_gregorian` | `seaice_bl99_dedd_two_subcycles_exponential_gregorian` | 2 x 2 | fluxes summed over the window, counter restart, gregorian |
| `hybrid_mushy_level_ponds` | `seaice_mushy_level_ponds_no_vertical_thermodynamics` | 2 x 2 | mushy: E3SM's frazil mass on colpkg's mush enthalpy |
| `hybrid_bl99_ccsm3_unrotated` | `seaice_bl99_hibler_strength_ccsm3_winter` | 1 x 2 | ccsm3 shortwave, no tilt, unrotated grid |

The dEdd cases need the opt-in glibc 2.39 transcription
(`WOOF_GLIBC239_LIBM64=1`, source checkouts), as the column step's own
parity tests do.
