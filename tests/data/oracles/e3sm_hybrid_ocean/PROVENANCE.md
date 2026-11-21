# E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration: ocean oracle

Written by `tools/mpas_ocean_oracle/build.sh`, then
`tools/e3sm_coupler_oracle/build.sh` (with `MPAS_V842_SOURCE_ROOT` and
`MPAS_OCEAN_ORACLE_BUILD` set) and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle hybrid` on pb (Linux
6.12 x86_64, glibc 2.44), 2026-11-21.

- MPAS-Model v8.4.2 (`f34984b2`), byte-unmodified, checked against
  `tools/mpas_ocean_oracle/SOURCES.sha256`: the ocean's forward mode as
  `run_forward` builds it, at RKIND 8, over `tools/mpas_ocean_oracle/stubs.F90`
  (pools by name, one block, halo exchanges the identity).
  `src/core_ocean/shared/mpas_ocn_constants.F` is compiled twice: with
  `MPAS_ESM_SHR_CONST` (shr_const_mod's `T0_Kelvin` 273.15 and `rho_sw` 1026,
  as E3SM builds MPAS: `run_hybrid_ocean_e3sm`) and without
  (`run_hybrid_ocean_mpas`).
- E3SM v3.2.0 (`648b167b7f95267a159a9fdee25faf604c2bfc4f`), byte-unmodified:
  `ocn_import_mct` and `ocn_export_mct` cut verbatim out of
  `components/mpas-ocean/driver/ocn_comp_mct.F` as for the coupling oracle,
  `mpaso_cpl_indices.F`, `shr_kind_mod`, `shr_const_mod`,
  `glc_zocnclass_mod`, compiled against v8.4.2's modules (so the import's
  `gravity` is v8.4.2's framework `9.80616_RKIND` and the export's freezing
  point is v8.4.2's `ocn_freezing_temperature`). MCT at `82b0071e`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, `-O0
  -ffp-contract=off`; build.sh checks the objects for fused multiply-adds.
- Driver: `run_hybrid_ocean.F90`: `ocn_init_mct`'s sequence for runtype
  'initial' (forcing built, the clock moved to the coupling alarm without a
  step, `ocn_time_average_coupled_init` and one accumulate, the export, the
  import of a zero x2o), then per window `ocn_run_mct`'s (the import, the
  means reset, `ocn_cpl_dt / config_dt` steps of: the absorbed fractions,
  frazil and land-ice arrays built, the clock advanced, `ocn_timestep`, the
  time levels shifted; then the export). o2x is MCT's persistent vector.
  It writes o2x after the init and each window, then every field.

## Cases

All real(8); the starts are the forward oracles' (`woof.verify.
mpas_ocean_forward*_oracle`), x2o is the G case's list
(`e3sm_ocean_merge_oracle.GCASE`) drawn per window (iceberg fluxes zero,
the sea-ice pressure past its cap in places).

| case | start | steps x windows | constants | what it covers |
|---|---|---|---|---|
| `hybrid_forward` | `split_three_steps` | 2 x 2 | e3sm | the loop, the means reset, the clock lag |
| `hybrid_frazil` | `frazil_split_three_steps` | 3 x 2 | e3sm | frazil sent (`Fioo_q`, `Fioo_frazil`) and reset |
| `hybrid_frazil_land_ice_mpas_constants` | `frazil_land_ice_nested_halos` | 1 x 3 | mpas | halos, land ice 'pressure_only', `config_density0` 1027.5, a positive freezing reference pressure |
| `hybrid_land_ice_coupled` | `coupled_jenkins_two_steps` | 2 x 2 | e3sm | land ice 'coupled': the boundary layer and effective density sent |
