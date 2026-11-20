# E3SM v3.2.0 MPAS-Ocean coupler import and export (ocn_import_mct, ocn_export_mct) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle coupling` on pb (Linux
6.12 x86_64, glibc 2.44), 2026-11-20.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `components/mpas-ocean/driver/ocn_comp_mct.F` (`ocn_import_mct` and
  `ocn_export_mct`, cut verbatim from their `subroutine` lines to their
  `end subroutine` lines into a module whose head carries the host module's
  own declarations of what they read: lines 107-109, 119, 125-127 and 129),
  `components/mpas-ocean/src/shared/mpas_ocn_equation_of_state.F`
  (`ocn_freezing_temperature`, cut the same way). Compiled verbatim:
  `components/mpas-ocean/driver/mpaso_cpl_indices.F`,
  `components/mpas-ocean/src/shared/mpas_ocn_constants.F` (with
  `MPAS_ESM_SHR_CONST`, as E3SM builds MPAS, so `rho_sw`, `cp_sw` and
  `T0_Kelvin` are `shr_const_mod`'s), `components/mpas-framework/src/framework/`
  `mpas_kind_types.F` and `mpas_constants.F` (its `gravity = 9.80616` is a
  default-real literal: 9.806159973144531), `share/util/shr_kind_mod.F90`,
  `share/util/shr_const_mod.F90` and `driver-mct/shr/glc_zocnclass_mod.F90`
  (with `NDEBUG`; `share/include/shr_assert.h`).
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc` (the commit v3.2.0 pins),
  byte-unmodified, as for the merge oracle: `mpaso_cpl_indices_set`'s field
  lookups and their stops.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- The MPAS framework is `tools/mpas_ocean_oracle/stubs.F90` (pools by name,
  one block, halo exchanges the identity, a field of a package that is off
  reports `isActive = .false.` and has no array). Stand-ins with no
  arithmetic: `ocean_standins.F90` (`ocn_config`'s namelist options and
  `ocn_diagnostics_variables`' land-ice boundary-layer arrays and indices, set
  by the driver) and `e3sm_standins.F90` (`seq_flds_mod`'s lists and switches,
  `seq_infodata_mod`'s `precip_fact` and `ocn_c2_glctf`, `shr_sys_abort`,
  `shr_log_mod`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, real(RKIND) = real(8). build.sh
  checks the cut routines, `ocn_constants` and the driver for fused
  multiply-adds (none) and the cut routines for libm calls other than
  `sqrt` (none).
- Driver: `run_ocean_coupling.F90`. It reads the switches, the coupler's
  field lists, x2o and o2x, and every config, package, dimension and array
  the two routines read; calls `glc_zocnclass_init`, `mpaso_cpl_indices_set`,
  `ocn_constants_init`, `ocn_import_mct` and `ocn_export_mct`; and writes o2x
  and every array.

## Cases

Arrays are `halo` cells longer than `nCellsSolve` (3, or 0 in
`tf3d_30`), so the cells E3SM leaves alone are checked. x2o is the G case's
list (`e3sm_ocean_merge_oracle.GCASE`), plus the waves and nutrients where
named; o2x starts as noise, so the columns not written are checked.

| case | cells | what it covers |
|---|---|---|
| `gcase` | 23 | frazil on without a land-ice mask; no waves or land ice (their arrays absent); signed zeros into `max(Foxx_swnet, 0)` and the frazil's `min(.., 0)`; the sea-ice pressure cap |
| `standalone` | 37 | land ice standalone with a mask of 0, 1 and 2; the removed Antarctic runoff on, just north of and south of 60S/57S; `precip_fact` 0.9873046875000123; the KPP theory-wave 10 m wind (`sqrt(So_duu10n)`, from 0); `config_density0` 1027.25 |
| `data` | 31 | land ice data; other open-ocean freezing coefficients and heat of fusion; `ocn_cpl_dt` 1237 |
| `coupled_tf3d` | 29 | land ice coupled (the boundary layer sent); 3-D thermal forcing with `glc_nzoc` 4 and `nGlcZLevels` 3, a mask of exactly 0.9 and just above; frazil off; the EKE 10 m wind; `ocn_lnd_one_way` |
| `tf3d_30` | 11 | `glc_nzoc` 30, all levels; no halo |
| `waves_active` | 27 | `wav_ocn_coup = 'twoway'` with `config_momentum_use_active_wave`; `rof2ocn_nutrients`; the removed runoff; four forcing arrays absent |
| `waves_passive` | 19 | the waves two ways, plain stress; four forcing arrays absent |
| `abort_missing_x2o` | 5 | no `Sg_blit`: MCT's `indexRA` stops |
| `abort_missing_x2o_icemask` | 5 | no `Sg_icemask_floating` (the third of three lookups into one index) |
| `abort_missing_o2x` | 5 | no `So_rhoeff` |
| `abort_missing_tf3d` | 5 | no `So_tf3d_mask03` |
| `abort_missing_wave` | 5 | twoway without `Sw_Charn` |
| `abort_missing_nutrient` | 5 | nutrients without `Foxx_rofDSi` |
| `abort_glc_nzoc` | 5 | `glc_nzoc` 5: `glc_zocnclass_init`'s `shr_sys_abort` |

gfortran's `MAX(a, b)` and `MIN(a, b)` keep `a` unless `b` is strictly
greater (smaller): `max(-0.0, 0.0)` is -0.0, `min(-0.0, 0.0)` is -0.0. MCT's
`die` exits with status 0; make_fixture.py counts a `FATAL` line with no
output as the stop. `aborts.json` keeps each stop's line.

## Grading

`tests/test_e3sm_ocean_coupling_parity.py` runs
`woof.hex.coupled.ocn_coupling_e3sm`: every o2x word and every array word
(halo cells included) must be bit-identical, and the index plan must refuse
each abort case with E3SM's message.
