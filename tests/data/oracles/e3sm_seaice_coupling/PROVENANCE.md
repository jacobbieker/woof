# E3SM v3.2.0 MPAS-Seaice coupler import and export (ice_import_mct, ice_export_mct) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` (with its fourth argument,
the Icepack tree) and `tools/e3sm_coupler_oracle/make_fixture.py --oracle
seaice` on pb (Linux 6.12 x86_64, glibc 2.44), 2026-11-20.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `components/mpas-seaice/driver/ice_comp_mct.F` (`ice_import_mct`,
  `ice_export_mct`, `basal_pressure` and `frazil_mass`, cut verbatim from
  their `subroutine` lines to their `end subroutine` lines into a module whose
  head carries the host module's own use lines, 58-68, 75 and 78, and its
  declaration of `domain`, line 124),
  `components/mpas-seaice/src/shared/mpas_seaice_mesh.F`
  (`seaice_latlon_vector_rotation_backward` and the five routines it calls,
  cut the same way). Compiled verbatim:
  `components/mpas-seaice/driver/mpassi_cpl_indices.F`,
  `components/mpas-seaice/src/shared/mpas_seaice_constants.F` (with
  `CCSMCOUPLED`, as E3SM builds it, so its constants are `shr_const_mod`'s:
  `rhoi` 917, `rhosw` 1026, `cp_ice` 2117.27, `cp_ocn` 3996, `Lfresh`
  333700, `Tffresh` 273.15, `gravity` 9.80616), `share/util/shr_kind_mod.F90`
  and `share/util/shr_const_mod.F90`, and the framework's
  `mpas_kind_types.F` and `mpas_constants.F`.
- Icepack: E3SM-Project/Icepack at `cf1d460fbb6a13d28988f798b33734f46d70a9dc`
  (the commit v3.2.0 pins as `components/mpas-seaice/src/icepack`),
  byte-unmodified, every `columnphysics/icepack_*.F90` (as `icepack_intfc`
  uses them all). The driver sets Icepack's parameters as
  `mpas_seaice_icepack.F` does from `seaice_constants` (`puny`, `rhos`,
  `rhoi`, `rhow`, `cp_ice`, `cp_ocn`, `depressT`, `Tocnfrz`, `Tffresh`,
  `Lsub`, `Lvap`), with the case's `ktherm` and `tfrz_option`.
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified:
  `mpassi_cpl_indices_set`'s field lookups and their stops.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for the three
  trees.
- The MPAS framework is `tools/mpas_ocean_oracle/stubs.F90` (pools by name,
  one block, halo exchanges the identity). `e3sm_standins.F90` holds
  `seq_flds_mod`'s lists, `wav_ice_coup` and `wav_nfreq`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, real(RKIND) = real(8). build.sh
  checks the cut routines, the mesh cut, Icepack's mushy physics and shared
  thermodynamics and the driver for fused multiply-adds (none) and the cut
  import and export for libm calls (none).
- Driver: `run_seaice_coupling.F90`. It reads the switches, the coupler's
  field lists, x2i and i2x, and every config, dimension and array the two
  routines read (every subpool exists); calls `mpassi_cpl_indices_set`,
  `icepack_init_parameters`, `ice_import_mct` and `ice_export_mct`; and
  writes i2x, every array, and per cell the stress rotation's eight
  trigonometric words (`trig`: sin and cos of `latCell`, `lonCell` and the
  rotated grid's latitude and longitude), made by E3SM's own
  `seaice_grid_rotation_forward` and `seaice_latlon_from_xyz` and glibc 2.44's
  `sin`, `cos`, `asin` and `atan2`.

## The rotation's libm words

E3SM's `seaice_latlon_vector_rotation_backward` calls `sin`, `cos`, `asin`
and `atan2`. The port takes their words as an input, and the parity tests
feed the ones recorded here, so the claim does not rest on the host's libm.
`ice_coupling_e3sm.rotation_trig` makes them with Python's `math` (the
platform libm): equal to these under glibc 2.44 (`TRIG_GLIBC`; tested
there only); under another glibc release or another libm (macOS's) the
words may differ in the last place, and it warns. A glibc
transcription of `atan2` behind `WOOF_GLIBC239_LIBM64` would close that if
it matters; it is not done, and would never be a default.

## Cases

Arrays are `halo` cells longer than `nCellsSolve` (3, or 0 in
`bl99_constant`), so the cells E3SM leaves alone are checked. i2x starts as
noise, so the export's zeroing is checked. The ice area's first cells are
0, -0, 1, 1.25, -1e-300, 3e-300 and 0.5; x2i's salinity and freezing
potential pairs reach both Si0new branches, both liquidus branches and a
potential of 0, -0 and below. The cells' positions come from Taylor
polynomials for sine and cosine (numpy multiplies and adds), not the host's
libm, so every host builds the same input bytes.

| case | cells | what it covers |
|---|---|---|
| `mushy_free` | 29 | E3SM's default: free surface, mushy thermodynamics and `tfrz_option` mushy, the rotated grid (with its pole and the geographic pole), column shortwave |
| `bl99_mushy_tfrz` | 23 | BL99 under `tfrz_option` mushy (the linear liquidus); unrotated grid; topographic ponds; no column shortwave; icebergs |
| `som_linear_salt` | 17 | a slab ocean (`frazilMassAdjust` left alone, still exported); `linear_salt` |
| `bl99_constant` | 13 | `constant`; BL99's frazil enthalpy; no halo |
| `zero_layer_minus1p8` | 11 | zero-layer thermodynamics; `minus1p8`; ponds |
| `aerosols_bulk` | 19 | the bulk aerosols |
| `aerosols_modal_zaer` | 21 | the modal aerosols and the z-aerosols |
| `zaer_bulk_column_bgc` | 15 | the bulk z-aerosols with the (uncoupled) column biogeochemistry |
| `waves_twoway` | 25 | `wav_ice_coup = 'twoway'`, 7 frequencies, column waves, floe sizes |
| `waves_twoway_off` | 9 | twoway without column waves or floe sizes (Si_ifloe stays 0) |
| `waves_column_only` | 9 | column waves without the waves coupled (nothing read) |
| `abort_missing_x2i` | 5 | no `So_dhdy`: MCT's `indexRA` stops |
| `abort_missing_aerosol` | 5 | no `Faxa_dstwet4` |
| `abort_missing_i2x` | 5 | no `Si_u10withgusts` |
| `abort_missing_ifloe` | 5 | twoway without `Si_ifloe` |
| `abort_missing_wavespec` | 5 | twoway with 12 frequencies, no `Sw_wavespec11` |

Not a case: the z-aerosols coupled (`config_couple_biogeochemistry_fields`)
without the coupled column biogeochemistry. E3SM's export reads
`oceanDustIronFlux` through the `biogeochemistry` pool pointer, which only
the coupled column biogeochemistry branch sets: the oracle segfaults. The
port refuses it, and refuses the coupled column biogeochemistry.

MCT's `die` exits with status 0; make_fixture.py counts a `FATAL` line with
no output as the stop. `aborts.json` keeps each stop's line.

NaN inputs are outside the word-for-word claim, and no case has one.

## Grading

`tests/test_e3sm_seaice_coupling_parity.py` runs
`woof.hex.coupled.ice_coupling_e3sm`: every i2x word and every array word
(halo cells included) must be bit-identical, and the index plan must refuse
each abort case with MCT's message.
