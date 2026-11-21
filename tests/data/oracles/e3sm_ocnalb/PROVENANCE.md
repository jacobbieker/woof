# E3SM v3.2.0 ocean albedo (seq_flux_ocnalb_mct) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle ocnalb` on pb (Linux
6.12 x86_64, glibc 2.44), 2026-11-21.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/seq_flux_mct.F90` (`seq_flux_ocnalb_mct`, cut verbatim
  from its `subroutine` line to its `end subroutine` line, lines 795-973,
  into the same module as the atmosphere-ocean flux's oracle, with the
  host module's public line 28 and a harness setter for the namelist's
  `seq_flux_mct_albdif` and `seq_flux_mct_albdir`).
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified, built
  with `--enable-mpiserial` (one task).
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- `flux_standins_base.F90` holds the rest, with no arithmetic:
  `shr_sys_abort` (stops with status 7), `seq_infodata_mod`'s
  `flux_albav`, `nextsw_cday` and orbital parameters, `component_type_mod`
  (the ocean's domain on the flux grid) and `shr_orb_mod`'s `shr_orb_decl`
  and `shr_orb_cosz` with E3SM's signatures, each stopping with status 9:
  the orbital branch is not part of the oracle.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle`
  pixi environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks
  the cut for fused multiply-adds (none) and that its only libm call is
  the orbital branch's `pow` (`cosz**1.7`), never reached.
- Driver: `run_flux_ocnalb.F90`. It reads `flux_albav`, the namelist's
  albedos, the flux grid's domain (`lat:lon:hgt` and its other fields), a2x,
  the fractions and xao; then per call `nextsw_cday` and the vectors (the
  domain on the first call only); calls `seq_flux_ocnalb_mct`; and writes
  xao and the fractions. E3SM's and MCT's stops go to the standard error.

Inputs are libm-free (numpy's generator, multiplies and adds), so every
host builds the same bytes.

## Cases

Every call runs on 64 points (seed per case): no sunlight as -0 (points 2,
3) and +0 (4); `So_anidr` exactly 1 (5, 9: the dark side), the doubles
either side of 1 (6, 7) and `So_anidr`/`So_anidf` 0 (8); `ifrac` -0 and
`ofrac` +0 (10). The shortwave bands and xao's albedos sit among other
fields, out of the lookups' order.

| case | switches | what it covers |
|---|---|---|
| `albav` | flux_albav, 2 calls (cday -1, 120.25) | the namelist's albedos, `ifrad`/`ofrad` copied; no orbital branch with flux_albav |
| `albav_zero_albedos` | flux_albav, albdif +0, albdir -0 | -0 passes the `< 0` check and is written |
| `computed` | 3 calls (cday -1, -1, -3) | the net shortwave, the dark side, the fractions untouched |
| `computed_cday_below_half` | cday the double below -0.5 | the orbital branch's edge, not taken |
| `computed_negative_zero_albedos` | albdif, albdir -0 | |
| `computed_without_ifrad` | fractions without `ifrad` | the computed branch never looks the radiative fractions up |
| `abort_albdif_negative`, `abort_albdir_negative` | albdif -1e-300; albdir -0.07 and no `So_anidr` | "namelist not initialized", ahead of MCT's lookups |
| `abort_xao_without_anidr`, `abort_xao_without_swup`, `abort_a2x_without_swvdf`, `abort_albav_without_ofrad` | | MCT stopping on a missing field |

`aborts.json` keeps each stop's line, the stand-in's `shr_sys_abort: `
prefix left out.

## Grading

`tests/test_e3sm_ocnalb_parity.py` runs `woof.hex.coupled.ocnalb_e3sm`:
every word of xao and the fractions after every call must be
bit-identical, each E3SM stop the same line and each MCT stop the same
missing attribute. The orbital branch (no flux_albav, `nextsw_cday >=
-0.5`) is refused by name by the port, and the rebuilt oracle is checked
to reach the stand-in (status 9) there.
