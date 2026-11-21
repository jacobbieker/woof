# E3SM v3.2.0 atmosphere-ocean flux (seq_flux_atmocn_mct) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle flux` on pb (Linux 6.12
x86_64, glibc 2.44), 2026-11-20.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/seq_flux_mct.F90` (`seq_flux_init_mct` and
  `seq_flux_atmocn_mct`, cut verbatim from their `subroutine` lines to their
  `end subroutine` lines into a module whose head is the host module's own
  lines 1-24, 28, 30 and 32-214: its use lines, module arrays and
  parameters; the other routines' public lines are left out, harness
  setters for the namelist's `seq_flux_atmocn_minwind`, `seq_flux_mct_albdif`
  and `seq_flux_mct_albdir` are added, and `seq_flux_ocnalb_mct` is cut
  alongside for its own oracle, `tests/data/oracles/e3sm_ocnalb`). Compiled verbatim:
  `share/util/shr_flux_mod.F90` (`shr_flux_atmOcn`,
  `shr_flux_update_stress`), `share/util/water_isotopes.F90`,
  `share/util/shr_kind_mod.F90` and `share/util/shr_const_mod.F90`.
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified, built
  with `--enable-mpiserial` (one task): the attribute vectors, the domain's
  general grid and the global segment map.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- `flux_standins_base.F90` and `flux_standins.F90` hold the rest, with no
  arithmetic: `shr_sys_abort` (stops with status 7), `shr_log_mod`
  (`shr_log_errMsg` as E3SM's: `ERROR in <file> at line <n>`),
  `seq_comm_mct` (the log is the standard error), `seq_flds_mod`'s and
  `seq_infodata_mod`'s switches, `component_type_mod` and `shr_orb_mod`
  (never reached).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks the cut
  and `shr_flux_mod.o` for fused multiply-adds (none), that
  `shr_flux_atmOcn` calls exactly `__powidf2`, `atan`, `exp`, `log` and
  `pow` (glibc 2.44's libm here; `log(zref/ztref)` is folded by gfortran),
  and that the cut calls no libm.
- Driver: `run_flux_atmocn.F90`. It reads the switches, the flux grid's
  domain (`lat:lon:hgt` and its other fields), its fractions, a2x, o2x and
  xao; builds the domain and segment map; calls `seq_flux_init_mct` and
  `seq_flux_atmocn_mct` once; and writes xao. E3SM's and MCT's stops go to
  the standard error.

Inputs are libm-free (numpy's generator, multiplies, adds and divides), so
every host builds the same bytes.

## Cases

Every case runs on the same 96 points (seed per case): the domain's mask 0
(points 3, 17), 0.5 (21, kept) and 0.4999 (22); no ocean or ice (5, 30),
+0 + -0 (31) and a tiny ocean fraction (32); cold air over warm water at
every fifth point; calm points moving with the ocean, stable (40), unstable
(41) and neutral (42, delt 0), a near-calm one (43), delt 0 (44), `zbot` at
the reference height (45); a density of 0 (50), between float32(1e-12) and
1e-12 (51), an ocean at 0.5 K (52) and at 1 K (53); no wind response and
the 0.99 * wind0 cap (60, 61), no gust (62).

| case | switches | what it covers |
|---|---|---|
| `explicit_default` | explicit, max -1 (2) | the E3SMv1 scheme as E3SM runs it by default |
| `implicit_stress` | implicit, tol 0.01 | the implicit stress, max -1 (30), the stress cap and its damping |
| `gustiness` | gust | the atmosphere's gustiness |
| `coldair_outbreak` | coldair | the cold-air outbreak scaling |
| `implicit_gust_coldair` | all three, tol 0.01 | together |
| `converges_early` | tol 1e-3, max 10 | points stopping before the cap |
| `dead_models`, `dead_implicit_gust` | dead | E3SM's constants, the mask set to 1, the diurnal fields 0 |
| `high_minwind` | minwind 4, max 3 | the minimum wind |
| `one_iteration` | max 1 | |
| `abort_no_iteration`, `abort_tolerance_one` | max 0; tol 1 | "No iterations performed" |
| `abort_sizes` | a2x of 95 points | "nloc sizes do not match" |
| `abort_a2x_without_pslv`, `abort_gust_without_ugust`, `abort_xao_without_ninc` | | MCT stopping on a missing field |
| `abort_scheme_3` | scheme 3 | shr_flux_atmOcn's scheme check |
| `abort_diurnal_ua` | scheme 2, diurnal | the diurnal cycle with the UA scheme |

`aborts.json` keeps each stop's line, the stand-in's `shr_sys_abort: `
prefix left out.

## Grading

`tests/test_e3sm_flux_atmocn_parity.py` runs `woof.hex.coupled.flux_atmocn_e3sm`
with the opt-in glibc 2.39 atan (`WOOF_GLIBC239_LIBM64=1`): every word of
xao must be bit-identical, each E3SM stop the same line and each MCT stop
the same missing attribute.
