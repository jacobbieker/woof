# E3SM v3.2.0 coupler interval (RASM_OPTION1, chained) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle interval` on pb (Linux
6.12 x86_64, glibc 2.44), 2026-11-21.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified. Cut verbatim
  from their `subroutine`/`function` lines to their `end` lines and linked
  into one program:
  - `driver-mct/main/prep_ocn_mod.F90`: `prep_ocn_mrg`, `prep_ocn_merge`,
    `prep_ocn_accum`, `prep_ocn_accum_avg`, `prep_ocn_calc_i2x_ox`,
    `prep_ocn_get_mapper_Fa2o` and `prep_ocn_get_mapper_SFi2o`, in a module
    of that name whose head is the host's own use lines (3-21) and module
    data (81-113), the data made public for the driver;
  - `driver-mct/main/prep_ice_mod.F90`: `prep_ice_mrg`, `prep_ice_merge`,
    `prep_ice_calc_a2x_ix`, `prep_ice_calc_o2x_ix` and
    `prep_ice_get_mapper_SFo2i`, likewise (use lines 3-20, data 62-78);
  - `driver-mct/main/seq_flux_mct.F90`, `seq_frac_mct.F90`, `seq_map_mod.F90`
    and `seq_map_type_mod.F90` as the flux and fractions oracles cut them
    (`e3sm_flux_atmocn`, `e3sm_ocnalb`, `e3sm_seq_frac`);
  - `share/util/mct_mod.F90`: `mct_rearr_rearrange_fldlist`,
    `mct_avect_accum` and `mct_aVect_avg`, in a `mct_mod` written from the
    file's own `use m_*, only:` renames and flags.
  Compiled verbatim: `share/util/shr_kind_mod.F90`, `shr_const_mod.F90`,
  `water_isotopes.F90` and `shr_flux_mod.F90`.
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified, built
  with `--enable-mpiserial` (one task).
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- `chain_standins_base.F90` and `chain_standins.F90` hold the rest, with no
  arithmetic: the union of the single-routine oracles' stand-ins (one
  instance of each component, one task's reductions, the timers, the
  switches as the driver sets them, `seq_flds_mod` public and re-exporting
  `seq_comm_setptrs` as E3SM's does).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle`
  pixi environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks the
  cut objects for fused multiply-adds (none) and the merges, accumulation,
  mapping and fractions for libm calls (none); the flux's are its own
  oracle's (exp, log, atan, pow).
- Driver: `run_coupler_interval.F90`. It sets the coupler up as
  `prep_ocn_init`, `prep_ice_init` and `seq_flux_init_mct` do (the mapped
  vectors and the accumulator zeroed, the copy_only ocean/sea-ice mappers,
  the atmosphere mapper of the case's kind, the flux's mask from the
  starting fractions), then per coupler step runs `cime_run_atmocn_setup`'s
  routines, `prep_ocn_accum_avg` on the ocean's run alarm,
  `cime_run_ice_setup_send`'s on the sea ice's, reads the sea ice's new
  export, runs `seq_frac_set`, and reads the ocean's new export on its next
  alarm; and writes xao, the accumulator's count and words, x2o, x2i and
  the three fractions after each step.

Inputs are libm-free (numpy's generator and separate multiplies and adds),
so every host builds the same bytes.

## Cases

16 points on the ocean's and sea ice's grid (11 on the atmosphere's for a
matrix mapper), the ocean and sea-ice merges' G-case field lists, the flux's
fields; seed per case.

| case | steps | alarms | what it covers |
|---|---|---|---|
| `ocean_every_third` | 7 | ocean every 3rd, ice every step | windows of 1, 3, 3; matrix i2a; nextsw_cday changing below -0.5; new r2x/g2x/w2x every other step |
| `albav_ice_every_other_epbal` | 4 | ocean and ice every 2nd | windows of 1, 2; `flux_albav`; `nextsw_cday` and `flux_epbalfact` changing every step; new r2x/g2x/w2x at step 3; a copy i2a; fractions held between the ice's runs |
| `implicit_gust` | 6 | ocean every 4th | the implicit stress and gustiness; windows of 1, 4, then one never averaged |
| `dead_models` | 3 | ocean every step | dead models (the mask set to 1) |
| `ocean_not_prognostic` | 4 | ocean every 3rd | no merge, no accumulation; a rearrange i2a |
| `no_component_maps` | 4 | ocean every 2nd | `ice_c2_ocn`, `ocn_c2_ice`, `atm_c2_ice` off: the mapped vectors stay zero |
| `ice_not_prognostic` | 3 | ocean every 3rd | no sea-ice merge; its export still read |
| `frac_abort_mid_run` | 3 | ocean every 3rd | the second step's sea-ice export below `seq_frac_check`'s bound: the chain stops in `seq_frac_set` (`aborts.json`) |
| `ocean_startup_lag` | 7 | ocean every 3rd, startup | `skip_ocean_run`: both ocean alarms held off until the run alarm's second ring; a first window of 4 |

Per step the stream also carries infodata's `nextsw_cday` and
`flux_epbalfact` and, when flagged, new runoff, land-ice and wave exports to
the ocean and sea ice; the driver sets them before the step's phases.

## Grading

`tests/test_e3sm_coupler_interval_parity.py` runs
`woof.hex.coupled.interval_e3sm.CouplerInterval` with
`WOOF_GLIBC239_LIBM64=1`: the count and every word of xao, the accumulator,
x2o, x2i and the three fractions after every step must be identical, and
the stopping case must stop in the same step with `seq_frac_check`'s line.
