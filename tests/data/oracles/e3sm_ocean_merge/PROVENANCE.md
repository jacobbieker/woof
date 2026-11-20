# E3SM v3.2.0 ocean import merge (prep_ocn_merge) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py` on pb (Linux 6.12 x86_64, glibc
2.44), 2026-11-20.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/prep_ocn_mod.F90` (`prep_ocn_merge`, cut verbatim from its
  `subroutine` line to its `end subroutine` line into a module of its own),
  `share/util/shr_kind_mod.F90` (compiled verbatim) and
  `share/util/mct_mod.F90` (its own `use m_AttrVect` renames and its
  `mct_usevector` flag; the rest of that file pulls in the MPI share layer).
- MCT: github.com/MCSclimate/MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`,
  the commit v3.2.0 pins as `externals/mct`, byte-unmodified, built by
  build.sh as it is (`configure --enable-mpiserial`, its own mpi-serial).
  MCT only moves data here: the field lists, `mct_aVect_zero` and the
  shared-name `mct_aVect_copy`.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- Stand-ins (`e3sm_standins.F90`, no arithmetic): `shr_sys_abort` (writes its
  message, then `error stop 7`), `seq_comm_mct` (one coupler task, the root),
  `seq_flds_mod`'s three switches (set by the driver from the stream) and
  `prep_glc_mod` (never called by `prep_ocn_merge`).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, for the merge, its stand-ins and MCT;
  real(r8); no FMA instruction in any object; no libm call.
- Driver: `run_ocean_merge.F90`. It reads the switches and the eight
  attribute vectors' field lists, then calls `prep_ocn_merge` once per call
  in the stream (the first call works out the plan, as in a run) and writes
  x2o after each.

## Cases

| case | points | calls | what it covers |
|---|---|---|---|
| `gcase` | 61 | 3 | field lists modelled on `seq_flds_mod` for MPAS ocean + sea ice with a data atmosphere and runoff, no wave or land-ice coupling; `flux_epbalfact` 1 |
| `gcase_epbal` | 40 | 2 | the same, `flux_epbalfact` 0.9873046875000123 |
| `branches` | 53 | 2 | `rof2ocn_nutrients`, `seq_flds_i2o_per_cat`, `wav_ocn_coup = 'twoway'` (wave states copied), `Sx_tref`/`Sx_qref` merged from the ice and the atmosphere/ocean fluxes, `Sx_shared` copied from the ice and then merged onto |
| `branches_wave_oneway` | 29 | 2 | the same lists with the switches off: the wave states are not copied |
| `abort_multiple_matches` | 5 | 1 | two atmosphere fields end in `_lwdn`: `ERROR multiple ka field matches` |
| `abort_atm_and_xao` | 5 | 1 | `Foxx_lat` matched by the atmosphere and the atmosphere/ocean fluxes: `ERROR aindx and xindx both non-zero` |
| `abort_merge_logic` | 5 | 1 | `Faxa_lwdn` replaced from the atmosphere but added from the ice: `ERROR aindx and iindx merge logic error` |
| `abort_missing_field` | 5 | 1 | no `Faxa_swvdr`: MCT's `indexRA` stops (`FATAL--attribute not found`) |

The fractions run through open ocean, full ice, partial ice, land (both
fractions zero, left as they are), sums other than one, and radiative
fractions that differ from the others. x2o is noise on entry, so the zeroing
is checked. MCT's `die` leaves through mpi-serial's `MPI_Abort`, which exits
with status 0; make_fixture.py counts a `FATAL` line with no output as the
stop. `aborts.json` keeps each stop's line.

## Grading

`tests/test_e3sm_ocean_merge_parity.py` runs
`woof.hex.coupled.prep_ocn_merge_e3sm`: every x2o word must be bit-identical,
and the port must refuse each abort case with E3SM's message.
