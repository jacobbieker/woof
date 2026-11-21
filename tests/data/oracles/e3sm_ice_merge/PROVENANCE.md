# E3SM v3.2.0 sea-ice import merge (prep_ice_merge) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle icemerge` on pb (Linux
6.12 x86_64, glibc 2.44), 2026-11-21.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/prep_ice_mod.F90` (`prep_ice_merge`, cut verbatim from its
  `subroutine` line to its `end subroutine` line into a module of its own),
  `share/util/shr_kind_mod.F90` (compiled verbatim) and
  `share/util/mct_mod.F90` (its own `use m_AttrVect` renames and its
  `mct_usevector` flag).
- MCT: `82b0071e69d14330b75d23b0bc68543ebea9aadc`, the commit v3.2.0 pins as
  `externals/mct`, byte-unmodified, built by build.sh as it is
  (`configure --enable-mpiserial`). MCT only moves data here: the field
  lists, the shared indices and the shared-name `mct_aVect_copy`.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- Stand-ins (`e3sm_standins.F90`, no arithmetic): `shr_sys_abort`,
  `seq_comm_mct` (one coupler task, the root) and `seq_flds_mod`'s
  `wav_ice_coup` (set by the driver).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks the
  merge's objects for fused multiply-adds (none) and for libm calls (none).
- Driver: `run_ice_merge.F90`. It reads `flux_epbalfact`, `wav_ice_coup` and
  the six attribute vectors' field lists, then calls `prep_ice_merge` once
  per call in the stream (x2i read in as well, as the merge does not zero
  it) and writes x2i after each.

## Cases

| case | points | calls | what it covers |
|---|---|---|---|
| `gcase` | 61 | 3 | lists modelled on `seq_flds_mod` for MPAS ocean + sea ice with a data atmosphere and runoff; `flux_epbalfact` 1; signed zeros in the sums (-0 + -0, -0 + 0) |
| `gcase_epbal` | 40 | 2 | the same, `flux_epbalfact` 0.9873046875000123 |
| `branches_wave_twoway` | 53 | 2 | `wav_ice_coup = 'twoway'` (wave states copied, w2x's `So_u` after o2x's), `Sx_shared` from o2x and a2x (a2x's lands), a2x's own `Faxa_rain` overwritten by the sum, `Sg_shared_ice` from g2x, `Sx_nobody` kept |
| `branches_wave_oneway` | 29 | 2 | the same lists, `wav_ice_coup = 'oneway'`: the waves are not copied |
| `abort_missing_a2x_snowc`, `abort_missing_r2x_rofi`, `abort_missing_x2i_rofi` | 5 | 1 | MCT's `indexRA` stopping on the first, a middle and the last lookup |

## Grading

`tests/test_e3sm_ice_merge_parity.py` runs `woof.hex.coupled.prep_ice_merge_e3sm`:
every word of x2i after every call must be bit-identical, and each stop
MCT's for the same field.
