# E3SM v3.2.0 ocean-input accumulation (prep_ocn_accum, prep_ocn_accum_avg) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle accum` on pb (Linux 6.12
x86_64, glibc 2.44), 2026-11-21.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/prep_ocn_mod.F90` (`prep_ocn_accum` and
  `prep_ocn_accum_avg`, cut verbatim from their `subroutine` lines to their
  `end subroutine` lines into a module of the same name whose head is the
  host's own lines 3, 8, 17, 18, 20 and 21 (what they use) and 106-107 and
  110 (the accumulator, its count and `mpicom_CPLID`), the accumulator made
  public for the driver) and `share/util/mct_mod.F90` (`mct_avect_accum` and
  `mct_aVect_avg`, cut verbatim into a `mct_mod` written from the file's own
  `use m_*, only:` renames and flags). Compiled verbatim:
  `share/util/shr_kind_mod.F90`.
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified, built
  with `--enable-mpiserial` (one task).
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- `accum_standins.F90` holds the rest, with no arithmetic: one ocean
  instance, the timers (nothing) and the components' x2o pointer.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle`
  pixi environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks the
  cut objects for fused multiply-adds (none) and libm calls (none).
- Driver: `run_ocn_accum.F90`. It reads x2o's fields and starting words,
  sets the accumulator up as `prep_ocn_init` does (x2o's fields, zeroed, a
  count of 0), then per operation either loads the x2o a merge would have
  written and calls `prep_ocn_accum`, or calls `prep_ocn_accum_avg`; and
  writes the count, the accumulator and x2o after each.

Inputs are libm-free (numpy's generator, multiplies and powers of 10 as
integers), so every host builds the same bytes.

## Cases

48 points, 14 of E3SM's x2o fields, magnitudes 1e-3 to 1e5 (seed per case).

| case | operations | what it covers |
|---|---|---|
| `window_of_three` | 3 accum, avg | `x * (1/3)`, which is not `x / 3` |
| `window_of_two` | 2 accum, avg | an exact reciprocal |
| `window_of_one` | accum, avg | no scaling at a count of 1 |
| `average_without_samples` | avg, 2 accum, avg, avg, accum, avg | the zeroed accumulator copied, the last average copied again, a first sample copied over it |
| `windows_of_seven_and_five` | 7 accum, avg, 5 accum, avg | 1/7 and 1/5 |
| `signed_zeros_subnormals_overflow` | 3 accum, avg, 2 accum, avg | -0 + -0, -0 + +0, subnormal sums and products, sums past the largest double |

No case stops (`aborts.json` is empty).

## Grading

`tests/test_e3sm_ocn_accum_parity.py` runs
`woof.hex.coupled.ocn_accum_e3sm.OceanAccumulator`: the count and every
word of the accumulator and x2o after every operation must be identical.
