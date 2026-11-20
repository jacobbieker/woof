# E3SM v3.2.0 coupler surface fractions update (seq_frac_set) oracle

Written by `tools/e3sm_coupler_oracle/build.sh` and
`tools/e3sm_coupler_oracle/make_fixture.py --oracle frac` on pb (Linux 6.12
x86_64, glibc 2.44), 2026-11-20.

- E3SM source: github.com/E3SM-Project/E3SM tag v3.2.0, commit
  `648b167b7f95267a159a9fdee25faf604c2bfc4f`, byte-unmodified:
  `driver-mct/main/seq_frac_mct.F90` (`seq_frac_set` and `seq_frac_check`,
  cut verbatim from their `subroutine` lines to their `end subroutine` lines
  into a module whose head is the host module's own lines 143-182 and
  184-228, its use lines, switches, `eps_*` parameters and mapper pointers;
  `seq_frac_init`'s public line is left out, as it is not cut, and a
  harness setter for the private `seq_frac_dead` is added),
  `driver-mct/main/seq_map_mod.F90` (`seq_map_map`, `seq_map_avNormAvF`,
  `seq_map_avNormArr`, cut the same way under the host's use lines 14-17
  and 19-23 and its `seq_map_avNorm` interface, lines 41-44),
  `driver-mct/main/seq_map_type_mod.F90` (the type `seq_map`) and
  `share/util/mct_mod.F90` (its `use m_*, only:` renames, its two flags and
  `mct_rearr_rearrange_fldlist`, cut verbatim). Compiled verbatim:
  `share/util/shr_kind_mod.F90` and `share/util/shr_const_mod.F90`.
- MCT at `82b0071e69d14330b75d23b0bc68543ebea9aadc`, byte-unmodified, built
  with `--enable-mpiserial` (one task): the attribute vectors, the global
  segment maps, the rearranger, `mct_sMatP_Init` (distributed, XandY) and
  `mct_sMat_avMult`.
- build.sh checks `tools/e3sm_coupler_oracle/SOURCES.sha256` for both trees.
- `frac_standins_base.F90` and `frac_standins.F90` hold the rest, with no
  arithmetic: `shr_sys_abort` (stops with status 7), the one-task MPI
  reductions, `seq_comm_mct` (the log is the standard error),
  `seq_infodata_mod`'s three switches, `component_type_mod`, the `prep_*`
  mapper getters, and `seq_nlmap_mod` (never reached).
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, real(r8). build.sh checks the cut
  routines, `mct_mod` and MCT's matrix multiply and rearranger for fused
  multiply-adds (none) and the cuts for libm calls (none).
- Driver: `run_seq_frac.F90`. It reads the switches, the two mappers' kinds
  and a sparse matrix's elements in the mapping file's order, i2x, the ice
  domain (`lat:lon:hgt` and its other fields) and the three fraction
  vectors; builds the global segment maps, a rearranger or the matrix as the
  coupler does on one task; calls `seq_frac_set`; and writes the three
  vectors. `seq_frac_check`'s lines go to the standard error.

Inputs are libm-free (numpy's generator, multiplies, adds and divides), so
every host builds the same bytes.

## Cases

| case | i2o | i2a | what it covers |
|---|---|---|---|
| `mpas_default` | rearrange | matrix | MPAS's maps; ice fractions 0, -0, 1, 1e-300, the lower bound `-eps` (its `ofrac` exactly `1 + eps`); land and coastal points; a shuffled matrix with repeated elements and empty rows |
| `ocean_matrix_atm_copy` | matrix | copy | an ocean grid of another size |
| `ocean_copy_atm_rearrange` | copy | rearrange | the other two kinds |
| `atm_absent`, `ocn_absent`, `ice_absent` | | | each component absent |
| `dead_reports_all` | rearrange | matrix | dead models: all three checks report, none stops |
| `abort_ice_bound`, `abort_ocn_bound`, `abort_atm_sum`, `abort_atm_lfrin` | | | each check stopping the coupler |
| `abort_ice_float32_eps`, `abort_atm_sum_float32_eps` | | | a bound and the sum between E3SM's single-precision eps and a double 0.01 |
| `abort_i2x_without_si_ifrac` | | | MCT's copy stopping on a missing `Si_ifrac` |

`aborts.json` keeps each stop's line; `<case>/reported` the stages the dead
models report.

Not a case: an ice domain without `frac`. `seq_frac_set` looks it up with
`perrWith`, so MCT only reports it and returns 0, and the product reads
`rAttr(0,:)`, outside the vector. The port refuses it.

NaN inputs are outside the word-for-word claim, and no case has one.

## Grading

`tests/test_e3sm_seq_frac_parity.py` runs `woof.hex.coupled.seq_frac_e3sm`:
every word of the three vectors must be bit-identical, the reported stages
the same, and each stop the same stage or MCT message.
