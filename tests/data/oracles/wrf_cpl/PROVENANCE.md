# WRF ocean-coupler exchange oracle, WRF v4.7.1

`wrf-cpl.csv` is written by `tools/wrf_cpl_wrf461_oracle/build.sh` from the
WRF v4.7.1 tree pinned at `f52c197ed39d12e087d02c50f412d90d418f6186`;
`build.sh` refuses any other commit and any edit to the two sources it reads.
(Re-pinned from v4.6.1, `d66e442f...`, on 2026-11-08: `frame/module_cpl.F`
and `share/module_model_constants.F` are byte-identical between the two
tags and `wrf-cpl.csv` rebuilt byte-identical; only the labels and
`build.sh`'s own digest moved.)

`frame/module_cpl.F` USEs WRF's Registry-generated `module_domain`, so it
cannot be compiled whole outside WRF.  `build.sh` cuts `cpl_snd2`
(`:298-382`, 85 lines) and `cpl_rcv` (`:408-455`, 48 lines) out of the pinned
file, from the `SUBROUTINE` line to the `END SUBROUTINE` line, and
`#include`s the cut text unchanged in `cpl_harness.F90`.  The harness supplies
a `TYPE(domain)` with exactly the components those bodies read, field-id and
send/receive switches that couple every field with every external domain, and
the OASIS put/get replaced by a copy into or out of a harness array.  Nothing
in it computes a flux.  `STBOLT` comes from the pinned
`share/module_model_constants.F`, compiled.

64 independent columns, two external ocean domains.  Inputs are spread by an
integer hash (no libm), plus named edge columns: no relative wind (1, 2) and a
relative wind below WRF's 1.e-7 floor (3), no friction velocity (4), night
with dewfall (5), a cloudburst (6), freezing and warm-pool SST (7, 8), a storm
over a strong current (9).  The masks cycle through none, either domain alone,
both, and partial overlaps, every pair summing to at most 1.

Every value is either a word `run_cpl.F90` placed in an input array or a word
`cpl_snd2` (via `cpl_sndfield`) or `cpl_rcv` placed in an output array.

`libm-report.txt`: the -O0 object calls libgcc's `__powisf2` (for
`SST**4`) and nothing from libm or libmvec.  `oracle-sha256sums.txt` hashes
the two WRF sources, the two cuts, the harness and the fixture.

Toolchain: gfortran 13.3.0, glibc 2.39-0ubuntu8.9 (Ubuntu 24.04 container),
2026-11-07.
