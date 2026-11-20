WRF v4.7.1 single-layer UCM with gridded morphology (the NUDAPT arm,
phys/module_sf_urban.F:649-788) and urban_var_init's keep/zero rule
(:2788-2833), from the byte-unmodified module (SHA-256 623868c7...0b0ea,
pinned in tools/urban_wrf471_oracle/SOURCES.sha256 with the two tables).

Built by tools/urban_wrf471_oracle/build_ucm_nudapt.sh with the driver
ucm_nudapt_oracle.F90 (driver-sha256sums.txt), gfortran 16.2.0
(conda-forge gcc 16.2.0-7) at -O0, glibc 2.44 (compiler.txt). The same
toolchain rebuilds every committed tests/data/oracles/urban/ucm fixture
byte for byte (they were written by gfortran 15.2.0 + glibc 2.43).

* nudapt-{nlcd,lcz}-main: every urban category x 4 morphologies x 8
  forcing scenarios x 6 carried steps; the wind turns through every
  lambda_f sector, exact axes included. The `snan` build equals `ref` on
  every row.
* nudapt-nlcd-probe: four degenerate morphologies. Sets 1 (LP > FRC,
  negative street width) and 2 (STDH = 0, Z0R = 0) stay finite and are
  graded; sets 3 (an LF sector = 0) and 4 (LP = 0 below LB) go non-finite
  in WRF itself, which is why woof/core/urban_param.py refuses them.
* init-opt{1,2,3}-lcz{0,1}.csv: urban_var_init over eight columns (kept,
  HGT = 0, HGT < 0, LCZ, non-urban) for each option.

WRF's undefined U10/V10 read at :709 is defined: the driver stores the
column's own U1/V1 into U10/V10 before each call (see the driver header).
