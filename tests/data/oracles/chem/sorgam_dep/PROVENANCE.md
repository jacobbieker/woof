# MADE/SORGAM dry-deposition and PM-sum oracle provenance

Built by `tools/chem_wrf471_oracle/sorgam/build.sh` together with the
`sorgam` fixtures (same WRF v4.7.1 tree `f52c197e`, same pins, same
compiler: GNU Fortran 13.3.0, Ubuntu GLIBC 2.39, `noah-oracle:u2404`, -O0).
`sorgam_depdriver`, `vdvg`, `vdvg_2` and `sum_pm_sorgam` are copied byte
for byte from `chem/module_aerosols_sorgam.F` (`routines.list`); none of
them is changed.  Driver: `tools/chem_wrf471_oracle/sorgam/run_sorgam_dep.F90`
(sha256 8b7f0000db02e828dc75c6b0af389e1dda03629f6700f252c781991771cb013e).  What is committed here is the oracle's numeric output only.
The rebuild that published these left the `sorgam` fixtures byte-identical.
`oracle-sha256sums.txt` is written by the build and checked by the test.
