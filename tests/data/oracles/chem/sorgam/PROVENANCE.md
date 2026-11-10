# MADE/SORGAM Fortran oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`share/module_model_constants.F`, `chem/module_data_radm2.F` and
`chem/module_data_sorgam.F` compile whole and unchanged.  The routines in
`tools/chem_wrf471_oracle/sorgam/routines.list` (sorgam_driver's call tree
and aerosols_sorgam_init) are copied byte for byte from
`chem/module_aerosols_sorgam.F` into one wrapper module, with one
documented change: aeroproc's `isorop=1` (line 1580) becomes `isorop=0`,
WRF's own alternative on line 1579.  isorop = 1 calls ISORROPIA
(`chem/isocom.F`), whose source carries "COPYRIGHT 1996-2006, UNIVERSITY OF
MIAMI, CARNEGIE MELLON UNIVERSITY, GEORGIA INSTITUTE OF TECHNOLOGY" and no
licence grant; WOOF's default build runs SORGAM's RPMARES branch.  All four
files are sha256-checked against `SOURCES.sha256` and are byte-identical in
v4.6.1 (`d66e442f`, the lane's first pin).  The build extracts WRF's code into its build directory
only; what is committed here is the oracle's numeric output (float32 arrays),
no code.

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.  libm words the reference calls: `expf`,
`logf`, `powf`, binary64 `exp` and `pow`, libgcc `__powisf2`; no libmvec
`_ZGV*` symbol in any -O0 object.  The cases execute 88 % of the extracted
lines (gfortran --coverage, `ORACLE_COVERAGE=1`); the unexecuted ones are
wrf_error_fatal paths, aerosols_sorgam_init past its chem_in_opt = 1 return,
and rare solver fallbacks.

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v WRF-4.7.1:/wrf:ro -v $PWD:/repo -v BUILD:/build \
    -w /repo noah-oracle:u2404 \
    bash tools/chem_wrf471_oracle/sorgam/build.sh /wrf /build/sorgam /build/sorgam/fixtures
```

then copy `/build/sorgam/fixtures/sorgam/*` here; the build writes this
directory's `oracle-sha256sums.txt`, which the parity test checks.

Driver: `tools/chem_wrf471_oracle/sorgam/run_sorgam.F90` (sha256 19602a664085603c193414d5a077b830719716345c0a6317f8080b724f76aa68).
