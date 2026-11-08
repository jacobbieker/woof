# RADM2 gas-phase Fortran oracle provenance

WRF v4.6.1, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`.  The four
compiled files (`share/module_model_constants.F`, `chem/module_data_radm2.F`,
`chem/module_data_sorgam.F`, `chem/module_radm.F`) are compiled whole and
unchanged, each sha256-checked against
`tools/chem_wrf461_oracle/radm2/SOURCES.sha256` before compilation; all four
are byte-identical in v4.7.1 (`f52c197e`).  `stub_radm2.F90` supplies only the
Registry-generated `module_state_description` / `module_configure` for the
radm2sorg package (`chem_opt = 2`).

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.  libm words the reference calls: `expf`,
`log10f`, `powf`; no libmvec `_ZGV*` symbol in any -O0 object (the positive
control produced one).

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v WRF-4.6.1:/wrf:ro -v $PWD:/repo -v BUILD:/build \
    -w /repo noah-oracle:u2404 \
    bash tools/chem_wrf461_oracle/radm2/build.sh /wrf /build/radm2 /build/radm2/fixtures
```

then copy `/build/radm2/fixtures/radm2/*` here.  The build writes
`oracle-sha256sums.txt` there: every fixture file's sha256, which the parity
test checks.

Driver: `tools/chem_wrf461_oracle/radm2/run_radm2.F90` (sha256
9b430e4649c34c39b0077cef8dce9f96eb58dcd0e2ff1755cf3e92b3a1eceaa6).  A rebuild from
this tree (2026-11-08) reproduced every fixture byte for byte.
