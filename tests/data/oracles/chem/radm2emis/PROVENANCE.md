# RADM2/SORGAM anthropogenic emission oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`share/module_model_constants.F`, `chem/module_data_radm2.F`,
`chem/module_data_sorgam.F` and `chem/module_emissions_anthropogenics.F`
are compiled whole and unchanged; `aerosols_sorgam_init` and
`sorgam_addemiss` are copied byte for byte from
`chem/module_aerosols_sorgam.F` (`routines.list`), nothing edited.  Each file
is sha256-checked against `tools/chem_wrf471_oracle/radm2emis/SOURCES.sha256`
before compilation; all five are byte-identical in v4.6.1 (`d66e442f`).
`stub_radm2emis.F90` supplies only the Registry-generated
`module_state_description` / `module_configure` (radm2sorg chem rows,
eradmsorg emission rows, chem_opt = 2, emiss_opt = 3) and stubs for branches
no case reaches.

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.  libm words the reference calls: `expf`, `powf`
and `__powisf2`, all inside `aerosols_sorgam_init` (the emission adds
themselves are products, quotients and sums); no libmvec `_ZGV*` symbol in
any -O0 object (the positive control produced one).
Driver: `tools/chem_wrf471_oracle/radm2emis/run_radm2emis.F90` (sha256 901d2e11170a3f5f10f65752092b46dc52f2d2c79ef674f7b7640cfdce41bfb1).

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v /path/to/WRF-4.7.1:/wrf:ro -v "$PWD":/repo -v /path/to/build:/build \
    -w /repo noah-oracle:u2404 \
    bash tools/chem_wrf471_oracle/radm2emis/build.sh /wrf /build/radm2emis /build/radm2emis/fixtures
```

then copy `/path/to/build/radm2emis/fixtures/radm2emis/*` here (both mounts are absolute host paths, not named volumes).  What is committed
here is the oracle's numeric output only.  `oracle-sha256sums.txt` is
written by the build and checked by the test.
