# activate oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`share/module_model_constants.F` compiles whole and unchanged. The routines in
`oracle/activate/routines.list` (`activate`, `maxsat`, `maxsat_init`,
`erfc_num_recipes`, `erf_alt`) are copied byte for byte from
`phys/module_mixactivate.F` into one wrapper module, with no change. Both files
are sha256-checked against `oracle/activate/SOURCES.sha256`. The build extracts
WRF's code into its own build directory only. What is committed here is the
oracle's numeric output (float32 and int32 arrays) and no code.

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.

Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).

Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1
-ffree-form -ffree-line-length-none -fallow-argument-mismatch`.

libm words the reference calls: `expf`, `logf`, `powf`, binary64 `exp`
(`erfc_num_recipes`), and libgcc `__powisf2`. No -O0 object has a libmvec
`_ZGV*` symbol.

The 112 cases (`run_activate.F90`, sha256
`b166554da875082001a91dc69cd46babeaa42808120dec9ae92f24bc8f442663`) execute 75 %
of the extracted lines (gfortran --coverage, measured with gfortran 16 on the
mac). The unexecuted lines are:

- the diagnostics (print and write);
- `all_bins_empty`;
- the cloud-top branch (`top` is hard-wired `.false.`);
- the integration's second, diagnostic pass when 200 steps do not suffice;
- `isectional` 3/4 (unreachable: `iquasisect_option = 2`);
- some sectional limits (fn or fm outside [0, 1]);
- `fn > 1.01` flux renormalisations.

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v WRF-4.7.1:/wrf:ro -v $PWD:/repo -v BUILD:/build \
    -w /repo noah-oracle:u2404 \
    bash optional/wrfchem_pnnl/oracle/activate/build.sh /wrf /build/activate /build/activate/fixtures
```

Then copy `/build/activate/fixtures/activate/*` here. The build writes this
directory's `oracle-sha256sums.txt`, which the parity test checks.
