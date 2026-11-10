# Fast-TUV photolysis Fortran oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.  Six files
are compiled whole and unchanged: `share/module_model_constants.F`,
`chem/module_data_radm2.F`, `chem/module_data_sorgam.F`,
`chem/module_wave_data.F`, `chem/module_ftuv_subs.F` and
`chem/module_ftuv_driver.F`.  Each is sha256-checked against
`tools/chem_wrf471_oracle/ftuv/SOURCES.sha256` before compilation.  The driver
differs from v4.6.1 only in ftuv_timestep_init, which only MOZART packages call.
`stub_ftuv.F90` supplies only the Registry-generated
`module_state_description` / `module_configure` (the radm2sorg species
indices and the package chem_opt values) and the wrf_debug, wrf_message and
wrf_abort entry points.  NETCDF is undefined.  It guards only MOZART code.

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.

libm words the reference calls:
- binary64 `exp`, `log`, `log10`, `pow`, `sin`, `cos`, `tan`, `atan` and `acos`;
- libgcc `__powidf2`;
- the driver's own float32 `expf` and `__powisf2`.

No -O0 object contains a libmvec `_ZGV*` symbol.

Cases: one `ftuv_driver` call each, after `ftuv_init` for domain 1 (photo_inti
with kte = 30, so the module nz is 45):

| case | chem_opt | aer_ra_feedback | julday | gmt | curr_secs |
|---|---|---|---|---|---|
| case1 | 2 | 0 | 172 | 12 | 0 |
| case2 | 2 | 0 | 172 | 12 | 5400 |
| case3 | 2 | 1 | 15 | 6 | 0 |
| case4 | 1 | 0 | 300 | 18 | 0 |
| case5 | 2 | 1 | 1 | 23 | 3000 |

The 12 columns cover:
- latitudes from -62 to 74 degrees;
- high sun, low sun, twilight and night;
- cloud water, including a value under the 1e-6 g/m3 floor;
- a saturated layer (the RH clamp);
- different terrain heights.

Each case writes its inputs and all 47 ph_* fields.  ph_* fields the RADM2
packages leave unassigned stay at the driver's -999 fill.

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v WRF-4.7.1:/wrf:ro -v $PWD:/repo -v BUILD:/build \
    -w /repo noah-oracle:u2404 \
    bash tools/chem_wrf471_oracle/ftuv/build.sh /wrf /build/ftuv /build/ftuv/fixtures
```

Then copy `/build/ftuv/fixtures/ftuv/*` here.  The build writes this
directory's `oracle-sha256sums.txt`, which the parity test checks.

Driver: `tools/chem_wrf471_oracle/ftuv/run_ftuv.F90` (sha256
2667b22afaf2b25a15048b22cdcaaf6d0153dea18ca53ed1f32c6f3e9dd40c78).  Only the
oracle's numeric output (float32 arrays) is committed here, no code.
