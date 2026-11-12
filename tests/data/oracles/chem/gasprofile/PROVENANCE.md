# WRF-Chem background gas profile oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`share/module_model_constants.F`, `chem/module_data_radm2.F`,
`chem/module_data_sorgam.F` and `share/module_chem_share.F` are compiled whole
and unchanged.  `chem/module_input_chem_data.F` (public domain, FSL notice) is
rebuilt as module `module_input_chem_data` from its own lines 1-321 (up to
`CONTAINS`: the notice, `#define CASENAME 4`, the USE lines and every module
DATA statement) and `setup_gasprofile_maps`, the four
`setup_gasprofile_map_*` routines, `vinterp_chem`, `input_chem_profile`,
`make_chem_profile`, `bdy_chem_value` and `gasprofile_init_pnnl`, each copied
byte for byte (`routines.list`), nothing edited.  Each file is sha256-checked
against `tools/chem_wrf471_oracle/gasprofile/SOURCES.sha256` before
compilation; module_input_chem_data.F differs from v4.6.1 (`d66e442f`) only in
a comment at :3846-3847, the other four are byte-identical.
`stub_gasprofile.F90` supplies only the Registry-generated
`module_state_description` (radm2 / radm2sorg species indices, package values),
a `module_domain` carrying ph_1, phb, chem, chem_opt and gas_ic_opt, and the
names the other USE lines import.

Cases (`run_gasprofile.F90`):

* `tables`: xl, zfa, zfa_bdy, dens, so4vaptoaer, grav (module_data_sorgam's
  9.80622, which input_chem_profile divides by), iref / fracref after
  setup_gasprofile_maps(chem_opt = 1), get_last_gas for both options.
* `ic_c1_nz{01,03,20,44,60}`, `ic_c2_nz{03,20}`: input_chem_profile on a 6 x 2
  grid with kde = nz + 1; ph_1, phb and the whole chem array after the call
  (rows past ho2 keep their -7 fill).
* `mk_c1`, `mk_c2`: make_chem_profile on 86 hand-picked mass heights.
* `bdy`: bdy_chem_value at the same heights for p_so2 .. p_ho2, chem_opt 1 and 2.
* `pnnl_ic_c1_nz20`, `pnnl_bdy`: gas_ic_opt = 101 (gasprofile_init_pnnl); the
  port refuses this option, the fixtures show what it changes (sulf only).

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.  The reference calls no libm routine (the
arithmetic is sums, products, quotients and MAX); no libmvec `_ZGV*` symbol in
any -O0 object (the positive control produced one).
Driver: `tools/chem_wrf471_oracle/gasprofile/run_gasprofile.F90` (sha256 59ae137c8a2ddf002d1a7f8f490812a8f7a7c5dd99307b05468500b50821312a).
Stub: `stub_gasprofile.F90` (sha256 4c950a41fba58c056d05be023b99cd81c80797df4b892e8b33ee42f17dd4bd0a).

Rebuild:

```sh
docker run --rm -u $(id -u):$(id -g) -v /path/to/WRF-4.7.1:/wrf:ro -v "$PWD":/repo -v /path/to/build:/build \
    -w /repo noah-oracle:u2404 \
    bash tools/chem_wrf471_oracle/gasprofile/build.sh /wrf /build/gasprofile /build/gasprofile/fixtures
```

then copy `/path/to/build/gasprofile/fixtures/gasprofile/*` here (both mounts
are absolute host paths, not named volumes).  What is committed here is the
oracle's numeric output only.  `oracle-sha256sums.txt` is written by the build
and checked by the test.
