# WRF-Chem background gas profile oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`share/module_model_constants.F`, `chem/module_data_radm2.F`,
`chem/module_data_sorgam.F` and `share/module_chem_share.F` are compiled whole
and unchanged.  `chem/module_input_chem_data.F` (public domain, FSL notice) is
rebuilt as module `module_input_chem_data` from its own lines 1-321 (up to
`CONTAINS`: the notice, `#define CASENAME 4`, the USE lines and every module
DATA statement) and `setup_gasprofile_maps`, the four
`setup_gasprofile_map_*` routines, `vinterp_chem`, `input_chem_profile`,
`make_chem_profile`, `bdy_chem_value_sorgam`, `bdy_chem_value` and
`gasprofile_init_pnnl`, each copied
byte for byte (`routines.list`), nothing edited.  Each file is sha256-checked
against `tools/chem_wrf471_oracle/gasprofile/SOURCES.sha256` before
compilation; module_input_chem_data.F differs from v4.6.1 (`d66e442f`) only in
a comment at :3846-3847, the other four are byte-identical.
`stub_gasprofile.F90` supplies only the Registry-generated
`module_state_description` (radm2 / radm2sorg species indices, package values),
a `module_domain` carrying ph_1, phb, chem, chem_opt and gas_ic_opt and a
`grid_config_rec_type` carrying chem_opt and aer_bc_opt, the names the other
USE lines import, and external stand-ins for `sorgam_set_aer_bc_pnnl` /
`sorgam_vbs_set_aer_bc_pnnl` (aer_bc_opt = 101, not extracted) that stop if
reached.

Cases (`run_gasprofile.F90`):

* `tables`: xl, zfa, zfa_bdy, dens, so4vaptoaer, grav (module_data_sorgam's
  9.80622, which input_chem_profile divides by), iref / fracref after
  setup_gasprofile_maps(chem_opt = 1), get_last_gas for both options.
* `ic_c1_nz{01,03,20,44,60}`, `ic_c2_nz{03,20}`: input_chem_profile on a 6 x 2
  grid with kde = nz + 1; ph_1, phb and the whole chem array after the call
  (rows past ho2 keep their -7 fill).
* `mk_c1`, `mk_c2`: make_chem_profile on 86 hand-picked mass heights.
* `bdy`: bdy_chem_value at the same heights for p_so2 .. p_ho2, chem_opt 1 and 2.
* `sorgam_bdy`: bdy_chem_value_sorgam (radm2sorg, aer_bc_opt = 1) for
  nch = p_so4aj .. p_corn (the 38 aerosol rows) at the `bdy` heights plus
  every zfa_bdy mass level (the routine's own grid), its float32 neighbours,
  midpoints, 10% points, -2 km, 35 km and 50 km, each in four atmospheres;
  t (perturbation theta), p, pb, alt and convfac are written, convfac made
  with flow_dep_bdy_chem's own two lines (:1751-1752).
* `sorgam_tables`: the module_data_sorgam PARAMETERs the routine reads as
  this compiler folds them (pirs, f6dpi, f6dpim9, the seven `*fac`, conmin,
  `dgini**3`, the constant parts of m3nuc / m3cor), and the module variables
  the routine sets (xxlsg*, l2sgini*, en1 / ea1 / ec1, esn36 / esa36 / esc36).
* `pnnl_ic_c1_nz20`, `pnnl_bdy`: gas_ic_opt = 101 (gasprofile_init_pnnl); the
  port refuses this option, the fixtures show what it changes (sulf only).

Reference compiler: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
Reference libc: Ubuntu GLIBC 2.39-0ubuntu8.9 (docker image `noah-oracle:u2404`).
Flags: `-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch`.  The profile routines call no libm routine
(sums, products, quotients and MAX); bdy_chem_value_sorgam calls `expf` for
en1 / ea1 / ec1 and libgcc's `__powisf2` for `en1**4` (its `log(sgini*)` and
`dgini*`**3 of PARAMETERs are folded at compile time, no `logf` is linked), and
the driver calls `expf` / `powf` to make its atmospheres; no libmvec `_ZGV*` symbol in
any -O0 object (the positive control produced one).
Driver: `tools/chem_wrf471_oracle/gasprofile/run_gasprofile.F90` (sha256 ee990404eb8441405c15c2e7cb7459dda945370f23908caf45e817d6837c6039).
Stub: `stub_gasprofile.F90` (sha256 6369ab5ece2564cd74f237de2d879598e85b5fed691d4efe7fa6f2463d7bf828).

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
