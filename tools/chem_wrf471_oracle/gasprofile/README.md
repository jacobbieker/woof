# WRF-Chem background gas profile oracle (chem_opt = 1/2, chem_in_opt = 0), WRF v4.7.1

`build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]` compiles share/module_model_constants.F,
chem/module_data_radm2.F, chem/module_data_sorgam.F and share/module_chem_share.F
whole and unchanged at `-O0`, and rebuilds module `module_input_chem_data` from
chem/module_input_chem_data.F's own lines 1 to `CONTAINS` (FSL notice,
`#define CASENAME 4`, USE lines, every module DATA statement) plus the routines in
`routines.list`, each copied byte for byte (nothing is edited).
`stub_gasprofile.F90` holds the Registry-generated `module_state_description`
(chem_opt / gas_ic_opt package values, the radm2 and radm2sorg species indices as
module variables), a `module_domain` with only the fields `input_chem_profile`
reads, and the names the remaining USE lines import.  `run_gasprofile.F90` calls
`input_chem_profile` (real.exe's chem_in_opt = 0 entry), `make_chem_profile` and
`bdy_chem_value` for chem_opt 1 and 2, then the gas_ic_opt = 101 path, and writes
one case per call through `oracle_io.F90` (a byte copy of the radm2emis lane's).

* `SOURCES.sha256` pins every compiled WRF file; a mismatch stops the build.
  module_input_chem_data.F differs from v4.6.1 only in a comment at :3846-3847,
  outside every compiled line; the other four are byte-identical.
* Every -O0 object is checked for libmvec `_ZGV*` symbols, and a positive
  control proves the check can fire.
* Fixtures and provenance: `tests/data/oracles/chem/gasprofile/`.  The port they
  grade is `woof/core/chem_gas_profile.py` (tables in
  `woof/core/chem_gas_profile_data.py`, written by
  `tools/chem_gas_profile/transcribe.py`).
