# RADM2/SORGAM anthropogenic emission oracle, WRF v4.7.1

`build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]` compiles
chem/module_emissions_anthropogenics.F and the data modules it USEs whole and
unchanged at `-O0`, and copies `aerosols_sorgam_init` and `sorgam_addemiss`
byte for byte from chem/module_aerosols_sorgam.F into a wrapper module named
`module_aerosols_sorgam` (`routines.list`, after `prelude.inc`; nothing is
edited).  `stub_radm2emis.F90` holds the Registry-generated index modules for
the radm2sorg chem package and the eradmsorg emission package (emiss_opt = 3),
stubs for the sea-salt, dust and PNNL-IC branches no case reaches, and the
service hooks.  `run_radm2emis.F90` calls add_anthropogenics then
sorgam_addemiss, in emissions_driver's order, for three cases (dtstep 120, 30,
7.5 s; kemit 4, 6 = the whole column, 1) and writes one case per call through
`oracle_io.F90` (a byte copy of the gocart lane's).

* `SOURCES.sha256` pins every compiled WRF file; a mismatch stops the build.
  All five are byte-identical in v4.6.1.
* Every -O0 object is checked for libmvec `_ZGV*` symbols, and a positive
  control proves the check can fire.
* Fixtures and provenance: `tests/data/oracles/chem/radm2emis/`.  The port
  they grade is `woof/core/chem_radm2_emiss.py` /
  `woof/core/kernels/chem_radm2_emiss.cu`.
