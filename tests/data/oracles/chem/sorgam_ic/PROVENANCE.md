# MADE/SORGAM initial-condition oracle provenance

Built by `tools/chem_wrf461_oracle/sorgam/build.sh` with the `sorgam` and
`sorgam_dep` fixtures (WRF v4.6.1 `d66e442f`, same pins and compiler:
GNU Fortran 13.3.0, Ubuntu GLIBC 2.39, `noah-oracle:u2404`, -O0).
`aerosols_sorgam_init` and `sorgam_init_aer_ic_pnnl` are copied byte for
byte and unchanged.  Driver: `tools/chem_wrf461_oracle/sorgam/run_sorgam_ic.F90`
(sha256 9ad99c9c8b035d64c5bf70fa37fd510f5136d7b075afa0b5a56692566740c016).  Numeric output only; the rebuild left the other two fixture
sets byte-identical.  `oracle-sha256sums.txt` is written by the build.
