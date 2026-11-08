# Noah-MP column oracle fixtures

Produced by `tools/noahmp_wrf461_oracle/column` (Ubuntu 24.04, gfortran
13.3.0, glibc 2.39, x86-64 baseline, no FMA in the binary), CPU only, from two
unmodified WRF trees. `build_column.sh` accepts either one and refuses any
other (`../noahmp_pins.sh`).

| | WRF v4.6.1 | WRF v4.7.1 |
|---|---|---|
| WRF commit | d66e442fccc04111067e29274c9f9eaccc3cef28 | f52c197ed39d12e087d02c50f412d90d418f6186 |
| phys/noahmp submodule | 848f54ad3d28c4303151fe5ad83724e232694422 | e5c0859874407859936739e8be8741f9aed369ee |
| phys/noahmp/src/module_sf_noahmplsm.F | bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282 | 74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b |
| phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F | 9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890 | 29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572 |
| phys/noahmp/src/module_sf_noahmp_glacier.F | bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579 | a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece |
| phys/noahmp/parameters/MPTABLE.TBL | 7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d | same |
| run/SOILPARM.TBL | 1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f | same |
| run/GENPARM.TBL | 9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2 | same |

Build: `build_column.sh <wrf> <dir> noopt` (-O0; no vector-math call in any
Noah-MP object).

| fixture (gzip -9 -n of) | tree | uncompressed sha256 |
|---|---|---|
| in-summer60.bin | | d7c27893a250c58b77b221c272cddb6a3132b16e48e8528ce720b219d7f4d84d |
| in-winter300.bin | | d2951adc02291910bf5038fc108efd73c6d34590851c6225e2fed1048e9d0710 |
| wrf461-O0-summer60.bin | 848f54ad | ab744240d8aa023663d4f086d6e49bd8a0dfb6a878b02965d35138bd00aa01fd |
| wrf461-O0-winter300.bin | 848f54ad | f0fa1e9087355034331f4037a9e211890a66f321937e2869cc811ca2296162ce |
| wrf471-O0-summer60.bin | e5c0859 | 6ac88daf7c14a218a987b2d19e13d9cab309c0fd3ea6d2d8838fd8fc420ac7c2 |
| wrf471-O0-winter300.bin | e5c0859 | 326745af893e3370ed596a5dbad97fd7be20c6dbfb46f4cd2f3aa4f49dae87b4 |

The `wrf461-O0-*` files were rebuilt from the 4.6.1 tree on 2026-11-08 and are
byte-identical to the committed ones. They stay the references for WOOF's
`noahmp_snow_age = "wrf_461"`.

The `wrf471-O0-*` files are the same inputs through the 4.7.1 tree, added
beside them because e5c0859 changes two things these columns observe:

* TAUSSXY on the night snow columns (coszen = 0: summer60 columns 23, 25, 26,
  27, 30, 33, 38 and 44; winter300 columns 23, 27, 28, 32, 33, 34, 35, 36 and
  44, 1-based as in `regimes-*.txt`). e5c0859 calls SNOW_AGE and
  SNOW_AGE_GLACIER before the `cosz > 0` test, so snow ages at night;
  848f54ad aged it only in daylight. 280 and 195 values.
* SOILENERGY and SNOWENERGY on the six glacier columns (43-48), every step
  after the cold start. e5c0859's NOAHMP_GLACIER returns HCPCT, so the driver
  no longer forms them from a local the glacier arm never set. 418 and 249
  values.

No other field moves. They are graded against WOOF's default
`noahmp_snow_age = "wrf_470"`, with the glacier SOILENERGY/SNOWENERGY
compared rather than excluded (`compare_column.GLACIER_UNDEFINED_E5C0859`);
only EFLXBXY stays excluded, because EFLXB is still never set on the glacier
arm. WOOF's device path matches both pairs word for word with no change
(`tests/test_noahmp_wrf461_column_oracle.py`).

The input files regenerate byte-identically from `gen_columns.py` (checked on
the box's Linux numpy and the desktop's Windows numpy). `regimes-*.txt` names
each column's regime, vegetation and soil class.
