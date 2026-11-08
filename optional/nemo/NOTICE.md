# NEMO opt-in: attribution and provenance

Everything under `optional/nemo/` is an **opt-in**: off by default, in no
wheel (`metal-woof`, `recast-woof`), in no default build, and excluded from the
release snapshot (`RELEASE-EXCLUDE.txt`).  It loads only through
`woof.optin.load("nemo")` with `WOOF_ENABLE_NEMO=1` in a source checkout.  The
project owner approved adding it on those terms on 2026-11-07.

## Licence

The code here is derived from NEMO, "Nucleus for European Modelling of the
Ocean", and is governed by the **CeCILL Free Software License Agreement,
version 2.0** (2006-09-05), whose full text is `LICENSE` in this directory,
copied verbatim (ISO-8859-1) from NEMO tag `5.0.2`
(`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`), `LICENSE` sha256
`2dcf0966f7eaae3af05f3e640215ac537c76d938d50ad02ad9d4260f1ca4bda5`.  NEMO's
source headers state "Software governed by the CeCILL license (see
./LICENSE)".  The rest of WOOF is Apache-2.0; this directory is not.

CeCILL is a copyleft.  This directory is modified software (a CUDA
transcription with a Python launcher, not NEMO's source) and is distributed
under CeCILL v2.0 (Article 5.3.2).  Code that runs in the same address space
as it is an Internal Module under the licence, so **a WOOF build or process
image that loads these kernels may be redistributed only under CeCILL v2.0,
or under the GNU GPL through Article 5.3.4, as a whole.**  That is why it is
never in a wheel.  The licence's warranty and liability limitations
(Articles 8 and 9) apply.

NEMO is developed by the NEMO System Team on behalf of the NEMO Consortium
(CNRS, Mercator Ocean International, UKMO, NOC, CMCC, INGV).  No endorsement
by them is implied.

## Source

| WOOF file | NEMO 5.0.2 source |
| --- | --- |
| `woof_nemo/traqsr.cu`, `traqsr.py` | `src/OCE/TRA/traqsr.F90`: `tra_qsr` (MLF path), `qsr_2BD`, `qsr_ext_lev`, and the np_2BD lines of `tra_qsr_init`; constants from `src/OCE/DOM/phycst.F90` |
| `woof_nemo/eos.cu`, `eos.py`, `eos_coeffs.py` | `src/OCE/TRA/eosbn2.F90`: `eos_insitu_pot_New_t` (non-stochastic), `rab_3d_t`, `bn2_t`, and `eos_init`'s TEOS-10 and EOS-80 coefficient tables (`eos_coeffs.py` is generated from the pinned file by `oracle/gen_eos_coeffs.py`, and `oracle/build_eos.sh` refuses a committed copy that differs) |
| `woof_nemo/tke.cu`, `tke.py` | `src/OCE/ZDF/zdfsh2.F90` (zdf_sh2); `src/OCE/ZDF/zdftke.F90` (tke_tke, tke_avn, zdf_tke_init's constants and htau, tke_rst's start from rest); `cfgs/SHARED/namelist_ref` (&namzdf_tke defaults) |
| `woof_nemo/trazdf.cu`, `trazdf.py` | `src/OCE/TRA/trazdf.F90` (tra_zdf, tra_zdf_imp) |
| `oracle/` | build harnesses, service stubs and drivers (`build.sh`, `build_eos.sh`, `build_tke.sh`, `build_trazdf.sh`); they compile the pinned files at build time and copy none of them |

`traqsr.cu` is preceded at load time by WOOF's
`woof/core/kernels/glibc_flt64.cuh` (glibc's binary64 exp/log words, MIT
licensed Arm code, see that file); `tke.cu` also by WOOF's own
`woof/core/kernels/f64cr.cuh` (correctly rounded binary64 sin).  Nothing of
either is copied here.

## Oracle

`oracle/build.sh NEMO_SOURCE_ROOT BUILD_DIR` cuts `tra_qsr`, `qsr_2BD` and
`qsr_ext_lev` byte for byte from the pinned `traqsr.F90`, compiles them at
`-O0` with NEMO's own `do_loop_substitute.h90`, `domzgr_substitute.h90`
(key_linssh, key_vco_3d) and `par_kind.F90`, and drives three leapfrog steps.
Toolchain: gfortran 13.3.0, glibc 2.39 (Ubuntu 24.04 container), 2026-11-08.
The gate is `tests/test_nemo_traqsr_parity.py`.
