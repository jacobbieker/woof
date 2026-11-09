# WOOF-OPTIN: nemo
"""NEMO's TKE vertical mixing on device (``tke.cu``).

A port of src/OCE/ZDF/zdfsh2.F90 (``zdf_sh2``) and src/OCE/ZDF/zdftke.F90
(``zdf_tke``: tke_tke then tke_avn, with zdf_tke_init's derived constants
and htau) from NEMO 5.0.2.  Accepted configuration: no sea-ice model, no
ice-shelf cavities, bottom friction on, Langmuir cells from the surface
stress, nn_etau 0 or 1, nn_htau 0 or 1, nn_mxl 0-3, nn_pdl 0 or 1, ln_mxl0
either way.  Anything else (an ice model, nn_etau 2 or 3, ln_mxhsw, wave
coupling, internal-wave mixing, ice-shelf cavities, ln_drg_OFF) is refused
by name.  Defaults are namelist_ref's &namzdf_tke.

Fields are float64 (jpk, jpj, jpi) device arrays with NEMO's one-cell halo;
the kernels write interior columns only, as NEMO's loops do.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 64
_MODULE = None
#: cfgs/SHARED/namelist_ref, &namzdf_tke (NEMO 5.0.2)
NAMELIST_REF = dict(rn_ediff=0.1, rn_ediss=0.7, rn_ebb=67.83, rn_emin=1.e-6, rn_emin0=1.e-4,
                    rn_bshear=1.e-20, nn_pdl=1, nn_mxl=3, ln_mxl0=True, nn_mxlice=0,
                    rn_mxlice=10., rn_mxl0=0.04, ln_mxhsw=False, ln_lc=True, rn_lc=0.15,
                    nn_etau=1, rn_efr=0.05, nn_htau=1, nn_eice=1, nn_bc_surf=1, nn_bc_bot=1)
TKE_CONST = ("rn_ediff", "rn_ediss", "rn_ebb", "rn_emin", "rn_emin0", "rn_bshear", "rn_mxl0",
             "rn_lc", "rn_efr", "rn_Dt", "rho0", "grav", "vkarmn", "rpi", "rsmall", "ri_cri",
             "rmxl_min")
#: phycst.F90 and eos_init
PHYCST = dict(rho0=1026.0, grav=9.80665, vkarmn=0.4, rpi=3.141592653589793,
              rsmall=float(np.float32(0.5) * np.finfo(np.float32).eps))


def _module():
    global _MODULE
    if _MODULE is None:
        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt64.cuh").read_text(encoding="utf-8") + "\n"
                  + (kernels / "f64cr.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "tke.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false",))
    return _MODULE


class TkeMixing:
    def __init__(self, *, rn_Dt, gphit, nn_ice=0, ln_isfcav=False, ln_drg_off=False,
                 ln_zdfiwm=False, wave_coupling=False, **namelist):
        import cupy as cp

        nl = {**NAMELIST_REF, **namelist}
        unknown = sorted(set(namelist) - set(NAMELIST_REF))
        if unknown:
            raise ValueError(f"not &namzdf_tke keys: {unknown}")
        if nn_ice != 0:
            raise NotImplementedError("TKE under sea ice (nn_eice, nn_mxlice with an ice model) "
                                      "is not ported: there is no sea-ice model")
        if ln_isfcav:
            raise NotImplementedError("ice-shelf cavities (ln_isfcav) are not ported")
        if ln_drg_off:
            raise NotImplementedError("ln_drg_OFF is not ported")
        if ln_zdfiwm:
            raise NotImplementedError("internal-wave-driven mixing (ln_zdfiwm) is not ported")
        if wave_coupling:
            raise NotImplementedError("wave coupling (Stokes drift, phioc) is not ported")
        if nl["ln_mxhsw"]:
            raise NotImplementedError("ln_mxhsw (mixing length from wave height) is not ported")
        if nl["nn_etau"] in (2, 3):
            raise NotImplementedError(f"nn_etau = {nl['nn_etau']} is not ported (0 and 1 are)")
        for key, ok in (("nn_mxl", (0, 1, 2, 3)), ("nn_pdl", (0, 1)), ("nn_htau", (0, 1)),
                        ("nn_etau", (0, 1))):
            if nl[key] not in ok:
                raise ValueError(f"{key} = {nl[key]} (zdf_tke_init stops)")
        # zdf_tke_init (:771, :826-844): with no ice model nn_eice = 0.
        ri_cri = 2.0 / (2.0 + nl["rn_ediss"] / nl["rn_ediff"])
        rmxl_min = 1.e-6 / (nl["rn_ediff"] * math.sqrt(nl["rn_emin"]))
        rn_mxl0 = rmxl_min if nl["ln_mxl0"] else nl["rn_mxl0"]
        consts = {**nl, **PHYCST, "rn_Dt": float(rn_Dt), "ri_cri": ri_cri,
                  "rmxl_min": rmxl_min, "rn_mxl0": rn_mxl0}
        self._cfd = cp.asarray(np.array([consts[k] for k in TKE_CONST], np.float64))
        self.flags = tuple(np.int32(v) for v in (nl["nn_mxl"], nl["nn_pdl"], int(nl["ln_mxl0"]),
                                                  int(nl["ln_lc"]), nl["nn_etau"]))
        self.rn_emin = nl["rn_emin"]
        jpj, jpi = map(int, gphit.shape)
        self.htau = cp.zeros((jpj, jpi), cp.float64)
        if nl["nn_etau"] != 0:
            _module().get_function("nemo_tke_htau")((((jpi * jpj + 255) // 256),), (256,), (
                gphit, self.htau, np.float64(PHYCST["rpi"]), np.int32(nl["nn_htau"]),
                np.int32(jpi * jpj)))

    def start_from_rest(self, wmask):
        """tke_rst 'READ' without a restart (:910-913): en = rn_emin*wmask,
        dissl = 1.e-12."""
        import cupy as cp

        en = cp.ascontiguousarray(self.rn_emin * wmask)
        dissl = cp.full(wmask.shape, 1.e-12, cp.float64)
        return en, dissl

    def sh2(self, *, avm, uu_b, uu_n, vv_b, vv_n, e3uw, e3vw, wumask, wvmask, umask, vmask, sh2):
        jpk, jpj, jpi = map(int, avm.shape)
        n = (jpi - 2) * (jpj - 2)
        _module().get_function("nemo_zdf_sh2")((((n + _TPB - 1) // _TPB),), (_TPB,), (
            avm, uu_b, uu_n, vv_b, vv_n, e3uw, e3vw, wumask, wvmask, umask, vmask, sh2,
            np.int32(jpi), np.int32(jpj), np.int32(jpk)))

    def step(self, *, sh2, avm, avt, en, dissl, rn2, rn2b, taum, rCdU_bot, uu_b, vv_b,
             umask, vmask, tmask, wmask, ssmask, mbkt, e3t, e3w, gdepw, avmb, avtb, avtb_2d):
        """One zdf_tke call (tke_tke, then tke_avn)."""
        jpk, jpj, jpi = map(int, avm.shape)
        if jpk >= 128:
            raise ValueError("the port carries at most 127 levels")
        n = (jpi - 2) * (jpj - 2)
        _module().get_function("nemo_zdf_tke")((((n + _TPB - 1) // _TPB),), (_TPB,), (
            self._cfd, *self.flags, sh2, avm, avt, en, dissl, rn2, rn2b, taum, rCdU_bot, uu_b,
            vv_b, umask, vmask, tmask, wmask, ssmask, mbkt, e3t, e3w, gdepw, self.htau, avmb,
            avtb, avtb_2d, np.int32(jpi), np.int32(jpj), np.int32(jpk)))
