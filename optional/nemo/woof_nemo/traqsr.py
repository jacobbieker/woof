# WOOF-OPTIN: nemo
"""NEMO's 2-band penetrative solar radiation on device (``traqsr.cu``).

A port of src/OCE/TRA/traqsr.F90 (NEMO 5.0.2) for ``ln_qsr_2bd`` on a linear
free surface (key_linssh) with a 3-D vertical grid (key_vco_3d), leapfrog
time stepping (no key_RK3).  The other light schemes (RGB, 5-band, bio) and
restarts of ``qsr_hc_b`` are refused by name.  ``qsr`` is the net solar flux
into the ocean (W/m2), WRF's SURF_NET_SOLAR on the far side of
:mod:`woof.ocean.wrf_cpl`.

Arrays are float64 device arrays: 2-D ``(jpj, jpi)``, 3-D ``(jpk, jpj, jpi)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TPB = 64
_MODULE = None

#: phycst.F90 (NEMO 5.0.2)
RHO0 = 1026.0
RCP = 3991.86795711963
#: namtra_qsr reference values
RN_ABS, RN_SI0, RN_SI1 = 0.58, 0.35, 23.0


def _module():
    global _MODULE
    if _MODULE is None:
        import cupy as cp

        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt64.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "traqsr.cu").read_text(encoding="utf-8"))
        _MODULE = cp.RawModule(code=source, options=("--fmad=false", "--ftz=false"))
    return _MODULE


class TraQsr2BD:
    """``tra_qsr`` with ``nqsr = np_2BD``; the init mirrors tra_qsr_init's
    np_2BD lines (traqsr.F90:1281-1283, 1318-1322, 1374-1379)."""

    def __init__(self, *, gdepw, e3t, tmask, wmask, rDt, rn_abs=RN_ABS,
                 rn_si0=RN_SI0, rn_si1=RN_SI1, scheme="2bd", restart=False):
        import cupy as cp

        if scheme != "2bd":
            raise NotImplementedError(
                f"NEMO light scheme {scheme!r} is not ported: only ln_qsr_2bd "
                "(np_2BD); RGB, 5-band and bio penetration are refused")
        if restart:
            raise NotImplementedError("reading qsr_hc_b from a NEMO restart is "
                                      "not ported")
        self.jpk, self.jpj, self.jpi = map(int, gdepw.shape)
        for name, a in (("gdepw", gdepw), ("e3t", e3t), ("tmask", tmask), ("wmask", wmask)):
            if tuple(a.shape) != (self.jpk, self.jpj, self.jpi) or a.dtype != cp.float64:
                raise ValueError(f"{name}: expected float64 {gdepw.shape}")
        self.gdepw, self.e3t, self.wmask = gdepw, e3t, wmask
        self.rn_abs = float(rn_abs)
        rho0_rcp = RHO0 * RCP
        self.r1_rho0_rcp = 1.0 / rho0_rcp
        self.r1_si0 = 1.0 / float(rn_si0)
        self.nk0 = self._ext_lev(gdepw, e3t, tmask, wmask, rn_si0, rn_abs, rho0_rcp, rDt)
        self.r1_si1 = 1.0 / float(rn_si1)
        zvlp = 1.0 - float(rn_abs)
        self.nkV = self._ext_lev(gdepw, e3t, tmask, wmask, rn_si1, zvlp, rho0_rcp, rDt)
        self.nksr = self.nkV
        n2 = self.jpi * self.jpj
        self.qsr_hc = cp.zeros((self.jpk, self.jpj, self.jpi), cp.float64)
        self.qsr_hc_b = cp.zeros_like(self.qsr_hc)
        self.fraqsr_1lev = cp.ones((self.jpj, self.jpi), cp.float64)
        self._first = True
        self._grid = ((n2 + _TPB - 1) // _TPB,)

    def _ext_lev(self, gdepw, e3t, tmask, wmask, pL, pfr, rho0_rcp, rDt) -> int:
        import cupy as cp

        out = cp.zeros((1,), cp.int32)
        _module().get_function("nemo_qsr_ext_lev")((1,), (1,), (
            gdepw, e3t, tmask, wmask, np.float64(pL), np.float64(pfr),
            np.float64(rho0_rcp), np.float64(rDt), out, np.int32(self.jpk),
            np.int32(self.jpi * self.jpj)))
        return int(out.get()[0])

    def step(self, qsr, ta_rhs) -> None:
        """One tra_qsr call: adds the solar trend to ``ta_rhs`` (K/s)."""
        _module().get_function("nemo_tra_qsr_2bd")(self._grid, (_TPB,), (
            qsr, self.gdepw, self.e3t, self.wmask, self.qsr_hc, self.qsr_hc_b,
            ta_rhs, self.fraqsr_1lev, np.float64(self.rn_abs),
            np.float64(self.r1_si0), np.float64(self.r1_si1),
            np.float64(self.r1_rho0_rcp), np.int32(self.nk0), np.int32(self.nkV),
            np.int32(self.nksr), np.int32(1 if self._first else 0),
            np.int32(self.jpk), np.int32(self.jpi * self.jpj)))
        self._first = False
