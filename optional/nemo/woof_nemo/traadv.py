# WOOF-OPTIN: nemo
"""NEMO's flux-corrected transport (FCT) tracer advection on device (``traadv.cu``).

A port of src/OCE/TRA/traadv_fct.F90 (``tra_adv_fct`` with ``fct_up1_1stp``,
``nonosc`` and ``interp_4th_cpt``) from NEMO 5.0.2: an upstream step, the
centred anti-diffusive fluxes (``nn_fct_h`` 2 or 4; ``nn_fct_v`` 2, or 4 for
the compact scheme), Zalesak's limiter, and the limited trend added to the
RHS.  Leapfrog (no key_RK3), linear free surface.  Refused by name:
adaptive-implicit vertical advection (ln_zad_Aimp, nn_fct_imp), ice-shelf
cavities, trend and transport diagnostics, and tiling (NEMO refuses it too).

``pU``/``pV``/``pW`` are the transports tra_adv hands the scheme (m3/s:
e2u*e3u*uu, e1v*e3v*vv, e1e2t*ww); the flow that makes them is NEMO's
dynamics, which is not ported.  Fields are float64 (jpk, jpj, jpi) device
arrays with NEMO 5.0's two-cell halo (``nn_hls = 2``); the boundaries are
closed, as NEMO's single-process lbc_lnk makes them (halo filled with 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 64
_MODULE = None
H = 2   # nn_hls (mppini.F90: nn_hls = MAX(2, nn_hls))


def _module():
    global _MODULE
    if _MODULE is None:
        source = (Path(__file__).resolve().parent / "traadv.cu").read_text(encoding="utf-8")
        _MODULE = raw_module(source, ("--fmad=false",))
    return _MODULE


class TracerAdvectionFct:
    def __init__(self, *, nn_fct_h=4, nn_fct_v=4, ln_zad_aimp=False, ln_isfcav=False,
                 trend_diagnostics=False, tiling=False, rk3=False):
        for name, on in (("adaptive-implicit vertical advection (ln_zad_Aimp)", ln_zad_aimp),
                         ("ice-shelf cavities (ln_isfcav)", ln_isfcav),
                         ("advective trend and transport diagnostics (l_trdtra, l_diaptr, "
                          "l_diaar5)", trend_diagnostics),
                         ("tiling (ln_tile; NEMO refuses FCT under tiling)", tiling),
                         ("the RK3 time stepping (key_RK3, fct_up1_2stp)", rk3)):
            if on:
                raise NotImplementedError(f"{name} is not ported in tra_adv_fct")
        for name, v in (("nn_fct_h", nn_fct_h), ("nn_fct_v", nn_fct_v)):
            if v not in (2, 4):
                raise ValueError(f"{name} = {v}: tra_adv_fct takes 2 or 4")
        self.kn_fct_h, self.kn_fct_v = int(nn_fct_h), int(nn_fct_v)
        self._work = None

    def _buffers(self, shape):
        import cupy as cp

        if self._work is None or self._work[0].shape != shape:
            self._work = tuple(cp.zeros(shape, cp.float64) for _ in range(6))
        return self._work

    def step(self, *, p2dt, pU, pV, pW, tb, tn, rhs, tmask, umask, vmask, wmask, e3t, e1e2t,
             r1_e1e2t, mbkt, mikt) -> None:
        """One tracer: adds the FCT advective trend to ``rhs`` (tb, tn the
        before and now fields; mbkt/mikt int32 (jpj, jpi))."""
        jpk, jpj, jpi = map(int, tb.shape)
        if jpk >= 128:
            raise ValueError("the port carries at most 127 levels")
        Fu, Fv, Fw, up1, bu, bd = self._buffers(tuple(tb.shape))
        k = _module()
        n = jpi * jpj
        grid, dims = (((n + _TPB - 1) // _TPB),), (np.int32(jpi), np.int32(jpj), np.int32(jpk),
                                                    np.int32(H))
        k.get_function("nemo_fct_upstream")(grid, (_TPB,), (pU, pV, pW, tb, wmask, Fu, Fv, Fw,
                                                              *dims))
        k.get_function("nemo_fct_up1")(grid, (_TPB,), (Fu, Fv, Fw, tb, e3t, tmask, r1_e1e2t,
                                                         rhs, up1, np.float64(p2dt), *dims))
        k.get_function("nemo_fct_antidiffusive")(grid, (_TPB,), (
            pU, pV, pW, tn, umask, vmask, wmask, mbkt, mikt, Fu, Fv, Fw,
            np.int32(self.kn_fct_h), np.int32(self.kn_fct_v), *dims))
        k.get_function("nemo_fct_beta")(grid, (_TPB,), (tb, up1, tmask, Fu, Fv, Fw, e1e2t, e3t,
                                                          bu, bd, np.float64(p2dt), *dims))
        k.get_function("nemo_fct_limit")(grid, (_TPB,), (bu, bd, Fu, Fv, Fw, *dims))
        k.get_function("nemo_fct_trend")(grid, (_TPB,), (Fu, Fv, Fw, e3t, tmask, r1_e1e2t, rhs,
                                                           *dims))
