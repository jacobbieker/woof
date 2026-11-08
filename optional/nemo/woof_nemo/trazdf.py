# WOOF-OPTIN: nemo
"""NEMO's implicit vertical diffusion of T and S on device (``trazdf.cu``).

A port of src/OCE/TRA/trazdf.F90 (``tra_zdf``) from NEMO 5.0.2 for a linear
free surface with avt from :mod:`woof_nemo.tke`.  Refused by name: iso-neutral
slopes (l_ldfslp), adaptive-implicit vertical advection (ln_zad_Aimp), the
mass-flux convection scheme (ln_zdfmfc) and double diffusion (ln_zdfddm,
which would give salinity its own avs matrix).  Fields are float64
(jpk, jpj, jpi) device arrays with NEMO's one-cell halo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 64
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        source = (Path(__file__).resolve().parent / "trazdf.cu").read_text(encoding="utf-8")
        _MODULE = raw_module(source, ("--fmad=false",))
    return _MODULE


class TracerVerticalDiffusion:
    def __init__(self, *, rDt, ln_ldfslp=False, ln_zad_aimp=False, ln_zdfmfc=False,
                 ln_zdfddm=False, ln_seos=False, rn_b0=7.6554e-1):
        for name, on in (("iso-neutral slopes (l_ldfslp)", ln_ldfslp),
                         ("adaptive-implicit vertical advection (ln_zad_Aimp)", ln_zad_aimp),
                         ("mass-flux convection (ln_zdfmfc)", ln_zdfmfc),
                         ("double diffusion (ln_zdfddm)", ln_zdfddm)):
            if on:
                raise NotImplementedError(f"{name} is not ported in tra_zdf")
        self.p2dt = float(rDt)
        # tra_zdf (:71): the salinity floor unless S-EOS with rn_b0 = 0
        self.salt_floor = not (ln_seos and rn_b0 == 0.0)

    def step(self, *, avt, e3t, e3w, tmask, tb, sb, trhs, srhs, ta, sa) -> None:
        jpk, jpj, jpi = map(int, avt.shape)
        if jpk >= 128:
            raise ValueError("the port carries at most 127 levels")
        n = (jpi - 2) * (jpj - 2)
        _module().get_function("nemo_tra_zdf")((((n + _TPB - 1) // _TPB),), (_TPB,), (
            avt, e3t, e3w, tmask, tb, sb, trhs, srhs, ta, sa, np.float64(self.p2dt),
            np.int32(self.salt_floor), np.int32(jpi), np.int32(jpj), np.int32(jpk)))
