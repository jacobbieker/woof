# WOOF-OPTIN: nemo
"""NEMO's equation of state on device (``eos.cu``).

A port of src/OCE/TRA/eosbn2.F90 (NEMO 5.0.2): ``insitu_pot`` is
eos_insitu_pot (the in situ density anomaly prd and the surface-referenced
potential density prhop), ``rab`` is rab_3d (the thermal and haline
expansion ratios), ``bn2`` is bn2 (the Brunt-Vaisala frequency squared), for
the three nameos choices: TEOS-10, EOS-80 and the simplified S-EOS with
nameos' reference coefficients.  The coefficient tables are eos_init's
(``eos_coeffs.py``).  The stochastic equation of state (ln_sto_eos) is
refused by name.  Fields are float64 (jpk, jpj, jpi) device arrays; NEMO
leaves level jpk untouched, and so does the port.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._compile import raw_module

from . import eos_coeffs

_TPB = 64
_MODULES: dict = {}
#: eosbn2.F90:66-80, nameos' S-EOS reference values
SEOS = dict(rn_T0=10.0, rn_S0=35.0, rn_a0=1.6550e-1, rn_b0=7.6554e-1, rn_lambda1=5.9520e-2,
            rn_lambda2=5.4914e-4, rn_mu1=1.4970e-4, rn_mu2=1.1090e-5, rn_nu=2.4341e-3)
#: eos_init (eosbn2.F90:1898) and phycst.F90:38
RHO0, GRAV = 1026.0, 9.80665
NEOS = {"teos10": -1, "eos80": 0, "seos": 1}


def _module():
    if "m" not in _MODULES:
        names = list(eos_coeffs.TEOS10)
        defines = "\n".join(f"#define {k} c[{i}]" for i, k in enumerate(names))
        source = defines + "\n" + (Path(__file__).resolve().parent / "eos.cu").read_text(encoding="utf-8")
        _MODULES["m"] = raw_module(source, ("--fmad=false",))
        _MODULES["names"] = names
    return _MODULES["m"]


class EquationOfState:
    def __init__(self, scheme: str = "teos10", *, stochastic: bool = False):
        import cupy as cp

        if stochastic:
            raise NotImplementedError("the stochastic equation of state (ln_sto_eos) is not ported")
        if scheme not in NEOS:
            raise ValueError(f"nameos offers {sorted(NEOS)}")
        self.neos = NEOS[scheme]
        _module()
        table = eos_coeffs.EOS80 if scheme == "eos80" else eos_coeffs.TEOS10
        self._c = cp.asarray(np.array([float(table[k]) for k in _MODULES["names"]], np.float64))
        self.rho0 = RHO0
        self.r1_rho0 = 1.0 / RHO0
        self._seos = tuple(np.float64(SEOS[k]) for k in (
            "rn_T0", "rn_S0", "rn_a0", "rn_b0", "rn_lambda1", "rn_lambda2", "rn_mu1", "rn_mu2", "rn_nu"))

    def _launch(self, name, args, n2):
        _module().get_function(name)((((n2 + _TPB - 1) // _TPB),), (_TPB,), args)

    def insitu_pot(self, tem, sal, gdept, tmask, prd, prhop):
        jpk, jpj, jpi = map(int, tem.shape)
        n2 = jpi * jpj
        self._launch("nemo_eos_insitu_pot", (self._c, tem, sal, gdept, tmask, prd, prhop,
                     np.int32(self.neos), *self._seos, np.float64(self.rho0),
                     np.float64(self.r1_rho0), np.int32(jpk), np.int32(n2)), n2)

    def rab(self, tem, sal, gdept, tmask, alpha, beta):
        jpk, jpj, jpi = map(int, tem.shape)
        n2 = jpi * jpj
        self._launch("nemo_rab", (self._c, tem, sal, gdept, tmask, alpha, beta,
                     np.int32(self.neos), *self._seos, np.float64(self.r1_rho0),
                     np.int32(jpk), np.int32(n2)), n2)

    def bn2(self, tem, sal, alpha, beta, gdept, gdepw, e3w, wmask, pn2):
        jpk, jpj, jpi = map(int, tem.shape)
        n2 = jpi * jpj
        self._launch("nemo_bn2", (tem, sal, alpha, beta, gdept, gdepw, e3w, wmask, pn2,
                     np.float64(GRAV), np.int32(jpk), np.int32(n2)), n2)
