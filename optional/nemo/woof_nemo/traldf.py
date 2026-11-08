# WOOF-OPTIN: nemo
"""NEMO's iso-level lateral tracer diffusion on device (``traldf.cu``).

A port of src/OCE/TRA/traldf_lev.F90 (NEMO 5.0.2): ``traldf_lev_lap``
(ln_traldf_lap with ln_traldf_lev) and ``traldf_lev_blp`` (ln_traldf_blp
with ln_traldf_lev), given the eddy diffusivities ahtu/ahtv (m2/s, or m4/s
with ldf_tra's negative sign for the bilaplacian).  Refused by name: the
iso-neutral operators (ln_traldf_iso, traldf_iso; ln_traldf_triad,
traldf_triad), ldf_tra_init's diffusivity fields (nn_aht_ijk_t; pass
ahtu/ahtv), and transport diagnostics.  Fields are float64 (jpk, jpj, jpi)
device arrays with NEMO 5.0's two-cell halo; NEMO adds no lbc_lnk here, so
the halo holds whatever the caller's last exchange left (closed: zero).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 64
_MODULE = None
H = 2


def _module():
    global _MODULE
    if _MODULE is None:
        source = (Path(__file__).resolve().parent / "traldf.cu").read_text(encoding="utf-8")
        _MODULE = raw_module(source, ("--fmad=false",))
    return _MODULE


class LateralDiffusionLevel:
    def __init__(self, *, operator="lap", direction="lev", transport_diagnostics=False):
        if direction in ("iso", "triad"):
            raise NotImplementedError(
                f"the iso-neutral operator (ln_traldf_{direction}, traldf_{direction}) is "
                "not ported; only the iso-level one (ln_traldf_lev)")
        if direction != "lev":
            raise ValueError("namtra_ldf direction: lev, iso or triad")
        if operator not in ("lap", "blp"):
            raise ValueError("namtra_ldf operator: lap or blp")
        if transport_diagnostics:
            raise NotImplementedError("lateral transport diagnostics (l_diaptr, ln_diaar5) "
                                      "are not ported")
        self.bilaplacian = operator == "blp"

    def step(self, *, ahtu, ahtv, e2_e1u, e1_e2v, e3u, e3v, e3t, r1_e1e2t, tb, rhs) -> None:
        """One tracer: adds the lateral diffusive trend of ``tb`` to ``rhs``."""
        jpk, jpj, jpi = map(int, tb.shape)
        n = (jpi - 2 * H) * (jpj - 2 * H)
        _module().get_function("nemo_traldf_lev")((((n + _TPB - 1) // _TPB),), (_TPB,), (
            ahtu, ahtv, e2_e1u, e1_e2v, e3u, e3v, e3t, r1_e1e2t, tb, rhs,
            np.int32(self.bilaplacian), np.int32(jpi), np.int32(jpj), np.int32(jpk), np.int32(H)))
