"""WRFDA's single (pseudo) observation (``num_pseudo = 1``), the standard check of ``B``.

``da_setup_pseudo_obs`` builds one report at the fractional grid position
``(pseudo_x, pseudo_y)`` (WRFDA's 1-based mass-point indices) and level
``pseudo_z`` (``zk``); ``pseudo_var`` picks the one of ``u v t p q`` that
carries the innovation ``pseudo_val`` with error ``pseudo_err``. The other
four fields have ``qc = missing_data`` and do not enter ``Jo``. The operator
(``da_transform_xtoy_pseudo``) is the linear interpolation of the analysis
increment at that point. No other observation is used (WRFDA switches every
other type off); the first guess is unchanged by it.

:func:`pseudo_obs` returns a :class:`~woof.da.var.obs_wrfda.WrfdaObs` that
:func:`woof.da.var.analysis.analyse` accepts in place of an ``ob.ascii``.

Derived from WRFDA 4.7.1 (``da_setup_pseudo_obs``, ``da_get_innov_vector_pseudo``,
``da_transform_xtoy_pseudo``, ``da_jo_and_grady_pseudo``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import math

import numpy as np

from .obs_wrfda import MISSING_DATA, MISSING_R, Field, ObsType, WrfdaObs

__all__ = ["pseudo_obs", "PSEUDO_FIELDS", "PSEUDO_VARS"]

#: ``iv%pseudo(n)%``'s fields, in ``da_jo_and_grady_pseudo``'s order.
PSEUDO_FIELDS = ("u", "v", "t", "p", "q")
#: ``pseudo_var`` values this port runs and tests (WRFDA also has ``p``, the hydrometeors, ``tpw``, ``ztd``, ``ref``).
PSEUDO_VARS = ("u", "v", "t", "q")


def pseudo_obs(var: str, x: float, y: float, z: float, val: float = 1.0, err: float = 1.0, *,
               lat: float = 0.0, lon: float = 0.0, analysis_minutes: float = 0.0) -> WrfdaObs:
    """One pseudo observation of ``var`` (``u``, ``v``, ``t``, ``p`` or ``q``).

    ``x``, ``y`` are WRFDA's 1-based fractional mass-point indices, ``z`` its
    fractional model level (1 is the lowest). ``int(x)``/``int(y)`` and the
    fractions are formed as in ``da_setup_pseudo_obs``; the interpolation
    needs ``int(x)+1`` and ``int(y)+1`` inside the tile.
    """
    var = var.strip().lower()
    if var not in PSEUDO_VARS:
        raise ValueError(f"pseudo_var {var!r}: expected one of {PSEUDO_VARS}")
    if not (math.isfinite(err) and err > 0.0 and math.isfinite(val)):
        raise ValueError(f"pseudo_err must be finite and positive, pseudo_val finite (got {err}, {val})")
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and x >= 1.0 and y >= 1.0 and z >= 1.0):
        raise ValueError(f"pseudo_x, pseudo_y, pseudo_z are 1-based grid indices and levels, got {x}, {y}, {z}")
    i, j = int(x), int(y)
    dx, dy = x - float(i), y - float(j)
    info = {
        "name": np.array(["pseudo"]), "platform": np.array(["pseudo"]), "id": np.array(["pseudo"]),
        "date": np.array([""]), "lat": np.array([lat]), "lon": np.array([lon]), "elv": np.array([0.0]),
        "x": np.array([x]), "y": np.array([y]), "i": np.array([i], dtype=np.int64),
        "j": np.array([j], dtype=np.int64), "dx": np.array([dx]), "dy": np.array([dy]),
        "dxm": np.array([1.0 - dx]), "dym": np.array([1.0 - dy]), "proc_domain": np.array([True]),
        "obs_global_index": np.array([1], dtype=np.int64), "levels": np.array([1], dtype=np.int64),
    }
    # da_setup_pseudo_obs: every field starts as inv = error = missing_r, qc = missing_data; only the selected
    # variable gets (pseudo_val, 0, pseudo_err).  The others never enter Jo (qc < 0) but H' still interpolates them.
    fields = {}
    for v in PSEUDO_FIELDS:
        sel = v == var
        fields[v] = Field(np.array([val if sel else MISSING_R]), np.array([0 if sel else MISSING_DATA], dtype=np.int64),
                          np.array([err if sel else MISSING_R]))
    t = ObsType("pseudo", info, {}, fields, {"zk": np.array([z])})
    return WrfdaObs({"pseudo": t}, {"pseudo": 1}, analysis_minutes)
