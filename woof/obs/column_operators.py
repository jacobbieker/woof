"""Column operators the engine's point observation operator needs.

``refractivity_n`` (the two-term Smith and Weintraub refractivity) and
``dewpoint_from_specific_humidity`` (Bolton 1980) moved here from woof global
(``arwen_global.obs_operators`` and ``arwen_global.surface_energy`` at
b54373992587) with their constants, unchanged, so
:mod:`woof.da.obs_point` no longer imports the global package.

``msl2geo1`` (WRFDA's ``da_msl2geo1``, geometric to geopotential height)
moved here from :mod:`woof.da.var.obs_wrfda` for the same reason.
:mod:`woof.da.obs_point` needs it for GNSS-RO tangent heights, and the
forecast path may not import ``woof.da.var`` (tests/test_var_import_graph.py).
"""
from __future__ import annotations

import numpy as np

REFRACTIVITY_DRY_K_HPA = 77.6
REFRACTIVITY_WET_K2_HPA = 3.73e5
EPSILON = 0.622
_BOLTON_A, _BOLTON_B, _BOLTON_C = 611.2, 17.67, 243.5

__all__ = ["REFRACTIVITY_DRY_K_HPA", "REFRACTIVITY_WET_K2_HPA", "EPSILON",
           "vapour_pressure_pa", "refractivity_n", "dewpoint_from_specific_humidity", "msl2geo1"]


def vapour_pressure_pa(q, p_pa):
    """Vapour pressure from specific humidity at pressure ``p_pa``."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p_pa, dtype=np.float64)
    return q * p / (EPSILON + (1.0 - EPSILON) * q)


def refractivity_n(p_pa, t_k, q):
    """Local refractivity in N-units from pressure (Pa), temperature (K)
    and specific humidity (kg/kg); the two-term Smith and Weintraub form."""
    p_hpa = np.asarray(p_pa, dtype=np.float64) / 100.0
    t = np.asarray(t_k, dtype=np.float64)
    e_hpa = vapour_pressure_pa(q, p_pa) / 100.0
    return REFRACTIVITY_DRY_K_HPA * p_hpa / t + REFRACTIVITY_WET_K2_HPA * e_hpa / (t * t)


def dewpoint_from_specific_humidity(q, p_pa):
    """2 m dewpoint (K) from specific humidity and pressure, Bolton 1980."""
    q = np.clip(np.asarray(q, dtype=np.float64), 1.0e-9, None)
    e = q * p_pa / (EPSILON + (1.0 - EPSILON) * q)
    ln = np.log(np.maximum(e, 1.0e-6) / _BOLTON_A)
    return _BOLTON_C * ln / (_BOLTON_B - ln) + 273.15


def msl2geo1(z_km: float, lat: float, cm=None) -> float:
    """``da_msl2geo1``: geometric to geopotential height (km).

    ``cm`` supplies ``sin`` and ``sqrt``.  The default is
    :mod:`woof.core.crmath`, whose correctly rounded ``sin`` gives the same
    words on every host.  ``-999`` (missing) passes through.
    """
    if cm is None:
        from woof.core import crmath as cm
    a, b = 6378.1370e3, 6356.7523142e3
    gp, ge = 9.8321849378, 9.7803253359
    omega, gm, grav, ecc = 7.292115e-5, 3.986004418e14, 9.80665, 0.081819
    flattening = (a - b) / a
    somigliana = (b / a) * (gp / ge) - 1.0
    grav_ratio = (omega * omega * a * a * b) / gm
    if z_km == -999.0:
        return -999.0
    latr = lat * (3.14159265358979 / 180.0)
    zm = z_km * 1000.0
    s = cm.sin(latr)
    sin2 = s * s
    termg = ge * ((1.0 + somigliana * sin2) / cm.sqrt(1.0 - ecc * ecc * sin2))
    termr = a / (1.0 + flattening + grav_ratio - 2.0 * flattening * sin2)
    return ((termg / grav) * ((termr * zm) / (termr + zm))) * 0.001
