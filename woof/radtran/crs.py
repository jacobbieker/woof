"""Absorption cross sections from libRadtran's tabulated files (read at run time).

Used where the REPTRAN lookup tables carry no cross section for a species
(``cross_section_source == 0``), as uvspec does by default:

* O3: Molina & Molina (1986) in the UV with the visible (Chappuis) continuation
  (``crs/crs_o3_all.dat``, 1e-20 cm^2, columns at 226, 263 and 298 K).  The
  temperature dependence is the quadratic through the three temperatures
  (linear through 226 and 298 K where the 263 K column is empty).
* NO2: Burrows et al. (1998), GOME, 273 K (``crs/crs_no2_gom.dat``, cm^2).
* O4: Greenblatt et al. (1990) (``crs/crs_o4_greenblatt.dat``, 1e-46 cm^5),
  applied to the column of n(O2)^2.

All are interpolated linearly in wavelength.  (The choice of files and the
interpolation rules were measured against uvspec run as a black box.)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from woof.radtran import data


def _read(path):
    rows = []
    for line in open(path):
        s = line.split()
        if not s or s[0].startswith("#") or not (s[0][0].isdigit() or s[0][0] == "."):
            continue
        try:
            rows.append([float(x) for x in s])
        except ValueError:
            continue
    return np.array(rows, dtype=np.float64)


@lru_cache(maxsize=None)
def _o3():
    return _read(data.path("crs", "crs_o3_all.dat"))


@lru_cache(maxsize=None)
def _no2():
    return _read(data.path("crs", "crs_no2_gom.dat"))


@lru_cache(maxsize=None)
def _o4():
    return _read(data.path("crs", "crs_o4_greenblatt.dat"))


O3_T = (226.0, 263.0, 298.0)


def o3(lam_nm: float, t_k):
    """O3 cross section [cm^2] at one wavelength for temperatures t_k."""
    a = _o3()
    t = np.asarray(t_k, dtype=np.float64)
    if lam_nm < a[0, 0] or lam_nm > a[-1, 0]:
        return np.zeros_like(t)
    c = [np.interp(lam_nm, a[:, 0], a[:, k]) * 1e-20 for k in (1, 2, 3)]
    # rows with an empty 263 K column carry two temperatures only
    i = int(np.searchsorted(a[:, 0], lam_nm))
    lo, hi = max(i - 1, 0), min(i, a.shape[0] - 1)
    if a[lo, 2] == 0.0 or a[hi, 2] == 0.0:
        return c[0] + (c[2] - c[0]) * (t - O3_T[0]) / (O3_T[2] - O3_T[0])
    t0, t1, t2 = O3_T
    l0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
    l1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
    l2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
    return c[0] * l0 + c[1] * l1 + c[2] * l2


def no2(lam_nm: float) -> float:
    a = _no2()
    if lam_nm < a[0, 0] or lam_nm > a[-1, 0]:
        return 0.0
    return float(np.interp(lam_nm, a[:, 0], a[:, 1]))


def o4(lam_nm: float) -> float:
    a = _o4()
    if lam_nm < a[0, 0] or lam_nm > a[-1, 0]:
        return 0.0
    return float(np.interp(lam_nm, a[:, 0], a[:, 1])) * 1e-46
