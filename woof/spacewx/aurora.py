"""The auroral oval on the GPU: Starkov 1994 boundaries on IGRF-14 dipole coordinates.

Layer L3b of the space-weather design (docs/spacewx.md).  For each column it
reports the geomagnetic latitude, the magnetic local time, the auroral zone
(0 equatorward of the diffuse aurora, 1 diffuse aurora, 2 auroral oval, 3
polar cap) and the distance from the diffuse aurora's equatorward boundary,
which is also the visibility line usually quoted for an aurora forecast.

The boundaries are Starkov's (1994, Geomagn. Aeron. 34, 331) expansion in
magnetic local time, whose coefficients are cubic in log10|AL|.  AL is either
observed (OMNI2) or converted from Kp with Starkov's own relation (Geomagn.
Aeron. 34, 101): ``|AL| = 18 - 12.3 Kp + 27.2 Kp^2 - 2 Kp^3`` nT.  Kp is
observed (GFZ) or estimated from the solar wind (woof.spacewx.solarwind).

The coefficient table is :data:`STARKOV`.  The values are the ones tabulated by
Sigernes et al. 2011 (Optica Pura y Aplicada 44, 599, Table 2), and an
independent tabulation agrees with them.  ocbpy (Burrell & Chisham;
BSD-3-Clause, ``ocbpy/boundaries/models.py``) carries the same table with one
difference: its equatorward-oval b0 of A0 is 1.16 where both other sources
give 1.61.  woof uses 1.61.  The numbers are the published fit of Starkov
1994; ocbpy's copy is redistributed under its BSD-3-Clause licence:

    Copyright (c) 2017, Angeline G. Burrell (AGB) and Gareth Chisham (GC)
    All rights reserved.  Redistribution and use in source and binary forms,
    with or without modification, are permitted provided that the conditions
    of the BSD 3-Clause licence are met (https://github.com/aburrell/ocbpy).

The geomagnetic coordinates are those of the centred dipole (IGRF-14 degree
1) at the date.  Starkov's boundaries are in corrected geomagnetic (CGM)
coordinates; the dipole latitude differs from CGM by a few degrees in some
sectors, which is the main approximation of this layer.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from woof.spacewx import igrf
from woof.spacewx.ionosphere import subsolar_point
from woof.spacewx.solar_irradiance import naive_utc as _naive_utc

BOUNDARIES = ("ocb", "eab", "diffuse")
COEFFICIENTS = ("A0", "A1", "A2", "A3", "alpha1", "alpha2", "alpha3")
#: Starkov 1994: [boundary][coefficient] = (b0, b1, b2, b3) of
#: ``b0 + b1 L + b2 L^2 + b3 L^3``, ``L = log10|AL|``.  A in degrees of
#: co-latitude, alpha in hours.
STARKOV = {
    "ocb": {"A0": (-0.07, 24.54, -12.53, 2.15), "A1": (-10.06, 19.83, -9.33, 1.24),
            "A2": (-4.44, 7.47, -3.01, 0.25), "A3": (-3.77, 7.90, -4.73, 0.91),
            "alpha1": (-6.61, 10.17, -5.80, 1.19), "alpha2": (6.37, -1.10, 0.34, -0.38),
            "alpha3": (-4.48, 10.16, -5.87, 0.98)},
    "eab": {"A0": (1.61, 23.21, -10.97, 2.03), "A1": (-9.59, 17.78, -7.20, 0.96),
            "A2": (-12.07, 17.49, -7.96, 1.15), "A3": (-6.56, 11.44, -6.73, 1.31),
            "alpha1": (-2.22, 1.50, -0.58, 0.08), "alpha2": (-23.98, 42.79, -26.96, 5.56),
            "alpha3": (-20.07, 36.67, -24.20, 5.11)},
    "diffuse": {"A0": (3.44, 29.77, -16.38, 3.35), "A1": (-2.41, 7.89, -4.32, 0.87),
                "A2": (-0.74, 3.94, -3.09, 0.72), "A3": (-2.12, 3.24, -1.67, 0.31),
                "alpha1": (-1.68, -2.48, 1.58, -0.28), "alpha2": (8.69, -20.73, 13.03, -2.14),
                "alpha3": (8.61, -5.34, -1.36, 0.76)},
}
OUTPUTS = ("mlat", "mlt", "zone", "eq_dist")
ZONES = {0: "equatorward of the diffuse aurora", 1: "diffuse aurora", 2: "auroral oval", 3: "polar cap"}
#: Starkov's fit covers |AL| from about 10 to 2000 nT.  Kp 0 to 9 gives 18 to
#: about 653 nT through :func:`kp_to_al`, whose cubic peaks near Kp 8.83 and
#: falls 0.1 % by Kp 9 (Kp 9 is then a hair smaller oval than Kp 8.67).
AL_RANGE = (10.0, 2000.0)


class AuroraError(ValueError):
    pass


def kp_to_al(kp: float) -> float:
    """Starkov 1994's |AL| (nT) for a Kp in [0, 9]."""
    kp = float(kp)
    if not 0.0 <= kp <= 9.0:
        raise AuroraError(f"Kp {kp} outside 0-9")
    return 18.0 - 12.3 * kp + 27.2 * kp * kp - 2.0 * kp ** 3


def coefficients(al: float) -> np.ndarray:
    """``(3, 7)``: A0..A3, alpha1..alpha3 of each boundary (ocb, eab, diffuse) at |AL|."""
    al = abs(float(al))
    if not np.isfinite(al):
        raise AuroraError(f"AL {al} is not finite")
    al = min(max(al, AL_RANGE[0]), AL_RANGE[1])
    x = np.log10(al)
    powers = np.array([1.0, x, x * x, x ** 3])
    return np.array([[np.dot(STARKOV[b][c], powers) for c in COEFFICIENTS] for b in BOUNDARIES])


def boundary_colatitude(coef_row, mlt):
    """Starkov's co-latitude (deg) of one boundary at ``mlt`` (h)."""
    a0, a1, a2, a3, p1, p2, p3 = coef_row
    t = np.asarray(mlt, dtype=np.float64)
    r = np.deg2rad
    return (a0 + a1 * np.cos(r(15.0 * (t + p1))) + a2 * np.cos(r(15.0 * (2 * t + p2)))
            + a3 * np.cos(r(15.0 * (3 * t + p3))))


def dipole_frame(model: igrf.IgrfModel, when: datetime) -> np.ndarray:
    """``(3, 3)``: the boreal dipole axis m and the unit vectors ex, ey (rows),
    Earth-fixed Cartesian, from the IGRF degree-1 coefficients at ``when``."""
    g, h = model.coefficients(igrf.decimal_year(when))
    m = -np.array([g[1, 1], h[1, 1], g[1, 0]], dtype=np.float64)
    m /= np.linalg.norm(m)
    ey = np.cross([0.0, 0.0, 1.0], m)
    if np.linalg.norm(ey) < 1e-12:          # an axial dipole: any meridian will do
        ey = np.array([0.0, 1.0, 0.0])
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, m)
    return np.stack([m, ex, ey])


def _unit(lat_deg, lon_deg, geodetic=True):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    if geodetic:
        lat = np.arctan2((1.0 - igrf.WGS84_F) ** 2 * np.sin(lat), np.cos(lat))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def dipole_latitude(frame: np.ndarray, lat_deg, lon_deg) -> np.ndarray:
    """Dipole geomagnetic latitude, deg, of geodetic points in ``frame``."""
    return np.rad2deg(np.arcsin(np.clip(_unit(lat_deg, lon_deg) @ frame[0], -1.0, 1.0)))


def sun_magnetic_longitude(frame: np.ndarray, when: datetime) -> float:
    lat, lon = subsolar_point(when)
    s = _unit(lat, lon, geodetic=False)
    return float(np.rad2deg(np.arctan2(s @ frame[2], s @ frame[1])))


def pole(frame: np.ndarray) -> tuple[float, float]:
    """Geographic latitude and longitude (deg) of the boreal dipole pole."""
    m = frame[0]
    return float(np.rad2deg(np.arcsin(m[2]))), float(np.rad2deg(np.arctan2(m[1], m[0])))


class Aurora:
    """The L3b oval on a fixed set of columns (lat/lon in degrees)."""

    def __init__(self, lat_deg, lon_deg, *, when: datetime, igrf_model=None):
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        if lat.shape != lon.shape or lat.size == 0:
            raise AuroraError("lat and lon must be non-empty and of one size")
        if not np.all(np.abs(lat) <= 90.0) or not np.all(np.isfinite(lon)):
            raise AuroraError("latitudes must lie in [-90, 90] and longitudes be finite")
        self.lat, self.lon = lat, lon
        model = igrf_model or igrf.load()
        self.igrf_sha256 = model.sha256
        self.frame = dipole_frame(model, _naive_utc(when))
        self._device = None

    def drivers(self, when: datetime, al: float) -> dict:
        return {"coef": coefficients(al), "mlon_sun": sun_magnetic_longitude(self.frame, _naive_utc(when)),
                "al": abs(float(al))}

    def evaluate(self, when: datetime, al: float) -> dict:
        """Device arrays (ncol,) of :data:`OUTPUTS` at ``when`` for |AL| ``al``."""
        import cupy as cp
        from woof.core.kernels import get_kernel

        drv = self.drivers(when, al)
        if self._device is None:
            self._device = tuple(cp.asarray(a, dtype=cp.float32) for a in
                                 (self.lat, self.lon, self.frame.ravel()))
        ncol = self.lat.size
        out = {k: cp.zeros(ncol, dtype=cp.float32) for k in OUTPUTS}
        coef = cp.asarray(drv["coef"].ravel(), dtype=cp.float32)
        threads = 128
        get_kernel("spacewx_aurora", "spacewx_aurora")(
            ((ncol + threads - 1) // threads,), (threads,),
            (*self._device, coef, np.float32(drv["mlon_sun"]), *(out[k] for k in OUTPUTS), np.int64(ncol)))
        return out

    def reference(self, when: datetime, al: float) -> dict:
        """The same model in NumPy float64 (the kernel's test oracle)."""
        drv = self.drivers(when, al)
        p = _unit(self.lat, self.lon)
        mlat = dipole_latitude(self.frame, self.lat, self.lon)
        mlon = np.rad2deg(np.arctan2(p @ self.frame[2], p @ self.frame[1]))
        mlt = np.mod(12.0 + (mlon - drv["mlon_sun"]) / 15.0, 24.0)
        colat = 90.0 - np.abs(mlat)
        ocb, eab, dif = (boundary_colatitude(drv["coef"][i], mlt) for i in range(3))
        zone = np.where(colat < ocb, 3.0, np.where(colat <= eab, 2.0, np.where(colat <= dif, 1.0, 0.0)))
        return {"mlat": mlat, "mlt": mlt, "zone": zone, "eq_dist": dif - colat}


def receipt(aurora: Aurora, driver: dict) -> dict:
    lat, lon = pole(aurora.frame)
    return {"spacewx_igrf_sha256": aurora.igrf_sha256, "spacewx_dipole_pole_deg": [lat, lon],
            "spacewx_oval_model": "Starkov 1994 (Sigernes et al. 2011 Table 2)", **driver}


__all__ = ["Aurora", "AuroraError", "STARKOV", "OUTPUTS", "ZONES", "kp_to_al", "coefficients",
           "boundary_colatitude", "dipole_frame", "sun_magnetic_longitude", "pole", "receipt"]
