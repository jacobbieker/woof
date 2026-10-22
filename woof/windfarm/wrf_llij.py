"""WRF's own turbine placement: real(4) Lambert ``latlon_to_ij`` and ``nint``.

WRF v4.7.1 places each turbine of ``windturbines.txt`` in
``phys/module_wind_fitch.F`` ``init_module_wind_fitch`` (lines 399-473):
it reads ``lat, lon`` as REAL, calls ``map_set(PROJ_LC, ...)`` anchored at
the float32 ``XLAT``/``XLONG`` of the tile's first mass point
``(its, jts)`` with ``knowni = REAL(its)``, ``knownj = REAL(jts)``, then
``latlon_to_ij`` and ``ival = nint(ts_rx)``, ``jval = nint(ts_ry)``.  All of
it is default REAL, real(4) in WRF's standard ``-DRWORDSIZE=4`` build, and
``share/module_llxy.F`` (``map_set``:267, ``set_lc``:1097, ``lc_cone``:1138,
``llij_lc``:1250) does the arithmetic.  This module is that arithmetic,
statement by statement in float32 and in gfortran's order of evaluation:
NumPy float32 scalar ``+ - * /`` are IEEE-754 correctly rounded as
gfortran's SSE code is, and ``COS``/``SIN``/``TAN``/``ALOG10``/``**`` call
the host C library's ``cosf``/``sinf``/``tanf``/``log10f``/``powf``
through ctypes, the functions gfortran's object code calls.  On a glibc
host that is the same machine word for word; on another C library both
the oracle and this module follow it.

It is held at max_ulp 0 against the unmodified WRF source compiled by
``tools/windfarm_layout_wrf471_oracle`` (``tests/test_windfarm_layout_wrf471_parity.py``).

Scope: Lambert conformal only (``map_proj = 1``), the projection woof's
wind-farm domains use; Mercator and polar stereographic are refused by
name.  The tile anchor is the domain's first mass point (``its = jts =
1``): WRF anchors each MPI patch at its own corner, so a decomposed WRF
run places turbines from a different float32 anchor and can differ in
the last bits; woof places once, as a serial WRF run does.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from woof.core.adaptive_timestep import nint as _fortran_nint

F = np.float32

#: module_llxy.F:137-152, REAL parameters (folded by gfortran in real(4)).
PI = F(3.141592653589793)
DEG_PER_RAD = F(F(180.0) / PI)
RAD_PER_DEG = F(PI / F(180.0))
EARTH_RADIUS_M = F(6370000.0)

#: module_llxy.F:156 PROJ_LC.
PROJ_LC = 1


@lru_cache(maxsize=1)
def _libm():
    name = ctypes.util.find_library("m") or "libm.so.6"
    library = ctypes.CDLL(name)
    for fn in ("cosf", "sinf", "tanf", "log10f"):
        getattr(library, fn).restype = ctypes.c_float
        getattr(library, fn).argtypes = [ctypes.c_float]
    library.powf.restype = ctypes.c_float
    library.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    return library


def _unary(name, x) -> np.float32:
    return F(getattr(_libm(), name)(ctypes.c_float(F(x))))


def cosf(x):
    return _unary("cosf", x)


def sinf(x):
    return _unary("sinf", x)


def tanf(x):
    return _unary("tanf", x)


def log10f(x):
    return _unary("log10f", x)


def powf(x, y):
    return F(_libm().powf(ctypes.c_float(F(x)), ctypes.c_float(F(y))))


def _wrap180(value: np.float32) -> np.float32:
    """map_set's lon1/stdlon loop (module_llxy.F:440-450)."""

    value = F(value)
    if abs(value) > F(180.0):
        for _ in range(10):
            if not abs(value) > F(180.0):
                break
            if value < F(-180.0):
                value = F(value + F(360.0))
            if value > F(180.0):
                value = F(value - F(360.0))
    return value


def lc_cone(truelat1, truelat2) -> np.float32:
    """module_llxy.F:1138-1171."""

    t1, t2 = F(truelat1), F(truelat2)
    if abs(F(t1 - t2)) > F(0.1):
        cone = F(log10f(cosf(F(t1 * RAD_PER_DEG)))
                 - log10f(cosf(F(t2 * RAD_PER_DEG))))
        a = log10f(tanf(F(F(F(45.0) - F(abs(t1) / F(2.0))) * RAD_PER_DEG)))
        b = log10f(tanf(F(F(F(45.0) - F(abs(t2) / F(2.0))) * RAD_PER_DEG)))
        return F(cone / F(a - b))
    return sinf(F(abs(t1) * RAD_PER_DEG))


@dataclass(frozen=True)
class WrfLambert:
    """``proj_info`` after ``map_set(PROJ_LC, ...)``: every member float32.

    ``lat1``/``lon1`` is the anchor (``XLAT``/``XLONG`` at ``(its, jts)``),
    ``knowni``/``knownj`` its 1-based mass index.
    """

    truelat1: np.float32
    truelat2: np.float32
    stdlon: np.float32
    lat1: np.float32
    lon1: np.float32
    knowni: np.float32
    knownj: np.float32
    dx: np.float32
    hemi: np.float32
    rebydx: np.float32
    cone: np.float32
    rsw: np.float32
    polei: np.float32
    polej: np.float32

    @classmethod
    def map_set(cls, *, truelat1, truelat2, stdlon, lat1, lon1, knowni,
                knownj, dx) -> "WrfLambert":
        """module_llxy.F map_set (PROJ_LC arm) then set_lc:1097-1135."""

        truelat1, truelat2 = F(truelat1), F(truelat2)
        lat1, lon1 = F(lat1), _wrap180(lon1)
        stdlon = _wrap180(stdlon)
        knowni, knownj, dx = F(knowni), F(knownj), F(dx)
        if not (dx > F(0.0) and abs(lat1) <= F(90.0)
                and abs(truelat1) <= F(90.0)):
            raise ValueError("WRF map_set refuses this Lambert grid")
        hemi = F(-1.0) if truelat1 < F(0.0) else F(1.0)
        rebydx = F(EARTH_RADIUS_M / dx)
        if abs(truelat2) > F(90.0):
            truelat2 = truelat1
        cone = lc_cone(truelat1, truelat2)
        deltalon1 = F(lon1 - stdlon)
        if deltalon1 > F(180.0):
            deltalon1 = F(deltalon1 - F(360.0))
        if deltalon1 < F(-180.0):
            deltalon1 = F(deltalon1 + F(360.0))
        ctl1r = cosf(F(truelat1 * RAD_PER_DEG))
        rsw = _radius(rebydx, ctl1r, cone, hemi, lat1, truelat1)
        arg = F(cone * F(deltalon1 * RAD_PER_DEG))
        polei = F(F(hemi * knowni) - F(F(hemi * rsw) * sinf(arg)))
        polej = F(F(hemi * knownj) + F(rsw * cosf(arg)))
        return cls(truelat1=truelat1, truelat2=truelat2, stdlon=stdlon,
                   lat1=lat1, lon1=lon1, knowni=knowni, knownj=knownj,
                   dx=dx, hemi=hemi, rebydx=rebydx, cone=cone, rsw=rsw,
                   polei=polei, polej=polej)

    def latlon_to_ij(self, lat, lon) -> tuple[np.float32, np.float32]:
        """module_llxy.F llij_lc:1250-1304 for one point."""

        lat, lon = F(lat), F(lon)
        deltalon = F(lon - self.stdlon)
        if deltalon > F(180.0):
            deltalon = F(deltalon - F(360.0))
        if deltalon < F(-180.0):
            deltalon = F(deltalon + F(360.0))
        ctl1r = cosf(F(self.truelat1 * RAD_PER_DEG))
        rm = _radius(self.rebydx, ctl1r, self.cone, self.hemi, lat,
                     self.truelat1)
        arg = F(self.cone * F(deltalon * RAD_PER_DEG))
        i = F(self.polei + F(F(self.hemi * rm) * sinf(arg)))
        j = F(self.polej - F(rm * cosf(arg)))
        return F(self.hemi * i), F(self.hemi * j)


def _radius(rebydx, ctl1r, cone, hemi, lat, truelat1) -> np.float32:
    """``rebydx * ctl1r/cone * (TAN(..lat..)/TAN(..truelat1..))**cone``."""

    num = tanf(F(F(F(F(F(90.0) * hemi) - lat) * RAD_PER_DEG) / F(2.0)))
    den = tanf(F(F(F(F(F(90.0) * hemi) - truelat1) * RAD_PER_DEG)
                 / F(2.0)))
    return F(F(F(rebydx * ctl1r) / cone) * powf(F(num / den), cone))


def nint(value) -> int:
    """Fortran NINT on a REAL (:func:`woof.core.adaptive_timestep.nint`).

    The float32 is widened first; ``x + 0.5`` is then exact in binary64
    for any float32 below 2**28, so the rounding is NINT's own.
    """

    x = float(F(value))
    if not math.isfinite(x):
        raise ValueError(f"NINT of {x}")
    return _fortran_nint(x)


__all__ = ["PROJ_LC", "WrfLambert", "lc_cone", "nint", "cosf", "sinf",
           "tanf", "log10f", "powf"]
