"""WRF's own turbine placement: real(4) ``map_set``, ``latlon_to_ij``, ``nint``.

WRF v4.7.1 places each turbine of ``windturbines.txt`` in
``phys/module_wind_fitch.F`` ``init_module_wind_fitch`` (lines 399-473):
it reads ``lat, lon`` as REAL and calls ``map_set`` for the domain's
projection -- ``PROJ_MERC`` (truelat1), ``PROJ_LC`` (truelat1, truelat2,
stand_lon) or ``PROJ_PS`` (truelat1, stand_lon) -- anchored at the float32
``XLAT``/``XLONG`` of the tile's first mass point ``(its, jts)`` with
``knowni = REAL(its)``, ``knownj = REAL(jts)`` and the domain's own ``dx``.
Then ``latlon_to_ij`` and ``ival = nint(ts_rx)``, ``jval = nint(ts_ry)``.
A nest is placed the same way from its own corner and its own ``dx``: its
parent start and ratio reach WRF only through that corner's
``XLAT``/``XLONG``.

All of it is default REAL, real(4) in WRF's standard ``-DRWORDSIZE=4``
build, and ``share/module_llxy.F`` does the arithmetic: ``map_init``:224,
``map_set``:267, ``set_ps``:696, ``llij_ps``:732, ``set_lc``:1097,
``lc_cone``:1138, ``llij_lc``:1250, ``set_merc``:1307, ``llij_merc``:1334.
This module is that arithmetic, statement by statement in float32 and in
gfortran's order of evaluation: NumPy float32 scalar ``+ - * /`` are
IEEE-754 correctly rounded as gfortran's SSE code is, and
``COS``/``SIN``/``TAN``/``ALOG``/``ALOG10``/``**`` call the host C library's
``cosf``/``sinf``/``tanf``/``logf``/``log10f``/``powf`` through ctypes, the
functions gfortran's object code calls.  On a glibc host that is the same
machine word for word; on another C library both the oracle and this
module follow it.  Held at max_ulp 0 against the unmodified WRF source by
``tests/test_windfarm_layout_wrf471_parity.py``.

Refused by name: any other projection.  WRF's wind-farm code has no other
arm (its Cassini and rotated-lat-lon branches are commented out, so
``latlon_to_ij`` would stop on an unset projection), and woof builds no
latitude-longitude domain.  The tile anchor is the domain's first mass
point (``its = jts = 1``): WRF anchors each MPI patch at its own corner, so
a decomposed WRF run places turbines from a different float32 anchor and
can differ in the last bits; woof places once, as a serial WRF run does.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
import math
import sys

import numpy as np

from woof.core.adaptive_timestep import nint as _fortran_nint

F = np.float32

#: module_llxy.F:137-152, REAL parameters (folded by gfortran in real(4)).
PI = F(3.141592653589793)
DEG_PER_RAD = F(F(180.0) / PI)
RAD_PER_DEG = F(PI / F(180.0))
EARTH_RADIUS_M = F(6370000.0)

#: module_llxy.F:155-157 projection codes WRF's wind-farm code handles.
PROJ_LC = 1
PROJ_PS = 2
PROJ_MERC = 3
#: woof map_proj name -> module_llxy code.
PROJECTION_CODES = {"lambert": PROJ_LC, "polar": PROJ_PS,
                    "mercator": PROJ_MERC}

#: map_init's fill for every unset REAL member (module_llxy.F:230-251).
UNSET = F(-999.9)


class WrfProjectionRefused(ValueError):
    """A projection WRF's wind-farm placement does not handle."""


class _Ported:
    """glibc's functions from woof's own word-exact ports, under the C
    names (woof.backend.host.libm32): the host C library is not glibc."""

    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        return getattr(self._lib, "woof_" + name)


@lru_cache(maxsize=1)
def _libm():
    from woof.backend.host import host_libc_is_glibc, libm32
    if not host_libc_is_glibc() and not sys.platform.startswith("linux"):
        # macOS / Windows: their libm is not the C library gfortran's WRF
        # called; glibc's words come from woof's ports instead.
        return _Ported(libm32())
    name = ctypes.util.find_library("m") or "libm.so.6"
    library = ctypes.CDLL(name)
    for fn in ("cosf", "sinf", "tanf", "logf", "log10f"):
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


def logf(x):
    return _unary("logf", x)


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
class WrfProjection:
    """``proj_info`` after ``map_set``: every REAL member float32.

    ``code`` is module_llxy's (1 Lambert, 2 polar stereographic, 3
    Mercator).  Members a projection does not set keep ``map_init``'s
    ``-999.9``; ``dlon`` (which ``map_init`` leaves undefined) is 0 except
    on Mercator.  ``lat1``/``lon1`` is the anchor (``XLAT``/``XLONG`` at
    ``(its, jts)``), ``knowni``/``knownj`` its 1-based mass index.
    """

    code: int
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
    dlon: np.float32

    @classmethod
    def for_grid(cls, grid) -> "WrfProjection":
        """What ``init_module_wind_fitch`` sets up for one woof grid (root
        or nest): the grid's own corner ``XLAT``/``XLONG(1, 1)`` -- mass
        point (1, 1) sits at projection coordinate (1, 1), and the scalar
        transform is the one woof's mass planes are built from -- and its
        own ``dx``."""

        name = getattr(grid, "map_proj", None)
        if name not in PROJECTION_CODES:
            raise WrfProjectionRefused(
                f"map_proj {name!r}: WRF v4.7.1's wind-farm placement "
                "(module_wind_fitch.F:408-438) handles Mercator, Lambert "
                "conformal and polar stereographic grids only")
        xlat, xlong = (float(v) for v in grid.ij_to_latlon(1.0, 1.0))
        return cls.map_set(
            PROJECTION_CODES[name], truelat1=grid.truelat1,
            truelat2=grid.truelat2, stdlon=grid.stand_lon, lat1=xlat,
            lon1=xlong, knowni=1.0, knownj=1.0, dx=grid.dx)

    @classmethod
    def map_set(cls, code, *, truelat1, truelat2=None, stdlon=None, lat1,
                lon1, knowni, knownj, dx) -> "WrfProjection":
        """module_llxy.F map_set, then set_lc / set_ps / set_merc, with
        the arguments module_wind_fitch.F:409-438 passes for ``code``."""

        if code not in (PROJ_LC, PROJ_PS, PROJ_MERC):
            raise WrfProjectionRefused(f"projection code {code}")
        truelat1, lat1, lon1 = F(truelat1), F(lat1), _wrap180(lon1)
        knowni, knownj, dx = F(knowni), F(knownj), F(dx)
        # Mercator is set without stdlon/truelat2; map_init's fill stays
        stdlon = UNSET if code == PROJ_MERC else _wrap180(stdlon)
        truelat2 = F(truelat2) if code == PROJ_LC else UNSET
        if not (dx > F(0.0) and abs(lat1) <= F(90.0)
                and abs(truelat1) <= F(90.0)):
            raise WrfProjectionRefused("WRF map_set refuses this grid")
        hemi = F(-1.0) if truelat1 < F(0.0) else F(1.0)
        rebydx = F(EARTH_RADIUS_M / dx)
        cone = rsw = polei = polej = UNSET
        dlon = F(0.0)
        if code == PROJ_LC:
            if abs(truelat2) > F(90.0):
                truelat2 = truelat1
            cone, rsw, polei, polej = _set_lc(
                truelat1, truelat2, stdlon, lat1, lon1, knowni, knownj,
                hemi, rebydx)
        elif code == PROJ_PS:
            rsw, polei, polej = _set_ps(truelat1, stdlon, lat1, lon1,
                                        knowni, knownj, hemi, rebydx)
        else:
            dlon, rsw = _set_merc(truelat1, lat1, dx)
        return cls(code=int(code), truelat1=truelat1, truelat2=truelat2,
                   stdlon=stdlon, lat1=lat1, lon1=lon1, knowni=knowni,
                   knownj=knownj, dx=dx, hemi=hemi, rebydx=rebydx,
                   cone=cone, rsw=rsw, polei=polei, polej=polej, dlon=dlon)

    def members(self) -> np.ndarray:
        """``proj_info``'s REAL members as the oracle writes them: cone,
        rebydx, rsw, polei, polej, hemi, dlon, stdlon, truelat1, truelat2,
        lat1, lon1, knowni, knownj, dx."""

        return np.array([self.cone, self.rebydx, self.rsw, self.polei,
                         self.polej, self.hemi, self.dlon, self.stdlon,
                         self.truelat1, self.truelat2, self.lat1, self.lon1,
                         self.knowni, self.knownj, self.dx], dtype=F)

    def latlon_to_ij(self, lat, lon) -> tuple[np.float32, np.float32]:
        """module_llxy.F latlon_to_ij for one point."""

        lat, lon = F(lat), F(lon)
        if self.code == PROJ_LC:
            return self._llij_lc(lat, lon)
        if self.code == PROJ_PS:
            return self._llij_ps(lat, lon)
        return self._llij_merc(lat, lon)

    def _llij_lc(self, lat, lon):
        """llij_lc:1250-1304."""

        deltalon = F(lon - self.stdlon)
        if deltalon > F(180.0):
            deltalon = F(deltalon - F(360.0))
        if deltalon < F(-180.0):
            deltalon = F(deltalon + F(360.0))
        ctl1r = cosf(F(self.truelat1 * RAD_PER_DEG))
        rm = _lc_radius(self.rebydx, ctl1r, self.cone, self.hemi, lat,
                        self.truelat1)
        arg = F(self.cone * F(deltalon * RAD_PER_DEG))
        i = F(self.polei + F(F(self.hemi * rm) * sinf(arg)))
        j = F(self.polej - F(rm * cosf(arg)))
        return F(self.hemi * i), F(self.hemi * j)

    def _llij_ps(self, lat, lon):
        """llij_ps:732-774 (no cut-zone wrap of lon - reflon, as WRF)."""

        reflon = F(self.stdlon + F(90.0))
        scale_top = F(F(1.0) + F(self.hemi
                                 * sinf(F(self.truelat1 * RAD_PER_DEG))))
        ala = F(lat * RAD_PER_DEG)
        rm = F(F(F(self.rebydx * cosf(ala)) * scale_top)
               / F(F(1.0) + F(self.hemi * sinf(ala))))
        alo = F(F(lon - reflon) * RAD_PER_DEG)
        i = F(self.polei + F(rm * cosf(alo)))
        j = F(self.polej + F(F(self.hemi * rm) * sinf(alo)))
        return i, j

    def _llij_merc(self, lat, lon):
        """llij_merc:1334-1355."""

        deltalon = F(lon - self.lon1)
        if deltalon < F(-180.0):
            deltalon = F(deltalon + F(360.0))
        if deltalon > F(180.0):
            deltalon = F(deltalon - F(360.0))
        i = F(self.knowni + F(deltalon / F(self.dlon * DEG_PER_RAD)))
        y = logf(tanf(F(F(0.5) * F(F(lat + F(90.0)) * RAD_PER_DEG))))
        j = F(F(self.knownj + F(y / self.dlon)) - self.rsw)
        return i, j


def _lc_radius(rebydx, ctl1r, cone, hemi, lat, truelat1) -> np.float32:
    """``rebydx * ctl1r/cone * (TAN(..lat..)/TAN(..truelat1..))**cone``."""

    num = tanf(F(F(F(F(F(90.0) * hemi) - lat) * RAD_PER_DEG) / F(2.0)))
    den = tanf(F(F(F(F(F(90.0) * hemi) - truelat1) * RAD_PER_DEG)
                 / F(2.0)))
    return F(F(F(rebydx * ctl1r) / cone) * powf(F(num / den), cone))


def _set_lc(truelat1, truelat2, stdlon, lat1, lon1, knowni, knownj, hemi,
            rebydx):
    """set_lc:1097-1135 -> (cone, rsw, polei, polej)."""

    cone = lc_cone(truelat1, truelat2)
    deltalon1 = F(lon1 - stdlon)
    if deltalon1 > F(180.0):
        deltalon1 = F(deltalon1 - F(360.0))
    if deltalon1 < F(-180.0):
        deltalon1 = F(deltalon1 + F(360.0))
    ctl1r = cosf(F(truelat1 * RAD_PER_DEG))
    rsw = _lc_radius(rebydx, ctl1r, cone, hemi, lat1, truelat1)
    arg = F(cone * F(deltalon1 * RAD_PER_DEG))
    polei = F(F(hemi * knowni) - F(F(hemi * rsw) * sinf(arg)))
    polej = F(F(hemi * knownj) + F(rsw * cosf(arg)))
    return cone, rsw, polei, polej


def _set_ps(truelat1, stdlon, lat1, lon1, knowni, knownj, hemi, rebydx):
    """set_ps:696-729 -> (rsw, polei, polej)."""

    reflon = F(stdlon + F(90.0))
    scale_top = F(F(1.0) + F(hemi * sinf(F(truelat1 * RAD_PER_DEG))))
    ala1 = F(lat1 * RAD_PER_DEG)
    rsw = F(F(F(rebydx * cosf(ala1)) * scale_top)
            / F(F(1.0) + F(hemi * sinf(ala1))))
    alo1 = F(F(lon1 - reflon) * RAD_PER_DEG)
    polei = F(knowni - F(rsw * cosf(alo1)))
    polej = F(knownj - F(F(hemi * rsw) * sinf(alo1)))
    return rsw, polei, polej


def _set_merc(truelat1, lat1, dx):
    """set_merc:1307-1331 -> (dlon, rsw)."""

    clain = cosf(F(RAD_PER_DEG * truelat1))
    dlon = F(dx / F(EARTH_RADIUS_M * clain))
    rsw = F(0.0)
    if lat1 != F(0.0):
        rsw = F(logf(tanf(F(F(0.5) * F(F(lat1 + F(90.0)) * RAD_PER_DEG))))
                / dlon)
    return dlon, rsw


def nint(value) -> int:
    """Fortran NINT on a REAL (:func:`woof.core.adaptive_timestep.nint`).

    The float32 is widened first; ``x + 0.5`` is then exact in binary64
    for any float32 below 2**28, so the rounding is NINT's own.
    """

    x = float(F(value))
    if not math.isfinite(x):
        raise ValueError(f"NINT of {x}")
    return _fortran_nint(x)


__all__ = ["PROJ_LC", "PROJ_PS", "PROJ_MERC", "PROJECTION_CODES", "UNSET",
           "WrfProjection", "WrfProjectionRefused", "lc_cone", "nint",
           "cosf", "sinf", "tanf", "logf", "log10f", "powf"]
