"""WRFDA's conventional observations from OBSPROC's ``obs_gts`` ASCII (EXPERIMENTAL).

A port of what WRFDA 4.7.1 does to ``ob.ascii`` before it computes an
innovation, for a regional Lambert-conformal ARW first guess on one tile, with
``ob_format = 2`` and ``thin_conv_ascii = .false.``:

* ``da_read_obs_ascii``: the header's own ``INFO``/``SRFC``/``EACH`` formats,
  the speed and direction to (u, v) rotation (``da_ffdduv``), the time window
  (``time_slots``), the domain check (``da_llxy``, ``da_llxy_wrf``,
  ``da_llxy_lc``, ``da_togrid``) and the per-type filling, including the
  sonde surface split, satem thickness layers, the geo/polar AMV split and the
  scatterometer error floor;
* ``da_obs_proc_station``: gross ``t``/``p``/``rh`` checks and ``q`` from
  ``rh`` (``da_tp_to_qs``);
* ``da_fill_obs_structures``: the ``q`` error from the ``rh`` error
  (``da_get_q_error``, ``f_qv_from_rh``) and the GNSS-RO geometric to
  geopotential height (``da_msl2geo1``);
* ``da_store_obs_grid_info``: ``proc_domain``.

Every arithmetic step is in WRFDA's order in binary64 (the oracle build is
``-fdefault-real-8``, no FMA). The transcendentals are correctly rounded by
default (:mod:`woof.da.var.crmath`), so every host gets the same words. With
``libm="host"`` on a glibc host, the result matches the oracle dump
(``woof_iv.bin``) word for word, except the GNSS-RO heights (below).

The output is :class:`WrfdaObs`: one :class:`ObsType` per WRFDA type, laid out
like WRFDA's ``iv`` (per-report info, per-level coordinates and
``(inv, qc, error)`` fields flattened report by report). At this stage
``inv`` holds the observed value; the innovation is computed later.
:meth:`WrfdaObs.to_obsset` exports the kinds that P0's
:class:`woof.grad.obs_cost.ObsSet` has a column for.

The background check (``check_max_iv``) needs ``H(x_b)``. It belongs to the
innovation step, not here.

GNSS-RO heights after ``da_msl2geo1`` differ from the oracle's by up to
four ulps. Each profile's gravity or radius term is one ulp off, and an
exhaustive search over the evaluation orders has not found where. GNSS-RO
is outside V1's operator list.
"""

from __future__ import annotations

import math
import re
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np


from . import crmath

#: :func:`read_obs_ascii`'s ``libm`` choices.
LIBMS = ("correctly-rounded", "host")


@lru_cache(maxsize=2)
def _libm(name: str = "correctly-rounded"):
    """The transcendentals WRFDA's gfortran build calls.

    ``"correctly-rounded"`` (the default, :mod:`woof.da.var.crmath`) gives the
    same words on every host, so the CUDA and Metal hosts preprocess
    identically. ``"host"`` is :func:`woof.core.host_libm.scalar`: the opt-in
    glibc 2.39 transcription where the bridge has it, else the host C
    library. On a glibc host that is the oracle's own libm.
    """
    if name == "correctly-rounded":
        return crmath
    if name != "host":
        raise ValueError(f"libm {name!r}: expected one of {LIBMS}")
    from woof.core.host_libm import scalar

    return types.SimpleNamespace(sin=scalar("sin"), cos=scalar("cos"), tan=scalar("tan"),
                                 exp=scalar("exp"), log10=scalar("log10"), pow_=scalar("pow"),
                                 sqrt=math.sqrt)


__all__ = [
    "MISSING", "MISSING_R", "MISSING_DATA", "TYPES", "LIBMS",
    "LambertWRFDA", "Field", "ObsType", "WrfdaObs",
    "read_obs_ascii", "fortran_format_fields", "julian_minutes",
]

# da_control.f90
MISSING = -888888
MISSING_R = -888888.0
MISSING_DATA = -88
PI = 3.1415926
RAD_PER_DEG = PI / 180.0
EARTH_RADIUS_M = 6370000.0
T_KELVIN = 273.15
ES_ALPHA, ES_BETA, ES_GAMMA = 611.2, 17.67, 243.5
RD_OVER_RV = 287.0 / 461.6
RD_OVER_RV1 = 1.0 - RD_OVER_RV
MAX_OB_LEVELS = 1001

# The types this port fills, in the oracle dump's order. ``surface`` types
# carry one level and no per-level coordinate.
TYPES = ("synop", "metar", "ships", "buoy", "sonde_sfc", "gpspw", "qscat",
         "sound", "airep", "pilot", "profiler", "geoamv", "polaramv", "satem", "gpsref")
SURFACE = ("synop", "metar", "ships", "buoy", "sonde_sfc")
_FIELDS = {
    "synop": ("u", "v", "t", "p", "q"), "metar": ("u", "v", "t", "p", "q"),
    "ships": ("u", "v", "t", "p", "q"), "buoy": ("u", "v", "t", "p", "q"),
    "sonde_sfc": ("u", "v", "t", "p", "q"), "gpspw": ("tpw",), "qscat": ("u", "v"),
    "sound": ("u", "v", "t", "q"), "airep": ("u", "v", "t", "q"), "pilot": ("u", "v"),
    "profiler": ("u", "v"), "geoamv": ("u", "v"), "polaramv": ("u", "v"),
    "satem": ("thickness",), "gpsref": ("ref", "p", "t", "q"),
}
# per-level coordinates each type keeps (h: height, p: pressure)
_COORDS = {
    "synop": ("h",), "metar": ("h",), "ships": ("h",), "buoy": ("h",), "sonde_sfc": ("h",),
    "gpspw": (), "qscat": ("h",), "sound": ("h", "p"), "airep": ("h", "p"),
    "pilot": ("h", "p"), "profiler": ("h", "p"), "geoamv": ("p",), "polaramv": ("p",),
    "satem": ("p",), "gpsref": ("h",),
}

# fm code -> WRFDA type (da_control's fm_index, the rows this port reads)
_FM = {12: "synop", 13: "ships", 17: "ships", 15: "metar", 16: "metar", 18: "buoy", 19: "buoy",
       32: "pilot", 33: "pilot", 34: "pilot", 35: "sound", 36: "sound", 37: "sound", 38: "sound",
       42: "airep", 96: "airep", 97: "airep", 86: "satem", 88: "geoamv", 111: "gpspw", 114: "gpspw",
       116: "gpsref", 132: "profiler", 281: "qscat"}


# --------------------------------------------------------------------------- formats
_TOKEN = re.compile(r"\s*(\d*)\s*(?:([,:()])|([Xx])|([AaIiFf])(\d*)(?:\.(\d+))?)")


def fortran_format_fields(fmt: str) -> list[tuple[str, int, int]]:
    """Expand a Fortran edit-descriptor list into ``(kind, width, decimals)``.

    Supports what OBSPROC writes: ``A``, ``I``, ``F``, ``nX``, ``:`` and
    repeated groups. ``kind`` is ``'A'``, ``'I'``, ``'F'`` or ``'X'``.
    """
    s = fmt.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"not a format: {fmt!r}")
    pos = 0

    def group(end_on_paren: bool) -> list:
        nonlocal pos
        out: list = []
        while pos < len(s):
            m = _TOKEN.match(s, pos)
            if not m:
                raise ValueError(f"cannot parse {fmt!r} at {s[pos:]!r}")
            rep, punct, xtok, tok, width, dec = m.groups()
            pos = m.end()
            if punct in (",", ":"):
                continue
            if punct == "(":
                inner = group(True)
                out.extend(inner * (int(rep) if rep else 1))
                continue
            if punct == ")":
                if end_on_paren:
                    return out
                raise ValueError(f"unbalanced {fmt!r}")
            if xtok:
                out.append(("X", int(rep) if rep else 1, 0))
                continue
            t = tok.upper()
            if not width:
                raise ValueError(f"{fmt!r}: {t} needs a width")
            n = int(rep) if rep else 1
            out.extend([(t, int(width), int(dec or 0))] * n)
        if end_on_paren:
            raise ValueError(f"unbalanced {fmt!r}")
        return out

    pos = 1
    s = s[:-1]
    return group(False)


def _read_record(line: str, fields: list[tuple[str, int, int]]) -> list:
    """List-free formatted READ with PAD='YES' and BLANK='NULL' (gfortran defaults)."""
    out = []
    col = 0
    for kind, w, d in fields:
        chunk = line[col : col + w].ljust(w) if col < len(line) else " " * w
        col += w
        if kind == "X":
            continue
        if kind == "A":
            out.append(chunk)
        elif kind == "I":
            t = chunk.replace(" ", "")
            out.append(int(t) if t not in ("", "+", "-") else 0)
        else:
            t = chunk.replace(" ", "")
            if t in ("", "+", "-", "."):
                out.append(0.0)
            elif "." in t or "e" in t.lower() or "d" in t.lower():
                out.append(float(t.lower().replace("d", "e")))
            else:
                # no decimal point: Fw.d scales by 10**-d (exact decimal, then rounded once)
                out.append(float(f"{t}e-{d}"))
    return out


def julian_minutes(year: int, month: int, day: int, hour: int, minute: int) -> float:
    """``da_get_julian_time`` (integer arithmetic truncating toward zero)."""
    def q(a, b):
        return int(a / b) if (a < 0) != (b < 0) else a // b
    a = q(month - 14, 12)
    iw3jdn = (day - 32075 + q(1461 * (year + 4800 + a), 4)
              + q(367 * (month - 2 - a * 12), 12)
              - q(3 * q(year + 4900 + a, 100), 4))
    ndays = iw3jdn - 2443510
    return float(ndays * 1440 + hour * 60 + minute)


def _date_minutes(text: str) -> float:
    # read(date,'(i4,4(1x,i2))') -- skips the separators
    vals = [int(text[0:4]), int(text[5:7]), int(text[8:10]), int(text[11:13]), int(text[14:16])]
    return julian_minutes(*vals)


# --------------------------------------------------------------------------- map projection
@dataclass(frozen=True)
class LambertWRFDA:
    """``da_map_set`` for ``PROJ_LC`` as ``da_setup_firstguess_wrf`` calls it.

    ``cen_lat``, ``cen_lon``, ``stand_lon``, ``truelat1``, ``truelat2``,
    ``moad_cen_lat`` and ``dx`` are the first guess's global attributes as the
    oracle reads them (``float32`` attributes widened to binary64); ``ide``
    and ``jde`` are the mass-point counts (``e_we - 1``, ``e_sn - 1``).
    """

    cen_lat: float
    cen_lon: float
    stand_lon: float
    truelat1: float
    truelat2: float
    moad_cen_lat: float
    dx: float
    ide: int
    jde: int
    libm: str = "correctly-rounded"
    cone: float = field(init=False)
    hemi: float = field(init=False)
    rebydx: float = field(init=False)
    polei: float = field(init=False)
    polej: float = field(init=False)
    knowni: float = field(init=False)
    knownj: float = field(init=False)

    def __post_init__(self):
        set_ = object.__setattr__
        cm = _libm(self.libm)
        hemi = -1.0 if self.truelat1 < 0.0 else 1.0
        rebydx = EARTH_RADIUS_M / self.dx
        t1, t2 = self.truelat1, self.truelat2 if abs(self.truelat2) <= 90.0 else self.truelat1
        if abs(t1 - t2) > 0.1:
            cone = cm.log10(cm.cos(t1 * RAD_PER_DEG)) - cm.log10(cm.cos(t2 * RAD_PER_DEG))
            cone = cone / (cm.log10(cm.tan((45.0 - abs(t1) / 2.0) * RAD_PER_DEG))
                           - cm.log10(cm.tan((45.0 - abs(t2) / 2.0) * RAD_PER_DEG)))
        else:
            cone = cm.sin(abs(t1) * RAD_PER_DEG)
        deltalon1 = self.cen_lon - self.stand_lon
        if deltalon1 > 180.0:
            deltalon1 = deltalon1 - 360.0
        if deltalon1 < -180.0:
            deltalon1 = deltalon1 + 360.0
        ctl1r = cm.cos(t1 * RAD_PER_DEG)
        ratio = (cm.tan((90.0 * hemi - self.cen_lat) * RAD_PER_DEG / 2.0)
                 / cm.tan((90.0 * hemi - t1) * RAD_PER_DEG / 2.0))
        rsw = rebydx * ctl1r / cone * cm.pow_(ratio, cone)
        arg = cone * (deltalon1 * RAD_PER_DEG)
        set_(self, "cone", cone)
        set_(self, "hemi", hemi)
        set_(self, "rebydx", rebydx)
        set_(self, "polei", 1.0 - hemi * rsw * cm.sin(arg))
        set_(self, "polej", 1.0 + rsw * cm.cos(arg))
        # real(ide - ids + 2) / 2.0 with ids = 1
        set_(self, "knowni", float(self.ide - 1 + 2) / 2.0)
        set_(self, "knownj", float(self.jde - 1 + 2) / 2.0)

    @classmethod
    def from_attributes(cls, attrs, libm: str = "correctly-rounded") -> "LambertWRFDA":
        """From a WRF first guess's global attributes (``CEN_LAT``, ``DX``...).

        The ``float32`` attribute values are widened to binary64, as
        WRFDA's ``-fdefault-real-8`` build widens them. Only ``MAP_PROJ = 1``
        (Lambert conformal) is ported.
        """
        if int(attrs["MAP_PROJ"]) != 1:
            raise NotImplementedError(f"MAP_PROJ {int(attrs['MAP_PROJ'])}: only Lambert conformal (1) is ported")

        def f(key: str) -> float:
            return float(np.float32(attrs[key]))

        return cls(cen_lat=f("CEN_LAT"), cen_lon=f("CEN_LON"), stand_lon=f("STAND_LON"),
                   truelat1=f("TRUELAT1"), truelat2=f("TRUELAT2"), moad_cen_lat=f("MOAD_CEN_LAT"),
                   dx=f("DX"), ide=int(attrs["WEST-EAST_GRID_DIMENSION"]) - 1,
                   jde=int(attrs["SOUTH-NORTH_GRID_DIMENSION"]) - 1, libm=libm)

    def llxy(self, lat: float, lon: float) -> tuple[float, float]:
        """``da_llxy_lc`` then ``da_llxy_wrf``'s shift to the known point."""
        cm = _libm(self.libm)
        deltalon = lon - self.stand_lon
        if deltalon > 180.0:
            deltalon = deltalon - 360.0
        if deltalon < -180.0:
            deltalon = deltalon + 360.0
        ctl1r = cm.cos(self.truelat1 * RAD_PER_DEG)
        ratio = (cm.tan((90.0 * self.hemi - lat) * RAD_PER_DEG / 2.0)
                 / cm.tan((90.0 * self.hemi - self.truelat1) * RAD_PER_DEG / 2.0))
        rm = self.rebydx * ctl1r / self.cone * cm.pow_(ratio, self.cone)
        arg = self.cone * (deltalon * RAD_PER_DEG)
        x = self.polei + self.hemi * rm * cm.sin(arg)
        y = self.polej - rm * cm.cos(arg)
        if self.hemi == -1.0:
            x = 2.0 - x
            y = 2.0 - y
        return x + self.knowni - 1.0, y + self.knownj - 1.0

    def ffdduv(self, speed: float, direction: float, lon: float) -> tuple[float, float]:
        """``da_ffdduv(convert_fd2uv)``: speed and direction to grid-relative u, v."""
        cm = _libm(self.libm)
        conv = 180.0 / PI
        aearth = direction / conv
        uearth = -speed * cm.sin(aearth)
        vearth = -speed * cm.cos(aearth)
        xlonrt = self.stand_lon - lon
        if xlonrt > 180.0:
            xlonrt = xlonrt - 360.0
        if xlonrt < -180.0:
            xlonrt = xlonrt + 360.0
        ang = xlonrt * self.cone / conv
        if self.moad_cen_lat < 0.0:
            ang = -ang
        sa, ca = cm.sin(ang), cm.cos(ang)
        return vearth * sa + uearth * ca, vearth * ca - uearth * sa


def _togrid(x: float, ib: int, ie: int) -> tuple[int, float, float]:
    i = int(x)
    if i < ib:
        i = ib
    elif i >= ie:
        i = ie - 1
    dx = x - float(i)
    return i, dx, 1.0 - dx


# --------------------------------------------------------------------------- physics
def tp_to_qs(t: float, p: float, libm: str = "correctly-rounded") -> tuple[float, float]:
    """``da_tp_to_qs``: saturation vapour pressure and specific humidity."""
    t_c = t - T_KELVIN
    cm = _libm(libm)
    es = ES_ALPHA * cm.exp(ES_BETA * t_c / (t_c + ES_GAMMA))
    qs = RD_OVER_RV * es / (p - RD_OVER_RV1 * es)
    return es, qs


def _f_qv_from_rh(rh: float, t_k: float, rh0: float, t_k0: float, p_pa0: float, libm: str) -> float:
    p_mb0 = p_pa0 / 100.0
    cm = _libm(libm)
    a = t_k0 - 273.15 + 243.5
    e = cm.exp(17.67 * (t_k0 - 273.15) / (t_k0 - 273.15 + 243.5))
    es = 6.112 * 17.67 * 243.5 * t_k / (a * a) * e
    es0 = 6.112 * e
    qs = 0.622 * (p_mb0 * es) / ((p_mb0 - es0) * (p_mb0 - es0))
    qs0 = 0.622 * es0 / (p_mb0 - es0)
    qv = 0.01 * (rh0 * qs + rh * qs0)
    if 0.01 * rh0 * qs0 < 0.0:
        qv = 0.0
    return qv


def get_q_error(p: float, t: float, q: float, t_error: float, rh_error: float,
                libm: str = "correctly-rounded") -> float:
    """``da_get_q_error``."""
    if abs(p - MISSING_R) > 1.0 and abs(t - MISSING_R) > 1.0 and abs(q - MISSING_R) > 1:
        _, qs = tp_to_qs(t, p, libm)
        rh = 100.0 * q / qs
        if rh > 100.0:
            rh = 100.0
        q_error = _f_qv_from_rh(rh_error, t_error, rh, t, p, libm)
    else:
        q_error = MISSING_R
    if q_error == 0.0:
        q_error = MISSING_R
    return q_error


def msl2geo1(z_km: float, lat: float, libm: str = "correctly-rounded") -> float:
    """``da_msl2geo1``: geometric to geopotential height (km)."""
    cm = _libm(libm)
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


# --------------------------------------------------------------------------- containers
class _F:
    """One ``field_type`` (inv, qc, error) while a report is processed."""

    __slots__ = ("inv", "qc", "error")

    def __init__(self, inv=MISSING_R, qc=MISSING, error=MISSING_R):
        self.inv, self.qc, self.error = inv, qc, error

    def copy(self) -> "_F":
        return _F(self.inv, self.qc, self.error)


@dataclass
class Field:
    inv: np.ndarray
    qc: np.ndarray
    error: np.ndarray

    def __len__(self) -> int:
        return len(self.inv)


@dataclass
class ObsType:
    """One WRFDA observation type, flattened like the oracle dump.

    ``info`` columns are per report (``name``, ``platform``, ``id``,
    ``date``, ``lat``, ``lon``, ``elv``, ``x``, ``y``, ``i``, ``j``, ``dx``,
    ``dy``, ``dxm``, ``dym``, ``proc_domain``, ``obs_global_index``,
    ``levels``); ``coords`` and ``fields`` are per level, report by report
    (``offsets[n]:offsets[n+1]`` are report ``n``'s levels). ``i`` and ``j``
    are WRFDA's 1-based indices.
    """

    name: str
    info: dict
    coords: dict
    fields: dict
    extra: dict = field(default_factory=dict)

    @property
    def nlocal(self) -> int:
        return len(self.info["levels"])

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.info["levels"])]).astype(np.int64)


@dataclass
class WrfdaObs:
    types: dict
    ntotal: dict
    analysis_minutes: float

    def __getitem__(self, name: str) -> ObsType:
        return self.types[name]

    def counts(self) -> dict:
        return {k: v.nlocal for k, v in self.types.items()}

    def to_obsset(self):
        """P0's ``ObsSet`` for the kinds it has (surface t, u, v; upper t, u, v).

        Only observations with ``qc >= 0`` are exported; winds are WRFDA's
        grid-relative components, as the variational operators use them.
        The row order is type by type in :data:`TYPES` order, then WRFDA's.
        """
        from woof.grad.obs_cost import ObsSet

        cols: dict[str, list] = {k: [] for k in ("platform", "kind", "station", "time", "lat", "lon",
                                                 "pressure_pa", "elevation_m", "value", "error")}

        def add(t: ObsType, n: int, kind: str, value: float, error: float, p: float):
            cols["platform"].append(t.name)
            cols["kind"].append(kind)
            cols["station"].append(t.info["id"][n].strip())
            d = t.info["date"][n]
            cols["time"].append(datetime(int(d[0:4]), int(d[5:7]), int(d[8:10]), int(d[11:13]),
                                         int(d[14:16]), int(d[17:19] or 0), tzinfo=timezone.utc))
            cols["lat"].append(t.info["lat"][n])
            cols["lon"].append(t.info["lon"][n])
            cols["pressure_pa"].append(p)
            cols["elevation_m"].append(t.info["elv"][n])
            cols["value"].append(value)
            cols["error"].append(error)

        surface = {"t": "surface_temperature", "u": "surface_wind_u", "v": "surface_wind_v"}
        upper = {"t": "upper_temperature", "u": "upper_wind_u", "v": "upper_wind_v"}
        for name in TYPES:
            t = self.types.get(name)
            if t is None:
                continue
            off = t.offsets
            for n in range(t.nlocal):
                for lv in range(off[n], off[n + 1]):
                    for var, f in t.fields.items():
                        table = surface if name in SURFACE or name == "qscat" else upper
                        if var not in table or f.qc[lv] < 0:
                            continue
                        if table is upper:
                            p = t.coords.get("p")
                            if p is None or p[lv] <= 0.0:
                                continue
                            add(t, n, table[var], f.inv[lv], f.error[lv], p[lv])
                        else:
                            add(t, n, table[var], f.inv[lv], f.error[lv], np.nan)
        return ObsSet.from_columns(**cols)


class _Builder:
    def __init__(self, name: str):
        self.name = name
        self.info: dict[str, list] = {k: [] for k in (
            "name", "platform", "id", "date", "lat", "lon", "elv", "x", "y", "i", "j",
            "dx", "dy", "dxm", "dym", "proc_domain", "obs_global_index", "levels")}
        self.coords: dict[str, list] = {k: [] for k in _COORDS[name]}
        self.fields: dict[str, list] = {k: [] for k in _FIELDS[name]}
        self.extra: dict[str, list] = {}

    def add_report(self, rep, loc, levels: int, global_index: int):
        inf = self.info
        inf["name"].append(rep["name"])
        inf["platform"].append(rep["platform"])
        inf["id"].append(rep["id"])
        inf["date"].append(rep["date"])
        inf["lat"].append(rep["lat"])
        inf["lon"].append(rep["lon"])
        inf["elv"].append(rep["elv"])
        for k in ("x", "y", "i", "j", "dx", "dy", "dxm", "dym", "proc_domain"):
            inf[k].append(loc[k])
        inf["obs_global_index"].append(global_index)
        inf["levels"].append(levels)

    def add_level(self, coords: dict, fields: dict):
        for k in self.coords:
            self.coords[k].append(coords[k])
        for k in self.fields:
            f = fields[k]
            self.fields[k].append((f.inv, f.qc, f.error))

    def build(self) -> ObsType:
        info = {}
        for k, v in self.info.items():
            if k in ("name", "platform", "id", "date"):
                info[k] = np.asarray(v, dtype=str) if v else np.zeros(0, dtype="U1")
            elif k in ("i", "j", "obs_global_index", "levels"):
                info[k] = np.asarray(v, dtype=np.int64)
            elif k == "proc_domain":
                info[k] = np.asarray(v, dtype=bool)
            else:
                info[k] = np.asarray(v, dtype=np.float64)
        coords = {k: np.asarray(v, dtype=np.float64) for k, v in self.coords.items()}
        fields = {}
        for k, v in self.fields.items():
            a = np.asarray(v, dtype=object).reshape(-1, 3) if v else np.zeros((0, 3))
            fields[k] = Field(np.asarray(a[:, 0], dtype=np.float64), np.asarray(a[:, 1], dtype=np.int64),
                              np.asarray(a[:, 2], dtype=np.float64))
        extra = {k: np.asarray(v, dtype=np.float64) for k, v in self.extra.items()}
        return ObsType(self.name, info, coords, fields, extra)


# --------------------------------------------------------------------------- the reader
_EACH_KEYS = ("p", "speed", "v", "height", "height_qc", "height_error", "t", "td", "rh")


def read_obs_ascii(path, proj: LambertWRFDA, *, analysis_date: str, time_window_min: str,
                   time_window_max: str, top_km_gpsro: float = 30.0, bot_km_gpsro: float = 0.0,
                   use: tuple[str, ...] = TYPES) -> WrfdaObs:
    """Read OBSPROC's ``obs_gts`` ASCII as WRFDA 4.7.1 does (one FGAT slot).

    ``analysis_date`` and the window bounds are WRFDA namelist strings
    (``2008-02-05_12:00:00``...). ``use`` lists the types to keep (WRFDA's
    ``use_*obs``); the default is every type this port fills.
    """
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    pos = 0
    while pos < len(lines) and lines[pos][:6] != "EACH  ":
        pos += 1
    if pos >= len(lines):
        raise ValueError(f"{path}: no 'EACH  ' header line")
    pos += 1
    fmts = []
    for _ in range(3):
        # read(fmt='(A,1X,A)') fmt_name(len=10), fmt_*
        fmts.append(lines[pos][11:].strip())
        pos += 1
    fmt_info, fmt_srfc, fmt_each = (fortran_format_fields(f) for f in fmts)
    uvq_direct = lines[pos][:21] == "MODIFIED FILTERED OBS"
    if uvq_direct:
        raise NotImplementedError("'MODIFIED FILTERED OBS' (uvq_direct) files are not ported")
    pos += 1

    t0 = _date_minutes(time_window_min)
    t1 = _date_minutes(time_window_max)
    tana = _date_minutes(analysis_date)
    ide, jde = proj.ide, proj.jde
    ids = jds = its = jts = 1
    ite, jte = ide, jde

    builders = {name: _Builder(name) for name in TYPES}
    ntotal = {name: 0 for name in TYPES}

    while pos < len(lines):
        rec = _read_record(lines[pos], fmt_info)
        pos += 1
        platform, date_char, name, nlev, lat, lon, elv, sid = rec
        if not platform.strip() and nlev == 0:
            break
        if lon == 180.0:
            lon = -180.0
        if lat < -89.9999 or lat > 89.9999:
            lon = 0.0
        fm = int(platform[3:5]) if platform[5] == " " else int(platform[3:6])
        slp_inv, slp_qc, slp_err, pw_inv, pw_qc, pw_err = _read_record(lines[pos], fmt_srfc)[:6]
        pos += 1
        each = []
        for i in range(nlev):
            line = lines[pos]
            pos += 1
            if i >= MAX_OB_LEVELS:
                continue
            v = _read_record(line, fmt_each)
            lev = {
                "p": _F(v[0], v[1], v[2]), "speed": _F(v[3], v[4], v[5]), "v": _F(v[6], v[7], v[8]),
                "height": v[9], "height_qc": v[10],
                "t": _F(v[12], v[13], v[14]), "td": _F(v[15], v[16], v[17]), "rh": _F(v[18], v[19], v[20]),
                "u": _F(MISSING_R, MISSING, MISSING_R), "q": _F(MISSING_R, MISSING, MISSING_R),
            }
            if fm not in (116, 118):
                if lev["speed"].qc != MISSING_DATA and lev["v"].qc != MISSING_DATA:
                    u, vv = proj.ffdduv(lev["speed"].inv, lev["v"].inv, lon)
                    lev["u"] = lev["speed"].copy()
                    lev["u"].inv = u
                    lev["v"].inv = vv
                else:
                    lev["u"] = _F(MISSING_R, MISSING_DATA, MISSING_R)
                    lev["v"] = _F(MISSING_R, MISSING_DATA, MISSING_R)
            each.append(lev)

        obs_time = _date_minutes(date_char)
        if obs_time < t0 or obs_time >= t1:
            continue

        x, y = proj.llxy(lat, lon)
        i, dxx, dxm = _togrid(x, its - 2, ite + 2)
        j, dyy, dym = _togrid(y, jts - 2, jte + 2)
        if int(x) < ids or int(x) > ide or int(y) < jds or int(y) > jde:
            continue  # outside_all
        outside = i < ids or i >= ide or j < jds or j >= jde or i < its - 1 or i > ite or j < jts - 1 or j > jte

        nlevels = min(nlev, MAX_OB_LEVELS)
        if nlevels < 1 and fm not in (111, 114):
            continue
        levs = nlevels

        if fm not in (86, 118):
            _proc_station(each, nlevels, proj.libm)

        kind = _FM.get(fm)
        if kind is None:
            continue
        if kind == "geoamv" and ("MODIS" in name or "modis" in name or "AVHRR" in sid):
            kind = "polaramv"
        if kind not in use:
            continue
        if kind == "satem" and pw_inv > 10.0:
            continue
        rep = {"name": name, "platform": platform, "id": sid, "date": date_char,
               "lat": lat, "lon": lon, "elv": elv}
        loc = {"x": x, "y": y, "i": i, "j": j, "dx": dxx, "dy": dyy, "dxm": dxm, "dym": dym,
               "proc_domain": its <= i <= ite and jts <= j <= jte}

        ntotal[kind] += 1
        if kind == "polaramv" and not outside:
            ntotal[kind] += 1  # WRFDA counts an inside polar AMV twice
        if kind == "sound":
            ntotal["sonde_sfc"] += 1
        if outside:
            continue
        b = builders[kind]
        is_surface = False

        if kind in ("synop", "ships", "metar", "buoy"):
            e = each[0]
            e["v"].error = e["u"].error
            b.add_level({"h": e["height"]}, {k: e[k] for k in _FIELDS[kind]})
        elif kind == "qscat":
            e = each[0]
            e["v"].error = e["u"].error
            u, vv = e["u"].copy(), e["v"].copy()
            u.error = max(e["u"].error, 1.0)
            vv.error = max(e["v"].error, 1.0)
            b.add_level({"h": e["height"]}, {"u": u, "v": vv})
        elif kind in ("pilot", "profiler", "geoamv", "polaramv", "airep"):
            for e in each[:nlevels]:
                e["v"].error = e["u"].error
                b.add_level({"h": e["height"], "p": e["p"].inv}, {k: e[k] for k in _FIELDS[kind]})
        elif kind == "sound":
            surface_level = 0
            for n_, e in enumerate(each[:nlevels], start=1):
                if elv != MISSING_R and e["height"] != MISSING_R and abs(elv - e["height"]) < 0.1:
                    is_surface = True
                    surface_level = n_
                    levs = nlevels - 1
                    break
            sb = builders["sonde_sfc"]
            if surface_level:
                e = each[surface_level - 1]
                e["v"].error = e["u"].error
                sb.add_level({"h": e["height"]}, {k: e[k] for k in _FIELDS["sonde_sfc"]})
            else:
                miss = _F(MISSING_R, MISSING, abs(MISSING_R))
                sb.add_level({"h": MISSING_R}, {k: miss for k in _FIELDS["sonde_sfc"]})
            for n_, e in enumerate(each[:nlevels], start=1):
                if n_ == surface_level:
                    continue
                e["v"].error = e["u"].error
                b.add_level({"h": e["height"], "p": e["p"].inv}, {k: e[k] for k in _FIELDS["sound"]})
        elif kind == "satem":
            b.extra.setdefault("ref_p", []).append(slp_inv)
            ts = [each[n_]["t"] for n_ in range(nlevels)]
            th = [ts[0].copy()]
            for n_ in range(1, nlevels):
                f = ts[n_].copy()
                if ts[n_].qc != MISSING_DATA and ts[n_ - 1].qc != MISSING_DATA:
                    f.inv = ts[n_].inv - ts[n_ - 1].inv
                else:
                    f.inv, f.qc = MISSING_R, MISSING_DATA
                th.append(f)
            for n_ in range(nlevels - 1, 0, -1):
                th[n_].error = math.sqrt(th[n_].error * th[n_].error + th[n_ - 1].error * th[n_ - 1].error)
            th[0].error = math.sqrt(th[0].error * th[0].error + pw_err * pw_err)
            for n_ in range(nlevels):
                b.add_level({"p": each[n_]["p"].inv}, {"thickness": th[n_]})
        elif kind == "gpspw":
            b.add_level({}, {"tpw": _F(pw_inv, pw_qc, pw_err)})
        elif kind == "gpsref":
            for e in each[:nlevels]:
                ref = e["td"].copy()
                if e["height"] > top_km_gpsro * 1000.0 or e["height"] < bot_km_gpsro * 1000.0:
                    ref.qc = -77
                b.add_level({"h": e["height"]}, {"ref": ref, "p": e["p"], "t": e["t"], "q": e["q"]})

        # WRFDA writes the info row when ``is_surface or gpspw or levs > 0``;
        # every report that gets here satisfies that (``nlevels >= 1`` or
        # gpspw, and a sounding's ``levs`` drops to 0 only with a surface level).
        b.add_report(rep, loc, 1 if kind in SURFACE or kind in ("qscat", "gpspw") else levs, ntotal[kind])
        if kind == "sound":
            builders["sonde_sfc"].add_report(rep, loc, 1, ntotal["sound"])

    types = {k: builders[k].build() for k in TYPES}
    _fill(types, proj.libm)
    return WrfdaObs(types=types, ntotal=ntotal, analysis_minutes=tana)


def _proc_station(each: list, nlevels: int, libm: str) -> None:
    """``da_obs_proc_station`` for ``ob_format_ascii``."""
    for e in each[:nlevels]:
        if e["t"].inv < 75.0 or e["t"].inv > 350.0:
            e["t"].inv, e["t"].qc = MISSING_R, MISSING
        if e["p"].inv <= 0.0 or e["p"].inv > 110000.0:
            e["p"].inv, e["p"].qc = MISSING_R, MISSING
        if e["rh"].inv < 0.0:
            e["rh"].inv, e["rh"].qc = MISSING_R, MISSING
        po, to, rho = e["p"].inv, e["t"].inv, e["rh"].inv
        if abs(po - MISSING_R) > 1.0 and abs(to - MISSING_R) > 1.0 and abs(rho - MISSING_R) > 1.0:
            _, qs = tp_to_qs(to, po, libm)
            qvo = qs if rho > 100.0 else rho * qs / 100.0
        else:
            qvo = MISSING_R
        e["q"] = _F(qvo, e["rh"].qc, e["rh"].error)


def _fill(types: dict, libm: str) -> None:
    """``da_fill_obs_structures``: q errors from rh errors; GNSS-RO heights."""
    for name in ("synop", "metar", "ships", "buoy", "sonde_sfc", "sound", "airep"):
        t = types[name]
        q, tt = t.fields["q"], t.fields["t"]
        p = t.fields["p"].inv if "p" in t.fields else t.coords["p"]
        for k in range(len(q)):
            qe = get_q_error(p[k], tt.inv[k], q.inv[k], tt.error[k], q.error[k], libm)
            q.error[k] = qe
            if qe == MISSING_R:
                q.qc[k] = MISSING_DATA
    g = types["gpsref"]
    off = g.offsets
    for n in range(g.nlocal):
        for k in range(off[n], off[n + 1]):
            g.coords["h"][k] = msl2geo1(g.coords["h"][k] / 1000.0, g.info["lat"][n], libm) * 1000.0
