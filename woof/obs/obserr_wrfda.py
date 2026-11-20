"""Observation errors as WRFDA's OBSPROC assigns them (``obs_err_afwa``, WRF v4.7.1).

OBSPROC gives every observation level an error taken from a table, ``obserr.txt``
(shipped here as ``data/wrfda_obserr.txt``, see its NOTICE).  This module ports
that table reader and the per-observation lookup, so a decoded report from any
source can carry the error WRFDA would give it.

How OBSPROC does it, and what is ported
---------------------------------------
* Tables are defined on 15 pressure levels (1000..10 hPa) for height, temperature,
  relative humidity and pressure, and on 26 levels (1100..10 hPa) for wind.  They
  start as the NCEP profile (``module_err_ncep``, 6 levels, interpolated onto the
  levels) and the file overrides them sensor by sensor.  Which file row feeds
  which platform is hard-wired in ``READ_OBSERR_*``; ``_FEEDS`` below is that map
  (rows that no case names fall into the ``other`` table, which nothing reads).
* The error of a level is a 1-D interpolation of the platform's table at the level's
  PRESSURE: ``intplin`` (linear in p) or ``intplog`` (linear in ln p), ending flat
  past the first and last level (``module_intp``; ``locate`` ported as is).  Which
  of the two is used per slot is the Fortran's, including that SYNOP height
  uses ``intplin`` while every other platform's height uses ``intplog``.
* Slots: ``p`` pressure (Pa), ``u``/``v`` wind components (m/s), ``speed``/``direction``
  (the speed and direction columns of obs_gts hold THESE errors, not u/v: all equal
  unless ``wind_sd`` selects the separate SPD and DIR tables, and the NCEP default
  gives the direction 35), ``height`` (m), ``t`` (K), ``td`` (= ``t``), ``rh`` (percent).
  SATEM also has ``thickness`` (= height error), which the obs_gts writer prints in the
  temperature column.  The writer also hard-codes a filler record for buoy reports
  with no levels (module_write: p 100 Pa, speed 1.4, direction 5, t/td 2, rh 10);
  that is not a table lookup and is not ported.
* FM code -> platform is ``fm_decoder``; an FM it does not know takes the NCEP
  profile (direction error 35 degrees, ``DEFAULT`` case).
* OBSPROC stops when a level has no pressure.  Pressure is recovered earlier
  (``recover_pressure_from_height``); for a single level that is
  ``floor(ref_pres(height))`` on the reference atmosphere (``module_mm5``), which is
  what ``errors`` does when only a height is given.  The multi-level recovery
  (hydrostatic interpolation between levels that have both) is not ported.

Not ported: SSMT1/SSMT2/SSMI/GPS refractivity and TC bogus (FM 135), whose
errors depend on more than a table lookup; ``errors`` raises for those FM
codes.  The qv error OBSPROC derives afterwards is not part of obs_gts errors
here either.

Precision: OBSPROC is ``real*4`` throughout (tables, pressures, interpolation);
this port is binary64 and follows the Fortran order of operations, so values
agree to single-precision rounding (obs_gts prints two decimals).
"""

from __future__ import annotations

import math
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "wrfda_obserr.txt"

# Pressure levels (Pa) of the tables, index 0 = first level (module_err_afwa 1.1/1.2).
LEVELS = [h * 100.0 for h in (1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10)]
WIND_LEVELS = [h * 100.0 for h in (1100, 1050, 1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500,
                                   450, 400, 350, 300, 250, 200, 150, 100, 50, 40, 30, 20, 10)]

# module_err_ncep: the NCEP profile every table starts from (Pa, m/s, K, %, Pa, m).
_NCEP = {
    "k": [100000.0, 70000.0, 50000.0, 30000.0, 10000.0, 5000.0],
    "uv": [1.4, 2.4, 2.8, 3.4, 2.5, 2.7],
    "t": [1.8, 1.3, 1.3, 2.0, 3.1, 4.0],
    "rh": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "p": [100.0, 72.7, 54.5, 36.4, 18.2, 13.6],
    "h": [7.0, 8.0, 8.6, 18.8, 39.4, 59.3],
}

GROUPS = ("synop", "ships", "buoys", "metar", "airep", "tamdar", "pilot", "profl", "sound", "satem",
          "satob", "ssmt1", "ssmt2", "ssmi", "tovs", "airs", "other")

# READ_OBSERR_*: block -> file row (sensor name) -> tables it fills.  A row whose name
# is not listed fills "other" (the DEFAULT case), and a later row overwrites an earlier one.
_WIND = {
    "RAOBS": ("sound", "synop", "ships", "metar"), "PIBALS": ("pilot",), "PROFL": ("profl",),
    "BUOY": ("buoys",), "AIREPS": ("airep",), "TAMDARS": ("tamdar",),
    "US LOW LEVEL WINDS": ("satob", "satem", "ssmt1", "ssmt2", "ssmi", "tovs"),
}
_DMSP = ("ssmt1", "ssmt2", "ssmi", "tovs")
_FEEDS = {
    "height": {
        "RAOBS": ("sound",), "PIBALS": ("pilot",), "AIREPS": ("airep",), "TAMDARS": ("tamdar",),
        "AIRSRET": ("airs",), "SURFACE LAND": ("synop", "metar"), "SURFACE SHIP": ("ships",),
        "BUOY": ("buoys",), "NOAA - A RETRIEVAL": ("satob",), "NOAA - CLEAR PATH RETRIEVAL": ("satem",),
        "DMSP - A RETRIEVAL": _DMSP, "PROFL": ("profl",),
    },
    "uv": _WIND, "spd": _WIND, "dir": _WIND,
    "t": {
        "SURFACE LAND": ("synop", "metar"), "SURFACE SHIP": ("ships",), "BUOY": ("buoys",),
        "RAOBS": ("sound",), "PIBALS": ("pilot", "profl"), "PROFL": ("pilot", "profl"),
        "NOAA - A RETRIEVAL": ("satob", "satem"), "AIREPS": ("airep",), "TAMDARS": ("tamdar",),
        "AIRSRET": ("airs",), "DMSP - A RETRIEVAL": _DMSP,
    },
    "rh": {
        "SURFACE LAND": ("synop", "ships", "buoys", "metar"), "RAOBS": ("sound",),
        "PIBALS": ("pilot", "profl"), "PROFL": ("pilot", "profl"),
        "NOAA - A RETRIEVAL": ("satob", "satem"), "AIREPS": ("airep",), "TAMDARS": ("tamdar",),
        "AIRSRET": ("airs",), "DMSP - A RETRIEVAL": _DMSP,
    },
    "p": {
        "SURFACE LAND": ("synop", "metar"), "SURFACE SHIP": ("ships",), "RAOBS": ("sound",),
        "PIBALS": ("pilot", "profl"), "PROFL": ("pilot", "profl"),
        "NOAA - A RETRIEVAL": ("satob", "satem"), "AIREPS": ("airep",), "TAMDARS": ("tamdar",),
        "AIRSRET": ("airs",), "DMSP - A RETRIEVAL": _DMSP,
    },
}
# block -> (keyword on its header row, values per row).  Rows are "(5(1X,F5.1))" (RH "F5.2")
# or, for wind, "(7(1X,F5.1))": fixed columns, as Fortran reads them.
_BLOCKS = {
    "height": ("HEIGHT SENSOR ERRORS", 5), "uv": ("UV SENSOR ERRORS", 7), "spd": ("SPD SENSOR ERRORS", 7),
    "dir": ("DIR SENSOR ERRORS", 7), "t": ("TEMP SENSOR ERRORS", 5), "rh": ("RH SENSOR ERRORS ( % )", 5),
    "p": ("PRESSURE SENSOR ERRORS", 5),
}
# The row text that ends a block's search: the next block's keyword (READ_OBSERR_* endword).
_END = {"height": "UV SENSOR ERRORS", "uv": "SPD SENSOR ERRORS", "spd": "DIR SENSOR ERRORS",
        "dir": "TEMP SENSOR ERRORS", "t": "RH SENSOR ERRORS ( % )", "rh": "PRESSURE SENSOR ERRORS", "p": None}

# fm_decoder: FM code -> platform name used by obs_err_afwa.
_PLATFORM = {12: "SYNOP", 14: "SYNOP", 13: "SHIP", 15: "METAR", 16: "METAR", 18: "BUOY", 19: "BUOY",
             32: "PILOT", 33: "PILOT", 34: "PILOT", 35: "SOUND", 36: "SOUND", 37: "SOUND", 38: "SOUND",
             42: "AMDAR", 86: "SATEM", 88: "SATOB", 96: "AIREP", 97: "AIREP", 101: "TAMDAR",
             131: "TOVS", 132: "PROFL", 133: "AIRSRET", 281: "QSCAT"}
_NOT_PORTED = {116, 118, 121, 122, 125, 126, 135}  # GPS refractivity, SSMT1/2, SSMI, TC bogus

# platform -> (table group, wind flag in the namelist) per slot; height interpolation is
# "lin" for SYNOP only.  AIRSRET has no wind and no dew point.
_TABLE = {"SYNOP": "synop", "SHIP": "ships", "BUOY": "buoys", "METAR": "metar", "PILOT": "pilot",
          "PROFL": "profl", "SOUND": "sound", "SATEM": "satem", "SATOB": "satob", "AIREP": "airep",
          "AMDAR": "airep", "TAMDAR": "tamdar", "QSCAT": "tovs", "TOVS": "tovs", "AIRSRET": "airs"}

# Reference atmosphere (module_mm5 ref_height/ref_pres) with the OBSPROC namelist defaults
# base_pres=100000, base_temp=290, base_lapse=50, base_strat_temp=215, base_tropo_pres=20000, ptop=1000.
_G, _GASR = 9.81, 287.04
_PS0, _TS0, _TLP, _PIS0, _TIS0, _PTOP = 100000.0, 290.0, 50.0, 20000.0, 215.0, 1000.0


def _ref_height(pres: float) -> float:
    if pres >= _PIS0:
        cc = math.log(pres / _PS0)
        return -(_GASR * _TS0 / _G * cc + _GASR * _TLP / (2.0 * _G) * cc * cc)
    return _ref_height(_PIS0) - _GASR * _TIS0 / _G * math.log(pres / _PIS0)


def ref_pres(height: float) -> float:
    """Reference-atmosphere pressure (Pa) at ``height`` (m), as ``module_mm5.ref_pres``."""
    htop = _ref_height(_PTOP)
    hh = min(htop, height) if htop > 0.0 else height
    h_tropo = _ref_height(_PIS0)
    if hh <= h_tropo:
        bb = _GASR * _TS0 / _G
        aa = _GASR * _TLP / (2.0 * _G)
        dd = (-bb + math.sqrt(bb * bb - 4.0 * aa * hh)) / (2.0 * aa)
        return _PS0 * math.exp(dd)
    return _PIS0 * math.exp((h_tropo - hh) / (_GASR * _TIS0 / _G))


def locate(x: float, xx: list[float]) -> int:
    """module_intp ``locate``: 1-based index jl with xx[jl] <= x < xx[jl+1] (0 or n outside)."""
    n = len(xx)
    ascnd = xx[n - 1] >= xx[0]
    jl, ju = 0, n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascnd == (x >= xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    if x == xx[0]:
        return 1
    if x == xx[n - 1]:
        return n - 1
    return jl


def intplin(x: float, xx: list[float], yy: list[float]) -> float:
    """module_intp ``intplin``: linear in x, the end value outside the table."""
    n = len(xx)
    jl = locate(x, xx)
    if jl <= 0:
        return yy[0]
    if jl >= n:
        return yy[n - 1]
    val = (xx[jl] - x) * yy[jl - 1] + (x - xx[jl - 1]) * yy[jl]
    return val / (xx[jl] - xx[jl - 1])


def intplog(x: float, xx: list[float], yy: list[float]) -> float:
    """module_intp ``intplog``: linear in ln x, the end value outside the table."""
    n = len(xx)
    jl = locate(x, xx)
    if jl <= 0:
        return yy[0]
    if jl >= n:
        return yy[n - 1]
    val = math.log(xx[jl] / x) * yy[jl - 1] + math.log(x / xx[jl - 1]) * yy[jl]
    return val / math.log(xx[jl] / xx[jl - 1])


def _fortran_floats(line: str, nfields: int) -> list[float]:
    """Read ``(n(1X,F5.d))``: fixed columns, a blank field is 0."""
    out = []
    for i in range(nfields):
        s = line[i * 6 + 1:(i + 1) * 6].strip()
        out.append(float(s) if s else 0.0)
    return out


def _sensor(line: str) -> str:
    """The row's name: text from the first character that is not blank, '.' or a digit."""
    for c, ch in enumerate(line.rstrip()):
        if ch not in " .0123456789":
            return line.rstrip()[c:]
    return ""


def _obstype(line: str) -> str:
    """The text after 'BOGUS' on a block's header row (empty elsewhere)."""
    c = line.find("BOGUS")
    return line[c + 5:].strip() if c >= 0 else ""


class ObsErrTable:
    """The OBSPROC error tables: ``tables[var][group]`` is a list over ``LEVELS`` (wind: ``WIND_LEVELS``)."""

    def __init__(self, tables: dict[str, dict[str, list[float]]]):
        self.tables = tables

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ObsErrTable":
        """NCEP defaults first (module_err_ncep), then the file, block by block (READ_OBSERR_*)."""
        k = _NCEP["k"]
        init = {
            "height": lambda p: intplin(p, k, _NCEP["h"]), "p": lambda p: intplin(p, k, _NCEP["p"]),
            "t": lambda p: intplog(p, k, _NCEP["t"]), "rh": lambda p: intplog(p, k, _NCEP["rh"]),
        }
        tables: dict[str, dict[str, list[float]]] = {}
        for var, f in init.items():
            tables[var] = {g: [f(p) for p in LEVELS] for g in GROUPS}
        for var in ("uv", "spd", "dir"):  # speed and direction start as the u error, too
            tables[var] = {g: [intplin(p, k, _NCEP["uv"]) for p in WIND_LEVELS] for g in GROUPS}

        lines = Path(path or DATA_FILE).read_text().split("\n")
        for var, (keyword, nvals) in _BLOCKS.items():
            # READ_OBSERR_*: every call scans the file from the top in entries of 3 rows (4 from the
            # UV header to the TEMP header, where the winds are); an entry is a row with the sensor
            # name and the rows that continue it.  Reading starts at the block's keyword row and
            # stops at the next block's keyword row (or at the "*." comment lines).
            i, wind, found = 0, False, False
            while i < len(lines):
                head, kind = lines[i], _obstype(lines[i])
                if kind in ("UV SENSOR ERRORS", "SPD SENSOR ERRORS", "DIR SENSOR ERRORS"):
                    wind = True
                elif kind == "TEMP SENSOR ERRORS":
                    wind = False
                rows = lines[i:i + (4 if wind else 3)]
                i += 4 if wind else 3
                if kind == keyword:
                    found = True
                elif kind == _END[var] or head[:2] == "*.":
                    break
                if not found or len(rows) < (4 if wind else 3):
                    continue
                vals: list[float] = []
                for r in rows:
                    vals += _fortran_floats(r, nvals)
                if wind:  # file order is 10 hPa first; 28 values = 26 levels reversed + 2 placeholders
                    vals = vals[:26][::-1]
                for g in _FEEDS[var].get(_sensor(head), ("other",)):
                    tables[var][g] = list(vals)
        return cls(tables)

    def errors(self, fm: int, pressure_pa: float | None, height_m: float | None = None,
               wind_sd: bool = False) -> dict[str, float]:
        """Errors OBSPROC assigns to one level of an FM-``fm`` report.

        Keys: ``p`` (Pa), ``u``, ``v``, ``speed``, ``direction`` (m/s; direction slot is an
        error in the units OBSPROC stores, see module doc), ``height`` (m), ``t`` and ``td`` (K),
        ``rh`` (%).  ``pressure_pa`` None falls back to ``floor(ref_pres(height_m))``.
        AIRSRET has only p, height, t, rh.
        """
        if fm in _NOT_PORTED:
            raise NotImplementedError(f"FM {fm}: error is not a plain table lookup, not ported")
        if pressure_pa is None:
            if height_m is None:
                raise ValueError("pressure or height is required")
            pressure_pa = float(math.floor(ref_pres(height_m)))
        pres = pressure_pa
        platform = _PLATFORM.get(fm)
        if platform is None:  # DEFAULT case: NCEP profile
            k = _NCEP["k"]
            u = intplin(pres, k, _NCEP["uv"])
            t = intplog(pres, k, _NCEP["t"])
            return {"p": intplin(pres, k, _NCEP["p"]), "u": u, "v": u, "speed": u, "direction": 35.0,
                    "height": intplin(pres, k, _NCEP["h"]), "t": t, "td": t,
                    "rh": intplog(pres, k, _NCEP["rh"])}
        g = _TABLE[platform]
        tb = self.tables
        t = intplog(pres, LEVELS, tb["t"][g])
        out = {"p": intplin(pres, LEVELS, tb["p"][g]),
               "height": (intplin if platform == "SYNOP" else intplog)(pres, LEVELS, tb["height"][g]),
               "t": t, "td": t, "rh": intplog(pres, LEVELS, tb["rh"][g])}
        if platform == "AIRSRET":
            del out["td"]
            return out
        u = intplin(pres, WIND_LEVELS, tb["uv"][g])
        out["u"] = out["v"] = u
        # SATEM always reads the SPD/DIR tables; the others only when wind_sd is on.
        separate = wind_sd or platform == "SATEM"
        out["speed"] = intplin(pres, WIND_LEVELS, tb["spd"][g]) if separate else u
        out["direction"] = intplin(pres, WIND_LEVELS, tb["dir"][g]) if separate else u
        if platform == "SATEM":  # the obs_gts writer puts the thickness in the temperature column
            out["thickness"] = out["height"]
        return out

