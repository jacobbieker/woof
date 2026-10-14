"""Regenerate ``woof/data/energy/turbine_library.json``.

Usage (from the repository root)::

    python -m tools.build_turbine_library --upstream-dir DIR   # offline
    python -m tools.build_turbine_library --fetch              # download
    python -m tools.build_turbine_library --upstream-dir DIR --check

``DIR`` holds ``turbine_data.csv`` and ``power_curves.csv`` from
windpowerlib's snapshot of the Open Energy Platform wind turbine library
(``windpowerlib/oedb/`` at commit :data:`UPSTREAM_COMMIT`).  ``--fetch``
downloads exactly those two files at that commit; nothing else touches the
network, and ``--check`` exits 1 when the committed file differs from what
the inputs produce.

The upstream data is the OEP "Wind Turbine Library" (ODbL-1.0,
(c) Reiner Lemoine Institut); the generated file is a derived database
under the same licence.  :data:`SUPPLEMENT` adds the rated power and rotor
diameter of common models the snapshot lacks, as published by their
manufacturers; those entries carry parametric curves
(:func:`woof.energy.turbine_library.parametric_fraction`) and say so.

Pinned corrections applied to the upstream rows (each is written into the
entry's ``notes``):

* the rating is taken from the ``turbine_type`` suffix (``GE103/2850`` ->
  2850 kW) when ``nominal_power`` disagrees with it;
* a curve point above the rating is clipped to the rating;
* zeros after the last positive point (the cut-out drop) are removed,
  because cut-out is applied with hysteresis at run time;
* a curve that starts at its cut-in with positive power is kept as is:
  below an entry's first point the run-time curve gives zero;
* a display name that parses to a different rating than the entry's own
  (``S140/3600`` is named "3.4M140 EBC") is not kept as an alias.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

import numpy as np

from woof.energy.turbine_library import (
    DEFAULT_TURBINE_LIBRARY, PARAMETRIC_CP, PARAMETRIC_CUT_IN_MS,
    PARAMETRIC_CUT_OUT_MS, PARAMETRIC_KNEE, TURBINE_LIBRARY_SCHEMA,
    load_turbine_library, parametric_fraction, parametric_speeds,
    parse_iec_class, parse_model, specific_power)

UPSTREAM_COMMIT = "393c206b325ca7e706473b8fcefe748ac1de4134"
UPSTREAM_BASE = ("https://raw.githubusercontent.com/wind-python/windpowerlib/"
                 f"{UPSTREAM_COMMIT}/windpowerlib/oedb/")
UPSTREAM_FILES = ("turbine_data.csv", "power_curves.csv")
OEP_TABLE = ("https://openenergyplatform.org/dataedit/view/supply/"
             "wind_turbine_library")

LICENSE = "ODbL-1.0"
ATTRIBUTION = (
    "Contains data from the Open Energy Platform Wind Turbine Library "
    "(c) Reiner Lemoine Institut, ODbL-1.0, via the windpowerlib oedb "
    "snapshot; supplementary ratings and rotor diameters from manufacturer "
    "publications. This derived database is made available under ODbL-1.0.")

#: Upstream rows of machines built for (or mainly sold) offshore.
OFFSHORE_IDS = frozenset({
    "AD116/5000", "AD132/5000", "AD135/5000", "AD135/5050", "AD180/8000",
    "GE150/6000", "S126/6150", "S152/6150", "S152/6330", "SCD168/8000",
    "SWT107/3600", "SWT120/3600", "SWT120/6000", "SWT154/6000",
    "SWT154/7000", "V164/8000", "V164/9500",
})

#: Further names for upstream rows.
EXTRA_ALIASES = {
    "SWT62/1300": ["Bonus 1.3 MW", "B62/1300"],
    "S126/6150": ["REpower 6M", "6M126", "6.2M126"],
    "GE150/6000": ["Haliade 150", "Haliade 150-6MW"],
    "V164/8000": ["V164-8.0 MW", "MHI Vestas V164-8.0"],
    "V164/9500": ["V164-9.5 MW", "V164-9.5"],
    "AD116/5000": ["Multibrid M5000", "M5000"],
}

_VESTAS = "https://www.vestas.com/en/energy-solutions/onshore-wind-turbines"
_VESTAS_OFF = "https://www.vestas.com/en/energy-solutions/offshore-wind-turbines"
_SGRE = "https://www.siemensgamesa.com/global/en/home/products-and-services.html"
_GE = "https://www.gevernova.com/wind-power"
_ENERCON = "https://www.enercon.de/en/turbines"
_NORDEX = "https://www.nordex-online.com/en/product/"
_GOLDWIND = "https://www.goldwind.com/en/"
_SUZLON = "https://www.suzlon.com/"
_MHI = "https://www.mhi.com/"
_CLIPPER = "https://www.clipperwind.com/"

#: (id, manufacturer, model, rated kW, rotor m, IEC class, offshore,
#:  aliases, source) for common models without an upstream curve.
SUPPLEMENT = [
    ("GE70/1500", "GE Wind", "GE 1.5s", 1500, 70.5, None, False,
     ["GE1.5s", "1.5s"], _GE),
    ("GE77/1500", "GE Wind", "GE 1.5sle", 1500, 77, "II", False,
     ["GE1.5sle", "1.5sle", "GE1.5-77", "GE 1.5-77", "1.5-77"], _GE),
    ("GE82/1500", "GE Wind", "GE 1.5xle", 1500, 82.5, "III", False,
     ["GE1.5xle", "1.5xle", "GE1.5-82.5", "GE 1.5-82.5", "1.5-82.5"], _GE),
    ("GE82/1600", "GE Wind", "GE 1.6-82.5", 1600, 82.5, "II", False,
     ["GE1.6-82.5", "1.6-82.5"], _GE),
    ("GE87/1850", "GE Wind", "GE 1.85-87", 1850, 87, None, False,
     ["GE1.85-87", "1.85-87"], _GE),
    ("GE100/1600", "GE Wind", "GE 1.6-100", 1600, 100, "III", False,
     ["GE1.6-100", "1.6-100"], _GE),
    ("GE100/1700", "GE Wind", "GE 1.7-100", 1700, 100, "III", False,
     ["GE1.7-100", "1.7-100"], _GE),
    ("GE100/1790", "GE Wind", "GE 1.79-100", 1790, 100, None, False,
     ["GE1.79-100", "1.79-100"], _GE),
    ("GE116/2300", "GE Wind", "GE 2.3-116", 2300, 116, None, False,
     ["GE2.3-116", "2.3-116"], _GE),
    ("GE127/2520", "GE Wind", "GE 2.52-127", 2520, 127, None, False,
     ["GE2.52-127", "2.52-127"], _GE),
    ("GE127/2820", "GE Wind", "GE 2.82-127", 2820, 127, None, False,
     ["GE2.82-127", "2.82-127"], _GE),
    ("GE220/12000", "GE Wind", "Haliade-X 12 MW", 12000, 220, "I", True,
     ["Haliade-X 12", "Haliade-X 12MW"], _GE),
    ("GE220/13000", "GE Wind", "Haliade-X 13 MW", 13000, 220, "I", True,
     ["Haliade-X 13", "Haliade-X 13MW"], _GE),
    ("GE220/14000", "GE Wind", "Haliade-X 14 MW", 14000, 220, "I", True,
     ["Haliade-X 14", "Haliade-X 14MW"], _GE),
    ("V47/660", "Vestas", "V47-660 kW", 660, 47, None, False,
     ["V47/660", "V47"], _VESTAS),
    ("V52/850", "Vestas", "V52-850 kW", 850, 52, None, False,
     ["V52/850", "V52"], _VESTAS),
    ("V66/1650", "Vestas", "V66-1.65 MW", 1650, 66, None, False,
     ["V66/1650"], _VESTAS),
    ("V82/1650", "Vestas", "V82-1.65 MW", 1650, 82, None, False,
     ["V82/1650", "V82"], _VESTAS),
    ("V90/1800", "Vestas", "V90-1.8 MW", 1800, 90, None, False,
     ["V90/1800"], _VESTAS),
    ("V150/6000", "Vestas", "V150-6.0 MW", 6000, 150, None, False,
     ["V150/6000"], _VESTAS),
    ("V162/6200", "Vestas", "V162-6.2 MW", 6200, 162, None, False,
     ["V162/6200"], _VESTAS),
    ("V164/10000", "Vestas", "V164-10.0 MW", 10000, 164, "I", True,
     ["V164/10000", "V164-10"], _VESTAS_OFF),
    ("V174/9500", "Vestas", "V174-9.5 MW", 9500, 174, None, True,
     ["V174/9500"], _VESTAS_OFF),
    ("V236/15000", "Vestas", "V236-15.0 MW", 15000, 236, None, True,
     ["V236/15000", "V236-15"], _VESTAS_OFF),
    ("SG114/2100", "Siemens Gamesa", "SG 2.1-114", 2100, 114, None, False,
     ["SG2.1-114"], _SGRE),
    ("SG114/2625", "Siemens Gamesa", "SG 2.6-114", 2625, 114, None, False,
     ["SG2.6-114"], _SGRE),
    ("SG132/3465", "Siemens Gamesa", "SG 3.4-132", 3465, 132, None, False,
     ["SG3.4-132"], _SGRE),
    ("SG145/4500", "Siemens Gamesa", "SG 4.5-145", 4500, 145, None, False,
     ["SG4.5-145"], _SGRE),
    ("SG170/6600", "Siemens Gamesa", "SG 6.6-170", 6600, 170, None, False,
     ["SG6.6-170"], _SGRE),
    ("SG167/8000", "Siemens Gamesa", "SG 8.0-167 DD", 8000, 167, "I", True,
     ["SG8.0-167", "SG 8.0-167"], _SGRE),
    ("SG200/11000", "Siemens Gamesa", "SG 11.0-200 DD", 11000, 200, "I",
     True, ["SG11.0-200", "SG 11.0-200"], _SGRE),
    ("SG222/14000", "Siemens Gamesa", "SG 14-222 DD", 14000, 222, "I", True,
     ["SG14-222", "SG 14-222"], _SGRE),
    ("SWT154/8000", "Siemens", "SWT-8.0-154", 8000, 154, "I", True,
     ["SWT8.0-154"], _SGRE),
    ("G52/850", "Gamesa", "G52-850", 850, 52, None, False, ["G52"], _SGRE),
    ("G58/850", "Gamesa", "G58-850", 850, 58, None, False, ["G58"], _SGRE),
    ("G80/2000", "Gamesa", "G80-2.0 MW", 2000, 80, "I", False,
     ["G80", "G80/2000"], _SGRE),
    ("G87/2000", "Gamesa", "G87-2.0 MW", 2000, 87, "II", False,
     ["G87", "G87/2000"], _SGRE),
    ("G90/2000", "Gamesa", "G90-2.0 MW", 2000, 90, "II", False,
     ["G90", "G90/2000"], _SGRE),
    ("G97/2000", "Gamesa", "G97-2.0 MW", 2000, 97, "III", False,
     ["G97", "G97/2000"], _SGRE),
    ("G114/2000", "Gamesa", "G114-2.0 MW", 2000, 114, "III", False,
     ["G114/2000"], _SGRE),
    ("MWT62/1000", "Mitsubishi", "MWT-1000A", 1000, 61.4, None, False,
     ["MWT1000A", "MWT-62/1.0"], _MHI),
    ("MWT95/2400", "Mitsubishi", "MWT95/2.4", 2400, 95, None, False,
     ["MWT-95/2.4", "MWT95"], _MHI),
    ("MWT102/2400", "Mitsubishi", "MWT102/2.4", 2400, 102, None, False,
     ["MWT-102/2.4", "MWT102"], _MHI),
    ("SZ82/1500", "Suzlon", "S82-1.5 MW", 1500, 82, None, False,
     ["S82", "S82/1500"], _SUZLON),
    ("SZ88/2100", "Suzlon", "S88-2.1 MW", 2100, 88, None, False,
     ["S88", "S88/2100"], _SUZLON),
    ("SZ97/2100", "Suzlon", "S97-2.1 MW", 2100, 97, None, False,
     ["S97", "S97/2100"], _SUZLON),
    ("SZ111/2100", "Suzlon", "S111-2.1 MW", 2100, 111, None, False,
     ["S111", "S111/2100"], _SUZLON),
    ("N60/1300", "Nordex", "N60/1300", 1300, 60, None, False, ["N60"],
     _NORDEX),
    ("N149/4500", "Nordex", "N149/4.5", 4500, 149, None, False,
     ["N149/4500"], _NORDEX),
    ("C96/2500", "Clipper", "Liberty C96", 2500, 96, None, False,
     ["C96", "C96/2500"], _CLIPPER),
    ("C99/2500", "Clipper", "Liberty C99", 2500, 99, None, False,
     ["C99", "C99/2500"], _CLIPPER),
    ("GW82/1500", "Goldwind", "GW82/1500", 1500, 82, None, False, [],
     _GOLDWIND),
    ("GW87/1500", "Goldwind", "GW87/1500", 1500, 87, None, False, [],
     _GOLDWIND),
    ("GW121/2500", "Goldwind", "GW121/2500", 2500, 121, None, False, [],
     _GOLDWIND),
    ("GW140/3000", "Goldwind", "GW140/3000", 3000, 140, None, False, [],
     _GOLDWIND),
    ("E-40/500", "Enercon", "E-40/500", 500, 40, None, False,
     ["E-40", "E40", "E-40/5.40"], _ENERCON),
    ("E-44/900", "Enercon", "E-44", 900, 44, None, False, ["E44"],
     _ENERCON),
    ("E-66/1800", "Enercon", "E-66/1800", 1800, 70, None, False,
     ["E-66", "E66", "E-66/18.70"], _ENERCON),
    ("E-103/2350", "Enercon", "E-103 EP2", 2350, 103, None, False,
     ["E-103", "E103"], _ENERCON),
    ("E-138/4200", "Enercon", "E-138 EP3", 4200, 138, None, False,
     ["E-138", "E138"], _ENERCON),
    ("RE126/5000", "Senvion/REpower", "REpower 5M", 5000, 126, "I", True,
     ["5M", "5M126", "Senvion 5M"], "https://www.senvion.com/"),
]

#: Generic class curves: specific power (W/m2) of a typical machine.
GENERIC = {
    "iec_I": {"specific_power_w_m2": 400.0, "iec_class": "I",
              "offshore": False,
              "description": "high-wind IEC class I onshore machine"},
    "iec_II": {"specific_power_w_m2": 320.0, "iec_class": "II",
               "offshore": False,
               "description": "medium-wind IEC class II onshore machine"},
    "iec_III": {"specific_power_w_m2": 250.0, "iec_class": "III",
                "offshore": False,
                "description": "low-wind IEC class III onshore machine"},
    "offshore": {"specific_power_w_m2": 350.0, "iec_class": "I",
                 "offshore": True,
                 "description": "large offshore machine"},
}
GENERIC_DEFAULT = {"onshore": "iec_II", "offshore": "offshore"}

MANUFACTURER_ALIASES = {
    "vestas": "vestas", "vestas wind systems": "vestas",
    "mhi vestas": "vestas", "mhi vestas offshore wind": "vestas",
    "neg micon": "vestas", "micon": "vestas",
    "siemens": "siemens-gamesa", "siemens wind power": "siemens-gamesa",
    "siemens gamesa": "siemens-gamesa",
    "siemens gamesa renewable energy": "siemens-gamesa",
    "sgre": "siemens-gamesa", "gamesa": "siemens-gamesa",
    "gamesa eolica": "siemens-gamesa", "bonus": "siemens-gamesa",
    "bonus energy": "siemens-gamesa",
    "enercon": "enercon",
    "nordex": "nordex", "nordex acciona": "nordex", "acciona": "nordex",
    "acciona windpower": "nordex", "acciona energia": "nordex",
    "ge": "ge", "ge wind": "ge", "ge wind energy": "ge",
    "general electric": "ge", "ge energy": "ge",
    "ge renewable energy": "ge", "ge vernova": "ge", "alstom": "ge",
    "alstom wind": "ge", "alstom power": "ge",
    "senvion": "senvion", "repower": "senvion", "repower systems": "senvion",
    "senvion repower": "senvion",
    "adwen": "adwen", "areva": "adwen", "areva wind": "adwen",
    "adwen areva": "adwen", "multibrid": "adwen",
    "eno": "eno", "eno energy": "eno", "vensys": "vensys",
    "aerodyn": "aerodyn",
    "goldwind": "goldwind", "xinjiang goldwind": "goldwind",
    "mitsubishi": "mitsubishi", "mitsubishi heavy industries": "mitsubishi",
    "mhi": "mitsubishi", "mitsubishi power": "mitsubishi",
    "suzlon": "suzlon", "suzlon energy": "suzlon",
    "clipper": "clipper", "clipper windpower": "clipper",
    "envision": "envision", "mingyang": "mingyang", "ming yang": "mingyang",
    "sany": "sany", "windey": "windey", "dongfang": "dongfang",
    "shanghai electric": "shanghai-electric", "inox": "inox",
    "inox wind": "inox",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(directory: Path) -> None:
    """Download the two pinned upstream files into ``directory``."""

    directory.mkdir(parents=True, exist_ok=True)
    for name in UPSTREAM_FILES:
        with urllib.request.urlopen(UPSTREAM_BASE + name, timeout=60) as r:
            (directory / name).write_bytes(r.read())


def _heights(text: str) -> list[float]:
    out = []
    for part in (text or "").split(";"):
        part = part.strip().replace(",", ".")
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            continue
        if 10.0 <= value <= 250.0:
            out.append(value)
    return sorted(set(out))


def _curve_stats(speeds: np.ndarray, power: np.ndarray, rated: float
                 ) -> dict:
    positive = np.flatnonzero(power > 0)
    rated_at = np.flatnonzero(power >= 0.99 * rated)
    return {
        "cut_in_ms": float(speeds[positive[0]]),
        "rated_ms": float(speeds[rated_at[0]]) if rated_at.size else None,
        "cut_out_ms": float(speeds[positive[-1]]),
    }


def _names_ok(name: str, rated_kw: float) -> bool:
    parsed = parse_model(name)
    return parsed.rated_kw is None or abs(parsed.rated_kw / rated_kw - 1) \
        <= 0.03


def _upstream(directory: Path) -> tuple[list[dict], dict]:
    with open(directory / "turbine_data.csv", newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(directory / "power_curves.csv", newline="",
              encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        grid = [float(v) for v in header[1:]]
        curves = {}
        for row in reader:
            points = [(s, float(p) / 1000.0) for s, p in zip(grid, row[1:])
                      if p.strip()]
            curves[row[0]] = points
    entries = []
    skipped = []
    for row in rows:
        turbine_type = (row.get("turbine_type") or "").strip()
        notes = []
        try:
            rated = float(row["nominal_power"]) / 1000.0
            rotor = float(row["rotor_diameter"])
        except (KeyError, TypeError, ValueError):
            skipped.append(turbine_type or "?")
            continue
        if not turbine_type or rated <= 0 or rotor <= 0:
            skipped.append(turbine_type or "?")
            continue
        parts = turbine_type.split("/")
        suffix = parts[1] if len(parts) > 1 else ""
        if suffix.isdigit() and abs(float(suffix) - rated) > 1.0:
            notes.append(f"nominal_power {rated:g} kW disagrees with the "
                         f"type suffix; rating {suffix} kW used")
            rated = float(suffix)
        names = [turbine_type]
        display = (row.get("name") or "").strip()
        if display:
            if _names_ok(display, rated):
                names.append(display)
            else:
                notes.append(f"display name {display!r} names another "
                             "rating; not kept as an alias")
        names += EXTRA_ALIASES.get(turbine_type, [])
        entry = {
            "id": turbine_type,
            "manufacturer": row["manufacturer"].strip() or None,
            "model": display if display and display in names
            else turbine_type,
            "names": sorted(set(names), key=names.index),
            "rated_kw": rated,
            "rotor_diameter_m": rotor,
            "hub_heights_m": _heights(row.get("hub_height", "")),
            "iec_class": parse_iec_class(row.get("wind_class_iec")),
            "offshore": turbine_type in OFFSHORE_IDS,
        }
        points = curves.get(turbine_type)
        if points and not any(p > 0 for _, p in points):
            notes.append("upstream curve has no positive point; ignored")
            points = None
        if points:
            speeds = np.array([s for s, _ in points])
            power = np.array([p for _, p in points])
            if power.max() > rated * 1.0001:
                notes.append(f"curve peaks at {power.max():g} kW above the "
                             f"{rated:g} kW rating; clipped")
                power = np.minimum(power, rated)
            last = int(np.flatnonzero(power > 0)[-1])
            if last < power.size - 1:
                speeds, power = speeds[:last + 1], power[:last + 1]
            entry.update(curve_source="oedb", **_curve_stats(speeds, power,
                                                             rated))
            entry["wind_speed_ms"] = [round(float(s), 3) for s in speeds]
            entry["power_kw"] = [round(float(p), 3) for p in power]
        else:
            entry.update(_parametric(rated, rotor))
            notes.append("no upstream curve; parametric curve at the "
                         "model's specific power")
        source = (row.get("source") or "").split(";")[0].strip()
        entry["source_url"] = source or OEP_TABLE
        entry["notes"] = notes
        entries.append(entry)
    if skipped:
        print(f"build_turbine_library: skipped malformed upstream rows "
              f"{skipped}", file=sys.stderr)
    return entries, {"rows": len(rows), "curves": len(curves),
                     "skipped": skipped}


def _parametric(rated: float, rotor: float) -> dict:
    speeds = parametric_speeds(PARAMETRIC_CUT_OUT_MS)
    sp = specific_power(rated / 1000.0, rotor)
    power = parametric_fraction(speeds, sp) * rated
    stats = _curve_stats(speeds, power, rated)
    return {"curve_source": "parametric", **stats,
            "wind_speed_ms": [round(float(s), 3) for s in speeds],
            "power_kw": [round(float(p), 3) for p in power]}


def _supplement() -> list[dict]:
    out = []
    for (ident, manufacturer, model, rated, rotor, iec, offshore, aliases,
         source) in SUPPLEMENT:
        entry = {"id": ident, "manufacturer": manufacturer, "model": model,
                 "names": [model] + list(aliases), "rated_kw": float(rated),
                 "rotor_diameter_m": float(rotor), "hub_heights_m": [],
                 "iec_class": iec, "offshore": offshore}
        entry.update(_parametric(float(rated), float(rotor)))
        entry["source_url"] = source
        entry["notes"] = ["manufacturer-published rating and rotor "
                          "diameter; parametric curve at the model's "
                          "specific power"]
        out.append(entry)
    return out


def build(directory: Path) -> dict:
    upstream, counts = _upstream(directory)
    ids = {entry["id"] for entry in upstream}
    supplement = [entry for entry in _supplement() if entry["id"] not in ids]
    turbines = sorted(upstream + supplement, key=lambda e: e["id"])
    measured = sum(entry["curve_source"] == "oedb" for entry in turbines)
    return {
        "schema": TURBINE_LIBRARY_SCHEMA,
        "description": (
            "Wind-turbine power curves (kW against hub-height wind speed at "
            "the IEC 61400-12-1 reference air density 1.225 kg/m3) for "
            "woof energy power. Speeds between points are linearly "
            "interpolated. curve_source 'oedb' is an upstream manufacturer "
            "curve; 'parametric' is P/P_r = s/(1+s^n)^(1/n), s = 0.5 rho0 "
            f"cp v^3 / SP, cp {PARAMETRIC_CP}, n {PARAMETRIC_KNEE:g}, zero "
            f"below {PARAMETRIC_CUT_IN_MS:g} m/s, at the model's specific "
            "power. Cut-out is applied with hysteresis at run time."),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "sources": [{
            "source": "Open Energy Platform Wind Turbine Library "
                      "(windpowerlib oedb snapshot)",
            "url": UPSTREAM_BASE, "table": OEP_TABLE,
            "commit": UPSTREAM_COMMIT, "license": LICENSE,
            "attribution": "(c) Reiner Lemoine Institut",
            "files": {name: _sha256(directory / name)
                      for name in UPSTREAM_FILES},
            "rows": counts["rows"], "curves": counts["curves"],
        }, {
            "source": "manufacturer publications (rated power, rotor "
                      "diameter)",
            "license": "facts; see each entry's source_url",
            "entries": len(supplement),
        }],
        "reference_density_kg_m3": 1.225,
        "counts": {"turbines": len(turbines), "measured_curves": measured,
                   "parametric_curves": len(turbines) - measured},
        "manufacturer_aliases": dict(sorted(MANUFACTURER_ALIASES.items())),
        "generic": GENERIC,
        "generic_default": GENERIC_DEFAULT,
        "turbines": turbines,
    }


def render(document: dict) -> str:
    """JSON with one line per turbine entry, so diffs stay readable."""

    head = {k: v for k, v in document.items() if k != "turbines"}
    text = json.dumps(head, indent=1, ensure_ascii=False)
    lines = [json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
             for entry in document["turbines"]]
    body = ',\n  '.join(lines)
    return text[:-2] + ',\n "turbines": [\n  ' + body + '\n ]\n}\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--upstream-dir", type=Path,
                       help="directory with turbine_data.csv and "
                            "power_curves.csv")
    where.add_argument("--fetch", action="store_true",
                       help=f"download them at commit {UPSTREAM_COMMIT[:12]}")
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_TURBINE_LIBRARY)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if --output differs from the build")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        directory = args.upstream_dir
        if args.fetch:
            directory = Path(scratch)
            fetch(directory)
        missing = [n for n in UPSTREAM_FILES
                   if not (directory / n).is_file()]
        if missing:
            print(f"build_turbine_library: {directory} lacks {missing}",
                  file=sys.stderr)
            return 2
        text = render(build(directory))
    if args.check:
        current = args.output.read_text(encoding="utf-8") \
            if args.output.is_file() else ""
        if current != text:
            print(f"{args.output} is out of date", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date", file=sys.stderr)
        return 0
    staging = args.output.with_name(args.output.name + ".partial")
    staging.write_text(text, encoding="utf-8")
    try:
        load_turbine_library(staging)       # the run-time checks, first
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    staging.replace(args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
