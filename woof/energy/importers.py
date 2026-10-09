"""Grid and renewable asset importers for ``woof energy import``.

``woof energy import`` reads one of four input formats and writes a
``woof-energy.assets.v1`` document (:mod:`woof.energy.contracts`),
optionally merged into an existing one.

``pypsa-eur``
    A PyPSA-Eur electricity network as CSV tables: ``buses.csv``,
    ``lines.csv`` and ``links.csv``, optionally ``transformers.csv``,
    ``converters.csv`` and ``generators.csv``, given as a directory or as
    the files themselves.  Three published layouts are recognised from their
    columns:

    ``osm-prebuilt``
        The OpenStreetMap-based prebuilt network on Zenodo
        (doi:10.5281/zenodo.13358976 and later versions; Xiong et al.).
        Buses: ``bus_id, voltage, dc, symbol, under_construction, [tags,]
        x, y, country, geometry``.  Lines: ``line_id, bus0, bus1, voltage,
        [i_nom,] circuits, [s_nom, r, x, b,] length, underground,
        under_construction, [type, tags,] geometry``.  Links: ``link_id,
        bus0, bus1, voltage, p_nom, length, underground,
        under_construction, [tags,] geometry``.  ODbL-1.0 (OSM-derived).
    ``gridkit``
        The older GridKit extract of the ENTSO-E transmission map shipped
        as ``data/entsoegridkit`` in PyPSA-Eur.  Buses: ``bus_id,
        station_id, voltage, dc, symbol, under_construction, tags, x, y``;
        lines as above without the electrical columns; links without
        ``voltage``/``p_nom``; ``generators.csv`` with ``generator_id,
        bus_id, technology, capacity, tags, geometry``.  ``tags`` is an
        hstore string.  CC-BY-4.0.
    ``pypsa-network``
        A network written by ``pypsa.Network.export_to_csv_folder`` (for
        example PyPSA-Eur's ``base.nc``): index column ``Bus``/``Line``/
        ``Link``/``Generator`` (or ``name``), ``v_nom`` instead of
        ``voltage``, ``num_parallel`` instead of ``circuits``, ``carrier``
        on links and generators.  CC-BY-4.0.

    Both the single-quoted (GridKit, prebuilt) and double-quoted (pandas)
    CSV dialects are read; ``t``/``f`` and ``True``/``False`` booleans both
    count.  Geometry is WKT (``POINT``, ``LINESTRING``,
    ``MULTILINESTRING``), parsed by :func:`parse_wkt` without shapely.  A
    line or link with no geometry becomes the straight segment between its
    buses, tagged ``woof:geometry=bus-to-bus``; a generator with no
    geometry is put at its bus, tagged ``woof:geometry=bus-location``.

    Buses become ``substation`` points.  Buses at one location, and buses
    joined by a transformer or converter less than 1 km long, are one
    substation carrying every voltage of its members (the prebuilt network
    has one bus per voltage level, and its v0.1-v0.4 releases offset those
    buses by ~140 m).  GridKit ``joint`` buses and buses whose symbol names
    a power plant are not substations; they still anchor lines.  Lines and
    links become ``line`` or ``cable`` by their ``underground`` flag; links
    carry ``carrier=DC``.  Generators become ``plant`` points.

``repd``
    The UK DESNZ Renewable Energy Planning Database quarterly extract CSV
    (cp1252 encoded; ``Ref ID``, ``Site Name``, ``Technology Type``,
    ``Installed Capacity (MWelec)``, ``Development Status (short)``,
    ``X-coordinate``/``Y-coordinate`` on the British National Grid, ...).
    Each row becomes a ``plant`` point.  Easting/northing goes to WGS84 by
    the inverse Transverse Mercator on Airy 1830 and the OS Helmert
    transformation (:func:`bng_to_wgs84`), which OS quotes as good to about
    3.5 m (95 %).  ``Height of Turbines (m)`` is kept as a tag and never
    used as a hub height: planning databases record height to blade tip.
    Some Northern Ireland rows are on the Irish Grid rather than the
    National Grid; a row whose converted position falls outside its
    country's box is refused, not guessed.  OGL-UK-3.0.

``geojson``
    Features with arbitrary properties.  The asset kind is the ``kind`` or
    OSM ``power`` property, else ``--kind``.  A file that already is a
    ``woof-energy.assets.v1`` document is read as one.

``csv``
    Points from ``--lat-col``/``--lon-col`` with ``--kind``; ``--id-col``
    names the identifier column (default: the 1-based data row number).

For ``geojson`` and ``csv`` the source is named ``<format>:<file stem>`` so
two files never share a ``(source, source_ref)`` pair.  Recognised property
names are the :class:`~woof.energy.contracts.Asset` field names (except
``asset_id``, ``source`` and ``source_ref``, which the importer assigns, so
such properties are kept as tags) plus the OSM keys ``voltage`` (volts,
``;``-separated; a value under 1000 is ambiguous and kept as a tag),
``frequency``, ``generator:source``/``plant:source``,
``plant:output:electricity``/``generator:output:electricity``
(``"50 MW"``), ``height:hub`` and ``rotor:diameter``; every other property
becomes a string tag.

Rows the importer cannot turn into a valid asset are refused one by one,
counted by reason and listed (first few) in the provenance record and the
summary; an import that yields no asset at all is refused as a whole.  The
per-row work here is dictionary and string handling over at most a few
hundred thousand rows; the only array math is the vectorised BNG transform
over the REPD rows and haversine checks inside :func:`merge_collections`,
small per-site vectors that the Python boundary allows in numpy.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woof.energy.contracts import (
    ASSET_KINDS,
    ASSETS_SCHEMA,
    GENERATOR_SOURCES,
    LINEAR_KINDS,
    Asset,
    AssetCollection,
    ContractError,
    dump_assets,
    load_assets,
    sha256_file,
)

FORMATS = ("pypsa-eur", "repd", "geojson", "csv")

PYPSA_SOURCE = "pypsa-eur"
REPD_SOURCE = "repd"
REPD_LICENSE = "OGL-UK-3.0"
REPD_ATTRIBUTION = (
    "Contains public sector information licensed under the Open Government "
    "Licence v3.0: Renewable Energy Planning Database, Department for "
    "Energy Security and Net Zero")

#: PyPSA-Eur layouts: licence and attribution recorded for each.
PYPSA_VARIANTS: dict[str, dict[str, str | None]] = {
    "osm-prebuilt": {
        "license": "ODbL-1.0",
        "attribution": (
            "PyPSA-Eur prebuilt electricity network based on OpenStreetMap "
            "data (Xiong et al., doi:10.5281/zenodo.13358976); "
            "(c) OpenStreetMap contributors, ODbL 1.0"),
    },
    "gridkit": {
        "license": "CC-BY-4.0",
        "attribution": (
            "PyPSA-Eur base network: GridKit extract of the ENTSO-E "
            "Transmission System Map (PyPSA-Eur authors), CC-BY-4.0"),
    },
    "pypsa-network": {
        "license": "CC-BY-4.0",
        "attribution": "PyPSA-Eur network export (PyPSA-Eur authors), "
                       "CC-BY-4.0",
    },
}

#: Duplicate tolerances used by :func:`merge_collections`.
MERGE_POINT_M = 50.0
MERGE_LINE_END_M = 100.0
MERGE_LINE_LENGTH_FRACTION = 0.05

#: Buses joined by a transformer/converter shorter than this are one
#: substation.
_SUBSTATION_JOIN_M = 1000.0
#: Mean Earth radius (IUGG) for tolerance checks; these are 50-100 m
#: duplicate tests, not geodesy.
_MEAN_EARTH_RADIUS_M = 6371008.8
_EXAMPLES = 10


class ImportRefused(ValueError):
    """``woof energy import`` cannot read these inputs as asked."""


class WKTError(ValueError):
    """A WKT string is malformed or of an unsupported geometry type."""


class _RowRefused(ValueError):
    """One input row cannot become an asset; the import goes on."""


def _refusal(error: Exception) -> str:
    """Row refusal text; the part before the first colon is the reason
    key the ledger counts by."""

    if isinstance(error, ContractError):
        return f"fails the assets contract: {error}"
    return str(error)


# --------------------------------------------------------------------------
# WKT


_WKT_HEAD = re.compile(
    r"^\s*(POINT|LINESTRING|MULTILINESTRING)\s*(ZM|Z|M)?\s*(.*)$",
    re.IGNORECASE | re.DOTALL)
_WKT_TOKEN = re.compile(r"\(|\)|,|[^\s(),]+")


def parse_wkt(text: str) -> dict[str, Any]:
    """WKT ``POINT``/``LINESTRING``/``MULTILINESTRING`` -> GeoJSON geometry.

    Coordinates are ``x y [z [m]]``; only ``x`` (longitude) and ``y``
    (latitude) are kept.  Surrounding quotes left by a CSV dialect are
    tolerated.  Anything else -- other geometry types, ``EMPTY``, a line
    with fewer than two positions, non-numeric or non-finite values,
    unbalanced parentheses, trailing text -- raises :class:`WKTError`.
    """

    if not isinstance(text, str):
        raise WKTError(f"WKT must be a string, got {type(text).__name__}")
    body = text.strip().strip("'\"").strip()
    if not body:
        raise WKTError("WKT is empty")
    match = _WKT_HEAD.match(body)
    if match is None:
        raise WKTError(f"unsupported WKT {body[:40]!r}: expected POINT, "
                       "LINESTRING or MULTILINESTRING")
    kind = match.group(1).upper()
    rest = match.group(3).strip()
    if rest.upper() == "EMPTY":
        raise WKTError(f"{kind} EMPTY has no coordinates")
    tokens = _WKT_TOKEN.findall(rest)
    position = 0

    def expect(token: str) -> None:
        nonlocal position
        if position >= len(tokens) or tokens[position] != token:
            found = tokens[position] if position < len(tokens) else "end"
            raise WKTError(f"{kind}: expected {token!r}, found {found!r}")
        position += 1

    def coordinate() -> list[float]:
        nonlocal position
        values = []
        while position < len(tokens) and tokens[position] not in "(),":
            try:
                number = float(tokens[position])
            except ValueError as error:
                raise WKTError(f"{kind}: {tokens[position]!r} is not a "
                               "number") from error
            if not math.isfinite(number):
                raise WKTError(f"{kind}: coordinate {tokens[position]!r} "
                               "is not finite")
            values.append(number)
            position += 1
        if not 2 <= len(values) <= 4:
            raise WKTError(f"{kind}: a position needs 2 to 4 numbers, got "
                           f"{len(values)}")
        return values[:2]

    def sequence() -> list[list[float]]:
        nonlocal position
        expect("(")
        points = [coordinate()]
        while position < len(tokens) and tokens[position] == ",":
            position += 1
            points.append(coordinate())
        expect(")")
        if len(points) < 2:
            raise WKTError(f"{kind}: a line needs at least two positions")
        return points

    if kind == "POINT":
        expect("(")
        geometry = {"type": "Point", "coordinates": coordinate()}
        expect(")")
    elif kind == "LINESTRING":
        geometry = {"type": "LineString", "coordinates": sequence()}
    else:
        expect("(")
        parts = [sequence()]
        while position < len(tokens) and tokens[position] == ",":
            position += 1
            parts.append(sequence())
        expect(")")
        geometry = {"type": "MultiLineString", "coordinates": parts}
    if position != len(tokens):
        raise WKTError(f"{kind}: unexpected trailing text "
                       f"{' '.join(tokens[position:])[:40]!r}")
    return geometry


# --------------------------------------------------------------------------
# British National Grid -> WGS84
#
# Ordnance Survey, "A guide to coordinate systems in Great Britain", v3.6
# (2020): Annexe A constants, Annexe C.2 inverse Transverse Mercator,
# Annexe B.1/B.2 geodetic <-> Cartesian, section 6.6 / Annexe D Helmert.

_AIRY_A = 6377563.396
_AIRY_B = 6356256.909
_GRS80_A = 6378137.0
_GRS80_B = 6356752.3141
_NG_F0 = 0.9996012717
_NG_LAT0 = math.radians(49.0)
_NG_LON0 = math.radians(-2.0)
_NG_E0 = 400000.0
_NG_N0 = -100000.0
#: WGS84/ETRS89 -> OSGB36 (tX, tY, tZ m; s ppm; rX, rY, rZ arc-seconds),
#: position-vector convention (OS guide table 4).
_HELMERT_WGS84_TO_OSGB36 = (-446.448, 125.157, -542.060, 20.4894,
                            -0.1502, -0.2470, -0.8421)


def _meridional_arc(phi: np.ndarray, *, a: float, b: float, f0: float,
                    phi0: float) -> np.ndarray:
    """OS guide equation C3."""

    n = (a - b) / (a + b)
    dphi = phi - phi0
    sphi = phi + phi0
    return b * f0 * (
        (1.0 + n + 1.25 * n**2 + 1.25 * n**3) * dphi
        - (3.0 * n + 3.0 * n**2 + 21.0 / 8.0 * n**3)
        * np.sin(dphi) * np.cos(sphi)
        + (15.0 / 8.0 * n**2 + 15.0 / 8.0 * n**3)
        * np.sin(2.0 * dphi) * np.cos(2.0 * sphi)
        - 35.0 / 24.0 * n**3 * np.sin(3.0 * dphi) * np.cos(3.0 * sphi))


def bng_to_osgb36(easting, northing) -> tuple[np.ndarray, np.ndarray]:
    """National Grid easting/northing (m) -> OSGB36 (lat, lon) degrees.

    OS guide Annexe C.2: iterate the latitude until the meridional arc
    matches the northing to 0.01 mm, then the VII-XIIA series.
    """

    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)
    a, b, f0 = _AIRY_A, _AIRY_B, _NG_F0
    e2 = (a * a - b * b) / (a * a)
    phi = (n - _NG_N0) / (a * f0) + _NG_LAT0
    arc = _meridional_arc(phi, a=a, b=b, f0=f0, phi0=_NG_LAT0)
    for _ in range(50):
        residual = n - _NG_N0 - arc
        if np.all(np.abs(residual) < 1e-5):
            break
        phi = phi + residual / (a * f0)
        arc = _meridional_arc(phi, a=a, b=b, f0=f0, phi0=_NG_LAT0)
    sin2 = np.sin(phi) ** 2
    nu = a * f0 / np.sqrt(1.0 - e2 * sin2)
    rho = a * f0 * (1.0 - e2) / (1.0 - e2 * sin2) ** 1.5
    eta2 = nu / rho - 1.0
    t = np.tan(phi)
    sec = 1.0 / np.cos(phi)
    vii = t / (2.0 * rho * nu)
    viii = t / (24.0 * rho * nu**3) * (5.0 + 3.0 * t**2 + eta2
                                       - 9.0 * t**2 * eta2)
    ix = t / (720.0 * rho * nu**5) * (61.0 + 90.0 * t**2 + 45.0 * t**4)
    x = sec / nu
    xi = sec / (6.0 * nu**3) * (nu / rho + 2.0 * t**2)
    xii = sec / (120.0 * nu**5) * (5.0 + 28.0 * t**2 + 24.0 * t**4)
    xiia = sec / (5040.0 * nu**7) * (61.0 + 662.0 * t**2 + 1320.0 * t**4
                                     + 720.0 * t**6)
    de = e - _NG_E0
    lat = phi - vii * de**2 + viii * de**4 - ix * de**6
    lon = _NG_LON0 + x * de - xi * de**3 + xii * de**5 - xiia * de**7
    return np.degrees(lat), np.degrees(lon)


def _geodetic_to_cartesian(lat_deg, lon_deg, height, *, a: float, b: float
                           ) -> np.ndarray:
    """OS guide B2-B5; returns shape (3, N)."""

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e2 = (a * a - b * b) / (a * a)
    nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.stack([(nu + height) * np.cos(lat) * np.cos(lon),
                     (nu + height) * np.cos(lat) * np.sin(lon),
                     ((1.0 - e2) * nu + height) * np.sin(lat)])


def _cartesian_to_geodetic(xyz: np.ndarray, *, a: float, b: float
                           ) -> tuple[np.ndarray, np.ndarray]:
    """OS guide B6-B8 (iterated to convergence); (lat, lon) degrees."""

    x, y, z = xyz
    e2 = (a * a - b * b) / (a * a)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(20):
        nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        new = np.arctan2(z + e2 * nu * np.sin(lat), p)
        if np.all(np.abs(new - lat) < 1e-13):
            lat = new
            break
        lat = new
    return np.degrees(lat), np.degrees(lon)


def _helmert_matrix() -> tuple[np.ndarray, np.ndarray]:
    tx, ty, tz, s_ppm, rx, ry, rz = _HELMERT_WGS84_TO_OSGB36
    arcsec = math.pi / (180.0 * 3600.0)
    rx, ry, rz = rx * arcsec, ry * arcsec, rz * arcsec
    s = 1.0 + s_ppm * 1e-6
    rotation = np.array([[s, -rz, ry], [rz, s, -rx], [-ry, rx, s]])
    return rotation, np.array([tx, ty, tz])


def osgb36_to_wgs84(lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """OSGB36 (lat, lon) -> WGS84 (lat, lon) degrees.

    Exact inverse of the OS WGS84->OSGB36 Helmert (section 6.6), at zero
    Airy ellipsoid height; the height's effect on position is millimetres.
    """

    lat = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
    osgb = _geodetic_to_cartesian(lat, lon, 0.0, a=_AIRY_A, b=_AIRY_B)
    rotation, translation = _helmert_matrix()
    wgs = np.linalg.solve(rotation, osgb - translation[:, None])
    return _cartesian_to_geodetic(wgs, a=_GRS80_A, b=_GRS80_B)


def bng_to_wgs84(easting, northing) -> tuple[np.ndarray, np.ndarray]:
    """National Grid easting/northing (m) -> WGS84 ``(lon, lat)`` degrees.

    Inverse Transverse Mercator on Airy 1830 then the OS Helmert; about
    3.5 m (95 %) against OSTN15 inside Great Britain, per the OS guide.
    """

    lat, lon = bng_to_osgb36(np.atleast_1d(easting), np.atleast_1d(northing))
    lat, lon = osgb36_to_wgs84(lat, lon)
    return lon, lat


# --------------------------------------------------------------------------
# small parsers


_NULLS = {"", "nan", "none", "null", "na", "n/a"}


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str)
                             and value.strip().lower() in _NULLS)


def _float(value: Any) -> float | None:
    """A finite float, or ``None`` for blanks and unparsable text."""

    if _blank(value):
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _float(value)
    return number if number is not None and number > 0.0 else None


def _count(value: Any) -> int | None:
    """A positive integer (``"2"``, ``"2.0"``), else ``None``."""

    number = _float(value)
    if number is None or number < 1.0 or number != int(number):
        return None
    return int(number)


def _bool(value: Any) -> bool | None:
    if _blank(value):
        return None
    text = str(value).strip().lower()
    if text in ("t", "true", "1", "yes", "y"):
        return True
    if text in ("f", "false", "0", "no", "n"):
        return False
    return None


_HSTORE_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=>\s*("((?:[^"\\]|\\.)*)"|NULL)')


def _hstore(text: str) -> dict[str, str]:
    """GridKit ``"key"=>"value", ...`` tags -> dict (NULLs dropped)."""

    out = {}
    for match in _HSTORE_PAIR.finditer(text or ""):
        if match.group(2) != "NULL":
            out[match.group(1)] = match.group(3)
    return out


def _haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = (np.radians(v) for v in (lon1, lat1, lon2, lat2))
    h = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * _MEAN_EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def _number_text(value: float) -> str:
    """Full-precision text for a number tag (no exponent for grid
    coordinates or capacities)."""

    return format(value, ".15g")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Ledger:
    """Per-source counts that end up in the provenance record."""

    rows: int = 0
    refused: Counter = field(default_factory=Counter)
    examples: list = field(default_factory=list)
    fallbacks: Counter = field(default_factory=Counter)
    derived: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    notes: list = field(default_factory=list)

    def refuse(self, where: str, reason: str) -> None:
        self.refused[reason.split(":")[0]] += 1
        if len(self.examples) < _EXAMPLES:
            self.examples.append(f"{where}: {reason}")

    def record(self) -> dict[str, Any]:
        return {"rows_read": self.rows,
                "rows_refused": sum(self.refused.values()),
                "refusals": dict(self.refused),
                "refusal_examples": list(self.examples),
                "rows_skipped": dict(self.skipped),
                "geometry_fallbacks": dict(self.fallbacks),
                "attributes_derived": dict(self.derived),
                "notes": list(self.notes)}


def _file_record(path: Path) -> dict[str, str]:
    return {"name": path.name, "sha256": sha256_file(path)}


# --------------------------------------------------------------------------
# CSV reading


def _read_text(path: Path) -> tuple[str, str]:
    """File text and the encoding that decoded it (UTF-8, then cp1252,
    then latin-1, which cannot fail)."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ImportRefused(f"{path}: cannot decode")  # unreachable: latin-1


def _read_csv(path: Path, *, quotechars: Sequence[str] = ("'", '"')
              ) -> tuple[list[str], list[dict[str, str]], str]:
    """Header, rows as dicts, and encoding.

    PyPSA-Eur ships two dialects: GridKit/prebuilt quote WKT and hstore
    fields with ``'``, pandas exports quote with ``"``.  The first quote
    character under which every row has the header's width wins.
    """

    text, encoding = _read_text(path)
    csv.field_size_limit(max(csv.field_size_limit(), 1 << 30))
    for quotechar in quotechars:
        reader = csv.reader(io.StringIO(text, newline=""),
                            quotechar=quotechar)
        rows = []
        ragged = False
        try:
            header = next(reader, None)
            if header is None:
                raise ImportRefused(f"{path} is empty")
            header = [" ".join(name.split()) for name in header]
            if len(set(header)) != len(header):
                raise ImportRefused(f"{path} repeats a column name in its "
                                    f"header {header}")
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) != len(header):
                    ragged = True
                    break
                rows.append(dict(zip(header, row)))
        except csv.Error as error:
            raise ImportRefused(f"{path} is not a readable CSV table: "
                                f"{error}") from error
        if not ragged:
            return header, rows, encoding
    raise ImportRefused(
        f"{path}: rows do not match the header width under either CSV "
        "quoting dialect; the file is malformed or not a CSV table")


def _pick(header: Sequence[str], *names: str) -> str | None:
    for name in names:
        if name in header:
            return name
    return None


# --------------------------------------------------------------------------
# generator source vocabularies


_REPD_TECHNOLOGY = {
    "solar photovoltaics": "solar",
    "wind onshore": "wind",
    "wind offshore": "wind",
    "battery": "battery",
    "anaerobic digestion": "biogas",
    "landfill gas": "biogas",
    "sewage sludge digestion": "biogas",
    "biomass (dedicated)": "biomass",
    "biomass (co-firing)": "biomass",
    "efw incineration": "waste",
    "advanced conversion technologies": "waste",
    "small hydro": "hydro",
    "large hydro": "hydro",
    "pumped storage hydroelectricity": "hydro",
    "tidal stream": "tidal",
    "tidal lagoon": "tidal",
    "tidal barrage and tidal stream": "tidal",
    "shoreline wave": "wave",
    "wave": "wave",
    "geothermal": "geothermal",
    "hot dry rocks (hdr)": "geothermal",
    "hydrogen": "other",
    "fuel cell (hydrogen)": "other",
    "liquid air energy storage": "other",
    "compressed air energy storage": "other",
    "flywheels": "other",
    "air source heat pumps": "other",
    "ground source heat pumps": "other",
    "solar thermal": "solar",
}

#: Keyword -> generator source for PyPSA carriers and GridKit technologies,
#: tried in order.
_PYPSA_KEYWORDS: tuple[tuple[str, str | None], ...] = (
    ("hydrogen", None), ("biogas", "biogas"), ("biomass", "biomass"),
    ("offwind", "wind"), ("onwind", "wind"), ("wind", "wind"),
    ("solar", "solar"), ("pv", "solar"),
    ("ror", "hydro"), ("hydro", "hydro"), ("phs", "hydro"),
    ("nuclear", "nuclear"), ("lignite", "coal"), ("coal", "coal"),
    ("ocgt", "gas"), ("ccgt", "gas"), ("gas", "gas"), ("oil", "oil"),
    ("geothermal", "geothermal"),
    ("waste", "waste"), ("battery", "battery"), ("tidal", "tidal"),
    ("wave", "wave"), ("other", "other"), ("mixed", "other"),
)

_OSM_SOURCE_ALIASES = {"photovoltaic": "solar", "pv": "solar",
                       "wind_turbine": "wind", "water": "hydro",
                       "diesel": "oil", "lignite": "coal"}


def _pypsa_generator_source(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    for keyword, source in _PYPSA_KEYWORDS:
        if keyword in lowered:
            return source
    return None


def _osm_generator_source(text: Any) -> str | None:
    if _blank(text):
        return None
    first = str(text).split(";")[0].strip().lower()
    first = _OSM_SOURCE_ALIASES.get(first, first)
    return first if first in GENERATOR_SOURCES else None


# --------------------------------------------------------------------------
# PyPSA-Eur


_PYPSA_TABLES = ("buses", "lines", "links", "transformers", "converters",
                 "generators")
_PYPSA_REQUIRED = ("buses", "lines", "links")
_PYPSA_EXPECTED = {
    "buses": "an id column (bus_id, Bus or name), x (longitude), y "
             "(latitude); optional voltage or v_nom, dc, symbol, carrier, "
             "under_construction, country, tags",
    "lines": "an id column (line_id, Line or name), bus0, bus1; optional "
             "voltage or v_nom, circuits or num_parallel, length, "
             "underground, under_construction, type, tags, geometry (WKT)",
    "links": "an id column (link_id, Link or name), bus0, bus1; optional "
             "voltage, p_nom, carrier, length, underground, "
             "under_construction, tags, geometry (WKT)",
    "transformers": "an id column (transformer_id, Transformer or name), "
                    "bus0, bus1",
    "converters": "an id column (converter_id or name), bus0, bus1",
    "generators": "an id column (generator_id, Generator or name), a bus "
                  "column (bus_id or bus), technology or carrier; optional "
                  "capacity or p_nom, geometry (WKT POINT)",
}
_ID_COLUMNS = {
    "buses": ("bus_id", "Bus", "name", ""),
    "lines": ("line_id", "Line", "name", ""),
    "links": ("link_id", "Link", "name", ""),
    "transformers": ("transformer_id", "Transformer", "name", ""),
    "converters": ("converter_id", "name", ""),
    "generators": ("generator_id", "Generator", "name", ""),
}


def _pypsa_files(paths: Sequence[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in paths:
        if path.is_dir():
            candidates = [path / f"{table}.csv" for table in _PYPSA_TABLES]
            candidates = [c for c in candidates if c.is_file()]
        elif path.is_file():
            candidates = [path]
        else:
            raise ImportRefused(f"{path} does not exist")
        for candidate in candidates:
            table = candidate.stem.lower()
            if table not in _PYPSA_TABLES:
                raise ImportRefused(
                    f"{candidate.name} is not a PyPSA-Eur network table; "
                    f"expected {', '.join(t + '.csv' for t in _PYPSA_TABLES)}")
            if table in found and found[table].resolve() != candidate.resolve():
                raise ImportRefused(
                    f"two {table}.csv inputs ({found[table]} and {candidate}):"
                    " import one network at a time and combine them with "
                    "--merge")
            found[table] = candidate
    missing = [t for t in _PYPSA_REQUIRED if t not in found]
    if missing:
        raise ImportRefused(
            "a PyPSA-Eur network needs buses.csv, lines.csv and links.csv; "
            f"missing {', '.join(m + '.csv' for m in missing)}")
    return found


def _require(table: str, header: Sequence[str], *groups: Sequence[str]
             ) -> list[str]:
    chosen = []
    for group in groups:
        column = _pick(header, *group)
        if column is None:
            raise ImportRefused(
                f"{table}.csv has columns {list(header)} but lacks "
                f"{' or '.join(repr(g) for g in group if g) or 'an id column'}"
                f"; this importer reads the PyPSA-Eur layouts with "
                f"{_PYPSA_EXPECTED[table]}")
        chosen.append(column)
    return chosen


def _pypsa_variant(header: Sequence[str]) -> str | None:
    if "station_id" in header:
        return "gridkit"
    if "v_nom" in header or "Bus" in header:
        return "pypsa-network"
    if "geometry" in header and "voltage" in header:
        return "osm-prebuilt"
    return None


@dataclass
class _Bus:
    bus_id: str
    lon: float
    lat: float
    voltage: float | None
    dc: bool
    substation: bool
    electric: bool
    tags: dict


class _UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _raw_tags(text: str) -> dict[str, str]:
    """PyPSA ``tags`` column: GridKit hstore or prebuilt OSM refs."""

    if _blank(text):
        return {}
    if "=>" in text:
        parsed = _hstore(text)
        out = {}
        if parsed.get("oid"):
            out["entsoe:oid"] = parsed["oid"]
        for key in ("name", "text_", "TSO", "symbol"):
            value = parsed.get(key, "").strip()
            if value and value.lower() != "none":
                out[f"entsoe:{key.rstrip('_')}"] = value
        return out
    return {"osm:refs": text.strip()}


def _import_pypsa(paths: Sequence[Path]) -> AssetCollection:
    files = _pypsa_files(paths)
    ledger = _Ledger()
    tables = {}
    encodings = set()
    for table, path in files.items():
        header, rows, encoding = _read_csv(path)
        encodings.add(encoding)
        tables[table] = (header, rows)
    bus_header, bus_rows = tables["buses"]
    variant = _pypsa_variant(bus_header)
    if variant is None:
        ledger.notes.append(
            "PyPSA-Eur layout not identified from buses.csv columns; "
            "licence not recorded -- check the network's origin")
        license_, attribution = None, None
    else:
        license_ = PYPSA_VARIANTS[variant]["license"]
        attribution = PYPSA_VARIANTS[variant]["attribution"]
    bus_id_col, x_col, y_col = _require("buses", bus_header,
                                        _ID_COLUMNS["buses"], ("x",), ("y",))
    v_col = _pick(bus_header, "voltage", "v_nom")
    symbol_col = _pick(bus_header, "symbol")
    carrier_col = _pick(bus_header, "carrier")

    assets: list[Asset] = []
    buses: dict[str, _Bus] = {}
    for index, row in enumerate(bus_rows, start=2):
        ledger.rows += 1
        bus_id = row[bus_id_col].strip()
        lon, lat = _float(row[x_col]), _float(row[y_col])
        if not bus_id:
            ledger.refuse(f"buses.csv line {index}", "no bus id")
            continue
        if bus_id in buses:
            ledger.refuse(f"buses.csv bus {bus_id}", "duplicate bus id")
            continue
        if (lon is None or lat is None or not -180.0 <= lon <= 180.0
                or not -90.0 <= lat <= 90.0):
            ledger.refuse(f"buses.csv bus {bus_id}",
                          f"no valid x/y: {row[x_col]!r}, {row[y_col]!r}")
            continue
        symbol = (row.get(symbol_col, "") if symbol_col else "").strip()
        carrier = (row.get(carrier_col, "") if carrier_col else "").strip()
        substation = (not symbol or "substation" in symbol.lower()
                      or "converter" in symbol.lower())
        electric = not carrier or carrier in ("AC", "DC")
        if not electric:
            substation = False
            ledger.skipped[f"bus carrier {carrier} (not a substation)"] += 1
        elif not substation:
            ledger.skipped[f"bus symbol {symbol} (not a substation)"] += 1
        tags = _raw_tags(row.get("tags", ""))
        if symbol:
            tags["pypsa:symbol"] = symbol
        if row.get("country", "").strip():
            tags["country"] = row["country"].strip()
        if _bool(row.get("under_construction")):
            tags["under_construction"] = "yes"
        dc = bool(_bool(row.get("dc"))) or carrier == "DC"
        buses[bus_id] = _Bus(bus_id, lon, lat,
                             _positive(row[v_col]) if v_col else None,
                             dc, substation, electric, tags)

    # Substations: electrical buses at one location, or joined by a short
    # transformer or converter, are one asset.  Non-electric buses (H2,
    # heat, battery stores) share their AC bus's coordinates in sector-
    # coupled networks and stay out of it.
    union = _UnionFind(buses)
    by_location: dict[tuple[float, float], str] = {}
    for bus in buses.values():
        if not bus.electric:
            continue
        key = (round(bus.lon, 6), round(bus.lat, 6))
        if key in by_location:
            union.union(by_location[key], bus.bus_id)
        else:
            by_location[key] = bus.bus_id
    joined = Counter()
    for table in ("transformers", "converters"):
        if table not in tables:
            continue
        header, rows = tables[table]
        _require(table, header, _ID_COLUMNS[table], ("bus0",), ("bus1",))
        for row in rows:
            ledger.rows += 1
            b0, b1 = buses.get(row["bus0"].strip()), buses.get(row["bus1"].strip())
            if b0 is None or b1 is None or not (b0.electric and b1.electric):
                joined[f"{table} with an unknown or non-electric bus "
                       "(ignored)"] += 1
                continue
            if float(_haversine_m(b0.lon, b0.lat, b1.lon, b1.lat)) \
                    <= _SUBSTATION_JOIN_M:
                union.union(b0.bus_id, b1.bus_id)
                joined[f"{table} joining buses into one substation"] += 1
            else:
                joined[f"{table} longer than 1 km (buses kept apart)"] += 1
    ledger.notes.extend(f"{count} {what}" for what, count in sorted(joined.items()))

    groups: dict[str, list[_Bus]] = {}
    for bus in buses.values():
        groups.setdefault(union.find(bus.bus_id), []).append(bus)
    for members in groups.values():
        stations = [b for b in members if b.substation]
        if not stations:
            continue
        voltages = sorted({b.voltage for b in members if b.voltage}, reverse=True)
        anchor = min(stations, key=lambda b: (-(b.voltage or 0.0), b.bus_id))
        ids = sorted(b.bus_id for b in members)
        tags = dict(anchor.tags)
        tags["pypsa:buses"] = ";".join(ids)
        if any(b.dc for b in members):
            tags["dc"] = "yes"
        name = tags.pop("entsoe:name", None)
        operator = tags.pop("entsoe:TSO", None)
        try:
            assets.append(Asset(
                asset_id=f"{PYPSA_SOURCE}:substation/{anchor.bus_id}",
                kind="substation", source=PYPSA_SOURCE,
                source_ref=f"bus/{anchor.bus_id}", name=name,
                operator=operator,
                geometry={"type": "Point", "coordinates": [anchor.lon,
                                                           anchor.lat]},
                voltage_kv=tuple(voltages), license=license_, tags=tags))
        except ContractError as error:
            ledger.refuse(f"buses.csv bus {anchor.bus_id}", _refusal(error))

    for table in ("lines", "links"):
        header, rows = tables[table]
        id_col, b0_col, b1_col = _require(table, header, _ID_COLUMNS[table],
                                          ("bus0",), ("bus1",))
        seen: set[str] = set()
        for index, row in enumerate(rows, start=2):
            ledger.rows += 1
            where = f"{table}.csv line {index}"
            try:
                if row[id_col].strip() in seen:
                    raise _RowRefused(f"duplicate id: {row[id_col].strip()}")
                seen.add(row[id_col].strip())
                asset = _pypsa_branch(table, row, id_col, b0_col, b1_col,
                                      header, buses, ledger, license_,
                                      length_scale=1000.0
                                      if variant == "pypsa-network" else 1.0)
            except (_RowRefused, ContractError) as error:
                ledger.refuse(where, _refusal(error))
                continue
            if asset is not None:
                assets.append(asset)

    if "generators" in tables:
        header, rows = tables["generators"]
        id_col, bus_col = _require("generators", header,
                                   _ID_COLUMNS["generators"],
                                   ("bus_id", "bus"))
        seen = set()
        for index, row in enumerate(rows, start=2):
            ledger.rows += 1
            try:
                if row[id_col].strip() in seen:
                    raise _RowRefused(f"duplicate id: {row[id_col].strip()}")
                seen.add(row[id_col].strip())
                assets.append(_pypsa_generator(row, id_col, bus_col, header,
                                               buses, ledger, license_))
            except (_RowRefused, ContractError) as error:
                ledger.refuse(f"generators.csv line {index}", _refusal(error))

    record = {"source": PYPSA_SOURCE, "license": license_,
              "attribution": attribution, "variant": variant,
              "files": [_file_record(p) for p in files.values()],
              "encoding": sorted(encodings), "imported_utc": _now_utc()}
    record.update(ledger.record())
    return _collection(assets, [record])


def _pypsa_branch(table: str, row: Mapping[str, str], id_col: str,
                  b0_col: str, b1_col: str, header: Sequence[str],
                  buses: Mapping[str, _Bus], ledger: _Ledger,
                  license_: str | None, *, length_scale: float = 1.0
                  ) -> Asset | None:
    """One ``lines.csv``/``links.csv`` row.  ``length_scale`` turns the
    ``length`` column into metres (PyPSA network exports store km)."""

    branch_id = row[id_col].strip()
    if not branch_id:
        raise _RowRefused("no id")
    carrier = row.get("carrier", "").strip()
    if table == "links" and carrier and carrier != "DC":
        ledger.skipped[f"link carrier {carrier} (not a DC link)"] += 1
        return None
    bus0, bus1 = row[b0_col].strip(), row[b1_col].strip()
    tags: dict[str, str] = {"pypsa:bus0": bus0, "pypsa:bus1": bus1}
    geometry = None
    wkt = row.get("geometry", "")
    if not _blank(wkt.strip("'\" ")):
        try:
            geometry = parse_wkt(wkt)
        except WKTError as error:
            raise _RowRefused(f"bad WKT geometry: {error}") from error
        if geometry["type"] == "Point":
            raise _RowRefused("bad WKT geometry: a branch cannot be a POINT")
    else:
        b0, b1 = buses.get(bus0), buses.get(bus1)
        if b0 is None or b1 is None:
            raise _RowRefused("no geometry and bus0/bus1 not in buses.csv")
        if (b0.lon, b0.lat) == (b1.lon, b1.lat):
            raise _RowRefused("no geometry and bus0/bus1 coincide")
        geometry = {"type": "LineString",
                    "coordinates": [[b0.lon, b0.lat], [b1.lon, b1.lat]]}
        tags["woof:geometry"] = "bus-to-bus"
        ledger.fallbacks["bus-to-bus"] += 1
    v_col = _pick(header, "voltage", "v_nom")
    voltage = _positive(row[v_col]) if v_col else None
    if voltage is None and buses.get(bus0) and buses[bus0].voltage:
        voltage = buses[bus0].voltage
        tags["woof:voltage"] = "from-bus0"
        ledger.derived["voltage_kv from bus0"] += 1
    circuits = None
    c_col = _pick(header, "circuits", "num_parallel")
    if c_col and not _blank(row[c_col]):
        circuits = _count(row[c_col])
        if circuits is None:
            tags[f"pypsa:{c_col}"] = row[c_col].strip()
    underground = bool(_bool(row.get("underground")))
    kind = "cable" if underground else "line"
    length = _positive(row.get("length"))
    if length is not None:
        tags["length_m"] = f"{length * length_scale:.1f}"
    if _bool(row.get("under_construction")):
        tags["under_construction"] = "yes"
    if row.get("type", "").strip():
        tags["pypsa:type"] = row["type"].strip()
    tags.update(_raw_tags(row.get("tags", "")))
    name = tags.pop("entsoe:name", None)
    if table == "links":
        tags["carrier"] = "DC"
        p_nom = _positive(row.get("p_nom"))
        if p_nom is not None:
            tags["p_nom_mw"] = _number_text(p_nom)
    singular = table[:-1]
    return Asset(asset_id=f"{PYPSA_SOURCE}:{singular}/{branch_id}",
                 kind=kind, geometry=geometry, source=PYPSA_SOURCE,
                 source_ref=f"{singular}/{branch_id}", name=name,
                 voltage_kv=(voltage,) if voltage else (),
                 circuits=circuits, license=license_, tags=tags)


def _pypsa_generator(row: Mapping[str, str], id_col: str, bus_col: str,
                     header: Sequence[str], buses: Mapping[str, _Bus],
                     ledger: _Ledger, license_: str | None) -> Asset:
    gen_id = row[id_col].strip()
    if not gen_id:
        raise _RowRefused("no id")
    bus_id = row[bus_col].strip()
    tags: dict[str, str] = {"pypsa:bus": bus_id}
    wkt = row.get("geometry", "")
    if not _blank(wkt.strip("'\" ")):
        try:
            geometry = parse_wkt(wkt)
        except WKTError as error:
            raise _RowRefused(f"bad WKT geometry: {error}") from error
        if geometry["type"] != "Point":
            raise _RowRefused("bad WKT geometry: a generator must be a POINT")
    else:
        bus = buses.get(bus_id)
        if bus is None:
            raise _RowRefused("no geometry and bus not in buses.csv")
        geometry = {"type": "Point", "coordinates": [bus.lon, bus.lat]}
        tags["woof:geometry"] = "bus-location"
        ledger.fallbacks["bus-location"] += 1
    technology = (row.get("technology") or row.get("carrier") or "").strip()
    source = _pypsa_generator_source(technology)
    if technology:
        tags["pypsa:technology"] = technology
    capacity = _positive(row.get("capacity") or row.get("p_nom"))
    tags.update(_raw_tags(row.get("tags", "")))
    name = tags.pop("entsoe:name", None)
    return Asset(asset_id=f"{PYPSA_SOURCE}:generator/{gen_id}", kind="plant",
                 geometry=geometry, source=PYPSA_SOURCE,
                 source_ref=f"generator/{gen_id}", name=name,
                 generator_source=source, capacity_mw=capacity,
                 license=license_, tags=tags)


# --------------------------------------------------------------------------
# REPD


_REPD_REQUIRED = ("Ref ID", "Site Name", "Technology Type",
                  "Installed Capacity (MWelec)", "X-coordinate",
                  "Y-coordinate")
_REPD_TAGS = {
    "Old Ref ID": "repd:old_ref_id",
    "Record Last Updated (dd/mm/yyyy)": "repd:last_updated",
    "Development Status": "repd:status_detail",
    "Storage Type": "repd:storage_type",
    "Mounting Type for Solar": "repd:solar_mounting",
    "No. of Turbines": "repd:turbines",
    "Turbine Capacity (MW)": "repd:turbine_capacity_mw",
    "Height of Turbines (m)": "repd:turbine_height_m",
    "Offshore Wind Round": "repd:offshore_wind_round",
    "CfD Capacity (MW)": "repd:cfd_capacity_mw",
    "County": "repd:county",
    "Region": "repd:region",
    "Country": "repd:country",
    "Operational": "repd:operational_date",
    "Solar Site Area (sqm)": "repd:solar_site_area_m2",
}
#: Generous boxes the converted positions must fall in (lon/lat degrees),
#: offshore wind included.
_GB_BOX = (-11.0, 49.0, 3.5, 62.0)
_NI_BOX = (-8.3, 53.9, -5.3, 55.5)


def _in_box(lon: float, lat: float, box: tuple[float, float, float, float]
            ) -> bool:
    return box[0] <= lon <= box[2] and box[1] <= lat <= box[3]


def _import_repd(path: Path) -> AssetCollection:
    header, rows, encoding = _read_csv(path, quotechars=('"',))
    missing = [c for c in _REPD_REQUIRED if c not in header]
    if missing:
        raise ImportRefused(
            f"{path.name} is not a REPD extract: lacks columns {missing}; "
            f"expected {list(_REPD_REQUIRED)} plus optional "
            "'Development Status (short)', 'Operator (or Applicant)', "
            "'Height of Turbines (m)', 'No. of Turbines', 'Turbine Capacity "
            "(MW)', 'Country'")
    ledger = _Ledger()
    ledger.notes.append(
        "positions: British National Grid -> WGS84 by inverse Transverse "
        "Mercator (Airy 1830) and the OS Helmert transformation, about "
        "3.5 m (95 %); not OSTN15")
    ledger.notes.append(
        "'Height of Turbines (m)' kept as tag repd:turbine_height_m and not "
        "used as hub height (planning heights are to blade tip)")
    candidates = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        ledger.rows += 1
        ref = row["Ref ID"].strip()
        where = f"line {index}" + (f" (Ref ID {ref})" if ref else "")
        if not ref:
            ledger.refuse(where, "no Ref ID")
            continue
        if ref in seen:
            ledger.refuse(where, "duplicate Ref ID")
            continue
        seen.add(ref)
        easting, northing = _float(row["X-coordinate"]), _float(row["Y-coordinate"])
        if easting is None or northing is None:
            ledger.refuse(where, "no X/Y coordinates")
            continue
        if not (-100000.0 <= easting <= 800000.0
                and -100000.0 <= northing <= 1400000.0):
            ledger.refuse(where, "X/Y outside the National Grid: "
                                 f"{easting:g}, {northing:g}")
            continue
        candidates.append((where, row, easting, northing))
    if candidates:
        lon, lat = bng_to_wgs84([c[2] for c in candidates],
                                [c[3] for c in candidates])
    else:
        lon = lat = np.zeros(0)
    assets = []
    statuses = Counter()
    for (where, row, easting, northing), x, y in zip(candidates, lon, lat):
        country = row.get("Country", "").strip()
        box = _NI_BOX if country == "Northern Ireland" else _GB_BOX
        if not _in_box(float(x), float(y), box):
            reason = ("X/Y not on the National Grid for a Northern Ireland "
                      "row (likely Irish Grid)"
                      if country == "Northern Ireland"
                      else "X/Y converts outside Great Britain")
            ledger.refuse(where, f"{reason}: {easting:g}, {northing:g}")
            continue
        technology = row["Technology Type"].strip()
        source = _REPD_TECHNOLOGY.get(technology.lower())
        if technology and source is None:
            ledger.derived[f"generator_source null: technology "
                           f"{technology!r} has no mapping"] += 1
        status = (row.get("Development Status (short)")
                  or row.get("Development Status") or "").strip()
        statuses[status or "unknown"] += 1
        tags = {"repd:technology": technology} if technology else {}
        if status:
            tags["repd:status"] = status
        tags["repd:easting"] = _number_text(easting)
        tags["repd:northing"] = _number_text(northing)
        for column, key in _REPD_TAGS.items():
            value = (row.get(column) or "").strip()
            if value:
                tags[key] = value
        if technology == "Wind Offshore":
            tags["offshore"] = "yes"
        ref = row["Ref ID"].strip()
        try:
            assets.append(Asset(
                asset_id=f"{REPD_SOURCE}:{ref}", kind="plant",
                geometry={"type": "Point",
                          "coordinates": [round(float(x), 7),
                                          round(float(y), 7)]},
                source=REPD_SOURCE, source_ref=ref,
                name=row["Site Name"].strip() or None,
                operator=(row.get("Operator (or Applicant)") or "").strip()
                or None,
                generator_source=source,
                capacity_mw=_positive(row["Installed Capacity (MWelec)"]),
                license=REPD_LICENSE, tags=tags))
        except ContractError as error:
            ledger.refuse(where, _refusal(error))
    record = {"source": REPD_SOURCE, "license": REPD_LICENSE,
              "attribution": REPD_ATTRIBUTION,
              "files": [_file_record(path)], "encoding": encoding,
              "crs": "EPSG:27700 (OSGB36 / British National Grid)",
              "imported_utc": _now_utc(),
              "status_counts": dict(statuses)}
    record.update(ledger.record())
    return _collection(assets, [record])


# --------------------------------------------------------------------------
# generic GeoJSON / CSV


_STEM_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_POWER_UNITS = {"w": 1e-6, "kw": 1e-3, "mw": 1.0, "gw": 1e3, "mwp": 1.0,
                "kwp": 1e-3, "mwe": 1.0, "mwel": 1.0}


def _generic_source(fmt: str, path: Path) -> str:
    return f"{fmt}:{_STEM_SAFE.sub('_', path.stem) or 'input'}"


def _power_mw(value: Any) -> float | None:
    """``"50 MW"``, ``"300 kW"``, ``12.5`` (MW) -> MW; else ``None``."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _positive(value)
    if _blank(value):
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]*)\s*", str(value))
    if match is None:
        return None
    number = _positive(match.group(1))
    unit = match.group(2).lower() or "mw"
    if number is None or unit not in _POWER_UNITS:
        return None
    return number * _POWER_UNITS[unit]


def _voltages(props: Mapping[str, Any], notes: Counter
              ) -> tuple[tuple[float, ...], dict[str, str]]:
    """Voltage list in kV from ``voltage_kv`` (kV) or OSM ``voltage`` (V)."""

    leftovers: dict[str, str] = {}
    if "voltage_kv" in props and props["voltage_kv"] is not None:
        raw = props["voltage_kv"]
        parts = raw if isinstance(raw, (list, tuple)) else str(raw).split(";")
        values = [_positive(p) for p in parts]
        if all(v is not None for v in values) and values:
            return tuple(values), leftovers
        notes["unparsed voltage_kv kept as tag"] += 1
        leftovers["voltage_kv"] = json.dumps(raw) if isinstance(raw, list) \
            else str(raw)
        return (), leftovers
    if "voltage" in props and not _blank(props["voltage"]):
        raw = str(props["voltage"])
        values = [_positive(p) for p in raw.split(";")]
        if values and all(v is not None and v >= 1000.0 for v in values):
            return tuple(v / 1000.0 for v in values), leftovers
        notes["voltage not in volts (ambiguous) kept as tag"] += 1
        leftovers["voltage"] = raw
    return (), leftovers


_GENERIC_CONSUMED = {
    "kind", "power", "name", "operator",
    "voltage_kv", "voltage", "circuits", "cables", "frequency_hz",
    "frequency", "generator_source", "generator:source", "plant:source",
    "capacity_mw", "plant:output:electricity", "generator:output:electricity",
    "hub_height_m", "height:hub", "rotor_diameter_m", "rotor:diameter",
    "tags", "license",
}


def _tag_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _number_text(value) if isinstance(value, float) \
            else str(value)
    return json.dumps(value, sort_keys=True)


def _generic_asset(props: Mapping[str, Any], geometry: Mapping[str, Any],
                   kind: str, source: str, ref: str, license_: str | None,
                   notes: Counter, skip: Iterable[str] = ()) -> Asset:
    tags: dict[str, str] = {}
    voltages, leftovers = _voltages(props, notes)
    tags.update(leftovers)

    def number(*keys: str, parse=_positive):
        for key in keys:
            if key in props and not _blank(props[key]):
                value = parse(props[key])
                if value is None:
                    notes[f"unparsed {key} kept as tag"] += 1
                    tags[key] = _tag_text(props[key])
                return value
        return None

    circuits = number("circuits", parse=_count)
    cables = number("cables", parse=_count)
    frequency = number("frequency_hz", "frequency")
    capacity = number("capacity_mw")
    if capacity is None and "capacity_mw" not in tags:
        capacity = number("plant:output:electricity",
                          "generator:output:electricity", parse=_power_mw)
    hub = number("hub_height_m", "height:hub")
    rotor = number("rotor_diameter_m", "rotor:diameter")
    generator_source = None
    for key in ("generator_source", "generator:source", "plant:source"):
        if key in props and not _blank(props[key]):
            generator_source = _osm_generator_source(props[key])
            if generator_source is None:
                notes[f"unmapped {key} kept as tag"] += 1
                tags[key] = _tag_text(props[key])
            break
    extra = props.get("tags")
    if isinstance(extra, Mapping):
        tags.update({str(k): _tag_text(v) for k, v in extra.items()
                     if v is not None})
    elif not _blank(extra):
        tags["tags"] = _tag_text(extra)
    if isinstance(props.get("license"), str) and not _blank(props["license"]):
        license_ = props["license"].strip()
    skip = set(skip)
    for key, value in props.items():
        if key in _GENERIC_CONSUMED or key in skip or value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        tags[str(key)] = _tag_text(value)
    name = props.get("name")
    operator = props.get("operator")
    return Asset(asset_id=f"{source}/{ref}", kind=kind, geometry=dict(geometry),
                 source=source, source_ref=ref,
                 name=None if _blank(name) else str(name),
                 operator=None if _blank(operator) else str(operator),
                 voltage_kv=voltages, circuits=circuits, cables=cables,
                 frequency_hz=frequency, generator_source=generator_source,
                 capacity_mw=capacity, hub_height_m=hub,
                 rotor_diameter_m=rotor, license=license_, tags=tags)


def _feature_kind(props: Mapping[str, Any], kind: str | None) -> str:
    for key in ("kind", "power"):
        value = props.get(key)
        if not _blank(value):
            if value in ASSET_KINDS:
                return value
            raise _RowRefused(f"{key}={value} is not an asset kind "
                              f"({', '.join(ASSET_KINDS)})")
    if kind is None:
        raise _RowRefused("no 'kind' or 'power' property and no --kind")
    return kind


def _import_geojson(path: Path, *, kind: str | None,
                    id_key: str | None) -> AssetCollection:
    text, _ = _read_text(path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ImportRefused(f"{path} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise ImportRefused(f"{path} is not a GeoJSON object")
    meta = document.get("woof")
    if isinstance(meta, dict) and meta.get("schema") == ASSETS_SCHEMA:
        if kind is not None or id_key is not None:
            raise ImportRefused(f"{path} is already a {ASSETS_SCHEMA} "
                                "document; --kind/--id-col do not apply")
        return AssetCollection.from_geojson(document)
    if document.get("type") == "Feature":
        features = [document]
    elif document.get("type") == "FeatureCollection":
        features = document.get("features") or []
    else:
        raise ImportRefused(f"{path} is neither a GeoJSON FeatureCollection "
                            "nor a Feature")
    source = _generic_source("geojson", path)
    ledger = _Ledger()
    notes = Counter()
    assets = []
    seen: set[str] = set()
    for index, feature in enumerate(features, start=1):
        ledger.rows += 1
        where = f"feature {index}"
        try:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise _RowRefused("not a GeoJSON Feature")
            props = dict(feature.get("properties") or {})
            if id_key is not None:
                if _blank(props.get(id_key)):
                    raise _RowRefused(f"no {id_key!r} property")
                ref = _tag_text(props[id_key])
            elif feature.get("id") is not None:
                ref = _tag_text(feature["id"])
            elif not _blank(props.get("@id")):
                ref = _tag_text(props["@id"])
            elif not _blank(props.get("id")):
                ref = _tag_text(props["id"])
            else:
                ref = str(index)
            if ref in seen:
                raise _RowRefused(f"duplicate id: {ref}")
            feature_kind = _feature_kind(props, kind)
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                raise _RowRefused("no geometry")
            assets.append(_generic_asset(
                props, geometry, feature_kind, source, ref, None, notes,
                skip=(id_key,) if id_key else ()))
            seen.add(ref)
        except (_RowRefused, ContractError) as error:
            ledger.refuse(where, _refusal(error))
    ledger.notes.extend(f"{n} {what}" for what, n in sorted(notes.items()))
    ledger.notes.append("licence not stated by the input; record it before "
                        "redistributing")
    record = {"source": source, "license": None,
              "files": [_file_record(path)], "imported_utc": _now_utc()}
    record.update(ledger.record())
    return _collection(assets, [record])


def _import_csv(path: Path, *, kind: str | None,
                column_map: Mapping[str, str]) -> AssetCollection:
    lat_col, lon_col = column_map.get("lat"), column_map.get("lon")
    id_col = column_map.get("id")
    if not lat_col or not lon_col:
        raise ImportRefused("--format csv needs --lat-col and --lon-col")
    if kind is None:
        raise ImportRefused("--format csv needs --kind (every row becomes "
                            "an asset of that kind)")
    if kind in LINEAR_KINDS:
        raise ImportRefused(f"--format csv makes Point assets, and a {kind} "
                            "is a line; import lines as GeoJSON")
    header, rows, encoding = _read_csv(path, quotechars=('"',))
    missing = [c for c in (lat_col, lon_col, id_col) if c and c not in header]
    if missing:
        raise ImportRefused(f"{path.name} lacks columns {missing}; it has "
                            f"{header}")
    source = _generic_source("csv", path)
    ledger = _Ledger()
    notes = Counter()
    assets = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        ledger.rows += 1
        ref = row[id_col].strip() if id_col else str(number)
        where = f"row {number}" + (f" ({id_col}={ref})" if id_col else "")
        try:
            if not ref:
                raise _RowRefused(f"empty id column: {id_col}")
            if ref in seen:
                raise _RowRefused(f"duplicate id: {ref}")
            lat, lon = _float(row[lat_col]), _float(row[lon_col])
            if lat is None or lon is None:
                raise _RowRefused(f"no {lat_col}/{lon_col}: "
                                  f"{row[lat_col]!r}, {row[lon_col]!r}")
            props = {k: v for k, v in row.items()
                     if k not in (lat_col, lon_col)}
            assets.append(_generic_asset(
                props, {"type": "Point", "coordinates": [lon, lat]}, kind,
                source, ref, None, notes, skip=(id_col,) if id_col else ()))
            seen.add(ref)
        except (_RowRefused, ContractError) as error:
            ledger.refuse(where, _refusal(error))
    ledger.notes.extend(f"{n} {what}" for what, n in sorted(notes.items()))
    ledger.notes.append("licence not stated by the input; record it before "
                        "redistributing")
    record = {"source": source, "license": None,
              "files": [_file_record(path)], "encoding": encoding,
              "imported_utc": _now_utc()}
    record.update(ledger.record())
    return _collection(assets, [record])


def _collection(assets: list[Asset], sources: list[dict]) -> AssetCollection:
    seen: dict[str, Asset] = {}
    for asset in assets:
        if asset.asset_id in seen:
            raise ImportRefused(f"two inputs produce asset_id "
                                f"{asset.asset_id!r}; rename one file")
        seen[asset.asset_id] = asset
    return AssetCollection(assets=assets, sources=sources)


# --------------------------------------------------------------------------
# public API


def import_assets(paths: Sequence[Path], *, fmt: str,
                  column_map: Mapping[str, str] | None = None,
                  kind: str | None = None) -> AssetCollection:
    """Read ``paths`` in format ``fmt``.  ``column_map`` keys: lat, lon, id.

    ``pypsa-eur`` reads one network from all ``paths`` together; the other
    formats read each path on its own and fold them with
    :func:`merge_collections`.  Per-row refusals, skips and geometry
    fallbacks are recorded in each provenance record; an import that
    yields no asset is refused (:class:`ImportRefused`).
    """

    if fmt not in FORMATS:
        raise ImportRefused(f"unknown format {fmt!r}; expected one of "
                            f"{', '.join(FORMATS)}")
    if kind is not None and kind not in ASSET_KINDS:
        raise ImportRefused(f"--kind {kind!r} is not one of "
                            f"{', '.join(ASSET_KINDS)}")
    paths = [Path(p) for p in paths]
    if not paths:
        raise ImportRefused("no input files given")
    column_map = dict(column_map or {})
    unknown = set(column_map) - {"lat", "lon", "id"}
    if unknown:
        raise ImportRefused(f"column_map keys are lat, lon, id; got "
                            f"{sorted(unknown)}")
    if fmt != "csv" and ("lat" in column_map or "lon" in column_map):
        raise ImportRefused("--lat-col/--lon-col apply to --format csv only")
    if fmt in ("pypsa-eur", "repd") and ("id" in column_map
                                         or kind is not None):
        raise ImportRefused(f"--id-col/--kind do not apply to --format {fmt}")

    if fmt == "pypsa-eur":
        collection = _import_pypsa(paths)
    else:
        collection = None
        stems: dict[str, Path] = {}
        for path in paths:
            if not path.is_file():
                raise ImportRefused(f"{path} is not a file")
            if fmt != "repd":
                name = _generic_source(fmt, path)
                if name in stems and stems[name].resolve() != path.resolve():
                    raise ImportRefused(
                        f"{stems[name]} and {path} would both be source "
                        f"{name!r}; rename one")
                stems[name] = path
            if fmt == "repd":
                part = _import_repd(path)
            elif fmt == "geojson":
                part = _import_geojson(path, kind=kind,
                                       id_key=column_map.get("id"))
            else:
                part = _import_csv(path, kind=kind, column_map=column_map)
            collection = part if collection is None \
                else merge_collections(collection, part)
    if not collection.assets:
        reasons = Counter()
        for record in collection.sources:
            reasons.update(record.get("refusals") or {})
        detail = (f"; refusals: {dict(reasons)}" if reasons
                  else "; the inputs hold no rows")
        examples = [e for r in collection.sources
                    for e in r.get("refusal_examples") or []][:3]
        if examples:
            detail += f"; e.g. {examples}"
        raise ImportRefused(f"no assets imported from {fmt} input{detail}")
    return collection


class _PointIndex:
    """Bucket points in ~100 m cells of an equirectangular projection."""

    def __init__(self, cell_m: float):
        self.cell = cell_m
        self.buckets: dict[tuple, list] = {}

    def _key(self, group: Any, lon: float, lat: float) -> tuple:
        y = math.radians(lat) * _MEAN_EARTH_RADIUS_M
        x = (math.radians(lon) * _MEAN_EARTH_RADIUS_M
             * max(math.cos(math.radians(lat)), 1e-6))
        return (group, int(math.floor(x / self.cell)),
                int(math.floor(y / self.cell)))

    def add(self, group: Any, lon: float, lat: float, item: Any) -> None:
        self.buckets.setdefault(self._key(group, lon, lat), []).append(item)

    def near(self, group: Any, lon: float, lat: float) -> Iterable[Any]:
        g, i, j = self._key(group, lon, lat)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                yield from self.buckets.get((g, i + di, j + dj), ())


def _line_parts(geometry: Mapping[str, Any]) -> list[list]:
    if geometry["type"] == "LineString":
        return [geometry["coordinates"]]
    return list(geometry["coordinates"])


def _line_summary(geometry: Mapping[str, Any]
                  ) -> tuple[tuple[float, float], tuple[float, float], float]:
    parts = _line_parts(geometry)
    length = 0.0
    for part in parts:
        xy = np.asarray([p[:2] for p in part], dtype=np.float64)
        if len(xy) > 1:
            length += float(np.sum(_haversine_m(xy[:-1, 0], xy[:-1, 1],
                                                xy[1:, 0], xy[1:, 1])))
    start = tuple(parts[0][0][:2])
    end = tuple(parts[-1][-1][:2])
    return start, end, length


def _close(a: Sequence[float], b: Sequence[float], tolerance: float) -> bool:
    return float(_haversine_m(a[0], a[1], b[0], b[1])) <= tolerance


def merge_with_report(base: AssetCollection, extra: AssetCollection
                      ) -> tuple[AssetCollection, dict[str, Any]]:
    """:func:`merge_collections` plus its counts.

    An ``extra`` asset is a duplicate of ``base`` when it has the same
    ``asset_id`` or the same ``(source, source_ref)``; when it is a Point
    of the same kind within 50 m of a ``base`` Point; or when it is a line
    of the same kind whose two ends lie within 100 m of a ``base`` line's
    ends (either direction) and whose length is within 5 % of it.
    """

    ids = {a.asset_id for a in base.assets}
    refs = {(a.source, a.source_ref) for a in base.assets
            if a.source_ref is not None}
    points = _PointIndex(MERGE_POINT_M * 2.0)
    lines = _PointIndex(MERGE_LINE_END_M)
    for asset in base.assets:
        geometry = asset.geometry
        if geometry["type"] == "Point":
            lon, lat = geometry["coordinates"][:2]
            points.add(asset.kind, lon, lat, (lon, lat))
        elif asset.kind in LINEAR_KINDS:
            start, end, length = _line_summary(geometry)
            entry = (start, end, length)
            lines.add(asset.kind, *start, entry)
            lines.add(asset.kind, *end, entry)
    kept: list[Asset] = []
    dropped = Counter()
    for asset in extra.assets:
        geometry = asset.geometry
        if asset.asset_id in ids:
            dropped["same asset_id"] += 1
            continue
        if asset.source_ref is not None and (asset.source,
                                             asset.source_ref) in refs:
            dropped["same source and source_ref"] += 1
            continue
        if geometry["type"] == "Point":
            lon, lat = geometry["coordinates"][:2]
            if any(_close((lon, lat), other, MERGE_POINT_M)
                   for other in points.near(asset.kind, lon, lat)):
                dropped[f"{asset.kind} point within {MERGE_POINT_M:g} m"] += 1
                continue
        elif asset.kind in LINEAR_KINDS:
            start, end, length = _line_summary(geometry)
            duplicate = False
            for b_start, b_end, b_length in lines.near(asset.kind, *start):
                ends = ((_close(start, b_start, MERGE_LINE_END_M)
                         and _close(end, b_end, MERGE_LINE_END_M))
                        or (_close(start, b_end, MERGE_LINE_END_M)
                            and _close(end, b_start, MERGE_LINE_END_M)))
                if ends and abs(length - b_length) \
                        <= MERGE_LINE_LENGTH_FRACTION * b_length:
                    duplicate = True
                    break
            if duplicate:
                dropped[f"{asset.kind} with matching ends and length"] += 1
                continue
        kept.append(asset)
    report = {"base_assets": len(base.assets),
              "offered": len(extra.assets), "kept": len(kept),
              "duplicates_dropped": sum(dropped.values()),
              "duplicates_by_rule": dict(dropped)}
    sources = [dict(r) for r in base.sources]
    for record in extra.sources:
        record = dict(record)
        record["merges"] = list(record.get("merges") or []) + [dict(report)]
        sources.append(record)
    return AssetCollection(assets=list(base.assets) + kept,
                           sources=sources), report


def merge_collections(base: AssetCollection, extra: AssetCollection
                      ) -> AssetCollection:
    """``base`` plus the assets of ``extra`` that duplicate none of it.

    ``base``'s copy of a duplicate is kept; the counts are recorded as a
    ``merge`` member on each of ``extra``'s provenance records, which are
    appended to ``base``'s.  Rules: :func:`merge_with_report`.
    """

    return merge_with_report(base, extra)[0]


# --------------------------------------------------------------------------
# CLI


def _summary(collection: AssetCollection, *, fmt: str, output: str,
             merge_into: str | None,
             merge_report: Mapping[str, Any] | None,
             imported: AssetCollection) -> dict[str, Any]:
    refused = Counter()
    fallbacks = Counter()
    derived = Counter()
    skipped = Counter()
    examples: list[str] = []
    notes: list[str] = []
    rows = 0
    for record in imported.sources:
        rows += record.get("rows_read", 0)
        refused.update(record.get("refusals") or {})
        fallbacks.update(record.get("geometry_fallbacks") or {})
        derived.update(record.get("attributes_derived") or {})
        skipped.update(record.get("rows_skipped") or {})
        examples.extend(record.get("refusal_examples") or [])
        notes.extend(f"{record['source']}: {n}" for n in record.get("notes")
                     or [])
    within = [m for r in imported.sources for m in r.get("merges") or []]
    return {
        "schema": "woof-energy.import-summary.v1",
        "format": fmt,
        "output": output,
        "assets": len(collection.assets),
        "by_kind": dict(Counter(a.kind for a in collection.assets)),
        "by_source": dict(Counter(a.source for a in collection.assets)),
        "imported": len(imported.assets),
        "rows_read": rows,
        "rows_refused": sum(refused.values()),
        "refusals": dict(refused),
        "refusal_examples": examples[:_EXAMPLES],
        "rows_skipped": dict(skipped),
        "geometry_fallbacks": dict(fallbacks),
        "attributes_derived": dict(derived),
        "duplicates_dropped": (
            (merge_report or {}).get("duplicates_dropped", 0)
            + sum(m.get("duplicates_dropped", 0) for m in within)),
        "merge": None if merge_report is None
        else {"into": merge_into, **merge_report},
        "sources": [{k: r.get(k) for k in ("source", "license", "variant")
                     if k in r} for r in collection.sources],
        "notes": notes,
    }


def main(args) -> int:
    column_map = {key: value for key, value in (("lat", args.lat_col),
                                                ("lon", args.lon_col),
                                                ("id", args.id_col))
                  if value}
    try:
        imported = import_assets([Path(p) for p in args.sources],
                                 fmt=args.format, column_map=column_map,
                                 kind=args.kind)
        collection, merge_report = imported, None
        if args.merge:
            base_path = Path(args.merge)
            if not base_path.is_file():
                raise ImportRefused(f"--merge {base_path} does not exist")
            collection, merge_report = merge_with_report(
                load_assets(base_path), imported)
        dump_assets(collection, args.output)
    except (ImportRefused, ContractError, OSError) as error:
        print(f"woof energy import: refused: {error}", file=sys.stderr)
        print(json.dumps({"schema": "woof-energy.import-summary.v1",
                          "format": args.format, "refusal": str(error)},
                         indent=2))
        return 2
    print(json.dumps(_summary(collection, fmt=args.format,
                              output=str(args.output),
                              merge_into=args.merge,
                              merge_report=merge_report, imported=imported),
                     indent=2, default=str))
    return 0


__all__ = [
    "FORMATS", "ImportRefused", "WKTError", "parse_wkt", "bng_to_osgb36",
    "osgb36_to_wgs84", "bng_to_wgs84", "import_assets", "merge_collections",
    "merge_with_report", "main", "MERGE_POINT_M", "MERGE_LINE_END_M",
    "MERGE_LINE_LENGTH_FRACTION",
]
