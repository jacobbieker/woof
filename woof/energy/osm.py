"""OpenStreetMap power infrastructure via the Overpass API.

``woof energy fetch`` asks Overpass for ``power=line|minor_line|cable|
substation|plant|generator|tower`` inside a bbox or polygon and writes a
``woof-energy.assets.v1`` document.  OSM data is (c) OpenStreetMap
contributors, ODbL 1.0; the attribution travels in the document's
``sources`` records.

**Query.**  One Overpass QL query per tile::

    [out:json][timeout:N];
    (
    way["power"~"^(cable|line)$"](S,W,N,E);
    node["power"~"^(generator|plant|substation)$"](S,W,N,E);
    way["power"~"^(generator|plant|substation)$"](S,W,N,E);
    relation["type"="multipolygon"]["power"~"^(generator|plant|substation)$"](S,W,N,E);
    node["power"~"^(tower)$"](S,W,N,E);
    );
    out body geom;

Lines and cables are ways; substations, plants and generators are nodes,
closed ways or multipolygon relations; towers are nodes.  An area given as
a polygon adds a ``(poly:"lat lon ...")`` filter per outer ring, chained
with the tile's bbox (Overpass intersects chained filters).  The polygon is
simplified (Douglas-Peucker, tolerance doubled until it fits) to at most
:data:`MAX_POLYGON_VERTICES` vertices, and inner rings are not excluded;
both are recorded in the output notes.

**Tiling.**  The area is cut along the integer-degree grid, so a tile is at
most 1 deg x 1 deg and interior tiles are the same query whatever area they
were asked for.  A tile the server cannot finish (``runtime error: Query
timed out`` or ``out of memory``) is split into quadrants, down to
:data:`MIN_TILE_DEG` on its longer side (quadrants that miss a polygon area
are dropped).  The split is remembered in the cache with the timeout that
hit it, so ``--offline`` and later runs walk the same tree; ``--refresh`` or
a longer ``--timeout-s`` asks for the whole tile again.  A polygon tile
that lies wholly inside the polygon is sent without its ``poly:`` filter,
so it is the same query, and the same cache entry, as a bbox fetch.

**Cache.**  Each tile's raw response lives under :func:`cache_root`
(``$WOOF_ENERGY_OSM_CACHE`` or ``~/.woof/cache/energy-osm/``) as
``tiles/<kk>/<sha256>.json``, keyed by the sha256 of the query text with
the server timeout left out (so the key does not depend on the endpoint or
on ``--timeout-s``).  The ``.meta.json`` sidecar beside it records
``retrieved_utc``, ``endpoint`` and ``osm_base``
(``osm3s.timestamp_osm_base``) and is written last, atomically: a body with
no sidecar is not a cache hit.  ``offline=True`` reads only the cache and
refuses with every missing tile named; ``refresh=True`` asks again.

**Network.**  Requests are form POSTs (``data=<query>``) through
:func:`woof.nomads_governor.paced_urlopen` and the tree's shared retry,
:func:`woof.fetch_endpoints.ask_along_ladder`, over :data:`ENDPOINTS` (an
``endpoint`` argument is asked first).  Requests to one host are spaced by
:data:`MIN_REQUEST_GAP_S`; 429/5xx answers and the server's own
"rate_limited"/dispatcher-busy pages back off and retry; and the whole fetch
holds to a wall-clock budget (``$WOOF_ENERGY_OSM_BUDGET_S``, default
:data:`DEFAULT_BUDGET_S`).  :class:`OverpassUnavailable` is raised when no
endpoint serves a tile.

**Normalization.**  ``asset_id`` is ``osm:<type>/<id>``; voltages are parsed
from volts (``"400000;132000"`` -> ``(400.0, 132.0)``) and anything that does
not parse is kept verbatim in ``tags``.  Unconsumed OSM tags are kept in
``tags`` too.  Multipolygon relations are assembled into MultiPolygons from
their member ways; when that fails the asset becomes a centroid Point
tagged ``woof:geometry=centroid-fallback``.

All of this is per-element scalar work on a few thousand to a few hundred
thousand small records, so it stays in Python; nothing here is a gridded
data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request

from woof.energy.contracts import (
    ASSET_KINDS,
    GENERATOR_SOURCES,
    LINEAR_KINDS,
    Asset,
    AssetCollection,
    ContractError,
    dump_assets,
    sha256_file,
)

SOURCE = "osm"
ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL 1.0"
LICENSE = "ODbL-1.0"
COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"

CACHE_ENV = "WOOF_ENERGY_OSM_CACHE"
BUDGET_ENV = "WOOF_ENERGY_OSM_BUDGET_S"
SUMMARY_SCHEMA = "woof-energy.fetch-summary.v1"
CACHE_META_SCHEMA = "woof-energy.osm-tile.v1"

#: Public Overpass interpreters, asked in this order.  Both serve the same
#: planet replica; a busy or failing one is a reason to ask the next.
ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

#: Overpass etiquette: never more than one request per host per this many
#: seconds from one process (overpass-api.de grants two slots per client).
MIN_REQUEST_GAP_S = 1.0

#: The wall-clock budget of one fetch, all tiles and retries included.
DEFAULT_BUDGET_S = 3600.0

#: The largest tile edge (the integer-degree grid) and the smallest a tile
#: the server cannot finish is split down to.
MAX_TILE_DEG = 1.0
MIN_TILE_DEG = 1.0 / 16.0

#: The most polygon vertices sent in ``poly:`` filters, all rings together.
MAX_POLYGON_VERTICES = 1000

#: Extra seconds the client waits beyond the server-side timeout.
HTTP_SLACK_S = 60.0

DEFAULT_KINDS = ("line", "cable", "substation", "plant", "generator")
_AREA_KINDS = ("substation", "plant", "generator")
_VOLTAGE_KINDS = ("line", "minor_line", "cable", "substation")

#: ``generator:source`` spellings that name a vocabulary entry under another
#: word.  Anything else outside :data:`GENERATOR_SOURCES` becomes ``other``;
#: in both cases the raw value is kept in ``tags``.
SOURCE_ALIASES = {"diesel": "oil", "biofuel": "biomass"}

GEOMETRY_TAG = "woof:geometry"
CENTROID_FALLBACK = "centroid-fallback"


def _user_agent() -> str:
    try:
        from woof import __version__ as version
    except Exception:                                  # noqa: BLE001
        version = "unknown"
    return (f"woof-energy-osm/{version} "
            "(+https://github.com/recastsystems/woof)")


# --------------------------------------------------------------------------
# errors


class OverpassUnavailable(RuntimeError):
    """No Overpass endpoint served a tile, or the fetch ran out of time."""


class OverpassCacheMiss(RuntimeError):
    """``offline=True`` and at least one tile has no cached response.

    ``missing`` lists ``(bbox, key)`` for each, ``bbox`` as W,S,E,N.
    """

    def __init__(self, message: str, missing: Sequence[tuple[tuple, str]]):
        super().__init__(message)
        self.missing = tuple(missing)


class OverpassTileTooLarge(RuntimeError):
    """A tile at the smallest split still exceeds the server's limits."""


class OverpassQueryError(RuntimeError):
    """The server rejected the query itself (HTTP 400): a bug, not a fault."""


class OsmAreaError(ValueError):
    """The requested area (bbox or polygon file) is not usable."""


class _TooLarge(Exception):
    """The server gave up on this tile (timeout/memory); split it."""


class _Busy(Exception):
    """The server's own rate-limit or dispatcher-busy answer.

    ``transient`` makes :func:`woof.fetch_endpoints.retry_delay` back off
    and ask again, exactly like a 429.
    """

    transient = True

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _OutOfTime(Exception):
    """The fetch's wall-clock budget is spent."""


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# cache location


def cache_root() -> Path:
    """``$WOOF_ENERGY_OSM_CACHE`` or ``~/.woof/cache/energy-osm``."""

    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".woof" / "cache" / "energy-osm"


def _budget_seconds() -> float:
    raw = os.environ.get(BUDGET_ENV)
    if not raw:
        return DEFAULT_BUDGET_S
    try:
        value = float(raw)
    except ValueError as error:
        raise OsmAreaError(f"${BUDGET_ENV} must be seconds, got {raw!r}") \
            from error
    if not math.isfinite(value) or value <= 0.0:
        raise OsmAreaError(f"${BUDGET_ENV} must be positive, got {raw!r}")
    return value


# --------------------------------------------------------------------------
# area: bbox, polygon, tiles


def _fmt(value: float) -> str:
    text = f"{float(value):.7f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


@dataclass(frozen=True)
class Area:
    """What to fetch: a bbox, or polygons (outer rings, ``(lon, lat)``)."""

    bbox: tuple[float, float, float, float]
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    notes: tuple[str, ...] = ()
    polygon_path: str | None = None
    polygon_sha256: str | None = None

    def describe(self) -> dict[str, Any]:
        if not self.polygons:
            return {"bbox": list(self.bbox)}
        return {"polygon": {"path": self.polygon_path,
                            "sha256": self.polygon_sha256,
                            "rings": len(self.polygons),
                            "vertices_sent": sum(len(r) for r in self.polygons)},
                "bbox": list(self.bbox)}


@dataclass(frozen=True)
class Tile:
    """One query cell, W,S,E,N in degrees, with its split depth."""

    west: float
    south: float
    east: float
    north: float
    depth: int = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def filter(self) -> str:
        """Overpass bbox filter: south,west,north,east."""

        return (f"({_fmt(self.south)},{_fmt(self.west)},"
                f"{_fmt(self.north)},{_fmt(self.east)})")

    def quadrants(self) -> tuple["Tile", ...]:
        mid_lon = (self.west + self.east) / 2.0
        mid_lat = (self.south + self.north) / 2.0
        depth = self.depth + 1
        return (Tile(self.west, self.south, mid_lon, mid_lat, depth),
                Tile(mid_lon, self.south, self.east, mid_lat, depth),
                Tile(self.west, mid_lat, mid_lon, self.north, depth),
                Tile(mid_lon, mid_lat, self.east, self.north, depth))

    def label(self) -> str:
        return ",".join(_fmt(v) for v in self.bbox)


def _check_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(v) for v in bbox)
    except (TypeError, ValueError) as error:
        raise OsmAreaError(f"bbox must be four numbers W,S,E,N, got "
                           f"{bbox!r}") from error
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise OsmAreaError(f"bbox must be finite, got {bbox!r}")
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise OsmAreaError(f"bbox needs west < east and south < north in "
                           f"range, got {bbox!r}")
    return west, south, east, north


def tiles_for(area: Area) -> list[Tile]:
    """The area's tiles: the integer-degree grid cut to its bbox.

    Ordered south to north, then west to east.  With polygons, tiles whose
    box meets no polygon are left out.  The result depends only on the
    area, never on the cache or the network.
    """

    west, south, east, north = area.bbox
    step = MAX_TILE_DEG
    tiles: list[Tile] = []
    lat = math.floor(south / step) * step
    while lat < north:
        lon = math.floor(west / step) * step
        while lon < east:
            tile = Tile(max(lon, west), max(lat, south),
                        min(lon + step, east), min(lat + step, north))
            if tile.east > tile.west and tile.north > tile.south:
                if not area.polygons or any(
                        _ring_meets_box(ring, tile.bbox)
                        for ring in area.polygons):
                    tiles.append(tile)
            lon += step
        lat += step
    return tiles


def _point_in_ring(lon: float, lat: float,
                   ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    count = len(ring)
    j = count - 1
    for i in range(count):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < cross:
                inside = not inside
        j = i
    return inside


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (value > 0) - (value < 0)

    def on(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), \
        orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(a, c, b)) or (o2 == 0 and on(a, d, b))
            or (o3 == 0 and on(c, a, d)) or (o4 == 0 and on(c, b, d)))


def _ring_meets_box(ring: Sequence[Sequence[float]],
                    box: tuple[float, float, float, float]) -> bool:
    west, south, east, north = box
    if any(west <= p[0] <= east and south <= p[1] <= north for p in ring):
        return True
    corners = ((west, south), (east, south), (east, north), (west, north))
    if any(_point_in_ring(x, y, ring) for x, y in corners):
        return True
    edges = list(zip(corners, corners[1:] + corners[:1]))
    closed = list(ring) + [ring[0]]
    for p, q in zip(closed, closed[1:]):
        for c, d in edges:
            if _segments_cross(p, q, c, d):
                return True
    return False


def _box_inside_ring(box: tuple[float, float, float, float],
                     ring: Sequence[Sequence[float]]) -> bool:
    """Is the whole box inside the ring (corners inside, no edge crossing)?

    Such a tile needs no ``poly:`` filter, so its query is the plain bbox
    query and shares its cache entry with every other area that covers it.
    """

    west, south, east, north = box
    corners = ((west, south), (east, south), (east, north), (west, north))
    if not all(_point_in_ring(x, y, ring) for x, y in corners):
        return False
    if any(west <= p[0] <= east and south <= p[1] <= north for p in ring):
        return False
    edges = list(zip(corners, corners[1:] + corners[:1]))
    closed = list(ring) + [ring[0]]
    return not any(_segments_cross(p, q, c, d)
                   for p, q in zip(closed, closed[1:]) for c, d in edges)


def _perpendicular(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) \
        / math.hypot(dx, dy)


def _douglas_peucker(points: Sequence[tuple[float, float]],
                     tolerance: float) -> list[tuple[float, float]]:
    """Iterative Douglas-Peucker on an open polyline; endpoints kept."""

    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        worst, index = -1.0, -1
        for i in range(first + 1, last):
            distance = _perpendicular(points[i], points[first], points[last])
            if distance > worst:
                worst, index = distance, i
        if index >= 0 and worst > tolerance:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, kept in zip(points, keep) if kept]


def _simplify_ring(ring: Sequence[tuple[float, float]],
                   tolerance: float) -> list[tuple[float, float]]:
    """An open ring (no closing repeat) simplified, at least 3 vertices."""

    if tolerance <= 0.0 or len(ring) <= 3:
        return list(ring)
    far = max(range(len(ring)), key=lambda i: math.hypot(
        ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]))
    if far == 0:
        return list(ring[:3])
    first = _douglas_peucker(list(ring[:far + 1]), tolerance)
    second = _douglas_peucker(list(ring[far:]) + [ring[0]], tolerance)
    out = first[:-1] + second[:-1]
    if len(out) < 3:
        # Everything within tolerance of the chord: keep the vertex
        # farthest from it so the ring still encloses an area.
        third = max((i for i in range(len(ring)) if i not in (0, far)),
                    key=lambda i: _perpendicular(ring[i], ring[0], ring[far]))
        out = [ring[i] for i in sorted((0, far, third))]
    return out


def _open_ring(coords: Sequence[Sequence[float]], what: str
               ) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for position in coords:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise OsmAreaError(f"{what} has a malformed position {position!r}")
        lon, lat = float(position[0]), float(position[1])
        if not (math.isfinite(lon) and math.isfinite(lat)
                and -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise OsmAreaError(f"{what} has an out-of-range position "
                               f"{position!r}")
        if not ring or ring[-1] != (lon, lat):
            ring.append((lon, lat))
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    if len(ring) < 3:
        raise OsmAreaError(f"{what} ring needs at least three distinct "
                           "vertices")
    return ring


def _polygons_of(document: Mapping[str, Any], what: str
                 ) -> tuple[list[list[tuple[float, float]]], int]:
    """Outer rings of every Polygon/MultiPolygon, and the inner-ring count."""

    kind = document.get("type")
    if kind == "FeatureCollection":
        outers: list[list[tuple[float, float]]] = []
        holes = 0
        for index, feature in enumerate(document.get("features") or []):
            found, inner = _polygons_of(feature, f"{what} feature {index}")
            outers.extend(found)
            holes += inner
        return outers, holes
    if kind == "Feature":
        geometry = document.get("geometry")
        if not isinstance(geometry, Mapping):
            raise OsmAreaError(f"{what} has no geometry")
        return _polygons_of(geometry, what)
    if kind == "Polygon":
        rings = document.get("coordinates") or []
        if not rings:
            raise OsmAreaError(f"{what} Polygon has no rings")
        return [_open_ring(rings[0], what)], len(rings) - 1
    if kind == "MultiPolygon":
        outers, holes = [], 0
        for polygon in document.get("coordinates") or []:
            if not polygon:
                raise OsmAreaError(f"{what} MultiPolygon has an empty part")
            outers.append(_open_ring(polygon[0], what))
            holes += len(polygon) - 1
        return outers, holes
    raise OsmAreaError(f"{what} must be a GeoJSON Polygon or MultiPolygon "
                       f"(or a Feature/FeatureCollection of them), got "
                       f"{kind!r}")


def load_polygon_area(path: str | Path) -> Area:
    """Read a GeoJSON polygon area and simplify it for ``poly:`` filters."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise OsmAreaError(f"polygon file {path} does not exist") from error
    except (OSError, json.JSONDecodeError) as error:
        raise OsmAreaError(f"polygon file {path} is not readable JSON: "
                           f"{error}") from error
    if not isinstance(document, Mapping):
        raise OsmAreaError(f"polygon file {path} is not a JSON object")
    outers, holes = _polygons_of(document, f"polygon file {path}")
    if not outers:
        raise OsmAreaError(f"polygon file {path} holds no polygon")
    original = sum(len(r) for r in outers)
    notes: list[str] = []
    rings = outers
    tolerance = 0.0
    if original > MAX_POLYGON_VERTICES:
        tolerance = 1e-5
        while True:
            rings = [_simplify_ring(r, tolerance) for r in outers]
            if sum(len(r) for r in rings) <= MAX_POLYGON_VERTICES:
                break
            if 3 * len(outers) > MAX_POLYGON_VERTICES:
                raise OsmAreaError(
                    f"polygon file {path} has {len(outers)} rings; at most "
                    f"{MAX_POLYGON_VERTICES // 3} can be sent")
            tolerance *= 2.0
        notes.append(
            f"polygon simplified from {original} to "
            f"{sum(len(r) for r in rings)} vertices (Douglas-Peucker, "
            f"tolerance {tolerance:g} deg); assets within about that distance "
            "of the boundary may be included or left out")
    if holes:
        notes.append(f"{holes} inner ring(s) of the polygon are not excluded: "
                     "Overpass poly filters take outer rings only")
    lons = [p[0] for r in outers for p in r]
    lats = [p[1] for r in outers for p in r]
    bbox = (min(lons), min(lats), max(lons), max(lats))
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise OsmAreaError(f"polygon file {path} encloses no area")
    return Area(bbox=bbox, polygons=tuple(tuple(r) for r in rings),
                notes=tuple(notes), polygon_path=str(path),
                polygon_sha256=sha256_file(path))


# --------------------------------------------------------------------------
# query text


def _kind_regex(kinds: Iterable[str]) -> str:
    return '"power"~"^(' + "|".join(sorted(kinds)) + ')$"'


def query_body(tile: Tile, kinds: Sequence[str],
               polygons: Sequence[Sequence[tuple[float, float]]] = ()) -> str:
    """The statements and output line of one tile's query (no settings)."""

    kinds = set(kinds)
    unknown = kinds - set(ASSET_KINDS)
    if unknown:
        raise OsmAreaError(f"unknown kinds {sorted(unknown)}; choose from "
                           f"{ASSET_KINDS}")
    if polygons and not any(_box_inside_ring(tile.bbox, ring)
                            for ring in polygons):
        filters = []
        for ring in polygons:
            if _ring_meets_box(ring, tile.bbox):
                points = " ".join(f"{_fmt(lat)} {_fmt(lon)}"
                                  for lon, lat in ring)
                filters.append(f'(poly:"{points}"){tile.filter()}')
    else:
        filters = [tile.filter()]
    linear = kinds & set(LINEAR_KINDS)
    areas = kinds & set(_AREA_KINDS)
    lines: list[str] = []
    for spatial in filters:
        if linear:
            lines.append(f"way[{_kind_regex(linear)}]{spatial};")
        if areas:
            regex = _kind_regex(areas)
            lines.append(f"node[{regex}]{spatial};")
            lines.append(f"way[{regex}]{spatial};")
            lines.append(f'relation["type"="multipolygon"][{regex}]'
                         f"{spatial};")
        if "tower" in kinds:
            lines.append(f"node[{_kind_regex(['tower'])}]{spatial};")
    return "(\n" + "\n".join(lines) + "\n);\nout body geom;\n"


def query_text(body: str, timeout_s: float | None) -> str:
    """The full query.  ``timeout_s=None`` gives the cache-key text."""

    timeout = "" if timeout_s is None else f"[timeout:{int(math.ceil(timeout_s))}]"
    return f"[out:json]{timeout};\n{body}"


def query_key(body: str) -> str:
    """sha256 of the endpoint- and timeout-independent query text."""

    return hashlib.sha256(query_text(body, None).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# cache


def _tile_paths(key: str) -> tuple[Path, Path]:
    folder = cache_root() / "tiles" / key[:2]
    return folder / f"{key}.json", folder / f"{key}.meta.json"


def _read_meta(key: str) -> dict | None:
    _body, meta_path = _tile_paths(key)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or meta.get("schema") != CACHE_META_SCHEMA \
            or meta.get("key") != key:
        return None
    return meta


def _read_cached(key: str, meta: dict | None = None
                 ) -> tuple[dict, dict] | None:
    """``(response, meta)`` of a complete cached tile, else ``None``.

    ``meta`` is the sidecar when the caller has already read it.
    """

    if meta is None:
        meta = _read_meta(key)
    if meta is None or meta.get("status") != "complete":
        return None
    body_path, _meta_path = _tile_paths(key)
    try:
        raw = body_path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != meta.get("body_sha256"):
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict) or not isinstance(
            document.get("elements"), list):
        return None
    return document, meta


def _write_meta(key: str, meta: Mapping[str, Any]) -> None:
    from woof.fetch_guard import atomic_write_text

    _body_path, meta_path = _tile_paths(key)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(meta_path, json.dumps(meta, indent=1) + "\n")


def _store(key: str, raw: bytes, meta: Mapping[str, Any]) -> None:
    from woof.fetch_guard import atomic_write_bytes

    body_path, _meta_path = _tile_paths(key)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(body_path, raw)
    _write_meta(key, meta)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# network


class _Budget:
    """A wall-clock deadline every request and every retry wait respects."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + float(seconds)
        self.seconds = float(seconds)

    def remaining(self) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0.0:
            raise _OutOfTime()
        return left

    def pause(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        if seconds >= self.remaining():
            raise _OutOfTime()
        from woof import fetch_pool
        fetch_pool.sleep_unless_stopped(seconds, sleep=time.sleep)


_HOST_LOCK = threading.Lock()
_LAST_REQUEST: dict[str, float] = {}


def _space_host(host: str, budget: _Budget) -> None:
    """Hold requests to one host :data:`MIN_REQUEST_GAP_S` apart."""

    with _HOST_LOCK:
        last = _LAST_REQUEST.get(host)
        now = time.monotonic()
        wait = 0.0 if last is None else last + MIN_REQUEST_GAP_S - now
        _LAST_REQUEST[host] = now + max(wait, 0.0)
    if wait > 0.0:
        budget.pause(wait)


def _open_url(request: Request, timeout: float):
    """The one urlopen this module makes (tests replace it)."""

    from woof.nomads_governor import paced_urlopen

    return paced_urlopen(request, timeout=timeout)


_BUSY_MARKERS = ("rate_limited", "Dispatcher_Client", "too many requests",
                 "Too Many Requests")
_TOO_LARGE_MARKERS = ("timed out", "out of memory", "run out of memory")


def _verify(raw: bytes) -> dict:
    """The parsed response, or why it cannot be used.

    A server-side give-up on the query raises :class:`_TooLarge`; the
    server's busy page :class:`_Busy`; anything else unusable a
    ``ValueError`` (the ladder asks the next endpoint).
    """

    text = raw.decode("utf-8", errors="replace")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if "runtime error" in lowered and any(
                m in lowered for m in _TOO_LARGE_MARKERS) \
                and "dispatcher" not in lowered:
            raise _TooLarge(_error_excerpt(text)) from None
        if any(marker in text for marker in _BUSY_MARKERS):
            raise _Busy(f"the server is busy: {_error_excerpt(text)}") from None
        raise ValueError(f"the answer is not JSON: {_error_excerpt(text)}") \
            from None
    if not isinstance(document, dict) or not isinstance(
            document.get("elements"), list):
        raise ValueError("the answer has no elements list")
    remark = str(document.get("remark") or "")
    if "runtime error" in remark:
        if any(m in remark.lower() for m in _TOO_LARGE_MARKERS) \
                and "Dispatcher" not in remark:
            raise _TooLarge(remark)
        if any(marker in remark for marker in _BUSY_MARKERS):
            raise _Busy(f"the server is busy: {remark}")
        raise ValueError(f"the server reported {remark!r}")
    return document


def _error_excerpt(text: str) -> str:
    match = re.search(r"Error</strong>:\s*(.*?)\s*</p>", text, re.S)
    found = match.group(1) if match else re.sub(r"<[^>]+>", " ", text)
    return " ".join(found.split())[:300]


def _endpoint(url: str):
    from woof.fetch_endpoints import Endpoint

    return Endpoint(name=urlsplit(url).netloc or url, base=url,
                    retention_hours=None,
                    why="a public Overpass API interpreter")


def endpoint_ladder(first: str | None = None) -> tuple[str, ...]:
    """``first`` (when given) then :data:`ENDPOINTS`, without repeats."""

    order: list[str] = []
    for url in ((first,) if first else ()) + ENDPOINTS:
        url = url.strip()
        if url and url not in order:
            order.append(url)
    for url in order:
        if urlsplit(url).scheme not in ("http", "https"):
            raise OsmAreaError(f"endpoint {url!r} is not an http(s) URL")
    return tuple(order)


_READ_CHUNK = 1 << 20


def _read_within(response, seconds: float, budget: _Budget) -> bytes:
    """Read the whole body, but no longer than ``seconds`` (and the budget).

    A socket timeout only bounds each read's idle time; a server that
    trickles a large answer steadily would otherwise run past both.
    """

    deadline = time.monotonic() + seconds
    chunks: list[bytes] = []
    while True:
        chunk = response.read(_READ_CHUNK)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        budget.remaining()
        if time.monotonic() > deadline:
            raise TimeoutError(f"the answer took longer than {seconds:.0f} s "
                               "to arrive")


def _download(tile: Tile, body: str, *, endpoints: Sequence[str],
              timeout_s: float, budget: _Budget) -> tuple[bytes, dict, str]:
    """``(raw, document, endpoint)`` for one tile, along the ladder."""

    from woof.fetch_endpoints import (
        FALLTHROUGH_STATUSES, TransferRefusal, ask_along_ladder)

    payload = urlencode({"data": query_text(body, timeout_s)}).encode("ascii")

    def transfer(endpoint):
        _space_host(endpoint.host, budget)
        request = Request(endpoint.base, data=payload, method="POST", headers={
            "User-Agent": _user_agent(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"})
        per_request = max(1.0, min(timeout_s + HTTP_SLACK_S,
                                   budget.remaining()))
        try:
            with _open_url(request, per_request) as response:
                raw = _read_within(response, per_request, budget)
        except HTTPError as error:
            if error.code == 400:
                try:
                    detail = _error_excerpt(error.read().decode(
                        "utf-8", errors="replace"))
                except Exception:                      # noqa: BLE001
                    detail = str(error)
                raise OverpassQueryError(
                    f"{endpoint.base} rejected the query for tile "
                    f"{tile.label()}: {detail}") from error
            if error.code not in FALLTHROUGH_STATUSES and error.code != 408:
                # 401/405/406/413 and the like: this endpoint will not
                # serve the query; another one may.  Asked once.
                raise ValueError(f"HTTP {error.code} {error.reason}") \
                    from error
            raise
        return raw, _verify(raw)

    try:
        used, (raw, document) = ask_along_ladder(
            tuple(_endpoint(url) for url in endpoints), transfer,
            label="woof energy fetch", name=f"tile {tile.label()}",
            progress=_progress, pause=budget.pause)
    except TransferRefusal as error:
        raise OverpassUnavailable(str(error)) from error
    except _OutOfTime as error:
        raise OverpassUnavailable(
            f"the {budget.seconds:g} s wall-clock budget (${BUDGET_ENV}) was "
            f"spent before tile {tile.label()} was served") from error
    return raw, document, used.base


# --------------------------------------------------------------------------
# tiles: cache walk and fetch


@dataclass
class TileResult:
    tile: Tile
    key: str
    document: dict
    meta: dict
    cached: bool


@dataclass
class FetchReport:
    """What one fetch did, beyond the assets it produced."""

    tiles: list[TileResult]
    splits: int = 0
    dropped_below_min_voltage: int = 0
    kept_without_voltage: int = 0
    centroid_fallbacks: int = 0
    skipped: dict | None = None
    notes: list[str] | None = None

    @property
    def tiles_cached(self) -> int:
        return sum(1 for t in self.tiles if t.cached)

    @property
    def tiles_fetched(self) -> int:
        return sum(1 for t in self.tiles if not t.cached)


def _quadrants_in(tile: Tile, area: Area) -> list[Tile]:
    """``tile``'s quadrants that still meet the area."""

    return [q for q in tile.quadrants()
            if not area.polygons
            or any(_ring_meets_box(ring, q.bbox) for ring in area.polygons)]


def _gather(tiles: Sequence[Tile], area: Area, kinds: Sequence[str], *,
            endpoints: Sequence[str], refresh: bool, offline: bool,
            timeout_s: float, budget: _Budget | None) -> tuple[list[TileResult],
                                                              int]:
    results: list[TileResult] = []
    missing: list[tuple[tuple, str]] = []
    splits = 0
    pending = list(tiles)
    while pending:
        tile = pending.pop(0)
        body = query_body(tile, kinds, area.polygons)
        key = query_key(body)
        meta = _read_meta(key)
        if meta is not None and meta.get("status") == "split" and (
                offline or not (refresh or timeout_s > float(
                    meta.get("timeout_s") or 0.0))):
            # The server gave up on this tile before, at this timeout or a
            # longer one.  ``refresh`` or a longer timeout asks again, so a
            # split caused by a busy server is not kept for good.
            pending[:0] = _quadrants_in(tile, area)
            splits += 1
            continue
        cached = None if refresh else _read_cached(key, meta)
        if cached is not None:
            results.append(TileResult(tile, key, cached[0], cached[1], True))
            continue
        if offline:
            missing.append((tile.bbox, key))
            continue
        assert budget is not None
        try:
            raw, document, used = _download(tile, body, endpoints=endpoints,
                                            timeout_s=timeout_s, budget=budget)
        except _TooLarge as error:
            if max(tile.east - tile.west, tile.north - tile.south) / 2.0 \
                    < MIN_TILE_DEG - 1e-12:
                raise OverpassTileTooLarge(
                    f"tile {tile.label()} still exceeds the server's limits "
                    f"at the smallest split ({MIN_TILE_DEG:g} deg): {error}. "
                    "Raise --timeout-s or narrow --kinds.") from error
            _progress(f"woof energy fetch: tile {tile.label()} is too dense "
                      f"for one query ({error}); splitting into quadrants")
            _write_meta(key, {"schema": CACHE_META_SCHEMA, "key": key,
                              "status": "split", "bbox": list(tile.bbox),
                              "reason": str(error),
                              "timeout_s": timeout_s,
                              "retrieved_utc": _utc_now()})
            pending[:0] = _quadrants_in(tile, area)
            splits += 1
            continue
        osm3s = document.get("osm3s") or {}
        meta = {"schema": CACHE_META_SCHEMA, "key": key, "status": "complete",
                "bbox": list(tile.bbox), "retrieved_utc": _utc_now(),
                "endpoint": used,
                "osm_base": osm3s.get("timestamp_osm_base"),
                "elements": len(document["elements"]),
                "bytes": len(raw),
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "query": query_text(body, None)}
        _store(key, raw, meta)
        results.append(TileResult(tile, key, document, meta, False))
    if missing:
        listed = "\n".join(f"  {','.join(_fmt(v) for v in bbox)}  key {key}"
                           for bbox, key in missing)
        raise OverpassCacheMiss(
            f"--offline: {len(missing)} of the area's tiles have no cached "
            f"Overpass response under {cache_root()} (W,S,E,N):\n{listed}\n"
            "  remedy: run once without --offline to fill the cache, or set "
            f"${CACHE_ENV} to the cache that holds them.", missing)
    return results, splits


# --------------------------------------------------------------------------
# tag parsing


_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*$")


def parse_voltage_kv(text: str | None) -> tuple[tuple[float, ...], bool]:
    """``("400000;132000")`` -> ``((400.0, 132.0), True)``.

    Values are volts unless written with ``V``/``kV``.  The flag is False
    when any part did not parse; the parsed parts are still returned, and
    the caller keeps the raw text.  Duplicates collapse, highest first.
    """

    if text is None:
        return (), True
    values: list[float] = []
    clean = True
    for part in str(text).split(";"):
        match = _NUMBER.match(part)
        if not match:
            clean = False
            continue
        number, unit = float(match.group(1)), match.group(2).lower()
        if unit in ("", "v"):
            kv = number / 1000.0
        elif unit == "kv":
            kv = number
        else:
            clean = False
            continue
        if kv <= 0.0:
            clean = False
            continue
        if kv not in values:
            values.append(kv)
    return tuple(sorted(values, reverse=True)), clean


_POWER_UNITS = {"w": 1e-6, "kw": 1e-3, "mw": 1.0, "gw": 1e3}


def parse_capacity_mw(text: str | None) -> float | None:
    """``"3 MW"`` -> 3.0, ``"500 kW"`` -> 0.5; ``"yes"``, a bare number or
    anything else -> ``None`` (the caller keeps the raw text).

    Units are W/kW/MW/GW, case-insensitive, with an optional peak ``p``
    (``"5 MWp"``, how solar output is tagged).
    """

    if text is None:
        return None
    match = _NUMBER.match(str(text))
    if not match:
        return None
    unit = match.group(2).lower()
    if unit.endswith("p"):
        unit = unit[:-1]
    scale = _POWER_UNITS.get(unit)
    if scale is None:
        return None
    value = float(match.group(1)) * scale
    return value if value > 0.0 else None


def parse_metres(text: str | None) -> float | None:
    """``"80"``/``"80 m"`` -> 80.0; other units or lists -> ``None``."""

    if text is None:
        return None
    match = _NUMBER.match(str(text))
    if not match or match.group(2).lower() not in ("", "m"):
        return None
    value = float(match.group(1))
    return value if value > 0.0 else None


def parse_positive_int(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped.isdigit():
        return None
    value = int(stripped)
    return value if value >= 1 else None


def parse_frequency_hz(text: str | None) -> float | None:
    """A single positive frequency; ``"0"`` (DC) and lists -> ``None``."""

    if text is None:
        return None
    match = _NUMBER.match(str(text))
    if not match or match.group(2).lower() not in ("", "hz"):
        return None
    value = float(match.group(1))
    return value if value > 0.0 else None


def map_generator_source(text: str | None) -> tuple[str | None, bool]:
    """``(vocabulary value, exact)``; unknown values map to ``"other"``."""

    if text is None:
        return None, True
    value = str(text).strip().lower()
    if value in GENERATOR_SOURCES:
        return value, True
    if value in SOURCE_ALIASES:
        return SOURCE_ALIASES[value], False
    return "other", False


# --------------------------------------------------------------------------
# geometry


def _lonlat(entry: Any) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    try:
        lon, lat = float(entry["lon"]), float(entry["lat"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    return lon, lat


def _way_coords(element: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    geometry = element.get("geometry")
    if not isinstance(geometry, list) or not geometry:
        return None
    coords = [_lonlat(entry) for entry in geometry]
    if any(c is None for c in coords):
        return None
    return coords  # type: ignore[return-value]


def _signed_area(ring: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _oriented(ring: list[tuple[float, float]], ccw: bool
              ) -> list[list[float]]:
    if (_signed_area(ring) > 0.0) != ccw:
        ring = ring[::-1]
    return [[lon, lat] for lon, lat in ring]


def _closed(ring: Sequence[tuple[float, float]]) -> bool:
    return len(ring) >= 4 and ring[0] == ring[-1]


def assemble_rings(segments: Sequence[Sequence[tuple[float, float]]]
                   ) -> list[list[tuple[float, float]]] | None:
    """Join way segments end to end into closed rings, or ``None``.

    A segment may join reversed.  Every segment must end up in a closed
    ring of at least four positions; one left dangling fails the lot.
    """

    pending = [list(s) for s in segments if s]
    rings: list[list[tuple[float, float]]] = []
    while pending:
        ring = pending.pop(0)
        while ring[0] != ring[-1]:
            for index, segment in enumerate(pending):
                if segment[0] == ring[-1]:
                    ring.extend(segment[1:])
                    break
                if segment[-1] == ring[-1]:
                    ring.extend(segment[-2::-1])
                    break
            else:
                return None
            pending.pop(index)
        if not _closed(ring):
            return None
        rings.append(ring)
    return rings


def _probe_point(inner: Sequence[tuple[float, float]],
                 outer: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """A point of ``inner`` to test against ``outer``.

    An inner ring may touch its outer at shared vertices, and a point on
    the boundary tests either way, so the first inner vertex the outer ring
    does not share is used, else the midpoint of an inner edge.
    """

    shared = set(outer)
    for point in inner:
        if point not in shared:
            return point
    (x1, y1), (x2, y2) = inner[0], inner[1]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _inside(point: tuple[float, float],
            ring: Sequence[tuple[float, float]]) -> bool:
    return _point_in_ring(point[0], point[1], ring)


def multipolygon_geometry(members: Sequence[Mapping[str, Any]]
                          ) -> dict | None:
    """A GeoJSON MultiPolygon from relation members with geometry, or None."""

    outer_segments: list[list[tuple[float, float]]] = []
    inner_segments: list[list[tuple[float, float]]] = []
    for member in members:
        if member.get("type") != "way":
            continue
        role = member.get("role") or "outer"
        if role not in ("outer", "inner"):
            continue
        coords = _way_coords(member)
        if coords is None or len(coords) < 2:
            return None
        (outer_segments if role == "outer" else inner_segments).append(coords)
    if not outer_segments:
        return None
    outers = assemble_rings(outer_segments)
    inners = assemble_rings(inner_segments) if inner_segments else []
    if outers is None or inners is None:
        return None
    holes: list[list[list[tuple[float, float]]]] = [[] for _ in outers]
    for inner in inners:
        owners = [i for i, outer in enumerate(outers)
                  if _inside(_probe_point(inner, outer), outer)]
        if not owners:
            return None
        owner = min(owners, key=lambda i: abs(_signed_area(outers[i])))
        holes[owner].append(inner)
    return {"type": "MultiPolygon",
            "coordinates": [[_oriented(outer, True)]
                            + [_oriented(h, False) for h in holes[i]]
                            for i, outer in enumerate(outers)]}


def _centroid(element: Mapping[str, Any]) -> tuple[float, float] | None:
    points: set[tuple[float, float]] = set()
    for member in element.get("members") or []:
        if member.get("type") == "node":
            point = _lonlat(member)
            if point is not None:
                points.add(point)
        for entry in member.get("geometry") or []:
            point = _lonlat(entry)
            if point is not None:
                points.add(point)
    for entry in element.get("geometry") or []:
        point = _lonlat(entry)
        if point is not None:
            points.add(point)
    if points:
        ordered = sorted(points)
        return (sum(p[0] for p in ordered) / len(ordered),
                sum(p[1] for p in ordered) / len(ordered))
    bounds = element.get("bounds")
    if isinstance(bounds, Mapping):
        try:
            return ((float(bounds["minlon"]) + float(bounds["maxlon"])) / 2.0,
                    (float(bounds["minlat"]) + float(bounds["maxlat"])) / 2.0)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _point(lonlat: tuple[float, float]) -> dict:
    return {"type": "Point", "coordinates": [lonlat[0], lonlat[1]]}


def element_geometry(element: Mapping[str, Any], kind: str
                     ) -> tuple[dict | None, str | None]:
    """``(geometry, why_skipped)``; a centroid fallback is a Point plus the
    reason ``CENTROID_FALLBACK`` in the second slot."""

    etype = element.get("type")
    if kind == "tower":
        if etype != "node":
            return None, "tower that is not a node"
        point = _lonlat(element)
        return (_point(point), None) if point else (None, "no coordinates")
    if kind in LINEAR_KINDS:
        if etype != "way":
            return None, f"{kind} that is not a way"
        coords = _way_coords(element)
        if coords is None or len(set(coords)) < 2:
            return None, "way without usable geometry"
        return {"type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in coords]}, None
    # substation, plant, generator
    if etype == "node":
        point = _lonlat(element)
        return (_point(point), None) if point else (None, "no coordinates")
    if etype == "way":
        coords = _way_coords(element)
        if coords is None:
            return None, "way without usable geometry"
        nodes = element.get("nodes") or []
        closed = (nodes[0] == nodes[-1]) if len(nodes) >= 2 else \
            coords[0] == coords[-1]
        if closed and _closed(coords) and len(set(coords)) >= 3:
            return {"type": "Polygon",
                    "coordinates": [_oriented(coords, True)]}, None
        centre = _centroid(element)
        return ((_point(centre), CENTROID_FALLBACK) if centre
                else (None, "way without usable geometry"))
    if etype == "relation":
        geometry = multipolygon_geometry(element.get("members") or [])
        if geometry is not None:
            return geometry, None
        centre = _centroid(element)
        return ((_point(centre), CENTROID_FALLBACK) if centre
                else (None, "relation without usable geometry"))
    return None, f"unexpected element type {etype!r}"


# --------------------------------------------------------------------------
# normalization


def element_to_asset(element: Mapping[str, Any]
                     ) -> tuple[Asset | None, str | None]:
    """One Overpass element as an :class:`Asset`, or why it was skipped.

    The second slot is ``CENTROID_FALLBACK`` when the asset was built with
    a fallback Point.
    """

    raw_tags = element.get("tags") or {}
    tags = {str(k): str(v) for k, v in raw_tags.items()}
    kind = tags.pop("power", None)
    if kind not in ASSET_KINDS:
        return None, f"power={kind!r} is not an asset kind"
    etype = element.get("type")
    asset_id = f"osm:{etype}/{element.get('id')}"
    geometry, why = element_geometry(element, kind)
    if geometry is None:
        return None, why
    if etype == "relation":
        tags.pop("type", None)
    name = tags.pop("name", None)
    operator = tags.pop("operator", None)

    voltages: tuple[float, ...] = ()
    if "voltage" in tags:
        voltages, clean = parse_voltage_kv(tags["voltage"])
        if clean and voltages:
            tags.pop("voltage")
    circuits = parse_positive_int(tags.get("circuits"))
    if circuits is not None:
        tags.pop("circuits")
    cables = parse_positive_int(tags.get("cables"))
    if cables is not None:
        tags.pop("cables")
    frequency = parse_frequency_hz(tags.get("frequency"))
    if frequency is not None:
        tags.pop("frequency")

    source_keys = (("plant:source", "generator:source") if kind == "plant"
                   else ("generator:source", "plant:source"))
    generator_source = None
    present = [key for key in source_keys if key in tags]
    mapped = {key: map_generator_source(tags[key]) for key in present}
    exact = [key for key in present if mapped[key][1]]
    if exact:
        generator_source = mapped[exact[0]][0]
        tags.pop(exact[0])
    elif present:
        generator_source = mapped[present[0]][0]
    output_keys = (("plant:output:electricity",
                    "generator:output:electricity") if kind == "plant"
                   else ("generator:output:electricity",
                         "plant:output:electricity"))
    capacity = None
    for key in output_keys:
        if key in tags:
            capacity = parse_capacity_mw(tags[key])
            if capacity is not None:
                tags.pop(key)
                break
    hub = parse_metres(tags.get("height:hub"))
    if hub is not None:
        tags.pop("height:hub")
    rotor = parse_metres(tags.get("rotor:diameter"))
    if rotor is not None:
        tags.pop("rotor:diameter")
    if why == CENTROID_FALLBACK:
        tags[GEOMETRY_TAG] = CENTROID_FALLBACK
    asset = Asset(asset_id=asset_id, kind=kind, geometry=geometry,
                  source=SOURCE, source_ref=f"{etype}/{element.get('id')}",
                  name=name, operator=operator, voltage_kv=voltages,
                  circuits=circuits, cables=cables, frequency_hz=frequency,
                  generator_source=generator_source, capacity_mw=capacity,
                  hub_height_m=hub, rotor_diameter_m=rotor, license=LICENSE,
                  tags=tags)
    return asset, why


_TYPE_ORDER = {"node": 0, "way": 1, "relation": 2}


def normalize(documents: Iterable[Mapping[str, Any]], kinds: Sequence[str],
              min_voltage_kv: float | None = None
              ) -> tuple[list[Asset], dict[str, Any]]:
    """Assets from Overpass responses, deduplicated by ``asset_id``.

    The first response to carry an element wins.  Returns the assets in
    (type, id) order and the counts :class:`FetchReport` keeps.
    """

    wanted = set(kinds)
    elements: dict[tuple[str, int], Mapping[str, Any]] = {}
    for document in documents:
        for element in document.get("elements") or []:
            key = (str(element.get("type")), int(element.get("id", -1)))
            elements.setdefault(key, element)
    assets: list[Asset] = []
    skipped: dict[str, int] = {}
    counts = {"dropped_below_min_voltage": 0, "kept_without_voltage": 0,
              "centroid_fallbacks": 0}
    for key in sorted(elements, key=lambda k: (_TYPE_ORDER.get(k[0], 9), k[1])):
        element = elements[key]
        power = (element.get("tags") or {}).get("power")
        if power not in wanted:
            continue
        try:
            asset, why = element_to_asset(element)
        except ContractError as error:
            asset, why = None, f"contract refused it ({error})"
        if asset is None:
            reason = why or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if why == CENTROID_FALLBACK:
            counts["centroid_fallbacks"] += 1
        if min_voltage_kv is not None and asset.kind in _VOLTAGE_KINDS:
            if asset.max_voltage_kv is None:
                counts["kept_without_voltage"] += 1
            elif asset.max_voltage_kv < min_voltage_kv:
                counts["dropped_below_min_voltage"] += 1
                continue
        assets.append(asset)
    counts["skipped"] = dict(sorted(skipped.items()))
    return assets, counts


# --------------------------------------------------------------------------
# entry points


def _check_kinds(kinds: Sequence[str]) -> tuple[str, ...]:
    if isinstance(kinds, str):
        kinds = tuple(k.strip() for k in kinds.split(",") if k.strip())
    kinds = tuple(dict.fromkeys(kinds))
    if not kinds:
        raise OsmAreaError("kinds must name at least one asset kind")
    unknown = [k for k in kinds if k not in ASSET_KINDS]
    if unknown:
        raise OsmAreaError(f"unknown kinds {unknown}; choose from "
                           f"{ASSET_KINDS}")
    return kinds


def fetch_with_report(*, bbox: tuple[float, float, float, float] | None = None,
                      polygon: Path | None = None,
                      kinds: Sequence[str] = DEFAULT_KINDS,
                      min_voltage_kv: float | None = None,
                      endpoint: str | None = None, refresh: bool = False,
                      offline: bool = False, timeout_s: float = 180.0
                      ) -> tuple[AssetCollection, FetchReport]:
    """:func:`fetch_assets`, plus what the fetch did (tiles, counts)."""

    if (bbox is None) == (polygon is None):
        raise OsmAreaError("give exactly one of bbox or polygon")
    if refresh and offline:
        raise OsmAreaError("refresh and offline contradict each other")
    if min_voltage_kv is not None and not (
            math.isfinite(float(min_voltage_kv)) and min_voltage_kv > 0.0):
        raise OsmAreaError(f"min_voltage_kv must be positive, got "
                           f"{min_voltage_kv!r}")
    if not (math.isfinite(float(timeout_s)) and timeout_s > 0.0):
        raise OsmAreaError(f"timeout_s must be positive, got {timeout_s!r}")
    kinds = _check_kinds(kinds)
    area = (load_polygon_area(polygon) if polygon is not None
            else Area(bbox=_check_bbox(bbox)))
    endpoints = endpoint_ladder(endpoint)
    tiles = tiles_for(area)
    budget = None if offline else _Budget(_budget_seconds())
    results, splits = _gather(tiles, area, kinds, endpoints=endpoints,
                              refresh=refresh, offline=offline,
                              timeout_s=float(timeout_s), budget=budget)
    assets, counts = normalize((r.document for r in results), kinds,
                               min_voltage_kv)
    notes = list(area.notes)
    if splits:
        notes.append(f"{splits} tile(s) were split into quadrants because the "
                     "server could not finish them in one query")
    if counts["centroid_fallbacks"]:
        notes.append(f"{counts['centroid_fallbacks']} asset(s) could not be "
                     "assembled into polygons and are centroid Points tagged "
                     f"{GEOMETRY_TAG}={CENTROID_FALLBACK}")
    if counts["skipped"]:
        notes.append("elements skipped: " + ", ".join(
            f"{reason} ({n})" for reason, n in counts["skipped"].items()))
    if min_voltage_kv is not None:
        notes.append(
            f"min_voltage_kv={min_voltage_kv:g}: dropped "
            f"{counts['dropped_below_min_voltage']} line/cable/substation "
            f"asset(s) below it and kept {counts['kept_without_voltage']} "
            "with no parsable voltage")
    retrieved = sorted(r.meta.get("retrieved_utc") for r in results
                       if r.meta.get("retrieved_utc"))
    bases = sorted(r.meta.get("osm_base") for r in results
                   if r.meta.get("osm_base"))
    record = {
        "source": SOURCE,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "url": COPYRIGHT_URL,
        "endpoints": sorted({r.meta.get("endpoint") for r in results
                             if r.meta.get("endpoint")}),
        "retrieved_utc": retrieved[0] if retrieved else None,
        "retrieved_utc_last": retrieved[-1] if retrieved else None,
        "osm_base": bases[0] if bases else None,
        "osm_base_last": bases[-1] if bases else None,
        "query": {"area": area.describe(), "kinds": list(kinds),
                  "min_voltage_kv": min_voltage_kv,
                  "tiles": [r.key for r in results]},
        "notes": notes,
    }
    collection = AssetCollection(assets=assets, sources=[record])
    report = FetchReport(tiles=results, splits=splits,
                         dropped_below_min_voltage=counts[
                             "dropped_below_min_voltage"],
                         kept_without_voltage=counts["kept_without_voltage"],
                         centroid_fallbacks=counts["centroid_fallbacks"],
                         skipped=counts["skipped"], notes=notes)
    return collection, report


def fetch_assets(*, bbox: tuple[float, float, float, float] | None = None,
                 polygon: Path | None = None,
                 kinds: Sequence[str] = DEFAULT_KINDS,
                 min_voltage_kv: float | None = None,
                 endpoint: str | None = None, refresh: bool = False,
                 offline: bool = False, timeout_s: float = 180.0
                 ) -> AssetCollection:
    """Fetch OSM power assets.  ``bbox`` is ``(west, south, east, north)``.

    Raises :class:`OverpassUnavailable` when no endpoint serves a tile,
    :class:`OverpassCacheMiss` when ``offline`` and a tile is not cached,
    :class:`OverpassTileTooLarge` when a tile cannot be split small enough,
    and :class:`OsmAreaError` for an unusable area or argument.
    """

    collection, _report = fetch_with_report(
        bbox=bbox, polygon=polygon, kinds=kinds,
        min_voltage_kv=min_voltage_kv, endpoint=endpoint, refresh=refresh,
        offline=offline, timeout_s=timeout_s)
    return collection


def main(args) -> int:
    polygon = Path(args.polygon) if getattr(args, "polygon", None) else None
    try:
        collection, report = fetch_with_report(
            bbox=getattr(args, "bbox", None), polygon=polygon,
            kinds=args.kinds, min_voltage_kv=args.min_voltage_kv,
            endpoint=args.endpoint, refresh=args.refresh,
            offline=args.offline, timeout_s=args.timeout_s)
    except (OverpassUnavailable, OverpassCacheMiss, OverpassTileTooLarge,
            OverpassQueryError, OsmAreaError, ContractError) as error:
        print(f"woof energy fetch: {error}", file=sys.stderr)
        return 2
    output = dump_assets(collection, args.output)
    by_kind: dict[str, int] = {}
    for asset in collection.assets:
        by_kind[asset.kind] = by_kind.get(asset.kind, 0) + 1
    record = collection.sources[0]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "output": str(output),
        "assets": len(collection),
        "by_kind": dict(sorted(by_kind.items())),
        "with_voltage": sum(1 for a in collection.assets if a.voltage_kv),
        "tiles": {"total": len(report.tiles), "fetched": report.tiles_fetched,
                  "cached": report.tiles_cached, "split": report.splits},
        "endpoints": record["endpoints"],
        "osm_base": record["osm_base"],
        "cache": str(cache_root()),
        "min_voltage_kv": args.min_voltage_kv,
        "dropped_below_min_voltage": report.dropped_below_min_voltage,
        "kept_without_voltage": report.kept_without_voltage,
        "centroid_fallbacks": report.centroid_fallbacks,
        "skipped": report.skipped,
        "notes": report.notes,
        "attribution": ATTRIBUTION,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = [
    "SOURCE", "ATTRIBUTION", "LICENSE", "CACHE_ENV", "BUDGET_ENV", "ENDPOINTS",
    "MIN_TILE_DEG", "MAX_TILE_DEG", "MAX_POLYGON_VERTICES",
    "OverpassUnavailable", "OverpassCacheMiss", "OverpassTileTooLarge",
    "OverpassQueryError", "OsmAreaError",
    "Area", "Tile", "TileResult", "FetchReport",
    "cache_root", "tiles_for", "load_polygon_area", "query_body",
    "query_text", "query_key", "endpoint_ladder",
    "parse_voltage_kv", "parse_capacity_mw", "parse_metres",
    "parse_positive_int", "parse_frequency_hz", "map_generator_source",
    "assemble_rings", "multipolygon_geometry", "element_geometry",
    "element_to_asset", "normalize", "fetch_with_report", "fetch_assets",
    "main",
]
