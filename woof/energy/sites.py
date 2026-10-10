"""Assets -> forecast sites.

``woof energy sites`` turns a ``woof-energy.assets.v1`` document into the
``woof-energy.sites.v1`` points a forecast is sampled at:

``line_sample``
    Lines, minor lines and cables (MultiLineString split into parts) are
    densified every ``--spacing-m`` by :func:`woof.energy.geometry.
    densify_polyline`; each sample carries the local conductor azimuth from
    :func:`woof.energy.geometry.segment_bearings` (normalised to [0, 360)),
    its chainage along the part and the asset's highest voltage.  The site
    id is ``f"{asset_id}@{part}:{chainage_m:.0f}"``; a rounding collision
    gets a ``~n`` suffix.
``tower``
    Only with ``--include-towers``.  A tower borrows the bearing (and, when
    it has no voltage of its own, the voltage) of the nearest densified
    line segment within :data:`TOWER_LINE_RADIUS_M`; with none in reach both
    stay null.  The search includes lines dropped by ``--min-voltage-kv``:
    a tower whose nearest line is below the threshold is dropped with it.
``substation``
    One site at the point, or at the area-weighted centroid of the polygon
    computed in :func:`woof.energy.geometry.local_projection`.
``turbine``
    A wind generator Point, with the asset's ``hub_height_m`` and
    ``capacity_mw``.  A missing hub height is counted in the provenance and
    left null; none is invented.
``pv``
    A solar generator or plant: its centroid, or with ``--pv-grid-m`` the
    points of a square grid of that spacing that fall inside the polygon
    (:func:`woof.energy.geometry.point_in_polygon`), the asset capacity
    split evenly across them.  A polygon too small to hold one grid point
    falls back to its centroid, and that fallback is counted.  A grid of
    more than :data:`MAX_PV_GRID_POINTS` points for one asset is refused.
``plant``
    Every other plant or generator, at its centroid.

A plant with a ``generator_source`` that has kept, in-region generators of
the same source inside its polygon, or naming it through ``tags['plant']``,
gets no site of its own: its members stand for it and its capacity is not
counted twice.  For wind this is the "wind plant with turbines" rule; the
same rule keeps a solar plant from doubling the capacity of its panels.

Filters run in this order: ``kinds`` (asset kinds); ``min_voltage_kv``
(lines, cables and substations, plus towers through their nearest line; an
asset with no voltage is kept and counted in ``kept_without_voltage``);
``region``; then the member-based decisions above and the near-duplicate
filter, so a site the region removes never justifies dropping another.  A
line sample is a near-duplicate when an already kept sample of a
*different* asset lies within ``spacing_m / 4`` and runs within
:data:`DEDUPE_MAX_ANGLE_DEG` of parallel (duplicated parallel OSM ways, a
line continuing through a junction); lines that cross are kept.  Every drop
is counted by reason in ``provenance["drops"]``.  A result with no sites is
refused (:class:`SitesRefused`) with that breakdown.

``heights_m`` is the sorted union of the requested heights and the distinct
hub heights (rounded to 0.1 m) of the turbines that survive the filters.

Python boundary (``docs/dev/static-rust-port.md``): this is per-site
bookkeeping on small vectors (a national grid at 100 m is a few hundred
thousand points), done with numpy and one scipy KD-tree; no gridded model
field is touched here, so nothing belongs in Rust.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woof.energy import geometry
from woof.energy.contracts import (
    LINEAR_KINDS,
    Asset,
    AssetCollection,
    ContractError,
    Site,
    SiteSet,
    dump_sites,
    file_ref,
    load_assets,
    validate_geometry,
)
from woof.static.projection import EARTH_RADIUS_M

#: A tower takes its bearing from a densified line no further than this.
TOWER_LINE_RADIUS_M = 50.0

#: Near-duplicate radius as a fraction of ``spacing_m``.
DEDUPE_FRACTION = 0.25

#: Two samples are near-duplicates only when their lines run within this
#: angle of parallel (direction-agnostic).
DEDUPE_MAX_ANGLE_DEG = 20.0

#: Refuse a ``--pv-grid-m`` grid with more points than this for one asset.
MAX_PV_GRID_POINTS = 100_000

_VOLTAGE_KINDS = LINEAR_KINDS + ("substation",)

Polygons = list[list[list[tuple[float, float]]]]


class SitesRefused(ValueError):
    """``woof energy sites`` cannot produce a usable site set."""


# --------------------------------------------------------------------------
# small geometry helpers (per-site vectors)


def _xyz(lon, lat) -> np.ndarray:
    """Points on the WRF sphere in metres, for chord-distance KD-trees.

    The chord between two points within a few kilometres differs from the
    arc by well under a millimetre, so chord radii stand for distances.
    """

    lon_r = np.radians(np.asarray(lon, dtype=np.float64))
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_r)
    return EARTH_RADIUS_M * np.column_stack(
        (cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)))


def _forward(plane, lon, lat) -> tuple[np.ndarray, np.ndarray]:
    x, y = plane.forward(np.asarray(lon, dtype=np.float64),
                         np.asarray(lat, dtype=np.float64))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _inverse(plane, x, y) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = plane.inverse(np.asarray(x, dtype=np.float64),
                             np.asarray(y, dtype=np.float64))
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _polygons_of(geom: Mapping[str, Any], counters: Counter | None = None
                 ) -> Polygons:
    """GeoJSON Polygon/MultiPolygon -> list of polygons (closed rings).

    The asset contract checks ring closure for a Polygon but not inside a
    MultiPolygon; an open ring is closed here and counted.
    """

    kind = geom["type"]
    coords = geom["coordinates"]
    raw = [coords] if kind == "Polygon" else list(coords)
    polygons: Polygons = []
    for polygon in raw:
        rings = []
        for ring in polygon:
            points = [(float(p[0]), float(p[1])) for p in ring]
            if points and points[0] != points[-1]:
                points.append(points[0])
                if counters is not None:
                    counters["open_ring_closed"] += 1
            rings.append(points)
        polygons.append(rings)
    return polygons


def _ring_area_centroid(x: np.ndarray, y: np.ndarray
                        ) -> tuple[float, float, float]:
    """Signed area and centroid of a closed ring (shoelace)."""

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * float(cross.sum())
    if area == 0.0:
        return 0.0, float(x[:-1].mean()), float(y[:-1].mean())
    cx = float(((x[:-1] + x[1:]) * cross).sum()) / (6.0 * area)
    cy = float(((y[:-1] + y[1:]) * cross).sum()) / (6.0 * area)
    return area, cx, cy


def _centroid(geom: Mapping[str, Any], counters: Counter | None = None
              ) -> tuple[float, float, bool]:
    """``(lon, lat, degenerate)`` of a Point or area-weighted polygon centroid.

    Holes subtract.  ``degenerate`` is True when the polygon encloses no
    area and the vertex mean was used instead.
    """

    if geom["type"] == "Point":
        return float(geom["coordinates"][0]), float(geom["coordinates"][1]), \
            False
    polygons = _polygons_of(geom, counters)
    first = polygons[0][0][0]
    plane = geometry.local_projection(first[1], first[0])
    total = sx = sy = 0.0
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for polygon in polygons:
        for index, ring in enumerate(polygon):
            x, y = _forward(plane, [p[0] for p in ring], [p[1] for p in ring])
            xs.append(x[:-1])
            ys.append(y[:-1])
            area, cx, cy = _ring_area_centroid(x, y)
            weight = abs(area) if index == 0 else -abs(area)
            total += weight
            sx += weight * cx
            sy += weight * cy
    if total <= 0.0:
        lon, lat = _inverse(plane, [np.concatenate(xs).mean()],
                            [np.concatenate(ys).mean()])
        return float(lon[0]), float(lat[0]), True
    lon, lat = _inverse(plane, [sx / total], [sy / total])
    return float(lon[0]), float(lat[0]), False


def _segment_distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from the origin to segment a-b, metres."""

    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, -(ax * dx + ay * dy)
                                                / length2))
    return math.hypot(ax + t * dx, ay + t * dy)


def _normalise_bearing(value: float) -> float:
    bearing = float(value) % 360.0
    return 0.0 if bearing >= 360.0 else bearing


def _axis_difference(a: np.ndarray | float, b: np.ndarray | float):
    """Angle between two line directions, ignoring sense, in [0, 90]."""

    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


# --------------------------------------------------------------------------
# region


def load_region(path: str | Path) -> Polygons:
    """Read a GeoJSON Polygon/MultiPolygon (bare, Feature or
    FeatureCollection) into the polygon list :mod:`woof.energy.geometry`
    takes, keeping holes with their outer ring.

    :func:`woof.domain_wizard.load_polygon_footprint` flattens every ring
    into one list (holes lose their owner) and words its errors for
    ``--polygon``, so the region is parsed here.
    """

    path = Path(path)
    if not path.is_file():
        raise SitesRefused(f"--region file does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SitesRefused(f"--region {path} is not readable GeoJSON: "
                           f"{error}") from error
    if not isinstance(document, dict):
        raise SitesRefused(f"--region {path} is not a GeoJSON object")

    geometries: list[Any] = []
    kind = document.get("type")
    if kind in ("Polygon", "MultiPolygon"):
        geometries.append(document)
    elif kind == "Feature":
        geometries.append(document.get("geometry"))
    elif kind == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list) or not features:
            raise SitesRefused(f"--region {path} FeatureCollection has no "
                               "features")
        for feature in features:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise SitesRefused(f"--region {path} holds a non-Feature "
                                   "member")
            geometries.append(feature.get("geometry"))
    else:
        raise SitesRefused(f"--region {path} type {kind!r} is not a Polygon, "
                           "MultiPolygon, Feature or FeatureCollection")

    polygons: Polygons = []
    for index, geom in enumerate(geometries):
        if not isinstance(geom, dict) or geom.get("type") not in (
                "Polygon", "MultiPolygon"):
            found = geom.get("type") if isinstance(geom, dict) else geom
            raise SitesRefused(f"--region {path} geometry {index} is "
                               f"{found!r}, not a Polygon/MultiPolygon")
        try:
            validate_geometry(geom, what=f"--region {path} geometry {index}")
            if geom["type"] == "MultiPolygon":
                for part, polygon in enumerate(geom["coordinates"]):
                    validate_geometry({"type": "Polygon",
                                       "coordinates": polygon},
                                      what=f"--region {path} geometry "
                                           f"{index} polygon {part}")
        except ContractError as error:
            raise SitesRefused(str(error)) from error
        polygons.extend(_polygons_of(geom))
    if not polygons:
        raise SitesRefused(f"--region {path} contains no polygons")
    return polygons


# --------------------------------------------------------------------------
# builder


@dataclass
class _Part:
    """One densified line part (kept, or dropped by voltage for towers)."""

    asset: Asset
    index: int
    lon: np.ndarray
    lat: np.ndarray
    chainage: np.ndarray
    bearing: np.ndarray
    kept: bool


class _Builder:
    def __init__(self, region: Polygons | None):
        self.region = region
        self.sites: list[dict[str, Any]] = []
        self.ids: set[str] = set()
        self.drops: Counter = Counter()
        self.counters: Counter = Counter()

    def inside(self, lon, lat) -> np.ndarray:
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        if self.region is None:
            return np.ones(lon.shape, dtype=bool)
        return np.asarray(geometry.point_in_polygon(lon, lat, self.region),
                          dtype=bool)

    def add(self, **record: Any) -> None:
        site_id = record["site_id"]
        candidate = site_id
        n = 1
        while candidate in self.ids:
            candidate = f"{site_id}~{n}"
            n += 1
        if candidate != site_id:
            self.counters["site_id_disambiguated"] += 1
        self.ids.add(candidate)
        record["site_id"] = candidate
        self.sites.append(record)

    def add_point_sites(self, records: list[dict[str, Any]]) -> None:
        """Region-filter and add non-line sites."""

        if not records:
            return
        inside = self.inside([r["lon"] for r in records],
                             [r["lat"] for r in records])
        self.drops["outside_region"] += int((~inside).sum())
        for record, ok in zip(records, inside):
            if ok:
                self.add(**record)


def _record(asset: Asset, kind: str, lon: float, lat: float, *,
            site_id: str | None = None, bearing_deg: float | None = None,
            chainage_m: float | None = None, voltage_kv: float | None = None,
            hub_height_m: float | None = None,
            capacity_mw: float | None = None) -> dict[str, Any]:
    return dict(site_id=site_id or asset.asset_id, asset_id=asset.asset_id,
                kind=kind, lon=float(lon), lat=float(lat),
                bearing_deg=bearing_deg, chainage_m=chainage_m,
                voltage_kv=voltage_kv, hub_height_m=hub_height_m,
                capacity_mw=capacity_mw)


def _is_turbine(asset: Asset) -> bool:
    return (asset.kind == "generator" and asset.generator_source == "wind"
            and asset.geometry["type"] == "Point")


def _line_parts(asset: Asset) -> list[list[tuple[float, float]]]:
    coords = asset.geometry["coordinates"]
    parts = [coords] if asset.geometry["type"] == "LineString" else coords
    return [[(float(p[0]), float(p[1])) for p in part] for part in parts]


def _pv_grid(asset: Asset, grid_m: float, counters: Counter
             ) -> tuple[np.ndarray, np.ndarray] | None:
    """Grid points of spacing ``grid_m`` inside the polygon, lattice
    anchored on the centroid; ``None`` when not one point falls inside."""

    lon0, lat0, _ = _centroid(asset.geometry, counters)
    polygons = _polygons_of(asset.geometry)
    plane = geometry.local_projection(lat0, lon0)
    xs, ys = [], []
    for polygon in polygons:
        x, y = _forward(plane, [p[0] for p in polygon[0]],
                        [p[1] for p in polygon[0]])
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    i0, i1 = math.floor(x_all.min() / grid_m), math.ceil(x_all.max() / grid_m)
    j0, j1 = math.floor(y_all.min() / grid_m), math.ceil(y_all.max() / grid_m)
    count = (i1 - i0 + 1) * (j1 - j0 + 1)
    if count > MAX_PV_GRID_POINTS:
        raise SitesRefused(
            f"--pv-grid-m {grid_m:g} would test {count:,} grid points for "
            f"{asset.asset_id} (limit {MAX_PV_GRID_POINTS:,}); choose a "
            "coarser spacing")
    gx, gy = np.meshgrid(np.arange(i0, i1 + 1) * grid_m,
                         np.arange(j0, j1 + 1) * grid_m)
    lon, lat = _inverse(plane, gx.ravel(), gy.ravel())
    inside = np.asarray(geometry.point_in_polygon(lon, lat, polygons),
                        dtype=bool)
    if not inside.any():
        return None
    return lon[inside], lat[inside]


def _densify(assets: Sequence[Asset], kept_ids: set[str], spacing_m: float,
             out: _Builder) -> list[_Part]:
    parts: list[_Part] = []
    for asset in assets:
        for part_index, part in enumerate(_line_parts(asset)):
            if len(part) < 2 or len(set(part)) < 2:
                if asset.asset_id in kept_ids:
                    out.drops["degenerate_line_part"] += 1
                continue
            lon, lat, chainage = (np.asarray(a, dtype=np.float64) for a in
                                  geometry.densify_polyline(part, spacing_m))
            bearing = np.asarray(geometry.segment_bearings(lon, lat),
                                 dtype=np.float64) % 360.0
            bearing[bearing >= 360.0] = 0.0
            parts.append(_Part(asset, part_index, lon, lat, chainage, bearing,
                               asset.asset_id in kept_ids))
    return parts


def _tower_match(tlon: float, tlat: float, candidates: Sequence[int],
                 parts: Sequence[_Part], part_of: np.ndarray,
                 offset: np.ndarray) -> tuple[_Part, float] | None:
    """Nearest densified segment within :data:`TOWER_LINE_RADIUS_M`."""

    plane = geometry.local_projection(tlat, tlon)
    best = TOWER_LINE_RADIUS_M
    match: tuple[_Part, float] | None = None
    for j in sorted(candidates):
        part = parts[int(part_of[j])]
        k = int(offset[j])
        for a, b in ((k - 1, k), (k, k + 1)):
            if a < 0 or b >= part.lon.size:
                continue
            x, y = _forward(plane, part.lon[[a, b]], part.lat[[a, b]])
            if x[0] == x[1] and y[0] == y[1]:
                continue
            distance = _segment_distance(x[0], y[0], x[1], y[1])
            if distance <= best:
                best = distance
                match = (part, _normalise_bearing(math.degrees(
                    math.atan2(x[1] - x[0], y[1] - y[0]))))
    return match


def build_sites(assets: AssetCollection, *, spacing_m: float = 100.0,
                kinds: Sequence[str] | None = None,
                min_voltage_kv: float | None = None,
                region: Sequence | None = None,
                heights_m: Sequence[float] = (10.0, 30.0, 100.0),
                include_towers: bool = False,
                pv_grid_m: float | None = None) -> SiteSet:
    """``region`` is a list of polygons as in :mod:`woof.energy.geometry`.

    The returned set has no ``assets_ref``; :func:`main` binds the file.
    Raises :class:`SitesRefused` when no site survives the filters.
    """

    if not (isinstance(spacing_m, (int, float)) and math.isfinite(spacing_m)
            and spacing_m > 0.0):
        raise SitesRefused(f"spacing_m must be a positive number, got "
                           f"{spacing_m!r}")
    if pv_grid_m is not None and not (math.isfinite(pv_grid_m)
                                      and pv_grid_m > 0.0):
        raise SitesRefused(f"pv_grid_m must be a positive number, got "
                           f"{pv_grid_m!r}")
    requested_heights = tuple(float(h) for h in heights_m)
    if not requested_heights or any(not math.isfinite(h) or h <= 0.0
                                    for h in requested_heights):
        raise SitesRefused("heights_m must name positive heights above "
                           f"ground, got {list(heights_m)!r}")

    spacing_m = float(spacing_m)
    out = _Builder(list(region) if region is not None else None)
    keep_kinds = set(kinds) if kinds is not None else None
    assets_by_kind = Counter(a.kind for a in assets.assets)
    kept_without_voltage: Counter = Counter()

    # ---- asset-level filters -------------------------------------------
    kept: list[Asset] = []
    low_voltage_lines: list[Asset] = []
    for asset in assets.assets:
        if keep_kinds is not None and asset.kind not in keep_kinds:
            out.drops["kind_filtered"] += 1
            continue
        if asset.kind == "tower" and not include_towers:
            out.drops["towers_not_requested"] += 1
            continue
        if asset.kind in _VOLTAGE_KINDS or asset.kind == "tower":
            top = asset.max_voltage_kv
            if top is None:
                if asset.kind != "tower":
                    kept_without_voltage[asset.kind] += 1
            elif min_voltage_kv is not None and top < min_voltage_kv:
                out.drops["below_min_voltage"] += 1
                if asset.kind in LINEAR_KINDS:
                    low_voltage_lines.append(asset)
                continue
        kept.append(asset)

    # ---- densify every line part (kept, and voltage-dropped for towers) --
    kept_line_ids = {a.asset_id for a in kept if a.kind in LINEAR_KINDS}
    towers = [a for a in kept if a.kind == "tower"]
    linear = [a for a in kept if a.kind in LINEAR_KINDS]
    if towers:
        linear += low_voltage_lines
    parts = _densify(linear, kept_line_ids, spacing_m, out)

    tree = None
    if parts:
        from scipy.spatial import cKDTree

        lon_all = np.concatenate([p.lon for p in parts])
        lat_all = np.concatenate([p.lat for p in parts])
        bearing_all = np.concatenate([p.bearing for p in parts])
        part_of = np.concatenate([np.full(p.lon.size, n)
                                  for n, p in enumerate(parts)])
        offset = np.concatenate([np.arange(p.lon.size) for p in parts])
        xyz = _xyz(lon_all, lat_all)
        tree = cKDTree(xyz)

    # ---- line samples: region, then near-duplicates ----------------------
    if parts:
        candidate = np.array([parts[int(n)].kept for n in part_of], dtype=bool)
        cand_index = np.flatnonzero(candidate)
        inside = out.inside(lon_all[cand_index], lat_all[cand_index])
        out.drops["outside_region"] += int((~inside).sum())
        cand_index = cand_index[inside]
        neighbours = tree.query_ball_point(xyz[cand_index],
                                           r=DEDUPE_FRACTION * spacing_m)
        accepted = np.zeros(lon_all.size, dtype=bool)
        owner = np.array([parts[int(n)].asset.asset_id for n in part_of],
                         dtype=object)
        for i, near in zip(cand_index, neighbours):
            near = np.asarray(near, dtype=np.int64)
            near = near[accepted[near] & (owner[near] != owner[i])]
            if near.size and (_axis_difference(bearing_all[near],
                                               bearing_all[i])
                              <= DEDUPE_MAX_ANGLE_DEG).any():
                out.drops["near_duplicate"] += 1
                continue
            accepted[i] = True
            part = parts[int(part_of[i])]
            k = int(offset[i])
            out.add(**_record(
                part.asset, "line_sample", lon_all[i], lat_all[i],
                site_id=f"{part.asset.asset_id}@{part.index}:"
                        f"{part.chainage[k]:.0f}",
                bearing_deg=float(bearing_all[i]),
                chainage_m=float(part.chainage[k]),
                voltage_kv=part.asset.max_voltage_kv))

    # ---- substations -----------------------------------------------------
    points: list[dict[str, Any]] = []
    for asset in kept:
        if asset.kind == "substation":
            lon, lat, degenerate = _centroid(asset.geometry, out.counters)
            out.counters["zero_area_centroid"] += int(degenerate)
            points.append(_record(asset, "substation", lon, lat,
                                  voltage_kv=asset.max_voltage_kv))

    # ---- towers -----------------------------------------------------------
    if towers:
        tower_lon = np.array([t.geometry["coordinates"][0] for t in towers],
                             dtype=np.float64)
        tower_lat = np.array([t.geometry["coordinates"][1] for t in towers],
                             dtype=np.float64)
        reach = TOWER_LINE_RADIUS_M + spacing_m
        nearby = (tree.query_ball_point(_xyz(tower_lon, tower_lat), r=reach)
                  if tree is not None else [[] for _ in towers])
        for asset, tlon, tlat, near in zip(towers, tower_lon, tower_lat,
                                           nearby):
            match = (_tower_match(float(tlon), float(tlat), near, parts,
                                  part_of, offset) if near else None)
            bearing: float | None = None
            voltage = asset.max_voltage_kv
            if match is not None:
                part, bearing = match
                if not part.kept:
                    out.drops["below_min_voltage"] += 1
                    continue
                if voltage is None:
                    voltage = part.asset.max_voltage_kv
            else:
                out.counters["towers_without_line"] += 1
            if voltage is None:
                kept_without_voltage["tower"] += 1
            points.append(_record(asset, "tower", tlon, tlat,
                                  bearing_deg=bearing, voltage_kv=voltage))

    # ---- generators and plants -------------------------------------------
    generators = [a for a in kept if a.kind == "generator"]
    member_pos = [_centroid(g.geometry, out.counters)[:2] for g in generators]
    member_in = (out.inside([p[0] for p in member_pos],
                            [p[1] for p in member_pos])
                 if generators else np.zeros(0, dtype=bool))
    covered: set[str] = set()
    for plant in (a for a in kept if a.kind == "plant"):
        source = plant.generator_source
        if source is None:
            continue
        members = [(g, pos) for g, pos, ok in
                   zip(generators, member_pos, member_in)
                   if ok and g.generator_source == source]
        if not members:
            continue
        if any(g.tags.get("plant") == plant.asset_id for g, _ in members):
            covered.add(plant.asset_id)
        elif plant.geometry["type"] in ("Polygon", "MultiPolygon"):
            hit = np.asarray(geometry.point_in_polygon(
                np.array([p[0] for _, p in members], dtype=np.float64),
                np.array([p[1] for _, p in members], dtype=np.float64),
                _polygons_of(plant.geometry)), dtype=bool)
            if hit.any():
                covered.add(plant.asset_id)

    for asset in kept:
        if asset.kind not in ("generator", "plant"):
            continue
        if asset.asset_id in covered:
            reason = ("wind_plant_with_turbines"
                      if asset.generator_source == "wind"
                      else "plant_with_member_generators")
            out.drops[reason] += 1
            continue
        if _is_turbine(asset):
            lon, lat = asset.geometry["coordinates"][:2]
            if out.inside(lon, lat)[0] and asset.hub_height_m is None:
                out.counters["turbines_missing_hub_height"] += 1
            points.append(_record(asset, "turbine", lon, lat,
                                  hub_height_m=asset.hub_height_m,
                                  capacity_mw=asset.capacity_mw))
            continue
        if asset.generator_source == "solar":
            grid = None
            if pv_grid_m is not None:
                if asset.geometry["type"] == "Point":
                    out.counters["pv_point_not_gridded"] += 1
                else:
                    grid = _pv_grid(asset, pv_grid_m, out.counters)
                    if grid is None:
                        out.counters["pv_grid_fallback_centroid"] += 1
            if grid is None:
                lon, lat, degenerate = _centroid(asset.geometry, out.counters)
                out.counters["zero_area_centroid"] += int(degenerate)
                points.append(_record(asset, "pv", lon, lat,
                                      voltage_kv=asset.max_voltage_kv,
                                      capacity_mw=asset.capacity_mw))
            else:
                glon, glat = grid
                share = (asset.capacity_mw / glon.size
                         if asset.capacity_mw is not None else None)
                for k in range(glon.size):
                    points.append(_record(
                        asset, "pv", glon[k], glat[k],
                        site_id=f"{asset.asset_id}#pv{k}",
                        voltage_kv=asset.max_voltage_kv, capacity_mw=share))
            continue
        lon, lat, degenerate = _centroid(asset.geometry, out.counters)
        out.counters["zero_area_centroid"] += int(degenerate)
        points.append(_record(asset, "plant", lon, lat,
                              voltage_kv=asset.max_voltage_kv,
                              capacity_mw=asset.capacity_mw))

    out.add_point_sites(points)

    drops = {reason: count for reason, count in sorted(out.drops.items())
             if count}
    if not out.sites:
        raise SitesRefused(
            "no sites remain after filtering; assets by kind "
            f"{dict(sorted(assets_by_kind.items()))}, dropped by reason "
            f"{drops or 'none'}")

    sites = [Site(**record) for record in out.sites]
    hubs = {round(s.hub_height_m, 1) for s in sites
            if s.kind == "turbine" and s.hub_height_m is not None}
    heights = tuple(sorted(set(requested_heights) | hubs))

    counts = Counter(s.kind for s in sites)
    c = out.counters
    notes = []
    if c["turbines_missing_hub_height"]:
        notes.append(f"{c['turbines_missing_hub_height']} turbine(s) have no "
                     "hub height; hub_height_m is null for them and no hub "
                     "height was assumed")
    if c["pv_grid_fallback_centroid"]:
        notes.append(f"{c['pv_grid_fallback_centroid']} PV polygon(s) were "
                     "smaller than --pv-grid-m and got one centroid site")
    if c["pv_point_not_gridded"]:
        notes.append(f"{c['pv_point_not_gridded']} PV asset(s) are points "
                     "and got one site")
    if c["zero_area_centroid"]:
        notes.append(f"{c['zero_area_centroid']} polygon(s) enclose no area; "
                     "their vertex mean was used")
    if c["open_ring_closed"]:
        notes.append(f"{c['open_ring_closed']} open polygon ring(s) were "
                     "closed from their first position")
    if c["towers_without_line"]:
        notes.append(f"{c['towers_without_line']} tower(s) had no line "
                     f"within {TOWER_LINE_RADIUS_M:g} m; their bearing is "
                     "null")
    if towers:
        notes.append("towers take bearing (and voltage, when untagged) from "
                     f"the nearest line segment within "
                     f"{TOWER_LINE_RADIUS_M:g} m")
    if c["site_id_disambiguated"]:
        notes.append(f"{c['site_id_disambiguated']} site id(s) collided "
                     "after rounding and carry a ~n suffix")

    provenance = {
        "builder": "woof.energy.sites",
        "settings": {
            "spacing_m": spacing_m,
            "kinds": sorted(keep_kinds) if keep_kinds is not None else None,
            "min_voltage_kv": min_voltage_kv,
            "region": region is not None,
            "heights_m_requested": sorted(set(requested_heights)),
            "include_towers": bool(include_towers),
            "pv_grid_m": pv_grid_m,
            "dedupe_radius_m": DEDUPE_FRACTION * spacing_m,
            "dedupe_max_angle_deg": DEDUPE_MAX_ANGLE_DEG,
            "tower_line_radius_m": TOWER_LINE_RADIUS_M,
        },
        "assets_by_kind": dict(sorted(assets_by_kind.items())),
        "counts_by_kind": dict(sorted(counts.items())),
        "drops": drops,
        "kept_without_voltage": dict(sorted(kept_without_voltage.items())),
        "turbines_missing_hub_height": int(c["turbines_missing_hub_height"]),
        "hub_heights_m": sorted(hubs),
        "notes": notes,
    }
    return SiteSet(sites=sites, heights_m=heights, spacing_m=spacing_m,
                   provenance=provenance)


def main(args) -> int:
    assets_path = Path(args.assets)
    output = Path(args.output)
    if not assets_path.is_file():
        raise SitesRefused(f"assets document does not exist: {assets_path}")
    collection = load_assets(assets_path)
    region = None
    if args.region is not None:
        region = load_region(args.region)
    site_set = build_sites(
        collection, spacing_m=args.spacing_m, kinds=args.kinds,
        min_voltage_kv=args.min_voltage_kv, region=region,
        heights_m=args.heights_m, include_towers=args.include_towers,
        pv_grid_m=args.pv_grid_m)
    output_dir = output.resolve().parent
    site_set.assets_ref = file_ref(assets_path, relative_to=output_dir)
    if args.region is not None:
        site_set.provenance["region_ref"] = file_ref(args.region,
                                                     relative_to=output_dir)
    written = dump_sites(site_set, output)
    print(json.dumps({
        "schema": "woof-energy.sites-summary.v1",
        "output": str(written),
        "count": len(site_set),
        "counts_by_kind": site_set.provenance["counts_by_kind"],
        "drops": site_set.provenance["drops"],
        "heights_m": list(site_set.heights_m),
        "spacing_m": site_set.spacing_m,
        "notes": site_set.provenance["notes"],
    }, indent=2, default=str))
    return 0


__all__ = ["SitesRefused", "TOWER_LINE_RADIUS_M", "DEDUPE_FRACTION",
           "DEDUPE_MAX_ANGLE_DEG", "MAX_PV_GRID_POINTS", "build_sites",
           "load_region", "main"]
