"""``woof energy partition``: a worldwide sites set as independent regional clusters.

The wrf-tiles and wrf-nests planners each emit ONE parent over the sites
they are given, so a worldwide ``woof-energy.sites.v1`` document (every
public wind farm, a few hundred thousand turbines) cannot be planned in
one go: the parent would span the globe, and the tile planner refuses
sites spread over more than 180 degrees of longitude.  This module cuts
such a set into regional clusters that are planned, run and extracted
independently, each with a parent of its own, and records the cut as a
``woof-energy.partition.v1`` document.

How the clusters are found
--------------------------
1. **Single linkage on the sphere.**  Two sites are linked when their
   great-circle distance on the WRF sphere
   (:data:`woof.static.projection.EARTH_RADIUS_M`) is at most
   ``linkage_km`` (default :data:`DEFAULT_LINKAGE_KM`); a cluster is a
   connected component of that graph.  Distances are compared as 3-D
   chords of unit vectors, so the antimeridian and the poles are not
   special.  The search is grid-hashed: sites are binned into cubes of
   edge ``chord / sqrt(3)`` (every pair inside one cube is linked without
   a test), only cube pairs whose point boxes can be within a chord are
   examined, a pair whose boxes are entirely within a chord is linked
   without a test, and the remaining pairs are tested point to point on
   the boundary strips only, skipping pairs already joined.  No step is
   O(n^2) in the site count.
2. **Size caps.**  A component is accepted when it has at most
   ``max_sites`` sites, its extent in a local azimuthal-equidistant frame
   (:class:`woof.energy.geometry.LocalProjection` about its centre) is at
   most ``max_extent_km`` on both axes, its shortest longitude arc is at
   most 180 degrees (the tile planner's and ``woof domain``'s rule; only
   sites spread round a pole break it), and a parent chain fits the card
   (below).  Otherwise it is bisected across its longer axis at the widest
   gap between consecutive sites in the middle half of the sites (so a
   wind farm is not cut when a gap between farms is available), each half
   is re-linked, and the pieces are examined in turn.
3. **Determinism.**  Sites are put in ``site_id`` order before anything
   else, so the partition is a function of the site SET, not of the
   document's order.  A cluster's ``cluster_id`` is ``c`` followed by the
   first :data:`CLUSTER_ID_HEX` hex digits of the sha256 of its sorted
   member ids joined by newlines; clusters are written sorted by id.

What each cluster records
-------------------------
* ``bbox``: the members' bounding box on their shortest longitude arc,
  whose centre is ``lon_frame``, so ``west`` may be below -180 or
  ``east`` above 180 for a cluster across the antimeridian;
  ``crosses_antimeridian`` says so.  ``region`` is the same box buffered
  by ``corridor_km`` as GeoJSON in [-180, 180]: a Polygon, or a
  MultiPolygon split at the antimeridian (RFC 7946 section 3.1.9).
* ``leaf_dx_m``: the leaf spacing, ``--leaf-dx-m`` (default
  :data:`DEFAULT_LEAF_DX_M`), refused outside
  [:data:`LEAF_DX_MIN_M`, :data:`LEAF_DX_MAX_M`].
* ``parent_chain``: the shortest chain of
  :data:`woof.energy.plan_wrf_tiles.DEFAULT_STEP_RATIO` steps (at most
  :data:`woof.energy.plan_wrf_tiles.MAX_CHAIN_STEPS`) whose parent, the
  cluster extent plus the buffer the tile planner adds
  (:func:`woof.energy.plan_wrf_tiles._level_margins` plus the parent's
  boundary clearance), fits the card.  A suggestion: the tile planner
  takes the same ladder and the ``woof domain`` wizard's own fit is the
  authority when the cluster is planned.
* ``projection``: :func:`woof.domain_wizard.auto_projection` at the
  bbox's centre latitude, the choice the wizard makes for the region.
* ``estimated_tiles``: the cover
  :func:`woof.energy.geometry.cover_with_rectangles` makes of the sites in
  the local frame at each tile level, with tiles no larger than the
  largest square the per-tile budget affords.
* ``topology``: ``wrf-nests`` for a cluster no wider than
  :data:`NESTS_MAX_EXTENT_KM` with at most :data:`NESTS_MAX_SITES` sites
  and one estimated leaf tile (one run, a 3/5 nest ladder from the
  source), ``wrf-tiles`` otherwise; ``--topology`` pins one for all.

What is pinned and what is estimated
------------------------------------
Square sizes come from :func:`woof.core.preflight.estimate_experiment` on
a bare :class:`woof.config.RunConfig` (the leaf with the LES recipe of
:data:`woof.energy.plan_wrf_tiles.LES_PHYSICS_RECIPE` at or finer than
250 m), against the ceiling ``woof downscale`` applies
(:func:`woof.downscale._budget_bytes`).  The planner later prices each
tile on the parent's emitted physics suite, which can be heavier, so the
tile counts are estimates (``estimate_basis`` says so) and the parent
chain a suggestion.  The card is ``--card``/``--vram-gib``/
``--hardware-json`` (a target snapshot, :mod:`woof.target_hardware`),
default :data:`woof.energy.plan_wrf_tiles.DEFAULT_CARD`; nothing here
probes the GPU of the machine the partition is written on.

Planning a cluster
------------------
:func:`plan_cluster` writes the cluster's sites as a ``sites.v1`` subset
(document order, provenance naming the partition) and hands it to
:func:`woof.energy.plan_wrf_tiles.build_plan` or
:func:`woof.energy.plan_wrf_nests.build_plan`, so every cluster gets its
own parent.  The card recorded in the partition is passed on unless the
caller names another.

Python boundary (docs/dev/static-rust-port.md): per-site vectors only (a
few hundred thousand points touched once at planning time) and no model
fields, so numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woof.energy.contracts import (
    ContractError,
    EnergyNotImplemented,
    Plan,
    SiteSet,
    _read_json,
    _schema_of,
    _write_json,
    dump_plan,
    dump_sites,
    load_sites,
    sha256_file,
    validate_geometry,
)

PARTITION_SCHEMA = "woof-energy.partition.v1"

#: Single-linkage distance: sites closer than this share a cluster.
DEFAULT_LINKAGE_KM = 30.0
#: Below this the grid hash's packed cube key would overflow int64.
MIN_LINKAGE_KM = 0.1

#: A cluster wider than this (either local axis) is bisected.
DEFAULT_MAX_EXTENT_KM = 250.0
#: A cluster with more sites than this is bisected.
DEFAULT_MAX_SITES = 20_000

#: Leaf spacing, and the range a reanalysis leaf may take.
DEFAULT_LEAF_DX_M = 200.0
LEAF_DX_MIN_M, LEAF_DX_MAX_M = 100.0, 300.0

#: Half-width of the high-resolution corridor around each site (the
#: planners' ``--corridor-km`` default).
DEFAULT_CORRIDOR_KM = 2.0

#: ``topology = auto`` picks wrf-nests for clusters at most this wide,
#: with at most this many sites and one estimated leaf tile.
NESTS_MAX_EXTENT_KM = 25.0
NESTS_MAX_SITES = 500

#: Hex digits of the member-id sha256 in a cluster id.
CLUSTER_ID_HEX = 16

#: Topologies a cluster may be planned with, and the CLI choice set.
CLUSTER_TOPOLOGIES = ("wrf-tiles", "wrf-nests")
TOPOLOGY_CHOICES = ("auto",) + CLUSTER_TOPOLOGIES

#: Projections the wizard selects from (:func:`woof.domain_wizard.
#: auto_projection`).
PROJECTIONS = ("lambert", "mercator", "polar")

#: The largest square the planner's own capacity search reaches
#: (:meth:`woof.energy.plan_wrf_tiles._Pricer.largest_square`).
MAX_SQUARE_CELLS = 4096

#: Smallest square the size search starts from.
_MIN_SQUARE_CELLS = 8

#: Pairwise distance block (points x points) for the exact linkage test.
_PAIR_BLOCK = 1 << 22

ESTIMATE_BASIS = (
    "woof.core.preflight.estimate_experiment on a bare RunConfig (the LES "
    "recipe on leaves at or finer than 250 m) against woof.downscale."
    "_budget_bytes; the planner prices tiles on the parent's emitted "
    "physics suite, which can be heavier, so tile counts are estimates")


class PartitionRefused(ValueError):
    """The partition cannot be built as asked."""


# --------------------------------------------------------------------------
# single linkage


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    from woof.energy.geometry import _unit_vectors as unit

    return unit(lon, lat)


def _chord_m(distance_m: float) -> float:
    from woof.static.projection import EARTH_RADIUS_M

    angle = min(float(distance_m) / EARTH_RADIUS_M, math.pi)
    return 2.0 * EARTH_RADIUS_M * math.sin(0.5 * angle)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb
        return True


def _any_within(a: np.ndarray, b: np.ndarray, lo_a: np.ndarray,
                hi_a: np.ndarray, lo_b: np.ndarray, hi_b: np.ndarray,
                chord2: float) -> bool:
    """Is some point of ``a`` within the chord of some point of ``b``?"""

    def near_box(points, lo, hi):
        gap = np.maximum(0.0, np.maximum(lo - points, points - hi))
        return points[np.einsum("ij,ij->i", gap, gap) <= chord2]

    a = near_box(a, lo_b, hi_b)
    if a.size == 0:
        return False
    b = near_box(b, lo_a, hi_a)
    if b.size == 0:
        return False
    step = max(1, _PAIR_BLOCK // max(1, len(b)))
    for start in range(0, len(a), step):
        block = a[start:start + step]
        diff = block[:, None, :] - b[None, :, :]
        if np.any(np.einsum("ijk,ijk->ij", diff, diff) <= chord2):
            return True
    return False


def linkage_labels(lon: Sequence[float], lat: Sequence[float],
                   linkage_km: float) -> np.ndarray:
    """Connected components of the great-circle single-linkage graph.

    Label ``k`` (``0..K-1``) per point; labels are numbered in order of
    each component's first point.  Exact: two points share a label iff a
    chain of points links them with every hop at most ``linkage_km``.
    """

    from woof.static.projection import EARTH_RADIUS_M

    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    n = lon.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    linkage_km = float(linkage_km)
    if not math.isfinite(linkage_km) or linkage_km < MIN_LINKAGE_KM:
        raise PartitionRefused(
            f"--linkage-km {linkage_km!r} must be at least {MIN_LINKAGE_KM:g}")
    chord = _chord_m(linkage_km * 1000.0)
    chord2 = chord * chord
    xyz = _unit_vectors(lon, lat) * EARTH_RADIUS_M
    edge = chord / math.sqrt(3.0)
    half = int(math.ceil(EARTH_RADIUS_M / edge)) + 3
    width = 2 * half + 1
    ijk = np.floor(xyz / edge).astype(np.int64) + half
    key = (ijk[:, 0] * width + ijk[:, 1]) * width + ijk[:, 2]
    order = np.argsort(key, kind="stable")
    cells, start = np.unique(key[order], return_index=True)
    count = np.diff(np.append(start, n))
    points = xyz[order]
    lo = np.minimum.reduceat(points, start, axis=0)
    hi = np.maximum.reduceat(points, start, axis=0)

    # Cube offsets (lexicographically positive half) that can hold a pair
    # within the chord: the nearest corners of cubes |d| apart along an
    # axis are (|d| - 1) edges apart.
    offsets = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                gap = sum(max(abs(d) - 1, 0) ** 2 for d in (dx, dy, dz))
                if edge * edge * gap <= chord2 * (1.0 + 1e-12):
                    offsets.append((dx * width + dy) * width + dz)

    cells_n = cells.size
    sure_a: list[np.ndarray] = []
    sure_b: list[np.ndarray] = []
    test_a: list[np.ndarray] = []
    test_b: list[np.ndarray] = []
    for delta in offsets:
        target = cells + delta
        pos = np.minimum(np.searchsorted(cells, target), cells_n - 1)
        hit = cells[pos] == target
        if not np.any(hit):
            continue
        a = np.flatnonzero(hit)
        b = pos[hit]
        gap = np.maximum(0.0, np.maximum(lo[a] - hi[b], lo[b] - hi[a]))
        near = np.einsum("ij,ij->i", gap, gap) <= chord2
        a, b = a[near], b[near]
        span = np.maximum(hi[a] - lo[b], hi[b] - lo[a])
        sure = np.einsum("ij,ij->i", span, span) <= chord2
        sure_a.append(a[sure])
        sure_b.append(b[sure])
        test_a.append(a[~sure])
        test_b.append(b[~sure])

    links = _UnionFind(cells_n)
    if sure_a:
        for a, b in zip(np.concatenate(sure_a).tolist(),
                        np.concatenate(sure_b).tolist()):
            links.union(a, b)
    if test_a:
        a_all = np.concatenate(test_a)
        b_all = np.concatenate(test_b)
        # Cheapest tests first, so cheap links can make dear tests moot.
        cost = count[a_all] * count[b_all]
        for k in np.lexsort((b_all, a_all, cost)).tolist():
            a, b = int(a_all[k]), int(b_all[k])
            if links.find(a) == links.find(b):
                continue
            if _any_within(points[start[a]:start[a] + count[a]],
                           points[start[b]:start[b] + count[b]],
                           lo[a], hi[a], lo[b], hi[b], chord2):
                links.union(a, b)

    roots = np.fromiter((links.find(c) for c in range(cells_n)),
                        dtype=np.int64, count=cells_n)
    per_point = np.empty(n, dtype=np.int64)
    per_point[order] = np.repeat(roots, count)
    _, first = np.unique(per_point, return_index=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    _, inverse = np.unique(per_point, return_inverse=True)
    return rank[inverse]


def _components(index: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                linkage_km: float) -> list[np.ndarray]:
    """``index`` split into its linkage components, each ascending."""

    if index.size <= 1:
        return [index]
    labels = linkage_labels(lon[index], lat[index], linkage_km)
    order = np.argsort(labels, kind="stable")
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return [np.sort(index[part]) for part in np.split(order, cuts)]


# --------------------------------------------------------------------------
# the local frame of a cluster


@dataclass(frozen=True)
class _Frame:
    lat0: float
    lon0: float
    x: np.ndarray
    y: np.ndarray

    @property
    def extent_m(self) -> tuple[float, float]:
        return (float(np.ptp(self.x)), float(np.ptp(self.y)))


def _frame(lon: np.ndarray, lat: np.ndarray) -> _Frame:
    """The local frame about the members' normalised mean unit vector
    (:func:`woof.energy.geometry._centre_of`); members spread round the
    globe, which have no such centre, are framed about their first site
    (the frame then only steers the bisection that splits them)."""

    from woof.energy.geometry import GeometryError, _centre_of, local_projection

    try:
        projection = _centre_of(_unit_vectors(lon, lat))
    except GeometryError:
        projection = local_projection(float(lat[0]), float(lon[0]))
    x, y = projection.forward(lon, lat)
    return _Frame(float(projection.lat0), float(projection.lon0),
                  np.asarray(x, dtype=np.float64),
                  np.asarray(y, dtype=np.float64))


def _lon_arc(lon: np.ndarray) -> tuple[float, float, float] | None:
    """(centre, west, east) of the shortest longitude arc holding ``lon``
    (:func:`woof.domain_wizard._minimum_longitude_arc`, the rule ``woof
    domain`` and the tile planner apply), or ``None`` when it is wider
    than 180 degrees (sites spread round a pole)."""

    from woof.domain_wizard import _minimum_longitude_arc

    try:
        return _minimum_longitude_arc(lon)
    except ValueError:
        return None


def _bisect(index: np.ndarray, frame: _Frame) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of ``index`` cut across the frame's longer axis at the
    widest gap in the middle half of the sites (nearest the median on a
    tie)."""

    ex, ey = frame.extent_m
    along, across = (frame.x, frame.y) if ex >= ey else (frame.y, frame.x)
    order = np.lexsort((np.arange(index.size), across, along))
    v = along[order]
    n = index.size
    first = max(0, n // 4 - 1)
    last = min(n - 2, (3 * n) // 4)
    cuts = np.arange(first, last + 1)
    gaps = v[cuts + 1] - v[cuts]
    widest = gaps == gaps.max()
    middle = 0.5 * (n - 2)
    pick = cuts[widest][np.argmin(np.abs(cuts[widest] - middle))]
    return np.sort(index[order[:pick + 1]]), np.sort(index[order[pick + 1:]])


# --------------------------------------------------------------------------
# sizing


def _parent_nz() -> int:
    from woof import domain_wizard

    return int(domain_wizard._SHARED_GRID_AND_DYNAMICS["nz"])


class _SquareBudget:
    """Largest square grid (cells per axis) a card affords, by level count
    and spacing, priced by the preflight estimator on a bare RunConfig."""

    def __init__(self, vram_gib: float):
        from woof.downscale import _budget_bytes

        self.vram_gib = float(vram_gib)
        self.limit = float(_budget_bytes(self.vram_gib)[1])
        self._cache: dict[tuple, int] = {}

    def _peak(self, n: int, *, dx_m: float, nz: int, les: bool) -> float:
        from datetime import datetime, timezone

        from woof.config import RunConfig, validate_run_config
        from woof.core.preflight import estimate_experiment
        from woof.energy.plan_wrf_tiles import LES_PHYSICS_RECIPE
        from woof.experiment import experiment_from_run_config

        settings = dict(nx=int(n), ny=int(n), nz=int(nz), dx=float(dx_m),
                        dy=float(dx_m), dt=max(0.5, 6.0 * dx_m / 1000.0),
                        ztop=20000.0, run_seconds=3600.0)
        if les:
            settings.update(LES_PHYSICS_RECIPE)
        config = RunConfig(**settings)
        validate_run_config(config)
        experiment = experiment_from_run_config(
            config, datetime(2000, 1, 1, tzinfo=timezone.utc))
        return float(estimate_experiment(
            experiment, vram_gib=self.vram_gib).peak_envelope_bytes)

    def largest(self, *, dx_m: float, nz: int, les: bool) -> int:
        """Cells per axis of the largest square under the ceiling (0 when
        not even :data:`_MIN_SQUARE_CELLS` fits)."""

        key = (round(float(dx_m), 6), int(nz), bool(les))
        if key not in self._cache:
            try:
                def fits(size: int) -> bool:
                    return self._peak(size, dx_m=dx_m, nz=nz,
                                      les=les) <= self.limit

                if not fits(_MIN_SQUARE_CELLS):
                    best = 0
                else:
                    low, high = _MIN_SQUARE_CELLS, 2 * _MIN_SQUARE_CELLS
                    while high <= MAX_SQUARE_CELLS and fits(high):
                        low, high = high, high * 2
                    high = min(high, MAX_SQUARE_CELLS + 1)
                    while low + 1 < high:
                        mid = (low + high) // 2
                        if fits(mid):
                            low = mid
                        else:
                            high = mid
                    best = low
            except (ValueError, NotImplementedError) as error:
                raise PartitionRefused(
                    f"the size estimate at {dx_m:g} m with {nz} levels "
                    f"failed: {error}") from error
            self._cache[key] = best
        return self._cache[key]


@dataclass(frozen=True)
class _Sizing:
    leaf_dx_m: float
    corridor_m: float
    leaf_nz: int
    parent_nz: int
    budget: _SquareBudget

    def leaf_les(self) -> bool:
        from woof.energy.plan_wrf_tiles import LES_CHILD_DX_M

        return self.leaf_dx_m <= LES_CHILD_DX_M + 1e-9

    def chain(self, extent_m: float) -> tuple[tuple[int, ...], int] | None:
        """(ratios, parent cells per axis) of the shortest chain whose
        parent holds ``extent_m`` plus the tile planner's buffer, or
        ``None``."""

        from woof.domain_wizard import _SPEC_BDY_WIDTH
        from woof.energy import plan_wrf_tiles as tiles

        clearance = (_SPEC_BDY_WIDTH + tiles.TERRAIN_BLEND_ROWS
                     + tiles.SINT_STENCIL_ROWS)
        for steps in range(1, tiles.MAX_CHAIN_STEPS + 1):
            ratios = (tiles.DEFAULT_STEP_RATIO,) * steps
            pdx = self.leaf_dx_m * math.prod(ratios)
            margins = tiles._level_margins(ratios, pdx, self.corridor_m)
            buffer_m = margins[0] + (clearance + 2) * pdx
            cells = int(math.ceil((extent_m + 2.0 * buffer_m) / pdx)) + 1
            if cells <= self.budget.largest(dx_m=pdx, nz=self.parent_nz,
                                             les=False):
                return ratios, cells
        return None

    def tiles(self, frame: _Frame, ratios: tuple[int, ...]
              ) -> tuple[list[int], int]:
        """(estimated tiles per level coarsest first, leaf tile cells)."""

        from woof.energy import plan_wrf_tiles as tiles
        from woof.energy.geometry import GeometryError, cover_with_rectangles

        pdx = self.leaf_dx_m * math.prod(ratios)
        margins = tiles._level_margins(ratios, pdx, self.corridor_m)
        counts = []
        size = 0
        dx = pdx
        for level, ratio in enumerate(ratios):
            align = dx
            dx = dx / ratio
            leaf = level == len(ratios) - 1
            size = self.budget.largest(
                dx_m=dx, nz=self.leaf_nz if leaf else self.parent_nz,
                les=leaf and self.leaf_les())
            max_cells = max(2 * ratio, size - 2 * ratio)
            try:
                rects = cover_with_rectangles(
                    frame.x, frame.y, margin_m=margins[level], dx_m=dx,
                    max_nx=max_cells, max_ny=max_cells, align_m=align)
            except GeometryError as error:
                raise PartitionRefused(
                    f"no {dx:g} m tile the card affords ({size} cells a "
                    f"side) can hold a site with its {margins[level]:g} m "
                    f"margin: {error}; use a larger --card/--vram-gib or "
                    "fewer --nz levels") from error
            counts.append(len(rects))
        return counts, size


# --------------------------------------------------------------------------
# the document


def cluster_id_for(site_ids: Sequence[str]) -> str:
    """``c`` + the first :data:`CLUSTER_ID_HEX` hex digits of the sha256 of
    the sorted member ids joined by newlines."""

    text = "\n".join(sorted(str(s) for s in site_ids))
    return "c" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:CLUSTER_ID_HEX]


@dataclass(frozen=True)
class Cluster:
    """One cluster of a ``partition.v1`` document."""

    cluster_id: str
    site_ids: tuple[str, ...]
    centre: dict
    bbox: dict
    crosses_antimeridian: bool
    region: dict
    extent_km: tuple[float, float]
    leaf_dx_m: float
    parent_chain: dict
    projection: str
    topology: str
    estimated_tiles: dict
    linkage_component_sites: int

    def __post_init__(self):
        what = f"cluster {self.cluster_id!r}"
        ids = tuple(str(s) for s in self.site_ids)
        if not ids:
            raise ContractError(f"{what} has no sites")
        if list(ids) != sorted(set(ids)):
            raise ContractError(f"{what} site_ids must be sorted and unique")
        object.__setattr__(self, "site_ids", ids)
        if self.cluster_id != cluster_id_for(ids):
            raise ContractError(f"{what} is not the hash of its site_ids "
                                f"({cluster_id_for(ids)})")
        leaf = _number(self.leaf_dx_m, f"{what} leaf_dx_m")
        if not LEAF_DX_MIN_M <= leaf <= LEAF_DX_MAX_M:
            raise ContractError(f"{what} leaf_dx_m {leaf:g} is outside "
                                f"[{LEAF_DX_MIN_M:g}, {LEAF_DX_MAX_M:g}]")
        if self.topology not in CLUSTER_TOPOLOGIES:
            raise ContractError(f"{what} topology {self.topology!r} is not "
                                f"one of {CLUSTER_TOPOLOGIES}")
        if self.projection not in PROJECTIONS:
            raise ContractError(f"{what} projection {self.projection!r} is "
                                f"not one of {PROJECTIONS}")
        box = {k: _number(self.bbox.get(k), f"{what} bbox {k}")
               for k in ("west", "south", "east", "north", "lon_frame")}
        if not (-90.0 <= box["south"] <= box["north"] <= 90.0):
            raise ContractError(f"{what} bbox latitudes are out of order "
                                "or range")
        if not (box["west"] <= box["east"] <= box["west"] + 360.0
                and -360.0 <= box["west"] and box["east"] <= 360.0):
            raise ContractError(f"{what} bbox longitudes are out of order "
                                "or range")
        crosses = box["west"] < -180.0 or box["east"] > 180.0
        if bool(self.crosses_antimeridian) != crosses:
            raise ContractError(f"{what} crosses_antimeridian disagrees with "
                                "its bbox")
        validate_geometry(self.region, what=f"{what} region")
        if self.region.get("type") not in ("Polygon", "MultiPolygon"):
            raise ContractError(f"{what} region must be a Polygon or "
                                "MultiPolygon")
        ratios = self.parent_chain.get("ratios")
        if (not isinstance(ratios, list) or not ratios
                or any(isinstance(r, bool) or not isinstance(r, int) or r < 2
                       for r in ratios)):
            raise ContractError(f"{what} parent_chain ratios must be integers "
                                "of at least 2")
        pdx = _number(self.parent_chain.get("parent_dx_m"),
                      f"{what} parent_dx_m")
        if abs(pdx - leaf * math.prod(ratios)) > 1e-6 * pdx:
            raise ContractError(f"{what} parent_dx_m is not leaf_dx_m times "
                                "the chain ratios")
        for key in ("leaf", "intermediate", "leaf_tile_cells"):
            value = self.estimated_tiles.get(key)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise ContractError(f"{what} estimated_tiles {key} must be "
                                    "a non-negative integer")
        if self.linkage_component_sites < len(ids):
            raise ContractError(f"{what} linkage_component_sites is smaller "
                                "than the cluster")

    def to_json(self) -> dict[str, Any]:
        return {"cluster_id": self.cluster_id,
                "site_count": len(self.site_ids),
                "centre": dict(self.centre),
                "bbox": dict(self.bbox),
                "crosses_antimeridian": bool(self.crosses_antimeridian),
                "region": self.region,
                "extent_km": list(self.extent_km),
                "leaf_dx_m": self.leaf_dx_m,
                "parent_chain": dict(self.parent_chain),
                "projection": self.projection,
                "topology": self.topology,
                "estimated_tiles": dict(self.estimated_tiles),
                "linkage_component_sites": self.linkage_component_sites,
                "site_ids": list(self.site_ids)}

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> "Cluster":
        if not isinstance(record, Mapping):
            raise ContractError("partition cluster is not a JSON object")
        try:
            site_ids = tuple(record["site_ids"])
            if record["site_count"] != len(site_ids):
                raise ContractError(
                    f"cluster {record.get('cluster_id')!r} site_count "
                    "disagrees with site_ids")
            return cls(
                cluster_id=str(record["cluster_id"]), site_ids=site_ids,
                centre=dict(record["centre"]), bbox=dict(record["bbox"]),
                crosses_antimeridian=bool(record["crosses_antimeridian"]),
                region=dict(record["region"]),
                extent_km=tuple(float(v) for v in record["extent_km"]),
                leaf_dx_m=float(record["leaf_dx_m"]),
                parent_chain=dict(record["parent_chain"]),
                projection=str(record["projection"]),
                topology=str(record["topology"]),
                estimated_tiles=dict(record["estimated_tiles"]),
                linkage_component_sites=int(
                    record["linkage_component_sites"]))
        except KeyError as error:
            raise ContractError(
                f"partition cluster lacks {error}") from error


def _number(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{what} must be a number, got {value!r}") from error
    if not math.isfinite(number):
        raise ContractError(f"{what} must be finite, got {value!r}")
    return number


_PARAMETER_KEYS = ("linkage_km", "max_extent_km", "max_sites", "leaf_dx_m",
                   "corridor_km", "leaf_nz", "parent_nz", "step_ratio",
                   "topology")


@dataclass
class Partition:
    """A ``partition.v1`` document.  ``sites_ref`` paths are relative to the
    partition file's directory."""

    sites_ref: dict
    parameters: dict
    capacity: dict
    clusters: list[Cluster]
    notes: list[str] = field(default_factory=list)
    estimate_basis: str = ESTIMATE_BASIS
    path: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.sites_ref, Mapping) or not all(
                isinstance(self.sites_ref.get(k), str)
                for k in ("path", "sha256")):
            raise ContractError("partition sites must be {path, sha256}")
        if Path(self.sites_ref["path"]).is_absolute():
            raise ContractError("partition sites path must be relative to "
                                "the partition file's directory")
        missing = [k for k in _PARAMETER_KEYS if k not in self.parameters]
        if missing:
            raise ContractError(f"partition parameters lack {missing}")
        capacity = _number(self.capacity.get("vram_gib"),
                           "partition capacity vram_gib")
        if capacity <= 0.0:
            raise ContractError("partition capacity vram_gib must be positive")
        ids = [c.cluster_id for c in self.clusters]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ContractError("partition clusters must be sorted by a "
                                "unique cluster_id")
        seen: set[str] = set()
        for cluster in self.clusters:
            overlap = seen.intersection(cluster.site_ids)
            if overlap:
                raise ContractError(
                    f"site(s) {sorted(overlap)[:5]} belong to more than one "
                    "cluster")
            seen.update(cluster.site_ids)

    @property
    def site_count(self) -> int:
        return sum(len(c.site_ids) for c in self.clusters)

    def cluster(self, cluster_id: str) -> Cluster:
        for candidate in self.clusters:
            if candidate.cluster_id == cluster_id:
                return candidate
        raise ContractError(f"partition has no cluster {cluster_id!r}")

    def sites_path(self) -> Path:
        if self.path is None:
            raise ContractError("partition has no file location to resolve "
                                "its sites document against")
        return self.path.parent / self.sites_ref["path"]

    def to_json(self) -> dict[str, Any]:
        return {"schema": PARTITION_SCHEMA,
                "sites": dict(self.sites_ref),
                "parameters": dict(self.parameters),
                "capacity": dict(self.capacity),
                "estimate_basis": self.estimate_basis,
                "count": len(self.clusters),
                "site_count": self.site_count,
                "notes": list(self.notes),
                "clusters": [c.to_json() for c in self.clusters]}

    @classmethod
    def from_json(cls, document: Mapping[str, Any], *,
                  path: str | Path | None = None) -> "Partition":
        _schema_of(document, PARTITION_SCHEMA, "partition document")
        records = document.get("clusters")
        if not isinstance(records, list):
            raise ContractError("partition document has no clusters list")
        clusters = [Cluster.from_json(r) for r in records]
        if document.get("count", len(clusters)) != len(clusters):
            raise ContractError("partition count disagrees with clusters")
        try:
            partition = cls(sites_ref=dict(document["sites"]),
                            parameters=dict(document["parameters"]),
                            capacity=dict(document["capacity"]),
                            clusters=clusters,
                            notes=list(document.get("notes") or []),
                            estimate_basis=str(document.get(
                                "estimate_basis", ESTIMATE_BASIS)),
                            path=Path(path) if path is not None else None)
        except KeyError as error:
            raise ContractError(
                f"partition document lacks {error}") from error
        if document.get("site_count", partition.site_count) \
                != partition.site_count:
            raise ContractError("partition site_count disagrees with "
                                "clusters")
        return partition


def load_partition(path: str | Path) -> Partition:
    path = Path(path)
    return Partition.from_json(_read_json(path, "partition document"),
                               path=path)


def dump_partition(partition: Partition, path: str | Path) -> Path:
    """Write atomically (``.partial`` then rename)."""

    partition.path = Path(path)
    return _write_json(partition.to_json(), path)


# --------------------------------------------------------------------------
# capacity


def resolve_capacity(*, card: str | None = None, vram_gib: float | None = None,
                     hardware_json: str | Path | None = None
                     ) -> tuple[float, dict, str | None]:
    """(GiB, the capacity record, a default note).  At most one of the
    three may be given; none means
    :data:`woof.energy.plan_wrf_tiles.DEFAULT_CARD`.  Never probes a GPU."""

    from woof.energy.plan_wrf_tiles import DEFAULT_CARD

    given = [name for name, value in (("--card", card),
                                      ("--vram-gib", vram_gib),
                                      ("--hardware-json", hardware_json))
             if value is not None]
    if len(given) > 1:
        raise PartitionRefused(f"choose one of {', '.join(given)}")
    if hardware_json is not None:
        from woof.core.preflight import GIB
        from woof.target_hardware import validate_sizing

        path = Path(hardware_json)
        try:
            payload = path.read_bytes()
            document = json.loads(payload)
            if not isinstance(document, dict):
                raise ValueError("not a JSON object")
            sizing = validate_sizing(document.get("sizing"))
        except (OSError, ValueError) as error:
            raise PartitionRefused(
                f"--hardware-json {path}: {error}") from error
        gib = float(sizing["total_bytes"]) / GIB
        return gib, {"vram_gib": round(gib, 6), "basis": "--hardware-json",
                     "device": sizing["profile"]["name"],
                     "sha256": hashlib.sha256(payload).hexdigest()}, None
    if vram_gib is not None:
        gib = float(vram_gib)
        if not math.isfinite(gib) or gib <= 0.0:
            raise PartitionRefused(f"--vram-gib {vram_gib!r} is not a size")
        return gib, {"vram_gib": gib, "basis": "--vram-gib"}, None
    from woof.domain_wizard import declared_card_gib

    note = None
    if card is None:
        card = DEFAULT_CARD
        note = (f"no --card/--vram-gib/--hardware-json given: sized for a "
                f"{DEFAULT_CARD} card (the tile planner's default); nothing "
                "probed the local GPU")
    try:
        gib = float(declared_card_gib(card))
    except ValueError as error:
        raise PartitionRefused(str(error)) from error
    return gib, {"vram_gib": gib, "basis": "--card", "card": str(card)}, note


# --------------------------------------------------------------------------
# building


def _region(west: float, south: float, east: float, north: float,
            buffer_m: float) -> dict:
    """The buffered box as GeoJSON in [-180, 180], split at the
    antimeridian when the local-frame box crosses it."""

    from woof.static.projection import EARTH_RADIUS_M

    dlat = math.degrees(buffer_m / EARTH_RADIUS_M)
    south = max(-89.9, south - dlat)
    north = min(89.9, north + dlat)
    cos_lat = max(math.cos(math.radians(max(abs(south), abs(north)))), 1e-3)
    dlon = min(math.degrees(buffer_m / (EARTH_RADIUS_M * cos_lat)), 10.0)
    from woof.energy.plan_wrf_tiles import lonlat_box

    return lonlat_box(west - dlon, south, east + dlon, north, digits=6)


def _cluster_record(index: np.ndarray, frame: _Frame, *, ids: np.ndarray,
                    lon: np.ndarray, lat: np.ndarray, sizing: _Sizing,
                    arc: tuple[float, float, float],
                    ratios: tuple[int, ...], parent_cells: int,
                    topology: str, component_sites: int) -> Cluster:
    from woof.domain_wizard import auto_projection

    member_ids = tuple(str(s) for s in ids[index])
    c_lon, c_lat = frame.lon0, frame.lat0
    lon_frame, west, east = (float(v) for v in arc)
    south, north = float(lat[index].min()), float(lat[index].max())
    counts, leaf_cells = sizing.tiles(frame, ratios)
    ex, ey = frame.extent_m
    if topology == "auto":
        small = (max(ex, ey) <= NESTS_MAX_EXTENT_KM * 1000.0
                 and len(member_ids) <= NESTS_MAX_SITES and counts[-1] == 1)
        topology = "wrf-nests" if small else "wrf-tiles"
    pdx = sizing.leaf_dx_m * math.prod(ratios)
    dxs = [pdx]
    for ratio in ratios:
        dxs.append(dxs[-1] / ratio)
    return Cluster(
        cluster_id=cluster_id_for(member_ids), site_ids=tuple(sorted(member_ids)),
        centre={"lat": round(c_lat, 6), "lon": round(c_lon, 6)},
        bbox={"west": round(west, 6), "south": round(south, 6),
              "east": round(east, 6), "north": round(north, 6),
              "lon_frame": round(lon_frame, 6)},
        crosses_antimeridian=bool(round(west, 6) < -180.0
                                  or round(east, 6) > 180.0),
        region=_region(west, south, east, north, sizing.corridor_m),
        extent_km=(round(ex / 1000.0, 3), round(ey / 1000.0, 3)),
        leaf_dx_m=sizing.leaf_dx_m,
        parent_chain={"ratios": [int(r) for r in ratios],
                      "parent_dx_m": pdx,
                      "dx_m": [round(d, 6) for d in dxs],
                      "parent_cells": int(parent_cells)},
        projection=auto_projection(0.5 * (south + north)),
        topology=topology,
        estimated_tiles={"leaf": int(counts[-1]),
                         "intermediate": int(sum(counts[:-1])),
                         "per_level": [int(c) for c in counts],
                         "leaf_tile_cells": int(leaf_cells)},
        linkage_component_sites=int(component_sites))


def build_partition(sites: SiteSet, *, sites_ref: Mapping[str, str],
                    linkage_km: float = DEFAULT_LINKAGE_KM,
                    max_extent_km: float = DEFAULT_MAX_EXTENT_KM,
                    max_sites: int = DEFAULT_MAX_SITES,
                    leaf_dx_m: float = DEFAULT_LEAF_DX_M,
                    corridor_km: float = DEFAULT_CORRIDOR_KM,
                    nz: int | None = None, topology: str = "auto",
                    card: str | None = None, vram_gib: float | None = None,
                    hardware_json: str | Path | None = None) -> Partition:
    """Cluster ``sites`` (see the module docstring) and return the
    partition; ``sites_ref`` is the ``{path, sha256}`` binding written
    into it (path relative to where the partition will be written)."""

    from woof.energy import plan_wrf_tiles as tiles

    if len(sites) == 0:
        raise PartitionRefused("the sites document has no sites to partition")
    leaf_dx_m = float(leaf_dx_m)
    if not (math.isfinite(leaf_dx_m)
            and LEAF_DX_MIN_M <= leaf_dx_m <= LEAF_DX_MAX_M):
        raise PartitionRefused(
            f"--leaf-dx-m {leaf_dx_m!r} is outside the reanalysis range "
            f"[{LEAF_DX_MIN_M:g}, {LEAF_DX_MAX_M:g}] m")
    for name, value in (("--max-extent-km", max_extent_km),
                        ("--corridor-km", corridor_km)):
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise PartitionRefused(f"{name} {value!r} must be positive")
    if isinstance(max_sites, bool) or int(max_sites) < 1:
        raise PartitionRefused(f"--max-sites {max_sites!r} must be at least 1")
    if nz is not None and int(nz) < 4:
        raise PartitionRefused("--nz must be at least 4 (the vertical "
                               "stencil width)")
    if topology not in TOPOLOGY_CHOICES:
        raise PartitionRefused(f"--topology {topology!r} is not one of "
                               f"{TOPOLOGY_CHOICES}")
    gib, capacity, card_note = resolve_capacity(
        card=card, vram_gib=vram_gib, hardware_json=hardware_json)
    parent_nz = _parent_nz()
    leaf_nz = (int(nz) if nz is not None
               else tiles.DEFAULT_LES_CHILD_NZ
               if leaf_dx_m <= tiles.LES_CHILD_DX_M else parent_nz)
    sizing = _Sizing(leaf_dx_m=leaf_dx_m, corridor_m=float(corridor_km) * 1000.0,
                     leaf_nz=leaf_nz, parent_nz=parent_nz,
                     budget=_SquareBudget(gib))
    if sizing.chain(0.0) is None:
        raise PartitionRefused(
            f"not even a one-site parent fits the {gib:g} GiB card at any "
            f"chain of {tiles.MAX_CHAIN_STEPS} or fewer x"
            f"{tiles.DEFAULT_STEP_RATIO} steps from {leaf_dx_m:g} m; use a "
            "larger --card/--vram-gib")

    arrays = sites.as_arrays()
    canonical = np.argsort(arrays["site_id"].astype(str), kind="stable")
    ids = arrays["site_id"][canonical]
    lon = arrays["lon"][canonical]
    lat = arrays["lat"][canonical]

    max_extent_m = float(max_extent_km) * 1000.0
    clusters: list[Cluster] = []
    split = 0
    stack: list[tuple[np.ndarray, int]] = [
        (part, part.size)
        for part in _components(np.arange(ids.size), lon, lat, linkage_km)]
    stack.reverse()
    while stack:
        index, component_sites = stack.pop()
        frame = _frame(lon[index], lat[index])
        extent = max(frame.extent_m)
        arc = _lon_arc(lon[index])
        fitted = (sizing.chain(extent)
                  if extent <= max_extent_m and arc is not None else None)
        if (index.size <= int(max_sites) and fitted is not None) \
                or index.size == 1:
            if fitted is None:
                raise PartitionRefused(
                    f"site {ids[index][0]} cannot be planned on the card")
            clusters.append(_cluster_record(
                index, frame, ids=ids, lon=lon, lat=lat, sizing=sizing,
                arc=arc, ratios=fitted[0], parent_cells=fitted[1],
                topology=topology, component_sites=component_sites))
            continue
        split += 1
        pieces = []
        for half in _bisect(index, frame):
            pieces.extend(_components(half, lon, lat, linkage_km))
        stack.extend((piece, component_sites) for piece in reversed(pieces))

    clusters.sort(key=lambda c: c.cluster_id)
    if len({c.cluster_id for c in clusters}) != len(clusters):
        raise PartitionRefused("two clusters hash to the same cluster_id; "
                               "raise CLUSTER_ID_HEX")
    notes = []
    if card_note:
        notes.append(card_note)
    notes.append(
        f"{len(clusters)} cluster(s) from {len(ids)} site(s): single linkage "
        f"at {float(linkage_km):g} km on the WRF sphere, {split} bisection(s) "
        f"to hold every cluster within {float(max_extent_km):g} km, "
        f"{int(max_sites)} sites and a parent chain that fits "
        f"{gib:g} GiB")
    crossing = sum(c.crosses_antimeridian for c in clusters)
    if crossing:
        notes.append(f"{crossing} cluster(s) cross the antimeridian; their "
                     "bbox is in a local longitude frame (lon_frame) and "
                     "their region is split at 180 degrees")
    notes.append("parent_chain and estimated_tiles are suggestions sized "
                 "by " + ESTIMATE_BASIS)
    return Partition(
        sites_ref=dict(sites_ref),
        parameters={"linkage_km": float(linkage_km),
                    "max_extent_km": float(max_extent_km),
                    "max_sites": int(max_sites), "leaf_dx_m": leaf_dx_m,
                    "corridor_km": float(corridor_km), "leaf_nz": leaf_nz,
                    "parent_nz": parent_nz,
                    "step_ratio": tiles.DEFAULT_STEP_RATIO,
                    "topology": topology},
        capacity=capacity, clusters=clusters, notes=notes)


# --------------------------------------------------------------------------
# planning one cluster


def _relpath(path: Path, start: Path) -> str:
    from woof.energy.plan_wrf_tiles import _relpath as relpath

    return relpath(path, start)


def load_partitioned_sites(partition: Partition) -> SiteSet:
    """The sites document the partition was built from, refused if its
    sha256 is not the one the partition recorded.  Load it once and pass
    it as ``sites=`` when planning many clusters."""

    source = partition.sites_path()
    digest = sha256_file(source)
    if digest != partition.sites_ref["sha256"]:
        raise ContractError(
            f"{source} has sha256 {digest}, not the "
            f"{partition.sites_ref['sha256']} the partition was built "
            "from; re-run woof energy partition")
    return load_sites(source)


def cluster_sites(partition: Partition | str | Path, cluster_id: str,
                  path: str | Path, *, sites: SiteSet | None = None
                  ) -> SiteSet:
    """The cluster's sites as a ``sites.v1`` set to be written at ``path``.

    Sites keep their order in the partitioned document; the provenance
    gains a ``partition`` record (path relative to ``path``'s directory,
    sha256, cluster id) and a relative assets path is rebased.  ``sites``
    is the partitioned document already loaded
    (:func:`load_partitioned_sites`); without it the document is loaded
    and its sha256 checked."""

    if not isinstance(partition, Partition):
        partition = load_partition(partition)
    cluster = partition.cluster(cluster_id)
    path = Path(path)
    source = partition.sites_path()
    if sites is None:
        sites = load_partitioned_sites(partition)
    wanted = set(cluster.site_ids)
    members = [site for site in sites.sites if site.site_id in wanted]
    if len(members) != len(wanted):
        found = {site.site_id for site in members}
        missing = sorted(wanted - found)[:5]
        raise ContractError(f"cluster {cluster_id} names site(s) {missing} "
                            f"the sites document {source} lacks")
    assets_ref = dict(sites.assets_ref) if sites.assets_ref else None
    if assets_ref and isinstance(assets_ref.get("path"), str) \
            and not Path(assets_ref["path"]).is_absolute():
        assets_ref["path"] = _relpath(source.parent / assets_ref["path"],
                                      path.parent)
    provenance = dict(sites.provenance)
    provenance["partition"] = {
        "path": (_relpath(partition.path, path.parent)
                 if partition.path is not None else None),
        "sha256": (sha256_file(partition.path)
                   if partition.path is not None
                   and Path(partition.path).exists() else None),
        "cluster_id": cluster_id}
    return SiteSet(sites=members, heights_m=sites.heights_m,
                   spacing_m=sites.spacing_m, assets_ref=assets_ref,
                   provenance=provenance)


def write_cluster_sites(partition: Partition | str | Path, cluster_id: str,
                        path: str | Path, *, sites: SiteSet | None = None
                        ) -> Path:
    """Write :func:`cluster_sites` at ``path`` (atomically)."""

    return dump_sites(cluster_sites(partition, cluster_id, path, sites=sites),
                      path)


def planner_capacity(partition: Partition, *, card: str | None = None,
                     vram_gib: float | None = None,
                     hardware_json: str | Path | None = None) -> dict:
    """The ``card``/``vram_gib`` keyword a planner takes: the caller's, or
    the card the partition was sized for."""

    if card is None and vram_gib is None and hardware_json is None:
        return {"vram_gib": float(partition.capacity["vram_gib"])}
    if card is not None and vram_gib is None and hardware_json is None:
        resolve_capacity(card=card)
        return {"card": card}
    gib, _, _ = resolve_capacity(card=card, vram_gib=vram_gib,
                                 hardware_json=hardware_json)
    return {"vram_gib": gib}


def plan_cluster(partition: Partition | str | Path, cluster_id: str, *,
                 outdir: str | Path, topology: str | None = None,
                 hardware_json: str | Path | None = None,
                 sites: SiteSet | None = None, **planner_kwargs) -> Plan:
    """Plan one cluster with its own parent; returns the written plan.

    ``sites`` is the partitioned document already loaded
    (:func:`load_partitioned_sites`), so a loop over many clusters reads
    and hashes it once.

    Writes ``outdir/sites.json`` (:func:`write_cluster_sites`) and runs the
    cluster's topology (or ``topology``) planner on it into ``outdir``.
    ``planner_kwargs`` go to the planner's ``build_plan`` (``start``,
    ``hours``, ``source``, ``card``, ``vram_gib``, ``max_domains``, ``nz``,
    ``parent_dx_m``, ``now`` ...); ``dx_m`` defaults to the cluster's leaf
    spacing and ``corridor_km`` to the partition's, and the card to the one
    the partition was sized for (``hardware_json`` is read for its
    capacity, as the partition reads it)."""

    if not isinstance(partition, Partition):
        partition = load_partition(partition)
    cluster = partition.cluster(cluster_id)
    topology = topology or cluster.topology
    if topology not in CLUSTER_TOPOLOGIES:
        raise PartitionRefused(f"topology {topology!r} is not one of "
                               f"{CLUSTER_TOPOLOGIES}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sites_path = outdir / "sites.json"
    subset = cluster_sites(partition, cluster_id, sites_path, sites=sites)
    dump_sites(subset, sites_path)
    kwargs = dict(planner_kwargs)
    capacity = planner_capacity(partition, card=kwargs.pop("card", None),
                                vram_gib=kwargs.pop("vram_gib", None),
                                hardware_json=hardware_json)
    kwargs.update(capacity)
    kwargs.setdefault("dx_m", cluster.leaf_dx_m)
    kwargs.setdefault("corridor_km",
                      float(partition.parameters["corridor_km"]))
    if topology == "wrf-tiles":
        from woof.energy import plan_wrf_tiles

        plan = plan_wrf_tiles.build_plan(subset, outdir=outdir, **kwargs)
    else:
        from woof.energy import plan_wrf_nests

        plan = plan_wrf_nests.build_plan(subset, outdir=outdir,
                                         sites_path=sites_path, **kwargs)
    plan.sites_ref = {"path": _relpath(sites_path, outdir),
                      "sha256": sha256_file(sites_path),
                      **(plan.sites_ref or {})}
    plan.notes.append(
        f"cluster {cluster_id} of partition "
        + (_relpath(partition.path, outdir) if partition.path is not None
           else "(unsaved)")
        + f": {len(cluster.site_ids)} site(s), {topology}, leaf "
        f"{float(kwargs['dx_m']):g} m, suggested parent chain "
        f"{cluster.parent_chain['ratios']}")
    dump_plan(plan, outdir / "plan.json")
    return plan


# --------------------------------------------------------------------------
# CLI


def _summary(partition: Partition, path: Path) -> dict:
    clusters = partition.clusters
    sizes = sorted((len(c.site_ids) for c in clusters), reverse=True)
    return {
        "schema": PARTITION_SCHEMA,
        "partition": str(path),
        "clusters": len(clusters),
        "sites": partition.site_count,
        "largest_cluster_sites": sizes[0] if sizes else 0,
        "crossing_antimeridian": sum(c.crosses_antimeridian for c in clusters),
        "topologies": {t: sum(c.topology == t for c in clusters)
                       for t in CLUSTER_TOPOLOGIES},
        "estimated_leaf_tiles": sum(c.estimated_tiles["leaf"]
                                    for c in clusters),
        "estimated_intermediate_tiles": sum(
            c.estimated_tiles["intermediate"] for c in clusters),
        "capacity": partition.capacity,
        "cluster_ids": [c.cluster_id for c in clusters],
        "notes": partition.notes,
    }


def main(args) -> int:
    """``woof energy partition``."""

    import sys

    sites_path = Path(args.sites)
    output = Path(args.output)
    try:
        sites = load_sites(sites_path)
        partition = build_partition(
            sites,
            sites_ref={"path": _relpath(sites_path, output.parent),
                       "sha256": sha256_file(sites_path)},
            linkage_km=args.linkage_km, max_extent_km=args.max_extent_km,
            max_sites=args.max_sites, leaf_dx_m=args.leaf_dx_m,
            corridor_km=args.corridor_km, nz=args.nz,
            topology=args.topology, card=args.card, vram_gib=args.vram_gib,
            hardware_json=args.hardware_json)
        dump_partition(partition, output)
    except (OSError, ValueError) as error:
        # ValueError: ContractError, PartitionRefused and the geometry and
        # wizard helpers' own refusals.
        print(f"woof energy partition: refused ({type(error).__name__}): "
              f"{error}", file=sys.stderr)
        return 2
    for note in partition.notes:
        print(f"woof energy partition: {note}", file=sys.stderr)
    print(json.dumps(_summary(partition, output), indent=2))
    return 0


def plan_cluster_main(args) -> int:
    """``woof energy plan-cluster``."""

    import contextlib
    import sys

    outdir = Path(args.outdir)
    try:
        partition = load_partition(args.partition)
        cluster = partition.cluster(args.cluster)
        topology = args.topology or cluster.topology
        planner_kwargs = {
            "start": args.start, "hours": args.hours, "source": args.source,
            "max_domains": args.max_domains, "nz": args.nz,
            "parent_dx_m": args.parent_dx_m,
            "dx_m": args.dx_m if args.dx_m is not None else cluster.leaf_dx_m,
            "corridor_km": (args.corridor_km if args.corridor_km is not None
                            else float(partition.parameters["corridor_km"])),
        }
        capacity = planner_capacity(partition, card=args.card,
                                    vram_gib=args.vram_gib,
                                    hardware_json=args.hardware_json)
        if args.dry_run:
            sites_path = write_cluster_sites(partition, args.cluster,
                                             outdir / "sites.json")
            print(json.dumps({
                "command": "energy plan-cluster", "dry_run": True,
                "cluster_id": cluster.cluster_id, "topology": topology,
                "sites": str(sites_path), "site_count": len(cluster.site_ids),
                "planner": f"woof.energy.plan_{topology.replace('-', '_')}"
                           ".build_plan",
                "planner_kwargs": {**planner_kwargs, **capacity,
                                   "outdir": str(outdir)},
                "parent_chain": cluster.parent_chain,
                "estimated_tiles": cluster.estimated_tiles,
                "region": cluster.region,
            }, indent=2, default=str))
            return 0
        with contextlib.redirect_stdout(sys.stderr):
            plan = plan_cluster(partition, args.cluster, outdir=outdir,
                                topology=topology,
                                **planner_kwargs, **capacity)
    except (OSError, ValueError, EnergyNotImplemented) as error:
        # ValueError: ContractError, PartitionRefused and the planners' own
        # refusals (TilePlanRefusal, PlanRefused, the wizard's ValueErrors).
        print(f"woof energy plan-cluster: refused ({type(error).__name__}): "
              f"{error}", file=sys.stderr)
        print(json.dumps({"command": "energy plan-cluster",
                          "cluster_id": args.cluster,
                          "refused": type(error).__name__,
                          "reason": str(error)}, indent=2))
        return 2
    print(json.dumps({
        "command": "energy plan-cluster",
        "cluster_id": cluster.cluster_id,
        "topology": plan.topology,
        "plan": str(outdir / "plan.json"),
        "sites": str(outdir / "sites.json"),
        "site_count": len(cluster.site_ids),
        "dx_m": plan.dx_m, "start": plan.start, "hours": plan.hours,
        "domains": len(plan.domains),
        "leaf_domains": sum(d.role == "child" for d in plan.domains),
        "notes": plan.notes,
    }, indent=2, default=str))
    return 0


__all__ = [
    "PARTITION_SCHEMA", "DEFAULT_LINKAGE_KM", "DEFAULT_MAX_EXTENT_KM",
    "DEFAULT_MAX_SITES", "DEFAULT_LEAF_DX_M", "LEAF_DX_MIN_M",
    "LEAF_DX_MAX_M", "CLUSTER_TOPOLOGIES", "TOPOLOGY_CHOICES",
    "PartitionRefused", "Cluster", "Partition", "cluster_id_for",
    "linkage_labels", "build_partition", "load_partition", "dump_partition",
    "resolve_capacity", "load_partitioned_sites", "cluster_sites",
    "write_cluster_sites", "plan_cluster", "main",
    "plan_cluster_main",
]
