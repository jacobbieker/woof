"""Geodesic helpers for grid assets.

Small-vector geometry on the WRF sphere
(:data:`woof.static.projection.EARTH_RADIUS_M`): local projections,
polyline densification and bearings, corridor buffers, point-in-polygon and
rectangle covers.  Coordinates are ``(lon, lat)`` degrees in and out unless
a function says it works in projected metres.

WHY THE WRF SPHERE.  The planners place 50-100 m WRF domains around the
points this module produces, and the model grids those domains become are
laid out by :mod:`woof.static.projection` on a sphere of radius
``EARTH_RADIUS_M`` (6 370 km, ``module_llxy.F``).  Measuring corridors and
chainages on any other figure would put a site a few metres from where the
grid thinks it is at 100 km from the projection centre.

WHY NUMPY HERE.  ``docs/dev/static-rust-port.md`` puts bulk data-path array
math in Rust.  Everything here works on per-asset or per-site vectors (a
national grid sampled every 100 m is a few hundred thousand points, touched
once at planning time), not on model fields, so vectorized numpy is the
right tool and nothing here runs per time step.

What each piece promises:

* :class:`LocalProjection` is the spherical azimuthal-equidistant
  projection, written in unit-vector form so it is exact at the centre and
  round-trips to well under a millimetre at 500 km.
* :func:`densify_polyline` samples at uniform great-circle chainage; the
  original vertices are not kept (they are not at multiples of the spacing),
  both ends are.
* :func:`segment_bearings` differences unit vectors in 3-D and reads the
  azimuth in the local tangent plane, so it is free of the longitude
  wrap-around and of the cos(latitude) factor.
* :func:`buffer_polyline` contours the distance field of the polyline on a
  fine lattice in a local projection (marching squares with linear
  interpolation), then simplifies the contour.  This is robust where an
  offset-curve construction is not: hairpins, self-approaching and
  self-crossing lines, closed loops (which produce holes) all come out as
  valid rings.  ``woof.hex.swath.geometry.swath_ring`` was not reused
  because it refuses exactly those shapes (it is built for smooth storm
  tracks on a 6 371 km sphere).
* :func:`point_in_polygon` is the even-odd rule, banded so a 250 000-point
  query against a several-thousand-vertex corridor stays fast.
* :func:`cover_with_rectangles` is a deterministic strip greedy cover with
  a merge pass, run along both axes, keeping the smaller cover.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Sequence

import numpy as np

from woof.static.projection import EARTH_RADIUS_M

LonLat = tuple[float, float]
#: A polygon: first ring is the outer boundary, any further rings are holes.
#: Rings are closed lists of (lon, lat).
Polygon = list[list[LonLat]]

#: Consecutive vertices closer than this (metres) are treated as one.
DUPLICATE_TOLERANCE_M = 1.0e-3

#: :func:`buffer_polyline` lattice step as a fraction of the half-width.
BUFFER_LATTICE_FRACTION = 1.0 / 8.0

#: :func:`buffer_polyline` output edges are no longer than this many
#: half-widths (straight runs are split; see the function).
BUFFER_MAX_EDGE_FRACTION = 2.0

#: :func:`buffer_polyline` refuses when the corridor would reach farther
#: than this from its projection centre; past it the azimuthal-equidistant
#: scale error (about 1.6 % at 2 000 km) would have to be absorbed into the
#: corridor width.
BUFFER_MAX_EXTENT_M = 2.0e6

#: :func:`buffer_polyline` refuses when its lattice would need more nodes
#: than this (a very long line with a very narrow corridor).
BUFFER_MAX_NODES = 60_000_000

#: :func:`cover_with_rectangles` runs the merge pass only up to this many
#: rectangles; above it the pairwise search is not worth its cost.
COVER_MERGE_LIMIT = 1500


class GeometryError(ValueError):
    """An input that this module cannot give a correct answer for."""


def _real(value) -> bool:
    """A finite real number that is not a bool."""

    return (isinstance(value, numbers.Real)
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def _positive(value) -> bool:
    return _real(value) and value > 0.0


# ---------------------------------------------------------------------------
# unit-vector primitives


def _unit_vectors(lon, lat) -> np.ndarray:
    lam = np.radians(np.asarray(lon, dtype=float))
    phi = np.radians(np.asarray(lat, dtype=float))
    cos_phi = np.cos(phi)
    return np.stack((cos_phi * np.cos(lam), cos_phi * np.sin(lam),
                     np.sin(phi)), axis=-1)


def _lonlat(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    return lon, lat


def _fold_lon(lon):
    """Longitude folded into [-180, 180); values already there unchanged."""

    lon = np.asarray(lon, dtype=float)
    return np.where((lon >= -180.0) & (lon < 180.0), lon,
                    (lon + 180.0) % 360.0 - 180.0)


def _east_north(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local east and north unit vectors at each unit vector."""

    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    lam = np.arctan2(y, x)
    phi = np.arctan2(z, np.hypot(x, y))
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.stack((-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam),
                      np.cos(phi)), axis=-1)
    return east, north


def _arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Central angle between unit vectors (radians), stable at small angles."""

    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def _slerp(a: np.ndarray, b: np.ndarray, omega: np.ndarray,
           t: np.ndarray) -> np.ndarray:
    """Great-circle interpolation; ``omega`` is the arc between ``a`` and
    ``b``.  Zero-length arcs return ``a``."""

    omega = omega[..., None]
    t = t[..., None]
    sin_omega = np.sin(omega)
    small = sin_omega < 1e-15
    safe = np.where(small, 1.0, sin_omega)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * omega) / safe)
    wb = np.where(small, t, np.sin(t * omega) / safe)
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _coords_array(coords: Sequence[LonLat], what: str) -> np.ndarray:
    try:
        arr = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{what}: coordinates are not numeric pairs") from exc
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 1:
        raise GeometryError(
            f"{what}: expected a sequence of (lon, lat) pairs, got shape "
            f"{arr.shape}")
    arr = arr[:, :2]
    if not np.all(np.isfinite(arr)):
        raise GeometryError(f"{what}: coordinates contain non-finite values")
    if np.any(np.abs(arr[:, 1]) > 90.0):
        raise GeometryError(f"{what}: latitude outside [-90, 90]")
    return arr


def _distinct(vectors: np.ndarray) -> np.ndarray:
    """Indices of the vertices kept when every vertex closer than
    :data:`DUPLICATE_TOLERANCE_M` to the last kept one is dropped.

    Distance is measured from the last KEPT vertex, so a long chain of
    sub-millimetre steps still keeps a vertex every millimetre instead of
    collapsing to one point.
    """

    if len(vectors) == 1:
        return np.array([0])
    step = _arc(vectors[:-1], vectors[1:]) * EARTH_RADIUS_M
    if np.all(step >= DUPLICATE_TOLERANCE_M):
        return np.arange(len(vectors))
    keep = [0]
    tol = DUPLICATE_TOLERANCE_M / EARTH_RADIUS_M
    for i in range(1, len(vectors)):
        if keep[-1] == i - 1 and step[i - 1] >= DUPLICATE_TOLERANCE_M:
            keep.append(i)
        elif float(_arc(vectors[keep[-1]], vectors[i])) >= tol:
            keep.append(i)
    return np.asarray(keep)


#: Legs closer than this (radians) to half a great circle are refused.
_ANTIPODAL_RAD = 1.0e-6


def _legs(vectors: np.ndarray, what: str) -> np.ndarray:
    """Arc of each leg, refusing (near-)antipodal legs, whose great circle
    is not unique."""

    arcs = _arc(vectors[:-1], vectors[1:])
    if np.any(arcs > math.pi - _ANTIPODAL_RAD):
        raise GeometryError(
            f"{what}: two consecutive vertices are (nearly) antipodal, so the "
            "great circle between them is not unique; add a vertex between "
            "them")
    return arcs


# ---------------------------------------------------------------------------
# projection


@dataclass(frozen=True)
class LocalProjection:
    """Azimuthal-equidistant projection about ``(lat0, lon0)``, metres."""

    lat0: float
    lon0: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lat0) and math.isfinite(self.lon0)):
            raise GeometryError(
                f"projection centre ({self.lat0}, {self.lon0}) is not finite")
        if abs(self.lat0) > 90.0:
            raise GeometryError(
                f"projection centre latitude {self.lat0} outside [-90, 90]")

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centre = _unit_vectors(self.lon0, self.lat0)
        east, north = _east_north(centre)
        return centre, east, north

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) degrees -> (x east, y north) metres."""

        centre, east, north = self._basis()
        p = _unit_vectors(lon, lat)
        xe = p @ east
        yn = p @ north
        sin_c = np.hypot(xe, yn)
        c = np.arctan2(sin_c, p @ centre)
        # c / sin(c), -> 1 at the centre.
        scale = np.where(sin_c > 1e-300, c / np.where(sin_c > 1e-300, sin_c, 1.0),
                         1.0)
        return (EARTH_RADIUS_M * scale * xe, EARTH_RADIUS_M * scale * yn)

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) metres -> (lon, lat) degrees."""

        centre, east, north = self._basis()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        c = np.hypot(x, y) / EARTH_RADIUS_M
        sinc = np.sinc(c / np.pi)  # sin(c)/c
        p = (np.cos(c)[..., None] * centre
             + (sinc / EARTH_RADIUS_M)[..., None]
             * (x[..., None] * east + y[..., None] * north))
        return _lonlat(p)


def local_projection(lat0: float, lon0: float) -> LocalProjection:
    """The :class:`LocalProjection` about ``(lat0, lon0)``."""

    return LocalProjection(float(lat0), float(lon0))


def _centre_of(vectors: np.ndarray) -> LocalProjection:
    """Projection about the normalized mean of unit vectors (pass evenly
    resampled vertices so the mean is weighted by length)."""

    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        raise GeometryError(
            "the vertices are spread round the globe; there is no local "
            "projection centre for them")
    lon, lat = _lonlat(mean / norm)
    return local_projection(float(lat), float(lon))


# ---------------------------------------------------------------------------
# polylines


def densify_polyline(coords: Sequence[LonLat], spacing_m: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ``spacing_m`` along a polyline, both ends included.

    Returns ``(lon, lat, chainage_m)``; the last step may be shorter.

    Chainage is great-circle distance along the polyline on the WRF sphere.
    Interior samples sit at exact multiples of ``spacing_m``, so original
    interior vertices are generally not among them.  Consecutive duplicate
    vertices are dropped.  A polyline of zero length (one point, or every
    vertex the same) returns that single point at chainage 0.  The first
    and last samples are the input end vertices exactly, with longitudes
    folded into [-180, 180) like every other sample.
    """

    if not _positive(spacing_m):
        raise GeometryError(f"densify spacing must be positive, not {spacing_m!r}")
    arr = _coords_array(coords, "densify_polyline")
    vectors = _unit_vectors(arr[:, 0], arr[:, 1])
    keep = _distinct(vectors)
    vectors = vectors[keep]
    if len(vectors) < 2:
        return (np.array([_fold_lon(arr[0, 0])]), np.array([arr[0, 1]]),
                np.array([0.0]))
    arcs = _legs(vectors, "densify_polyline")
    lengths = arcs * EARTH_RADIUS_M
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    count = max(1, int(math.ceil(total / spacing_m - 1e-9)))
    chainage = np.concatenate((spacing_m * np.arange(count), [total]))
    segment = np.clip(np.searchsorted(cumulative, chainage, side="right") - 1,
                      0, len(lengths) - 1)
    t = np.clip((chainage - cumulative[segment]) / lengths[segment], 0.0, 1.0)
    points = _slerp(vectors[segment], vectors[segment + 1], arcs[segment], t)
    lon, lat = _lonlat(points)
    lon[0], lat[0] = _fold_lon(arr[0, 0]), arr[0, 1]
    lon[-1], lat[-1] = _fold_lon(arr[-1, 0]), arr[-1, 1]
    return lon, lat, chainage


def segment_bearings(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Azimuth of the polyline at each point, degrees clockwise from north
    in [0, 360); centred differences inside, one-sided at the ends.

    The difference is taken between unit vectors in 3-D and resolved into
    the local east/north plane, so it is the great-circle azimuth.  Runs of
    coincident points share one bearing.  Where a centred difference
    vanishes (the line doubles straight back on itself) the forward
    one-sided difference is used.  Fewer than two distinct points have no
    direction and are refused.
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise GeometryError("segment_bearings: lon and lat must be 1-D arrays "
                            "of the same length")
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        raise GeometryError("segment_bearings: non-finite coordinates")
    if lon.size == 0:
        raise GeometryError("segment_bearings: no points")
    vectors = _unit_vectors(lon, lat)
    starts = _distinct(vectors)
    if len(starts) < 2:
        raise GeometryError(
            "segment_bearings: fewer than two distinct points, so the line "
            "has no direction")
    q = vectors[starts]
    m = len(q)
    tangent = np.empty_like(q)
    tangent[0] = q[1] - q[0]
    tangent[-1] = q[-1] - q[-2]
    forward = np.empty_like(q)
    forward[:-1] = q[1:] - q[:-1]
    forward[-1] = tangent[-1]
    if m > 2:
        tangent[1:-1] = q[2:] - q[:-2]
    east, north = _east_north(q)
    te = np.sum(tangent * east, axis=-1)
    tn = np.sum(tangent * north, axis=-1)
    reverse = np.hypot(te, tn) < 1e-12
    if np.any(reverse):
        te = np.where(reverse, np.sum(forward * east, axis=-1), te)
        tn = np.where(reverse, np.sum(forward * north, axis=-1), tn)
    bearing = np.degrees(np.arctan2(te, tn)) % 360.0
    bearing = np.where(bearing >= 360.0, 0.0, bearing)
    run = np.cumsum(np.isin(np.arange(len(lon)), starts)) - 1
    return bearing[run]


def _resample_keep_vertices(vectors: np.ndarray, step_m: float) -> np.ndarray:
    """Densify so no leg exceeds ``step_m``, keeping every vertex."""

    if len(vectors) < 2:
        return vectors
    arcs = _legs(vectors, "buffer_polyline")
    pieces = np.maximum(1, np.ceil(arcs * EARTH_RADIUS_M / step_m)).astype(int)
    segment = np.repeat(np.arange(len(arcs)), pieces)
    offset = np.arange(len(segment)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    t = offset / pieces[segment]
    out = _slerp(vectors[segment], vectors[segment + 1], arcs[segment], t)
    return np.concatenate((out, vectors[-1:]), axis=0)


# ---------------------------------------------------------------------------
# corridor buffer


def _build_case_table() -> dict[tuple[int, bool], list[tuple[int, int]]]:
    """Oriented marching-squares segments, interior on the left.

    Corners are numbered counter-clockwise (0 lower-left, 1 lower-right,
    2 upper-right, 3 upper-left); edge ``k`` joins corner ``k`` to corner
    ``k+1``.  Walking the cell boundary counter-clockwise, an edge that
    goes inside->outside is an exit and outside->inside an entry.  A
    segment runs from each exit to the nearest entry clockwise, except in a
    saddle whose centre is inside, where it runs to the nearest entry
    counter-clockwise (the two inside corners are joined).
    """

    table: dict[tuple[int, bool], list[tuple[int, int]]] = {}
    for case in range(16):
        inside = [bool(case >> k & 1) for k in range(4)]
        exits = [k for k in range(4) if inside[k] and not inside[(k + 1) % 4]]
        entries = {k for k in range(4)
                   if not inside[k] and inside[(k + 1) % 4]}
        for centre_inside in (False, True):
            segs = []
            for k in exits:
                step = 1 if (centre_inside and case in (5, 10)) else -1
                m = (k + step) % 4
                while m not in entries:
                    m = (m + step) % 4
                segs.append((k, m))
            table[(case, centre_inside)] = segs
    return table


_CASES = _build_case_table()


def _segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    safe = np.where(length2 > 0.0, length2, 1.0)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / safe, 0.0, 1.0)
    t = np.where(length2 > 0.0, t, 0.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _distance_nodes(xy: np.ndarray, step: float, reach: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lattice nodes (integer i, j at ``step``) within ``reach`` of the
    projected polyline ``xy`` and their distance to it."""

    a = xy[:-1] if len(xy) > 1 else xy
    b = xy[1:] if len(xy) > 1 else xy
    lo_i = np.floor((np.minimum(a[:, 0], b[:, 0]) - reach) / step).astype(np.int64)
    hi_i = np.ceil((np.maximum(a[:, 0], b[:, 0]) + reach) / step).astype(np.int64)
    lo_j = np.floor((np.minimum(a[:, 1], b[:, 1]) - reach) / step).astype(np.int64)
    hi_j = np.ceil((np.maximum(a[:, 1], b[:, 1]) + reach) / step).astype(np.int64)
    ni = int(np.max(hi_i - lo_i)) + 1
    nj = int(np.max(hi_j - lo_j)) + 1
    total = len(a) * ni * nj
    if total > BUFFER_MAX_NODES:
        raise GeometryError(
            f"buffer_polyline would evaluate {total:,} lattice nodes (limit "
            f"{BUFFER_MAX_NODES:,}); the corridor is too narrow for the line's "
            "length. Split the line or widen the corridor")
    oi, oj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    oi = oi.ravel()
    oj = oj.ravel()
    chunk = max(1, 4_000_000 // (ni * nj))
    keys_i, keys_j, values = [], [], []
    for s in range(0, len(a), chunk):
        sl = slice(s, s + chunk)
        ii = lo_i[sl, None] + oi[None, :]
        jj = lo_j[sl, None] + oj[None, :]
        valid = (ii <= hi_i[sl, None]) & (jj <= hi_j[sl, None])
        d = _segment_distance(ii * step, jj * step, a[sl, 0:1], a[sl, 1:2],
                              b[sl, 0:1], b[sl, 1:2])
        mask = valid & (d <= reach)
        keys_i.append(ii[mask])
        keys_j.append(jj[mask])
        values.append(d[mask])
    ii = np.concatenate(keys_i)
    jj = np.concatenate(keys_j)
    dd = np.concatenate(values)
    i0, j0 = ii.min() - 2, jj.min() - 2
    width = int(jj.max() - j0) + 4
    key = (ii - i0) * width + (jj - j0)
    order = np.lexsort((dd, key))
    key = key[order]
    dd = dd[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    key = key[first]
    dd = dd[first]
    return key, dd, np.array([i0, j0, width], dtype=np.int64)


def _contour_rings(key: np.ndarray, value: np.ndarray, frame: np.ndarray,
                   level: float, step: float) -> list[np.ndarray]:
    """Marching-squares rings of ``value < level`` on the sparse lattice,
    interior on the left (outer rings counter-clockwise, holes clockwise)."""

    i0, j0, width = (int(v) for v in frame)
    inside_keys = key[value < level]
    if inside_keys.size == 0:
        return []
    far = level + 4.0 * step

    def lookup(k: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(key, k)
        pos_c = np.minimum(pos, len(key) - 1)
        found = key[pos_c] == k
        return np.where(found, value[pos_c], far)

    # Every cell that has an inside corner (lower-left node key).
    cells = np.unique(np.concatenate((inside_keys, inside_keys - 1,
                                      inside_keys - width,
                                      inside_keys - width - 1)))
    corner_keys = np.stack((cells, cells + width, cells + width + 1, cells + 1),
                           axis=1)  # ll, lr, ur, ul (i is the slow axis)
    corner_vals = lookup(corner_keys)
    inside = corner_vals < level
    case = (inside * (1 << np.arange(4))).sum(axis=1)
    centre_inside = corner_vals.mean(axis=1) < level
    active = (case != 0) & (case != 15)
    cells, corner_keys, corner_vals = cells[active], corner_keys[active], corner_vals[active]
    case, centre_inside = case[active], centre_inside[active]

    # Edge ids: horizontal (x-direction) edge from node n is 2n, vertical
    # (y-direction) edge from node n is 2n+1.  Cell edges: 0 bottom (ll->lr,
    # horizontal from ll), 1 right (lr->ur, vertical from lr), 2 top
    # (ul->ur, horizontal from ul), 3 left (ll->ul, vertical from ll).
    edge_ids = np.stack((2 * corner_keys[:, 0], 2 * corner_keys[:, 1] + 1,
                         2 * corner_keys[:, 3], 2 * corner_keys[:, 0] + 1), axis=1)
    edge_a = np.array([0, 1, 3, 0])  # corner index of each edge's base node
    edge_b = np.array([1, 2, 2, 3])
    src, dst = [], []
    for (c, cin), segs in _CASES.items():
        if not segs:
            continue
        sel = np.flatnonzero((case == c) & (centre_inside == cin))
        if sel.size == 0:
            continue
        for e_from, e_to in segs:
            src.append(np.stack((sel, np.full(sel.size, e_from)), axis=1))
            dst.append(np.stack((sel, np.full(sel.size, e_to)), axis=1))
    src_ce = np.concatenate(src)
    dst_ce = np.concatenate(dst)
    src_id = edge_ids[src_ce[:, 0], src_ce[:, 1]]
    dst_id = edge_ids[dst_ce[:, 0], dst_ce[:, 1]]

    # Crossing point of each source edge.
    va = corner_vals[src_ce[:, 0], edge_a[src_ce[:, 1]]]
    vb = corner_vals[src_ce[:, 0], edge_b[src_ce[:, 1]]]
    t = np.clip((level - va) / (vb - va), 0.0, 1.0)
    base = corner_keys[src_ce[:, 0], edge_a[src_ce[:, 1]]]
    bi = base // width + i0
    bj = base % width + j0
    horizontal = (src_id % 2) == 0
    px = (bi + np.where(horizontal, t, 0.0)) * step
    py = (bj + np.where(horizontal, 0.0, t)) * step

    order = np.argsort(src_id, kind="stable")
    sorted_src = src_id[order]
    pos = np.searchsorted(sorted_src, dst_id)
    if np.any(pos >= len(sorted_src)) or np.any(sorted_src[np.minimum(pos, len(sorted_src) - 1)] != dst_id):
        raise GeometryError("buffer_polyline: contour did not close (internal "
                            "error)")
    nxt = order[pos]
    visited = np.zeros(len(src_id), dtype=bool)
    rings: list[np.ndarray] = []
    for start in range(len(src_id)):
        if visited[start]:
            continue
        idx = []
        k = start
        while not visited[k]:
            visited[k] = True
            idx.append(k)
            k = int(nxt[k])
        if k != start:
            raise GeometryError("buffer_polyline: contour did not close "
                                "(internal error)")
        rings.append(np.stack((px[idx], py[idx]), axis=1))
    return rings


def _simplify_open(xy: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker keep-mask for an open chain (ends kept)."""

    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        inner = xy[s + 1:e]
        d = _segment_distance(inner[:, 0], inner[:, 1], xy[s, 0], xy[s, 1],
                              xy[e, 0], xy[e, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = s + 1 + k
            keep[m] = True
            stack.append((s, m))
            stack.append((m, e))
    return keep


def _simplify_closed(xy: np.ndarray, tol: float) -> np.ndarray:
    if len(xy) <= 3:
        return xy
    far = int(np.argmax(np.hypot(xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1])))
    if far == 0:
        return xy[:1]
    first = _simplify_open(xy[:far + 1], tol)
    second = _simplify_open(np.concatenate((xy[far:], xy[:1])), tol)
    keep = np.zeros(len(xy), dtype=bool)
    keep[:far + 1] |= first
    keep[far:] |= second[:-1]
    return xy[keep]


def _split_long_edges(xy: np.ndarray, max_len: float) -> np.ndarray:
    """Insert evenly spaced vertices so no closed-ring edge exceeds
    ``max_len``."""

    nxt = np.roll(xy, -1, axis=0)
    length = np.hypot(nxt[:, 0] - xy[:, 0], nxt[:, 1] - xy[:, 1])
    pieces = np.maximum(1, np.ceil(length / max_len)).astype(int)
    if np.all(pieces == 1):
        return xy
    edge = np.repeat(np.arange(len(xy)), pieces)
    t = (np.arange(len(edge)) - np.repeat(np.cumsum(pieces) - pieces, pieces)) \
        / pieces[edge]
    return xy[edge] + t[:, None] * (nxt[edge] - xy[edge])


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def buffer_polyline(coords: Sequence[LonLat], half_width_m: float, *,
                    segments_per_quarter: int = 4) -> Polygon:
    """Corridor polygon within ``half_width_m`` of the polyline.

    The returned polygon contains every point within ``half_width_m`` of the
    polyline (great-circle distance) and nothing farther than about 1.1x
    ``half_width_m``.  Most of the boundary lies within 1.05x; the worst
    case is where the corridor's own edges meet on the inside of a bend,
    where the contour cuts the corner (measured at most 1.065x for
    ``segments_per_quarter=8``, 1.09x for the default 4 and 1.097x for 3 on
    several hundred random polylines).  Below 3 the simplification
    tolerance alone, ``half_width_m * (1 / cos(pi / (4 *
    segments_per_quarter)) - 1)``, pushes past 1.1x.  Ends and joins are
    round; ``segments_per_quarter`` sets the vertex density on the round
    parts (a quarter circle is drawn with about that many chords).

    The first ring is the outer boundary, counter-clockwise; further rings
    are holes (a closed loop or a line that wraps round), clockwise.  Rings
    are closed (first vertex repeated).

    Method: in an azimuthal-equidistant projection about the line's centre,
    the distance to the (densified) line is evaluated on a lattice of step
    ``half_width_m / 8`` near the line, contoured by marching squares at a
    level just above the half-width, and the contour is simplified by
    Douglas-Peucker.  The level absorbs the contour, simplification and
    projection-scale errors, so containment holds by construction.  Edges
    are then split to at most two half-widths, because GeoJSON (and
    :func:`point_in_polygon`) reads an edge as straight in (lon, lat), and
    a long edge straight in the projection bows away from that.

    Limitations: holes smaller than the simplification tolerance (a few
    per cent of the half-width) are dropped, which only adds area within
    that tolerance of the corridor.  Corridors reaching more than
    :data:`BUFFER_MAX_EXTENT_M` from their centre, or needing more than
    :data:`BUFFER_MAX_NODES` lattice nodes, are refused.
    """

    if not _positive(half_width_m):
        raise GeometryError(f"buffer half-width must be positive, not {half_width_m!r}")
    if (isinstance(segments_per_quarter, bool)
            or not isinstance(segments_per_quarter, (int, np.integer))
            or segments_per_quarter < 1):
        raise GeometryError(
            f"segments_per_quarter must be a positive integer, not "
            f"{segments_per_quarter!r}")
    arr = _coords_array(coords, "buffer_polyline")
    vectors = _unit_vectors(arr[:, 0], arr[:, 1])
    vectors = vectors[_distinct(vectors)]
    # Short legs so a projected chord follows its great circle and every
    # leg's lattice box stays small.
    hw = float(half_width_m)
    vectors = _resample_keep_vertices(vectors, hw)
    proj = _centre_of(vectors)
    lon, lat = _lonlat(vectors)
    x, y = proj.forward(lon, lat)
    xy = np.stack((x, y), axis=1)

    tol_frac = 1.0 / math.cos(math.pi / (4.0 * segments_per_quarter)) - 1.0
    step = hw * BUFFER_LATTICE_FRACTION
    # Farthest the lattice reaches from the centre, using a scale bound
    # (1.02) above the true one at BUFFER_MAX_EXTENT_M (1.0164).
    reach_bound = 1.02 * hw * (1.0 + 1.0 / 128.0 + tol_frac) + 2.0 * step
    rho = float(np.max(np.hypot(x, y))) + reach_bound
    if rho > BUFFER_MAX_EXTENT_M:
        raise GeometryError(
            f"buffer_polyline: the corridor reaches {rho / 1000:.0f} km from "
            f"its projection centre (limit {BUFFER_MAX_EXTENT_M / 1000:.0f} km); "
            "split the line")
    for pole_lat in (90.0, -90.0):
        px, py = proj.forward(0.0, pole_lat)
        if float(np.min(_segment_distance(
                float(px), float(py), xy[:-1, 0] if len(xy) > 1 else xy[:, 0],
                xy[:-1, 1] if len(xy) > 1 else xy[:, 1],
                xy[1:, 0] if len(xy) > 1 else xy[:, 0],
                xy[1:, 1] if len(xy) > 1 else xy[:, 1]))) <= reach_bound:
            raise GeometryError(
                "buffer_polyline: the corridor reaches a pole, where a ring "
                "of (lon, lat) vertices with straight edges cannot describe "
                "it")
    # Azimuthal scale factor at the corridor's far edge: projected distances
    # exceed true ones by at most this factor, so widen by it.
    c = rho / EARTH_RADIUS_M
    scale = c / math.sin(c) if c > 0 else 1.0
    hw_eff = hw * scale
    tol = hw_eff * tol_frac
    level = hw_eff * (1.0 + 1.0 / 128.0) + tol

    key, value, frame = _distance_nodes(xy, step, level + 2.0 * step)
    rings = _contour_rings(key, value, frame, level, step)
    simplified = []
    for ring in rings:
        ring = _simplify_closed(ring, tol)
        if len(ring) >= 3:
            simplified.append(ring)
    outers = [r for r in simplified if _signed_area(r) > 0.0]
    holes = [r for r in simplified if _signed_area(r) < 0.0]
    if len(outers) != 1:
        raise GeometryError(
            f"buffer_polyline: expected one outer ring, found {len(outers)} "
            "(internal error)")
    holes.sort(key=lambda r: (float(r[:, 0].min()), float(r[:, 1].min())))
    polygon: Polygon = []
    for ring in [outers[0]] + holes:
        # GeoJSON edges are straight in (lon, lat), not in the projection;
        # the two differ by about s**2 tan(lat) / (8 R) over an edge of
        # length s, so long straight runs are split to keep that well
        # inside the level's 1/128 allowance.
        ring = _split_long_edges(ring, BUFFER_MAX_EDGE_FRACTION * hw)
        rlon, rlat = proj.inverse(ring[:, 0], ring[:, 1])
        pts = [(float(a), float(b)) for a, b in zip(rlon, rlat)]
        pts.append(pts[0])
        polygon.append(pts)
    return polygon


# ---------------------------------------------------------------------------
# point in polygon


def _ring_edges(ring, ref_lon: float) -> np.ndarray:
    arr = np.asarray(ring, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        raise GeometryError("point_in_polygon: a ring needs at least three "
                            "(lon, lat) vertices")
    arr = arr[:, :2].copy()
    if not np.all(np.isfinite(arr)):
        raise GeometryError("point_in_polygon: ring has non-finite vertices")
    arr[:, 0] = (arr[:, 0] - ref_lon + 180.0) % 360.0 - 180.0 + ref_lon
    closed = np.concatenate((arr, arr[:1]), axis=0)
    return np.concatenate((closed[:-1], closed[1:]), axis=1)  # x0 y0 x1 y1


def point_in_polygon(lon: np.ndarray, lat: np.ndarray,
                     polygons: Sequence[Polygon]) -> np.ndarray:
    """Boolean mask of points inside any polygon (holes excluded).

    The even-odd rule over all of a polygon's rings, with edges straight in
    (lon, lat) as GeoJSON specifies.  Longitudes are unwrapped about each
    polygon's first vertex, so a polygon across the antimeridian works if
    its vertices are continuous there.  Rings may be given closed or open.
    Points exactly on an edge may fall either way.
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape != lat.shape:
        raise GeometryError("point_in_polygon: lon and lat shapes differ")
    shape = lon.shape
    lon = lon.ravel()
    lat = lat.ravel()
    result = np.zeros(lon.shape, dtype=bool)
    finite = np.isfinite(lon) & np.isfinite(lat)
    for polygon in polygons:
        if not polygon:
            raise GeometryError("point_in_polygon: empty polygon")
        ref = float(np.asarray(polygon[0], dtype=float)[0, 0])
        edges = np.concatenate([_ring_edges(r, ref) for r in polygon], axis=0)
        qx = (lon - ref + 180.0) % 360.0 - 180.0 + ref
        x_lo = min(edges[:, 0].min(), edges[:, 2].min())
        x_hi = max(edges[:, 0].max(), edges[:, 2].max())
        y_lo = min(edges[:, 1].min(), edges[:, 3].min())
        y_hi = max(edges[:, 1].max(), edges[:, 3].max())
        cand = np.flatnonzero(finite & ~result & (qx >= x_lo) & (qx <= x_hi)
                              & (lat >= y_lo) & (lat <= y_hi))
        if cand.size == 0:
            continue
        # Horizontal edges never count under the half-open rule; drop them.
        edges = edges[edges[:, 1] != edges[:, 3]]
        e_lo = np.minimum(edges[:, 1], edges[:, 3])
        e_hi = np.maximum(edges[:, 1], edges[:, 3])
        nbands = int(np.clip(len(edges) // 4, 1, 4096))
        band_h = (y_hi - y_lo) / nbands if y_hi > y_lo else 1.0
        py = lat[cand]
        px = qx[cand]
        band = np.clip(((py - y_lo) / band_h).astype(np.int64), 0, nbands - 1)
        b_lo = np.clip(((e_lo - y_lo) / band_h).astype(np.int64), 0, nbands - 1)
        b_hi = np.clip(((e_hi - y_lo) / band_h).astype(np.int64), 0, nbands - 1)
        point_order = np.argsort(band, kind="stable")
        bounds = np.searchsorted(band[point_order], np.arange(nbands + 1))
        # Edges per band: expand each edge over the bands it spans.
        span = b_hi - b_lo + 1
        edge_idx = np.repeat(np.arange(len(edges)), span)
        edge_band = np.repeat(b_lo, span) + (
            np.arange(len(edge_idx)) - np.repeat(np.cumsum(span) - span, span))
        e_order = np.argsort(edge_band, kind="stable")
        edge_idx = edge_idx[e_order]
        e_bounds = np.searchsorted(edge_band[e_order], np.arange(nbands + 1))
        parity = np.zeros(cand.size, dtype=bool)
        for b in range(nbands):
            p_sel = point_order[bounds[b]:bounds[b + 1]]
            if p_sel.size == 0:
                continue
            e_sel = edge_idx[e_bounds[b]:e_bounds[b + 1]]
            if e_sel.size == 0:
                continue
            e = edges[e_sel]
            rows = max(1, 4_000_000 // max(1, e_sel.size))
            for s in range(0, p_sel.size, rows):
                sel = p_sel[s:s + rows]
                yy = py[sel, None]
                xx = px[sel, None]
                straddle = (e[None, :, 1] > yy) != (e[None, :, 3] > yy)
                x_cross = e[None, :, 0] + (yy - e[None, :, 1]) * (
                    e[None, :, 2] - e[None, :, 0]) / (e[None, :, 3] - e[None, :, 1])
                hits = straddle & (xx < x_cross)
                parity[sel] = (np.count_nonzero(hits, axis=1) % 2) == 1
        result[cand] |= parity
    return result.reshape(shape)


# ---------------------------------------------------------------------------
# rectangle cover


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in projected metres covering ``members``.

    ``nx``/``ny`` count mass points at the requested ``dx_m``; the extent is
    ``x_min + i*dx`` for ``i in range(nx)`` (likewise y).
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    nx: int
    ny: int
    members: np.ndarray


class _Axis:
    """Extent arithmetic along one axis of :func:`cover_with_rectangles`."""

    def __init__(self, margin: float, dx: float, max_n: int,
                 align: float | None):
        self.margin = margin
        self.dx = dx
        self.max_n = max_n
        self.align = align
        width = (max_n - 1) * dx
        if align is None:
            self.width = width
            self.guaranteed = width - 2.0 * margin
        else:
            self.width = math.floor(width / align + 1e-9) * align
            self.guaranteed = self.width - 2.0 * margin - align

    def extent(self, lo_m, hi_m):
        """Snapped (lo, hi, n) for member extents (scalars or arrays)."""

        lo = np.asarray(lo_m, dtype=float) - self.margin
        hi = np.asarray(hi_m, dtype=float) + self.margin
        if self.align is None:
            n = np.ceil((hi - lo) / self.dx - 1e-9).astype(np.int64) + 1
            hi = lo + (n - 1) * self.dx
        else:
            lo = np.floor(lo / self.align) * self.align
            hi = np.ceil(hi / self.align) * self.align
            n = np.rint((hi - lo) / self.dx).astype(np.int64) + 1
        return lo, hi, n

    def window_end(self, anchor: float) -> float:
        """Largest member coordinate that fits with ``anchor`` as minimum."""

        if self.align is None:
            end = anchor + self.width - 2.0 * self.margin
        else:
            start = math.floor((anchor - self.margin) / self.align) * self.align
            end = start + self.width - self.margin
        end -= 1e-6 * max(1.0, self.dx)
        return max(anchor, end)


def _strip_groups(u: np.ndarray, v: np.ndarray, au: _Axis, av: _Axis
                  ) -> list[np.ndarray]:
    order = np.argsort(u, kind="stable")
    us = u[order]
    groups: list[np.ndarray] = []
    i, n = 0, len(us)
    while i < n:
        j = int(np.searchsorted(us, au.window_end(float(us[i])), side="right"))
        strip = order[i:j]
        sub = np.argsort(v[strip], kind="stable")
        strip = strip[sub]
        vs = v[strip]
        k, m = 0, len(vs)
        while k < m:
            e = int(np.searchsorted(vs, av.window_end(float(vs[k])), side="right"))
            groups.append(strip[k:e])
            k = e
        i = j
    return groups


def _merge_groups(groups: list[np.ndarray], x: np.ndarray, y: np.ndarray,
                  ax: _Axis, ay: _Axis) -> list[np.ndarray]:
    k = len(groups)
    if k < 2 or k > COVER_MERGE_LIMIT:
        return groups
    groups = list(groups)
    ext = np.array([[x[g].min(), x[g].max(), y[g].min(), y[g].max()]
                    for g in groups])
    alive = np.ones(k, dtype=bool)

    def cost_row(i: int) -> np.ndarray:
        x_lo = np.minimum(ext[i, 0], ext[:, 0])
        x_hi = np.maximum(ext[i, 1], ext[:, 1])
        y_lo = np.minimum(ext[i, 2], ext[:, 2])
        y_hi = np.maximum(ext[i, 3], ext[:, 3])
        _, _, nx = ax.extent(x_lo, x_hi)
        _, _, ny = ay.extent(y_lo, y_hi)
        cost = (nx * ny).astype(float)
        cost[(nx > ax.max_n) | (ny > ay.max_n) | ~alive] = np.inf
        cost[i] = np.inf
        return cost

    cost = np.full((k, k), np.inf)
    for i in range(k):
        cost[i] = cost_row(i)
    # Per-row minimum kept up to date, so each merge costs O(k), not O(k^2).
    row_min = cost.min(axis=1)
    row_arg = cost.argmin(axis=1)
    while True:
        i = int(np.argmin(row_min))
        if not np.isfinite(row_min[i]):
            break
        j = int(row_arg[i])
        i, j = min(i, j), max(i, j)
        groups[i] = np.concatenate((groups[i], groups[j]))
        ext[i] = (min(ext[i, 0], ext[j, 0]), max(ext[i, 1], ext[j, 1]),
                  min(ext[i, 2], ext[j, 2]), max(ext[i, 3], ext[j, 3]))
        alive[j] = False
        cost[j, :] = np.inf
        cost[:, j] = np.inf
        row_min[j] = np.inf
        row = cost_row(i)
        cost[i, :] = row
        cost[:, i] = row
        row_min[i] = row.min()
        row_arg[i] = int(row.argmin())
        stale = np.flatnonzero(alive & ((row_arg == i) | (row_arg == j)))
        for r in stale:
            if r != i:
                row_min[r] = cost[r].min()
                row_arg[r] = int(cost[r].argmin())
        better = alive & (row < row_min)
        better[i] = False
        row_min[better] = row[better]
        row_arg[better] = i
    return [g for g, a in zip(groups, alive) if a]


def cover_with_rectangles(x: np.ndarray, y: np.ndarray, *, margin_m: float,
                          dx_m: float, max_nx: int, max_ny: int,
                          align_m: float | None = None) -> list[Rect]:
    """Greedy cover of projected points by rectangles.

    Every point lies at least ``margin_m`` inside some rectangle; no
    rectangle exceeds ``max_nx`` x ``max_ny`` mass points at ``dx_m``;
    corners snap to multiples of ``align_m`` (a parent cell) when given.
    Each point is a member of exactly one rectangle.

    Method (deterministic): sort along one axis and cut strips anchored at
    the first uncovered point, each as wide as a rectangle may be; inside a
    strip cut runs along the other axis the same way; then merge pairs of
    groups whose union still fits, smallest union first.  Both axis orders
    are tried and the cover with fewer rectangles (then smaller total
    area, then x-first) is returned.  Each rectangle is shrunk to its
    members plus the margin, snapped outward to ``align_m`` when given
    (which must be a whole multiple of ``dx_m``), and otherwise grown at
    its upper edge to a whole number of ``dx_m`` steps.  Rectangles may
    overlap one another (they always do by about ``2 * margin_m`` where
    neighbours meet); membership never does.  Rectangles come back sorted
    by ``(x_min, y_min)`` with ``members`` ascending.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise GeometryError("cover_with_rectangles: x and y lengths differ")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise GeometryError("cover_with_rectangles: non-finite coordinates")
    for name, val in (("dx_m", dx_m), ("margin_m", margin_m)):
        if not _real(val):
            raise GeometryError(f"cover_with_rectangles: {name}={val!r} is not "
                                "a finite number")
    if dx_m <= 0.0:
        raise GeometryError(f"cover_with_rectangles: dx_m must be positive, "
                            f"not {dx_m}")
    if margin_m < 0.0:
        raise GeometryError(f"cover_with_rectangles: margin_m must not be "
                            f"negative, not {margin_m}")
    for name, val in (("max_nx", max_nx), ("max_ny", max_ny)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) \
                or val < 2:
            raise GeometryError(f"cover_with_rectangles: {name} must be an "
                                f"integer of at least 2, not {val!r}")
    if align_m is not None:
        if not _positive(align_m):
            raise GeometryError(f"cover_with_rectangles: align_m must be "
                                f"positive, not {align_m!r}")
        ratio = align_m / dx_m
        if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio):
            raise GeometryError(
                f"cover_with_rectangles: align_m={align_m} is not a whole "
                f"multiple of dx_m={dx_m}, so snapped corners would not fall "
                "on mass points")
        align = float(align_m)
    else:
        align = None
    ax = _Axis(float(margin_m), float(dx_m), int(max_nx), align)
    ay = _Axis(float(margin_m), float(dx_m), int(max_ny), align)
    for name, axis in (("x", ax), ("y", ay)):
        if axis.guaranteed < 0.0:
            snap = f" plus one align_m={align} cell" if align else ""
            raise GeometryError(
                f"cover_with_rectangles: twice the margin ({2 * margin_m:g} m)"
                f"{snap} exceeds the largest {name} extent "
                f"({axis.width:g} m = (max_n{name} - 1) * dx_m"
                f"{' snapped to align_m' if align else ''}); no rectangle can "
                "hold even one point")
    if x.size == 0:
        return []

    best = None
    for swap in (False, True):
        if swap:
            groups = [g for g in _strip_groups(y, x, ay, ax)]
        else:
            groups = _strip_groups(x, y, ax, ay)
        groups = _merge_groups(groups, x, y, ax, ay)
        rects = []
        for g in groups:
            g = np.sort(g)
            x_lo, x_hi, nx = ax.extent(x[g].min(), x[g].max())
            y_lo, y_hi, ny = ay.extent(y[g].min(), y[g].max())
            if nx > max_nx or ny > max_ny:
                raise GeometryError("cover_with_rectangles: a rectangle "
                                    "exceeded its size cap (internal error)")
            rects.append(Rect(float(x_lo), float(y_lo), float(x_hi),
                              float(y_hi), int(nx), int(ny), g))
        area = sum(r.nx * r.ny for r in rects)
        score = (len(rects), area)
        if best is None or score < best[0]:
            best = (score, rects)
    rects = best[1]
    rects.sort(key=lambda r: (r.x_min, r.y_min, int(r.members[0])))
    return rects


__all__ = [
    "LonLat", "Polygon", "LocalProjection", "Rect", "GeometryError",
    "local_projection", "densify_polyline", "segment_bearings",
    "buffer_polyline", "point_in_polygon", "cover_with_rectangles",
]
