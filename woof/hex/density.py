"""Density rasters for the hex mesh generator: ``woof-hex.density.v1``.

A density raster is the requested cell SPACING of a variable-resolution
mesh, sampled on a regular latitude/longitude grid.  The mesh generator
(``woof mesh --density-raster`` and a mesh spec region
``{"shape": "raster", "path": P}``) reads it in place of the cap and
polygon rungs a hand-written spec carries, so a corridor, a terrain or a
forecast field can ask for resolution exactly where it is needed.

THE FORMAT, frozen across the hex units.  CF netCDF with

* 1-D ``lat`` and ``lon`` coordinates in degrees, strictly ascending;
* ``spacing_km`` (float64, dims ``lat, lon``), the requested spacing in km;
* global attributes ``schema = "woof-hex.density.v1"`` and
  ``min_spacing_km``, plus (this module) ``max_spacing_km``,
  ``background_km``, ``sources`` and ``limiter`` (JSON strings).

THREE BUILDERS, ONE PIPELINE.

* energy corridors (:func:`corridor_from_sites`, :func:`corridor_from_assets`):
  ``fine_km`` within ``corridor_km`` of a line, site or asset;
* terrain (:func:`terrain_from_glo30`, :func:`terrain_from_dem`): finer
  spacing where the terrain, seen at the finest scale the mesh resolves,
  is steep or strongly curved;
* forecast fields (:func:`density_from_fields`): a generic Python API for
  the adaptive cycle driver -- per-point scalar criteria and thresholds in,
  spacing out.

Every source paints onto one raster with ``min``; the result then goes
through ONE gradient limiter, which is also what grades a corridor outward:
spacing grows by at most the per-cell bound, so it is geometric in cell
count and linear in distance, ``h(d) = fine + g d`` capped at background.

THE GRADIENT BOUND.  ``rw_mpas_mesh`` measures the steepest per-cell change
``|h(p + h(p) e) / h(p) - 1|`` (``mesh/density.rs``,
``steepest_gradient_reading_of``) and refuses a spec past the transition-band
ceiling (:data:`woof.hex.mesh_spec_gates.MAX_GRADIENT_PER_CELL`, 12.25 %/cell).
``woof mesh`` refuses earlier, at the provisional smoothness bound in
``woof/data/mpas/mesh-sizing.json`` (3.06 %/cell).  A field that is
``g``-Lipschitz in km/km reads exactly ``<= g`` on that instrument, so the
limiter enforces a Lipschitz envelope: by default at the stricter 3.06 %,
at the 12.25 % ceiling only under ``allow_rough`` (reported as a workaround,
exactly as ``woof mesh --allow-rough-mesh`` reports it).  The envelope is the
exact chamfer (8-neighbour) Lipschitz envelope of the raster, so every axis
and diagonal neighbour pair differs by at most ``g`` times its distance; it
is computed in two raster sweeps.  A one-percent safety margin absorbs the
bilinear reconstruction and sphere-curvature residue between raster rows.

RASTER RESOLUTION.  The target is :data:`DEFAULT_CELLS_PER_FINEST` raster
cells per finest spacing.  The raster must also reach background at its edge
(a ramp at ``g`` per cell from 0.1 km to 25 km is 814 km long at 3.06 %), so
the target can exceed any sensible size.  The raster is then coarsened, never
past the point where it could misplace spacing by more than
``interp_tolerance`` of the finest spacing (``g * cell / sqrt(2)``: a
``g``-Lipschitz field cannot change faster than that across a cell), and
refused beyond ``max_raster_cells``.  The summary says which rule chose it.

Bulk work here is raster preparation in vectorized numpy/scipy; the DEM
decode and area-average run in the Rust static-fields library through the
existing GLO-30 fetch machinery (:mod:`woof.static.highres_fetch`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .errors import MpasPortError

#: The frozen contract name.
SCHEMA = "woof-hex.density.v1"
#: What :meth:`DensityRaster.summary` prints.
SUMMARY_SCHEMA = "woof-hex.density-summary.v1"

#: ``rw-mpas`` ``mesh/density.rs::EARTH_RADIUS_M`` (the MPAS sphere), in km.
EARTH_RADIUS_KM = 6371.229
KM_PER_DEG = EARTH_RADIUS_KM * math.pi / 180.0
#: Area of a regular hexagon of centre-to-centre spacing ``h`` is
#: ``(sqrt 3 / 2) h^2``: the cell estimate is the integral of dA over that.
HEX_AREA_FACTOR = math.sqrt(3.0) / 2.0

DEFAULT_CORRIDOR_KM = 2.0
DEFAULT_CELLS_PER_FINEST = 4.0
#: About 2 GB of working memory at float64 with the copies the summary needs.
DEFAULT_MAX_RASTER_CELLS = 30_000_000
DEFAULT_INTERP_TOLERANCE = 0.10
#: The limiter enforces ``(1 - LIMITER_SAFETY) * g`` so the bilinear
#: reconstruction between raster rows (whose km-per-degree differ by the
#: cosine of latitude) still reads ``<= g`` on the generator's instrument.
LIMITER_SAFETY = 0.01
#: A regular lat/lon raster beyond this latitude is refused: its columns
#: collapse and its antimeridian handling is not part of the contract.
MAX_ABS_LAT = 85.0
#: GLO-30's native spacing, the floor of a terrain analysis grid.
GLO30_SPACING_KM = 0.03
#: Terrain heuristics.  NOT MEASURED against any mesh run: they say what
#: "steep" and "complex" mean for the default mapping and nothing more.
DEFAULT_SLOPE_COARSE_DEG = 5.0
DEFAULT_SLOPE_FINE_DEG = 25.0
DEFAULT_CURVATURE_COARSE_PER_M = 2.0e-4
DEFAULT_CURVATURE_FINE_PER_M = 2.0e-3

_ROW_BLOCK = 1024
#: Limiter sweeps before it gives up converging; the result is verified
#: against the bound either way, so a cap reached is a refusal, never a
#: silently rough raster.
MAX_LIMITER_SWEEPS = 8


class DensityRefusal(MpasPortError):
    """A density raster that cannot be built as asked, said with numbers."""


# ---------------------------------------------------------------------------
# the gradient policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradientPolicy:
    """The per-cell spacing-change bound the limiter enforces."""

    #: The bound in force (fraction per cell).
    bound_per_cell: float
    #: Where the bound comes from, in words a receipt can carry.
    bound_source: str
    #: ``measured`` / ``provisional`` / ``derived``.
    bound_status: str
    #: The grade the limiter grades and limits at (fraction, ``<= bound``).
    grade_per_cell: float
    allow_rough: bool = False
    workaround: str | None = None

    @property
    def enforced_per_cell(self) -> float:
        return self.grade_per_cell * (1.0 - LIMITER_SAFETY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bound_percent_per_cell": self.bound_per_cell * 100.0,
            "bound_source": self.bound_source,
            "bound_status": self.bound_status,
            "grade_percent_per_cell": self.grade_per_cell * 100.0,
            "enforced_percent_per_cell": self.enforced_per_cell * 100.0,
            "safety_fraction": LIMITER_SAFETY,
            "allow_rough": self.allow_rough,
            "workaround": self.workaround,
            "method": "exact 8-neighbour chamfer Lipschitz envelope, "
                      "two raster sweeps",
        }


def gradient_policy(*, allow_rough: bool = False,
                    grade_percent_per_cell: float | None = None
                    ) -> GradientPolicy:
    """The bound the generator gates at, read where the gates live.

    Default: the ``woof mesh`` smoothness refusal bound from
    ``mesh-sizing.json`` (3.06 %/cell, provisional).  ``allow_rough``: the
    ``rw_mpas_mesh`` transition-band ceiling
    (:data:`~woof.hex.mesh_spec_gates.MAX_GRADIENT_PER_CELL`), which no flag
    lifts.  ``grade_percent_per_cell`` asks for a gentler grade and is
    refused above the bound in force.
    """

    from woof.hex.mesh_spec_gates import MAX_GRADIENT_PER_CELL
    from woof.mpas_mesh import load_sizing

    smooth = load_sizing().smoothness
    if allow_rough:
        bound = MAX_GRADIENT_PER_CELL
        source = ("rw_mpas_mesh transition-band ceiling "
                  "(woof.hex.mesh_spec_gates.MAX_GRADIENT_PER_CELL)")
        status = "derived"
        workaround = (
            "WORKAROUND: --allow-rough was passed, so the "
            f"{smooth.refuse_above_percent_per_cell:.2f} %/cell smoothness "
            "bound woof mesh enforces was NOT applied; the raster is limited "
            f"at the generator's {MAX_GRADIENT_PER_CELL * 100.0:.2f} %/cell "
            "transition-band ceiling instead.  woof mesh will refuse it "
            "unless --allow-rough-mesh is passed there too.  "
            f"{smooth.evidence}")
    else:
        bound = min(smooth.refuse_above_percent_per_cell / 100.0,
                    MAX_GRADIENT_PER_CELL)
        source = ("woof mesh smoothness bound (mesh-sizing.json "
                  "smoothness.refuse_above_percent_per_cell)")
        status = smooth.status
        workaround = None
    if grade_percent_per_cell is None:
        grade = bound
    else:
        grade = float(grade_percent_per_cell) / 100.0
        if not math.isfinite(grade) or grade <= 0.0:
            raise DensityRefusal(
                f"--grade-percent-per-cell must be positive, got "
                f"{grade_percent_per_cell!r}")
        if grade > bound * (1.0 + 1e-12):
            raise DensityRefusal(
                f"--grade-percent-per-cell {grade * 100.0:.3f} is steeper "
                f"than the {bound * 100.0:.3f} %/cell bound in force "
                f"({source}); the generator would refuse the raster")
    return GradientPolicy(bound_per_cell=bound, bound_source=source,
                          bound_status=status, grade_per_cell=grade,
                          allow_rough=allow_rough, workaround=workaround)


# ---------------------------------------------------------------------------
# the raster grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterGrid:
    """A regular lat/lon grid of cell CENTRES, south-west edge first."""

    lat0: float  # south edge, degrees
    lon0: float  # west edge, degrees
    dlat: float
    dlon: float
    ny: int
    nx: int

    @property
    def lat(self) -> np.ndarray:
        return self.lat0 + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dlat

    @property
    def lon(self) -> np.ndarray:
        return self.lon0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dlon

    @property
    def cells(self) -> int:
        return self.ny * self.nx

    @property
    def dy_km(self) -> float:
        return self.dlat * KM_PER_DEG

    @property
    def dx_km_rows(self) -> np.ndarray:
        return self.dlon * KM_PER_DEG * np.cos(np.radians(self.lat))

    def extent(self) -> dict[str, float]:
        return {"lat_min": self.lat0, "lat_max": self.lat0 + self.ny * self.dlat,
                "lon_min": self.lon0, "lon_max": self.lon0 + self.nx * self.dlon}

    def rows_of(self, lat: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(lat) - self.lat0) / self.dlat).astype(np.int64)

    def cols_of(self, lon: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(lon) - self.lon0) / self.dlon).astype(np.int64)

    def window(self, south: float, north: float, west: float, east: float,
               pad_km: float = 0.0) -> tuple[slice, slice] | None:
        """Row/column slices of the cells within a box (padded by km)."""

        pad_lat = pad_km / KM_PER_DEG
        cos_pole = math.cos(math.radians(min(
            MAX_ABS_LAT, max(abs(south), abs(north)) + pad_lat)))
        pad_lon = pad_km / (KM_PER_DEG * max(cos_pole, 1e-6))
        r0 = int(math.floor((south - pad_lat - self.lat0) / self.dlat)) - 1
        r1 = int(math.ceil((north + pad_lat - self.lat0) / self.dlat)) + 1
        c0 = int(math.floor((west - pad_lon - self.lon0) / self.dlon)) - 1
        c1 = int(math.ceil((east + pad_lon - self.lon0) / self.dlon)) + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.ny), min(c1, self.nx)
        if r0 >= r1 or c0 >= c1:
            return None
        return slice(r0, r1), slice(c0, c1)


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    lam = np.radians(np.asarray(lon, dtype=np.float64))
    c = np.cos(phi)
    return np.stack((c * np.cos(lam), c * np.sin(lam), np.sin(phi)), axis=-1)


def _chord(distance_km: float) -> float:
    """Unit-sphere chord of a great-circle distance."""

    return 2.0 * math.sin(min(distance_km / (2.0 * EARTH_RADIUS_KM), math.pi / 2))


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


class DensitySource(Protocol):
    """Something that can paint requested spacing onto a raster."""

    name: str

    def finest_km(self) -> float: ...

    def bounds(self) -> tuple[float, float, float, float]:
        """``(south, north, west, east)`` of where it asks for < background."""

    def paint(self, grid: RasterGrid, spacing: np.ndarray,
              background_km: float) -> None:
        """Lower ``spacing`` in place (``np.minimum``) where it asks."""

    def provenance(self) -> dict[str, Any]: ...


def _paint_near_points(grid: RasterGrid, spacing: np.ndarray,
                       lat: np.ndarray, lon: np.ndarray, values: np.ndarray,
                       radius_km: float, *, tree_lat: np.ndarray | None = None,
                       tree_lon: np.ndarray | None = None,
                       tree_values: np.ndarray | None = None) -> int:
    """``spacing = min(spacing, value of the nearest point)`` within a radius.

    ``lat/lon/values`` are the points that ask for refinement; the nearest
    point is searched among ``tree_*`` (default: the same points), so a
    raster cell takes the value of the point whose Voronoi cell it is in.
    The raster cell containing each point always takes its value, so a
    feature thinner than one raster cell is never lost.  Returns the number
    of raster cells written.
    """

    from scipy.ndimage import maximum_filter
    from scipy.spatial import cKDTree

    if lat.size == 0:
        return 0
    rows = grid.rows_of(lat)
    cols = grid.cols_of(lon)
    inside = (rows >= 0) & (rows < grid.ny) & (cols >= 0) & (cols < grid.nx)
    rows, cols, own = rows[inside], cols[inside], values[inside]
    if rows.size == 0:
        return 0
    np.minimum.at(spacing, (rows, cols), own)
    if tree_lat is None:
        tree_lat, tree_lon, tree_values = lat, lon, values
    tree = cKDTree(_xyz(tree_lat, tree_lon))
    chord = _chord(radius_km)
    dx_rows = grid.dx_km_rows
    kr = int(math.ceil(radius_km / grid.dy_km)) + 1
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    lat_axis, lon_axis = grid.lat, grid.lon
    written = 0
    # Candidates are found and queried one block of raster rows at a time,
    # so working memory follows the block, not the whole footprint.
    block = max(_ROW_BLOCK // 4, 2 * kr)
    first = max(int(rows[0]) - kr, 0)
    last = min(int(rows[-1]) + kr + 1, grid.ny)
    for b0 in range(first, last, block):
        b1 = min(b0 + block, last)
        lo = np.searchsorted(rows, b0 - kr, side="left")
        hi = np.searchsorted(rows, b1 + kr, side="left")
        if lo == hi:
            continue
        s_rows, s_cols = rows[lo:hi], cols[lo:hi]
        e0, e1 = max(b0 - kr, 0), min(b1 + kr, grid.ny)
        kc = int(math.ceil(radius_km / float(np.min(dx_rows[e0:e1])))) + 1
        c0 = max(int(s_cols.min()) - kc, 0)
        c1 = min(int(s_cols.max()) + kc + 1, grid.nx)
        seeds = np.zeros((e1 - e0, c1 - c0), dtype=bool)
        seeds[s_rows - e0, s_cols - c0] = True
        near = maximum_filter(seeds, size=(2 * kr + 1, 2 * kc + 1),
                              mode="constant")[b0 - e0:b1 - e0]
        cand_r, cand_c = np.nonzero(near)
        if cand_r.size == 0:
            continue
        dist, index = tree.query(
            _xyz(lat_axis[cand_r + b0], lon_axis[cand_c + c0]), k=1,
            distance_upper_bound=chord, workers=-1)
        hit = np.isfinite(dist)
        rr = cand_r[hit] + b0
        cc = cand_c[hit] + c0
        spacing[rr, cc] = np.minimum(spacing[rr, cc], tree_values[index[hit]])
        written += int(hit.sum())
    return written + int(rows.size)


@dataclass
class CorridorSource:
    """``fine_km`` within ``half_width_km`` of polylines, points and areas."""

    polylines: list[tuple[np.ndarray, np.ndarray]]  # (lat, lon) per line
    points: list[tuple[float, float]]  # (lat, lon)
    areas: list[list[tuple[float, float]]]  # outer rings, (lon, lat)
    fine_km: float
    half_width_km: float
    origin: dict[str, Any] = field(default_factory=dict)
    name: str = "corridor"
    _painted_cells: int = 0

    def __post_init__(self) -> None:
        _require_positive(self.fine_km, "fine_km")
        _require_positive(self.half_width_km, "corridor half-width (km)")
        if not (self.polylines or self.points or self.areas):
            raise DensityRefusal(
                f"the {self.name} source holds no line, site or area: an "
                "empty corridor asks for nothing, so there is no raster to "
                "build from it")

    def _all_lat_lon(self) -> tuple[np.ndarray, np.ndarray]:
        lats = [np.asarray(lat, dtype=np.float64) for lat, _ in self.polylines]
        lons = [np.asarray(lon, dtype=np.float64) for _, lon in self.polylines]
        lats.append(np.array([p[0] for p in self.points], dtype=np.float64))
        lons.append(np.array([p[1] for p in self.points], dtype=np.float64))
        for ring in self.areas:
            lats.append(np.array([v[1] for v in ring], dtype=np.float64))
            lons.append(np.array([v[0] for v in ring], dtype=np.float64))
        return np.concatenate(lats), np.concatenate(lons)

    def finest_km(self) -> float:
        return float(self.fine_km)

    def bounds(self) -> tuple[float, float, float, float]:
        lat, lon = self._all_lat_lon()
        pad_lat = self.half_width_km / KM_PER_DEG
        cos_pole = math.cos(math.radians(min(MAX_ABS_LAT,
                                             float(np.max(np.abs(lat))))))
        pad_lon = self.half_width_km / (KM_PER_DEG * max(cos_pole, 1e-6))
        return (float(lat.min()) - pad_lat, float(lat.max()) + pad_lat,
                float(lon.min()) - pad_lon, float(lon.max()) + pad_lon)

    def paint(self, grid: RasterGrid, spacing: np.ndarray,
              background_km: float) -> None:
        from woof.energy.geometry import densify_polyline

        step_km = max(min(grid.dy_km, float(np.min(grid.dx_km_rows)),
                          self.half_width_km) / 2.0, 1e-3)
        lats: list[np.ndarray] = []
        lons: list[np.ndarray] = []
        rings = [[(lon, lat) for lon, lat in ring] + [ring[0]]
                 for ring in self.areas]
        lines = [list(zip(np.asarray(lon, dtype=float).tolist(),
                          np.asarray(lat, dtype=float).tolist()))
                 for lat, lon in self.polylines] + rings
        for coords in lines:
            d_lon, d_lat, _ = densify_polyline(coords, step_km * 1000.0)
            lats.append(d_lat)
            lons.append(d_lon)
        if self.points:
            lats.append(np.array([p[0] for p in self.points], dtype=np.float64))
            lons.append(np.array([p[1] for p in self.points], dtype=np.float64))
        lat = np.concatenate(lats)
        lon = np.concatenate(lons)
        values = np.full(lat.shape, float(self.fine_km))
        written = _paint_near_points(grid, spacing, lat, lon, values,
                                     self.half_width_km)
        written += self._fill_areas(grid, spacing)
        self._painted_cells = written

    def _fill_areas(self, grid: RasterGrid, spacing: np.ndarray) -> int:
        if not self.areas:
            return 0
        from woof.energy.geometry import point_in_polygon

        written = 0
        for ring in self.areas:
            lon_r = np.array([v[0] for v in ring])
            lat_r = np.array([v[1] for v in ring])
            win = grid.window(float(lat_r.min()), float(lat_r.max()),
                              float(lon_r.min()), float(lon_r.max()))
            if win is None:
                continue
            la, lo = np.meshgrid(grid.lat[win[0]], grid.lon[win[1]],
                                 indexing="ij")
            inside = point_in_polygon(lo, la, [[list(ring)]])
            block = spacing[win]
            block[inside] = np.minimum(block[inside], self.fine_km)
            written += int(inside.sum())
        return written

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "fine_km": self.fine_km,
            "corridor_half_width_km": self.half_width_km,
            "polylines": len(self.polylines),
            "points": len(self.points),
            "areas": len(self.areas),
            "raster_cells_painted": self._painted_cells,
            **self.origin,
        }


def _require_positive(value: float, what: str) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0.0):
        raise DensityRefusal(f"{what} must be a positive number, got {value!r}")


def corridor_from_sites(path: str | Path, *, fine_km: float,
                        corridor_km: float = DEFAULT_CORRIDOR_KM
                        ) -> CorridorSource:
    """A corridor over a ``woof-energy.sites.v1`` document.

    Sites are grouped into ordered per-asset chains exactly as the hex
    planner groups them (:func:`woof.energy.plan_hex.corridor_chains`,
    split at :data:`~woof.energy.plan_hex.CHAIN_GAP_KM`), so a line's
    corridor is continuous between its samples rather than a string of
    beads; a single-site chain is a point.
    """

    from woof.energy.contracts import load_sites, sha256_file
    from woof.energy.plan_hex import CHAIN_GAP_KM, _Aeqd, _centre, corridor_chains

    site_set = load_sites(path)
    if len(site_set) == 0:
        raise DensityRefusal(
            f"sites document {path} holds no sites: there is nothing to "
            "refine around")
    arrays = site_set.as_arrays()
    lat0, lon0 = _centre(arrays["lat"], arrays["lon"])
    chains = corridor_chains(site_set, _Aeqd(lat0, lon0))
    polylines: list[tuple[np.ndarray, np.ndarray]] = []
    points: list[tuple[float, float]] = []
    for chain in chains:
        if chain.is_point:
            points.append((float(chain.lat[0]), float(chain.lon[0])))
        else:
            polylines.append((np.asarray(chain.lat), np.asarray(chain.lon)))
    return CorridorSource(
        polylines=polylines, points=points, areas=[], fine_km=fine_km,
        half_width_km=corridor_km, name="sites-corridor",
        origin={"path": str(path), "sha256": sha256_file(path),
                "schema": "woof-energy.sites.v1", "sites": len(site_set),
                "chains": len(chains), "chain_gap_km": CHAIN_GAP_KM})


def corridor_from_assets(path: str | Path, *, fine_km: float,
                         corridor_km: float = DEFAULT_CORRIDOR_KM
                         ) -> CorridorSource:
    """A corridor over a ``woof-energy.assets.v1`` GeoJSON document.

    Lines and cables are densified along their geometry; point assets are
    points; polygon assets (substations, plants) take their whole area plus
    the corridor around their outline.
    """

    from woof.energy.contracts import load_assets, sha256_file

    collection = load_assets(path)
    polylines: list[tuple[np.ndarray, np.ndarray]] = []
    points: list[tuple[float, float]] = []
    areas: list[list[tuple[float, float]]] = []
    holes = 0
    for asset in collection.assets:
        geometry = asset.geometry
        kind = geometry["type"]
        coords = geometry["coordinates"]
        if kind == "Point":
            points.append((float(coords[1]), float(coords[0])))
        elif kind == "LineString":
            polylines.append(_line_arrays(coords))
        elif kind == "MultiLineString":
            polylines.extend(_line_arrays(part) for part in coords)
        elif kind == "Polygon":
            areas.append([(float(v[0]), float(v[1])) for v in coords[0]])
            holes += len(coords) - 1
        elif kind == "MultiPolygon":
            for part in coords:
                areas.append([(float(v[0]), float(v[1])) for v in part[0]])
                holes += len(part) - 1
        else:  # pragma: no cover - the contract admits no other geometry
            raise DensityRefusal(f"{asset.asset_id}: geometry {kind} is not "
                                 "one an asset document carries")
    if not collection.assets:
        raise DensityRefusal(
            f"assets document {path} holds no assets: there is nothing to "
            "refine around")
    origin = {"path": str(path), "sha256": sha256_file(path),
              "schema": "woof-energy.assets.v1",
              "assets": len(collection.assets)}
    if holes:
        origin["note"] = (f"{holes} polygon hole(s) ignored: the whole outer "
                          "ring is refined, which only adds cells")
    return CorridorSource(polylines=polylines, points=points, areas=areas,
                          fine_km=fine_km, half_width_km=corridor_km,
                          name="assets-corridor", origin=origin)


def _line_arrays(coords) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=np.float64)
    return arr[:, 1].copy(), arr[:, 0].copy()


@dataclass
class GriddedSource:
    """Spacing already sampled on a rectilinear lat/lon analysis grid.

    ``lat`` (ascending, need not be uniform) and ``lon`` (ascending) are the
    analysis cell centres.  Painting is conservative: when the raster is
    coarser than the analysis grid each raster cell takes the MINIMUM of
    the analysis cells it covers, so a narrow steep valley is never
    averaged away.
    """

    lat: np.ndarray
    lon: np.ndarray
    spacing_km: np.ndarray  # (lat, lon), NaN = no request
    name: str
    region: list[list[tuple[float, float]]] | None = None  # rings (lon, lat)
    origin: dict[str, Any] = field(default_factory=dict)
    background_km: float | None = None
    _painted_cells: int = 0

    def __post_init__(self) -> None:
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.spacing_km = np.asarray(self.spacing_km, dtype=np.float64)
        if self.spacing_km.shape != (self.lat.size, self.lon.size):
            raise DensityRefusal(
                f"{self.name}: spacing shape {self.spacing_km.shape} does not "
                f"match lat {self.lat.size} x lon {self.lon.size}")
        if self.lat.size < 2 or self.lon.size < 2 or np.any(
                np.diff(self.lat) <= 0) or np.any(np.diff(self.lon) <= 0):
            raise DensityRefusal(
                f"{self.name}: analysis lat/lon must be strictly ascending "
                "with at least two of each")
        finite = self.spacing_km[np.isfinite(self.spacing_km)]
        if finite.size and float(finite.min()) <= 0.0:
            raise DensityRefusal(f"{self.name}: spacing must be positive")

    def finest_km(self) -> float:
        finite = self.spacing_km[np.isfinite(self.spacing_km)]
        if finite.size == 0:
            raise DensityRefusal(f"{self.name}: every analysis cell is no data")
        return float(finite.min())

    def bounds(self) -> tuple[float, float, float, float]:
        half_lat = 0.5 * float(np.max(np.diff(self.lat)))
        half_lon = 0.5 * float(np.max(np.diff(self.lon)))
        return (float(self.lat[0]) - half_lat, float(self.lat[-1]) + half_lat,
                float(self.lon[0]) - half_lon, float(self.lon[-1]) + half_lon)

    def paint(self, grid: RasterGrid, spacing: np.ndarray,
              background_km: float) -> None:
        from scipy.ndimage import minimum_filter

        south, north, west, east = self.bounds()
        win = grid.window(south, north, west, east)
        if win is None:
            return
        source = np.where(np.isfinite(self.spacing_km), self.spacing_km,
                          background_km)
        # How many analysis cells one raster cell spans, per axis.
        a_dlat = float(np.min(np.diff(self.lat)))
        a_dlon = float(np.min(np.diff(self.lon)))
        kr = max(1, int(math.ceil(grid.dlat / a_dlat)))
        kc = max(1, int(math.ceil(grid.dlon / a_dlon)))
        if kr > 1 or kc > 1:
            source = minimum_filter(source, size=(kr | 1, kc | 1),
                                    mode="nearest")
        lat_c = grid.lat[win[0]]
        lon_c = grid.lon[win[1]]
        keep_r = (lat_c >= south) & (lat_c <= north)
        keep_c = (lon_c >= west) & (lon_c <= east)
        ri = _nearest_index(self.lat, lat_c)
        ci = _nearest_index(self.lon, lon_c)
        values = source[np.ix_(ri, ci)]
        mask = keep_r[:, None] & keep_c[None, :]
        if self.region is not None:
            from woof.energy.geometry import point_in_polygon

            la, lo = np.meshgrid(lat_c, lon_c, indexing="ij")
            mask &= point_in_polygon(lo, la, [[list(r)] for r in self.region])
        block = spacing[win]
        lowered = mask & (values < block)
        block[lowered] = values[lowered]
        self._painted_cells = int(lowered.sum())

    def provenance(self) -> dict[str, Any]:
        return {"kind": self.name,
                "analysis_shape": [int(self.lat.size), int(self.lon.size)],
                "finest_km": self.finest_km(),
                "raster_cells_painted": self._painted_cells,
                **self.origin}


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(axis, values), 1, axis.size - 1)
    left = axis[idx - 1]
    right = axis[idx]
    idx -= (values - left) < (right - values)
    return idx


@dataclass
class PointSource:
    """Per-point requested spacing (e.g. forecast criteria on mesh cells).

    Each raster cell within ``support_km`` of a point takes the spacing of
    its NEAREST point, so the points' own Voronoi cells carry their
    requests onto the raster.  A point asks for nothing when its spacing is
    NaN or, given ``background_km``, at or above it; only asking points set
    the raster extent.
    """

    lat: np.ndarray
    lon: np.ndarray
    spacing_km: np.ndarray
    support_km: float
    name: str = "fields"
    origin: dict[str, Any] = field(default_factory=dict)
    background_km: float | None = None
    _painted_cells: int = 0

    def __post_init__(self) -> None:
        self.lat = np.asarray(self.lat, dtype=np.float64).ravel()
        self.lon = np.asarray(self.lon, dtype=np.float64).ravel()
        self.spacing_km = np.asarray(self.spacing_km, dtype=np.float64).ravel()
        if not (self.lat.size == self.lon.size == self.spacing_km.size):
            raise DensityRefusal(
                f"{self.name}: lat, lon and spacing differ in size")
        _require_positive(self.support_km, f"{self.name} support_km")

    def _fine(self, background_km: float | None = None) -> np.ndarray:
        values = self.spacing_km
        mask = np.isfinite(values)
        background = background_km if background_km is not None \
            else self.background_km
        if background is not None:
            mask &= values < background
        return mask

    def finest_km(self) -> float:
        asking = self.spacing_km[self._fine()]
        if asking.size == 0:
            raise DensityRefusal(f"{self.name}: no point asks for refinement")
        return float(asking.min())

    def bounds(self) -> tuple[float, float, float, float]:
        mask = self._fine()
        lat, lon = self.lat[mask], self.lon[mask]
        if lat.size == 0:
            raise DensityRefusal(f"{self.name}: no point asks for refinement")
        pad_lat = self.support_km / KM_PER_DEG
        cos_pole = math.cos(math.radians(min(MAX_ABS_LAT,
                                             float(np.max(np.abs(lat))))))
        pad_lon = self.support_km / (KM_PER_DEG * max(cos_pole, 1e-6))
        return (float(lat.min()) - pad_lat, float(lat.max()) + pad_lat,
                float(lon.min()) - pad_lon, float(lon.max()) + pad_lon)

    def paint(self, grid: RasterGrid, spacing: np.ndarray,
              background_km: float) -> None:
        fine = self._fine(background_km)
        tree_values = np.where(np.isfinite(self.spacing_km),
                               np.minimum(self.spacing_km, background_km),
                               background_km)
        self._painted_cells = _paint_near_points(
            grid, spacing, self.lat[fine], self.lon[fine], tree_values[fine],
            self.support_km, tree_lat=self.lat, tree_lon=self.lon,
            tree_values=tree_values)

    def provenance(self) -> dict[str, Any]:
        return {"kind": self.name, "points": int(self.lat.size),
                "points_asking": int(self._fine().sum()),
                "support_km": self.support_km,
                "finest_km": self.finest_km(),
                "raster_cells_painted": self._painted_cells,
                **self.origin}


# ---------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------


def terrain_spacing(lat: np.ndarray, lon: np.ndarray, elevation_m: np.ndarray,
                    *, fine_km: float, background_km: float,
                    slope_coarse_deg: float = DEFAULT_SLOPE_COARSE_DEG,
                    slope_fine_deg: float = DEFAULT_SLOPE_FINE_DEG,
                    curvature_coarse_per_m: float = DEFAULT_CURVATURE_COARSE_PER_M,
                    curvature_fine_per_m: float = DEFAULT_CURVATURE_FINE_PER_M,
                    smooth_cells: float = 1.0
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Spacing from slope and curvature of a rectilinear lat/lon DEM.

    The DEM is Gaussian-smoothed by ``smooth_cells`` analysis cells first
    (the analysis grid is already area-averaged at the scale the mesh can
    resolve).  Each criterion scores 0 at its ``coarse`` threshold and 1 at
    its ``fine`` threshold; the larger score wins and maps log-linearly,
    ``h = background * (fine / background) ** score``.  Curvature is
    ``|laplacian(z)|`` in 1/m.  NaN elevations (no data) ask for nothing.
    """

    from scipy.ndimage import gaussian_filter

    if not slope_fine_deg > slope_coarse_deg >= 0.0:
        raise DensityRefusal(
            f"--slope-fine-deg ({slope_fine_deg}) must exceed "
            f"--slope-coarse-deg ({slope_coarse_deg}) >= 0")
    if not curvature_fine_per_m > curvature_coarse_per_m >= 0.0:
        raise DensityRefusal(
            f"--curvature-fine ({curvature_fine_per_m}) must exceed "
            f"--curvature-coarse ({curvature_coarse_per_m}) >= 0")
    z = np.asarray(elevation_m, dtype=np.float64)
    holes = ~np.isfinite(z)
    if holes.all():
        raise DensityRefusal("the DEM holds no finite elevation")
    if holes.any():
        # Fill each hole from its nearest valid cell, so no artificial cliff
        # to some global mean is differentiated at the hole's rim.
        from scipy.ndimage import distance_transform_edt

        _, (ri, ci) = distance_transform_edt(holes, return_indices=True)
        filled = z[ri, ci]
    else:
        filled = z
    if smooth_cells > 0:
        filled = gaussian_filter(filled, sigma=smooth_cells, mode="nearest")
    y_m = np.asarray(lat, dtype=np.float64) * KM_PER_DEG * 1000.0
    dx_m = (np.gradient(np.asarray(lon, dtype=np.float64)) * KM_PER_DEG * 1000.0
            )[None, :] * np.cos(np.radians(lat))[:, None]
    dz_dy = np.gradient(filled, y_m, axis=0)
    dz_dx = np.gradient(filled, axis=1) / dx_m
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    d2y = np.gradient(dz_dy, y_m, axis=0)
    d2x = np.gradient(dz_dx, axis=1) / dx_m
    curvature = np.abs(d2x + d2y)
    s_slope = np.clip((slope_deg - slope_coarse_deg)
                      / (slope_fine_deg - slope_coarse_deg), 0.0, 1.0)
    s_curv = np.clip((curvature - curvature_coarse_per_m)
                     / (curvature_fine_per_m - curvature_coarse_per_m), 0.0, 1.0)
    score = np.maximum(s_slope, s_curv)
    spacing = background_km * (fine_km / background_km) ** score
    spacing[holes] = np.nan
    valid = ~holes
    diagnostics = {
        "slope_deg_p50": float(np.percentile(slope_deg[valid], 50)),
        "slope_deg_p99": float(np.percentile(slope_deg[valid], 99)),
        "slope_deg_max": float(slope_deg[valid].max()),
        "curvature_per_m_p99": float(np.percentile(curvature[valid], 99)),
        "fraction_at_fine": float(np.mean(score[valid] >= 1.0)),
        "fraction_refined": float(np.mean(score[valid] > 0.0)),
        "no_data_cells": int(holes.sum()),
        "thresholds": {
            "slope_coarse_deg": slope_coarse_deg,
            "slope_fine_deg": slope_fine_deg,
            "curvature_coarse_per_m": curvature_coarse_per_m,
            "curvature_fine_per_m": curvature_fine_per_m,
            "status": "heuristic, not measured against any mesh run",
        },
        "smooth_cells": smooth_cells,
    }
    return spacing, diagnostics


def terrain_from_dem(lat: np.ndarray, lon: np.ndarray, elevation_m: np.ndarray,
                     *, fine_km: float, background_km: float,
                     region: list[list[tuple[float, float]]] | None = None,
                     origin: Mapping[str, Any] | None = None,
                     **thresholds: Any) -> GriddedSource:
    """A terrain source from an in-memory rectilinear lat/lon DEM."""

    _require_positive(fine_km, "terrain fine_km")
    spacing, diagnostics = terrain_spacing(
        lat, lon, elevation_m, fine_km=fine_km, background_km=background_km,
        **thresholds)
    spacing = np.where(spacing >= background_km * (1.0 - 1e-12), np.nan,
                       spacing)
    if not np.isfinite(spacing).any():
        raise DensityRefusal(
            "the terrain asks for no refinement anywhere in the region at "
            "these thresholds (nothing is steeper than "
            f"{diagnostics['thresholds']['slope_coarse_deg']} deg or more "
            "curved than "
            f"{diagnostics['thresholds']['curvature_coarse_per_m']} /m); "
            "lower the thresholds or drop --terrain-gradient")
    return GriddedSource(
        lat=lat, lon=lon, spacing_km=spacing, name="terrain-gradient",
        region=region, origin={"terrain_fine_km": fine_km,
                               "diagnostics": diagnostics,
                               **dict(origin or {})})


class _OfflineMiss(RuntimeError):
    pass


def _offline_urlopen(url: str, offset: int):
    raise _OfflineMiss(url)


def fetch_glo30_elevation(west: float, south: float, east: float, north: float,
                          *, analysis_km: float, cache_root: Path | None = None,
                          offline: bool = False, max_cells: int
                          = DEFAULT_MAX_RASTER_CELLS
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     dict[str, Any]]:
    """GLO-30 elevation area-averaged onto a lat/lon-rectilinear grid.

    Tiles come from :func:`woof.static.highres_fetch.fetch_copernicus_dem_tiles`
    (cached per user in ``~/.cache/woof/highres-cache``, resumable, hashed at
    fetch) and are mosaicked by
    :func:`~woof.static.highres_fetch.derive_global_terrain_window`.  The
    window is area-averaged by the Rust static-fields library onto a
    Mercator grid of ``analysis_km`` spacing at the box centre -- a
    Mercator grid's rows are parallels and its columns meridians, so its
    mass points form a rectilinear lat/lon grid.  ``offline`` reads the
    cache only and refuses on the first tile it does not hold.
    """

    from woof.static import rust_bridge
    from woof.static.highres_fetch import (
        COPERNICUS_DEM_LAT_MAX, FootprintBBox, derive_global_terrain_window,
        fetch_copernicus_dem_tiles)
    from woof.static.highres_production import default_highres_cache_root
    from woof.static.projection import MercatorGrid

    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise DensityRefusal(
            f"terrain box W,S,E,N = {west},{south},{east},{north} needs "
            "west < east and south < north in range")
    if max(abs(south), abs(north)) > min(COPERNICUS_DEM_LAT_MAX, MAX_ABS_LAT):
        raise DensityRefusal(
            f"terrain box reaches latitude {max(abs(south), abs(north))}; "
            f"GLO-30 and this raster stop at {COPERNICUS_DEM_LAT_MAX}")
    analysis_km = max(float(analysis_km), GLO30_SPACING_KM)
    root = Path(cache_root) if cache_root is not None else \
        default_highres_cache_root()
    bbox = FootprintBBox(lat_min=south, lat_max=north, lon_min=west,
                         lon_max=east)
    started = time.perf_counter()
    try:
        tiles, absent = fetch_copernicus_dem_tiles(
            bbox, root, urlopen=_offline_urlopen if offline else None)
    except _OfflineMiss as error:
        raise DensityRefusal(
            f"--offline was given and the GLO-30 tile {error} is not in the "
            f"cache {root}; run once without --offline to fetch it") from error
    if not tiles:
        raise DensityRefusal(
            f"GLO-30 publishes no tile over W,S,E,N = {west},{south},{east},"
            f"{north} (absent: {', '.join(absent)}); open sea has no terrain "
            "gradient to refine on")
    window, audit = derive_global_terrain_window(tiles, bbox, root,
                                                 sea_level_fill=0.0)
    fetched_s = time.perf_counter() - started
    lat_c = 0.5 * (south + north)
    lon_c = 0.5 * (west + east)
    dx_m = analysis_km * 1000.0
    width_km = (east - west) * KM_PER_DEG * math.cos(math.radians(lat_c))
    # Mercator rows stretch by 1/cos(lat) away from the true latitude.
    y_lo = math.log(math.tan(math.radians(45.0 + south / 2.0)))
    y_hi = math.log(math.tan(math.radians(45.0 + north / 2.0)))
    height_km = (y_hi - y_lo) * EARTH_RADIUS_KM * math.cos(math.radians(lat_c))
    # One spare mass row/column beyond covering the box, so rounding can
    # never leave an edge just outside the outermost centre.
    e_we = int(math.ceil(width_km / analysis_km)) + 3
    e_sn = int(math.ceil(height_km / analysis_km)) + 3
    if (e_we - 1) * (e_sn - 1) > max_cells:
        raise DensityRefusal(
            f"the terrain analysis grid would be {(e_we - 1) * (e_sn - 1):,} "
            f"cells at {analysis_km} km, past the {max_cells:,} limit; use a "
            "smaller box or a coarser --terrain-scale-km")
    # Centre the grid on the Mercator-y midpoint of the box, not the
    # arithmetic mid-latitude: rows are uniform in y, and y's midpoint lies
    # poleward of lat_c, so centring on lat_c would shift the grid
    # equatorward and leave the box's poleward strip unjudged.
    lat_mid = math.degrees(2.0 * math.atan(math.exp(0.5 * (y_lo + y_hi)))
                           - math.pi / 2.0)
    grid = MercatorGrid(lat_mid, lon_c, lat_c, lat_c, lon_c, dx_m, dx_m,
                        e_we, e_sn)
    bridge = rust_bridge
    reason = bridge.unavailable_reason()
    if reason is not None:
        raise DensityRefusal(
            "the GLO-30 window is decoded and area-averaged by the Rust "
            f"static-fields library, which is not available: {reason}")
    fields, _ = bridge.highres_resample({
        "kind": "continuous", "grid_spec": grid._rust_spec(),
        "method": "average",
        "source": {"path": str(window.path), "sha256": window.sha256,
                   "scale_factor": 1.0, "expected_bytes": int(window.bytes)},
    })
    values = np.asarray(fields["VALUES"], dtype=np.float64)
    ny, nx = values.shape
    lat, _ = grid.ij_to_latlon(np.ones(ny), np.arange(1, ny + 1, dtype=float))
    _, lon = grid.ij_to_latlon(np.arange(1, nx + 1, dtype=float), np.ones(nx))
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat[0] > south or lat[-1] < north or lon[0] > west or lon[-1] < east:
        raise DensityRefusal(  # pragma: no cover - a grid-placement defect
            f"the terrain analysis grid spans {lat[0]:.4f}..{lat[-1]:.4f} N, "
            f"{lon[0]:.4f}..{lon[-1]:.4f} E and does not cover the box "
            f"W,S,E,N = {west},{south},{east},{north}")
    provenance = {
        "terrain_source": "copernicus-dem-glo30",
        "tiles": [item.path.name for item in tiles],
        "tiles_absent": list(absent),
        "tiles_fetched_bytes": int(sum(item.bytes for item in tiles
                                       if not item.cache_hit)),
        "cache_root": "default per-user highres cache" if cache_root is None
        else str(cache_root),
        "window_sha256": window.sha256,
        "window_audit": audit,
        "offline": bool(offline),
        "analysis_km": analysis_km,
        "analysis_grid": "mercator mass points (rectilinear lat/lon), "
                         "area-averaged by static-fields",
        "box_wsen": [west, south, east, north],
        "fetch_seconds": round(fetched_s, 3),
    }
    return lat, lon, values, provenance


def terrain_from_glo30(west: float, south: float, east: float, north: float,
                       *, fine_km: float, background_km: float,
                       scale_km: float | None = None,
                       region: list[list[tuple[float, float]]] | None = None,
                       cache_root: Path | None = None, offline: bool = False,
                       max_cells: int = DEFAULT_MAX_RASTER_CELLS,
                       **thresholds: Any) -> GriddedSource:
    """Terrain-gradient source over a box (optionally clipped to polygons).

    ``scale_km`` is the length the terrain is judged at (default
    ``fine_km``): the DEM is area-averaged onto a grid of half that and
    smoothed by one cell, because terrain finer than the finest mesh
    spacing cannot be represented by it.
    """

    scale = float(fine_km if scale_km is None else scale_km)
    _require_positive(scale, "--terrain-scale-km")
    lat, lon, z, provenance = fetch_glo30_elevation(
        west, south, east, north, analysis_km=scale / 2.0,
        cache_root=cache_root, offline=offline, max_cells=max_cells)
    provenance["terrain_scale_km"] = scale
    return terrain_from_dem(lat, lon, z, fine_km=fine_km,
                            background_km=background_km, region=region,
                            origin=provenance, **thresholds)


# ---------------------------------------------------------------------------
# forecast fields (the adaptive cycle driver's entry point)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """One per-point scalar criterion and how it maps to spacing.

    ``coarse_at`` scores 0 (background), ``fine_at`` scores 1 (``fine_km``);
    either order is accepted, so "higher is finer" (wind speed, icing rate)
    and "lower is finer" (visibility, ceiling) are both one row.  Between
    the two the spacing is log-linear.  ``fine_km`` overrides the build's
    finest spacing for this criterion.
    """

    coarse_at: float
    fine_at: float
    fine_km: float | None = None

    def score(self, values: np.ndarray) -> np.ndarray:
        if not (math.isfinite(self.coarse_at) and math.isfinite(self.fine_at)) \
                or self.coarse_at == self.fine_at:
            raise DensityRefusal(
                f"criterion thresholds must be two distinct finite numbers, "
                f"got coarse_at={self.coarse_at!r} fine_at={self.fine_at!r}")
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return np.clip((values - self.coarse_at)
                           / (self.fine_at - self.coarse_at), 0.0, 1.0)


def criteria_spacing(criteria_values: Mapping[str, Any],
                     thresholds: Mapping[str, Criterion | Sequence[float]],
                     *, fine_km: float, background_km: float
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-point spacing: the minimum over criteria of each one's mapping.

    NaN values ask for nothing for that criterion at that point.
    """

    if not criteria_values:
        raise DensityRefusal("density_from_fields needs at least one criterion")
    missing = sorted(set(criteria_values) - set(thresholds))
    if missing:
        raise DensityRefusal(f"criteria {missing} have no thresholds")
    spacing: np.ndarray | None = None
    report: dict[str, Any] = {}
    for name, raw in criteria_values.items():
        rule = thresholds[name]
        if not isinstance(rule, Criterion):
            rule = Criterion(*rule)
        fine = float(rule.fine_km if rule.fine_km is not None else fine_km)
        _require_positive(fine, f"criterion {name} fine_km")
        if fine >= background_km:
            raise DensityRefusal(
                f"criterion {name} fine_km {fine} is not finer than the "
                f"{background_km} km background")
        values = np.asarray(raw, dtype=np.float64).ravel()
        if spacing is not None and values.shape != spacing.shape:
            raise DensityRefusal("criteria arrays differ in size")
        score = rule.score(values)
        h = background_km * (fine / background_km) ** score
        h = np.where(np.isfinite(score), h, background_km)
        spacing = h if spacing is None else np.minimum(spacing, h)
        report[name] = {"coarse_at": rule.coarse_at, "fine_at": rule.fine_at,
                        "fine_km": fine,
                        "points_refined": int(np.sum(score > 0.0)),
                        "points_at_fine": int(np.sum(score >= 1.0))}
    assert spacing is not None
    return spacing, report


def _median_neighbour_km(lat: np.ndarray, lon: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if lat.size < 2:
        raise DensityRefusal("support_km cannot be inferred from one point; "
                             "pass support_km")
    sample = np.linspace(0, lat.size - 1, min(lat.size, 200_000)).astype(int)
    tree = cKDTree(_xyz(lat, lon))
    dist, _ = tree.query(_xyz(lat[sample], lon[sample]), k=2, workers=-1)
    chord = float(np.median(dist[:, 1]))
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(chord / 2.0, 1.0))


def density_from_fields(lat: Any, lon: Any,
                        criteria_values: Mapping[str, Any],
                        thresholds: Mapping[str, Criterion | Sequence[float]],
                        *, fine_km: float, background_km: float,
                        support_km: float | None = None,
                        extra_sources: Sequence[DensitySource] = (),
                        allow_rough: bool = False,
                        grade_percent_per_cell: float | None = None,
                        cells_per_finest: float = DEFAULT_CELLS_PER_FINEST,
                        max_raster_cells: int = DEFAULT_MAX_RASTER_CELLS,
                        interp_tolerance: float = DEFAULT_INTERP_TOLERANCE,
                        origin: Mapping[str, Any] | None = None
                        ) -> "DensityRaster":
    """A density raster from per-point scalar criteria (forecast fields).

    ``lat``/``lon`` are point coordinates in degrees (any shape; mesh cell
    centres, or a meshgrid of a regular field), and each
    ``criteria_values[name]`` holds one value per point.
    ``thresholds[name]`` is a :class:`Criterion` or ``(coarse_at, fine_at)``
    / ``(coarse_at, fine_at, fine_km)``.  Each point asks for the minimum
    spacing over its criteria; each raster cell within ``support_km``
    (default: the points' median neighbour distance) takes its nearest
    point's request.  ``extra_sources`` (e.g. a corridor from
    :func:`corridor_from_assets`) are combined by ``min``.  The result is
    limited and resolved exactly like every other builder's.
    """

    lat_a = np.asarray(lat, dtype=np.float64).ravel()
    lon_a = np.asarray(lon, dtype=np.float64).ravel()
    if lat_a.shape != lon_a.shape or lat_a.size == 0:
        raise DensityRefusal("lat and lon must be non-empty and the same size")
    for name, values in criteria_values.items():
        if np.asarray(values).size != lat_a.size:
            raise DensityRefusal(
                f"criterion {name} has {np.asarray(values).size} values for "
                f"{lat_a.size} points")
    _check_spacings(fine_km, background_km)
    spacing, report = criteria_spacing(criteria_values, thresholds,
                                       fine_km=fine_km,
                                       background_km=background_km)
    if support_km is None:
        support_km = _median_neighbour_km(lat_a, lon_a)
    _require_positive(support_km, "support_km")
    sources: list[DensitySource] = list(extra_sources)
    if np.any(spacing < background_km * (1.0 - 1e-12)):
        point_source = PointSource(
            lat=lat_a, lon=lon_a,
            spacing_km=np.where(spacing < background_km, spacing, np.nan),
            support_km=float(support_km), background_km=float(background_km),
            origin={"criteria": report, **dict(origin or {})})
        sources.append(point_source)
    if not sources:
        raise DensityRefusal(
            "no point meets any criterion's coarse threshold, so the fields "
            "ask for no refinement; there is no raster to build")
    return build_density(
        sources, background_km=background_km,
        policy=gradient_policy(allow_rough=allow_rough,
                               grade_percent_per_cell=grade_percent_per_cell),
        cells_per_finest=cells_per_finest, max_raster_cells=max_raster_cells,
        interp_tolerance=interp_tolerance)


# ---------------------------------------------------------------------------
# the limiter and the instruments that read it
# ---------------------------------------------------------------------------


def limit_gradient(spacing: np.ndarray, grid: RasterGrid,
                   per_cell: float) -> None:
    """Lower ``spacing`` in place to its ``per_cell``-Lipschitz envelope.

    ``h(x) = min_y h(y) + g d(x, y)`` with ``d`` the 8-neighbour chamfer
    distance on the raster (axis steps of ``dy`` and the row's own ``dx``,
    diagonal steps of their hypotenuse).  Each sweep takes, row by row, the
    previous row's three neighbours and then an exact in-row envelope in
    both directions.  Because ``dx`` narrows poleward, a shortest path may
    detour poleward and come back, so sweeps run POLEWARD FIRST and repeat
    in alternating directions until a sweep changes nothing (two on a
    one-hemisphere raster; a raster across the equator can need a third).
    Spacing only ever goes DOWN: a request is never coarsened.
    """

    ny, nx = spacing.shape
    g = float(per_cell)
    dy = grid.dy_km
    dx = grid.dx_km_rows
    idx = np.arange(nx, dtype=np.float64)

    def in_row(row: np.ndarray, step_km: float) -> None:
        ramp = (g * step_km) * idx
        np.minimum(row, ramp + np.minimum.accumulate(row - ramp), out=row)
        np.minimum(row, np.minimum.accumulate((row + ramp)[::-1])[::-1] - ramp,
                   out=row)

    floor = float(np.min(spacing))
    gy = g * dy
    north_first = float(np.mean(grid.lat)) >= 0.0
    up, down = range(ny), range(ny - 1, -1, -1)
    orders = (up, down) if north_first else (down, up)
    for sweep in range(MAX_LIMITER_SWEEPS):
        changed = False
        previous = None
        for r in orders[sweep % 2]:
            row = spacing[r]
            before = row.copy() if sweep >= 1 else None
            if previous is not None:
                prev = spacing[previous]
                np.minimum(row, prev + gy, out=row)
                gd = g * math.hypot(dy, min(dx[r], dx[previous]))
                np.minimum(row[1:], prev[:-1] + gd, out=row[1:])
                np.minimum(row[:-1], prev[1:] + gd, out=row[:-1])
            in_row(row, float(dx[r]))
            if before is not None and not changed:
                changed = bool(np.any(row < before * (1.0 - 1e-13)))
            previous = r
        if sweep >= 1 and not changed:
            break
    # The envelope never goes below the input's own minimum; the ramp
    # arithmetic can round a few ulps under it, which would misreport the
    # finest request.
    np.maximum(spacing, floor, out=spacing)


def max_neighbour_gradient(spacing: np.ndarray, grid: RasterGrid) -> float:
    """Largest ``|dh| / distance`` over axis and diagonal neighbour pairs."""

    ny, nx = spacing.shape
    dy = grid.dy_km
    dx = grid.dx_km_rows
    worst = 0.0

    def row_max(diff: np.ndarray) -> np.ndarray:
        # Distances are constant along a row (pair), so the max |dh| per
        # row divided by that row's distance is the max ratio.
        np.abs(diff, out=diff)
        return diff.max(axis=1)

    for start in range(0, ny, _ROW_BLOCK):
        stop = min(start + _ROW_BLOCK + 1, ny)
        block = spacing[start:stop]
        dxb = dx[start:stop]
        if nx > 1:
            worst = max(worst, float(np.max(
                row_max(np.diff(block, axis=1)) / dxb)))
        if block.shape[0] > 1:
            worst = max(worst, float(np.max(
                row_max(np.diff(block, axis=0)))) / dy)
            if nx > 1:
                dd = np.hypot(dy, np.minimum(dxb[1:], dxb[:-1]))
                worst = max(worst, float(np.max(
                    row_max(block[1:, 1:] - block[:-1, :-1]) / dd)))
                worst = max(worst, float(np.max(
                    row_max(block[1:, :-1] - block[:-1, 1:]) / dd)))
    return worst


def probe_gradient_per_cell(spacing: np.ndarray, grid: RasterGrid,
                            background_km: float,
                            max_points: int = 2_000_000) -> float:
    """The generator's own instrument, read off the raster.

    ``rw_mpas_mesh`` (``steepest_gradient_reading_of``) steps the local
    spacing ``h`` east, west, north and south and takes the largest
    ``|h(q)/h(p) - 1|``.  Here ``h(q)`` is the raster's bilinear
    reconstruction.  Like the generator's reading it is a max over a probe
    set -- every refined cell, strided down to ``max_points`` -- so it is a
    lower bound on the peak; the neighbour bound is the guarantee.
    """

    from scipy.ndimage import map_coordinates

    stride = max(1, int(math.ceil(math.sqrt(spacing.size / max_points))))
    rows, cols = np.nonzero(
        spacing[::stride, ::stride] < background_km * (1.0 - 1e-12))
    if rows.size == 0:
        return 0.0
    rows *= stride
    cols *= stride
    h = spacing[rows, cols]
    dx = grid.dx_km_rows[rows]
    worst = 0.0
    for dr, dc in ((h / grid.dy_km, 0.0), (-h / grid.dy_km, 0.0),
                   (0.0, h / dx), (0.0, -h / dx)):
        q = map_coordinates(spacing, [rows + dr, cols + dc], order=1,
                            mode="nearest", prefilter=False)
        worst = max(worst, float(np.max(np.abs(q / h - 1.0))))
    return worst


def estimate_cells(spacing: np.ndarray, grid: RasterGrid) -> float:
    """``integral dA / (0.866 h^2)`` over the raster (hexagon cells)."""

    area_rows = (grid.dy_km * grid.dlon * KM_PER_DEG
                 * np.cos(np.radians(grid.lat)))
    total = 0.0
    for start in range(0, spacing.shape[0], _ROW_BLOCK):
        block = spacing[start:start + _ROW_BLOCK]
        total += float(np.sum(area_rows[start:start + _ROW_BLOCK, None]
                              / (HEX_AREA_FACTOR * block * block)))
    return total


# ---------------------------------------------------------------------------
# resolution and extent
# ---------------------------------------------------------------------------


def _check_spacings(fine_km: float, background_km: float) -> None:
    _require_positive(fine_km, "--fine-km")
    _require_positive(background_km, "--background-km")
    if fine_km >= background_km:
        raise DensityRefusal(
            f"--fine-km {fine_km} must be finer than --background-km "
            f"{background_km}")


@dataclass(frozen=True)
class GridPlan:
    grid: RasterGrid
    cell_km: float
    target_cell_km: float
    floor_cell_km: float
    coarsened_for_size: bool
    finest_km: float
    margin_km: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dlat_deg": self.grid.dlat, "dlon_deg": self.grid.dlon,
            "cell_km": self.cell_km,
            "cell_km_narrowest": float(np.min(self.grid.dx_km_rows)),
            "target_cell_km": self.target_cell_km,
            "target_cells_per_finest": self.finest_km / self.target_cell_km,
            "cells_per_finest": self.finest_km / self.cell_km,
            "coarsest_admissible_cell_km": self.floor_cell_km,
            "coarsened_for_size": self.coarsened_for_size,
            "rule": ("coarsened to fit the raster size limit; spacing "
                     "placement error bounded by the interpolation tolerance"
                     if self.coarsened_for_size else
                     "target cells per finest spacing"),
            "ramp_margin_km": self.margin_km,
        }


def plan_grid(sources: Sequence[DensitySource], *, background_km: float,
              per_cell: float, cells_per_finest: float = DEFAULT_CELLS_PER_FINEST,
              max_raster_cells: int = DEFAULT_MAX_RASTER_CELLS,
              interp_tolerance: float = DEFAULT_INTERP_TOLERANCE) -> GridPlan:
    """Extent (sources + the ramp to background) and resolution."""

    _require_positive(cells_per_finest, "--cells-per-finest")
    _require_positive(interp_tolerance, "--interp-tolerance")
    if int(max_raster_cells) < 16:
        raise DensityRefusal("--max-raster-cells must be at least 16")
    finest = min(source.finest_km() for source in sources)
    if finest >= background_km:
        raise DensityRefusal(
            f"the finest requested spacing {finest} km is not finer than the "
            f"{background_km} km background")
    south = north = west = east = None
    margin_max = 0.0
    for source in sources:
        s, n, w, e = source.bounds()
        margin = max(background_km - source.finest_km(), 0.0) / per_cell
        margin_max = max(margin_max, margin)
        pad_lat = margin / KM_PER_DEG
        lat_s, lat_n = s - pad_lat, n + pad_lat
        if max(abs(lat_s), abs(lat_n)) > MAX_ABS_LAT:
            raise DensityRefusal(
                f"the {source.name} source with its {margin:.0f} km ramp to "
                f"background reaches latitude {max(abs(lat_s), abs(lat_n)):.1f}"
                f"; a regular lat/lon density raster stops at {MAX_ABS_LAT}")
        cos_pole = math.cos(math.radians(max(abs(lat_s), abs(lat_n))))
        pad_lon = margin / (KM_PER_DEG * cos_pole)
        lon_w, lon_e = w - pad_lon, e + pad_lon
        south = lat_s if south is None else min(south, lat_s)
        north = lat_n if north is None else max(north, lat_n)
        west = lon_w if west is None else min(west, lon_w)
        east = lon_e if east is None else max(east, lon_e)
    assert south is not None and north is not None
    assert west is not None and east is not None
    if west < -180.0 or east > 180.0:
        raise DensityRefusal(
            f"the raster with its ramp to background spans longitudes "
            f"{west:.2f}..{east:.2f}, across the antimeridian; the "
            f"{SCHEMA} contract is one ascending longitude axis in "
            "-180..180, so this region cannot be expressed")
    cos_eq = 1.0 if south <= 0.0 <= north else math.cos(
        math.radians(min(abs(south), abs(north))))

    def make(cell_km: float) -> RasterGrid:
        dlat = cell_km / KM_PER_DEG
        dlon = cell_km / (KM_PER_DEG * cos_eq)
        lat0 = south - 2 * dlat
        lon0 = max(west - 2 * dlon, -180.0)
        ny = int(math.ceil((north - south) / dlat)) + 4
        nx = int(math.ceil((min(east + 2 * dlon, 180.0) - lon0) / dlon))
        return RasterGrid(lat0=lat0, lon0=lon0, dlat=dlat, dlon=dlon,
                          ny=ny, nx=nx)

    target = finest / cells_per_finest
    floor = max(target, math.sqrt(2.0) * interp_tolerance * finest / per_cell)
    grid = make(target)
    coarsened = False
    cell = target
    if grid.cells > max_raster_cells:
        coarsened = True
        cell = target * math.sqrt(grid.cells / max_raster_cells)
        grid = make(cell)
        while grid.cells > max_raster_cells:
            cell *= 1.01
            grid = make(cell)
        if cell > floor * (1.0 + 1e-9):
            at_floor = make(floor)
            raise DensityRefusal(
                f"the raster cannot be built inside the limits: it spans "
                f"{(north - south) * KM_PER_DEG:,.0f} km north-south (sources "
                f"plus a {margin_max:,.0f} km ramp to the {background_km} km "
                f"background at {per_cell * 100.0:.2f} %/cell).  "
                f"{cells_per_finest:g} cells per {finest} km spacing needs "
                f"{make(target).cells:,} raster cells; the coarsest admissible "
                f"cell ({floor:.4f} km, interpolation tolerance "
                f"{interp_tolerance:g}) still needs {at_floor.cells:,}, past "
                f"--max-raster-cells {max_raster_cells:,}.  Raise "
                "--max-raster-cells, coarsen --fine-km, lower "
                "--background-km, or relax --interp-tolerance")
    return GridPlan(grid=grid, cell_km=cell, target_cell_km=target,
                    floor_cell_km=floor, coarsened_for_size=coarsened,
                    finest_km=finest, margin_km=margin_max)


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


@dataclass
class DensityRaster:
    """A built raster: grid, spacing, provenance and the summary."""

    grid: RasterGrid
    spacing_km: np.ndarray
    background_km: float
    policy: GradientPolicy
    plan: GridPlan
    sources: list[dict[str, Any]]
    limiter: dict[str, Any]
    timings_s: dict[str, float]

    @property
    def lat(self) -> np.ndarray:
        return self.grid.lat

    @property
    def lon(self) -> np.ndarray:
        return self.grid.lon

    @property
    def min_spacing_km(self) -> float:
        return float(np.min(self.spacing_km))

    @property
    def max_spacing_km(self) -> float:
        return float(np.max(self.spacing_km))

    def attributes(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "min_spacing_km": self.min_spacing_km,
            "max_spacing_km": self.max_spacing_km,
            "background_km": float(self.background_km),
            "sources": json.dumps(self.sources, sort_keys=True,
                                  default=_json_default),
            "limiter": json.dumps(self.limiter, sort_keys=True,
                                  default=_json_default),
            "Conventions": "CF-1.8",
            "title": "woof hex requested mesh spacing",
            "source": "woof hex density",
        }

    def write(self, path: str | Path) -> Path:
        """Write the CF netCDF (atomically: a partial file, then rename)."""

        import netCDF4

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        try:
            self._write_to(partial, netCDF4)
            os.replace(partial, path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return path

    def _write_to(self, partial: Path, netCDF4: Any) -> None:
        ny, nx = self.spacing_km.shape
        with netCDF4.Dataset(partial, "w", format="NETCDF4") as ds:
            ds.createDimension("lat", ny)
            ds.createDimension("lon", nx)
            lat = ds.createVariable("lat", "f8", ("lat",))
            lat.units = "degrees_north"
            lat.standard_name = "latitude"
            lat.long_name = "latitude of raster cell centre"
            lat[:] = self.lat
            lon = ds.createVariable("lon", "f8", ("lon",))
            lon.units = "degrees_east"
            lon.standard_name = "longitude"
            lon.long_name = "longitude of raster cell centre"
            lon[:] = self.lon
            var = ds.createVariable(
                "spacing_km", "f8", ("lat", "lon"), zlib=True, complevel=2,
                shuffle=True, chunksizes=(min(ny, 512), min(nx, 512)))
            var.units = "km"
            var.long_name = "requested mesh cell spacing"
            var[:, :] = self.spacing_km
            ds.setncatts(self.attributes())

    def summary(self, path: str | Path | None = None) -> dict[str, Any]:
        return {
            "schema": SUMMARY_SCHEMA,
            "raster_schema": SCHEMA,
            "path": None if path is None else str(path),
            "extent": self.grid.extent(),
            "shape": [self.grid.ny, self.grid.nx],
            "raster_cells": self.grid.cells,
            "resolution": self.plan.as_dict(),
            "min_spacing_km": self.min_spacing_km,
            "max_spacing_km": self.max_spacing_km,
            "background_km": self.background_km,
            "estimated_cells": self.limiter["estimated_cells_after"],
            "estimated_cells_note": (
                "integral of dA / (0.866 h^2) over the raster extent only; "
                "the rest of the mesh is at background"),
            "limiter": self.limiter,
            "sources": self.sources,
            "timings_s": self.timings_s,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def build_density(sources: Sequence[DensitySource], *, background_km: float,
                  policy: GradientPolicy | None = None,
                  cells_per_finest: float = DEFAULT_CELLS_PER_FINEST,
                  max_raster_cells: int = DEFAULT_MAX_RASTER_CELLS,
                  interp_tolerance: float = DEFAULT_INTERP_TOLERANCE,
                  progress: Callable[[str], None] | None = None
                  ) -> DensityRaster:
    """Paint every source with ``min``, limit, verify, and summarise."""

    if not sources:
        raise DensityRefusal("a density raster needs at least one source")
    _require_positive(background_km, "--background-km")
    policy = policy or gradient_policy()
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now
        if progress is not None:
            progress(f"{name}: {timings[name]:.2f} s")

    g = policy.enforced_per_cell
    plan = plan_grid(sources, background_km=background_km, per_cell=g,
                     cells_per_finest=cells_per_finest,
                     max_raster_cells=max_raster_cells,
                     interp_tolerance=interp_tolerance)
    grid = plan.grid
    lap("plan")
    spacing = np.full((grid.ny, grid.nx), float(background_km),
                      dtype=np.float64)
    for source in sources:
        source.paint(grid, spacing, background_km)
    lap("paint")
    raw_cells = estimate_cells(spacing, grid)
    raw_gradient = max_neighbour_gradient(spacing, grid)
    raw = spacing.copy()
    lap("measure_raw")
    limit_gradient(spacing, grid, g)
    lap("limit")
    lowered = int(np.count_nonzero(spacing < raw * (1.0 - 1e-12)))
    del raw
    after_gradient = max_neighbour_gradient(spacing, grid)
    if after_gradient > policy.grade_per_cell * (1.0 + 1e-9):
        raise DensityRefusal(  # pragma: no cover - a limiter defect
            f"the limited raster reads {after_gradient * 100.0:.4f} %/cell "
            f"between neighbours, past the {policy.grade_per_cell * 100.0:.4f}"
            " %/cell it was limited to; refusing to write it")
    edges = np.concatenate((spacing[0], spacing[-1], spacing[:, 0],
                            spacing[:, -1]))
    if float(edges.min()) < background_km * (1.0 - 1e-9):
        raise DensityRefusal(  # pragma: no cover - a planning defect
            f"the raster edge reads {float(edges.min()):.4f} km, finer than "
            f"the {background_km} km background, so the generator would see "
            "a jump there; refusing to write it")
    probe = probe_gradient_per_cell(spacing, grid, background_km)
    after_cells = estimate_cells(spacing, grid)
    lap("measure_limited")
    limiter = {
        **policy.as_dict(),
        "max_neighbour_gradient_percent_before": raw_gradient * 100.0,
        "max_neighbour_gradient_percent_after": after_gradient * 100.0,
        "probe_gradient_percent_per_cell_after": probe * 100.0,
        "probe_note": ("rw_mpas_mesh's instrument (step h east/west/north/"
                       "south, bilinear) read on the raster; a max over a "
                       "probe set, so a lower bound on the peak"),
        "raster_cells_lowered": lowered,
        "fraction_lowered": lowered / grid.cells,
        "estimated_cells_before": raw_cells,
        "estimated_cells_after": after_cells,
        "cell_ratio_after_over_before": (after_cells / raw_cells
                                         if raw_cells > 0 else None),
    }
    return DensityRaster(
        grid=grid, spacing_km=spacing, background_km=float(background_km),
        policy=policy, plan=plan,
        sources=[source.provenance() for source in sources],
        limiter=limiter, timings_s=timings)


# ---------------------------------------------------------------------------
# reading (the contract, checked)
# ---------------------------------------------------------------------------


def read_density_raster(path: str | Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   dict[str, Any]]:
    """``(lat, lon, spacing_km, attrs)`` of a raster, refusing off-contract."""

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        attrs = {name: ds.getncattr(name) for name in ds.ncattrs()}
        if attrs.get("schema") != SCHEMA:
            raise DensityRefusal(
                f"{path}: schema {attrs.get('schema')!r} is not {SCHEMA!r}")
        for name in ("lat", "lon", "spacing_km"):
            if name not in ds.variables:
                raise DensityRefusal(f"{path}: variable {name} is missing")
        if "min_spacing_km" not in attrs:
            raise DensityRefusal(f"{path}: attribute min_spacing_km is missing")
        var = ds.variables["spacing_km"]
        if var.dimensions != ("lat", "lon") or var.dtype != np.float64:
            raise DensityRefusal(
                f"{path}: spacing_km must be float64 on (lat, lon), found "
                f"{var.dtype} on {var.dimensions}")
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        spacing = np.asarray(var[:, :], dtype=np.float64)
    for name, axis in (("lat", lat), ("lon", lon)):
        if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
            raise DensityRefusal(f"{path}: {name} must be 1-D and ascending")
    if not np.all(np.isfinite(spacing)) or float(spacing.min()) <= 0.0:
        raise DensityRefusal(f"{path}: spacing_km must be finite and positive")
    return lat, lon, spacing, attrs


# ---------------------------------------------------------------------------
# the door (argparse lives in woof.hex.cli; this is the handler body)
# ---------------------------------------------------------------------------


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise DensityRefusal(f"--bbox takes west,south,east,north; got {value!r}")
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError as error:
        raise DensityRefusal(f"--bbox values must be numbers; got {value!r}"
                             ) from error
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise DensityRefusal(
            f"--bbox needs west < east and south < north in range; got {value!r}")
    return west, south, east, north


def _load_polygon(path: str | Path
                  ) -> tuple[list[list[tuple[float, float]]], int]:
    """Outer rings of a GeoJSON polygon file, and how many holes it had."""

    from woof.energy.osm import _polygons_of

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DensityRefusal(f"--polygon {path} is not readable JSON: {error}"
                             ) from error
    outers, holes = _polygons_of(document, f"--polygon {path}")
    if not outers:
        raise DensityRefusal(f"--polygon {path} holds no polygon")
    return outers, holes


#: Terrain-only flags and their defaults: given without --terrain-gradient
#: they would do nothing, so they are refused rather than ignored.
_TERRAIN_ONLY_DEFAULTS = {
    "terrain_fine_km": None, "terrain_scale_km": None,
    "slope_coarse_deg": DEFAULT_SLOPE_COARSE_DEG,
    "slope_fine_deg": DEFAULT_SLOPE_FINE_DEG,
    "curvature_coarse": DEFAULT_CURVATURE_COARSE_PER_M,
    "curvature_fine": DEFAULT_CURVATURE_FINE_PER_M,
    "cache_root": None, "offline": False, "bbox": None, "polygon": None,
}


def run_density(arguments: Any) -> int:
    """``woof hex density``: build, write and print the JSON summary."""

    import sys

    _check_spacings(arguments.fine_km, arguments.background_km)
    started = time.perf_counter()
    policy = gradient_policy(allow_rough=arguments.allow_rough,
                             grade_percent_per_cell=arguments.grade_percent_per_cell)
    sources: list[DensitySource] = []
    if arguments.sites is not None:
        sources.append(corridor_from_sites(
            arguments.sites, fine_km=arguments.fine_km,
            corridor_km=arguments.corridor_km))
    if arguments.assets is not None:
        sources.append(corridor_from_assets(
            arguments.assets, fine_km=arguments.fine_km,
            corridor_km=arguments.corridor_km))
    if arguments.terrain_gradient:
        region = None
        holes = 0
        if arguments.polygon is not None:
            region, holes = _load_polygon(arguments.polygon)
            lons = [v[0] for ring in region for v in ring]
            lats = [v[1] for ring in region for v in ring]
            box = (min(lons), min(lats), max(lons), max(lats))
        elif arguments.bbox is not None:
            box = _parse_bbox(arguments.bbox)
        else:
            raise DensityRefusal(
                "--terrain-gradient needs the region to judge: pass --bbox "
                "W,S,E,N or --polygon FILE.geojson")
        terrain_fine = (arguments.terrain_fine_km
                        if arguments.terrain_fine_km is not None
                        else arguments.fine_km)
        _check_spacings(terrain_fine, arguments.background_km)
        sources.append(terrain_from_glo30(
            *box, fine_km=terrain_fine, background_km=arguments.background_km,
            scale_km=arguments.terrain_scale_km, region=region,
            cache_root=arguments.cache_root, offline=arguments.offline,
            max_cells=arguments.max_raster_cells,
            slope_coarse_deg=arguments.slope_coarse_deg,
            slope_fine_deg=arguments.slope_fine_deg,
            curvature_coarse_per_m=arguments.curvature_coarse,
            curvature_fine_per_m=arguments.curvature_fine))
        if holes:
            sources[-1].origin["note"] = (
                f"{holes} polygon hole(s) in --polygon ignored: terrain "
                "inside them is judged too, which only adds cells")
    else:
        given = sorted(
            "--" + name.replace("_", "-")
            for name, default in _TERRAIN_ONLY_DEFAULTS.items()
            if getattr(arguments, name, default) != default)
        if given:
            raise DensityRefusal(
                f"{', '.join(given)} only shape the --terrain-gradient "
                "source and mean nothing without it")
    if not sources:
        raise DensityRefusal(
            "name at least one source: --sites, --assets or "
            "--terrain-gradient")
    sources_s = time.perf_counter() - started
    raster = build_density(
        sources, background_km=arguments.background_km, policy=policy,
        cells_per_finest=arguments.cells_per_finest,
        max_raster_cells=arguments.max_raster_cells,
        interp_tolerance=arguments.interp_tolerance)
    clock = time.perf_counter()
    path = raster.write(arguments.output)
    raster.timings_s["sources"] = round(sources_s, 3)
    raster.timings_s["write"] = round(time.perf_counter() - clock, 3)
    raster.timings_s["total"] = round(time.perf_counter() - started, 3)
    if policy.workaround:
        print(policy.workaround, file=sys.stderr)
    print(json.dumps(raster.summary(path), indent=2, sort_keys=True,
                     default=_json_default))
    return 0
