"""``woof hex static-highres`` -- 30 m terrain and 100 m land use on a hex static.

WHY THIS DOOR EXISTS.  A hex static comes out of ``rw_mpas_static``, which
reads WPS_GEOG at 30 arc-seconds (about 900 m of latitude).  On a 50-100 m
cell that is the same terrain height replicated across a dozen cells: the
valley the energy corridor follows is not in the file at all.  This door
post-processes a static file, cell by cell, from the sources the WRF
high-resolution path already fetches, caches and hashes
(:mod:`woof.static.highres_fetch`):

* ``--terrain glo30``: Copernicus DEM GLO-30 (1 arc-second, about 30 m),
  area-averaged over each MPAS cell's Voronoi polygon into ``ter``;
* ``--landuse cglc``: CGLC-MODIS-LCZ (100 m), the per-cell area MODE for
  ``ivgtyp``/``lu_index`` through the same crosswalk the WRF path uses
  (:data:`woof.static.highres.CGLC_MODIS_LCZ_TO_MODIS21`), with
  ``landmask`` re-derived exactly the way ``rw_mpas_static`` derives it
  (``ivgtyp != iswater_lu``) and the land-masked climatologies carried
  across every land/water flip.

ORDER.  This must run BEFORE ``woof hex vertical`` (the vertical build
smooths ``ter`` into its height coordinate, so terrain changed afterwards is
terrain the model never sees consistently) and BEFORE ``woof hex register``
(the mesh-row registry pins the static's bytes by SHA-256).  A static that
already carries a vertical grid, or that this door already processed, is
refused by name.

HOW A CELL AVERAGE IS COMPUTED, and where the work runs.  The raster side
-- tile decode, mosaic, clip and the warp onto a sample lattice -- runs in
the Rust static-fields library through the existing seams
(:func:`woof.static.highres_fetch.derive_global_terrain_window`,
:func:`woof.static.highres_fetch.derive_landcover_window` and the
``continuous`` kind of ``gpuwm_static_highres_resample``), exactly as the
WRF path does.  The lattice is a Mercator grid over the footprint at a
spacing no coarser than the source pixel and fine enough that every cell
holds samples; each sample is the source pixel under it (nearest), so the
lattice is a rasterization of the piecewise-constant source.  Each sample
is then given to the cell whose CENTRE is nearest on the unit sphere
(``cKDTree``) -- which is exactly the Voronoi assignment, because an MPAS
cell is the Voronoi region of its generator -- and the per-cell sums are
accumulated with ``np.bincount`` weighted by each sample's true area.  At
the edge of the processed set (a culled mesh's outer ring, or the boundary
of ``--max-spacing-km``'s selection) the nearest-centre rule alone would
over-claim samples that belong to cells not in the set, so samples given to
those cells are additionally tested against the cell's own polygon.

WHAT IS NOT RECOMPUTED, by design.  ``var2d``, ``con``, ``oa1..4`` and
``ol1..4`` are the orographic gravity-wave-drag statistics.  ``rw_mpas_static``
transcribes ``mpas_init_atm_gwd.F``, which defines them over a lat/lon BOX
of the 30-arc-second topography sized from the cell's mean ``dcEdge``, and
the drag scheme was tuned against that definition (see
``rw-mpas/src/static_gwd.rs``).  A 30 m statistic would be a different
quantity under the same name, and at 50-100 m the drag scheme is meant to
be off anyway, so they are kept and the receipt says so
(``fields_kept_from_30s``).  Soil texture, albedo, green fraction, snow
albedo and deep-soil temperature stay on their 30-arc-second sources except
where a land/water flip forces a value (see :func:`_carry_land_masked`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MpasPortError

#: The schema this door writes into the static's global attributes and its
#: receipt; a static that already carries it is refused (double application
#: would average an average and mislabel its provenance).
SCHEMA = "woof-hex.static-highres.v1"

#: ``--terrain`` choices -> :data:`woof.static.highres_fetch.TERRAIN_SOURCES`
#: ids.  Rows, not code paths: another terrain source is another row here
#: plus a fetcher in the shared fetch module.
TERRAIN_CHOICES: Mapping[str, str] = {"glo30": "copernicus-dem-glo30"}
#: ``--landuse`` choices -> :data:`woof.static.highres_fetch.LANDCOVER_SOURCES`.
LANDUSE_CHOICES: Mapping[str, str] = {"cglc": "cglc-modis-lcz"}

#: MPAS's Earth radius (``mpas_constants``); a unit-sphere file is scaled to it.
MPAS_EARTH_RADIUS_M = 6_371_229.0
#: The sphere WPS projections and the static-fields warp work on.
WPS_EARTH_RADIUS_M = 6_370_000.0

#: Nominal source pixel sizes (metres of latitude) that bound the lattice:
#: sampling coarser than the source would discard source pixels.
SOURCE_PIXEL_M: Mapping[str, float] = {
    "copernicus-dem-glo30": 30.0,
    "cglc-modis-lcz": 100.0,
}

#: Cells coarser than this keep their 30-arc-second fields by default: at a
#: few kilometres the two area means agree to within the source noise, and a
#: global parent mesh would otherwise be sampled at 30 m over the planet.
DEFAULT_MAX_SPACING_KM = 2.0

#: The largest lattice this door samples before refusing.  At the measured
#: rate this is tens of minutes of work, and a footprint that needs more is
#: almost always a global parent whose ``--max-spacing-km`` selects too much.
MAX_SAMPLES = 1_000_000_000

#: Samples per warp call: bounds the resident lattice (values, xyz, cells,
#: about 100 bytes a sample).  Each call re-verifies the window's SHA-256
#: and reads only the source blocks under its band, so fewer, larger bands
#: cost memory, not decode work.
BAND_SAMPLES = 8_000_000

#: Lattice spacing as a fraction of the finest selected cell's INRADIUS.  A
#: disc of radius r always contains a point of a square lattice of spacing
#: s <= r * sqrt(2); 1.2 leaves margin for the Mercator scale change across
#: the footprint, so no selected cell can fall between samples.
_INRADIUS_TO_SPACING = 1.2

#: Samples across the median cell (about 55 per hexagon).  Measured on a
#: synthetic 100 m hex patch under a 0.45 m/m plane: the lattice-vs-hexagon
#: quantisation leaves a median |error| of 2.2 m at 4 across, 1.1 m at 8,
#: 0.45 m at 20 (tests/test_static_highres.py).  At the 30 m source pixel the
#: source itself is quantised more coarsely than that.
_SAMPLES_ACROSS_CELL = 8.0

#: A land-use mode needs at least this share of its cell's area classified;
#: below it (the open sea past CGLC's coastal zone is unclassified) the
#: 30-arc-second category is kept.
LANDUSE_MIN_VALID_FRACTION = 0.5

#: Variables whose presence means a vertical grid has been built from this
#: file's terrain (vertical_spec / init outputs), so changing ``ter`` now
#: would desynchronise the two.
VERTICAL_MARKERS = ("zgrid", "zz", "zxu", "hx", "rdzw", "fzm")

#: The land-use inventory the CGLC crosswalk targets.
MODIS_MMINLU = "MODIFIED_IGBP_MODIS_NOAH"

#: Land-masked climatologies ``rw_mpas_static`` writes as 0 over water.
LAND_MASKED_FIELDS = ("greenfrac", "albedo12m", "snoalb", "soiltemp",
                      "soilcomp")
#: Soil categories, carried from a donor across a land/water flip.
SOIL_CATEGORY_FIELDS = ("isltyp", "soilcat_top", "soilcl1", "soilcl2",
                        "soilcl3", "soilcl4")
#: Orographic GWD statistics: 30-arc-second box statistics by definition.
GWD_FIELDS = ("var2d", "con", "oa1", "oa2", "oa3", "oa4",
              "ol1", "ol2", "ol3", "ol4")

ORDER_NOTE = (
    "run before `woof hex vertical` and `woof hex register`: the vertical "
    "build smooths ter into its height coordinate and the mesh-row registry "
    "pins the static's SHA-256, so both must see the post-processed file")


class StaticHighresRefusal(MpasPortError):
    """A named refusal of the hex high-resolution static door."""


# ---------------------------------------------------------------------------
# Mesh geometry
# ---------------------------------------------------------------------------

def _unit_xyz(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    cos_lat = np.cos(lat_rad)
    return np.stack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad),
                     np.sin(lat_rad)), axis=-1)


@dataclass
class CellGeometry:
    """The parts of an MPAS mesh the cell average needs, on the unit sphere."""

    xyz: np.ndarray            # (nCells, 3) cell centres
    vertex_xyz: np.ndarray     # (nVertices, 3)
    n_edges: np.ndarray        # (nCells,)
    vertices: np.ndarray       # (nCells, maxEdges) 0-based, -1 invalid
    neighbours: np.ndarray     # (nCells, maxEdges) 0-based, -1 invalid
    radius_m: float

    @property
    def n_cells(self) -> int:
        return int(self.xyz.shape[0])


def read_cell_geometry(dataset) -> CellGeometry:
    """Geometry from an open netCDF4 static (or grid) dataset.

    Connectivity is MPAS's 1-based convention with 0 (or anything past the
    dimension) marking a missing entry, as a culled mesh's outer ring
    carries.  Coordinates must be radians, as every MPAS file writes them.
    """
    names = set(dataset.variables)
    needed = ("latCell", "lonCell", "latVertex", "lonVertex",
              "nEdgesOnCell", "verticesOnCell", "cellsOnCell")
    missing = [name for name in needed if name not in names]
    if missing:
        raise StaticHighresRefusal(
            f"the static carries no {missing}; a cell average needs each "
            "cell's polygon (verticesOnCell over latVertex/lonVertex) and its "
            "neighbours, which every rw_mpas_static output writes")

    def read(name, dtype):
        return np.asarray(dataset.variables[name][:], dtype=dtype)

    lat_c, lon_c = read("latCell", np.float64), read("lonCell", np.float64)
    lat_v, lon_v = read("latVertex", np.float64), read("lonVertex", np.float64)
    if (np.nanmax(np.abs(lat_c)) > math.pi / 2 + 1e-6
            or np.nanmax(np.abs(lat_v)) > math.pi / 2 + 1e-6):
        raise StaticHighresRefusal(
            "latCell/latVertex exceed pi/2: the file's coordinates are not "
            "radians, which is the only unit MPAS writes")
    n_cells, n_vertices = lat_c.size, lat_v.size
    n_edges = read("nEdgesOnCell", np.int64)
    vertices = read("verticesOnCell", np.int64) - 1
    neighbours = read("cellsOnCell", np.int64) - 1
    slot = np.arange(vertices.shape[1])[None, :]
    within = slot < n_edges[:, None]
    vertices = np.where(within & (vertices >= 0) & (vertices < n_vertices),
                        vertices, -1)
    neighbours = np.where(
        within & (neighbours >= 0) & (neighbours < n_cells), neighbours, -1)
    radius = float(getattr(dataset, "sphere_radius", MPAS_EARTH_RADIUS_M))
    if not radius > 1000.0:
        # A unit-sphere grid file: the geometry is the same, the metres are
        # MPAS's own Earth.
        radius = MPAS_EARTH_RADIUS_M
    return CellGeometry(
        xyz=_unit_xyz(lat_c, lon_c), vertex_xyz=_unit_xyz(lat_v, lon_v),
        n_edges=n_edges, vertices=vertices, neighbours=neighbours,
        radius_m=radius)


def _polygon_edges(geometry: CellGeometry, cells: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The polygon edges of ``cells``: ``(a, b, valid, complete)``.

    ``a[m, k]`` and ``b[m, k]`` are the unit vectors of vertices k and k+1
    of cell ``cells[m]``; ``valid`` is False past ``nEdgesOnCell`` or where
    either vertex is missing; ``complete`` is True for a cell every one of
    whose ``nEdgesOnCell`` edges is valid.  A culled mesh's outermost ring
    can lose vertices, and such a polygon is open: it bounds neither an
    area nor a point test, so every caller treats it as no polygon at all.
    """
    cells = np.asarray(cells, dtype=np.int64)
    vertices = geometry.vertices[cells]
    n_edges = geometry.n_edges[cells]
    slots = vertices.shape[1]
    live = np.arange(slots)[None, :] < n_edges[:, None]
    following = (np.arange(slots)[None, :] + 1) % np.maximum(n_edges, 1)[:, None]
    nxt = np.take_along_axis(vertices, following, axis=1)
    valid = live & (vertices >= 0) & (nxt >= 0)
    complete = (valid == live).all(axis=1) & (n_edges >= 3)
    a = geometry.vertex_xyz[np.where(valid, vertices, 0)]
    b = geometry.vertex_xyz[np.where(valid, nxt, 0)]
    return a, b, valid, complete


def _polygon_normals(geometry: CellGeometry, cells: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Edge-plane normals of the given cells' polygons, oriented inward.

    Returns ``(normals, valid)``: ``normals[m, k]`` is the unit normal of the
    great circle through vertices k and k+1 of cell ``cells[m]``, signed so
    the cell centre lies on its positive side; ``valid`` marks the live
    edges of a COMPLETE polygon (an open one has none, see
    :func:`_polygon_edges`).
    """
    cells = np.asarray(cells, dtype=np.int64)
    a, b, valid, complete = _polygon_edges(geometry, cells)
    valid &= complete[:, None]
    normals = np.cross(a, b)
    norm = np.linalg.norm(normals, axis=-1)
    valid &= norm > 0.0
    normals = normals / np.where(norm > 0.0, norm, 1.0)[..., None]
    side = np.einsum("mkj,mj->mk", normals, geometry.xyz[cells])
    normals *= np.where(side < 0.0, -1.0, 1.0)[..., None]
    return normals, valid


def cell_inradius_m(geometry: CellGeometry, cells: np.ndarray | None = None,
                    *, chunk: int = 250_000) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest edge of its polygon.

    About half the cell's ``dcEdge``.  NaN for an open polygon (a culled
    outer-ring cell that lost vertices).
    """
    cells = (np.arange(geometry.n_cells) if cells is None
             else np.asarray(cells, dtype=np.int64))
    out = np.full(cells.size, np.nan)
    for start in range(0, cells.size, chunk):
        part = cells[start:start + chunk]
        normals, valid = _polygon_normals(geometry, part)
        dots = np.einsum("mkj,mj->mk", normals, geometry.xyz[part])
        angle = np.where(valid, np.arcsin(np.clip(dots, -1.0, 1.0)), np.inf)
        nearest = angle.min(axis=1)
        out[start:start + chunk] = np.where(
            valid.any(axis=1) & np.isfinite(nearest),
            nearest * geometry.radius_m, np.nan)
    return out


def cell_spacing_m(geometry: CellGeometry, cells: np.ndarray | None = None,
                   *, chunk: int = 250_000) -> np.ndarray:
    """Area-equivalent cell spacing (m): the centre-to-centre distance of
    the regular hexagon with the cell's area, ``sqrt(2 A / sqrt(3))``.

    Area, not inradius, because a cell can be close to one neighbour and
    still reach far in another direction (the outer cells of a refined
    patch); its spacing is what it covers.  NaN for an open polygon, whose
    area is unknown -- such a cell is never selected.
    """
    cells = (np.arange(geometry.n_cells) if cells is None
             else np.asarray(cells, dtype=np.int64))
    out = np.full(cells.size, np.nan)
    for start in range(0, cells.size, chunk):
        part = cells[start:start + chunk]
        a, b, valid, complete = _polygon_edges(geometry, part)
        centre = geometry.xyz[part][:, None, :]
        area = 0.5 * np.linalg.norm(np.cross(a - centre, b - centre), axis=-1)
        area = np.where(valid, area, 0.0).sum(axis=1) * geometry.radius_m ** 2
        out[start:start + chunk] = np.where(
            complete, np.sqrt(2.0 * area / math.sqrt(3.0)), np.nan)
    return out


def select_cells(geometry: CellGeometry, spacing_m: np.ndarray,
                 max_spacing_km: float) -> np.ndarray:
    """Cells whose area-equivalent spacing is at most ``max_spacing_km``."""
    spacing_km = np.asarray(spacing_m) / 1000.0
    return np.isfinite(spacing_km) & (spacing_km <= float(max_spacing_km))


class VoronoiAssigner:
    """Give unit-sphere points to the selected cell whose polygon holds them.

    Nearest selected centre is the Voronoi assignment for every selected
    cell whose neighbours are all selected: its polygon is the intersection
    of the half-spaces against exactly those neighbours.  A selected cell
    with a neighbour outside the set (or missing, on a culled outer ring) is
    an EDGE cell, and a point nearest to it may in truth belong to the
    absent neighbour; those points are kept only when they fall inside the
    edge cell's own polygon.
    """

    def __init__(self, geometry: CellGeometry, selected: np.ndarray):
        from scipy.spatial import cKDTree

        selected = np.asarray(selected, dtype=bool)
        self.cells = np.flatnonzero(selected)
        if self.cells.size == 0:
            raise StaticHighresRefusal("no cell is selected for processing")
        self.geometry = geometry
        self._tree = cKDTree(geometry.xyz[self.cells])
        neighbours = geometry.neighbours[self.cells]
        slots = np.arange(neighbours.shape[1])[None, :]
        live = slots < geometry.n_edges[self.cells][:, None]
        outside = live & ((neighbours < 0)
                          | ~selected[np.where(neighbours < 0, 0, neighbours)])
        self.edge_local = outside.any(axis=1)
        edge = np.flatnonzero(self.edge_local)
        self._edge_slot = np.full(self.cells.size, -1, dtype=np.int64)
        self._edge_slot[edge] = np.arange(edge.size)
        self._edge_normals, self._edge_valid = _polygon_normals(
            geometry, self.cells[edge])
        open_polygons = ~self._edge_valid.any(axis=1)
        if open_polygons.any():
            # select_cells never selects an open polygon (its spacing is
            # NaN); one arriving here would accept every point near it.
            raise StaticHighresRefusal(
                f"{int(open_polygons.sum())} selected edge cell(s) have an "
                "open polygon (missing vertices), so no point test bounds them")

    @property
    def edge_cell_count(self) -> int:
        return int(np.count_nonzero(self.edge_local))

    def assign(self, xyz: np.ndarray) -> np.ndarray:
        """LOCAL selected-cell index per point (into ``self.cells``), -1 for
        a point that belongs to no selected cell."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        _, local = self._tree.query(xyz, workers=-1)
        local = np.asarray(local, dtype=np.int64)
        test = np.flatnonzero(self.edge_local[local])
        if test.size:
            slot = self._edge_slot[local[test]]
            side = np.einsum("mkj,mj->mk", self._edge_normals[slot], xyz[test])
            inside = np.all((side >= -1e-12) | ~self._edge_valid[slot], axis=1)
            local[test[~inside]] = -1
        return local


# ---------------------------------------------------------------------------
# The sample lattice
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleLattice:
    """A Mercator lattice over the footprint, cut into row bands.

    Mercator because its mass points are separable (longitude is linear in
    column, latitude a function of row), so every sample's position and
    area follow from two 1-D vectors, and because it is a projection the
    static-fields warp already serves.  ``dx_m`` is true at the footprint's
    equatorward edge and finer poleward of it; a sample's area is ``(dx_m * cos(lat) /
    cos(truelat))**2``.
    """

    lat_min: float
    lon_min: float
    truelat: float
    dx_m: float
    nx: int
    ny: int

    @classmethod
    def over(cls, bbox, dx_m: float) -> "SampleLattice":
        from woof.static.projection import MercatorGrid

        # dx is TRUE at the equatorward edge, so no row is sampled coarser
        # than asked (Mercator ground spacing is dx*cos(lat)/cos(truelat)).
        if bbox.lat_min <= 0.0 <= bbox.lat_max:
            truelat = 0.0
        elif bbox.lat_min > 0.0:
            truelat = float(bbox.lat_min)
        else:
            truelat = float(bbox.lat_max)
        probe = MercatorGrid(bbox.lat_min, bbox.lon_min, truelat, truelat,
                             bbox.lon_min, dx_m, dx_m, 3, 3,
                             known_x=1.0, known_y=1.0)
        i_max, j_max = probe.latlon_to_ij(bbox.lat_max, bbox.lon_max)
        nx = int(math.ceil(float(i_max) - 1.0)) + 1
        ny = int(math.ceil(float(j_max) - 1.0)) + 1
        return cls(float(bbox.lat_min), float(bbox.lon_min), float(truelat),
                   float(dx_m), max(nx, 1), max(ny, 1))

    @property
    def samples(self) -> int:
        return self.nx * self.ny

    def grid(self, row0: int = 0, rows: int | None = None):
        """The Mercator grid of rows ``row0 .. row0+rows-1`` (0-based, south
        first).  The reference point stays the lattice's south-west mass
        point, so every band shares one projection exactly."""
        from woof.static.projection import MercatorGrid

        rows = self.ny - row0 if rows is None else rows
        return MercatorGrid(self.lat_min, self.lon_min, self.truelat,
                            self.truelat, self.lon_min, self.dx_m, self.dx_m,
                            self.nx + 1, rows + 1,
                            known_x=1.0, known_y=1.0 - row0)

    def longitudes(self) -> np.ndarray:
        grid = self.grid()
        _, lon = grid.ij_to_latlon(np.arange(1, self.nx + 1, dtype=np.float64),
                                   np.ones(self.nx))
        return np.asarray(lon, dtype=np.float64)

    def latitudes(self, row0: int, rows: int) -> np.ndarray:
        grid = self.grid()
        y = np.arange(row0 + 1, row0 + rows + 1, dtype=np.float64)
        lat, _ = grid.ij_to_latlon(np.ones(rows), y)
        return np.asarray(lat, dtype=np.float64)

    def area_m2(self, lat_deg: np.ndarray) -> np.ndarray:
        scale = np.cos(np.radians(lat_deg)) / math.cos(math.radians(self.truelat))
        return (self.dx_m * scale) ** 2

    def bands(self, band_samples: int = BAND_SAMPLES):
        rows = max(1, int(band_samples) // max(self.nx, 1))
        for row0 in range(0, self.ny, rows):
            yield row0, min(rows, self.ny - row0)


#: ``sampler(bound_raster, grid) -> (ny, nx) float array, south row first``.
Sampler = Callable[[Any, Any], np.ndarray]


def rust_sampler(bound, grid) -> np.ndarray:
    """The source pixel under every mass point of ``grid`` (nearest), through
    the static-fields warp; non-finite where the source has no data."""
    from woof.static import rust_bridge
    from woof.static.highres import _raster_spec

    bridge = rust_bridge.route("hex static-highres lattice warp")
    if bridge is None:
        raise StaticHighresRefusal(
            "the Rust static-fields bridge is not in use "
            f"({rust_bridge.unavailable_reason() or 'WOOF_STATIC_PYTHON is set'}); "
            "this door decodes and warps rasters only through it and has no "
            "pure-Python body.  Build or stage the static-fields library "
            "(see `woof doctor`) and unset WOOF_STATIC_PYTHON")
    fields, _ = bridge.highres_resample({
        "kind": "continuous",
        "grid_spec": grid._rust_spec(),
        "method": "nearest",
        "source": _raster_spec(bound),
    })
    values = np.asarray(fields["VALUES"], dtype=np.float64)
    expected = (grid.e_sn - 1, grid.e_we - 1)
    if values.shape != expected:
        raise StaticHighresRefusal(
            f"the warp returned a {values.shape} plane for a {expected} "
            "lattice band")
    return values


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

@dataclass
class CellAccumulators:
    """Per-selected-cell sums over the samples each cell received."""

    n: int
    categories: int = 0
    area: np.ndarray = field(init=False)
    terrain_area: np.ndarray = field(init=False)
    terrain_sum: np.ndarray = field(init=False)
    terrain_sumsq: np.ndarray = field(init=False)
    landuse: np.ndarray | None = field(init=False)
    samples: int = 0
    samples_kept: int = 0

    def __post_init__(self):
        self.area = np.zeros(self.n)
        self.terrain_area = np.zeros(self.n)
        self.terrain_sum = np.zeros(self.n)
        self.terrain_sumsq = np.zeros(self.n)
        self.landuse = (np.zeros((self.n, self.categories))
                        if self.categories else None)

    def add_area(self, cells, weights):
        self.area += np.bincount(cells, weights, minlength=self.n)

    def add_terrain(self, cells, weights, values):
        ok = np.isfinite(values)
        c, w, v = cells[ok], weights[ok], values[ok]
        self.terrain_area += np.bincount(c, w, minlength=self.n)
        self.terrain_sum += np.bincount(c, w * v, minlength=self.n)
        self.terrain_sumsq += np.bincount(c, w * v * v, minlength=self.n)

    def add_landuse(self, cells, weights, categories):
        ok = categories > 0
        flat = cells[ok] * self.categories + (categories[ok] - 1)
        self.landuse += np.bincount(
            flat, weights[ok], minlength=self.n * self.categories
        ).reshape(self.n, self.categories)


def _category_lookup(crosswalk: Mapping[int, int], nodata: float | None
                     ) -> np.ndarray:
    size = max(int(k) for k in crosswalk) + 1
    lookup = np.full(max(size, 256), -1, dtype=np.int64)
    for raw, target in crosswalk.items():
        lookup[int(raw)] = int(target)
    if nodata is not None and 0 <= int(nodata) < lookup.size:
        lookup[int(nodata)] = 0
    return lookup


def map_categories(raw: np.ndarray, lookup: np.ndarray, *, source_id: str
                   ) -> np.ndarray:
    """Raw source classes -> target categories; 0 for no data.  An unmapped
    class is refused, as the WRF path's mapped-category resample refuses."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.zeros(raw.shape, dtype=np.int64)
    finite = np.isfinite(raw)
    if not finite.any():
        return out
    ints = raw[finite].astype(np.int64)
    inside = (ints >= 0) & (ints < lookup.size)
    mapped = np.full(ints.shape, -1, dtype=np.int64)
    mapped[inside] = lookup[ints[inside]]
    unknown = np.unique(ints[mapped < 0])
    if unknown.size:
        raise StaticHighresRefusal(
            f"land-use source {source_id!r} carries unmapped categories "
            f"{unknown.tolist()} inside the footprint")
    out[finite] = mapped
    return out


def accumulate(lattice: SampleLattice, assigner: VoronoiAssigner, *,
               terrain_bound=None, landuse_bound=None,
               landuse_lookup: np.ndarray | None = None,
               landuse_source_id: str = "", categories: int = 0,
               sampler: Sampler = rust_sampler,
               band_samples: int = BAND_SAMPLES,
               progress: Callable[[str], None] | None = None
               ) -> CellAccumulators:
    """Walk the lattice band by band and accumulate per-cell sums."""
    acc = CellAccumulators(assigner.cells.size,
                           categories if landuse_bound is not None else 0)
    lon = np.radians(lattice.longitudes())
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)
    bands = list(lattice.bands(band_samples))
    step = max(1, len(bands) // 10)
    for index, (row0, rows) in enumerate(bands):
        if progress is not None and (index % step == 0 or index == len(bands) - 1):
            progress(f"band {index + 1}/{len(bands)} ({lattice.samples:,} "
                     f"samples at {lattice.dx_m:.3g} m)")
        lat_deg = lattice.latitudes(row0, rows)
        lat = np.radians(lat_deg)
        xyz = np.empty((rows, lattice.nx, 3))
        xyz[..., 0] = np.cos(lat)[:, None] * cos_lon[None, :]
        xyz[..., 1] = np.cos(lat)[:, None] * sin_lon[None, :]
        xyz[..., 2] = np.sin(lat)[:, None]
        cells = assigner.assign(xyz.reshape(-1, 3))
        acc.samples += cells.size
        keep = cells >= 0
        if not keep.any():
            continue
        acc.samples_kept += int(np.count_nonzero(keep))
        weights = np.broadcast_to(lattice.area_m2(lat_deg)[:, None],
                                  (rows, lattice.nx)).reshape(-1)[keep]
        kept = cells[keep]
        acc.add_area(kept, weights)
        grid = None
        if terrain_bound is not None or landuse_bound is not None:
            grid = lattice.grid(row0, rows)
        if terrain_bound is not None:
            values = sampler(terrain_bound, grid).reshape(-1)[keep]
            acc.add_terrain(kept, weights, values)
        if landuse_bound is not None:
            raw = sampler(landuse_bound, grid).reshape(-1)[keep]
            acc.add_landuse(kept, weights, map_categories(
                raw, landuse_lookup, source_id=landuse_source_id))
    return acc


# ---------------------------------------------------------------------------
# Field updates
# ---------------------------------------------------------------------------

def terrain_update(acc: CellAccumulators, baseline: np.ndarray
                   ) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """New terrain for the selected cells plus coverage counts and the
    per-cell sub-cell standard deviation (NaN where uncovered).

    The area the source does not cover (an unpublished all-water tile)
    takes the cell's own 30-arc-second height, so a partly covered cell is
    the area mean of the merged field, and a wholly uncovered one keeps its
    baseline exactly.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    covered = acc.terrain_area > 0.0
    new = baseline.copy()
    uncovered_area = np.maximum(acc.area - acc.terrain_area, 0.0)
    new[covered] = ((acc.terrain_sum[covered]
                     + uncovered_area[covered] * baseline[covered])
                    / acc.area[covered])
    mean = np.divide(acc.terrain_sum, acc.terrain_area,
                     out=np.full(acc.n, np.nan), where=covered)
    second = np.divide(acc.terrain_sumsq, acc.terrain_area,
                       out=np.full(acc.n, np.nan), where=covered)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    full = covered & (acc.terrain_area >= acc.area * (1.0 - 1e-9))
    counts = {
        "cells_fully_covered": int(np.count_nonzero(full)),
        "cells_partly_covered": int(np.count_nonzero(covered & ~full)),
        "cells_uncovered_kept_baseline": int(np.count_nonzero(~covered)),
    }
    return new, counts, std


def landuse_mode(acc: CellAccumulators, baseline: np.ndarray
                 ) -> tuple[np.ndarray, dict[str, int]]:
    """Per-cell area mode of the mapped categories (lowest category on a
    tie).

    The area the source leaves unclassified (CGLC's 0: the open sea past
    its coastal zone) votes for the cell's own 30-arc-second category, so
    a half-classified coastal cell is not handed to the land by default;
    and a cell with under :data:`LANDUSE_MIN_VALID_FRACTION` of its area
    classified keeps the baseline outright.
    """
    baseline = np.asarray(baseline, dtype=np.int64)
    classified = acc.landuse.sum(axis=1)
    share = np.divide(classified, acc.area, out=np.zeros(acc.n),
                      where=acc.area > 0.0)
    apply = share >= LANDUSE_MIN_VALID_FRACTION
    votes = acc.landuse.copy()
    in_range = (baseline >= 1) & (baseline <= acc.categories)
    rows = np.flatnonzero(in_range)
    votes[rows, baseline[rows] - 1] += np.maximum(
        acc.area[rows] - classified[rows], 0.0)
    mode = np.argmax(votes, axis=1) + 1
    new = np.where(apply, mode, baseline)
    return new, {
        "cells_mode_applied": int(np.count_nonzero(apply)),
        "cells_kept_baseline_unclassified": int(np.count_nonzero(~apply)),
        "cells_category_changed": int(np.count_nonzero(new != baseline)),
    }


#: STATSGO's land-ice soil category, which WRF's consistency check pairs
#: with the land-use ice category.
STATSGO_LAND_ICE = 16


def _carry_land_masked(arrays: dict[str, np.ndarray], xyz: np.ndarray,
                       old_land: np.ndarray, new_land: np.ndarray, *,
                       old_category: np.ndarray | None = None,
                       new_category: np.ndarray | None = None,
                       isice: int | None = None) -> dict[str, int]:
    """Keep the land-masked fields consistent with a changed ``landmask``.

    ``rw_mpas_static`` averages green fraction, albedo, snow albedo, deep
    soil temperature and soil composition over LAND only and leaves 0 over
    water.  A cell that becomes land takes every such field, and its soil
    categories, from the nearest cell that was land (the WRF merge's
    nearest-climatology rule, :func:`woof.static.highres.merge_highres_overrides`);
    a cell that becomes water gets the water 0 and the soil categories of
    the nearest cell that was water (kept, and counted, when the file had
    no water cell to copy).  Every donor value is read from the fields as
    they were BEFORE this call, so one flipped cell never donates another
    flipped cell's new value.

    Land ice gets the pairing WRF's consistency check enforces: a cell that
    becomes the ice category takes the land-ice soil category, and a cell
    that stops being ice takes the soil categories of the nearest land
    cell that was not ice.  ``shdmin``/``shdmax`` are re-derived from the
    green fraction of every changed cell, exactly as the static writer
    derives them.
    """
    from scipy.spatial import cKDTree

    before = {name: np.array(arrays[name], copy=True)
              for name in LAND_MASKED_FIELDS + SOIL_CATEGORY_FIELDS
              if name in arrays}
    newly_land = new_land & ~old_land
    newly_water = ~new_land & old_land
    audit = {"newly_land_cells": int(newly_land.sum()),
             "newly_water_cells": int(newly_water.sum()),
             "newly_water_soil_kept_no_water_donor": 0,
             "newly_ice_cells": 0, "no_longer_ice_cells": 0}

    def donate(targets, donors, names):
        _, nearest = cKDTree(xyz[donors]).query(xyz[targets], workers=-1)
        source = donors[nearest]
        for name in names:
            if name in before:
                arrays[name][targets] = before[name][source]

    if newly_land.any():
        donors = np.flatnonzero(old_land)
        if donors.size == 0:
            raise StaticHighresRefusal(
                "the high-resolution land use makes land of a file that has "
                "no land cell at all, so there is no land climatology "
                "(green fraction, albedo, soil) to give the new land cells")
        donate(newly_land, donors, LAND_MASKED_FIELDS + SOIL_CATEGORY_FIELDS)
    if newly_water.any():
        for name in LAND_MASKED_FIELDS:
            if name in arrays:
                arrays[name][newly_water] = 0
        donors = np.flatnonzero(~old_land)
        if donors.size:
            donate(newly_water, donors, SOIL_CATEGORY_FIELDS)
        else:
            audit["newly_water_soil_kept_no_water_donor"] = int(
                newly_water.sum())
    if isice is not None and old_category is not None and new_category is not None:
        old_ice = np.asarray(old_category) == isice
        new_ice = np.asarray(new_category) == isice
        became_ice = new_ice & ~old_ice
        left_ice = old_ice & ~new_ice & new_land
        audit["newly_ice_cells"] = int(became_ice.sum())
        audit["no_longer_ice_cells"] = int(left_ice.sum())
        for name in ("isltyp", "soilcat_top"):
            if name in arrays:
                arrays[name][became_ice] = STATSGO_LAND_ICE
        if left_ice.any():
            donors = np.flatnonzero(old_land & ~old_ice)
            if donors.size:
                donate(left_ice, donors, SOIL_CATEGORY_FIELDS)
    changed = newly_land | newly_water
    if changed.any() and "greenfrac" in arrays:
        green = np.asarray(arrays["greenfrac"])
        if "shdmin" in arrays:
            arrays["shdmin"][changed] = green[changed].min(axis=1)
        if "shdmax" in arrays:
            arrays["shdmax"][changed] = green[changed].max(axis=1)
    return audit


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _footprint(geometry: CellGeometry, cells: np.ndarray, pad_deg: float):
    from woof.static.highres_fetch import (FootprintBBox,
                                           _continued_longitude_range)

    vertices = geometry.vertices[cells]
    points = np.concatenate((geometry.xyz[cells],
                             geometry.vertex_xyz[vertices[vertices >= 0]]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lon_min, lon_max = _continued_longitude_range(lon)
    if lon_max - lon_min >= 180.0:
        raise StaticHighresRefusal(
            "the selected cells span 180 degrees of longitude or more; a "
            "high-resolution window is one rectangle, so this would sample "
            "a hemisphere at source resolution.  Lower --max-spacing-km so "
            "only the refined region is selected, or cull the region first")
    return FootprintBBox(float(lat.min()), float(lat.max()),
                         lon_min, lon_max).padded(pad_deg)


def _offline_cached(path: Path, url: str):
    from woof.static.highres_fetch import _cache_hit

    return _cache_hit(Path(path), url)


def _offline_refusal(what: str, missing: Sequence[str], folder: Path):
    from woof.static.highres_fetch import HighresFetchRefusal

    return HighresFetchRefusal(
        f"[hex static-highres] --offline and {len(missing)} {what} "
        f"{'is' if len(missing) == 1 else 'are'} not in the cache: "
        f"{', '.join(missing)}.  Offline, a file never fetched and a file the "
        "source does not publish (an all-water GLO-30 tile) cannot be told "
        "apart, so nothing is assumed about either",
        remedy=(f"remedy: run the same command once without --offline to "
                f"fill {folder}, or stage the files there with their "
                ".sha256.json sidecars"),
        folders=(folder,))


def fetch_terrain(bbox, cache_root: Path, source_id: str, *, offline: bool,
                  urlopen=None) -> tuple[Any, dict[str, Any]]:
    """Fetch (or read cached) terrain tiles and derive the window; return
    ``(bound_raster, manifest)``."""
    from woof.static import highres_fetch as hf
    from woof.static.highres import BoundRaster

    coverage = hf.terrain_source_coverage(source_id)
    try:
        coverage.require_reach(bbox)
    except hf.CoverageError as error:
        raise StaticHighresRefusal(f"no terrain coverage: {error}") from error
    if source_id != "copernicus-dem-glo30":  # pragma: no cover - table guard
        raise StaticHighresRefusal(f"no fetcher wired for {source_id!r}")
    cache = Path(cache_root) / "copernicus_dem_glo30"
    tile_ids = hf.copernicus_dem_tile_ids(bbox)
    if offline:
        tiles, missing = [], []
        for tile in tile_ids:
            url = hf.COPERNICUS_DEM_TILE_URL.format(tile=tile)
            hit = _offline_cached(
                cache / f"Copernicus_DSM_COG_10_{tile}_DEM.tif", url)
            (tiles.append(hit) if hit is not None else missing.append(tile))
        if missing:
            raise _offline_refusal("GLO-30 tile(s)", missing, cache)
        absent: tuple[str, ...] = ()
    else:
        try:
            tiles, absent = hf.fetch_copernicus_dem_tiles(bbox, cache_root,
                                                          urlopen=urlopen)
        except (hf.CoverageError, hf.HighresRefusal) as error:
            raise StaticHighresRefusal(f"no terrain tiles: {error}") from error
    if not tiles:
        raise StaticHighresRefusal(
            f"no terrain coverage: Copernicus GLO-30 publishes none of the "
            f"tiles {list(tile_ids)} over the footprint {bbox.as_dict()} (it "
            "omits all-water tiles), so there is no 30 m terrain to average.  "
            "Drop --terrain, or check the footprint")
    try:
        window, window_audit = hf.derive_global_terrain_window(
            tiles, bbox, cache_root, sea_level_fill=None)
    except hf.HighresRefusal as error:
        raise StaticHighresRefusal(str(error)) from error
    bound = BoundRaster(
        path=Path(window.path), sha256=window.sha256, source_id=source_id,
        role="terrain", source_url=coverage.source_url,
        license_id=coverage.license_id, license_url=coverage.license_url,
        nominal_resolution=coverage.nominal_resolution,
        expected_bytes=int(window.bytes))
    manifest = {
        "terrain_source": source_id,
        "terrain_tiles": [{"tile": Path(t.path).name, "sha256": t.sha256,
                           "bytes": int(t.bytes), "url": t.url,
                           "cache_hit": bool(t.cache_hit)} for t in tiles],
        "terrain_tiles_absent": list(absent),
        "terrain_window": {"name": Path(window.path).name,
                           "sha256": window.sha256, "bytes": int(window.bytes)},
        "terrain_window_audit": window_audit,
        "terrain_vertical_datum": hf.COPERNICUS_DEM_VERTICAL_DATUM,
        "terrain_attribution": coverage.attribution,
        "terrain_license": coverage.license_id,
    }
    return bound, manifest


def fetch_landuse(bbox, cache_root: Path, source_id: str, *, offline: bool,
                  urlopen=None) -> tuple[Any, dict[str, Any], Any]:
    """Fetch (or read cached) the land-cover raster and derive the window;
    return ``(bound_raster, manifest, source_row)``."""
    from woof.static import highres_fetch as hf
    from woof.static.highres import BoundRaster

    source = hf.landcover_source(source_id)
    try:
        source.coverage.require_reach(bbox)
    except hf.CoverageError as error:
        raise StaticHighresRefusal(f"no land-use coverage: {error}") from error
    if source.fetch != "whole-geotiff":  # pragma: no cover - table guard
        raise StaticHighresRefusal(
            f"land-use source {source_id!r} is not a whole GeoTIFF")
    year = source.first_year
    if offline:
        path = Path(cache_root) / source.cache_dir / source.file_name
        raster = _offline_cached(path, source.url)
        if raster is None:
            raise _offline_refusal(f"{source.label} raster", [source.file_name],
                                   path.parent)
        downloaded = (raster,)
    else:
        try:
            downloaded, raster = hf.fetch_landcover(source, year, cache_root,
                                                    urlopen=urlopen)
        except (hf.CoverageError, hf.HighresRefusal) as error:
            raise StaticHighresRefusal(
                f"no land-use raster: {error}") from error
    try:
        window = hf.derive_landcover_window(raster, bbox, cache_root)
    except hf.CoverageError as error:
        raise StaticHighresRefusal(f"no land-use coverage: {error}") from error
    except hf.HighresRefusal as error:
        raise StaticHighresRefusal(str(error)) from error
    bound = BoundRaster(
        path=Path(window.path), sha256=window.sha256,
        source_id=source.bound_id(year), role="landcover",
        source_url=source.coverage.source_url,
        license_id=source.coverage.license_id,
        license_url=source.coverage.license_url,
        nominal_resolution=source.coverage.nominal_resolution,
        expected_bytes=int(window.bytes), reference_year=year,
        nodata_override=source.nodata)
    manifest = {
        "landuse_source": source.source_id,
        "landuse_year": year,
        "landuse_raster": {"name": Path(raster.path).name,
                           "sha256": raster.sha256, "bytes": int(raster.bytes),
                           "url": raster.url},
        "landuse_window": {"name": Path(window.path).name,
                           "sha256": window.sha256, "bytes": int(window.bytes)},
        "landuse_window_audit": hf.landcover_window_audit(window),
        "landuse_crosswalk": "CGLC_MODIS_LCZ_TO_MODIS21 (LCZ 51-61 -> 13 "
                             "urban; 17 sea and 21 inland water kept)",
        "landuse_attribution": source.coverage.attribution,
        "landuse_license": source.coverage.license_id,
        "landuse_downloads": len(downloaded),
    }
    return bound, manifest, source


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {"mean": round(float(values.mean()), 3),
            "std": round(float(values.std()), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3)}


def _char_text(variable) -> str:
    raw = np.asarray(variable[:])
    if raw.dtype.kind in ("S", "U"):
        return b"".join(np.atleast_1d(raw).astype("S1").tolist()).decode(
            "ascii", "ignore").strip("\x00 ").strip()
    return str(raw)


def _progress(message: str) -> None:
    print(f"[hex static-highres] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    from woof.static.highres import sha256_file

    return sha256_file(path)


def run_static_highres(static: Path, out: Path, *,
                       terrain: str | None = "glo30",
                       landuse: str | None = None,
                       cache_root: Path | None = None,
                       offline: bool = False,
                       max_spacing_km: float = DEFAULT_MAX_SPACING_KM,
                       sample_m: float | None = None,
                       clobber: bool = False,
                       receipt_path: Path | None = None,
                       sampler: Sampler = rust_sampler,
                       urlopen=None,
                       band_samples: int = BAND_SAMPLES) -> dict[str, Any]:
    """Post-process ``static`` into ``out``; return the receipt."""
    import netCDF4

    started = time.perf_counter()
    static, out = Path(static), Path(out)
    if terrain is None and landuse is None:
        raise StaticHighresRefusal(
            "neither --terrain nor --landuse selects a source; there is "
            "nothing to post-process")
    if terrain is not None and terrain not in TERRAIN_CHOICES:
        raise StaticHighresRefusal(
            f"--terrain {terrain!r} is not one of {sorted(TERRAIN_CHOICES)}")
    if landuse is not None and landuse not in LANDUSE_CHOICES:
        raise StaticHighresRefusal(
            f"--landuse {landuse!r} is not one of {sorted(LANDUSE_CHOICES)}")
    if not static.is_file():
        raise StaticHighresRefusal(f"--static {static} does not exist")
    if out.resolve() == static.resolve():
        raise StaticHighresRefusal(
            "-o names the input static; this door writes a new file so the "
            "30-arc-second original (and any registry row pinning its "
            "SHA-256) stays intact")
    if out.exists() and not clobber:
        raise StaticHighresRefusal(f"{out} exists; pass --clobber to replace it")
    if not (math.isfinite(max_spacing_km) and max_spacing_km > 0.0):
        raise StaticHighresRefusal("--max-spacing-km must be positive")
    if sample_m is not None and not (math.isfinite(sample_m) and sample_m > 0):
        raise StaticHighresRefusal("--sample-m must be positive")
    if cache_root is None:
        from woof.static.highres_production import default_highres_cache_root

        cache_root = default_highres_cache_root()
    cache_root = Path(cache_root)

    with netCDF4.Dataset(static, "r") as dataset:
        dataset.set_auto_mask(False)
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        if "highres_schema" in attrs:
            raise StaticHighresRefusal(
                f"{static} was already post-processed "
                f"({attrs['highres_schema']}, terrain "
                f"{attrs.get('highres_terrain_source', 'none')}); applying "
                "it again would average an average.  Start from the "
                "rw_mpas_static output")
        built = [name for name in VERTICAL_MARKERS if name in dataset.variables]
        if built:
            raise StaticHighresRefusal(
                f"{static} already carries a vertical grid ({built}); the "
                "vertical build smoothed the terrain this door would replace. "
                f"Post-process the static first: {ORDER_NOTE}")
        if "ter" not in dataset.variables and terrain is not None:
            raise StaticHighresRefusal(f"{static} carries no 'ter'")
        geometry = read_cell_geometry(dataset)
        arrays: dict[str, np.ndarray] = {}
        if terrain is not None:
            arrays["ter"] = np.array(dataset.variables["ter"][:],
                                     dtype=np.float64)
        iswater = isice = None
        if landuse is not None:
            for name in ("ivgtyp", "landmask"):
                if name not in dataset.variables:
                    raise StaticHighresRefusal(
                        f"{static} carries no {name!r}; --landuse rewrites it")
            mminlu = (_char_text(dataset.variables["mminlu"])
                      if "mminlu" in dataset.variables
                      else str(attrs.get("mminlu", "")))
            if mminlu and mminlu != MODIS_MMINLU:
                raise StaticHighresRefusal(
                    f"the static's land-use inventory is {mminlu!r}; the CGLC "
                    f"crosswalk targets {MODIS_MMINLU!r} (MODIS 21 classes), "
                    "and writing MODIS numbers into another inventory would "
                    "relabel every cell")
            iswater = (int(np.asarray(dataset.variables["iswater_lu"][:]).ravel()[0])
                       if "iswater_lu" in dataset.variables else 17)
            from woof.static.highres import MODIS21_ISICE, MODIS21_ISWATER

            isice = (int(np.asarray(dataset.variables["isice_lu"][:]).ravel()[0])
                     if "isice_lu" in dataset.variables else MODIS21_ISICE)

            if iswater != MODIS21_ISWATER:
                raise StaticHighresRefusal(
                    f"iswater_lu is {iswater}, not MODIS's {MODIS21_ISWATER}")
            for name in (("ivgtyp", "lu_index", "landmask")
                         + LAND_MASKED_FIELDS + SOIL_CATEGORY_FIELDS
                         + ("shdmin", "shdmax")):
                if name in dataset.variables:
                    arrays[name] = np.array(dataset.variables[name][:])

    spacing = cell_spacing_m(geometry)
    selected = select_cells(geometry, spacing, max_spacing_km)
    if not selected.any():
        finest = np.nanmin(spacing) / 1000.0
        raise StaticHighresRefusal(
            f"no cell is {max_spacing_km:g} km or finer (the finest is "
            f"{finest:.3g} km); raise --max-spacing-km to post-process "
            "coarser cells")
    assigner = VoronoiAssigner(geometry, selected)
    cells = assigner.cells
    sel_inradius = cell_inradius_m(geometry, cells)
    sel_spacing = spacing[cells]
    pixel = min(SOURCE_PIXEL_M[TERRAIN_CHOICES[terrain]] if terrain else math.inf,
                SOURCE_PIXEL_M[LANDUSE_CHOICES[landuse]] if landuse else math.inf)
    automatic = min(pixel, _INRADIUS_TO_SPACING * float(np.nanmin(sel_inradius)),
                    float(np.nanmedian(sel_spacing)) / _SAMPLES_ACROSS_CELL)
    dx = float(sample_m) if sample_m is not None else automatic
    # WPS-sphere metres: the lattice lives on the warp's sphere, the cell
    # metrics on MPAS's; the 0.02 % difference is far inside the margin.
    bbox = _footprint(geometry, cells, pad_deg=2.0 * dx / 111_000.0)
    lattice = SampleLattice.over(bbox, dx)
    if lattice.samples > MAX_SAMPLES:
        raise StaticHighresRefusal(
            f"the footprint {bbox.as_dict()} at a {dx:.3g} m lattice is "
            f"{lattice.samples:,} samples, beyond this door's "
            f"{MAX_SAMPLES:,}.  Lower --max-spacing-km so only the refined "
            "region is processed, cull the region first, or pass a coarser "
            "--sample-m")

    _progress(f"{cells.size:,} of {geometry.n_cells:,} cells selected; "
              f"footprint {bbox.as_dict()}; fetching sources")
    terrain_bound = landuse_bound = None
    manifest: dict[str, Any] = {}
    lookup = None
    landuse_source_id = ""
    categories = 0
    if terrain is not None:
        terrain_bound, terrain_manifest = fetch_terrain(
            bbox, cache_root, TERRAIN_CHOICES[terrain], offline=offline,
            urlopen=urlopen)
        manifest.update(terrain_manifest)
    if landuse is not None:
        from woof.static.highres import MODIS21_CATEGORY_COUNT

        landuse_bound, landuse_manifest, source = fetch_landuse(
            bbox, cache_root, LANDUSE_CHOICES[landuse], offline=offline,
            urlopen=urlopen)
        manifest.update(landuse_manifest)
        lookup = _category_lookup(source.crosswalk, source.nodata)
        landuse_source_id = source.source_id
        categories = MODIS21_CATEGORY_COUNT

    sample_started = time.perf_counter()
    acc = accumulate(lattice, assigner, terrain_bound=terrain_bound,
                     landuse_bound=landuse_bound, landuse_lookup=lookup,
                     landuse_source_id=landuse_source_id,
                     categories=categories, sampler=sampler,
                     band_samples=band_samples, progress=_progress)
    sample_seconds = time.perf_counter() - sample_started
    empty = int(np.count_nonzero(acc.area <= 0.0))
    if empty:
        raise StaticHighresRefusal(
            f"{empty} selected cell(s) received no lattice sample at "
            f"{dx:.3g} m; their average would be invented.  Use a finer "
            "--sample-m (or leave it automatic)")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "order": ORDER_NOTE,
        "static": str(static.resolve()),
        "static_sha256": _sha256(static),
        "out": str(out.resolve()),
        "cache_root": str(cache_root),
        "offline": bool(offline),
        "cells": geometry.n_cells,
        "cells_selected": int(cells.size),
        "max_spacing_km": float(max_spacing_km),
        "selected_spacing_km": {
            "min": round(float(np.nanmin(sel_spacing)) / 1000.0, 4),
            "median": round(float(np.nanmedian(sel_spacing)) / 1000.0, 4),
            "max": round(float(np.nanmax(sel_spacing)) / 1000.0, 4)},
        "edge_cells_polygon_tested": assigner.edge_cell_count,
        "cells_open_polygon_kept_baseline": int(np.count_nonzero(
            ~np.isfinite(spacing))),
        "footprint": bbox.as_dict(),
        "lattice": {"projection": "mercator", "dx_m": round(dx, 4),
                    "truelat": lattice.truelat, "nx": lattice.nx,
                    "ny": lattice.ny, "samples": acc.samples,
                    "samples_in_selected_cells": acc.samples_kept,
                    "resampling": "nearest (source pixel under each sample)",
                    "automatic": sample_m is None},
        "method": ("voronoi-area-mean: nearest-centre (cKDTree, unit sphere) "
                   "assignment of a Mercator sample lattice, polygon-tested "
                   "at the edge of the processed set, area-weighted "
                   "np.bincount; rasters decoded and warped by the Rust "
                   "static-fields library"),
        "fields_kept_from_30s": {
            "fields": [name for name in GWD_FIELDS],
            "why": ("orographic GWD statistics are defined by "
                    "mpas_init_atm_gwd.F over a 30-arc-second lat/lon box "
                    "sized from dcEdge, and the drag scheme is tuned to that "
                    "definition; a 30 m statistic would be another quantity "
                    "under the same name")},
        "sources": manifest,
        "fields_replaced": [],
    }

    written: dict[str, np.ndarray] = {}
    if terrain is not None:
        before = arrays["ter"][cells]
        new_ter, coverage_counts, sub_std = terrain_update(acc, before)
        after = arrays["ter"].copy()
        after[cells] = new_ter
        written["ter"] = after
        delta = new_ter - before
        receipt["terrain"] = {
            "source": TERRAIN_CHOICES[terrain],
            **coverage_counts,
            "before": _stats(before), "after": _stats(new_ter),
            "delta_mean_abs_m": round(float(np.abs(delta).mean()), 3),
            "delta_max_abs_m": round(float(np.abs(delta).max()), 3),
            "sub_cell_std_mean_m": (round(float(np.nanmean(sub_std)), 3)
                                    if np.isfinite(sub_std).any() else None),
        }
        receipt["fields_replaced"].append("ter")
    if landuse is not None:
        baseline_cat = np.asarray(arrays["ivgtyp"], dtype=np.int64)
        new_cat_sel, mode_counts = landuse_mode(acc, baseline_cat[cells])
        new_cat = baseline_cat.copy()
        new_cat[cells] = new_cat_sel
        old_land = np.asarray(arrays["landmask"]).astype(np.int64) != 0
        new_land = new_cat != iswater
        for name in LAND_MASKED_FIELDS + SOIL_CATEGORY_FIELDS + (
                "shdmin", "shdmax"):
            if name in arrays:
                arrays[name] = np.array(arrays[name], copy=True)
        carry = _carry_land_masked(arrays, geometry.xyz, old_land, new_land,
                                   old_category=baseline_cat,
                                   new_category=new_cat, isice=isice)
        written["ivgtyp"] = new_cat.astype(arrays["ivgtyp"].dtype)
        if "lu_index" in arrays:
            written["lu_index"] = new_cat.astype(arrays["lu_index"].dtype)
        written["landmask"] = new_land.astype(arrays["landmask"].dtype)
        for name in LAND_MASKED_FIELDS + SOIL_CATEGORY_FIELDS + (
                "shdmin", "shdmax"):
            if name in arrays and (carry["newly_land_cells"]
                                   + carry["newly_water_cells"]
                                   + carry["newly_ice_cells"]
                                   + carry["no_longer_ice_cells"]):
                written[name] = arrays[name]
        receipt["landuse"] = {
            "source": LANDUSE_CHOICES[landuse],
            "rule": (f"per-cell area mode; cells with under "
                     f"{LANDUSE_MIN_VALID_FRACTION:g} of their area "
                     "classified keep the 30-arc-second category; landmask "
                     f"= ivgtyp != iswater_lu ({iswater}), as rw_mpas_static "
                     "derives it"),
            **mode_counts, **carry,
        }
        receipt["fields_replaced"].extend(sorted(written.keys() - {"ter"}))

    _write(static, out, written, receipt, attrs)
    receipt["out_sha256"] = _sha256(out)
    receipt["seconds"] = {"sampling": round(sample_seconds, 3),
                          "total": round(time.perf_counter() - started, 3)}
    target = (Path(receipt_path) if receipt_path is not None
              else out.with_name(out.name + ".static-highres.json"))
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    receipt["receipt"] = str(target.resolve())
    return receipt


def _write(static: Path, out: Path, written: Mapping[str, np.ndarray],
           receipt: Mapping[str, Any], attrs: Mapping[str, Any]) -> None:
    """Copy the static, replace the named fields, stamp provenance, and
    publish atomically.  Every other byte of the original stays."""
    import netCDF4

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial-{os.getpid()}")
    shutil.copyfile(static, partial)
    try:
        with netCDF4.Dataset(partial, "r+") as dataset:
            for name, values in written.items():
                variable = dataset.variables[name]
                variable[:] = np.asarray(values).astype(variable.dtype,
                                                        copy=False)
            sources = receipt["sources"]
            stamp = {
                "highres_schema": SCHEMA,
                "highres_order": ORDER_NOTE,
                "highres_parent_static": static.name,
                "highres_parent_static_sha256": receipt["static_sha256"],
                "highres_method": receipt["method"],
                "highres_lattice": json.dumps(receipt["lattice"],
                                              sort_keys=True),
                "highres_fields_replaced": ",".join(receipt["fields_replaced"]),
                "highres_fields_kept_30s": ",".join(
                    receipt["fields_kept_from_30s"]["fields"]),
                "highres_max_spacing_km": float(receipt["max_spacing_km"]),
                "highres_cells_selected": int(receipt["cells_selected"]),
            }
            if "terrain_source" in sources:
                stamp.update({
                    "highres_terrain_source": sources["terrain_source"],
                    "highres_terrain_tiles": json.dumps(
                        [{"tile": t["tile"], "sha256": t["sha256"]}
                         for t in sources["terrain_tiles"]], sort_keys=True),
                    "highres_terrain_tiles_absent": json.dumps(
                        sources["terrain_tiles_absent"]),
                    "highres_terrain_window_sha256":
                        sources["terrain_window"]["sha256"],
                    "highres_terrain_vertical_datum":
                        sources["terrain_vertical_datum"],
                    "highres_terrain_attribution":
                        sources["terrain_attribution"],
                })
            if "landuse_source" in sources:
                stamp.update({
                    "highres_landuse_source": sources["landuse_source"],
                    "highres_landuse_raster_sha256":
                        sources["landuse_raster"]["sha256"],
                    "highres_landuse_window_sha256":
                        sources["landuse_window"]["sha256"],
                    "highres_landuse_crosswalk": sources["landuse_crosswalk"],
                    "highres_landuse_attribution":
                        sources["landuse_attribution"],
                })
            for key, value in stamp.items():
                dataset.setncattr(key, value)
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _choice(value: str | None) -> str | None:
    return None if value in (None, "none") else value


def run_cli(arguments: argparse.Namespace) -> int:
    """The ``woof hex static-highres`` handler (parser: :mod:`woof.hex.cli`,
    which stays importable without numpy)."""
    from woof.ingest.source_coverage import PreparationRefusal
    from woof.static.highres_refusal import HighresRefusal

    try:
        receipt = run_static_highres(
            arguments.static, arguments.output,
            terrain=_choice(arguments.terrain),
            landuse=_choice(arguments.landuse),
            cache_root=arguments.cache_root, offline=arguments.offline,
            max_spacing_km=arguments.max_spacing_km,
            sample_m=arguments.sample_m, clobber=arguments.clobber,
            receipt_path=arguments.receipt)
    except PreparationRefusal as refusal:
        print(f"woof hex static-highres: {refusal}", file=sys.stderr)
        print(refusal.remedy, file=sys.stderr)
        return 2
    except HighresRefusal as refusal:
        print(f"woof hex static-highres: {refusal}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"RECEIPT {receipt['receipt']}", flush=True)
    return 0


__all__ = [
    "CellGeometry", "DEFAULT_MAX_SPACING_KM", "LANDUSE_CHOICES", "SCHEMA",
    "SampleLattice", "StaticHighresRefusal", "TERRAIN_CHOICES",
    "VoronoiAssigner", "accumulate", "run_cli",
    "cell_inradius_m", "cell_spacing_m", "landuse_mode", "read_cell_geometry",
    "run_static_highres", "select_cells", "terrain_update",
]
