"""Building morphology on the model grid, from GEM-15 and footprint datasets.

Opt-in input for two consumers (neither is a WRF default):

* resolved obstacles at LES / high resolution (``obstacle_opt``, the
  porous immersed obstacles of the WRF route): per model column, the plan
  area fraction of buildings by height bin (``bldg_hist``), each bin's
  mean height (``bldg_hbin``) and the sub-grid frontal area index for flow
  along the model x and y axes (``frontal_x``/``frontal_y``).  The model
  turns these into a solid volume fraction and frontal-area density on its
  own levels at initialisation;
* the gridded urban morphology WRF's urban canopy schemes read from
  NUDAPT's ``URB_PARAM`` (:func:`urb_param`), at exactly the indices WRF
  4.7.1 ``module_initialize_real.F:3182-3215`` reads: 91 LP_URB2D, 92
  MH_URB2D, 93 STDH_URB2D, 94 HGT_URB2D, 95 LB_URB2D, 96-99 LF_URB2D and
  118-132 HI_URB2D.

Inputs, which may be combined:

* a building-height raster from GEM-15's surface minus terrain model
  (nDSM, 15 m; :func:`ndsm_heights` filters it: height window, optional
  canopy-spread test, optional footprint mask, minimum object area);
* building footprints with heights (Overture Maps, Microsoft
  GlobalMLBuildingFootprints, Google Open Buildings, OpenStreetMap) from a
  GeoJSON or GeoParquet file (:func:`read_footprints`), rasterised with
  their own height, else the GEM-15 nDSM median inside the footprint,
  else levels x storey height, else a declared default
  (:func:`rasterize_footprints`).

Every source keeps its licence and attribution (:data:`BUILDING_SOURCES`);
the product carries the ones that contributed.  Nothing here fetches
footprints or knows any place: geography arrives from the caller.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = "woof-buildings.v1"

#: Height-bin edges (m above ground) of the LES product: 2 m to 20 m, 5 m
#: to 100 m, 10 m to 300 m, and one bin to 400 m.  A building taller than
#: the last edge is refused by :func:`ndsm_heights`'s height window and
#: clipped into the last bin otherwise.
LES_BIN_EDGES = np.array(
    [*range(0, 20, 2), *range(20, 100, 5), *range(100, 300, 10), 300, 400],
    dtype=np.float64)

#: WRF's BEP height classes: ``hb_u(k) = 5 k`` m, k = 1..15
#: (module_sf_bep.F:3450-3453); a building goes to the nearest class.
HI_CLASS_HEIGHTS = 5.0 * np.arange(1, 16, dtype=np.float64)

#: 1-based URB_PARAM indices WRF 4.7.1 reads (module_initialize_real.F).
URB_PARAM_SIZE = 132
URB_PARAM_INDEX = {
    "LP_URB2D": 91, "MH_URB2D": 92, "STDH_URB2D": 93, "HGT_URB2D": 94,
    "LB_URB2D": 95, "LF_URB2D": 96, "HI_URB2D": 118,
}

#: Licences and attributions of the building sources this module reads.
#: Overture, Microsoft and OSM data are ODbL 1.0: a model input derived
#: from them is a Produced Work that must carry the attribution, and a
#: derived database shared publicly must itself be ODbL.
BUILDING_SOURCES = {
    "gem15-ndsm": dict(
        name="GEM-15 v1 surface minus terrain model (15 m, experimental)",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution=("This dataset is produced by Google. GEM-15 v1 "
                     "(Batchu et al. 2026, doi:10.31223/X55N75): building "
                     "heights are its dsm minus dtm, filtered by woof."),
        url=("https://developers.google.com/earth-engine/datasets/catalog/"
             "projects_climate-and-sustainability_assets_gem-15_v1")),
    "overture": dict(
        name="Overture Maps Foundation buildings theme",
        license_id="ODbL-1.0",
        license_url="https://opendatacommons.org/licenses/odbl/1-0/",
        attribution=("© OpenStreetMap contributors, Overture Maps "
                     "Foundation (ODbL 1.0)"),
        url="https://docs.overturemaps.org/guides/buildings/"),
    "microsoft": dict(
        name="Microsoft GlobalMLBuildingFootprints",
        license_id="ODbL-1.0",
        license_url="https://opendatacommons.org/licenses/odbl/1-0/",
        attribution="Microsoft GlobalMLBuildingFootprints (ODbL 1.0)",
        url="https://github.com/microsoft/GlobalMLBuildingFootprints"),
    "google-open-buildings": dict(
        name="Google Open Buildings (v3 polygons; 2.5D Temporal heights)",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution=("Google Open Buildings (CC BY 4.0; v3 polygons are "
                     "also offered under ODbL 1.0)"),
        url="https://sites.research.google/gr/open-buildings/"),
    "osm": dict(
        name="OpenStreetMap buildings",
        license_id="ODbL-1.0",
        license_url="https://opendatacommons.org/licenses/odbl/1-0/",
        attribution="© OpenStreetMap contributors (ODbL 1.0)",
        url="https://www.openstreetmap.org/copyright"),
    "user": dict(
        name="user-supplied footprints", license_id="unknown",
        license_url="", attribution="as supplied by the user", url=""),
}

#: Attribute names read for a footprint's height (metres) and storey count,
#: in order: Overture/OSM/Microsoft ``height``; Overture ``num_floors``,
#: OSM ``building:levels``.
HEIGHT_FIELDS = ("height", "building:height")
LEVEL_FIELDS = ("num_floors", "building:levels", "levels")


# --------------------------------------------------------------------------
# Rasters and grids
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HeightRaster:
    """Building height above ground on a north-up regular raster.

    ``height`` is (rows, cols), metres; 0 is open ground and NaN is unknown
    (a hole: excluded from every area, not counted as open).  ``x0``/``y0``
    are the upper-left corner, ``px``/``py`` positive pixel sizes in the
    units of ``crs`` (metres for UTM; ``crs=None`` means the model's own
    Cartesian plane, as an idealised case uses).
    """

    height: np.ndarray
    x0: float
    y0: float
    px: float
    py: float
    crs: str | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        h = np.asarray(self.height, dtype=np.float32)
        if h.ndim != 2:
            raise ValueError("building height raster must be 2-D")
        if not (self.px > 0 and self.py > 0):
            raise ValueError("pixel sizes must be positive")
        object.__setattr__(self, "height", h)
        for source in self.sources:
            if source not in BUILDING_SOURCES:
                raise KeyError(f"unknown building source {source!r}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape

    def centres(self):
        rows, cols = self.shape
        x = self.x0 + (np.arange(cols) + 0.5) * self.px
        y = self.y0 - (np.arange(rows) + 0.5) * self.py
        return x, y

    def supersampled(self, factor: int) -> "HeightRaster":
        """Each pixel split into ``factor`` x ``factor`` (nearest)."""
        factor = int(factor)
        if factor <= 1:
            return self
        h = np.repeat(np.repeat(self.height, factor, axis=0), factor, axis=1)
        return HeightRaster(h, self.x0, self.y0, self.px / factor,
                            self.py / factor, self.crs, self.sources)


class CartesianGrid:
    """Model mass cells on the raster's own plane: cell (j, i) spans
    ``[x0 + i dx, x0 + (i+1) dx) x [y0 + j dy, y0 + (j+1) dy)``."""

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 x0: float = 0.0, y0: float = 0.0, crs: str | None = None):
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        self.x0, self.y0 = float(x0), float(y0)
        self.crs = crs

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def fractional_ij(self, x, y, crs):
        if crs != self.crs:
            raise ValueError(f"raster CRS {crs!r} differs from the grid's "
                             f"{self.crs!r}")
        return ((np.asarray(x, dtype=np.float64) - self.x0) / self.dx,
                (np.asarray(y, dtype=np.float64) - self.y0) / self.dy)

    def axis_rotation(self, raster: HeightRaster) -> float:
        return 0.0


class ProjectionGrid:
    """A WRF domain (:class:`woof.static.projection.ProjectedGrid`): raster
    pixel centres go to lon/lat with pyproj and then through the domain's
    own ``latlon_to_ij`` (mass point (j, i) sits at x = i + 1, y = j + 1)."""

    def __init__(self, projection):
        self.projection = projection
        self.nx = int(projection.e_we) - 1
        self.ny = int(projection.e_sn) - 1
        self.dx, self.dy = float(projection.dx), float(projection.dy)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def _lonlat(self, x, y, crs):
        if crs in (None, "EPSG:4326", "OGC:CRS84"):
            if crs is None:
                raise ValueError("a projected domain needs a georeferenced "
                                 "raster (crs=None is the model plane)")
            return np.asarray(x, np.float64), np.asarray(y, np.float64)
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return tr.transform(np.asarray(x, np.float64),
                            np.asarray(y, np.float64))

    def fractional_ij(self, x, y, crs):
        lon, lat = self._lonlat(x, y, crs)
        gx, gy = self.projection.latlon_to_ij(lat, lon)
        return np.asarray(gx) - 0.5, np.asarray(gy) - 0.5

    def axis_rotation(self, raster: HeightRaster) -> float:
        """Angle (radians) from the raster's +x axis to the model's +x axis
        at the raster centre (UTM grid convergence plus map rotation)."""
        rows, cols = raster.shape
        xc = raster.x0 + 0.5 * cols * raster.px
        yc = raster.y0 - 0.5 * rows * raster.py
        step = 10.0 * raster.px
        gx, gy = self.fractional_ij([xc, xc + step], [yc, yc], raster.crs)
        return -math.atan2(float(gy[1] - gy[0]), float(gx[1] - gx[0]))


class VoronoiGrid:
    """The cells of an MPAS (hex) mesh as a one-row grid, shape (1, nCells).

    A point belongs to the cell whose centre is nearest (a Voronoi cell is
    exactly that set), found with a KD-tree on unit vectors.  The model axes
    x/y are local east/north, so ``frontal_x``/``frontal_y`` face flow along
    east and north.  A georeferenced raster goes to lon/lat with pyproj; an
    idealised raster (``crs=None``) lies on the plane tangent at
    ``plane_origin`` (lat, lon in degrees; azimuthal equidistant on the
    mesh's own ``radius``), x east and y north, metres.  ``dx``/``dy`` are
    the mean cell spacing, which only sets the raster supersampling.

    A point farther than twice its nearest cell's equivalent radius from
    that cell's centre is outside the grid (-1): on a limited-area mesh the
    boundary cells do not take everything beyond the domain.  Neighbours
    come from ``cellsOnCell`` (``from_mesh``) or else from the spherical
    Delaunay triangulation of the centres.
    """

    def __init__(self, lat_cell, lon_cell, area_cell, *, radius: float,
                 plane_origin: tuple[float, float] | None = None,
                 cells_on_cell=None, n_edges_on_cell=None):
        lat = np.asarray(lat_cell, np.float64)
        lon = np.asarray(lon_cell, np.float64)
        area = np.asarray(area_cell, np.float64)
        self.n_cells = int(lat.size)
        self.nx, self.ny = self.n_cells, 1
        self.radius = float(radius)
        spacing = float(np.sqrt(np.mean(area)))
        self.dx = self.dy = spacing
        self.plane_origin = plane_origin
        # Chord reach on the unit sphere: twice the equivalent-disc radius.
        self._reach = 2.0 * np.sqrt(area / math.pi) / self.radius
        self._transformers: dict = {}
        from scipy.spatial import cKDTree
        unit = _unit(lat, lon)
        self._tree = cKDTree(unit)
        if cells_on_cell is not None:
            coc = np.asarray(cells_on_cell, np.int64)
            used = np.arange(coc.shape[1])[None, :] < np.asarray(
                n_edges_on_cell, np.int64)[:, None]
            used &= (coc >= 0) & (coc < self.n_cells)
            rows = np.broadcast_to(np.arange(self.n_cells)[:, None], coc.shape)
            pairs = np.stack([rows[used], coc[used]], axis=1)
        else:
            from scipy.spatial import ConvexHull
            tri = ConvexHull(unit).simplices
            pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        self._pairs = np.unique(np.sort(pairs, axis=1), axis=0)

    @classmethod
    def from_mesh(cls, mesh, **kwargs) -> "VoronoiGrid":
        """From an MPAS mesh object (``arrays`` by MPAS name, ``attrs``)."""
        a = mesh.arrays
        radius = kwargs.pop("radius", None)
        if radius is None:
            radius = float(mesh.attrs["sphere_radius"])
        if "cellsOnCell" in a and "nEdgesOnCell" in a:
            kwargs.setdefault("cells_on_cell", a["cellsOnCell"])
            kwargs.setdefault("n_edges_on_cell", a["nEdgesOnCell"])
        return cls(a["latCell"], a["lonCell"], a["areaCell"], radius=radius,
                   **kwargs)

    def dilate(self, mask) -> np.ndarray:
        """``mask`` (1, nCells) grown by one ring of mesh neighbours."""
        m = np.asarray(mask, bool).reshape(-1)
        out = m.copy()
        a, b = self._pairs[:, 0], self._pairs[:, 1]
        out[a[m[b]]] = True
        out[b[m[a]]] = True
        return out.reshape(1, -1)

    @property
    def shape(self) -> tuple[int, int]:
        return 1, self.n_cells

    def _lonlat(self, x, y, crs):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        if crs is None:
            if self.plane_origin is None:
                raise ValueError("an idealised raster (crs=None) needs the "
                                 "VoronoiGrid's plane_origin")
            lat0, lon0 = (math.radians(v) for v in self.plane_origin)
            up = np.array([math.cos(lat0) * math.cos(lon0),
                           math.cos(lat0) * math.sin(lon0), math.sin(lat0)])
            east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
            north = np.cross(up, east)
            dist = np.hypot(x, y)
            ang = dist / self.radius
            with np.errstate(invalid="ignore", divide="ignore"):
                ux = np.where(dist > 0, x / dist, 0.0)
                uy = np.where(dist > 0, y / dist, 0.0)
            p = (np.cos(ang)[..., None] * up
                 + np.sin(ang)[..., None] * (ux[..., None] * east
                                             + uy[..., None] * north))
            return (np.degrees(np.arctan2(p[..., 1], p[..., 0])),
                    np.degrees(np.arcsin(np.clip(p[..., 2], -1.0, 1.0))))
        if _is_geographic(crs):
            return x, y
        tr = self._transformers.get(crs)
        if tr is None:
            from pyproj import Transformer
            tr = self._transformers[crs] = Transformer.from_crs(
                crs, "EPSG:4326", always_xy=True)
        return tr.transform(x, y)

    def cells(self, x, y, crs) -> np.ndarray:
        lon, lat = self._lonlat(x, y, crs)
        dist, index = self._tree.query(_unit(np.radians(lat), np.radians(lon)))
        index = np.asarray(index, np.int64)
        return np.where(np.asarray(dist) <= self._reach[index], index, -1)

    def fractional_ij(self, x, y, crs):
        cell = self.cells(x, y, crs)
        return cell + 0.5, np.full(cell.shape, 0.5)

    def model_cells(self, raster: HeightRaster, densify: int = 64) -> np.ndarray:
        """The (1, nCells) mask of cells holding a sample of the raster,
        sampled at least every half cell spacing (and at ``densify`` points)
        along each axis."""
        rows, cols = raster.shape
        per_axis = lambda n, step: min(n, max(densify, int(math.ceil(
            2.0 * n * step / self.dx)) + 1))
        r = np.linspace(0, rows - 1, per_axis(rows, raster.py)).round().astype(int)
        c = np.linspace(0, cols - 1, per_axis(cols, raster.px)).round().astype(int)
        xs, ys = raster.centres()
        X, Y = np.meshgrid(xs[c], ys[r])
        cell = self.cells(X.ravel(), Y.ravel(), raster.crs)
        mask = np.zeros(self.n_cells, bool)
        mask[cell[cell >= 0]] = True
        return mask[None]

    def axis_rotation(self, raster: HeightRaster) -> float:
        """Angle from the raster's +x axis to local east at its centre."""
        if raster.crs is None or _is_geographic(raster.crs):
            return 0.0
        rows, cols = raster.shape
        xc = raster.x0 + 0.5 * cols * raster.px
        yc = raster.y0 - 0.5 * rows * raster.py
        step = 10.0 * raster.px
        lon, lat = self._lonlat([xc, xc + step], [yc, yc], raster.crs)
        east = (lon[1] - lon[0]) * math.cos(math.radians(lat[0]))
        return -math.atan2(float(lat[1] - lat[0]), float(east))


def _unit(lat, lon):
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _is_geographic(crs) -> bool:
    if crs in ("EPSG:4326", "OGC:CRS84"):
        return True
    from pyproj import CRS
    return CRS.from_user_input(crs).is_geographic


def utm_raster(lon_min: float, lat_min: float, lon_max: float,
               lat_max: float, resolution_m: float,
               sources: tuple[str, ...] = ()) -> HeightRaster:
    """An empty (all open ground) north-up raster in the UTM zone of the
    box centre that covers the lon/lat box, for burning footprints into
    when no GEM-15 window gives the grid."""
    from pyproj import Transformer

    lon_c = 0.5 * (lon_min + lon_max)
    lat_c = 0.5 * (lat_min + lat_max)
    zone = int((lon_c + 180.0) // 6.0) % 60 + 1
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    # Densified edges: UTM parallels curve, so an edge's midpoint can lie
    # outside the box its corners span.
    xmin, ymin, xmax, ymax = tr.transform_bounds(
        lon_min, lat_min, lon_max, lat_max, densify_pts=64)
    res = float(resolution_m)
    x0 = math.floor(xmin / res) * res
    y1 = math.ceil(ymax / res) * res
    cols = int(math.ceil((xmax - x0) / res))
    rows = int(math.ceil((y1 - ymin) / res))
    return HeightRaster(np.zeros((rows, cols), np.float32), x0, y1, res, res,
                        f"EPSG:{epsg}", sources)


def _cells(grid, x, y, crs):
    """Flat cell index of each point, -1 outside the grid."""
    fi, fj = grid.fractional_ij(x, y, crs)
    i = np.floor(fi).astype(np.int64)
    j = np.floor(fj).astype(np.int64)
    inside = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
    return np.where(inside, j * grid.nx + i, -1)


# --------------------------------------------------------------------------
# GEM-15 nDSM filtering
# --------------------------------------------------------------------------

def ndsm_heights(dsm, dtm, *, pixel_area_m2: float, min_height: float = 2.5,
                 max_height: float = 400.0, min_area_m2: float = 200.0,
                 spread=None, max_spread_fraction: float = 0.5,
                 footprint_mask=None) -> tuple[np.ndarray, dict]:
    """Building heights from a surface and a terrain model on one raster.

    ``dsm - dtm`` is kept where it lies in ``[min_height, max_height]``.
    Vegetation is the main false positive; three optional filters act on
    it: ``spread`` (e.g. GEM-15's ``dsm_percentile_95 - dsm_percentile_5``
    or ``dsm_std``) rejects pixels whose within-pixel spread exceeds
    ``max_spread_fraction`` of their height (roofs are flat, crowns are
    not); ``footprint_mask`` keeps only pixels inside known footprints;
    connected objects (4-neighbour) smaller than ``min_area_m2`` go.
    Returns ``(height, receipt)``: NaN where either model is NaN.
    """
    from scipy import ndimage

    dsm = np.asarray(dsm, dtype=np.float64)
    dtm = np.asarray(dtm, dtype=np.float64)
    if dsm.shape != dtm.shape:
        raise ValueError(f"dsm {dsm.shape} and dtm {dtm.shape} differ")
    hole = ~(np.isfinite(dsm) & np.isfinite(dtm))
    ndsm = np.where(hole, 0.0, dsm - dtm)
    keep = (~hole) & (ndsm >= min_height) & (ndsm <= max_height)
    counts = {"pixels": int(dsm.size), "holes": int(hole.sum()),
              "in_height_window": int(keep.sum())}
    if spread is not None:
        spread = np.asarray(spread, dtype=np.float64)
        rough = np.isfinite(spread) & (spread > max_spread_fraction * ndsm)
        counts["rejected_spread"] = int((keep & rough).sum())
        keep &= ~rough
    if footprint_mask is not None:
        mask = np.asarray(footprint_mask, dtype=bool)
        counts["rejected_outside_footprints"] = int((keep & ~mask).sum())
        keep &= mask
    if min_area_m2 > 0 and keep.any():
        labels, n = ndimage.label(keep)
        area = np.bincount(labels.ravel(), minlength=n + 1) * pixel_area_m2
        small = area < min_area_m2
        small[0] = False
        drop = small[labels]
        counts["rejected_small_objects"] = int(drop.sum())
        keep &= ~drop
    counts["building_pixels"] = int(keep.sum())
    height = np.where(keep, ndsm, 0.0).astype(np.float32)
    height[hole] = np.nan
    receipt = dict(min_height_m=float(min_height),
                   max_height_m=float(max_height),
                   min_area_m2=float(min_area_m2),
                   max_spread_fraction=(float(max_spread_fraction)
                                        if spread is not None else None),
                   footprint_mask=footprint_mask is not None, **counts)
    return height, receipt


# --------------------------------------------------------------------------
# Footprints
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Footprint:
    """One building: ``polygons`` is a list of polygons, each a list of
    rings (exterior first) of (x, y) pairs in ``crs``."""

    polygons: tuple
    height: float = math.nan
    levels: float = math.nan
    source: str = "user"

    def bounds(self):
        pts = np.concatenate([np.asarray(r, np.float64).reshape(-1, 2)
                              for poly in self.polygons for r in poly])
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def geojson(self) -> dict:
        polys = [[[list(map(float, p)) for p in ring] for ring in poly]
                 for poly in self.polygons]
        if len(polys) == 1:
            return {"type": "Polygon", "coordinates": polys[0]}
        return {"type": "MultiPolygon", "coordinates": polys}


def _number(value) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) and number > 0 else math.nan


def _from_properties(props) -> tuple[float, float]:
    props = props or {}
    height = next((h for h in (_number(props.get(k)) for k in HEIGHT_FIELDS)
                   if not math.isnan(h)), math.nan)
    levels = next((v for v in (_number(props.get(k)) for k in LEVEL_FIELDS)
                   if not math.isnan(v)), math.nan)
    return height, levels


def _rings_from_geojson(geom) -> tuple:
    kind = geom.get("type")
    coords = geom.get("coordinates")
    if kind == "Polygon":
        polys = [coords]
    elif kind == "MultiPolygon":
        polys = coords
    else:
        return ()
    return tuple(tuple(tuple((float(p[0]), float(p[1])) for p in ring)
                       for ring in poly) for poly in polys)


def parse_wkb(data: bytes) -> tuple:
    """Polygons of an ISO/EWKB Polygon or MultiPolygon (Z/M dropped).

    Other geometry types return ``()``.  Written here so footprint input
    needs no shapely/GEOS.
    """
    view = memoryview(bytes(data))

    def read(offset):
        order = "<" if view[offset] == 1 else ">"
        (code,) = struct.unpack_from(order + "I", view, offset + 1)
        offset += 5
        has_srid = bool(code & 0x20000000)
        dims = 2 + bool(code & 0x80000000) + bool(code & 0x40000000)
        code &= 0x0FFFFFFF
        if has_srid:
            offset += 4
        base, iso = code % 1000, code // 1000
        dims += {0: 0, 1: 1, 2: 1, 3: 2}[iso]
        return order, base, dims, offset

    def polygon(offset, order, dims):
        (nrings,) = struct.unpack_from(order + "I", view, offset)
        offset += 4
        rings = []
        for _ in range(nrings):
            (npts,) = struct.unpack_from(order + "I", view, offset)
            offset += 4
            pts = np.frombuffer(view, dtype=order + "f8", count=npts * dims,
                                offset=offset).reshape(npts, dims)[:, :2]
            offset += 8 * npts * dims
            rings.append(tuple(map(tuple, pts.tolist())))
        return tuple(rings), offset

    order, base, dims, offset = read(0)
    if base == 3:
        poly, _ = polygon(offset, order, dims)
        return (poly,)
    if base == 6:
        (n,) = struct.unpack_from(order + "I", view, offset)
        offset += 4
        out = []
        for _ in range(n):
            sub_order, sub_base, sub_dims, offset = read(offset)
            if sub_base != 3:
                raise ValueError("MultiPolygon member is not a Polygon")
            poly, offset = polygon(offset, sub_order, sub_dims)
            out.append(poly)
        return tuple(out)
    return ()


def _bbox_hit(fp: Footprint, bbox) -> bool:
    if bbox is None:
        return True
    x0, y0, x1, y1 = fp.bounds()
    return not (x1 < bbox[0] or x0 > bbox[2] or y1 < bbox[1] or y0 > bbox[3])


def read_footprints(path, *, source: str = "user", bbox=None
                    ) -> tuple[list[Footprint], str]:
    """Footprints from a GeoJSON FeatureCollection or a GeoParquet file.

    ``bbox`` = (xmin, ymin, xmax, ymax) in the file's CRS keeps only
    footprints that touch it.  Returns ``(footprints, crs)``; GeoJSON is
    OGC:CRS84 (RFC 7946) and GeoParquet declares its CRS in the ``geo``
    metadata (OGC:CRS84 when absent, as the specification says).
    """
    if source not in BUILDING_SOURCES:
        raise KeyError(f"unknown building source {source!r}")
    path = Path(path)
    suffix = path.suffix.lower()
    out: list[Footprint] = []
    if suffix in (".geojson", ".json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        features = doc.get("features", []) if doc.get("type") == \
            "FeatureCollection" else [doc]
        for feature in features:
            geom = feature.get("geometry") or {}
            polys = _rings_from_geojson(geom)
            if not polys:
                continue
            height, levels = _from_properties(feature.get("properties"))
            fp = Footprint(polys, height, levels, source)
            if _bbox_hit(fp, bbox):
                out.append(fp)
        return out, "OGC:CRS84"
    if suffix in (".parquet", ".geoparquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        meta = (table.schema.metadata or {}).get(b"geo")
        geo = json.loads(meta) if meta else {}
        column = geo.get("primary_column", "geometry")
        info = geo.get("columns", {}).get(column, {})
        if info.get("encoding", "WKB").upper() != "WKB":
            raise ValueError(f"GeoParquet geometry encoding "
                             f"{info.get('encoding')!r} is not WKB")
        crs = _geoparquet_crs(info, path)
        names = set(table.column_names)
        geoms = table.column(column).to_pylist()
        props = {name: table.column(name).to_pylist()
                 for name in (*HEIGHT_FIELDS, *LEVEL_FIELDS) if name in names}
        for row, wkb in enumerate(geoms):
            if wkb is None:
                continue
            polys = parse_wkb(wkb)
            if not polys:
                continue
            height, levels = _from_properties(
                {name: values[row] for name, values in props.items()})
            fp = Footprint(polys, height, levels, source)
            if _bbox_hit(fp, bbox):
                out.append(fp)
        return out, crs
    raise ValueError(f"{path.name}: footprints must be .geojson/.json or "
                     ".parquet (GeoParquet)")


def _geoparquet_crs(info: dict, path) -> str:
    """A GeoParquet column's CRS: an omitted key is OGC:CRS84 (the spec's
    default); an explicit null is an unknown CRS and refuses; a PROJJSON
    definition is kept whole (as WKT) when it carries no authority id."""
    if "crs" not in info:
        return "OGC:CRS84"
    crs = info["crs"]
    if crs is None:
        raise ValueError(f"{Path(path).name}: the GeoParquet CRS is null "
                         "(unknown); reproject the footprints to a known CRS")
    if isinstance(crs, dict):
        ident = crs.get("id") or {}
        if ident.get("authority") and ident.get("code") is not None:
            return f"{ident['authority']}:{ident['code']}"
        from pyproj import CRS
        return CRS.from_json_dict(crs).to_wkt()
    return str(crs)


def _transform_footprints(footprints, src_crs, dst_crs):
    if src_crs == dst_crs:
        return footprints
    if dst_crs is None or src_crs is None:
        raise ValueError(
            f"footprints in {src_crs!r} cannot burn onto a raster in "
            f"{dst_crs!r}: one of them is the model's own plane (crs=None), "
            "which has no georeference; burn georeferenced footprints onto a "
            "georeferenced raster (utm_raster)")
    from pyproj import Transformer
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out = []
    for fp in footprints:
        polys = []
        for poly in fp.polygons:
            rings = []
            for ring in poly:
                arr = np.asarray(ring, np.float64)
                x, y = tr.transform(arr[:, 0], arr[:, 1])
                rings.append(tuple(zip(np.asarray(x).tolist(),
                                       np.asarray(y).tolist())))
            polys.append(tuple(rings))
        out.append(Footprint(tuple(polys), fp.height, fp.levels, fp.source))
    return out


def _footprint_median(fp: Footprint, values: np.ndarray, like: HeightRaster
                      ) -> float:
    """Median of ``values`` (>0, finite) over ``fp``'s own pixels, read in
    a window around the footprint so other footprints cannot hide them."""
    from rasterio import features
    from rasterio.transform import from_origin

    rows, cols = like.shape
    minx, miny, maxx, maxy = fp.bounds()
    c0 = max(0, int(math.floor((minx - like.x0) / like.px)))
    c1 = min(cols, int(math.ceil((maxx - like.x0) / like.px)))
    r0 = max(0, int(math.floor((like.y0 - maxy) / like.py)))
    r1 = min(rows, int(math.ceil((like.y0 - miny) / like.py)))
    if c1 <= c0 or r1 <= r0:
        return math.nan
    mask = features.rasterize(
        [(fp.geojson(), 1)], out_shape=(r1 - r0, c1 - c0),
        transform=from_origin(like.x0 + c0 * like.px,
                              like.y0 - r0 * like.py, like.px, like.py),
        fill=0, dtype="uint8").astype(bool)
    v = values[r0:r1, c0:c1][mask]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if v.size else math.nan


def _same_crs(a, b) -> bool:
    if a == b:
        return True
    if a is None or b is None:
        return False
    from pyproj import CRS
    return CRS.from_user_input(a) == CRS.from_user_input(b)


def _same_lattice(a: HeightRaster, b: HeightRaster) -> bool:
    """Same CRS, origin and pixel size (to a millionth of a pixel)."""
    return (_same_crs(a.crs, b.crs)
            and math.isclose(a.px, b.px, rel_tol=1e-9)
            and math.isclose(a.py, b.py, rel_tol=1e-9)
            and abs(a.x0 - b.x0) <= 1e-6 * b.px
            and abs(a.y0 - b.y0) <= 1e-6 * b.py)


def rasterize_footprints(footprints: Sequence[Footprint], footprint_crs,
                         like: HeightRaster, *, ndsm=None,
                         ndsm_sources: tuple[str, ...] = ("gem15-ndsm",),
                         storey_height_m: float = 3.0,
                         default_height_m: float | None = None,
                         ndsm_medians: Sequence[float] | None = None
                         ) -> tuple[HeightRaster, dict]:
    """Burn footprints into ``like``'s raster (pixel-centre rule).

    Each building's height is, in order: its own ``height`` attribute; the
    median of ``ndsm`` (a building-height raster on ``like``'s grid, e.g.
    from :func:`ndsm_heights` with no area filter, as an array or a
    :class:`HeightRaster`) over its own pixels, whatever other footprints
    cover them; its storey count x ``storey_height_m``;
    ``default_height_m``.  A building with none of these is dropped and
    counted.  Overlaps keep the taller, whatever the input order.  When an
    nDSM median supplies any height, the nDSM's sources (a HeightRaster's
    own, else ``ndsm_sources``) join the result's.  ``ndsm_medians``
    (one value per footprint, NaN for none) gives those medians already
    read elsewhere, e.g. from GEM-15 windows on their own lattices, in
    place of ``ndsm``.
    """
    from rasterio import features
    from rasterio.transform import from_origin

    footprints = _transform_footprints(list(footprints), footprint_crs,
                                       like.crs)
    rows, cols = like.shape
    transform = from_origin(like.x0, like.y0, like.px, like.py)
    if isinstance(ndsm, HeightRaster):
        if ndsm.shape != like.shape or not _same_lattice(ndsm, like):
            raise ValueError("ndsm must be on like's grid (same CRS, origin "
                             "and pixel size); a same-sized window from "
                             "other ground would give every footprint the "
                             "wrong height")
        ndsm_sources = ndsm.sources or tuple(ndsm_sources)
        ndsm = ndsm.height
    values = None if ndsm is None else np.asarray(ndsm, np.float64)
    if values is not None and values.shape != (rows, cols):
        raise ValueError("ndsm must be on like's grid")
    median = np.full(len(footprints) + 1, np.nan)
    if ndsm_medians is not None:
        if values is not None:
            raise ValueError("give ndsm or ndsm_medians, not both")
        given = np.asarray(ndsm_medians, np.float64)
        if given.shape != (len(footprints),):
            raise ValueError("ndsm_medians needs one value per footprint")
        median[1:] = given
    if values is not None:
        for n, fp in enumerate(footprints, start=1):
            if math.isnan(fp.height):
                median[n] = _footprint_median(fp, values, like)
    heights = np.zeros(len(footprints) + 1)
    provenance = {"attribute": 0, "ndsm_median": 0, "storeys": 0,
                  "default": 0, "dropped": 0}
    keys: list[str] = []
    for n, fp in enumerate(footprints, start=1):
        if not math.isnan(fp.height):
            heights[n], key = fp.height, "attribute"
        elif not math.isnan(median[n]):
            heights[n], key = median[n], "ndsm_median"
        elif not math.isnan(fp.levels):
            heights[n], key = fp.levels * storey_height_m, "storeys"
        elif default_height_m is not None:
            heights[n], key = float(default_height_m), "default"
        else:
            heights[n], key = 0.0, "dropped"
        provenance[key] += 1
        keys.append(key)
    # Taller buildings win overlaps: burn again in ascending height order.
    order = np.argsort(heights[1:], kind="stable")
    shapes = [(footprints[k].geojson(), float(heights[k + 1]))
              for k in order if heights[k + 1] > 0]
    if shapes:
        burnt = features.rasterize(shapes, out_shape=(rows, cols),
                                   transform=transform, fill=0.0,
                                   dtype="float32")
        # Credit only footprints that show: burn their indices in the same
        # order, so a building hidden under a taller one, dropped, or off
        # the raster contributes no source.
        index = features.rasterize(
            [(footprints[k].geojson(), int(k) + 1) for k in order
             if heights[k + 1] > 0], out_shape=(rows, cols),
            transform=transform, fill=0, dtype="int32")
        shown = np.unique(index[index > 0]) - 1
    else:
        burnt = np.zeros((rows, cols), np.float32)
        shown = np.zeros(0, np.int64)
    used = {footprints[int(k)].source for k in shown}
    if any(keys[int(k)] == "ndsm_median" for k in shown):
        used.update(ndsm_sources)
    used = sorted(used)
    raster = HeightRaster(burnt, like.x0, like.y0, like.px, like.py,
                          like.crs, tuple(used))
    receipt = dict(footprints=len(footprints), height_from=provenance,
                   storey_height_m=float(storey_height_m),
                   default_height_m=default_height_m, sources=used)
    return raster, receipt


# --------------------------------------------------------------------------
# Aggregation onto the model grid
# --------------------------------------------------------------------------

@dataclass
class BuildingMorphology:
    """Per-cell building statistics on a (ny, nx) model grid."""

    sampled_area: np.ndarray          # m2 of known ground under each cell
    lambda_p: np.ndarray              # plan area fraction
    mean_height: np.ndarray           # plan-area-weighted mean height (m)
    std_height: np.ndarray            # plan-area-weighted std (m)
    lambda_b: np.ndarray              # building surface / plan area
    frontal_x: np.ndarray             # frontal area index, flow along x
    frontal_y: np.ndarray             # frontal area index, flow along y
    frontal_d45: np.ndarray           # flow along +x+y (WRF LF sector 2)
    frontal_d135: np.ndarray          # flow along +x-y (WRF LF sector 4)
    hi: np.ndarray                    # (15, ny, nx) BEP class fractions
    bldg_hist: np.ndarray             # (nb, ny, nx) plan fraction per bin
    bldg_hbin: np.ndarray             # (nb, ny, nx) mean height per bin
    bin_edges: np.ndarray = field(default_factory=lambda: LES_BIN_EDGES)
    sources: tuple[str, ...] = ()
    receipt: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.lambda_p.shape

    def lambda_f_sectors(self) -> np.ndarray:
        """LF_URB2D's four wind sectors (module_sf_urban.F:708-733, sector
        by WDR = atan2(U10, V10)): 1 flow along y (N-S), 2 along +x+y,
        3 along x (E-W), 4 along +x-y; each is projected from the walls'
        own orientations."""
        return np.stack([self.frontal_y, self.frontal_d45, self.frontal_x,
                         self.frontal_d135])

    def solid_fraction(self, z_bottom, z_top) -> np.ndarray:
        """Building volume fraction of the layer [z_bottom, z_top) (m above
        ground; scalars or arrays broadcasting to (ny, nx))."""
        zb = np.asarray(z_bottom, np.float64)
        zt = np.asarray(z_top, np.float64)
        depth = np.maximum(zt - zb, 1e-9)
        frac = np.zeros(self.shape)
        for b in range(self.bldg_hist.shape[0]):
            h = self.bldg_hbin[b]
            overlap = np.clip(np.minimum(h, zt) - zb, 0.0, None) / depth
            frac += self.bldg_hist[b] * np.minimum(overlap, 1.0)
        return frac


class _Sums:
    def __init__(self, ncell: int, nbins: int):
        self.n = ncell
        self.area = np.zeros(ncell)
        self.bldg = np.zeros(ncell)
        self.h1 = np.zeros(ncell)
        self.h2 = np.zeros(ncell)
        self.wall = np.zeros(ncell)
        self.fx = np.zeros(ncell)
        self.fy = np.zeros(ncell)
        self.f45 = np.zeros(ncell)
        self.f135 = np.zeros(ncell)
        self.hi = np.zeros((15, ncell))
        self.bin_area = np.zeros((nbins, ncell))
        self.bin_h = np.zeros((nbins, ncell))

    def add(self, target, cells, weights):
        ok = cells >= 0
        target += np.bincount(cells[ok], weights=weights[ok],
                              minlength=self.n)


def _accumulate(sums: _Sums, raster: HeightRaster, grid, edges, rotation):
    h = raster.height.astype(np.float64)
    known = np.isfinite(h)
    xs, ys = raster.centres()
    pixel = raster.px * raster.py
    rows = h.shape[0]
    # Row blocks keep the projected coordinates small.
    block = max(1, 2_000_000 // max(1, h.shape[1]))
    nb = len(edges) - 1
    # A raster-x wall's normal lies at -rotation in the model plane, a
    # raster-y wall's at 90 degrees more.  Flow along angle a sees half a
    # wall's area times |cos(normal - a)| (its windward faces).
    def seen(alpha):
        return (abs(math.cos(rotation + alpha)),
                abs(math.sin(rotation + alpha)))
    planes = ((sums.fx, seen(0.0)), (sums.fy, seen(0.5 * math.pi)),
              (sums.f45, seen(0.25 * math.pi)),
              (sums.f135, seen(0.75 * math.pi)))
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block + 1)        # one extra row for y edges
        hb = h[r0:r1]
        kb = known[r0:r1]
        X, Y = np.meshgrid(xs, ys[r0:r1])
        own = slice(0, min(block, rows - r0))
        cells = _cells(grid, X[own].ravel(), Y[own].ravel(), raster.crs)
        hv = hb[own].ravel()
        kv = kb[own].ravel()
        c = np.where(kv, cells, -1)
        sums.add(sums.area, c, np.full(hv.shape, pixel))
        built = kv & (hv > 0)
        cb = np.where(built, cells, -1)
        hw = np.where(built, hv, 0.0)
        sums.add(sums.bldg, cb, np.full(hv.shape, pixel))
        sums.add(sums.h1, cb, hw * pixel)
        sums.add(sums.h2, cb, hw * hw * pixel)
        cls = np.clip(np.rint(hw / 5.0).astype(np.int64) - 1, 0, 14)
        for k in range(15):
            sel = np.where(cls == k, cb, -1)
            sums.add(sums.hi[k], sel, np.full(hv.shape, pixel))
        bins = np.clip(np.searchsorted(edges, hw, side="right") - 1, 0, nb - 1)
        for b in np.unique(bins[built]):
            sel = np.where(bins == b, cb, -1)
            sums.add(sums.bin_area[b], sel, np.full(hv.shape, pixel))
            sums.add(sums.bin_h[b], sel, hw * pixel)
        # Wall faces between x neighbours (length py) and y neighbours
        # (length px), assigned to the cell under the face midpoint.
        hx = hb[own]
        kx = kb[own]
        dxh = np.abs(hx[:, 1:] - hx[:, :-1])
        okx = kx[:, 1:] & kx[:, :-1] & (dxh > 0)
        if okx.any():
            mx = 0.5 * (X[own][:, 1:] + X[own][:, :-1])
            cx = _cells(grid, mx[okx], Y[own][:, 1:][okx], raster.crs)
            area = dxh[okx] * raster.py
            sums.add(sums.wall, cx, area)
            for target, (from_x, _from_y) in planes:
                sums.add(target, cx, 0.5 * area * from_x)
        hy, ky, Yy, Xy = hb, kb, Y, X
        dyh = np.abs(hy[1:] - hy[:-1])
        oky = ky[1:] & ky[:-1] & (dyh > 0)
        if oky.any():
            my = 0.5 * (Yy[1:] + Yy[:-1])
            cy = _cells(grid, Xy[1:][oky], my[oky], raster.crs)
            area = dyh[oky] * raster.px
            sums.add(sums.wall, cy, area)
            for target, (_from_x, from_y) in planes:
                sums.add(target, cy, 0.5 * area * from_y)


def _grid_key(raster: HeightRaster):
    """Rasters sharing a key lie on one pixel lattice and can be mosaicked."""
    def phase(origin, step):
        return round((origin / step) % 1.0, 6) % 1.0
    return (raster.crs, raster.px, raster.py, phase(raster.x0, raster.px),
            phase(raster.y0, raster.py))


def _mosaic(rasters: Sequence[HeightRaster]) -> HeightRaster:
    """One raster from rasters on one lattice: where two know the same
    pixel the taller wins (the footprints' rule); a pixel none knows is a
    hole (NaN).  Walls across the old seams are then seen once."""
    if len(rasters) == 1:
        return rasters[0]
    first = rasters[0]
    px, py = first.px, first.py
    x0 = min(r.x0 for r in rasters)
    y0 = max(r.y0 for r in rasters)
    cols = max(int(round((r.x0 - x0) / px)) + r.shape[1] for r in rasters)
    rows = max(int(round((y0 - r.y0) / py)) + r.shape[0] for r in rasters)
    out = np.full((rows, cols), np.nan, np.float32)
    sources: set[str] = set()
    for r in rasters:
        c = int(round((r.x0 - x0) / px))
        w = int(round((y0 - r.y0) / py))
        view = out[w:w + r.shape[0], c:c + r.shape[1]]
        view[...] = np.fmax(view, r.height)
        sources.update(r.sources)
    return HeightRaster(out, x0, y0, px, py, first.crs,
                        tuple(sorted(sources)))


def _model_cells(raster: HeightRaster, grid, densify: int = 256):
    """The model cells the raster's extent reaches (a boolean (ny, nx)
    mask), from its densified outline mapped onto the grid (a projected
    edge can bow).  An outline lying exactly on a cell edge does not reach
    the cell beyond it; a raster off the grid reaches none."""
    custom = getattr(grid, "model_cells", None)
    if custom is not None:
        return custom(raster)
    from affine import Affine
    from rasterio import features

    rows, cols = raster.shape
    x0, x1 = raster.x0, raster.x0 + cols * raster.px
    y1, y0 = raster.y0, raster.y0 - rows * raster.py
    t = np.linspace(0.0, 1.0, densify + 1)[:-1]
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full_like(t, x1),
                         x1 - (x1 - x0) * t, np.full_like(t, x0)])
    ys = np.concatenate([np.full_like(t, y0), y0 + (y1 - y0) * t,
                         np.full_like(t, y1), y1 - (y1 - y0) * t])
    fi, fj = grid.fractional_ij(xs, ys, raster.crs)
    fi, fj = np.asarray(fi, np.float64), np.asarray(fj, np.float64)
    ci, cj = fi.mean(), fj.mean()
    shrink = 1.0 - 1e-7
    ring = list(zip((ci + (fi - ci) * shrink).tolist(),
                    (cj + (fj - cj) * shrink).tolist()))
    ring.append(ring[0])
    ny, nx = grid.shape
    return features.rasterize(
        [({"type": "Polygon", "coordinates": [ring]}, 1)],
        out_shape=(ny, nx), transform=Affine.identity(), fill=0,
        all_touched=True, dtype="uint8").astype(bool)


def aggregate(rasters: Iterable[HeightRaster], grid, *,
              bin_edges=LES_BIN_EDGES, min_samples: int = 9
              ) -> BuildingMorphology:
    """Per-cell morphology of one or more building-height rasters.

    Rasters are supersampled (nearest) until at least ``min_samples``
    pixels fall in a cell, so a 15 m source still resolves a 10 m grid.
    Rasters on one pixel lattice (same CRS, pixel size and alignment) are
    mosaicked first: where two know the same pixel the taller wins, and
    walls across their seams count once.  Rasters on different lattices
    must lie at least a model cell apart (the cells their densified
    outlines reach are compared; an overlap or a shared seam refuses, a
    raster off the grid is ignored): resample them onto one lattice
    first.  Only rasters that sample some cell are credited as sources.
    Pixels are assigned by centre; a wall face (a height step between two
    neighbouring known pixels) by its midpoint.  Frontal area index is
    half the wall area facing each model axis per plan area (the windward
    faces of closed buildings), with the raster's rotation onto the model
    axes taken at its centre.  Unknown (NaN) pixels count nowhere.
    """
    edges = np.asarray(bin_edges, np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    ny, nx = grid.shape
    sums = _Sums(ny * nx, edges.size - 1)
    sources: set[str] = set()
    factors = []
    groups: dict = {}
    for raster in rasters:
        if raster.crs is not None and _is_geographic(raster.crs):
            raise ValueError(f"building rasters must be in metres; "
                             f"{raster.crs} is geographic (use the source's "
                             "UTM grid, :func:`utm_raster`)")
        # A raster that reaches no cell, or knows no pixel, contributes
        # nothing: it is neither mosaicked, checked nor credited.
        cells = _model_cells(raster, grid)
        if not cells.any() or not np.isfinite(raster.height).any():
            continue
        key = _grid_key(raster)
        group = groups.setdefault(key, ([], np.zeros(grid.shape, bool)))
        group[0].append(raster)
        group[1][...] |= cells
    mosaics = [_mosaic(members) for members, _ in groups.values()]
    if len(mosaics) > 1:
        from scipy import ndimage
        # Each lattice reaches the union of its members' cells, not the
        # box around them, so a gap between tiles stays open.
        reach = [cells for _, cells in groups.values()]
        # A grid with its own adjacency (a hex mesh) grows by its neighbours.
        dilate = getattr(grid, "dilate", None)
        near = [dilate(m) if dilate is not None
                else ndimage.binary_dilation(m, np.ones((3, 3), bool))
                for m in reach]
    for a in range(len(mosaics)):
        for b in range(a):
            # Touching counts too: a seam between lattices would lose the
            # walls along it, so a raster may not reach a cell beside one
            # the other reaches.
            if np.any(near[a] & reach[b]):
                raise ValueError(
                    "building rasters on different pixel lattices overlap "
                    "or touch on the model grid; their ground would count "
                    "twice or the walls along their seam would be lost. "
                    "Resample them onto one lattice (same CRS, pixel size "
                    "and alignment) so they mosaic")
    for raster in mosaics:
        per_cell = grid.dx * grid.dy / (raster.px * raster.py)
        factor = max(1, int(math.ceil(math.sqrt(
            min_samples / max(per_cell, 1e-12)))))
        factors.append(factor)
        raster = raster.supersampled(factor)
        sampled = float(sums.area.sum())
        _accumulate(sums, raster, grid, edges, grid.axis_rotation(raster))
        # Only a raster that put ground into some cell is credited.
        if float(sums.area.sum()) > sampled:
            sources.update(raster.sources)

    def plane(values):
        return values.reshape(ny, nx)

    area = sums.area
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_area = np.where(area > 0, 1.0 / area, 0.0)
        inv_bldg = np.where(sums.bldg > 0, 1.0 / sums.bldg, 0.0)
        lambda_p = sums.bldg * inv_area
        mean_h = sums.h1 * inv_bldg
        var = np.maximum(sums.h2 * inv_bldg - mean_h ** 2, 0.0)
        hist = sums.bin_area * inv_area
        hbin = np.where(sums.bin_area > 0, sums.bin_h / np.where(
            sums.bin_area > 0, sums.bin_area, 1.0), 0.0)
        hi = sums.hi * inv_bldg
    lambda_b = lambda_p + sums.wall * inv_area
    f32 = np.float32
    return BuildingMorphology(
        sampled_area=plane(area).astype(f32),
        lambda_p=plane(lambda_p).astype(f32),
        mean_height=plane(mean_h).astype(f32),
        std_height=plane(np.sqrt(var)).astype(f32),
        lambda_b=plane(lambda_b).astype(f32),
        frontal_x=plane(sums.fx * inv_area).astype(f32),
        frontal_y=plane(sums.fy * inv_area).astype(f32),
        frontal_d45=plane(sums.f45 * inv_area).astype(f32),
        frontal_d135=plane(sums.f135 * inv_area).astype(f32),
        hi=hi.reshape(15, ny, nx).astype(f32),
        bldg_hist=hist.reshape(-1, ny, nx).astype(f32),
        bldg_hbin=hbin.reshape(-1, ny, nx).astype(f32),
        bin_edges=edges, sources=tuple(sorted(sources)),
        receipt=dict(supersample_factors=factors,
                     cells_sampled=int((area > 0).sum()),
                     cells_with_buildings=int((sums.bldg > 0).sum())))


def urb_param(morph: BuildingMorphology) -> np.ndarray:
    """The (132, ny, nx) NUDAPT URB_PARAM block WRF 4.7.1's real init reads.

    Only the indices WRF reads are filled; the rest are zero.  MH_URB2D and
    HGT_URB2D are both the plan-area-weighted mean here (a raster has no
    building identity; NUDAPT's 92 is a per-building mean).  HI_URB2D is
    the plan-area fraction of building pixels in BEP's 5 m classes,
    normalised over buildings (BEP renormalises it anyway).
    """
    ny, nx = morph.shape
    out = np.zeros((URB_PARAM_SIZE, ny, nx), np.float32)
    put = lambda index, value: out.__setitem__(index - 1, value)
    put(URB_PARAM_INDEX["LP_URB2D"], morph.lambda_p)
    put(URB_PARAM_INDEX["MH_URB2D"], morph.mean_height)
    put(URB_PARAM_INDEX["STDH_URB2D"], morph.std_height)
    put(URB_PARAM_INDEX["HGT_URB2D"], morph.mean_height)
    put(URB_PARAM_INDEX["LB_URB2D"], morph.lambda_b)
    lf = morph.lambda_f_sectors()
    base = URB_PARAM_INDEX["LF_URB2D"] - 1
    out[base:base + 4] = lf
    base = URB_PARAM_INDEX["HI_URB2D"] - 1
    out[base:base + 15] = morph.hi
    return out


# --------------------------------------------------------------------------
# The product file
# --------------------------------------------------------------------------

_PLANES = ("sampled_area", "lambda_p", "mean_height", "std_height",
           "lambda_b", "frontal_x", "frontal_y", "frontal_d45",
           "frontal_d135")


def write_product(path, morph: BuildingMorphology, *, grid_note: str = "",
                  extra_receipt: dict | None = None) -> Path:
    """Write the ``woof-buildings.v1`` NetCDF (model grid, compressed)."""
    import netCDF4

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = morph.shape
    receipt = dict(morph.receipt, **(extra_receipt or {}))
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createDimension("bin", morph.bldg_hist.shape[0])
        ds.createDimension("edge", morph.bin_edges.size)
        ds.createDimension("hi", 15)
        ds.createDimension("urb_param", URB_PARAM_SIZE)
        for name in _PLANES:
            var = ds.createVariable(name, "f4", ("y", "x"), zlib=True)
            var[:] = getattr(morph, name)
        for name in ("bldg_hist", "bldg_hbin"):
            var = ds.createVariable(name, "f4", ("bin", "y", "x"), zlib=True)
            var[:] = getattr(morph, name)
        ds.createVariable("bin_edges", "f8", ("edge",))[:] = morph.bin_edges
        ds.createVariable("hi", "f4", ("hi", "y", "x"), zlib=True)[:] = \
            morph.hi
        ds.createVariable("URB_PARAM", "f4", ("urb_param", "y", "x"),
                          zlib=True)[:] = urb_param(morph)
        ds.schema = SCHEMA
        ds.grid = grid_note
        ds.sources = json.dumps(list(morph.sources))
        ds.attribution = " | ".join(
            BUILDING_SOURCES[s]["attribution"] for s in morph.sources)
        ds.licenses = json.dumps({s: BUILDING_SOURCES[s]["license_id"]
                                  for s in morph.sources})
        ds.receipt = json.dumps(receipt, sort_keys=True, default=str)
    return path


def read_product(path) -> BuildingMorphology:
    """Read a ``woof-buildings.v1`` file back (refuses another schema)."""
    import netCDF4

    with netCDF4.Dataset(Path(path)) as ds:
        if getattr(ds, "schema", None) != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        get = lambda name: np.asarray(ds.variables[name][:])
        return BuildingMorphology(
            **{name: get(name).astype(np.float32) for name in _PLANES},
            hi=get("hi").astype(np.float32),
            bldg_hist=get("bldg_hist").astype(np.float32),
            bldg_hbin=get("bldg_hbin").astype(np.float32),
            bin_edges=get("bin_edges").astype(np.float64),
            sources=tuple(json.loads(ds.sources)),
            receipt=json.loads(ds.receipt))


# --------------------------------------------------------------------------
# Idealised obstacles and GEM-15 windows
# --------------------------------------------------------------------------

def block_raster(nx: int, ny: int, dx: float, dy: float, blocks, *,
                 resolution: float | None = None) -> HeightRaster:
    """Rectangular buildings on a model-plane raster (idealised cases).

    ``blocks`` holds ``(x0, x1, y0, y1, height)`` in metres from the grid's
    south-west corner.  ``resolution`` defaults to a tenth of the smaller
    grid spacing.
    """
    res = float(resolution or min(dx, dy) / 10.0)
    cols = int(round(nx * dx / res))
    rows = int(round(ny * dy / res))
    h = np.zeros((rows, cols), np.float32)
    xc = (np.arange(cols) + 0.5) * res
    yc = ny * dy - (np.arange(rows) + 0.5) * res
    for x0, x1, y0, y1, height in blocks:
        sx = (xc >= x0) & (xc < x1)
        sy = (yc >= y0) & (yc < y1)
        h[np.ix_(sy, sx)] = np.maximum(h[np.ix_(sy, sx)], float(height))
    return HeightRaster(h, 0.0, ny * dy, res, res, None)


def gem15_building_rasters(dsm_windows, dtm_windows, **filters):
    """Pair GEM-15 ``dsm``/``dtm`` UTM windows (paths of the GeoTIFFs
    :func:`woof.static.gem15.fetch_gem15_windows` writes, same cell and
    block range) and filter each pair with :func:`ndsm_heights`.

    Returns ``(rasters, receipts)``; a pair whose grids disagree refuses.
    """
    import rasterio

    def opened(path):
        with rasterio.open(path) as src:
            t = src.transform
            if t.b != 0 or t.d != 0:
                raise ValueError(f"{path}: rotated rasters are not supported")
            return (src.read(1).astype(np.float64), float(t.c), float(t.f),
                    float(t.a), float(-t.e), src.crs.to_string())

    if len(dsm_windows) != len(dtm_windows):
        raise ValueError("GEM-15 dsm and dtm window counts differ")
    rasters, receipts = [], []
    for dsm_path, dtm_path in zip(dsm_windows, dtm_windows):
        dsm, x0, y0, px, py, crs = opened(dsm_path)
        dtm, *frame = opened(dtm_path)
        if (x0, y0, px, py, crs) != tuple(frame) or dsm.shape != dtm.shape:
            raise ValueError(f"{Path(dsm_path).name} and "
                             f"{Path(dtm_path).name} are not one grid")
        height, receipt = ndsm_heights(dsm, dtm, pixel_area_m2=px * py,
                                       **filters)
        rasters.append(HeightRaster(height, x0, y0, px, py, crs,
                                    ("gem15-ndsm",)))
        receipts.append(dict(receipt, window=Path(dsm_path).name, crs=crs))
    return rasters, receipts


__all__ = [
    "BUILDING_SOURCES", "BuildingMorphology", "CartesianGrid", "Footprint",
    "HI_CLASS_HEIGHTS", "HeightRaster", "LES_BIN_EDGES", "ProjectionGrid",
    "SCHEMA", "URB_PARAM_INDEX", "URB_PARAM_SIZE", "aggregate",
    "block_raster", "gem15_building_rasters", "ndsm_heights", "parse_wkb",
    "rasterize_footprints", "read_footprints", "read_product", "urb_param",
    "utm_raster", "VoronoiGrid", "write_product",
]
