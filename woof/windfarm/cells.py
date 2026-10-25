"""Turbines on the columns of an unstructured mesh (hex's Voronoi cells).

WRF places a turbine in a projected grid's cell with its own ``map_set``
and ``latlon_to_ij`` (:mod:`woof.windfarm.wrf_llij`).  An MPAS mesh has no
projection: a point belongs to the cell whose generator (cell centre) is
nearest on the sphere, which is what a Voronoi cell IS and the rule MPAS
itself applies to a point location (MPAS's ``nearest_cell`` walks
``cellsOnCell`` to the cell centre at the least great-circle distance, as
init_atmosphere does for its point locations; the static builder sends
each terrain pixel to its nearest cell the same way).  This module is that
rule and nothing else:

* :class:`CellPlacement`, a mesh's columns as the physics sees them: the
  cell centres' latitude and longitude (degrees), each column's horizontal
  area (m2, MPAS ``areaCell``) and which columns are modelled interior
  (``False`` for a limited-area mesh's specified and relaxation zones,
  ``bdyMaskCell > 0``, and for the padded garbage column);
* :func:`nearest_cells`, the placement: unit vectors in binary64, the
  largest dot product with the turbine's unit vector (the smallest
  great-circle distance), ties to the lowest column index.  MPAS walks
  from a starting cell toward the point and so returns one of a tie's
  cells by its path; the lowest index makes the choice independent of any
  path and is documented here as the one difference.

A turbine whose nearest column is not interior is refused by name (the
boundary columns are forced, not modelled: the rule
:func:`woof.windfarm.layout.build_layout` applies to a WRF domain's
outermost row and column), and so is a turbine farther from its nearest
centre than twice that cell's equal-area radius ``sqrt(area/pi)``, which on
a limited-area mesh means outside the mesh altogether.

The turbines' curves, hub heights, rotors and grouping are
:mod:`woof.windfarm.layout`'s own (:func:`woof.windfarm.layout.build_cell_layout`):
only where a turbine stands is decided here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How far from its nearest cell centre a turbine may stand, in units of
#: that cell's equal-area radius sqrt(area/pi).  A Voronoi cell's farthest
#: corner from its centre is about 1.1 radii on a quasi-uniform mesh and
#: under 2 on any mesh MPAS builds; farther is outside the mesh.
OUTSIDE_RADII = 2.0
#: MPAS's sphere radius (mpas_constants ``a``), m: the great-circle
#: distance a refusal reports.
SPHERE_RADIUS_M = 6371229.0


class CellPlacementRefused(ValueError):
    """Turbines cannot be placed on this mesh's columns as given."""


def _unit_vectors(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64).reshape(-1))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64).reshape(-1))
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon),
                     np.sin(lat)], axis=1)


@dataclass(frozen=True, eq=False)
class CellPlacement:
    """A mesh's physics columns: where they are, how big, which are modelled.

    ``latitude_deg``/``longitude_deg``/``area_m2``/``interior`` have one
    value per column, in the physics column order (hex's cell order, plus a
    limited-area mesh's padded garbage column, which is never interior).
    ``nominal_dx_m`` is the mesh's nominal spacing, recorded in a layout's
    summary and never used to size a cell.
    """

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    area_m2: np.ndarray
    interior: np.ndarray
    nominal_dx_m: float = 0.0
    _xyz: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        put = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        lat = np.array(self.latitude_deg, dtype=np.float64).reshape(-1)
        lon = np.array(self.longitude_deg, dtype=np.float64).reshape(-1)
        area = np.array(self.area_m2, dtype=np.float64).reshape(-1)
        interior = np.array(self.interior, dtype=bool).reshape(-1)
        n = lat.size
        if n < 1 or any(a.size != n for a in (lon, area, interior)):
            raise ValueError(
                "latitude_deg, longitude_deg, area_m2 and interior must have "
                f"one value per column (got {lat.size}, {lon.size}, "
                f"{area.size}, {interior.size})")
        if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
            raise ValueError("cell centres must be finite")
        if not (np.isfinite(area).all() and (area > 0).all()):
            raise ValueError("cell areas must be finite and positive")
        for name, value in (("latitude_deg", lat), ("longitude_deg", lon),
                            ("area_m2", area), ("interior", interior)):
            value.setflags(write=False)
            put(name, value)
        put("nominal_dx_m", float(self.nominal_dx_m))
        xyz = _unit_vectors(lat, lon)
        xyz.setflags(write=False)
        put("_xyz", xyz)

    @property
    def n_columns(self) -> int:
        return int(self.latitude_deg.size)

    @property
    def shape(self) -> tuple[int, int]:
        """The physics column grid, ``(1, n_columns)``."""
        return (1, self.n_columns)


def _tree(placement: CellPlacement):
    """A k-d tree over the unit cell centres, built once per placement."""
    tree = placement.__dict__.get("_kdtree")
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(placement._xyz)
        object.__setattr__(placement, "_kdtree", tree)
    return tree


def nearest_cells(placement: CellPlacement, lat_deg, lon_deg) -> tuple[
        np.ndarray, np.ndarray]:
    """``(column, distance_m)`` of each point: its nearest cell centre on
    the sphere (ties to the lowest column) and the great-circle distance to
    it.  Every column is a candidate; interiority is the caller's rule.

    The chord between unit vectors grows with the great-circle distance, so
    the k-d tree's few nearest by chord hold the nearest on the sphere; among
    them the largest binary64 dot product wins, the lowest column on a tie.
    """
    points = _unit_vectors(lat_deg, lon_deg)
    xyz = placement._xyz
    k = min(8, xyz.shape[0])
    _, candidates = _tree(placement).query(points, k=k)
    candidates = np.asarray(candidates).reshape(points.shape[0], k)
    columns = np.empty(points.shape[0], dtype=np.int64)
    distance = np.empty(points.shape[0], dtype=np.float64)
    for n, p in enumerate(points):
        near = np.sort(candidates[n])
        dots = xyz[near] @ p
        best = int(near[int(np.argmax(dots))])   # first maximum: lowest index
        columns[n] = best
        distance[n] = SPHERE_RADIUS_M * float(
            np.arccos(np.clip(xyz[best] @ p, -1.0, 1.0)))
    return columns, distance


def place_on_cells(placement: CellPlacement, lat_deg, lon_deg, names, *,
                   what: str = "turbines") -> np.ndarray:
    """The column of each point, refused by name when it is not modelled.

    ``names`` labels each point in a refusal (turbine ids or list rows).
    """
    from woof.explain import layered

    columns, distance = nearest_cells(placement, lat_deg, lon_deg)
    radius = np.sqrt(placement.area_m2[columns] / np.pi)
    names = [str(name) for name in names]
    outside = [f"{names[n]} ({distance[n] / 1000.0:.3g} km from the nearest "
               f"cell centre, column {int(columns[n])})"
               for n in range(columns.size)
               if distance[n] > OUTSIDE_RADII * radius[n]]
    if outside:
        raise CellPlacementRefused(layered(
            f"{what} outside this mesh: {', '.join(outside[:12])}"
            + (f" and {len(outside) - 12} more" if len(outside) > 12 else "")
            + ".\n  remedy: drop them from the table, or run a mesh that "
              "covers them",
            f"A point farther than {OUTSIDE_RADII:g} equal-area radii "
            "(sqrt(areaCell/pi)) from its nearest cell centre is in no cell "
            "of the mesh; placing it in the boundary cell nearest to it "
            "would move it silently."))
    boundary = [f"{names[n]} (column {int(columns[n])})"
                for n in range(columns.size)
                if not placement.interior[columns[n]]]
    if boundary:
        raise CellPlacementRefused(layered(
            f"{what} in this mesh's boundary columns: "
            f"{', '.join(boundary[:12])}"
            + (f" and {len(boundary) - 12} more" if len(boundary) > 12 else "")
            + ".\n  remedy: use a mesh whose interior holds them",
            "A limited-area mesh's specified and relaxation cells "
            "(bdyMaskCell > 0) are forced by the lateral boundary, not "
            "modelled; WRF's grid refuses its outermost row and column on "
            "the same rule."))
    return columns


__all__ = ["CellPlacement", "CellPlacementRefused", "OUTSIDE_RADII",
           "nearest_cells", "place_on_cells"]
