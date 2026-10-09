"""Geodesic helpers for grid assets (unit 3, stub).

Small-vector geometry on the WRF sphere
(:data:`woof.static.projection.EARTH_RADIUS_M`): local projections,
polyline densification and bearings, corridor buffers, point-in-polygon and
rectangle covers.  Coordinates are ``(lon, lat)`` degrees in and out unless
a function says it works in projected metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from woof.energy.contracts import EnergyNotImplemented

LonLat = tuple[float, float]
#: A polygon: first ring is the outer boundary, any further rings are holes.
#: Rings are closed lists of (lon, lat).
Polygon = list[list[LonLat]]


@dataclass(frozen=True)
class LocalProjection:
    """Azimuthal-equidistant projection about ``(lat0, lon0)``, metres."""

    lat0: float
    lon0: float

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) degrees -> (x east, y north) metres."""
        raise EnergyNotImplemented("woof.energy.geometry.LocalProjection")

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) metres -> (lon, lat) degrees."""
        raise EnergyNotImplemented("woof.energy.geometry.LocalProjection")


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


def local_projection(lat0: float, lon0: float) -> LocalProjection:
    raise EnergyNotImplemented("woof.energy.geometry.local_projection")


def densify_polyline(coords: Sequence[LonLat], spacing_m: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ``spacing_m`` along a polyline, both ends included.

    Returns ``(lon, lat, chainage_m)``; the last step may be shorter.
    """

    raise EnergyNotImplemented("woof.energy.geometry.densify_polyline")


def segment_bearings(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Azimuth of the polyline at each point, degrees clockwise from north
    in [0, 360); centred differences inside, one-sided at the ends."""

    raise EnergyNotImplemented("woof.energy.geometry.segment_bearings")


def buffer_polyline(coords: Sequence[LonLat], half_width_m: float, *,
                    segments_per_quarter: int = 4) -> Polygon:
    """Corridor polygon within ``half_width_m`` of the polyline."""

    raise EnergyNotImplemented("woof.energy.geometry.buffer_polyline")


def point_in_polygon(lon: np.ndarray, lat: np.ndarray,
                     polygons: Sequence[Polygon]) -> np.ndarray:
    """Boolean mask of points inside any polygon (holes excluded)."""

    raise EnergyNotImplemented("woof.energy.geometry.point_in_polygon")


def cover_with_rectangles(x: np.ndarray, y: np.ndarray, *, margin_m: float,
                          dx_m: float, max_nx: int, max_ny: int,
                          align_m: float | None = None) -> list[Rect]:
    """Greedy cover of projected points by rectangles.

    Every point lies at least ``margin_m`` inside some rectangle; no
    rectangle exceeds ``max_nx`` x ``max_ny`` mass points at ``dx_m``;
    corners snap to multiples of ``align_m`` (a parent cell) when given.
    Each point is a member of exactly one rectangle.
    """

    raise EnergyNotImplemented("woof.energy.geometry.cover_with_rectangles")
