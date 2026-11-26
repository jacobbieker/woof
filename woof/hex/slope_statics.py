"""slope_rad's SLOPE and SLP_AZI on a hex mesh, from each cell's own neighbours.

WRF builds them once per domain from HGT on its structured grid (``dyn_em/start_em.F:1539-1577``,
:func:`woof.core.topo_radiation.slope_geometry`).  The stencil there is a centred difference across
the two neighbours along each grid axis (one-sided at the domain edge), scaled to true distance by the
map factor:

    hx = (HT(i+1,j) - HT(i-1,j)) * msftx / dx / 2      hy likewise along j
    SLOPE = atan(sqrt(hx**2 + hy**2))
    SLP_AZI = atan2(hx, hy) + pi, rotated from grid to geographic north (sina, cosa)
    SLOPE < 1e-4 -> SLOPE = SLP_AZI = 0

The hex columns ride a ``(1, nCells)`` layout whose index neighbours are not the mesh's, so that
stencil would difference unrelated cells (the gpuwm-hex 0.6.1 breakage the refusal names).  Here
the terrain gradient of a cell is the least-squares plane through its ``cellsOnCell`` neighbours,
in the cell's local tangent plane:

* each neighbour's displacement is its unit position vector minus the cell's, projected on the cell's
  local east and north unit vectors and scaled so its length is the great-circle distance
  (``sphere_radius`` times the central angle);
* ``(hx, hy)`` minimise ``sum_j (h_j - h_i - hx*de_j - hy*dn_j)**2`` over the neighbours present (a
  limited-area mesh's boundary cell has fewer; one with fewer than two independent directions gets
  zero gradient);
* SLOPE, SLP_AZI then follow WRF's formulas above, already in geographic east/north, so no
  rotation is needed; the ``1e-4`` flat threshold is WRF's.

On a regular hexagonal mesh the fit equals the mean of the three centred differences across opposite
neighbours, and it reproduces any planar terrain exactly on any mesh.  It differs from WRF's stencil
in three ways, all of geometry rather than of the slope physics:

* six (or five, seven) neighbours at 60-degree spacing instead of four on two axes, so on curved
  terrain it is a different, more isotropic, discrete gradient of the same surface;
* the spacing is each neighbour's own distance (variable resolution included), not one ``dx``
  with a map factor;
* at a limited-area boundary it fits the neighbours present instead of WRF's one-sided difference.

Computed in float64 on the host once per run and handed to the engine as float32 words (WRF's
REAL), as ``topo_slope`` / ``topo_slope_azimuth``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: WRF's flat threshold (start_em.F: ``IF (SLOPE < 1.E-4) SLOPE=0, SLP_AZI=0``).
FLAT_SLOPE = 1.0e-4


def _unit(lat, lon):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def slope_aspect(lat_rad, lon_rad, cells_on_cell, n_edges_on_cell, height_m, *, radius_m: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``(slope, azimuth)`` in radians (float32, one per cell); see the module docstring.

    ``cells_on_cell`` is zero-based ``[nCells, maxEdges]``; an entry outside ``[0, nCells)`` (MPAS's
    nCells+1 "outside" index on a limited-area mesh, or a negative sentinel) is a missing neighbour.
    """
    lat = np.asarray(lat_rad, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon_rad, dtype=np.float64).reshape(-1)
    h = np.asarray(height_m, dtype=np.float64).reshape(-1)
    n = lat.size
    cells = np.asarray(cells_on_cell, dtype=np.int64)
    counts = np.asarray(n_edges_on_cell, dtype=np.int64).reshape(-1)
    if cells.ndim != 2 or cells.shape[0] != n or counts.shape != (n,) or h.shape != (n,):
        raise ValueError(f"mesh arrays do not agree: {n} cells, cellsOnCell {cells.shape}, "
                         f"nEdgesOnCell {counts.shape}, height {h.shape}")
    if not (np.isfinite(radius_m) and radius_m > 0):
        raise ValueError(f"sphere radius {radius_m!r} must be finite and positive")
    if not np.all(np.isfinite(h)):
        raise ValueError("terrain height must be finite")
    slot = np.arange(cells.shape[1])[None, :]
    present = (slot < counts[:, None]) & (cells >= 0) & (cells < n)
    neighbour = np.where(present, cells, np.arange(n)[:, None])
    p = _unit(lat, lon)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    d = p[neighbour] - p[:, None, :]
    de = np.einsum("ijk,ik->ij", d, east)
    dn = np.einsum("ijk,ik->ij", d, north)
    chord_plane = np.hypot(de, dn)
    angle = 2.0 * np.arcsin(np.clip(np.linalg.norm(d, axis=-1) / 2.0, 0.0, 1.0))
    scale = np.where(chord_plane > 0, radius_m * angle / np.where(chord_plane > 0, chord_plane, 1.0), 0.0)
    de = np.where(present, de * scale, 0.0)
    dn = np.where(present, dn * scale, 0.0)
    dh = np.where(present, h[neighbour] - h[:, None], 0.0)
    a11 = np.sum(de * de, axis=1)
    a12 = np.sum(de * dn, axis=1)
    a22 = np.sum(dn * dn, axis=1)
    b1 = np.sum(de * dh, axis=1)
    b2 = np.sum(dn * dh, axis=1)
    det = a11 * a22 - a12 * a12
    # Fewer than two independent directions: no plane, no gradient.
    solvable = det > 1e-12 * np.maximum(a11 * a22, np.finfo(np.float64).tiny)
    safe = np.where(solvable, det, 1.0)
    hx = np.where(solvable, (a22 * b1 - a12 * b2) / safe, 0.0)
    hy = np.where(solvable, (a11 * b2 - a12 * b1) / safe, 0.0)
    slope = np.arctan(np.hypot(hx, hy))
    azimuth = np.arctan2(hx, hy) + np.pi
    flat = slope < FLAT_SLOPE
    slope = np.where(flat, 0.0, slope).astype(np.float32)
    azimuth = np.where(flat, 0.0, azimuth).astype(np.float32)
    return slope, azimuth


def mesh_slope_aspect(mesh: Any, height_m) -> tuple[np.ndarray, np.ndarray]:
    """:func:`slope_aspect` on a hex mesh object (``latCell``, ``lonCell``, ``cellsOnCell``,
    ``nEdgesOnCell``, ``sphere_radius``).  The loaded mesh's ``cellsOnCell`` is zero-based, as the
    topo_wind laplacian reads it (run_cuda_v841_forecast._topo_wind_cell_neighbours)."""

    from woof.hex.cuda_physics_v841 import _mesh_value

    on_sphere = _mesh_value(mesh, "on_a_sphere", "YES")
    if str(on_sphere).strip().upper() not in ("YES", "TRUE", "1"):
        raise ValueError("slope_rad's mesh slope is built on a spherical mesh; this mesh is planar "
                         "(on_a_sphere = NO), whose periodic neighbours it does not unwrap")
    lat = np.asarray(_mesh_value(mesh, "latCell"), dtype=np.float64).reshape(-1)
    lon = np.asarray(_mesh_value(mesh, "lonCell"), dtype=np.float64).reshape(-1)
    counts = np.asarray(_mesh_value(mesh, "nEdgesOnCell"), dtype=np.int64).reshape(-1)
    cells = np.asarray(_mesh_value(mesh, "cellsOnCell"), dtype=np.int64)
    if cells.ndim == 2 and cells.shape[0] != counts.size and cells.shape[1] == counts.size:
        cells = cells.T
    if cells.ndim != 2 or cells.shape[0] != counts.size:
        raise ValueError(f"cellsOnCell {cells.shape} does not fit nEdgesOnCell {counts.shape}")
    radius = float(_mesh_value(mesh, "sphere_radius", 6371229.0))
    height = np.asarray(height_m, dtype=np.float64).reshape(-1)[: lat.size]
    return slope_aspect(lat, lon, cells, counts, height, radius_m=radius)


def derive(cfg, context: Mapping[str, Any], which: int) -> np.ndarray:
    """The surface-input contract's derivation (woof.hex.surface_inputs): ``context["slope_mesh"]``
    is ``{"mesh": ..., "terrain_height_m": ...}`` of the run's own cells."""

    del cfg
    staged = context.get("slope_mesh")
    if not staged:
        raise ValueError("slope_rad's SLOPE/SLP_AZI are built from the run's mesh and terrain, "
                         "and bind was given neither")
    key = ("mesh", id(staged["mesh"]))
    cache = staged.setdefault("_computed", {})
    if key not in cache:
        cache[key] = mesh_slope_aspect(staged["mesh"], staged["terrain_height_m"])
    return cache[key][which]
