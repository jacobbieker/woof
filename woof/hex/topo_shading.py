"""topo_shading's terrain shadow search on a hex mesh: the statics it marches over.

WRF's ``toposhad`` (``phys/module_radiation_driver.F:4610-4862``, transcribed in
``woof/core/kernels/topo_radiation.cu`` as ``topo_shadow_scan``) walks from each mass point toward the
sun one grid row at a time, up to ``shadlen`` metres, interpolating HT linearly between the two grid
points the ray crosses.  At each step it asks whether the terrain there rises above the sun:

    topoelev = atan((HT_ray - HT) / dxabs);   shadowed where sin(topoelev) >= cos(zenith)

and raises HT_SHAD to the height the sun would have to clear.  A hex column has no grid rows, so the
march is the mesh's own (``topo_shadow_scan_mesh``, :func:`woof.core.topo_radiation.mesh_terrain_shadow`):

* the sun (zenith cosine at the radiation interval's midpoint, azimuth) is WRF's, statement for
  statement, at the cell's own latitude, longitude and geographic north;
* the k-th sample lies ``k * dxc`` metres from the cell along the sun azimuth, for WRF's
  ``gpshad = INT(shadlen/dxc + 1)`` samples, where ``dxc`` is the cell's mean distance to its neighbours
  (WRF's ``dx``; variable resolution included);
* HT at a sample is the barycentric (linear) interpolant on the Delaunay triangle that holds it: the
  sample's nearest cell and two consecutive ``cellsOnCell`` neighbours, the mesh's counterpart of WRF's
  two-point interpolation across a grid row;
* ``dxabs`` is the sample's horizontal distance from the cell, and the test and HT_SHAD update are WRF's;
* a sample whose nearest cell touches a limited-area mesh's outer edge ends the scan, as leaving the
  domain ends WRF's.

On planar terrain both marches read the plane exactly, so the mask is WRF's whenever the sun azimuth and
the plane do not put a sample within rounding of the threshold
(tests/test_hex_topo_shading.py).  On curved terrain the samples sit at different points of the same
surface (WRF's at grid rows, ``dx / cos`` apart along the ray; these at ``dxc``), so the two masks are
the same physics on two discretisations, not the same words.

This module builds, once per run on the host in float64, what the kernel reads per cell:

* ``topo_shadow_cells`` ``int32 [1, max_edges, nCells]``: zero-based ``cellsOnCell`` in MPAS's
  counter-clockwise order, ``-1`` past ``nEdgesOnCell``, ``-2`` for a neighbour outside a limited-area mesh;
* ``topo_shadow_geometry`` ``float32 [1, 7, nCells]``: the cell centre in metres on the sphere as
  float32 high and low words (x, y, z high, then low; their sum is the float64 position, so a difference
  of nearby centres keeps millimetres in single precision, which Metal requires), then ``dxc``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: The kernel's code for a neighbour slot past ``nEdgesOnCell``.
UNUSED = -1
#: The kernel's code for a neighbour outside a limited-area mesh.
OUTSIDE = -2


def shadow_statics(lat_rad, lon_rad, cells_on_cell, n_edges_on_cell, *, radius_m: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``(topo_shadow_cells, topo_shadow_geometry)``; see the module docstring.

    ``cells_on_cell`` is zero-based ``[nCells, maxEdges]``; an entry outside ``[0, nCells)`` within
    ``nEdgesOnCell`` is a neighbour outside the mesh.
    """
    lat = np.asarray(lat_rad, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon_rad, dtype=np.float64).reshape(-1)
    n = lat.size
    cells = np.asarray(cells_on_cell, dtype=np.int64)
    counts = np.asarray(n_edges_on_cell, dtype=np.int64).reshape(-1)
    if cells.ndim != 2 or cells.shape[0] != n or counts.shape != (n,) or lon.shape != (n,):
        raise ValueError(f"mesh arrays do not agree: {n} cells, cellsOnCell {cells.shape}, "
                         f"nEdgesOnCell {counts.shape}")
    if not (np.isfinite(radius_m) and radius_m > 0):
        raise ValueError(f"sphere radius {radius_m!r} must be finite and positive")
    if np.any(counts < 0) or np.any(counts > cells.shape[1]):
        raise ValueError("nEdgesOnCell must lie in [0, maxEdges]")
    slots = np.arange(cells.shape[1])[None, :]
    used = slots < counts[:, None]
    inside = (cells >= 0) & (cells < n)
    table = np.where(used, np.where(inside, cells, OUTSIDE), UNUSED).astype(np.int32)
    position = radius_m * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                                    np.sin(lat)])
    high = position.astype(np.float32)
    low = (position - high.astype(np.float64)).astype(np.float32)
    # dxc: the mean great-circle distance to the neighbours present.
    unit = position / radius_m
    total = np.zeros(n)
    present = np.zeros(n)
    for slot in range(cells.shape[1]):
        ok = used[:, slot] & inside[:, slot]
        other = np.where(ok, cells[:, slot], np.arange(n))
        cosine = np.clip(np.einsum("ki,ki->i", unit, unit[:, other]), -1.0, 1.0)
        total += np.where(ok, radius_m * np.arccos(cosine), 0.0)
        present += ok
    if np.any(present == 0):
        raise ValueError(f"{int(np.count_nonzero(present == 0))} cells have no neighbour on the mesh; "
                         "the shadow search has no spacing to march at")
    spacing = (total / present).astype(np.float32)
    geometry = np.concatenate([high, low, spacing[None, :]]).astype(np.float32)
    return table[None, :, :].transpose(0, 2, 1).copy(), geometry[None, :, :].copy()


def mesh_shadow_statics(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """:func:`shadow_statics` on a hex mesh object (``latCell``, ``lonCell``, ``cellsOnCell``,
    ``nEdgesOnCell``, ``sphere_radius``), read as woof.hex.slope_statics reads it."""

    from woof.hex.cuda_physics_v841 import _mesh_value

    on_sphere = _mesh_value(mesh, "on_a_sphere", "YES")
    if str(on_sphere).strip().upper() not in ("YES", "TRUE", "1"):
        raise ValueError("topo_shading's shadow search marches a spherical mesh; this mesh is planar "
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
    return shadow_statics(lat, lon, cells, counts, radius_m=radius)


def derive(cfg, context: Mapping[str, Any], which: int) -> np.ndarray:
    """The surface-input contract's derivation (woof.hex.surface_inputs), from the same
    ``context["slope_mesh"]`` slope_rad's statics read."""

    del cfg
    staged = context.get("slope_mesh") or {}
    if staged.get("mesh") is None:
        raise ValueError("topo_shading's shadow statics are built from the run's mesh, "
                         "and bind was given none")
    if "_shadow" not in staged:
        staged["_shadow"] = mesh_shadow_statics(staged["mesh"])
    return staged["_shadow"][which]
