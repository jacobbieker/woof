"""aer_opt = 2 (user aerosol optics, auxinput15) on the hex mesh.

WRF reads AOD5502D / ANGEXP2D / AERSSA2D / AERASY2D from the auxinput15 file on
its own grid.  On hex the radiation columns are Voronoi cells, so each record
the run's radiation calls read (the same file, record and interval rule as WRF:
:func:`woof.core.aer_opt2.auxinput15_offset`) is remapped to the cells by the
first-order conservative operator of :mod:`woof.hex.moving_remap` (the
hydro exchange's, grid donors onto mesh cells), ``q_c = sum_g W[c, g] q_g / A_c``, and sealed into the constructor.  The
optics themselves are :mod:`woof.core.aer_opt2`, the one kernel WRF runs.

The WRF grid's cell corners are the file's ``XLAT_C``/``XLONG_C`` when it has
them, else they are derived from the cell centres (``XLAT``/``XLONG``): the
mean of the four surrounding centres inside, linear extrapolation at the edge.
Every mesh cell must lie under the grid (the operator's ``target_subdomain``
coverage test): an aerosol value is never invented for a cell the file does not
cover, so a mesh larger than the grid is refused.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .errors import ConfigurationRefusal

def run_offsets(cfg, run_seconds: float) -> list[float]:
    """The auxinput15 offsets the run's radiation calls read, from 0 to the end."""
    from woof.core.aer_opt2 import auxinput15_offset

    interval = int(cfg.aer_auxinput15_interval)
    if interval <= 0:
        raise ConfigurationRefusal(
            "aer_auxinput15_interval", interval,
            "hex stages the auxinput15 records the run reads before it starts, so it "
            "needs the file interval; 0 reads a record at every radiation time",
            "a positive aer_auxinput15_interval (seconds)")
    last = auxinput15_offset(cfg, max(float(run_seconds) - 1.0e-3, 0.0))
    return [float(k * interval) for k in range(int(last // interval) + 1)]


def grid_corners(path):
    """``(lat_c, lon_c)`` (ny+1, nx+1) degrees of the WRF grid in ``path``."""
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if "XLAT_C" in ds.variables and "XLONG_C" in ds.variables:
            lat = np.asarray(ds.variables["XLAT_C"][:], np.float64)
            lon = np.asarray(ds.variables["XLONG_C"][:], np.float64)
            return lat.reshape(lat.shape[-2:]), lon.reshape(lon.shape[-2:])
        for name in ("XLAT", "XLONG"):
            if name not in ds.variables:
                raise ConfigurationRefusal(
                    "aer_auxinput15", str(path),
                    f"the file carries neither XLAT_C/XLONG_C nor {name}, so the WRF grid's "
                    "cells cannot be placed on the sphere", "an auxinput15 file with its grid")
        lat = np.asarray(ds.variables["XLAT"][:], np.float64)
        lon = np.asarray(ds.variables["XLONG"][:], np.float64)
    lat, lon = lat.reshape(lat.shape[-2:]), lon.reshape(lon.shape[-2:])
    from .moving_remap import sphere_xyz

    xyz = sphere_xyz(np.radians(lat), np.radians(lon))
    ny, nx = lat.shape
    padded = np.empty((ny + 2, nx + 2, 3))
    padded[1:-1, 1:-1] = xyz
    padded[0, 1:-1] = 2 * xyz[0] - xyz[1]
    padded[-1, 1:-1] = 2 * xyz[-1] - xyz[-2]
    padded[:, 0] = 2 * padded[:, 1] - padded[:, 2]
    padded[:, -1] = 2 * padded[:, -2] - padded[:, -3]
    corner = (padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]) / 4.0
    corner /= np.linalg.norm(corner, axis=-1, keepdims=True)
    return (np.degrees(np.arcsin(np.clip(corner[..., 2], -1, 1))),
            np.degrees(np.arctan2(corner[..., 1], corner[..., 0])))


def stage_user_aerosol(cfg, start_time, *, mesh, run_seconds: float, cells=None):
    """``(frames, offsets)``: the run's auxinput15 records on the mesh cells.

    ``frames`` is float32 ``[records, fields, nCells]``, ``offsets`` float64.
    """
    from woof.core.aer_opt2 import (auxinput15_name, read_auxinput15,
                                    required_fields)
    from .hydro_remap import grid_cells, mesh_cells
    from .moving_remap import ConservativeMap

    fields = required_fields(cfg)
    offsets = run_offsets(cfg, run_seconds)
    cells = mesh_cells(mesh) if cells is None else cells
    exchange = None
    shape = None
    frames = []
    for offset in offsets:
        name, valid = auxinput15_name(cfg, start_time, offset)
        record = read_auxinput15(cfg, name, valid)
        if exchange is None:
            shape = np.asarray(record[fields[0]]).shape
            lat_c, lon_c = grid_corners(name)
            if lat_c.shape != (shape[0] + 1, shape[1] + 1):
                raise ConfigurationRefusal(
                    "aer_auxinput15", name,
                    f"the grid's corners {lat_c.shape} do not enclose the fields' {shape}",
                    "an auxinput15 file whose grid matches its fields")
            try:
                weights = ConservativeMap.build(
                    grid_cells(lat_c, lon_c), cells, target_subdomain=True).overlap.tocsr()
            except ValueError as error:
                raise ConfigurationRefusal(
                    "aer_auxinput15", name,
                    f"the auxinput15 grid does not cover the mesh ({error}); an aerosol value "
                    "is never invented for a cell the file does not cover",
                    "an auxinput15 grid that covers the whole mesh") from None
            exchange = (weights, np.asarray(weights.sum(axis=1)).ravel())
        planes = []
        for key in fields:
            value = np.ma.filled(np.ma.asarray(record[key], dtype=np.float64), np.nan)
            if value.shape != shape:
                raise ValueError(f"auxinput15 {key} shape {value.shape} differs from {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"auxinput15 {key} carries non-finite values")
            planes.append(((exchange[0] @ value.reshape(-1)) / exchange[1]).astype(np.float32))
        frames.append(np.stack(planes))
    return np.stack(frames).astype(np.float32), np.asarray(offsets, np.float64)
