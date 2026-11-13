"""``woof hex surface-statics`` -- the per-cell scheme inputs a selection reads.

The hex static builder half of the surface-input contract
(:mod:`woof.hex.surface_inputs`): it writes ``<mesh>.surface_statics.nc``, a
sidecar of the mesh's static file holding, per cell, the inputs WRF column
schemes read beyond the init's own fields.  The forecast binds the file for a
generic-seam selection that declares them.  The static itself is not touched
(the mesh-row registry pins its SHA-256).

FIELDS, and how WRF's geogrid/real.exe would build each one on these cells.

``LAKEMASK`` (CLM lake, ``sf_lake_physics = 1``).  real.exe marks a lake where
the dominant category of the with-lakes MODIS land use is ``ISLAKE`` (21) and
then turns the cell into water (``ISWATER``); its ``LAKEFLAG`` is 1.  On the
mesh the dominant category is the cell's own vote, so the mask is taken, in
order, from:

1. the static's ``tc_lakemask`` (``rw_mpas_static --preserve-lake-mask``: the
   Rust static builder's own dominant-category vote of the with-lakes land use,
   ``== islake``);
2. the static's ``ivgtyp == ISLAKE`` (a static voted from the with-lakes land
   use without folding lakes, which carries category 21 itself);
3. a vote of ``modis_landuse_20class_30s_with_lakes`` under ``--geog-root``: every
   30-arc-second pixel goes to the Voronoi cell whose centre is nearest
   (:class:`woof.hex.static_highres.VoronoiAssigner`, MPAS init's rule), and the
   cell's dominant category (lowest category on a tie, as geogrid's) decides.

A lake is a water cell (real.exe converts it to ``ISWATER``), so the mask is
intersected with the static's own water cells (``landmask == 0``); the count
that rule removed is recorded.

``LAKE_DEPTH`` (``use_lakedepth = 1``).  GEOGRID.TBL's row (WPS v4.6.0
GEOGRID.TBL.ARW:430-438, :data:`woof.static.lake.LAKE_DEPTH_ROW`):
``average_gcell(1.0)+search(5)``, ``masked = land``, ``fill_missing = 10``.  On
the mesh: the mean of the valid bathymetry pixels whose centres fall in the
cell (the same nearest-centre assignment); a water cell that holds none takes
geogrid's breadth-first search of up to 5 source pixels from the pixel under
the cell centre; a land cell, and a water cell the search does not reach, get
the fill.  ``LAKE_DEPTH_FLAG`` is 1.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import MpasPortError
from .surface_inputs import FILE_SUFFIX, SCHEMA, builder_default_path

#: WPS GEOGRID.TBL LAKE_DEPTH row (woof.static.lake.LAKE_DEPTH_ROW).
LAKE_DEPTH_FILL_M = 10.0
LAKE_DEPTH_SEARCH = 5
LAKE_DEPTH_DATASET = "lake_depth"
LAKE_LANDUSE_DATASET = "modis_landuse_20class_30s_with_lakes"
#: MODIS's lake class (WPS index ``islake``), when the static names none.
MODIS_ISLAKE = 21
#: Source pixels read per band: bounds the band's values, indices, centres and
#: assignment temporaries (a few hundred bytes a pixel) on a global source.
_BAND_PIXELS = 4_000_000


class SurfaceStaticsRefusal(MpasPortError):
    """A named refusal of the surface statics door."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_sha256(dataset_dir: Path) -> str:
    return sha256((dataset_dir / "index").read_bytes()).hexdigest()


def _footprint(geometry) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) in degrees, longitudes unwrapped
    about the mesh's first cell so a footprint across 180 stays contiguous."""
    xyz = np.concatenate([geometry.xyz, geometry.vertex_xyz])
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    centre = lon[0]
    lon = centre + (np.mod(lon - centre + 180.0, 360.0) - 180.0)
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def _source_window(dataset, footprint, pad: int):
    """The 1-based inclusive source index window over the footprint."""
    lat_min, lat_max, lon_min, lon_max = footprint
    if lon_max - lon_min >= 359.0 or lat_max - lat_min >= 179.0:
        x0, x1 = 1, int(dataset.nx_global)
    else:
        xa, _ = dataset.latlon_to_xy(0.0, lon_min)
        width = (lon_max - lon_min) / dataset.index.dx
        x0 = int(math.floor(float(xa))) - pad
        x1 = x0 + int(math.ceil(width)) + 2 * pad + 1
    _, ya = dataset.latlon_to_xy(lat_min, 0.0)
    _, yb = dataset.latlon_to_xy(lat_max, 0.0)
    y0 = max(1, int(math.floor(float(min(ya, yb)))) - pad)
    y1 = min(int(dataset.ny_global), int(math.ceil(float(max(ya, yb)))) + pad)
    return x0, x1, y0, y1


def _pixel_xyz(dataset, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit vectors of the centres of source pixels ``(x, y)`` (1-based)."""
    lat, lon = dataset.xy_to_latlon(np.asarray(x, np.float64),
                                    np.asarray(y, np.float64))
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)), axis=-1)


def _bands(window):
    """Row bands of ``window`` holding at most ``_BAND_PIXELS`` pixels each."""
    x0, x1, y0, y1 = window
    rows = max(1, _BAND_PIXELS // (x1 - x0 + 1))
    for start in range(y0, y1 + 1, rows):
        yield x0, x1, start, min(y1, start + rows - 1)


def gcell_sums(geometry, assigner, dataset, *, pad: int, label: str,
               progress: Callable[[str], None] | None = None):
    """geogrid's average_gcell numerator and denominator on the mesh: the sum
    and count of the valid source pixels whose centres fall in each Voronoi
    cell (nearest-centre assignment).  Returns ``(total, count, window)``."""
    n = geometry.n_cells
    total = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    window = _source_window(dataset, _footprint(geometry), pad)
    for bx0, bx1, by0, by1 in _bands(window):
        values = dataset.read_window(bx0, bx1, by0, by1).values()
        valid = np.flatnonzero(np.isfinite(values.reshape(-1)))
        if valid.size == 0:
            continue
        nx = bx1 - bx0 + 1
        rows, cols = np.divmod(valid, nx)
        local = assigner.assign(_pixel_xyz(dataset, bx0 + cols, by0 + rows))
        keep = local >= 0
        cells = assigner.cells[local[keep]]
        total += np.bincount(cells, values.reshape(-1)[valid][keep], minlength=n)
        count += np.bincount(cells, minlength=n)
        if progress is not None:
            progress(f"{label} rows {by0}-{by1}: {int(valid.size)} valid pixels")
    return total, count, window


def _cell_source_xy(geometry, dataset, cells):
    lat = np.degrees(np.arcsin(np.clip(geometry.xyz[cells, 2], -1, 1)))
    lon = np.degrees(np.arctan2(geometry.xyz[cells, 1], geometry.xyz[cells, 0]))
    return dataset.latlon_to_xy(lat, lon)


#: WPS v4.6.0 GEOGRID.TBL.ARW:511-518 (IRRIGATION).
IRRIGATION_DATASET = "irrigation/fao"
IRRIGATION_FILL = 0.0


def irrigation_cells(geometry, assigner, land: np.ndarray, dataset, *,
                     progress: Callable[[str], None] | None = None
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """GEOGRID's IRRIGATION on the mesh.

    ``average_gcell(0.0)+average_4pt+nearest_neighbor``, ``masked = water``,
    ``fill_missing = 0``: the mean of the valid source pixels in the cell;
    a cell finer than the source (no pixel centre in it) takes the mean of
    the four source points around its centre, then the nearest point; water
    cells hold the fill.
    """
    n = geometry.n_cells
    total, count, window = gcell_sums(geometry, assigner, dataset, pad=2,
                                      label="irrigation", progress=progress)
    out = np.full(n, IRRIGATION_FILL)
    gcell = land & (count > 0)
    out[gcell] = total[gcell] / count[gcell]
    rest = np.flatnonzero(land & (count == 0))
    four = nearest = 0
    if rest.size:
        xs, ys = _cell_source_xy(geometry, dataset, rest)
        for k, cell in enumerate(rest):
            i0, j0 = int(np.floor(xs[k])), int(np.floor(ys[k]))
            patch = dataset.read_window(i0, i0 + 1, j0, j0 + 1).values()
            valid = patch[np.isfinite(patch)]
            if valid.size:
                # average_4pt averages the valid points among the four.
                out[cell] = float(valid.mean())
                four += 1
                continue
            ci, cj = int(np.rint(xs[k])), int(np.rint(ys[k]))
            value = dataset.read_window(ci, ci, cj, cj).values().reshape(-1)[0]
            if np.isfinite(value):
                out[cell] = float(value)
                nearest += 1
    return out.astype(np.float32), {
        "method": "average_gcell(0.0)+average_4pt+nearest_neighbor, masked=water, "
                  "fill 0 (GEOGRID.TBL IRRIGATION) over each Voronoi cell",
        "source_window": list(window),
        "land_cells": int(np.count_nonzero(land)),
        "cells_averaged": int(np.count_nonzero(gcell)),
        "cells_average_4pt": int(four),
        "cells_nearest": int(nearest),
    }


def lake_depth_cells(geometry, assigner, water: np.ndarray, dataset, *,
                     progress: Callable[[str], None] | None = None
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """GEOGRID's LAKE_DEPTH on the mesh (module docstring)."""
    n = geometry.n_cells
    total, count, window = gcell_sums(
        geometry, assigner, dataset, pad=LAKE_DEPTH_SEARCH + 1,
        label="lake_depth", progress=progress)
    depth = np.full(n, LAKE_DEPTH_FILL_M)
    gcell = water & (count > 0)
    depth[gcell] = total[gcell] / count[gcell]
    searched = np.flatnonzero(water & (count == 0))
    found = 0
    if searched.size:
        lat = np.degrees(np.arcsin(np.clip(geometry.xyz[searched, 2], -1, 1)))
        lon = np.degrees(np.arctan2(geometry.xyz[searched, 1], geometry.xyz[searched, 0]))
        xs, ys = dataset.latlon_to_xy(lat, lon)
        offsets = [(di, dj) for dj in range(-LAKE_DEPTH_SEARCH, LAKE_DEPTH_SEARCH + 1)
                   for di in range(-LAKE_DEPTH_SEARCH, LAKE_DEPTH_SEARCH + 1)
                   if abs(di) + abs(dj) <= LAKE_DEPTH_SEARCH]
        for k, cell in enumerate(searched):
            ci, cj = int(np.rint(xs[k])), int(np.rint(ys[k]))
            patch = dataset.read_window(
                ci - LAKE_DEPTH_SEARCH, ci + LAKE_DEPTH_SEARCH,
                max(1, cj - LAKE_DEPTH_SEARCH), cj + LAKE_DEPTH_SEARCH).values()
            base_j = max(1, cj - LAKE_DEPTH_SEARCH)
            best = None
            for di, dj in offsets:
                jj = cj + dj - base_j
                if jj < 0 or jj >= patch.shape[0]:
                    continue
                value = patch[jj, di + LAKE_DEPTH_SEARCH]
                if not np.isfinite(value):
                    continue
                # geogrid's search: the shallowest breadth-first level that
                # holds a valid pixel, then the pixel nearest the point.
                key = (abs(di) + abs(dj),
                       (ci + di - xs[k]) ** 2 + (cj + dj - ys[k]) ** 2)
                if best is None or key < best[0]:
                    best = (key, value)
            if best is not None:
                depth[cell] = best[1]
                found += 1
    receipt = {
        "method": "average_gcell(1.0)+search(5), masked=land, fill 10 m "
                  "(GEOGRID.TBL LAKE_DEPTH) over each Voronoi cell",
        "source_window": list(window),
        "water_cells": int(np.count_nonzero(water)),
        "cells_averaged": int(np.count_nonzero(gcell)),
        "cells_searched": int(searched.size),
        "cells_search_found": int(found),
        "cells_filled": int(n - np.count_nonzero(gcell) - found),
    }
    return depth.astype(np.float32), receipt


def category_counts(geometry, assigner, candidates: np.ndarray, dataset, *,
                    progress: Callable[[str], None] | None = None
                    ) -> tuple[np.ndarray, int, tuple[int, int, int, int]]:
    """Per-cell pixel counts of each category of a categorical dataset.

    Every source pixel goes to the Voronoi cell whose centre is nearest (MPAS
    init's and geogrid's gcell rule); only ``candidates`` cells are counted.
    A candidate cell no pixel centre falls in (a cell finer than the source)
    takes the category of the pixel under its centre, geogrid's nearest
    neighbour.  Returns ``(counts[n, categories], category_min, window)``.
    """
    index = dataset.index
    cmin = int(index.category_min or 1)
    cmax = int(index.category_max)
    categories = cmax - cmin + 1
    n = geometry.n_cells
    counts = np.zeros((n, categories), dtype=np.int64)
    window = _source_window(dataset, _footprint(geometry), 1)
    for bx0, bx1, by0, by1 in _bands(window):
        raw = dataset.read_window(bx0, bx1, by0, by1).raw[0].reshape(-1)
        nx = bx1 - bx0 + 1
        rows, cols = np.divmod(np.arange(raw.size), nx)
        local = assigner.assign(_pixel_xyz(dataset, bx0 + cols, by0 + rows))
        keep = local >= 0
        cells = assigner.cells[local[keep]]
        cats = raw[keep].astype(np.int64)
        ok = candidates[cells] & (cats >= cmin) & (cats <= cmax)
        flat = cells[ok] * categories + (cats[ok] - cmin)
        counts += np.bincount(flat, minlength=n * categories).reshape(n, categories)
        if progress is not None:
            progress(f"{dataset.path.name} rows {by0}-{by1}")
    empty = np.flatnonzero(candidates & (counts.sum(axis=1) == 0))
    if empty.size:
        lat = np.degrees(np.arcsin(np.clip(geometry.xyz[empty, 2], -1, 1)))
        lon = np.degrees(np.arctan2(geometry.xyz[empty, 1], geometry.xyz[empty, 0]))
        xs, ys = dataset.latlon_to_xy(lat, lon)
        for k, cell in enumerate(empty):
            ci, cj = int(np.rint(xs[k])), int(np.rint(ys[k]))
            value = int(dataset.read_window(ci, ci, cj, cj).raw[0].reshape(-1)[0])
            if cmin <= value <= cmax:
                counts[cell, value - cmin] = 1
    return counts, cmin, window


def lake_mask_vote(geometry, assigner, water: np.ndarray, dataset, *,
                   progress: Callable[[str], None] | None = None
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    """Dominant with-lakes category == ISLAKE over each WATER cell."""
    lake = dataset.index.islake
    if lake is None or lake <= 0:
        raise SurfaceStaticsRefusal(
            f"{dataset.path} declares no positive islake, so it carries no lake class")
    counts, cmin, window = category_counts(geometry, assigner, water, dataset,
                                           progress=progress)
    voted = counts.sum(axis=1) > 0
    dominant = np.argmax(counts, axis=1) + cmin
    mask = voted & (dominant == lake)
    return mask, {
        "method": f"dominant category of {dataset.path.name} over each water "
                  "cell (nearest-centre pixel assignment, lowest category on a "
                  "tie) == islake",
        "islake": int(lake), "source_window": list(window),
        "water_cells_voted": int(np.count_nonzero(voted)),
    }


#: The field groups ``--fields`` names: each is one scheme's inputs.
FIELD_GROUPS = ("lake", "urban", "landusef", "irrigation", "soilctop", "soilcbot")
#: geogrid's soil-category fraction fields and their WPS datasets (GEOGRID.TBL
#: SOILCTOP/SOILCBOT, dest_type categorical, 16 categories).
SOIL_FRACTION_DATASETS = {"soilctop": ("SOILCTOP", "soiltype_top_30s"),
                          "soilcbot": ("SOILCBOT", "soiltype_bot_30s")}


def landuse_fractions(counts: np.ndarray, cmin: int
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """geogrid's LANDUSEF on the mesh: ``[nLandCat, nCells]`` float32, the
    share of each category among the pixels a cell holds (planes from
    category 1; a cell with no pixel holds the one under its centre)."""
    total = counts.sum(axis=1, keepdims=True)
    fractions = np.where(total > 0, counts / np.maximum(total, 1), 0.0)
    planes = np.zeros((cmin - 1 + fractions.shape[1], fractions.shape[0]))
    planes[cmin - 1:] = fractions.T
    return planes.astype(np.float32), {
        "method": "pixel shares of each category per Voronoi cell "
                  "(nearest-centre assignment), geogrid's LANDUSEF",
        "categories": int(planes.shape[0]),
        "cells_without_pixels": int(np.count_nonzero(total == 0)),
    }


def urban_fraction_cells(counts: np.ndarray, cmin: int, ivgtyp, landmask
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """FRC_URB2D for SLUCM, as WOOF's WRF preparation estimates it.

    The cell's land-use category FRACTIONS (``counts``, the pixel shares of
    the static's own MODIS land use) weighted by each urban category's
    URBPARM.TBL FRC_URB, on the cells whose dominant category is urban: the
    Rust estimate the WRF path runs (woof.static.urban_fraction, algorithm
    ``landcover-area-urbparm-weighted-v1``), here on a (1, nCells) plane.
    """
    from woof.static.urban_fraction import estimate_urban_fraction

    total = counts.sum(axis=1, keepdims=True)
    fractions = np.where(total > 0, counts / np.maximum(total, 1), 0.0)
    from woof.core.urban_tables import urban_category_set
    from woof.static.highres import MODIS21_ISURBAN

    # The estimate weighs every category the urban table set names (ISURBAN
    # and the LCZ_1..LCZ_11 rows, 51-61), so the plane stack spans them; a
    # 21-class source has no pixels in the LCZ planes.
    deepest = max(urban_category_set(isurban=MODIS21_ISURBAN).urban)
    planes = np.zeros((max(cmin - 1 + fractions.shape[1], deepest),
                       fractions.shape[0]))
    planes[cmin - 1:cmin - 1 + fractions.shape[1]] = fractions.T
    frc, audit = estimate_urban_fraction(
        {"LANDUSEF": planes[:, None, :],
         "LU_INDEX": np.asarray(ivgtyp, np.float64)[None, :],
         "LANDMASK": np.asarray(landmask, np.float64)[None, :]},
        option=1, use_wudapt_lcz=0)
    frc = np.asarray(frc, dtype=np.float64).reshape(-1)
    from types import SimpleNamespace

    from .surface_inputs import urban_table_sha256

    return frc.astype(np.float32), {
        # The table estimate_urban_fraction read (load_urban_params' default);
        # the forecast refuses a run whose urban model reads another.
        "urbparm_sha256": urban_table_sha256(SimpleNamespace()),
        "method": "LANDUSEF (pixel shares per Voronoi cell) x URBPARM.TBL "
                  "FRC_URB on dominant-urban land cells (woof.static."
                  "urban_fraction, the WRF path's SLUCM estimate)",
        "audit": audit,
        "urban_cells": int(np.count_nonzero(frc > 0)),
    }


def build(static: Path, out: Path, *, geog_root: Path | None,
          lake_depth: bool = True, fields: tuple[str, ...] = ("lake",),
          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write ``out`` from ``static`` (and ``geog_root``); return the receipt."""
    import netCDF4

    from woof.static.geog import GeogDataset
    from .static_highres import VoronoiAssigner, cell_spacing_m, read_cell_geometry

    static = Path(static).expanduser().resolve(strict=True)
    with netCDF4.Dataset(static, "r") as dataset:
        geometry = read_cell_geometry(dataset)
        lat = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
        landmask = np.asarray(dataset.variables["landmask"][:]).reshape(-1)
        ivgtyp = np.asarray(dataset.variables["ivgtyp"][:]).reshape(-1)
        tc_lakemask = (np.asarray(dataset.variables["tc_lakemask"][:]).reshape(-1)
                       if "tc_lakemask" in dataset.variables else None)
        replaced_landuse = (dataset.getncattr("highres_landuse_source")
                            if "highres_landuse_source" in dataset.ncattrs() else None)
    n = geometry.n_cells
    water = landmask == 0
    selected = np.isfinite(cell_spacing_m(geometry))
    assigner = None

    def need_assigner():
        nonlocal assigner
        if assigner is None:
            assigner = VoronoiAssigner(geometry, selected)
        return assigner

    def geog(name: str):
        if geog_root is None:
            raise SurfaceStaticsRefusal(
                f"{name} is needed and no --geog-root was given")
        path = Path(geog_root).expanduser() / name
        if not (path / "index").is_file():
            raise SurfaceStaticsRefusal(
                f"the WPS_GEOG dataset {name} is not at {path}; stage it with "
                f"`woof fetch-geog --datasets {name} --root {geog_root}`")
        return GeogDataset(path)

    groups = tuple(dict.fromkeys(fields))
    unknown = sorted(set(groups) - set(FIELD_GROUPS))
    if unknown or not groups:
        raise SurfaceStaticsRefusal(
            f"--fields names {unknown or 'nothing'}; it takes {list(FIELD_GROUPS)}")
    provenance: dict[str, Any] = {
        "field_groups": list(groups),
        "parent_static": static.name,
        "parent_static_sha256": _file_sha256(static),
        "cells": int(n),
        "cells_with_closed_polygon": int(np.count_nonzero(selected)),
    }
    written: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    if "lake" in groups:
        if tc_lakemask is not None:
            mask = tc_lakemask == 1
            provenance["lakemask_source"] = (
                "static tc_lakemask (rw_mpas_static --preserve-lake-mask: the "
                "with-lakes dominant category == islake)")
        elif np.any(ivgtyp == MODIS_ISLAKE):
            mask = ivgtyp == MODIS_ISLAKE
            provenance["lakemask_source"] = (
                f"static ivgtyp == {MODIS_ISLAKE} (a with-lakes static that keeps "
                "the lake class)")
        else:
            dataset = geog(LAKE_LANDUSE_DATASET)
            mask, vote = lake_mask_vote(geometry, need_assigner(), water, dataset,
                                        progress=progress)
            vote["index_sha256"] = _index_sha256(dataset.path)
            provenance["lakemask_source"] = "with-lakes land-use vote on the mesh"
            provenance["lakemask_vote"] = vote
        on_land = mask & ~water
        if np.any(on_land) and provenance["lakemask_source"].startswith("static ivgtyp"):
            raise SurfaceStaticsRefusal(
                "the static's lake cells are not water cells: a lake is water "
                "(real.exe turns ISLAKE into ISWATER)")
        mask = mask & water
        provenance["lake_cells"] = int(np.count_nonzero(mask))
        provenance["lake_cells_on_static_land_dropped"] = int(np.count_nonzero(on_land))
        written["LAKEMASK"] = mask.astype(np.int32)
        attrs["LAKEFLAG"] = np.int32(1)
        if lake_depth:
            dataset = geog(LAKE_DEPTH_DATASET)
            depth, depth_receipt = lake_depth_cells(
                geometry, need_assigner(), water, dataset, progress=progress)
            depth_receipt["index_sha256"] = _index_sha256(dataset.path)
            if mask.any():
                values = depth[mask]
                depth_receipt["lake_depth_on_lake_cells_m"] = {
                    "min": float(values.min()), "max": float(values.max()),
                    "mean": float(values.mean())}
            provenance["lake_depth"] = depth_receipt
            written["LAKE_DEPTH"] = depth
            attrs["LAKE_DEPTH_FLAG"] = np.int32(1)
    if replaced_landuse is not None and ("urban" in groups or "landusef" in groups):
        raise SurfaceStaticsRefusal(
            f"the static's land use was replaced by `woof hex static-highres` "
            f"({replaced_landuse}); the category shares here count the "
            "30-arc-second MODIS pixels, which would disagree with that static's "
            "cells. Build the urban fraction and LANDUSEF from a static carrying "
            "the 30-arc-second land use (the high-resolution source is not wired "
            "here yet)")
    landuse = None
    if "urban" in groups or "landusef" in groups:
        # One pass over the with-lakes land use serves both groups.
        dataset = geog(LAKE_LANDUSE_DATASET)
        counts, cmin, window = category_counts(
            geometry, need_assigner(), np.ones(n, dtype=bool), dataset,
            progress=progress)
        landuse = (dataset, counts, cmin, window)
    if "landusef" in groups:
        dataset, counts, cmin, window = landuse
        written["LANDUSEF"], provenance["landusef"] = landuse_fractions(
            counts, cmin)
        provenance["landusef"].update(
            dataset=dataset.path.name, source_window=list(window),
            index_sha256=_index_sha256(dataset.path))
    if "urban" in groups:
        dataset, counts, cmin, window = landuse
        frc, urban = urban_fraction_cells(counts, cmin, ivgtyp, landmask)
        urban.update(dataset=dataset.path.name, source_window=list(window),
                     index_sha256=_index_sha256(dataset.path))
        provenance["urban"] = urban
        written["FRC_URB2D"] = frc
    for group, (variable, name) in SOIL_FRACTION_DATASETS.items():
        if group not in groups:
            continue
        dataset = geog(name)
        counts, cmin, window = category_counts(
            geometry, need_assigner(), np.ones(n, dtype=bool), dataset,
            progress=progress)
        written[variable], provenance[group] = landuse_fractions(counts, cmin)
        provenance[group].update(
            dataset=dataset.path.name, source_window=list(window),
            index_sha256=_index_sha256(dataset.path))
    if "irrigation" in groups:
        dataset = geog(IRRIGATION_DATASET)
        written["IRRIGATION"], provenance["irrigation"] = irrigation_cells(
            geometry, need_assigner(), ~water, dataset, progress=progress)
        provenance["irrigation"]["index_sha256"] = _index_sha256(dataset.path)
    fields = written
    out = Path(out).expanduser().absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial-{os.getpid()}")
    try:
        with netCDF4.Dataset(partial, "w", format="NETCDF4") as target:
            target.createDimension("nCells", n)
            # The cells' own centres, as the static stores them: the forecast
            # holds the file to its mesh's cells and their order.
            for name, values in (("latCell", lat), ("lonCell", lon)):
                target.createVariable(name, "f8", ("nCells",))[:] = values
            for name, values in fields.items():
                if values.ndim == 2:
                    # geogrid's own category dimension names.
                    dim = "nSoilCat" if name.startswith("SOILC") else "nLandCat"
                    if dim not in target.dimensions:
                        target.createDimension(dim, values.shape[0])
                    dims = (dim, "nCells")
                else:
                    dims = ("nCells",)
                variable = target.createVariable(name, values.dtype, dims)
                variable[:] = values
            target.setncattr("schema", SCHEMA)
            for name, value in attrs.items():
                target.setncattr(name, value)
            target.setncattr("provenance_json", json.dumps(provenance, sort_keys=True))
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"out": str(out), "schema": SCHEMA, "fields": sorted(fields),
            "flags": {name: int(value) for name, value in attrs.items()},
            **provenance}


def add_surface_statics_parser(commands) -> None:
    parser = commands.add_parser(
        "surface-statics",
        help="build the per-cell scheme inputs a WRF column selection reads "
             "(CLM lake mask and bathymetry, the SLUCM urban fraction) beside a "
             "hex static",
        description=__doc__.split("\n\n")[1])
    parser.add_argument("--static", type=Path, required=True, metavar="S",
                        help="the mesh's static file (rw_mpas_static output)")
    parser.add_argument("-o", "--output", type=Path, default=None, metavar="F",
                        help=f"the file to write (default: <S stem>{FILE_SUFFIX} "
                             "beside S, with a trailing .static dropped)")
    parser.add_argument("--geog-root", type=Path, default=None, metavar="DIR",
                        help="WPS_GEOG root holding lake_depth (and, for a static "
                             f"without a lake mask, {LAKE_LANDUSE_DATASET})")
    parser.add_argument("--fields", default="lake", metavar="GROUPS",
                        help="comma-separated field groups to build: "
                             f"{', '.join(FIELD_GROUPS)} (default lake); urban is "
                             "SLUCM's FRC_URB2D")
    parser.add_argument("--no-lake-depth", action="store_true",
                        help="stage the lake mask only (for use_lakedepth = 0)")
    parser.set_defaults(handler=run_cli)


def run_cli(arguments: argparse.Namespace) -> int:
    static = Path(arguments.static)
    out = arguments.output
    if out is None:
        out = builder_default_path(static)
    receipt = build(static, out, geog_root=arguments.geog_root,
                    lake_depth=not arguments.no_lake_depth,
                    fields=tuple(item.strip() for item in arguments.fields.split(",")
                                 if item.strip()),
                    progress=lambda message: print(message, flush=True))
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0
