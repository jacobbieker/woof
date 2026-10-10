"""Sample MPAS (hex) history at sites and heights.

Same output contract as :func:`woof.energy.sample.sample_wrfout`: the keys,
shapes and units in that module's docstring, returned as a
:class:`~woof.energy.sample.SampleResult`.

Two history dialects are read, by variable name:

* native MPAS-A history (``uReconstructZonal``, ``t2m``, ``xtime``, ...);
* ``woof hex forecast`` output (``cuda-history.<YYYY-MM-DD_HH.MM.SS>.nc``:
  ``u_zonal``, ``v_meridional``, ``t2``, ``pressure``, ``swdown``; the
  valid time is ``xtime`` when the frame carries it, else the label in the
  file name, which frames written before ``xtime`` was added rely on).

Mesh fields (``latCell``/``lonCell`` in radians, ``bdyMaskCell``,
``cellsOnVertex``, ``ter``) are read from ``mesh_path`` first and from the
first history file second.  ``zgrid`` is read from the first history file
that carries one -- ``woof hex forecast`` writes the run's own ``zgrid`` in
every frame -- and from ``mesh_path`` (the culled init, the plan's
``mesh["mesh_path"]``) only when no history file does.  History files whose
``zgrid_sha256`` attributes disagree come from different vertical grids and
are refused.

Horizontal: barycentric interpolation over the Delaunay triangle (the three
cells around one Voronoi vertex, ``cellsOnVertex``) that contains the site,
solved on unit-sphere vectors.  Without ``cellsOnVertex`` the nearest cell
is used and the result's notes say so.

Vertical: each of the cells is interpolated linearly in height above its own
terrain, at layer midpoints ``0.5 * (zgrid[k] + zgrid[k+1]) - zgrid[0]``
(``w`` is destaggered to the same midpoints), then the cells are weighted.
A height below the lowest or above the highest midpoint is NaN: nothing is
extrapolated, horizontally or vertically.

``inside`` is False -- and every value NaN -- for a site no triangle of the
mesh contains, or whose triangle touches a limited-area boundary cell
(``bdyMaskCell > 0``, the seven relaxation rings driven by the boundary
files rather than solved).

Python boundary (``docs/dev/static-rust-port.md``): this is per-site work
-- a KD-tree over triangle centres, 3x3 solves per site, and gathers of the
few cells the sites touch -- in numpy/scipy, which the boundary admits for
small per-site vectors.  Only the cells the sites' triangles touch are
read from each history variable; no field-wide arithmetic is done.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from woof.energy.sample import (
    PROFILE_VARS,
    SURFACE_VARS,
    SampleResult,
    SampleUnavailable,
)

#: Sphere radius MPAS meshes are generated on, metres.
SPHERE_RADIUS_M = 6_371_229.0

#: Output key -> history names, first match wins.  ``PRES`` also accepts
#: ``pressure_p + pressure_base`` (native MPAS when ``pressure`` is absent).
PROFILE_SOURCES: dict[str, tuple[str, ...]] = {
    "U": ("uReconstructZonal", "u_zonal"),
    "V": ("uReconstructMeridional", "v_meridional"),
    "W": ("w",),
    "THETA": ("theta",),
    "PRES": ("pressure",),
    "QVAPOR": ("qv",),
    "QCLOUD": ("qc",),
    "QRAIN": ("qr",),
    "QICE": ("qi",),
    "QSNOW": ("qs",),
    "QGRAUP": ("qg",),
}
SURFACE_SOURCES: dict[str, tuple[str, ...]] = {
    "U10": ("u10",),
    "V10": ("v10",),
    "T2": ("t2m", "t2"),
    "Q2": ("q2",),
    "PSFC": ("surface_pressure",),
    "SWDOWN": ("swdnb", "swdown"),
    "SWDDNI": ("swddni",),
    "SWDDIF": ("swddif",),
    "RAINNC": ("rainnc",),
    "RAINC": ("rainc",),
    "COSZEN": ("coszr",),
}
PRESSURE_PARTS = ("pressure_p", "pressure_base")

_MESH_NAMES = ("latCell", "lonCell", "bdyMaskCell", "cellsOnVertex",
               "ter")
_LABEL = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[.:](\d{2})[.:](\d{2})")


def _dataset(path: Path):
    try:
        from netCDF4 import Dataset
    except ImportError as error:  # core dependency; refuse clearly anyway
        raise SampleUnavailable("netCDF4 is required to read MPAS history"
                                ) from error
    try:
        return Dataset(str(path))
    except OSError as error:
        raise SampleUnavailable(f"{path} is not a readable netCDF file: "
                                f"{error}") from error


def _array(variable, index: Any = Ellipsis) -> np.ndarray:
    data = variable[index]
    if np.ma.isMaskedArray(data):
        if data.dtype.kind in "iu":
            return np.asarray(data.filled(0))
        return np.asarray(data.astype(np.float64).filled(np.nan))
    return np.asarray(data)


def _source_names(paths: Sequence[Path], mesh_path: Path | None
                  ) -> tuple[set[str], set[str]]:
    """Variable names in the first history file, and in the mesh file."""

    if not paths:
        raise SampleUnavailable("no MPAS history files were given")
    with _dataset(Path(paths[0])) as ds:
        history = set(ds.variables)
    mesh: set[str] = set()
    if mesh_path is not None:
        with _dataset(Path(mesh_path)) as ds:
            mesh = set(ds.variables)
    return history, mesh


def _resolve(key: str, history: set[str]) -> tuple[str, ...] | None:
    sources = PROFILE_SOURCES.get(key) or SURFACE_SOURCES.get(key) or ()
    for name in sources:
        if name in history:
            return (name,)
    if key == "PRES" and all(part in history for part in PRESSURE_PARTS):
        return PRESSURE_PARTS
    return None


def available_variables(paths: Sequence[Path], *,
                        mesh_path: Path | None = None) -> set[str]:
    """Output keys the history (plus mesh) can supply.

    Profile keys need ``zgrid`` (from ``mesh_path`` or the history itself)
    to be placed in height, so without it none are reported.
    """

    history, mesh = _source_names(paths, mesh_path)
    has_z = "zgrid" in history or "zgrid" in mesh
    out = set()
    for key in SURFACE_VARS:
        if _resolve(key, history):
            out.add(key)
    if has_z:
        for key in PROFILE_VARS:
            if _resolve(key, history):
                out.add(key)
    return out


# --------------------------------------------------------------------------
# mesh and horizontal weights


@dataclass
class _Mesh:
    lat: np.ndarray            # radians
    lon: np.ndarray
    zgrid: np.ndarray | None   # (nCells, nVertLevelsP1)
    bdy: np.ndarray | None     # (nCells,)
    cells_on_vertex: np.ndarray | None   # (nVertices, 3), zero-based, -1 none
    ter: np.ndarray | None
    notes: list[str]

    @property
    def n_cells(self) -> int:
        return int(self.lat.shape[0])

    def xyz(self) -> np.ndarray:
        return np.column_stack([np.cos(self.lat) * np.cos(self.lon),
                                np.cos(self.lat) * np.sin(self.lon),
                                np.sin(self.lat)])


def _read_mesh(paths: Sequence[Path], mesh_path: Path | None) -> _Mesh:
    found: dict[str, np.ndarray] = {}
    history_path = Path(paths[0])
    sources = ([Path(mesh_path)] if mesh_path is not None else []) + \
        [history_path]
    # zgrid: the first history file that carries one (the run's own
    # vertical grid; every other frame carrying one must agree with it by
    # zgrid_sha256, checked in sample_mpas), the mesh file only when no
    # history file does.
    zgrid_from = None
    zgrid_sources = [(Path(p), "history") for p in paths]
    if mesh_path is not None:
        zgrid_sources.append((Path(mesh_path), "mesh"))
    for source, kind in zgrid_sources:
        with _dataset(source) as ds:
            if "zgrid" in ds.variables:
                variable = ds.variables["zgrid"]
                index = tuple(0 if d == "Time" else slice(None)
                              for d in variable.dimensions)
                found["zgrid"] = _array(variable, index)
                zgrid_from = kind
                break
    for source in sources:
        with _dataset(source) as ds:
            for name in _MESH_NAMES:
                if name in found or name not in ds.variables:
                    continue
                variable = ds.variables[name]
                if "Time" in variable.dimensions:
                    index = tuple(0 if d == "Time" else slice(None)
                                  for d in variable.dimensions)
                    found[name] = _array(variable, index)
                else:
                    found[name] = _array(variable)
    if "latCell" not in found or "lonCell" not in found:
        raise SampleUnavailable(
            "neither the mesh file nor the history carries latCell/lonCell, "
            "so the sites cannot be placed on the mesh")
    notes: list[str] = []
    lat = found["latCell"].astype(np.float64)
    lon = found["lonCell"].astype(np.float64)
    if np.nanmax(np.abs(lat)) > np.pi / 2.0 + 1e-3:
        lat, lon = np.radians(lat), np.radians(lon)
        notes.append("latCell/lonCell read as degrees (values exceed pi/2)")
    zgrid = found.get("zgrid")
    if zgrid_from is not None:
        notes.append(f"zgrid read from the {zgrid_from} file")
    if zgrid is not None:
        zgrid = zgrid.astype(np.float64)
        if zgrid.shape[0] != lat.shape[0] and zgrid.shape[-1] == lat.shape[0]:
            zgrid = zgrid.T
        if zgrid.shape[0] != lat.shape[0]:
            raise SampleUnavailable(
                f"zgrid has shape {zgrid.shape} for {lat.shape[0]} cells")
    bdy = found.get("bdyMaskCell")
    cov = found.get("cellsOnVertex")
    if cov is not None:
        cov = cov.astype(np.int64) - 1          # MPAS indices are one-based
    ter = found.get("ter")
    return _Mesh(lat, lon, zgrid,
                 None if bdy is None else bdy.astype(np.int64),
                 cov, None if ter is None else ter.astype(np.float64), notes)


def _site_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)])


def horizontal_weights(mesh: _Mesh, lat_deg: np.ndarray, lon_deg: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """``(cells (S, 3), weights (S, 3), inside (S,), method)``.

    Unused slots carry cell 0 with weight 0.  ``inside`` is False where no
    triangle contains the site or a used cell is a boundary cell.
    """

    from scipy.spatial import cKDTree

    xyz = mesh.xyz()
    sites = _site_xyz(lat_deg, lon_deg)
    count = len(sites)
    cells = np.zeros((count, 3), dtype=np.int64)
    weights = np.zeros((count, 3))
    inside = np.zeros(count, dtype=bool)
    triangles = None
    if mesh.cells_on_vertex is not None:
        cov = mesh.cells_on_vertex
        ok = np.all((cov >= 0) & (cov < mesh.n_cells), axis=1)
        if cov.ndim == 2 and cov.shape[1] == 3 and np.any(ok):
            triangles = cov[ok]
    if triangles is not None:
        method = "barycentric over cellsOnVertex triangles"
        centres = xyz[triangles].mean(axis=1)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        k = min(12, len(triangles))
        _, candidates = cKDTree(centres).query(sites, k=k)
        candidates = np.asarray(candidates).reshape(count, k)
        for s in range(count):
            for t in candidates[s]:
                tri = triangles[t]
                matrix = xyz[tri].T
                try:
                    w = np.linalg.solve(matrix, sites[s])
                except np.linalg.LinAlgError:
                    continue
                total = w.sum()
                if total <= 0.0:
                    continue
                w = w / total
                if np.all(w >= -1e-9):
                    cells[s] = tri
                    weights[s] = np.clip(w, 0.0, None)
                    weights[s] /= weights[s].sum()
                    inside[s] = True
                    break
    else:
        method = "nearest cell (no cellsOnVertex in the mesh or history)"
        tree = cKDTree(xyz)
        distance, nearest = tree.query(sites)
        neighbour, _ = tree.query(xyz[nearest], k=2)
        spacing = np.asarray(neighbour)[:, 1]
        cells[:, 0] = nearest
        weights[:, 0] = 1.0
        inside = np.asarray(distance) <= spacing
    if mesh.bdy is not None:
        touches = np.any((mesh.bdy[cells] > 0) & (weights > 0.0), axis=1)
        inside &= ~touches
    weights[~inside] = 0.0
    return cells, weights, inside, method


# --------------------------------------------------------------------------
# vertical


def _vertical_plan(zgrid: np.ndarray, used: np.ndarray,
                   heights: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    """Per used cell and height: lower level, fraction, valid mask."""

    z = zgrid[used]
    mid = 0.5 * (z[:, :-1] + z[:, 1:]) - z[:, :1]          # (C, L)
    levels = mid.shape[1]
    if levels < 2:
        raise SampleUnavailable("zgrid has fewer than two mass levels")
    upper = np.sum(mid[:, :, None] < heights[None, None, :], axis=1)  # (C, H)
    valid = ((heights[None, :] >= mid[:, :1])
             & (heights[None, :] <= mid[:, -1:]))
    lower = np.clip(upper - 1, 0, levels - 2)
    z0 = np.take_along_axis(mid, lower, axis=1)
    z1 = np.take_along_axis(mid, lower + 1, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(z1 > z0, (heights[None, :] - z0) / (z1 - z0), 0.0)
    return lower, frac, valid


# --------------------------------------------------------------------------
# times


def _times(dataset, path: Path) -> list[np.datetime64]:
    count = len(dataset.dimensions["Time"]) if "Time" in dataset.dimensions \
        else 1
    if "xtime" in dataset.variables:
        from netCDF4 import chartostring

        raw = dataset.variables["xtime"]
        raw.set_auto_mask(False)
        values = np.asarray(raw[:])
        if values.dtype.kind == "S" and values.dtype.itemsize == 1:
            values = chartostring(values.reshape(count, -1))
        return [_parse_label(str(text if not isinstance(text, bytes)
                                 else text.decode("ascii", "ignore")), path)
                for text in np.atleast_1d(values)]
    match = _LABEL.search(path.name)
    if match and count == 1:
        return [_parse_label(match.group(0), path)]
    raise SampleUnavailable(
        f"{path} carries no xtime and its name has no "
        "YYYY-MM-DD_HH.MM.SS label, so its valid time is unknown")


def _parse_label(text: str, path: Path) -> np.datetime64:
    match = _LABEL.search(text)
    if not match:
        raise SampleUnavailable(f"{path}: cannot read a valid time from "
                                f"{text!r}")
    day, hh, mm, ss = match.groups()
    stamp = datetime.strptime(f"{day} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    return np.datetime64(stamp, "s")


# --------------------------------------------------------------------------
# the sampler


def _cell_axis(variable, n_cells: int) -> list[str]:
    dims = list(variable.dimensions)
    if "nCells" not in dims:
        raise SampleUnavailable(f"{variable.name} is not on cells "
                                f"(dimensions {dims})")
    if len(variable.dimensions) and variable.shape[dims.index("nCells")] \
            != n_cells:
        raise SampleUnavailable(
            f"{variable.name} has {variable.shape[dims.index('nCells')]} "
            f"cells but the mesh has {n_cells}: the mesh file is not this "
            "history's mesh")
    return dims


def _read_field(dataset, name: str, time_index: int, n_cells: int,
                cells: np.ndarray, ndim: int) -> np.ndarray:
    """``(C,)`` or ``(C, levels)`` float64 at the sorted ``cells``, one time.

    Only those cells are read (netCDF4 orthogonal indexing), never the
    whole field.
    """

    variable = dataset.variables[name]
    dims = _cell_axis(variable, n_cells)
    kept = [d for d in dims if d != "Time"]
    if len(kept) != ndim:
        raise SampleUnavailable(
            f"{name} has dimensions {dims}; a "
            f"{'profile' if ndim == 2 else 'surface'} field needs "
            f"{'(nCells, levels)' if ndim == 2 else '(nCells,)'}")
    index = tuple(time_index if d == "Time"
                  else (cells if d == "nCells" else slice(None))
                  for d in dims)
    data = _array(variable, index).astype(np.float64)
    order = [kept.index("nCells")] + [i for i, d in enumerate(kept)
                                      if d != "nCells"]
    return np.transpose(data, order)


def sample_mpas(paths: Sequence[Path], lat: np.ndarray, lon: np.ndarray,
                heights_m: Sequence[float],
                profile_vars: Sequence[str] = PROFILE_VARS,
                surface_vars: Sequence[str] = SURFACE_VARS, *,
                mesh_path: Path | None = None) -> SampleResult:
    """Sample MPAS history files (time-ordered) at the sites."""

    paths = [Path(p) for p in paths]
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    heights = np.asarray(list(heights_m), dtype=np.float64)
    if lat.shape != lon.shape or lat.ndim != 1:
        raise SampleUnavailable("lat and lon must be 1-D and the same length")
    unknown = ([v for v in profile_vars if v not in PROFILE_SOURCES]
               + [v for v in surface_vars if v not in SURFACE_SOURCES])
    if unknown:
        raise SampleUnavailable(
            f"unknown sample keys {unknown}: profile keys are "
            f"{sorted(PROFILE_SOURCES)}, surface keys {sorted(SURFACE_SOURCES)}")
    history, _ = _source_names(paths, None)
    mesh = _read_mesh(paths, mesh_path)
    notes = list(mesh.notes)
    cells, weights, inside, method = horizontal_weights(mesh, lat, lon)
    notes.append(f"horizontal: {method}")

    profile_keys = [k for k in profile_vars if _resolve(k, history)]
    surface_keys = [k for k in surface_vars if _resolve(k, history)]
    if profile_keys and mesh.zgrid is None:
        notes.append("no zgrid in the mesh file or history: profile "
                     f"variables {profile_keys} cannot be placed in height "
                     "and were not sampled")
        profile_keys = []
    skipped = [k for k in list(profile_vars) + list(surface_vars)
               if k not in profile_keys and k not in surface_keys]
    if skipped:
        notes.append(f"not in this history, not sampled: {sorted(skipped)}")

    # Only cells carrying weight are read; zero-weight slots index cell 0 of
    # the gathered block and are masked out of every sum.
    used = np.unique(cells[weights > 0.0]) if np.any(inside) else \
        np.zeros(0, dtype=np.int64)
    position = np.full(mesh.n_cells, -1, dtype=np.int64)
    position[used] = np.arange(len(used))
    local = position[cells]                          # (S, 3)
    local[(local < 0) | ~inside[:, None]] = 0
    if profile_keys and len(used):
        lower, frac, valid = _vertical_plan(mesh.zgrid, used, heights)
        weighted = weights[inside] > 0.0                       # (S_in, 3)
        covered = np.where(weighted[:, :, None], valid[local[inside]], True)
        out_of_column = int(np.sum(~covered.all(axis=(1, 2))))
        if out_of_column:
            notes.append(f"{out_of_column} inside site(s) have heights below "
                         "the lowest or above the highest mass level; those "
                         "values are NaN (not extrapolated)")

    def to_sites(per_cell: np.ndarray) -> np.ndarray:
        """Weighted sum over the three cells; NaN outside.

        A zero-weight slot contributes nothing even when its value is NaN.
        """

        gathered = per_cell[local]
        if per_cell.ndim == 1:
            gathered = np.where(weights > 0.0, gathered, 0.0)
            values = np.einsum("sk,sk->s", weights, gathered)
        else:
            gathered = np.where(weights[:, :, None] > 0.0, gathered, 0.0)
            values = np.einsum("sk,skh->sh", weights, gathered)
        values[~inside] = np.nan
        return values

    times: list[np.datetime64] = []
    profile: dict[str, list[np.ndarray]] = {k: [] for k in profile_keys}
    surface: dict[str, list[np.ndarray]] = {k: [] for k in surface_keys}
    count = len(lat)
    zgrid_digests: dict[str, Path] = {}
    for path in paths:
        with _dataset(path) as ds:
            if "zgrid_sha256" in ds.ncattrs():
                zgrid_digests.setdefault(str(ds.getncattr("zgrid_sha256")),
                                         path)
                if len(zgrid_digests) > 1:
                    first, second = list(zgrid_digests.values())[:2]
                    raise SampleUnavailable(
                        f"{first} and {second} carry different zgrid "
                        "(zgrid_sha256 differs): the history files come "
                        "from different vertical grids and cannot be "
                        "sampled as one series")
            file_times = _times(ds, path)
            names = set(ds.variables)
            for t, stamp in enumerate(file_times):
                times.append(stamp)
                for key in profile_keys:
                    source = _resolve(key, names)
                    if source is None:
                        raise SampleUnavailable(
                            f"{path} lacks {PROFILE_SOURCES[key]} that the "
                            "first history file carries")
                    if not len(used):
                        profile[key].append(np.full((count, len(heights)),
                                                    np.nan))
                        continue
                    column = sum(_read_field(ds, n, t, mesh.n_cells, used, 2)
                                 for n in source)           # (C, L)
                    if key == "W" and column.shape[1] == mesh.zgrid.shape[1]:
                        column = 0.5 * (column[:, :-1] + column[:, 1:])
                    if column.shape[1] != mesh.zgrid.shape[1] - 1:
                        raise SampleUnavailable(
                            f"{path}: {source} has {column.shape[1]} levels "
                            f"but zgrid describes {mesh.zgrid.shape[1] - 1}")
                    v0 = np.take_along_axis(column, lower, axis=1)
                    v1 = np.take_along_axis(column, lower + 1, axis=1)
                    at_heights = np.where(valid, v0 + frac * (v1 - v0),
                                          np.nan)            # (C, H)
                    profile[key].append(to_sites(at_heights))
                for key in surface_keys:
                    source = _resolve(key, names)
                    if source is None:
                        raise SampleUnavailable(
                            f"{path} lacks {SURFACE_SOURCES[key]} that the "
                            "first history file carries")
                    if not len(used):
                        surface[key].append(np.full(count, np.nan))
                        continue
                    surface[key].append(to_sites(_read_field(
                        ds, source[0], t, mesh.n_cells, used, 1)))

    time_array = np.asarray(times, dtype="datetime64[s]")
    order = np.argsort(time_array, kind="stable")
    if np.any(order != np.arange(len(order))):
        notes.append("history files were not in time order; sorted")
    if len(np.unique(time_array)) != len(time_array):
        raise SampleUnavailable("two history frames share a valid time")

    if len(used) and mesh.zgrid is not None:
        terrain = to_sites(mesh.zgrid[used, 0])
    elif len(used) and mesh.ter is not None:
        terrain = to_sites(mesh.ter[used])
    else:
        terrain = np.full(count, np.nan)
        if len(used):
            notes.append("no zgrid or ter: terrain height unknown")

    return SampleResult(
        times=time_array[order],
        heights_m=heights,
        profile={k: np.stack(v)[order] for k, v in profile.items()},
        surface={k: np.stack(v)[order] for k, v in surface.items()},
        inside=inside,
        terrain_m=terrain,
        dx_m=_spacing_m(mesh, used),
        source=", ".join(str(p) for p in paths),
        notes=notes,
    )


def _spacing_m(mesh: _Mesh, used: np.ndarray) -> float:
    """Median centre-to-nearest-centre distance around the sites, metres."""

    from scipy.spatial import cKDTree

    xyz = mesh.xyz()
    probe = used if len(used) else np.arange(min(mesh.n_cells, 2000))
    if mesh.n_cells < 2 or not len(probe):
        return float("nan")
    chord, _ = cKDTree(xyz).query(xyz[probe], k=2)
    chord = np.asarray(chord)[:, 1]
    return float(np.median(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
                 * SPHERE_RADIUS_M)


__all__ = ["PROFILE_SOURCES", "SURFACE_SOURCES", "available_variables",
           "horizontal_weights", "sample_mpas"]
