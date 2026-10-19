"""Crop a global regular latitude/longitude NetCDF source to the run's area.

A whole-globe source published as NetCDF (one variable per object, every
object on one regular grid) is decoded WHOLE by the mapped engine: its
NetCDF path decodes the whole series before carving it by valid time, and
it cannot window the decode the way its GRIB2 path does.  A small domain
then pays for the globe at every level of every valid time -- a 27-hour,
3-hourly window of a 10 km global grid with 37 pressure levels grew the
engine past 68 GB of resident memory, for a domain about 90 km wide.

This stage runs in ``woof prep`` before the input manifest is authored, so
the manifest binds the files the engine actually reads.  It keeps the
rows and columns of a box around the run -- the ``[fetch] area`` the
experiment config declares, or else the namelist's domains plus
:data:`DEFAULT_MARGIN_DEG` -- plus :data:`STENCIL_PAD_CELLS` cells each
side, and writes each object again with every other byte unchanged:
variables, attributes, dtypes, compression and fill values are copied;
only the latitude and longitude dimensions are shorter.

Longitude is cut on the source's own circle.  A box crossing the source
axis's seam (the antimeridian for a -180..180 axis, the prime meridian for
a 0..360 axis) keeps its columns in eastward order and continues past the
seam by adding 360 degrees, so the cropped axis stays uniform and
increasing: the shape every regular-grid consumer requires
(:func:`woof.ingest.horiz._regular_coordinates`).  The values that move
are exact multiples of the source increment, so a cropped terrain
supplement and a cropped primary still match coordinate for coordinate.

The stage is DECLARED by a packaged profile (``input_crop``) rather than
listed here, and :func:`crops_at_preparation` is what lets ``woof fetch``
accept ``--area`` for such a source: the fetch still transfers whole
published objects (there is no subsetting service in front of them) and
this stage is where the area is honoured.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np

#: The one crop kind this module implements.
CROP_KIND = "regular-latlon-netcdf-v1"

#: The receipt written beside the cropped objects.
RECEIPT_SCHEMA = "gpuwm-regional-input-crop-v1"
RECEIPT_NAME = "crop-receipt.json"

#: Degrees added round the namelist's domains when the experiment config
#: declares no ``[fetch] area``: the margin ``woof domain`` adds to a
#: regular-grid source's fetch box (``domain_wizard._FETCH_MARGIN_DEG``).
DEFAULT_MARGIN_DEG = 2.0

#: Source cells kept beyond the box on every side, so a box drawn exactly
#: on the domain still holds every interpolation stencil.
STENCIL_PAD_CELLS = 2

_READ_BLOCK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def declared_crop(source: str) -> str | None:
    """The crop kind ``source``'s packaged profile declares, or ``None``."""

    from woof import fetch_routes
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_profile

    try:
        adapter = get_source_adapter(fetch_routes.canonical_source(source))
    except (KeyError, ValueError):
        return None
    if adapter.packaged_profile is None:
        return None
    try:
        kind = packaged_profile(str(adapter.packaged_profile)).get("input_crop")
    except (KeyError, ValueError):
        return None
    return None if kind is None else str(kind)


def crops_at_preparation(source: str) -> bool:
    """Whether ``woof prep`` crops ``source``'s inputs to an area."""

    return declared_crop(source) is not None


def declared_crop_stage(kind: str) -> dict[str, object]:
    """What a dry run says about this stage: it opens no input."""

    _check_kind(kind)
    return {"stage": "regional_input_crop", "kind": kind,
            "margin_degrees": DEFAULT_MARGIN_DEG,
            "stencil_pad_cells": STENCIL_PAD_CELLS,
            "crop_before_manifest": True, "dry_run": True}


def _check_kind(kind: str) -> None:
    if kind != CROP_KIND:
        raise ValueError(f"unknown input crop kind {kind!r}; this WOOF "
                         f"implements {CROP_KIND!r}")


# ---------------------------------------------------------------------------
# The box
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropBox:
    """A latitude band and an eastward longitude arc, in degrees.

    ``west`` is in [0, 360) and the arc runs ``span`` degrees east of it,
    so a box over the antimeridian or the prime meridian needs no special
    spelling.  ``span >= 360`` keeps every longitude.
    """

    south: float
    north: float
    west: float
    span: float

    def __post_init__(self):
        values = (self.south, self.north, self.west, self.span)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("crop box bounds must be finite")
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError("crop box latitudes must satisfy "
                             "-90 <= south < north <= 90")
        if not 0.0 <= self.west < 360.0 or self.span <= 0.0:
            raise ValueError("crop box longitude arc must start in [0, 360) "
                             "and have a positive span")

    @property
    def full_circle(self) -> bool:
        return self.span >= 360.0

    def holds_longitude(self, longitude) -> np.ndarray:
        if self.full_circle:
            return np.ones(np.shape(longitude), dtype=bool)
        offset = np.mod(np.asarray(longitude, dtype=np.float64) - self.west, 360.0)
        return offset <= self.span

    def contains(self, other: "CropBox") -> bool:
        if other.south < self.south or other.north > self.north:
            return False
        if self.full_circle:
            return True
        if other.full_circle:
            return False
        start = (other.west - self.west) % 360.0
        return start + other.span <= self.span

    def widened(self, degrees: float) -> "CropBox":
        span = self.span + 2.0 * degrees
        return CropBox(max(-90.0, self.south - degrees),
                       min(90.0, self.north + degrees),
                       (self.west - degrees) % 360.0,
                       360.0 if span >= 360.0 else span)

    def document(self) -> dict[str, float]:
        east = self.west + min(self.span, 360.0)
        return {"south": self.south, "north": self.north,
                "west": self.west if self.west < 180.0 else self.west - 360.0,
                "east": east if east <= 180.0 else east - 360.0,
                "span": min(self.span, 360.0)}


def box_from_points(latitudes: Sequence[float], longitudes: Sequence[float]) -> CropBox:
    """The smallest band and arc holding every point.

    The arc is the circle minus its largest empty gap, so it does not
    depend on whether the points are spelled -180..180 or 0..360.
    """

    lat = np.asarray(latitudes, dtype=np.float64).ravel()
    lon = np.asarray(longitudes, dtype=np.float64).ravel()
    if lat.size == 0 or lat.size != lon.size:
        raise ValueError("domain points must be nonempty and paired")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("domain points must be finite")
    ordered = np.unique(np.mod(lon, 360.0))
    if ordered.size == 1:
        west, span = float(ordered[0]), 0.0
    else:
        gaps = np.diff(np.append(ordered, ordered[0] + 360.0))
        cut = int(np.argmax(gaps))
        west = float(ordered[(cut + 1) % ordered.size])
        span = 360.0 - float(gaps[cut])
    south, north = float(lat.min()), float(lat.max())
    # A degenerate band or arc (a single point) still needs a positive
    # extent to be a box; the stencil pad widens it to whole cells.
    if north <= south:
        north = min(90.0, north + 1.0e-9)
        south = max(-90.0, south - 1.0e-9)
    return CropBox(south, north, west % 360.0, max(span, 1.0e-9))


def box_from_area(area) -> CropBox:
    """A :class:`woof.fetch.Area` (a ``[fetch] area``) as a crop box."""

    return CropBox(float(area.lat_south), float(area.lat_north),
                   float(area.lon_west) % 360.0,
                   float(area.longitude_span_degrees))


def domain_box(wps_namelist: Path) -> CropBox:
    """The box every mass, staggered and corner point of every domain needs."""

    from woof.static.projection import grids_from_wps_namelist

    lat: list[np.ndarray] = []
    lon: list[np.ndarray] = []
    for grid in grids_from_wps_namelist(Path(wps_namelist)):
        for pair in (grid.latlon_c(), grid.latlon_mass()):
            lat.append(np.asarray(pair[0], dtype=np.float64).ravel())
            lon.append(np.asarray(pair[1], dtype=np.float64).ravel())
    return box_from_points(np.concatenate(lat), np.concatenate(lon))


def declared_area(experiment_config: Path | None):
    """The experiment config's ``[fetch] area`` (or point/radius), if any."""

    if experiment_config is None:
        return None
    import tomllib

    from types import SimpleNamespace

    from woof.fetch import _resolve_area

    path = Path(experiment_config)
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    table = document.get("fetch")
    if not isinstance(table, Mapping):
        return None
    radius = table.get("radius_km")
    return _resolve_area(SimpleNamespace(
        area=None if table.get("area") is None else str(table["area"]),
        point=None if table.get("point") is None else str(table["point"]),
        radius_km=None if radius is None else float(radius)))


def crop_box_for(wps_namelist: Path, area=None, *,
                 margin_deg: float = DEFAULT_MARGIN_DEG) -> CropBox:
    """The box this run crops to.

    A declared area is honoured as it is, and must hold every point of
    every domain: an area that leaves part of the run outside would be
    refused at the interpolation, after the decode it was meant to
    shrink, so it is refused here by name.  Without one, the domains'
    own box widened by ``margin_deg``.
    """

    need = domain_box(wps_namelist)
    if area is None:
        return need.widened(margin_deg)
    box = box_from_area(area)
    if not box.contains(need):
        found = need.document()
        raise ValueError(
            "the experiment config's [fetch] area "
            f"{area.lat_south:g},{area.lon_west:g},{area.lat_north:g},"
            f"{area.lon_east:g} does not hold the namelist's domains, which "
            f"reach latitude {found['south']:.3f}..{found['north']:.3f} and "
            f"longitude {found['west']:.3f}..{found['east']:.3f}; widen the "
            "area (woof domain prints one that fits) or remove it so the "
            "preparation crops to the domains plus "
            f"{DEFAULT_MARGIN_DEG:g} degrees")
    return box


# ---------------------------------------------------------------------------
# Index plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropPlan:
    """Which rows and columns of the source grid a crop keeps.

    ``rows`` is one contiguous run.  ``column_runs`` are contiguous runs
    of source columns in output order (two when the arc crosses the
    axis's seam), and ``column_shift`` the degrees each run's longitudes
    move by so the output axis keeps increasing.
    """

    source_shape: tuple[int, int]
    rows: tuple[int, int]
    column_runs: tuple[tuple[int, int], ...]
    column_shift: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows[1] - self.rows[0],
                sum(stop - start for start, stop in self.column_runs))

    @property
    def is_whole(self) -> bool:
        return (self.shape == self.source_shape
                and self.column_runs == ((0, self.source_shape[1]),))

    def document(self) -> dict[str, object]:
        return {"source_shape": list(self.source_shape),
                "rows": list(self.rows),
                "column_runs": [list(run) for run in self.column_runs],
                "column_shift_degrees": list(self.column_shift),
                "shape": list(self.shape)}


def _uniform_step(values: np.ndarray, name: str, *, single: bool) -> float:
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"source {name} must be a 1-D axis of at least two cells")
    steps = np.diff(values)
    # A float32 axis is judged at the rounding of its own values: four
    # float32 ulps of its largest coordinate.
    atol = (4.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(values)))
            if single else 1.0e-12)
    if (not np.allclose(steps, steps[0], rtol=0.0 if single else 1.0e-9, atol=atol)
            or steps[0] == 0.0):
        raise ValueError(f"source {name} axis is not uniform; only a regular "
                         "latitude/longitude grid is cropped")
    return float(steps[0])


def plan_crop(latitude, longitude, box: CropBox, *,
              pad: int = STENCIL_PAD_CELLS) -> CropPlan:
    """The rows and columns of a regular grid that hold ``box`` plus ``pad``."""

    from woof.ingest.horiz import global_longitude_period_columns

    if pad < 1:
        # A box narrower than a cell keeps the one nearest row or column;
        # the pad is what brackets a target point between two of them.
        raise ValueError("the stencil pad must keep at least one cell each side")
    # The plan only decides which cells to keep, so a float32 axis is
    # judged at float32 precision (a 0.1-degree float32 axis is not
    # uniform to float64 rounding); the consumers keep their own checks.
    single = any(np.asarray(axis).dtype == np.float32 for axis in (latitude, longitude))
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    _uniform_step(lat, "latitude", single=single)
    step = _uniform_step(lon, "longitude", single=single)
    if step < 0.0:
        raise ValueError("a decreasing longitude axis is not cropped")
    ny, nx = lat.size, lon.size

    inside = np.flatnonzero((lat >= box.south) & (lat <= box.north))
    if inside.size == 0:
        # A band narrower than one row: the row nearest its middle.
        inside = np.array([int(np.argmin(np.abs(lat - 0.5 * (box.south + box.north))))])
    rows = (max(0, int(inside.min()) - pad), min(ny, int(inside.max()) + 1 + pad))

    if single:
        # The mean step of the whole axis: one float32 difference can be
        # off by an ulp, enough to miss a 3600-column circle by 0.05.
        mean = (lon[-1] - lon[0]) / (nx - 1)
        columns = 360.0 / mean
        period = int(round(columns))
        if period < 2 or abs(columns - period) > 1.0e-3 or nx < period:
            period = None
    else:
        period = global_longitude_period_columns(lon)
    if box.full_circle:
        return CropPlan((ny, nx), rows, ((0, nx),), (0.0,))
    held = box.holds_longitude(lon)
    if period is None:
        columns = np.flatnonzero(held)
        if columns.size == 0:
            raise ValueError(
                "the crop box reaches none of this regional source's "
                "longitudes; the source does not cover the run")
        if columns.max() - columns.min() + 1 != columns.size:
            raise ValueError(
                "the crop box crosses this regional source's longitude "
                "seam; a regional axis is not continued across it")
        run = (max(0, int(columns.min()) - pad), min(nx, int(columns.max()) + 1 + pad))
        return CropPlan((ny, nx), rows, (run,), (0.0,))

    # A global axis: walk east from the first held column.
    offset = np.mod(lon[:period] - box.west, 360.0)
    first = int(np.argmin(np.where(held[:period], offset, np.inf)))
    if not held[first]:
        # An arc narrower than one column: the column nearest its middle.
        middle = (box.west + 0.5 * box.span) % 360.0
        distance = np.abs(np.mod(lon[:period] - middle + 180.0, 360.0) - 180.0)
        first, count = int(np.argmin(distance)), 1
    else:
        count = 0
        while count < period and held[(first + count) % period]:
            count += 1
    total = count + 2 * pad
    if total >= period:
        return CropPlan((ny, nx), rows, ((0, nx),), (0.0,))
    start = (first - pad) % period
    runs: list[tuple[int, int]] = []
    shifts: list[float] = []
    shift = 0.0
    position = start
    remaining = total
    while remaining:
        length = min(remaining, period - position)
        runs.append((position, position + length))
        shifts.append(shift)
        remaining -= length
        position = 0
        shift += 360.0
    return CropPlan((ny, nx), rows, tuple(runs), tuple(shifts))


# ---------------------------------------------------------------------------
# One object
# ---------------------------------------------------------------------------

def _axis_variable(dataset, standard_name: str, axis: str):
    """The 1-D coordinate variable naming a horizontal dimension."""

    found = []
    for name, variable in dataset.variables.items():
        if variable.ndim != 1 or variable.dimensions != (name,):
            continue
        attributes = {key: variable.getncattr(key) for key in variable.ncattrs()}
        if (attributes.get("standard_name") == standard_name
                or str(attributes.get("axis", "")).upper() == axis):
            found.append(name)
    if len(found) != 1:
        raise ValueError(
            f"{dataset.filepath()} must carry exactly one 1-D {standard_name} "
            f"coordinate, not {found}")
    return found[0]


def read_axes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(latitude, longitude)`` of one NetCDF object, as stored."""

    import netCDF4

    with netCDF4.Dataset(str(path)) as dataset:
        dataset.set_auto_maskandscale(False)
        lat_name = _axis_variable(dataset, "latitude", "Y")
        lon_name = _axis_variable(dataset, "longitude", "X")
        return (np.asarray(dataset.variables[lat_name][:]),
                np.asarray(dataset.variables[lon_name][:]))


def _take(variable, dimensions, lat_dim, lon_dim, plan: CropPlan,
          shift: bool) -> np.ndarray:
    """Read only the planned hyperslabs of ``variable`` and join them."""

    def slab(columns: tuple[int, int] | None):
        key = []
        for dimension in dimensions:
            if dimension == lat_dim:
                key.append(slice(*plan.rows))
            elif dimension == lon_dim and columns is not None:
                key.append(slice(*columns))
            else:
                key.append(slice(None))
        return np.asarray(variable[tuple(key)])

    if lon_dim not in dimensions:
        return slab(None)
    axis = dimensions.index(lon_dim)
    pieces = []
    for run, degrees in zip(plan.column_runs, plan.column_shift):
        piece = slab(run)
        if shift and degrees:
            piece = piece + piece.dtype.type(degrees)
        pieces.append(piece)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=axis)


#: Compression filters a cropped object keeps, by netCDF4's ``compression``
#: name.  Anything else (szip, blosc) is rewritten with zlib: lossless
#: either way, and the receipt records the cropped bytes' own digest.
_COMPRESSIONS = ("zlib", "zstd", "bzip2")


def _storage_options(variable, dimensions, sizes) -> dict[str, object]:
    filters = variable.filters() or {}
    options: dict[str, object] = {"endian": variable.endian()}
    compression = next((name for name in _COMPRESSIONS if filters.get(name)), None)
    if compression is None and any(filters.get(name) for name in ("szip", "blosc")):
        compression = "zlib"
    if compression is not None:
        options.update(compression=compression,
                       complevel=int(filters.get("complevel") or 4),
                       shuffle=bool(filters.get("shuffle")))
    if filters.get("fletcher32"):
        options["fletcher32"] = True
    chunking = variable.chunking() if dimensions else "contiguous"
    if chunking == "contiguous":
        if compression is None and not options.get("fletcher32"):
            options["contiguous"] = bool(dimensions)
    else:
        options["chunksizes"] = tuple(
            max(1, min(int(chunk), sizes[dimension]))
            for chunk, dimension in zip(chunking, dimensions))
    return options


def crop_object(source: Path, destination: Path, plan: CropPlan, *,
                axes: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """Write ``source`` again over ``plan``'s rows and columns only.

    Every variable, attribute, dtype, fill value, byte order, chunk shape
    and compression filter is copied; variables on the latitude or
    longitude dimension are read over the plan alone (the stored chunks
    outside it are never decompressed), and the longitude coordinate and
    its bounds take the plan's seam shift.  Raw values are copied: no
    mask, scale or offset is applied on either side.  ``axes``, when
    given, are the latitude and longitude the plan was made from, and the
    object must carry exactly those: the same shape on another axis would
    cut another place on the globe.
    """

    import netCDF4

    with netCDF4.Dataset(str(source)) as src:
        if src.groups:
            raise ValueError(f"{source} has groups; only a flat NetCDF object is cropped")
        src.set_auto_maskandscale(False)
        lat_dim = _axis_variable(src, "latitude", "Y")
        lon_dim = _axis_variable(src, "longitude", "X")
        shape = (len(src.dimensions[lat_dim]), len(src.dimensions[lon_dim]))
        if shape != plan.source_shape:
            raise ValueError(
                f"{source} is on a {shape[0]} x {shape[1]} grid, not the "
                f"{plan.source_shape[0]} x {plan.source_shape[1]} grid the "
                "crop was planned on; every object of one source must share "
                "its grid")
        if axes is not None and not (
                np.array_equal(np.asarray(src.variables[lat_dim][:]), axes[0])
                and np.array_equal(np.asarray(src.variables[lon_dim][:]), axes[1])):
            raise ValueError(
                f"{source} has the grid's shape but not its latitude/longitude "
                "values; every object of one source must share its grid")
        bounds = {str(src.variables[lon_dim].getncattr("bounds"))
                  if "bounds" in src.variables[lon_dim].ncattrs() else None}
        shifted = {lon_dim} | {name for name in bounds if name in src.variables}
        sizes = {lat_dim: plan.shape[0], lon_dim: plan.shape[1]}
        with netCDF4.Dataset(str(destination), "w",
                             format=src.data_model) as dst:
            dst.set_auto_maskandscale(False)
            dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, None if dimension.isunlimited()
                    else sizes.get(name, len(dimension)))
            for name, variable in src.variables.items():
                dimensions = variable.dimensions
                attributes = {key: variable.getncattr(key)
                              for key in variable.ncattrs()}
                fill = attributes.pop("_FillValue", None)
                options = _storage_options(variable, dimensions, {
                    dimension: sizes.get(dimension, len(src.dimensions[dimension]))
                    for dimension in dimensions})
                created = dst.createVariable(
                    name, variable.datatype, dimensions,
                    fill_value=fill if fill is not None else False,
                    **options)
                created.setncatts(attributes)
                if not dimensions:
                    created.assignValue(variable.getValue())
                    continue
                created[...] = _take(variable, dimensions, lat_dim, lon_dim,
                                     plan, shift=name in shifted)


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


#: The digest list a verified fetch writes beside its objects.
_DIGEST_LIST = "SHA256SUMS"


def _input_digest(path: Path, cache: dict[Path, dict[str, str]]) -> tuple[str, str]:
    """An input's SHA-256, and where it came from.

    A fetch folder lists every object it verified in ``SHA256SUMS``, and
    hashing a whole-globe window again (gigabytes) only to restate those
    digests is the dominant cost of a crop, so the listed digest is used
    and named as such.  An object the list does not name is hashed.
    """

    folder = path.parent
    if folder not in cache:
        listed: dict[str, str] = {}
        try:
            lines = (folder / _DIGEST_LIST).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            digest, _sep, name = line.partition("  ")
            if len(digest) == 64 and name:
                listed[name.strip()] = digest
        cache[folder] = listed
    digest = cache[folder].get(path.name)
    if digest is not None:
        return digest, f"{_DIGEST_LIST} beside the input"
    return _sha256(path), "hashed"


def _output_names(paths: Sequence[Path]) -> dict[Path, str]:
    names: dict[Path, str] = {}
    taken: set[str] = set()
    for index, path in enumerate(paths):
        name = path.name
        if name in taken:
            name = f"{index:04d}-{path.name}"
        taken.add(name)
        names[path] = name
    return names


def _reusable(directory: Path, identity: Mapping[str, object],
              names: Mapping[Path, str]) -> bool:
    """Whether an earlier crop of this exact request is still intact."""

    try:
        receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(receipt, dict) or receipt.get("identity") != identity:
        return False
    rows = {row.get("cropped"): row for row in receipt.get("objects", ())
            if isinstance(row, dict)}
    if set(rows) != set(names.values()):
        return False
    for name, row in rows.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != row.get("cropped_sha256"):
            return False
    return True


def crop_inputs(paths: Sequence[Path], box: CropBox, destination_root: Path, *,
                source: str, kind: str = CROP_KIND,
                pad: int = STENCIL_PAD_CELLS,
                progress=None) -> tuple[dict[Path, Path] | None, dict[str, object]]:
    """Crop every object of ``paths`` over one plan, into a keyed folder.

    Returns the map from each input to its cropped copy (``None`` when the
    box keeps the whole grid, so the inputs are read as they are) and the
    receipt.  The folder is named from the request -- the box, the pad and
    every input's path, size and modification time -- and reused when its
    receipt names the same request and every cropped object still hashes
    to what it recorded; otherwise it is built beside and moved into
    place, so a reader never sees half a crop.
    """

    _check_kind(kind)
    ordered = [Path(path).resolve() for path in paths]
    if not ordered:
        raise ValueError("there are no inputs to crop")
    if len(set(ordered)) != len(ordered):
        raise ValueError("an input is listed twice")
    latitude, longitude = read_axes(ordered[0])
    plan = plan_crop(latitude, longitude, box, pad=pad)
    summary = {"schema": RECEIPT_SCHEMA, "kind": kind, "source": source,
               "box": box.document(), "stencil_pad_cells": pad,
               "plan": plan.document()}
    if plan.is_whole:
        return None, {**summary, "cropped": False}
    stats = []
    for path in ordered:
        info = path.stat()
        stats.append([str(path), info.st_size, info.st_mtime_ns])
    identity = {"kind": kind, "source": source, "box": box.document(),
                "pad": pad, "plan": plan.document(), "inputs": stats}
    key = hashlib.sha256(_canonical(identity)).hexdigest()[:16]
    names = _output_names(ordered)
    root = Path(destination_root).resolve()
    directory = root / f"{source}-crop-{key}"
    if not _reusable(directory, identity, names):
        root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f".{source}-crop-", dir=root))
        try:
            objects = []
            recorded: dict[Path, dict[str, str]] = {}
            for count, path in enumerate(ordered, 1):
                target = work / names[path]
                crop_object(path, target, plan, axes=(latitude, longitude))
                digest, digest_from = _input_digest(path, recorded)
                objects.append({
                    "input": str(path), "input_bytes": path.stat().st_size,
                    "input_sha256": digest, "input_sha256_from": digest_from,
                    "cropped": names[path],
                    "cropped_bytes": target.stat().st_size,
                    "cropped_sha256": _sha256(target)})
                if progress is not None:
                    progress(count, len(ordered))
            receipt = {**summary, "cropped": True, "identity": identity,
                       "objects": objects}
            (work / RECEIPT_NAME).write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            try:
                os.rename(work, directory)
            except OSError:
                # Taken meanwhile (a concurrent preparation of the same
                # request), or a damaged earlier crop: keep an intact one,
                # replace anything else.
                if _reusable(directory, identity, names):
                    shutil.rmtree(work, ignore_errors=True)
                else:
                    shutil.rmtree(directory, ignore_errors=True)
                    os.rename(work, directory)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    return {path: directory / names[path] for path in ordered}, receipt


def _netcdf_supplement_roles(composition: Path) -> set[str]:
    """Supplement roles the composition reads as NetCDF on the primary grid."""

    document = json.loads(Path(composition).read_text(encoding="utf-8"))
    return {str(row["data_role"]) for row in (document.get("supplements") or {}).values()
            if isinstance(row, Mapping) and row.get("format") == "netcdf"
            and row.get("data_role")}


def crop_packaged_inputs(kind: str, args, *, progress=None) -> dict[str, object]:
    """Source front-door hook, after input-list expansion and before authoring.

    Crops the primary inputs and every NetCDF supplement the composition
    binds on the primary grid, and rebinds ``args.mapped_inputs``,
    ``args.input_list`` and ``args.supplement`` to the cropped copies.
    The folder sits beside the input manifest being authored, as the
    normalization stage's does.  Nothing is changed when the box keeps
    the whole grid.
    """

    _check_kind(kind)
    if not getattr(args, "wps_namelist", None):
        raise ValueError("the input crop needs --wps-namelist to know the run's area")
    manifest = getattr(args, "author_input_manifest", None)
    if manifest is None:
        raise ValueError("the input crop needs --author-input-manifest: a pinned "
                         "manifest binds the uncropped objects")
    area = declared_area(getattr(args, "experiment_config", None))
    box = crop_box_for(Path(args.wps_namelist), area)
    roles = _netcdf_supplement_roles(Path(args.composition))
    bindings = [str(value) for value in (args.supplement or ())]
    supplements = []
    for binding in bindings:
        role, _sep, path = binding.partition("=")
        if role in roles:
            supplements.append(Path(path).resolve())
    from woof import fetch_routes

    primary = [Path(path).resolve() for path in args.mapped_inputs]
    everything = list(dict.fromkeys(primary + supplements))
    source = fetch_routes.canonical_source(str(args.source))
    cropped, receipt = crop_inputs(
        everything, box, Path(manifest).resolve().parent,
        source=source, kind=kind, progress=progress)
    result = {"stage": "regional_input_crop", "kind": kind, "source": source,
              "area_from": "experiment [fetch] area" if area is not None
              else f"namelist domains + {DEFAULT_MARGIN_DEG:g} degrees",
              "box": receipt["box"], "plan": receipt["plan"],
              "cropped": cropped is not None, "inputs": len(everything)}
    if cropped is None:
        return result
    directory = next(iter(cropped.values())).parent
    args.mapped_inputs = [cropped[path] for path in primary]
    input_list = directory / "inputs.txt"
    text = "".join(f"{path}\n" for path in args.mapped_inputs)
    if not input_list.is_file() or input_list.read_text(encoding="utf-8") != text:
        input_list.write_text(text, encoding="utf-8")
    args.input_list = input_list
    rebound = []
    for binding in bindings:
        role, _sep, path = binding.partition("=")
        if role in roles:
            binding = f"{role}={cropped[Path(path).resolve()]}"
        rebound.append(binding)
    args.supplement = rebound
    # The manifest now binds bytes that depend on the run's area, so it is
    # authored inside the area-keyed folder.  At the fetch's shared path a
    # second domain prepared from the same download (or one the fetch's
    # earlier, uncropped handoff already bound) would be refused as a
    # different manifest.
    args.author_input_manifest = directory / Path(manifest).name
    result.update(directory=str(directory), receipt=str(directory / RECEIPT_NAME),
                  input_manifest=str(args.author_input_manifest),
                  input_bytes=sum(row["input_bytes"] for row in receipt["objects"]),
                  cropped_bytes=sum(row["cropped_bytes"] for row in receipt["objects"]))
    return result


__all__ = ["CROP_KIND", "RECEIPT_SCHEMA", "RECEIPT_NAME", "DEFAULT_MARGIN_DEG",
           "STENCIL_PAD_CELLS", "CropBox", "CropPlan", "declared_crop",
           "crops_at_preparation", "declared_crop_stage", "box_from_points",
           "box_from_area", "domain_box", "declared_area", "crop_box_for",
           "plan_crop", "read_axes", "crop_object", "crop_inputs",
           "crop_packaged_inputs"]
