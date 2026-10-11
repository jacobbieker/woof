"""The Met Office global deterministic 10 km model onto WPS intermediates:
``woof hex intermediate --source met-office-global``.

THE SOURCE.  The Met Office publishes its global Unified Model run on AWS
open data (``registry.opendata.aws/met-office-global-deterministic``) as one
CF/Iris NetCDF-4 file per variable per lead::

    global-deterministic-10km/<RUN YYYYMMDDTHHMMZ>/<VALID YYYYMMDDTHHMMZ>-PT<HHHH>H00M-<variable>.nc

on a regular 2560 x 1920 latitude-longitude grid (0.140625 x 0.09375 deg,
cell centres from -179.93 E and -89.95 N), with scalar time coordinates.
The valid time comes first in the file name.  ``woof fetch --source
met-office-global`` lays the files out under those names; this door reads
them from ``--input-dir`` (flat, or under a ``<RUN>/`` subdirectory).

WHY ITS OWN DOOR.  The rows of :mod:`woof.hex.met_intermediate_door` are
GRIB decoded by ``met_intermediate``; these files are NetCDF, wind comes as
speed and direction, and the product carries no soil, no land-sea mask and
no sea ice.  So, like the wrfout door (:mod:`woof.hex.wrfout_intermediate`),
Python reads and converts and :func:`woof.hex.hrrr_intermediate.wps_record_bytes`
writes, and the missing land state is a declared supplement from the
same-cycle NCEP GDAS (or GFS) run through ``met_intermediate``.

THE ROAD.

* CROP -- the box is the cull's: ``--mesh-grid`` (a culled MPAS grid, static
  or init file: the extent of its cells and vertices), ``--from-plan``,
  ``--cull-region`` with ``--halo-km``, or ``--point`` with ``--radius-km``,
  each widened by ``--margin-km`` and then by :data:`HALO_CELLS` source cells
  for the init's 16-point stencil.  Only that window is read (``netCDF4``
  hyperslabs over the 128 x 128 chunks), one pressure level at a time, so a
  global file is never held whole.  The grid is the source's own: nothing
  is regridded, and a crop across the antimeridian is read in two pieces.
* CONVERT, vectorised over the window --
  ``wind_speed/wind_direction_on_pressure_levels`` -> ``UU``/``VV`` with the
  meteorological "from" convention (``u = -s sin d``, ``v = -s cos d``);
  ``geopotential_height_on_pressure_levels`` (``height_ASL_on_pressure_levels``
  before the January 2026 rename) -> ``GHT``; temperature -> ``TT``;
  relative humidity (a fraction) -> ``RH`` in percent;
  ``pressure_at_mean_sea_level`` -> ``PMSL``; ``temperature_at_surface`` ->
  ``SKINTEMP``; ``height_of_orography`` (published at step 0 only) ->
  ``SOILHGT``; ``snow_depth_water_equivalent`` -> ``SNOW``; screen (1.5 m)
  temperature and humidity -> the 2 m surface record (level 200100), the
  height recorded in the receipt; 10 m speed and direction -> surface
  ``UU``/``VV``.
* LEVELS -- relative humidity is published on 33 levels (to 10 hPa) and the
  rest on 37 (to 0.4 hPa).  Only levels where all of TT, GHT, RH, UU and VV
  exist are written, so the column tops at 10 hPa (about 31 km, above the
  30 km hex model top) and the levels above are named in the receipt.
* SUPPLEMENT -- ``met_intermediate --map-source ncep-gfs`` over the
  same-cycle GDAS (``--supplement-source gfs`` for leads past GDAS's f009)
  with a subset of the packaged ``Vtable.GFS.rw`` that keeps only the soil
  temperature and moisture layers, ``LANDSEA`` and ``SEAICE``.  Those records
  are cropped to the same box on their own 0.25-degree grid and appended to
  the file.  ``rw_mpas_init`` reads every record on its own projection
  (``init/metfile.rs``) and masks soil and sea ice with ``LANDSEA`` index by
  index, so the masked fields and the mask must share one grid: they all
  come from the supplement, and validation checks it.
* SURFACE PRESSURE -- not written.  ``rw_mpas_init`` interpolates the
  isobaric column (whose below-ground levels the producer extrapolates and
  flags) to the model's lowest level, and ``rw_mpas_lbc`` reads no PSFC.
* VALIDATE -- every file is read back through this tree's reader and the
  init door's met check (:func:`woof.hex.init_door.scan_met_file`) via
  :func:`woof.hex.met_intermediate_door.validate_intermediate`, whose
  single-grid rule is relaxed for the declared supplement records only.
  The init engine silently zeroes a missing PMSL, SKINTEMP, SNOW or SEAICE,
  so their presence is checked here.

LICENCE.  The Met Office data are British Crown copyright, released under CC
BY-SA 4.0; the share-alike term applies to derived outputs, so the receipt
carries the attribution beside the GDAS one.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA = "gpuwm-hex.metoffice-intermediate/v1"
SOURCE_NAME = "met-office-global"
#: The record header's 32-character producing-centre text.
MAP_SOURCE = "Met Office UM Global 10km"
LICENCE = ("British Crown copyright, the Met Office, CC BY-SA 4.0 "
           "(share-alike applies to derived outputs)")
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
DATA_URL = "https://registry.opendata.aws/met-office-global-deterministic/"
S3_BASE = ("https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com/"
           "global-deterministic-10km")
SUPPLEMENT_ATTRIBUTION = (
    "NOAA/NCEP GDAS/GFS 0.25-degree analysis and forecast (pgrb2.0p25), a U.S. "
    "Government work distributed by NOAA NOMADS/NODD without restriction"
)
SURFACE_LEVEL = 200100.0
MSL_LEVEL = 201300.0
#: Source cells kept beyond the box on every side: the init's 16-point
#: stencil reaches one cell before the donor and two after it, plus one.
HALO_CELLS = 3
#: Every field the supplement contributes, and nothing else.
SUPPLEMENT_SOIL = ("ST000010", "ST010040", "ST040100", "ST100200",
                   "SM000010", "SM010040", "SM040100", "SM100200")
SUPPLEMENT_FIELDS = frozenset({"LANDSEA", "SEAICE", *SUPPLEMENT_SOIL})
#: Fields ``rw_mpas_init`` zero-fills without a word when they are absent.
SILENTLY_ZEROED = ("PMSL", "SKINTEMP", "SNOW", "SEAICE")
COORDINATE_TOLERANCE_DEG = 1.0e-4


def _refusal() -> type:
    from .met_intermediate_door import GlobalIntermediateRefusal

    return GlobalIntermediateRefusal


def refuse(message: str) -> None:
    raise _refusal()(message)


# ---------------------------------------------------------------------------
# the source row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NetcdfSource:
    """A per-variable NetCDF source this module reads directly."""

    name: str
    description: str
    #: Leads published, by cycle hour.
    max_hour_by_cycle_hour: Mapping[int, int]
    default_interval_hours: int
    #: The init switch the published moisture implies (read by init_switches).
    use_spechumd: str = "no"
    notes: str = ""


METOFFICE_SOURCES: Mapping[str, NetcdfSource] = {
    SOURCE_NAME: NetcdfSource(
        name=SOURCE_NAME,
        description="Met Office global deterministic 10 km (Unified Model; AWS open "
                    "data, one NetCDF-4 file per variable per lead, 2560 x 1920 "
                    "regular lat-lon)",
        max_hour_by_cycle_hour={0: 168, 6: 69, 12: 168, 18: 69},
        default_interval_hours=3,
        notes="no soil, land-sea mask or sea ice is published: those records are a "
              "declared same-cycle GDAS/GFS supplement on its own grid",
    ),
}


@dataclass(frozen=True)
class Variable:
    """One published variable: the file-name spellings it has had, in order."""

    key: str
    names: tuple[str, ...]
    #: Published units -> the factor onto the WPS record's units.
    units: Mapping[str, float]
    levels: bool = False
    #: Published at step 0 only; later leads read the step-0 file.
    step_zero_only: bool = False


_K = {"K": 1.0}
_FRACTION_TO_PERCENT = {"1": 100.0, "%": 1.0}
_SPEED = {"m s-1": 1.0}
_DEGREES = {"degrees": 1.0, "degree": 1.0}
#: Snow water equivalent: a liquid-water-equivalent thickness in metres is
#: 1000 kg m-2 per metre (the Met Office publishes
#: lwe_thickness_of_surface_snow_amount in m); open_field's standard-name
#: check keeps a physical depth out of this factor.
_SWE = {"kg m-2": 1.0, "m": 1000.0}

VARIABLES: tuple[Variable, ...] = (
    Variable("T", ("temperature_on_pressure_levels",), _K, levels=True),
    Variable("Z", ("geopotential_height_on_pressure_levels", "height_ASL_on_pressure_levels"),
             {"m": 1.0}, levels=True),
    Variable("RH", ("relative_humidity_on_pressure_levels",), _FRACTION_TO_PERCENT, levels=True),
    Variable("WS", ("wind_speed_on_pressure_levels",), _SPEED, levels=True),
    Variable("WD", ("wind_direction_on_pressure_levels",), _DEGREES, levels=True),
    Variable("PMSL", ("pressure_at_mean_sea_level",), {"Pa": 1.0, "hPa": 100.0}),
    Variable("TSK", ("temperature_at_surface",), _K),
    Variable("OROG", ("height_of_orography",), {"m": 1.0}, step_zero_only=True),
    Variable("SNOW", ("snow_depth_water_equivalent",), _SWE),
    Variable("T15", ("temperature_at_screen_level",), _K),
    Variable("RH15", ("relative_humidity_at_screen_level",), _FRACTION_TO_PERCENT),
    Variable("WS10", ("wind_speed_at_10m",), _SPEED),
    Variable("WD10", ("wind_direction_at_10m",), _DEGREES),
)
VARIABLE_BY_KEY = {variable.key: variable for variable in VARIABLES}


def file_name(cycle: datetime, lead: int, name: str) -> str:
    valid = cycle + timedelta(hours=int(lead))
    return f"{valid:%Y%m%dT%H%MZ}-PT{int(lead):04d}H00M-{name}.nc"


# ---------------------------------------------------------------------------
# finding the files
# ---------------------------------------------------------------------------
def _roots(input_dir: Path, cycle: datetime) -> list[Path]:
    root = Path(input_dir).expanduser()
    return [root, root / f"{cycle:%Y%m%dT%H%MZ}"]


def _hits(input_dir: Path, cycle: datetime, lead: int, variable: Variable) -> list[Path]:
    found: list[Path] = []
    for root in _roots(input_dir, cycle):
        for name in variable.names:
            candidate = root / file_name(cycle, lead, name)
            if candidate.is_file() and candidate.resolve() not in {p.resolve() for p in found}:
                found.append(candidate)
    return found


def locate_files(input_dir: Path, cycle: datetime, leads: Sequence[int]) -> dict[int, dict[str, Path]]:
    """Every lead's file per variable, or one refusal naming every gap."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        refuse(f"--input-dir {root} is not a directory")
    missing: list[str] = []
    ambiguous: list[str] = []
    found: dict[int, dict[str, Path]] = {}
    for lead in leads:
        per: dict[str, Path] = {}
        for variable in VARIABLES:
            hits = _hits(root, cycle, lead, variable)
            if not hits and variable.step_zero_only and lead != 0:
                hits = _hits(root, cycle, 0, variable)
            if not hits:
                lead_for_name = 0 if variable.step_zero_only else lead
                missing.append(file_name(cycle, lead_for_name, variable.names[0]))
            elif len(hits) > 1:
                ambiguous.append(", ".join(str(p) for p in hits))
            else:
                per[variable.key] = hits[0]
        found[lead] = per
    if ambiguous:
        refuse(
            f"{root} holds more than one file for one variable and lead ({'; '.join(ambiguous)}); "
            f"two spellings or two copies cannot both be the input, so move one aside"
        )
    if missing:
        unique = list(dict.fromkeys(missing))
        refuse(
            f"{root} is missing {len(unique)} Met Office file(s) for the "
            f"{cycle:%Y-%m-%dT%H}Z run: {', '.join(unique[:8])}"
            f"{', ...' if len(unique) > 8 else ''}.  `woof fetch --source {SOURCE_NAME} "
            f"--cycle {cycle:%Y-%m-%dT%H} ... --out {root}` writes them under these names, "
            f"the object names of {S3_BASE}/{cycle:%Y%m%dT%H%MZ}/"
        )
    return found


# ---------------------------------------------------------------------------
# the grid and the crop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegularGrid:
    """A regular lat-lon grid, south row first, x fastest."""

    south: float
    west: float
    dlat: float
    dlon: float
    nx: int
    ny: int

    @property
    def north(self) -> float:
        return self.south + self.dlat * (self.ny - 1)

    @property
    def periodic(self) -> bool:
        return abs(self.nx * self.dlon - 360.0) < 0.5 * self.dlon

    def as_dict(self) -> dict[str, Any]:
        return {"south": self.south, "west": self.west, "north": self.north,
                "dlat": self.dlat, "dlon": self.dlon, "nx": self.nx, "ny": self.ny,
                "periodic_in_longitude": self.periodic}


@dataclass(frozen=True)
class Crop:
    """The rows and columns of a source grid the box needs."""

    grid: RegularGrid
    j0: int
    j1: int
    columns: np.ndarray
    halo_cells: int

    @property
    def nx(self) -> int:
        return int(self.columns.size)

    @property
    def ny(self) -> int:
        return self.j1 - self.j0 + 1

    def segments(self) -> list[slice]:
        """Contiguous column runs: one, or two when the crop crosses the seam."""

        breaks = np.flatnonzero(np.diff(self.columns) != 1) + 1
        return [slice(int(run[0]), int(run[-1]) + 1) for run in np.split(self.columns, breaks)]

    def target(self) -> Any:
        from .hrrr_intermediate import LatLonTarget

        west = self.grid.west + self.grid.dlon * float(self.columns[0])
        west = (west + 180.0) % 360.0 - 180.0
        return LatLonTarget(
            south=self.grid.south + self.grid.dlat * self.j0, west=west,
            dlat=self.grid.dlat, dlon=self.grid.dlon, nx=self.nx, ny=self.ny,
        )

    def take(self, slab: np.ndarray) -> np.ndarray:
        """The crop of a whole ``(ny, nx)`` slab already in memory."""

        return np.ascontiguousarray(slab[self.j0:self.j1 + 1][:, self.columns])

    def as_dict(self) -> dict[str, Any]:
        target = self.target()
        return {
            "rows_zero_based_inclusive": [self.j0, self.j1],
            "columns_zero_based": [int(self.columns[0]), int(self.columns[-1])],
            "crosses_the_grid_seam": len(self.segments()) > 1,
            "shape": [self.ny, self.nx], "halo_cells": self.halo_cells,
            "target": target.as_dict(),
        }


def crop_for(grid: RegularGrid, bounds: tuple[float, float, float, float], *,
             halo_cells: int, what: str) -> Crop:
    """The window of ``grid`` covering ``bounds`` (south, west, north, east)
    plus ``halo_cells`` on every side, or a refusal naming the shortfall."""

    south, west, north, east = (float(v) for v in bounds)
    if not (north > south and east > west):
        refuse(f"the box {south:g}..{north:g} N, {west:g}..{east:g} E is empty")
    j0 = int(math.floor((south - grid.south) / grid.dlat + 1e-9)) - halo_cells
    j1 = int(math.ceil((north - grid.south) / grid.dlat - 1e-9)) + halo_cells
    if j0 < 0:
        if grid.south - grid.dlat > -90.0 + 1e-6:
            refuse(f"the box reaches {south:.3f} N and the {what} grid starts at "
                   f"{grid.south:.3f} N: it does not cover the box plus its "
                   f"{halo_cells}-cell halo")
        j0 = 0
    if j1 > grid.ny - 1:
        if grid.north + grid.dlat < 90.0 - 1e-6:
            refuse(f"the box reaches {north:.3f} N and the {what} grid ends at "
                   f"{grid.north:.3f} N: it does not cover the box plus its "
                   f"{halo_cells}-cell halo")
        j1 = grid.ny - 1
    span = east - west
    if grid.periodic and span + 2.0 * (halo_cells + 1) * grid.dlon >= 360.0:
        columns = np.arange(grid.nx, dtype=np.int64)
    else:
        offset = ((west - grid.west) % 360.0) / grid.dlon
        i0 = int(math.floor(offset + 1e-9)) - halo_cells
        width = int(math.ceil(span / grid.dlon + (offset - math.floor(offset + 1e-9)) - 1e-9)) \
            + 2 * halo_cells + 1
        columns = i0 + np.arange(width, dtype=np.int64)
        if grid.periodic:
            columns %= grid.nx
        elif columns.min() < 0 or columns.max() > grid.nx - 1:
            refuse(
                f"the box {west:.3f}..{east:.3f} E plus its {halo_cells}-cell halo leaves "
                f"the {what} grid ({grid.west:.3f}..{grid.west + grid.dlon * (grid.nx - 1):.3f} E); "
                f"fetch an area that covers it"
            )
    return Crop(grid=grid, j0=j0, j1=j1, columns=columns, halo_cells=halo_cells)


def mesh_bounds(path: Path, margin_km: float) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    """The lat-lon box of an MPAS mesh's cells and vertices, plus a margin."""

    from .wrfout_intermediate import _netcdf
    from .hrrr_intermediate import KM_PER_DEG

    netCDF4 = _netcdf()
    path = Path(path).expanduser()
    if not path.is_file():
        refuse(f"--mesh-grid {path} is not a file")
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    try:
        with netCDF4.Dataset(str(path), "r") as dataset:
            dataset.set_auto_mask(False)
            for lat_name, lon_name in (("latCell", "lonCell"), ("latVertex", "lonVertex")):
                if lat_name in dataset.variables and lon_name in dataset.variables:
                    lats.append(np.degrees(np.asarray(dataset.variables[lat_name][:], dtype=np.float64)))
                    lons.append(np.degrees(np.asarray(dataset.variables[lon_name][:], dtype=np.float64)))
    except OSError as error:
        refuse(f"--mesh-grid {path} is not a readable netCDF file: {error}")
    if not lats:
        refuse(f"--mesh-grid {path} carries no latCell/lonCell; it is not an MPAS mesh file")
    lat = np.concatenate(lats)
    lon = np.concatenate(lons) % 360.0
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        refuse(f"--mesh-grid {path}: the cell coordinates carry non-finite values")
    # The shortest longitude arc covering every point: the complement of
    # the widest empty gap between consecutive sorted longitudes.
    ordered = np.unique(lon)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    widest = int(np.argmax(gaps))
    west = float(ordered[(widest + 1) % ordered.size])
    east = float(ordered[widest])
    if east < west:
        east += 360.0
    if gaps[widest] < 1.0:
        # Every longitude is occupied (a polar cap, or a global mesh): the
        # window is the whole circle, and crop_for reads every column.
        west, east = -180.0, 180.0
    half_lat = float(margin_km) / KM_PER_DEG
    south, north = float(lat.min()) - half_lat, float(lat.max()) + half_lat
    cos_lat = max(math.cos(math.radians(max(abs(south), abs(north)))), 0.05)
    half_lon = float(margin_km) / (KM_PER_DEG * cos_lat)
    if west > 180.0:
        west -= 360.0
        east -= 360.0
    bounds = (max(south, -90.0), west - half_lon, min(north, 90.0), east + half_lon)
    return bounds, {"mesh_grid": str(path), "points": int(lat.size),
                    "mesh_extent_deg": [float(lat.min()), west, float(lat.max()), east]}


# ---------------------------------------------------------------------------
# reading one file
# ---------------------------------------------------------------------------
_EPOCH = datetime(1970, 1, 1)


def _scalar_time(dataset: Any, name: str, path: Path) -> datetime | float:
    if name not in dataset.variables:
        refuse(f"{path} carries no scalar {name}; it is not a Met Office CF/Iris file")
    variable = dataset.variables[name]
    value = np.asarray(variable[...]).reshape(-1)
    if value.size != 1:
        refuse(f"{path}: {name} is not a scalar coordinate (it has {value.size} values); "
               f"one file is one time")
    units = str(getattr(variable, "units", ""))
    if name == "forecast_period":
        if units not in ("seconds", "s"):
            refuse(f"{path}: forecast_period is in {units!r}, not seconds")
        return float(value[0])
    if not units.startswith("seconds since 1970-01-01"):
        refuse(f"{path}: {name} is in {units!r}, not seconds since 1970-01-01")
    return _EPOCH + timedelta(seconds=int(value[0]))


def data_variable(dataset: Any, path: Path) -> Any:
    """The one variable carrying a grid mapping: the field itself."""

    candidates = [v for v in dataset.variables.values()
                  if hasattr(v, "grid_mapping") and v.ndim >= 2]
    if len(candidates) != 1:
        refuse(f"{path} carries {len(candidates)} gridded data variables "
               f"({', '.join(v.name for v in candidates) or 'none'}); a Met Office file "
               f"carries exactly one")
    return candidates[0]


def grid_of(dataset: Any, path: Path) -> RegularGrid:
    for name in ("latitude", "longitude"):
        if name not in dataset.variables:
            refuse(f"{path} carries no {name} coordinate")
    lat = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
    lon = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
    if lat.size < 2 or lon.size < 2:
        refuse(f"{path}: the latitude/longitude axes are degenerate")
    dlat = (lat[-1] - lat[0]) / (lat.size - 1)
    dlon = (lon[-1] - lon[0]) / (lon.size - 1)
    if not dlat > 0.0:
        refuse(f"{path}: latitude does not ascend; the door reads the south-first grid "
               f"the Met Office publishes")
    lat_error = float(np.max(np.abs(lat - (lat[0] + dlat * np.arange(lat.size)))))
    lon_error = float(np.max(np.abs(lon - (lon[0] + dlon * np.arange(lon.size)))))
    if lat_error > COORDINATE_TOLERANCE_DEG or lon_error > COORDINATE_TOLERANCE_DEG:
        refuse(f"{path}: the grid is not regular (latitude off by {lat_error:.3g} deg, "
               f"longitude by {lon_error:.3g} deg); a WPS lat-lon record cannot carry it")
    return RegularGrid(south=float(lat[0]), west=float(lon[0]), dlat=float(dlat),
                       dlon=float(dlon), nx=int(lon.size), ny=int(lat.size))


def _same_grid(a: RegularGrid, b: RegularGrid) -> bool:
    return (a.nx, a.ny) == (b.nx, b.ny) and all(
        abs(x - y) <= COORDINATE_TOLERANCE_DEG
        for x, y in ((a.south, b.south), (a.west, b.west), (a.dlat, b.dlat), (a.dlon, b.dlon)))


@dataclass
class OpenField:
    """One opened file: identity checked, the data variable and its levels."""

    variable: Variable
    path: Path
    dataset: Any
    data: Any
    units: str
    levels_pa: tuple[float, ...] | None
    height_m: float | None

    @property
    def scale(self) -> float:
        return float(self.variable.units[self.units])

    def slab(self, crop: Crop, level_index: int | None = None) -> np.ndarray:
        """The cropped ``(ny, nx)`` slab, read as one or two hyperslabs."""

        rows = slice(crop.j0, crop.j1 + 1)
        pieces = []
        try:
            for columns in crop.segments():
                if level_index is None:
                    pieces.append(np.asarray(self.data[rows, columns], dtype=np.float64))
                else:
                    pieces.append(np.asarray(self.data[level_index, rows, columns], dtype=np.float64))
        except (RuntimeError, OSError) as error:
            refuse(f"{self.path}: reading {self.data.name} failed ({error}); a truncated or "
                   f"corrupt download is refused, fetch the file again")
        values = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        fill = [getattr(self.data, name) for name in ("_FillValue", "missing_value")
                if hasattr(self.data, name)]
        bad = ~np.isfinite(values)
        for marker in fill:
            bad |= values == float(np.asarray(marker).reshape(-1)[0])
        if bad.any():
            where = "" if level_index is None else f" at {self.levels_pa[level_index]:g} Pa"
            refuse(f"{self.path}: {int(bad.sum())} missing or non-finite value(s){where} inside "
                   f"the crop; the door writes no record with a hole in it")
        return values * self.scale if self.scale != 1.0 else values


def open_field(variable: Variable, path: Path, *, cycle: datetime, valid: datetime,
               lead: int) -> OpenField:
    from .wrfout_intermediate import _netcdf

    netCDF4 = _netcdf()
    try:
        dataset = netCDF4.Dataset(str(path), "r")
    except OSError as error:
        refuse(f"{path} is not a readable NetCDF file: {error}")
    dataset.set_auto_mask(False)
    try:
        reference = _scalar_time(dataset, "forecast_reference_time", path)
        moment = _scalar_time(dataset, "time", path)
        period = _scalar_time(dataset, "forecast_period", path)
        expect_valid = cycle if variable.step_zero_only and moment == cycle else valid
        expect_period = 0.0 if expect_valid == cycle else lead * 3600.0
        if reference != cycle or moment != expect_valid or abs(float(period) - expect_period) > 0.5:
            refuse(
                f"{path}: its own coordinates say run {reference}, valid {moment}, lead "
                f"{float(period) / 3600.0:g} h; the file name and the request say run "
                f"{cycle}, valid {expect_valid}.  A renamed or mis-filed file is refused"
            )
        data = data_variable(dataset, path)
        units = str(getattr(data, "units", ""))
        if units not in variable.units:
            refuse(f"{path}: {data.name} is in {units!r}; the door converts "
                   f"{' or '.join(repr(u) for u in variable.units)}")
        standard = str(getattr(data, "standard_name", ""))
        if variable.key == "SNOW" and units == "m" and not standard.startswith("lwe_thickness"):
            refuse(f"{path}: {data.name} is in m with standard_name {standard!r}; only a "
                   f"liquid-water-equivalent thickness (lwe_thickness_...) converts to "
                   f"kg m-2 by 1000, a physical snow depth does not")
        levels: tuple[float, ...] | None = None
        if variable.levels:
            if data.ndim != 3 or data.dimensions[0] not in dataset.variables:
                refuse(f"{path}: {data.name} has dimensions {data.dimensions}; a "
                       f"pressure-level file is (pressure, latitude, longitude)")
            axis = dataset.variables[data.dimensions[0]]
            scale = {"Pa": 1.0, "hPa": 100.0}.get(str(getattr(axis, "units", "")))
            if scale is None:
                refuse(f"{path}: the vertical axis {axis.name} is in "
                       f"{getattr(axis, 'units', None)!r}, not Pa or hPa")
            levels = tuple(float(v) * scale for v in np.asarray(axis[:], dtype=np.float64))
        elif data.ndim != 2:
            refuse(f"{path}: {data.name} has dimensions {data.dimensions}; a surface field "
                   f"is (latitude, longitude)")
        height = None
        if "height" in dataset.variables:
            height = float(np.asarray(dataset.variables["height"][...]).reshape(-1)[0])
        return OpenField(variable=variable, path=path, dataset=dataset, data=data,
                         units=units, levels_pa=levels, height_m=height)
    except BaseException:
        dataset.close()
        raise


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def wind_components(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Earth-relative u, v from speed and the direction the wind blows FROM."""

    radians = np.radians(direction_deg)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def _floor_zero(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _record(name: str, units: str, description: str, level: float, values: np.ndarray) -> dict[str, Any]:
    return {"field": name, "units": units, "description": description, "level": float(level),
            "values": np.asarray(values, dtype=np.float32)}


def common_levels(opened: Mapping[str, OpenField]) -> tuple[list[float], dict[str, list[float]]]:
    """Pressure levels every profile variable carries, top-down sorted, and
    what each variable carries beyond them."""

    sets = {key: set(opened[key].levels_pa or ()) for key in ("T", "Z", "RH", "WS", "WD")}
    shared = set.intersection(*sets.values())
    if len(shared) < 2:
        refuse(f"the profile files share {len(shared)} pressure level(s); a column needs two")
    dropped = {key: sorted(levels - shared, reverse=True) for key, levels in sets.items()
               if levels - shared}
    return sorted(shared, reverse=True), dropped


def lead_records(opened: Mapping[str, OpenField], crop: Crop, levels: Sequence[float]) -> Iterator[dict[str, Any]]:
    """The Met Office records of one lead, one pressure level at a time."""

    index = {key: {p: i for i, p in enumerate(opened[key].levels_pa or ())}
             for key in ("T", "Z", "RH", "WS", "WD")}
    for pressure in levels:
        u, v = wind_components(opened["WS"].slab(crop, index["WS"][pressure]),
                               opened["WD"].slab(crop, index["WD"][pressure]))
        yield _record("GHT", "m", "Height", pressure, opened["Z"].slab(crop, index["Z"][pressure]))
        yield _record("TT", "K", "Temperature", pressure, opened["T"].slab(crop, index["T"][pressure]))
        yield _record("RH", "%", "Relative Humidity", pressure,
                      _floor_zero(opened["RH"].slab(crop, index["RH"][pressure])))
        yield _record("UU", "m s-1", "U", pressure, u)
        yield _record("VV", "m s-1", "V", pressure, v)
    screen = opened["T15"].height_m
    yield _record("TT", "K", f"Temperature at {screen:g} m (screen level)", SURFACE_LEVEL,
                  opened["T15"].slab(crop))
    yield _record("RH", "%", f"Relative Humidity at {screen:g} m (screen)", SURFACE_LEVEL,
                  _floor_zero(opened["RH15"].slab(crop)))
    u10, v10 = wind_components(opened["WS10"].slab(crop), opened["WD10"].slab(crop))
    yield _record("UU", "m s-1", "U at 10 m", SURFACE_LEVEL, u10)
    yield _record("VV", "m s-1", "V at 10 m", SURFACE_LEVEL, v10)
    yield _record("PMSL", "Pa", "Sea-level Pressure", MSL_LEVEL, opened["PMSL"].slab(crop))
    yield _record("SKINTEMP", "K", "Skin temperature", SURFACE_LEVEL, opened["TSK"].slab(crop))
    yield _record("SOILHGT", "m", "Terrain field of source analysis", SURFACE_LEVEL,
                  opened["OROG"].slab(crop))
    yield _record("SNOW", "kg m-2", "Water equivalent snow depth", SURFACE_LEVEL,
                  _floor_zero(opened["SNOW"].slab(crop)))


def check_physical(record: Mapping[str, Any]) -> None:
    """Plausibility windows no real value approaches."""

    windows = {"TT": (150.0, 350.0), "SKINTEMP": (150.0, 400.0), "PMSL": (8.0e4, 1.2e5),
               "RH": (0.0, 150.0), "UU": (-200.0, 200.0), "VV": (-200.0, 200.0),
               "SNOW": (0.0, 1.0e6)}
    window = windows.get(str(record["field"]))
    if window is None:
        return
    values = record["values"]
    low, high = float(values.min()), float(values.max())
    if low < window[0] or high > window[1]:
        refuse(f"{record['field']} at level {record['level']:g} spans {low:g}..{high:g}, "
               f"outside {window[0]:g}..{window[1]:g}; that is not a field this door converts "
               f"(a unit change upstream?)")


# ---------------------------------------------------------------------------
# the same-cycle GDAS/GFS supplement
# ---------------------------------------------------------------------------
def supplement_vtable_text(packaged: str) -> str:
    """``Vtable.GFS.rw`` cut to the supplement's rows: soil, LANDSEA, SEAICE."""

    out: list[str] = []
    kept: set[str] = set()
    for line in packaged.splitlines():
        cells = line.split("|")
        is_row = len(cells) >= 7 and not line.lstrip().startswith(("#", "-"))
        name = cells[4].strip() if is_row else ""
        if is_row and name not in ("metgrid", "Name"):
            if name not in SUPPLEMENT_FIELDS:
                continue
            kept.add(name)
        out.append(line)
    if kept != SUPPLEMENT_FIELDS:
        refuse(f"the packaged Vtable.GFS.rw has no row for {sorted(SUPPLEMENT_FIELDS - kept)}; "
               f"the supplement cannot be cut from it")
    out.append("#  woof hex intermediate --source met-office-global: the same-cycle")
    out.append("#  GDAS/GFS supplement subset (soil layers, LANDSEA, SEAICE) of Vtable.GFS.rw.")
    return "\n".join(out) + "\n"


def grid_of_record(record: Any) -> RegularGrid:
    projection = record.projection
    if projection.name != "latlon":
        refuse(f"the supplement record {record.field} is on a {projection.name} grid, not lat-lon")
    dlat = float(projection.delta_latitude)
    south = float(projection.start_latitude)
    if dlat < 0.0:
        south = south + dlat * (record.ny - 1)
        dlat = -dlat
    return RegularGrid(south=south, west=float(projection.start_longitude), dlat=dlat,
                       dlon=float(projection.delta_longitude), nx=int(record.nx), ny=int(record.ny))


def south_first(record: Any) -> np.ndarray:
    values = np.asarray(record.values, dtype=np.float32).T
    return values[::-1] if float(record.projection.delta_latitude) < 0.0 else values


@dataclass
class SupplementLead:
    records: list[dict[str, Any]]
    crop: Crop
    provenance: dict[str, Any]


def run_supplement(*, engine: Path, vtable: Path, source: Any, grib: Path, hdate: str,
                   lead: int, bounds: tuple[float, float, float, float],
                   work_dir: Path, halo_cells: int = HALO_CELLS) -> SupplementLead:
    """One lead's soil, LANDSEA and SEAICE from GDAS/GFS, cropped to the box."""

    from .met_intermediate_door import parse_engine_receipt
    from .wps_intermediate import WpsIntermediateReader

    scratch = work_dir / f".supplement.f{lead:03d}.partial"
    argv = [str(engine), "--vtable", str(vtable), "--date", hdate, "--map-source",
            source.map_source, "--out", str(scratch), str(grib)]
    started = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    log_path = work_dir / f"met_intermediate.supplement.f{lead:03d}.log"
    log_path.write_text(
        f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {done.returncode}]\n"
        f"--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}\n", encoding="utf-8")
    try:
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()
            refuse(f"met_intermediate exited {done.returncode} on the {source.name} supplement "
                   f"for f{lead:03d}; its log is {log_path}" + (f".  It said: {tail[-1]}" if tail else ""))
        receipt = parse_engine_receipt(done.stdout)
        if list(receipt.get("matched_valid_times") or []) != [hdate]:
            refuse(f"the {source.name} supplement {grib.name} matched messages valid at "
                   f"{receipt.get('matched_valid_times') or 'no time'}, not {hdate}; the "
                   f"supplement must be the same valid time as the Met Office lead")
        records: list[dict[str, Any]] = []
        crop: Crop | None = None
        seen: set[str] = set()
        with WpsIntermediateReader(scratch) as reader:
            for item in reader.iter_fields():
                if item.field not in SUPPLEMENT_FIELDS or item.field in seen:
                    refuse(f"the supplement wrote {item.field} "
                           f"{'twice' if item.field in seen else 'outside its declared set'}")
                seen.add(item.field)
                grid = grid_of_record(item)
                if crop is None:
                    crop = crop_for(grid, bounds, halo_cells=halo_cells, what=f"{source.name} supplement")
                elif not _same_grid(grid, crop.grid):
                    refuse(f"the supplement's {item.field} is on another grid from its "
                           f"{records[0]['field']}; the land-sea mask must share the grid of "
                           f"every field it masks")
                values = crop.take(south_first(item))
                if item.field == "LANDSEA":
                    values = np.where(values >= 0.5, np.float32(1.0), np.float32(0.0))
                records.append({"field": item.field, "units": item.units,
                                "description": item.description, "level": float(item.level),
                                "values": np.asarray(values, dtype=np.float32),
                                "map_source": item.map_source})
        missing = SUPPLEMENT_FIELDS - seen
        if missing:
            refuse(f"the {source.name} supplement {grib.name} carries no "
                   f"{', '.join(sorted(missing))}; the init needs every soil layer, the land-sea "
                   f"mask and sea ice")
        assert crop is not None
        provenance = {
            "lead": lead, "valid_time": hdate, "seconds": round(elapsed, 2),
            "log": str(log_path),
            "fields": sorted(seen), "crop": crop.as_dict(), "source_grid": crop.grid.as_dict(),
            "engine_receipt": {key: receipt.get(key) for key in (
                "schema", "map_source", "grib_messages_read", "grib_messages_matched",
                "records_written", "matched_valid_times", "bitmap_masked_points",
                "masked_point_value")},
        }
        return SupplementLead(records=records, crop=crop, provenance=provenance)
    finally:
        scratch.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_records(path: Path, groups: Iterable[tuple[Any, Iterable[Mapping[str, Any]]]], *,
                  hdate: str, lead: int) -> list[dict[str, Any]]:
    """Every record of every ``(target, records)`` group, each on its own grid,
    then read back through this tree's reader before anything is claimed."""

    from .hrrr_intermediate import wps_record_bytes
    from .wps_intermediate import inventory

    written: list[dict[str, Any]] = []
    with open(path, "wb") as handle:
        for target, records in groups:
            for item in records:
                check_physical(item)
                handle.write(wps_record_bytes(
                    hdate=hdate, xfcst=float(lead),
                    map_source=str(item.get("map_source") or MAP_SOURCE),
                    field_name=str(item["field"]), units=str(item["units"]),
                    description=str(item["description"]), level=float(item["level"]),
                    target=target, values=item["values"],
                ))
                written.append({"field": str(item["field"]), "level": float(item["level"]),
                                "nx": target.nx, "ny": target.ny})
    seen = inventory(path)
    if int(seen["field_records"]) != len(written):
        refuse(f"{path} reads back as {seen['field_records']} records after {len(written)} "
               f"were written; the writer and the reader disagree")
    for got, want in zip(seen["records"], written):
        if (got["field"], int(got["nx"]), int(got["ny"])) != (want["field"], want["nx"], want["ny"]) \
                or abs(float(got["level"]) - want["level"]) > 1e-3:
            refuse(f"{path}: record {want['field']}@{want['level']:g} read back as "
                   f"{got['field']}@{got['level']:g} ({got['nx']}x{got['ny']})")
    return written


def require_complete(written: Sequence[Mapping[str, Any]], levels: Sequence[float], path: Path) -> None:
    """Everything the init and boundary engines read, present."""

    have = {(item["field"], float(item["level"])) for item in written}
    names = {item["field"] for item in written}
    gaps = [name for name in (*SILENTLY_ZEROED, "SOILHGT", "LANDSEA") if name not in names]
    gaps += [f"{name}@200100" for name in ("TT", "RH", "UU", "VV") if (name, SURFACE_LEVEL) not in have]
    for pressure in levels:
        gaps += [f"{name}@{pressure:g}" for name in ("GHT", "TT", "RH", "UU", "VV")
                 if (name, float(pressure)) not in have]
    if gaps:
        refuse(f"{path.name} lacks {', '.join(gaps[:10])}{', ...' if len(gaps) > 10 else ''}; "
               f"rw_mpas_init zero-fills a missing PMSL/SKINTEMP/SNOW/SEAICE without a word, "
               f"so the door refuses instead")


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
@dataclass
class MetOfficeRequest:
    source: NetcdfSource
    input_dir: Path
    cycle: datetime
    leads: tuple[int, ...]
    interval_hours: int
    bounds: tuple[float, float, float, float]
    out_dir: Path
    supplement_dir: Path
    supplement_source: str = "gdas"
    prefix: str = "MET"
    engine: Path | None = None
    #: Concurrent met_intermediate supplement decodes.
    workers: int = 4
    target_basis: Mapping[str, Any] = field(default_factory=dict)


def _check_leads(source: NetcdfSource, cycle: datetime, leads: Sequence[int]) -> None:
    if cycle.minute or cycle.second:
        refuse(f"--cycle {cycle} is not on the hour; Met Office runs start at 00/06/12/18Z")
    horizon = source.max_hour_by_cycle_hour.get(cycle.hour)
    if horizon is None:
        refuse(f"--cycle {cycle:%Y-%m-%dT%H}Z is not a {source.name} run; it runs at "
               f"{', '.join(f'{h:02d}' for h in sorted(source.max_hour_by_cycle_hour))}Z")
    if leads[-1] > horizon:
        refuse(f"the {cycle:%H}Z {source.name} run publishes leads to PT{horizon:04d}H; "
               f"--hours asks for PT{leads[-1]:04d}H")


def locate_supplements(request: MetOfficeRequest) -> tuple[Any, dict[int, Path]]:
    from .met_intermediate_door import GLOBAL_SOURCES, GlobalRequest, locate_inputs

    key = str(request.supplement_source).strip().lower()
    if key not in ("gdas", "gfs"):
        refuse(f"--supplement-source {request.supplement_source!r} is not gdas or gfs; the "
               f"supplement is the same-cycle NCEP run read with Vtable.GFS.rw")
    source = GLOBAL_SOURCES[key]
    if source.max_hour is not None and request.leads[-1] > source.max_hour:
        refuse(f"the same-cycle {key} publishes leads to f{source.max_hour:03d} and --hours asks "
               f"for f{request.leads[-1]:03d}; pass --supplement-source gfs for a longer series")
    probe = GlobalRequest(source=source, grib_dir=Path(request.supplement_dir), cycle=request.cycle,
                          leads=tuple(request.leads), interval_hours=request.interval_hours,
                          out_dir=Path(request.out_dir))
    try:
        found = locate_inputs(probe)
    except _refusal() as error:
        refuse(f"the same-cycle {key} supplement: {error}")
    return source, {lead: hits[0].absolute() for lead, (hits, _inv) in found.items()}


def build_metoffice_intermediates(request: MetOfficeRequest, *,
                                  log: Callable[[str], None] = print) -> dict[str, Any]:
    from .met_intermediate_door import (
        LAPSE_RATE_TOP_PA,
        _described,
        _partial,
        init_switches,
        packaged_vtable,
        refuse_occupied_out_dir,
        resolve_engine,
        validate_intermediate,
    )

    source = request.source
    _check_leads(source, request.cycle, request.leads)
    out_dir = Path(request.out_dir).expanduser().absolute()
    supplement_source, supplement_files = locate_supplements(request)
    files_by_lead = locate_files(request.input_dir, request.cycle, request.leads)
    engine = resolve_engine(request.engine)
    refuse_occupied_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packaged = packaged_vtable(supplement_source.vtable)
    vtable = out_dir / "Vtable.GFS.supplement.rw"
    vtable.write_text(supplement_vtable_text(packaged.read_text(encoding="utf-8")), encoding="utf-8")
    log(f"{source.name}: {len(request.leads)} valid time(s) from the "
        f"{request.cycle:%Y-%m-%dT%H}Z run, soil/LANDSEA/SEAICE from the same-cycle "
        f"{supplement_source.name}")

    digests: dict[Path, dict[str, Any]] = {}
    planned: list[tuple[int, Path, dict[str, Any]]] = []
    grid0: RegularGrid | None = None
    levels0: list[float] | None = None
    dropped0: dict[str, list[float]] = {}
    screen_height = None
    workers = max(1, min(int(request.workers), len(request.leads)))

    def supplement_for(lead: int) -> tuple[int, SupplementLead]:
        valid = request.cycle + timedelta(hours=int(lead))
        return lead, run_supplement(
            engine=engine, vtable=vtable, source=supplement_source,
            grib=supplement_files[lead], hdate=valid.strftime("%Y-%m-%d_%H:%M:%S"),
            lead=lead, bounds=request.bounds, work_dir=out_dir)

    try:
        # The supplement decodes are independent subprocesses: run them
        # together first, then read the Met Office files lead by lead.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            supplements = dict(pool.map(supplement_for, request.leads))
        for lead in request.leads:
            started = time.perf_counter()
            valid = request.cycle + timedelta(hours=int(lead))
            hdate = valid.strftime("%Y-%m-%d_%H:%M:%S")
            out = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
            opened: dict[str, OpenField] = {}
            try:
                for key, path in files_by_lead[lead].items():
                    opened[key] = open_field(VARIABLE_BY_KEY[key], path, cycle=request.cycle,
                                             valid=valid, lead=lead)
                grids = {key: grid_of(item.dataset, item.path) for key, item in opened.items()}
                grid = grids["T"]
                strays = [opened[k].path.name for k, g in grids.items() if not _same_grid(g, grid)]
                if strays or (grid0 is not None and not _same_grid(grid, grid0)):
                    refuse(f"the f{lead:03d} files are not on one grid "
                           f"({', '.join(strays) or 'differs from the first lead'}); one series is "
                           f"one source grid")
                grid0 = grid0 or grid
                levels, dropped = common_levels(opened)
                if levels0 is not None and levels != levels0:
                    refuse(f"f{lead:03d} shares pressure levels {levels} and the first lead "
                           f"{levels0}; --nfglevels is one number for the series")
                levels0, dropped0 = levels, dropped
                for key, expected in (("T15", 1.5), ("RH15", 1.5), ("WS10", 10.0), ("WD10", 10.0)):
                    height = opened[key].height_m
                    if height is None or abs(height - expected) > 0.01:
                        refuse(f"{opened[key].path}: its height coordinate is {height!r} m, not "
                               f"the {expected:g} m the record description declares")
                screen_height = opened["T15"].height_m
                crop = crop_for(grid, request.bounds, halo_cells=HALO_CELLS, what=source.name)
                supplement = supplements[lead]
                written = write_records(_partial(out), [
                    (crop.target(), lead_records(opened, crop, levels)),
                    (supplement.crop.target(), supplement.records),
                ], hdate=hdate, lead=lead)
            finally:
                for item in opened.values():
                    item.dataset.close()
            require_complete(written, levels, out)
            checked = validate_intermediate(_partial(out), hdate=hdate,
                                            supplement_fields=SUPPLEMENT_FIELDS)
            checked["lead"] = lead
            checked["seconds"] = round(time.perf_counter() - started, 2)
            checked["crop"] = crop.as_dict()
            checked["inputs"] = {key: _described(path, digests)
                                 for key, path in sorted(files_by_lead[lead].items())}
            checked["orography_from"] = (
                "this lead" if f"PT{lead:04d}H" in files_by_lead[lead]["OROG"].name else
                "the step-0 file (the Met Office publishes height_of_orography at step 0 only)")
            checked["supplement"] = dict(supplement.provenance,
                                         input=_described(supplement_files[lead], digests))
            planned.append((lead, out, checked))
            log(f"WROTE {out.name}: {checked['field_records']} records "
                f"({len(written) - len(supplement.records)} Met Office on "
                f"{crop.ny}x{crop.nx}, {len(supplement.records)} {supplement_source.name} on "
                f"{supplement.crop.ny}x{supplement.crop.nx}), {checked['profile_levels']} "
                f"profile levels, {checked['seconds']} s")
        switches = init_switches(source, [item for _l, _o, item in planned])
    except BaseException:
        for lead in request.leads:
            valid = request.cycle + timedelta(hours=int(lead))
            _partial(out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}").unlink(missing_ok=True)
        raise
    files: list[dict[str, Any]] = []
    for _lead, out, checked in planned:
        os.replace(_partial(out), out)
        checked["path"] = str(out)
        files.append(checked)
    assert grid0 is not None and levels0 is not None
    receipt = {
        "schema": SCHEMA,
        "source": source.name,
        "source_description": source.description,
        "source_notes": source.notes,
        "licence": {
            "met_office": LICENCE, "met_office_licence_url": LICENCE_URL,
            "met_office_data": DATA_URL,
            "supplement": SUPPLEMENT_ATTRIBUTION,
        },
        "cycle": f"{request.cycle:%Y-%m-%d_%H:%M:%S}",
        "leads": list(request.leads),
        "interval_hours": int(request.interval_hours),
        "valid_times": [item["valid_time"] for item in files],
        "input_dir": str(Path(request.input_dir).expanduser().absolute()),
        "source_grid": grid0.as_dict(),
        "regridding": "none: the Met Office window is written on the source's own "
                      "0.140625 x 0.09375 degree grid, the supplement on its own "
                      "0.25 degree grid",
        "box": {"south_west_north_east": list(request.bounds), "basis": dict(request.target_basis),
                "halo_cells": HALO_CELLS},
        "levels": {
            "isobaric_pa": levels0,
            "count_with_surface": len(levels0) + 1,
            "not_written_pa": dropped0,
            "why_not_written": "a level is written only where TT, GHT, RH, UU and VV all "
                               "exist; relative humidity is published to 10 hPa, so the "
                               "temperature, height and wind levels above it are left out "
                               "rather than written with no moisture",
            "screen_level_height_m": screen_height,
            "screen_level_note": f"the {screen_height:g} m screen temperature and relative "
                                 f"humidity are written at the WPS 2 m surface level 200100",
            "below_ground": "isobaric values below the surface are the producer's own "
                            "extrapolation (its flag ancillary variable), written as published",
        },
        "conversions": {
            "UU": "-speed * sin(direction): direction is where the wind blows from, "
                  "relative to true north",
            "VV": "-speed * cos(direction)",
            "GHT": "geopotential_height_on_pressure_levels (height_ASL_on_pressure_levels "
                   "before January 2026), as published",
            "RH": "relative_humidity * 100 when published as a fraction (units 1), "
                  "floored at 0",
            "TT": "temperature_on_pressure_levels; the 200100 record from temperature_at_screen_level",
            "PMSL": "pressure_at_mean_sea_level at level 201300",
            "SKINTEMP": "temperature_at_surface", "SOILHGT": "height_of_orography",
            "SNOW": "snow_depth_water_equivalent: lwe_thickness_of_surface_snow_amount "
                    "in m times 1000 kg m-3 (or kg m-2 as is), floored at 0",
        },
        "not_carried": ["PSFC", "SPECHUMD", "SNOWH", "SST"],
        "why_not_carried": "PSFC: rw_mpas_init interpolates surface pressure from the "
                           "isobaric column (whose below-ground levels the producer "
                           "extrapolates) and rw_mpas_lbc reads none; the moisture is RH; "
                           "SNOWH and SST are optional to the init",
        "supplement": {
            "source": supplement_source.name,
            "supplement_dir": str(Path(request.supplement_dir).expanduser().absolute()),
            "map_source": supplement_source.map_source,
            "fields": sorted(SUPPLEMENT_FIELDS),
            "declared_as": "per-record-grid supplement: these records alone may sit on "
                           "a grid other than the Met Office window's, and they share one "
                           "grid among themselves because LANDSEA masks the soil and sea "
                           "ice index by index",
            "vtable": _described(vtable, digests),
            "vtable_cut_from": _described(packaged, digests),
            "engine": _described(engine, digests),
        },
        "files": files,
        "init_switches": switches,
        "init_switches_note": (
            f"the column tops at {switches['column_top_pa']:g} Pa; --extrap-airtemp "
            f"constant is only for a top below {LAPSE_RATE_TOP_PA:g} Pa"),
    }
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8", newline="\n")
    log(f"RECEIPT {receipt_path}")
    return receipt


# ---------------------------------------------------------------------------
# argparse glue (the parser itself lives in hrrr_intermediate)
# ---------------------------------------------------------------------------
def request_from_arguments(arguments: argparse.Namespace) -> MetOfficeRequest:
    from .hrrr_intermediate import parse_cycle
    from .met_intermediate_door import parse_lead_range
    from .wrfout_intermediate import bounds_from_arguments

    key = str(arguments.source).strip().lower()
    source = METOFFICE_SOURCES[key]
    foreign = [flag for flag, attribute in (
        ("--grib-dir", "grib_dir"), ("--wrfout-glob", "wrfout_glob"),
        ("--wrf-edge-cells", "wrf_edge_cells"), ("--vtable", "vtable"),
        ("--spacing-deg", "spacing_deg"),
    ) if getattr(arguments, attribute, None) is not None]
    if foreign:
        refuse(f"{', '.join(foreign)} belong to another --source; --source {source.name} reads "
               f"NetCDF from --input-dir on its own grid, with the GDAS/GFS supplement from "
               f"--supplement-dir")
    for flag, attribute in (("--input-dir", "input_dir"), ("--cycle", "cycle"),
                            ("--supplement-dir", "supplement_dir")):
        if getattr(arguments, attribute, None) is None:
            refuse(f"--source {source.name} needs {flag}; it has no default")
    mesh_grid = getattr(arguments, "mesh_grid", None)
    margin = float(getattr(arguments, "margin_km", 0.0) or 0.0)
    if not margin >= 0.0:
        refuse(f"--margin-km {margin} is negative")
    if mesh_grid is not None:
        given = [flag for flag, attribute in (("--from-plan", "from_plan"),
                                               ("--cull-region", "cull_region"),
                                               ("--point", "point"), ("--radius-km", "radius_km"),
                                               ("--halo-km", "halo_km"))
                 if getattr(arguments, attribute, None) is not None]
        if given:
            refuse(f"--mesh-grid and {', '.join(given)} were both given; the box comes from one "
                   f"(a culled mesh already carries its boundary rings)")
        bounds, basis = mesh_bounds(Path(mesh_grid), margin)
        basis["margin_km"] = margin
    else:
        given = [getattr(arguments, attribute, None)
                 for attribute in ("from_plan", "cull_region", "point")]
        if all(value is None for value in given):
            refuse("no box was given: pass --mesh-grid (the culled grid), --from-plan, "
                   "--cull-region with --halo-km, or --point with --radius-km; the "
                   "intermediate covers the cull and nothing here guesses one")
        bounds, basis = bounds_from_arguments(arguments)
    interval = getattr(arguments, "interval_hours", None)
    interval = source.default_interval_hours if interval is None else int(interval)
    leads = parse_lead_range(getattr(arguments, "hours", None), interval)
    workers = getattr(arguments, "workers", None)
    workers = 4 if workers is None else int(workers)
    if workers < 1:
        refuse(f"--workers {workers} is not a worker count")
    return MetOfficeRequest(
        source=source, input_dir=Path(arguments.input_dir), cycle=parse_cycle(arguments.cycle),
        leads=leads, interval_hours=interval, bounds=bounds, out_dir=Path(arguments.out_dir),
        supplement_dir=Path(arguments.supplement_dir),
        supplement_source=str(getattr(arguments, "supplement_source", None) or "gdas"),
        prefix=str(getattr(arguments, "prefix", "MET") or "MET"),
        engine=getattr(arguments, "decoder", None), workers=workers, target_basis=basis,
    )


def run_metoffice_intermediate(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    receipt = build_metoffice_intermediates(request)
    switches = receipt["init_switches"]
    out_dir = Path(request.out_dir).expanduser().absolute()
    first = receipt["files"][0]["path"]
    flags = (f"--nfglevels {switches['--nfglevels']} --nfgsoillevels "
             f"{switches['--nfgsoillevels']} --use-spechumd {switches['--use-spechumd']} "
             f"--extrap-airtemp {switches['--extrap-airtemp']}")
    summary: dict[str, Any] = {
        "files": [item["path"] for item in receipt["files"]],
        "receipt": str(out_dir / "intermediate-receipt.json"),
        "init_switches": switches,
        "next": f"woof hex init --met {first} ... {flags}",
    }
    if len(receipt["files"]) > 1:
        summary["next_lbc"] = (
            f"woof hex lbc --grid INIT.nc --met-dir {out_dir} --out-dir LBC --start-time "
            f"{receipt['valid_times'][0]} --stop-time {receipt['valid_times'][-1]} "
            f"--nfglevels {switches['--nfglevels']} --use-spechumd "
            f"{switches['--use-spechumd']} --extrap-airtemp {switches['--extrap-airtemp']}"
        )
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "HALO_CELLS",
    "LICENCE",
    "METOFFICE_SOURCES",
    "SCHEMA",
    "SOURCE_NAME",
    "SUPPLEMENT_FIELDS",
    "VARIABLES",
    "Crop",
    "MetOfficeRequest",
    "NetcdfSource",
    "RegularGrid",
    "build_metoffice_intermediates",
    "common_levels",
    "crop_for",
    "file_name",
    "locate_files",
    "mesh_bounds",
    "request_from_arguments",
    "run_metoffice_intermediate",
    "supplement_vtable_text",
    "wind_components",
]
