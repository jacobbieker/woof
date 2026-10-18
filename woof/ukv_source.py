"""UKV (Met Office UK deterministic 2 km) from the Source Cooperative archive.

The archive is one Icechunk repository on the Source Cooperative open-data
bucket: the UKV's hourly state from 2024-01-01 00Z to 2025-08-13 22Z on the
UK 2 km standard grid, a Lambert azimuthal equal-area plane.  The mapped
engine reads regular latitude/longitude and Lambert conformal grids only, so
this module does the one thing a mapping document cannot: it reads the
requested window out of the store, crops it to the domain plus a margin,
interpolates it onto a regular 0.02-degree latitude/longitude grid and turns
speed and from-direction into earth-relative wind components.  It writes one
CF NetCDF per valid time, and the packaged profile ``ukv-sc-netcdf-v1``
(``authorities/rw-wps-ukv-sc.*.json``) decodes those files with no UKV code
in the engine.

The store publishes no soil, no sea ice and no orography.  The land state and
sea ice come from ERA5 at the same valid time (Google's anonymous ARCO copy),
copied cell for cell by nearest index onto the same grid, the semantics of the
composition's ``nearest_coordinate_index`` borrow: nothing is interpolated and
the borrowed land/sea boundary follows the 0.25-degree ERA5 coastline.  The
terrain is UKV's own: the height of each column's published surface pressure
on its own pressure-level heights, so terrain and surface pressure agree.

Three facts of the archive were MEASURED (2026-10-18) and are encoded here,
because the store states none of them:

* wind_from_direction is relative to TRUE north.  Regressing the angle
  between the published 300 hPa direction and the geostrophic direction
  computed on the plane against the grid convergence gave a slope of -0.24
  for true north and -1.24 for grid north over five dates, so no rotation is
  applied.
* the pressure-level ``wind_from_direction`` array is one record short: the
  2025-05-16 15Z record is absent and every later record sits one index
  early (matched against the height-level winds by bisection).  That hour
  is refused and later hours read index - 1.
* values are float16 except pressure: temperature is quantized to 0.125 K
  below 256 K and 0.25 K above, and geopotential height to 16 m above
  16384 m (about 10 hPa to 100 hPa).  Values are used as published.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from woof import fetch_guard
from woof.era5_arco import ARCO_STORE
from woof.explain import layered

SOURCE_ID = "ukv"
CONTRACT = "ukv-source-coop-icechunk-regrid-v1"
MANIFEST_SCHEMA = "gpuwm-ukv-source-coop-fetch-manifest-v1"
REQUEST_NAME = "ukv-request.json"
FILE_SCHEMA = "gpuwm-ukv-regular-netcdf-v1"

#: The archive.  Icechunk format 1: the dated siblings
#: (``..._29092024``/``..._16052025``) are format 2 and do not open with
#: the icechunk this tree is tested against.
STORE_BUCKET = "us-west-2.opendata.source.coop"
STORE_PREFIX = "bkr/metoffice/metoffice_uk_deterministic_2km.icechunk"
STORE_REGION = "us-west-2"
STORE_URL = f"s3://{STORE_BUCKET}/{STORE_PREFIX}"
STORE_DOCUMENTATION = "https://source.coop/bkr/metoffice"

#: First and last hourly record (MEASURED: 14183 hourly steps, no gaps).
COVERAGE_START = datetime(2024, 1, 1, 0)
COVERAGE_END = datetime(2025, 8, 13, 22)
CADENCE_HOURS = 1
#: The hour the pressure-level wind direction is missing (see module doc).
WIND_DIRECTION_GAP = datetime(2025, 5, 16, 15)

#: The UK 2 km standard grid: Lambert azimuthal equal-area on the WGS84
#: ellipsoid, read from the store's ``lambert_azimuthal_equal_area``
#: grid-mapping attributes and its projection coordinates (cell centres).
LAEA_SEMI_MAJOR = 6378137.0
LAEA_SEMI_MINOR = 6356752.314140356
LAEA_LATITUDE_ORIGIN = 54.9
LAEA_LONGITUDE_ORIGIN = -2.5
GRID_X0 = -1158000.0
GRID_Y0 = -1036000.0
GRID_SPACING = 2000.0
GRID_NX = 1042
GRID_NY = 970

#: The regular grid the fetch writes, and how far beyond the requested area
#: it reaches so the preparation's horizontal interpolation has a halo.
TARGET_STEP_DEGREES = 0.02
MARGIN_DEGREES = 0.25

#: The 33 published pressure levels (Pa); the profile's ladder is all of them.
PRESSURE_LEVELS_PA = (
    1000, 2000, 3000, 4000, 5000, 7000, 10000, 12500, 15000, 17500, 20000,
    22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 45000, 50000,
    55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 92500, 95000,
    97500, 100000)

#: Store variable -> (output name, units, standard name).  Winds are read
#: as speed and direction and written as components, below.
_LEVEL_FIELDS = {
    "air_temperature": ("air_temperature", "K", "air_temperature"),
    "geopotential_height": ("geopotential_height", "m", "geopotential_height"),
    "relative_humidity": ("relative_humidity", "1", "relative_humidity"),
}
_SURFACE_FIELDS = {
    "surface_air_pressure": ("surface_air_pressure", "Pa", "surface_air_pressure"),
    "surface_temperature": ("surface_temperature", "K", "surface_temperature"),
    "air_temperature_1.5m": ("air_temperature_1p5m", "K", "air_temperature"),
    "relative_humidity_1.5m": ("relative_humidity_1p5m", "1", "relative_humidity"),
    "lwe_thickness_of_surface_snow_amount": (
        "lwe_thickness_of_surface_snow_amount", "m",
        "lwe_thickness_of_surface_snow_amount"),
}
_WIND_PAIRS = {
    # store speed, store direction -> output u, v
    ("wind_speed", "wind_from_direction"): ("eastward_wind", "northward_wind"),
    ("wind_speed_10.0m", "wind_from_direction_10.0m"): (
        "eastward_wind_10m", "northward_wind_10m"),
}

#: The ERA5 land state and sea ice, borrowed from Google's ARCO copy.  Zarr
#: v2, one blosc/lz4 chunk per field per hour on the global 0.25-degree grid
#: (latitude 90 -> -90, longitude 0 -> 359.75), time in hours since 1900.
ERA5_STORE = ARCO_STORE
ERA5_ORIGIN = datetime(1900, 1, 1)
ERA5_SHAPE = (721, 1440)
ERA5_STEP = 0.25
_ERA5_FIELDS = {
    "land_sea_mask": ("era5_land_sea_mask", "1", "land_area_fraction"),
    "sea_ice_cover": ("era5_sea_ice_cover", "1", "sea_ice_area_fraction"),
    **{f"soil_temperature_level_{k}": (f"era5_soil_temperature_level_{k}", "K",
                                       "soil_temperature") for k in range(1, 5)},
    **{f"volumetric_soil_water_layer_{k}": (
        f"era5_volumetric_soil_water_layer_{k}", "m3 m-3",
        "volume_fraction_of_condensed_water_in_soil") for k in range(1, 5)},
}

#: The composition's in-band terrain role: every primary file carries the
#: same derived terrain, so the files are their own terrain supplement.
TERRAIN_ROLE = "ukv_sc_in_band_surface"

_RD = 287.04
_GRAVITY = 9.80665


# --------------------------------------------------------------------------
# The plane: ellipsoidal Lambert azimuthal equal-area (Snyder 1987, 24)
# --------------------------------------------------------------------------

_E2 = 1.0 - (LAEA_SEMI_MINOR / LAEA_SEMI_MAJOR) ** 2
_E = math.sqrt(_E2)


def _authalic_q(phi):
    s = np.sin(phi)
    return (1.0 - _E2) * (s / (1.0 - _E2 * s * s)
                          - np.log((1.0 - _E * s) / (1.0 + _E * s)) / (2.0 * _E))


_QP = float(_authalic_q(np.pi / 2.0))
_RQ = LAEA_SEMI_MAJOR * math.sqrt(_QP / 2.0)
_PHI1 = math.radians(LAEA_LATITUDE_ORIGIN)
_BETA1 = math.asin(float(_authalic_q(_PHI1)) / _QP)
_D = (LAEA_SEMI_MAJOR * math.cos(_PHI1)
      / math.sqrt(1.0 - _E2 * math.sin(_PHI1) ** 2) / (_RQ * math.cos(_BETA1)))
_LAMBDA0 = math.radians(LAEA_LONGITUDE_ORIGIN)


def laea_forward(latitude, longitude):
    """Projection metres (x, y) of geographic degrees on the UKV plane."""

    phi = np.radians(np.asarray(latitude, dtype=np.float64))
    delta = np.radians(np.asarray(longitude, dtype=np.float64)) - _LAMBDA0
    beta = np.arcsin(np.clip(_authalic_q(phi) / _QP, -1.0, 1.0))
    scale = _RQ * np.sqrt(2.0 / (1.0 + math.sin(_BETA1) * np.sin(beta)
                                 + math.cos(_BETA1) * np.cos(beta) * np.cos(delta)))
    x = scale * _D * np.cos(beta) * np.sin(delta)
    y = (scale / _D) * (math.cos(_BETA1) * np.sin(beta)
                        - math.sin(_BETA1) * np.cos(beta) * np.cos(delta))
    return x, y


def laea_inverse(x, y):
    """Geographic degrees (latitude, longitude) of UKV plane metres."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.hypot(x / _D, _D * y)
    centre = rho == 0.0
    safe = np.where(centre, 1.0, rho)
    ce = 2.0 * np.arcsin(np.clip(rho / (2.0 * _RQ), -1.0, 1.0))
    beta = np.where(centre, _BETA1, np.arcsin(np.clip(
        np.cos(ce) * math.sin(_BETA1) + _D * y * np.sin(ce) * math.cos(_BETA1) / safe,
        -1.0, 1.0)))
    lam = _LAMBDA0 + np.arctan2(
        x * np.sin(ce),
        _D * rho * math.cos(_BETA1) * np.cos(ce) - _D * _D * y * math.sin(_BETA1) * np.sin(ce))
    e4, e6 = _E2 * _E2, _E2 * _E2 * _E2
    phi = (beta + (_E2 / 3.0 + 31.0 * e4 / 180.0 + 517.0 * e6 / 5040.0) * np.sin(2.0 * beta)
           + (23.0 * e4 / 360.0 + 251.0 * e6 / 3780.0) * np.sin(4.0 * beta)
           + (761.0 * e6 / 45360.0) * np.sin(6.0 * beta))
    return np.degrees(phi), np.degrees(lam)


def coverage_box(margin_cells: int = 1) -> tuple[float, float, float, float]:
    """The largest latitude/longitude box inside the UKV plane, (S, W, N, E).

    Measured from the plane's own edges, ``margin_cells`` inside them so
    a bilinear stencil never leaves the grid; a box this size is what a
    regular coverage window can say about a projected domain.
    """

    lo_x = GRID_X0 + margin_cells * GRID_SPACING
    hi_x = GRID_X0 + (GRID_NX - 1 - margin_cells) * GRID_SPACING
    lo_y = GRID_Y0 + margin_cells * GRID_SPACING
    hi_y = GRID_Y0 + (GRID_NY - 1 - margin_cells) * GRID_SPACING
    xs = np.linspace(lo_x, hi_x, 2001)
    ys = np.linspace(lo_y, hi_y, 2001)
    south_lat, _ = laea_inverse(xs, np.full_like(xs, lo_y))
    north_lat, _ = laea_inverse(xs, np.full_like(xs, hi_y))
    _, west_lon = laea_inverse(np.full_like(ys, lo_x), ys)
    _, east_lon = laea_inverse(np.full_like(ys, hi_x), ys)
    return (float(south_lat.max()), float(west_lon.max()),
            float(north_lat.min()), float(east_lon.min()))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

def _archive_refusal(what: str) -> str:
    return layered(
        f"{what}  The UKV archive holds hourly states from "
        f"{COVERAGE_START:%Y-%m-%d %H}Z to {COVERAGE_END:%Y-%m-%d %H}Z; choose "
        "a cycle and --hours inside it, or another source (era5, gfs) for "
        "dates outside it.",
        f"The Source Cooperative copy ({STORE_URL}) is a closed archive: "
        "nothing before or after those hours exists in it, so a window "
        "reaching past either end has no state to initialize or force from.")


def validate_window(cycle: datetime, hours: int,
                    cadence: int | None = None) -> tuple[datetime, ...]:
    """The valid times of one window, or a refusal naming why not."""

    spacing = CADENCE_HOURS if cadence is None else cadence
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("UKV states are hourly; the cycle must fall on a whole UTC hour")
    if type(spacing) is not int or spacing < CADENCE_HOURS:
        raise ValueError("--cadence must be a positive whole number of hours for --source ukv")
    if type(hours) is not int or hours < 0 or hours % spacing:
        raise ValueError(
            f"--hours must be a nonnegative multiple of the {spacing} h cadence")
    times = tuple(cycle + timedelta(hours=step) for step in range(0, hours + 1, spacing))
    if times[0] < COVERAGE_START or times[-1] > COVERAGE_END:
        raise ValueError(_archive_refusal(
            f"--source ukv window {times[0]:%Y-%m-%dT%H}Z..{times[-1]:%Y-%m-%dT%H}Z "
            "leaves the archive."))
    if WIND_DIRECTION_GAP in times:
        raise ValueError(layered(
            f"--source ukv has no pressure-level wind direction at "
            f"{WIND_DIRECTION_GAP:%Y-%m-%dT%H}Z; start the window after it, end "
            "it before it, or choose a cadence that steps over it.",
            "The archive's pressure-level wind_from_direction array is one "
            "record short: that hour is absent and every later record is "
            "stored one index early (measured against the height-level "
            "winds).  The later hours are read at their true time; this one "
            "has no wind to read."))
    return times


def latest_cycle(hours: int, cadence: int | None = None) -> datetime:
    """The latest cycle whose window still ends inside the archive."""

    cycle = COVERAGE_END - timedelta(hours=hours)
    validate_window(cycle, hours, cadence)
    return cycle


def target_axes(area) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of the regular grid covering ``area`` + margin."""

    if area is None:
        raise ValueError(layered(
            "--source ukv needs --area (or --point with --radius-km): the "
            "fetch interpolates only the domain it is given.",
            "Each UKV hour is a ~150 MB set of whole-domain chunks; the fetch "
            "reads them, crops to the domain plus a margin and regrids, so "
            "without an area it would have to write the whole plane at "
            "0.02 degrees."))
    if area.crosses_antimeridian:
        raise ValueError(_domain_refusal("crosses the antimeridian"))
    step = TARGET_STEP_DEGREES

    def axis(low, high):
        first = math.floor((low - MARGIN_DEGREES) / step + 1e-9)
        last = math.ceil((high + MARGIN_DEGREES) / step - 1e-9)
        return np.round(np.arange(first, last + 1) * step, 6)

    west = ((area.lon_west + 180.0) % 360.0) - 180.0
    east = west + area.longitude_span_degrees
    return axis(area.lat_south, area.lat_north), axis(west, east)


def _domain_refusal(detail: str) -> str:
    south, west, north, east = coverage_box()
    return layered(
        f"--source ukv cannot serve this domain: {detail}.  Keep the domain "
        f"(plus its {MARGIN_DEGREES:g}-degree margin) inside the UKV grid, "
        f"which wholly contains the box {south:.2f}..{north:.2f}N, "
        f"{west:.2f}..{east:.2f}E, or use a global source (era5, gfs).",
        "The UKV publishes one regional plane, the UK 2 km standard grid "
        "(Lambert azimuthal equal-area centred on 54.9N 2.5W, "
        f"{GRID_NX} x {GRID_NY} cells).  Outside it there is no UKV state, "
        "and nothing is extrapolated.")


class Regridder:
    """Bilinear interpolation from the UKV plane to one regular grid."""

    def __init__(self, latitude: np.ndarray, longitude: np.ndarray) -> None:
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        lat2, lon2 = np.meshgrid(self.latitude, self.longitude, indexing="ij")
        x, y = laea_forward(lat2, lon2)
        fx = (x - GRID_X0) / GRID_SPACING
        fy = (y - GRID_Y0) / GRID_SPACING
        outside = ~((fx >= 0.0) & (fx <= GRID_NX - 1) & (fy >= 0.0) & (fy <= GRID_NY - 1))
        if outside.any():
            j, i = np.argwhere(outside)[0]
            raise ValueError(_domain_refusal(
                f"{int(outside.sum())} of {outside.size} grid points leave the UKV "
                f"plane, the first at {self.latitude[j]:.2f}N "
                f"{self.longitude[i]:.2f}E"))
        i0 = np.minimum(np.floor(fx).astype(np.int64), GRID_NX - 2)
        j0 = np.minimum(np.floor(fy).astype(np.int64), GRID_NY - 2)
        self.window = (int(j0.min()), int(j0.max()) + 2, int(i0.min()), int(i0.max()) + 2)
        self.j0 = j0 - self.window[0]
        self.i0 = i0 - self.window[2]
        self.wx = fx - i0
        self.wy = fy - j0

    def native_slices(self) -> tuple[slice, slice]:
        return slice(self.window[0], self.window[1]), slice(self.window[2], self.window[3])

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """Interpolate a cropped (..., y, x) native field onto the grid."""

        values = np.asarray(field, dtype=np.float64)
        j, i, wx, wy = self.j0, self.i0, self.wx, self.wy
        return ((values[..., j, i] * (1.0 - wx) + values[..., j, i + 1] * wx) * (1.0 - wy)
                + (values[..., j + 1, i] * (1.0 - wx) + values[..., j + 1, i + 1] * wx) * wy)


def wind_components(speed: np.ndarray, from_direction: np.ndarray):
    """Earth-relative (u, v) from speed and meteorological FROM direction.

    The direction is relative to true north (measured; see the module
    doc), so the components need no rotation.
    """

    radians = np.radians(np.asarray(from_direction, dtype=np.float64))
    speed = np.asarray(speed, dtype=np.float64)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def surface_height(levels_pa, geopotential_height, temperature, relative_humidity,
                   surface_pressure, screen_temperature, screen_humidity) -> np.ndarray:
    """Each column's height at its published surface pressure (m).

    The hypsometric equation from the first published level above the
    surface down to the surface pressure, with the layer's mean virtual
    temperature between that level and the screen level.  A surface below
    the deepest level (pressure above 1000 hPa) integrates down from it.
    Humidities are fractions; saturation is WRF real's Bolton relation.
    """

    levels = np.asarray(levels_pa, dtype=np.float64)
    order = np.argsort(-levels, kind="stable")          # deepest first
    ladder = levels[order]
    heights = np.asarray(geopotential_height, dtype=np.float64)[order]
    temps = np.asarray(temperature, dtype=np.float64)[order]
    humid = np.asarray(relative_humidity, dtype=np.float64)[order]
    pressure = np.asarray(surface_pressure, dtype=np.float64)
    if not (np.isfinite(pressure).all() and (pressure > ladder[-1]).all()):
        raise ValueError("UKV surface pressure is missing or above the highest published level")
    above = ladder[None, None, :] < pressure[..., None]
    level = np.argmax(above, axis=-1)                    # first level above ground
    rows, cols = np.indices(pressure.shape)
    p_k = ladder[level]

    def virtual(t, rh, p):
        es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        e = np.clip(rh, 0.0, 1.0) * es
        q = 0.622 * e / (p - 0.378 * e)
        return t * (1.0 + 0.608 * q)

    tv_level = virtual(temps[level, rows, cols], humid[level, rows, cols], p_k)
    tv_surface = virtual(np.asarray(screen_temperature, dtype=np.float64),
                         np.asarray(screen_humidity, dtype=np.float64), pressure)
    thickness = _RD * 0.5 * (tv_level + tv_surface) / _GRAVITY * np.log(pressure / p_k)
    return heights[level, rows, cols] - thickness


# --------------------------------------------------------------------------
# Reading the archives
# --------------------------------------------------------------------------

def open_store():
    """The archive's read-only root group and the snapshot it resolves to."""

    try:
        import icechunk
        import zarr
    except ImportError as error:
        raise RuntimeError(layered(
            "--source ukv reads an Icechunk archive and needs the icechunk "
            "and zarr packages: `pixi install` (they are in this checkout's "
            "environment) or `pip install icechunk zarr`.",
            f"The import failed: {error}.")) from error
    storage = icechunk.s3_storage(bucket=STORE_BUCKET, prefix=STORE_PREFIX,
                                  region=STORE_REGION, anonymous=True)
    session = icechunk.Repository.open(storage).readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r"), str(session.snapshot_id)


def store_indices(group, times) -> dict[datetime, tuple[int, int]]:
    """Per valid time: the record index, and the pressure-level direction's.

    Refuses when the store no longer has the layout this module measured,
    rather than read a record an hour away from the one it names.
    """

    stamps = np.asarray(group["time"][:], dtype=np.int64)
    expected = (int((COVERAGE_END - COVERAGE_START).total_seconds() // 3600) + 1)
    origin = int((COVERAGE_START - datetime(1970, 1, 1)).total_seconds() * 1000)
    x = np.asarray(group["projection_x_coordinate"][:], dtype=np.float64)
    y = np.asarray(group["projection_y_coordinate"][:], dtype=np.float64)
    plane = (GRID_NY, GRID_NX)
    shapes = {name: (expected, len(PRESSURE_LEVELS_PA), *plane) for name in
              (*_LEVEL_FIELDS, "wind_speed")}
    shapes["wind_from_direction"] = (expected - 1, len(PRESSURE_LEVELS_PA), *plane)
    shapes.update({name: (expected, *plane) for name in
                   (*_SURFACE_FIELDS, "wind_speed_10.0m", "wind_from_direction_10.0m")})
    units = {store: unit for store, (_out, unit, _std) in
             {**_LEVEL_FIELDS, **_SURFACE_FIELDS}.items()}
    for speed, direction in _WIND_PAIRS:
        units[speed], units[direction] = "m s-1", "degrees"
    # Everything a record index, a level label or a unit here assumes:
    # the time axis, the pressure ladder in store order, the plane's
    # origin and spacing, each array's shape and each field's units.
    checks = {
        "hourly time axis 2024-01-01 00Z..2025-08-13 22Z": (
            stamps.size == expected and int(stamps[0]) == origin
            and bool(np.all(np.diff(stamps) == 3_600_000))
            and str(group["time"].attrs.get("units")) == "milliseconds since 1970-01-01"),
        "33 pressure levels in Pa, top first": (
            np.asarray(group["pressure"][:], dtype=np.float64).tolist()
            == [float(level) for level in PRESSURE_LEVELS_PA]
            and str(group["pressure"].attrs.get("units")) == "Pa"),
        "the 2 km plane's origin and spacing": (
            x.size == GRID_NX and y.size == GRID_NY
            and x[0] == GRID_X0 and y[0] == GRID_Y0
            and bool(np.all(np.diff(x) == GRID_SPACING))
            and bool(np.all(np.diff(y) == GRID_SPACING))),
        "array shapes (pressure-level direction one record short)": all(
            tuple(group[name].shape) == shape for name, shape in shapes.items()),
        "field units": all(str(group[name].attrs.get("units")) == unit
                           for name, unit in units.items()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(layered(
            f"The UKV archive at {STORE_URL} no longer has the layout this "
            f"woof measured ({'; '.join(failed)} changed); update woof.",
            "Every record index, level label and unit here is taken from that "
            "measured layout, and a store that has been rewritten could put a "
            "different hour, level or unit behind the same index."))
    result = {}
    for valid in times:
        index = int((valid - COVERAGE_START).total_seconds() // 3600)
        direction = index - 1 if valid > WIND_DIRECTION_GAP else index
        result[valid] = (index, direction)
    return result


def read_ukv(group, index: int, direction_index: int, regrid: Regridder,
             *, workers: int = 4) -> dict[str, np.ndarray]:
    """One UKV hour on the regular grid, as float64 arrays keyed by output name."""

    rows, cols = regrid.native_slices()
    reads = {**{name: (name, index, True) for name in _LEVEL_FIELDS},
             **{name: (name, index, False) for name in _SURFACE_FIELDS}}
    for (speed, direction), _ in _WIND_PAIRS.items():
        levelled = speed == "wind_speed"
        reads[speed] = (speed, index, levelled)
        reads[direction] = (direction, direction_index if levelled else index, levelled)

    def read(item):
        name, record, levelled = item
        selection = (record, slice(None), rows, cols) if levelled else (record, rows, cols)
        values = np.asarray(group[name][selection], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"UKV {name} has missing values at record {record} inside the domain")
        return values

    with ThreadPoolExecutor(max_workers=workers) as pool:
        native = dict(zip(reads, pool.map(read, reads.values())))
    fields = {}
    for name, (output, _units, _standard) in {**_LEVEL_FIELDS, **_SURFACE_FIELDS}.items():
        fields[output] = regrid(native[name])
    for (speed, direction), (u_name, v_name) in _WIND_PAIRS.items():
        u, v = wind_components(native[speed], native[direction])
        fields[u_name], fields[v_name] = regrid(u), regrid(v)
    return fields


def _http_get(url: str) -> bytes:
    last = None
    for attempt in range(5):
        try:
            with urlopen(url, timeout=180) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last = error
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                break
            time.sleep(min(4 * (attempt + 1), 20))
    raise RuntimeError(f"ERA5 ARCO read failed after bounded retries: {url}: {last}")


def era5_nearest_indices(latitude: np.ndarray, longitude: np.ndarray):
    """Nearest ERA5 row and column per target row and column (lower on ties)."""

    rows = np.ceil((90.0 - np.asarray(latitude, dtype=np.float64)) / ERA5_STEP - 0.5)
    cols = np.ceil(np.mod(np.asarray(longitude, dtype=np.float64), 360.0) / ERA5_STEP - 0.5)
    return rows.astype(np.int64), (cols.astype(np.int64) % ERA5_SHAPE[1])


def read_era5(valid: datetime, latitude: np.ndarray, longitude: np.ndarray,
              *, workers: int = 4) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """ERA5's land state and sea ice at ``valid``, by nearest index onto the grid."""

    import numcodecs

    index = int((valid - ERA5_ORIGIN).total_seconds() // 3600)
    rows, cols = era5_nearest_indices(latitude, longitude)
    codec = numcodecs.Blosc()

    def read(name):
        raw = codec.decode(_http_get(f"{ERA5_STORE}/{name}/{index}.0.0"))
        field = np.frombuffer(raw, dtype="<f4").reshape(ERA5_SHAPE)
        return field[np.ix_(rows, cols)].astype(np.float64)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = dict(zip(_ERA5_FIELDS, pool.map(read, _ERA5_FIELDS)))
    clamped = 0
    fields = {}
    for name, (output, _units, _standard) in _ERA5_FIELDS.items():
        field = values[name]
        if name != "sea_ice_cover" and not np.isfinite(field).all():
            raise ValueError(f"ERA5 {name} has missing values at {valid:%Y-%m-%dT%H}Z inside the domain")
        if name.startswith("volumetric_soil_water"):
            # ERA5 packs slightly negative water contents over the sea
            # (down to -0.01 measured); a negative volume fraction is not
            # a state, so it is floored at zero and counted.
            clamped += int((field < 0.0).sum())
            field = np.maximum(field, 0.0)
        fields[output] = field
    receipt = {"store": ERA5_STORE, "time_index": index, "valid_time": valid.isoformat(),
               "negative_soil_water_floored": clamped}
    return fields, receipt


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _units_and_names():
    table = {output: (units, standard) for output, units, standard in
             (*_LEVEL_FIELDS.values(), *_SURFACE_FIELDS.values(), *_ERA5_FIELDS.values())}
    for u_name, v_name in _WIND_PAIRS.values():
        table[u_name] = ("m s-1", "eastward_wind")
        table[v_name] = ("m s-1", "northward_wind")
    table["surface_altitude"] = ("m", "surface_altitude")
    return table


_UNITS = _units_and_names()
_LEVELLED = {"air_temperature", "geopotential_height", "relative_humidity",
             "eastward_wind", "northward_wind"}


def write_netcdf(path: Path, valid: datetime, latitude, longitude,
                 fields: dict[str, np.ndarray], attributes: dict[str, str]) -> None:
    """One valid time as a CF NetCDF-4 file the packaged profile reads."""

    from netCDF4 import Dataset

    temporary = path.with_name(path.name + ".partial")
    with Dataset(temporary, "w", format="NETCDF4") as dataset:
        dataset.setncatts({"Conventions": "CF-1.8", **attributes})
        dataset.createDimension("time", 1)
        dataset.createDimension("pressure", len(PRESSURE_LEVELS_PA))
        dataset.createDimension("latitude", len(latitude))
        dataset.createDimension("longitude", len(longitude))
        variable = dataset.createVariable("time", "f8", ("time",))
        variable.setncatts({"standard_name": "time", "calendar": "proleptic_gregorian",
                            "units": "seconds since 1970-01-01 00:00:00", "axis": "T"})
        variable[:] = [(valid - datetime(1970, 1, 1)).total_seconds()]
        variable = dataset.createVariable("pressure", "f8", ("pressure",))
        variable.setncatts({"standard_name": "air_pressure", "long_name": "pressure",
                            "units": "Pa", "positive": "down", "axis": "Z"})
        variable[:] = PRESSURE_LEVELS_PA
        for name, values, units in (("latitude", latitude, "degrees_north"),
                                    ("longitude", longitude, "degrees_east")):
            variable = dataset.createVariable(name, "f8", (name,))
            variable.setncatts({"standard_name": name, "units": units,
                                "axis": "Y" if name == "latitude" else "X"})
            variable[:] = values
        variable = dataset.createVariable("crs", "i4")
        variable.setncatts({"grid_mapping_name": "latitude_longitude",
                            "semi_major_axis": LAEA_SEMI_MAJOR,
                            "semi_minor_axis": LAEA_SEMI_MINOR})
        for name in sorted(fields):
            units, standard = _UNITS[name]
            dims = (("time", "pressure", "latitude", "longitude") if name in _LEVELLED
                    else ("time", "latitude", "longitude"))
            missing = name == "era5_sea_ice_cover"
            variable = dataset.createVariable(
                name, "f4", dims, zlib=True, complevel=1,
                fill_value=np.float32(np.nan) if missing else False)
            variable.setncatts({"standard_name": standard, "units": units,
                                "grid_mapping": "crs"})
            variable[:] = np.asarray(fields[name], dtype=np.float32)[None, ...]
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, document) -> None:
    fetch_guard.atomic_write_text(
        path, json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _document(path: Path) -> dict | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def file_name(valid: datetime) -> str:
    return f"ukv-{valid:%Y%m%dT%H}Z.nc"


def request_identity(times, area) -> dict:
    return {"schema": "gpuwm-ukv-source-coop-request-v1", "source": SOURCE_ID,
            "contract": CONTRACT, "store": STORE_URL, "era5_store": ERA5_STORE,
            "valid_times": [value.isoformat() for value in times],
            "area": area.as_manifest(), "grid_step_degrees": TARGET_STEP_DEGREES,
            "margin_degrees": MARGIN_DEGREES}


def check_prior_request(out: Path, *, cycle: datetime, hours: int,
                        cadence: int | None, area) -> None:
    """Refuse a cache directory that holds another request's files."""

    expected = request_identity(validate_window(cycle, hours, cadence), area)
    if _document(Path(out) / REQUEST_NAME) != expected:
        raise ValueError("The UKV cache belongs to another acquisition request")


def _reusable(out: Path, identity: dict) -> dict | None:
    from woof import fetch_routes
    manifest = _document(out / fetch_routes.MANIFEST_NAME)
    if (manifest is None or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("request") != identity):
        return None
    try:
        for entry in manifest["files"]:
            if _sha256(out / Path(entry["path"]).name) != entry["sha256"]:
                return None
    except (OSError, KeyError, TypeError):
        return None
    return manifest


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + ".fetch.json")


def _resumable(path: Path, identity: dict) -> dict | None:
    """The stamp of an hour an interrupted run of THIS request finished."""

    stamp = _document(_stamp(path))
    try:
        if (stamp is not None and stamp.get("request") == identity
                and _sha256(path) == stamp["entry"]["sha256"]):
            return stamp
    except (OSError, KeyError, TypeError):
        pass
    return None


def _owned_names(out: Path) -> list[str]:
    """Every file name an earlier request in ``out`` may have written."""

    from woof import fetch_routes
    names = [REQUEST_NAME, fetch_routes.MANIFEST_NAME, fetch_routes.SHA256SUMS_NAME,
             fetch_routes.INPUT_LIST_NAME, fetch_routes.PREP_COMMAND_NAME,
             fetch_routes.PREP_ARGUMENTS_NAME]
    for entry in (_document(out / fetch_routes.MANIFEST_NAME) or {}).get("files", []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            names.append(Path(entry["path"]).name)
    for value in (_document(out / REQUEST_NAME) or {}).get("valid_times", []):
        try:
            names.append(file_name(datetime.fromisoformat(value)))
        except (TypeError, ValueError):
            continue
    # Only plain names inside ``out``: a recorded path never reaches outside.
    names = [name for name in names if name and name not in (".", "..")]
    return [*names, *(_stamp(Path(name)).name for name in names if name.endswith(".nc"))]


def _publish_front_door(out: Path, files: list[dict], first: datetime) -> None:
    """The ordered input list, checksums and bound prep arguments."""

    from woof import fetch_routes
    inputs = out / fetch_routes.INPUT_LIST_NAME
    fetch_guard.atomic_write_text(
        inputs, "".join(str(out / entry["path"]) + "\n" for entry in files))
    fetch_guard.atomic_write_text(
        out / fetch_routes.SHA256SUMS_NAME,
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in files))
    tokens = ["--source", SOURCE_ID, "--input-list", str(inputs)]
    for entry in files:
        tokens += ["--supplement", f"{TERRAIN_ROLE}={out / entry['path']}"]
    tokens += ["--author-input-manifest", str(out / "inputs.json")]
    fetch_routes.write_prep_arguments(out, source=SOURCE_ID, prep_source=SOURCE_ID,
                                      cycle=first, tokens=tokens)
    fetch_guard.atomic_write_text(out / fetch_routes.PREP_COMMAND_NAME,
                                  fetch_routes.render_prep_command(tokens) + "\n")


def _file_attributes(snapshot: str, first: datetime) -> dict[str, str]:
    return {
        "title": "UKV (Met Office UK deterministic 2 km) regridded for woof",
        "institution": "Met Office (UKV); ECMWF/Copernicus (ERA5 land state)",
        "source": f"{STORE_URL} snapshot {snapshot}; land state and sea ice from {ERA5_STORE}",
        "woof_file_schema": FILE_SCHEMA,
        "history": (f"{CONTRACT}: bilinear from the UK 2 km Lambert azimuthal "
                    f"equal-area grid to {TARGET_STEP_DEGREES} degrees; winds from "
                    "speed and true-north FROM direction; surface_altitude is "
                    f"the hypsometric height of the {first:%Y-%m-%dT%H}Z surface "
                    "pressure; ERA5 fields by nearest index"),
        "licence": ("UKV: Met Office, British Crown copyright, as distributed by "
                    "Source Cooperative; ERA5: Copernicus Climate Change Service, "
                    "contains modified Copernicus information"),
    }


def _read_terrain(path: Path) -> np.ndarray:
    from netCDF4 import Dataset
    with Dataset(path) as dataset:
        return np.asarray(dataset.variables["surface_altitude"][0], dtype=np.float64)


def acquire(*, cycle: datetime, hours: int, cadence: int | None, area,
            out: Path, workers: int = 4, force: bool = False, progress=None) -> dict:
    """Fetch, regrid and publish one window plus its preparation front door.

    An interrupted run of the same request resumes: each finished hour
    carries a stamp naming the request and its bytes, and is kept.
    """

    from woof import fetch_routes

    say = progress or (lambda text: print(text, file=sys.stderr))
    if type(workers) is not int or workers <= 0:
        raise ValueError("--fetch-workers must be a positive whole number")
    times = validate_window(cycle, hours, cadence)
    latitude, longitude = target_axes(area)
    regrid = Regridder(latitude, longitude)
    identity = request_identity(times, area)
    out = Path(out).resolve()
    with fetch_guard.hold("fetch-out", out):
        out.mkdir(parents=True, exist_ok=True)
        if not force:
            prior = _reusable(out, identity)
            if prior is not None:
                _publish_front_door(out, prior["files"], times[0])
                say(f"fetch ukv: reused verified inputs in {out}")
                return prior
        if force or _document(out / REQUEST_NAME) != identity:
            for name in dict.fromkeys(_owned_names(out)):
                path = out / name
                if path.is_file():
                    fetch_guard.quarantine(path, tag="previous-request")
        _json(out / REQUEST_NAME, identity)
        started = time.monotonic()
        group = snapshot = indices = None
        terrain = None
        files, era5_receipts = [], []
        with ThreadPoolExecutor(max_workers=1) as background:
            for number, valid in enumerate(times, start=1):
                path = out / file_name(valid)
                stamp = _resumable(path, identity)
                if stamp is not None:
                    say(f"fetch ukv: {valid:%Y-%m-%dT%H}Z {number}/{len(times)} kept from "
                        "an interrupted run")
                    files.append(stamp["entry"])
                    era5_receipts.append(stamp["era5"])
                    if terrain is None:
                        terrain = _read_terrain(path)
                    continue
                if group is None:
                    say(f"fetch ukv: opening {STORE_URL}")
                    group, snapshot = open_store()
                    indices = store_indices(group, times)
                say(f"fetch ukv: {valid:%Y-%m-%dT%H}Z {number}/{len(times)} reading UKV and ERA5")
                index, direction_index = indices[valid]
                # The two archives are independent reads: ERA5 runs beside UKV.
                era5_read = background.submit(read_era5, valid, latitude, longitude,
                                              workers=workers)
                fields = read_ukv(group, index, direction_index, regrid, workers=workers)
                era5, receipt = era5_read.result()
                if terrain is None:
                    # One terrain for the whole window, from its first hour, so
                    # every file's in-band supplement is byte-identical.
                    terrain = surface_height(
                        PRESSURE_LEVELS_PA, fields["geopotential_height"],
                        fields["air_temperature"], fields["relative_humidity"],
                        fields["surface_air_pressure"], fields["air_temperature_1p5m"],
                        fields["relative_humidity_1p5m"])
                fields.update(era5)
                fields["surface_altitude"] = terrain
                write_netcdf(path, valid, latitude, longitude, fields,
                             _file_attributes(snapshot, times[0]))
                entry = {"path": path.name, "bytes": path.stat().st_size,
                         "sha256": _sha256(path), "valid_time": valid.isoformat(),
                         "store_record": index, "wind_direction_record": direction_index,
                         "store_snapshot": snapshot}
                _json(_stamp(path), {"request": identity, "entry": entry, "era5": receipt})
                files.append(entry)
                era5_receipts.append(receipt)
        _publish_front_door(out, files, times[0])
        manifest = {
            "schema": MANIFEST_SCHEMA, "source": SOURCE_ID, "contract": CONTRACT,
            "request": identity,
            "store_snapshots": sorted({entry["store_snapshot"] for entry in files}),
            "files": files, "valid_times": identity["valid_times"],
            "grid": {"latitude": [float(latitude[0]), float(latitude[-1]), len(latitude)],
                     "longitude": [float(longitude[0]), float(longitude[-1]), len(longitude)],
                     "native_window": list(regrid.window)},
            "terrain": {"derived_from": times[0].isoformat(),
                        "min_m": float(terrain.min()), "max_m": float(terrain.max())},
            "era5": era5_receipts,
            "total_bytes": sum(entry["bytes"] for entry in files),
            "wall_seconds": time.monotonic() - started, "workers": workers,
        }
        _json(out / fetch_routes.MANIFEST_NAME, manifest)
        say(f"fetch ukv: published {len(files)} valid time(s) in {out}")
        return manifest


def cli(args) -> int:
    """``woof fetch --source ukv``."""

    from woof.fetch import _resolve_area, parse_cycle
    unsupported = ["member", "validate", "transport", "engine", "mode", "cache_dir",
                   "p_top_pa", "forecast_start_hour", "bridge", "wps_namelist",
                   "experiment_config", "static_input", "static_receipt", "manifest_out"]
    supplied = ["--" + key.replace("_", "-") for key in unsupported
                if getattr(args, key, None) is not None]
    supplied += ["--" + key.replace("_", "-") for key in
                 ("retrieve", "all_levels", "author_front_door_manifest",
                  "accept_inventory_change") if getattr(args, key, False)]
    if supplied:
        raise ValueError(
            f"{', '.join(supplied)} does not describe a UKV archive window; use "
            "--cycle, --hours, --cadence and --area")
    if args.cycle is None or args.hours is None or args.out is None:
        raise ValueError("fetch requires --cycle, --hours and --out")
    if args.cycle == "latest":
        cycle = latest_cycle(args.hours, args.cadence)
    else:
        cycle = parse_cycle(args.cycle, SOURCE_ID)
    manifest = acquire(cycle=cycle, hours=args.hours, cadence=args.cadence,
                       area=_resolve_area(args), out=args.out,
                       workers=args.fetch_workers or 4, force=args.force_refetch)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0
