"""UKV (Met Office UK deterministic 2 km) from its two public stores.

* The Source Cooperative archive: one Icechunk repository holding each
  hour's own analysis (its run's T+0) from 2024-01-01 00Z to 2025-08-13 22Z.
* The Met Office open-data bucket: every run as NetCDF-4, one object per
  field per lead, a rolling two years.  A window outside the archive is
  read from ONE run whose leads hold every valid time.

Both publish the UK 2 km standard grid, a Lambert azimuthal equal-area
plane.  The mapped engine reads regular latitude/longitude and Lambert
conformal grids only, so this module does the one thing a mapping document
cannot: it reads the requested window, crops it to the domain plus a
margin, interpolates it onto a regular 0.02-degree latitude/longitude grid
and turns speed and from-direction into earth-relative wind components.  It
writes one CF NetCDF per valid time, and the packaged profile
``ukv-sc-netcdf-v1`` (``authorities/rw-wps-ukv-sc.*.json``) decodes those
files with no UKV code in the engine.

Neither store publishes soil or sea ice.  They come from ERA5 at the same
valid time (Google's anonymous ARCO copy) when ERA5 holds the window, and
from GFS's 0.25-degree land state past ERA5's last published hour, copied
cell for cell by nearest index onto the same grid on Noah's four layers:
nothing is interpolated and the borrowed land/sea boundary follows the
0.25-degree donor coastline.  The terrain is UKV's own: a live run's
published orography, or in the archive, which publishes none, the height
of the first hour's surface pressure on its own pressure-level heights.

Facts of the stores were MEASURED (2026-10-18) and are encoded here,
because neither store states them:

* wind_from_direction is relative to TRUE north.  Regressing the angle
  between the published 300 hPa direction and the geostrophic direction
  computed on the plane against the grid convergence gave a slope of -0.24
  for true north and -1.24 for grid north over five dates, so no rotation is
  applied.
* the archive's pressure-level ``wind_from_direction`` array is one record
  short: the 2025-05-16 15Z record is absent and every later record sits one
  index early (bisected against the height-level winds and confirmed against
  the live T+0).  That hour is served live or refused, and later hours read
  index - 1.
* archive values are float16 except pressure: temperature is quantized to
  0.125 K below 256 K and 0.25 K above, and geopotential height to 16 m
  above 16384 m.  The live objects are float32.  Values are used as
  published.
* the live runs' leads depend on the run hour (see ``run_leads``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from woof import fetch_guard
from woof.era5_arco import ARCO_STORE
from woof.explain import layered

SOURCE_ID = "ukv"
CONTRACT = "ukv-regrid-v2"
MANIFEST_SCHEMA = "gpuwm-ukv-fetch-manifest-v2"
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

#: The Met Office's own open-data bucket: every UKV run as NetCDF-4, one
#: object per field per lead, ``<run>/<valid>-PT<HHHH>H00M-<field>.nc``,
#: float32 on the same 2 km plane.  A rolling two years, anonymous.  The
#: archive above is each hourly run's T+0 from these same objects
#: (MEASURED 2026-10-18: the 2025-06-01 00Z run's T+0 equals the archive
#: hour to float16 rounding, and its T+1/T+3 neighbours do not).
LIVE_BUCKET = "met-office-atmospheric-model-data"
LIVE_HOST = f"https://{LIVE_BUCKET}.s3.eu-west-2.amazonaws.com"
LIVE_PREFIX = "uk-deterministic-2km/"
LIVE_URL = f"s3://{LIVE_BUCKET}/{LIVE_PREFIX}"
#: Two rolling years: the oldest run listed on 2026-10-18 was 20241016T2200Z.
LIVE_RETENTION_HOURS = 17520
#: Every UTC hour runs.  Leads by run hour (MEASURED 2026-10-12..18 from the
#: bucket's listings, the pressure-level temperature objects): 03Z and 15Z
#: run hourly to T+54 then three-hourly to T+120; 00, 06, 09, 12, 18 and
#: 21Z hourly to T+54; every other hour hourly to T+12.
LIVE_LONG_RUN_HOURS = (3, 15)
LIVE_MAIN_RUN_HOURS = (0, 6, 9, 12, 18, 21)

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
_ERA5_FIELDS = ("land_sea_mask", "sea_ice_cover",
                *(f"soil_temperature_level_{k}" for k in range(1, 5)),
                *(f"volumetric_soil_water_layer_{k}" for k in range(1, 5)))
#: ERA5's four soil layers (m), remapped to Noah's by layer-mean overlap.
ERA5_SOIL_LAYERS = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))

#: Past ERA5's last published hour the land state comes from NCEP GFS on
#: its 0.25-degree grid: the four Noah layers, the land mask and sea ice,
#: byte ranges of the AWS pgrb2 objects read through their ``.idx``.
GFS_HOST = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS_RECORDS = ("LAND:surface", "ICEC:surface",
               *(f"{name}:{layer} m below ground" for layer in ("0-0.1", "0.1-0.4", "0.4-1", "1-2")
                 for name in ("TSOIL", "SOILW")))
#: GFS posts hourly leads to f120, three-hourly to f384.
GFS_HOURLY_THROUGH = 120

#: The land state as the files carry it, whichever product supplied it:
#: Noah's four layers (0-0.1, 0.1-0.4, 0.4-1, 1-2 m).
NOAH_SOIL_LAYERS = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
_LAND_STATE_FIELDS = {
    "land_area_fraction": ("1", "land_area_fraction"),
    "sea_ice_area_fraction": ("1", "sea_ice_area_fraction"),
    **{f"soil_temperature_layer_{k}": ("K", "soil_temperature") for k in range(1, 5)},
    **{f"soil_moisture_layer_{k}": ("m3 m-3", "volume_fraction_of_condensed_water_in_soil")
       for k in range(1, 5)},
}
#: Fields a product may leave missing (water cells): written with a NaN fill.
_MASKED_FIELDS = {"sea_ice_area_fraction",
                  *(f"soil_temperature_layer_{k}" for k in range(1, 5)),
                  *(f"soil_moisture_layer_{k}" for k in range(1, 5))}

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


def run_leads(run_hour: int) -> frozenset[int]:
    """The leads a live run of this UTC hour publishes (measured ladders)."""

    if run_hour in LIVE_LONG_RUN_HOURS:
        return frozenset(range(0, 55)) | frozenset(range(57, 121, 3))
    if run_hour in LIVE_MAIN_RUN_HOURS:
        return frozenset(range(0, 55))
    return frozenset(range(0, 13))


@dataclass(frozen=True)
class WindowPlan:
    """Which store serves a window, and (live) which runs could."""

    times: tuple[datetime, ...]
    store: str
    #: Live runs whose ladder holds every valid time, newest first.
    runs: tuple[datetime, ...] = ()


def _window_times(cycle: datetime, hours: int, cadence: int | None) -> tuple[datetime, ...]:
    spacing = CADENCE_HOURS if cadence is None else cadence
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("UKV states are hourly; the cycle must fall on a whole UTC hour")
    if type(spacing) is not int or spacing < CADENCE_HOURS:
        raise ValueError("--cadence must be a positive whole number of hours for --source ukv")
    if type(hours) is not int or hours < 0 or hours % spacing:
        raise ValueError(
            f"--hours must be a nonnegative multiple of the {spacing} h cadence")
    return tuple(cycle + timedelta(hours=step) for step in range(0, hours + 1, spacing))


def covering_runs(times, *, now: datetime | None = None) -> tuple[datetime, ...]:
    """Live runs whose published ladder holds every time, newest first.

    Only the ladder is asked here, never the bucket: a run that has not
    posted yet still plans, and the fetch checks publication.
    """

    now = now or datetime.utcnow()
    oldest = now - timedelta(hours=LIVE_RETENTION_HOURS)
    first = times[0]
    runs = []
    for back in range(0, 121):
        run = first - timedelta(hours=back)
        if run < oldest:
            break
        leads = run_leads(run.hour)
        if all(int((valid - run).total_seconds() // 3600) in leads for valid in times):
            runs.append(run)
    return tuple(runs)


def _no_run_refusal(times, cadence, *, now: datetime | None = None) -> str:
    span = int((times[-1] - times[0]).total_seconds() // 3600)
    hint = ""
    if (cadence or 1) % 3 and covering_runs(_window_times(times[0], span - span % 3, 3),
                                            now=now):
        hint = (" A 3 h cadence (--cadence 3) is served by the 03Z/15Z runs"
                + ("" if span % 3 == 0 else f", over {span - span % 3} h") + ".")
    return layered(
        f"--source ukv: no UKV run publishes every valid time of "
        f"{times[0]:%Y-%m-%dT%H}Z..{times[-1]:%Y-%m-%dT%H}Z at this cadence.{hint}  "
        "Shorten the window, or start it closer to a run.",
        "Outside the Source Cooperative archive a window is served by ONE run's "
        "forecast from the Met Office open-data bucket, and the runs reach: 03Z "
        "and 15Z hourly to T+54 then three-hourly to T+120; 00, 06, 09, 12, 18 "
        "and 21Z hourly to T+54; every other hour to T+12.  The bucket keeps two "
        "rolling years of runs.")


def check_window_shape(hours: int, cadence: int | None = None) -> None:
    """Refuse a window no live run serves from its own T+0, before any date."""

    _window_times(datetime(2000, 1, 1), hours, cadence)
    spacing = CADENCE_HOURS if cadence is None else cadence
    leads = range(0, hours + 1, spacing)
    if not any(all(lead in run_leads(hour) for lead in leads) for hour in range(24)):
        raise ValueError(_no_run_refusal(
            tuple(datetime(2000, 1, 1, 3) + timedelta(hours=lead) for lead in leads),
            cadence, now=datetime(2000, 1, 2)))


def plan_window(cycle: datetime, hours: int, cadence: int | None = None, *,
                now: datetime | None = None) -> WindowPlan:
    """The valid times of one window and the store that serves them.

    The Source Cooperative archive (each hour's own analysis) serves a
    window that lies inside it and avoids its missing hour; anything else
    is served by one live run's forecast leads.
    """

    times = _window_times(cycle, hours, cadence)
    in_archive = times[0] >= COVERAGE_START and times[-1] <= COVERAGE_END
    if in_archive and WIND_DIRECTION_GAP not in times:
        return WindowPlan(times, "archive")
    runs = covering_runs(times, now=now)
    if runs:
        return WindowPlan(times, "live", runs)
    if in_archive:
        raise ValueError(layered(
            f"--source ukv has no pressure-level wind direction at "
            f"{WIND_DIRECTION_GAP:%Y-%m-%dT%H}Z; start the window after it, end "
            "it before it, or choose a cadence that steps over it.",
            "The archive's pressure-level wind_from_direction array is one "
            "record short: that hour is absent and every later record is "
            "stored one index early (measured against the height-level "
            "winds).  The later hours are read at their true time; this one "
            "has no wind to read, and no live run covers the window either."))
    oldest = (now or datetime.utcnow()) - timedelta(hours=LIVE_RETENTION_HOURS)
    if times[0] < oldest:
        raise ValueError(_archive_refusal(
            f"--source ukv window {times[0]:%Y-%m-%dT%H}Z..{times[-1]:%Y-%m-%dT%H}Z "
            "leaves the archive and is older than the open-data bucket's two "
            "rolling years."))
    raise ValueError(_no_run_refusal(times, cadence, now=now))


def validate_window(cycle: datetime, hours: int,
                    cadence: int | None = None) -> tuple[datetime, ...]:
    """The valid times of one window, or a refusal naming why not."""

    return plan_window(cycle, hours, cadence).times


def latest_cycle(hours: int, cadence: int | None = None) -> datetime:
    """The newest window start a PUBLISHED live run serves from its T+0."""

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    check_window_shape(hours, cadence)
    for back in range(0, 49):
        run = now - timedelta(hours=back)
        try:
            plan = plan_window(run, hours, cadence)
        except ValueError:
            continue
        if plan.store == "live" and plan.runs[0] == run and not live_missing(run, plan.times):
            return run
    raise ValueError(layered(
        f"--source ukv: no run published in the last two days serves a "
        f"{hours} h window from its own start; name the cycle explicitly.",
        "`--cycle latest` starts the window at the newest complete run.  The "
        "long (T+120) runs are 03Z and 15Z, the others reach T+54 or T+12."))


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


# --------------------------------------------------------------------------
# The live bucket
# --------------------------------------------------------------------------

#: Object field name -> (variable, units, on pressure levels).  The output
#: name is the archive's, so both stores write the same files.
_LIVE_OBJECTS = {
    "temperature_on_pressure_levels": ("air_temperature", "K", True),
    "geopotential_height_on_pressure_levels": ("geopotential_height", "m", True),
    "relative_humidity_on_pressure_levels": ("relative_humidity", "1", True),
    "wind_speed_on_pressure_levels": ("wind_speed", "m s-1", True),
    "wind_direction_on_pressure_levels": ("wind_from_direction", "degrees", True),
    "pressure_at_surface": ("surface_air_pressure", "Pa", False),
    "temperature_at_surface": ("surface_temperature", "K", False),
    "temperature_at_screen_level": ("air_temperature", "K", False),
    "relative_humidity_at_screen_level": ("relative_humidity", "1", False),
    "wind_speed_at_10m": ("wind_speed", "m s-1", False),
    "wind_direction_at_10m": ("wind_from_direction", "degrees", False),
    "snow_depth_water_equivalent": ("lwe_thickness_of_surface_snow_amount", "m", False),
    "height_of_orography": ("surface_altitude", "m", False),
}
_LIVE_OUTPUTS = {
    "temperature_on_pressure_levels": "air_temperature",
    "geopotential_height_on_pressure_levels": "geopotential_height",
    "relative_humidity_on_pressure_levels": "relative_humidity",
    "pressure_at_surface": "surface_air_pressure",
    "temperature_at_surface": "surface_temperature",
    "temperature_at_screen_level": "air_temperature_1p5m",
    "relative_humidity_at_screen_level": "relative_humidity_1p5m",
    "snow_depth_water_equivalent": "lwe_thickness_of_surface_snow_amount",
    "height_of_orography": "surface_altitude",
}
_LIVE_WINDS = {
    ("wind_speed_on_pressure_levels", "wind_direction_on_pressure_levels"):
        ("eastward_wind", "northward_wind"),
    ("wind_speed_at_10m", "wind_direction_at_10m"): ("eastward_wind_10m", "northward_wind_10m"),
}


#: netCDF4/HDF5 is built without thread safety: one decode at a time.
_NETCDF_LOCK = threading.Lock()


def live_key(run: datetime, valid: datetime, product: str) -> str:
    lead = int((valid - run).total_seconds() // 3600)
    return (f"{LIVE_PREFIX}{run:%Y%m%dT%H%MZ}/{valid:%Y%m%dT%H%MZ}"
            f"-PT{lead:04d}H00M-{product}.nc")


def live_listing(run: datetime, valid: datetime) -> set[str]:
    """The object keys of one lead of one run, from the bucket's own listing."""

    import re
    from urllib.parse import urlencode
    keys, token = set(), None
    while True:
        query = {"list-type": "2",
                 "prefix": f"{LIVE_PREFIX}{run:%Y%m%dT%H%MZ}/{valid:%Y%m%dT%H%MZ}-"}
        if token:
            query["continuation-token"] = token
        text = _http_get(f"{LIVE_HOST}/?{urlencode(query)}").decode("utf-8")
        keys.update(re.findall(r"<Key>([^<]+)</Key>", text))
        token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if token is None:
            return keys
        token = token.group(1)


def live_missing(run: datetime, times) -> list[str]:
    """The objects a run would need for ``times`` that it does not hold."""

    missing = []
    for valid in times:
        listed = live_listing(run, valid)
        missing += [key for product in _LIVE_OBJECTS
                    if (key := live_key(run, valid, product)) not in listed]
        if missing:
            # One absent lead decides it; the rest need not be listed.
            break
    return missing


def select_live_run(plan: WindowPlan, pinned: datetime | None = None) -> datetime:
    """The newest covering run that has published every object it needs."""

    now = datetime.utcnow()
    candidates = (pinned,) if pinned is not None else tuple(
        run for run in plan.runs if run <= now)
    if not candidates:
        raise ValueError(layered(
            f"--source ukv: the newest run serving {plan.times[0]:%Y-%m-%dT%H}Z.."
            f"{plan.times[-1]:%Y-%m-%dT%H}Z starts at {plan.runs[0]:%Y-%m-%dT%H}Z, "
            "which has not run yet; fetch after it has posted (a few hours after "
            "its start).",
            f"The window is read from one run of {LIVE_URL}; a run that has not "
            "started has nothing to read."))
    first_missing = None
    for run in candidates:
        missing = live_missing(run, plan.times)
        if not missing:
            return run
        if first_missing is None:
            first_missing = (run, missing)
    run, missing = first_missing
    raise ValueError(layered(
        f"--source ukv: no UKV run serving {plan.times[0]:%Y-%m-%dT%H}Z.."
        f"{plan.times[-1]:%Y-%m-%dT%H}Z is complete in the Met Office bucket; the "
        f"newest ({run:%Y-%m-%dT%H}Z) lacks {len(missing)} object(s), the first "
        f"{missing[0]}.  Wait for the run to finish posting and fetch again.",
        f"The window is read from one run of {LIVE_URL}.  A run posts its leads "
        "over a few hours after its nominal time, and older runs leave the "
        "bucket after two years."))


def read_live(run: datetime, valid: datetime, regrid: Regridder, staging: Path,
              *, workers: int = 4) -> dict[str, np.ndarray]:
    """One lead of one live run on the regular grid, keyed like ``read_ukv``."""

    from netCDF4 import Dataset

    rows, cols = regrid.native_slices()
    staging.mkdir(parents=True, exist_ok=True)

    def read(product):
        variable, units, levelled = _LIVE_OBJECTS[product]
        key = live_key(run, valid, product)
        path = staging / Path(key).name
        path.write_bytes(_http_get(f"{LIVE_HOST}/{key}"))
        # Downloads overlap; HDF5 is not thread-safe, so decoding does not.
        try:
            with _NETCDF_LOCK, Dataset(path) as dataset:
                data = dataset.variables[variable]
                x = np.asarray(dataset.variables["projection_x_coordinate"][:], dtype=np.float64)
                y = np.asarray(dataset.variables["projection_y_coordinate"][:], dtype=np.float64)
                stamp = int(dataset.variables["time"][...])
                reference = int(dataset.variables["forecast_reference_time"][...])
                plane = ("projection_y_coordinate", "projection_x_coordinate")
                layout = (str(getattr(data, "units", "")) == units
                          and tuple(data.dimensions) == ((("pressure",) if levelled else ())
                                                         + plane)
                          and x.size == GRID_NX and y.size == GRID_NY
                          and x[0] == GRID_X0 and y[0] == GRID_Y0
                          and bool(np.all(np.diff(x) == GRID_SPACING))
                          and bool(np.all(np.diff(y) == GRID_SPACING))
                          and stamp == int((valid - datetime(1970, 1, 1)).total_seconds())
                          and reference == int((run - datetime(1970, 1, 1)).total_seconds()))
                levels = None
                if levelled:
                    levels = np.asarray(dataset.variables["pressure"][:], dtype=np.float64)
                    layout = layout and sorted(levels.tolist()) == sorted(
                        float(level) for level in PRESSURE_LEVELS_PA)
                if not layout:
                    raise ValueError(layered(
                        f"{key} does not have the layout this woof measured "
                        f"(units {units}, the 2 km plane, valid {valid:%Y-%m-%dT%H}Z of "
                        f"the {run:%Y-%m-%dT%H}Z run, 33 pressure levels); update woof.",
                        "Every level label, unit and grid index here is taken from the "
                        "measured layout of the Met Office's UKV objects."))
                selection = (slice(None), rows, cols) if levelled else (rows, cols)
                values = np.ma.filled(np.ma.asarray(data[selection], dtype=np.float64), np.nan)
        finally:
            path.unlink(missing_ok=True)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} has missing values inside the domain")
        if levels is not None:
            # The objects store the ladder bottom first; the files carry it
            # in PRESSURE_LEVELS_PA order, matched by value.
            position = {float(level): index for index, level in enumerate(levels)}
            values = values[[position[float(level)] for level in PRESSURE_LEVELS_PA]]
        return values

    with ThreadPoolExecutor(max_workers=workers) as pool:
        native = dict(zip(_LIVE_OBJECTS, pool.map(read, _LIVE_OBJECTS)))
    fields = {output: regrid(native[product]) for product, output in _LIVE_OUTPUTS.items()}
    for (speed, direction), (u_name, v_name) in _LIVE_WINDS.items():
        u, v = wind_components(native[speed], native[direction])
        fields[u_name], fields[v_name] = regrid(u), regrid(v)
    return fields


# --------------------------------------------------------------------------
# The land state
# --------------------------------------------------------------------------

def _http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    from urllib.request import Request
    last = None
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers or {}), timeout=180) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last = error
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                break
            time.sleep(min(4 * (attempt + 1), 20))
    raise RuntimeError(f"read failed after bounded retries: {url}: {last}")


def nearest_axis_indices(axis: np.ndarray, targets: np.ndarray, *,
                         cyclic: bool = False) -> np.ndarray:
    """Per target coordinate, the nearest axis index (lower index on ties)."""

    axis = np.asarray(axis, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    distance = targets[:, None] - axis[None, :]
    if cyclic:
        distance = (distance + 180.0) % 360.0 - 180.0
    return np.argmin(np.abs(distance), axis=1)


def era5_nearest_indices(latitude: np.ndarray, longitude: np.ndarray):
    """Nearest ERA5 row and column per target row and column (lower on ties)."""

    rows = nearest_axis_indices(90.0 - ERA5_STEP * np.arange(ERA5_SHAPE[0]), latitude)
    cols = nearest_axis_indices(ERA5_STEP * np.arange(ERA5_SHAPE[1]), longitude, cyclic=True)
    return rows, cols


def layer_mean_remap(values, source_layers, target_layers) -> list[np.ndarray]:
    """Layer means on ``target_layers`` from overlap-weighted source layers.

    The composition's ``conservative_layer_means`` arithmetic; the source
    must cover every target layer.
    """

    result = []
    for top, bottom in target_layers:
        total = 0.0
        acc = None
        for (s_top, s_bottom), value in zip(source_layers, values):
            overlap = min(bottom, s_bottom) - max(top, s_top)
            if overlap > 0.0:
                acc = value * overlap if acc is None else acc + value * overlap
                total += overlap
        if acc is None or abs(total - (bottom - top)) > 1e-9:
            raise ValueError(f"soil layers do not cover {top}..{bottom} m")
        result.append(acc / total)
    return result


def era5_last_hour() -> datetime:
    """The last hour Google's ARCO ERA5 copy publishes.

    Its own bounds document, read through the registry's ARCO archive
    window (cached, the same reading the date guidance uses); when that
    cannot be read, the window's measured publication lag behind now.
    """

    from woof.source_adapters import get_source_adapter
    from woof.source_availability import published_stop
    window = next(row for row in get_source_adapter("era5").archive_windows
                  if row.transport == "arco")
    stop = published_stop(window)
    if stop is None:
        stop = datetime.utcnow() - timedelta(hours=window.publication_lag_hours or 168.0)
    return stop


def land_state_source(times, *, era5_through: datetime | None = None) -> str:
    """``era5`` when ARCO holds the whole window, else ``gfs``."""

    era5_through = era5_last_hour() if era5_through is None else era5_through
    return "era5" if times[-1] <= era5_through else "gfs"


def read_era5(valid: datetime, latitude: np.ndarray, longitude: np.ndarray,
              *, workers: int = 4) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """ERA5's land state and sea ice at ``valid``, by nearest index, on Noah layers."""

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
    for name, field in values.items():
        if name != "sea_ice_cover" and not np.isfinite(field).all():
            raise ValueError(f"ERA5 {name} has missing values at {valid:%Y-%m-%dT%H}Z inside the domain")
    # ERA5 packs slightly negative water contents over the sea (down to
    # -0.01 measured); a negative volume fraction is not a state, so it is
    # floored at zero and counted.
    water = [values[f"volumetric_soil_water_layer_{k}"] for k in range(1, 5)]
    clamped = int(sum((layer < 0.0).sum() for layer in water))
    water = [np.maximum(layer, 0.0) for layer in water]
    temperature = [values[f"soil_temperature_level_{k}"] for k in range(1, 5)]
    fields = {"land_area_fraction": values["land_sea_mask"],
              "sea_ice_area_fraction": values["sea_ice_cover"]}
    for k, layer in enumerate(layer_mean_remap(temperature, ERA5_SOIL_LAYERS, NOAH_SOIL_LAYERS), 1):
        fields[f"soil_temperature_layer_{k}"] = layer
    for k, layer in enumerate(layer_mean_remap(water, ERA5_SOIL_LAYERS, NOAH_SOIL_LAYERS), 1):
        fields[f"soil_moisture_layer_{k}"] = layer
    receipt = {"source": "era5", "store": ERA5_STORE, "time_index": index,
               "valid_time": valid.isoformat(), "negative_soil_water_floored": clamped,
               "soil_layers": "ERA5 0-7/7-28/28-100/100-289 cm, layer-mean remapped to Noah"}
    return fields, receipt


def gfs_cycle_for(run: datetime) -> datetime:
    """The GFS cycle at or before ``run``: where its land state starts looking."""

    return run.replace(hour=run.hour - run.hour % 6)


def gfs_object(cycle: datetime, lead: int) -> str:
    return (f"{GFS_HOST}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}")


def choose_gfs_cycle(start: datetime, times) -> datetime:
    """The newest GFS cycle at or before ``start`` that has posted ``times``.

    GFS posts three to five hours after its cycle, later than a UKV run
    of the same hour, so the cycle a run rounds down to may not be out
    yet: up to two earlier cycles are asked before refusing.
    """

    from urllib.request import Request
    cycle = gfs_cycle_for(start)
    for _ in range(3):
        url = gfs_object(cycle, gfs_lead(cycle, times[-1])) + ".idx"
        try:
            with urlopen(Request(url, method="HEAD"), timeout=60):
                return cycle
        except HTTPError as error:
            if error.code not in (403, 404):
                raise
        cycle -= timedelta(hours=6)
    raise ValueError(layered(
        f"--source ukv: no GFS cycle at or before {start:%Y-%m-%dT%H}Z has posted "
        f"the land state for {times[-1]:%Y-%m-%dT%H}Z; fetch again later.",
        "Past ERA5's last published hour the soil, land mask and sea ice come "
        f"from GFS ({GFS_HOST}); the three newest cycles were asked."))


def gfs_lead(cycle: datetime, valid: datetime) -> int:
    lead = int((valid - cycle).total_seconds() // 3600)
    # Past f120 GFS posts every third hour; soil moves slowly enough that
    # the earlier posted lead stands in, and the receipt says which.
    return lead if lead <= GFS_HOURLY_THROUGH else lead - lead % 3


def read_gfs(valid: datetime, cycle: datetime, latitude: np.ndarray, longitude: np.ndarray,
             staging: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """GFS's land mask, sea ice and Noah soil at ``valid``, by nearest index."""

    from woof import mapped_source

    lead = gfs_lead(cycle, valid)
    url = gfs_object(cycle, lead)
    index = _http_get(url + ".idx").decode("utf-8").splitlines()
    offsets = [(int(line.split(":")[1]), ":".join(line.split(":")[3:5])) for line in index if line]
    ranges = {}
    for position, (start, record) in enumerate(offsets):
        if record in GFS_RECORDS and record not in ranges:
            end = offsets[position + 1][0] - 1 if position + 1 < len(offsets) else None
            ranges[record] = (start, end)
    missing = [record for record in GFS_RECORDS if record not in ranges]
    if missing:
        raise ValueError(f"GFS {url} lacks {missing}")
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / f"gfs-land-{cycle:%Y%m%dT%H}-f{lead:03d}.grib2"
    with path.open("wb") as stream:
        for record in GFS_RECORDS:
            start, end = ranges[record]
            span = f"bytes={start}-" + ("" if end is None else str(end))
            stream.write(_http_get(url, headers={"Range": span}))
    try:
        inventory_tool, dump_tool = mapped_source._build_grib2_tools()
        rows = mapped_source._grib2_inventory(path, inventory_tool)
        records = mapped_source._grib2_records(
            path, inventory_tool, dump_tool, {int(row["index"]) for row in rows})
    finally:
        path.unlink(missing_ok=True)
    if len(records) != len(GFS_RECORDS):
        raise ValueError(f"GFS {url}: decoded {len(records)} of {len(GFS_RECORDS)} records")
    first = records[0]
    lat_index = nearest_axis_indices(first.latitude, latitude)
    lon_index = nearest_axis_indices(first.longitude, longitude, cyclic=True)
    values = {}
    for record, name in zip(records, GFS_RECORDS):
        field = np.asarray(record.values, dtype=np.float64)[np.ix_(lat_index, lon_index)]
        values[name] = field
    soil = {"TSOIL": "soil_temperature_layer_", "SOILW": "soil_moisture_layer_"}
    fields = {"land_area_fraction": values["LAND:surface"],
              "sea_ice_area_fraction": values["ICEC:surface"]}
    for k, layer in enumerate(("0-0.1", "0.1-0.4", "0.4-1", "1-2"), 1):
        for short, prefix in soil.items():
            fields[f"{prefix}{k}"] = values[f"{short}:{layer} m below ground"]
    land = fields["land_area_fraction"] >= 0.5
    for k in range(1, 5):
        for prefix in soil.values():
            if not np.isfinite(fields[f"{prefix}{k}"][land]).all():
                raise ValueError(f"GFS {url}: soil is missing over land inside the domain")
    receipt = {"source": "gfs", "object": url, "cycle": cycle.isoformat(), "lead": lead,
               "valid_time": valid.isoformat()}
    return fields, receipt


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _units_and_names():
    table = {output: (units, standard) for output, units, standard in
             (*_LEVEL_FIELDS.values(), *_SURFACE_FIELDS.values())}
    table.update(_LAND_STATE_FIELDS)
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
    with _NETCDF_LOCK, Dataset(temporary, "w", format="NETCDF4") as dataset:
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
            missing = name in _MASKED_FIELDS
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


def request_identity(times, area, store: str) -> dict:
    return {"schema": "gpuwm-ukv-request-v2", "source": SOURCE_ID,
            "contract": CONTRACT, "store": STORE_URL if store == "archive" else LIVE_URL,
            "valid_times": [value.isoformat() for value in times],
            "area": area.as_manifest(), "grid_step_degrees": TARGET_STEP_DEGREES,
            "margin_degrees": MARGIN_DEGREES}


def check_prior_request(out: Path, *, cycle: datetime, hours: int,
                        cadence: int | None, area) -> None:
    """Refuse a cache directory that holds another request's files."""

    plan = plan_window(cycle, hours, cadence)
    expected = request_identity(plan.times, area, plan.store)
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


def _file_attributes(store: str, origin: str, land: str, first: datetime) -> dict[str, str]:
    land_store = ERA5_STORE if land == "era5" else GFS_HOST
    terrain = ("the hypsometric height of the "
               f"{first:%Y-%m-%dT%H}Z surface pressure" if store == "archive"
               else "the run's published height_of_orography")
    return {
        "title": "UKV (Met Office UK deterministic 2 km) regridded for woof",
        "institution": ("Met Office (UKV); " + ("ECMWF/Copernicus (ERA5" if land == "era5"
                                                 else "NOAA NCEP (GFS") + " land state)"),
        "source": f"{origin}; land state and sea ice from {land_store}",
        "woof_file_schema": FILE_SCHEMA,
        "land_state_source": land,
        "history": (f"{CONTRACT}: bilinear from the UK 2 km Lambert azimuthal "
                    f"equal-area grid to {TARGET_STEP_DEGREES} degrees; winds from "
                    f"speed and true-north FROM direction; surface_altitude is {terrain}; "
                    f"{land.upper()} land state by nearest index on Noah's four layers"),
        "licence": ("UKV: Met Office, British Crown copyright; "
                    + ("ERA5: Copernicus Climate Change Service, contains modified "
                       "Copernicus information" if land == "era5" else
                       "GFS: NOAA NCEP, United States government work")),
    }


def _read_terrain(path: Path) -> np.ndarray:
    from netCDF4 import Dataset
    with _NETCDF_LOCK, Dataset(path) as dataset:
        return np.asarray(dataset.variables["surface_altitude"][0], dtype=np.float64)


def acquire(*, cycle: datetime, hours: int, cadence: int | None, area,
            out: Path, workers: int = 4, force: bool = False, progress=None) -> dict:
    """Fetch, regrid and publish one window plus its preparation front door.

    An interrupted run of the same request resumes: each finished hour
    carries a stamp naming the request, its bytes, the live run and the
    land-state product it used, and a resume keeps both choices.
    """

    import shutil
    from woof import fetch_routes

    say = progress or (lambda text: print(text, file=sys.stderr))
    if type(workers) is not int or workers <= 0:
        raise ValueError("--fetch-workers must be a positive whole number")
    plan = plan_window(cycle, hours, cadence)
    times = plan.times
    latitude, longitude = target_axes(area)
    regrid = Regridder(latitude, longitude)
    identity = request_identity(times, area, plan.store)
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
        stamps = {valid: _resumable(out / file_name(valid), identity) for valid in times}
        kept = [stamp for stamp in stamps.values() if stamp is not None]
        # A resume keeps the run and the land-state product it began with,
        # so one window never mixes two runs or two soil analyses.
        run = (datetime.fromisoformat(kept[0]["run"]) if kept and kept[0].get("run")
               else None)
        land = kept[0]["land_state"]["source"] if kept else None
        if plan.store == "live" and any(stamp is None for stamp in stamps.values()):
            say(f"fetch ukv: choosing a run of {LIVE_URL}")
            run = select_live_run(plan, run)
        land = land or land_state_source(times)
        gfs_cycle = None
        if land == "gfs" and any(stamp is None for stamp in stamps.values()):
            gfs_cycle = (datetime.fromisoformat(kept[0]["land_state"]["cycle"]) if kept
                         else choose_gfs_cycle(run or times[0], times))
        origin = (f"{LIVE_URL} run {run:%Y-%m-%dT%H}Z" if plan.store == "live" else STORE_URL)
        say(f"fetch ukv: {plan.store} store"
            + (f", the {run:%Y-%m-%dT%H}Z run" if run is not None else "")
            + f"; land state from {land.upper()}")
        group = indices = None
        staging = out / ".ukv-staging"
        terrain = None
        files, land_receipts = [], []
        try:
            with ThreadPoolExecutor(max_workers=1) as background:
                for number, valid in enumerate(times, start=1):
                    path = out / file_name(valid)
                    stamp = stamps[valid]
                    if stamp is not None:
                        say(f"fetch ukv: {valid:%Y-%m-%dT%H}Z {number}/{len(times)} kept "
                            "from an interrupted run")
                        files.append(stamp["entry"])
                        land_receipts.append(stamp["land_state"])
                        if terrain is None:
                            terrain = _read_terrain(path)
                        continue
                    say(f"fetch ukv: {valid:%Y-%m-%dT%H}Z {number}/{len(times)} reading UKV "
                        f"and {land.upper()}")
                    # The two archives are independent reads: the land state
                    # is read beside the atmosphere.
                    if land == "era5":
                        land_read = background.submit(read_era5, valid, latitude, longitude,
                                                      workers=workers)
                    else:
                        land_read = background.submit(
                            read_gfs, valid, gfs_cycle, latitude, longitude, staging / "gfs")
                    record = {"valid_time": valid.isoformat()}
                    if plan.store == "archive":
                        if group is None:
                            say(f"fetch ukv: opening {STORE_URL}")
                            group, snapshot = open_store()
                            indices = store_indices(group, times)
                            origin = f"{STORE_URL} snapshot {snapshot}"
                        index, direction_index = indices[valid]
                        fields = read_ukv(group, index, direction_index, regrid, workers=workers)
                        record.update(store_record=index, wind_direction_record=direction_index,
                                      store_snapshot=snapshot)
                    else:
                        fields = read_live(run, valid, regrid, staging, workers=workers)
                        published = fields.pop("surface_altitude")
                        record.update(run=run.isoformat(),
                                      lead=int((valid - run).total_seconds() // 3600))
                    land_fields, receipt = land_read.result()
                    if terrain is None:
                        # One terrain for the whole window, so every file's
                        # in-band supplement is byte-identical: the run's own
                        # orography live, the first hour's derived height in
                        # the archive, which publishes none.
                        terrain = published if plan.store == "live" else surface_height(
                            PRESSURE_LEVELS_PA, fields["geopotential_height"],
                            fields["air_temperature"], fields["relative_humidity"],
                            fields["surface_air_pressure"], fields["air_temperature_1p5m"],
                            fields["relative_humidity_1p5m"])
                    fields.update(land_fields)
                    fields["surface_altitude"] = terrain
                    write_netcdf(path, valid, latitude, longitude, fields,
                                 _file_attributes(plan.store, origin, land, times[0]))
                    entry = {"path": path.name, "bytes": path.stat().st_size,
                             "sha256": _sha256(path), **record}
                    _json(_stamp(path), {"request": identity, "entry": entry,
                                         "land_state": receipt,
                                         "run": None if run is None else run.isoformat()})
                    files.append(entry)
                    land_receipts.append(receipt)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        _publish_front_door(out, files, times[0])
        manifest = {
            "schema": MANIFEST_SCHEMA, "source": SOURCE_ID, "contract": CONTRACT,
            "request": identity, "store": plan.store,
            "run": None if run is None else run.isoformat(),
            "store_snapshots": sorted({entry["store_snapshot"] for entry in files
                                       if "store_snapshot" in entry}),
            "files": files, "valid_times": identity["valid_times"],
            "grid": {"latitude": [float(latitude[0]), float(latitude[-1]), len(latitude)],
                     "longitude": [float(longitude[0]), float(longitude[-1]), len(longitude)],
                     "native_window": list(regrid.window)},
            "terrain": {"from": ("derived from the surface pressure at " + times[0].isoformat()
                                 if plan.store == "archive" else "published orography"),
                        "min_m": float(terrain.min()), "max_m": float(terrain.max())},
            "land_state_source": land, "land_state": land_receipts,
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
