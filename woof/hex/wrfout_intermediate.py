"""A WOOF WRF run's own wrfout history onto the regular lat-lon WPS
intermediate the hex init and boundary engines read:
``woof hex intermediate --source wrfout --wrfout-glob GLOB``.

WHAT THIS IS FOR.  One-way forcing of a hex limited-area run from a WRF run
this tree already made: a 1-3 km ``wrf-nests`` / ``wrf-tiles`` parent from
``woof energy`` drives a 100 m hex corridor through ``woof hex init`` and
``woof hex lbc``, exactly as a GRIB source does.  The parent's output never
leaves its own numbers: no reanalysis is interposed.

THE ROAD, and what is borrowed from where.

* READ -- ``netCDF4`` hyperslab reads of only the source window every target
  stencil touches (plus one staggered column/row for U/V), per time.  The
  projection is rebuilt from the file's own global attributes with
  :func:`woof.static.projection.projection_class` (Lambert, Mercator, polar
  stereographic: the WPS ``module_llxy`` transcription) and is proved against
  the file's own ``XLAT``/``XLONG`` before anything is moved; a file whose
  attributes do not reproduce its coordinates is refused.
* DERIVE -- on the source grid, in FP64: U/V destaggered to mass points and
  rotated to the earth basis with the file's own ``SINALPHA``/``COSALPHA``;
  ``theta = T + 300``; ``p = P + PB``; ``TT = theta (p / 1e5)^(R/cp)`` with
  WRF's ``R/cp = 2/7``; geopotential height ``(PH + PHB) / g`` averaged to mass
  levels with WRF's ``g = 9.81``; mixing ratios become specific humidity
  ``q / (1 + q)`` (3-D and 2 m).
* REGRID -- the engine's projected-source operator
  (``woof.ingest.hrrr._ProjectedCpuPlan``: WPS overlapping-parabolic with FP64
  donor selection, run by the Rust indexed-donor entry when the CPU bridge
  exports it), handed this grid's donors instead of HRRR's.  Every record is
  then produced by the HRRR route's own :func:`regrid_hour` from the
  ``wrfout`` row of :data:`woof.hex.hrrr_intermediate.SOURCE_ROWS`.
* WRITE -- :func:`woof.hex.hrrr_intermediate.write_intermediate`: the WPS
  version-5 layout, read back through this tree's reader before the receipt
  is signed.

WHAT THE RESULT IS.  Every WRF mass level is carried as a level-indexed slab
(``xlvl = 1..nz``, bottom first) with the 3-D ``PRESSURE`` beside it, plus the
surface level 200100; soil is the four Noah layers WRF itself ran, so the
init reads exactly the column the parent integrated.  Cloud water, rain, ice,
snow and graupel are written (``QC QR QI QS QG``) when the parent carried
them; ``rw_mpas_init`` and ``rw_mpas_lbc`` read past them today (the same
statement the HRRR receipt makes), so they are there for a later init, not
used by this one.

SUB-HOURLY HISTORY.  The init and boundary engines take each file's valid time
from its own header and ``woof hex lbc`` passes every file by path with its
time, so a sub-hourly series is readable end to end.  Files are named
``PREFIX:YYYY-MM-DD_HH`` when every time is on the hour and
``PREFIX:YYYY-MM-DD_HH:MM`` (ungrib's own sub-hourly spelling) otherwise.
MPAS Fortran's ``config_met_prefix`` lookup builds hourly names only; nothing
in this chain uses that lookup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import glob as globbing
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .hrrr_intermediate import (
    DEFAULT_MARGIN_KM,
    INTERMEDIATE_SCHEMA,
    KM_PER_DEG,
    IntermediateRefusal,
    LatLonTarget,
    SourceRow,
    SourceWindow,
    regrid_hour,
    sha256_file,
    source_row,
    write_intermediate,
)

#: WRF ``module_model_constants``: r_d = 287, cp = 7 r_d / 2, g = 9.81, and
#: the perturbation potential temperature offset t0 = 300.
WRF_RCP = 2.0 / 7.0
WRF_G = 9.81
WRF_P0 = 1.0e5
WRF_T0 = 300.0
#: The Noah / Noah-MP soil column WRF runs: layer thicknesses (m).  The four
#: ``ST/SM`` names of the row ARE these layers, so a parent on another land
#: model (RUC's nine levels, PX's two, the slab's five) is refused rather than
#: re-layered by a guess.
NOAH_DZS_M = (0.1, 0.3, 0.6, 1.0)
NOAH_LAND_MODELS = {2: "Noah", 4: "Noah-MP"}
#: Rows next to the WRF lateral boundary that carry relaxed boundary data,
#: not the parent's own solution (WRF's default spec_bdy_width).
DEFAULT_WRF_EDGE_CELLS = 5
#: The engine operator's four-point stencil reaches one cell before the donor
#: and two after it.
STENCIL_BEFORE = 1
STENCIL_AFTER = 2
#: How far the rebuilt projection may sit from the file's own XLAT/XLONG
#: (float32 in the file: ~4e-6 deg measured on a 1 km polar-stereographic
#: run); a wrong projection is off by whole cells.
COORDINATE_TOLERANCE_DEG = 1.0e-3

REQUIRED_VARIABLES = (
    "Times", "XLAT", "XLONG", "SINALPHA", "COSALPHA",
    "U", "V", "T", "P", "PB", "PH", "PHB", "QVAPOR",
    "PSFC", "T2", "Q2", "U10", "V10", "TSK", "HGT",
    "LANDMASK", "SEAICE", "SNOW", "TSLB", "SMOIS",
)
#: wrfout name -> the row's window key, for what a parent may or may not carry.
OPTIONAL_3D = {"QCLOUD": "QC", "QRAIN": "QR", "QICE": "QI", "QSNOW": "QS", "QGRAUP": "QG"}
OPTIONAL_SURFACE = {"SST": "SST", "SNOWH": "SNOWH"}
REQUIRED_ATTRIBUTES = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
    "DX", "DY", "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
)
_MAP_PROJ_NAMES = {1: "lambert", 2: "polar", 3: "mercator"}


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------
def _netcdf():
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - the env carries it
        raise IntermediateRefusal(
            f"netCDF4 is not importable ({error}); a wrfout cannot be read without it"
        ) from error
    return netCDF4


@dataclass(frozen=True)
class WrfGrid:
    """One WRF domain as its wrfout declares it."""

    map_proj: int
    truelat1: float
    truelat2: float
    stand_lon: float
    cen_lat: float
    cen_lon: float
    dx: float
    dy: float
    e_we: int
    e_sn: int
    nz: int
    soil_layers: int
    land_model: int | None

    @property
    def nx(self) -> int:
        return self.e_we - 1

    @property
    def ny(self) -> int:
        return self.e_sn - 1

    def projection(self):
        from woof.static.projection import projection_class

        name = _MAP_PROJ_NAMES.get(self.map_proj)
        if name is None:
            raise IntermediateRefusal(
                f"wrfout MAP_PROJ {self.map_proj} is not Lambert (1), polar "
                f"stereographic (2) or Mercator (3); a latitude-longitude or "
                f"rotated WRF grid is not a projection this door rebuilds"
            )
        return projection_class(name)(
            self.cen_lat, self.cen_lon, self.truelat1, self.truelat2, self.stand_lon,
            self.dx, self.dy, self.e_we, self.e_sn,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "map_proj": self.map_proj, "projection": _MAP_PROJ_NAMES.get(self.map_proj),
            "truelat1": self.truelat1, "truelat2": self.truelat2,
            "stand_lon": self.stand_lon, "cen_lat": self.cen_lat, "cen_lon": self.cen_lon,
            "dx_m": self.dx, "dy_m": self.dy, "mass_shape": [self.ny, self.nx],
            "levels": self.nz, "soil_layers": self.soil_layers,
            "land_model": NOAH_LAND_MODELS.get(self.land_model or -1, self.land_model),
        }


@dataclass(frozen=True)
class WrfFrame:
    """One wrfout time: which file, which record."""

    valid_time: datetime
    path: Path
    index: int


def _grid_of(dataset: Any, path: Path) -> WrfGrid:
    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(dataset, name)]
    if missing:
        raise IntermediateRefusal(
            f"{path} carries no global attribute {missing}; it is not a WRF "
            f"wrfout and its projection cannot be rebuilt"
        )
    dims = dataset.dimensions
    for name in ("bottom_top", "west_east", "south_north", "soil_layers_stag"):
        if name not in dims:
            raise IntermediateRefusal(f"{path} has no {name} dimension; it is not a wrfout")
    land = getattr(dataset, "SF_SURFACE_PHYSICS", None)
    grid = WrfGrid(
        map_proj=int(dataset.MAP_PROJ),
        truelat1=float(dataset.TRUELAT1), truelat2=float(dataset.TRUELAT2),
        stand_lon=float(dataset.STAND_LON),
        cen_lat=float(dataset.CEN_LAT), cen_lon=float(dataset.CEN_LON),
        dx=float(dataset.DX), dy=float(dataset.DY),
        e_we=int(getattr(dataset, "WEST-EAST_GRID_DIMENSION")),
        e_sn=int(getattr(dataset, "SOUTH-NORTH_GRID_DIMENSION")),
        nz=len(dims["bottom_top"]), soil_layers=len(dims["soil_layers_stag"]),
        land_model=None if land is None else int(land),
    )
    if (len(dims["west_east"]), len(dims["south_north"])) != (grid.nx, grid.ny):
        raise IntermediateRefusal(
            f"{path}: the west_east/south_north dimensions "
            f"({len(dims['west_east'])}x{len(dims['south_north'])}) disagree with the "
            f"grid-dimension attributes ({grid.nx}x{grid.ny})"
        )
    return grid


def _times_of(dataset: Any, path: Path) -> list[datetime]:
    variable = dataset.variables["Times"]
    variable.set_auto_chartostring(False)
    raw = np.asarray(variable[:])
    if raw.dtype.kind != "S":
        raise IntermediateRefusal(f"{path}: Times is not a character variable")
    out: list[datetime] = []
    for row in np.atleast_2d(raw):
        stamp = row.tobytes().decode("ascii", errors="replace").strip("\x00 ")
        try:
            out.append(datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S"))
        except ValueError as error:
            raise IntermediateRefusal(
                f"{path}: Times entry {stamp!r} is not YYYY-MM-DD_HH:MM:SS"
            ) from error
    return out


def scan_wrfout(paths: Sequence[Path]) -> tuple[WrfGrid, list[WrfFrame], list[dict[str, Any]], dict[str, bool]]:
    """Every time in every file, refused on a mixed grid, a missing variable
    or a duplicate time."""

    if not paths:
        raise IntermediateRefusal("no wrfout file was named")
    netCDF4 = _netcdf()
    grid: WrfGrid | None = None
    frames: list[WrfFrame] = []
    sources: list[dict[str, Any]] = []
    carried: dict[str, bool] | None = None
    seen: dict[datetime, Path] = {}
    for path in paths:
        try:
            dataset = netCDF4.Dataset(str(path), "r")
        except OSError as error:
            raise IntermediateRefusal(f"{path} is not a readable netCDF file: {error}") from error
        with dataset:
            missing = [name for name in REQUIRED_VARIABLES if name not in dataset.variables]
            if missing:
                raise IntermediateRefusal(
                    f"{path} carries no {', '.join(missing)}; the intermediate needs every "
                    f"one of {', '.join(REQUIRED_VARIABLES)} and nothing here invents one.  "
                    f"Add the variables to the parent's history stream (WRF registry "
                    f"'h' io) and re-run it"
                )
            this_grid = _grid_of(dataset, path)
            if grid is None:
                grid = this_grid
            elif this_grid != grid:
                raise IntermediateRefusal(
                    f"{path} is a different WRF grid from the first file "
                    f"({this_grid.as_dict()} vs {grid.as_dict()}); one door call "
                    f"converts one domain"
                )
            present = {name: name in dataset.variables for name in (*OPTIONAL_3D, *OPTIONAL_SURFACE)}
            if carried is None:
                carried = present
            else:
                carried = {name: carried[name] and present[name] for name in carried}
            times = _times_of(dataset, path)
        for index, valid in enumerate(times):
            if valid in seen:
                raise IntermediateRefusal(
                    f"{valid:%Y-%m-%d_%H:%M:%S} appears in both {seen[valid]} and {path}"
                    + (f" (record {index})" if seen[valid] == path else "")
                    + "; two first guesses for one valid time would be one silently "
                      "dropped.  Narrow --wrfout-glob to one run"
                )
            if valid.second or valid.microsecond:
                raise IntermediateRefusal(
                    f"{path}: valid time {valid:%Y-%m-%d_%H:%M:%S} is not on a whole minute; "
                    f"neither the WPS file name nor the boundary interval can carry it"
                )
            seen[valid] = path
            frames.append(WrfFrame(valid_time=valid, path=path, index=index))
        sources.append({"path": str(path), "bytes": path.stat().st_size,
                        "times": [f"{t:%Y-%m-%d_%H:%M:%S}" for t in times]})
    assert grid is not None and carried is not None
    frames.sort(key=lambda frame: frame.valid_time)
    return grid, frames, sources, carried


def sign_sources(sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sha256 of every wrfout, taken after every cheap refusal has run:
    hashing a glob of multi-GB history first would make a refusal of the
    cull or the soil column arrive minutes late."""

    return [{**source, "sha256": sha256_file(Path(source["path"]))} for source in sources]


def refuse_stale_out_dir(out_dir: Path) -> None:
    """A directory that already holds intermediates is refused: `woof hex lbc`
    reads every WPS file in it by its header, so a survivor from an earlier
    run would join this run's boundary series without saying so."""

    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise IntermediateRefusal(f"--out-dir {out_dir} exists and is not a directory")
    from .init_door import _probe_wps_header

    held = sorted(p.name for p in out_dir.iterdir() if p.is_file() and _probe_wps_header(p))
    if held or (out_dir / "intermediate-receipt.json").exists():
        raise IntermediateRefusal(
            f"--out-dir {out_dir} already holds intermediates "
            f"({', '.join(held[:4]) or 'intermediate-receipt.json'}"
            f"{', ...' if len(held) > 4 else ''}); `woof hex lbc --met-dir` would read "
            f"them beside this run's.  Pass a fresh --out-dir"
        )


def resolve_glob(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in globbing.glob(str(Path(pattern).expanduser())))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise IntermediateRefusal(f"--wrfout-glob {pattern!r} matches no file")
    return paths


# ---------------------------------------------------------------------------
# the projection, proved against the file
# ---------------------------------------------------------------------------
def prove_projection(grid: WrfGrid, path: Path) -> tuple[Any, dict[str, float]]:
    """The rebuilt projection, refused unless it reproduces XLAT/XLONG."""

    netCDF4 = _netcdf()
    projection = grid.projection()
    with netCDF4.Dataset(str(path), "r") as dataset:
        dataset.set_auto_mask(False)
        lat = np.asarray(dataset.variables["XLAT"][0], dtype=np.float64)
        lon = np.asarray(dataset.variables["XLONG"][0], dtype=np.float64)
        sina = np.asarray(dataset.variables["SINALPHA"][0], dtype=np.float64)
        cosa = np.asarray(dataset.variables["COSALPHA"][0], dtype=np.float64)
    j, i = np.mgrid[0:grid.ny, 0:grid.nx]
    rebuilt_lat, rebuilt_lon = projection.ij_to_latlon(i + 1.0, j + 1.0)
    dlat = float(np.max(np.abs(np.asarray(rebuilt_lat) - lat)))
    dlon = float(np.max(np.abs((np.asarray(rebuilt_lon) - lon + 180.0) % 360.0 - 180.0)))
    rotation = float(np.max(np.abs(sina * sina + cosa * cosa - 1.0)))
    if not (dlat <= COORDINATE_TOLERANCE_DEG and dlon <= COORDINATE_TOLERANCE_DEG):
        raise IntermediateRefusal(
            f"{path}: the projection rebuilt from its global attributes misses the "
            f"file's own XLAT/XLONG by {dlat:.3g} / {dlon:.3g} deg (tolerance "
            f"{COORDINATE_TOLERANCE_DEG:g}); every regridded point would be misplaced"
        )
    if not rotation <= 1.0e-3:
        raise IntermediateRefusal(
            f"{path}: SINALPHA^2 + COSALPHA^2 departs from 1 by {rotation:.3g}; the "
            f"wind rotation the file carries is not a rotation"
        )
    return projection, {"max_lat_error_deg": dlat, "max_lon_error_deg": dlon,
                        "max_rotation_norm_error": rotation}


# ---------------------------------------------------------------------------
# the target box
# ---------------------------------------------------------------------------
def _cap_bounds(centre: tuple[float, float], reach_km: float) -> tuple[float, float, float, float]:
    lat, lon = centre
    half_lat = reach_km / KM_PER_DEG
    north = lat + half_lat
    south = lat - half_lat
    cos_lat = max(math.cos(math.radians(max(abs(south), abs(north)))), 0.05)
    half_lon = reach_km / (KM_PER_DEG * cos_lat)
    return south, lon - half_lon, north, lon + half_lon


def _polygon_bounds(vertices: Sequence[Sequence[float]], margin_km: float) -> tuple[float, float, float, float]:
    if len(vertices) < 3:
        raise IntermediateRefusal("a polygon cull region needs at least three vertices")
    lats = [float(v[0]) for v in vertices]
    first = float(vertices[0][1])
    lons = [first + ((float(v[1]) - first + 180.0) % 360.0 - 180.0) for v in vertices]
    half_lat = margin_km / KM_PER_DEG
    south, north = min(lats) - half_lat, max(lats) + half_lat
    cos_lat = max(math.cos(math.radians(max(abs(south), abs(north)))), 0.05)
    half_lon = margin_km / (KM_PER_DEG * cos_lat)
    return south, min(lons) - half_lon, north, max(lons) + half_lon


def region_bounds(region: Mapping[str, Any], *, halo_km: float, margin_km: float,
                  origin: str) -> tuple[float, float, float, float]:
    kind = region.get("kind")
    if kind == "cap":
        centre = (float(region["center_deg"][0]), float(region["center_deg"][1]))
        return _cap_bounds(centre, float(region["radius_km"]) + halo_km + margin_km)
    if kind == "polygon":
        return _polygon_bounds(region.get("vertices_deg") or [], halo_km + margin_km)
    raise IntermediateRefusal(
        f"{origin} is a {kind!r} region; this door reads a cap "
        f"({{'kind': 'cap', 'center_deg', 'radius_km'}}) or a polygon "
        f"({{'kind': 'polygon', 'vertices_deg'}}), the two shapes `woof hex cull` "
        f"and `woof energy plan` write"
    )


def bounds_from_arguments(arguments: argparse.Namespace) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    """The lat-lon box the cull needs, from exactly one of the three sources."""

    given = [flag for flag, value in (("--from-plan", arguments.from_plan),
                                      ("--cull-region", getattr(arguments, "cull_region", None)),
                                      ("--point", arguments.point)) if value is not None]
    if len(given) > 1:
        raise IntermediateRefusal(f"{' and '.join(given)} were both given; the box comes from one")
    margin = float(arguments.margin_km)
    if not margin >= 0.0:
        raise IntermediateRefusal(f"--margin-km {margin} is negative")
    if arguments.from_plan is not None or getattr(arguments, "cull_region", None) is not None:
        path = Path(arguments.from_plan if arguments.from_plan is not None else arguments.cull_region)
        flag = given[0]
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntermediateRefusal(f"{flag} {path} is not readable JSON: {error}") from error
        if not isinstance(document, dict):
            raise IntermediateRefusal(f"{flag} {path} is not a JSON object")
        if flag == "--from-plan":
            if arguments.halo_km is not None:
                raise IntermediateRefusal(
                    "--halo-km was given with --from-plan; the plan carries its own halo"
                )
            plan = document.get("plan", document)
            region = plan.get("cull_region") or {}
            halo = float((plan.get("cull") or {}).get("halo_km") or 0.0)
        else:
            if arguments.radius_km is not None:
                raise IntermediateRefusal("--radius-km belongs to --point; a cull region has its own shape")
            if arguments.halo_km is None:
                raise IntermediateRefusal(
                    "--cull-region needs --halo-km: the boundary rings outside the cut have a "
                    "width the region file does not carry.  Pass the plan's halo_km (the "
                    "cull estimate `woof energy plan` prints), or 0 if the region already "
                    "includes the rings"
                )
            region = document.get("cull_region", document)
            halo = float(arguments.halo_km)
        if not isinstance(region, dict):
            raise IntermediateRefusal(
                f"{flag} {path}: cull_region is {type(region).__name__} {region!r}, not a "
                f"cap or polygon object; pass the cull_region.json itself"
            )
        if not halo >= 0.0:
            raise IntermediateRefusal(f"the halo {halo} km is negative")
        bounds = region_bounds(region, halo_km=halo, margin_km=margin, origin=f"{flag} {path}")
        basis: dict[str, Any] = {flag.lstrip("-").replace("-", "_"): str(path),
                                 "region_kind": region.get("kind"), "halo_km": halo}
    elif arguments.point is not None:
        from .mesh_point import parse_point

        if arguments.radius_km is None:
            raise IntermediateRefusal("--point needs --radius-km: the box has no reach without one")
        centre = parse_point(arguments.point)
        halo = 0.0 if arguments.halo_km is None else float(arguments.halo_km)
        if not halo >= 0.0:
            raise IntermediateRefusal(f"--halo-km {halo} is negative")
        bounds = _cap_bounds(centre, float(arguments.radius_km) + halo + margin)
        basis = {"centre_deg": list(centre), "reach_km": float(arguments.radius_km),
                 "halo_km": halo}
    else:
        raise IntermediateRefusal(
            "neither --from-plan nor --point (nor --cull-region) was given; the "
            "intermediate covers a box and nothing here guesses one"
        )
    south, west, north, east = bounds
    if south < -90.0 or north > 90.0:
        raise IntermediateRefusal(
            f"the box {south:.3f}..{north:.3f} N crosses a pole; a regular lat-lon "
            f"intermediate cannot carry it"
        )
    basis["margin_km"] = margin
    return bounds, basis


def target_for_bounds(bounds: tuple[float, float, float, float], *, dlat: float, dlon: float) -> LatLonTarget:
    south, west, north, east = bounds
    if not (dlat > 0.0 and dlon > 0.0):
        raise IntermediateRefusal(f"target spacing {dlat} x {dlon} deg is not positive")
    ny = int(math.ceil((north - south) / dlat)) + 1
    nx = int(math.ceil((east - west) / dlon)) + 1
    centre_lat = 0.5 * (south + north)
    centre_lon = 0.5 * (west + east)
    south = round(centre_lat - 0.5 * (ny - 1) * dlat, 6)
    west = round(centre_lon - 0.5 * (nx - 1) * dlon, 6)
    if west < -180.0:
        west += 360.0
    elif west >= 180.0:
        west -= 360.0
    return LatLonTarget(south=south, west=west, dlat=dlat, dlon=dlon, nx=nx, ny=ny)


def spacing_for(grid: WrfGrid, bounds: tuple[float, float, float, float],
                explicit_deg: float | None) -> tuple[float, float, str]:
    """Lat-lon spacing comparable to the WRF dx: dx meridionally, the same
    ground distance zonally at the box's centre latitude."""

    if explicit_deg is not None:
        if not explicit_deg > 0.0:
            raise IntermediateRefusal(f"--spacing-deg {explicit_deg} is not positive")
        return float(explicit_deg), float(explicit_deg), "--spacing-deg"
    dlat = round(min(grid.dx, grid.dy) / 1000.0 / KM_PER_DEG, 6)
    centre = 0.5 * (bounds[0] + bounds[2])
    dlon = round(dlat / max(math.cos(math.radians(centre)), 0.05), 6)
    return dlat, dlon, f"WRF dx {grid.dx:g} m at the box centre latitude {centre:.3f}"


# ---------------------------------------------------------------------------
# the window and the plan
# ---------------------------------------------------------------------------
def window_for(projection: Any, grid: WrfGrid, target: LatLonTarget, *,
               edge_cells: int) -> tuple[SourceWindow, np.ndarray, np.ndarray]:
    """The source window every target stencil needs, refused unless it lies
    inside the WRF interior (the relaxed boundary rows excluded)."""

    lat, lon = target.mesh()
    x, y = projection.latlon_to_ij(lat, lon)
    zero_x = np.asarray(x, dtype=np.float64) - 1.0
    zero_y = np.asarray(y, dtype=np.float64) - 1.0
    if not (np.isfinite(zero_x).all() and np.isfinite(zero_y).all()):
        raise IntermediateRefusal(
            "the target box maps to non-finite WRF grid coordinates; it lies "
            "outside the projection's domain"
        )
    window = SourceWindow(
        i_start=int(np.floor(zero_x.min())) - STENCIL_BEFORE,
        i_end=int(np.floor(zero_x.max())) + STENCIL_AFTER,
        j_start=int(np.floor(zero_y.min())) - STENCIL_BEFORE,
        j_end=int(np.floor(zero_y.max())) + STENCIL_AFTER,
    )
    low = int(edge_cells)
    high_i = grid.nx - 1 - int(edge_cells)
    high_j = grid.ny - 1 - int(edge_cells)
    if window.i_start < low or window.j_start < low or window.i_end > high_i or window.j_end > high_j:
        raise IntermediateRefusal(
            f"the target box ({target.south:.3f}..{target.north:.3f} N, "
            f"{target.west:.3f}..{target.east:.3f} E) plus its interpolation stencil "
            f"needs WRF mass cells i={window.i_start}..{window.i_end}, "
            f"j={window.j_start}..{window.j_end}, and the wrfout interior is "
            f"i={low}..{high_i}, j={low}..{high_j} ({grid.nx}x{grid.ny} cells less "
            f"{edge_cells} relaxed boundary rows per side).  The cull is not inside "
            f"the parent: shrink --margin-km or the cull, or force it from a parent "
            f"domain that covers it"
        )
    return window, zero_x, zero_y


def projected_plan(zero_x: np.ndarray, zero_y: np.ndarray, window: SourceWindow, backend: Any) -> Any:
    """The engine's projected operator over this grid's donors.

    ``_ProjectedCpuPlan`` resolves its donors from the HRRR projection in its
    constructor; everything after that -- donor/fraction carriage, the Rust
    indexed-donor entry, the NumPy reference, the nearest pick -- reads only
    the attributes set below.  They are set exactly as that constructor sets
    them (``x = global_x - i_start``, ``ix = floor(global_x) - i_start``,
    ``fx = global_x - floor(global_x)`` in FP32), so the operator runs
    unchanged on a WRF grid.
    """

    try:
        from woof.ingest import hrrr as engine
    except ImportError as error:
        raise IntermediateRefusal(
            f"the engine's ingest is not importable ({error}); this door drives the "
            f"engine's own projected operator and refuses to substitute one"
        ) from error
    cls = getattr(engine, "_ProjectedCpuPlan", None)
    announce = getattr(engine, "_announce_projected_numpy_fallback", None)
    if cls is None or announce is None:
        raise IntermediateRefusal(
            "the installed engine's ingest does not expose _ProjectedCpuPlan; this "
            "road drives the operator the engine ships"
        )
    global_ix = np.floor(zero_x).astype(np.int64)
    global_iy = np.floor(zero_y).astype(np.int64)
    plan = cls.__new__(cls)
    plan.route = "interpolated"
    plan.source_shape = (window.ny, window.nx)
    plan.target_shape = tuple(map(int, zero_x.shape))
    plan.x_host = zero_x - window.i_start
    plan.y_host = zero_y - window.j_start
    plan.ix = (global_ix - window.i_start).astype(np.int32)
    plan.iy = (global_iy - window.j_start).astype(np.int32)
    plan.fx = np.asarray(zero_x - global_ix, dtype=np.float32)
    plan.fy = np.asarray(zero_y - global_iy, dtype=np.float32)
    plan.nearest_ix = (np.floor(zero_x + 0.5).astype(np.int64) - window.i_start).astype(np.int32)
    plan.nearest_iy = (np.floor(zero_y + 0.5).astype(np.int64) - window.j_start).astype(np.int32)
    plan._native = None
    plan.operator = engine.PROJECTED_OPERATOR_NUMPY
    if (int(plan.ix.min()) - STENCIL_BEFORE < 0 or int(plan.ix.max()) + STENCIL_AFTER >= window.nx
            or int(plan.iy.min()) - STENCIL_BEFORE < 0 or int(plan.iy.max()) + STENCIL_AFTER >= window.ny):
        raise IntermediateRefusal("the WRF window lacks the four-point interpolation halo")
    builder = getattr(backend, "indexed_donor_plan", None)
    if builder is None:
        announce("this preprocessing backend has no indexed-donor plan")
    elif not getattr(backend, "indexed_donor_interp", False):
        announce("the loaded CPU preprocessing bridge does not export gpuwm_indexed_interp_f32")
    else:
        plan._native = builder(plan.source_shape, plan.iy, plan.ix, plan.fy, plan.fx)
        plan.operator = engine.PROJECTED_OPERATOR_RUST
    return plan


# ---------------------------------------------------------------------------
# one time, on the source grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WrfSnapshot:
    """One time's derived fields on the window, under the row's window keys."""

    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    sina: np.ndarray
    cosa: np.ndarray


def read_snapshot(frame: WrfFrame, window: SourceWindow, carried: Mapping[str, bool]) -> WrfSnapshot:
    netCDF4 = _netcdf()
    t = frame.index
    js = slice(window.j_start, window.j_end + 1)
    is_ = slice(window.i_start, window.i_end + 1)
    with netCDF4.Dataset(str(frame.path), "r") as dataset:
        dataset.set_auto_mask(False)
        v = dataset.variables

        def mass(name: str) -> np.ndarray:
            data = v[name]
            if data.ndim == 4:
                return np.asarray(data[t, :, js, is_], dtype=np.float64)
            return np.asarray(data[t, js, is_], dtype=np.float64)

        u_stag = np.asarray(v["U"][t, :, js, window.i_start:window.i_end + 2], dtype=np.float64)
        v_stag = np.asarray(v["V"][t, :, window.j_start:window.j_end + 2, is_], dtype=np.float64)
        ph = np.asarray(v["PH"][t, :, js, is_], dtype=np.float64) \
            + np.asarray(v["PHB"][t, :, js, is_], dtype=np.float64)
        pressure = mass("P") + mass("PB")
        theta = mass("T") + WRF_T0
        qv = np.maximum(mass("QVAPOR"), 0.0)
        q2 = np.maximum(mass("Q2"), 0.0)
        sina = mass("SINALPHA")
        cosa = mass("COSALPHA")
        surface = {key: mass(name) for name, key in (
            ("T2", "T2"), ("PSFC", "PSFC"), ("TSK", "TSK"), ("HGT", "TERRAIN"),
            ("SNOW", "SNOW"), ("LANDMASK", "LANDMASK"), ("SEAICE", "SEAICE"),
            ("U10", "U10"), ("V10", "V10"),
        )}
        soil_t = mass("TSLB")
        soil_m = mass("SMOIS")
        optional = {key: mass(name) for name, key in {**OPTIONAL_3D, **OPTIONAL_SURFACE}.items()
                    if carried.get(name)}
    if ph.shape[0] != pressure.shape[0] + 1:
        raise IntermediateRefusal(
            f"{frame.path}: PH has {ph.shape[0]} levels for {pressure.shape[0]} mass levels; "
            f"it must have one more (the w-staggered column)"
        )
    if (pressure <= 0.0).any():
        raise IntermediateRefusal(f"{frame.path}: P + PB is not positive everywhere")
    u_mass = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
    v_mass = 0.5 * (v_stag[..., :-1, :] + v_stag[..., 1:, :])
    fields: dict[str, np.ndarray] = {
        "P_FULL": pressure,
        "Z_MASS": 0.5 * (ph[:-1] + ph[1:]) / WRF_G,
        "TEMP": theta * (pressure / WRF_P0) ** WRF_RCP,
        "QSPEC": qv / (1.0 + qv),
        "U_MASS": u_mass,
        "V_MASS": v_mass,
        "T2": surface["T2"],
        "Q2SPEC": q2 / (1.0 + q2),
        "PSFC": surface["PSFC"],
        "TSK": surface["TSK"],
        "TERRAIN": surface["TERRAIN"],
        "SNOW": surface["SNOW"],
        "LANDMASK": surface["LANDMASK"],
        "SEAICE": surface["SEAICE"],
        "U10_MASS": surface["U10"],
        "V10_MASS": surface["V10"],
        "SOILT": soil_t,
        "SOILW": soil_m,
    }
    fields.update(optional)
    return WrfSnapshot(
        valid_time=frame.valid_time,
        fields={name: np.asarray(values, dtype=np.float32) for name, values in fields.items()},
        sina=sina, cosa=cosa,
    )


def require_physical(source: Mapping[str, np.ndarray]) -> None:
    """Plausibility windows no real value approaches, plus finiteness."""

    for name, values in source.items():
        if not np.isfinite(values).all():
            raise IntermediateRefusal(f"the wrfout field behind {name} carries a non-finite value")
    windows = {"TEMP": (150.0, 350.0), "T2": (150.0, 350.0), "TSK": (150.0, 400.0),
               "QSPEC": (0.0, 0.1), "Q2SPEC": (0.0, 0.1), "PSFC": (3.0e4, 1.2e5),
               "LANDMASK": (0.0, 1.0), "SEAICE": (0.0, 1.0)}
    for name, (low, high) in windows.items():
        values = source[name]
        if float(values.min()) < low or float(values.max()) > high:
            raise IntermediateRefusal(
                f"the wrfout field behind {name} spans {float(values.min()):g}..{float(values.max()):g}, "
                f"outside {low:g}..{high:g}; that is not a WRF state this door converts"
            )


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
@dataclass
class WrfoutRequest:
    source: SourceRow
    paths: tuple[Path, ...]
    bounds: tuple[float, float, float, float]
    out_dir: Path
    spacing_deg: float | None = None
    edge_cells: int = DEFAULT_WRF_EDGE_CELLS
    prefix: str = "MET"
    target_basis: Mapping[str, Any] | None = None
    backend: Any = None


def run_row(row: SourceRow, grid: WrfGrid, carried: Mapping[str, bool]) -> SourceRow:
    """The row for this parent: its level count and what it carries."""

    if grid.soil_layers != len(row.soil_layer_names):
        raise IntermediateRefusal(
            f"the parent carries {grid.soil_layers} soil layers; the row writes the "
            f"{len(row.soil_layer_names)} Noah layers "
            f"({', '.join(t for t, _ in row.soil_layer_names)}) and refuses to re-layer "
            f"another land model's column by a guess"
        )
    if grid.land_model is not None and grid.land_model not in NOAH_LAND_MODELS:
        raise IntermediateRefusal(
            f"the parent ran SF_SURFACE_PHYSICS={grid.land_model}; its four soil levels "
            f"are not the Noah/Noah-MP layers ({', '.join(f'{d:g} m' for d in NOAH_DZS_M)}) "
            f"the ST/SM names declare"
        )
    three_d = dict(row.atmosphere_3d)
    surface = dict(row.surface)
    for table, declared, into in ((OPTIONAL_3D, row.optional_3d, three_d),
                                  (OPTIONAL_SURFACE, row.optional_surface, surface)):
        for name, key in table.items():
            if not carried.get(name):
                continue
            if key not in declared:
                raise IntermediateRefusal(
                    f"wrfout {name} maps to window key {key!r} and the {row.name} row "
                    f"declares no such optional field; the row is where its WPS name lives"
                )
            into[key] = declared[key]
    return replace(row, levels=grid.nz, atmosphere_3d=three_d, surface=surface)


def _identity_layers(values: np.ndarray, nodes: Sequence[float], midpoints: Sequence[float]) -> np.ndarray:
    if len(nodes) != len(midpoints) or values.shape[0] != len(midpoints):
        raise IntermediateRefusal(
            f"the soil column has {values.shape[0]} layers and the row names {len(midpoints)}"
        )
    return values


def intermediate_name(prefix: str, valid: datetime, *, sub_hourly: bool) -> str:
    return f"{prefix}:{valid:%Y-%m-%d_%H:%M}" if sub_hourly else f"{prefix}:{valid:%Y-%m-%d_%H}"


def build_wrfout_intermediates(request: WrfoutRequest, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    out_dir = Path(request.out_dir).expanduser().absolute()
    refuse_stale_out_dir(out_dir)
    grid, frames, sources, carried = scan_wrfout(request.paths)
    projection, proof = prove_projection(grid, request.paths[0])
    row = run_row(request.source, grid, carried)
    dlat, dlon, spacing_basis = spacing_for(grid, request.bounds, request.spacing_deg)
    target = target_for_bounds(request.bounds, dlat=dlat, dlon=dlon)
    window, zero_x, zero_y = window_for(projection, grid, target, edge_cells=request.edge_cells)
    backend = request.backend
    if backend is None:
        try:
            from woof.ingest.preprocess_backend import resolve_preprocess_backend
        except ImportError as error:
            raise IntermediateRefusal(
                f"the engine's preprocessing backend is not importable ({error})"
            ) from error
        backend = resolve_preprocess_backend("cpu")
    plan = projected_plan(zero_x, zero_y, window, backend)
    sources = sign_sources(sources)
    noah_mid = tuple(float(np.sum(NOAH_DZS_M[:k]) + 0.5 * NOAH_DZS_M[k]) for k in range(len(NOAH_DZS_M)))
    pieces = {
        "plan": lambda snapshot, lat, lon, _backend: plan,
        "ranges": require_physical,
        "rotation": lambda snapshot: (snapshot.sina, snapshot.cosa),
        "interp_nodes": _identity_layers,
        "soil_nodes_m": noah_mid,
        "noah_midpoints_m": noah_mid,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_hourly = any(frame.valid_time.minute for frame in frames)
    first = frames[0].valid_time
    files: list[dict[str, Any]] = []
    for frame in frames:
        started = time.perf_counter()
        snapshot = read_snapshot(frame, window, carried)
        records, measured = regrid_hour(row, pieces, snapshot, target, backend=backend)
        path = out_dir / intermediate_name(request.prefix, frame.valid_time, sub_hourly=sub_hourly)
        lead = (frame.valid_time - first).total_seconds() / 3600.0
        written = write_intermediate(
            path, records, valid_time=frame.valid_time, forecast_hour=lead,
            map_source=row.map_source, target=target,
        )
        written["regrid_seconds"] = round(time.perf_counter() - started, 2)
        written["levels"] = measured["levels"]
        written["source"] = {"path": str(frame.path), "record": frame.index}
        files.append(written)
        log(f"WROTE {path.name}: {written['field_records']} records, "
            f"{written['bytes'] / 1e6:.1f} MB, {written['regrid_seconds']} s")
    interval = sorted({int((b.valid_time - a.valid_time).total_seconds())
                       for a, b in zip(frames, frames[1:])})
    carried_3d = [OPTIONAL_3D[name] for name in OPTIONAL_3D if carried.get(name)]
    receipt = {
        "schema": INTERMEDIATE_SCHEMA,
        "source": row.name,
        "source_description": row.description,
        "wrfout": sources,
        "wrf_grid": grid.as_dict(),
        "projection_proof": proof,
        "valid_times": [f"{frame.valid_time:%Y-%m-%d_%H:%M:%S}" for frame in frames],
        "interval_seconds": interval,
        "file_naming": "PREFIX:YYYY-MM-DD_HH:MM (sub-hourly)" if sub_hourly else "PREFIX:YYYY-MM-DD_HH",
        "target": target.as_dict(),
        "target_spacing_basis": spacing_basis,
        "target_basis": dict(request.target_basis or {}),
        "source_window": window.as_dict(),
        "interior_edge_cells": int(request.edge_cells),
        "regrid": {
            "operator": plan.operator,
            "engine": "woof.ingest.hrrr._ProjectedCpuPlan (WPS overlapping-parabolic, FP64 "
                      "donors) over WRF donors from the file's own projection; winds "
                      "destaggered and rotated to the earth basis with the file's "
                      "SINALPHA/COSALPHA on the source grid",
            "derived": {
                "TT": "(T + 300) * ((P + PB) / 1e5) ** (2/7)",
                "GHT": "mass-level mean of (PH + PHB) / 9.81",
                "PRESSURE": "P + PB",
                "SPECHUMD": "QVAPOR / (1 + QVAPOR); 2 m from Q2 likewise",
                "SOILHGT": "HGT", "SKINTEMP": "TSK", "LANDSEA": "LANDMASK (nearest)",
                "SEAICE": "SEAICE (nearest)",
            },
            "soil": "the parent's own four Noah layers (TSLB, SMOIS), moved by nearest neighbour",
        },
        "levels": {
            "count": row.levels + 1,
            "convention": "level-indexed WRF mass levels (xlvl 1..N, bottom first) with the "
                          "3-D PRESSURE field carried, plus the surface level 200100.0",
            "init_switches": {
                "--nfglevels": row.levels + 1, "--nfgsoillevels": len(row.soil_layer_names),
                "--use-spechumd": "yes", "--extrap-airtemp": "constant",
                "why_constant": "the WRF column tops at its p_top, below the 30 km model top; "
                                "lapse-rate extrapolation above the first-guess top is the "
                                "Fortran's own fatal (docs/source-matrix.md, IFS row)",
            },
        },
        "hydrometeors_carried": carried_3d,
        "files": files,
        "not_carried": ["SH2O", "PMSL"] + [OPTIONAL_3D[n] for n in OPTIONAL_3D if not carried.get(n)],
        "why_not_carried": "rw_mpas_init has no SH2O or PMSL rule for a 3-D-pressure first "
                           "guess (it rebuilds liquid soil water itself); the hydrometeors "
                           "written are read past by today's init and boundary engines",
    }
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8", newline="\n")
    log(f"RECEIPT {receipt_path}")
    return receipt


def request_from_arguments(arguments: argparse.Namespace) -> WrfoutRequest:
    row = source_row(arguments.source)
    for flag, value in (("--grib-dir", arguments.grib_dir), ("--cycle", arguments.cycle),
                        ("--decoder", arguments.decoder), ("--hours", arguments.hours),
                        ("--workers", arguments.workers)):
        if value is not None:
            raise IntermediateRefusal(
                f"{flag} belongs to a GRIB row; --source {row.name} reads its valid times "
                f"from the wrfout files named by --wrfout-glob"
            )
    if arguments.wrfout_glob is None:
        raise IntermediateRefusal(f"--source {row.name} needs --wrfout-glob: the WRF history to convert")
    edge = DEFAULT_WRF_EDGE_CELLS if arguments.wrf_edge_cells is None else int(arguments.wrf_edge_cells)
    if edge < 0:
        raise IntermediateRefusal(f"--wrf-edge-cells {edge} is negative")
    bounds, basis = bounds_from_arguments(arguments)
    return WrfoutRequest(
        source=row, paths=tuple(resolve_glob(arguments.wrfout_glob)), bounds=bounds,
        out_dir=arguments.out_dir, spacing_deg=arguments.spacing_deg, edge_cells=edge,
        prefix=str(arguments.prefix), target_basis=basis,
    )


def run_wrfout_intermediate(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    receipt = build_wrfout_intermediates(request)
    levels = receipt["levels"]
    print(json.dumps({
        "files": [item["path"] for item in receipt["files"]],
        "target": receipt["target"],
        "source_window": receipt["source_window"],
        "next": (
            f"woof hex init --met {receipt['files'][0]['path']} ... --nfglevels "
            f"{levels['count']} --nfgsoillevels {levels['init_switches']['--nfgsoillevels']} "
            f"--use-spechumd yes --extrap-airtemp constant"
        ),
    }, indent=2))
    return 0


__all__ = [
    "DEFAULT_WRF_EDGE_CELLS",
    "REQUIRED_VARIABLES",
    "WrfFrame",
    "WrfGrid",
    "WrfSnapshot",
    "WrfoutRequest",
    "bounds_from_arguments",
    "build_wrfout_intermediates",
    "intermediate_name",
    "projected_plan",
    "prove_projection",
    "read_snapshot",
    "region_bounds",
    "request_from_arguments",
    "run_wrfout_intermediate",
    "scan_wrfout",
    "spacing_for",
    "target_for_bounds",
    "window_for",
]
