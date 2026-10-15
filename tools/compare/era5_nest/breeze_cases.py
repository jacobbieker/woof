"""The two published Breeze/NumericalEarth real-data cases, as woof reproduces them.

Every number marked ``read`` below is copied from the cited source line at
the commit named in :data:`SOURCES`; every number marked ``derived`` is
computed here from those, and the function that computes it is the
citation.  Nothing in this module touches the network or a GPU.

What "the same domain" means.  Breeze runs both cases on a constant-degree
``LatitudeLongitudeGrid`` at 1/9 degree (about 12.35 km north-south, and
``12.35 cos(lat)`` km east-west).  woof's regional grids are WRF map
projections; it has no lat-lon regional grid.  The reproduction therefore
keeps what sets the cost of a step -- the cell counts ``Nx x Ny x Nz``, the
centre, the nominal 12 km spacing, the 50-level column to about 20 km and
the hourly ERA5 boundary cadence -- on a Lambert conformal grid at a uniform
12 km.  The ground covered differs (see :func:`ground_coverage_km`) and so
does the smallest horizontal spacing, which is what bounds Breeze's
advective time step.  Both are stated beside every number this harness
reports.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime

#: Source commits the facts below were read at.
SOURCES = {
    "NumericalEarth.jl": (
        "https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac/"
        "examples/breeze_downscaling_era5.jl"),
    "AtmosphericRivers.jl": (
        "https://github.com/glwagner/AtmosphericRivers.jl/blob/7cc5735d/README.md"),
    "NumericalEarth PR 454 timing comment": (
        "https://github.com/NumericalEarth/NumericalEarth.jl/pull/454"
        "#issuecomment-5012436787"),
}

#: Breeze's 1/9-degree spacing (read: breeze_downscaling_era5.jl:74,
#: AtmosphericRivers case.jl:33-35).
BREEZE_DEGREES = 1.0 / 9.0

#: Mean Earth radius used for the spacing arithmetic (km).
EARTH_RADIUS_KM = 6371.0


@dataclasses.dataclass(frozen=True)
class PublishedTiming:
    """A wall-clock figure exactly as its source states it.

    A source gives either a whole run (``simulated_hours`` in
    ``wall_hours``) or a per-iteration cost; whichever it does not give
    stays ``None``.
    """

    hardware: str
    gpus: int
    quote: str
    source: str
    simulated_hours: float | None = None
    wall_hours: float | None = None
    iterations: int | None = None
    seconds_per_iteration: float | None = None
    #: True when the run wrote output during the timed interval.
    with_output: bool | None = None
    #: The card as a key of nest_throughput.BANDWIDTH_GBPS, or None when the
    #: source does not say which variant it was.
    card: str | None = None


@dataclasses.dataclass(frozen=True)
class Case:
    key: str
    title: str
    #: Breeze's lat-lon box (read), degrees: (west, east), (south, north).
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    nx: int
    ny: int
    nz: int
    start: datetime
    #: The published run length (read), hours.
    hours: int
    #: The nominal spacing both models are labelled with, metres.
    dx_m: float
    published: tuple[PublishedTiming, ...]
    notes: tuple[str, ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.lat_bounds[0] + self.lat_bounds[1]),
                0.5 * (self.lon_bounds[0] + self.lon_bounds[1]))

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cycle(self) -> str:
        return self.start.strftime("%Y-%m-%dT%H")


# MC3E: breeze_downscaling_era5.jl:64-80 (read).  Centre (36.605, -97.485),
# extent 16.7 x 15.1 degrees, Nx = round(16.7*9) = 150, Ny = round(15.1*9) =
# 136, start 2011-05-20T00, duration 3 hours.
_MC3E_LAT0, _MC3E_LON0 = 36.605, -97.485
MC3E = Case(
    key="mc3e",
    title="MC3E squall line, ARM SGP (Oklahoma), 20 May 2011",
    lon_bounds=(_MC3E_LON0 - 16.7 / 2, _MC3E_LON0 + 16.7 / 2),
    lat_bounds=(_MC3E_LAT0 - 15.1 / 2, _MC3E_LAT0 + 15.1 / 2),
    nx=150, ny=136, nz=50,
    start=datetime(2011, 5, 20, 0),
    hours=3,
    dx_m=12000.0,
    published=(
        PublishedTiming(
            hardware="NVIDIA A100 (memory size not stated)",
            gpus=1,
            simulated_hours=1.0,
            wall_hours=7.2 / 60.0,
            quote=("The simulated hour cost 7.2 minutes wall -- the full "
                   "18-hour hindcast projects to ~2.2 hours."),
            source=SOURCES["NumericalEarth PR 454 timing comment"],
            with_output=True),
    ),
    notes=(
        "The checked-in example runs 3 h; the PR 454 timing was an 18 h "
        "configuration of the same domain on registered Breeze 0.9.0.",
    ),
)

# AR: AtmosphericRivers.jl README.md:53-64, case.jl:12-35 (read).  Corridor
# 170W-110W, 25N-60N at 1/9 degree = 540 x 315 x 50, start 2025-12-07T12,
# 72 h.
AR = Case(
    key="ar",
    title="Atmospheric river, NE Pacific / Pacific Northwest, 7-10 Dec 2025",
    lon_bounds=(-170.0, -110.0),
    lat_bounds=(25.0, 60.0),
    nx=540, ny=315, nz=50,
    start=datetime(2025, 12, 7, 12),
    hours=72,
    dx_m=12000.0,
    published=(
        PublishedTiming(
            hardware="NVIDIA A100-40GB",
            card="a100-40gb",
            gpus=1,
            simulated_hours=72.0,
            wall_hours=3.77,
            iterations=47600,
            quote=("The full 72 h hindcast (Dec 7 12Z -> Dec 10 12Z) completed "
                   "in 3.77 h of stepping on one A100-40GB (47,600 iterations, "
                   "~19x real time) with the production recipe: "
                   "AR_RELAX_WIDTH=8 AR_BLEND_LENGTH=240000 and the streaming "
                   "ERA5 parent."),
            source=SOURCES["AtmosphericRivers.jl"] + "#L62-L64",
            with_output=True),
        PublishedTiming(
            hardware="NVIDIA A100-40GB",
            card="a100-40gb",
            gpus=1,
            seconds_per_iteration=0.267,
            quote=("Strong scaling 1->2 GPUs: 1.50x speedup (75% efficiency) -- "
                   "0.178 vs 0.267 s/iter on the 12 km corridor."),
            source=SOURCES["AtmosphericRivers.jl"] + "#L77-L78",
            with_output=False),
        PublishedTiming(
            hardware="NVIDIA A100-40GB",
            card="a100-40gb",
            gpus=2,
            seconds_per_iteration=0.178,
            quote=("Strong scaling 1->2 GPUs: 1.50x speedup (75% efficiency) -- "
                   "0.178 vs 0.267 s/iter on the 12 km corridor."),
            source=SOURCES["AtmosphericRivers.jl"] + "#L77-L78",
            with_output=False),
    ),
    notes=(
        "Run on an unpublished NumericalEarth branch (glw/cleanup, v0.6.1) "
        "with Breeze 0.8.0 (AtmosphericRivers.jl Manifest.toml).",
        "The per-iteration figures come from 6 h runs with no output "
        "writers (slurm/scale2.batch, AR_HOURS=6).",
    ),
)

CASES = {case.key: case for case in (MC3E, AR)}


# ---------------------------------------------------------------------------
# Derived geometry
# ---------------------------------------------------------------------------

def breeze_spacing_km(case: Case) -> dict[str, float]:
    """Breeze's realised cell sizes in km (derived from the lat-lon box).

    ``dy`` is constant; ``dx`` shrinks with ``cos(lat)``, so the smallest
    zonal spacing sits on the poleward edge.
    """

    dlon = (case.lon_bounds[1] - case.lon_bounds[0]) / case.nx
    dlat = (case.lat_bounds[1] - case.lat_bounds[0]) / case.ny
    per_degree = math.pi * EARTH_RADIUS_KM / 180.0
    south, north = case.lat_bounds
    center = case.center[0]
    return {
        "dy": dlat * per_degree,
        "dx_equatorward_edge": dlon * per_degree * math.cos(math.radians(south)),
        "dx_center": dlon * per_degree * math.cos(math.radians(center)),
        "dx_poleward_edge": dlon * per_degree * math.cos(math.radians(north)),
    }


def lat_lon_box_area_km2(case: Case) -> float:
    """Exact spherical area of Breeze's lat-lon box (derived)."""

    west, east = case.lon_bounds
    south, north = case.lat_bounds
    return (EARTH_RADIUS_KM ** 2 * math.radians(east - west)
            * (math.sin(math.radians(north)) - math.sin(math.radians(south))))


def ground_coverage_km(case: Case) -> dict[str, float]:
    """Footprint of the two grids, km and km^2 (derived)."""

    woof_x = case.nx * case.dx_m / 1000.0
    woof_y = case.ny * case.dx_m / 1000.0
    breeze_area = lat_lon_box_area_km2(case)
    return {
        "woof_extent_x_km": woof_x,
        "woof_extent_y_km": woof_y,
        "woof_area_km2": woof_x * woof_y,
        "breeze_area_km2": breeze_area,
        "area_ratio_woof_over_breeze": woof_x * woof_y / breeze_area,
    }


# ---------------------------------------------------------------------------
# The vertical column
# ---------------------------------------------------------------------------

def breeze_z_interfaces() -> list[float]:
    """Breeze's 51 cell interfaces in metres (derived, exactly as Oceananigans
    builds them).

    ``ReferenceToStretchedDiscretization(extent=19525, bias=:left,
    constant_spacing=60, constant_spacing_extent=60, maximum_spacing=490,
    stretching=LinearStretching(0.15))`` (read: breeze_downscaling_era5.jl:
    85-91, identical in AtmosphericRivers downscale.jl:61-76), stepped with
    Oceananigans' ``compute_stretched_interfaces`` (src/Grids/
    coordinate_utils.jl:250-307): one 60 m surface cell, then each spacing
    1.15 times the last, capped at 490 m, every face rounded to 0.01 m, until
    the face reaches the extent.  The loop overshoots 19525 m to ~20003 m.
    """

    extent, spacing0, maximum = 19525.0, 60.0, 490.0
    faces = [0.0, spacing0]
    while faces[-1] < extent:
        previous = faces[-1] - faces[-2]
        step = min(maximum, 1.15 * previous)
        faces.append(round(faces[-1] + step, 2))
    return faces


def standard_pressure_pa(z_m: float) -> float:
    """US Standard Atmosphere 1976 pressure (Pa) up to 32 km geopotential."""

    g0, r_air = 9.80665, 287.053
    if z_m <= 11000.0:
        t = 288.15 - 0.0065 * z_m
        return 101325.0 * (t / 288.15) ** (g0 / (r_air * 0.0065))
    p11 = standard_pressure_pa(11000.0)
    if z_m <= 20000.0:
        return p11 * math.exp(-g0 * (z_m - 11000.0) / (r_air * 216.65))
    if z_m <= 32000.0:
        p20 = standard_pressure_pa(20000.0)
        t = 216.65 + 0.001 * (z_m - 20000.0)
        return p20 * (t / 216.65) ** (-g0 / (r_air * 0.001))
    raise ValueError("standard_pressure_pa covers 0-32 km")


def breeze_matched_eta() -> tuple[list[float], float]:
    """Full eta levels and p_top (Pa) that put woof's 50 layers on Breeze's
    interface heights over a standard-atmosphere column at sea level
    (derived).

    eta_k = (p(z_k) - p_top) / (p_sfc - p_top) with p_top = p(z_top).  Over
    terrain the two coordinates differ by construction -- WRF's hybrid eta
    follows dry hydrostatic pressure, Breeze's Gal-Chen sigma follows height
    -- so this matches the column, not every cell.
    """

    faces = breeze_z_interfaces()
    p_surface = standard_pressure_pa(0.0)
    p_top = standard_pressure_pa(faces[-1])
    eta = [(standard_pressure_pa(z) - p_top) / (p_surface - p_top) for z in faces]
    eta[0], eta[-1] = 1.0, 0.0
    return [round(value, 6) for value in eta], p_top
