"""``woof track plan``: a nested, track-following forecast from a 3-D path.

One :class:`~woof.track.model.Track3D` in, one ready-to-run ``woof go``
configuration out, with everything a batch machine needs beside it:

* a three-domain tree built by the ``woof domain`` machinery
  (:mod:`woof.domain_wizard`: its projection, its domain tables, its
  vertical column, its fetch hints and its namelist.wps), never a second
  copy of the projection arithmetic.  d01 is static and covers the
  track's envelope plus a margin; d02 slides and d03 follows the track;
* a 256-level band-biased column (:mod:`woof.vertical_bands`) whose top
  sits a fixed headroom above the track's peak;
* d03 born from its live parent shortly before launch
  (``start_from = "parent"``), writing history only while the track is
  airborne;
* the relocation itinerary that keeps d03 on the track and d02 around
  d03 (``woof relocation from-track``), where this woof has it;
* ``[output]`` set for an Icechunk export of d03 alone (``--products
  none``);
* ``plan.json``: cells, GPU memory, cell-steps, a wall-time range and the
  output volume; and, on request, a ``woof.go-job.v1`` job document.

Nothing here knows a dataset.  Names come from the caller.

Pieces of the plan that need features this woof may not have yet degrade
to something that still runs and SAY so in ``plan.json`` and on the
terminal: with no relocation itinerary module the nests stay where they
are born; with no tree physics suite the nest columns are written
explicitly; a regional source whose column stops below the model top
gives way to GFS when the source was chosen automatically.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from woof.explain import layered
from woof.track.model import Track3D


PLAN_SCHEMA = "woof.track-plan.v1"

#: Horizontal spacing of d01, d02 and d03 (km).
DEFAULT_DX_KM = (1.5, 0.3, 0.1)
#: Window sides (km).  d03's mass-cell count is rounded to a whole number
#: of its parent's cells (and an even count of them), so 32.4 km at 100 m
#: is 324 cells, 108 of d02's.
DEFAULT_D03_WINDOW_KM = 32.4
DEFAULT_D02_WINDOW_KM = 72.0
#: Ground d01 keeps around the track envelope on every side (km).
DEFAULT_MARGIN_KM = 60.0
#: Spin-up of the whole tree before launch, and of d03 (hours).
DEFAULT_SPINUP_H = 3.0
DEFAULT_D03_SPINUP_H = 1.0
#: d03's history interval while the track is airborne (s).
DEFAULT_HISTORY_S = 120.0
#: Bands the default column guarantees: the boundary layer the launch
#: climbs out of, and the lower stratosphere a high track floats in.
DEFAULT_ETA_BANDS = ("0:2000:41:agl", "15000:25000:100")
#: Model top: the reference-atmosphere pressure this far above the
#: track's peak, clipped to P_TOP_RANGE_PA.  The headroom keeps the
#: peak below the damping layer (``zdamp`` = 5 km in the emitted
#: ``[shared]``).
P_TOP_HEADROOM_M = 7000.0
P_TOP_RANGE_PA = (300.0, 5000.0)
#: Below this share of the flight inside a static d03 the plan's note is
#: a warning: most of d03's exported history would not hold the track.
LOW_COVERAGE = 0.5
#: Itinerary cadence (model seconds): how often d03 may move.
DEFAULT_RELOCATION_CADENCE_S = 300.0
#: Distance (d03 cells) the track keeps from d03's edges.  40 cells is
#: 4 km at 100 m, clear of the relaxation zone and of the strip a move
#: cold-starts.
DEFAULT_TRACK_MARGIN_CELLS = 40
#: Thinnest layer the track-height bias may leave in the free atmosphere
#: (above FREE_ATMOSPHERE_AGL_M over the launch ground), in m.  A long
#: float packs the biased levels to ~15 m in the lower stratosphere; on
#: the default ladder's 7.5 s root step a 15 m layer is crossed by 2 m/s
#: of vertical motion, which mountain waves exceed, and a 1.5 km root
#: over steep ground collapsed such a layer (non-positive thickness) an
#: hour in.  50 m keeps a 6.7 m/s Courant margin on that step.
DEFAULT_ETA_MIN_FREE_DZ_M = 50.0
FREE_ATMOSPHERE_AGL_M = 3000.0
#: Halvings of the bias weight tried before the bias is dropped.
ETA_BIAS_HALVINGS = 6
#: Resource model: device memory per mass cell (KiB) for a 256-level
#: tree, plus a fixed context; throughput on a 48 GB card in cell-steps
#: per second.  A range, because both depend on physics and on how much
#: of the tree is moving.
GPU_KIB_PER_CELL = (0.54, 0.84)
GPU_BASE_GIB = 3.0
CELL_STEPS_PER_S = (12.0e6, 20.0e6)
#: The AWS tiers of docs/public/AWS-TRACK-RUNS.md: an L40S worker admits
#: jobs declaring up to this many GiB, the H100 tier the rest up to its
#: card.
L40S_TIER_MAX_GIB = 44.0
H100_TIER_MAX_GIB = 80.0
#: Placeholders a ``--icechunk-url`` may carry (``woof go`` fills
#: ``{name}`` from ``[experiment] name``).
ICECHUNK_DEFAULT_URL = "{name}.icechunk"
#: Sources ``--source`` takes.  ``auto`` reads the track's own suggestion.
SOURCE_CHOICES = ("auto", "hrrr", "ukv", "gfs", "era5")
#: The global source an automatic choice falls back to: published to
#: public object storage, no account needed, any model top.
FALLBACK_SOURCE = "gfs"
RELOCATION_CHOICES = ("auto", "track", "static")
#: Output presets tried in order for ``--output-preset auto``.
OUTPUT_PRESET_PREFERENCE = ("balloon", "minimal")
#: The explicit physics this module writes when no tree suite is
#: registered: the root's scheme choices (a ``woof physics-catalog``
#: mix) and every nest's columns.  The nests run large-eddy: PBL off,
#: 1.5-order TKE closure, full 3-D diffusion, surface fluxes from the
#: surface layer, isotropic mixing length (docs/public/LES.md).
EXPLICIT_ROOT_PHYSICS = {
    "microphysics": "thompson-mp8", "pbl": "mynn",
    "surface_layer": "mynn", "land_surface": "noah-mp",
    "radiation": "rte-rrtmgp", "cumulus": "off",
}
EXPLICIT_NEST_PHYSICS = {
    "bl_pbl_physics": 0, "km_opt": 2, "diff_opt": 2, "isfflx": 1,
    "mix_isotropic": 1,
}
#: The tree suite ``--physics auto`` takes when this woof registers one.
TREE_SUITE_NEST_KEYS = {"bl_pbl_physics": 0, "km_opt": 2}


class TrackPlanRefused(ValueError):
    """A plan this module will not write as asked."""


# --------------------------------------------------------------------------
# options


@dataclass(frozen=True)
class PlanOptions:
    """Everything ``woof track plan`` takes besides the track."""

    source: str = "auto"
    dx_km: tuple[float, float, float] = DEFAULT_DX_KM
    d03_window_km: float = DEFAULT_D03_WINDOW_KM
    d02_window_km: float = DEFAULT_D02_WINDOW_KM
    margin_km: float = DEFAULT_MARGIN_KM
    spinup_h: float = DEFAULT_SPINUP_H
    d03_spinup_h: float = DEFAULT_D03_SPINUP_H
    history_s: float = DEFAULT_HISTORY_S
    #: ``LO:HI:N[:msl|agl]`` strings; ``None`` takes DEFAULT_ETA_BANDS,
    #: fitted to the column (a default band above the model top is
    #: trimmed; an explicit one is refused).
    eta_bands: tuple[str, ...] | None = None
    #: Bias the column toward the heights the track spends time at.
    eta_bias_track: bool = True
    #: Largest ratio between adjacent layer depths (None: the generator's).
    eta_max_stretch: float | None = None
    #: First layer depth (m) for the band generator (None: its default).
    dzbot_m: float | None = None
    #: Free-atmosphere layer floor the track bias must respect (m); 0 off.
    eta_min_free_dz_m: float = DEFAULT_ETA_MIN_FREE_DZ_M
    #: In-model hub-height wind diagnostic heights (m AGL): interval
    #: max/min/mean wind at each, i.e. gusts by height.
    hub_heights_m: tuple[float, ...] | None = None
    #: Points sampled every hub_sample_interval_s: (site_id, lat, lon).
    hub_sites: tuple[tuple[str, float, float], ...] = ()
    #: ``None`` is auto (P_TOP_HEADROOM_M above the peak, clipped).
    p_top_pa: float | None = None
    icechunk_url: str | None = None
    output_preset: str = "auto"
    relocation: str = "auto"
    relocation_cadence_s: float = DEFAULT_RELOCATION_CADENCE_S
    track_margin_cells: int = DEFAULT_TRACK_MARGIN_CELLS
    #: ``auto`` (the registered tree suite, else explicit keys),
    #: ``explicit``, or a tree suite id.
    physics: str = "auto"
    gpu_mem_gib: float | None = None


# --------------------------------------------------------------------------
# pieces


def _floor_to(moment: datetime, seconds: int) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=timezone.utc)


def _ceil_to(moment: datetime, seconds: int) -> datetime:
    floor = _floor_to(moment, seconds)
    return floor if floor == moment else floor + timedelta(seconds=seconds)


def _naive(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def model_top_pa(track: Track3D, requested: float | None = None) -> tuple[float, str]:
    """``(p_top, why)``: the requested top, or the reference-atmosphere
    pressure P_TOP_HEADROOM_M above the track's peak, clipped to
    P_TOP_RANGE_PA."""

    from woof.vertical_bands import reference_pressure

    if requested is not None:
        return float(requested), "requested"
    # An AGL track carries no terrain height, so its heights are read as
    # MSL over sea-level ground (the column below is built the same way).
    peak = float(track.max_height)
    raw = float(reference_pressure(peak + P_TOP_HEADROOM_M))
    lo, hi = P_TOP_RANGE_PA
    top = min(max(raw, lo), hi)
    why = (f"{P_TOP_HEADROOM_M / 1000:g} km above the {peak / 1000:.2f} km "
           f"peak is {raw:.0f} Pa in the standard atmosphere")
    if top != raw:
        why += f", clipped to {top:g} Pa"
    return round(top, 1), why


def _source_reach_pa(source: str) -> float | None:
    """The highest model top (lowest pressure) SOURCE can initialise, or
    ``None`` when nothing bounds it.

    The registry row's reach, as :func:`woof.domain_wizard.
    source_model_top_refusal` reads it: the composite column's when this
    woof carries another source above a regional one
    (``woof.source_adapters.composite_source_top_pa``), else the row's
    fetch extension or certified top.  A REGIONAL row that declares no
    top at all is held to the wizard's default emission
    (:data:`woof.domain_wizard.DEFAULT_MODEL_TOP_PA`), the one top every
    route is known to prepare for it.
    """

    from woof import source_adapters
    from woof.domain_wizard import DEFAULT_MODEL_TOP_PA

    adapter = source_adapters.get_source_adapter(source)
    composite = getattr(source_adapters, "composite_source_top_pa", None)
    upper = getattr(adapter, "upper_source", None)
    if composite is not None:
        reach = composite(source, upper)
    else:
        reach = adapter.extendable_source_top_pa
        if reach is None:
            reach = adapter.certified_source_top_pa
    if reach is None and upper is None and adapter.coverage_window is not None:
        return DEFAULT_MODEL_TOP_PA
    return None if reach is None else float(reach)


def _source_registered(source: str) -> bool:
    from woof.source_adapters import get_source_adapter

    try:
        return get_source_adapter(source).runnable
    except ValueError:
        return False


def choose_source(track: Track3D, requested: str, p_top_pa: float
                  ) -> tuple[str, list[str]]:
    """``(source id, notes)`` for the boundary source.

    ``auto`` takes the track's suggestion (hrrr, ukv or era5: see
    :func:`woof.track.model.boundary_source`) where this woof can prepare
    it to ``p_top_pa``, and :data:`FALLBACK_SOURCE` otherwise -- era5's
    place included, because GFS needs no account and serves any top.  A
    named source that cannot is refused.
    """

    from woof.domain_wizard import resolve_source

    notes: list[str] = []
    if requested not in SOURCE_CHOICES and not _source_registered(requested):
        raise TrackPlanRefused(
            f"--source {requested!r} is neither one of "
            f"{', '.join(SOURCE_CHOICES)} nor a runnable registered source")
    if requested == "auto":
        wanted = track.boundary_source
        candidate = wanted if wanted != "era5" else FALLBACK_SOURCE
        if wanted == "era5":
            notes.append("source: the track is outside the regional grids; "
                         f"using {FALLBACK_SOURCE} (global, no account) in "
                         "place of era5 (pass --source era5 to keep it)")
        if not _source_registered(candidate):
            notes.append(f"source: {candidate} is not a source this woof "
                         f"can prepare; using {FALLBACK_SOURCE}")
            candidate = FALLBACK_SOURCE
        reach = _source_reach_pa(candidate)
        if reach is not None and p_top_pa < reach:
            notes.append(
                f"source: {candidate}'s column stops at {reach:g} Pa, above "
                f"the {p_top_pa:g} Pa model top this track needs, and this "
                f"woof carries no other column above it; using "
                f"{FALLBACK_SOURCE} for the whole column")
            candidate = FALLBACK_SOURCE
        return resolve_source(candidate), notes
    if not _source_registered(requested):
        raise TrackPlanRefused(layered(
            f"--source {requested}: this woof has no runnable {requested} "
            "source.\n"
            f"  remedy: pass --source {FALLBACK_SOURCE} (or --source auto).",
            "The registry (`woof prep --list-sources`) lists what this "
            "build can prepare; a plan for anything else could not run."))
    source = resolve_source(requested)
    reach = _source_reach_pa(source)
    if reach is not None and p_top_pa < reach:
        raise TrackPlanRefused(layered(
            f"--source {source} reaches {reach:g} Pa, and this track needs "
            f"a model top of {p_top_pa:g} Pa.\n"
            f"  remedy: pass --source {FALLBACK_SOURCE} or --source auto, "
            f"or --p-top-pa {reach:g} if the track's peak fits under it.",
            "The model top sits a fixed headroom above the track's peak so "
            "the peak stays below the damping layer; a source whose column "
            "stops lower cannot initialise the levels in between."))
    return source, notes


def forcing_cadence_s(source: str) -> int:
    from woof.domain_wizard import SOURCE_FORCING_INTERVAL_S

    interval = SOURCE_FORCING_INTERVAL_S.get(source)
    if interval is None:
        from woof.source_adapters import source_forcing_interval_seconds
        interval = source_forcing_interval_seconds(source)
    return int(interval)


@dataclass(frozen=True)
class PlanTimes:
    """The run's clock, all naive UTC."""

    start: datetime
    end: datetime
    launch: datetime
    landing: datetime
    d03_start: datetime
    cadence_s: int
    cycle: datetime
    start_hour: int
    #: d03's history interval: the window opens and closes on its multiples
    #: (from d03's start), which are whole d03 steps on every ladder
    #: (_fit_root_step) and whole seconds.
    history_s: float = 1.0

    @property
    def run_seconds(self) -> int:
        return int((self.end - self.start).total_seconds())

    @property
    def d03_offset_s(self) -> int:
        return int((self.d03_start - self.start).total_seconds())

    def history_window(self) -> tuple[int, int]:
        """d03's history_begin_s / history_end_s, from d03's start."""

        grid = max(int(round(self.history_s)), 1)
        begin = (self.launch - self.d03_start).total_seconds()
        end = (self.landing - self.d03_start).total_seconds()
        begin = int(math.floor(begin / grid)) * grid
        end = int(math.ceil(end / grid)) * grid
        return max(begin, 0), end


def plan_times(track: Track3D, source: str, options: PlanOptions, *,
               cadence_s: int | None = None) -> PlanTimes:
    """Start, end, d03's birth and the forcing cycle for ``track``, on the
    source's forcing cadence or ``cadence_s``."""

    from woof.source_adapters import get_source_adapter

    if not 0.0 <= options.d03_spinup_h <= options.spinup_h:
        raise TrackPlanRefused(
            f"--d03-spinup-h {options.d03_spinup_h:g} must lie between 0 "
            f"and --spinup-h {options.spinup_h:g}: d03 is born from its "
            "parent, so it cannot start before the tree")
    cadence = int(cadence_s or forcing_cadence_s(source))
    launch, landing = track.launch.time, track.landing.time
    if not landing > launch:
        raise TrackPlanRefused(
            f"track {track.name} lands at or before its launch; there is "
            "nothing to follow")
    start = _floor_to(launch - timedelta(hours=options.spinup_h), cadence)
    end = _ceil_to(landing, cadence)
    # d03 is born on a whole minute before launch: every step in the tree
    # (7.5, 1.5, 0.5 s by default) divides a minute, so its birth lands on
    # its parent's step lattice.
    d03 = launch - timedelta(hours=options.d03_spinup_h)
    d03 = d03.replace(second=0, microsecond=0)
    d03 = max(d03, start)
    adapter = get_source_adapter(source)
    hours = (end - start).total_seconds() / 3600.0
    cycle, start_hour = _naive(start), 0
    if adapter.max_forecast_hour > 0:
        cycle, start_hour = _cycle_for(source, _naive(start), hours)
    return PlanTimes(start=_naive(start), end=_naive(end),
                     launch=_naive(launch), landing=_naive(landing),
                     d03_start=_naive(d03), cadence_s=cadence, cycle=cycle,
                     start_hour=start_hour, history_s=float(options.history_s))


def _align_d03_start(times: PlanTimes, exp) -> PlanTimes:
    """``times`` with d03's birth moved back, if it must, onto a whole
    second that is also a whole number of d02 steps from the start (the
    loader's delayed-start rule).  A minute serves every default ladder."""

    dt2 = Fraction(exp.dt_exact(2))
    lattice = math.lcm(60, dt2.numerator)
    offset = times.d03_offset_s - times.d03_offset_s % lattice
    if offset == times.d03_offset_s:
        return times
    return replace(times, d03_start=times.start + timedelta(seconds=offset))


def _cycle_for(source: str, start: datetime, hours: float
               ) -> tuple[datetime, int]:
    """The latest cycle of SOURCE at or before ``start`` whose forecast
    reaches the end of the run, and the lead the run starts at."""

    from woof.source_adapters import get_source_adapter
    from woof.source_cycles import cycle_grid_for

    adapter = get_source_adapter(source)
    grid = cycle_grid_for(source)
    hours_of_day = grid.hours if grid is not None else tuple(range(24))
    for back in range(0, 49):
        cycle = start - timedelta(hours=back)
        if cycle.hour not in hours_of_day:
            continue
        horizon = (grid.horizon(cycle) if grid is not None else None) \
            or adapter.max_forecast_hour
        if back + hours <= horizon:
            return cycle, back
    raise TrackPlanRefused(
        f"no {source} cycle in the 48 h before {start:%Y-%m-%dT%H}Z "
        f"forecasts {hours:g} h ahead; shorten --spinup-h or pick another "
        "--source")


def _snap_cells(km: float, dx_km: float, ratio: int) -> int:
    """Mass cells for a ``km`` window at ``dx_km``: a whole, even number
    of parent cells (the wizard's child quantum, 2 x ratio)."""

    quantum = 2 * ratio
    return max(quantum, quantum * round(km / dx_km / quantum))


def _ratios(dx_km: Sequence[float]) -> tuple[int, int]:
    if len(dx_km) != 3 or any(not (value > 0) for value in dx_km):
        raise TrackPlanRefused(
            f"--dx-km takes three positive spacings (d01,d02,d03), got "
            f"{','.join(f'{v:g}' for v in dx_km)}")
    ratios = []
    for parent, child in zip(dx_km, dx_km[1:]):
        ratio = parent / child
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 2:
            raise TrackPlanRefused(
                f"--dx-km {parent:g} -> {child:g} is a ratio of {ratio:.4g}; "
                "each nest must refine its parent by a whole number >= 2 "
                "(3 or 5 are WRF's usual choices)")
        ratios.append(int(round(ratio)))
    return ratios[0], ratios[1]


@dataclass(frozen=True)
class TreeLayout:
    projection: dict
    dims: tuple[tuple[int, int], ...]
    ratios: tuple[int, int]
    root_dx_m: float
    #: Initial (i_parent_start, j_parent_start) of d02 and d03.
    placements: dict
    margin_km: float
    notes: tuple[str, ...] = ()


def tree_layout(track: Track3D, source: str, options: PlanOptions,
                times: PlanTimes) -> TreeLayout:
    """d01 around the envelope, d02 and d03 centred where the track is
    when d03 is born."""

    from woof import domain_wizard as dw
    from woof.static.source_defaults import align_source_projection
    from woof.source_coverage import _root_grid

    ratios = _ratios(options.dx_km)
    root_dx_m = options.dx_km[0] * 1000.0
    notes: list[str] = []
    n3 = _snap_cells(options.d03_window_km, options.dx_km[2], ratios[1])
    n2 = _snap_cells(options.d02_window_km, options.dx_km[1], ratios[0])
    # d03 must fit inside d02 with the wizard's clearance (boundary zone
    # plus blend) on every side.
    clearance = dw._CLEARANCE_ROWS
    if n3 // ratios[1] + 2 * clearance > n2:
        raise TrackPlanRefused(
            f"d03 ({n3 * options.dx_km[2]:g} km) does not fit in d02 "
            f"({n2 * options.dx_km[1]:g} km) with {clearance} d02 cells of "
            "boundary and blend on each side; widen --d02-window-km or "
            "narrow --d03-window-km")
    south, west, north, east = track.envelope
    lat = np.asarray(track.lat, dtype=float)
    lon = np.asarray(track.lon, dtype=float)
    projection = dw._projection_entries(0.5 * (south + north),
                                        0.5 * (west + east))
    # d01 must hold d02 wherever the track takes it: half of d02, d01's
    # own boundary and blend rows, and two cells of slack.
    d02_half_km = 0.5 * n2 * options.dx_km[1]
    floor_km = d02_half_km + (clearance + 2) * options.dx_km[0]
    margin_km = float(options.margin_km)
    if margin_km < floor_km:
        notes.append(f"layout: --margin-km {margin_km:g} grown to "
                     f"{floor_km:.1f} so d01 holds d02 at the envelope's edge")
        margin_km = floor_km
    grid = _root_grid(projection, 2, 2, root_dx_m)
    i, j = grid.latlon_to_ij(lat, lon)
    i, j = np.asarray(i, dtype=float), np.asarray(j, dtype=float)
    if not (np.all(np.isfinite(i)) and np.all(np.isfinite(j))):
        raise TrackPlanRefused(
            f"track {track.name} cannot be placed in the "
            f"{projection['map_proj']} projection centred on its envelope")
    margin_cells = margin_km / options.dx_km[0]
    half_x = float(np.max(np.abs(i - grid.known_x))) + margin_cells
    half_y = float(np.max(np.abs(j - grid.known_y))) + margin_cells
    n1x = max(2, 2 * math.ceil(half_x))
    n1y = max(2, 2 * math.ceil(half_y))
    centred = projection
    for _ in range(4):
        # A source whose static grid serves this spacing moves d01 onto
        # its own mass points and projection; the track is placed again
        # in the grid that results, and d01 grown until it holds the
        # margin there too.
        projection = align_source_projection(centred, (n1x, n1y), root_dx_m,
                                             source)
        if projection is centred:
            break
        x, y = (np.asarray(v, dtype=float) for v in _root_grid(
            projection, n1x, n1y, root_dx_m).latlon_to_ij(lat, lon))
        short_x = max(0.5 + margin_cells - x.min(),
                      x.max() - (n1x + 0.5 - margin_cells), 0.0)
        short_y = max(0.5 + margin_cells - y.min(),
                      y.max() - (n1y + 0.5 - margin_cells), 0.0)
        if not (short_x or short_y):
            break
        n1x += 2 * math.ceil(short_x)
        n1y += 2 * math.ceil(short_y)
    else:
        raise TrackPlanRefused(
            f"d01 cannot be placed on {source}'s own grid around track "
            f"{track.name} with the {margin_km:g} km margin")
    dims = ((n1x, n1y), (n2, n2), (n3, n3))
    dw._pole_clearance_refusal(projection, n1x, n1y, root_dx_m,
                               target_option="the track")
    gap = dw.source_coverage_refusal(projection, n1x, n1y, source=source,
                                     root_dx_m=root_dx_m,
                                     target_option="the track")
    if gap is not None:
        raise TrackPlanRefused(gap)
    # Centre d02 and d03 where the track is when d03 is born.
    seconds = np.array([(_naive(t) - times.d03_start).total_seconds()
                        for t in track.times])
    at_lat = float(np.interp(0.0, seconds, lat))
    at_lon = float(np.interp(0.0, seconds, lon))
    root = _root_grid(projection, n1x, n1y, root_dx_m)
    x, y = root.latlon_to_ij(at_lat, at_lon)
    placements = _centred_placements(float(x), float(y), dims, ratios,
                                     clearance)
    return TreeLayout(projection=projection, dims=dims, ratios=ratios,
                      root_dx_m=root_dx_m, placements=placements,
                      margin_km=margin_km, notes=tuple(notes))


def _centred_placements(x: float, y: float, dims, ratios, clearance: int
                        ) -> dict:
    """d02 and d03 placements centred on root mass point (x, y), clamped
    to keep each nest ``clearance`` parent cells inside its parent."""

    (n1x, n1y), (n2x, n2y), (n3x, n3y) = dims
    r2, r3 = ratios

    def place(centre: float, parent_n: int, span: int) -> int:
        start = int(round(centre - span / 2.0 + 0.5))
        lo, hi = 1 + clearance, parent_n - span + 1 - clearance
        return min(max(start, lo), hi)

    i2 = place(x, n1x, n2x // r2)
    j2 = place(y, n1y, n2y // r2)
    # The point in d02 cells, measured from d02's first mass point.
    x2 = (x - i2) * r2 + (r2 + 1) / 2.0
    y2 = (y - j2) * r2 + (r2 + 1) / 2.0
    i3 = place(x2, n2x, n3x // r3)
    j3 = place(y2, n2y, n3y // r3)
    return {2: (i2, j2), 3: (i3, j3)}


def check_followable(track: Track3D, layout: TreeLayout, options: PlanOptions,
                     times: PlanTimes, *, refuse: bool = True) -> dict:
    """The farthest the track moves over one relocation cadence, against
    what d03 can follow, as a summary; with ``refuse`` (d03 will follow
    the track), refuses a track no itinerary on this cadence can keep
    inside d03."""

    cadence = float(options.relocation_cadence_s)
    dx3_km = options.dx_km[2]
    n3 = min(layout.dims[2])
    usable_km = (n3 - 2 * options.track_margin_cells) * dx3_km
    if usable_km <= 0:
        raise TrackPlanRefused(
            f"--d03-window-km {n3 * dx3_km:g} leaves no room inside the "
            f"{options.track_margin_cells}-cell track margin on each side")
    t = np.array([(_naive(x) - times.d03_start).total_seconds()
                  for x in track.times])
    lat = np.asarray(track.lat, dtype=float)
    lon = np.asarray(track.lon, dtype=float)
    keep = t >= 0.0
    worst_km, worst_at = 0.0, None
    if keep.sum() >= 2:
        t, lat, lon = t[keep], lat[keep], lon[keep]
        # What the track spans inside every cadence window, per axis
        # (d03's margins bound each axis), on a local tangent plane: a
        # track that doubles back spans its whole excursion, not only its
        # distance from where the window began.
        from woof.track.model import EARTH_RADIUS_KM
        x = np.radians(lon) * EARTH_RADIUS_KM
        y = np.radians(lat) * EARTH_RADIUS_KM
        shrink = np.cos(np.radians(lat))
        ends = np.searchsorted(t, t + cadence, side="left")
        for k in range(t.size):
            window = slice(k, max(ends[k], k + 1))
            east = float(np.ptp(x[window]) * shrink[k])
            north = float(np.ptp(y[window]))
            spread = max(east, north)
            if spread > worst_km:
                worst_km, worst_at = spread, t[k]
    speed = worst_km * 1000.0 / cadence
    summary = {"cadence_s": cadence, "worst_window_km": round(worst_km, 3),
               "worst_window_speed_m_s": round(speed, 2),
               "usable_d03_km": round(usable_km, 3)}
    # Static nests follow nothing, so no cadence can be too slow for them.
    if worst_km > usable_km and refuse:
        suggestion = _largest_cadence(worst_km, cadence, usable_km)
        when = times.d03_start + timedelta(seconds=float(worst_at or 0.0))
        raise TrackPlanRefused(layered(
            f"track {track.name} covers {worst_km:.1f} km in one "
            f"{cadence:g} s relocation window (from {when:%Y-%m-%dT%H:%M}Z, "
            f"~{speed:.0f} m/s); d03 keeps only {usable_km:.1f} km of "
            f"track inside its {options.track_margin_cells}-cell margins.\n"
            + (f"  remedy: pass --relocation-cadence-s {suggestion:g} "
               "(or less), or widen --d03-window-km."
               if suggestion else
               "  remedy: widen --d03-window-km or coarsen --dx-km; no "
               "cadence of a whole minute keeps up with this track."),
            "d03 moves only at cadence opportunities, so everything the "
            "track does inside one window must fit between d03's margins."))
    return summary


def _largest_cadence(worst_km: float, cadence: float, usable_km: float
                     ) -> float | None:
    """Whole minutes at which the worst window's speed fits d03."""

    speed = worst_km / cadence
    minutes = math.floor(usable_km / speed / 60.0)
    return float(minutes * 60) if minutes >= 1 else None


def vertical_column(track: Track3D, source: str, p_top: float,
                    options: PlanOptions) -> tuple[dict, list[str]]:
    """The 256-level band-biased column (``woof domain --eta-band``)."""

    from woof import domain_wizard as dw
    from woof.vertical_bands import (DEFAULT_DAMPING_DEPTH_M, HeightBand,
                                     parse_band, reference_height)

    notes: list[str] = []
    ground = float(track.launch.height) if track.height_datum == "msl" else 0.0
    if options.eta_bands is None:
        # A default band may reach into the damping layer (its levels
        # are a floor, not a ceiling) but stops half a sponge below the
        # top, so the top layers stay free to coarsen.
        top_m = float(reference_height(p_top))
        ceiling = top_m - 0.5 * DEFAULT_DAMPING_DEPTH_M
        bands = []
        for text in DEFAULT_ETA_BANDS:
            band = parse_band(text)
            lo, hi = band.msl(ground)
            if lo >= ceiling:
                notes.append(f"vertical: default band {band.label()} lies "
                             "too close to this model top; dropped")
                continue
            if hi > ceiling:
                kept = (ceiling - lo) / (hi - lo)
                count = max(1, int(round(band.min_levels * kept)))
                new_hi = ceiling - (ground if band.datum == "agl" else 0.0)
                notes.append(f"vertical: default band {band.label()} trimmed "
                             f"to {new_hi:.0f} m, half the damping layer "
                             f"below the model top ({count} levels)")
                band = HeightBand(band.lo_m, round(new_hi, 1), count,
                                  band.datum)
            bands.append(band)
    else:
        bands = [parse_band(text) for text in options.eta_bands]
    heights = None
    if options.eta_bias_track:
        launch, landing = track.launch.time, track.landing.time
        heights = tuple(float(h) for t, h in zip(track.times, track.height)
                        if launch <= t <= landing)
    def build(bias_heights, weight=None):
        try:
            return dw.resolve_vertical_column(
                source=source, p_top_pa=p_top, eta_band=bands,
                eta_bias_heights=bias_heights or None, eta_bias_weight=weight,
                eta_surface_m=ground, eta_max_stretch=options.eta_max_stretch,
                dzbot=options.dzbot_m)
        except dw.VerticalColumnRefused as error:
            raise TrackPlanRefused(str(error)) from error

    column = build(heights)
    floor = float(options.eta_min_free_dz_m or 0.0)
    if heights and floor > 0.0:
        thinnest = _thinnest_free_layer(column, ground)
        if thinnest < floor:
            from woof.vertical_bands import DEFAULT_BIAS_WEIGHT

            weight, found = float(DEFAULT_BIAS_WEIGHT), thinnest
            for _ in range(ETA_BIAS_HALVINGS):
                weight /= 2.0
                column = build(heights, weight)
                thinnest = _thinnest_free_layer(column, ground)
                if thinnest >= floor:
                    break
            else:
                column = build(None)
                thinnest, weight = _thinnest_free_layer(column, ground), 0.0
            notes.append(
                f"vertical: the track-height bias left a {found:.0f} m "
                f"layer above {FREE_ATMOSPHERE_AGL_M / 1000:g} km AGL, "
                f"thinner than the {floor:g} m free-atmosphere floor "
                f"(--eta-min-free-dz); bias weight {weight:g} gives "
                f"{thinnest:.0f} m")
    return column, notes


def _thinnest_free_layer(column: dict, ground_m: float) -> float:
    """Thinnest layer (m) of ``column`` above FREE_ATMOSPHERE_AGL_M over
    ``ground_m``, in the reference column the generator measured."""

    from woof.vertical_bands import dz_profile

    profile = dz_profile(column["eta_levels"], p_top_pa=column["p_top"],
                         surface_elevation_m=ground_m)
    above = profile["z_full_m"][:-1] >= ground_m + FREE_ATMOSPHERE_AGL_M
    return float(profile["dz_m"][above].min()) if above.any() else math.inf


def resolve_output_preset(requested: str) -> tuple[str, list[str]]:
    from woof.io.history_selection import HISTORY_PRESETS

    if requested != "auto":
        if requested not in HISTORY_PRESETS:
            raise TrackPlanRefused(
                f"--output-preset {requested!r} is not one of "
                f"{', '.join(sorted(HISTORY_PRESETS))}")
        return requested, []
    for name in OUTPUT_PRESET_PREFERENCE:
        if name in HISTORY_PRESETS:
            notes = ([] if name == OUTPUT_PRESET_PREFERENCE[0] else
                     [f"output: this woof has no {OUTPUT_PRESET_PREFERENCE[0]!r} "
                      f"history preset; using {name!r}"])
            return name, notes
    return "full", ["output: no lean history preset; writing the full tape"]


def resolve_physics(requested: str, source: str
                    ) -> tuple[dict, list[str]]:
    """How the tree's physics is written.

    ``{"profile": id, "nest_physics": {...}}`` for a registered tree suite
    (render_config writes both), or ``{"mix": request, "nest": keys}`` for
    the explicit fallback.
    """

    from woof import domain_wizard as dw

    suites = {}
    try:
        from woof.physics_compat import TREE_PHYSICS_PROFILES
        suites = dict(TREE_PHYSICS_PROFILES)
    except ImportError:
        pass
    can_render_nests = "nest_physics" in inspect.signature(
        dw.render_config).parameters
    if requested not in ("auto", "explicit"):
        if requested not in suites or not can_render_nests:
            raise TrackPlanRefused(
                f"--physics {requested!r} is not a tree physics suite this "
                "woof registers"
                + (f" (it has: {', '.join(sorted(suites))})" if suites else "")
                + "; pass --physics explicit")
        tree = suites[requested]
        return {"profile": tree.root, "nest_physics": tree.nest_overrides(),
                "label": requested}, []
    if requested == "auto" and can_render_nests:
        for tree_id, tree in sorted(suites.items()):
            nest = tree.nest_overrides()
            if all(nest.get(key) == value
                   for key, value in TREE_SUITE_NEST_KEYS.items()):
                return {"profile": tree.root, "nest_physics": nest,
                        "label": tree_id}, []
    notes = ([] if requested == "explicit" else
             ["physics: this woof registers no LES-nest tree suite; the root "
              "schemes and nest columns are written explicitly"])
    mix = dw.physics_mix_request(json.dumps(EXPLICIT_ROOT_PHYSICS),
                                 source=source, profile=None)
    return {"mix": mix, "nest": dict(EXPLICIT_NEST_PHYSICS),
            "label": "explicit"}, notes


# --------------------------------------------------------------------------
# the config


def _fetch_hints(source: str, times: PlanTimes, layout: TreeLayout,
                 config_path: Path) -> dict | None:
    from woof import domain_wizard as dw
    from woof.fetch import validate_fetch_hints
    from woof.runplan import drivability_for

    hours = times.run_seconds / 3600.0
    n1x, n1y = layout.dims[0]
    area = dw.fetch_area_hint(layout.projection, n1x, n1y, source=source,
                              root_dx_m=layout.root_dx_m,
                              target_option="the track", notes=[],
                              coverage_notes=[])
    cadence = dw._fetch_cadence_h(source, times.start_hour, hours)
    if cadence is not None:
        # A window planned on a coarser spacing than the source's own
        # (fetch_times) is fetched on that spacing.
        cadence = max(cadence, times.cadence_s // 3600)
    local = bool(drivability_for(source).get("requires_source_root"))
    if cadence is None and local:
        cadence = max(1, times.cadence_s // 3600)
    hints = {
        "source": source, "cycle": times.cycle.strftime("%Y-%m-%dT%H"),
        "hours": (math.ceil(hours) if cadence is None
                  else max(cadence, math.ceil(hours / cadence) * cadence)),
        "area": area, "out": "data",
    }
    if cadence is not None:
        hints["cadence"] = cadence
    if times.start_hour:
        hints["forecast_start_hour"] = times.start_hour
    if not (dw.source_has_fetch_front_door(source) or local):
        return None
    if not dw.source_fetch_takes_a_crop_box(source):
        hints.pop("area")
    if local:
        hints.pop("out")
        hints["source_root"] = "data"
    validate_fetch_hints(hints, source=str(config_path))
    return hints


def _era5_case_data(name: str, config_path: Path, hints: dict | None) -> dict:
    from woof import domain_wizard as dw
    from woof.fetch import era5_combined_name
    from woof.source_adapters import source_forcing_interval_seconds

    combined = era5_combined_name((hints or {}).get("era5_provider"))
    return {
        "forcing": [f"data/{combined}"],
        "vtable": dw._PACKAGED_VTABLE.name,
        "forcing_interval_s": source_forcing_interval_seconds("era5"),
        "wps_namelist": f"{config_path.stem}.namelist.wps",
        "geog_root": "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG",
        "sfcp_to_sfcp": True,
        "output_domain": 1,
        "output_title": f"woof {name}",
    }


def fetch_times(track: Track3D, source: str, options: PlanOptions,
                times: PlanTimes, layout: TreeLayout, config_path: Path
                ) -> tuple[PlanTimes, dict | None, list[str]]:
    """``(times, fetch hints, notes)``: the plan's window on the source's
    own spacing when its fetch serves that, else on the first of 3 h and
    6 h that it does (a source whose long runs step three-hourly past some
    lead serves a long window only on that spacing)."""

    from woof.fetch import fetch_accepts_cadence

    try:
        return times, _fetch_hints(source, times, layout, config_path), []
    except ValueError as error:
        refusal = error
    if fetch_accepts_cadence(source):
        for hours in (3, 6):
            if hours * 3600 <= times.cadence_s:
                continue
            coarser = plan_times(track, source, options,
                                 cadence_s=hours * 3600)
            coarser = replace(coarser, d03_start=times.d03_start)
            try:
                hints = _fetch_hints(source, coarser, layout, config_path)
            except ValueError:
                continue
            return coarser, hints, [
                f"source: {source} serves this {coarser.run_seconds / 3600:g} h "
                f"window only every {hours} h; the boundaries are "
                f"{hours}-hourly"]
    raise TrackPlanRefused(str(refusal)) from refusal


def render_tree(name: str, source: str, times: PlanTimes,
                layout: TreeLayout, vertical: dict, physics: dict,
                config_path: Path, hints: dict | None
                ) -> tuple[str, dict]:
    """``(header, document)``: the wizard's emission for this layout,
    parsed, before the track-specific tables go in."""

    from woof import domain_wizard as dw
    from woof.static.source_defaults import with_source_static_defaults_text

    case_data = (_era5_case_data(name, config_path, hints)
                 if source == "era5" else None)
    kwargs = {}
    if "nest_physics" in physics:
        kwargs["nest_physics"] = physics["nest_physics"]
    text = dw.render_config(
        name=name, start_time=times.start,
        hours=times.run_seconds // 3600, projection=layout.projection,
        dims=list(layout.dims), ratios=layout.ratios, fetch_hints=hints,
        case_data=case_data, root_dx_m=layout.root_dx_m,
        profile=physics.get("profile", dw.DEFAULT_PHYSICS_PROFILE),
        cumulus_requested=False, vertical=vertical,
        history_interval_s=None, nest_history_interval_s=None,
        physics_mix=physics.get("mix"), clock="fixed", **kwargs)
    text = with_source_static_defaults_text(text, source)
    return "\n".join(_first_comment_block(text)), tomllib.loads(text)


def _first_comment_block(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return lines



def _fit_root_step(domains: list[dict], history_s: float) -> None:
    """Shorten the root step, if it must, so every domain's step divides
    that domain's radiation period and d03's step divides its history
    interval and a minute.

    The wizard picks the root step for stability alone; on a ladder other
    than the default one the steps can come out as odd fractions (e.g. a
    d03 step of 100/69 s) that no history interval or radiation period is
    a whole number of.  The step kept is the longest d03 step of the form
    gcd(history, 60 s)/n that meets every period and is not longer than
    the wizard's, so stability is never loosened.  A radt the wizard
    snapped to its own clock (not a whole minute) is restored to the
    nearest whole minute first, and every period met is written back."""

    by_id = {int(t["grid_id"]): t for t in domains}
    root = by_id[1]
    root_dt = Fraction(int(root.get("time_step", 0))) + Fraction(
        int(root.get("time_step_fract_num", 0)),
        int(root.get("time_step_fract_den", 1)) or 1)
    # factor[g]: domain g's step over d03's step.
    factor, gid, below = {3: Fraction(1)}, 3, Fraction(1)
    while gid != 1:
        table = by_id[gid]
        below *= int(table.get("parent_time_step_ratio",
                               table.get("parent_grid_ratio", 1)))
        gid = int(table.get("parent_id", gid - 1))
        factor[gid] = below
    # The wizard snaps the root's radt to whole steps of ITS clock (4.5 km:
    # 27 x 22.5 s = 10.125 min); once this clock moves that cadence is
    # stale, so a period off the whole minute goes back to the nearest
    # whole minute, which the fit below then lands on exactly.
    periods = {}
    for g in factor:
        if not by_id[g].get("radt"):
            continue
        seconds = Fraction(str(float(by_id[g]["radt"]))) * 60
        if seconds.denominator != 1 or seconds % 60:
            seconds = Fraction(60 * max(1, round(seconds / 60)))
        periods[g] = seconds

    def fits(d03_dt: Fraction) -> bool:
        # Every grid writes history on the same interval.
        if any((Fraction(history_s) / (d03_dt * f)).denominator != 1
               for f in factor.values()):
            return False
        return all((period / (d03_dt * factor[g])).denominator == 1
                   for g, period in periods.items())

    def write_periods() -> None:
        for g, period in periods.items():
            by_id[g]["radt"] = float(period / 60)

    wizard = root_dt / factor[1]
    if wizard <= 0 or fits(wizard):
        write_periods()
        return
    base = Fraction(math.gcd(int(round(history_s)), 60))
    n = math.ceil(base / wizard)
    while not fits(base / n):
        n += 1
    new_root = base / n * factor[1]
    root["time_step"] = int(new_root)
    remainder = new_root - int(new_root)
    root["time_step_fract_num"] = remainder.numerator
    root["time_step_fract_den"] = remainder.denominator if remainder else 1
    write_periods()


def apply_track_tables(document: dict, *, times: PlanTimes,
                       layout: TreeLayout, physics: dict, options: PlanOptions,
                       preset: str, icechunk_url: str,
                       relocation: dict | None,
                       placements: dict) -> dict:
    """The parsed wizard document with d02/d03 placed, d03 born late from
    its parent, history only in flight, the nests' physics, ``[output]``
    and ``[relocation]``."""

    document = copy.deepcopy(document)
    document["experiment"]["start_time"] = times.start
    # A physics mix can replace the land scheme the source recipe's
    # land-scoped settings were filled for (render_config fills them
    # before the mix): drop the ones the scheme now running never reads.
    from woof.physics_source_defaults import LAND_SCOPED_RECIPE_SETTINGS
    shared = document["shared"]
    # A source recipe may pin the clock to its operational model's (the
    # HRRR recipe does), turning the measured terrain rules into advice.
    # Track trees put 100 m nests over whatever ground the track crosses,
    # and on steep ground those rules are what holds the integration (six
    # acoustic substeps where four fail), so the plan lets them apply.
    if shared.get("terrain_clock") == "pinned":
        shared["terrain_clock"] = "measured"
    for key, schemes in LAND_SCOPED_RECIPE_SETTINGS.items():
        if key in shared and shared.get("sf_surface_physics") not in schemes:
            del shared[key]
    domains = document["domain"]
    for table in domains:
        gid = int(table["grid_id"])
        if gid in placements:
            table["i_parent_start"], table["j_parent_start"] = (
                int(placements[gid][0]), int(placements[gid][1]))
        if gid > 1 and "nest" in physics:
            table.update(physics["nest"])
    _fit_root_step(domains, options.history_s)
    d03 = next(table for table in domains if int(table["grid_id"]) == 3)
    for key in ("start_time", "start_from", "history_begin_s",
                "history_end_s"):
        d03.pop(key, None)
    if times.d03_start > times.start:
        d03["start_time"] = times.d03_start
        d03["start_from"] = "parent"
    begin, end = times.history_window()
    d03["history_interval_s"] = float(options.history_s)
    if begin:
        d03["history_begin_s"] = begin
    d03["history_end_s"] = end
    # d01 and d02 write the same flight window on the same interval, from
    # the tree's start (they are never born late), so every grid lands in
    # the Icechunk repository and none writes during the spin-up.
    born = (times.d03_start - times.start).total_seconds()
    for table in domains:
        if int(table["grid_id"]) == 3:
            continue
        table["history_interval_s"] = float(options.history_s)
        table.pop("history_begin_s", None)
        if begin + born:
            table["history_begin_s"] = int(begin + born)
        table["history_end_s"] = int(end + born)
    document["output"] = {
        "preset": preset,
        **({"hub_heights_m": [float(h) for h in options.hub_heights_m]}
           if options.hub_heights_m else {}),
        **({"hub_sites": [{"site_id": n, "lat": la, "lon": lo}
                          for n, la, lo in options.hub_sites]}
           if options.hub_sites else {}),
        "icechunk": {"url": icechunk_url,
                     "domains": sorted(int(t["grid_id"]) for t in domains),
                     "delete_after_commit": True, "compress": True},
    }
    document.pop("relocation", None)
    if relocation:
        document["relocation"] = relocation
    return document


def _relocation_module():
    try:
        from woof import relocation_itinerary
    except ImportError:
        return None
    if not hasattr(relocation_itinerary, "itinerary_from_track"):
        return None
    return relocation_itinerary


def plan_relocation(track: Track3D, exp, times: PlanTimes,
                    options: PlanOptions) -> tuple[dict | None, dict | None,
                                                   list[str], dict]:
    """``(relocation table, placements, notes, summary)``.

    The itinerary of :mod:`woof.relocation_itinerary` when this woof has
    it and ``--relocation`` allows it; ``None`` (static nests) otherwise.
    """

    notes: list[str] = []
    if options.relocation == "static":
        return None, None, ["relocation: static nests (--relocation static)"], {
            "mode": "static"}
    module = _relocation_module()
    if module is None:
        if options.relocation == "track":
            raise TrackPlanRefused(layered(
                "--relocation track needs the track itinerary "
                "(woof relocation from-track), which this woof does not "
                "have.\n"
                "  remedy: pass --relocation static (or auto) for nests that "
                "stay where they are born, or update woof.",
                "Without the itinerary d03 stays centred where the track is "
                "at its birth and sees only the part of the flight inside "
                "it."))
        notes.append("relocation: this woof has no track itinerary "
                     "(woof relocation from-track); d02 and d03 stay where "
                     "they are born, centred on the track at d03's start")
        return None, None, notes, {"mode": "static", "reason": "unavailable"}
    fixes = [(t, la, lo, h) for t, la, lo, h in
             zip(track.times, track.lat, track.lon, track.height)
             if _naive(t) >= times.d03_start]
    if len(fixes) < 2:
        fixes = list(zip(track.times, track.lat, track.lon, track.height))
    try:
        plan = module.itinerary_from_track(
            [_naive(f[0]) for f in fixes], [f[1] for f in fixes],
            [f[2] for f in fixes], exp, times.start,
            float(options.relocation_cadence_s),
            int(options.track_margin_cells),
            heights=[f[3] for f in fixes])
    except ValueError as error:
        raise TrackPlanRefused(str(error)) from error
    table = plan.relocation_table() or None
    summary = {"mode": "track" if table else "static",
               "moves": len(getattr(plan, "moves", ()) or ()),
               "containment_slides": int(getattr(plan, "containment_slides", 0)),
               "worst_margin_cells": float(getattr(plan, "worst_margin_cells", 0.0))}
    if not table:
        notes.append("relocation: the track never leaves d03's margins; no "
                     "moves are needed")
    return table, dict(plan.placements), notes, summary


def d03_coverage(track: Track3D, layout: TreeLayout, placements: dict,
                 times: PlanTimes, margin_cells: int) -> float:
    """The fraction of the flight (launch to landing, sampled every 10 s)
    a STATIC d03 at ``placements`` holds the track ``margin_cells``
    inside its edges."""

    from woof.source_coverage import _root_grid

    (n1x, n1y), _, (n3x, n3y) = layout.dims
    r2, r3 = layout.ratios
    t = np.array([(_naive(x) - times.launch).total_seconds()
                  for x in track.times])
    span = (times.landing - times.launch).total_seconds()
    samples = np.arange(0.0, span + 10.0, 10.0)
    lat = np.interp(samples, t, np.asarray(track.lat, dtype=float))
    lon = np.interp(samples, t, np.asarray(track.lon, dtype=float))
    root = _root_grid(layout.projection, n1x, n1y, layout.root_dx_m)
    x, y = (np.asarray(v, dtype=float) for v in root.latlon_to_ij(lat, lon))
    (i2, j2), (i3, j3) = placements[2], placements[3]
    # Parent mass coordinate -> child mass coordinate (both 1-based): the
    # child's west edge is the parent's at i_parent_start - 0.5.
    x3 = ((x - i2 + 0.5) * r2 + 0.5 - i3 + 0.5) * r3 + 0.5
    y3 = ((y - j2 + 0.5) * r2 + 0.5 - j3 + 0.5) * r3 + 0.5
    inside = ((x3 >= 0.5 + margin_cells) & (x3 <= n3x + 0.5 - margin_cells)
              & (y3 >= 0.5 + margin_cells) & (y3 <= n3y + 0.5 - margin_cells))
    return float(inside.mean()) if inside.size else 1.0


def _patch_wps(text: str, placements: dict) -> str:
    """namelist.wps with &geogrid i/j_parent_start set for the grids
    named (entry g - 1 for grid g)."""

    lines = text.splitlines()
    for key, axis in (("i_parent_start", 0), ("j_parent_start", 1)):
        pattern = re.compile(rf"^(\s*{key}\s*=\s*)(.*?)(,?)\s*$")
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue
            values = [v.strip() for v in match.group(2).split(",") if v.strip()]
            for gid, ij in placements.items():
                values[int(gid) - 1] = str(int(ij[axis]))
            lines[index] = match.group(1) + ", ".join(values) + ","
            break
        else:
            raise TrackPlanRefused(f"namelist.wps carries no {key}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the estimate


def estimate(exp, times: PlanTimes, layout: TreeLayout, options: PlanOptions
             ) -> dict:
    """Cells, GPU memory, cell-steps, wall time and output volume."""

    nz = exp.vertical.mass_level_count
    domains = []
    total_cells = 0
    cell_steps = 0.0
    for dc in exp.domains:
        gid = int(dc.grid_id)
        nx, ny = int(dc.run.nx), int(dc.run.ny)
        levels = nz or int(dc.run.nz)
        cells = nx * ny * levels
        dt = float(exp.dt_exact(gid))
        active = float(times.run_seconds)
        if gid == 3:
            active -= times.d03_offset_s
        steps = active / dt
        total_cells += cells
        cell_steps += cells * steps
        domains.append({"grid_id": gid, "nx": nx, "ny": ny, "nz": levels,
                        "dx_m": float(dc.run.dx), "dt_s": dt,
                        "cells": cells, "active_s": active,
                        "steps": round(steps, 1)})
    lo_kib, hi_kib = GPU_KIB_PER_CELL
    mem = (GPU_BASE_GIB + total_cells * lo_kib / 1024 ** 2,
           GPU_BASE_GIB + total_cells * hi_kib / 1024 ** 2)
    slow, fast = CELL_STEPS_PER_S
    wall_h = (cell_steps / fast / 3600.0, cell_steps / slow / 3600.0)
    output = _output_bytes(exp, times, options)
    gpu_mem_gib = (float(options.gpu_mem_gib) if options.gpu_mem_gib
                   else math.ceil(mem[1] * 2.0) / 2.0)
    tier = ("l40s" if gpu_mem_gib <= L40S_TIER_MAX_GIB else
            "h100" if gpu_mem_gib <= H100_TIER_MAX_GIB else "none")
    return {
        "domains": domains,
        "total_cells": total_cells,
        "gpu_mem_gib_range": [round(mem[0], 2), round(mem[1], 2)],
        "gpu_mem_gib": gpu_mem_gib,
        "gpu_mem_model": {"kib_per_cell": list(GPU_KIB_PER_CELL),
                          "base_gib": GPU_BASE_GIB},
        "tier": tier,
        "cell_steps": cell_steps,
        "cell_steps_per_s": list(CELL_STEPS_PER_S),
        "wall_h_range": [round(wall_h[0], 2), round(wall_h[1], 2)],
        "output": output,
    }


def _output_bytes(exp, times: PlanTimes, options: PlanOptions) -> dict:
    """Every grid's exported history (uncompressed), from the history
    writer's own pricing where it answers, else 4 bytes x volumes x cells."""

    begin, end = times.history_window()
    frames = int((end - begin) // options.history_s) + 1
    per_grid: dict[int, int] = {}
    basis = "history writer pricing"
    try:
        from woof.output_disk import forecast_projection
        for d in forecast_projection(exp)["domains"]:
            if d.get("history_frames"):
                per_grid[int(d["grid_id"])] = int(
                    d["history_bytes"] / d["history_frames"])
    except Exception:  # noqa: BLE001 - pricing is advisory
        per_grid = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if gid not in per_grid:
            per_grid[gid] = 4 * 14 * int(dc.run.nx) * int(dc.run.ny) * int(
                dc.run.nz)
            basis = "4 bytes x 14 volumes per cell"
    per_frame = sum(per_grid.values())
    return {"frames": frames,
            "bytes_per_frame_by_grid": {f"d{g:02d}": b
                                        for g, b in sorted(per_grid.items())},
            "bytes_per_frame": int(per_frame),
            "bytes": int(per_frame * frames), "basis": basis,
            "compressed": "smaller: the export compresses (Blosc zstd)"}


# --------------------------------------------------------------------------
# the whole plan


@dataclass
class PlannedRun:
    name: str
    directory: Path
    config: Path
    companions: list[Path]
    plan: dict
    notes: list[str] = field(default_factory=list)


def plan_track(track: Track3D, name: str, out_dir: Path,
               options: PlanOptions = PlanOptions()) -> PlannedRun:
    """Write one track's config, companions and ``plan.json`` into
    ``out_dir/name/``; return what was written."""

    from woof import domain_wizard as dw
    from woof.track.ingest import check_name

    name = check_name(name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise TrackPlanRefused(layered(
            f"run name {name!r} has characters a job id cannot carry.\n"
            "  remedy: pass --name with only letters, digits, '.', '_' "
            "and '-'.",
            "The name becomes the config's [experiment] name, the job id "
            "and the Icechunk repository's key."))
    notes: list[str] = []
    p_top, top_why = model_top_pa(track, options.p_top_pa)
    if not P_TOP_RANGE_PA[0] <= p_top < 100000.0:
        raise TrackPlanRefused(f"--p-top-pa {p_top:g} is outside "
                               f"[{P_TOP_RANGE_PA[0]:g}, 100000) Pa")
    source, source_notes = choose_source(track, options.source, p_top)
    notes += source_notes
    times = plan_times(track, source, options)
    layout = tree_layout(track, source, options, times)
    notes += layout.notes
    will_follow = (options.relocation != "static"
                   and (_relocation_module() is not None
                        or options.relocation == "track"))
    follow = check_followable(track, layout, options, times,
                              refuse=will_follow)
    vertical, vertical_notes = vertical_column(track, source, p_top, options)
    notes += vertical_notes
    preset, preset_notes = resolve_output_preset(options.output_preset)
    notes += preset_notes
    physics, physics_notes = resolve_physics(options.physics, source)
    notes += physics_notes
    icechunk_url = options.icechunk_url or ICECHUNK_DEFAULT_URL
    if source == "era5":
        notes.append("source: era5 forcing is staged under data/ beside the "
                     "config; run woof go from the config's directory")

    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f"{name}.toml"
    times, hints, cadence_notes = fetch_times(track, source, options, times,
                                              layout, config_path)
    notes += cadence_notes
    wizard_header, document = render_tree(
        name, source, times, layout, vertical, physics, config_path, hints)
    # The loader refuses a delayed d03 start that is off d02's step lattice,
    # and the lattice is only known once the tree loads: load it first with
    # d03 born at the start, align, then lay the real birth in.
    probe = apply_track_tables(
        document, times=replace(times, d03_start=times.start),
        layout=layout, physics=physics, options=options, preset=preset,
        icechunk_url=icechunk_url, relocation=None,
        placements=layout.placements)
    times = _align_d03_start(times, _load(probe, config_path))
    document = apply_track_tables(
        document, times=times, layout=layout, physics=physics,
        options=options, preset=preset, icechunk_url=icechunk_url,
        relocation=None, placements=layout.placements)
    exp = _load(document, config_path)
    relocation, placements, relocation_notes, relocation_summary = \
        plan_relocation(track, exp, times, options)
    notes += relocation_notes
    placements = placements or layout.placements
    if relocation is not None and times.d03_start > times.start:
        # A moving d03 starts with the tree.  A parent-born nest cannot be a
        # relocation target (a resume re-births it on its declared ground
        # under a nest that has since moved), and an analysis-born one under
        # a sliding d02 needs its analysis prepared on d02's live footprint,
        # which a fixed prepared cache does not hold; both are refused by
        # the forecast.
        times = replace(times, d03_start=times.start)
        notes.append("d03: starts with the tree (a moving nest cannot be "
                     "born late), so its spin-up is the tree's")
    if relocation is None:
        covered = d03_coverage(track, layout, placements, times,
                               options.track_margin_cells)
        relocation_summary["d03_coverage"] = round(covered, 4)
        if covered < 0.999:
            loud = "WARNING " if covered < LOW_COVERAGE else ""
            notes.append(f"{loud}relocation: static d03 holds the track "
                         f"(inside its {options.track_margin_cells}-cell "
                         f"margins) for {100 * covered:.0f}% of the flight"
                         + ("; d03's history covers the rest of the flight "
                            "without the track in it" if loud else ""))
    document = apply_track_tables(
        document, times=times, layout=layout, physics=physics,
        options=options, preset=preset, icechunk_url=icechunk_url,
        relocation=relocation, placements=placements)
    header = _header(name, track, source, times, layout, vertical, physics,
                     preset, relocation_summary, wizard_header, notes)
    text = header + "\n" + emit_config(document)
    from woof.explain import muted_warnings
    # The loader's warnings about this file are `woof go`'s to print; a
    # batch of plans would repeat them per run.
    try:
        with muted_warnings():
            exp = dw.experiment_from_text(text, source=str(config_path))
    except ValueError as error:
        raise TrackPlanRefused(
            f"the planned config does not load: {error}") from error

    written = []
    dw._write_atomic(config_path, text)
    wps_text = _patch_wps(dw.render_wps_namelist(
        layout.projection, list(layout.dims), layout.ratios,
        root_dx_m=layout.root_dx_m, source=source,
        forcing_interval_seconds=times.cadence_s), placements)
    if source == "hrrr":
        from woof.hrrr_route_inputs import write_hrrr_route_inputs
        with muted_warnings():
            written += write_hrrr_route_inputs(
                config_path, exp, wps_text=wps_text, writer=dw._write_atomic)
    else:
        wps_path = run_dir / f"{name}.namelist.wps"
        dw._write_atomic(wps_path, wps_text)
        written.append(wps_path)
    eta_path = run_dir / f"{name}.eta.txt"
    dw._write_atomic(eta_path, dw._eta_levels_text(vertical))
    written.append(eta_path)
    if source == "era5":
        vtable = run_dir / dw._PACKAGED_VTABLE.name
        vtable.write_bytes(dw._PACKAGED_VTABLE.read_bytes())
        written.append(vtable)

    sizes = estimate(exp, times, layout, options)
    plan = {
        "schema": PLAN_SCHEMA,
        "name": name,
        "config": config_path.name,
        "companions": [path.name for path in written],
        "track": {"launch": _iso(times.launch), "landing": _iso(times.landing),
                  "launch_observed": track.launch_observed,
                  "landing_observed": track.landing_observed,
                  "duration_h": round(track.duration_h, 3),
                  "max_height_m": float(track.max_height),
                  "height_datum": track.height_datum,
                  "envelope": list(track.envelope),
                  "suggested_source": track.boundary_source},
        "source": source,
        "fetch": hints,
        "times": {"start": _iso(times.start), "end": _iso(times.end),
                  "run_seconds": times.run_seconds,
                  "forcing_cadence_s": times.cadence_s,
                  "cycle": _iso(times.cycle), "forecast_start_hour":
                      times.start_hour,
                  "d03_start": _iso(times.d03_start),
                  "d03_history_window_s": list(times.history_window())},
        "layout": {"projection": layout.projection,
                   "dims": [list(d) for d in layout.dims],
                   "ratios": list(layout.ratios),
                   "dx_m": [layout.root_dx_m / math.prod(layout.ratios[:k])
                            for k in range(3)],
                   "placements": {f"d{g:02d}": list(ij)
                                  for g, ij in sorted(placements.items())},
                   "margin_km": layout.margin_km},
        "vertical": {"nz": int(vertical["nz"]), "p_top_pa": float(vertical["p_top"]),
                     "p_top_reason": top_why, "eta_file": eta_path.name},
        "physics": physics.get("label"),
        "output": {"preset": preset, "icechunk_url": icechunk_url,
                   "domains": [1, 2, 3], "products": "none"},
        "relocation": relocation_summary,
        "follow_check": follow,
        "estimate": sizes,
        "notes": notes,
    }
    plan_path = run_dir / "plan.json"
    dw._write_atomic(plan_path, json.dumps(plan, indent=1, default=str) + "\n")
    return PlannedRun(name=name, directory=run_dir, config=config_path,
                      companions=written, plan=plan, notes=notes)


def _load(document: dict, config_path: Path):
    from woof import domain_wizard as dw
    from woof.explain import muted_warnings

    try:
        with muted_warnings():
            return dw.experiment_from_text(emit_config(document),
                                           source=str(config_path))
    except ValueError as error:
        raise TrackPlanRefused(
            f"the planned config does not load: {error}") from error


#: The order the sections of an emitted config are written in; anything
#: else follows in the document's own order.
_SECTION_ORDER = ("experiment", "projection", "shared", "domain", "fetch",
                  "case_data", "static", "output", "relocation")


def emit_config(document: dict) -> str:
    """``document`` as TOML, section by section in :data:`_SECTION_ORDER`,
    with the eta ladder five levels to a line as ``woof domain`` writes
    it; refuses text that does not parse back to ``document``."""

    from woof import domain_wizard as dw
    from woof.toml_document import emit_experiment_toml

    keys = [key for key in _SECTION_ORDER if key in document]
    keys += [key for key in document if key not in keys]
    chunks = []
    for key in keys:
        chunk = emit_experiment_toml({key: document[key]})
        if key == "shared" and "eta_levels" in document[key]:
            ladder = dw._toml_value(tuple(document[key]["eta_levels"]))
            chunk = re.sub(r"^eta_levels = \[.*\]$",
                           lambda _: f"eta_levels = {ladder}", chunk,
                           count=1, flags=re.M)
        chunks.append(chunk)
    text = "\n".join(chunks)
    if tomllib.loads(text) != document:
        raise TrackPlanRefused("the planned config did not survive its own "
                               "TOML round trip")
    return text


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


#: Wizard header statements this header carries over (each with its
#: continuation lines), and the ones that end such a statement: the
#: wizard's own provenance, its column (stated again below) and its
#: single-PBL advisories, which do not describe LES nests.
_KEPT_STATEMENTS = ("# PHYSICS", "# NOCTURNAL", "# NOT NOCTURNALLY",
                    "# JUSTIFY", "# TREE SUITE", "# NEST PHYSICS")
_DROPPED_STATEMENTS = ("# Emitted by", "# Taken verbatim", "# The switches",
                       "# The root is taken", "# forecast runner",
                       "# the parent chain", "# VERTICAL", "# CUMULUS",
                       "# GRAY", "# CLOCK", "# TROPICAL", "# NO [fetch]")


def _wizard_physics_lines(wizard_header: str) -> list[str]:
    """The wizard header's physics statements, whole."""

    kept, keeping = [], False
    for line in _first_comment_block(wizard_header):
        if line.startswith(_KEPT_STATEMENTS):
            keeping = True
        elif line.startswith(_DROPPED_STATEMENTS):
            keeping = False
        if keeping:
            kept.append(line)
    return kept


def _header(name, track, source, times, layout, vertical, physics, preset,
            relocation_summary, wizard_header, notes) -> str:
    from woof import domain_wizard as dw

    chain = "-".join(f"{layout.root_dx_m / math.prod(layout.ratios[:k]) / 1000:g}"
                     for k in range(3))
    begin, end = times.history_window()
    lines = [
        f"# Emitted by `woof track plan` for run {name}: ladder {chain} km, "
        f"source {source}.",
        f"# d01 static over the track envelope + {layout.margin_km:g} km; "
        "d02 and d03 centred on the track at d03's start.",
        f"# d03 born from its parent at {times.d03_start:%Y-%m-%dT%H:%M}Z; "
        f"history from {begin} s to {end} s after that (launch to landing).",
        "# relocation: " + ("d03 follows the track, d02 slides to contain it."
                            if relocation_summary.get("mode") == "track"
                            else "none (static nests)."),
        f"# output: preset {preset}, d03 exported to Icechunk; run with "
        "--products none.",
    ]
    if "nest" in physics:
        lines.append("# NEST PHYSICS: every nest runs large-eddy: "
                     + ", ".join(f"{k} = {v}" for k, v in physics["nest"].items())
                     + ".")
    lines += _wizard_physics_lines(wizard_header)
    for line in dw.vertical_column_header(vertical):
        lines.append(f"# {line}")
    for note in notes:
        lines.append(f"# NOTE {note}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# job documents


def job_document(run: PlannedRun, *, inputs_prefix: str | None = None,
                 icechunk_url: str | None = None,
                 result_prefix: str | None = None) -> dict:
    """A ``woof.go-job.v1`` document for ``run``: inline, or by URL under
    ``inputs_prefix`` (``<prefix>/<name>/<file>``)."""

    from woof.go_job import JOB_SCHEMA, parse_job

    if run.plan["source"] == "era5":
        raise TrackPlanRefused(layered(
            f"{run.name}: an era5 config declares its forcing beside the "
            "config, and woof go-job stages the config in inputs/ but runs "
            "woof go from its workdir, so the two paths would disagree.\n"
            "  remedy: plan job runs with --source gfs (or auto).",
            "ERA5's [case_data] forcing is relative to the config and its "
            "[fetch] out to the launch directory; the woof domain emission "
            "assumes they are the same directory."))
    document: dict[str, Any] = {"schema": JOB_SCHEMA, "job_id": run.name}
    if inputs_prefix:
        base = f"{inputs_prefix.rstrip('/')}/{run.name}"
        document["config"] = f"{base}/{run.config.name}"
        document["files"] = [{"url": f"{base}/{path.name}", "path": path.name}
                             for path in run.companions]
    else:
        document["config_toml"] = run.config.read_text(encoding="utf-8")
        document["config_name"] = run.config.name
        document["files"] = [{"path": path.name,
                              "content": path.read_text(encoding="utf-8")}
                             for path in run.companions]
    if icechunk_url:
        document["icechunk_url"] = icechunk_url.replace("{name}", run.name)
    if result_prefix:
        document["result_url"] = (f"{result_prefix.rstrip('/')}/"
                                  f"{run.name}.result.json")
    document["products"] = "none"
    document["resources"] = {"gpu_mem_gib": run.plan["estimate"]["gpu_mem_gib"]}
    document["description"] = (
        f"woof track plan {run.name}: {run.plan['source']}, "
        f"{run.plan['times']['start']} to {run.plan['times']['end']}")
    document["labels"] = {"tier": run.plan["estimate"]["tier"],
                          "source": run.plan["source"]}
    try:
        # Without an icechunk_url the worker names the store from
        # WOOF_ENERGY_OUTPUT_URL; validate as it would.
        parse_job(json.loads(json.dumps(document)), source=run.name,
                  output_prefix=None if icechunk_url else "file:///validate")
    except ValueError as error:
        raise TrackPlanRefused(f"the job document for {run.name} does not "
                               f"validate: {error}") from error
    return document


def summary_table(runs: Sequence[PlannedRun]) -> str:
    """One row per planned run."""

    width = max([len("run")] + [len(run.name) for run in runs])
    lines = [f"{'run':<{width}}  {'source':<6}  {'start (UTC)':<16}  "
             f"{'hours':>5}  {'d01':>9}  {'Mcells':>7}  {'GiB':>11}  "
             f"{'wall h':>11}  {'out GB':>7}  reloc"]
    for run in runs:
        plan = run.plan
        est = plan["estimate"]
        d01 = plan["layout"]["dims"][0]
        gib = est["gpu_mem_gib_range"]
        wall = est["wall_h_range"]
        lines.append(
            f"{run.name:<{width}}  {plan['source']:<6}  "
            f"{plan['times']['start'][:16].replace('T', ' '):<16}  "
            f"{plan['times']['run_seconds'] / 3600:>5.1f}  "
            f"{f'{d01[0]}x{d01[1]}':>9}  {est['total_cells'] / 1e6:>7.1f}  "
            f"{f'{gib[0]:.1f}-{gib[1]:.1f}':>11}  "
            f"{f'{wall[0]:.1f}-{wall[1]:.1f}':>11}  "
            f"{est['output']['bytes'] / 1e9:>7.1f}  "
            f"{plan['relocation'].get('mode')}")
    return "\n".join(lines)
