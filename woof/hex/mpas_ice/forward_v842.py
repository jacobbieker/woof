"""MALI v8.4.2: the land-ice forward mode around the forward-Euler step, on the GPU.

:func:`cold_start` and :func:`advance` are what ``li_simulation_clock_init``,
``li_core_init`` and ``li_core_run`` (``mpas_li_core.F``) do to the state:

- the clock (:class:`Clock`): ``config_start_time`` and ``config_dt`` on
  ``config_calendar_type``'s calendar, the stop time from
  ``config_run_duration`` (or ``config_stop_time``), and the
  ``adaptiveTimestepForceInterval`` alarm, in whole seconds;
- ``li_core_init``: ``li_setup_config_options`` (a thermal solver advects its
  tracers), ``xtime``, ``simulationStartTime``, ``daysSinceStart`` and
  ``deltat``; ``landice_init_block``: the second time level of ``vertexMask``
  cleared, ``li_setup_vertical_grid``, ``li_setup_sign_and_index_fields``,
  ``li_velocity_block_init`` (``li_sia_block_init``'s barycentric weights
  with ``from_vertex_barycentric``, or 'simple''s prescribed velocity,
  :mod:`~woof.hex.mpas_ice.velocity_simple_v842`),
  ``li_calculate_mask_init`` and ``thicknessOld``; ``li_thermal_init``
  (:mod:`~woof.hex.mpas_ice.thermal_init_v842`) and ``li_SGH_init``;
- ``li_core_initial_solve``: the masks and geometry, ``li_velocity_solve``
  and ``li_calculate_diagnostic_vars``
  (:mod:`~woof.hex.mpas_ice.time_integration_fe_v842`), the statistics with
  ``config_write_stats_on_startup`` (:mod:`~woof.hex.mpas_ice.statistics_v842`),
  and the geometry's time levels shifted;
- each step (``li_timestep``): ``deltat`` from the clock (or, adaptive, from
  the step's CFL limits and the seconds to the force alarm's next ring), the
  forward-Euler step, the clock advanced (by the adaptive ``deltat``),
  ``xtime`` and ``daysSinceStart``, ``timestepNumber``, the time levels
  shifted and the force alarm reset when it rings; then the statistics
  every ``config_stats_interval``-th step;
- the analysis members (:mod:`~woof.hex.mpas_ice.analysis_v842`):
  ``globalStats`` and ``regionalStats`` on startup (after the startup
  statistics) and after each step's statistics, as their alarms ring.

Each statistics report (:class:`~woof.hex.mpas_ice.statistics_v842.Statistics`)
goes to ``on_statistics`` when given, else to the ``woof.hex.mpas_ice``
logger at INFO, as the model writes it to its log. Each analysis-member
compute leaves its words in its pool (``globalStatsAM``,
``regionalStatsAM``) and goes to ``on_analysis`` when given; ``streams``
names the streams that exist with their reference times, and
``stream_ringing`` says when a stream's output alarm rings (see the module).

The mesh-scale initialisation (the vertical grid, the edge signs, the
barycentric weights) runs on the host at RKIND, one correctly rounded
operation at a time in the Fortran's order; the per-cell work on the GPU
(``forward_v842.cu`` and the units it calls). Every word is held to v8.4.2's
land-ice core, its ``mpas_timekeeping`` and ESMF time library compiled
verbatim at RKIND 4 and 8 (``tools/mali_oracle/run_forward.F90``,
``tests/test_mali_forward_parity.py``).

:func:`write_restart` and :func:`read_restart` round-trip a run: every pool
array word for word and the clock with its calendar and alarm; a run resumed
from them continues bitwise as if never stopped.

One v8.4.2 quirk has a named opt-in correction (``woof_sgh_init_mask``):
``li_SGH_init`` sets the ocean's pressure where the cell mask says floating,
or no ice below sea level, but at that point the mask carries only
``li_calculate_mask_init``'s initial-extent bit, so grounded ice below sea
level is given the ocean's pressure as well. 'v842' (the default) keeps the
model's words; 'corrected' computes ``li_calculate_mask`` first. The
advection's enthalpy units have one too (``woof_enthalpy_advection_units``,
:mod:`woof.hex.mpas_ice.advection_v842`), and so does the 'simple' velocity
solver's circular-shelf bed (``woof_simple_velocity_bed``,
:mod:`woof.hex.mpas_ice.velocity_simple_v842`), and so do the statistics'
extrema (``woof_statistics_extrema``, :mod:`woof.hex.mpas_ice.statistics_v842`),
and so do the analysis members' empty means (``woof_analysis_empty``,
:mod:`woof.hex.mpas_ice.analysis_v842`).

Refused by name, beyond what the step refuses: a restart start
(``config_do_restart``; :func:`read_restart` is this port's restart),
time integrations other than forward Euler, analysis members other than
globalStats and regionalStats, the Albany mesh writer, intervals with months, years on the
Gregorian calendar, fractional seconds, and ``config_year_digits`` other than
4. Output and restart streams change no state and are the caller's.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_ocean.forward_v842 import calendar_of, format_time, parse_time
from woof import quirk_corrections

from . import (advection_v842, analysis_v842, geometry_v842, rkind, statistics_v842, subglacial_hydro_v842, thermal_init_v842,
               velocity_simple_v842)
from . import time_integration_fe_v842 as fe

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_core.F:62-263,279-514,616-787,811-933,949-1077",
    "src/core_landice/mode_forward/mpas_li_time_integration.F:74-231",
    "src/core_landice/shared/mpas_li_setup.F:84-330",
    "src/core_landice/shared/mpas_li_mask.F:172-223",
    "src/core_landice/mode_forward/mpas_li_sia.F:126-211",
    "src/core_landice/mode_forward/mpas_li_velocity.F:157-208",
    "src/core_landice/mode_forward/mpas_li_statistics.F",
    "src/core_landice/analysis_members/mpas_li_analysis_driver.F",
    "src/core_landice/mode_forward/mpas_li_subglacial_hydro.F:75-191,1281-1361",
    "src/operators/mpas_geometry_utils.F:203-236,311-372,423-637,710-749",
    "src/operators/mpas_vector_operations.F:79-121",
    "src/framework/mpas_timekeeping.F",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
SGH_INIT_MASK = ("v842", "corrected")
SECONDS_PER_DAY = 86400
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("forward_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"li_forward_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision,
                      _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


# ---------------------------------------------------------------- the clock


def parse_interval(text: str, knob: str) -> tuple[int, int, int]:
    """``mpas_set_timeInterval``'s string ('[[YYYY-]MM-]DD_hh:mm:ss',
    '[[hh:]mm:]ss', with optional decimals of a second) as (years, months,
    whole seconds). Decimals other than zero are refused by name."""
    whole, dot, decimals = text.strip().partition(".")
    if dot and decimals.strip() and int(decimals) > 0:
        _refuse(knob, text, "the clock counts whole seconds", f"{knob} without fractions of a second")
    date, _, clock = whole.rpartition("_")
    years = months = days = 0
    if date:
        parts = [int(x) for x in date.split("-")]
        if len(parts) == 3:
            years, months, days = parts
        elif len(parts) == 2:
            months, days = parts
        else:
            days = parts[0]
    hms = [int(x) for x in clock.split(":")]
    if not 1 <= len(hms) <= 3:
        raise ValueError(f"{knob}: invalid TimeInterval string {text!r}")
    h, m, s = [0] * (3 - len(hms)) + hms
    return years, months, ((days * 24 + h) * 60 + m) * 60 + s


def interval_seconds(text: str, calendar: str, knob: str) -> int:
    """An interval string in whole seconds on ``calendar``: a year is 365
    days on the 'gregorian_noleap' calendar; months, and years on the
    Gregorian calendar (whose length depends on the date), are refused."""
    years, months, seconds = parse_interval(text, knob)
    if months:
        _refuse(knob, text, "a month's length depends on the date; the clock carries whole seconds",
                f"{knob} without months")
    if years and calendar != "gregorian_noleap":
        _refuse(knob, text, f"a year's length on the {calendar} calendar depends on the date; the clock carries "
                "whole seconds", f"{knob} without years, or config_calendar_type = 'gregorian_noleap'")
    return years * 365 * SECONDS_PER_DAY + seconds


def interval_dt(seconds: int, real):
    """``mpas_get_timeInterval(dt=)`` at RKIND: (real(days) * 24 * 60 * 60) +
    real(seconds) + real(sn)/real(sd), left to right, the days and seconds
    split toward zero as ESMF's."""
    days = int(seconds / SECONDS_PER_DAY) if seconds >= 0 else -int(-seconds // SECONDS_PER_DAY)
    rest = seconds - days * SECONDS_PER_DAY
    head = ((real(days) * real(24.0)) * real(60.0)) * real(60.0)
    return (head + real(rest)) + real(0.0) / real(1.0)


@dataclass
class Clock:
    """The land-ice clock on whole seconds since 0001-01-01_00:00:00 of its
    calendar: the start, current and stop times, the time step, and the
    ``adaptiveTimestepForceInterval`` alarm's interval and previous ring
    (``None`` without an adaptive time step, which alone reads it)."""
    start: int
    now: int
    step: int
    stop: int
    calendar: str
    force_interval: int | None = None
    force_prev: int | None = None

    @property
    def time_stamp(self) -> str:
        return format_time(self.now, self.calendar)

    @property
    def stopped(self) -> bool:
        """``mpas_is_clock_stop_time``."""
        return self.now >= self.stop

    def seconds_to_next_force_time(self) -> int:
        """``mpas_alarm_get_next_ring_time`` minus now."""
        return self.force_prev + self.force_interval - self.now

    def reset_force_alarm(self) -> None:
        """``mpas_reset_clock_alarm`` on the force alarm when it rings."""
        if self.force_interval is not None and self.now >= self.force_prev + self.force_interval:
            self.force_prev = self.now - (self.now - self.force_prev) % self.force_interval


def clock_init(config: dict) -> Clock:
    """``li_simulation_clock_init``."""
    calendar = calendar_of(config)
    if config["config_start_time"] == "file":
        _refuse("config_start_time", "file", "the restart timestamp belongs to a restart start; read_restart is "
                "this port's restart", "config_start_time = a date")
    start = parse_time(config["config_start_time"], calendar)
    step = interval_seconds(config["config_dt"], calendar, "config_dt")
    if not config["config_adaptive_timestep"] and step <= 0:
        _refuse("config_dt", config["config_dt"], "a time step of no time never reaches the stop time",
                "config_dt = a positive interval")
    if config["config_run_duration"] != "none":
        stop = start + interval_seconds(config["config_run_duration"], calendar, "config_run_duration")
        if stop < start:
            _refuse("config_run_duration", config["config_run_duration"], "the clock would run backwards",
                    "config_run_duration = a positive interval")
    elif config["config_stop_time"] != "none":
        stop = parse_time(config["config_stop_time"], calendar, "config_stop_time")
        if stop < start:
            _refuse("config_stop_time", config["config_stop_time"], "config_stop_time is earlier than "
                    "config_start_time! (the clock would run backwards)", "config_stop_time >= config_start_time")
    else:
        raise ValueError("Neither config_run_duration nor config_stop_time were specified.")
    clock = Clock(start, start, step, stop, calendar)
    if config["config_adaptive_timestep"]:
        interval = interval_seconds(config["config_adaptive_timestep_force_interval"], calendar,
                                    "config_adaptive_timestep_force_interval")
        if interval <= 0:
            raise ValueError("config_adaptive_timestep_force_interval must be positive")
        # the alarm starts at the start time with its previous ring one interval
        # before; it rings there at once and is reset to the start
        clock.force_interval, clock.force_prev = interval, start - interval
        clock.reset_force_alarm()
    return clock


# ---------------------------------------------------------------- configs


def check_config(config: dict, precision: str, velocity: dict) -> None:
    """Refuse by name what the forward mode does not carry (the step's own
    refusals included)."""
    if config.get("config_do_restart", False):
        _refuse("config_do_restart", True, "the forward mode's restart is its own (read_restart): the model's "
                "restart stream is not read", "config_do_restart = .false. with read_restart to resume")
    if config["config_time_integration"] != "forward_euler":
        _refuse("config_time_integration", config["config_time_integration"],
                "v8.4.2 carries forward Euler only (rk4 is logged as unsupported)",
                "config_time_integration = 'forward_euler'")
    statistics_v842.check_config(config)
    if config.get("config_write_albany_ascii_mesh", False):
        _refuse("config_write_albany_ascii_mesh", True, "Albany is not ported", "config_write_albany_ascii_mesh = "
                ".false.")
    analysis_v842.check_config(config)
    if int(config.get("config_year_digits", 4)) != 4:
        _refuse("config_year_digits", config["config_year_digits"], "the time stamps carry 4-digit years",
                "config_year_digits = 4")
    if quirk_corrections.option(config, "woof_sgh_init_mask") not in SGH_INIT_MASK:
        _refuse("woof_sgh_init_mask", config.get("woof_sgh_init_mask"), "not an option of this port",
                "woof_sgh_init_mask = 'v842' | 'corrected'")
    if config.get("config_SGH", False) and int(config.get("config_num_halos", 2)) < 1:
        raise ValueError("Subglacial hydrology requires config_num_halos >= 1")
    if (config["config_velocity_solver"] == "sia"
            and config["config_sia_tangent_slope_calculation"] == "from_vertex_barycentric"
            and config.get("on_a_sphere", False)):
        _refuse("on_a_sphere", True, "the barycentric weights on the sphere call glibc's tan and atan",
                "on_a_sphere = .false. with 'from_vertex_barycentric'")
    fe._check_config(config, precision, velocity)


def setup_config_options(config: dict) -> dict:
    """``li_setup_config_options``: a thermal solver advects its tracers."""
    config = dict(config)
    if config["config_thermal_solver"] in ("temperature", "enthalpy") and config["config_tracer_advection"] != "fo":
        config["config_tracer_advection"] = "fo"
    return config


# ---------------------------------------------------------------- the mesh's initialisation (host, RKIND)


def vertical_grid(fractions: np.ndarray, real) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``li_setup_vertical_grid`` on a cold start: (layerThicknessFractions,
    layerCenterSigma, layerInterfaceSigma). A sum more than 0.001 from one is
    the model's error."""
    f = np.array(fractions, real)
    n = f.size
    total = real(0.0)
    for k in range(n):
        total = total + f[k]
    if total != real(1.0):
        if abs(total - real(1.0)) > real(0.001):
            raise ValueError("The sum of layerThicknessFractions is different from 1.0 by more than 0.001.")
        f[0] = f[0] - (total - real(1.0))
    half = real(0.5)
    centre = np.empty(n, real)
    interface = np.empty(n + 1, real)
    centre[0] = half * f[0]
    interface[0] = real(0.0)
    for k in range(1, n):
        centre[k] = (centre[k - 1] + half * f[k - 1]) + half * f[k]
        interface[k] = interface[k - 1] + f[k - 1]
    interface[n] = real(1.0)
    return f, centre, interface


def edge_sign_on_cell(n_edges_on_cell: np.ndarray, edges_on_cell: np.ndarray, cells_on_edge: np.ndarray) -> np.ndarray:
    """``li_setup_sign_and_index_fields``' edgeSignOnCell: -1 where the cell
    is the edge's first, 1 otherwise, 0 past nEdgesOnCell."""
    n_cells, max_edges = edges_on_cell.shape
    sign = np.zeros((n_cells, max_edges), np.int32)
    for i in range(n_cells):
        for j in range(int(n_edges_on_cell[i])):
            sign[i, j] = -1 if i + 1 == cells_on_edge[edges_on_cell[i, j] - 1, 0] else 1
    return sign


def _cross(p, q):
    return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])


def _mag(x):
    return np.sqrt((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2])


def _dot0(x, n, real):
    """gfortran's sum(x * n) over three words: from zero, left to right."""
    return ((real(0.0) + x[0] * n[0]) + x[1] * n[1]) + x[2] * n[2]


def _point_in_triangle(point, corners, real) -> bool:
    """``mpas_point_in_polygon`` on the plane."""
    normal = (real(0.0), real(0.0), real(1.0))
    eps = real(1.0e-12)
    for i in range(3):
        a, b = corners[i], corners[(i + 1) % 3]
        vec1 = tuple(b[k] - a[k] for k in range(3))
        vec2 = tuple(point[k] - a[k] for k in range(3))
        if _dot0(_cross(vec1, vec2), normal, real) < (real(0.0) - eps):
            return False
    return True


def _distance(a, b, real):
    """``mpas_distance_plane``: sqrt(sum((a - b)**2))."""
    d = tuple(a[k] - b[k] for k in range(3))
    return np.sqrt(((real(0.0) + d[0] * d[0]) + d[1] * d[1]) + d[2] * d[2])


def _signed_area(a, b, c, real):
    """``mpas_triangle_signed_area_plane`` with the normal (0, 0, 1)."""
    ab = tuple(b[k] - a[k] for k in range(3))
    ac = tuple(c[k] - a[k] for k in range(3))
    cross = _cross(ab, ac)
    if _mag(cross) == real(0.0):
        return real(0.0)
    mag = _mag(cross)
    unit = tuple(cross[k] / mag for k in range(3))
    return (real(0.5) * _mag(cross)) * _dot0(unit, (real(0.0), real(0.0), real(1.0)), real)


def barycentric_weights(mesh: dict, real) -> tuple[np.ndarray, np.ndarray]:
    """``mpas_calculate_barycentric_weights_for_points`` at every vertex,
    searching from the vertex itself (``li_sia_block_init``), on the plane:
    (baryCellsOnVertex, baryWeightsOnVertex) in the (nVertices, 3) layout.
    Host numpy arrays in; a weight sum away from one (across a periodic
    mesh's seam) is only logged by the model, so it is kept."""
    cov = np.asarray(mesh["cellsOnVertex"])
    voc = np.asarray(mesh["verticesOnCell"])
    neoc = np.asarray(mesh["nEdgesOnCell"])
    n_vertices, degree = cov.shape
    n_cells = neoc.shape[0]
    if degree != 3:
        raise ValueError("Barycentric weights can only be calculated if vertexDegree is 3")
    cells = [tuple(real(np.asarray(mesh[k])[i]) for k in ("xCell", "yCell", "zCell")) for i in range(n_cells)]
    verts = [tuple(real(np.asarray(mesh[k])[i]) for k in ("xVertex", "yVertex", "zVertex"))
             for i in range(n_vertices)]
    huge = real(np.finfo(real).max)
    bary_cells = np.empty((n_vertices, 3), np.int32)
    bary_weights = np.empty((n_vertices, 3), real)

    def triangle(v):
        c1, c2, c3 = (int(x) for x in cov[v])
        if c1 <= n_cells and c2 <= n_cells and c3 <= n_cells:
            return (cells[c1 - 1], cells[c2 - 1], cells[c3 - 1])
        return None

    for v in range(n_vertices):
        point = verts[v]
        bary_cells[v] = cov[v]
        tri = triangle(v)
        if tri is not None:
            inside = _point_in_triangle(point, tri, real)
            nearest_distance, nearest = _distance(point, verts[v], real), v
        else:
            inside, nearest_distance, nearest = False, huge, v
        if not inside:
            for c in range(3):
                i_cell = int(cov[v, c])
                if i_cell > n_cells:
                    continue
                found = False
                for t in range(int(neoc[i_cell - 1])):
                    i_tri = int(voc[i_cell - 1, t])
                    if i_tri > n_vertices:
                        continue
                    tri = triangle(i_tri - 1)
                    if tri is not None:
                        if _point_in_triangle(point, tri, real):
                            inside = True
                            bary_cells[v] = cov[i_tri - 1]
                            found = True
                            break
                        distance = _distance(point, verts[i_tri - 1], real)
                    else:
                        distance = huge
                    if distance < nearest_distance:
                        nearest_distance, nearest = distance, i_tri - 1
                if found:
                    break
        if not inside:
            bary_cells[v] = cov[nearest]
        a, b, c = (cells[int(x) - 1] for x in bary_cells[v])
        area = _signed_area(a, b, c, real)
        if area > real(0.0):
            bary_weights[v] = (_signed_area(point, b, c, real) / area, _signed_area(point, c, a, real) / area,
                               _signed_area(point, a, b, real) / area)
        else:
            bary_weights[v] = real(0.0)
    return bary_cells, bary_weights


# ---------------------------------------------------------------- the forward mode


def _mesh_view(pools: dict) -> dict:
    """The mesh pool as the step takes it: with nCellsSolve and nEdgesSolve
    (one block owns every cell and edge)."""
    m = pools["mesh"]
    return dict(m, nCellsSolve=m["nEdgesOnCell"].shape[0], nEdgesSolve=m["cellsOnEdge"].shape[0])


def _set_days_since_start(pools: dict, clock: Clock, real) -> None:
    mesh = pools["mesh"]
    start = parse_time(mesh["simulationStartTime"], clock.calendar, "simulationStartTime")
    mesh["daysSinceStart"][...] = interval_dt(clock.now - start, real) / real(float(SECONDS_PER_DAY))


def _shift_time_levels(geometry: dict) -> None:
    """``mpas_pool_shift_time_levels`` on the geometry pool: vertexMask's two
    levels trade arrays."""
    geometry["vertexMask"], geometry["vertexMask@2"] = geometry["vertexMask@2"], geometry["vertexMask"]


def _sgh_init(precision: str, mesh: dict, geometry: dict, hydro: dict, config: dict) -> None:
    """``li_SGH_init`` for one block."""
    real = rkind.PRECISIONS[precision]
    g, h = geometry, hydro
    n_cells = g["thickness"].shape[0]
    masks = fe._mask_configs(config)
    hydro["deltatSGH"][...] = real(1.0e-4)
    if quirk_corrections.option(config, "woof_sgh_init_mask") == "corrected":
        geometry_v842.calculate_mask(mesh=mesh, geometry=g, **masks)
    _function(precision, "li_sgh_init_v842")(_grid(n_cells), (_BLOCK,), (
        h["waterThickness"], h["tillWaterThickness"], h["waterPressure"], g["thickness"], g["bedTopography"],
        g["cellMask"], real(config["config_ice_density"]), real(config["config_ocean_density"]),
        real(config["config_sea_level"]), real(config["config_SGH_till_max"]), np.int32(n_cells)))
    geometry_v842.calculate_mask(mesh=mesh, geometry=g, **masks)
    _function(precision, "li_sgh_pressure_diag_v842")(_grid(n_cells), (_BLOCK,), (
        h["effectivePressure"], h["hydropotentialBase"], h["hydropotential"], h["waterPressure"],
        h["waterThickness"], g["thickness"], g["bedTopography"], g["cellMask"], real(config["config_ice_density"]),
        np.int32(n_cells)))


def _log_statistics(stats: statistics_v842.Statistics) -> None:
    logging.getLogger("woof.hex.mpas_ice").info("%s", stats.report)


def _statistics(pools: dict, config: dict, on_statistics) -> None:
    """``li_compute_statistics`` at the current timestepNumber."""
    m, g = pools["mesh"], pools["geometry"]
    precision = rkind.precision_of(g["thickness"])
    n_cells, n_levels = g["layerThickness"].shape

    def layer_thickness() -> None:
        advection_v842._function(precision, "li_calculate_layer_thickness_v842")(_grid(n_cells), (_BLOCK,), (
            g["layerThickness"], g["thickness"], m["layerThicknessFractions"], np.int32(n_cells), np.int32(n_levels)))

    stats = statistics_v842.compute_statistics(
        mesh=m, geometry=g, thermal=pools["thermal"], velocity=pools["velocity"], config=config,
        itimestep=int(m["timestepNumber"].get()[0]), xtime=m["xtime"], apply_layer_thickness=layer_thickness)
    (on_statistics or _log_statistics)(stats)


def cold_start(*, pools: dict, config: dict, on_statistics=None, on_analysis=None,
               streams: dict[str, str] | None = None) -> Clock:
    """``li_simulation_clock_init``, ``li_core_init`` and ``li_core_run``'s
    initial solve on a cold start, in place on ``pools`` (MALI's pools by
    name: ``mesh``, ``geometry``, ``thermal``, ``velocity``, ``hydro``;
    device arrays, one-word device arrays for the scalars, host strings for
    ``xtime`` and ``simulationStartTime``; ``vertexMask@2`` is the second
    time level). Returns the clock.

    The mesh pool's ``edgeNormalVectors`` and ``coeffs_reconstruct`` are
    taken as given: ``mpas_rbf_interp_initialize`` and
    ``mpas_init_reconstruct`` (which li_core_init calls) are not ported, and
    the oracle takes both from its input too. A mesh without them, or with
    all-zero reconstruction coefficients (as MPAS static files declare
    them), is refused by name."""
    import cupy as cp
    m, g, t, v = pools["mesh"], pools["geometry"], pools["thermal"], pools["velocity"]
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    check_config(config, precision, v)
    for name in ("edgeNormalVectors", "coeffs_reconstruct"):
        if name not in m or not bool(cp.any(m[name] != 0)):
            _refuse(name, "missing" if name not in m else "all zero",
                    "mpas_rbf_interp_initialize and mpas_init_reconstruct are not ported: the mesh must carry them",
                    f"a mesh pool with {name} (MPAS writes it to its output and restart streams)")
    if config["config_velocity_solver"] == "simple":
        velocity_simple_v842.require_fields(mesh=m, velocity=v)
        velocity_simple_v842.check_config(config)
    config = setup_config_options(config)
    clock = clock_init(config)
    analysis = analysis_v842.init(config, clock, streams)   # li_core_init's li_analysis_init
    n_cells, n_levels = g["layerThickness"].shape

    # li_core_init: the time stamps and the time step
    m["xtime"] = clock.time_stamp
    if m["simulationStartTime"].strip() == "no_date_available":
        m["simulationStartTime"] = clock.time_stamp
    _set_days_since_start(pools, clock, real)
    m["deltat"][...] = interval_dt(clock.step, real)
    m["cellProcID"][...] = 0

    # landice_init_block
    g["vertexMask@2"][...] = g["vertexMask"]   # mpas_pool_initialize_time_levels
    g["vertexMask@2"][...] = 0
    fractions, centre, interface = vertical_grid(cp.asnumpy(m["layerThicknessFractions"]), real)
    m["layerThicknessFractions"][...] = cp.asarray(fractions)
    m["layerCenterSigma"][...] = cp.asarray(centre)
    m["layerInterfaceSigma"][...] = cp.asarray(interface)
    host = {k: cp.asnumpy(m[k]) for k in ("nEdgesOnCell", "edgesOnCell", "cellsOnEdge")}
    m["edgeSignOnCell"][...] = cp.asarray(edge_sign_on_cell(host["nEdgesOnCell"], host["edgesOnCell"],
                                                            host["cellsOnEdge"]))
    if (config["config_velocity_solver"] == "sia"
            and config["config_sia_tangent_slope_calculation"] == "from_vertex_barycentric"):
        bary_mesh = {k: cp.asnumpy(m[k]) for k in ("cellsOnVertex", "verticesOnCell", "nEdgesOnCell", "xCell",
                                                    "yCell", "zCell", "xVertex", "yVertex", "zVertex")}
        cells, weights = barycentric_weights(bary_mesh, real)
        m["baryCellsOnVertex"][...] = cp.asarray(cells)
        m["baryWeightsOnVertex"][...] = cp.asarray(weights)
    elif config["config_velocity_solver"] == "simple":
        velocity_simple_v842.block_init(mesh=m, geometry=g, velocity=v, config=config)
    _function(precision, "li_mask_init_v842")(_grid(n_cells), (_BLOCK,), (g["cellMask"], g["thickness"],
                                                                          np.int32(n_cells)))
    g["thicknessOld"][...] = g["thickness"]

    # li_thermal_init, li_SGH_init
    thermal_init_v842.thermal_init(
        thermal=t, thickness=g["thickness"], upper_surface=g["upperSurface"],
        layer_center_sigma=m["layerCenterSigma"], n_cells_solve=n_cells,
        config_ice_density=config["config_ice_density"], config_temperature_init=config["config_temperature_init"],
        config_surface_air_temperature_source=config["config_surface_air_temperature_source"],
        config_surface_air_temperature_value=config["config_surface_air_temperature_value"],
        config_surface_air_temperature_lapse_rate=config["config_surface_air_temperature_lapse_rate"],
        config_basal_heat_flux_source=config["config_basal_heat_flux_source"],
        config_basal_heat_flux_value=config["config_basal_heat_flux_value"])
    mesh = _mesh_view(pools)
    if config.get("config_SGH", False):
        subglacial_hydro_v842.check_config(config, precision)
        _sgh_init(precision, mesh, g, pools["hydro"], config)

    # li_core_run: timestepNumber, then li_core_initial_solve
    m["timestepNumber"][...] = 0
    geometry_v842.calculate_mask(mesh=mesh, geometry=g, **fe._mask_configs(config))
    fe._update_geometry(mesh, g, config)
    fe.velocity_solve(mesh=mesh, geometry=g, thermal=t, velocity=v, config=config)
    fe.diagnostic_vars(mesh=mesh, geometry=g, thermal=t, config=config, deltat=real(m["deltat"].get()[0]))
    if config.get("config_write_stats_on_startup", False):
        _statistics(pools, config, on_statistics)
    analysis_v842.compute_startup(analysis, pools=pools, config=config, on_analysis=on_analysis)
    _shift_time_levels(g)
    return clock


def advance(*, pools: dict, config: dict, clock: Clock, n_steps: int | None = None, on_statistics=None,
            on_analysis=None, streams: dict[str, str] | None = None, stream_ringing=None) -> int:
    """``li_core_run``'s time-step loop, in place, to the clock's stop time
    or ``n_steps`` steps, whichever is first. Returns the steps taken. The
    analysis members' alarms are rebuilt from their reference times at the
    clock's time (:mod:`~woof.hex.mpas_ice.analysis_v842`)."""
    m, g = pools["mesh"], pools["geometry"]
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    check_config(config, precision, pools["velocity"])
    config = setup_config_options(config)
    adaptive = bool(config["config_adaptive_timestep"])
    if adaptive and clock.force_interval is None:
        raise ValueError("an adaptive time step needs the clock's force alarm (clock_init with "
                         "config_adaptive_timestep)")
    mesh = _mesh_view(pools)
    analysis = analysis_v842.init(config, clock, streams)
    taken = 0
    while not clock.stopped and (n_steps is None or taken < n_steps):
        counters = (m["timestepNumber"].copy(), m["deltat"].copy())
        m["timestepNumber"][...] += 1
        # li_timestep
        m["deltat"][...] = real(0.0) if adaptive else interval_dt(clock.step, real)
        try:
            # an adaptive step's time step is refused (a whole positive number of
            # seconds, as the clock counts) as soon as set_timestep forms it,
            # before the step changes any prognostic field
            fe.forward_euler_step(
                mesh=mesh, geometry=g, thermal=pools["thermal"], velocity=pools["velocity"], config=config,
                hydro=pools.get("hydro"),
                seconds_to_next_force_time=(interval_dt(clock.seconds_to_next_force_time(), real) if adaptive
                                            else None))
        except BaseException:
            m["timestepNumber"][...], m["deltat"][...] = counters
            raise
        if adaptive:
            clock.step = int(float(m["deltat"].get()[0]))
        clock.now += clock.step
        m["xtime"] = clock.time_stamp
        _set_days_since_start(pools, clock, real)
        _shift_time_levels(g)
        clock.reset_force_alarm()
        taken += 1
        if statistics_v842.due(config, int(m["timestepNumber"].get()[0])):
            _statistics(pools, config, on_statistics)
        analysis_v842.compute(analysis, pools=pools, config=config, clock=clock, stream_ringing=stream_ringing,
                              on_analysis=on_analysis)
    return taken


def run(*, pools: dict, config: dict, on_statistics=None, on_analysis=None,
        streams: dict[str, str] | None = None, stream_ringing=None) -> Clock:
    """A whole cold-start run: :func:`cold_start`, then :func:`advance` to
    the stop time. Returns the clock."""
    clock = cold_start(pools=pools, config=config, on_statistics=on_statistics, on_analysis=on_analysis,
                       streams=streams)
    advance(pools=pools, config=config, clock=clock, on_statistics=on_statistics, on_analysis=on_analysis,
            streams=streams, stream_ringing=stream_ringing)
    return clock


# ---------------------------------------------------------------- restart


def write_restart(path, *, pools: dict, clock: Clock) -> None:
    """Every pool array word for word and the clock (its calendar and force
    alarm included) to ``path`` (.npz)."""
    import cupy as cp
    force = (-1, -1) if clock.force_interval is None else (clock.force_interval, clock.force_prev)
    arrays: dict[str, np.ndarray] = {
        "clock": np.array([clock.start, clock.now, clock.step, clock.stop, *force], np.int64),
        "clock_calendar": np.array(clock.calendar)}
    for pool, fields in pools.items():
        for name, value in fields.items():
            key = f"{pool}/{name}"
            if isinstance(value, str):
                arrays[f"str:{key}"] = np.array(value)
            elif isinstance(value, np.ndarray):
                arrays[f"host:{key}"] = value
            else:
                arrays[f"dev:{key}"] = cp.asnumpy(value)
    np.savez(Path(path), **arrays)


def read_restart(path, *, config: dict, precision: str | None = None) -> tuple[dict, Clock]:
    """The pools and clock :func:`write_restart` wrote. The clock resumes on
    its own calendar only: another ``config_calendar_type`` is refused. The
    configuration is checked against the stored run at load: its refusals
    (:func:`check_config`, at the stored arrays' precision), and the time
    step and the adaptive time step's force interval, which the stored clock
    carries; ``precision``, when given, must be the stored one."""
    import cupy as cp
    pools: dict[str, dict] = {}
    with np.load(Path(path)) as data:
        for key in data.files:
            if key in ("clock", "clock_calendar"):
                continue
            kind, full = key.split(":", 1)
            pool, name = full.split("/", 1)
            value = data[key]
            pools.setdefault(pool, {})[name] = {"str": lambda x: str(x), "host": lambda x: x.copy(),
                                                "dev": cp.asarray}[kind](value)
        start, now, step, stop, interval, prev = (int(x) for x in data["clock"])
        calendar = str(data["clock_calendar"])
    if calendar_of(config) != calendar:
        _refuse("config_calendar_type", config.get("config_calendar_type"),
                f"the restart's clock runs on the {calendar} calendar", f"config_calendar_type = '{calendar}'")
    clock = Clock(start, now, step, stop, calendar, None if interval < 0 else interval, None if interval < 0 else prev)
    stored = rkind.precision_of(pools["geometry"]["thickness"])
    if precision is not None and precision != stored:
        _refuse("precision", precision, f"the restart's arrays are RKIND {stored}", f"precision = '{stored}'")
    check_config(config, stored, pools["velocity"])
    if bool(config["config_adaptive_timestep"]) != (clock.force_interval is not None):
        _refuse("config_adaptive_timestep", config["config_adaptive_timestep"],
                "the restart's clock was " + ("adaptive" if clock.force_interval is not None else "fixed"),
                f"config_adaptive_timestep = {'.true.' if clock.force_interval is not None else '.false.'}")
    if clock.force_interval is not None:
        interval = interval_seconds(config["config_adaptive_timestep_force_interval"], calendar,
                                    "config_adaptive_timestep_force_interval")
        if interval != clock.force_interval:
            _refuse("config_adaptive_timestep_force_interval", config["config_adaptive_timestep_force_interval"],
                    f"the restart's force alarm rings every {clock.force_interval} s",
                    "the force interval the run started with")
    elif interval_seconds(config["config_dt"], calendar, "config_dt") != clock.step:
        _refuse("config_dt", config["config_dt"], f"the restart's clock steps {clock.step} s",
                "the config_dt the run started with")
    return pools, clock
