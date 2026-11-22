"""MPAS-Ocean v8.4.2: the forward mode around the split-explicit step, on the GPU.

:func:`cold_start` and :func:`advance` are what ``ocn_forward_mode_init`` and
``ocn_forward_mode_run`` (``mpas_ocn_forward_mode.F``) do to the state:

- the clock (:class:`Clock`): ``config_start_time`` and ``config_dt`` on
  ``config_calendar_type``'s calendar (``mpas_timekeeping_init``): the
  Registry default ``gregorian_noleap``, a 365-day year without 29 February,
  or ``gregorian`` (proleptic); in whole seconds;
- ``ocn_time_integration_split_init``: the barotropic subcycle count from
  ``config_dt`` and ``config_btr_dt`` (:func:`btr_subcycle_count`) and, on a
  cold start, the initial velocity split into barotropic and baroclinic
  (``config_filter_btr_mode`` with its top-layer reset);
- the state part of ``ocn_init_routines_block``: the boundary layer depth
  (half the top layer, where the diagnostics carry it), ``ocn_diagnostic_solve``,
  the missing cell's and the land levels' sentinels, the transport velocity
  (with ``config_use_standardGM`` GM's bolus velocity,
  :mod:`~woof.hex.mpas_ocean.gm_v842`, added, and its vectors on cells,
  ``ocn_reconstruct_gm_vectors``), the time levels initialised (the mesh
  part, the signs, ``derivTwo``, the advection and reconstruction
  coefficients and the mesh scaling, is the input's; ``mpas_reconstruct`` of
  the velocity is not run, as the step's own overwrites it);
- each step: the clock advanced, the absorbed fractions
  (:func:`~woof.hex.mpas_ocean.forcing_v842.fraction_absorbed`), the
  land-ice fluxes (:func:`~woof.hex.mpas_ocean.land_ice_v842.build_arrays`,
  'standalone'), the frazil arrays (:func:`~woof.hex.mpas_ocean.frazil_v842.build_arrays`, under
  ``config_use_frazil_ice_formation``), the step
  (:mod:`~woof.hex.mpas_ocean.split_step_v842`), ``xtime`` and
  ``daysSinceStartOfSim`` (``ocn_timestep``), and the state's two time levels
  shifted (``mpas_pool_shift_time_levels``: ``NAME`` and ``NAME@2`` trade
  arrays).

:func:`write_restart` and :func:`read_restart` round-trip a run: every pool
array, word for word, and the clock with its calendar; a run restarted from
them continues bitwise as if never stopped. A restart is resumed only on its
own calendar (``config_calendar_type``), and one written before the clock
carried its calendar only under an explicit ``'gregorian'``.

Held word for word over N steps to v8.4.2 (``ocn_timestep`` and
``ocn_time_integration_split_init`` cut verbatim, every module the step
reaches compiled verbatim; ``tools/mpas_ocean_oracle/run_forward.F90``,
``tests/test_mpas_ocean_forward_parity.py``).

Refused by name, beyond what the step refuses: RK4, and a restart start
(``config_do_restart``; :func:`read_restart` is this port's restart).
Analysis members and output streams change no state and are not run.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import diagnostics_v842, forcing_v842, frazil_v842, gm_v842, land_ice_v842, split_step_v842
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/mode_forward/mpas_ocn_forward_mode.F:460-700",
    "src/core_ocean/mode_forward/mpas_ocn_time_integration.F:80-164",
    "src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F:1940-2096",
    "src/core_ocean/shared/mpas_ocn_init_routines.F:529-700",
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:1844-1920",
) + land_ice_v842.SOURCE_FILES[1:2]
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128
#: the state's fields with two time levels (state pool, tracers pool)
TWO_LEVEL = {"state": ("layerThickness", "normalVelocity", "ssh", "normalBarotropicVelocity",
                       "normalBaroclinicVelocity", "sshSubcycle", "normalBarotropicVelocitySubcycle"),
             "tracers": ("activeTracers",)}


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("forward_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_forward_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = launch_options()
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


# ---------------------------------------------------------------- the clock


#: the calendars mpas_timekeeping_init sets (mpas_timekeeping.F:87-127) that the
#: clock carries; v8.4.2's built-in ESMF has no 360-day calendar
CALENDARS = ("gregorian", "gregorian_noleap")
#: config_calendar_type's Registry default (both cores)
DEFAULT_CALENDAR = "gregorian_noleap"
_NOLEAP_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def calendar_of(configs: dict) -> str:
    """``config_calendar_type`` (the Registry default when absent), refused by
    name unless the clock carries it."""
    calendar = configs.get("config_calendar_type", DEFAULT_CALENDAR)
    if calendar not in CALENDARS:
        raise ConfigurationRefusal(knob="config_calendar_type", value=calendar,
                                   reason="the clock carries MPAS's gregorian and gregorian_noleap calendars only",
                                   declaration="config_calendar_type = 'gregorian_noleap' | 'gregorian'")
    return calendar


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _days_from_civil(y: int, m: int, d: int, calendar: str) -> int:
    """Days since 0001-01-01 on the proleptic Gregorian calendar, or on the
    365-day year (``gregorian_noleap``)."""
    if calendar == "gregorian_noleap":
        return (y - 1) * 365 + _NOLEAP_MONTH_START[m - 1] + d - 1
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 306


def parse_time(text: str, calendar: str, knob: str = "config_start_time") -> int:
    """An MPAS date-time string 'YYYY-MM-DD_hh:mm:ss' in seconds since
    0001-01-01 on ``calendar`` (one of :data:`CALENDARS`; there is no
    default). A date the calendar does not have (29 February of the 365-day
    year) is refused, as ``isValidDate`` rejects it
    (``mpas_timekeeping.F:2146-2185``), naming ``knob``, where the string came from."""
    if calendar not in CALENDARS:
        raise ValueError(f"calendar {calendar!r} is not one of {CALENDARS}")
    date, clock = text.split("_")
    y, m, d = (int(x) for x in date.split("-"))
    hh, mm, ss = (int(x) for x in clock.split(":"))
    days = _MONTH_DAYS[m - 1] + (calendar == "gregorian" and m == 2 and _is_leap(y)) if 1 <= m <= 12 else 0
    if not 1 <= d <= days:
        raise ConfigurationRefusal(knob=knob, value=text,
                                   reason=f"MPAS_setTime: {text} is not a date of the {calendar} calendar",
                                   declaration="a date of config_calendar_type's calendar")
    return ((_days_from_civil(y, m, d, calendar) * 24 + hh) * 60 + mm) * 60 + ss


def format_time(seconds: int, calendar: str) -> str:
    """Seconds since 0001-01-01 on ``calendar`` as 'YYYY-MM-DD_hh:mm:ss'."""
    days, rest = divmod(seconds, 86400)
    clock = f"{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    if calendar == "gregorian_noleap":
        y, yday = divmod(days, 365)
        m = max(i for i, first in enumerate(_NOLEAP_MONTH_START) if first <= yday)
        return f"{y + 1:04d}-{m + 1:02d}-{yday - _NOLEAP_MONTH_START[m] + 1:02d}_{clock}"
    if calendar != "gregorian":
        raise ValueError(f"calendar {calendar!r} is not one of {CALENDARS}")
    z = days + 306
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    y += m <= 2
    return f"{y:04d}-{m:02d}-{d:02d}_{clock}"


def interval_seconds(text: str) -> int:
    """An MPAS interval string '[DDDD_]hh:mm:ss' in seconds."""
    days, _, clock = text.rpartition("_")
    h, m, s = (int(x) for x in clock.split(":"))
    return ((int(days or 0) * 24 + h) * 60 + m) * 60 + s


@dataclass
class Clock:
    """MPAS's clock on whole seconds since 0001-01-01_00:00:00 of its
    calendar, which every clock names (one of :data:`CALENDARS`)."""
    start: int
    now: int
    step: int
    calendar: str

    def __post_init__(self):
        if self.calendar not in CALENDARS:
            raise ValueError(f"calendar {self.calendar!r} is not one of {CALENDARS}")

    @classmethod
    def from_configs(cls, configs: dict) -> "Clock":
        calendar = calendar_of(configs)
        start = parse_time(configs["config_start_time"], calendar)
        return cls(start, start, interval_seconds(configs["config_dt"]), calendar)

    @property
    def time_stamp(self) -> str:
        return format_time(self.now, self.calendar)

    def seconds(self, text: str, knob: str = "config_start_time") -> int:
        """A date-time string in seconds on the clock's calendar (``knob``
        names where it came from, for a refusal)."""
        return parse_time(text, self.calendar, knob)

    def check_calendar(self, configs: dict) -> None:
        """Refuse a clock on another calendar than ``config_calendar_type``'s:
        its seconds count other dates."""
        calendar = calendar_of(configs)
        if self.calendar != calendar:
            raise ConfigurationRefusal(
                knob="config_calendar_type", value=calendar,
                reason=f"the clock runs on the {self.calendar} calendar (its restart's or its own), and its "
                       f"seconds are dates of that calendar only",
                declaration=f"config_calendar_type = '{self.calendar}', or a cold start")


def btr_subcycle_count(config_dt: str, config_btr_dt: str) -> int:
    """``ocn_time_integration_split_init``'s nBtrSubcycles: config_dt over
    config_btr_dt, one more for a remainder."""
    n, rem = divmod(interval_seconds(config_dt), interval_seconds(config_btr_dt))
    return n + (1 if rem > 0 else 0)


# ---------------------------------------------------------------- the run


def check_config(configs: dict) -> None:
    if configs["config_time_integrator"] == "RK4":
        raise ConfigurationRefusal(knob="config_time_integrator", value="RK4",
                                   reason="the RK4 integrator (mpas_ocn_time_integration_rk4.F) is not ported",
                                   declaration="config_time_integrator = 'split_explicit' | 'unsplit_explicit'")
    if configs.get("config_do_restart", False):
        raise ConfigurationRefusal(knob="config_do_restart", value=True,
                                   reason="this port restarts from its own restart (read_restart), not MPAS's "
                                          "restart stream",
                                   declaration="config_do_restart = .false.; restart with read_restart")


def _two_level(pools: dict) -> dict:
    """The two-level fields present: :data:`TWO_LEVEL` and the frazil and land-ice packages'."""
    names = dict(TWO_LEVEL)
    packages = frazil_v842.TWO_LEVEL + land_ice_v842.TWO_LEVEL_FLUXES + land_ice_v842.TWO_LEVEL_COUPLING
    present = tuple(n for n in packages if pools["state"].get(n) is not None)
    if present:
        names["state"] = names["state"] + present
    return names


def _state(pools: dict, level: int) -> dict:
    st = pools["state"]
    return {n: st[n if level == 1 else f"{n}@2"] for n in ("layerThickness", "normalVelocity", "ssh")}


def _tracers(pools: dict, level: int) -> dict:
    tr = pools["tracers"]
    return {"activeTracers": tr["activeTracers" if level == 1 else "activeTracers@2"],
            "index_temperature": tr["index_temperature"], "index_salinity": tr["index_salinity"]}


def cold_start(*, pools: dict, configs: dict) -> Clock:
    """``ocn_time_integration_split_init`` and the state part of
    ``ocn_init_routines_block`` on a cold start; the clock at its start."""
    check_config(configs)
    m, st, tr, d = pools["mesh"], pools["state"], pools["tracers"], pools["diagnostics"]
    precision = _precision_of(st["layerThickness"])
    real = PRECISIONS[precision]
    i32 = np.int32
    nc, ne, nl = int(m["nCells"]), int(m["nEdges"]), int(m["nVertLevels"])
    nt = int(tr["activeTracers"].shape[2])
    clock = Clock.from_configs(configs)
    dt = float(clock.step)

    # ocn_time_integration_split_init
    integrator = configs["config_time_integrator"]
    if integrator == "unsplit_explicit":
        _launch(precision, "ocn_fw_unsplit_init_v842", ne, (
            st["normalBarotropicVelocity"], st["normalBaroclinicVelocity"], st["normalVelocity"], i32(ne), i32(nl)))
    elif integrator == "split_explicit":
        if configs.get("config_filter_btr_mode", False):
            top = real(m["refBottomDepth"][0].item())
            _launch(precision, "ocn_fw_filter_thickness_v842", nc, (st["layerThickness"], top, i32(nc), i32(nl)))
        _launch(precision, "ocn_fw_split_init_v842", ne, (
            st["normalBarotropicVelocity"], st["normalBaroclinicVelocity"], st["normalVelocity"],
            st["layerThickness"], m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(ne), i32(nl)))
        if configs.get("config_filter_btr_mode", False):
            st["normalVelocity"][...] = st["normalBaroclinicVelocity"]
            st["normalBarotropicVelocity"][...] = 0

    # ocn_init_routines_block's state part
    gm = bool(configs.get("config_use_standardGM", False))
    if d.get("boundaryLayerDepth") is not None:
        # read by GM's surface-layer tapering (and by KPP, which is refused) alone
        d["boundaryLayerDepth"][:nc] = st["layerThickness"][:nc, 0] * real(0.5)
    diagnostics_v842.diagnostic_solve(dt=dt, mesh=m, state=_state(pools, 1), tracers=_tracers(pools, 1),
                                      diagnostics=d, forcing=pools["forcing"], configs=configs)
    m["areaCell"][nc] = real(-1.0e34)
    st["layerThickness"][nc] = 0
    _launch(precision, "ocn_fw_init_edges_v842", ne, (
        st["normalVelocity"], d["normalTransportVelocity"], m["maxLevelEdgeTop"], m["maxLevelEdgeBot"], i32(ne),
        i32(nl)))
    _launch(precision, "ocn_fw_init_cells_v842", nc, (tr["activeTracers"], m["maxLevelCell"], i32(nc), i32(nl),
                                                       i32(nt)))
    if gm:
        # mpas_ocn_init_routines.F:656-664 and 685-687
        gm_v842.compute_bolus_velocity(mesh=m, diagnostics=d, configs=configs)
        d["normalTransportVelocity"][:ne] += d["normalGMBolusVelocity"][:ne]
        split_step_v842.reconstruct_gm_vectors(mesh=m, diagnostics=d, configs=configs)
    for pool, names in _two_level(pools).items():
        for name in names:
            pools[pool][f"{name}@2"][...] = pools[pool][name]
    # the land-ice fluxes for potential output at startup (mpas_ocn_init_routines.F:690-696)
    forcing_v842.fraction_absorbed(mesh=m, layer_thickness=st["layerThickness"], diagnostics=d,
                                   forcing=pools["forcing"])
    land_ice_v842.build_arrays(dt=dt, mesh=m, state=st, diagnostics=d, forcing=pools["forcing"], configs=configs)
    return clock


def _shift(pools: dict) -> None:
    """``mpas_pool_shift_time_levels``: each two-level field's arrays trade names."""
    for pool, names in _two_level(pools).items():
        p = pools[pool]
        for name in names:
            p[name], p[f"{name}@2"] = p[f"{name}@2"], p[name]


def advance(*, pools: dict, configs: dict, clock: Clock, n_steps: int) -> None:
    """``ocn_forward_mode_run``'s loop, ``n_steps`` times, in place on the pools and clock."""
    import cupy as cp
    check_config(configs)
    st, d = pools["state"], pools["diagnostics"]
    precision = _precision_of(st["layerThickness"])
    real = PRECISIONS[precision]
    n_btr = btr_subcycle_count(configs["config_dt"], configs["config_btr_dt"])
    dt = float(clock.step)
    clock.check_calendar(configs)
    start = clock.seconds(configs["diagnostics:simulationStartTime"], "diagnostics:simulationStartTime")
    days_per_second = real(real(1.0) / real(86400.0))
    for _ in range(n_steps):
        clock.now += clock.step
        forcing_v842.fraction_absorbed(mesh=pools["mesh"], layer_thickness=st["layerThickness"], diagnostics=d,
                                       forcing=pools["forcing"])
        land_ice_v842.build_arrays(dt=dt, mesh=pools["mesh"], state=st, diagnostics=d, forcing=pools["forcing"],
                                   configs=configs)
        frazil_v842.build_arrays(dt=dt, mesh=pools["mesh"], state=st, tracers=_tracers(pools, 1), diagnostics=d,
                                 forcing=pools["forcing"], configs=configs)
        split_step_v842.split_step(dt=dt, n_btr_subcycles=n_btr, pools=pools, configs=configs)
        # ocn_timestep: xtime, and the days since the simulation's start
        d["xtime"] = clock.time_stamp
        days = real(real(clock.now - start) * days_per_second)
        d["daysSinceStartOfSim"][...] = cp.asarray(days)
        _shift(pools)


# ---------------------------------------------------------------- restart


def write_restart(path, *, pools: dict, clock: Clock) -> None:
    """Every pool array word for word, every dimension and the clock with its
    calendar, to ``path`` (.npz). Host pools (numpy arrays, strings and
    integers) need no GPU."""
    arrays: dict[str, np.ndarray] = {"clock": np.array([clock.start, clock.now, clock.step], np.int64),
                                     "clock_calendar": np.array(clock.calendar)}
    for pool, fields in pools.items():
        for name, value in fields.items():
            key = f"{pool}/{name}"
            if value is None:
                continue
            if isinstance(value, str):
                arrays[f"str:{key}"] = np.array(value)
            elif isinstance(value, (int, np.integer)):
                arrays[f"int:{key}"] = np.array(int(value), np.int64)
            elif isinstance(value, np.ndarray):
                arrays[f"host:{key}"] = value
            else:
                import cupy as cp
                arrays[f"dev:{key}"] = cp.asnumpy(value)
    np.savez(Path(path), **arrays)


def _device(value):
    import cupy as cp
    return cp.asarray(value)


#: a restart written before the clock carried its calendar counts Gregorian seconds
LEGACY_RESTART_CALENDAR = "gregorian"


def read_restart(path, *, like: dict, configs: dict) -> tuple[dict, Clock]:
    """The pools and clock :func:`write_restart` wrote; ``like`` (pools of the
    same layout) names the fields that are absent (``None``).

    The clock's calendar must be ``configs``' (``config_calendar_type``, the
    Registry default when absent): a restart resumed on another calendar is
    refused by name. A restart written before the clock carried its calendar
    counts Gregorian seconds: it resumes only under an explicit
    ``config_calendar_type = 'gregorian'``, and is refused otherwise (the
    default would silently read its seconds as other dates)."""
    with np.load(Path(path)) as data:
        pools: dict[str, dict] = {pool: {name: None for name in fields} for pool, fields in like.items()}
        for key in data.files:
            if key in ("clock", "clock_calendar"):
                continue
            kind, full = key.split(":", 1)
            pool, name = full.split("/", 1)
            value = data[key]
            pools.setdefault(pool, {})[name] = {"str": lambda v: str(v), "int": lambda v: int(v),
                                                "host": lambda v: v.copy(),
                                                "dev": _device}[kind](value)
        start, now, step = (int(x) for x in data["clock"])
        legacy = "clock_calendar" not in data.files
        calendar = LEGACY_RESTART_CALENDAR if legacy else str(data["clock_calendar"])
    if legacy and configs.get("config_calendar_type") != LEGACY_RESTART_CALENDAR:
        raise ConfigurationRefusal(
            knob="config_calendar_type", value=configs.get("config_calendar_type"),
            reason=f"{Path(path).name} was written before the clock carried its calendar and counts seconds on "
                   "the Gregorian calendar; resuming it needs config_calendar_type set to 'gregorian' explicitly "
                   "(the default gregorian_noleap would read its seconds as other dates)",
            declaration="config_calendar_type = 'gregorian' to resume it, or a cold start on another calendar")
    clock = Clock(start, now, step, calendar)
    clock.check_calendar(configs)
    return pools, clock
