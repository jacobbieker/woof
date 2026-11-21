"""MALI v8.4.2's analysis members: ``globalStats`` and ``regionalStats``.

``li_analysis_setup_packages`` turns a member's package on when
``config_AM_<member>_enable``; ``li_analysis_init`` (in ``li_core_init``)
gives each enabled member whose ``config_AM_<member>_compute_interval`` is
not 'output_interval', and whose ``config_AM_<member>_stream_name`` is not
'none', a recurring clock alarm (``<member>CMPALRM``) on its stream's
reference time every interval ('dt': the clock's time step), reset at once.
A stream a member names must exist. Then:

- on a cold start, after the startup statistics and before the time levels
  shift, every enabled member with ``config_AM_<member>_compute_on_startup``
  computes (:func:`compute_startup`);
- after every step's statistics (:func:`compute`), a member at
  'output_interval' computes when its stream's output alarm rings (the
  caller's streams: ``stream_ringing``, never by default); any other member
  computes when its alarm rings, after resetting it.

Each compute leaves its words in its pool (``globalStatsAM``: one word per
field; ``regionalStatsAM``: one per region), named as in the Registry, and
goes to ``on_analysis`` as a :class:`Compute` when given. ``regionalStats``
reads the ``regions`` pool's ``regionCellMasks`` (cells by regions).

Both members are reproduced on the host at RKIND, one correctly rounded
operation at a time in the Fortran's order: every term formed as the
Fortran writes it (the masks as reals, multiplied in place), the cell and
edge sums from 0 in sequence (``np.add.accumulate``, never pairwise; the
grounding-line flux edge by edge, level by level), the extrema by strict
comparison from the Fortran's sentinels (100000 m for the global minimum
thickness, 10000 m for a region's, 0 for the maxima). A division of 0 by 0
gives the NaN x86-64 gives (the sign bit set), whatever the host.

On one rank every reduction is the identity. Held to v8.4.2's driver and
members compiled verbatim (``tools/mali_oracle/run_forward.F90``,
``tests/test_mali_forward_parity.py``).

One v8.4.2 quirk has a named opt-in correction (``woof_analysis_empty``):
the means divide by an area that is 0 when no ice (or no grounded, or no
floating ice) is counted, so they are NaN; and the minimum thickness of no
ice is the sentinel (100000 m, a region's 10000 m). 'v842' (the default)
keeps the model's words; 'corrected' reports 0 for both.

The alarms are not stored: MPAS's restart does not store them either, and
an alarm rebuilt from its reference time at any time between two steps is
the one that ran there (its previous ring is the latest ring time not
after now), so :func:`init` rebuilds them wherever a run resumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import rkind

SOURCE_FILES = ("src/core_landice/analysis_members/mpas_li_analysis_driver.F:84-460",
                "src/core_landice/analysis_members/mpas_li_global_stats.F:131-512",
                "src/core_landice/analysis_members/mpas_li_regional_stats.F:121-598",
                "src/core_landice/mode_forward/mpas_li_core.F:237-238,410-418,733-741",
                "src/framework/mpas_timekeeping.F:490-566,814-985")
MEMBERS = ("globalStats", "regionalStats")
#: globalStatsAM's fields, Registry order
GLOBAL_FIELDS = ("totalIceVolume", "volumeAboveFloatation", "totalIceArea", "floatingIceVolume", "floatingIceArea",
                 "groundedIceVolume", "groundedIceArea", "iceThicknessMean", "iceThicknessMax", "iceThicknessMin",
                 "totalSfcMassBal", "avgNetAccumulation", "totalBasalMassBal", "totalGroundedBasalMassBal",
                 "avgGroundedBasalMelt", "totalFloatingBasalMassBal", "avgSubshelfMelt", "totalCalvingFlux",
                 "groundingLineFlux", "surfaceSpeedMax", "basalSpeedMax")
#: regionalStatsAM's fields (nRegions each), Registry order
REGIONAL_FIELDS = ("regionalIceArea", "regionalIceVolume", "regionalVolumeAboveFloatation",
                   "regionalGroundedIceArea", "regionalGroundedIceVolume", "regionalFloatingIceArea",
                   "regionalFloatingIceVolume", "regionalIceThicknessMin", "regionalIceThicknessMax",
                   "regionalIceThicknessMean", "regionalSumSfcMassBal", "regionalAvgNetAccumulation",
                   "regionalSumBasalMassBal", "regionalSumGroundedBasalMassBal", "regionalAvgGroundedBasalMelt",
                   "regionalSumFloatingBasalMassBal", "regionalAvgSubshelfMelt", "regionalSumCalvingFlux",
                   "regionalSumGroundingLineFlux", "regionalSurfaceSpeedMax", "regionalBasalSpeedMax")
FIELDS = {"globalStats": GLOBAL_FIELDS, "regionalStats": REGIONAL_FIELDS}
#: the Registry's defaults
DEFAULTS = {"enable": False, "compute_interval": "output_interval", "compute_on_startup": True,
            "write_on_startup": True}
#: woof_analysis_empty: 'v842' (NaN means, sentinel minima) or 'corrected' (0)
ANALYSIS_EMPTY = ("v842", "corrected")
MASK_ICE, MASK_FLOATING, MASK_GROUNDING_LINE = 32, 4, 256


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def option(config: dict, member: str, name: str):
    """``config_AM_<member>_<name>``, at the Registry's default when absent."""
    default = f"{member}Output" if name == "stream_name" else DEFAULTS[name]
    return config.get(f"config_AM_{member}_{name}", default)


def enabled(config: dict) -> tuple[str, ...]:
    """The enabled members, in the driver's order."""
    return tuple(m for m in MEMBERS if bool(option(config, m, "enable")))


def check_config(config: dict) -> None:
    """Refuse by name what the members do not carry."""
    if config.get("woof_analysis_empty", "v842") not in ANALYSIS_EMPTY:
        _refuse("woof_analysis_empty", config["woof_analysis_empty"], "not an option of this port",
                "woof_analysis_empty = 'v842' | 'corrected'")
    for knob in config:
        if knob.startswith("config_AM_") and not any(
                knob == f"config_AM_{m}_{n}" for m in MEMBERS for n in (*DEFAULTS, "stream_name")):
            _refuse(knob, config[knob], "v8.4.2's land-ice core has the globalStats and regionalStats analysis "
                    "members only", "config_AM_globalStats_* or config_AM_regionalStats_*")


@dataclass
class Alarm:
    """A recurring MPAS clock alarm on whole seconds (forward clock)."""
    interval: int
    prev: int

    def ringing(self, now: int) -> bool:
        """``mpas_in_ringing_envelope``: prevRingTime + interval <= now."""
        return self.prev + self.interval <= now

    def reset(self, now: int) -> None:
        """``mpas_reset_clock_alarm``: when ringing, the previous ring
        becomes now less the remainder of (now - prev) by the interval."""
        if self.ringing(now):
            self.prev = now - (now - self.prev) % self.interval


@dataclass
class Analysis:
    """The driver's state: the enabled members and their compute alarms."""
    members: tuple[str, ...]
    alarms: dict[str, Alarm] = field(default_factory=dict)


@dataclass(frozen=True)
class Compute:
    """One member's compute: the member, the xtime it computed at, and its
    words (its fields in Registry order, a region's words consecutive within
    a field)."""
    member: str
    xtime: str
    words: np.ndarray


def init(config: dict, clock, streams: dict[str, str] | None = None) -> Analysis:
    """``li_analysis_init`` at the clock's current time. ``streams`` maps
    each stream that exists to its reference time (a time stamp); by default
    every stream a member names exists with the run's start as its reference
    time (MPAS's default for a stream without one)."""
    from . import forward_v842 as fwd
    check_config(config)
    analysis = Analysis(enabled(config))
    for member in analysis.members:
        interval = str(option(config, member, "compute_interval"))
        stream = str(option(config, member, "stream_name"))
        if interval == "dt":
            # the clock's time step at li_core_init: config_dt (an adaptive step changes it later)
            seconds = fwd.interval_seconds(config["config_dt"], clock.calendar, "config_dt")
        elif interval != "output_interval":
            seconds = fwd.interval_seconds(interval, clock.calendar, f"config_AM_{member}_compute_interval")
            if seconds <= 0:
                _refuse(f"config_AM_{member}_compute_interval", interval, "an alarm every no time rings at every "
                        "check and its reset divides by zero", f"config_AM_{member}_compute_interval = a positive "
                        "interval, 'dt' or 'output_interval'")
        if stream == "none":
            continue
        if streams is not None and stream not in streams:
            raise ValueError(f"MPAS-landice: ERROR: Stream {stream} does not exist. Exiting...")
        if interval == "output_interval":
            continue
        reference = clock.start if streams is None else fwd.parse_time(streams[stream], clock.calendar)
        alarm = Alarm(seconds, reference - seconds)
        alarm.reset(clock.now)
        analysis.alarms[member] = alarm
    return analysis


def compute_startup(analysis: Analysis, *, pools: dict, config: dict, on_analysis=None) -> None:
    """``li_analysis_compute_startup`` (a cold start's)."""
    for member in analysis.members:
        if bool(option(config, member, "compute_on_startup")):
            _compute(member, pools, config, on_analysis)


def compute(analysis: Analysis, *, pools: dict, config: dict, clock, stream_ringing=None, on_analysis=None) -> None:
    """``li_analysis_compute`` after a step: ``stream_ringing(name)`` says
    whether a stream's output alarm rings (never when not given)."""
    for member in analysis.members:
        stream = str(option(config, member, "stream_name"))
        if option(config, member, "compute_interval") == "output_interval" and stream != "none":
            if stream_ringing is not None and stream_ringing(stream):
                _compute(member, pools, config, on_analysis)
        elif member in analysis.alarms and analysis.alarms[member].ringing(clock.now):
            analysis.alarms[member].reset(clock.now)
            _compute(member, pools, config, on_analysis)


# ---------------------------------------------------------------- the members (host, RKIND)


def _host(array) -> np.ndarray:
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def _sum(terms: np.ndarray, real):
    """``s = 0; do i: s = s + terms(i)``."""
    return np.add.accumulate(np.concatenate([np.zeros(1, real), np.asarray(terms, real).ravel()]))[-1]


def _quiet(x, real):
    bits = np.array(x, real).view(np.uint32 if real is np.float32 else np.uint64)
    bits = bits | (np.uint32(0x00400000) if real is np.float32 else np.uint64(0x0008000000000000))
    return bits.view(real)[()]


def _div(a, b, real):
    """``a / b`` as x86-64's ``divss``/``divsd``: a NaN operand comes back
    quieted (the first one), an invalid quotient (0/0, inf/inf) is the
    default NaN with its sign bit set."""
    a, b = real(a), real(b)
    if np.isnan(a):
        return _quiet(a, real)
    if np.isnan(b):
        return _quiet(b, real)
    with np.errstate(all="ignore"):
        q = real(a / b)
    if np.isnan(q):
        return np.array(0xFFC00000 if real is np.float32 else 0xFFF8000000000000,
                        np.uint32 if real is np.float32 else np.uint64).view(real)[()]
    return q


def _running(values: np.ndarray, assigned: np.ndarray, init, larger: bool, positive: bool = False):
    """``if (values(i) > best [.and. values(i) > 0]) best = assigned(i)``
    over i in order (``<`` when not ``larger``)."""
    best = init
    candidates = values > 0 if positive or larger else values < init
    for i in np.flatnonzero(candidates):
        v = values[i]
        if (v > best) if larger else (v < best and (not positive or v > 0)):
            best = assigned[i]
    return best


def _inputs(pools: dict, config: dict, real) -> dict:
    m, g, v = pools["mesh"], pools["geometry"], pools["velocity"]
    out = {k: _host(m[k]).astype(real, copy=False) for k in ("areaCell", "dvEdge")}
    out["cellsOnEdge"] = _host(m["cellsOnEdge"])
    out["deltat"] = real(_host(m["deltat"]).ravel()[0])
    for k in ("thickness", "bedTopography", "sfcMassBal", "basalMassBal", "groundedBasalMassBal",
              "floatingBasalMassBal", "calvingThickness", "layerThicknessEdge"):
        out[k] = _host(g[k]).astype(real, copy=False)
    for k in ("cellMask", "edgeMask"):
        out[k] = _host(g[k])
    for k in ("surfaceSpeed", "basalSpeed", "layerNormalVelocity"):
        out[k] = _host(v[k]).astype(real, copy=False)
    out["rhoi"], out["rhow"] = real(config["config_ice_density"]), real(config["config_ocean_density"])
    out["sea_level"] = real(config["config_sea_level"])
    out["scyr"] = real(31536000.0)
    mask = out["cellMask"]
    ice = (mask & MASK_ICE) == MASK_ICE
    out["ice"] = ice.astype(real)
    out["floating"] = ((mask & MASK_FLOATING) // MASK_FLOATING).astype(real)
    out["grounded"] = np.where(ice, 1 - (mask & MASK_FLOATING) // MASK_FLOATING, 0).astype(real)
    out["grounded_ice"] = ice & ((mask & MASK_FLOATING) != MASK_FLOATING)
    # min(0.0_RKIND, bed - sea level)
    below = out["bedTopography"] - out["sea_level"]
    out["below"] = np.where(below < 0, below, real(0.0)).astype(real)
    return out


def _gl_terms(x: dict, real) -> tuple[np.ndarray, np.ndarray]:
    """Each grounding-line edge's level terms of the flux (kg/yr), edge by
    edge, and the edges (0-based)."""
    edges = np.flatnonzero((x["edgeMask"] & MASK_GROUNDING_LINE) == MASK_GROUNDING_LINE)
    cell1 = x["cellsOnEdge"][edges, 0]
    if edges.size and (cell1.min() < 1 or x["cellsOnEdge"][edges, 1].min() < 1):
        raise ValueError("a grounding-line edge lies on the mesh's boundary: cellsOnEdge names no cell there")
    sign = np.where(x["grounded_ice"][cell1 - 1], real(1.0), real(-1.0)).astype(real)
    terms = (((((sign[:, None] * x["layerNormalVelocity"][edges]) * x["dvEdge"][edges, None])
               * x["layerThicknessEdge"][edges]) * x["scyr"]) * x["rhoi"])
    return terms.astype(real), edges


def global_stats(pools: dict, config: dict) -> dict:
    """``li_compute_global_stats``: its fields' words at RKIND."""
    real = rkind.PRECISIONS[rkind.precision_of(pools["geometry"]["thickness"])]
    x = _inputs(pools, config, real)
    a, t, ice, gr, fl, scyr = x["areaCell"], x["thickness"], x["ice"], x["grounded"], x["floating"], x["scyr"]
    ratio = real(x["rhow"] / x["rhoi"])
    sums = [
        _sum(ice * a, real), _sum((ice * a) * t, real), _sum(gr * a, real), _sum((gr * a) * t, real),
        _sum(fl * a, real), _sum((fl * a) * t, real), _sum(((ice * a) * x["sfcMassBal"]) * scyr, real),
        _sum(((ice * a) * x["basalMassBal"]) * scyr, real),
        _sum(((ice * a) * x["groundedBasalMassBal"]) * scyr, real),
        _sum(((fl * a) * x["floatingBasalMassBal"]) * scyr, real),
        _sum(((x["calvingThickness"] * a) * x["rhoi"]) / real(x["deltat"] / scyr), real),
        _sum((gr * a) * (t + ratio * x["below"]), real), _sum(_gl_terms(x, real)[0], real)]
    area, vol, garea, gvol, farea, fvol, smb, bmb, gbmb, fbmb, calving, vaf, gl = sums
    rhoi = x["rhoi"]
    t_max = _running(t, t, real(0.0), True)
    t_min = _running(t, t, real(100000.0), False, positive=True)
    s_max = _running(x["surfaceSpeed"], x["surfaceSpeed"], real(0.0), True)
    b_max = _running(x["basalSpeed"], x["basalSpeed"], real(0.0), True)
    out = {
        "totalIceArea": area, "totalIceVolume": vol, "groundedIceArea": garea, "groundedIceVolume": gvol,
        "floatingIceArea": farea, "floatingIceVolume": fvol, "totalSfcMassBal": smb, "totalBasalMassBal": bmb,
        "totalGroundedBasalMassBal": gbmb, "totalFloatingBasalMassBal": fbmb, "totalCalvingFlux": calving,
        "volumeAboveFloatation": vaf, "groundingLineFlux": gl,
        "iceThicknessMean": _div(vol, area, real),
        "avgNetAccumulation": _div(_div(smb, area, real), rhoi, real),
        "avgGroundedBasalMelt": _div(_div(real(real(-1.0) * gbmb), garea, real), rhoi, real),
        "avgSubshelfMelt": _div(_div(real(real(-1.0) * fbmb), farea, real), rhoi, real),
        "iceThicknessMin": t_min, "iceThicknessMax": t_max,
        "surfaceSpeedMax": real(s_max * scyr), "basalSpeedMax": real(b_max * scyr)}
    if config.get("woof_analysis_empty", "v842") == "corrected":
        for name, over in (("iceThicknessMean", area), ("avgNetAccumulation", area),
                           ("avgGroundedBasalMelt", garea), ("avgSubshelfMelt", farea)):
            if over == 0:
                out[name] = real(0.0)
        if t_min == real(100000.0) and not bool(np.any(t > 0)):
            out["iceThicknessMin"] = real(0.0)
    return {k: np.array([out[k]], real) for k in GLOBAL_FIELDS}


def regional_stats(pools: dict, config: dict) -> dict:
    """``li_compute_regional_stats``: its fields' words at RKIND, one per
    region (``regions/regionCellMasks``: cells by regions)."""
    real = rkind.PRECISIONS[rkind.precision_of(pools["geometry"]["thickness"])]
    if "regionCellMasks" not in pools.get("regions", {}):
        _refuse("regions/regionCellMasks", "missing", "regionalStats sums over each region's cells",
                "a 'regions' pool with regionCellMasks (nCells x nRegions, 0 or 1)")
    masks = _host(pools["regions"]["regionCellMasks"])
    x = _inputs(pools, config, real)
    a, t, ice, gr, fl, scyr, rhoi = (x["areaCell"], x["thickness"], x["ice"], x["grounded"], x["floating"],
                                     x["scyr"], x["rhoi"])
    ratio = real(x["rhow"] / x["rhoi"])
    gl_terms, gl_edges = _gl_terms(x, real)
    cells_on_edge = x["cellsOnEdge"][gl_edges]
    n_regions = masks.shape[1]
    out = {k: np.zeros(n_regions, real) for k in REGIONAL_FIELDS}
    empty = config.get("woof_analysis_empty", "v842") == "corrected"
    for r in range(n_regions):
        mr = masks[:, r].astype(real)
        area = _sum((mr * a) * ice, real)
        vol = _sum(((mr * a) * t) * ice, real)
        vaf = _sum(((gr * mr) * a) * (t + ratio * x["below"]), real)
        garea = _sum((mr * a) * gr, real)
        gvol = _sum(((mr * a) * t) * gr, real)
        farea = _sum((mr * a) * fl, real)
        fvol = _sum(((mr * a) * t) * fl, real)
        smb = _sum((((mr * ice) * a) * x["sfcMassBal"]) * scyr, real)
        bmb = _sum((((mr * ice) * a) * x["basalMassBal"]) * scyr, real)
        fbmb = _sum((((mr * fl) * a) * x["floatingBasalMassBal"]) * scyr, real)
        gbmb = _sum((((mr * ice) * a) * x["groundedBasalMassBal"]) * scyr, real)
        calving = _sum((((mr * x["calvingThickness"]) * a) * rhoi) / real(x["deltat"] / scyr), real)
        thick = (t * mr) * ice
        t_max = _running(thick, t, real(0.0), True)
        t_min = _running(thick, t, real(10000.0), False, positive=True)
        s_max = _running((x["surfaceSpeed"] * mr) * ice, x["surfaceSpeed"], real(0.0), True)
        b_max = _running((x["basalSpeed"] * mr) * ice, x["basalSpeed"], real(0.0), True)
        inside = (masks[cells_on_edge[:, 0] - 1, r] + masks[cells_on_edge[:, 1] - 1, r]) > 0
        gl = _sum(gl_terms[inside], real)
        values = {
            "regionalIceArea": area, "regionalIceVolume": vol, "regionalGroundedIceArea": garea,
            "regionalGroundedIceVolume": gvol, "regionalFloatingIceArea": farea, "regionalFloatingIceVolume": fvol,
            "regionalSumSfcMassBal": smb, "regionalSumBasalMassBal": bmb, "regionalSumGroundedBasalMassBal": gbmb,
            "regionalSumFloatingBasalMassBal": fbmb, "regionalSumCalvingFlux": calving,
            "regionalVolumeAboveFloatation": vaf, "regionalSumGroundingLineFlux": gl,
            "regionalIceThicknessMean": _div(vol, area, real),
            "regionalAvgNetAccumulation": _div(_div(smb, area, real), rhoi, real),
            "regionalAvgGroundedBasalMelt": _div(_div(real(real(-1.0) * gbmb), garea, real), rhoi, real),
            "regionalAvgSubshelfMelt": _div(_div(real(real(-1.0) * fbmb), farea, real), rhoi, real),
            "regionalIceThicknessMin": t_min, "regionalIceThicknessMax": t_max,
            "regionalSurfaceSpeedMax": real(s_max * scyr), "regionalBasalSpeedMax": real(b_max * scyr)}
        if empty:
            for name, over in (("regionalIceThicknessMean", area), ("regionalAvgNetAccumulation", area),
                               ("regionalAvgGroundedBasalMelt", garea), ("regionalAvgSubshelfMelt", farea)):
                if over == 0:
                    values[name] = real(0.0)
            if not bool(np.any(thick > 0)):
                values["regionalIceThicknessMin"] = real(0.0)
        for k, value in values.items():
            out[k][r] = value
    return out


def _compute(member: str, pools: dict, config: dict, on_analysis) -> None:
    words = (global_stats if member == "globalStats" else regional_stats)(pools, config)
    pool = pools.setdefault(f"{member}AM", {})
    for name, value in words.items():
        current = pool.get(name)
        if current is not None and not isinstance(current, np.ndarray):
            import cupy as cp
            current[...] = cp.asarray(value).reshape(current.shape)
        else:
            pool[name] = value.copy()
    if on_analysis is not None:
        xtime = pools["mesh"]["xtime"]
        on_analysis(Compute(member, str(xtime), np.concatenate([words[k].ravel() for k in FIELDS[member]])))
