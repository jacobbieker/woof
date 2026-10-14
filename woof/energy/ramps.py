"""Ramps and wind-energy products: ``wind-resource`` and ``ramps``.

Two ``woof energy rating`` products and the vectorised NumPy (and xarray)
functions behind them.  Everything works on ``(time, site[, height])``
arrays at any time step: an hourly forecast, the 15-minute reanalysis
output, or 1-5 minute series that resolve the ramps themselves.  Times are
seconds (any spacing, strictly increasing); nothing assumes a fixed step.

What is pinned
--------------

Rotor geometry.  The rotor disc of diameter ``D`` centred at the hub height
is cut into ``n_segments`` (default 10) horizontal strips of equal height.
Each strip's wind is the profile at its centre height, interpolated linearly
in ln(height) between the bracketing forecast heights (nearest height
outside them), and its weight is its exact share of the disc area.  When the
forecast carries no ``rotor_diameter_m`` for a site, ``D`` is
``--rotor-diameter-m`` if given, else the hub height (hub height / rotor
diameter is about 0.8-1.2 for current turbines); the notes say how many
sites were assumed.

``rotor_equivalent_wind_speed`` (REWS)
    IEC 61400-12-1:2017 (Ed. 2) eq. 4: ``(sum_i (v_i cos phi_i)^3 A_i / A)
    ^ (1/3)`` over the strips, with ``phi_i`` the wind direction of strip
    ``i`` relative to the hub-height direction when ``u`` and ``v`` (or
    ``wind_from_direction``) are present, and ``cos phi = 1`` otherwise.
    ``cos phi`` is clipped at 0 (a strip blowing more than 90 degrees from
    the hub wind contributes nothing).

``shear_exponent``
    The power-law ``alpha`` in ``U(z) = U_ref (z / z_ref)^alpha``: the least
    squares slope of ln U against ln z over the forecast heights between the
    lower and upper rotor tip, or over the two forecast heights nearest the
    hub (in ln z) when fewer than two lie within the rotor.  Heights with
    ``U <= 0`` are left out.

``wind_veer``
    The wind direction at the upper tip minus that at the lower tip,
    wrapped to (-180, 180] degrees; positive is clockwise turning with
    height (veering), from ``u``/``v`` interpolated in ln(height).  NaN when
    either tip is calm (below 1e-6 m/s) or the forecast has no direction.

``turbulence_intensity``
    ``sigma_u / U`` at the hub.  From TKE when the data carry it
    (``sigma_u = sqrt(2/3 TKE)``, isotropic turbulence; MYNN/UW ``TKE_PBL``
    is the subgrid TKE), otherwise the resolved standard deviation of the
    hub-height wind over a trailing window (default 10 min, at least 3
    samples) divided by its mean.  ``forecast.v1`` carries no TKE, so the
    product uses the resolved estimate when the time step allows at least 3
    samples per window and is NaN (with a note) otherwise.  NaN where the
    mean wind is below 1 m/s.

``rews_power_mw``
    The normalised generic power curve (``--power-curve``, default the
    table's default) at the density-corrected REWS ``REWS (rho /
    1.225)^(1/3)`` (IEC 61400-12-1 for pitch-regulated turbines; ``rho`` at
    the hub), times ``capacity_mw``.  When a ``turbine`` site carries both
    ``capacity_mw`` and ``rotor_diameter_m``, the curve's speed axis is
    stretched between cut-in and rated so it reaches rated power at
    ``(2 P / (A rho Cp))^(1/3)`` with ``Cp = 0.45`` (the specific-power
    scaling the packaged curves' ``specific_power_w_m2`` follow); cut-in and
    cut-out stay.  No wakes, availability, curtailment or turbulence losses.

``capacity_factor``
    ``mean_t(rews_power_mw) / capacity_mw`` over the finite times.

Ramps.  The ramp signal is ``rews_power_mw / capacity_mw``, the power as a
fraction of capacity, which needs no capacity (``ramp_signal``).  Two
definitions run side by side:

``threshold``
    For each window ``W`` (default 15 min, 1 h, 4 h) with threshold ``X``
    (default 0.2 of capacity): ``ramp_delta(t) = P(t) - P(t - W)``, with
    ``P(t - W)`` linearly interpolated in time.  Every time with
    ``|delta| >= X`` marks the interval ``[t - W, t]`` (from the sample at
    or before ``t - W``); overlapping intervals of the same direction are
    merged, and each merged interval is trimmed to its largest rise (up) or
    fall (down): the end is the first sample of the largest change, the
    start the last sample of the extreme it leaves.  An event can therefore
    last longer than ``W`` when windows chain.  Windows shorter than the
    median time step are refused.
``swinging_door``
    The swinging-door algorithm (Bristol 1990; applied to wind power ramps
    by Florita, Hodge and Orwig 2013, NREL/CP-5500-58060) segments the
    signal into straight lines through retained samples, each within
    ``--sda-tolerance`` (default 0.05 of capacity) of the samples it
    covers.  Consecutive segments with the same sign of slope and a rate of
    at least 0.05 per hour are merged; a merged run is a ramp when its
    change is at least the threshold (``--ramp-threshold``).  Because a
    swinging door retains a sample only once the door closes, each end is
    then moved to the local extreme: forward/backward while the signal does
    not advance in the ramp direction (trims plateaus), then outward while
    it strictly does.  No maximum duration is imposed.

Per-timestep ``ramp_flag`` (per window) and ``ramp_flag_sda`` are +1 inside
an up event, -1 inside a down event, 0 elsewhere; where events of opposite
direction overlap, the larger one wins.  ``ramp_rate`` is the
sample-to-sample rate ``dP/dt`` per hour (NaN at the first time).  The event
table (``ramp_event`` dimension) is :data:`RAMP_EVENT_COLUMNS` and is read
back, validated, by :func:`woof.energy.contracts.load_ramp_events`.

Reusable helpers: :func:`density_corrected_speed`,
:func:`curve_power_fraction`, :func:`rated_speed_for_specific_power`,
:func:`rotor_segments` and :func:`rotor_equivalent_wind_speed` (NumPy),
:func:`detect_ramps` (NumPy), :func:`ramps_dataset` and
:func:`wind_resource_dataset` (xarray; through ``xarray.apply_ufunc`` with
``dask="parallelized"``, so dask-backed or lazily loaded Icechunk arrays are
computed chunk-wise where the computation is local in time; the ramp event
table and resolved TI need whole time series per site).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from woof.energy.contracts import RAMP_EVENT_COLUMNS, ContractError
#: IEC 61400-12-1 reference air density (kg m-3), shared with wind-power.
from woof.energy.products import IEC_REFERENCE_DENSITY  # noqa: E402

#: Rotor power coefficient at rated speed behind the specific-power scaling.
CURVE_POWER_COEFFICIENT = 0.45

DEFAULT_RAMP_WINDOWS = "15m,1h,4h"
DEFAULT_RAMP_THRESHOLD = 0.2
DEFAULT_SDA_TOLERANCE = 0.05
DEFAULT_SDA_FLAT_RATE_PER_H = 0.05
DEFAULT_TI_WINDOW_MIN = 10.0
DEFAULT_REWS_SEGMENTS = 10
RESOLVED_TI_MIN_SAMPLES = 3
#: Below this mean wind (m/s) turbulence intensity is NaN.
TI_MIN_SPEED = 1.0
#: Below this speed (m/s) a wind direction is undefined.
CALM_SPEED = 1.0e-6
#: Names a TKE variable may carry in an xarray dataset.
TKE_NAMES = ("turbulent_kinetic_energy", "tke", "TKE_PBL", "TKE")

#: Output variables of the two products: name -> (dims, units, long_name).
PRODUCT_VARIABLES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "rotor_equivalent_wind_speed": (("time", "site"), "m s-1",
                                    "rotor-equivalent wind speed"),
    "shear_exponent": (("time", "site"), "1",
                       "power-law wind shear exponent across the rotor"),
    "wind_veer": (("time", "site"), "degree",
                  "wind direction change from lower to upper rotor tip"),
    "turbulence_intensity": (("time", "site"), "1",
                             "hub-height turbulence intensity"),
    "rews_power_mw": (("time", "site"), "MW",
                      "generic-curve power at the density-corrected REWS"),
    "capacity_factor": (("site",), "1",
                        "mean rews_power_mw over capacity_mw"),
    "ramp_window": (("ramp_window",), "min", "threshold ramp window"),
    "ramp_threshold": (("ramp_window",), "1",
                       "threshold ramp magnitude, fraction of capacity"),
    "ramp_signal": (("time", "site"), "1",
                    "generic-curve power as a fraction of capacity"),
    "ramp_rate": (("time", "site"), "h-1",
                  "sample-to-sample rate of the ramp signal"),
    "ramp_delta": (("time", "site", "ramp_window"), "1",
                   "ramp signal change over the window ending at each time"),
    "ramp_flag": (("time", "site", "ramp_window"), "1",
                  "threshold ramp flag: +1 up, -1 down, 0 none"),
    "ramp_flag_sda": (("time", "site"), "1",
                      "swinging-door ramp flag: +1 up, -1 down, 0 none"),
}
WIND_RESOURCE_VARIABLES = ("rotor_equivalent_wind_speed", "shear_exponent",
                           "wind_veer", "turbulence_intensity",
                           "rews_power_mw", "capacity_factor")
RAMPS_VARIABLES = ("ramp_window", "ramp_threshold", "ramp_signal",
                   "ramp_rate", "ramp_delta", "ramp_flag", "ramp_flag_sda")


class RampsRefused(ContractError):
    """A ramp or wind-resource computation cannot honour its inputs."""


# --------------------------------------------------------------------------
# power curves and density


def density_corrected_speed(speed: Any, density: Any,
                            reference_density: float = IEC_REFERENCE_DENSITY
                            ) -> np.ndarray:
    """IEC 61400-12-1 density normalisation ``v (rho / rho_ref)^(1/3)``."""

    return np.asarray(speed, dtype=float) * (
        np.asarray(density, dtype=float) / reference_density) ** (1.0 / 3.0)


def specific_power(capacity_mw: Any, rotor_diameter_m: Any) -> np.ndarray:
    """Rated power over swept area, W m-2."""

    diameter = np.asarray(rotor_diameter_m, dtype=float)
    return np.asarray(capacity_mw, dtype=float) * 1e6 / (
        math.pi * diameter ** 2 / 4.0)


def rated_speed_for_specific_power(
        specific_power_w_m2: Any, *,
        power_coefficient: float = CURVE_POWER_COEFFICIENT,
        density: float = IEC_REFERENCE_DENSITY) -> np.ndarray:
    """Rated wind speed ``(2 P/A / (rho Cp))^(1/3)`` (m/s)."""

    return (2.0 * np.asarray(specific_power_w_m2, dtype=float)
            / (density * power_coefficient)) ** (1.0 / 3.0)


def curve_power_fraction(speed: Any, curve: Mapping[str, Any], *,
                         rated_ms: Any = None) -> np.ndarray:
    """Power as a fraction of rated from a normalised curve.

    ``curve`` is :func:`woof.energy.products.load_power_curve` output.
    ``speed`` should already be density-corrected.  Zero below cut-in and
    above cut-out, NaN where ``speed`` is NaN.  ``rated_ms`` (broadcastable
    to ``speed``; NaN keeps the curve's own) stretches the speed axis
    between cut-in and rated so the curve reaches 1 at ``rated_ms``, and
    holds 1 from there to cut-out.
    """

    v = np.asarray(speed, dtype=float)
    cut_in = float(curve["cut_in_ms"])
    cut_out = float(curve["cut_out_ms"])
    x = v
    if rated_ms is not None:
        base = float(curve["rated_ms"])
        rated = np.broadcast_to(np.asarray(rated_ms, dtype=float), v.shape)
        rated = np.where(np.isfinite(rated), rated, base)
        if np.any(rated <= cut_in) or np.any(rated >= cut_out):
            raise RampsRefused(
                f"rated speed must lie between cut-in {cut_in:g} and cut-out "
                f"{cut_out:g} m/s for curve {curve.get('key', '?')!r}")
        stretched = cut_in + (v - cut_in) * (base - cut_in) / (rated - cut_in)
        x = np.where(v < rated, stretched, base)
    fraction = np.interp(x, curve["speeds"], curve["fraction"])
    fraction = np.where((v < cut_in) | (v > cut_out), 0.0, fraction)
    return np.where(np.isfinite(v), fraction, np.nan)


# --------------------------------------------------------------------------
# rotor and profile


def _area_below(h: np.ndarray) -> np.ndarray:
    """Fraction of a unit disc below the chord at height ``h`` (in radii)."""

    h = np.clip(h, -1.0, 1.0)
    return (h * np.sqrt(1.0 - h * h) + np.arcsin(h) + math.pi / 2.0) / math.pi


def rotor_segments(hub_height_m: Any, rotor_diameter_m: Any,
                   n_segments: int = DEFAULT_REWS_SEGMENTS
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Centre heights ``(site, n)`` and area fractions ``(n,)`` of equal-height
    horizontal rotor strips; the fractions sum to 1."""

    if int(n_segments) < 1:
        raise RampsRefused("n_segments must be at least 1")
    hub = np.atleast_1d(np.asarray(hub_height_m, dtype=float))
    radius = np.atleast_1d(np.asarray(rotor_diameter_m, dtype=float)) / 2.0
    edges = np.linspace(-1.0, 1.0, int(n_segments) + 1)
    weights = np.diff(_area_below(edges))
    centres = (edges[:-1] + edges[1:]) / 2.0
    return hub[:, None] + radius[:, None] * centres[None, :], weights


def interp_profile(field: Any, heights: Any, targets: Any, *,
                   log: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a ``(time, site, height)`` field to per-site heights.

    ``targets`` is ``(site, k)``; returns ``(time, site, k)`` and the
    ``(site, k)`` mask of targets outside the forecast heights (those take
    the nearest height).  Linear in ln(height) with ``log=True``; non-positive
    targets are NaN under ``log``.
    """

    field = np.asarray(field, dtype=float)
    heights = np.asarray(heights, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n_time, n_site, n_height = field.shape
    outside = (targets < heights[0]) | (targets > heights[-1])
    bad = ~np.isfinite(targets) | ((targets <= 0.0) if log else False)
    clipped = np.clip(np.where(bad, heights[0], targets), heights[0],
                      heights[-1])
    if n_height == 1:
        out = np.broadcast_to(field[:, :, :1],
                              (n_time, n_site, targets.shape[1])).copy()
        out[:, bad] = np.nan
        return out, outside | (targets != heights[0])
    z = np.log(heights) if log else heights
    zt = np.log(clipped) if log else clipped
    hi = np.clip(np.searchsorted(z, zt), 1, n_height - 1)
    lo = hi - 1
    weight = (zt - z[lo]) / (z[hi] - z[lo])
    shape = (n_time,) + targets.shape
    below = np.take_along_axis(field, np.broadcast_to(lo[None], shape), 2)
    above = np.take_along_axis(field, np.broadcast_to(hi[None], shape), 2)
    out = below + (above - below) * weight[None]
    out[:, bad] = np.nan
    return out, outside


def _direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological from-direction, degrees; NaN when calm."""

    with np.errstate(invalid="ignore"):
        direction = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
        return np.where(np.hypot(u, v) < CALM_SPEED, np.nan, direction)


def rotor_equivalent_wind_speed(speed: Any, heights: Any, hub_height_m: Any,
                                rotor_diameter_m: Any, *, u: Any = None,
                                v: Any = None,
                                n_segments: int = DEFAULT_REWS_SEGMENTS
                                ) -> tuple[np.ndarray, np.ndarray]:
    """IEC 61400-12-1:2017 rotor-equivalent wind speed, ``(time, site)``.

    ``speed`` (and optional ``u``, ``v`` for the veer term) are
    ``(time, site, height)``.  Returns REWS and the per-site count of strip
    centres outside the forecast heights.
    """

    centres, weights = rotor_segments(hub_height_m, rotor_diameter_m,
                                      n_segments)
    segment, outside = interp_profile(speed, heights, centres)
    if u is not None and v is not None:
        hub = np.atleast_1d(np.asarray(hub_height_m, dtype=float))[:, None]
        us, _ = interp_profile(u, heights, centres)
        vs, _ = interp_profile(v, heights, centres)
        uh, _ = interp_profile(u, heights, hub)
        vh, _ = interp_profile(v, heights, hub)
        norm = np.hypot(us, vs) * np.hypot(uh, vh)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (us * uh + vs * vh) / norm
        cosine = np.where(norm > 0.0, np.clip(cosine, 0.0, 1.0), 1.0)
        segment = segment * cosine
    cubed = np.sum(np.maximum(segment, 0.0) ** 3 * weights, axis=-1)
    return np.cbrt(cubed), outside.sum(axis=1)


def shear_exponent(speed: Any, heights: Any, hub_height_m: Any,
                   rotor_diameter_m: Any = None) -> np.ndarray:
    """Least-squares power-law shear exponent, ``(time, site)``."""

    speed = np.asarray(speed, dtype=float)
    heights = np.asarray(heights, dtype=float)
    hub = np.atleast_1d(np.asarray(hub_height_m, dtype=float))
    lnz = np.log(heights)
    if rotor_diameter_m is None:
        within = np.ones((hub.size, heights.size), dtype=bool)
    else:
        radius = np.atleast_1d(np.asarray(rotor_diameter_m, dtype=float)) / 2
        within = ((heights[None, :] >= (hub - radius)[:, None])
                  & (heights[None, :] <= (hub + radius)[:, None]))
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.abs(lnz[None, :] - np.log(hub)[:, None])
    distance = np.where(np.isfinite(distance), distance, np.inf)
    rank = np.argsort(np.argsort(distance, axis=1, kind="stable"), axis=1)
    select = np.where((within.sum(axis=1) >= 2)[:, None], within, rank < 2)
    select &= np.isfinite(hub)[:, None]
    if rotor_diameter_m is not None:
        # an unknown (NaN) rotor has no shear, as for every rotor quantity
        select &= np.isfinite(radius)[:, None]
    valid = select[None] & np.isfinite(speed) & (speed > 0.0)
    n = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log(np.where(valid, speed, 1.0))
        xbar = np.sum(np.where(valid, lnz, 0.0), axis=-1) / n
        ybar = np.sum(np.where(valid, y, 0.0), axis=-1) / n
        dx = np.where(valid, lnz - xbar[..., None], 0.0)
        dy = np.where(valid, y - ybar[..., None], 0.0)
        sxx = np.sum(dx * dx, axis=-1)
        alpha = np.sum(dx * dy, axis=-1) / sxx
    return np.where((n >= 2) & (sxx > 0.0), alpha, np.nan)


def wind_veer(u: Any, v: Any, heights: Any, hub_height_m: Any,
              rotor_diameter_m: Any) -> np.ndarray:
    """Direction at the upper tip minus the lower tip, (-180, 180] degrees."""

    hub = np.atleast_1d(np.asarray(hub_height_m, dtype=float))
    radius = np.atleast_1d(np.asarray(rotor_diameter_m, dtype=float)) / 2.0
    tips = np.stack([hub - radius, hub + radius], axis=1)
    ut, _ = interp_profile(u, heights, tips)
    vt, _ = interp_profile(v, heights, tips)
    direction = _direction(ut, vt)
    veer = np.mod(direction[..., 1] - direction[..., 0] + 180.0, 360.0) \
        - 180.0
    return np.where(veer == -180.0, 180.0, veer)


def turbulence_intensity_from_tke(tke: Any, speed: Any) -> np.ndarray:
    """``sqrt(2/3 TKE) / U``; NaN below :data:`TI_MIN_SPEED`."""

    tke = np.asarray(tke, dtype=float)
    speed = np.asarray(speed, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ti = np.sqrt(2.0 / 3.0 * np.maximum(tke, 0.0)) / speed
    return np.where(speed >= TI_MIN_SPEED, ti, np.nan)


def resolved_turbulence_intensity(speed: Any, time_s: Any, window_s: float,
                                  min_samples: int = RESOLVED_TI_MIN_SAMPLES
                                  ) -> np.ndarray:
    """Std/mean of ``speed`` ``(time, site)`` over the trailing window
    ``(t - window_s, t]``; NaN with fewer than ``min_samples`` finite samples
    or a mean below :data:`TI_MIN_SPEED`."""

    speed = np.asarray(speed, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    finite = np.isfinite(speed)
    # Running sums of the anomaly from a per-site reference keep the
    # variance free of cancellation over long series.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.nan_to_num(np.nanmean(speed, axis=0))
    x = np.where(finite, speed - reference, 0.0)
    zero = np.zeros((1,) + speed.shape[1:])
    c0 = np.concatenate([zero, np.cumsum(finite, axis=0)])
    c1 = np.concatenate([zero, np.cumsum(x, axis=0)])
    c2 = np.concatenate([zero, np.cumsum(x * x, axis=0)])
    start = np.searchsorted(time_s, time_s - float(window_s), side="right")
    end = np.arange(time_s.size) + 1
    n = c0[end] - c0[start]
    s1 = c1[end] - c1[start]
    s2 = c2[end] - c2[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n + reference
        var = np.maximum(s2 - s1 * s1 / n, 0.0) / (n - 1)
        ti = np.sqrt(var) / mean
    return np.where((n >= min_samples) & (mean >= TI_MIN_SPEED), ti, np.nan)


def capacity_factor(power: Any, capacity: Any, axis: int = 0) -> np.ndarray:
    """Mean power over capacity along ``axis`` (time), ignoring NaN."""

    power = np.asarray(power, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(power, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(capacity > 0.0, mean / capacity, np.nan)


# --------------------------------------------------------------------------
# ramps


@dataclass(frozen=True)
class RampWindow:
    minutes: float
    threshold: float


_DURATION = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(s|m|min|h)?\s*$")


def parse_ramp_windows(text: str | Sequence[Any], *,
                       threshold: float = DEFAULT_RAMP_THRESHOLD
                       ) -> tuple[RampWindow, ...]:
    """``"15m,1h,4h"`` or ``"15m:0.1,1h:0.2"`` -> windows.

    A bare number is minutes; ``s``, ``m``/``min`` and ``h`` are units.  A
    ``:X`` suffix sets that window's threshold (fraction of capacity, or
    signal units), otherwise ``threshold``.
    """

    items = text.split(",") if isinstance(text, str) else list(text)
    windows: list[RampWindow] = []
    for item in items:
        if isinstance(item, RampWindow):
            windows.append(item)
            continue
        item = str(item).strip()
        if not item:
            continue
        span, _, limit = item.partition(":")
        match = _DURATION.match(span)
        if not match:
            raise RampsRefused(f"ramp window {item!r} is not a duration "
                               "(e.g. 15m, 1h, 90s)")
        value = float(match.group(1))
        minutes = value * {"s": 1 / 60, "h": 60.0}.get(match.group(2) or "m",
                                                       1.0)
        try:
            limit_value = float(limit) if limit.strip() else float(threshold)
        except ValueError as error:
            raise RampsRefused(f"ramp window {item!r}: threshold "
                               f"{limit!r} is not a number") from error
        if not (minutes > 0 and math.isfinite(minutes)
                and limit_value > 0 and math.isfinite(limit_value)):
            raise RampsRefused(f"ramp window {item!r} needs a positive "
                               "duration and threshold")
        windows.append(RampWindow(minutes, limit_value))
    if len({w.minutes for w in windows}) != len(windows):
        raise RampsRefused("ramp windows repeat a duration")
    return tuple(sorted(windows, key=lambda w: w.minutes))


@dataclass
class RampResult:
    """:func:`detect_ramps` output; arrays are ``(time, site[, window])``.

    ``events`` rows hold ``site`` (index), ``method``, ``window_min`` (NaN
    for swinging_door), ``direction``, ``start``/``end`` (time indices),
    ``magnitude``, ``duration_min`` and ``max_rate`` (per hour).
    """

    windows: tuple[RampWindow, ...]
    delta: np.ndarray
    flag: np.ndarray
    rate: np.ndarray
    sda_flag: np.ndarray
    sda_breakpoints: np.ndarray
    events: list[dict[str, Any]]


def _check_series(signal: Any, time_s: Any) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]
    time_s = np.asarray(time_s, dtype=float)
    if signal.ndim != 2 or signal.shape[0] != time_s.size:
        raise RampsRefused(f"ramp signal shape {signal.shape} is not "
                           f"(time={time_s.size}, site)")
    if time_s.size and (not np.all(np.isfinite(time_s))
                        or np.any(np.diff(time_s) <= 0)):
        raise RampsRefused("ramp times must be finite and strictly "
                           "increasing")
    return signal, time_s


def ramp_rate(signal: Any, time_s: Any) -> np.ndarray:
    """Sample-to-sample rate per hour, ``(time, site)``; NaN at time 0."""

    signal, time_s = _check_series(signal, time_s)
    rate = np.full(signal.shape, np.nan)
    if time_s.size > 1:
        rate[1:] = np.diff(signal, axis=0) / np.diff(time_s)[:, None] * 3600.0
    return rate


def _start_index(time_s: np.ndarray, window_s: float) -> np.ndarray:
    """Index of the sample at or before ``t - W`` (-1 before the series)."""

    return np.searchsorted(time_s, time_s - window_s, side="right") - 1


def window_delta(signal: Any, time_s: Any, window_s: float) -> np.ndarray:
    """``P(t) - P(t - W)``, ``P(t - W)`` linear in time; NaN before start."""

    signal, time_s = _check_series(signal, time_s)
    n = time_s.size
    out = np.full(signal.shape, np.nan)
    if n < 2:
        return out
    target = time_s - float(window_s)
    lo = np.clip(_start_index(time_s, window_s), 0, n - 2)
    hi = lo + 1
    weight = (target - time_s[lo]) / (time_s[hi] - time_s[lo])
    past = signal[lo] + (signal[hi] - signal[lo]) * weight[:, None]
    valid = target >= time_s[0]
    out[valid] = signal[valid] - past[valid]
    return out


def _max_rate(rate: np.ndarray, site: int, start: int, end: int,
              direction: int) -> float:
    values = direction * rate[start + 1:end + 1, site]
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


def _event(site: int, method: str, window_min: float, direction: int,
           start: int, end: int, signal: np.ndarray, time_s: np.ndarray,
           rate: np.ndarray) -> dict[str, Any]:
    return {
        "site": int(site), "method": method, "window_min": float(window_min),
        "direction": int(direction), "start": int(start), "end": int(end),
        "magnitude": float(abs(signal[end, site] - signal[start, site])),
        "duration_min": float((time_s[end] - time_s[start]) / 60.0),
        "max_rate": max(_max_rate(rate, site, start, end, direction), 0.0),
    }


def _paint(flag: np.ndarray, events: list[dict[str, Any]]) -> None:
    for event in sorted(events, key=lambda e: e["magnitude"]):
        flag[event["start"]:event["end"] + 1, event["site"]] = \
            event["direction"]


def threshold_ramps(signal: Any, time_s: Any, window: RampWindow, *,
                    rate: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Threshold-over-window ramps: ``(delta, flag, events)``."""

    signal, time_s = _check_series(signal, time_s)
    window_s = window.minutes * 60.0
    rate = ramp_rate(signal, time_s) if rate is None else rate
    delta = window_delta(signal, time_s, window_s)
    start_index = np.maximum(_start_index(time_s, window_s), 0)
    events: list[dict[str, Any]] = []
    with np.errstate(invalid="ignore"):
        exceed = {1: delta >= window.threshold,
                  -1: delta <= -window.threshold}
    for direction, mask in exceed.items():
        for site in np.flatnonzero(mask.any(axis=0)):
            ends = np.flatnonzero(mask[:, site])
            starts = start_index[ends]
            group = np.concatenate([[True], starts[1:] > ends[:-1]])
            first = np.flatnonzero(group)
            last = np.concatenate([first[1:] - 1, [ends.size - 1]])
            for a, b in zip(starts[first], ends[last]):
                seg = direction * signal[a:b + 1, site]
                finite = np.isfinite(seg)
                running = np.minimum.accumulate(np.where(finite, seg, np.inf))
                rise = np.where(finite, seg - running, -np.inf)
                end_rel = int(np.argmax(rise))
                low = running[end_rel]
                start_rel = int(np.flatnonzero(
                    np.where(finite, seg, np.inf)[:end_rel + 1] == low)[-1])
                if end_rel > start_rel:
                    events.append(_event(site, "threshold", window.minutes,
                                         direction, a + start_rel,
                                         a + end_rel, signal, time_s, rate))
    flag = np.zeros(signal.shape, dtype=np.int8)
    _paint(flag, events)
    return delta, flag, events


def swinging_door(signal: Any, time_s: Any, tolerance: float) -> np.ndarray:
    """Swinging-door segmentation: boolean ``(time, site)`` of the retained
    samples (segment ends).  NaN samples split segments."""

    signal, time_s = _check_series(signal, time_s)
    if not tolerance > 0:
        raise RampsRefused("the swinging-door tolerance must be positive")
    n_time, n_site = signal.shape
    keep = np.zeros(signal.shape, dtype=bool)
    finite = np.isfinite(signal)
    anchor = np.full(n_site, -1)
    upper = np.full(n_site, np.inf)
    lower = np.full(n_site, -np.inf)
    for t in range(n_time):
        ok = finite[t]
        active = anchor >= 0
        gap = active & ~ok
        if gap.any():
            keep[t - 1, gap] = True
            anchor[gap] = -1
        new = ok & ~active
        if new.any():
            keep[t, new] = True
            anchor[new] = t
            upper[new] = np.inf
            lower[new] = -np.inf
        cols = np.flatnonzero(ok & active)
        if not cols.size:
            continue
        a = anchor[cols]
        dt = time_s[t] - time_s[a]
        pa = signal[a, cols]
        p = signal[t, cols]
        up = np.minimum(upper[cols], (p + tolerance - pa) / dt)
        lo = np.maximum(lower[cols], (p - tolerance - pa) / dt)
        closed = lo > up
        open_cols = cols[~closed]
        upper[open_cols] = up[~closed]
        lower[open_cols] = lo[~closed]
        if closed.any():
            shut = cols[closed]
            keep[t - 1, shut] = True
            anchor[shut] = t - 1
            step = time_s[t] - time_s[t - 1]
            upper[shut] = (signal[t, shut] + tolerance
                           - signal[t - 1, shut]) / step
            lower[shut] = (signal[t, shut] - tolerance
                           - signal[t - 1, shut]) / step
    if n_time:
        keep[n_time - 1, anchor >= 0] = True
    return keep


def _refine(series: np.ndarray, start: int, end: int, direction: int
            ) -> tuple[int, int]:
    """Move swinging-door ends to the local extremes (see module doc)."""

    p = direction * series
    n = p.size
    while start < end and p[start + 1] <= p[start]:
        start += 1
    while end > start and p[end - 1] >= p[end]:
        end -= 1
    while start > 0 and np.isfinite(p[start - 1]) and p[start - 1] < p[start]:
        start -= 1
    while end < n - 1 and np.isfinite(p[end + 1]) and p[end + 1] > p[end]:
        end += 1
    return start, end


def swinging_door_ramps(signal: Any, time_s: Any, *, tolerance: float,
                        threshold: float,
                        flat_rate: float = DEFAULT_SDA_FLAT_RATE_PER_H,
                        rate: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray,
                                   list[dict[str, Any]]]:
    """Swinging-door ramps: ``(breakpoints, flag, events)``."""

    signal, time_s = _check_series(signal, time_s)
    if not threshold > 0:
        raise RampsRefused("the swinging-door ramp threshold must be "
                           "positive")
    rate = ramp_rate(signal, time_s) if rate is None else rate
    keep = swinging_door(signal, time_s, tolerance)
    events: list[dict[str, Any]] = []
    for site in range(signal.shape[1]):
        points = np.flatnonzero(keep[:, site])
        if points.size < 2:
            continue
        series = signal[:, site]
        nan_count = np.concatenate([[0], np.cumsum(~np.isfinite(series))])
        a, b = points[:-1], points[1:]
        valid = nan_count[b + 1] - nan_count[a] == 0
        slope = (series[b] - series[a]) / (time_s[b] - time_s[a]) * 3600.0
        direction = np.where(valid & (np.abs(slope) >= flat_rate),
                             np.sign(slope), 0).astype(int)
        runs: list[tuple[int, int, int]] = []
        k = 0
        while k < direction.size:
            if direction[k] == 0:
                k += 1
                continue
            j = k
            while j + 1 < direction.size and direction[j + 1] == direction[k]:
                j += 1
            sign = int(direction[k])
            start, end = _refine(series, int(a[k]), int(b[j]), sign)
            # Outward refinement can carry two runs split by a slow but
            # monotone stretch onto the same samples: one ramp, not two.
            if runs and runs[-1][0] == sign and start <= runs[-1][2]:
                _, first, last = runs.pop()
                start, end = min(start, first), max(end, last)
            runs.append((sign, start, end))
            k = j + 1
        for sign, start, end in runs:
            if end > start and sign * (series[end] - series[start]) \
                    >= threshold:
                events.append(_event(site, "swinging_door", math.nan, sign,
                                     start, end, signal, time_s, rate))
    flag = np.zeros(signal.shape, dtype=np.int8)
    _paint(flag, events)
    return keep, flag, events


def detect_ramps(signal: Any, time_s: Any, *,
                 windows: str | Sequence[Any] = DEFAULT_RAMP_WINDOWS,
                 threshold: float = DEFAULT_RAMP_THRESHOLD,
                 sda_tolerance: float = DEFAULT_SDA_TOLERANCE,
                 sda_threshold: float | None = None,
                 sda_flat_rate: float = DEFAULT_SDA_FLAT_RATE_PER_H
                 ) -> RampResult:
    """Both ramp definitions on ``signal`` ``(time, site)`` at ``time_s``.

    Thresholds are in signal units: pass a fraction of capacity for the
    defaults to mean what they say.  Windows shorter than the median time
    step are refused.
    """

    signal, time_s = _check_series(signal, time_s)
    parsed = parse_ramp_windows(windows, threshold=threshold)
    if time_s.size > 1:
        step = float(np.median(np.diff(time_s)))
        short = [w.minutes for w in parsed if w.minutes * 60.0 < step]
        if short:
            raise RampsRefused(
                f"ramp window(s) {', '.join(f'{m:g} min' for m in short)} "
                f"are shorter than the median time step {step / 60:g} min")
    rate = ramp_rate(signal, time_s)
    n_time, n_site = signal.shape
    delta = np.full((n_time, n_site, len(parsed)), np.nan)
    flag = np.zeros((n_time, n_site, len(parsed)), dtype=np.int8)
    events: list[dict[str, Any]] = []
    for i, window in enumerate(parsed):
        delta[..., i], flag[..., i], found = threshold_ramps(
            signal, time_s, window, rate=rate)
        events += found
    keep, sda_flag, found = swinging_door_ramps(
        signal, time_s, tolerance=sda_tolerance,
        threshold=threshold if sda_threshold is None else sda_threshold,
        flat_rate=sda_flat_rate, rate=rate)
    events += found
    events.sort(key=lambda e: (e["site"], e["start"], e["method"],
                               e["window_min"] if e["method"] == "threshold"
                               else 0.0))
    return RampResult(windows=parsed, delta=delta, flag=flag, rate=rate,
                      sda_flag=sda_flag, sda_breakpoints=keep, events=events)


def ramp_event_columns(events: Sequence[Mapping[str, Any]],
                       site_ids: Sequence[Any], times: Any
                       ) -> dict[str, np.ndarray]:
    """Event rows -> :data:`RAMP_EVENT_COLUMNS` arrays; ``times`` indexed by
    the event start/end indices."""

    times = np.asarray(times)
    start = np.array([e["start"] for e in events], dtype=int)
    end = np.array([e["end"] for e in events], dtype=int)
    return {
        "ramp_event_site_id": np.array(
            [str(site_ids[e["site"]]) for e in events], dtype=object),
        "ramp_event_method": np.array([e["method"] for e in events],
                                      dtype=object),
        "ramp_event_window": np.array([e["window_min"] for e in events],
                                      dtype=float),
        "ramp_event_direction": np.array([e["direction"] for e in events],
                                         dtype=np.int8),
        "ramp_event_start": times[start] if start.size else times[:0],
        "ramp_event_end": times[end] if end.size else times[:0],
        "ramp_event_magnitude": np.array([e["magnitude"] for e in events],
                                         dtype=float),
        "ramp_event_duration": np.array([e["duration_min"] for e in events],
                                        dtype=float),
        "ramp_event_max_rate": np.array([e["max_rate"] for e in events],
                                        dtype=float),
    }


def ramp_event_summary(methods: Sequence[Any], windows: Sequence[float]
                       ) -> dict[str, int]:
    """Event counts keyed ``threshold 60 min`` / ``swinging_door``."""

    keys = [str(m) if not (isinstance(w, float) and math.isfinite(w))
            else f"{m} {w:g} min" for m, w in zip(methods, windows)]
    return dict(sorted(Counter(keys).items()))


# --------------------------------------------------------------------------
# xarray


def _seconds(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "M":
        return (values - values[0]) / np.timedelta64(1, "s")
    return np.asarray(values, dtype=float)


def ramps_dataset(signal: Any, *, capacity: Any = None,
                  windows: str | Sequence[Any] = DEFAULT_RAMP_WINDOWS,
                  threshold: float = DEFAULT_RAMP_THRESHOLD,
                  sda_tolerance: float = DEFAULT_SDA_TOLERANCE,
                  sda_threshold: float | None = None,
                  sda_flat_rate: float = DEFAULT_SDA_FLAT_RATE_PER_H,
                  time_dim: str = "time", site_dim: str = "site"):
    """Ramps of an xarray DataArray ``(time[, site])`` as a Dataset.

    ``capacity`` (scalar or DataArray over ``site``) divides the signal
    first, so thresholds are fractions of capacity; without it they are in
    the signal's units (e.g. m/s for a hub-height wind series).  The whole
    time series of each site is loaded; the result carries ``ramp_delta``,
    ``ramp_flag``, ``ramp_rate``, ``ramp_flag_sda``, ``ramp_signal`` and
    the :data:`RAMP_EVENT_COLUMNS` table with datetime start/end.
    """

    import xarray as xr

    da = signal
    if site_dim not in da.dims:
        da = da.expand_dims({site_dim: [str(da.name or "series")]})
    if capacity is not None:
        da = da / capacity
    da = da.transpose(time_dim, site_dim)
    times = da[time_dim].values
    result = detect_ramps(da.values, _seconds(times), windows=windows,
                          threshold=threshold, sda_tolerance=sda_tolerance,
                          sda_threshold=sda_threshold,
                          sda_flat_rate=sda_flat_rate)
    coords = {time_dim: da[time_dim], site_dim: da[site_dim],
              "ramp_window": [w.minutes for w in result.windows]}
    site_ids = [str(s) for s in da[site_dim].values]
    columns = ramp_event_columns(result.events, site_ids, times)
    out = xr.Dataset(
        {
            "ramp_signal": ((time_dim, site_dim), da.values),
            "ramp_rate": ((time_dim, site_dim), result.rate),
            "ramp_delta": ((time_dim, site_dim, "ramp_window"),
                           result.delta),
            "ramp_flag": ((time_dim, site_dim, "ramp_window"), result.flag),
            "ramp_flag_sda": ((time_dim, site_dim), result.sda_flag),
            "ramp_threshold": (("ramp_window",),
                               [w.threshold for w in result.windows]),
            **{name: (("ramp_event",), values)
               for name, values in columns.items()},
        },
        coords=coords)
    for name, (_, units, long_name) in PRODUCT_VARIABLES.items():
        if name in out.variables and name != "ramp_window":
            out[name].attrs.update(units=units, long_name=long_name)
    for name, (units, meaning) in RAMP_EVENT_COLUMNS.items():
        if units not in ("time", "-"):
            out[name].attrs["units"] = units
        out[name].attrs["long_name"] = meaning
    out.attrs.update(sda_tolerance=float(sda_tolerance),
                     sda_flat_rate_per_h=float(sda_flat_rate))
    return out


def _find_tke(ds: Any) -> str | None:
    for name in TKE_NAMES:
        if name in ds.variables:
            return name
    return None


def wind_resource_dataset(ds: Any, *, hub_height_m: Any,
                          rotor_diameter_m: Any, curve: Mapping[str, Any]
                          | None = None, capacity_mw: Any = None,
                          ti_window_min: float = DEFAULT_TI_WINDOW_MIN,
                          n_segments: int = DEFAULT_REWS_SEGMENTS,
                          time_dim: str = "time", site_dim: str = "site",
                          height_dim: str = "height"):
    """Shear, veer, TI, REWS and generic-curve power from an xarray Dataset.

    ``ds`` carries ``wind_speed`` ``(time, site, height)`` with a height
    coordinate (m above ground), and optionally ``u``/``v`` (veer and the
    REWS veer term), ``air_density`` (density correction, else 1.225) and a
    TKE variable named one of :data:`TKE_NAMES` (TI from TKE, else the
    resolved trailing-window estimate).  ``hub_height_m`` and
    ``rotor_diameter_m`` are scalars or per-site arrays.  Time-local
    quantities go through ``xarray.apply_ufunc(dask="parallelized")``.
    """

    import xarray as xr

    heights = np.asarray(ds[height_dim].values, dtype=float)
    n_site = ds.sizes[site_dim]
    hub = xr.DataArray(np.broadcast_to(np.asarray(hub_height_m, float),
                                       (n_site,)).copy(), dims=(site_dim,))
    diameter = xr.DataArray(np.broadcast_to(
        np.asarray(rotor_diameter_m, float), (n_site,)).copy(),
        dims=(site_dim,))
    has_uv = "u" in ds.variables and "v" in ds.variables

    def pointwise(func, *arrays, n_out=1):
        """Run ``func`` with every (time, site) pair as its own site."""

        def inner(speed, *rest):
            lead = speed.shape[:-1]
            flat = speed.reshape((1, -1, speed.shape[-1]))
            pieces = [np.broadcast_to(r, lead + r.shape[len(lead):])
                      .reshape((1, -1) + r.shape[len(lead):])
                      if r.ndim > len(lead) else
                      np.broadcast_to(r, lead).reshape(-1) for r in rest]
            out = func(flat, *pieces)
            return out.reshape(lead)
        return inner

    core = [[height_dim]]

    def apply(func, arrays, cores):
        return xr.apply_ufunc(func, *arrays, input_core_dims=cores,
                              dask="parallelized", output_dtypes=[float])

    def rews_func(speed, hub_, d_, *uv):
        u_, v_ = (uv[0], uv[1]) if uv else (None, None)
        return rotor_equivalent_wind_speed(speed, heights, hub_, d_, u=u_,
                                           v=v_, n_segments=n_segments)[0]

    uv_args = [ds["u"], ds["v"]] if has_uv else []
    rews = apply(pointwise(rews_func), [ds["wind_speed"], hub, diameter,
                                        *uv_args],
                 core + [[], []] + core * len(uv_args))
    shear = apply(pointwise(lambda s, h, d: shear_exponent(s, heights, h, d)),
                  [ds["wind_speed"], hub, diameter], core + [[], []])

    def hub_value(field, h):
        return interp_profile(field, heights, h[:, None])[0][..., 0]

    hub_speed = apply(pointwise(hub_value), [ds["wind_speed"], hub],
                      core + [[]])
    out = {"rotor_equivalent_wind_speed": rews, "shear_exponent": shear}
    if has_uv:
        out["wind_veer"] = apply(
            pointwise(lambda u_, v_, h, d: wind_veer(u_, v_, heights, h, d)),
            [ds["u"], ds["v"], hub, diameter], core + core + [[], []])
    tke_name = _find_tke(ds)
    if tke_name is not None:
        tke_hub = apply(pointwise(hub_value), [ds[tke_name], hub],
                        core + [[]])
        out["turbulence_intensity"] = xr.apply_ufunc(
            turbulence_intensity_from_tke, tke_hub, hub_speed,
            dask="parallelized", output_dtypes=[float])
        ti_method = f"sqrt(2/3 {tke_name}) / hub wind"
    else:
        time_s = _seconds(ds[time_dim].values)
        window_s = float(ti_window_min) * 60.0

        def resolved(speed):
            flat = speed.reshape((-1, speed.shape[-1])).T
            ti = resolved_turbulence_intensity(flat, time_s, window_s)
            return ti.T.reshape(speed.shape)

        out["turbulence_intensity"] = xr.apply_ufunc(
            resolved, hub_speed, input_core_dims=[[time_dim]],
            output_core_dims=[[time_dim]], dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True}
        ).transpose(*hub_speed.dims)
        ti_method = (f"resolved std/mean of hub wind over a trailing "
                     f"{ti_window_min:g} min window")
    if curve is not None:
        density = ds["air_density"] if "air_density" in ds.variables else None
        if density is not None:
            # linear in height, as the products path interpolates density
            def hub_linear(field, h):
                return interp_profile(field, heights, h[:, None],
                                      log=False)[0][..., 0]

            rho = apply(pointwise(hub_linear), [density, hub], core + [[]])
        else:
            rho = IEC_REFERENCE_DENSITY
        equivalent = rews * (rho / IEC_REFERENCE_DENSITY) ** (1.0 / 3.0)
        fraction = xr.apply_ufunc(lambda s: curve_power_fraction(s, curve),
                                  equivalent, dask="parallelized",
                                  output_dtypes=[float])
        if capacity_mw is not None:
            capacity = xr.DataArray(np.broadcast_to(
                np.asarray(capacity_mw, float), (n_site,)).copy(),
                dims=(site_dim,))
            out["rews_power_mw"] = fraction * capacity
            out["capacity_factor"] = (out["rews_power_mw"].mean(
                time_dim, skipna=True) / capacity.where(capacity > 0))
        out["ramp_signal"] = fraction
    result = xr.Dataset(out)
    for name, var in result.data_vars.items():
        _, units, long_name = PRODUCT_VARIABLES[name]
        var.attrs.update(units=units, long_name=long_name)
    result["turbulence_intensity"].attrs["method"] = ti_method
    return result


# --------------------------------------------------------------------------
# woof energy rating products


def _turbine_sites(forecast, notes, product: str,
                   rotor_diameter_m: float | None):
    from woof.energy.products import _coord_or_nan, _wind_mask

    kind = forecast.coord("kind").astype(str)
    hub = _coord_or_nan(forecast, "hub_height_m")
    mask = _wind_mask(kind, hub)
    no_hub = mask & ~np.isfinite(hub)
    if no_hub.any():
        notes.add(f"{product}: {int(no_hub.sum())} turbine site(s) have no "
                  "hub_height_m; their values are NaN")
    idx = np.flatnonzero(mask & np.isfinite(hub))
    diameter = _coord_or_nan(forecast, "rotor_diameter_m")[idx]
    given = np.isfinite(diameter) & (diameter > 0)
    if (~given).any():
        fill = rotor_diameter_m if rotor_diameter_m else None
        diameter = np.where(given, diameter,
                            fill if fill is not None else hub[idx])
        source = (f"--rotor-diameter-m {fill:g} m" if fill is not None
                  else "the hub height")
        notes.add(f"{product}: {int((~given).sum())} site(s) have no "
                  f"rotor_diameter_m; rotor diameter assumed equal to "
                  f"{source}")
    low = hub[idx] - diameter / 2.0 <= 0.0
    if low.any():
        diameter = np.where(low, np.nan, diameter)
        notes.add(f"{product}: {int(low.sum())} site(s) have a rotor reaching "
                  "the ground (diameter >= 2 x hub height); their rotor "
                  "quantities are NaN")
    return idx, hub[idx], diameter, given


def _uv(forecast, idx):
    v = forecast.variables
    if "u" in v and "v" in v:
        return v["u"][:, idx], v["v"][:, idx], "u/v"
    if "wind_from_direction" in v:
        speed = v["wind_speed"][:, idx]
        rad = np.radians(v["wind_from_direction"][:, idx])
        return -speed * np.sin(rad), -speed * np.cos(rad), \
            "wind_from_direction"
    return None, None, None


def _hub_density(forecast, idx, hub, notes, product):
    from woof.energy.products import _R_DRY

    v = forecast.variables
    target = hub[:, None]
    if "air_density" in v:
        return interp_profile(v["air_density"][:, idx], forecast.heights,
                              target, log=False)[0][..., 0]
    p = interp_profile(v["air_pressure"][:, idx], forecast.heights, target,
                       log=False)[0][..., 0]
    t = interp_profile(v["air_temperature"][:, idx], forecast.heights,
                       target, log=False)[0][..., 0]
    notes.add(f"{product}: no air_density in the forecast; density from "
              "air_pressure / (R_d air_temperature), dry air")
    return p / (_R_DRY * t)


@dataclass
class _Rotor:
    idx: np.ndarray
    hub: np.ndarray
    diameter: np.ndarray
    given: np.ndarray
    rews: np.ndarray
    fraction: np.ndarray
    u: np.ndarray | None
    v: np.ndarray | None


def rotor_power(forecast, curve, notes, product, *,
                rotor_diameter_m: float | None = None,
                n_segments: int = DEFAULT_REWS_SEGMENTS) -> _Rotor:
    """REWS and generic-curve power fraction at the turbine sites of a
    :class:`products.Forecast`, shared by ``wind-resource`` and ``ramps``;
    ``product`` labels the notes."""

    from woof.energy.products import _coord_or_nan

    idx, hub, diameter, given = _turbine_sites(forecast, notes, product,
                                               rotor_diameter_m)
    speed = forecast.variables["wind_speed"][:, idx]
    u, v, source = _uv(forecast, idx)
    rews, outside = rotor_equivalent_wind_speed(
        speed, forecast.heights, hub, diameter, u=u, v=v,
        n_segments=n_segments)
    if (outside > 0).any():
        notes.add(f"{product}: the rotor extends beyond the forecast heights "
                  f"at {int((outside > 0).sum())} site(s); the nearest "
                  "height was used there")
    if source is None:
        notes.add(f"{product}: no u/v or wind_from_direction; REWS has no "
                  "veer term" + (" and wind_veer is NaN"
                                 if "wind-resource" in product else ""))
    rho = _hub_density(forecast, idx, hub, notes, product)
    equivalent = density_corrected_speed(rews, rho)
    kind = forecast.coord("kind").astype(str)[idx]
    capacity = _coord_or_nan(forecast, "capacity_mw")[idx]
    scaled = given & (kind == "turbine") & np.isfinite(capacity)
    rated = np.where(scaled, rated_speed_for_specific_power(
        specific_power(np.where(scaled, capacity, 1.0),
                       np.where(scaled, diameter, 1.0))), np.nan)
    rated = np.where(scaled & (rated > float(curve["cut_in_ms"]))
                     & (rated < float(curve["cut_out_ms"])), rated, np.nan)
    if np.isfinite(rated).any():
        notes.add(f"{product}: rated speed scaled by specific power "
                  f"(Cp {CURVE_POWER_COEFFICIENT:g}) for "
                  f"{int(np.isfinite(rated).sum())} turbine site(s) with "
                  "capacity_mw and rotor_diameter_m")
    fraction = curve_power_fraction(equivalent, curve,
                                    rated_ms=rated[None, :])
    return _Rotor(idx=idx, hub=hub, diameter=diameter, given=given,
                  rews=rews, fraction=fraction, u=u, v=v)


def _output(name: str, data: np.ndarray, **attrs: Any):
    from woof.energy.products import ProductOutput

    dims, units, long_name = PRODUCT_VARIABLES[name]
    merged = {"units": units, "long_name": long_name}
    merged.update(attrs)
    return ProductOutput(name, dims, data, merged)


def _curve_note(curve: Mapping[str, Any]) -> str:
    return (f"{curve['key']} (cut-in {curve['cut_in_ms']:g}, rated "
            f"{curve['rated_ms']:g}, cut-out {curve['cut_out_ms']:g} m/s)")


def compute_wind_resource(forecast, curve: Mapping[str, Any], notes, *,
                          rotor_diameter_m: float | None = None,
                          ti_window_min: float = DEFAULT_TI_WINDOW_MIN,
                          n_segments: int = DEFAULT_REWS_SEGMENTS,
                          rotor: _Rotor | None = None) -> list:
    """The ``wind-resource`` product on a :class:`products.Forecast`.

    ``rotor`` is a :func:`rotor_power` result to reuse (it is computed when
    not given).
    """

    from woof.energy.products import _coord_or_nan, _full

    product = "wind-resource"
    if rotor is None:
        rotor = rotor_power(forecast, curve, notes, product,
                            rotor_diameter_m=rotor_diameter_m,
                            n_segments=n_segments)
    idx = rotor.idx
    speed = forecast.variables["wind_speed"][:, idx]
    fields = {name: _full(forecast) for name in
              ("rotor_equivalent_wind_speed", "shear_exponent", "wind_veer",
               "turbulence_intensity", "rews_power_mw")}
    capacity = _coord_or_nan(forecast, "capacity_mw")
    if idx.size:
        fields["rotor_equivalent_wind_speed"][:, idx] = rotor.rews
        fields["shear_exponent"][:, idx] = shear_exponent(
            speed, forecast.heights, rotor.hub, rotor.diameter)
        if rotor.u is not None:
            fields["wind_veer"][:, idx] = wind_veer(
                rotor.u, rotor.v, forecast.heights, rotor.hub,
                rotor.diameter)
        hub_speed = interp_profile(speed, forecast.heights,
                                   rotor.hub[:, None])[0][..., 0]
        step = float(np.median(np.diff(forecast.time_seconds))) \
            if forecast.time_seconds.size > 1 else math.inf
        window_s = ti_window_min * 60.0
        if step * (RESOLVED_TI_MIN_SAMPLES - 1) < window_s:
            fields["turbulence_intensity"][:, idx] = \
                resolved_turbulence_intensity(hub_speed,
                                              forecast.time_seconds, window_s)
        else:
            notes.add(f"{product}: turbulence_intensity is NaN: forecast.v1 "
                      "has no TKE and the time step "
                      f"({step / 60:g} min) gives fewer than "
                      f"{RESOLVED_TI_MIN_SAMPLES} samples per "
                      f"{ti_window_min:g} min window")
        cap = capacity[idx]
        if (~np.isfinite(cap)).any():
            notes.add(f"{product}: {int((~np.isfinite(cap)).sum())} site(s) "
                      "have no capacity_mw; rews_power_mw and "
                      "capacity_factor are NaN there")
        fields["rews_power_mw"][:, idx] = rotor.fraction * cap[None, :]
    cf = np.full(forecast.n_site, np.nan)
    cf[idx] = capacity_factor(fields["rews_power_mw"][:, idx], capacity[idx])
    rotor_note = (f"rotor split into {n_segments} equal-height strips; "
                  "profile linear in ln(height); rotor_diameter_m from the "
                  "forecast, else assumed (see notes)")
    return [
        _output("rotor_equivalent_wind_speed",
                fields["rotor_equivalent_wind_speed"],
                method="IEC 61400-12-1:2017 eq. 4, (sum (v_i cos phi_i)^3 "
                       "A_i/A)^(1/3)", assumptions=rotor_note),
        _output("shear_exponent", fields["shear_exponent"],
                method="least-squares slope of ln U on ln z over forecast "
                       "heights within the rotor (else the two nearest the "
                       "hub)"),
        _output("wind_veer", fields["wind_veer"],
                method="direction(upper tip) - direction(lower tip), "
                       "wrapped to (-180, 180]; positive veering"),
        _output("turbulence_intensity", fields["turbulence_intensity"],
                method=f"resolved std/mean of hub wind over a trailing "
                       f"{ti_window_min:g} min window (>= "
                       f"{RESOLVED_TI_MIN_SAMPLES} samples); NaN below "
                       f"{TI_MIN_SPEED:g} m/s",
                assumptions="forecast.v1 carries no TKE, so subgrid "
                            "turbulence is not included"),
        _output("rews_power_mw", fields["rews_power_mw"],
                method=f"normalised power curve {_curve_note(curve)} at "
                       "REWS (rho/1.225)^(1/3), scaled by capacity_mw",
                power_curve_sha256=curve["sha256"],
                assumptions="no wake, availability, curtailment or "
                            "turbulence losses"),
        _output("capacity_factor", cf,
                method="mean over finite times of rews_power_mw / "
                       "capacity_mw"),
    ]


def compute_ramps(forecast, curve: Mapping[str, Any], notes, *,
                  windows: str | Sequence[Any] = DEFAULT_RAMP_WINDOWS,
                  threshold: float = DEFAULT_RAMP_THRESHOLD,
                  sda_tolerance: float = DEFAULT_SDA_TOLERANCE,
                  sda_flat_rate: float = DEFAULT_SDA_FLAT_RATE_PER_H,
                  rotor_diameter_m: float | None = None,
                  n_segments: int = DEFAULT_REWS_SEGMENTS,
                  rotor: _Rotor | None = None) -> list:
    """The ``ramps`` product on a :class:`products.Forecast`.

    ``rotor`` is a :func:`rotor_power` result to reuse.
    """

    from woof.energy.products import _full

    product = "ramps"
    parsed = parse_ramp_windows(windows, threshold=threshold)
    time_s = forecast.time_seconds
    step = float(np.median(np.diff(time_s))) if time_s.size > 1 else math.inf
    kept = tuple(w for w in parsed if w.minutes * 60.0 >= step)
    dropped = [w.minutes for w in parsed if w not in kept]
    if dropped:
        notes.add(f"{product}: window(s) "
                  f"{', '.join(f'{m:g} min' for m in dropped)} skipped, "
                  f"shorter than the {step / 60:g} min time step")
    if not kept:
        listed = ", ".join(f"{w.minutes:g} min" for w in parsed)
        raise RampsRefused(
            f"every ramp window ({listed}) is shorter than the forecast "
            f"time step ({step / 60:g} min)")
    if rotor is None:
        rotor = rotor_power(forecast, curve, notes, product,
                            rotor_diameter_m=rotor_diameter_m,
                            n_segments=n_segments)
    idx = rotor.idx
    signal = _full(forecast)
    signal[:, idx] = rotor.fraction
    n_time, n_site = signal.shape
    delta = np.full((n_time, n_site, len(kept)), np.nan)
    flag = np.zeros((n_time, n_site, len(kept)), dtype=np.int8)
    rate = _full(forecast)
    sda_flag = np.zeros((n_time, n_site), dtype=np.int8)
    events: list[dict[str, Any]] = []
    if idx.size:
        result = detect_ramps(rotor.fraction, time_s, windows=kept,
                              threshold=threshold,
                              sda_tolerance=sda_tolerance,
                              sda_flat_rate=sda_flat_rate)
        delta[:, idx] = result.delta
        flag[:, idx] = result.flag
        rate[:, idx] = result.rate
        sda_flag[:, idx] = result.sda_flag
        events = [dict(e, site=int(idx[e["site"]])) for e in result.events]
    times = np.asarray(forecast.time_var["_values"], dtype=float)
    columns = ramp_event_columns(events, list(forecast.coord("site_id")),
                                 times)
    time_attrs = {key: forecast.time_var[key] for key in ("units", "calendar")
                  if key in forecast.time_var}
    signal_note = (f"rews_power_mw / capacity_mw: {_curve_note(curve)} at the "
                   "density-corrected REWS")
    outputs = [
        _output("ramp_window", np.array([w.minutes for w in kept])),
        _output("ramp_threshold", np.array([w.threshold for w in kept])),
        _output("ramp_signal", signal, method=signal_note,
                power_curve_sha256=curve["sha256"]),
        _output("ramp_rate", rate,
                method="(P(t) - P(t-1)) / dt, per hour"),
        _output("ramp_delta", delta,
                method="P(t) - P(t - W), P(t - W) linear in time"),
        _output("ramp_flag", flag,
                method="|ramp_delta| >= ramp_threshold marks [t - W, t]; "
                       "same-direction intervals merged and trimmed to the "
                       "largest change", flag_values="-1 0 1",
                flag_meanings="down none up"),
        _output("ramp_flag_sda", sda_flag,
                method=f"swinging door, tolerance {sda_tolerance:g}, "
                       "same-direction segments merged (rate >= "
                       f"{sda_flat_rate:g} h-1), change >= {threshold:g}",
                flag_values="-1 0 1", flag_meanings="down none up",
                sda_tolerance=float(sda_tolerance),
                sda_flat_rate_per_h=float(sda_flat_rate),
                sda_threshold=float(threshold)),
    ]
    from woof.energy.products import ProductOutput

    for name, (units, meaning) in RAMP_EVENT_COLUMNS.items():
        attrs: dict[str, Any] = {"long_name": meaning}
        if units == "time":
            attrs.update(time_attrs)
        elif units != "-":
            attrs["units"] = units
        outputs.append(ProductOutput(name, ("ramp_event",), columns[name],
                                     attrs))
    return outputs


__all__ = [
    "IEC_REFERENCE_DENSITY", "CURVE_POWER_COEFFICIENT", "PRODUCT_VARIABLES",
    "WIND_RESOURCE_VARIABLES", "RAMPS_VARIABLES", "RampsRefused",
    "RampWindow", "RampResult", "density_corrected_speed", "specific_power",
    "rated_speed_for_specific_power", "curve_power_fraction",
    "rotor_segments", "interp_profile", "rotor_equivalent_wind_speed",
    "shear_exponent", "wind_veer", "turbulence_intensity_from_tke",
    "resolved_turbulence_intensity", "capacity_factor", "parse_ramp_windows",
    "ramp_rate", "window_delta", "threshold_ramps", "swinging_door",
    "swinging_door_ramps", "detect_ramps", "ramp_event_columns",
    "ramp_event_summary", "ramps_dataset", "wind_resource_dataset",
    "compute_wind_resource", "compute_ramps", "rotor_power",
]
