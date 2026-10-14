"""Turbine-aware wind power at wind farms: ``woof energy power``.

Reads a ``woof-energy.forecast.v1`` file (and, optionally, the
``woof-energy.assets.v1`` document its sites came from) and writes
``woof-energy.power.v1``: power per plant on ``(time, plant)``, with the
turbine table (curve match and provenance per turbine) on ``turbine`` and,
with ``--per-turbine``, per-turbine series on ``(time, turbine)``.

What is computed, in order, for every wind row (a ``turbine`` site, or a
``plant`` site standing for ``count`` identical turbines):

1. **Curve.**  :meth:`woof.energy.turbine_library.TurbineLibrary.match`
   (exact model, fuzzy model, rating + rotor, generic) on what the asset
   says.  A plant-level row's turbine count comes from a count tag, else
   from the matched model's rating, else from a typical unit
   (:data:`TYPICAL_UNIT_KW`), and its unit rating is ``capacity / count``.
2. **Inflow.**  Hub-height wind by linear-in-ln(z) interpolation of
   ``wind_speed``; or the IEC 61400-12-1:2017 rotor-equivalent wind speed
   ``(sum_i A_i/A v_i^3)^(1/3)`` over the forecast heights inside the rotor
   disc when there are at least :data:`REWS_MIN_HEIGHTS` of them (``--hub-
   only`` turns this off).  Density at hub height from ``air_density`` (or
   ``air_pressure / (R_d air_temperature)``).
3. **Wakes** (per plant).  With two or more turbines at known positions, a
   directional table of each turbine's waked/free speed ratio is built on
   :data:`WAKE_DIRECTIONS_DEG` x :data:`WAKE_SPEEDS_MS` by the Jensen/PARK
   top-hat model (linear expansion ``k``, partial rotor overlap) or the
   Bastankhah & Porte-Agel (2014) Gaussian model, deficits combined by
   root-sum-square, each turbine's thrust coefficient taken from its curve
   at its own waked speed (:meth:`TurbineCurve.thrust_coefficient`) in
   upstream-to-downstream order, and the table smoothed over direction by a
   Gaussian of :data:`WAKE_DIRECTION_SIGMA_DEG`.  At run time the table is
   read bilinearly at the plant's mean free-stream speed and circular-mean
   hub direction.  A plant-level row uses :data:`FALLBACK_WAKE_LOSS` (by
   turbine count, plus :data:`OFFSHORE_EXTRA_WAKE_LOSS` offshore) as a
   uniform speed deficit ``1 - (1 - L)^(1/3)``, so ``L`` is the loss in
   the cubic part of the curve and vanishes above rated.
4. **Density correction.**  IEC 61400-12-1: ``v_eq = v (rho / 1.225)^(1/3)``
   on the waked speed, then the curve.
5. **High-wind shutdown with hysteresis.**  A turbine stops when its
   free-stream speed exceeds cut-out (above rated the thrust, and so the
   wake, is small; deciding on the waked speed would let a waked row run
   where the free-stream row has stopped), or with ``--gust-shutdown`` when the gust
   exceeds the gust limit, and restarts only once the speed falls below
   the restart speed (cut-out minus :data:`~woof.energy.turbine_library.
   RESTART_MARGIN_MS` unless the library says otherwise).  The state is
   carried forward along time, vectorised; a turbine is assumed running at
   the first step unless that step is a shutdown.
6. **Losses.**  ``power = waked x availability x (1 - icing_loss) x
   (1 - electrical_losses)``; ``gross_power_mw`` is free-stream power with
   the same shutdown logic and no losses.

Plants sum their rows; any NaN row makes the plant NaN for that step (and is
noted) rather than being dropped.  Everything is deterministic numpy over
``(time, row)``; the series may be any spacing (15-min or finer).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import datetime as _dt
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from woof.energy.contracts import (
    Asset,
    AssetCollection,
    ContractError,
    load_assets,
    sha256_file,
)
from woof.energy.products import _R_DRY, _Notes
from woof.energy.turbine_library import (
    PLANT_TAGS,
    REFERENCE_DENSITY,
    TurbineCurve,
    TurbineLibrary,
    TurbineMatch,
    TurbineSpec,
    asset_offshore,
    load_turbine_library,
    spec_from_asset,
)

POWER_SCHEMA = "woof-energy.power.v1"
MATCHES_SCHEMA = "woof-energy.turbine-matches.v1"
WAKE_MODELS = ("jensen", "bastankhah", "none")

# pinned engineering constants
JENSEN_K = {"onshore": 0.075, "offshore": 0.04}
BASTANKHAH_K = {"onshore": 0.04, "offshore": 0.025}
WAKE_DIRECTIONS_DEG = np.arange(0.0, 360.0, 5.0)
WAKE_SPEEDS_MS = np.arange(0.0, 31.0, 1.0)
WAKE_DIRECTION_SIGMA_DEG = 5.0
#: (largest turbine count, wake loss) rows; first row whose count is >= n.
FALLBACK_WAKE_LOSS = ((1, 0.0), (5, 0.03), (20, 0.06), (50, 0.09),
                      (100, 0.11), (10 ** 9, 0.13))
OFFSHORE_EXTRA_WAKE_LOSS = 0.02
REWS_MIN_HEIGHTS = 3
DEFAULT_ELECTRICAL_LOSSES = 0.02
DEFAULT_AVAILABILITY = 1.0
#: Unit rating assumed for a plant-level asset with no count or model.
TYPICAL_UNIT_KW = {"onshore": 3000.0, "offshore": 8000.0}
GUST_VARIABLES = ("wind_speed_of_gust", "wind_gust", "gust", "wind_gust_10m")
ICING_VARIABLE = "icing_loss_fraction"

TIME_PLANT = ("time", "plant")
TIME_TURBINE = ("time", "turbine")

#: ``power.v1`` data variables: name -> (dims, units, long_name).
POWER_VARIABLES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "power_mw": (TIME_PLANT, "MW", "net plant power"),
    "gross_power_mw": (TIME_PLANT, "MW",
                       "free-stream plant power without losses"),
    "wake_loss_mw": (TIME_PLANT, "MW", "power lost to wakes"),
    "capacity_factor": (TIME_PLANT, "1", "net power over capacity"),
    "hub_wind_speed": (TIME_PLANT, "m s-1",
                       "mean free-stream hub-height wind speed"),
    "running_fraction": (TIME_PLANT, "1",
                         "capacity fraction not in high-wind shutdown"),
}
#: Optional ``--per-turbine`` variables.
TURBINE_VARIABLES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "turbine_power_mw": (TIME_TURBINE, "MW", "net turbine-row power"),
    "turbine_wind_speed": (TIME_TURBINE, "m s-1",
                           "free-stream hub or rotor-equivalent wind speed"),
    "turbine_waked_wind_speed": (TIME_TURBINE, "m s-1",
                                 "waked wind speed"),
    "turbine_equivalent_wind_speed": (TIME_TURBINE, "m s-1",
                                      "waked density-corrected wind speed"),
    "turbine_running": (TIME_TURBINE, "1",
                        "1 running, 0 in high-wind shutdown"),
}
#: ``power.v1`` coordinates: name -> (dims, units-or-None, long_name).
POWER_COORDINATES: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    "time": (("time",), None, "valid time (UTC)"),
    "plant_id": (("plant",), None, "plant identifier"),
    "plant_name": (("plant",), None, "plant name"),
    "plant_lat": (("plant",), "degree_north", "plant mean latitude"),
    "plant_lon": (("plant",), "degree_east", "plant mean longitude"),
    "capacity_mw": (("plant",), "MW", "sum of turbine ratings"),
    "n_turbines": (("plant",), "1", "turbines in the plant"),
    "offshore": (("plant",), "1", "1 offshore, 0 onshore, -1 unknown"),
    "wake_method": (("plant",), None, "wake model applied"),
    "fallback_wake_loss": (("plant",), "1",
                           "table wake loss (NaN unless used)"),
    "turbine_id": (("turbine",), None, "turbine row identifier"),
    "site_id": (("turbine",), None, "forecast site"),
    "asset_id": (("turbine",), None, "asset identifier"),
    "plant_index": (("turbine",), "1", "index on the plant dimension"),
    "count": (("turbine",), "1", "turbines the row stands for"),
    "turbine_lat": (("turbine",), "degree_north", "latitude"),
    "turbine_lon": (("turbine",), "degree_east", "longitude"),
    "rated_mw": (("turbine",), "MW", "rating of one turbine"),
    "rotor_diameter_m": (("turbine",), "m", "rotor diameter"),
    "hub_height_m": (("turbine",), "m", "hub height"),
    "curve_id": (("turbine",), None, "power curve used"),
    "curve_source": (("turbine",), None, "oedb, parametric or generic"),
    "match_quality": (("turbine",), None,
                      "exact, fuzzy, size, generic or none"),
    "match_score": (("turbine",), "1", "match score"),
    "speed_method": (("turbine",), None, "hub or rews"),
}


class PowerRefused(ContractError):
    """``woof energy power`` cannot honour the request."""


# --------------------------------------------------------------------------
# inflow


def rotor_equivalent_wind_speed(speed: Any, heights: Any, hub_height_m: Any,
                                rotor_diameter_m: Any, *,
                                min_heights: int = REWS_MIN_HEIGHTS
                                ) -> tuple[np.ndarray, np.ndarray]:
    """IEC 61400-12-1:2017 rotor-equivalent wind speed.

    ``speed`` is ``(time, n, height)``.  Each forecast height inside the
    rotor disc owns the horizontal band of the disc between the midpoints
    to its neighbours (the outermost bands reach the disc edge);
    ``v_eq = (sum_i A_i / A v_i^3)^(1/3)``.  Returns ``(rews (time, n),
    used (n,))``; ``used`` is False (and ``rews`` NaN) where fewer than
    ``min_heights`` heights fall inside the disc.

    Same name and leading arguments as
    ``woof.energy.ramps.rotor_equivalent_wind_speed``, which integrates over
    strips interpolated from the profile instead and returns a count of
    strips outside the forecast heights as its second value.
    """

    heights = np.asarray(heights, dtype=float)
    hub = np.atleast_1d(np.asarray(hub_height_m, dtype=float))[:, None]
    radius = 0.5 * np.atleast_1d(np.asarray(rotor_diameter_m,
                                            dtype=float))[:, None]
    z = heights[None, :] - hub
    inside = np.abs(z) <= radius
    used = (inside.sum(axis=1) >= min_heights) & np.isfinite(radius[:, 0]) \
        & np.isfinite(hub[:, 0])
    mids = 0.5 * (heights[:-1] + heights[1:])
    mid_lo = np.concatenate([[-np.inf], mids])[None, :] - hub
    mid_hi = np.concatenate([mids, [np.inf]])[None, :] - hub
    prev_in = np.concatenate([np.zeros_like(inside[:, :1]), inside[:, :-1]],
                             axis=1)
    next_in = np.concatenate([inside[:, 1:], np.zeros_like(inside[:, :1])],
                             axis=1)
    lower = np.where(prev_in, mid_lo, -radius)
    upper = np.where(next_in, mid_hi, radius)
    lower = np.clip(lower, -radius, radius)
    upper = np.clip(upper, -radius, radius)

    def above(level):
        ratio = np.clip(level / np.where(radius > 0, radius, 1.0), -1.0, 1.0)
        return radius ** 2 * (np.arccos(ratio)
                              - ratio * np.sqrt(1.0 - ratio ** 2))

    with np.errstate(invalid="ignore"):
        weight = np.where(inside, above(lower) - above(upper), 0.0)
        weight = weight / np.where(radius > 0, math.pi * radius ** 2, 1.0)
        weight = np.where(used[:, None], weight, 0.0)
        cubes = np.where(weight[None, :, :] > 0.0,
                         np.asarray(speed, float) ** 3, 0.0)
        cube = np.einsum("tnh,nh->tn", cubes, weight)
    rews = np.cbrt(cube)
    return np.where(used[None, :], rews, np.nan), used


def density_corrected_speed(speed: Any, density: Any,
                            reference_density: float = REFERENCE_DENSITY
                            ) -> np.ndarray:
    """IEC 61400-12-1 ``v (rho / rho_ref)^(1/3)`` (pitch-regulated); the
    signature of ``woof.energy.ramps.density_corrected_speed``."""

    return np.asarray(speed, dtype=float) * np.cbrt(
        np.asarray(density, dtype=float) / reference_density)


def running_state(speed: np.ndarray, cut_out: np.ndarray,
                  restart: np.ndarray, *, gust: np.ndarray | None = None,
                  gust_limit: np.ndarray | None = None) -> np.ndarray:
    """High-wind shutdown with hysteresis, ``(time, n)`` booleans.

    Off when ``speed > cut_out`` (or ``gust > gust_limit``), on when
    ``speed < restart``, otherwise the previous state; running at the first
    step unless it is a shutdown.  NaN speed holds the previous state.
    """

    speed = np.asarray(speed, dtype=float)
    with np.errstate(invalid="ignore"):
        off = speed > np.asarray(cut_out)[None, :]
        if gust is not None and gust_limit is not None:
            off |= np.asarray(gust) > np.asarray(gust_limit)[None, :]
        on = (speed < np.asarray(restart)[None, :]) & ~off
    event = np.where(off, 0, np.where(on, 1, -1)).astype(np.int8)
    steps = np.arange(speed.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(event >= 0, steps, -1), axis=0)
    carried = np.take_along_axis(event, np.clip(last, 0, None), axis=0)
    return np.where(last >= 0, carried, 1).astype(bool)


# --------------------------------------------------------------------------
# wakes


def fallback_wake_loss(count: int, offshore: bool | None) -> float:
    """Plant wake loss from :data:`FALLBACK_WAKE_LOSS` (no layout known)."""

    loss = next(value for limit, value in FALLBACK_WAKE_LOSS
                if count <= limit)
    if offshore and count > 1:
        loss += OFFSHORE_EXTRA_WAKE_LOSS
    return loss


def _circle_overlap(rw: np.ndarray, r: np.ndarray, d: np.ndarray
                    ) -> np.ndarray:
    """Fraction of the circle of radius ``r`` inside one of radius ``rw``
    whose centre is ``d`` away."""

    rw, r, d = np.broadcast_arrays(rw, r, d)
    out = np.zeros(rw.shape)
    full = d + r <= rw
    out[full] = 1.0
    inner = d + rw <= r
    out[inner & ~full] = (rw[inner & ~full] / r[inner & ~full]) ** 2
    part = ~full & ~inner & (d < rw + r)
    if part.any():
        a, b, c = rw[part], r[part], np.maximum(d[part], 1e-9)
        alpha = np.arccos(np.clip((c ** 2 + a ** 2 - b ** 2) / (2 * c * a),
                                  -1, 1))
        beta = np.arccos(np.clip((c ** 2 + b ** 2 - a ** 2) / (2 * c * b),
                                 -1, 1))
        area = (a ** 2 * (alpha - np.sin(2 * alpha) / 2)
                + b ** 2 * (beta - np.sin(2 * beta) / 2))
        out[part] = area / (math.pi * b ** 2)
    return out


def wake_ratio_table(x_m: np.ndarray, y_m: np.ndarray, hub_m: np.ndarray,
                     curves: Sequence[TurbineCurve], rotor_m: np.ndarray, *,
                     model: str = "jensen", offshore: bool = False,
                     directions_deg: np.ndarray = WAKE_DIRECTIONS_DEG,
                     speeds_ms: np.ndarray = WAKE_SPEEDS_MS,
                     sigma_deg: float = WAKE_DIRECTION_SIGMA_DEG
                     ) -> np.ndarray:
    """Waked/free speed ratio ``(direction, speed, turbine)`` for a layout.

    ``x_m``/``y_m`` are east/north metres; directions are where the wind
    comes FROM, clockwise from north.  See the module docstring.
    """

    if model not in ("jensen", "bastankhah"):
        raise PowerRefused(f"wake model {model!r} has no table")
    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    hub = np.asarray(hub_m, float)
    rotor = np.asarray(rotor_m, float)
    n = x.size
    theta = np.radians(np.asarray(directions_deg, float))[:, None]
    downstream = -x[None, :] * np.sin(theta) - y[None, :] * np.cos(theta)
    across = x[None, :] * np.cos(theta) - y[None, :] * np.sin(theta)
    n_dir, n_speed = theta.shape[0], len(speeds_ms)
    free = np.asarray(speeds_ms, float)[None, :]
    grid_step = 0.1
    grid = np.arange(0.0, 40.0 + grid_step / 2, grid_step)
    ct_grid = np.stack([curve.thrust_coefficient(grid) for curve in curves])
    order = np.argsort(downstream, axis=1, kind="stable")
    deficit_sq = np.zeros((n_dir, n_speed, n))
    ratio = np.ones((n_dir, n_speed, n))
    rows = np.arange(n_dir)
    k = (JENSEN_K if model == "jensen" else BASTANKHAH_K)[
        "offshore" if offshore else "onshore"]
    for rank in range(n):
        i = order[:, rank]
        own = 1.0 - np.sqrt(deficit_sq[rows, :, i])
        own = np.clip(own, 0.0, 1.0)
        ratio[rows, :, i] = own
        speed_i = free * own
        pos = np.clip(speed_i / grid_step, 0.0, grid.size - 1.000001)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        ct = (ct_grid[i[:, None], lo] * (1 - frac)
              + ct_grid[i[:, None], lo + 1] * frac)
        ct = np.clip(ct, 0.0, 0.999)
        dx = downstream - downstream[rows, i][:, None]
        dr = np.hypot(across - across[rows, i][:, None],
                      hub[None, :] - hub[i][:, None])
        behind = dx > 1e-6
        radius_i = 0.5 * rotor[i][:, None]
        if model == "jensen":
            wake_r = radius_i + k * np.where(behind, dx, 0.0)
            shape = (radius_i / wake_r) ** 2 * _circle_overlap(
                wake_r, 0.5 * rotor[None, :], dr)
            shape = np.where(behind, shape, 0.0)
            deficit = (1.0 - np.sqrt(1.0 - ct))[:, :, None] \
                * shape[:, None, :]
        else:
            diameter = rotor[i][:, None, None]
            root = np.sqrt(1.0 - ct)
            beta = 0.5 * (1.0 + root) / root
            eps = 0.2 * np.sqrt(beta)
            sigma_d = (k * np.where(behind, dx, 0.0)[:, None, :]
                       / diameter + eps[:, :, None])
            centre = 1.0 - np.sqrt(np.clip(
                1.0 - ct[:, :, None] / (8.0 * sigma_d ** 2), 0.0, 1.0))
            sigma = sigma_d * diameter
            deficit = centre * np.exp(-dr[:, None, :] ** 2
                                      / (2.0 * sigma ** 2))
            deficit = np.where(behind[:, None, :], deficit, 0.0)
        deficit_sq += deficit ** 2
    if sigma_deg > 0 and n_dir > 1:
        step = 360.0 / n_dir
        reach = int(math.ceil(3 * sigma_deg / step))
        offsets = np.arange(-reach, reach + 1)
        weights = np.exp(-0.5 * (offsets * step / sigma_deg) ** 2)
        weights /= weights.sum()
        ratio = sum(w * np.roll(ratio, -o, axis=0)
                    for o, w in zip(offsets, weights))
    return ratio


def apply_wake_table(table: np.ndarray, speed: np.ndarray,
                     direction: np.ndarray | None, *,
                     directions_deg: np.ndarray = WAKE_DIRECTIONS_DEG,
                     speeds_ms: np.ndarray = WAKE_SPEEDS_MS) -> np.ndarray:
    """Read ``table`` at per-step ambient ``speed`` and ``direction``.

    Bilinear in (direction, speed), circular in direction, clamped at the
    speed ends; with no direction the direction-mean table is used.
    Returns ``(time, turbine)``; NaN where ``speed`` (or ``direction``) is.
    """

    speed = np.asarray(speed, float)
    speeds_ms = np.asarray(speeds_ms, float)
    step_v = speeds_ms[1] - speeds_ms[0]
    pos_v = np.clip((np.nan_to_num(speed) - speeds_ms[0]) / step_v, 0.0,
                    speeds_ms.size - 1.000001)
    v0 = np.floor(pos_v).astype(int)
    wv = (pos_v - v0)[:, None]
    if direction is None:
        mean = table.mean(axis=0)
        out = mean[v0] * (1 - wv) + mean[v0 + 1] * wv
        return np.where(np.isfinite(speed)[:, None], out, np.nan)
    direction = np.asarray(direction, float)
    n_dir = len(directions_deg)
    step_d = 360.0 / n_dir
    pos_d = np.mod(np.nan_to_num(direction) - directions_deg[0], 360.0) \
        / step_d
    d0 = np.floor(pos_d).astype(int) % n_dir
    d1 = (d0 + 1) % n_dir
    wd = (pos_d - np.floor(pos_d))[:, None]
    out = ((table[d0, v0] * (1 - wv) + table[d0, v0 + 1] * wv) * (1 - wd)
           + (table[d1, v0] * (1 - wv) + table[d1, v0 + 1] * wv) * wd)
    ok = np.isfinite(speed) & np.isfinite(direction)
    return np.where(ok[:, None], out, np.nan)


def _local_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray,
                                                         np.ndarray]:
    from woof.energy.geometry import local_projection

    projection = local_projection(float(np.mean(lat)), float(np.mean(lon)))
    x, y = projection.forward(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x, float), np.asarray(y, float)


def _circular_mean_deg(direction: np.ndarray, idx: np.ndarray
                       ) -> np.ndarray:
    """Per-step circular mean of ``direction[:, idx]`` (NaN if none)."""

    rad = np.radians(direction)
    s = _nanmean_rows(np.sin(rad), idx)
    c = _nanmean_rows(np.cos(rad), idx)
    return np.mod(np.degrees(np.arctan2(s, c)), 360.0)


# --------------------------------------------------------------------------
# rows and plants


@dataclass
class Row:
    """One wind row: a turbine site, or a plant site of ``count`` turbines."""

    row_id: str
    site_index: int
    site_id: str
    asset_id: str
    plant_key: str
    count: int
    spec: TurbineSpec
    match: TurbineMatch
    hub_height_m: float
    rotor_diameter_m: float
    lon: float
    lat: float
    positioned: bool
    speed_method: str = "hub"

    @property
    def rated_kw(self) -> float:
        return float(self.match.rated_kw) if self.match.rated_kw is not None \
            else float("nan")


@dataclass
class Plant:
    plant_id: str
    name: str
    rows: list[int] = field(default_factory=list)
    offshore: bool | None = None
    wake_method: str = "none"
    fallback_loss: float = float("nan")


@dataclass
class PowerOptions:
    wake_model: str = "jensen"
    rews: bool = True
    availability: float = DEFAULT_AVAILABILITY
    electrical_losses: float = DEFAULT_ELECTRICAL_LOSSES
    gust_shutdown: bool = False
    per_turbine: bool = False

    def __post_init__(self):
        if self.wake_model not in WAKE_MODELS:
            raise PowerRefused(f"wake model {self.wake_model!r} is not one "
                               f"of {WAKE_MODELS}")
        if not 0.0 < self.availability <= 1.0:
            raise PowerRefused("availability must be in (0, 1]")
        if not 0.0 <= self.electrical_losses < 1.0:
            raise PowerRefused("electrical_losses must be in [0, 1)")


def plant_keys(turbines: Sequence[Asset], wind_plants: Sequence[Asset]
               ) -> dict[str, str]:
    """Turbine asset_id -> plant key.

    A plant tag (:data:`~woof.energy.turbine_library.PLANT_TAGS`) naming a
    wind plant asset by id, or by name (case-insensitive), joins that plant;
    otherwise a turbine inside a wind plant polygon joins it (first plant by
    asset_id); otherwise the tag value, or the turbine alone.
    """

    from woof.energy.geometry import point_in_polygon
    from woof.energy.sites import _polygons_of

    ids = {plant.asset_id for plant in wind_plants}
    by_name: dict[str, str] = {}
    for plant in sorted(wind_plants, key=lambda p: p.asset_id):
        if plant.name:
            by_name.setdefault(plant.name.strip().lower(), plant.asset_id)
    keys: dict[str, str] = {}
    fallback: dict[str, str] = {}
    pending: list[Asset] = []
    for asset in turbines:
        tag = next((asset.tags[k] for k in PLANT_TAGS
                    if asset.tags.get(k)), None)
        if tag in ids:
            keys[asset.asset_id] = tag
        elif tag and tag.strip().lower() in by_name:
            keys[asset.asset_id] = by_name[tag.strip().lower()]
        else:
            fallback[asset.asset_id] = tag or asset.asset_id
            if asset.geometry["type"] == "Point":
                pending.append(asset)
    if pending:
        lon = np.array([a.geometry["coordinates"][0] for a in pending], float)
        lat = np.array([a.geometry["coordinates"][1] for a in pending], float)
        free = np.ones(len(pending), dtype=bool)
        for plant in sorted(wind_plants, key=lambda p: p.asset_id):
            if plant.geometry["type"] not in ("Polygon", "MultiPolygon"):
                continue
            hit = np.asarray(point_in_polygon(
                lon, lat, _polygons_of(plant.geometry)), dtype=bool) & free
            for i in np.flatnonzero(hit):
                keys[pending[i].asset_id] = plant.asset_id
            free &= ~hit
            if not free.any():
                break
    for asset_id, key in fallback.items():
        keys.setdefault(asset_id, key)
    return keys


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def build_rows(forecast, assets: AssetCollection | None,
               library: TurbineLibrary, notes: _Notes
               ) -> tuple[list[Row], list[Plant]]:
    """Wind rows of a forecast, matched to curves and grouped into plants."""

    kind = np.asarray(forecast.coord("kind")).astype(str)
    site_ids = np.asarray(forecast.coord("site_id")).astype(str)
    asset_ids = (np.asarray(forecast.coords["asset_id"]).astype(str)
                 if "asset_id" in forecast.coords else site_ids)
    hub_coord = (np.asarray(forecast.coords["hub_height_m"], float)
                 if "hub_height_m" in forecast.coords
                 else np.full(kind.size, np.nan))
    cap_coord = (np.asarray(forecast.coords["capacity_mw"], float)
                 if "capacity_mw" in forecast.coords
                 else np.full(kind.size, np.nan))
    lat = np.asarray(forecast.coords.get("lat", np.full(kind.size, np.nan)),
                     float)
    lon = np.asarray(forecast.coords.get("lon", np.full(kind.size, np.nan)),
                     float)
    by_id = {a.asset_id: a for a in assets.assets} if assets else {}
    wind_plants = [a for a in (assets.assets if assets else [])
                   if a.kind == "plant" and a.generator_source == "wind"]
    plants_by_id = {a.asset_id: a for a in wind_plants}
    turbine_assets = [by_id[asset_ids[s]] for s in range(kind.size)
                      if kind[s] == "turbine" and asset_ids[s] in by_id]
    keys = plant_keys(turbine_assets, wind_plants)

    candidates = []
    for s in range(kind.size):
        asset = by_id.get(asset_ids[s])
        if kind[s] == "turbine":
            candidates.append((s, asset, False))
        elif kind[s] == "plant":
            if assets is not None and asset is not None:
                if asset.generator_source == "wind":
                    candidates.append((s, asset, True))
            elif np.isfinite(hub_coord[s]):
                candidates.append((s, asset, True))
    if assets is not None:
        missing = sorted({asset_ids[s] for s, asset, _ in candidates
                          if asset is None})
        if missing:
            raise PowerRefused(
                f"{len(missing)} wind site(s) name assets the assets "
                f"document does not hold (first: {missing[:3]}); pass the "
                "assets document the sites were built from")
    else:
        notes.add("no assets document: curves matched from the forecast's "
                  "capacity_mw alone (generic), each turbine its own plant")

    rows: list[Row] = []
    for s, asset, plant_level in candidates:
        cap_mw = _finite_or_none(cap_coord[s])
        hub = _finite_or_none(hub_coord[s])
        if not plant_level:
            spec = spec_from_asset(asset) if asset else TurbineSpec()
            if spec.rated_kw is None and cap_mw is not None:
                spec = replace(spec, rated_kw=cap_mw * 1000.0)
            match = library.match(spec)
            count = 1
            plant_key = keys.get(asset_ids[s], asset_ids[s])
            positioned = asset is not None and \
                asset.geometry["type"] == "Point"
        else:
            spec = spec_from_asset(asset, per_turbine=False) if asset \
                else TurbineSpec()
            capacity_mw = (asset.capacity_mw if asset and asset.capacity_mw
                           else cap_mw)
            match, count = _plant_level_match(
                spec, capacity_mw, library)
            plant_key = asset.asset_id if asset else asset_ids[s]
            positioned = False
        rotor = spec.rotor_diameter_m or (
            match.curve.rotor_diameter_m if match.curve else None)
        hub = hub or spec.hub_height_m
        if hub is None and match.curve is not None and \
                match.curve.hub_heights_m:
            hub = float(np.median(match.curve.hub_heights_m))
            notes.add(f"{asset_ids[s]}: no hub height; the median library "
                      f"hub height {hub:g} m of {match.curve.curve_id} was "
                      "used")
        rows.append(Row(
            row_id=asset_ids[s] + ("#plant" if plant_level else ""),
            site_index=s, site_id=site_ids[s], asset_id=asset_ids[s],
            plant_key=plant_key, count=count, spec=spec, match=match,
            hub_height_m=hub if hub is not None else float("nan"),
            rotor_diameter_m=rotor if rotor is not None else float("nan"),
            lon=float(lon[s]), lat=float(lat[s]), positioned=positioned))

    # turbines and a plant-level row of the same plant: keep the turbines
    turbine_plants = {r.plant_key for r in rows if not r.row_id.endswith(
        "#plant")}
    dropped = [r for r in rows if r.row_id.endswith("#plant")
               and r.plant_key in turbine_plants]
    if dropped:
        notes.add(f"{len(dropped)} plant-level site(s) skipped because "
                  "their turbines are sites too")
        rows = [r for r in rows if r not in dropped]

    plants: dict[str, Plant] = {}
    for index, row in enumerate(rows):
        plant = plants.get(row.plant_key)
        if plant is None:
            asset = plants_by_id.get(row.plant_key) or by_id.get(
                row.plant_key)
            name = (asset.name if asset is not None and asset.name else
                    row.plant_key)
            offshore = asset_offshore(asset) if asset is not None else None
            plant = plants[row.plant_key] = Plant(
                plant_id=row.plant_key, name=name, offshore=offshore)
        plant.rows.append(index)
        if row.spec.offshore and plant.offshore is None:
            plant.offshore = True
    ordered = list(plants.values())
    order = [i for plant in ordered for i in plant.rows]
    rows = [rows[i] for i in order]
    start = 0
    for plant in ordered:
        size = len(plant.rows)
        plant.rows = list(range(start, start + size))
        start += size
    return rows, ordered


def _plant_level_match(spec: TurbineSpec, capacity_mw: float | None,
                       library: TurbineLibrary
                       ) -> tuple[TurbineMatch, int]:
    """Typical turbine of a plant known only by its capacity."""

    if capacity_mw is None:
        return (TurbineMatch("none", None, 0.0,
                             "plant-level asset with no capacity_mw", None),
                1)
    capacity_kw = capacity_mw * 1000.0
    if spec.count:
        count = spec.count
        how = "turbine count tag"
    else:
        probe = library.match(replace(spec, rated_kw=None)) \
            if spec.model else None
        if probe is not None and probe.quality in ("exact", "fuzzy"):
            unit = probe.curve.rated_kw
            how = f"model rating {unit:g} kW"
        else:
            unit = TYPICAL_UNIT_KW["offshore" if spec.offshore
                                   else "onshore"]
            how = f"typical {'offshore' if spec.offshore else 'onshore'} " \
                  f"unit {unit:g} kW"
        count = max(1, int(round(capacity_kw / unit)))
    unit_kw = capacity_kw / count
    match = library.match(replace(spec, rated_kw=unit_kw))
    prefix = (f"plant-level: {count} x {unit_kw:g} kW from {how}; ")
    match = TurbineMatch(match.quality, match.curve, match.score,
                         prefix + match.reason, match.rated_kw)
    return match, count


# --------------------------------------------------------------------------
# computing


@dataclass
class PowerResult:
    times: list
    time_var: dict
    rows: list[Row]
    plants: list[Plant]
    plant_vars: dict[str, np.ndarray]
    turbine_vars: dict[str, np.ndarray]
    notes: list[str]
    attrs: dict[str, Any]


def _at_hub(field_: np.ndarray, heights: np.ndarray, hub: np.ndarray, *,
            log: bool) -> tuple[np.ndarray, np.ndarray]:
    """(time, row, height) -> (time, row) at each row's hub height."""

    from woof.energy.products import interp_height

    target = np.where(np.isfinite(hub), hub, heights[0])
    value, outside = interp_height(field_, heights, target, log=log)
    value = np.where(np.isfinite(hub)[None, :], value, np.nan)
    return value, outside & np.isfinite(hub)


def _nanmean_rows(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
    block = data[:, idx]
    ok = np.isfinite(block)
    total = np.where(ok, block, 0.0).sum(axis=1)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def compute_power(forecast, assets: AssetCollection | None,
                  library: TurbineLibrary, options: PowerOptions, *,
                  gust: np.ndarray | None = None,
                  icing_loss: np.ndarray | None = None,
                  notes: _Notes | None = None) -> PowerResult:
    """Plant and turbine power from a read ``forecast.v1``.

    ``gust`` and ``icing_loss`` are optional ``(time, site)`` arrays on the
    forecast's site axis.
    """

    notes = notes if notes is not None else _Notes()
    rows, plants = build_rows(forecast, assets, library, notes)
    if not rows:
        raise PowerRefused("the forecast has no wind sites (kind turbine, or "
                           "kind plant with a wind asset or hub height)")
    v = forecast.variables
    if "wind_speed" not in v:
        raise PowerRefused("the forecast has no wind_speed")
    nt = len(forecast.times)
    n = len(rows)
    sites = np.array([r.site_index for r in rows])
    hub = np.array([r.hub_height_m for r in rows])
    rotor = np.array([r.rotor_diameter_m for r in rows])
    count = np.array([r.count for r in rows], dtype=float)
    rated_kw = np.array([r.rated_kw for r in rows])

    no_hub = ~np.isfinite(hub)
    if no_hub.any():
        notes.add(f"{int(no_hub.sum())} wind row(s) have no hub height; "
                  "their power is NaN")
    no_curve = [r.row_id for r in rows if r.match.curve is None]
    if no_curve:
        notes.add(f"{len(no_curve)} wind row(s) have no rating to scale a "
                  f"curve (first: {no_curve[:3]}); their power is NaN")

    heights = forecast.heights

    def at_hub(name: str, log: bool) -> np.ndarray:
        return _at_hub(v[name][:, sites, :], heights, hub, log=log)[0]

    hub_speed, outside = _at_hub(v["wind_speed"][:, sites, :], heights, hub,
                                 log=True)
    if outside.any():
        notes.add(f"hub height outside the forecast heights for "
                  f"{int(outside.sum())} row(s); the nearest height was used")
    speed = hub_speed
    if options.rews:
        rews, used = rotor_equivalent_wind_speed(
            v["wind_speed"][:, sites, :], forecast.heights, hub, rotor)
        if used.any():
            speed = np.where(used[None, :], rews, hub_speed)
            for i in np.flatnonzero(used):
                rows[i].speed_method = "rews"
            notes.add(f"rotor-equivalent wind speed used for "
                      f"{int(used.sum())} of {n} row(s) (at least "
                      f"{REWS_MIN_HEIGHTS} forecast heights across the "
                      "rotor); hub height for the rest")
        else:
            notes.add("hub-height wind for every row: no rotor spans "
                      f"{REWS_MIN_HEIGHTS} forecast heights")
    if "air_density" in v:
        rho = at_hub("air_density", False)
    elif "air_pressure" in v and "air_temperature" in v:
        rho = at_hub("air_pressure", False) / (
            _R_DRY * at_hub("air_temperature", False))
        notes.add("no air_density in the forecast; density from "
                  "air_pressure / (R_d air_temperature), dry air")
    else:
        raise PowerRefused("the forecast needs air_density, or air_pressure "
                           "and air_temperature, for the density correction")
    direction = None
    if "u" in v and "v" in v:
        direction = np.mod(np.degrees(np.arctan2(
            -at_hub("u", True), -at_hub("v", True))), 360.0)
    elif "wind_from_direction" in v:
        rad = np.radians(v["wind_from_direction"][:, sites, :])
        sin_, _ = _at_hub(np.sin(rad), heights, hub, log=False)
        cos_, _ = _at_hub(np.cos(rad), heights, hub, log=False)
        direction = np.mod(np.degrees(np.arctan2(sin_, cos_)), 360.0)

    # wakes
    ratio = np.ones((nt, n))
    any_layout = False
    for plant in plants:
        idx = np.array(plant.rows)
        offshore = bool(plant.offshore)
        if options.wake_model == "none":
            plant.wake_method = "none"
            continue
        positioned = [i for i in idx if rows[i].positioned
                      and rows[i].match.curve is not None
                      and np.isfinite(rotor[i]) and np.isfinite(hub[i])]
        if len(idx) >= 2 and len(positioned) == len(idx):
            x, y = _local_xy(np.array([rows[i].lon for i in idx]),
                             np.array([rows[i].lat for i in idx]))
            table = wake_ratio_table(
                x, y, hub[idx], [rows[i].match.curve for i in idx],
                rotor[idx], model=options.wake_model, offshore=offshore)
            ambient = _nanmean_rows(speed, idx)
            plant_dir = (_circular_mean_deg(direction, idx)
                         if direction is not None else None)
            ratio[:, idx] = apply_wake_table(table, ambient, plant_dir)
            plant.wake_method = f"layout-{options.wake_model}"
            any_layout = True
        elif len(idx) == 1 and rows[idx[0]].count > 1:
            loss = fallback_wake_loss(rows[idx[0]].count, plant.offshore)
            ratio[:, idx] = np.cbrt(1.0 - loss)
            plant.wake_method = "fallback-table"
            plant.fallback_loss = loss
        elif len(idx) >= 2:
            total = int(count[idx].sum())
            loss = fallback_wake_loss(total, plant.offshore)
            ratio[:, idx] = np.cbrt(1.0 - loss)
            plant.wake_method = "fallback-table"
            plant.fallback_loss = loss
            notes.add(f"plant {plant.plant_id}: some turbines lack a point "
                      "position or curve; fallback wake table used")
        else:
            plant.wake_method = "none"
    if any_layout and direction is None:
        notes.add("no u/v or wind_from_direction in the forecast: wake "
                  "tables averaged over all directions")

    waked = speed * ratio
    veq = density_corrected_speed(waked, rho)
    veq_free = density_corrected_speed(speed, rho)
    cut_out = np.array([r.match.curve.cut_out_ms if r.match.curve else np.inf
                        for r in rows])
    restart = np.array([r.match.curve.restart if r.match.curve else np.inf
                        for r in rows])
    gust_limit = np.array([r.match.curve.gust_limit if r.match.curve
                           else np.inf for r in rows])
    gust_rows = gust[:, sites] if (gust is not None
                                   and options.gust_shutdown) else None
    # Shutdown is decided on the free-stream speed: above rated the thrust,
    # and so the wake, is small, and a waked row must not keep running
    # where the free-stream row has stopped.
    running = running_state(speed, cut_out, restart, gust=gust_rows,
                            gust_limit=gust_limit)
    running_free = running
    fraction = np.full((nt, n), np.nan)
    fraction_free = np.full((nt, n), np.nan)
    for i, row in enumerate(rows):
        if row.match.curve is None:
            continue
        fraction[:, i] = row.match.curve.fraction(veq[:, i])
        fraction_free[:, i] = row.match.curve.fraction(veq_free[:, i])
    row_mw = count * rated_kw / 1000.0
    waked_mw = fraction * running * row_mw[None, :]
    gross_mw = fraction_free * running_free * row_mw[None, :]
    net_mw = waked_mw * options.availability * \
        (1.0 - options.electrical_losses)
    if icing_loss is not None:
        loss = icing_loss[:, sites]
        bad = ~np.isfinite(loss)
        if bad.any():
            notes.add(f"icing loss is NaN at {int(bad.sum())} row-step(s); "
                      "no icing loss applied there")
        net_mw = net_mw * (1.0 - np.where(bad, 0.0, loss))

    # aggregate
    starts = np.array([plant.rows[0] for plant in plants])
    plant_power = np.add.reduceat(net_mw, starts, axis=1)
    plant_gross = np.add.reduceat(gross_mw, starts, axis=1)
    plant_waked = np.add.reduceat(waked_mw, starts, axis=1)
    capacity = np.add.reduceat(row_mw, starts)
    weight = count[None, :] * np.isfinite(speed)
    plant_speed = np.add.reduceat(np.nan_to_num(speed) * weight, starts,
                                  axis=1)
    weight_sum = np.add.reduceat(weight, starts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        plant_speed = np.where(weight_sum > 0, plant_speed / weight_sum,
                               np.nan)
        running_cap = np.add.reduceat(running * row_mw[None, :], starts,
                                      axis=1)
        plant_running = running_cap / capacity[None, :]
        cf = plant_power / capacity[None, :]
    nan_plants = [plant.plant_id for k, plant in enumerate(plants)
                  if np.isnan(plant_power[:, k]).any()]
    if nan_plants:
        notes.add(f"{len(nan_plants)} plant(s) have NaN power at some step "
                  f"(a row without hub height, rating or forecast data; "
                  f"first: {nan_plants[:3]})")
    plant_vars = {
        "power_mw": plant_power,
        "gross_power_mw": plant_gross,
        "wake_loss_mw": plant_gross - plant_waked,
        "capacity_factor": cf,
        "hub_wind_speed": plant_speed,
        "running_fraction": plant_running,
    }
    turbine_vars = {}
    if options.per_turbine:
        turbine_vars = {
            "turbine_power_mw": net_mw,
            "turbine_wind_speed": speed,
            "turbine_waked_wind_speed": waked,
            "turbine_equivalent_wind_speed": veq,
            "turbine_running": running.astype(np.int8),
        }
    return PowerResult(times=forecast.times, time_var=forecast.time_var,
                       rows=rows, plants=plants, plant_vars=plant_vars,
                       turbine_vars=turbine_vars, notes=list(notes),
                       attrs={})


# --------------------------------------------------------------------------
# writing and reading power.v1


def _plant_coords(result: PowerResult) -> dict[str, np.ndarray]:
    rows, plants = result.rows, result.plants
    out: dict[str, np.ndarray] = {
        "plant_id": np.array([p.plant_id for p in plants], dtype=object),
        "plant_name": np.array([p.name for p in plants], dtype=object),
        "plant_lat": np.array([np.nanmean([rows[i].lat for i in p.rows])
                               for p in plants]),
        "plant_lon": np.array([np.nanmean([rows[i].lon for i in p.rows])
                               for p in plants]),
        "capacity_mw": np.array([sum(rows[i].count * rows[i].rated_kw
                                     for i in p.rows) / 1000.0
                                 for p in plants]),
        "n_turbines": np.array([sum(rows[i].count for i in p.rows)
                                for p in plants], dtype=np.int32),
        "offshore": np.array([-1 if p.offshore is None else int(p.offshore)
                              for p in plants], dtype=np.int8),
        "wake_method": np.array([p.wake_method for p in plants],
                                dtype=object),
        "fallback_wake_loss": np.array([p.fallback_loss for p in plants]),
    }
    return out


def _turbine_coords(result: PowerResult) -> dict[str, np.ndarray]:
    rows = result.rows
    plant_index = np.zeros(len(rows), dtype=np.int32)
    for k, plant in enumerate(result.plants):
        plant_index[plant.rows] = k

    def text(values):
        return np.array([("" if value is None else str(value))
                         for value in values], dtype=object)

    return {
        "turbine_id": text(r.row_id for r in rows),
        "site_id": text(r.site_id for r in rows),
        "asset_id": text(r.asset_id for r in rows),
        "plant_index": plant_index,
        "count": np.array([r.count for r in rows], dtype=np.int32),
        "turbine_lat": np.array([r.lat for r in rows]),
        "turbine_lon": np.array([r.lon for r in rows]),
        "rated_mw": np.array([r.rated_kw / 1000.0 for r in rows]),
        "rotor_diameter_m": np.array([r.rotor_diameter_m for r in rows]),
        "hub_height_m": np.array([r.hub_height_m for r in rows]),
        "curve_id": text(r.match.curve.curve_id if r.match.curve else ""
                         for r in rows),
        "curve_source": text(r.match.curve.curve_source if r.match.curve
                             else "" for r in rows),
        "match_quality": text(r.match.quality for r in rows),
        "match_score": np.array([r.match.score for r in rows]),
        "speed_method": text(r.speed_method
                             for r in rows),
    }


def write_power(result: PowerResult, path: str | Path) -> Path:
    """Write ``power.v1`` atomically (``.partial`` then rename)."""

    import netCDF4

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        with netCDF4.Dataset(str(tmp), "w", format="NETCDF4") as ds:
            ds.createDimension("time", len(result.times))
            ds.createDimension("plant", len(result.plants))
            ds.createDimension("turbine", len(result.rows))
            tattrs = dict(result.time_var)
            values = tattrs.pop("_values")
            dtype = tattrs.pop("_dtype")
            var = ds.createVariable("time", dtype, ("time",))
            var.setncatts(tattrs)
            var[:] = values
            coords = {**_plant_coords(result), **_turbine_coords(result)}
            for name, data in coords.items():
                dims, units, long_name = POWER_COORDINATES[name]
                if data.dtype == object:
                    var = ds.createVariable(name, str, dims)
                    var[:] = data
                elif data.dtype.kind in "iu":
                    var = ds.createVariable(name, data.dtype.str[1:], dims)
                    var[:] = data
                else:
                    var = ds.createVariable(name, "f8", dims,
                                            fill_value=np.nan)
                    var[:] = data
                var.long_name = long_name
                if units:
                    var.units = units
            tables = {**POWER_VARIABLES, **TURBINE_VARIABLES}
            for name, data in {**result.plant_vars,
                               **result.turbine_vars}.items():
                dims, units, long_name = tables[name]
                if data.dtype == np.int8:
                    var = ds.createVariable(name, "i1", dims, zlib=True)
                else:
                    var = ds.createVariable(name, "f8", dims,
                                            fill_value=np.nan, zlib=True)
                var[:] = data
                var.units = units
                var.long_name = long_name
            ds.setncatts({key: value for key, value in result.attrs.items()
                          if value is not None})
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


@dataclass
class PowerDocument:
    path: Path
    attrs: dict[str, Any]
    sizes: dict[str, int]
    data: dict[str, np.ndarray]


def load_power(path: str | Path) -> PowerDocument:
    """Read and check a ``woof-energy.power.v1`` netCDF."""

    import netCDF4

    path = Path(path)
    if not path.is_file():
        raise PowerRefused(f"power file {path} does not exist")
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        if getattr(ds, "schema", None) != POWER_SCHEMA:
            raise PowerRefused(f"{path} is not a {POWER_SCHEMA} file")
        sizes = {name: len(dim) for name, dim in ds.dimensions.items()}
        for dim in ("time", "plant", "turbine"):
            if dim not in sizes:
                raise PowerRefused(f"{path} has no {dim!r} dimension")
        data: dict[str, np.ndarray] = {}
        required = {**POWER_COORDINATES, **POWER_VARIABLES}
        optional = TURBINE_VARIABLES
        for name, (dims, units, _) in {**required, **optional}.items():
            if name not in ds.variables:
                if name in required:
                    raise PowerRefused(f"{path} lacks {name!r}")
                continue
            var = ds.variables[name]
            if tuple(var.dimensions) != tuple(dims):
                raise PowerRefused(f"{path}: {name} has dims "
                                   f"{var.dimensions}, expected {dims}")
            if units and name != "time" and getattr(var, "units", None) != \
                    units:
                raise PowerRefused(f"{path}: {name} units "
                                   f"{getattr(var, 'units', None)!r}, "
                                   f"expected {units!r}")
            data[name] = np.asarray(var[:])
        attrs = {key: ds.getncattr(key) for key in ds.ncattrs()}
    index = data["plant_index"]
    if index.size and (index.min() < 0 or index.max() >= sizes["plant"]):
        raise PowerRefused(f"{path}: plant_index out of range")
    quality = set(np.asarray(data["match_quality"]).astype(str))
    from woof.energy.turbine_library import MATCH_QUALITIES

    if not quality <= set(MATCH_QUALITIES):
        raise PowerRefused(f"{path}: unknown match_quality "
                           f"{sorted(quality - set(MATCH_QUALITIES))}")
    return PowerDocument(path=path, attrs=attrs, sizes=sizes, data=data)


def matches_document(result: PowerResult, library: TurbineLibrary
                     ) -> dict[str, Any]:
    plant_ids = {i: plant.plant_id for plant in result.plants
                 for i in plant.rows}
    records = []
    for i, row in enumerate(result.rows):
        record = {
            "turbine_id": row.row_id, "site_id": row.site_id,
            "asset_id": row.asset_id, "plant_id": plant_ids[i],
            "count": row.count,
            "manufacturer": row.spec.manufacturer, "model": row.spec.model,
            "asset_rated_kw": row.spec.rated_kw,
            "asset_rotor_diameter_m": row.spec.rotor_diameter_m,
            "hub_height_m": (None if not math.isfinite(row.hub_height_m)
                             else row.hub_height_m),
            "speed_method": row.speed_method,
        }
        record.update(row.match.as_record())
        records.append(record)
    return {"schema": MATCHES_SCHEMA,
            "library": {"path": library.path.name, "sha256": library.sha256,
                        "license": library.license,
                        "attribution": library.attribution},
            "count": len(records), "turbines": records}


def load_matches(path: str | Path) -> dict[str, Any]:
    """Read and check a ``woof-energy.turbine-matches.v1`` document."""

    from woof.energy.turbine_library import MATCH_QUALITIES

    try:
        document = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PowerRefused(f"matches {path} cannot be read: {error}") \
            from error
    if document.get("schema") != MATCHES_SCHEMA:
        raise PowerRefused(f"{path} is not a {MATCHES_SCHEMA} document")
    rows = document.get("turbines")
    if not isinstance(rows, list) or document.get("count") != len(rows):
        raise PowerRefused(f"{path}: turbines/count disagree")
    for row in rows:
        if row.get("match_quality") not in MATCH_QUALITIES:
            raise PowerRefused(f"{path}: {row.get('turbine_id')!r} has "
                               f"match_quality {row.get('match_quality')!r}")
    return document


# --------------------------------------------------------------------------
# CLI


def _read_site_field(path: Path, names: Sequence[str], what: str,
                     forecast) -> tuple[str, np.ndarray]:
    """A (time, site) field from a netCDF on the forecast's axes."""

    import netCDF4

    from woof.energy.products import _strings

    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        name = next((n for n in names if n in ds.variables), None)
        if name is None:
            raise PowerRefused(f"{path} has no {what} variable (looked for "
                               f"{', '.join(names)})")
        var = ds.variables[name]
        dims = tuple(var.dimensions)
        if dims != ("time", "site"):
            raise PowerRefused(f"{path}: {name} has dims {dims}; expected "
                               "(time, site)")
        data = np.asarray(var[:], dtype=float)
        fill = getattr(var, "_FillValue", None)
        if fill is not None:
            data = np.where(data == fill, np.nan, data)
        if "site_id" in ds.variables:
            ids = _strings(ds.variables["site_id"]).astype(str)
            if list(ids) != list(np.asarray(forecast.coord("site_id"))
                                 .astype(str)):
                raise PowerRefused(f"{path}: site_id differs from the "
                                   "forecast's")
    if data.shape[0] != len(forecast.times) or \
            data.shape[1] != forecast.n_site:
        raise PowerRefused(f"{path}: {name} is {data.shape[:2]}, the "
                           f"forecast is ({len(forecast.times)}, "
                           f"{forecast.n_site})")
    return name, data


def _summary(result: PowerResult, output: Path, matches_path: Path | None
             ) -> dict[str, Any]:
    qualities: dict[str, int] = {}
    for row in result.rows:
        qualities[row.match.quality] = qualities.get(row.match.quality, 0) \
            + row.count
    seconds = np.array([(t - result.times[0]).total_seconds()
                        for t in result.times])
    dt_h = np.diff(seconds) / 3600.0
    step_h = np.append(dt_h, np.median(dt_h) if dt_h.size else 0.0)
    plants = []
    coords = _plant_coords(result)
    for k, plant in enumerate(result.plants):
        power = result.plant_vars["power_mw"][:, k]
        gross = result.plant_vars["gross_power_mw"][:, k]
        wake = result.plant_vars["wake_loss_mw"][:, k]
        finite = np.isfinite(power)
        plants.append({
            "plant_id": plant.plant_id, "name": plant.name,
            "capacity_mw": round(float(coords["capacity_mw"][k]), 3),
            "n_turbines": int(coords["n_turbines"][k]),
            "wake_method": plant.wake_method,
            "mean_power_mw": (round(float(power[finite].mean()), 4)
                              if finite.any() else None),
            "mean_capacity_factor": (
                round(float(np.nanmean(result.plant_vars["capacity_factor"]
                                       [:, k])), 4) if finite.any()
                else None),
            "energy_mwh": round(float(np.nansum(power * step_h)), 3),
            "wake_loss_fraction": (
                round(float(np.nansum(wake) / np.nansum(gross)), 4)
                if np.nansum(gross) > 0 else None),
        })
    return {
        "schema": POWER_SCHEMA, "output": str(output),
        "matches": str(matches_path) if matches_path else None,
        "times": len(result.times),
        "time_step_s": (float(np.median(np.diff(seconds)))
                        if seconds.size > 1 else None),
        "plants": len(result.plants), "turbine_rows": len(result.rows),
        "turbines": int(sum(r.count for r in result.rows)),
        "match_quality": dict(sorted(qualities.items())),
        "per_plant": plants,
        "notes": list(result.notes),
    }


def run(forecast_path: Path, *, output: Path, assets_path: Path | None = None,
        library_path: Path | None = None, options: PowerOptions | None = None,
        icing_path: str | None = None, gust_var: str | None = None,
        matches_path: Path | None = None) -> tuple[PowerResult, dict]:
    from woof.energy.products import read_forecast, read_forecast_header

    options = options or PowerOptions()
    forecast_path = Path(forecast_path)
    output = Path(output)
    if output.resolve() == forecast_path.resolve():
        raise PowerRefused("the output would overwrite the forecast")
    header = read_forecast_header(forecast_path)
    if "wind_speed" not in header.variables:
        raise PowerRefused(f"forecast {forecast_path} lacks wind_speed")
    if "air_density" not in header.variables and not {
            "air_pressure", "air_temperature"} <= header.variables:
        raise PowerRefused(f"forecast {forecast_path} lacks air_density (or "
                           "air_pressure and air_temperature)")
    needed = [name for name in ("wind_speed", "air_density", "air_pressure",
                                "air_temperature", "u", "v",
                                "wind_from_direction")
              if name in header.variables]
    if "air_density" in needed:
        needed = [n for n in needed if n not in ("air_pressure",
                                                 "air_temperature")]
    forecast = read_forecast(forecast_path, needed=needed)
    assets = load_assets(assets_path) if assets_path else None
    library = load_turbine_library(library_path)
    notes = _Notes()
    gust = None
    gust_name = None
    if options.gust_shutdown:
        names = (gust_var,) if gust_var else GUST_VARIABLES
        gust_name, gust = _read_site_field(forecast_path, names, "gust",
                                           forecast)
        notes.add(f"gust shutdown from forecast variable {gust_name!r} "
                  "(used as given, not adjusted to hub height)")
    icing = None
    icing_ref = None
    if icing_path:
        if Path(icing_path).is_file() or ":" not in icing_path:
            path_text, var = icing_path, ""
        else:
            path_text, _, var = icing_path.rpartition(":")
        icing_file = Path(path_text)
        if not icing_file.is_file():
            raise PowerRefused(f"icing loss file {icing_file} does not exist")
        name, icing = _read_site_field(icing_file,
                                       (var or ICING_VARIABLE,),
                                       "icing loss", forecast)
        finite = icing[np.isfinite(icing)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            raise PowerRefused(f"{icing_file}: {name} must lie in [0, 1]")
        icing_ref = f"{icing_file.name}:{name}"
    result = compute_power(forecast, assets, library, options, gust=gust,
                           icing_loss=icing, notes=notes)
    result.attrs = {
        "schema": POWER_SCHEMA,
        "title": "woof energy power",
        "forecast": forecast_path.name,
        "forecast_sha256": sha256_file(forecast_path),
        "assets": assets_path.name if assets_path else "",
        "assets_sha256": sha256_file(assets_path) if assets_path else "",
        "turbine_library": library.path.name,
        "turbine_library_sha256": library.sha256,
        "turbine_library_license": library.license,
        "turbine_library_attribution": library.attribution,
        "wake_model": options.wake_model,
        "rews": int(options.rews),
        "availability": options.availability,
        "electrical_losses": options.electrical_losses,
        "gust_shutdown": gust_name or "",
        "icing_loss": icing_ref or "",
        "reference_density_kg_m3": REFERENCE_DENSITY,
        "notes": "\n".join(result.notes),
        "created": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    for key in ("plan_sha256", "sites_sha256"):
        if key in forecast.attrs:
            result.attrs[key] = forecast.attrs[key]
    write_power(result, output)
    if matches_path:
        from woof.energy.contracts import _write_json

        _write_json(matches_document(result, library), matches_path)
    return result, _summary(result, output, matches_path)


def main(args) -> int:
    try:
        options = PowerOptions(
            wake_model=args.wake_model, rews=not args.hub_only,
            availability=args.availability,
            electrical_losses=args.electrical_losses,
            gust_shutdown=args.gust_shutdown or bool(args.gust_var),
            per_turbine=args.per_turbine)
        _, summary = run(
            Path(args.forecast), output=Path(args.output),
            assets_path=Path(args.assets) if args.assets else None,
            library_path=Path(args.library) if args.library else None,
            options=options, icing_path=args.icing_loss,
            gust_var=args.gust_var,
            matches_path=Path(args.matches) if args.matches else None)
    except ContractError as error:
        print(json.dumps({"schema": POWER_SCHEMA, "refused": str(error)},
                         indent=2))
        return 2
    for note in summary["notes"]:
        print(f"woof energy power: {note}", file=sys.stderr)
    print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = [
    "POWER_SCHEMA", "MATCHES_SCHEMA", "WAKE_MODELS", "POWER_VARIABLES",
    "TURBINE_VARIABLES", "POWER_COORDINATES", "PowerRefused",
    "PowerOptions", "PowerResult", "PowerDocument",
    "rotor_equivalent_wind_speed", "density_corrected_speed",
    "running_state", "fallback_wake_loss", "wake_ratio_table",
    "apply_wake_table", "build_rows", "compute_power", "write_power",
    "load_power", "matches_document", "load_matches", "run", "main",
]
