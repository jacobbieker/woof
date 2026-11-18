"""Storm-time correction of the L3a ionosphere: woof's own empirical fit.

The CCIR/URSI maps (woof.spacewx.ionosphere) are a monthly climatology driven
by F10.7, so they have no geomagnetic storms.  During a storm the F2 layer's
density rises (a positive phase, mostly at low and middle latitudes and early
in the storm) or falls (a negative phase, as composition changes spread from
the auroral zone, strongest in summer and at night-to-morning local times).

The correction is fitted here, the way STORM (Araujo-Pradere et al. 2002)
was, but on open data, and nothing of STORM is used:

* the driver is the filtered ap of Wrenn (1987),
  ``ap_tau(t) = (1 - tau) * sum_i tau^i ap(t_i)``, over the 3-hour ap
  intervals that have ENDED by t (i = 0 the latest), so that it never reads
  the future;
* the target is ``log(TEC / TEC_q)`` at the nodes of CODE's final global
  ionosphere maps (IONEX, AIUB/University of Bern), where TEC_q is the median
  of the 27 days before at the same UT and node: the storm's departure from
  the recent quiet ionosphere, independent of the climatology's own bias;
* the model is ``log R = a x + b x^2`` with ``x = ap_tau / 100``, without an
  intercept so that a quiet day gets R near 1, and with (a, b) interpolated
  linearly between nodes of geomagnetic latitude |MLAT| (IGRF dipole), local
  solar time (periodic) and season (+1 local summer, 0 equinox, -1 local
  winter);
* R scales NmF2, which is the TEC's scale for an unchanged slab, so foF2 is
  multiplied by sqrt(R).  hmF2 then follows the CCIR relations as usual.

``tools/spacewx_storm_fit.py`` fits it on training years and grades it on
others; the coefficients and their record are in
``woof/data/spacewx/storm_fit_v1.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from woof.spacewx.indices import naive_utc

FIT_PATH = Path(__file__).resolve().parent.parent / "data" / "spacewx" / "storm_fit_v1.json"
#: Hours per ap interval.
AP_HOURS = 3
#: The latest ended intervals that must have ap: a storm's onset is in them.
LATEST_REQUIRED = 2
#: Within this magnetic latitude, deg, of the dip equator the season coordinate
#: goes linearly to 0, so the two hemispheres' seasons meet without a jump.
SEASON_TAPER_DEG = 10.0


class StormError(ValueError):
    pass


@dataclass(frozen=True)
class StormFit:
    tau: float                       # Wrenn filter constant per 3-hour interval
    nlag: int                        # intervals summed
    mlat_nodes: tuple                # |MLAT| nodes, deg
    lt_nodes: tuple                  # local-time nodes, h (periodic in 24 h)
    season_nodes: tuple              # -1 local winter, 0 equinox, +1 local summer
    coef: np.ndarray                 # (n_mlat, n_lt, n_season, 2): a, b
    x_max: float                     # largest x of the training data; x is held below it
    ratio_range: tuple               # (min, max) of R
    sha256: str
    record: dict


def _shape(fit_or_nodes) -> tuple[int, int, int]:
    return len(fit_or_nodes[0]), len(fit_or_nodes[1]), len(fit_or_nodes[2])


def load_fit(path=None) -> StormFit:
    """The shipped fit, or a file the fitting tool wrote."""
    return _load(str(Path(path) if path else FIT_PATH))


def _increasing(nodes) -> bool:
    return len(nodes) >= 2 and bool(np.all(np.diff(np.asarray(nodes, dtype=np.float64)) > 0))


def _load(path: str) -> StormFit:
    raw = Path(path).read_bytes()
    try:
        doc = json.loads(raw)
        nodes = (tuple(doc["mlat_nodes"]), tuple(doc["lt_nodes"]), tuple(doc["season_nodes"]))
        coef = np.asarray(doc["coef"], dtype=np.float64)
        if coef.shape != (*_shape(nodes), 2) or not np.all(np.isfinite(coef)):
            raise StormError(f"coefficients of shape {coef.shape}, expected {(*_shape(nodes), 2)}")
        tau, nlag, x_max = float(doc["tau"]), int(doc["nlag"]), float(doc["x_max"])
        lo, hi = (float(v) for v in doc["ratio_range"])
        if not (0.0 < tau < 1.0 and nlag >= LATEST_REQUIRED and x_max > 0.0 and 0.0 < lo <= 1.0 <= hi):
            raise StormError(f"tau {tau}, nlag {nlag}, x_max {x_max} or ratio_range {(lo, hi)} out of range")
        if not all(_increasing(n) for n in nodes):
            raise StormError("node lists must be increasing and hold at least two nodes each")
        coef.setflags(write=False)
        return StormFit(tau, nlag, *nodes, coef, x_max, (lo, hi), hashlib.sha256(raw).hexdigest(),
                        dict(doc.get("record", {})))
    except (KeyError, TypeError, ValueError) as exc:
        raise StormError(f"{path}: not a storm-time fit ({exc})") from None


# ------------------------------------------------------------------ the driver

def last_ended_interval(when: datetime) -> datetime:
    """The start of the latest 3-hour ap interval that has ended by ``when`` (naive UTC)."""
    when = naive_utc(when)
    now = when.replace(minute=0, second=0, microsecond=0)
    now -= timedelta(hours=now.hour % AP_HOURS)            # the interval in progress starts here
    return now - timedelta(hours=AP_HOURS)


def _ap_series(geo, when: datetime, nlag: int) -> np.ndarray:
    """ap of the ``nlag`` 3-hour intervals that have ended by ``when``, latest
    first (NaN where GFZ has none)."""
    last = last_ended_interval(when)
    out = np.full(nlag, np.nan)
    for i in range(nlag):
        start = last - timedelta(hours=AP_HOURS * i)
        try:
            row = geo._row(start.date())
        except (KeyError, ValueError, IndexError):
            continue
        out[i] = geo.ap[row, start.hour // AP_HOURS]
    return out


def filtered_ap(geo, when: datetime, tau: float, nlag: int, min_weight: float = 0.8) -> float:
    """Wrenn's ap_tau at ``when`` from the intervals that have ended by then.
    The latest :data:`LATEST_REQUIRED` intervals must be present (a storm's
    onset is there); an older missing interval is left out and the weights
    renormalised, and fewer than ``min_weight`` of the weight present is an
    error."""
    ap = _ap_series(geo, when, nlag)
    w = (1.0 - tau) * tau ** np.arange(nlag)
    good = np.isfinite(ap)
    if not good[:LATEST_REQUIRED].all():
        raise StormError(f"GFZ has no ap yet for the 3-hour intervals that end {AP_HOURS * LATEST_REQUIRED} h "
                         f"before {naive_utc(when):%Y-%m-%dT%H:%MZ}; the storm correction needs them")
    if w[good].sum() < min_weight * w.sum():
        raise StormError(f"GFZ has too little ap in the {nlag * AP_HOURS} h before {when:%Y-%m-%dT%H:%MZ}")
    return float(np.dot(w[good], ap[good]) / w[good].sum())


# ------------------------------------------------------------------ the geometry

def magnetic_latitude(lat_deg, lon_deg, when: datetime, model=None) -> np.ndarray:
    """Dipole geomagnetic latitude, deg, of geodetic points (the IGRF-14 dipole at ``when``)."""
    from woof.spacewx import aurora, igrf
    return aurora.dipole_latitude(aurora.dipole_frame(model or igrf.load(), when), lat_deg, lon_deg)


def local_time(lon_deg, when: datetime) -> np.ndarray:
    """Local mean solar time, h."""
    when = naive_utc(when)
    ut = when.hour + when.minute / 60.0 + when.second / 3600.0
    return np.mod(ut + np.asarray(lon_deg, dtype=np.float64) / 15.0, 24.0)


def season_coordinate(when: datetime, mlat_deg) -> np.ndarray:
    """+1 at the solstice of the magnetic hemisphere's summer, -1 at its winter
    one, 0 at the equinoxes; tapered linearly to 0 within
    :data:`SEASON_TAPER_DEG` of the dip equator, so that it is continuous there."""
    when = naive_utc(when)
    doy = when.timetuple().tm_yday + when.hour / 24.0
    north = np.cos(2.0 * np.pi * (doy - 172.0) / 365.25)
    return north * np.clip(np.asarray(mlat_deg, dtype=np.float64) / SEASON_TAPER_DEG, -1.0, 1.0)


def _hats(x, nodes, period=None) -> np.ndarray:
    """(n, k) linear-interpolation weights of ``x`` on ``nodes`` (clamped at the
    ends, or periodic with ``period``)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nodes = np.asarray(nodes, dtype=np.float64)
    k = nodes.size
    w = np.zeros((x.size, k))
    if period is not None:
        ext = np.concatenate([nodes, [nodes[0] + period]])
        xm = np.mod(x - nodes[0], period) + nodes[0]
        i = np.clip(np.searchsorted(ext, xm, side="right") - 1, 0, k - 1)
        f = (xm - ext[i]) / (ext[i + 1] - ext[i])
        rows = np.arange(x.size)
        w[rows, i] += 1.0 - f
        w[rows, (i + 1) % k] += f
        return w
    xc = np.clip(x, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, xc, side="right") - 1, 0, k - 2)
    f = (xc - nodes[i]) / (nodes[i + 1] - nodes[i])
    rows = np.arange(x.size)
    w[rows, i] = 1.0 - f
    w[rows, i + 1] = f
    return w


def basis(nodes, mlat_deg, lt_h, season) -> np.ndarray:
    """(n, n_mlat * n_lt * n_season) interpolation weights of each point."""
    wm = _hats(np.abs(mlat_deg), nodes[0])
    wl = _hats(lt_h, nodes[1], period=24.0)
    ws = _hats(season, nodes[2])
    return np.einsum("ni,nj,nk->nijk", wm, wl, ws).reshape(wm.shape[0], -1)


def log_ratio(fit: StormFit, mlat_deg, lt_h, season, ap_tau) -> np.ndarray:
    """log(R) at points; ``ap_tau`` a scalar or one per point."""
    x = np.minimum(np.asarray(ap_tau, dtype=np.float64) / 100.0, fit.x_max)
    b = basis((fit.mlat_nodes, fit.lt_nodes, fit.season_nodes), mlat_deg, lt_h, season)
    c = fit.coef.reshape(-1, 2)
    return (b @ c[:, 0]) * x + (b @ c[:, 1]) * x * x


def nmf2_factor(fit: StormFit, mlat_deg, lt_h, season, ap_tau) -> np.ndarray:
    """R = NmF2(storm) / NmF2(quiet), clipped to the fit's range."""
    lo, hi = fit.ratio_range
    return np.clip(np.exp(log_ratio(fit, mlat_deg, lt_h, season, ap_tau)), lo, hi)


def fof2_scale(fit: StormFit, lat_deg, lon_deg, when: datetime, ap_tau: float, *, mlat_deg=None,
               model=None) -> np.ndarray:
    """The factor on foF2, sqrt(R), at geodetic columns and ``when``."""
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    mlat = magnetic_latitude(lat, lon, when, model) if mlat_deg is None else np.asarray(mlat_deg).ravel()
    return np.sqrt(nmf2_factor(fit, mlat, local_time(lon, when), season_coordinate(when, mlat), ap_tau))


def receipt(fit: StormFit, source: str) -> dict:
    """The fit and the ap record; each time's ap_tau goes with that time."""
    return {"spacewx_storm_model": "woof storm-time fit v1 (CODE GIM TEC, Wrenn filtered ap)",
            "spacewx_storm_fit_sha256": fit.sha256, "spacewx_storm_tau": fit.tau,
            "spacewx_storm_ap_source": source}


__all__ = ["FIT_PATH", "StormError", "StormFit", "load_fit", "last_ended_interval", "filtered_ap", "magnetic_latitude", "local_time",
           "season_coordinate", "basis", "log_ratio", "nmf2_factor", "fof2_scale", "receipt"]
