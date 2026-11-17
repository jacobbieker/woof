"""The L3b auroral oval in a forecast's history (``spacewx_aurora_opt = 1``).

A diagnostic layer above the model top, like the ionosphere rows
(woof.spacewx.history).  It reads nothing the forecast integrates and writes
nothing back, so a run with the option on integrates the same atmosphere and
adds four surface rows to each wrfout frame: the auroral zone, the distance
from the diffuse aurora's equatorward edge (the visibility line), the
geomagnetic latitude and the magnetic local time (woof.spacewx.aurora) over the
domain's columns at the frame's valid time.

The activity is held for the run at the value the START sees, which is
persistence: a forecast has no observed Kp for its future.  ``spacewx_kp_source``
picks it:

* ``"gfz"`` (or ``""``): the latest observed Kp at or before the start in the
  GFZ index file (fetched, CC BY 4.0); a local file in GFZ's format also works;
* ``"omni"`` or ``"swpc"``: the Kp estimated from the solar wind
  (woof.spacewx.solarwind) for the latest 3-hour interval at or before the
  start, from NASA SPDF OMNI2 or NOAA SWPC's real-time feed;
* a number in [0, 9]: that Kp.

Kp becomes |AL| through Starkov's relation.  The magnetic frame (the IGRF-14
dipole) is taken at the start; the subsolar point, and so the MLT, moves with
each frame.  The columns are the radiation geometry's, so a run needs a
radiation scheme on.  Hex meshes and streamed (tiled) domains refuse the option
for now; each says so by name.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

#: wrfout name -> (aurora output, description, units).
HISTORY_FIELDS = {
    "AURORA_ZONE": ("zone", "AURORAL ZONE: 0 EQUATORWARD, 1 DIFFUSE, 2 OVAL, 3 POLAR CAP (STARKOV 1994)", "-"),
    "AURORA_DIST": ("eq_dist", "DEGREES POLEWARD OF THE DIFFUSE AURORA EQUATORWARD EDGE (STARKOV 1994)", "degree"),
    "MAG_LAT": ("mlat", "GEOMAGNETIC LATITUDE (IGRF-14 CENTRED DIPOLE)", "degree_north"),
    "MAG_LT": ("mlt", "MAGNETIC LOCAL TIME (IGRF-14 CENTRED DIPOLE)", "hour"),
}
#: How far back from the start an observed or estimated Kp is looked for.
LOOKBACK_HOURS = 24


class AuroraHistoryError(ValueError):
    pass


def enabled(cfg) -> bool:
    return bool(int(getattr(cfg, "spacewx_aurora_opt", 0) or 0))


def history_shapes(cfg) -> dict[str, tuple[int, int]]:
    """wrfout name -> shape of the rows the option adds (empty when off)."""
    if not enabled(cfg):
        return {}
    return {name: (int(cfg.ny), int(cfg.nx)) for name in HISTORY_FIELDS}


def device_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The float32 device arrays a domain holds while the option is on: the
    two column vectors, the frame, every kernel output and the four rows
    (preflight's memory inventory)."""
    if not enabled(cfg):
        return {}
    from woof.spacewx.aurora import OUTPUTS
    ncol = int(cfg.ny) * int(cfg.nx)
    shapes = {f"spacewx_aurora/{name}": (ncol,) for name in ("lat", "lon", *OUTPUTS)}
    shapes["spacewx_aurora/frame"] = (9,)
    shapes["spacewx_aurora/history"] = (len(HISTORY_FIELDS), int(cfg.ny), int(cfg.nx))
    return shapes


def parse_kp_source(source: str):
    """``("gfz", None)``, ``("omni", None)``, ``("swpc", None)``, ``("value", kp)``
    or ``("file", path)``."""
    text = str(source or "").strip()
    if text in ("", "gfz"):
        return "gfz", None
    if text in ("omni", "swpc"):
        return text, None
    try:
        value = float(text)
    except ValueError:
        return "file", text
    if not np.isfinite(value) or not 0.0 <= value <= 9.0:
        raise AuroraHistoryError(f"spacewx_kp_source={source!r} is not a Kp in [0, 9]")
    return "value", value


def _latest(lookup, start: datetime, what: str) -> tuple[float, datetime]:
    """The value of ``lookup`` for the latest 3-hour interval at or before ``start``."""
    when = start.replace(minute=0, second=0, microsecond=0)
    when -= timedelta(hours=when.hour % 3)
    first = when
    errors = []
    while start - when <= timedelta(hours=LOOKBACK_HOURS):
        try:
            return float(lookup(when)), when
        except ValueError as exc:
            errors.append(str(exc))
        when -= timedelta(hours=3)
    raise AuroraHistoryError(
        f"no {what} in the {LOOKBACK_HOURS} h before {start:%Y-%m-%dT%H:%MZ} "
        f"(latest interval tried {first:%Y-%m-%dT%H}Z: {errors[0] if errors else 'none'})")


def activity(source: str, start: datetime, *, indices_loader=None, solarwind_loader=None) -> dict:
    """The run's held Kp, its |AL| and the receipt of where it came from."""
    from woof.spacewx.aurora import kp_to_al

    kind, value = parse_kp_source(source)
    if kind == "value":
        kp, record = value, {"spacewx_kp_source": "value"}
    elif kind in ("omni", "swpc"):
        from woof.spacewx import solarwind
        load = solarwind_loader or solarwind.load
        wind = load(kind, start - timedelta(hours=LOOKBACK_HOURS + 2), start)
        kp, at = _latest(wind.kp_estimate, start, f"{kind} solar wind")
        record = {"spacewx_kp_source": wind.source, "spacewx_kp_sha256": wind.sha256,
                  "spacewx_kp_interval": f"{at:%Y-%m-%dT%H:%M}Z", "spacewx_kp_estimated": True}
    else:
        from woof.spacewx import indices
        load = indices_loader or indices.load
        # The start's own day: the latest Kp is the one that matters here.
        geo = load("gfz" if kind == "gfz" else value, start)
        kp, at = _latest(geo.kp_at, start, "observed Kp")
        record = {"spacewx_kp_source": geo.source, "spacewx_kp_sha256": geo.sha256,
                  "spacewx_kp_interval": f"{at:%Y-%m-%dT%H:%M}Z"}
    return {"kp": float(kp), "al": float(kp_to_al(kp)), "record": record}


class AuroraHistory:
    """The option's state on one domain: rebuilt from the RunConfig, stateless."""

    @classmethod
    def for_run(cls, cfg, radiation, *, indices_loader=None, solarwind_loader=None, igrf_model=None):
        if not enabled(cfg):
            return None
        from woof.spacewx.history import _host, _naive_utc
        start = getattr(radiation, "start_time", None)
        lat = getattr(radiation, "latitude_deg", None)
        lon = getattr(radiation, "longitude_deg", None)
        if radiation is None or start is None or lat is None or lon is None:
            raise AuroraHistoryError(
                "spacewx_aurora_opt = 1 places the auroral oval on the radiation "
                "geometry's columns, and this domain's radiation publishes no start "
                "time or latitude/longitude")
        start = _naive_utc(start)
        drive = activity(getattr(cfg, "spacewx_kp_source", ""), start,
                         indices_loader=indices_loader, solarwind_loader=solarwind_loader)
        from woof.spacewx.aurora import Aurora
        lat, lon = _host(lat), _host(lon)
        oval = Aurora(lat, lon, when=start, igrf_model=igrf_model)
        return cls(oval, start, drive, lat.shape)

    def __init__(self, oval, start_time: datetime, drive: dict, shape):
        self.aurora = oval
        self.start_time = start_time
        self.kp = drive["kp"]
        self.al = drive["al"]
        self.record = drive["record"]
        self.shape = tuple(int(n) for n in shape)
        self._rows = None
        self._elapsed = None

    def valid_time(self, elapsed_seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=float(elapsed_seconds))

    def receipt(self) -> dict:
        from woof.spacewx.aurora import receipt
        return receipt(self.aurora, {**self.record, "spacewx_kp": self.kp, "spacewx_al_nt": self.al})

    def history_fields(self, elapsed_seconds: float) -> dict:
        """wrfout name -> (ny, nx) float32 device row at the frame's valid time
        (one stable allocation, refilled when the valid time moves)."""
        import cupy as cp
        if self._rows is None:
            rows = cp.zeros((len(HISTORY_FIELDS), *self.shape), dtype=cp.float32)
            self._rows = dict(zip(HISTORY_FIELDS, (rows[i] for i in range(len(HISTORY_FIELDS)))))
        if self._elapsed != float(elapsed_seconds):
            out = self.aurora.evaluate(self.valid_time(elapsed_seconds), self.al)
            for name, (key, _desc, _units) in HISTORY_FIELDS.items():
                self._rows[name][...] = out[key].reshape(self.shape)
            self._elapsed = float(elapsed_seconds)
        return dict(self._rows)


__all__ = ["HISTORY_FIELDS", "AuroraHistory", "AuroraHistoryError", "activity", "enabled",
           "history_shapes", "device_shapes", "parse_kp_source"]
