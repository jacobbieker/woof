"""The L3a ionosphere in a forecast's history (``spacewx_ionosphere_opt = 1``).

A diagnostic layer above the model top.  It reads nothing the forecast
integrates and writes nothing back, so a run with the option on integrates the
same atmosphere and adds five surface rows to each wrfout frame: the vertical
TEC and the F2/E-layer parameters of the climatological ionosphere
(woof.spacewx.ionosphere) over the domain's columns at the frame's valid time.

The solar driver is fixed for the run: the F10.7 the run's START would use
(``spacewx_f107_mode``: the previous UT day's observed flux, or the trailing 81
days ending there), taken from the GFZ index file (fetched, CC BY 4.0), a local
file in that format or a number.  That is persistence, which is the only
forecast of F10.7 a run can use without a flux forecast; it holds the profile
on the measured solar activity rather than letting a multi-day forecast run
off the end of the observed record.  The magnetic coordinates (IGRF-14) are
also taken at the start.  The UT, season, monthly-map blend and solar zenith
angle move with each frame.

The domain's columns are the radiation geometry's latitude and longitude, so a
run needs a radiation scheme on.  Hex meshes and streamed (tiled) domains
refuse the option for now; each says so by name.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from woof.spacewx.solar_irradiance import _naive_utc

#: wrfout name -> (ionosphere output, description, units).
HISTORY_FIELDS = {
    "IONO_TEC": ("tec", "VERTICAL TOTAL ELECTRON CONTENT, 60-2000 KM (CLIMATOLOGICAL IONOSPHERE)", "TECU"),
    "IONO_FOF2": ("fof2", "F2-LAYER CRITICAL FREQUENCY (CLIMATOLOGICAL IONOSPHERE)", "MHz"),
    "IONO_HMF2": ("hmf2", "F2-LAYER PEAK HEIGHT (CLIMATOLOGICAL IONOSPHERE)", "km"),
    "IONO_NMF2": ("nmf2", "F2-LAYER PEAK ELECTRON DENSITY (CLIMATOLOGICAL IONOSPHERE)", "m-3"),
    "IONO_FOE": ("foe", "E-LAYER CRITICAL FREQUENCY (CLIMATOLOGICAL IONOSPHERE)", "MHz"),
}


class IonosphereHistoryError(ValueError):
    pass


def enabled(cfg) -> bool:
    return bool(int(getattr(cfg, "spacewx_ionosphere_opt", 0) or 0))


def history_shapes(cfg) -> dict[str, tuple[int, int]]:
    """wrfout name -> shape of the rows the option adds (empty when off)."""
    if not enabled(cfg):
        return {}
    return {name: (int(cfg.ny), int(cfg.nx)) for name in HISTORY_FIELDS}


def device_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The float32 device arrays a domain holds while the option is on: the
    four column-geometry vectors, every kernel output and the five history
    rows (preflight's memory inventory)."""
    if not enabled(cfg):
        return {}
    from woof.spacewx.ionosphere import OUTPUTS
    ncol = int(cfg.ny) * int(cfg.nx)
    shapes = {f"spacewx_ionosphere/{name}": (ncol,)
              for name in ("lat", "lon", "modip", "dip_latitude", *OUTPUTS)}
    shapes["spacewx_ionosphere/history"] = (len(HISTORY_FIELDS), int(cfg.ny), int(cfg.nx))
    return shapes


def parse_f107_source(source: str):
    """``("gfz", None)``, ``("value", sfu)`` or ``("file", path)``."""
    text = str(source or "").strip()
    if text in ("", "gfz"):
        return "gfz", None
    try:
        value = float(text)
    except ValueError:
        return "file", text
    if not np.isfinite(value) or value <= 0.0:
        raise IonosphereHistoryError(f"spacewx_f107_source={source!r} is not a solar flux in sfu")
    return "value", value


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array, dtype=np.float64)


class IonosphereHistory:
    """The option's state on one domain: rebuilt from the RunConfig, stateless."""

    @classmethod
    def for_run(cls, cfg, radiation, *, indices_loader=None, igrf_model=None, map_loader=None):
        if not enabled(cfg):
            return None
        start = getattr(radiation, "start_time", None)
        lat = getattr(radiation, "latitude_deg", None)
        lon = getattr(radiation, "longitude_deg", None)
        if radiation is None or start is None or lat is None or lon is None:
            raise IonosphereHistoryError(
                "spacewx_ionosphere_opt = 1 places the ionosphere on the radiation "
                "geometry's columns, and this domain's radiation publishes no start "
                "time or latitude/longitude")
        start = _naive_utc(start)
        kind, value = parse_f107_source(getattr(cfg, "spacewx_f107_source", ""))
        mode = getattr(cfg, "spacewx_f107_mode", "81day")
        if kind == "value":
            f107, record = value, {"spacewx_f107_source": "value"}
        else:
            from woof.spacewx import indices
            load = indices_loader or indices.load
            # F10.7 reads through the previous UTC day, so a cache through it is enough.
            geo = load("gfz" if kind == "gfz" else value, start - timedelta(days=1))
            try:
                f107 = geo.f107_driver(start, mode)
            except indices.IndicesError as first:
                if kind != "gfz":
                    raise IonosphereHistoryError(str(first)) from None
                # The cached file may carry the previous day's row before its
                # flux was released (-1): fetch again if the file does not
                # reach the start's own day, then refuse by name.
                geo = load("gfz", start)
                try:
                    f107 = geo.f107_driver(start, mode)
                except indices.IndicesError as exc:
                    raise IonosphereHistoryError(
                        f"spacewx_ionosphere_opt = 1: {exc}; GFZ has not released the flux "
                        "this start needs, so set spacewx_f107_source to a number (sfu) "
                        "or use spacewx_f107_mode = '81day'") from None
            record = {"spacewx_f107_source": geo.source, "spacewx_f107_sha256": geo.sha256}
        from woof.spacewx.ionosphere import Ionosphere, month_weights
        lat, lon = _host(lat), _host(lon)
        iono = Ionosphere(lat, lon, when=start, family=getattr(cfg, "spacewx_ionosphere_maps", "ccir"),
                          igrf_model=igrf_model, map_loader=map_loader)
        # Load every monthly map the run's frames blend now, so a run crossing
        # a mid-month boundary does not fetch one in the middle of the forecast.
        end = start + timedelta(seconds=float(getattr(cfg, "run_seconds", 0.0) or 0.0))
        when = start
        while True:
            for month, _weight in month_weights(min(when, end)):
                iono.maps(month)
            if when >= end:
                break
            when += timedelta(days=1)
        return cls(iono, start, f107, mode, lat.shape, record)

    def __init__(self, iono, start_time: datetime, f107: float, mode: str, shape, record: dict):
        self.ionosphere = iono
        self.start_time = start_time
        self.f107 = float(f107)
        self.f107_mode = mode
        self.shape = tuple(int(n) for n in shape)
        self.record = record
        self._rows = None
        self._elapsed = None

    def valid_time(self, elapsed_seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=float(elapsed_seconds))

    def receipt(self) -> dict:
        from woof.spacewx.ionosphere import receipt
        out = receipt(self.ionosphere, self.record["spacewx_f107_source"], self.f107_mode)
        out.update(self.record, spacewx_f107_sfu=self.f107)
        return out

    def history_fields(self, elapsed_seconds: float) -> dict:
        """wrfout name -> (ny, nx) float32 device row at the frame's valid time.

        The rows are one stable allocation, refilled when the valid time moves
        (a frame's resident and streamed views read the same arrays)."""
        import cupy as cp
        if self._rows is None:
            rows = cp.zeros((len(HISTORY_FIELDS), *self.shape), dtype=cp.float32)
            self._rows = dict(zip(HISTORY_FIELDS, (rows[i] for i in range(len(HISTORY_FIELDS)))))
        if self._elapsed != float(elapsed_seconds):
            out = self.ionosphere.evaluate(self.valid_time(elapsed_seconds), self.f107)
            for name, (key, _desc, _units) in HISTORY_FIELDS.items():
                self._rows[name][...] = out[key].reshape(self.shape)
            self._elapsed = float(elapsed_seconds)
        return dict(self._rows)


__all__ = ["HISTORY_FIELDS", "IonosphereHistory", "IonosphereHistoryError", "enabled",
           "history_shapes", "device_shapes", "parse_f107_source"]
