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

``spacewx_ionosphere_opt = 2`` adds the storm-time correction
(woof.spacewx.storm, the fit ``woof spacewx ionosphere --storm`` uses): foF2
is multiplied by sqrt(R), R = NmF2(storm) / NmF2(quiet), from Wrenn's filtered
ap, and a sixth row, ``IONO_STORM``, carries R.  ap comes from the same GFZ
index file as F10.7 (so a numeric ``spacewx_f107_source`` is refused).  At
the start, ap_tau is the CLI's: the 3-hour intervals that have ended by then,
the latest two required.  A forecast has no ap for its future, so every
interval that ends after the start takes the latest observed ap
(persistence of the index, as the aurora rows hold the latest Kp) and the
filter runs on with each frame.  The dipole magnetic latitude is taken at the
start; local time and season move with each frame.  The ap series is held
per (source, start) in the space-weather cache, so a resume, and every domain
of a nested run, uses the ap the first read found.
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


#: The storm row of ``spacewx_ionosphere_opt = 2``: wrfout name -> (description, units).
STORM_HISTORY_FIELDS = {
    "IONO_STORM": ("STORM-TIME NMF2 RATIO R, FOF2 SCALED BY SQRT(R) (WOOF STORM FIT V1)", "-"),
}


class IonosphereHistoryError(ValueError):
    pass


def enabled(cfg) -> bool:
    return bool(int(getattr(cfg, "spacewx_ionosphere_opt", 0) or 0))


def storm_enabled(cfg) -> bool:
    return int(getattr(cfg, "spacewx_ionosphere_opt", 0) or 0) == 2


def row_names(cfg) -> tuple[str, ...]:
    """The wrfout names the option writes (none when off)."""
    if not enabled(cfg):
        return ()
    return tuple(HISTORY_FIELDS) + (tuple(STORM_HISTORY_FIELDS) if storm_enabled(cfg) else ())


def history_shapes(cfg) -> dict[str, tuple[int, int]]:
    """wrfout name -> shape of the rows the option adds (empty when off)."""
    return {name: (int(cfg.ny), int(cfg.nx)) for name in row_names(cfg)}


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
    if storm_enabled(cfg):
        shapes["spacewx_ionosphere/fo_scale"] = (ncol,)
    shapes["spacewx_ionosphere/history"] = (len(row_names(cfg)), int(cfg.ny), int(cfg.nx))
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


def _held_ap(load, name: str, start: datetime, nlag: int, memo) -> tuple:
    """(the index record, ap of the ``nlag`` intervals ended by the start,
    latest first; the receipt fields).  With ``memo`` (a directory) the series
    is kept per (source, start), so a resume and every nest reads the same ap."""
    import hashlib
    import json
    from pathlib import Path
    from woof.spacewx import storm
    memo_path = None
    if memo is not None:
        key = hashlib.sha256(f"{name}|{start.isoformat()}|{nlag}".encode()).hexdigest()[:24]
        memo_path = Path(memo) / f"{name}-{start:%Y%m%dT%H%M%S}-{key}.json"
        if memo_path.is_file():
            try:
                held = json.loads(memo_path.read_text(encoding="utf-8"))
                ap = np.array([np.nan if v is None else float(v) for v in held["ap"]], dtype=np.float64)
                if ap.size == nlag:
                    return ap, dict(held["record"], spacewx_storm_ap_held_from=str(memo_path))
            except (ValueError, KeyError, TypeError):
                pass                                   # damaged: read the source again
    # GFZ adds ap during the day: a cache without the latest ended interval
    # is fetched again, and kept if GFZ cannot be reached.
    try:
        geo = load(name, storm.last_ended_interval(start), kp=True)
    except OSError:
        geo = load(name, None)
    ap = storm.ap_series(geo, start, nlag)
    record = {"spacewx_storm_ap_source": geo.source, "spacewx_storm_ap_sha256": geo.sha256}
    if memo_path is not None:
        import os
        memo_path.parent.mkdir(parents=True, exist_ok=True)
        part = memo_path.with_name(f"{memo_path.name}.{os.getpid()}.part")
        doc = {"ap": [None if not np.isfinite(v) else float(v) for v in ap], "record": record}
        part.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
        part.replace(memo_path)
    return ap, record


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array, dtype=np.float64)


class IonosphereHistory:
    """The option's state on one domain: rebuilt from the RunConfig, stateless."""

    @classmethod
    def for_run(cls, cfg, radiation, *, indices_loader=None, igrf_model=None, map_loader=None,
                storm_fit=None):
        if not enabled(cfg):
            return None
        start = getattr(radiation, "start_time", None)
        lat = getattr(radiation, "latitude_deg", None)
        lon = getattr(radiation, "longitude_deg", None)
        if radiation is None or start is None or lat is None or lon is None:
            raise IonosphereHistoryError(
                f"spacewx_ionosphere_opt = {int(cfg.spacewx_ionosphere_opt)} places the ionosphere on the radiation "
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
        storm_state = None
        if storm_enabled(cfg):
            storm_state = cls._storm_for_run(cfg, kind, value, start, lat, lon, indices_loader=indices_loader,
                                             igrf_model=igrf_model, storm_fit=storm_fit)
            record = {**record, **storm_state["record"]}
        return cls(iono, start, f107, mode, lat.shape, record, storm=storm_state)

    @staticmethod
    def _storm_for_run(cfg, kind, value, start, lat, lon, *, indices_loader, igrf_model, storm_fit):
        from woof.spacewx import indices, storm
        if kind == "value":
            raise IonosphereHistoryError(
                "spacewx_ionosphere_opt = 2 reads ap from the GFZ index file, so "
                "spacewx_f107_source must be 'gfz' or a GFZ-format file, not a number")
        try:
            fit = storm_fit if storm_fit is not None else storm.load_fit()
        except (storm.StormError, OSError) as exc:
            raise IonosphereHistoryError(f"spacewx_ionosphere_opt = 2: the storm-time fit: {exc}") from None
        memo = None
        if indices_loader is None and kind == "gfz":
            from woof.spacewx.solar_irradiance import cache_root
            memo = cache_root() / "storm-activity"
        name = "gfz" if kind == "gfz" else value
        ap, record = _held_ap(indices_loader or indices.load, name, start, fit.nlag, memo)
        try:
            ap_tau = storm.persisted_ap_tau(ap, start, start, fit.tau)
        except storm.StormError as exc:
            raise IonosphereHistoryError(f"spacewx_ionosphere_opt = 2: {exc}") from None
        mlat = storm.magnetic_latitude(lat.ravel(), lon.ravel(), start, model=igrf_model)
        record = {**record, **storm.receipt(fit, record["spacewx_storm_ap_source"]),
                  "spacewx_storm_ap_tau_start": ap_tau,
                  "spacewx_storm_ap_latest": float(ap[0]),
                  "spacewx_storm_ap_interval": f"{storm.last_ended_interval(start):%Y-%m-%dT%H:%M}Z"}
        return {"fit": fit, "ap": ap, "mlat": mlat, "lat": lat.ravel(), "lon": lon.ravel(), "record": record}

    def __init__(self, iono, start_time: datetime, f107: float, mode: str, shape, record: dict, storm=None):
        self.ionosphere = iono
        self.start_time = start_time
        self.f107 = float(f107)
        self.f107_mode = mode
        self.shape = tuple(int(n) for n in shape)
        self.record = record
        self.storm = storm
        self.names = tuple(HISTORY_FIELDS) + (tuple(STORM_HISTORY_FIELDS) if storm is not None else ())
        self._rows = None
        self._elapsed = None

    def valid_time(self, elapsed_seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=float(elapsed_seconds))

    def storm_scale(self, when: datetime) -> tuple[float, np.ndarray]:
        """(ap_tau, the foF2 factor sqrt(R) on the columns, float64) at ``when``."""
        from woof.spacewx import storm
        s = self.storm
        try:
            ap_tau = storm.persisted_ap_tau(s["ap"], self.start_time, when, s["fit"].tau)
        except storm.StormError as exc:
            raise IonosphereHistoryError(f"spacewx_ionosphere_opt = 2: {exc}") from None
        return ap_tau, storm.fof2_scale(s["fit"], s["lat"], s["lon"], when, ap_tau, mlat_deg=s["mlat"])

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
            rows = cp.zeros((len(self.names), *self.shape), dtype=cp.float32)
            self._rows = dict(zip(self.names, (rows[i] for i in range(len(self.names)))))
        if self._elapsed != float(elapsed_seconds):
            when = self.valid_time(elapsed_seconds)
            scale = None
            if self.storm is not None:
                # Host float64, once per frame (as the CLI's --storm does).
                _ap_tau, scale = self.storm_scale(when)
            out = self.ionosphere.evaluate(when, self.f107, fo_scale=scale)
            for name, (key, _desc, _units) in HISTORY_FIELDS.items():
                self._rows[name][...] = out[key].reshape(self.shape)
            if scale is not None:
                self._rows["IONO_STORM"][...] = cp.asarray((scale * scale).astype(np.float32).reshape(self.shape))
            self._elapsed = float(elapsed_seconds)
        return dict(self._rows)


__all__ = ["HISTORY_FIELDS", "STORM_HISTORY_FIELDS", "IonosphereHistory", "IonosphereHistoryError", "enabled",
           "storm_enabled", "row_names", "history_shapes", "device_shapes", "parse_f107_source"]
