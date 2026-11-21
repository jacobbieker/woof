"""The L3b auroral oval in a forecast's history (``spacewx_aurora_opt = 1``),
and with ``spacewx_aurora_opt = 2`` its particle energy flux.

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

* ``"gfz"`` (or ``""``): the observed Kp of the latest 3-hour interval that
  has ended by the start, in the GFZ index file (fetched, CC BY 4.0); a local
  file in GFZ's format also works;
* ``"omni"`` or ``"swpc"``: the Kp estimated from the solar wind
  (woof.spacewx.solarwind) for the latest 3-hour interval whose fitted hours
  have ended by the start, from NASA SPDF OMNI2 or NOAA SWPC's real-time feed;
* a number in [0, 9]: that Kp.

Kp becomes |AL| through Starkov's relation.  The magnetic frame (the IGRF-14
dipole) is taken at the start; the subsolar point, and so the MLT, moves with
each frame.  The columns are the radiation geometry's, so a run needs a
radiation scheme on.  Hex meshes and streamed (tiled) domains refuse the option
for now; each says so by name.

``spacewx_aurora_opt = 2`` adds three rows from OVATION Prime 2010
(woof.spacewx.ovation): the precipitating electron energy flux, its mean
energy and the ion energy flux.  OVATION is driven by the solar wind, so this
needs ``spacewx_kp_source = "omni"`` or ``"swpc"``.  Its dPhi/dt is OVATION's
4-hour weighted mean taken on the hour of the start, where the hour in
progress has weight 0, so only hours that have ended count; it is held for the
run like the Kp, and so is the season.  Each frame looks the held grid up at
the oval's geomagnetic latitude and MLT.

``spacewx_aurora_opt = 3`` adds the Joule power of Knipp et al. 2004
(woof.spacewx.ovation.joule_power: Sol. Phys. 224, 495, Table I), a seasonal
quadratic in |PC(N)| and |Dst| fitted to AMIE, on top of option 2's rows.
PC(N) and Dst come from OMNI2 (words 52 and 41), so it needs
``spacewx_kp_source = "omni"``; SWPC's real-time feed carries neither.  Both
are taken from the latest OMNI2 hour that has ended by the start and has
both values, held for the run like the Kp, and so is Knipp's season (that of
the start).  The fit is for the northern hemisphere and has no horizontal
structure, so ``JOULE_POWER`` holds that one number (GW) in every column of
every frame; Knipp et al. double it for the globe, and the receipt carries
both.  It is the hemisphere-integrated Joule power, not a local heating rate.
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
#: The energy-flux rows of ``spacewx_aurora_opt = 2``: wrfout name ->
#: (woof.spacewx.ovation column field, description, units).
ENERGY_HISTORY_FIELDS = {
    "AURORA_EFLUX": ("eflux", "ELECTRON PRECIPITATION ENERGY FLUX (OVATION PRIME 2010)", "mW m-2"),
    "AURORA_EMEAN": ("mean_energy", "ELECTRON PRECIPITATION MEAN ENERGY (OVATION PRIME 2010)", "keV"),
    "AURORA_IFLUX": ("eflux_ion", "ION PRECIPITATION ENERGY FLUX (OVATION PRIME 2010)", "mW m-2"),
}
#: The Joule row of ``spacewx_aurora_opt = 3``: wrfout name -> (description, units).
JOULE_HISTORY_FIELDS = {
    "JOULE_POWER": ("NORTHERN-HEMISPHERE JOULE POWER (KNIPP ET AL. 2004), HELD FROM THE START", "GW"),
}
#: How far back from the start an observed or estimated Kp is looked for.
LOOKBACK_HOURS = 24


class AuroraHistoryError(ValueError):
    pass


def enabled(cfg) -> bool:
    return bool(int(getattr(cfg, "spacewx_aurora_opt", 0) or 0))


def energy_enabled(cfg) -> bool:
    return int(getattr(cfg, "spacewx_aurora_opt", 0) or 0) >= 2


def joule_enabled(cfg) -> bool:
    return int(getattr(cfg, "spacewx_aurora_opt", 0) or 0) == 3


def row_names(cfg) -> tuple[str, ...]:
    """The wrfout names the option writes (none when off)."""
    if not enabled(cfg):
        return ()
    return (tuple(HISTORY_FIELDS) + (tuple(ENERGY_HISTORY_FIELDS) if energy_enabled(cfg) else ())
            + (tuple(JOULE_HISTORY_FIELDS) if joule_enabled(cfg) else ()))


def history_shapes(cfg) -> dict[str, tuple[int, int]]:
    """wrfout name -> shape of the rows the option adds (empty when off)."""
    return {name: (int(cfg.ny), int(cfg.nx)) for name in row_names(cfg)}


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
    shapes["spacewx_aurora/history"] = (len(row_names(cfg)), int(cfg.ny), int(cfg.nx))
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


def _interval_start(when: datetime) -> datetime:
    when = when.replace(minute=0, second=0, microsecond=0)
    return when - timedelta(hours=when.hour % 3)


def _latest(lookup, start: datetime, what: str, complete_hours: int) -> tuple[float, datetime]:
    """The value of ``lookup`` for the latest 3-hour interval T whose data end by
    ``start`` (T + ``complete_hours`` <= start): nothing after the start is read."""
    when = _interval_start(start)
    while when + timedelta(hours=complete_hours) > start:
        when -= timedelta(hours=3)
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


def _coupling_at_start(wind, start: datetime) -> float:
    """OVATION's dPhi/dt from the hours that have ended by the start: taken on
    the start's hour, with that hour and any later one removed from the record,
    so the three hours before must all be good and nothing after is read."""
    import dataclasses
    from woof.spacewx import ovation
    hour = start.replace(minute=0, second=0, microsecond=0)
    ended = wind.hours < np.datetime64(hour, "h")
    past = dataclasses.replace(wind, **{k: np.where(ended, getattr(wind, k), np.nan)
                                        for k in ("by", "bz", "speed")})
    try:
        return ovation.coupling(past, hour)
    except ovation.OvationError as exc:
        raise AuroraHistoryError(f"spacewx_aurora_opt = 2: no OVATION coupling at the start: {exc}") from None


def _joule_at_start(wind, start: datetime) -> dict:
    """Knipp et al. 2004's drivers held for the run: PC(N) and Dst of the latest
    hour that has ended by the start (hour h + 1 <= start) and carries both,
    looked for back to :data:`LOOKBACK_HOURS`; nothing after the start is read."""
    from woof.spacewx import ovation
    hour = start.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    first = hour
    while start - hour <= timedelta(hours=LOOKBACK_HOURS):
        try:
            pcn, dst = wind.pcn_at(hour), wind.dst_at(hour)
        except ValueError:
            pcn = dst = float("nan")
        if np.isfinite(pcn) and np.isfinite(dst):
            return {"pcn": float(pcn), "dst": float(dst), "hour": f"{hour:%Y-%m-%dT%H:%M}Z",
                    "season": ovation.knipp_season(start), "north_gw": ovation.joule_power(pcn, dst, start)}
        hour -= timedelta(hours=1)
    raise AuroraHistoryError(
        f"spacewx_aurora_opt = 3: no OMNI2 hour with both PC(N) and Dst in the {LOOKBACK_HOURS} h "
        f"before {start:%Y-%m-%dT%H:%MZ} (latest hour tried {first:%Y-%m-%dT%H}Z; {wind.source})")


def activity(source: str, start: datetime, *, indices_loader=None, solarwind_loader=None,
             memo=None, energy: bool = False, joule: bool = False) -> dict:
    """The run's held Kp, its |AL| and the receipt of where it came from; with
    ``energy``, also OVATION's dPhi/dt (a solar-wind source only); with
    ``joule``, also Knipp et al. 2004's PC(N), Dst and Joule power (OMNI2 only).

    With ``memo`` (a directory), a fetched source's result is kept per
    (source, start): a resumed run, and every domain of a nested one, holds
    the Kp the first read found, even after the live feed has moved on."""
    from woof.spacewx.aurora import kp_to_al

    kind, value = parse_kp_source(source)
    if energy and kind not in ("omni", "swpc"):
        raise AuroraHistoryError(
            f"spacewx_aurora_opt = 2 drives OVATION Prime with the solar wind, so "
            f"spacewx_kp_source must be 'omni' or 'swpc', not {source!r}")
    if joule and kind != "omni":
        raise AuroraHistoryError(
            f"spacewx_aurora_opt = 3 reads PC(N) and Dst, which only OMNI2 carries, so "
            f"spacewx_kp_source must be 'omni', not {source!r}")
    if kind == "value":
        return {"kp": float(value), "al": float(kp_to_al(value)),
                "record": {"spacewx_kp_source": "value"}}
    memo_path = held = None
    if memo is not None and kind in ("gfz", "omni", "swpc"):
        import hashlib
        import json
        from pathlib import Path
        key = hashlib.sha256(f"{kind}|{start.isoformat()}".encode()).hexdigest()[:24]
        memo_path = Path(memo) / f"{kind}-{start:%Y%m%dT%H%M%S}-{key}.json"
        if memo_path.is_file():
            try:
                held = json.loads(memo_path.read_text(encoding="utf-8"))
                out = {"kp": float(held["kp"]), "al": float(held["al"]),
                       "record": dict(held["record"], spacewx_kp_held_from=str(memo_path))}
                if held.get("dphidt") is not None:
                    out["dphidt"] = float(held["dphidt"])
                if isinstance(held.get("joule"), dict):
                    out["joule"] = {k: held["joule"][k] for k in ("pcn", "dst", "hour", "season", "north_gw")}
                if (not energy or "dphidt" in out) and (not joule or "joule" in out):
                    return out
                # Held by a run without the energy or Joule rows: keep its Kp, add what is missing.
            except (ValueError, KeyError, TypeError):
                held = None                            # damaged: read the source again
    if kind in ("omni", "swpc"):
        from woof.spacewx import solarwind
        load = solarwind_loader or solarwind.load
        wind = load(kind, start - timedelta(hours=LOOKBACK_HOURS + 4), start)
        if held is not None:
            # The hold predates its dPhi/dt or its Joule drivers: read the solar
            # wind for the start again (refused, by name, if the feed no longer
            # covers it).
            added, record = {}, dict(held["record"])
            if energy and "dphidt" not in out:
                out["dphidt"] = added["dphidt"] = _coupling_at_start(wind, start)
                out["record"]["spacewx_dphidt_sha256"] = record["spacewx_dphidt_sha256"] = wind.sha256
            if joule and "joule" not in out:
                out["joule"] = added["joule"] = _joule_at_start(wind, start)
                out["record"]["spacewx_joule_sha256"] = record["spacewx_joule_sha256"] = wind.sha256
            _write_memo(memo_path, {**held, **added, "record": record})
            return out
        # Hours T-1, T and T+1 must have ended by the start.
        kp, at = _latest(wind.kp_estimate, start, f"{kind} solar wind", complete_hours=2)
        record = {"spacewx_kp_source": wind.source, "spacewx_kp_sha256": wind.sha256,
                  "spacewx_kp_interval": f"{at:%Y-%m-%dT%H:%M}Z", "spacewx_kp_estimated": True}
        # Kept whenever it exists, so that a resume with the energy rows finds it.
        try:
            dphidt = _coupling_at_start(wind, start)
        except AuroraHistoryError:
            if energy:
                raise
            dphidt = None
        # Knipp's drivers, kept whenever OMNI2 has them (as the coupling is).
        joule_drive = None
        if kind == "omni":
            try:
                joule_drive = _joule_at_start(wind, start)
                record["spacewx_joule_sha256"] = wind.sha256
            except AuroraHistoryError:
                if joule:
                    raise
    else:
        from woof.spacewx import indices
        load = indices_loader or indices.load
        name = "gfz" if kind == "gfz" else value
        # The latest interval that has ended by the start; a cached GFZ file
        # without it is fetched again, and kept if GFZ cannot be reached.
        newest = _interval_start(start) - timedelta(hours=3)
        try:
            geo = load(name, newest, kp=True)
        except OSError:
            geo = load(name, None)
        kp, at = _latest(geo.kp_at, start, "observed Kp", complete_hours=3)
        record = {"spacewx_kp_source": geo.source, "spacewx_kp_sha256": geo.sha256,
                  "spacewx_kp_interval": f"{at:%Y-%m-%dT%H:%M}Z"}
    out = {"kp": float(kp), "al": float(kp_to_al(kp)), "record": record}
    if kind in ("omni", "swpc") and dphidt is not None:
        out["dphidt"] = float(dphidt)
    if kind == "omni" and joule_drive is not None:
        out["joule"] = joule_drive
    if memo_path is not None:
        _write_memo(memo_path, out)
    return out


def _write_memo(memo_path, out: dict) -> None:
    import json
    import os
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    part = memo_path.with_name(f"{memo_path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(out, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(memo_path)


_TABLES: dict = {}


def energy_tables(coef, dphidt: float, start: datetime) -> dict:
    """Per hemisphere, the (96, 80) grid of each energy row (its column value is
    the grid's cell, as woof.spacewx.ovation.energy_columns gives it), and the
    hemispheric powers.  Held per (tables, dPhi/dt, start), so the domains of a
    nested run share one."""
    from woof.spacewx import ovation
    key = (coef.sha256, coef.source, float(dphidt), start)
    if key not in _TABLES:
        grids = ovation.hemisphere_grids(coef, dphidt, start)
        out = {"power": ovation.power_scalars(grids)}
        for h, g in grids.items():
            eflux = g["eflux_diff"] + g["eflux_mono"] + g["eflux_wave"]
            nflux = g["nflux_diff"] + g["nflux_mono"] + g["nflux_wave"]
            cells = {"eflux": eflux, "mean_energy": ovation.mean_energy_kev(eflux, nflux),
                     "eflux_ion": g["eflux_ion"]}
            out[h] = {name: cells[key] for name, (key, _d, _u) in ENERGY_HISTORY_FIELDS.items()}
        _TABLES.clear()
        _TABLES[key] = out
    return _TABLES[key]


class AuroraHistory:
    """The option's state on one domain: rebuilt from the RunConfig, stateless."""

    @classmethod
    def for_run(cls, cfg, radiation, *, indices_loader=None, solarwind_loader=None, igrf_model=None,
                ovation_coefficients=None):
        if not enabled(cfg):
            return None
        from woof.spacewx.history import _host, _naive_utc
        start = getattr(radiation, "start_time", None)
        lat = getattr(radiation, "latitude_deg", None)
        lon = getattr(radiation, "longitude_deg", None)
        if radiation is None or start is None or lat is None or lon is None:
            raise AuroraHistoryError(
                f"spacewx_aurora_opt = {int(cfg.spacewx_aurora_opt)} places the auroral oval on the radiation "
                "geometry's columns, and this domain's radiation publishes no start "
                "time or latitude/longitude")
        start = _naive_utc(start)
        memo = None
        if indices_loader is None and solarwind_loader is None:
            from woof.spacewx.solar_irradiance import cache_root
            memo = cache_root() / "aurora-activity"
        energy, joule = energy_enabled(cfg), joule_enabled(cfg)
        drive = activity(getattr(cfg, "spacewx_kp_source", ""), start, indices_loader=indices_loader,
                         solarwind_loader=solarwind_loader, memo=memo, energy=energy, joule=joule)
        from woof.spacewx.aurora import Aurora
        lat, lon = _host(lat), _host(lon)
        oval = Aurora(lat, lon, when=start, igrf_model=igrf_model)
        tables = None
        if energy:
            from woof.spacewx import ovation
            try:
                coef = ovation_coefficients if ovation_coefficients is not None else ovation.load()
            except (ovation.OvationError, OSError, ValueError) as exc:
                raise AuroraHistoryError(
                    f"spacewx_aurora_opt = 2 needs the OVATION Prime coefficients (NOAA's ovation-prime "
                    f"2.3.0 zip, fetched into the space-weather cache): {exc}") from None
            tables = energy_tables(coef, drive["dphidt"], start)
            drive = dict(drive, record=dict(drive["record"], spacewx_ovation_sha256=coef.sha256,
                                            spacewx_ovation_source=coef.source))
        if not joule:
            drive = {k: v for k, v in drive.items() if k != "joule"}
            drive["record"] = {k: v for k, v in drive["record"].items() if k != "spacewx_joule_sha256"}
        return cls(oval, start, drive, lat.shape, tables=tables)

    def __init__(self, oval, start_time: datetime, drive: dict, shape, tables=None):
        self.aurora = oval
        self.start_time = start_time
        self.kp = drive["kp"]
        self.al = drive["al"]
        self.record = drive["record"]
        self.dphidt = drive.get("dphidt")
        self.tables = tables
        self.joule = drive.get("joule")
        # The summer fit's negative |Dst| term can take a near-zero PC(N) and
        # Dst a fraction of a GW below 0; a power is not negative, so the row
        # and the receipt's north/global values hold max(fit, 0), and the
        # receipt keeps the fit's own value beside them.
        self.joule_gw = max(float(self.joule["north_gw"]), 0.0) if self.joule is not None else None
        self.names = (tuple(HISTORY_FIELDS) + (tuple(ENERGY_HISTORY_FIELDS) if tables is not None else ())
                      + (tuple(JOULE_HISTORY_FIELDS) if self.joule is not None else ()))
        self.shape = tuple(int(n) for n in shape)
        self._rows = None
        self._elapsed = None

    def valid_time(self, elapsed_seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=float(elapsed_seconds))

    def receipt(self) -> dict:
        from woof.spacewx.aurora import receipt
        extra = {}
        if self.tables is not None:
            extra = {"spacewx_ovation_dphidt": self.dphidt, "spacewx_energy_model": "OVATION Prime 2010",
                     **{f"spacewx_{k}": v for k, v in self.tables["power"].items()}}
        if self.joule is not None:
            extra.update({"spacewx_joule_model": "Knipp et al. 2004 Table I, northern hemisphere, |PC(N)| and |Dst|",
                          "spacewx_joule_pcn": self.joule["pcn"], "spacewx_joule_dst_nt": self.joule["dst"],
                          "spacewx_joule_hour": self.joule["hour"], "spacewx_joule_season": self.joule["season"],
                          "spacewx_joule_fit_gw": self.joule["north_gw"],
                          "spacewx_joule_north_gw": self.joule_gw,
                          "spacewx_joule_global_gw": 2.0 * self.joule_gw})
        return receipt(self.aurora, {**self.record, "spacewx_kp": self.kp, "spacewx_al_nt": self.al, **extra})

    def history_fields(self, elapsed_seconds: float) -> dict:
        """wrfout name -> (ny, nx) float32 device row at the frame's valid time
        (one stable allocation, refilled when the valid time moves)."""
        import cupy as cp
        if self._rows is None:
            rows = cp.zeros((len(self.names), *self.shape), dtype=cp.float32)
            self._rows = dict(zip(self.names, (rows[i] for i in range(len(self.names)))))
            if self.joule is not None:
                # Held for the run: the same number in every column and frame.
                self._rows["JOULE_POWER"][...] = np.float32(self.joule_gw)
        if self._elapsed != float(elapsed_seconds):
            out = self.aurora.evaluate(self.valid_time(elapsed_seconds), self.al)
            for name, (key, _desc, _units) in HISTORY_FIELDS.items():
                self._rows[name][...] = out[key].reshape(self.shape)
            if self.tables is not None:
                # OVATION's bins on the frame's columns: an integer lookup of the
                # held tables, binned on the host in float64 from the kernel's
                # mlat and MLT exactly as woof.spacewx.ovation.on_columns does
                # (once per history frame, two column vectors each way).
                from woof.spacewx import ovation
                mlat = cp.asnumpy(out["mlat"]).astype(np.float64)
                mlt = cp.asnumpy(out["mlt"]).astype(np.float64)
                north = ovation.on_columns(self.tables["north"], mlat, mlt)
                south = ovation.on_columns(self.tables["south"], mlat, mlt)
                for name in ENERGY_HISTORY_FIELDS:
                    col = np.where(mlat >= 0.0, north[name], south[name])
                    self._rows[name][...] = cp.asarray(col.astype(np.float32).reshape(self.shape))
            self._elapsed = float(elapsed_seconds)
        return dict(self._rows)


__all__ = ["HISTORY_FIELDS", "ENERGY_HISTORY_FIELDS", "JOULE_HISTORY_FIELDS", "AuroraHistory",
           "AuroraHistoryError", "activity", "enabled", "energy_enabled", "joule_enabled", "row_names",
           "history_shapes", "device_shapes", "parse_kp_source"]
