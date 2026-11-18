"""Observation readers for the observation cost (EXPERIMENTAL).

Every reader returns a :class:`woof.grad.obs_cost.ObsSet` for one time
window and one latitude/longitude box, read-only and anonymously from the
Source Cooperative icechunk stores the hex referees already read
(``us-west-2.opendata.source.coop``, prefix ``bkr/obs``).  Nothing is
cached inside the tree; ``woof grad obs`` writes the window's set to a file
the caller names, outside the repository.

* ``wis2_land`` (``bkr/obs/wis2_land.icechunk``): WIS2 SYNOP reports.  2 m
  temperature and dewpoint (sensor 1 to 3 m above ground, or unstated) and
  10 m wind as components and speed (anemometer 8 to 12 m, or unstated,
  the station wind referee's rule).  One row per station and time: a
  corrected report (higher ``update_sequence_number``) replaces the
  original.
* ``wis2_upper_air`` (``bkr/obs/wis2_upper_air.icechunk``): radiosondes at
  the sixteen mandatory levels, temperature and wind.  Each level keeps
  its own time (launch plus time since launch) and position (launch plus
  drift), so FGAT pairs a level with the frame nearest the moment it was
  measured.  A level comes from a reported level within 0.5 hPa of it, or
  from ln p interpolation between reported levels at most 15 % apart (the
  radiosonde referee's rule, :func:`woof.hex.sonde_profile_referee.mandatory_from_levels`,
  applied to every quantity).  A station reporting twice (two bulletins)
  keeps the sounding with more levels.
* ``wis2_aircraft`` (``bkr/obs/wis2_aircraft.icechunk``): AMDAR
  temperature and wind at the pressure of the reported flight level
  (pressure altitude, ICAO standard atmosphere), one row per report id.
* ``metop_sg_ro`` (``bkr/obs/metop_sg_ro.icechunk``): Metop-SG GNSS
  radio occultation.  The store carries bending angle against impact
  parameter; refractivity comes from the Abel inversion
  (:func:`abel_refractivity`) and is placed at each level's tangent point,
  at height ``a / n - r_curve - undulation`` above mean sea level, thinned
  to one level per ``ro_spacing_m``.
* GOES cloud water path: a ``gpuwm-obs.goes-grid.v1`` file (the product
  :mod:`woof.obs.goes_grid` writes), one column observation per
  observed cell, with the product's own error.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from woof.grad.obs_cost import ObsCostError, ObsSet, _utc
# The radiosonde referee's levels and temperature bounds, shared so the cost and the scorecard
# select the same levels (tests/test_grad_obs_cost.py holds mandatory_levels to its rule).
from woof.hex.sonde_profile_referee import LEVELS_HPA as MANDATORY_HPA, T_BOUNDS
from woof.hex.metoffice_intermediate import wind_components

BUCKET = "us-west-2.opendata.source.coop"
STORES = {
    "wis2_land": "bkr/obs/wis2_land.icechunk",
    "wis2_upper_air": "bkr/obs/wis2_upper_air.icechunk",
    "wis2_aircraft": "bkr/obs/wis2_aircraft.icechunk",
    "metop_sg_ro": "bkr/obs/metop_sg_ro.icechunk",
}
SPEED_BOUNDS = (0.0, 120.0)
T2_SENSOR_M = (1.0, 3.0)
WIND_SENSOR_M = (8.0, 12.0)
# A WIS2 sounding filed for hour H is launched up to about 75 min before it.
SONDE_LAUNCH_LEAD = timedelta(minutes=150)
# Two bulletins of one station launched this close together are one sounding.
SONDE_DUPLICATE_S = 3600.0


def open_store(name):
    """The store ``name`` as an xarray Dataset (anonymous, read-only)."""
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(bucket=BUCKET, prefix=STORES[name], region="us-west-2", anonymous=True)
    return xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store, consolidated=False)


def _in_box(lat, lon, box):
    """Inside (south, north, west, east); longitudes are taken in [west, west + 360)."""
    south, north, west, east = box
    lon = ((np.asarray(lon, float) - west) % 360.0) + west
    return (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)


def _ids(raw, prefix, index):
    """Text ids, with a unique stand-in where the store has none (so missing ids never merge)."""
    out = []
    for value, i in zip(np.asarray(raw).tolist(), np.asarray(index).tolist()):
        text = value.decode() if isinstance(value, bytes) else str(value)
        text = text.strip()
        out.append(text if text and text.lower() not in ("nan", "none") else f"{prefix}-{int(i)}")
    return np.asarray(out, dtype=str)


def _window(start, end):
    return (np.datetime64(_utc(start).replace(tzinfo=None), "ns"),
            np.datetime64(_utc(end).replace(tzinfo=None), "ns"))


def _rows(ds, index, names):
    """Columns ``names`` at rows ``index`` (sorted): one contiguous block when the rows are dense,
    else only the selected rows (a late correction far from the rest does not pull the span)."""
    if index.size == 0:
        return {name: np.zeros(0) for name in names}
    lo, hi = int(index.min()), int(index.max()) + 1
    if hi - lo <= 4 * index.size:
        return {name: np.asarray(ds[name][lo:hi].values)[index - lo] for name in names}
    return {name: np.asarray(ds[name].isel({ds[name].dims[0]: index}).values) for name in names}


def pressure_from_flight_level(height_m):
    """ICAO standard-atmosphere pressure (Pa) of a pressure altitude (m), troposphere and lower stratosphere."""
    h = np.asarray(height_m, float)
    low = 101325.0 * np.power(np.clip(1.0 - 2.25577e-5 * h, 1e-9, None), 5.25588)
    high = 22632.06 * np.exp(-1.576883e-4 * (h - 11000.0))
    return np.where(h <= 11000.0, low, high)


# --------------------------------------------------------------------------- SYNOP
def wis2_land(start, end, box, *, ds=None) -> ObsSet:
    ds = open_store("wis2_land") if ds is None else ds
    lo, hi = _window(start, end)
    time = ds["time"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    index = np.nonzero((time >= lo) & (time <= hi) & _in_box(lat, lon, box))[0]
    names = ("station_id", "station_elevation", "2m_temperature", "2m_dewpoint_temperature",
             "temperature_sensor_height", "10m_wind_speed", "10m_wind_direction", "wind_sensor_height",
             "update_sequence_number")
    cols = _rows(ds, index, names)
    cols.update(time=time[index], lat=lat[index], lon=lon[index], row=index)
    return land_rows(cols)


def land_rows(cols) -> ObsSet:
    """The SYNOP selection on plain columns (what :func:`wis2_land` reads)."""
    sid = _ids(cols["station_id"], "wis2-land", cols.get("row", np.arange(len(cols["station_id"]))))
    time = np.asarray(cols["time"]).astype("datetime64[s]")
    seq = np.asarray(cols["update_sequence_number"], float)
    best = {}
    for i in np.lexsort((np.nan_to_num(seq, nan=-1.0), time, sid)):
        best[(sid[i], time[i])] = i  # the highest sequence number of a (station, time) wins
    keep = np.array(sorted(best.values()), dtype=np.int64)
    lat = np.asarray(cols["lat"], float)[keep]
    lon = np.asarray(cols["lon"], float)[keep]
    elev = np.asarray(cols["station_elevation"], float)[keep]
    t2 = np.asarray(cols["2m_temperature"], float)[keep]
    td = np.asarray(cols["2m_dewpoint_temperature"], float)[keep]
    th = np.asarray(cols["temperature_sensor_height"], float)[keep]
    spd = np.asarray(cols["10m_wind_speed"], float)[keep]
    wdir = np.asarray(cols["10m_wind_direction"], float)[keep]
    wh = np.asarray(cols["wind_sensor_height"], float)[keep]
    sid, time = sid[keep], time[keep]
    t_ok = ~np.isfinite(th) | ((th >= T2_SENSOR_M[0]) & (th <= T2_SENSOR_M[1]))
    w_ok = ~np.isfinite(wh) | ((wh >= WIND_SENSOR_M[0]) & (wh <= WIND_SENSOR_M[1]))
    good_t = t_ok & np.isfinite(t2) & (t2 >= T_BOUNDS[0]) & (t2 <= T_BOUNDS[1])
    good_td = t_ok & np.isfinite(td) & (td >= T_BOUNDS[0]) & (td <= T_BOUNDS[1]) & ~(td > t2 + 0.05)
    good_s = w_ok & np.isfinite(spd) & (spd >= SPEED_BOUNDS[0]) & (spd <= SPEED_BOUNDS[1])
    good_uv = good_s & np.isfinite(wdir) & (wdir >= 0) & (wdir <= 360) & ~((spd > 0) & (wdir == 0))
    u, v = wind_components(spd, wdir)
    parts = []
    for kind, sel, value in (("surface_temperature", good_t, t2), ("surface_dewpoint", good_td, td),
                             ("surface_wind_speed", good_s, spd), ("surface_wind_u", good_uv, u),
                             ("surface_wind_v", good_uv, v)):
        parts.append(ObsSet.from_columns(platform=np.full(sel.sum(), "synop"), kind=np.full(sel.sum(), kind),
                                         station=sid[sel], time=time[sel], lat=lat[sel], lon=lon[sel],
                                         elevation_m=elev[sel], value=value[sel]))
    return ObsSet.concat(parts)


# --------------------------------------------------------------------------- radiosondes
def mandatory_levels(pressure_pa, values: dict) -> dict:
    """{level_hpa: {name: value}} at the mandatory levels, by the referee's rule, for every column in ``values``.

    A reported level within 0.5 hPa is taken as is; otherwise ln p
    interpolation between the bracketing reported levels when they are at
    most 15 % apart in pressure.  Each quantity is interpolated over the
    levels where it is finite, so a missing wind does not remove a
    temperature.
    """
    p = np.asarray(pressure_pa, float)
    out = {}
    for name, column in values.items():
        x = np.asarray(column, float)
        ok = np.isfinite(p) & (p > 0) & np.isfinite(x)
        pp, xx = p[ok], x[ok]
        if pp.size == 0:
            continue
        order = np.argsort(pp)
        pp, xx = pp[order], xx[order]
        for lev in MANDATORY_HPA:
            target = lev * 100.0
            near = np.nonzero(np.abs(pp - target) <= 50.0)[0]
            if near.size:
                out.setdefault(lev, {})[name] = float(xx[near[np.argmin(np.abs(pp[near] - target))]])
                continue
            hi = np.searchsorted(pp, target)
            if 0 < hi < pp.size and pp[hi] / pp[hi - 1] <= 1.15:
                w = (np.log(target) - np.log(pp[hi - 1])) / (np.log(pp[hi]) - np.log(pp[hi - 1]))
                out.setdefault(lev, {})[name] = float(xx[hi - 1] + w * (xx[hi] - xx[hi - 1]))
    return out


def wis2_upper_air(start, end, box, *, ds=None) -> ObsSet:
    ds = open_store("wis2_upper_air") if ds is None else ds
    lo, hi = _window(start, end)
    launch = ds["launch_time"].values
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    rows = np.nonzero((launch >= lo - np.timedelta64(SONDE_LAUNCH_LEAD)) & (launch <= hi)
                      & _in_box(lat, lon, box))[0]
    if rows.size == 0:
        return ObsSet.empty()
    first = np.asarray(ds["first_level"].values)[rows]
    count = np.asarray(ds["level_count"].values)[rows]
    a, b = int(first.min()), int((first + count).max())
    level = {name: np.asarray(ds[name][a:b].values, float) for name in
             ("pressure", "temperature", "wind_speed", "wind_direction", "time_since_launch",
              "latitude_displacement", "longitude_displacement")}
    sid = _ids(np.asarray(ds["station_id"].values)[rows], "wis2", rows)
    elev = np.asarray(ds["station_elevation"].values, float)[rows]
    kept = []  # (name, launch, n, r, profile); bulletins of one launch keep the one with more levels
    for n, r in enumerate(rows):
        s0, s1 = int(first[n]) - a, int(first[n] + count[n]) - a
        sl = {k: v[s0:s1] for k, v in level.items()}
        t = np.where((sl["temperature"] >= T_BOUNDS[0]) & (sl["temperature"] <= T_BOUNDS[1]), sl["temperature"], np.nan)
        spd = np.where((sl["wind_speed"] >= SPEED_BOUNDS[0]) & (sl["wind_speed"] <= SPEED_BOUNDS[1]),
                       sl["wind_speed"], np.nan)
        u, v = wind_components(spd, sl["wind_direction"])
        disp_lat = np.nan_to_num(sl["latitude_displacement"])
        disp_lon = np.nan_to_num(sl["longitude_displacement"])
        since = np.nan_to_num(sl["time_since_launch"])
        profile = mandatory_levels(sl["pressure"], {"t": t, "u": u, "v": v, "dlat": disp_lat, "dlon": disp_lon,
                                                    "since": since})
        if not profile:
            continue
        name = sid[n]
        same = [k for k, row in enumerate(kept) if row[0] == name
                and abs((row[1] - launch[r]) / np.timedelta64(1, "s")) < SONDE_DUPLICATE_S]
        if not same:
            kept.append((name, launch[r], n, r, profile))
        elif len(profile) > len(kept[same[0]][4]):
            kept[same[0]] = (name, launch[r], n, r, profile)
    columns = {k: [] for k in ("station", "time", "lat", "lon", "pressure_pa", "elevation_m", "kind", "value")}
    for name, _launch, n, r, profile in sorted(kept, key=lambda row: (row[0], row[1])):
        for lev, q in sorted(profile.items(), reverse=True):
            when = launch[r] + np.timedelta64(int(round(q.get("since", 0.0))), "s")
            if not (lo <= when <= hi):
                continue
            for kind, key in (("upper_temperature", "t"), ("upper_wind_u", "u"), ("upper_wind_v", "v")):
                if key in q and np.isfinite(q[key]):
                    columns["station"].append(name)
                    columns["time"].append(when)
                    columns["lat"].append(lat[r] + q.get("dlat", 0.0))
                    columns["lon"].append(lon[r] + q.get("dlon", 0.0))
                    columns["pressure_pa"].append(lev * 100.0)
                    columns["elevation_m"].append(elev[n])
                    columns["kind"].append(kind)
                    columns["value"].append(q[key])
    n = len(columns["value"])
    return ObsSet.from_columns(platform=np.full(n, "sonde"), **columns)


# --------------------------------------------------------------------------- aircraft
def wis2_aircraft(start, end, box, *, ds=None) -> ObsSet:
    ds = open_store("wis2_aircraft") if ds is None else ds
    lo, hi = _window(start, end)
    time = ds["time"].values
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    index = np.nonzero((time >= lo) & (time <= hi) & _in_box(lat, lon, box))[0]
    cols = _rows(ds, index, ("report_id", "flight_level", "temperature", "wind_speed", "wind_direction",
                             "aircraft_identifier"))
    rid = _ids(cols["report_id"], "wis2-aircraft", index)  # a missing id is its own report
    _, first = np.unique(rid, return_index=True)
    keep = np.sort(first)
    fl = np.asarray(cols["flight_level"], float)[keep]
    t = np.asarray(cols["temperature"], float)[keep]
    spd = np.asarray(cols["wind_speed"], float)[keep]
    wdir = np.asarray(cols["wind_direction"], float)[keep]
    when = time[index][keep]
    la, lo_ = lat[index][keep], lon[index][keep]
    ident = np.asarray([str(s) for s in np.asarray(cols["aircraft_identifier"]).tolist()])[keep]
    p = pressure_from_flight_level(fl)
    ok = np.isfinite(fl) & (fl > -500) & (fl < 20000)
    good_t = ok & np.isfinite(t) & (t >= T_BOUNDS[0]) & (t <= T_BOUNDS[1])
    good_w = ok & np.isfinite(spd) & (spd >= SPEED_BOUNDS[0]) & (spd <= SPEED_BOUNDS[1]) & np.isfinite(wdir)
    u, v = wind_components(spd, wdir)
    parts = []
    for kind, sel, value in (("upper_temperature", good_t, t), ("upper_wind_u", good_w, u),
                             ("upper_wind_v", good_w, v)):
        parts.append(ObsSet.from_columns(platform=np.full(sel.sum(), "aircraft"), kind=np.full(sel.sum(), kind),
                                         station=ident[sel], time=when[sel], lat=la[sel], lon=lo_[sel],
                                         pressure_pa=p[sel], value=value[sel]))
    return ObsSet.concat(parts)


# --------------------------------------------------------------------------- radio occultation
def abel_refractivity(impact_parameter_m, bending_rad):
    """Refractive index ``n`` at each impact parameter, by the Abel inversion.

    ``ln n(a_i) = (1/pi) * integral_{a_i}^{a_top} alpha(a) / sqrt(a^2 - a_i^2) da``,
    integrated in ``x = sqrt(a^2 - a_i^2)`` (where the integrand
    ``alpha / a`` is smooth) by the trapezoid rule over the reported
    levels.  The profile above its last level is taken as zero, which
    underestimates ``n - 1`` by less than about 0.1 N-unit below 30 km
    for a profile reaching 60 km.  Returns NaN where the input is not
    finite.
    """
    a = np.asarray(impact_parameter_m, float)
    alpha = np.asarray(bending_rad, float)
    ok = np.isfinite(a) & np.isfinite(alpha)
    out = np.full(a.shape, np.nan)
    if ok.sum() < 3:
        return out
    order = np.nonzero(ok)[0][np.argsort(a[ok])]
    aa, al = a[order], alpha[order]
    f = al / aa
    ln_n = np.zeros(aa.size)
    for i in range(aa.size - 1):
        x = np.sqrt(np.clip(aa[i:] ** 2 - aa[i] ** 2, 0.0, None))
        ln_n[i] = np.trapezoid(f[i:], x) / np.pi
    out[order] = np.exp(ln_n)
    return out


def ro_profile_rows(impact_parameter_m, bending_rad, lat_tp, lon_tp, *, r_curve, undulation, time,
                    station="", spacing_m=500.0, max_height_m=30000.0) -> ObsSet:
    """Refractivity rows of one occultation: N = 1e6 (n - 1) at height a/n - r_curve - undulation."""
    n = abel_refractivity(impact_parameter_m, bending_rad)
    a = np.asarray(impact_parameter_m, float)
    height = a / n - float(r_curve) - float(undulation)
    refr = 1.0e6 * (n - 1.0)
    ok = np.isfinite(height) & np.isfinite(refr) & (refr > 0) & (height >= 0) & (height <= max_height_m) \
        & np.isfinite(lat_tp) & np.isfinite(lon_tp)
    index = np.nonzero(ok)[0]
    index = index[np.argsort(height[index])]
    chosen, last = [], -np.inf
    for i in index:
        if height[i] - last >= spacing_m:
            chosen.append(i)
            last = height[i]
    chosen = np.asarray(chosen, np.int64)
    m = chosen.size
    return ObsSet.from_columns(platform=np.full(m, "ro"), kind=np.full(m, "refractivity"),
                               station=np.full(m, station), time=[time] * m,
                               lat=np.asarray(lat_tp, float)[chosen], lon=np.asarray(lon_tp, float)[chosen],
                               height_m=height[chosen], value=refr[chosen])


def metop_sg_ro(start, end, box, *, ds=None, spacing_m=500.0, quality_ok_value=None) -> ObsSet:
    """Occultations whose mean tangent point lies in ``box``.

    ``quality_ok_value`` keeps only occultations whose ``overall_quality_ok``
    equals it (None keeps every occultation; the store does not document
    the flag's convention).
    """
    ds = open_store("metop_sg_ro") if ds is None else ds
    lo, hi = _window(start, end)
    time = ds["time"].values
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    sel = (time >= lo) & (time <= hi) & _in_box(lat, lon, box)
    if quality_ok_value is not None:
        sel &= np.asarray(ds["overall_quality_ok"].values, float) == quality_ok_value
    parts = []
    for i in np.nonzero(sel)[0]:
        one = ds.isel(time=int(i))
        parts.append(ro_profile_rows(one["impact_parameter"].values, one["bending_angle"].values,
                                     one["lat_tp"].values, one["lon_tp"].values,
                                     r_curve=float(one["r_curve"].values), undulation=float(one["undulation"].values),
                                     time=time[i], station=f"{one['platform_name'].values}-{int(i)}",
                                     spacing_m=spacing_m))
    out = ObsSet.concat(parts)
    return out.take(np.nonzero(_in_box(out.lat, out.lon, box))[0]) if len(out) else out


# --------------------------------------------------------------------------- GOES CWP
def goes_cwp(path) -> ObsSet:
    """Column observations of one ``gpuwm-obs.goes-grid.v1`` file, with the product's own error."""
    from woof.obs.goes_grid import read_goes_grid

    document = read_goes_grid(path)
    arrays = document["variables"]
    valid = document.get("valid_time")
    if not valid:
        raise ObsCostError(f"{path} names no valid_time")
    when = datetime.fromisoformat(str(valid).replace("Z", "+00:00"))
    mask = np.asarray(arrays["cwp_mask"]).astype(bool)
    value = np.asarray(arrays["cwp_obs"], float)[mask]
    m = value.size
    return ObsSet.from_columns(platform=np.full(m, "goes"), kind=np.full(m, "cloud_water_path"),
                               time=[when] * m, lat=np.asarray(arrays["XLAT"], float)[mask],
                               lon=np.asarray(arrays["XLONG"], float)[mask], value=value,
                               error=np.asarray(arrays["cwp_err"], float)[mask])


READERS = {"wis2_land": wis2_land, "wis2_upper_air": wis2_upper_air, "wis2_aircraft": wis2_aircraft,
           "metop_sg_ro": metop_sg_ro}


def read_window(stores, start, end, box) -> tuple[ObsSet, dict]:
    """Every named store's observations in the window and box, with per-store counts."""
    parts, counts = [], {}
    for name in stores:
        if name not in READERS:
            raise ObsCostError(f"unknown store {name!r}; known: {sorted(READERS)}")
        part = READERS[name](start, end, box)
        counts[name] = len(part)
        parts.append(part)
    return ObsSet.concat(parts), counts


def box_of(lat, lon, margin_deg: float = 0.0) -> tuple[float, float, float, float]:
    """(south, north, west, east) of model columns, widened by ``margin_deg``."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    centre = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(lon))), np.mean(np.cos(np.radians(lon))))))
    rel = ((lon - centre + 180.0) % 360.0) - 180.0
    return (float(lat.min()) - margin_deg, float(lat.max()) + margin_deg,
            centre + float(rel.min()) - margin_deg, centre + float(rel.max()) + margin_deg)


__all__ = ["BUCKET", "MANDATORY_HPA", "READERS", "STORES", "abel_refractivity", "box_of", "goes_cwp",
           "land_rows", "mandatory_levels", "metop_sg_ro", "open_store", "pressure_from_flight_level",
           "read_window", "ro_profile_rows", "wind_components", "wis2_aircraft", "wis2_land", "wis2_upper_air"]
