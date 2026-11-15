"""Radiosonde profile referee for MPAS-mesh history (woof hex or native MPAS-A).

The surface referees (ASOS, MRMS, Stage-IV) cannot see model temperature aloft, so the
upper-band drift in components/hex/docs/declared-divergences.md (section 1) needs a profile
referee. This one scores the temperature of MPAS-layout history frames against radiosondes
at the sixteen mandatory levels (1000 to 10 hPa), by latitude band and lead time.

Observations, read-only, from the Source Cooperative icechunk stores (anonymous S3):

* ``igra16``: ``bkr/obs/igra_v2_16_levels.icechunk``, IGRA v2 interpolated to the sixteen
  mandatory levels (station x time x level; the store ends mid-2025);
* ``wis2``: ``bkr/obs/wis2_upper_air.icechunk``, WMO WIS2 BUFR soundings, ragged (from
  late October 2026).

A sounding is matched to the frame whose valid time is its nominal synoptic hour (IGRA's
time axis is nominal; a WIS2 launch within ``LAUNCH_WINDOW`` before the hour counts as that
hour). Model side: the nearest cell to the station (great circle), temperature
``theta * (p / 1e5) ** (Rd / cp)`` interpolated linearly in ln p between the two model
levels that bracket the sonde level; a level outside the column is dropped, never
extrapolated. Cells with ``bdyMaskCell > 0`` (a regional mesh's relaxation zone) are never
used.

Quality control: a station whose initial-time sounding departs from the initial state (the
analysis every arm shares) by more than ``INITIAL_MEDIAN_DEPARTURE_K`` in median over its
levels is dropped from every lead of the case; the record lists the stations dropped.

Statistics: per (band, level, lead) the count, bias (model - obs) and RMSE; the drift is the
least-squares slope of bias against lead (K per 24 h). ``compare`` pairs two arms over the
same cases and resamples whole cases (the independent blocks) with a fixed seed, so station
and level counts never inflate the confidence. No acquisition writes anywhere but the
caller's output directory.

    python -m woof.hex.sonde_profile_referee score --history DIR --sondes wis2 \
        --case CASE --arm ARM --out OUT.json [--pairs OUT.csv]
    python -m woof.hex.sonde_profile_referee compare A1.json [A2.json ...] --against B1.json [...] \
        --out CMP.json [--replicates 5000] [--seed 20261115]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

LEVELS_HPA = (1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10)
RD_OVER_CP = 287.0 / 1004.5
BANDS = {"SP": (-90.0, -60.0), "SH": (-60.0, -20.0), "TR": (-20.0, 20.0), "NH": (20.0, 60.0),
         "NP": (60.0, 90.0), "GL": (-90.0, 90.0)}
STORES = {"igra16": "bkr/obs/igra_v2_16_levels.icechunk", "wis2": "bkr/obs/wis2_upper_air.icechunk"}
BUCKET = "us-west-2.opendata.source.coop"
# A WIS2 sounding filed for hour H is launched up to about 75 min before it.
LAUNCH_WINDOW = (timedelta(minutes=-150), timedelta(minutes=60))
# Gross bounds; a value outside them is instrument garbage and is dropped, never repaired.
T_BOUNDS = (150.0, 340.0)
INITIAL_MEDIAN_DEPARTURE_K = 5.0
INITIAL_MIN_LEVELS = 3


# ----------------------------------------------------------------------------- model side
def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def nearest_cells(cell_xyz, lat_deg, lon_deg, usable=None):
    """Index of the nearest usable cell to each point (maximum dot product of unit vectors)."""
    points = unit_vectors(lat_deg, lon_deg)
    out = np.empty(len(points), dtype=np.int64)
    cells = np.arange(len(cell_xyz)) if usable is None else np.flatnonzero(usable)
    if cells.size == 0:
        raise ValueError("no usable model cell to pair against")
    xyz = np.asarray(cell_xyz)[cells]
    for start in range(0, len(points), 256):
        out[start:start + 256] = cells[np.argmax(points[start:start + 256] @ xyz.T, axis=1)]
    return out


def column_temperature_at(theta_col, pressure_col, level_pa):
    """Temperature (K) of one model column at ``level_pa``, linear in ln p; NaN outside the column."""
    t = theta_col * (pressure_col / 1.0e5) ** RD_OVER_CP
    lnp = np.log(pressure_col)
    target = np.log(level_pa)
    if not (lnp.min() <= target <= lnp.max()):
        return np.nan
    order = np.argsort(lnp)
    return float(np.interp(target, lnp[order], t[order]))


def read_frame(path):
    import netCDF4

    with netCDF4.Dataset(path) as d:
        v = d.variables
        stamp = "".join(c.decode() if isinstance(c, bytes) else c for c in np.ma.filled(v["xtime"][0], b" ")).strip()
        frame = {
            "path": path,
            "valid": datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc),
            "theta": np.asarray(v["theta"][0], dtype=np.float64),
            "pressure": np.asarray(v["pressure"][0], dtype=np.float64),
        }
        if "latCell" in v:
            frame["lat"] = np.degrees(np.asarray(v["latCell"][:], dtype=np.float64))
            frame["lon"] = np.degrees(np.asarray(v["lonCell"][:], dtype=np.float64))
        if "bdyMaskCell" in v:
            frame["bdy"] = np.asarray(v["bdyMaskCell"][:])
    return frame


def history_frames(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*history.*.nc")))
    if not paths:
        raise FileNotFoundError(f"no *history.*.nc under {folder}")
    return paths


# ----------------------------------------------------------------------------- observations
def _store(name):
    import icechunk

    storage = icechunk.s3_storage(bucket=BUCKET, prefix=STORES[name], region="us-west-2", anonymous=True)
    return icechunk.Repository.open(storage).readonly_session("main").store


def soundings_igra16(valid_times):
    """{valid: list of (station, lat, lon, {level_hpa: T})} from the IGRA16 store."""
    import xarray as xr

    ds = xr.open_zarr(_store("igra16"), consolidated=False)
    times = ds["time"].values
    levels = [int(round(float(p) / 100.0)) if float(p) > 2000 else int(round(float(p)))
              for p in ds["pressure_level"].values]
    stations = ds["station"].values
    out = {}
    for valid in valid_times:
        key = np.datetime64(valid.replace(tzinfo=None), "ns")
        idx = np.nonzero(times == key)[0]
        out[valid] = []
        if not len(idx):
            continue
        sub = ds.isel(time=int(idx[0]))
        temp = np.asarray(sub["temperature"].values, dtype=np.float64)
        lat = np.asarray(sub["latitude"].values, dtype=np.float64)
        lon = np.asarray(sub["longitude"].values, dtype=np.float64)
        for s in range(len(stations)):
            if not (np.isfinite(lat[s]) and np.isfinite(lon[s])):
                continue
            profile = {lev: float(temp[s, j]) for j, lev in enumerate(levels)
                       if lev in LEVELS_HPA and np.isfinite(temp[s, j]) and T_BOUNDS[0] <= temp[s, j] <= T_BOUNDS[1]}
            if profile:
                out[valid].append((str(stations[s]), float(lat[s]), float(lon[s]), profile))
    return out


def mandatory_from_levels(pressure_pa, temperature_k):
    """{level_hpa: T} from one sounding's reported levels: a reported mandatory level when
    present (|dp| <= 0.5 hPa), else ln-p interpolation between reported levels at most 15 %
    apart in pressure. Levels with bad values are dropped first."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t >= T_BOUNDS[0]) & (t <= T_BOUNDS[1])
    p, t = p[ok], t[ok]
    if p.size == 0:
        return {}
    order = np.argsort(p)
    p, t = p[order], t[order]
    out = {}
    for lev in LEVELS_HPA:
        target = lev * 100.0
        near = np.nonzero(np.abs(p - target) <= 50.0)[0]
        if near.size:
            out[lev] = float(t[near[np.argmin(np.abs(p[near] - target))]])
            continue
        hi = np.searchsorted(p, target)
        if 0 < hi < p.size and p[hi] / p[hi - 1] <= 1.15:
            w = (np.log(target) - np.log(p[hi - 1])) / (np.log(p[hi]) - np.log(p[hi - 1]))
            out[lev] = float(t[hi - 1] + w * (t[hi] - t[hi - 1]))
    return out


def soundings_wis2(valid_times):
    """{valid: list of (station, lat, lon, {level_hpa: T})} from the WIS2 ragged store. A station
    reporting twice for one hour (two bulletins) keeps the sounding with more levels."""
    import xarray as xr

    ds = xr.open_zarr(_store("wis2"), consolidated=False)
    launch = ds["launch_time"].values
    first = ds["first_level"].values
    count = ds["level_count"].values
    out = {}
    for valid in valid_times:
        key = np.datetime64(valid.replace(tzinfo=None), "ns")
        lo, hi = key + np.timedelta64(LAUNCH_WINDOW[0]), key + np.timedelta64(LAUNCH_WINDOW[1])
        rows = np.nonzero((launch >= lo) & (launch <= hi))[0]
        best = {}
        if rows.size:
            a, b = int(first[rows].min()), int((first[rows] + count[rows]).max())
            pres = np.asarray(ds["pressure"][a:b].values, dtype=np.float64)
            temp = np.asarray(ds["temperature"][a:b].values, dtype=np.float64)
            sid = ds["station_id"].values[rows]
            lat = np.asarray(ds["latitude"].values[rows], dtype=np.float64)
            lon = np.asarray(ds["longitude"].values[rows], dtype=np.float64)
            for n, r in enumerate(rows):
                s0, s1 = int(first[r]) - a, int(first[r] + count[r]) - a
                profile = mandatory_from_levels(pres[s0:s1], temp[s0:s1])
                if not profile or not (np.isfinite(lat[n]) and np.isfinite(lon[n])):
                    continue
                name = str(sid[n]) or f"wis2-{r}"
                if name not in best or len(profile) > len(best[name][3]):
                    best[name] = (name, float(lat[n]), float(lon[n]), profile)
        out[valid] = sorted(best.values())
    return out


SOURCES = {"igra16": soundings_igra16, "wis2": soundings_wis2}


# ----------------------------------------------------------------------------- scoring
def pair_frame(frame, soundings, cell_xyz, usable):
    """Rows (station, lat, lon, level_hpa, obs, model) for one frame."""
    if not soundings:
        return []
    lat = np.array([s[1] for s in soundings])
    lon = np.array([s[2] for s in soundings])
    cells = nearest_cells(cell_xyz, lat, lon, usable)
    rows = []
    for (station, slat, slon, profile), cell in zip(soundings, cells):
        theta, pres = frame["theta"][cell], frame["pressure"][cell]
        for lev, obs in sorted(profile.items(), reverse=True):
            model = column_temperature_at(theta, pres, lev * 100.0)
            if np.isfinite(model):
                rows.append((station, slat, slon, lev, obs, model))
    return rows


def initial_blacklist(pairs):
    """Stations whose sounding at the initial time departs from the initial state by more than
    ``INITIAL_MEDIAN_DEPARTURE_K`` in median over its levels (at least ``INITIAL_MIN_LEVELS``).

    At lead 0 the model state is the analysis every arm of a case shares, so the decision is
    the same for every arm and never depends on the forecast being judged. A station that far
    from the analysis through its whole column is a bad report or bad metadata, not weather
    (WIS2, 2026-10-28 00Z: a column 10-14 K off the analysis at every level at two stations).
    Such a station is dropped from every lead of the case."""
    by_station = {}
    for lead, station, _lat, _lon, _lev, obs, model in pairs:
        if lead == 0:
            by_station.setdefault(station, []).append(abs(model - obs))
    return {station for station, dev in by_station.items()
            if len(dev) >= INITIAL_MIN_LEVELS and float(np.median(dev)) > INITIAL_MEDIAN_DEPARTURE_K}


def band_of(lat):
    return [name for name, (lo, hi) in BANDS.items() if name != "GL" and lo <= lat < hi or (name == "NP" and lat == 90.0)]


def summarise(pairs):
    """{band: {level: {lead: [n, sum_err, sum_sq]}}} from rows with a lead."""
    acc = {}
    for lead, station, lat, lon, lev, obs, model in pairs:
        err = model - obs
        for band in band_of(lat) + ["GL"]:
            cell = acc.setdefault(band, {}).setdefault(str(lev), {}).setdefault(str(lead), [0, 0.0, 0.0])
            cell[0] += 1
            cell[1] += err
            cell[2] += err * err
    table = {}
    for band, levels in acc.items():
        for lev, leads in levels.items():
            for lead, (n, s, q) in leads.items():
                table.setdefault(band, {}).setdefault(lev, {})[lead] = {
                    "n": n, "bias": s / n, "rmse": (q / n) ** 0.5}
    for band, levels in table.items():
        for lev, leads in levels.items():
            x = np.array([float(k) for k in leads])
            y = np.array([leads[k]["bias"] for k in leads])
            if x.size >= 2 and np.ptp(x) > 0:
                leads["drift_k_per_24h"] = float(np.polyfit(x, y, 1)[0] * 24.0)
    return table


def score(history_dir, source, case, arm, *, pairs_out=None, soundings=None):
    paths = history_frames(history_dir)
    frames = [read_frame(p) for p in paths]
    start = min(f["valid"] for f in frames)
    synoptic = [f for f in frames if f["valid"].hour in (0, 12) and f["valid"].minute == 0]
    geo = frames[0]
    if "lat" not in geo:
        raise ValueError(f"{paths[0]} carries no latCell/lonCell")
    cell_xyz = unit_vectors(geo["lat"], geo["lon"])
    usable = (geo["bdy"] == 0) if "bdy" in geo else None
    if soundings is None:
        soundings = SOURCES[source]([f["valid"] for f in synoptic])
    pairs = []
    for f in synoptic:
        lead = int(round((f["valid"] - start).total_seconds() / 3600.0))
        for row in pair_frame(f, soundings.get(f["valid"], []), cell_xyz, usable):
            pairs.append((lead,) + row)
    rejected = initial_blacklist(pairs)
    pairs = [p for p in pairs if p[1] not in rejected]
    if pairs_out:
        with open(pairs_out, "w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["case", "arm", "lead_h", "station", "lat", "lon", "level_hpa", "obs_k", "model_k"])
            for p in pairs:
                w.writerow([case, arm, *p[:4], p[4], f"{p[5]:.3f}", f"{p[6]:.3f}"])
    return {"schema": "woof.hex.sonde-profile-referee/v1", "case": case, "arm": arm, "source": source,
            "store": STORES.get(source), "start": start.strftime("%Y-%m-%dT%H:%MZ"),
            "frames": [os.path.basename(f["path"]) for f in synoptic],
            "soundings": {f["valid"].strftime("%Y-%m-%dT%HZ"): len(soundings.get(f["valid"], [])) for f in synoptic},
            "pairs": len(pairs), "rejected_stations": sorted(rejected),
            "qc": {"initial_median_departure_k": INITIAL_MEDIAN_DEPARTURE_K, "min_levels": INITIAL_MIN_LEVELS},
            "table": summarise(pairs)}


# ----------------------------------------------------------------------------- comparison
def _metric(record, band, level, lead, stat):
    cell = record["table"].get(band, {}).get(str(level), {}).get(str(lead))
    if not isinstance(cell, dict) or not cell.get("n"):
        return None
    return abs(cell["bias"]) if stat == "absbias" else cell[stat]


def paired_interval(diff, *, replicates, seed):
    """Mean and 95 % bootstrap interval of the finite case-paired differences.

    Only the cases that supply this metric are resampled; with fewer than
    two of them there is no interval, and with fewer than four it is
    flagged exploratory.
    """
    finite = np.asarray(diff, float)
    finite = finite[np.isfinite(finite)]
    record = {"mean_diff": float(finite.mean()), "ci95": None, "cases": int(finite.size),
              "exploratory": bool(finite.size < 4)}
    if finite.size >= 2:
        draws = np.random.default_rng(seed).integers(0, finite.size, size=(replicates, finite.size))
        lo, hi = np.percentile(finite[draws].mean(axis=1), [2.5, 97.5])
        record["ci95"] = [float(lo), float(hi)]
    return record


def compare(arm_a, arm_b, *, replicates=5000, seed=20261115, stats=("bias", "absbias", "rmse")):
    """Case-paired difference A - B per (band, level, lead, stat), resampling whole cases."""
    by_case_a = {r["case"]: r for r in arm_a}
    by_case_b = {r["case"]: r for r in arm_b}
    cases = sorted(set(by_case_a) & set(by_case_b))
    if not cases:
        raise ValueError("the two arms share no case")
    out = {"cases": cases, "replicates": replicates, "seed": seed,
           "arm_a": sorted({r["arm"] for r in arm_a}), "arm_b": sorted({r["arm"] for r in arm_b}), "table": {}}
    bands = sorted(set().union(*[r["table"] for r in arm_a + arm_b]))
    leads = sorted({int(k) for r in arm_a + arm_b for b in r["table"].values() for l in b.values()
                    for k in l if k.lstrip("-").isdigit()})
    for band in bands:
        for lev in LEVELS_HPA:
            for lead in leads:
                for stat in stats:
                    diff = []
                    for c in cases:
                        a = _metric(by_case_a[c], band, lev, lead, stat)
                        b = _metric(by_case_b[c], band, lev, lead, stat)
                        diff.append(np.nan if a is None or b is None else a - b)
                    diff = np.array(diff)
                    if not np.isfinite(diff).any():
                        continue
                    out["table"].setdefault(band, {}).setdefault(str(lev), {}).setdefault(str(lead), {})[stat] = {
                        **paired_interval(diff, replicates=replicates, seed=seed),
                        "per_case": [None if not np.isfinite(d) else float(d) for d in diff]}
    out["note"] = ("fewer than four independent cases: the interval is exploratory"
                   if len(cases) < 4 else "cases are the resampled blocks")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m woof.hex.sonde_profile_referee", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--history", required=True, help="directory holding history.*.nc frames")
    s.add_argument("--sondes", choices=sorted(SOURCES), required=True)
    s.add_argument("--case", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--pairs")
    c = sub.add_parser("compare")
    c.add_argument("arm_a", nargs="+")
    c.add_argument("--against", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--replicates", type=int, default=5000)
    c.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args(argv)
    if a.command == "score":
        record = score(a.history, a.sondes, a.case, a.arm, pairs_out=a.pairs)
    else:
        load = lambda paths: [json.load(open(p, encoding="utf-8")) for p in paths]  # noqa: E731
        record = compare(load(a.arm_a), load(a.against), replicates=a.replicates, seed=a.seed)
    with open(a.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
