"""Surface-station 10 m wind referee for MPAS-mesh history (woof hex or native MPAS-A).

Scores the history's ``u10``/``v10`` wind speed against land synoptic reports by latitude
band and lead time, so a 10 m wind bias is measured against observations on a global mesh,
where the ASOS referee (CONUS only) cannot reach.

Observations, read-only, from the Source Cooperative icechunk store ``bkr/obs/wis2_land.icechunk``
(WMO WIS2 land SYNOP reports, flat: one row per station report, from late October 2026).

A report is matched to the frame whose valid time it carries to the minute (SYNOP reports
are filed on the hour). Only anemometers at 8 to 12 m (or with no stated height, which in
SYNOP means the standard 10 m) are used, and a station whose elevation differs from its
owning cell's terrain by more than ``MAX_TERRAIN_DIFF_M`` is dropped: such a cell does not
represent the site's exposure, and the wind it carries is not the station's wind. The cell
ownership rule (full-mesh owner, then the boundary mask, then the footprint bound) is the
radiosonde referee's, so the two referees agree on which station a cell may represent.

Quality control: a station whose initial-time report departs from the initial state (which
every arm of a case shares) by more than ``INITIAL_DEPARTURE_M_S`` is dropped from every
lead of the case; the record lists the stations dropped.

Statistics: per (band, lead) the count, bias (model - obs), RMSE and the mean observed
speed; ``compare`` pairs two arms over the same cases and resamples whole cases with a
fixed seed (the radiosonde referee's ``paired_interval``).

    python -m woof.hex.station_wind_referee score --history DIR --case CASE --arm ARM --out OUT.json
    python -m woof.hex.station_wind_referee compare A1.json [A2.json ...] --against B1.json [...] \
        --out CMP.json [--replicates 5000] [--seed 20261115]
"""
from __future__ import annotations

import argparse
import json
from datetime import timezone

import numpy as np

from .sonde_profile_referee import (BANDS, BUCKET, band_of, history_frames, nearest_cells, paired_interval,
                                    unit_vectors, usable_cells)

STORE = "bkr/obs/wis2_land.icechunk"
SENSOR_HEIGHT_M = (8.0, 12.0)
SPEED_BOUNDS = (0.0, 75.0)
MAX_TERRAIN_DIFF_M = 300.0
INITIAL_DEPARTURE_M_S = 15.0
FIELDS = ("station_id", "latitude", "longitude", "station_elevation", "wind_sensor_height", "10m_wind_speed", "time")


# ----------------------------------------------------------------------------- model side
def read_frame(path):
    import netCDF4
    from datetime import datetime

    with netCDF4.Dataset(path) as d:
        v = d.variables
        stamp = "".join(c.decode() if isinstance(c, bytes) else c for c in np.ma.filled(v["xtime"][0], b" ")).strip()
        frame = {
            "path": path,
            "valid": datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc),
            "speed": np.hypot(np.asarray(v["u10"][0], np.float64), np.asarray(v["v10"][0], np.float64)),
        }
        if "latCell" in v:
            frame["lat"] = np.degrees(np.asarray(v["latCell"][:], dtype=np.float64))
            frame["lon"] = np.degrees(np.asarray(v["lonCell"][:], dtype=np.float64))
        if "ter" in v:
            frame["ter"] = np.asarray(v["ter"][:], dtype=np.float64).reshape(-1)
        if "bdyMaskCell" in v:
            frame["bdy"] = np.asarray(v["bdyMaskCell"][:])
    return frame


# ----------------------------------------------------------------------------- observations
def select_reports(table, valid_times):
    """{valid: list of (station, lat, lon, elevation, speed)} from a flat report table.

    ``table`` maps FIELDS to equal-length arrays. A report counts for a valid time when its
    time equals it to the minute; one station reporting twice for an hour (an update) keeps
    the last row. Speeds outside SPEED_BOUNDS and anemometers outside SENSOR_HEIGHT_M are
    dropped, never repaired."""
    time = np.asarray(table["time"]).astype("datetime64[m]")
    speed = np.asarray(table["10m_wind_speed"], np.float64)
    height = np.asarray(table["wind_sensor_height"], np.float64)
    lat = np.asarray(table["latitude"], np.float64)
    lon = np.asarray(table["longitude"], np.float64)
    elev = np.asarray(table["station_elevation"], np.float64)
    sid = [str(x) for x in np.asarray(table["station_id"]).tolist()]
    ok = (np.isfinite(speed) & (speed >= SPEED_BOUNDS[0]) & (speed <= SPEED_BOUNDS[1])
          & np.isfinite(lat) & np.isfinite(lon)
          & (~np.isfinite(height) | ((height >= SENSOR_HEIGHT_M[0]) & (height <= SENSOR_HEIGHT_M[1]))))
    out = {}
    for valid in valid_times:
        key = np.datetime64(valid.replace(tzinfo=None), "m")
        best = {}
        for i in np.nonzero(ok & (time == key))[0]:
            best[sid[i]] = (sid[i], float(lat[i]), float(lon[i]), float(elev[i]), float(speed[i]))
        out[valid] = sorted(best.values())
    return out


def reports_wis2(valid_times):
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(bucket=BUCKET, prefix=STORE, region="us-west-2", anonymous=True)
    ds = xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store, consolidated=False)
    times = ds["time"].values
    keys = np.array([np.datetime64(v.replace(tzinfo=None), "m") for v in valid_times])
    rows = np.nonzero(np.isin(times.astype("datetime64[m]"), keys))[0]
    if rows.size == 0:
        return {v: [] for v in valid_times}
    lo, hi = int(rows.min()), int(rows.max()) + 1
    table = {name: ds[name][lo:hi].values for name in FIELDS}
    return select_reports(table, valid_times)


# ----------------------------------------------------------------------------- scoring
def pair_frame(frame, reports, cell_xyz, usable, ter):
    """Rows (station, lat, lon, obs, model) for one frame."""
    if not reports:
        return []
    cells = nearest_cells(cell_xyz, [r[1] for r in reports], [r[2] for r in reports], usable)
    rows = []
    for (station, lat, lon, elev, obs), cell in zip(reports, cells):
        if cell < 0:
            continue
        if ter is not None and (not np.isfinite(elev) or abs(elev - ter[cell]) > MAX_TERRAIN_DIFF_M):
            continue
        rows.append((station, lat, lon, obs, float(frame["speed"][cell])))
    return rows


def initial_blacklist(pairs):
    """Stations whose initial-time report departs from the shared initial state by more than
    INITIAL_DEPARTURE_M_S: a bad report or bad metadata, dropped from every lead of the case."""
    return {p[1] for p in pairs if p[0] == 0 and abs(p[5] - p[4]) > INITIAL_DEPARTURE_M_S}


def summarise(pairs):
    """{band: {lead: {n, bias, rmse, obs_mean}}} plus each band's drift (m/s per 24 h)."""
    acc = {}
    for lead, station, lat, lon, obs, model in pairs:
        for band in band_of(lat) + ["GL"]:
            cell = acc.setdefault(band, {}).setdefault(str(lead), [0, 0.0, 0.0, 0.0])
            err = model - obs
            cell[0] += 1
            cell[1] += err
            cell[2] += err * err
            cell[3] += obs
    table = {}
    for band, leads in acc.items():
        table[band] = {lead: {"n": n, "bias": s / n, "rmse": (q / n) ** 0.5, "obs_mean": o / n}
                       for lead, (n, s, q, o) in leads.items()}
        x = np.array([float(k) for k in table[band]])
        y = np.array([table[band][k]["bias"] for k in table[band]])
        if x.size >= 2 and np.ptp(x) > 0:
            table[band]["drift_m_s_per_24h"] = float(np.polyfit(x, y, 1)[0] * 24.0)
    return table


def score(history_dir, case, arm, *, reports=None, mask_file=None):
    frames = [read_frame(p) for p in history_frames(history_dir)]
    start = min(f["valid"] for f in frames)
    hourly = [f for f in frames if f["valid"].minute == 0 and f["valid"].second == 0]
    geo = frames[0]
    if "lat" not in geo:
        raise ValueError(f"{geo['path']} carries no latCell/lonCell")
    cell_xyz = unit_vectors(geo["lat"], geo["lon"])
    usable = usable_cells(geo["lat"], geo["lon"], geo.get("bdy"), mask_file)
    if reports is None:
        reports = reports_wis2([f["valid"] for f in hourly])
    pairs = []
    for f in hourly:
        lead = int(round((f["valid"] - start).total_seconds() / 3600.0))
        for row in pair_frame(f, reports.get(f["valid"], []), cell_xyz, usable, geo.get("ter")):
            pairs.append((lead,) + row)
    rejected = initial_blacklist(pairs)
    pairs = [p for p in pairs if p[1] not in rejected]
    return {"schema": "woof.hex.station-wind-referee/v1", "case": case, "arm": arm, "store": STORE,
            "start": start.strftime("%Y-%m-%dT%H:%MZ"),
            "frames": [f["path"].rsplit("/", 1)[-1] for f in hourly],
            "reports": {f["valid"].strftime("%Y-%m-%dT%HZ"): len(reports.get(f["valid"], [])) for f in hourly},
            "pairs": len(pairs), "rejected_stations": sorted(rejected),
            "qc": {"sensor_height_m": list(SENSOR_HEIGHT_M), "max_terrain_diff_m": MAX_TERRAIN_DIFF_M,
                   "initial_departure_m_s": INITIAL_DEPARTURE_M_S},
            "table": summarise(pairs)}


# ----------------------------------------------------------------------------- comparison
def compare(arm_a, arm_b, *, replicates=5000, seed=20261115, stats=("bias", "absbias", "rmse")):
    """Case-paired difference A - B per (band, lead, stat), resampling whole cases."""
    by_a = {r["case"]: r for r in arm_a}
    by_b = {r["case"]: r for r in arm_b}
    cases = sorted(set(by_a) & set(by_b))
    if not cases:
        raise ValueError("the two arms share no case")

    def metric(record, band, lead, stat):
        cell = record["table"].get(band, {}).get(str(lead))
        if not isinstance(cell, dict) or not cell.get("n"):
            return None
        return abs(cell["bias"]) if stat == "absbias" else cell[stat]

    leads = sorted({int(k) for r in arm_a + arm_b for b in r["table"].values() for k in b if k.lstrip("-").isdigit()})
    out = {"cases": cases, "replicates": replicates, "seed": seed,
           "arm_a": sorted({r["arm"] for r in arm_a}), "arm_b": sorted({r["arm"] for r in arm_b}), "table": {}}
    for band in BANDS:
        for lead in leads:
            for stat in stats:
                diff = []
                for c in cases:
                    a, b = metric(by_a[c], band, lead, stat), metric(by_b[c], band, lead, stat)
                    diff.append(np.nan if a is None or b is None else a - b)
                diff = np.array(diff)
                if not np.isfinite(diff).any():
                    continue
                out["table"].setdefault(band, {}).setdefault(str(lead), {})[stat] = {
                    **paired_interval(diff, replicates=replicates, seed=seed),
                    "per_case": [None if not np.isfinite(d) else float(d) for d in diff]}
    out["note"] = ("fewer than four independent cases: the interval is exploratory"
                   if len(cases) < 4 else "cases are the resampled blocks")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m woof.hex.station_wind_referee", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--history", required=True, help="directory holding history.*.nc frames")
    s.add_argument("--case", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mask", help="MPAS file of the same mesh carrying bdyMaskCell (regional history without it)")
    c = sub.add_parser("compare")
    c.add_argument("arm_a", nargs="+")
    c.add_argument("--against", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--replicates", type=int, default=5000)
    c.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args(argv)
    if a.command == "score":
        record = score(a.history, a.case, a.arm, mask_file=a.mask)
    else:
        load = lambda paths: [json.load(open(p, encoding="utf-8")) for p in paths]  # noqa: E731
        record = compare(load(a.arm_a), load(a.against), replicates=a.replicates, seed=a.seed)
    with open(a.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
