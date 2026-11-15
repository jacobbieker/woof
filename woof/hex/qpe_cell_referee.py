"""24 h precipitation referee for MPAS-layout history against gridded QPE (Stage-IV).

The observation grid is far finer than a global or regional hex cell (4.8 km
Stage-IV pixels against 15-120 km cells), so the comparison is made on the
model's own cells: every valid observation pixel is assigned to its nearest
cell and the cell's observed accumulation is the mean of its pixels.  A cell
is scored only when it is fully inside the observed area (its pixel count at
least ``COVERAGE`` of the median full cell and at least ``VALID`` of its
pixels valid), so a cell on the radar edge is not compared with a partial mean.

The model accumulation is ``rainc + rainnc`` (whichever are present) differenced
between two history frames whose valid times bracket the observation window.

Per window: cell count, model and observed means, bias, RMSE, and the frequency
bias and equitable threat score at fixed thresholds.  ``compare`` is the same
case-paired block bootstrap the sonde referee uses: whole cases are resampled.

    python -m woof.hex.qpe_cell_referee score --history DIR --qpe DIR --case C --arm A --out A.json
    python -m woof.hex.qpe_cell_referee compare A1.json A2.json --against B1.json B2.json --out d.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from woof.hex.sonde_profile_referee import paired_interval

THRESHOLDS_MM = (1.0, 5.0, 10.0, 25.0)
COVERAGE = 0.7
VALID = 0.9
STATS = ("bias", "absbias", "rmse", "ratio", "fbi_1", "fbi_10", "ets_1", "ets_10")


def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


# A pixel belongs to a cell only inside that cell's footprint: within
# FOOTPRINT times the distance to the cell's nearest neighbour. A regular
# hexagon's circumradius is 1/sqrt(3) = 0.577 of the centre spacing; 0.75
# leaves room for the irregular cells of a variable-resolution mesh.
FOOTPRINT = 0.75


def assign_pixels(cell_lat, cell_lon, pixel_lat, pixel_lon, usable=None):
    """Owning (usable) cell of every pixel; -1 for a pixel outside every usable cell's footprint."""
    from scipy.spatial import cKDTree

    xyz = unit_vectors(cell_lat, cell_lon)
    cells = np.arange(len(xyz)) if usable is None else np.flatnonzero(usable)
    pixels = unit_vectors(np.ravel(pixel_lat), np.ravel(pixel_lon))
    if cells.size == 0:
        return np.full(len(pixels), -1, dtype=np.int64)
    if len(xyz) < 2:
        raise ValueError("a footprint needs at least two model cells")
    spacing = cKDTree(xyz).query(xyz[cells], k=2)[0][:, 1]
    distance, index = cKDTree(xyz[cells]).query(pixels)
    owner = cells[index].astype(np.int64)
    owner[distance > FOOTPRINT * spacing[index]] = -1
    return owner


def cell_means(assignment, values, valid, ncells):
    """``(mean, scored)`` per cell: the valid-pixel mean, and whether the cell is fully observed."""
    assignment = np.ravel(assignment)
    inside = assignment >= 0
    assignment = assignment[inside]
    valid = np.ravel(valid).astype(bool)[inside]
    values = np.ravel(values).astype(np.float64)[inside]
    total = np.bincount(assignment, minlength=ncells).astype(np.float64)
    count = np.bincount(assignment[valid], minlength=ncells).astype(np.float64)
    summed = np.bincount(assignment[valid], weights=values[valid], minlength=ncells)
    touched = total > 0
    full = np.median(total[touched]) if touched.any() else 0.0
    scored = touched & (total >= COVERAGE * full) & (count >= VALID * total) & (count > 0)
    mean = np.full(ncells, np.nan)
    mean[count > 0] = summed[count > 0] / count[count > 0]
    return mean, scored


def contingency(model, obs, threshold):
    hits = float(np.sum((model >= threshold) & (obs >= threshold)))
    false = float(np.sum((model >= threshold) & (obs < threshold)))
    misses = float(np.sum((model < threshold) & (obs >= threshold)))
    n = float(model.size)
    fbi = (hits + false) / (hits + misses) if hits + misses > 0 else None
    chance = (hits + false) * (hits + misses) / n if n else 0.0
    denom = hits + false + misses - chance
    ets = (hits - chance) / denom if denom > 0 else None
    return fbi, ets


def window_stats(model, obs):
    diff = model - obs
    out = {"cells": int(model.size), "model_mean": float(model.mean()), "obs_mean": float(obs.mean()),
           "bias": float(diff.mean()), "rmse": float(np.sqrt(np.mean(diff ** 2)))}
    out["absbias"] = abs(out["bias"])
    out["ratio"] = out["model_mean"] / out["obs_mean"] if out["obs_mean"] > 0 else None
    for t in THRESHOLDS_MM:
        fbi, ets = contingency(model, obs, t)
        out[f"fbi_{t:g}"], out[f"ets_{t:g}"] = fbi, ets
    return out


def read_accumulation(path):
    """``(valid time, lat, lon, accumulated precipitation mm, bdyMaskCell or None)`` of one frame."""
    import netCDF4

    with netCDF4.Dataset(path) as d:
        v = d.variables
        stamp = "".join(c.decode() if isinstance(c, bytes) else c for c in np.ma.filled(v["xtime"][0], b" ")).strip()
        valid = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        names = [n for n in ("rainc", "rainnc") if n in v]
        if not names:
            raise KeyError(f"{path} carries neither rainc nor rainnc")
        total = sum(np.asarray(v[n][0], dtype=np.float64) for n in names)
        lat = np.degrees(np.asarray(v["latCell"][:], dtype=np.float64)) if "latCell" in v else None
        lon = np.degrees(np.asarray(v["lonCell"][:], dtype=np.float64)) if "lonCell" in v else None
        bdy = np.asarray(v["bdyMaskCell"][:]) if "bdyMaskCell" in v else None
    return valid, lat, lon, total, bdy


def qpe_windows(folder, hours):
    """``{end valid time: pack path}`` of the decoded ``hours``-hour QPE packs under ``folder``."""
    from woof.obs.obspack import read_pack

    found = {}
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.obspack"), recursive=True)):
        meta = read_pack(path).meta
        if meta.get("schema") != "gpuwm-obs.obs-grid.v1" or int(meta.get("accumulation_hours", -1)) != hours:
            continue
        end = datetime.fromisoformat(str(meta["valid_time"])).replace(tzinfo=timezone.utc)
        found[end] = path
    return found


def score(history_dir, qpe_dir, case, arm, *, hours=24):
    from woof.obs.obspack import read_geo_pack, read_grid_pack

    frames = {}
    lat = lon = bdy = None
    for path in sorted(glob.glob(os.path.join(history_dir, "*history.*.nc"))):
        valid, la, lo, total, b = read_accumulation(path)
        frames[valid] = total
        if la is not None:
            lat, lon, bdy = la, lo, b
    if not frames or lat is None:
        raise FileNotFoundError(f"no history frame with cell coordinates under {history_dir}")
    windows = qpe_windows(qpe_dir, hours)
    geometry = sorted(glob.glob(os.path.join(qpe_dir, "**", "geometry.obspack"), recursive=True))
    if not geometry:
        raise FileNotFoundError(f"no geometry.obspack under {qpe_dir}")
    plat, plon = read_geo_pack(geometry[0])
    usable = None if bdy is None else (bdy == 0)
    assignment = assign_pixels(lat, lon, plat, plon, usable)
    init = min(frames)
    record = {"case": case, "arm": arm, "hours": hours, "init": init.isoformat(), "windows": {}}
    for end, pack_path in sorted(windows.items()):
        start = end - timedelta(hours=hours)
        if start not in frames or end not in frames:
            continue
        pack = read_grid_pack(pack_path)
        obs, scored = cell_means(assignment, pack.array("values"), pack.array("valid"), len(lat))
        model = frames[end] - frames[start]
        keep = scored & np.isfinite(obs) & np.isfinite(model)
        if not keep.any():
            continue
        lead = int((end - init).total_seconds() // 3600)
        stats = window_stats(model[keep], obs[keep])
        stats["end"] = end.isoformat()
        record["windows"][str(lead)] = stats
    if not record["windows"]:
        raise ValueError("no QPE window is bracketed by two history frames")
    return record


def _case_metric(record, stat, leads):
    values = [record["windows"][k].get(stat) for k in leads]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _shared_windows(a, b):
    """Window keys both arms scored over the same window (same init, hours and end)."""
    if a["init"] != b["init"] or a["hours"] != b["hours"]:
        raise ValueError(f"case {a['case']!r}: the arms differ in init or window length "
                         f"({a['init']}/{a['hours']} h against {b['init']}/{b['hours']} h)")
    shared = sorted(set(a["windows"]) & set(b["windows"]), key=int)
    for k in shared:
        if a["windows"][k].get("end") != b["windows"][k].get("end"):
            raise ValueError(f"case {a['case']!r}: window {k} ends at different times in the two arms")
    return shared


def compare(arm_a, arm_b, *, replicates=5000, seed=20261115, stats=STATS):
    """Case-paired A - B per stat (each case's windows averaged first), resampling whole cases."""
    by_a = {r["case"]: r for r in arm_a}
    by_b = {r["case"]: r for r in arm_b}
    cases = sorted(set(by_a) & set(by_b))
    if not cases:
        raise ValueError("the two arms share no case")
    shared = {c: _shared_windows(by_a[c], by_b[c]) for c in cases}
    out = {"cases": cases, "replicates": replicates, "seed": seed,
           "arm_a": sorted({r["arm"] for r in arm_a}), "arm_b": sorted({r["arm"] for r in arm_b}),
           "windows": shared, "stats": {}}
    for stat in stats:
        a = np.array([np.nan if (x := _case_metric(by_a[c], stat, shared[c])) is None else x for c in cases])
        b = np.array([np.nan if (x := _case_metric(by_b[c], stat, shared[c])) is None else x for c in cases])
        diff = a - b
        pair = np.isfinite(diff)
        if not pair.any():
            continue
        out["stats"][stat] = {
            "a": float(a[pair].mean()), "b": float(b[pair].mean()),
            **paired_interval(diff, replicates=replicates, seed=seed),
            "per_case": [None if not np.isfinite(d) else float(d) for d in diff]}
    out["note"] = ("fewer than four independent cases: the interval is exploratory"
                   if len(cases) < 4 else "cases are the resampled blocks")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m woof.hex.qpe_cell_referee", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--history", required=True, help="directory holding history.*.nc frames")
    s.add_argument("--qpe", required=True, help="directory holding decoded QPE packs and geometry.obspack")
    s.add_argument("--hours", type=int, default=24)
    s.add_argument("--case", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("arm_a", nargs="+")
    c.add_argument("--against", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--replicates", type=int, default=5000)
    c.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args(argv)
    if a.command == "score":
        result = score(a.history, a.qpe, a.case, a.arm, hours=a.hours)
    else:
        def load(paths):
            return [json.load(open(p, encoding="utf-8")) for p in paths]
        result = compare(load(a.arm_a), load(a.against), replicates=a.replicates, seed=a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
