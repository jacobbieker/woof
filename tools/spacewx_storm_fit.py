"""Fit and grade the storm-time ionosphere correction (woof.spacewx.storm).

Input: CODE's final global ionosphere maps, one IONEX file a day
(AIUB, University of Bern; ``http://ftp.aiub.unibe.ch/CODE/YYYY/``, served from
``https://zhw-b.s3.cloud.switch.ch/aiub/CODE/YYYY/``: ``CODGDDD0.YYI.Z`` up to
day 330 of 2022, ``COD0OPSFIN_YYYYDDD0000_01D_01H_GIM.INX.gz`` after), and
GFZ's Kp/ap file.

For every 2-hourly map, at nodes every 5 deg of latitude (|lat| <= 85) and
10 deg of longitude:
- the quiet reference TEC_q is the median of the 27 days before at the same UT
  and node (at least 20 of them), and the target is y = log(TEC / TEC_q) where
  TEC_q >= 3 TECU;
- the driver is Wrenn's filtered ap of the intervals ended by the map's epoch,
  for each candidate tau.

The training years give least-squares coefficients (a small ridge keeps
empty corners at 0) for each tau and for two variants, log R linear in
x = ap_tau / 100, or linear and quadratic.  A nuisance term in the day
before's F10.7 over its 27-day mean, per |MLAT| node, is fitted alongside
and not shipped: it keeps a flux change that comes with a storm out of the
ap coefficients.  The variant and tau are chosen by leave-one-year-out RMSE
on the rows with ap_tau >= 30 for tau = 0.8, the same rows for every
candidate (the training RMSE favours the slowest filter,
which follows the solar rotation rather than storms), then refitted on all
training years.  The test years are then graded against the zero correction (TEC =
TEC_q): RMSE of y, and of TEC in TECU, overall and when ap_tau (tau = 0.8)
>= 30 and >= 80, by |MLAT| band.

    python -m tools.spacewx_storm_fit --gim DIR --indices Kp_ap_Ap_SN_F107_since_1932.txt \\
        --train 2015 2017 2023 --test 2024 2025 --out storm_fit_v1.json --json stats.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys

import numpy as np

TAUS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98)
NLAG = 40                                   # 120 h
MLAT_NODES = (0.0, 20.0, 35.0, 50.0, 62.5, 75.0)
LT_NODES = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
SEASON_NODES = (-1.0, 0.0, 1.0)
QUIET_DAYS, QUIET_MIN = 27, 20
TEC_Q_MIN = 3.0
RIDGE = 1.0e-3
BANDS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 90.0))
STORM_LEVELS = (0.0, 30.0, 80.0)
STORM_SELECT = 30.0                         # the rows that choose the variant and tau
#: The filter whose ap_tau marks storm rows, for every candidate tau alike
#: (selecting on each tau's own ap_tau would score them on different rows).
REF_TAU = 0.8
X_CAP = 4.0                                 # ap_tau 400: beyond any record


def gim_files(root: Path, year: int) -> list[tuple[int, Path]]:
    out = []
    for f in sorted(root.iterdir()):
        m = re.fullmatch(r"CODG(\d{3})0\.(\d\d)I\.Z", f.name) or re.fullmatch(
            r"COD0OPSFIN_(\d{4})(\d{3})0000_01D_01H_GIM\.INX\.gz", f.name)
        if not m:
            continue
        if f.name.startswith("CODG"):
            doy, yy = int(m.group(1)), int(m.group(2))
            y = 2000 + yy
        else:
            y, doy = int(m.group(1)), int(m.group(2))
        if y == year:
            out.append((doy, f))
    return sorted(out)


def epochs_of(path: Path):
    """(epoch, lat (nlat,), lon (nlon,), tec (nlat, nlon)) for the 2-hourly maps of
    the file's own day, on the 5 x 10 deg node grid."""
    from tools.spacewx_gim_validate import read_ionex
    d = read_ionex(path)
    lat, lon = d["lat"], d["lon"]
    li = np.flatnonzero((np.abs(np.mod(lat, 5.0)) < 1e-6) & (np.abs(lat) <= 85.0))
    lj = np.flatnonzero((np.abs(np.mod(lon, 10.0)) < 1e-6) & (lon < 180.0))
    day = d["epochs"][0].date()
    for epoch, grid in zip(d["epochs"], d["tec"]):
        if epoch.date() != day or epoch.minute or epoch.hour % 2:
            continue
        yield epoch, lat[li], lon[lj], grid[np.ix_(li, lj)]


def rows(gim: Path, years, geo, taus=TAUS, model=None):
    """Per usable map: (epoch, mlat, lt, season, y, tec, tec_q, {tau: ap_tau}) on its
    nodes with a quiet reference."""
    from woof.spacewx import storm
    for year in years:
        history: dict[int, deque] = defaultdict(lambda: deque(maxlen=QUIET_DAYS))
        last_day = None
        for doy, path in gim_files(gim, year):
            day = datetime(year, 1, 1) + timedelta(days=doy - 1)
            if last_day is not None and (day - last_day).days > 1:
                history.clear()                      # a gap: the reference restarts
            last_day = day
            try:
                maps = list(epochs_of(path))
            except Exception as exc:                 # a damaged file is skipped, and said
                print(f"skip {path.name}: {exc}", file=sys.stderr)
                history.clear()
                continue
            for epoch, lat, lon, tec in maps:
                lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
                past = history[epoch.hour]
                if len(past) >= QUIET_MIN:
                    with np.errstate(all="ignore"):
                        tec_q = np.nanmedian(np.stack(past), axis=0)
                    ok = np.isfinite(tec) & np.isfinite(tec_q) & (tec_q >= TEC_Q_MIN) & (tec > 0.1)
                    try:
                        aps = {t: storm.filtered_ap(geo, epoch, t, NLAG) for t in taus}
                    except storm.StormError:
                        ok[:] = False
                    if ok.any():
                        mlat = storm.magnetic_latitude(lat2[ok], lon2[ok], epoch, model)
                        yield (epoch, mlat, storm.local_time(lon2[ok], epoch),
                               storm.season_coordinate(epoch, mlat), np.log(tec[ok] / tec_q[ok]),
                               tec[ok], tec_q[ok], aps)
                past.append(tec)


def _design(mlat, lt, season, x, solar):
    """Columns: every a, every b, then the solar nuisance per |MLAT| node."""
    from woof.spacewx import storm
    b = storm.basis((MLAT_NODES, LT_NODES, SEASON_NODES), mlat, lt, season)
    wm = storm._hats(np.abs(mlat), MLAT_NODES)
    return np.concatenate([b * x, b * x * x, wm * solar], axis=1)


def solar_anomaly(geo, epoch: datetime) -> float:
    """log(F10.7 of the day before / its mean over the 27 days before): the
    flux the quiet reference has not seen.  A nuisance term of the fit, so that
    an ap coefficient does not take up a flux change that comes with a storm."""
    day = geo._row((epoch - timedelta(days=1)).date())
    window = geo.f107[max(day - 26, 0):day + 1]
    window = window[np.isfinite(window) & (window > 0)]
    f = geo.f107[day]
    return float(np.log(f / window.mean())) if window.size >= 20 and np.isfinite(f) and f > 0 else 0.0


def _solve(xtx, xty, cols):
    xtx, xty = xtx[np.ix_(cols, cols)], xty[cols]
    lam = RIDGE * np.trace(xtx) / xtx.shape[0]
    return np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), xty)


def _sse(c, cols, xtx, xty, yy):
    return float(yy - 2.0 * c @ xty[cols] + c @ xtx[np.ix_(cols, cols)] @ c)


def fit(gim: Path, years, geo, model=None) -> dict:
    """Normal equations per training year, tau and subset (all rows; rows with
    ap_tau >= STORM_SELECT) for the full design; each variant (linear, or
    linear and quadratic in x) and tau is scored by leave-one-year-out RMSE on
    the storm rows, and the best is refitted on every training year."""
    k = len(MLAT_NODES) * len(LT_NODES) * len(SEASON_NODES)
    nm = len(MLAT_NODES)
    width = 2 * k + nm
    variants = {"linear": np.r_[np.arange(k), 2 * k + np.arange(nm)], "quadratic": np.arange(width)}

    def blank():
        return [np.zeros((width, width)), np.zeros(width), 0.0, 0]
    acc = {(y, t, sub): blank() for y in years for t in TAUS for sub in ("all", "storm")}
    x_max, ys = dict.fromkeys(TAUS, 0.0), []
    for epoch, mlat, lt, season, y, _tec, _tq, aps in rows(gim, years, geo, model=model):
        ys.append(np.percentile(y, (0.1, 99.9)))
        solar = solar_anomaly(geo, epoch)
        stormy = aps[REF_TAU] >= STORM_SELECT
        for t, ap in aps.items():
            x = min(ap / 100.0, X_CAP)
            a = _design(mlat, lt, season, x, solar)
            ata, aty, yy = a.T @ a, a.T @ y, float(y @ y)
            for sub in ("all", "storm") if stormy else ("all",):
                h = acc[(epoch.year, t, sub)]
                h[0] += ata
                h[1] += aty
                h[2] += yy
                h[3] += y.size
            x_max[t] = max(x_max[t], x)

    def total(t, sub, skip=None):
        parts = [acc[(y, t, sub)] for y in years if y != skip]
        return [sum(p[i] for p in parts) for i in range(4)]
    scores = {}
    for name, cols in variants.items():
        for t in TAUS:
            sse_cv = n_cv = 0.0
            for held in years if len(years) > 1 else ():
                c = _solve(*total(t, "all", skip=held)[:2], cols)
                xtx, xty, yy, n = acc[(held, t, "storm")]
                sse_cv += _sse(c, cols, xtx, xty, yy)
                n_cv += n
            xtx, xty, yy, n = total(t, "all")
            c = _solve(xtx, xty, cols)
            s_all = total(t, "storm")
            scores[(name, t)] = {
                "coef": c,
                "rmse_log_cv_storm": float(np.sqrt(max(sse_cv, 0.0) / n_cv)) if n_cv else None,
                "rmse_log_train": float(np.sqrt(max(_sse(c, cols, xtx, xty, yy), 0.0) / max(n, 1))),
                "rmse_log_train_zero": float(np.sqrt(yy / max(n, 1))),
                "rmse_log_train_storm_zero": float(np.sqrt(s_all[2] / max(s_all[3], 1))),
                "rows": int(n), "storm_rows": int(s_all[3])}
    key = "rmse_log_cv_storm" if len(years) > 1 else "rmse_log_train"
    best = min(scores, key=lambda v: scores[v][key])
    name, tau = best
    c = np.zeros(width)
    c[variants[name]] = scores[best]["coef"]
    coef = np.stack([c[:k], c[k:2 * k]], axis=-1).reshape(len(MLAT_NODES), len(LT_NODES), len(SEASON_NODES), 2)
    lo = float(np.exp(np.min([v[0] for v in ys]))) if ys else 0.25
    hi = float(np.exp(np.max([v[1] for v in ys]))) if ys else 2.5
    return {"tau": tau, "nlag": NLAG, "mlat_nodes": list(MLAT_NODES), "lt_nodes": list(LT_NODES),
            "season_nodes": list(SEASON_NODES), "coef": np.round(coef, 6).tolist(),
            "x_max": round(x_max[tau], 4), "ratio_range": [round(max(lo, 0.1), 3), round(min(hi, 10.0), 3)],
            "variant": name, "solar_nuisance": np.round(c[2 * k:], 6).tolist(),
            "training": {f"{v} {t}": {kk: vv for kk, vv in sc.items() if kk != "coef"}
                         for (v, t), sc in scores.items()}}


def grade(gim: Path, years, geo, fit_obj, model=None) -> dict:
    from woof.spacewx import storm
    cells = defaultdict(lambda: np.zeros(5))        # n, sse_log fit, sse_log zero, sse_tec fit, sse_tec zero
    for _epoch, mlat, lt, season, y, tec, tq, aps in rows(gim, years, geo, taus=(fit_obj.tau, REF_TAU),
                                                            model=model):
        r = storm.nmf2_factor(fit_obj, mlat, lt, season, aps[fit_obj.tau])
        lr = np.log(r)
        for lo_ap in STORM_LEVELS:              # binned by the common filter, whatever the fit's tau
            if aps[REF_TAU] < lo_ap:
                continue
            for lo, hi in BANDS + ((0.0, 90.0),):
                sel = (np.abs(mlat) >= lo) & (np.abs(mlat) < hi)
                if not sel.any():
                    continue
                c = cells[(lo_ap, lo, hi)]
                c += [sel.sum(), ((lr - y)[sel] ** 2).sum(), (y[sel] ** 2).sum(),
                      ((tq * r - tec)[sel] ** 2).sum(), ((tq - tec)[sel] ** 2).sum()]
    out = []
    for (lo_ap, lo, hi), c in sorted(cells.items()):
        n = c[0]
        out.append({"ap_tau_min": lo_ap, "mlat": [lo, hi], "nodes": int(n),
                    "rmse_log_fit": float(np.sqrt(c[1] / n)), "rmse_log_quiet": float(np.sqrt(c[2] / n)),
                    "rmse_tecu_fit": float(np.sqrt(c[3] / n)), "rmse_tecu_quiet": float(np.sqrt(c[4] / n))})
    return {"years": list(years), "cells": out}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.spacewx_storm_fit", description=__doc__.split("\n")[0])
    parser.add_argument("--gim", type=Path, required=True, help="directory of CODE daily IONEX files")
    parser.add_argument("--indices", required=True, help="GFZ's Kp/ap file")
    parser.add_argument("--train", type=int, nargs="+", default=[])
    parser.add_argument("--test", type=int, nargs="*", default=[])
    parser.add_argument("--fit", type=Path, help="grade this fit instead of fitting")
    parser.add_argument("--out", type=Path, help="write the fit here")
    parser.add_argument("--json", type=Path, help="write the statistics here")
    args = parser.parse_args(argv)
    from woof.spacewx import igrf, indices, storm
    geo = indices.read(args.indices)
    model = igrf.load()
    doc = {"gim": "CODE final GIM (AIUB)", "indices": f"{geo.source} sha256:{geo.sha256}"}
    if args.fit is None:
        if not args.train:
            parser.error("--train years, or --fit")
        made = fit(args.gim, args.train, geo, model)
        made["record"] = {"train_years": args.train, "test_years": args.test, "gim": doc["gim"],
                          "indices_sha256": geo.sha256, "tool": "tools/spacewx_storm_fit.py",
                          "source": "CODE final global ionosphere maps (AIUB, University of Bern), daily IONEX",
                          "indices": "GFZ Potsdam Kp/ap (CC BY 4.0)",
                          "note": "woof own fit: tools/spacewx_storm_fit.py; selection by leave-one-year-out "
                                  "RMSE on storm rows (ap_tau(0.8) >= 30)"}
        out = args.out or Path("storm_fit.json")
        out.write_text(json.dumps(made, indent=1) + "\n", encoding="utf-8")
        doc["fit"] = {k: made[k] for k in ("tau", "variant", "x_max", "ratio_range", "solar_nuisance", "training")}
        print(json.dumps(doc["fit"], indent=1))
        fit_path = out
    else:
        fit_path = args.fit
    fitted = storm.load_fit(fit_path)
    if args.test:
        doc["test"] = grade(args.gim, args.test, geo, fitted, model)
        for c in doc["test"]["cells"]:
            print(f"ap_tau>={c['ap_tau_min']:4.0f} |MLAT| {c['mlat'][0]:4.0f}-{c['mlat'][1]:3.0f}: "
                  f"{c['nodes']:9d} nodes  RMSE log {c['rmse_log_fit']:.3f} (quiet ref {c['rmse_log_quiet']:.3f})  "
                  f"TECU {c['rmse_tecu_fit']:6.2f} (quiet ref {c['rmse_tecu_quiet']:6.2f})")
    if args.json is not None:
        args.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
