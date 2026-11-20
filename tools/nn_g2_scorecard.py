"""The G2 observation scorecard for NN column physics: SYNOP 2 m temperature, paired by station and lead.

``python tools/nn_g2_scorecard.py --obs OBS.npz --base RUN --test RUN [--test RUN ...] --mesh GRID``

* ``OBS.npz`` is a window's observation set from ``woof grad obs --stores wis2_land`` (read-only from the
  Source Cooperative stores; keep it outside the repository).
* ``RUN`` is a forecast's history directory (or files), read by :func:`woof.grad.obs_cost.read_frames`.
  ``--base`` is the physics-only run; each ``--test`` is scored against it.

Pairing is :func:`woof.grad.obs_cost.pair` on the base run's frames (FGAT: the frame nearest each report,
the nearest usable cell, the error table's 200 m station-elevation limit).  The model's T2 is moved to the
station's elevation with a 6.5 K/km lapse rate.  A pair whose base departure exceeds ``--gross`` K is dropped.
Every arm is scored on the same pairs.

The test is a station-block bootstrap: stations are drawn with replacement (``--draws``, seed ``--seed``), all
of a drawn station's pairs come with it, and the RMSE difference (test minus base) is taken on each draw.
The report gives the RMSE of each arm, the difference and its two-sided ``--level`` interval.  A difference
above zero means the test run is worse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAPSE_K_PER_M = 0.0065


def _frames(paths, mesh):
    from woof.grad.obs_cost import read_frames

    files = []
    for value in paths:
        path = Path(value)
        files.extend(sorted(path.glob("history.*.nc")) if path.is_dir() else [path])
    if not files:
        raise SystemExit(f"no history files in {paths}")
    return read_frames([str(f) for f in files], route="mpas", mesh=mesh)


def _model_t2(obs, frames, pairing):
    from woof.grad.obs_cost import _gather

    out = np.full(len(pairing.obs_index), np.nan)
    for fi in np.unique(pairing.frame_index):
        rows = np.nonzero(pairing.frame_index == fi)[0]
        frame = frames[int(fi)]
        col = pairing.column[rows]
        t2 = _gather(frame.t2, col)
        out[rows] = t2 + LAPSE_K_PER_M * (frame.terrain[col] - obs.elevation_m[pairing.obs_index[rows]])
    return out


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def rmse(d):
    return float(np.sqrt(np.mean(np.square(d)))) if d.size else float("nan")


def station_bootstrap(station, base_err, test_err, *, draws=5000, seed=20261119, level=0.95):
    """The RMSE difference (test - base) and its station-block bootstrap interval."""
    ids, inverse = np.unique(station, return_inverse=True)
    sq_b = np.bincount(inverse, weights=base_err ** 2, minlength=ids.size)
    sq_t = np.bincount(inverse, weights=test_err ** 2, minlength=ids.size)
    n = np.bincount(inverse, minlength=ids.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, ids.size, size=(draws, ids.size))
    counts = np.zeros((draws, ids.size))
    np.add.at(counts, (np.repeat(np.arange(draws), ids.size), pick.ravel()), 1.0)
    total = counts @ n
    diff = np.sqrt((counts @ sq_t) / total) - np.sqrt((counts @ sq_b) / total)
    tail = (1.0 - level) / 2.0
    return {"difference": rmse(test_err) - rmse(base_err),
            "interval": [float(np.quantile(diff, tail)), float(np.quantile(diff, 1.0 - tail))],
            "level": level, "draws": draws, "seed": seed, "stations": int(ids.size)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--obs", required=True, help="an observation set from woof grad obs (.npz)")
    parser.add_argument("--base", nargs="+", required=True, help="the physics-only run's history")
    parser.add_argument("--test", nargs="+", action="append", required=True, metavar="RUN",
                        help="a run to score against the base (repeat for several)")
    parser.add_argument("--label", action="append", default=None, help="a name per --test, in order")
    parser.add_argument("--mesh", default=None, help="the MPAS grid/static/init file (geometry the history lacks)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--gross", type=float, default=10.0, help="drop pairs whose base departure exceeds this (K)")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args(argv)

    from woof.grad.obs_cost import ObsSet, pair, read_error_table

    obs = ObsSet.load(args.obs)
    obs = obs.take(np.nonzero(obs.kind == "surface_temperature")[0])
    base = _frames(args.base, args.mesh)
    pairing = pair(obs, base, read_error_table(), start=args.start, end=args.end)
    y = obs.value[pairing.obs_index]
    base_err = _model_t2(obs, base, pairing) - y
    good = np.isfinite(base_err) & (np.abs(base_err) <= args.gross)
    station = obs.station[pairing.obs_index]
    t0 = base[0].valid.timestamp()
    lead_h = np.array([(base[int(f)].valid.timestamp() - t0) / 3600.0 for f in pairing.frame_index])
    report = {"schema": "woof.nn.g2-scorecard/v1", "obs": str(Path(args.obs).resolve()),
              "base": [str(Path(p).resolve()) for p in args.base], "lapse_K_per_m": LAPSE_K_PER_M,
              "gross_K": args.gross, "counts": pairing.counts, "tests": {}}
    labels = args.label or [f"test{i}" for i in range(len(args.test))]
    if len(labels) != len(args.test):
        raise SystemExit(f"{len(labels)} --label for {len(args.test)} --test")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--label names repeat: {labels}")
    errors = []
    for runs in args.test:
        frames = _frames(runs, args.mesh)
        if [f.valid for f in frames] != [f.valid for f in base]:
            raise SystemExit(f"{runs}: its frames' valid times are not the base run's")
        if not np.array_equal(_host(frames[0].terrain), _host(base[0].terrain)):
            raise SystemExit(f"{runs}: its cells are not the base run's (terrain differs)")
        errors.append(_model_t2(obs, frames, pairing) - y)
    ok = good & np.all([np.isfinite(e) for e in errors], axis=0)       # every arm on the same pairs
    if not ok.any():
        raise SystemExit("no pair survives the gross check in every arm")
    for label, runs, test_err in zip(labels, args.test, errors):
        result = station_bootstrap(station[ok], base_err[ok], test_err[ok], draws=args.draws, seed=args.seed,
                                   level=args.level)
        result.update(runs=[str(Path(p).resolve()) for p in runs], pairs=int(ok.sum()),
                      rmse_base=rmse(base_err[ok]), rmse_test=rmse(test_err[ok]),
                      bias_base=float(np.mean(base_err[ok])), bias_test=float(np.mean(test_err[ok])),
                      lead_hours=[float(lead_h[ok].min()), float(lead_h[ok].max())] if ok.any() else None)
        report["tests"][label] = result
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
