"""Summarise a real-data coupled-fire run from its wrfout tape.

Reads one domain's history (the fire-mesh FIRE_AREA, LFN, TIGN_G, ROS, UF,
VF and the mass-grid GRNHFX, W, T, SMOKE) and reports, per frame:

* the burned area (fire-mesh nodes with LFN <= 0) and its growth;
* the head-fire spread rate: the advance of the burned region's furthest
  extent along the mean fire-mesh wind between frames;
* the rate WRF's own fire_ros gives for that fuel, wind and slope
  (woof.fire.wrf_sfire, the WRF v4.7.1 port), for a front facing the wind:
  the expectation the observed rate is compared with;
* the atmosphere's answer: the peak fire heat flux, the strongest updraft
  and the warmest lowest-level theta anomaly over the burning columns, and
  the smoke column above the fire.

Run::

    pixi run python tools/fire_real_e2e_report.py RUN_DIR/wrfout/wrfout_d03_* \\
        --sr 10 --ignition-x 16650 --ignition-y 12450
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _frames(paths):
    from netCDF4 import Dataset

    for path in sorted(paths):
        with Dataset(path) as ds:
            ntime = ds.dimensions["Time"].size
            for t in range(ntime):
                rec = {"file": Path(path).name,
                       "time": b"".join(ds["Times"][t]).decode()}
                for name in ("FIRE_AREA", "LFN", "TIGN_G", "ROS", "UF", "VF",
                             "FGRNHFX", "GRNHFX", "W", "T", "SMOKE",
                             "SMOKE_COL", "PH", "PHB", "HGT"):
                    if name in ds.variables:
                        rec[name] = np.asarray(ds[name][t])
                rec["DX"] = float(ds.DX)
                rec["DY"] = float(ds.DY)
                yield rec


def expected_head_ros(uf, vf, nfuel_cat, dzdxf=0.0, dzdyf=0.0):
    """fire_ros for a front facing the wind (WRF v4.7.1, woof's port)."""
    from woof.fire import wrf_sfire

    speed = float(np.hypot(uf, vf))
    if speed == 0.0:
        return 0.0
    params = wrf_sfire.set_fire_params(np.array([[float(nfuel_cat)]]))
    rb, rw, rs = wrf_sfire.fire_ros(
        params, np.array([[uf]], np.float32), np.array([[vf]], np.float32),
        np.array([[dzdxf]], np.float32), np.array([[dzdyf]], np.float32),
        np.array([[uf / speed]], np.float32),
        np.array([[vf / speed]], np.float32), fire_advection=1)
    return float(np.asarray(rb + rw + rs).ravel()[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("wrfout", nargs="+")
    parser.add_argument("--sr", type=int, default=10)
    parser.add_argument("--ignition-x", type=float, required=True)
    parser.add_argument("--ignition-y", type=float, required=True)
    parser.add_argument("--fuel", type=int, default=1,
                        help="the ignition cell's Anderson category")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = [p for pattern in args.wrfout for p in glob.glob(pattern)]
    records = []
    prev = None
    for frame in _frames(paths):
        if "LFN" not in frame:
            continue
        dxf = frame["DX"] / args.sr
        dyf = frame["DY"] / args.sr
        lfn = frame["LFN"]
        burned = lfn <= 0
        nyf, nxf = lfn.shape
        xf = (np.arange(nxf) + 0.5) * dxf
        yf = (np.arange(nyf) + 0.5) * dyf
        rec = {"time": frame["time"],
               "burned_area_m2": float(burned.sum()) * dxf * dyf}
        if burned.any():
            jj, ii = np.nonzero(burned)
            near = ((np.abs(xf[None, :] - args.ignition_x) < 300.0)
                    & (np.abs(yf[:, None] - args.ignition_y) < 300.0))
            uf = float(frame["UF"][near].mean())
            vf = float(frame["VF"][near].mean())
            speed = float(np.hypot(uf, vf))
            rec["fire_wind_m_s"] = [uf, vf]
            ex, ey = (uf / speed, vf / speed) if speed > 0 else (1.0, 0.0)
            along = ((xf[ii] - args.ignition_x) * ex
                     + (yf[jj] - args.ignition_y) * ey)
            rec["head_extent_m"] = float(along.max())
            rec["back_extent_m"] = float(-along.min())
            rec["ros_max_m_s"] = float(frame["ROS"].max())
            # The rate the level set spread at on the front itself: the
            # nodes the perimeter crosses (0 < FIRE_AREA < 1).
            front = (frame["FIRE_AREA"] > 0) & (frame["FIRE_AREA"] < 1)
            if front.any():
                ros = frame["ROS"][front]
                rec["ros_front_median_m_s"] = float(np.median(ros))
                rec["ros_front_p90_m_s"] = float(np.percentile(ros, 90))
            rec["equivalent_radius_m"] = float(
                np.sqrt(rec["burned_area_m2"] / np.pi))
            rec["expected_head_ros_flat_m_s"] = expected_head_ros(
                uf, vf, args.fuel)
            if prev is not None and "head_extent_m" in prev:
                from datetime import datetime
                t0 = datetime.strptime(prev["time"], "%Y-%m-%d_%H:%M:%S")
                t1 = datetime.strptime(frame["time"], "%Y-%m-%d_%H:%M:%S")
                dt = (t1 - t0).total_seconds()
                rec["head_spread_m_s"] = (rec["head_extent_m"]
                                          - prev["head_extent_m"]) / dt
                rec["radius_growth_m_s"] = (rec["equivalent_radius_m"]
                                            - prev["equivalent_radius_m"]) / dt
            fire_cols = frame["GRNHFX"] > 0
            rec["grnhfx_max_W_m2"] = float(frame["GRNHFX"].max())
            rec["fgrnhfx_max_W_m2"] = float(frame["FGRNHFX"].max())
            w = frame["W"]
            rec["w_max_domain_m_s"] = float(w.max())
            if fire_cols.any():
                rec["w_max_over_fire_m_s"] = float(w[:, fire_cols].max())
                t0 = frame["T"][0]
                rec["theta_anomaly_k0_over_fire_K"] = float(
                    t0[fire_cols].max() - np.median(t0))
                if "SMOKE_COL" in frame:
                    rec["smoke_col_max_ug_m2"] = float(
                        frame["SMOKE_COL"].max())
        records.append(rec)
        prev = rec
        print(json.dumps(rec), flush=True)
    if args.output:
        args.output.write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
