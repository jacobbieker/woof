"""``woof grad``: observation cost and adjoint-free sensitivities (EXPERIMENTAL).

Reached only through the component door (:mod:`woof.components`), so no
other command imports this package.

    woof grad obs  --history RUN_OR_FILES --out OBS.npz [--stores wis2_land,...] [--goes-cwp FILE ...]
    woof grad cost --history RUN_OR_FILES --obs OBS.npz --out J.json [--pairs-out PAIRS.npz] [--accept REF.npz]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def history_files(entries) -> list[str]:
    """History files named directly or found in a directory (WRF ``wrfout_d01_*`` or MPAS ``*history*.nc``)."""
    out = []
    for entry in entries:
        if os.path.isdir(entry):
            found = sorted(glob.glob(os.path.join(entry, "wrfout_d01_*")))
            if not found:
                found = sorted(glob.glob(os.path.join(entry, "**", "*history*.nc"), recursive=True))
            if not found:
                found = sorted(glob.glob(os.path.join(entry, "**", "wrfout_d01_*"), recursive=True))
            if not found:
                raise SystemExit(f"woof grad: no wrfout_d01_* or *history*.nc under {entry}")
            out.extend(found)
        else:
            out.append(entry)
    return out


def _time(text):
    if text is None:
        return None
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _frames(args):
    from woof.grad.obs_cost import read_frames
    return read_frames(history_files(args.history), route=args.route, mesh=args.mesh, rim=args.rim)


def _add_history(parser):
    parser.add_argument("--history", nargs="+", required=True, metavar="RUN|FILE",
                        help="history files, or run directories holding them")
    parser.add_argument("--route", choices=("auto", "wrf", "mpas"), default="auto")
    parser.add_argument("--mesh", default=None, help="MPAS grid/static/init file for geometry the history lacks")
    parser.add_argument("--rim", type=int, default=5, help="WRF boundary rows left unused (spec_bdy_width)")
    parser.add_argument("--start", default=None, help="window start (ISO, UTC); default the first frame")
    parser.add_argument("--end", default=None, help="window end (ISO, UTC); default the last frame")


def command_obs(args) -> int:
    from woof.grad import obs_stores
    from woof.grad.obs_cost import ObsSet

    frames = _frames(args)
    start = _time(args.start) or frames[0].valid
    end = _time(args.end) or frames[-1].valid
    box = obs_stores.frames_box(frames, args.margin_deg)
    stores = [s for s in args.stores.split(",") if s]
    obs, counts = obs_stores.read_window(stores, start, end, box)
    for path in args.goes_cwp or ():
        part = obs_stores.goes_cwp(path)
        counts[f"goes_cwp:{Path(path).name}"] = len(part)
        obs = ObsSet.concat([obs, part])
    obs.save(args.out)
    receipt = {"schema": "woof.grad.obs-window/v1", "start": start.isoformat(), "end": end.isoformat(),
               "box_south_north_west_east": list(box), "stores": counts, "rows": len(obs),
               "obs_sha256": obs.digest(), "file": str(args.out)}
    print(json.dumps(receipt, indent=2))
    return 0


def command_cost(args) -> int:
    from woof.grad.obs_cost import ObsSet, accepted_from, evaluate, read_error_table

    frames = _frames(args)
    obs = ObsSet.load(args.obs)
    table = read_error_table(args.error_table)
    accept = accepted_from(args.accept) if args.accept else None
    result = evaluate(obs, frames, table=table, start=_time(args.start), end=_time(args.end),
                      max_offset_s=None if args.max_offset_minutes is None else 60.0 * args.max_offset_minutes,
                      accept=accept)
    summary = result.summary()
    summary["error_table"] = table.path
    summary["accepted_from"] = args.accept
    if args.pairs_out:
        result.save_pairs(args.pairs_out)
        summary["pairs"] = str(args.pairs_out)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text if not args.quiet else f"J = {result.J:.6g} over {result.n} observations")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="woof grad", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    obs = sub.add_parser("obs", help="read a forecast window's observations from the bkr/obs stores into one file")
    _add_history(obs)
    obs.add_argument("--stores", default="wis2_land,wis2_upper_air,wis2_aircraft,metop_sg_ro",
                     help="comma-separated stores: wis2_land, wis2_upper_air, wis2_aircraft, metop_sg_ro")
    obs.add_argument("--goes-cwp", nargs="*", default=None, metavar="FILE",
                     help="gpuwm-obs.goes-grid.v1 files to add as cloud water path observations")
    obs.add_argument("--margin-deg", type=float, default=0.0)
    obs.add_argument("--out", required=True, help="the observation set (.npz); keep it outside the repository")
    obs.set_defaults(handler=command_obs)

    cost = sub.add_parser("cost", help="the observation cost J_o of a forecast against an observation set")
    _add_history(cost)
    cost.add_argument("--obs", required=True, help="an observation set written by woof grad obs")
    cost.add_argument("--error-table", default=None, help="a woof.grad.obs-cost-types/v1 table")
    cost.add_argument("--max-offset-minutes", type=float, default=None,
                      help="FGAT pairing distance; default half the history interval")
    cost.add_argument("--accept", default=None,
                      help="a reference run's pairs file: score exactly its accepted observations, no background check")
    cost.add_argument("--pairs-out", default=None, help="write the accepted pairs (.npz) for later --accept")
    cost.add_argument("--out", default=None, help="write the summary JSON here")
    cost.add_argument("--quiet", action="store_true")
    cost.set_defaults(handler=command_cost)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
