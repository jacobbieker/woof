"""``woof grad``: observation cost and adjoint-free sensitivities (EXPERIMENTAL).

Reached only through the component door (:mod:`woof.components`), so no
other command imports this package.

    woof grad obs  --history RUN_OR_FILES --out OBS.npz [--stores wis2_land,...] [--goes-cwp FILE ...]
    woof grad cost --history RUN_OR_FILES --obs OBS.npz --out J.json [--pairs-out PAIRS.npz] [--accept REF.npz]
    woof grad direction --checkpoint CKPT --field thp --blob K J I --radius-cells R --depth-levels D --out DIR.npz
    woof grad direction --mesh GRID.nc --field theta --at LAT LON LEVEL --radius-km R --out DIR.npz
    woof grad perturb --checkpoint CKPT|STATE.nc --direction DIR.npz --scale EPS --out PERTURBED
    woof grad fd --leg EPS PLUS.json MINUS.json [--leg ...] [--reference J0.json] [--zero JZ.json]
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
    from woof.grad.obs_cost import ObsSet, history_extent

    lat, lon, first, last = history_extent(history_files(args.history), route=args.route, mesh=args.mesh)
    start = _time(args.start) or first
    end = _time(args.end) or last
    box = obs_stores.box_of(lat, lon, args.margin_deg)
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
    summary["accepted_from"] = str(Path(args.accept).resolve()) if args.accept else None
    if args.accept:
        import hashlib
        summary["accepted_sha256"] = hashlib.sha256(Path(args.accept).read_bytes()).hexdigest()
    if args.pairs_out:
        result.save_pairs(args.pairs_out)
        summary["pairs"] = str(args.pairs_out)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text if not args.quiet else f"J = {result.J:.6g} over {result.n} observations")
    return 0


def command_direction(args) -> int:
    from woof.grad.sensitivity import (cell_blob, checkpoint_fields, gaussian_blob, load_direction, mesh_cells,
                                       save_direction)

    fields = {}
    if args.add_to:
        fields, _meta = load_direction(args.add_to)
    if args.blob is not None:
        if not args.checkpoint:
            raise SystemExit("woof grad direction: --blob needs --checkpoint")
        shape = checkpoint_fields(args.checkpoint, [args.field])[args.field].shape
        k, j, i = args.blob
        blob = gaussian_blob(shape, k=k, j=j, i=i, radius_cells=args.radius_cells,
                             depth_levels=args.depth_levels, amplitude=args.amplitude)
        added = {"field": args.field, "kji": [k, j, i], "radius_cells": args.radius_cells}
    else:
        if not args.mesh:
            raise SystemExit("woof grad direction: --at needs --mesh")
        cells = mesh_cells(args.mesh)
        nz = args.levels or cells["nz"]
        if not nz:
            raise SystemExit("woof grad direction: the mesh file has no nVertLevels; pass --levels")
        interior = None if cells["bdy"] is None else cells["bdy"] == 0
        lat0, lon0, level = args.at
        blob = cell_blob(cells["lat"], cells["lon"], nz, lat0=lat0, lon0=lon0, level=level,
                         radius_km=args.radius_km, depth_levels=args.depth_levels, amplitude=args.amplitude,
                         interior=interior)
        added = {"field": args.field, "lat_lon_level": [lat0, lon0, level], "radius_km": args.radius_km}
    fields[args.field] = fields.get(args.field, 0.0) + blob
    added.update(depth_levels=args.depth_levels, amplitude=args.amplitude)
    sha = save_direction(args.out, fields, {"kind": "gaussian_blob", "source": str(args.checkpoint or args.mesh),
                                             "added": added, "added_to": args.add_to})
    print(json.dumps({"direction": str(args.out), "sha256": sha, "fields": sorted(fields)}, indent=2))
    return 0


def command_perturb(args) -> int:
    from woof.grad.sensitivity import load_direction, perturb_checkpoint, perturb_mpas_file

    fields, meta = load_direction(args.direction)
    if str(args.checkpoint).endswith(".nc"):
        receipt = perturb_mpas_file(args.checkpoint, fields, args.scale, args.out)
    else:
        receipt = perturb_checkpoint(args.checkpoint, fields, args.scale, args.out, mp_physics=args.mp_physics)
    receipt["direction"] = str(args.direction)
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    if args.receipt:
        Path(args.receipt).write_text(text + "\n")
    print(text if not args.quiet else f"wrote {args.out} (scale {args.scale:g}, direction {meta['sha256'][:12]})")
    return 0


def command_fd(args) -> int:
    from woof.grad.sensitivity import fd_summary

    legs = [(float(scale), plus, minus) for scale, plus, minus in args.leg]
    result = fd_summary(legs, reference=args.reference, zero=args.zero)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
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

    direction = sub.add_parser("direction", help="write a perturbation direction: a Gaussian blob on one state field")
    where = direction.add_mutually_exclusive_group(required=True)
    where.add_argument("--blob", nargs=3, type=float, metavar=("K", "J", "I"),
                       help="WRF route: centre in the field's own (level, row, column) indices; needs --checkpoint")
    where.add_argument("--at", nargs=3, type=float, metavar=("LAT", "LON", "LEVEL"),
                       help="hex: centre in degrees and level; needs --mesh")
    direction.add_argument("--checkpoint", default=None, help="a WRF-route restart checkpoint (gives the field's shape)")
    direction.add_argument("--mesh", default=None, help="hex: a grid, static or init file of the mesh")
    direction.add_argument("--levels", type=int, default=None, help="hex: level count when the mesh file has none")
    direction.add_argument("--field", required=True, help="state field, e.g. thp, u, v, qv (WRF) or theta (hex)")
    direction.add_argument("--radius-cells", type=float, default=4.0)
    direction.add_argument("--radius-km", type=float, default=30.0)
    direction.add_argument("--depth-levels", type=float, default=3.0)
    direction.add_argument("--amplitude", type=float, default=1.0, help="peak value, in the field's unit")
    direction.add_argument("--add-to", default=None, help="an existing direction file to add this blob to")
    direction.add_argument("--out", required=True)
    direction.set_defaults(handler=command_direction)

    perturb = sub.add_parser("perturb", help="write CHECKPOINT + SCALE * DIRECTION as a new restart state")
    perturb.add_argument("--checkpoint", required=True,
                         help="a WRF-route restart checkpoint (.npz) or a hex MPAS-layout state file (.nc)")
    perturb.add_argument("--direction", required=True)
    perturb.add_argument("--scale", type=float, required=True, help="signed multiple of the direction")
    perturb.add_argument("--mp-physics", type=int, default=None, help="the run's mp_physics, for the moment guard")
    perturb.add_argument("--out", required=True, help="the perturbed checkpoint (a new file)")
    perturb.add_argument("--receipt", default=None, help="write the writer's receipt JSON here")
    perturb.add_argument("--quiet", action="store_true")
    perturb.set_defaults(handler=command_perturb)

    fd = sub.add_parser("fd", help="central-difference derivative of J from scored +/- runs")
    fd.add_argument("--leg", nargs=3, action="append", required=True, metavar=("SCALE", "PLUS", "MINUS"),
                    help="a scale and the cost summaries of its + and - runs; repeat for several scales")
    fd.add_argument("--reference", default=None, help="cost summary of the run written with --scale 0 (the writer's own repairs, no perturbation)")
    fd.add_argument("--zero", default=None, help="cost summary of a restart from the untouched checkpoint (its distance from --reference is reported)")
    fd.add_argument("--out", default=None)
    fd.set_defaults(handler=command_fd)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
