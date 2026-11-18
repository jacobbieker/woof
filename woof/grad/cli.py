"""``woof grad``: observation cost and adjoint-free sensitivities (EXPERIMENTAL).

Reached only through the component door (:mod:`woof.components`), so no
other command imports this package.

    woof grad obs  --history RUN_OR_FILES --out OBS.npz [--stores wis2_land,...] [--goes-cwp FILE ...]
    woof grad cost --history RUN_OR_FILES --obs OBS.npz --out J.json [--pairs-out PAIRS.npz] [--accept REF.npz]
    woof grad direction --checkpoint CKPT --field thp --blob K J I --radius-cells R --depth-levels D --out DIR.npz
    woof grad direction --mesh GRID.nc --field theta --at LAT LON LEVEL --radius-km R --out DIR.npz
    woof grad perturb --checkpoint CKPT|STATE.nc --direction DIR.npz --scale EPS --out PERTURBED
    woof grad fd --leg EPS PLUS.json MINUS.json [--leg ...] [--reference J0.json] [--zero JZ.json]
    woof grad members --checkpoint CKPT --members K --seed S --out-dir DIR
    woof grad ens --states DIR_k.npz... --costs J_k.json... --out ES.npz [--regress FIELD K J I --direction-out P.npz]
    woof grad letkf --obs OBS.npz --accept PAIRS.npz --history H_k... --checkpoints C_k... --out-dir DIR
    woof grad mean --checkpoints C_k... --out MEAN.npz
    woof grad efso --analysis DIR/efso-inputs.npz --obs OBS.npz --forecast-pairs P_k.npz... --out IMPACT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
        from woof.file_digest import sha256_file
        summary["accepted_sha256"] = sha256_file(args.accept)
    if args.pairs_out:
        result.save_pairs(args.pairs_out)
        summary["pairs"] = str(args.pairs_out)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text if not args.quiet else f"J = {result.J:.6g} over {result.n} observations")
    return 0


def command_direction(args) -> int:
    from woof.grad.sensitivity import (cell_blob, checkpoint_fields, direction_path, gaussian_blob, load_direction,
                                       mesh_cells, save_direction)

    fields, blobs = {}, []
    if args.add_to:
        fields, previous = load_direction(args.add_to)
        # Carry the earlier blobs' provenance; a pre-history file records one blob under "added".
        blobs = list(previous.get("blobs") or ([previous["added"]] if previous.get("added") else []))
    if args.blob is not None:
        if not args.checkpoint:
            raise SystemExit("woof grad direction: --blob needs --checkpoint")
        shape = checkpoint_fields(args.checkpoint, [args.field])[args.field].shape
        k, j, i = args.blob
        blob = gaussian_blob(shape, k=k, j=j, i=i, radius_cells=args.radius_cells,
                             depth_levels=args.depth_levels, amplitude=args.amplitude, rim=args.rim)
        added = {"field": args.field, "kji": [k, j, i], "radius_cells": args.radius_cells, "rim": args.rim}
    else:
        if not args.mesh:
            raise SystemExit("woof grad direction: --at needs --mesh")
        cells = mesh_cells(args.mesh)
        nz = args.levels or cells["nz"]
        if not nz:
            raise SystemExit("woof grad direction: the mesh file has no nVertLevels; pass --levels")
        interior = None if cells["bdy"] is None else cells["bdy"] == 0
        if interior is None:
            print(f"woof grad direction: {args.mesh} carries no bdyMaskCell, so the blob is not zeroed in a "
                  "regional mesh's relaxation rings; pass the static or init file of a regional mesh",
                  file=sys.stderr)
        lat0, lon0, level = args.at
        blob = cell_blob(cells["lat"], cells["lon"], nz, lat0=lat0, lon0=lon0, level=level,
                         radius_km=args.radius_km, depth_levels=args.depth_levels, amplitude=args.amplitude,
                         interior=interior)
        added = {"field": args.field, "lat_lon_level": [lat0, lon0, level], "radius_km": args.radius_km}
    fields[args.field] = fields.get(args.field, 0.0) + blob
    added.update(depth_levels=args.depth_levels, amplitude=args.amplitude,
                 source=str(args.checkpoint or args.mesh))
    sha = save_direction(args.out, fields, {"kind": "gaussian_blob", "blobs": blobs + [added]})
    print(json.dumps({"direction": str(direction_path(args.out)), "sha256": sha, "fields": sorted(fields),
                      "blobs": len(blobs) + 1}, indent=2))
    return 0


def command_perturb(args) -> int:
    from woof.grad.sensitivity import load_direction, perturb_checkpoint, perturb_mpas_file

    fields, meta = load_direction(args.direction)
    if str(args.checkpoint).endswith(".nc"):
        receipt = perturb_mpas_file(args.checkpoint, fields, args.scale, args.out)
    else:
        receipt = perturb_checkpoint(args.checkpoint, fields, args.scale, args.out, mp_physics=args.mp_physics,
                                     morr_rimed_ice=args.morr_rimed_ice)
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


def _shapes(checkpoint):
    from woof.grad.sensitivity import checkpoint_fields
    return {name: value.shape for name, value in checkpoint_fields(checkpoint, ["thp", "u", "v"]).items()}


def command_members(args) -> int:
    from woof.grad.ensemble import centred, grf_cells, grf_increments
    from woof.grad.sensitivity import mesh_cells, save_direction

    if bool(args.checkpoint) == bool(args.mesh):
        raise SystemExit("woof grad members: pass exactly one of --checkpoint (WRF route) or --mesh (hex)")
    if args.checkpoint:
        shapes = _shapes(args.checkpoint)
        draws = [grf_increments(shapes, seed=args.seed + m, dx_km=args.dx_km, length_scale_km=args.length_km,
                                vertical_scale_levels=args.vertical_levels, thp_k=args.thp_k, wind_ms=args.wind_ms,
                                rim=args.rim) for m in range(args.members)]
    else:
        cells = mesh_cells(args.mesh)
        nz = args.levels or cells["nz"]
        if not nz:
            raise SystemExit("woof grad members: the mesh file has no nVertLevels; pass --levels")
        interior = None if cells["bdy"] is None else cells["bdy"] == 0
        draws = [{"theta": grf_cells(cells["lat"], cells["lon"], nz, seed=args.seed + m, dx_km=args.dx_km,
                                     length_scale_km=args.length_km, vertical_scale_levels=args.vertical_levels,
                                     sigma=args.thp_k, interior=interior)} for m in range(args.members)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: getattr(args, k) for k in ("seed", "dx_km", "length_km", "vertical_levels", "thp_k", "wind_ms", "rim")}
    meta["route"] = "wrf" if args.checkpoint else "hex"
    for m, member in enumerate(centred(draws) if args.members > 1 else draws):
        save_direction(out / f"member_{m:03d}.npz", member, dict(meta, kind="grf_member", member=m,
                                                                 members=args.members, centred=args.members > 1))
    print(f"wrote {args.members} member perturbations to {out}")
    return 0


def _member_states(paths, names):
    """Member states (K, ...) per field from direction files or checkpoints."""
    from woof.grad.sensitivity import checkpoint_fields, load_direction

    stacks = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            is_direction = "schema" in data.files
        fields = load_direction(path)[0] if is_direction else checkpoint_fields(path, names)
        for name in names:
            stacks.setdefault(name, []).append(np.asarray(fields[name], np.float64))
    return {name: np.stack(values) for name, values in stacks.items()}


def command_ens(args) -> int:
    from woof.grad.ensemble import ensemble_sensitivity, regression_direction
    from woof.grad.sensitivity import _cost, save_direction

    if len(args.states) != len(args.costs):
        raise SystemExit(f"woof grad ens: {len(args.states)} states but {len(args.costs)} costs")
    J = np.array([_cost(path)[0] for path in args.costs])
    X = _member_states(args.states, args.fields)
    report = {"schema": "woof.grad.ensemble-sensitivity/v1", "members": len(J), "J_mean": float(J.mean()),
              "J_std": float(J.std(ddof=1)), "fields": {}}
    arrays = {}
    for name in args.fields:
        es = ensemble_sensitivity(J, X[name])
        arrays.update({f"{name}/S": es["S"], f"{name}/r": es["r"], f"{name}/std": es["std"]})
        flat = np.nan_to_num(np.abs(es["S"] * es["std"]), nan=-1.0)
        top = np.unravel_index(np.argsort(flat, axis=None)[::-1][:args.top], flat.shape)
        report["fields"][name] = {
            "max_abs_r": float(np.nanmax(np.abs(es["r"]))),
            "top_points_by_S_times_std": [{"index": [int(a[n]) for a in top], "S": float(es["S"][tuple(a[n] for a in top)]),
                                           "r": float(es["r"][tuple(a[n] for a in top)])} for n in range(len(top[0]))]}
    np.savez(args.out, J=J, **arrays)
    if args.regress:
        from woof.grad.ensemble import regression_direction_cells
        from woof.grad.sensitivity import mesh_cells

        if not args.direction_out:
            raise SystemExit("woof grad ens: --regress needs --direction-out")
        field = args.regress[0]
        index = tuple(map(int, args.regress[1:]))
        if args.mesh:
            cells = mesh_cells(args.mesh)
            direction, info = regression_direction_cells(X, field=field, cell=index[0], level=index[1],
                                                         lat=cells["lat"], lon=cells["lon"],
                                                         horizontal_km=args.localization_km,
                                                         vertical_levels=args.localization_levels)
        else:
            if "thp" not in X or not args.dx_km or len(index) != 3:
                raise SystemExit("woof grad ens: on the WRF route --regress takes FIELD K J I and needs thp "
                                 "among --fields and --dx-km")
            direction, info = regression_direction(X, field=field, index=index, mass_shape=X["thp"].shape[1:],
                                                   dx_m=args.dx_km * 1e3, horizontal_m=args.localization_km * 1e3,
                                                   vertical_levels=args.localization_levels)
        S = float(ensemble_sensitivity(J, X[field][(slice(None),) + index])["S"])
        info.update(predicted_dJ_per_unit_scale=S)
        sha = save_direction(args.direction_out, direction, info)
        report["regression"] = dict(info, direction=str(args.direction_out), sha256=sha)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n")
    print(text)
    return 0


def command_letkf(args) -> int:
    from woof.da.letkf import GridGeometry, LetkfConfig, Localization
    from woof.da.radar_assimilation import _RESTAGGER, _mass_field
    from woof.grad.ensemble import layered_batches, letkf_with_obs_space, obs_gridpoints
    from woof.grad.obs_cost import COLUMN_KINDS, ObsSet, accepted_from, evaluate, read_error_table, read_frames
    from woof.grad.sensitivity import checkpoint_fields
    from woof.ensemble.increments import apply_increments_to_checkpoint, publish_staged_analysis

    if len(args.history) != len(args.checkpoints):
        raise SystemExit("woof grad letkf: one --history file per --checkpoints member")
    obs = ObsSet.load(args.obs)
    table = read_error_table(args.error_table)
    accept = accepted_from(args.accept)
    results = []
    frame = None
    for path in args.history:
        frames = read_frames([path], rim=args.rim)
        if len(frames) != 1:
            raise SystemExit(f"woof grad letkf: {path} has {len(frames)} frames; give each member's history at "
                             "the analysis time only (one frame), so every observation is placed in that frame")
        frame = frame or frames[0]
        results.append(evaluate(obs, frames, table=table, accept=accept,
                                max_offset_s=accept.max_offset_s))
    common = set(results[0].obs_index.tolist())
    for result in results[1:]:
        common &= set(result.obs_index.tolist())
    common_rows = np.array(sorted(common), np.int64)
    column_kind = np.isin(obs.kind[common_rows], COLUMN_KINDS)
    rows = common_rows[~column_kind]
    pick = [np.searchsorted(r.obs_index, rows) for r in results]
    hb = np.stack([r.h[p] for r, p in zip(results, pick)])
    first = results[0]
    sigma, y, column = first.sigma[pick[0]], first.y[pick[0]], first.column[pick[0]]
    names = tuple(args.fields)
    states = [checkpoint_fields(path, names) for path in args.checkpoints]
    prior = {name: np.stack([_mass_field(name, state, where="grad letkf") for state in states]) for name in names}
    nz, ny, nx = prior[names[0]].shape[1:]
    k = obs_gridpoints(obs, rows, column, frame)
    flat = k * ny * nx + column
    batches, layers = layered_batches(flat, y, sigma, hb, shape=(nz, ny, nx), name="obs")
    grid = GridGeometry(dx_m=args.dx_km * 1e3, dy_m=args.dx_km * 1e3,
                        heights_m=np.asarray(frame.z).reshape(nz, ny, nx),
                        lat_deg=np.asarray(frame.lat).reshape(ny, nx), lon_deg=np.asarray(frame.lon).reshape(ny, nx))
    config = LetkfConfig(localization=Localization(args.localization_km * 1e3, args.vertical_km * 1e3),
                         analysis_fields=names, rtps_alpha=0.0,
                         memory_budget_mib=args.memory_budget_mib)
    import time
    t0 = time.perf_counter()
    increments, ya = letkf_with_obs_space(prior, batches, layers, grid, config)
    seconds = time.perf_counter() - t0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    for m, path in enumerate(args.checkpoints):
        member = {name: (_RESTAGGER[name](increments[name][m]) if name in _RESTAGGER else increments[name][m])
                  for name in names}
        target = out / f"member_{m:03d}" / Path(path).name.split("__")[0]
        target = target.with_suffix(".npz") if target.suffix != ".npz" else target
        receipt = apply_increments_to_checkpoint(path, member, target, publish=False, mp_physics=args.mp_physics)
        staged.append((receipt["staged"], target))
    # Every member is staged before any is published, so a failure leaves no half-analysed ensemble.
    written = [str(publish_staged_analysis(source, target)) for source, target in staged]
    hb_mean = hb.mean(axis=0)
    np.savez(out / "efso-inputs.npz", obs_index=rows, innovation=y - hb_mean, sigma=sigma, hb=hb, ya=ya,
             lat=obs.lat[rows], lon=obs.lon[rows], k=k, column=column, obs_sha256=np.asarray(obs.digest()))
    report = {"schema": "woof.grad.letkf/v1", "members": len(states), "observations": int(len(rows)),
              "observations_by_type": dict(sorted(Counter(f"{p}/{k}" for p, k in
                                                          zip(obs.platform[rows], obs.kind[rows])).items())),
              "batches": len(batches), "fields": list(names), "localization_km": args.localization_km,
              "vertical_km": args.vertical_km, "rtps_alpha": 0.0, "analysis_seconds": seconds,
              "left_out_column_kinds": int(column_kind.sum()),
              "innovation_rms_over_sigma": float(np.sqrt(np.mean(((y - hb_mean) / sigma) ** 2))),
              "spread_hb_over_sigma": float(np.mean(hb.std(axis=0, ddof=1) / sigma)),
              "spread_ya_over_sigma": float(np.mean(ya.std(axis=0, ddof=1) / sigma)),
              "analysis_checkpoints": written}
    (out / "letkf.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "analysis_checkpoints"}, indent=2))
    return 0


def command_mean(args) -> int:
    from woof.ensemble.increments import apply_increments_to_checkpoint
    from woof.ensemble.state_sha import serialized_state_attrs
    from woof.grad.sensitivity import checkpoint_fields

    contract = set(serialized_state_attrs())
    first = checkpoint_fields(args.checkpoints[0])
    names = sorted(name for name in first if name in contract)
    total = {name: np.zeros(first[name].shape, np.float64) for name in names}
    for path in args.checkpoints:
        fields = checkpoint_fields(path, names)
        for name in names:
            total[name] += fields[name]
    increments = {name: total[name] / len(args.checkpoints) - first[name] for name in names}
    receipt = apply_increments_to_checkpoint(args.checkpoints[0], increments, args.out, publish=True,
                                             mp_physics=args.mp_physics)
    print(json.dumps({"mean_of": len(args.checkpoints), "fields": len(names), "out": str(args.out),
                      "moments": receipt.get("moments"), "saturation": receipt.get("saturation")},
                     indent=2, default=str))
    return 0


def command_efso(args) -> int:
    from woof.da.letkf import gaspari_cohn
    from woof.grad.ensemble import efso, impact_table
    from woof.grad.obs_cost import ObsSet
    from woof.grad.sensitivity import great_circle_km

    obs = ObsSet.load(args.obs)
    with np.load(args.analysis, allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
    if str(a["obs_sha256"]) != obs.digest():
        raise SystemExit("woof grad efso: the analysis assimilated another observation file")
    pairs = []
    for path in args.forecast_pairs:
        with np.load(path, allow_pickle=False) as data:
            if str(data["obs_sha256"]) != obs.digest():
                raise SystemExit(f"woof grad efso: {path} was scored against another observation file")
            pairs.append({key: data[key] for key in ("obs_index", "h", "y", "sigma")})
    common = set(pairs[0]["obs_index"].tolist())
    for p in pairs[1:]:
        common &= set(p["obs_index"].tolist())
    verify = np.array(sorted(common), np.int64)
    jv = []
    for p in pairs:
        sel = np.searchsorted(p["obs_index"], verify)
        jv.append(0.5 * ((p["h"][sel] - p["y"][sel]) / p["sigma"][sel]) ** 2)
    jv = np.stack(jv)
    rho = None
    if args.localization_km:
        lat_v, lon_v = obs.lat[verify][None, :], obs.lon[verify][None, :]

        def rho(rows):
            km = great_circle_km(lat_v, lon_v, a["lat"][rows][:, None], a["lon"][rows][:, None])
            return np.asarray(gaspari_cohn(km, args.localization_km))
    impact = efso(a["innovation"], a["sigma"], a["ya"], jv if rho is not None else jv.sum(axis=1), rho=rho)
    rows = a["obs_index"]
    table = impact_table(impact, obs.platform[rows], obs.kind[rows])
    report = {"schema": "woof.grad.efso/v1", "members": int(jv.shape[0]), "assimilated": int(len(rows)),
              "verifying": int(len(verify)), "localization_km": args.localization_km,
              "J_forecast_mean": float(jv.sum(axis=1).mean()), "J_forecast_std": float(jv.sum(axis=1).std(ddof=1)),
              "predicted_dJ": float(impact.sum()), "by_type": table}
    if args.impacts_out:
        np.savez(args.impacts_out, obs_index=rows, impact=impact)
    text = json.dumps(report, indent=2, sort_keys=True)
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
    direction.add_argument("--rim", type=int, default=5,
                           help="WRF route: rows and columns zeroed at each edge (the specified and relaxation zone)")
    direction.add_argument("--add-to", default=None, help="an existing direction file to add this blob to")
    direction.add_argument("--out", required=True)
    direction.set_defaults(handler=command_direction)

    perturb = sub.add_parser("perturb", help="write CHECKPOINT + SCALE * DIRECTION as a new restart state")
    perturb.add_argument("--checkpoint", required=True,
                         help="a WRF-route restart checkpoint (.npz) or a hex MPAS-layout state file (.nc)")
    perturb.add_argument("--direction", required=True)
    perturb.add_argument("--scale", type=float, required=True, help="signed multiple of the direction")
    perturb.add_argument("--mp-physics", type=int, default=None, help="the run's mp_physics, for the moment guard")
    perturb.add_argument("--morr-rimed-ice", type=int, default=None, choices=(0, 1),
                         help="the run's morr_rimed_ice (Morrison: 0 graupel, 1 hail), for the moment repair; "
                              "default: the analysis writer's")
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

    members = sub.add_parser("members", help="write K centred Gaussian-random-field perturbations (thp and non-divergent wind)")
    members.add_argument("--checkpoint", default=None, help="WRF route: a restart checkpoint (gives the shapes)")
    members.add_argument("--mesh", default=None, help="hex: a grid, static or init file (theta-only members)")
    members.add_argument("--levels", type=int, default=None, help="hex: level count when the mesh file has none")
    members.add_argument("--members", type=int, required=True)
    members.add_argument("--seed", type=int, required=True)
    members.add_argument("--dx-km", type=float, required=True)
    members.add_argument("--length-km", type=float, default=100.0)
    members.add_argument("--vertical-levels", type=float, default=4.0)
    members.add_argument("--thp-k", type=float, default=0.5, help="theta perturbation standard deviation (K)")
    members.add_argument("--wind-ms", type=float, default=1.0, help="WRF route: wind perturbation standard deviation (m/s)")
    members.add_argument("--rim", type=int, default=5, help="WRF route: boundary rows the perturbation tapers to zero over")
    members.add_argument("--out-dir", required=True)
    members.set_defaults(handler=command_members)

    ens = sub.add_parser("ens", help="ensemble sensitivity of J to the members' states, and regression directions")
    ens.add_argument("--states", nargs="+", required=True,
                     help="member perturbations (woof grad members) or member checkpoints, one per member")
    ens.add_argument("--costs", nargs="+", required=True, help="each member's cost summary, same order")
    ens.add_argument("--fields", nargs="+", default=["thp", "u", "v"], help="thp u v (WRF route) or theta (hex)")
    ens.add_argument("--top", type=int, default=5, help="report this many strongest points per field")
    ens.add_argument("--out", required=True, help="the sensitivity maps (.npz)")
    ens.add_argument("--report", default=None)
    ens.add_argument("--regress", nargs="+", default=None, metavar="FIELD INDEX",
                     help="also write the localised regression of the state on one point as a direction: "
                          "FIELD K J I (WRF route) or FIELD CELL LEVEL (hex, with --mesh)")
    ens.add_argument("--mesh", default=None, help="hex: a grid, static or init file (the regression's geometry)")
    ens.add_argument("--direction-out", default=None)
    ens.add_argument("--dx-km", type=float, default=None)
    ens.add_argument("--localization-km", type=float, default=300.0, help="Gaspari-Cohn cutoff of the regression")
    ens.add_argument("--localization-levels", type=float, default=15.0)
    ens.set_defaults(handler=command_ens)

    letkf = sub.add_parser("letkf", help="LETKF analysis of member checkpoints with the paired observations; keeps Y^a for EFSO")
    letkf.add_argument("--obs", required=True)
    letkf.add_argument("--accept", required=True, help="pairs file fixing the assimilated observations (one frame)")
    letkf.add_argument("--history", nargs="+", required=True, help="each member's history file at the analysis time")
    letkf.add_argument("--checkpoints", nargs="+", required=True, help="each member's checkpoint at the analysis time")
    letkf.add_argument("--error-table", default=None)
    letkf.add_argument("--fields", nargs="+", default=["u", "v", "thp", "qv"])
    letkf.add_argument("--dx-km", type=float, required=True)
    letkf.add_argument("--localization-km", type=float, default=200.0)
    letkf.add_argument("--vertical-km", type=float, default=3.0)
    letkf.add_argument("--rim", type=int, default=5)
    letkf.add_argument("--memory-budget-mib", type=float, default=512.0, help="the filter's batched-solve scratch ceiling")
    letkf.add_argument("--mp-physics", type=int, default=None)
    letkf.add_argument("--out-dir", required=True)
    letkf.set_defaults(handler=command_letkf)

    mean = sub.add_parser("mean", help="write the ensemble-mean restart checkpoint of member checkpoints")
    mean.add_argument("--checkpoints", nargs="+", required=True)
    mean.add_argument("--mp-physics", type=int, default=None)
    mean.add_argument("--out", required=True)
    mean.set_defaults(handler=command_mean)

    efso_p = sub.add_parser("efso", help="each assimilated observation's impact on the verifying cost (EFSO)")
    efso_p.add_argument("--analysis", required=True, help="efso-inputs.npz from woof grad letkf")
    efso_p.add_argument("--obs", required=True)
    efso_p.add_argument("--forecast-pairs", nargs="+", required=True,
                        help="pairs files of the analysis members' forecasts (cost --accept --pairs-out), member order")
    efso_p.add_argument("--localization-km", type=float, default=None,
                        help="Gaspari-Cohn cutoff between assimilated and verifying observations; default none")
    efso_p.add_argument("--impacts-out", default=None)
    efso_p.add_argument("--out", default=None)
    efso_p.set_defaults(handler=command_efso)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
