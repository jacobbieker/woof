"""``woof track plan``: path files in, track-following ``woof go`` configs out.

The planning itself is :mod:`woof.track.plan`; this is its command line:
reading each path with the ``woof track inspect`` column options, naming
the runs, writing the job documents and printing one summary row per
run.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from woof.explain import layered


def _floats(text: str) -> tuple[float, ...]:
    import argparse

    from woof.cli_numbers import positive_float

    values = tuple(positive_float(part.strip())
                   for part in str(text).split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"{text!r}: give three spacings, d01,d02,d03 (km)")
    return values


def _heights(text: str) -> tuple[float, ...]:
    import argparse

    from woof.cli_numbers import positive_float

    values = tuple(positive_float(part.strip())
                   for part in str(text).split(",") if part.strip())
    if not 0 < len(values) <= 32 or list(values) != sorted(set(values)):
        raise argparse.ArgumentTypeError(
            f"{text!r}: give 1-32 strictly increasing heights (m)")
    return values


def _p_top(text: str) -> float | None:
    import argparse

    if str(text).strip().lower() == "auto":
        return None
    try:
        return float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--p-top-pa {text!r} is not 'auto' or a pressure in Pa") from error


def _gpu_mem(text: str) -> float | None:
    import argparse

    if str(text).strip().lower() == "auto":
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--gpu-mem-gib {text!r} is not 'auto' or a number") from error
    if not value > 0:
        raise argparse.ArgumentTypeError("--gpu-mem-gib must be positive")
    return value


def safe_stem(path: Path) -> str:
    """A file stem as a run name: every run of characters a job id cannot
    carry (anything but letters, digits, '.', '_', '-') becomes one '-',
    and separators are trimmed from both ends."""

    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")


def register_plan(commands) -> None:
    from woof.cli_numbers import (nonnegative_float, positive_float,
                                  positive_int)
    from woof.explain import add_explain_flag
    from woof.track import plan as tp
    from woof.track.cli import add_mapping_arguments
    from woof.vertical_bands import parse_band

    parser = commands.add_parser(
        "plan", help="write a nested, track-following woof go config (and "
                     "optionally an AWS job document) for each path",
        description=(
            "For each path: a three-domain tree (d01 static over the "
            "track's envelope, d02 sliding, d03 following the track), a "
            "256-level column topped above the track's peak, d03 born from "
            "its parent before launch and writing history only in flight, "
            "an Icechunk export of d03, a plan.json with cells, GPU memory, "
            "wall time and output volume, and with --job-json a "
            "woof.go-job.v1 document.  Everything lands in OUT/<name>/; job "
            "documents in OUT/jobs/<tier>/.  Run each config with woof go "
            "CONFIG --products none."))
    parser.add_argument("paths", type=Path, nargs="+", metavar="PATHFILE",
                        help="path file (.csv, .tsv, .json, .parquet), or a "
                             "directory of them")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="the run's name: [experiment] name, job id and "
                             "Icechunk key (only with a single PATHFILE)")
    parser.add_argument("--names-from", choices=("stem",), default=None,
                        help="name each run after its file stem, with "
                             "characters a job id cannot carry replaced by "
                             "'-' (needed for several PATHFILEs)")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="directory the runs are written into")
    add_mapping_arguments(parser)
    parser.add_argument("--source", default="auto", metavar="SOURCE",
                        help="boundary source: auto (the track's suggestion "
                             "where this woof can prepare it to the model "
                             "top, else gfs), "
                             f"{', '.join(tp.SOURCE_CHOICES[1:])}, or any "
                             "runnable registered source id")
    parser.add_argument("--dx-km", type=_floats, default=tp.DEFAULT_DX_KM,
                        metavar="D1,D2,D3",
                        help="spacing of d01, d02, d03 in km (default "
                             + ",".join(f"{v:g}" for v in tp.DEFAULT_DX_KM)
                             + ")")
    parser.add_argument("--d03-window-km", type=positive_float,
                        default=tp.DEFAULT_D03_WINDOW_KM, metavar="KM",
                        help=f"d03's side (default {tp.DEFAULT_D03_WINDOW_KM:g})")
    parser.add_argument("--d02-window-km", type=positive_float,
                        default=tp.DEFAULT_D02_WINDOW_KM, metavar="KM",
                        help=f"d02's side (default {tp.DEFAULT_D02_WINDOW_KM:g})")
    parser.add_argument("--margin-km", type=positive_float,
                        default=tp.DEFAULT_MARGIN_KM, metavar="KM",
                        help="ground d01 keeps around the track envelope "
                             f"(default {tp.DEFAULT_MARGIN_KM:g}; grown when "
                             "d02 would not fit at the envelope's edge)")
    parser.add_argument("--spinup-h", type=nonnegative_float,
                        default=tp.DEFAULT_SPINUP_H,
                        metavar="H",
                        help="tree spin-up before launch, rounded out to the "
                             f"forcing cadence (default {tp.DEFAULT_SPINUP_H:g})")
    parser.add_argument("--d03-spinup-h", type=nonnegative_float,
                        default=tp.DEFAULT_D03_SPINUP_H, metavar="H",
                        help="d03's spin-up before launch, born from d02 "
                             f"(default {tp.DEFAULT_D03_SPINUP_H:g})")
    parser.add_argument("--history-s", type=positive_float,
                        default=tp.DEFAULT_HISTORY_S, metavar="S",
                        help="d03 history interval during the flight "
                             f"(default {tp.DEFAULT_HISTORY_S:g})")
    parser.add_argument("--eta-band", action="append", default=None,
                        metavar="LO:HI:N[:msl|agl]", type=str,
                        help="height band holding at least N levels; "
                             "repeatable (default "
                             + " and ".join(tp.DEFAULT_ETA_BANDS)
                             + ", fitted under the model top)")
    parser.add_argument("--no-eta-bias", action="store_true",
                        help="do not bias the column toward the heights the "
                             "track spends time at")
    parser.add_argument("--p-top-pa", type=_p_top, default=None,
                        metavar="PA|auto",
                        help="model top (default auto: the standard-"
                             f"atmosphere pressure {tp.P_TOP_HEADROOM_M / 1000:g} "
                             "km above the track's peak, clipped to "
                             f"{tp.P_TOP_RANGE_PA[0]:g}-{tp.P_TOP_RANGE_PA[1]:g})")
    parser.add_argument("--icechunk-url", default=None, metavar="URL",
                        help="the run's Icechunk repository, e.g. "
                             "s3://BUCKET/PREFIX/{name}.icechunk ({name} is "
                             "the run name; default {name}.icechunk beside "
                             "the config)")
    parser.add_argument("--eta-max-stretch", type=positive_float,
                        default=None, metavar="R",
                        help="largest ratio between adjacent layer depths "
                             "(default: the band generator's 1.08); raise it "
                             "to pack many levels low and still reach the "
                             "model top")
    parser.add_argument("--dzbot", type=positive_float, default=None,
                        metavar="M",
                        help="first layer depth in m for the band generator "
                             "(default 10); lower it when a band packs "
                             "levels finer than that near the ground")
    parser.add_argument("--eta-min-free-dz",
                        type=nonnegative_float, default=tp.DEFAULT_ETA_MIN_FREE_DZ_M,
                        metavar="M",
                        help="thinnest layer in m the track-height bias may "
                             "leave above 3 km AGL (default 50); the bias "
                             "is weakened until the column meets it. 0 "
                             "turns the floor off")
    parser.add_argument("--hub-heights-m", type=_heights, default=None,
                        metavar="H1,H2,...",
                        help="heights above ground (m) for the in-model "
                             "hub-wind diagnostic: interval max/min/mean wind "
                             "at each, i.e. gusts by height (at most 32)")
    parser.add_argument("--hub-site", action="append", default=[],
                        metavar="ID:LAT:LON",
                        help="a point sampled by the hub-wind diagnostic "
                             "(e.g. a weather station); repeatable")
    parser.add_argument("--output-preset", default="auto", metavar="PRESET",
                        help="history preset (default auto: the lean "
                             "track preset where this woof has one)")
    parser.add_argument("--relocation", choices=tp.RELOCATION_CHOICES,
                        default="auto",
                        help="auto: d03 follows the track where this woof "
                             "has the itinerary, else static nests; track: "
                             "require it; static: never move")
    parser.add_argument("--relocation-cadence-s", type=positive_float,
                        default=tp.DEFAULT_RELOCATION_CADENCE_S, metavar="S",
                        help="how often d03 may move (default "
                             f"{tp.DEFAULT_RELOCATION_CADENCE_S:g})")
    parser.add_argument("--track-margin-cells", type=positive_int,
                        default=tp.DEFAULT_TRACK_MARGIN_CELLS, metavar="N",
                        help="d03 cells the track keeps from d03's edges "
                             f"(default {tp.DEFAULT_TRACK_MARGIN_CELLS})")
    parser.add_argument("--physics", default="auto", metavar="auto|explicit|SUITE",
                        help="auto: the registered LES-nest tree suite, else "
                             "explicit keys; explicit; or a tree suite id")
    parser.add_argument("--gpu-mem-gib", type=_gpu_mem, default=None,
                        metavar="GIB|auto",
                        help="the job's declared GPU memory (default auto: "
                             "the estimate's upper bound)")
    parser.add_argument("--job-json", action="store_true",
                        help="also write a woof.go-job.v1 document per run "
                             "into OUT/jobs/<tier>/")
    parser.add_argument("--inputs-prefix", default=None, metavar="URL",
                        help="with --job-json: reference the config and "
                             "companions as URLs under URL/<name>/ (push them "
                             "with the printed woof go-job-push command) "
                             "instead of carrying them inline")
    parser.add_argument("--result-prefix", default=None, metavar="URL",
                        help="with --job-json: write each job's result JSON "
                             "to URL/<name>.result.json")
    parser.set_defaults(func=plan_main, parse_band=parse_band)
    add_explain_flag(parser, nested=True)


def _options(args):
    from woof.track.plan import PlanOptions

    bands = None
    if args.eta_band:
        for text in args.eta_band:
            args.parse_band(text)
        bands = tuple(args.eta_band)
    return PlanOptions(
        source=args.source, dx_km=tuple(args.dx_km),
        d03_window_km=args.d03_window_km, d02_window_km=args.d02_window_km,
        margin_km=args.margin_km, spinup_h=args.spinup_h,
        d03_spinup_h=args.d03_spinup_h, history_s=args.history_s,
        eta_bands=bands, eta_bias_track=not args.no_eta_bias,
        eta_max_stretch=args.eta_max_stretch, dzbot_m=args.dzbot,
        eta_min_free_dz_m=args.eta_min_free_dz,
        hub_heights_m=(tuple(args.hub_heights_m)
                       if args.hub_heights_m else None),
        hub_sites=tuple(_hub_site(text) for text in args.hub_site),
        p_top_pa=args.p_top_pa, icechunk_url=args.icechunk_url,
        output_preset=args.output_preset, relocation=args.relocation,
        relocation_cadence_s=args.relocation_cadence_s,
        track_margin_cells=args.track_margin_cells, physics=args.physics,
        gpu_mem_gib=args.gpu_mem_gib)


def _hub_site(text: str) -> tuple[str, float, float]:
    try:
        name, lat, lon = text.rsplit(":", 2)
        return name, float(lat), float(lon)
    except ValueError:
        raise SystemExit(f"--hub-site {text!r}: expected ID:LAT:LON") from None


def plan_main(args) -> int:
    from woof.explain import explain_enabled, render
    from woof.go_job import check_url
    from woof.track import plan as tp
    from woof.track.cli import mapping_from_args
    from woof.track.ingest import load_track, supported

    paths = supported(args.paths)
    if not paths:
        raise ValueError(f"no path files in {', '.join(map(str, args.paths))}")
    if args.name is not None and (len(paths) > 1 or args.names_from):
        raise ValueError(layered(
            f"--name names one run but {len(paths)} files were given"
            + (" (or --names-from too)" if len(paths) == 1 else "") + ".\n"
            "  remedy: drop --name and pass --names-from stem, or pass a "
            "single PATHFILE.",
            "Each run's directory, job id and Icechunk repository are its "
            "name, so one name for several runs would overwrite them."))
    if args.name is None and args.names_from is None:
        raise ValueError(layered(
            "no run name: pass --name NAME (one PATHFILE) or --names-from "
            "stem.\n"
            "  remedy: name the run explicitly.",
            "The name is the config's [experiment] name, the job id and the "
            "key of the run's Icechunk repository; it is never guessed."))
    if (args.inputs_prefix or args.result_prefix) and not args.job_json:
        raise ValueError("--inputs-prefix and --result-prefix describe job "
                         "documents; add --job-json")
    for flag, url in (("--inputs-prefix", args.inputs_prefix),
                      ("--result-prefix", args.result_prefix)):
        if url:
            check_url(url, flag, schemes=("s3", "gs", "file"))
    if (len(paths) > 1 and args.icechunk_url
            and "{name}" not in args.icechunk_url):
        raise ValueError(layered(
            f"--icechunk-url {args.icechunk_url} names one repository for "
            f"{len(paths)} runs.\n"
            "  remedy: put {name} in it, e.g. "
            "s3://BUCKET/PREFIX/{name}.icechunk.",
            "Every run writes its own repository; runs sharing one would "
            "commit over each other."))
    if args.icechunk_url and "://" in args.icechunk_url:
        check_url(args.icechunk_url.replace("{name}", "x"), "--icechunk-url",
                  schemes=("s3", "gs", "file"))
    options = _options(args)
    mapping = mapping_from_args(args)
    explain = explain_enabled(args)
    runs: list[tp.PlannedRun] = []
    failed = 0
    seen: set[str] = set()
    for path in paths:
        name = args.name if args.name is not None else safe_stem(path)
        try:
            if name in seen:
                raise tp.TrackPlanRefused(
                    f"two path files are named {name}; plan them separately "
                    "with --name")
            seen.add(name)
            track = load_track(path, mapping, name=name,
                               min_climb_m=args.min_climb)
            runs.append(tp.plan_track(track, name, args.out, options))
        except (ValueError, OSError, csv.Error) as error:
            # One path that cannot be planned must not hide the others.
            failed += 1
            print(f"woof track plan: {path}: "
                  + render(str(error), explain=explain,
                           command="woof track plan"), file=sys.stderr)
    if not runs:
        return 2
    print(tp.summary_table(runs))
    for run in runs:
        for note in run.notes:
            print(f"  {run.name}: {note}")
    if args.job_json:
        failed += _write_jobs(runs, args, explain=explain)
    print(f"wrote {len(runs)} run(s) under {args.out}; run each with: woof go "
          f"{args.out}/<name>/<name>.toml --products none")
    if len(runs) > 1:
        summary = {"schema": tp.PLAN_SCHEMA + "-summary",
                   "runs": [run.plan for run in runs]}
        (Path(args.out) / "plan-summary.json").write_text(
            json.dumps(summary, indent=1, default=str) + "\n",
            encoding="utf-8")
    return 2 if failed else 0


def _write_jobs(runs, args, *, explain: bool = False) -> int:
    """Write each run's job document; the number that could not be."""

    from woof.explain import render
    from woof.track import plan as tp

    jobs_root = Path(args.out) / "jobs"
    tiers: dict[str, int] = {}
    failed = 0
    pushed = []
    for run in runs:
        try:
            document = tp.job_document(run, inputs_prefix=args.inputs_prefix,
                                       icechunk_url=args.icechunk_url,
                                       result_prefix=args.result_prefix)
        except ValueError as error:
            failed += 1
            print("woof track plan: " + render(str(error), explain=explain,
                                               command="woof track plan"),
                  file=sys.stderr)
            continue
        tier = run.plan["estimate"]["tier"]
        if tier == "none":
            failed += 1
            print(f"woof track plan: {run.name} declares "
                  f"{document['resources']['gpu_mem_gib']:g} GiB, more than "
                  f"the {tp.H100_TIER_MAX_GIB:g} GiB card of the largest "
                  "tier; no job document written.\n"
                  "  remedy: shrink the windows (--d03-window-km, "
                  "--d02-window-km) or --margin-km.", file=sys.stderr)
            continue
        pushed.append(run)
        directory = jobs_root / tier
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{run.name}.json").write_text(
            json.dumps(document, indent=1) + "\n", encoding="utf-8")
        tiers[tier] = tiers.get(tier, 0) + 1
    if not tiers:
        return failed
    print("job documents: " + ", ".join(
        f"{count} in {jobs_root / tier}" for tier, count in sorted(tiers.items())))
    if args.inputs_prefix:
        prefix = args.inputs_prefix.rstrip("/")
        print("push the inputs first:")
        for run in pushed:
            print(f"  woof go-job-push {run.directory} {prefix}/{run.name}/")
    for tier in sorted(tiers):
        print(f"then: woof energy queue push QUEUE-{tier.upper()} "
              f"{jobs_root / tier} --schema woof.go-job.v1")
    return failed
