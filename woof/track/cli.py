"""``woof track``: inspect 3-D path files and write their Track3D JSON."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from woof.explain import layered
from woof.track.ingest import ColumnMapping, load_track, parse_drop_when, supported
from woof.track.model import DEFAULT_MIN_CLIMB_M, HEIGHT_DATUMS, Track3D


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "track", help="inspect 3-D path files (time, lat, lon, height): "
                      "launch, landing, extent and boundary source",
        description="Read time/lat/lon/height paths from CSV, JSON or "
                    "Parquet, clean them, and report each track's launch, "
                    "duration, peak, horizontal extent and the regional "
                    "analysis that can force a domain around it.")
    commands = parser.add_subparsers(dest="track_command", required=True)
    inspect = commands.add_parser(
        "inspect", help="print one row per track; optionally write one "
                        "Track3D JSON per track",
        description="One row per track: name, launch (UTC), duration, peak "
                    "height, east-west by north-south extent, whether launch "
                    "and landing were observed, and the suggested boundary "
                    "source (hrrr, ukv or era5).  Column options override "
                    "the --mapping file; both default to columns named "
                    "time, lat, lon and height.")
    inspect.add_argument("paths", type=Path, nargs="+", metavar="FILE",
                         help="track file (.csv, .tsv, .json, .parquet), or "
                              "a directory of them")
    inspect.add_argument("--name", default=None, metavar="NAME",
                         help="the track's name (default: the file stem); "
                              "only with a single FILE")
    add_mapping_arguments(inspect)
    inspect.add_argument("--json", type=Path, default=None, metavar="OUT",
                         help="directory to write <name>.json Track3D documents "
                              "into (created if missing)")
    inspect.set_defaults(func=track_main)
    from woof.explain import add_explain_flag
    add_explain_flag(inspect, nested=True)
    from woof.track.plan_cli import register_plan
    register_plan(commands)
    from woof.track.splice import register_splice
    register_splice(commands)


def add_mapping_arguments(parser) -> None:
    """The column options `woof track inspect`, `woof track plan` and
    `woof estimate-tracks` share."""

    parser.add_argument("--mapping", type=Path, default=None, metavar="FILE",
                        help="JSON or TOML column mapping with keys time, lat, "
                             "lon, height, height_fallback, time_format, "
                             "height_datum, drop_when")
    parser.add_argument("--time-col", default=None, metavar="COL",
                        help="time column (default: time)")
    parser.add_argument("--lat-col", default=None, metavar="COL",
                        help="latitude column, degrees (default: lat)")
    parser.add_argument("--lon-col", default=None, metavar="COL",
                        help="longitude column, degrees (default: lon)")
    parser.add_argument("--height-col", default=None, metavar="COL",
                        help="height column, metres (default: height)")
    parser.add_argument("--height-fallback-col", action="append", default=None,
                        metavar="COL",
                        help="column used where the height is blank; "
                             "repeatable, tried in order")
    parser.add_argument("--time-format", default=None, metavar="FMT",
                        help="a Python strptime format, or 'epoch' for Unix "
                             "seconds (default: ISO 8601, date and time "
                             "joined by a space or T); times without a "
                             "zone are UTC")
    parser.add_argument("--height-datum", choices=HEIGHT_DATUMS, default=None,
                        help="heights above mean sea level or above ground "
                             "(default: msl)")
    parser.add_argument("--drop-when", action="append", default=None,
                        metavar="COL=VALUE",
                        help="drop rows whose COL equals VALUE (e.g. a "
                             "receiver's no-fix flag); repeatable")
    parser.add_argument("--min-climb", type=float, default=DEFAULT_MIN_CLIMB_M,
                        metavar="M",
                        help="climb above the starting ground, metres, that "
                             f"makes a launch (default: {DEFAULT_MIN_CLIMB_M:g})")


def mapping_from_args(args) -> ColumnMapping:
    """The ``--mapping`` file (or the defaults) with the column options
    laid over it."""

    base = (ColumnMapping() if args.mapping is None
            else ColumnMapping.from_file(args.mapping))
    return base.with_overrides(
        time=args.time_col, lat=args.lat_col, lon=args.lon_col,
        height=args.height_col,
        height_fallback=(None if args.height_fallback_col is None
                         else tuple(args.height_fallback_col)),
        time_format=args.time_format, height_datum=args.height_datum,
        drop_when=(None if args.drop_when is None
                   else parse_drop_when(args.drop_when)))


def track_main(args) -> int:
    from woof.explain import explain_enabled, render

    mapping = mapping_from_args(args)
    paths = supported(args.paths)
    if not paths:
        raise ValueError(f"no track files in {', '.join(map(str, args.paths))}")
    if args.name is not None and len(paths) > 1:
        raise ValueError(layered(
            f"--name names one track but {len(paths)} files were given.\n"
            "  remedy: drop --name (each track is then named by its file "
            "stem) or pass a single FILE.",
            "Every track's JSON is written as <name>.json, so one name for "
            "several tracks would write them over each other."))
    tracks: list[Track3D] = []
    failed = 0
    for path in paths:
        try:
            tracks.append(load_track(path, mapping, name=args.name,
                                     min_climb_m=args.min_climb))
        except (ValueError, OSError, csv.Error) as error:
            # One unreadable file must not hide the others; it is named
            # here and makes the exit status non-zero.
            failed += 1
            print(f"woof track: {path}: "
                  + render(str(error), explain=explain_enabled(args),
                           command="woof track inspect"), file=sys.stderr)
    names = [track.name for track in tracks]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated and args.json is not None:
        raise ValueError(layered(
            f"two files share the track name {', '.join(duplicated)}.\n"
            "  remedy: inspect them separately with --name, or rename one.",
            "Each track's JSON is written as <name>.json in OUT; two tracks "
            "with one name would write over each other."))
    if tracks:
        print(track_table(tracks))
    if explain_enabled(args):
        print()
        for track in tracks:
            print(_explain_line(track))
    if args.json is not None and tracks:
        args.json.mkdir(parents=True, exist_ok=True)
        for track in tracks:
            (args.json / f"{track.name}.json").write_text(
                track.to_json(indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(tracks)} track(s) to {args.json}")
    return 2 if failed else 0


def track_table(tracks: list[Track3D]) -> str:
    """The `woof track inspect` table, one row per track."""

    width = max([len("track")] + [len(track.name) for track in tracks])
    lines = [f"{'track':<{width}}  {'launch (UTC)':<16}  {'dur h':>6}  "
             f"{'max km':>6}  {'E-W x N-S km':>13}  {'seen':<11}  source"]
    sources: dict[str, int] = {}
    for track in tracks:
        extent = f"{track.extent_ew_km:.0f} x {track.extent_ns_km:.0f}"
        seen = "+".join(word for word, flag in (("launch", track.launch_observed),
                                                ("land", track.landing_observed))
                        if flag) or "-"
        lines.append(
            f"{track.name:<{width}}  "
            f"{track.launch.time.strftime('%Y-%m-%d %H:%M'):<16}  "
            f"{track.duration_h:>6.2f}  {track.max_height / 1000.0:>6.2f}  "
            f"{extent:>13}  {seen:<11}  {track.boundary_source}")
        sources[track.boundary_source] = sources.get(track.boundary_source, 0) + 1
    lines.append("")
    lines.append(f"{len(tracks)} track(s); boundary source: "
                 + ", ".join(f"{name} {count}"
                             for name, count in sorted(sources.items())))
    return "\n".join(lines)


def _explain_line(track: Track3D) -> str:
    dropped = ", ".join(f"{key} {value}" for key, value in sorted(track.qc.items())
                        if key not in ("rows", "fixes"))
    launch = (f"launch window {track.launch.window_s:.0f} s"
              if track.launch_observed else "launch before the record (first contact)")
    landing = (f"landing window {track.landing.window_s:.0f} s"
               if track.landing_observed else "landing not observed (last contact)")
    return (f"{track.name}: {track.qc.get('fixes', len(track.times))} of "
            f"{track.qc.get('rows', '?')} rows kept"
            + (f" (dropped/repaired: {dropped})" if dropped else "")
            + f"; {launch}; {landing}; peak {track.peak_class}, "
            f"heights {track.height_datum.upper()}")
