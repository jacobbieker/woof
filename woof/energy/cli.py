"""``woof energy``: forecasts along power lines, substations and renewables.

This module owns argument parsing only.  Each subcommand hands its parsed
arguments to ``main(args)`` in the module that does the work, imported when
the subcommand runs::

    woof energy fetch     woof.energy.osm             OSM power data via Overpass
    woof energy import    woof.energy.importers       PyPSA-Eur, REPD, GeoJSON, CSV
    woof energy sites     woof.energy.sites           assets -> forecast sites
    woof energy plan      woof.energy.plan_wrf_nests  --topology wrf-nests
                          woof.energy.plan_wrf_tiles  --topology wrf-tiles
                          woof.energy.plan_hex        --topology hex-swath
    woof energy run       woof.energy.run             run every plan domain
    woof energy extract   woof.energy.extract         plan runs -> forecast.v1
    woof energy rating    woof.energy.products        line rating, icing, power
    woof energy hindcast  woof.energy.hindcast        year-long segment jobs
"""

from __future__ import annotations

import argparse
from importlib import import_module

from woof.cli_numbers import positive_float, positive_int

_KINDS = ("line", "minor_line", "cable", "substation", "plant", "generator",
          "tower")
_SITE_KINDS = ("line", "minor_line", "cable", "substation", "plant",
               "generator", "tower")
_TOPOLOGY_MODULES = {
    "wrf-nests": "woof.energy.plan_wrf_nests",
    "wrf-tiles": "woof.energy.plan_wrf_tiles",
    "hex-swath": "woof.energy.plan_hex",
}
_PRODUCTS = ("dlr", "icing", "wind-power", "pv-power")


def _bbox(value: str) -> tuple[float, float, float, float]:
    """``W,S,E,N`` in degrees (the GeoJSON order ``woof obs`` also uses)."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--bbox takes west,south,east,north; got {value!r}")
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--bbox values must be numbers; got {value!r}") from error
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise argparse.ArgumentTypeError(
            f"--bbox needs west < east and south < north in range; "
            f"got {value!r}")
    return west, south, east, north


def _choices_list(allowed: tuple[str, ...], flag: str):
    def parse(value: str) -> tuple[str, ...]:
        names = tuple(part.strip() for part in value.split(",") if part.strip())
        bad = [name for name in names if name not in allowed]
        if not names or bad:
            raise argparse.ArgumentTypeError(
                f"{flag} takes a comma list of {', '.join(allowed)}; "
                f"got {value!r}")
        return names
    return parse


def _heights(value: str) -> tuple[float, ...]:
    try:
        heights = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--heights-m takes a comma list of metres; got {value!r}") from error
    if not heights or any(h <= 0.0 for h in heights):
        raise argparse.ArgumentTypeError(
            f"--heights-m values must be positive metres above ground; "
            f"got {value!r}")
    return tuple(sorted(set(heights)))


def _names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected a comma list of names")
    return names


def _dispatch(module: str):
    def run(args) -> int:
        return import_module(module).main(args)
    return run


#: ``woof energy plan`` flags only the hex-swath planner reads.
_HEX_ONLY_FLAGS = (("hex_experimental_dt", "--hex-experimental-dt"),
                   ("hex_density", "--hex-density"),
                   ("hex_forcing_wrfout", "--hex-forcing-wrfout"))


def _plan(args) -> int:
    if args.topology != "hex-swath":
        given = [flag for attribute, flag in _HEX_ONLY_FLAGS
                 if getattr(args, attribute, None) not in (None, False)]
        if given:
            import sys

            print(f"woof energy plan: REFUSED: {', '.join(given)} applies "
                  f"only to --topology hex-swath; --topology {args.topology} "
                  "would ignore it", file=sys.stderr)
            return 2
    return import_module(_TOPOLOGY_MODULES[args.topology]).main(args)


def _energy_overview(args) -> int:
    print(__doc__.strip())
    return 0


def _register_fetch(sub) -> None:
    p = sub.add_parser(
        "fetch", help="fetch OpenStreetMap power infrastructure (Overpass)")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--bbox", type=_bbox, metavar="W,S,E,N",
                       help="area to fetch, degrees west,south,east,north "
                            "(write --bbox=-3.4,51.6,-3.0,51.8 when west is "
                            "negative)")
    where.add_argument("--polygon", metavar="FILE.geojson",
                       help="area to fetch as a GeoJSON Polygon/MultiPolygon")
    p.add_argument("--kinds", type=_choices_list(_KINDS, "--kinds"),
                   default=("line", "cable", "substation", "plant",
                            "generator"),
                   help="asset kinds to fetch (comma list; default "
                        "line,cable,substation,plant,generator)")
    p.add_argument("--min-voltage-kv", type=positive_float, default=None,
                   help="drop lines, cables and substations below this "
                        "voltage (assets with no voltage tag are kept)")
    p.add_argument("--endpoint", default=None, metavar="URL",
                   help="Overpass interpreter URL to ask first "
                        "(default: the built-in mirror ladder)")
    p.add_argument("--refresh", action="store_true",
                   help="revalidate cached Overpass responses")
    p.add_argument("--offline", action="store_true",
                   help="use only cached responses; refuse if any tile is "
                        "missing")
    p.add_argument("--timeout-s", type=positive_float, default=180.0,
                   help="server-side Overpass timeout per tile (default 180)")
    p.add_argument("-o", "--output", required=True, metavar="ASSETS.geojson",
                   help="where to write the woof-energy.assets.v1 document")
    p.set_defaults(func=_dispatch("woof.energy.osm"))


def _register_import(sub) -> None:
    p = sub.add_parser(
        "import", help="import grid or renewable assets from files")
    p.add_argument("sources", nargs="+", metavar="SRC",
                   help="input files (PyPSA-Eur CSV directory or files, "
                        "REPD CSV, GeoJSON, CSV)")
    p.add_argument("--format", required=True,
                   choices=("pypsa-eur", "repd", "geojson", "csv"),
                   help="input format")
    p.add_argument("--merge", default=None, metavar="ASSETS.geojson",
                   help="existing assets document to merge into (duplicates "
                        "by source reference and proximity are dropped)")
    p.add_argument("--lat-col", default=None,
                   help="CSV latitude column (--format csv)")
    p.add_argument("--lon-col", default=None,
                   help="CSV longitude column (--format csv)")
    p.add_argument("--id-col", default=None,
                   help="CSV identifier column (--format csv)")
    p.add_argument("--kind", choices=_KINDS, default=None,
                   help="asset kind for every row (--format csv/geojson "
                        "when the input does not say)")
    p.add_argument("-o", "--output", required=True, metavar="ASSETS.geojson",
                   help="where to write the woof-energy.assets.v1 document")
    p.set_defaults(func=_dispatch("woof.energy.importers"))


def _register_sites(sub) -> None:
    p = sub.add_parser("sites", help="turn assets into forecast sites")
    p.add_argument("assets", metavar="ASSETS.geojson",
                   help="woof-energy.assets.v1 document")
    p.add_argument("--spacing-m", type=positive_float, default=100.0,
                   help="sample spacing along lines and cables (default 100)")
    p.add_argument("--kinds", type=_choices_list(_SITE_KINDS, "--kinds"),
                   default=None,
                   help="asset kinds to keep (comma list; default all)")
    p.add_argument("--min-voltage-kv", type=positive_float, default=None,
                   help="drop lines and substations below this voltage")
    p.add_argument("--region", default=None, metavar="FILE.geojson",
                   help="keep only sites inside this Polygon/MultiPolygon")
    p.add_argument("--heights-m", type=_heights, default=(10.0, 30.0, 100.0),
                   help="heights above ground to sample every site at "
                        "(comma list; default 10,30,100); turbine hub "
                        "heights are added")
    p.add_argument("--include-towers", action="store_true",
                   help="add a site at every power=tower node")
    p.add_argument("--pv-grid-m", type=positive_float, default=None,
                   help="sample solar farm polygons on a grid of this "
                        "spacing (default: one site at the centroid)")
    p.add_argument("-o", "--output", required=True, metavar="SITES.json",
                   help="where to write the woof-energy.sites.v1 document")
    p.set_defaults(func=_dispatch("woof.energy.sites"))


def _register_plan(sub) -> None:
    p = sub.add_parser(
        "plan", help="plan high-resolution domains covering the sites")
    p.add_argument("sites", metavar="SITES.json",
                   help="woof-energy.sites.v1 document")
    p.add_argument("--topology", required=True, choices=tuple(_TOPOLOGY_MODULES),
                   help="wrf-nests: sibling nests in one run; wrf-tiles: one "
                        "parent run plus offline child tiles (no count "
                        "limit); hex-swath: MPAS corridor mesh")
    p.add_argument("--dx-m", type=positive_float, default=100.0,
                   help="target grid spacing over the sites (default 100)")
    p.add_argument("--corridor-km", type=positive_float, default=2.0,
                   help="half-width of the high-resolution corridor around "
                        "each site (default 2)")
    p.add_argument("--parent-dx-m", type=positive_float, default=None,
                   help="outer parent grid spacing (default: chosen by the "
                        "planner)")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DDTHH",
                   help="forecast start, UTC (default: the most recent "
                        "00/06/12/18 cycle)")
    p.add_argument("--hours", type=positive_float, default=24.0,
                   help="forecast length in hours (default 24)")
    p.add_argument("--source", default=None,
                   help="initial/boundary condition source (default: the "
                        "one woof domain emits)")
    capacity = p.add_mutually_exclusive_group()
    capacity.add_argument("--card", default=None,
                          help="size each domain for this GPU tier")
    capacity.add_argument("--vram-gib", type=positive_float, default=None,
                          help="size each domain for this many GiB")
    p.add_argument("--max-domains", type=positive_int, default=None,
                   help="refuse plans with more high-resolution domains")
    p.add_argument("--nz", type=positive_int, default=None,
                   help="vertical levels (default: the planner's ladder; "
                        "hex-swath: 55, or 3..80 priced at that column)")
    hex_lane = p.add_argument_group(
        "hex-swath experimental lane",
        "read only with --topology hex-swath; refused with any other")
    hex_lane.add_argument(
        "--hex-experimental-dt", action="store_true",
        help="EXPERIMENTAL: lift the hex-swath spacing floor (about 819 m) "
             "by planning on an unanchored sub-5 s timestep (Courant, clock "
             "and radiation checks still apply) and emit the full chain: "
             "mesh, statics, vertical, cull, met, init, boundaries and an "
             "LES forecast.  Every output is labelled "
             "experimental-unanchored")
    hex_lane.add_argument(
        "--hex-density", choices=("raster", "polygons"), default=None,
        help="with --hex-experimental-dt: hand the mesh generator a density "
             "raster built from the sites (raster, the default) or the "
             "planner's corridor polygon rows (polygons)")
    hex_lane.add_argument(
        "--hex-forcing-wrfout", default=None, metavar="GLOB",
        help="with --hex-experimental-dt: force the hex corridor from this "
             "WRF parent run's wrfout files instead of --source")
    p.add_argument("-o", "--outdir", required=True, metavar="DIR",
                   help="directory for plan.json and emitted configs")
    p.set_defaults(func=_plan)


def _register_run(sub) -> None:
    p = sub.add_parser("run", help="run every domain of a plan")
    p.add_argument("plan", metavar="PLAN.json",
                   help="woof-energy.plan.v1 document")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    p.add_argument("--only", type=_names, default=None, metavar="ID[,ID]",
                   help="run only these domain_ids (their parents must "
                        "already have run)")
    p.add_argument("--resume", action="store_true",
                   help="skip domains whose run manifest says complete")
    p.set_defaults(func=_dispatch("woof.energy.run"))


def _register_extract(sub) -> None:
    p = sub.add_parser(
        "extract", help="sample plan runs at the sites (forecast.v1)")
    p.add_argument("plan", metavar="PLAN.json",
                   help="woof-energy.plan.v1 document whose runs finished")
    p.add_argument("--sites", default=None, metavar="SITES.json",
                   help="sites document (default: the one the plan names)")
    p.add_argument("--heights-m", type=_heights, default=None,
                   help="override the sites document's heights")
    p.add_argument("--vars", type=_names, default=None, metavar="NAME[,NAME]",
                   help="forecast.v1 variables to write (default all "
                        "the runs can supply)")
    p.add_argument("--format", default="netcdf",
                   choices=("netcdf", "zarr", "icechunk", "csv"),
                   help="output format (default netcdf)")
    p.add_argument("-o", "--output", required=True, metavar="PATH",
                   help="output file or store")
    p.set_defaults(func=_dispatch("woof.energy.extract"))


def _register_rating(sub) -> None:
    p = sub.add_parser(
        "rating", help="line rating, icing and power from a forecast.v1")
    p.add_argument("forecast", metavar="FORECAST.nc",
                   help="woof-energy.forecast.v1 netCDF file")
    p.add_argument("--conductor", default="auto",
                   help="conductor name from the table, or auto (by line "
                        "voltage; default)")
    p.add_argument("--conductor-table", default=None, metavar="FILE.json",
                   help="conductor table replacing the built-in one")
    p.add_argument("--products", type=_choices_list(_PRODUCTS, "--products"),
                   default=_PRODUCTS,
                   help="products to compute (comma list; default "
                        "dlr,icing,wind-power,pv-power)")
    p.add_argument("-o", "--output", required=True, metavar="PRODUCTS.nc",
                   help="where to write the products netCDF")
    p.set_defaults(func=_dispatch("woof.energy.products"))


def _hours_of_day(value: str) -> int:
    hour = int(value)
    if not 0 <= hour <= 23:
        raise argparse.ArgumentTypeError(f"--cycle-hour {value} is not 0..23")
    return hour


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more; got {value}")
    return number


def _register_hindcast(sub) -> None:
    hindcast = sub.add_parser(
        "hindcast", help="a long reanalysis as re-initialised segment jobs")
    hindcast.set_defaults(func=_dispatch("woof.energy.hindcast"))
    verbs = hindcast.add_subparsers(dest="hindcast_command", required=False)

    p = verbs.add_parser(
        "plan", help="expand clusters x [start, end) into segment jobs")
    what = p.add_argument_group("what to plan")
    what.add_argument("--sites", default=None, metavar="SITES.json",
                      help="sites document planned as one cluster (or the "
                           "sites a --partition refers to)")
    what.add_argument("--partition", default=None, metavar="PARTITION.json",
                      help="woof-energy.partition.v1 document: one job "
                           "chain per cluster")
    what.add_argument("--clusters", type=_names, default=None,
                      metavar="ID[,ID]",
                      help="only these partition clusters")
    what.add_argument("--cluster-id", default=None,
                      help="name of the single --sites cluster (default: "
                           "the sites file's stem)")
    when = p.add_argument_group("segments (UTC)")
    when.add_argument("--start", required=True, metavar="YYYY-MM-DD[THH]",
                      help="first segment's day (its run starts at "
                           "--cycle-hour)")
    when.add_argument("--end", required=True, metavar="YYYY-MM-DD[THH]",
                      help="exclusive bound on segment run starts")
    when.add_argument("--segment-hours", type=positive_int, default=30,
                      help="run length of every segment (default 30)")
    when.add_argument("--spinup-hours", type=_non_negative_int, default=6,
                      help="leading hours discarded as spin-up (default 6); "
                           "each segment keeps segment - spinup hours")
    when.add_argument("--cycle-hour", type=_hours_of_day, default=0,
                      help="UTC hour runs start from (default 0)")
    grid = p.add_argument_group("planner passthrough")
    grid.add_argument("--topology", default="wrf-nests",
                      choices=("wrf-nests", "wrf-tiles"),
                      help="planner for clusters that do not name one "
                           "(default wrf-nests)")
    grid.add_argument("--dx-m", type=positive_float, default=None,
                      help="leaf spacing (default: the cluster's, else 100)")
    grid.add_argument("--corridor-km", type=positive_float, default=2.0,
                      help="corridor half-width (default 2)")
    grid.add_argument("--parent-dx-m", type=positive_float, default=None,
                      help="outer parent spacing (default: the cluster "
                           "chain's, else the planner's)")
    grid.add_argument("--nz", type=positive_int, default=None,
                      help="vertical levels (default: the planner's)")
    grid.add_argument("--p-top-pa", type=positive_float, default=5000.0,
                      help="model top the source must reach and the plan "
                           "must carry (default 5000 Pa, about 20 km)")
    capacity = grid.add_mutually_exclusive_group()
    capacity.add_argument("--card", default=None,
                          help="size each domain for this GPU tier")
    capacity.add_argument("--vram-gib", type=positive_float, default=None,
                          help="size each domain for this many GiB")
    grid.add_argument("--max-domains", type=positive_int, default=None,
                      help="refuse plans with more high-resolution domains")
    forcing = p.add_argument_group("boundary source")
    forcing.add_argument("--source", default=None,
                         help="one source for every segment (default: "
                              "chosen per segment by the source selector)")
    forcing.add_argument("--source-priority", type=_names, default=None,
                         metavar="SRC[,SRC]",
                         help="preference order handed to the selector")
    out = p.add_argument_group("history and outputs")
    out.add_argument("--leaf-history-interval-s", type=positive_float,
                     default=900.0,
                     help="leaf history cadence (default 900; finer such "
                          "as 300 allowed)")
    out.add_argument("--parent-history-interval-s", type=positive_float,
                     default=None,
                     help="wrf-nests parent cadence (default 3600)")
    out.add_argument("--output-root", default="icechunk", metavar="URL",
                     help="root of the output stores, e.g. s3://bucket/x "
                          "(default: icechunk/ in the hindcast directory)")
    out.add_argument("--gridded-url",
                     default="{root}/gridded/{cluster_id}/{leaf}.icechunk",
                     metavar="TEMPLATE",
                     help="gridded store per cluster leaf")
    out.add_argument("--sites-url",
                     default="{root}/sites/{cluster_id}.icechunk",
                     metavar="TEMPLATE", help="site-series store per cluster")
    p.add_argument("--dry-run", action="store_true",
                   help="print the manifest and an example job; write "
                        "nothing")
    p.add_argument("--resume", action="store_true",
                   help="merge into an existing manifest, keeping statuses")
    p.add_argument("-o", "--outdir", required=True, metavar="DIR",
                   help="hindcast directory (manifest, jobs, clusters)")

    p = verbs.add_parser("status", help="job counts, coverage and sources")
    p.add_argument("dir", metavar="DIR", help="hindcast directory")
    p.add_argument("--cluster", default=None, help="only this cluster")
    p.add_argument("--status", default=None,
                   choices=("pending", "materialized", "running", "complete",
                            "failed"),
                   help="list the jobs in this state")
    p.add_argument("--jobs", action="store_true", help="list every job")

    for verb, text in (("materialize", "write a job's segment plan"),
                       ("run-local", "run a job on this machine")):
        p = verbs.add_parser(verb, help=text)
        p.add_argument("job", metavar="JOB",
                       help="job.json, its directory, or a job id with --dir")
        p.add_argument("--dir", default=None, metavar="DIR",
                       help="hindcast directory a bare job id is in")
        p.add_argument("--force", action="store_true",
                       help="re-plan (earlier plan moved aside)"
                       if verb == "materialize" else
                       "run a job the manifest calls complete")
    p.add_argument("--dry-run", action="store_true",
                   help="materialize and print woof energy run's argv")
    p.add_argument("--resume", action="store_true",
                   help="pass --resume to woof energy run")
    p.add_argument("--no-extract", action="store_true",
                   help="skip extracting the kept window at the sites")
    p.add_argument("--writer", default=None, metavar="MODULE:CALLABLE",
                   help="callable handed the kept-window forecast and "
                        "targets (e.g. the Icechunk writer)")


def register_cli(subparsers) -> None:
    """Add ``woof energy`` to the product CLI."""

    energy = subparsers.add_parser(
        "energy",
        help="high-resolution forecasts along power lines and renewables")
    energy.set_defaults(func=_energy_overview)
    sub = energy.add_subparsers(dest="energy_command", required=False)
    _register_fetch(sub)
    _register_import(sub)
    _register_sites(sub)
    _register_plan(sub)
    _register_run(sub)
    _register_extract(sub)
    _register_rating(sub)
    _register_hindcast(sub)


__all__ = ["register_cli"]
