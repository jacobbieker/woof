"""``woof energy``: forecasts along power lines, substations and renewables.

This module owns argument parsing only.  Each subcommand hands its parsed
arguments to ``main(args)`` in the module that does the work, imported when
the subcommand runs::

    woof energy fetch     woof.energy.osm             OSM power data via Overpass
    woof energy import    woof.energy.importers       PyPSA-Eur, REPD, GeoJSON, CSV,
                                                      USWTDB, GEM, EMODnet, OSM pbf
    woof energy inventory woof.energy.inventory_global  merge / download wind inventories
    woof energy sites     woof.energy.sites           assets -> forecast sites
    woof energy plan      woof.energy.plan_wrf_nests  --topology wrf-nests
                          woof.energy.plan_wrf_tiles  --topology wrf-tiles
                          woof.energy.plan_hex        --topology hex-swath
    woof energy run       woof.energy.run             run every plan domain
    woof energy extract   woof.energy.extract         plan runs -> forecast.v1
    woof energy rating    woof.energy.products        line rating, icing, power
    woof energy partition woof.energy.partition       worldwide sites -> clusters
    woof energy plan-cluster  woof.energy.partition   plan one cluster
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
_IMPORT_FORMATS = ("pypsa-eur", "repd", "geojson", "csv",
                   "uswtdb", "gem", "emodnet", "osm-pbf")
#: ``woof:status`` vocabulary (woof.energy.inventory_global.STATUSES).
_STATUSES = ("operating", "construction", "pre-construction", "announced",
             "shelved", "mothballed", "cancelled", "retired", "unknown")
_INVENTORY_DATASETS = ("uswtdb", "emodnet", "gem", "osm")


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


def _dispatch(module: str, function: str = "main"):
    def run(args) -> int:
        return getattr(import_module(module), function)(args)
    return run


#: ``woof energy plan`` flags only the hex-swath planner reads.
_HEX_ONLY_FLAGS = (("hex_experimental_dt", "--hex-experimental-dt"),
                   ("hex_density", "--hex-density"),
                   ("hex_forcing_wrfout", "--hex-forcing-wrfout"))


#: Topologies whose planners take ``--source auto`` and
#: ``--source-priority`` (woof.energy.source_select).
_AUTO_SOURCE_TOPOLOGIES = ("wrf-nests", "wrf-tiles")


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
    if args.topology not in _AUTO_SOURCE_TOPOLOGIES:
        from woof.energy.source_select import is_auto

        given = (["--source auto"] if is_auto(args.source) else []) + (
            ["--source-priority"]
            if getattr(args, "source_priority", None) is not None else [])
        if given:
            import sys

            print(f"woof energy plan: REFUSED: {', '.join(given)} applies "
                  f"only to --topology {' or '.join(_AUTO_SOURCE_TOPOLOGIES)}"
                  f"; name the source for --topology {args.topology}",
                  file=sys.stderr)
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
                        "REPD CSV, GeoJSON, CSV, USWTDB CSV/JSON, GEM "
                        "xlsx/CSV, EMODnet GeoJSON, OSM .osm.pbf)")
    p.add_argument("--format", required=True, choices=_IMPORT_FORMATS,
                   help="input format")
    p.add_argument("--status", type=_choices_list(_STATUSES, "--status"),
                   default=None,
                   help="development statuses to keep (comma list; "
                        "default operating for uswtdb/gem/emodnet/osm-pbf, "
                        "no filter for repd; other formats refuse it)")
    p.add_argument("--osm-index", default=None, metavar="IDX",
                   help="pyosmium node-location index for --format osm-pbf "
                        "(default flex_mem; sparse_file_array,FILE for the "
                        "planet)")
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


def _register_inventory(sub) -> None:
    p = sub.add_parser(
        "inventory", help="merge and fetch global wind-farm inventories")
    inv = p.add_subparsers(dest="inventory_command", required=True)
    m = inv.add_parser(
        "merge", help="deduplicate wind farms across assets documents")
    m.add_argument("assets", nargs="+", metavar="ASSETS.geojson",
                   help="woof-energy.assets.v1 documents to merge")
    m.add_argument("--status", type=_choices_list(_STATUSES, "--status"),
                   default=("operating",),
                   help="development statuses to keep (comma list; "
                        "default operating)")
    m.add_argument("--keep-non-wind", action="store_true",
                   help="pass non-wind assets through unmatched instead of "
                        "dropping them")
    m.add_argument("--turbine-match-m", type=positive_float, default=50.0,
                   help="turbines of two sources this close are one "
                        "(default 50)")
    m.add_argument("--plant-near-m", type=positive_float, default=1000.0,
                   help="plants this close match unless name or capacity "
                        "disagree (default 1000)")
    m.add_argument("--plant-match-m", type=positive_float, default=5000.0,
                   help="plants this close match when their names agree "
                        "(default 5000)")
    m.add_argument("--name-threshold", type=float, default=0.7,
                   help="name similarity that counts as agreement "
                        "(default 0.7)")
    m.add_argument("--name-floor", type=float, default=0.3,
                   help="name similarity below which names disagree "
                        "(default 0.3)")
    m.add_argument("--capacity-tolerance", type=float, default=0.25,
                   help="relative capacity difference still agreeing "
                        "(default 0.25)")
    m.add_argument("--plant-link-m", type=positive_float, default=10000.0,
                   help="link turbines to a plant this far away when the "
                        "project names agree (default 10000)")
    m.add_argument("--plant-link-near-m", type=positive_float,
                   default=2000.0,
                   help="link unnamed turbines to a plant this close "
                        "(default 2000)")
    m.add_argument("--turbine-coverage", type=float, default=0.8,
                   help="fraction of plant capacity linked turbines must "
                        "cover for the plant to be turbine-level "
                        "(default 0.8)")
    m.add_argument("--prefer", type=_names, default=None,
                   metavar="SRC[,SRC]",
                   help="source order for each duplicate cluster's winner "
                        "(default osm,uswtdb,emodnet,repd,gem)")
    m.add_argument("-o", "--output", required=True, metavar="ASSETS.geojson",
                   help="where to write the merged woof-energy.assets.v1 "
                        "document")
    m.set_defaults(func=_dispatch("woof.energy.inventory_global"))
    d = inv.add_parser(
        "download", help="fetch a public inventory dataset (network)")
    d.add_argument("--dataset", required=True, choices=_INVENTORY_DATASETS,
                   help="uswtdb (API JSON) or emodnet (WFS GeoJSON); gem "
                        "and osm print where to get them")
    d.add_argument("--timeout-s", type=positive_float, default=300.0,
                   help="network timeout (default 300)")
    d.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="where to save the download")
    d.set_defaults(func=_dispatch("woof.energy.inventory_global"))


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
                        "one woof domain emits); auto (wrf-tiles, "
                        "wrf-nests) picks the first source of "
                        "--source-priority that covers the sites, holds "
                        "the --start cycle and reaches --hours, and "
                        "writes the woof-energy.source-choice.v1 receipt")
    p.add_argument("--source-priority", type=_names, default=None,
                   metavar="SRC[,SRC]",
                   help="with --source auto: candidate sources, best "
                        "first (default hrrr-native,hrrr,rap,"
                        "met-office-global,ecmwf-open-data,gfs,era5)")
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


#: ``woof energy partition`` defaults, held equal to the constants of
#: :mod:`woof.energy.partition` by its tests (the module is imported only
#: when the subcommand runs).
_PARTITION_DEFAULTS = {"linkage_km": 30.0, "max_extent_km": 250.0,
                       "max_sites": 20_000, "leaf_dx_m": 200.0,
                       "corridor_km": 2.0}
_PARTITION_TOPOLOGIES = ("auto", "wrf-tiles", "wrf-nests")


def _register_partition(sub) -> None:
    d = _PARTITION_DEFAULTS

    def capacity(parser, default: str) -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--card", default=None,
                           help="size for this GPU tier or model (no local "
                                "probe); " + default)
        group.add_argument("--vram-gib", type=positive_float, default=None,
                           help="size for this many GiB (no local probe)")
        group.add_argument("--hardware-json", default=None,
                           metavar="FILE.json",
                           help="size for a selected target hardware "
                                "snapshot's measured GPU capacity (no local "
                                "probe)")

    p = sub.add_parser(
        "partition",
        help="cut a worldwide sites set into independent regional clusters")
    p.add_argument("sites", metavar="SITES.json",
                   help="woof-energy.sites.v1 document")
    p.add_argument("--linkage-km", type=positive_float,
                   default=d["linkage_km"],
                   help="single-linkage distance: sites closer than this "
                        f"share a cluster (default {d['linkage_km']:g})")
    p.add_argument("--max-extent-km", type=positive_float,
                   default=d["max_extent_km"],
                   help="bisect clusters wider than this "
                        f"(default {d['max_extent_km']:g})")
    p.add_argument("--max-sites", type=positive_int,
                   default=d["max_sites"],
                   help="bisect clusters with more sites than this "
                        f"(default {d['max_sites']})")
    p.add_argument("--leaf-dx-m", type=positive_float,
                   default=d["leaf_dx_m"],
                   help="leaf grid spacing, 100 to 300 m "
                        f"(default {d['leaf_dx_m']:g})")
    p.add_argument("--corridor-km", type=positive_float,
                   default=d["corridor_km"],
                   help="half-width of the high-resolution corridor around "
                        f"each site (default {d['corridor_km']:g})")
    p.add_argument("--nz", type=positive_int, default=None,
                   help="leaf vertical levels the tile sizes are estimated "
                        "at (default: the tile planner's ladder)")
    p.add_argument("--topology", choices=_PARTITION_TOPOLOGIES, default="auto",
                   help="planner every cluster is recommended for (auto: "
                        "wrf-nests for a single small farm, else wrf-tiles)")
    capacity(p, "default 24gb")
    p.add_argument("-o", "--output", required=True, metavar="PARTITION.json",
                   help="where to write the woof-energy.partition.v1 "
                        "document")
    p.set_defaults(func=_dispatch("woof.energy.partition"))

    q = sub.add_parser(
        "plan-cluster",
        help="plan one cluster of a partition with its own parent")
    q.add_argument("partition", metavar="PARTITION.json",
                   help="woof-energy.partition.v1 document")
    q.add_argument("--cluster", required=True, metavar="ID",
                   help="cluster_id to plan")
    q.add_argument("--topology", choices=("wrf-tiles", "wrf-nests"),
                   default=None,
                   help="planner to use (default: the cluster's "
                        "recommendation)")
    q.add_argument("--dx-m", type=positive_float, default=None,
                   help="leaf grid spacing (default: the cluster's)")
    q.add_argument("--corridor-km", type=positive_float, default=None,
                   help="corridor half-width (default: the partition's)")
    q.add_argument("--parent-dx-m", type=positive_float, default=None,
                   help="outer parent grid spacing (default: chosen by the "
                        "planner)")
    q.add_argument("--start", default=None, metavar="YYYY-MM-DDTHH",
                   help="run start, UTC (default: the planner's)")
    q.add_argument("--hours", type=positive_float, default=24.0,
                   help="run length in hours (default 24)")
    q.add_argument("--source", default=None,
                   help="initial/boundary condition source (default: the "
                        "one woof domain emits)")
    capacity(q, "default: the card the partition was sized for")
    q.add_argument("--max-domains", type=positive_int, default=None,
                   help="refuse plans with more high-resolution domains")
    q.add_argument("--nz", type=positive_int, default=None,
                   help="vertical levels (default: the planner's ladder)")
    q.add_argument("--dry-run", action="store_true",
                   help="write the cluster's sites.json and print the "
                        "planner call without planning")
    q.add_argument("-o", "--outdir", required=True, metavar="DIR",
                   help="directory for sites.json, plan.json and emitted "
                        "configs")
    q.set_defaults(func=_dispatch("woof.energy.partition", "plan_cluster_main"))


def register_cli(subparsers) -> None:
    """Add ``woof energy`` to the product CLI."""

    energy = subparsers.add_parser(
        "energy",
        help="high-resolution forecasts along power lines and renewables")
    energy.set_defaults(func=_energy_overview)
    sub = energy.add_subparsers(dest="energy_command", required=False)
    _register_fetch(sub)
    _register_import(sub)
    _register_inventory(sub)
    _register_sites(sub)
    _register_plan(sub)
    _register_run(sub)
    _register_extract(sub)
    _register_rating(sub)
    _register_partition(sub)


__all__ = ["register_cli"]
