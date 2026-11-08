"""The ``woof hex`` console script: every front door this package ships."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from .errors import MpasPortError

# Every door module is imported only when its parser is built, and a door
# command builds only its own parser (``build_parser(command)``).  THE
# BREAKAGE THIS ENDS (member-leg start-up measure, RTX 5090,
# 2026-10-28): every ``woof hex forecast`` imported all 36 doors (country,
# swath, cycle, ensemble, global, ...) before the door opened, and every
# spawn compile worker of a cold start re-ran this module as ``__mp_main__``
# and so imported the forecast, init and cull doors too.  ``.mesh`` and
# ``.oracle`` stay inside the handlers that use them: the ``render`` door must
# run on a render node whose Python has neither numpy nor netCDF4.
#
# The checkout probe lives with the forecast door (``forecast_door.
# checkout_root``), stated once; the mesh defaults below are read from it on
# first use (module ``__getattr__``), not at import.
_CHECKOUT_DEFAULTS = {
    "DEFAULT_GRID": ("data", "meshes", "x1.2562", "x1.2562.grid.nc"),
    "DEFAULT_STATIC": ("data", "meshes", "x1.2562", "x1.2562.static.nc"),
    "DEFAULT_ORACLE": ("oracle", "x1.2562"),
}


def __getattr__(name: str):
    if name == "PROJECT_ROOT":
        from .forecast_door import PROJECT_ROOT

        return PROJECT_ROOT
    if name in _CHECKOUT_DEFAULTS:
        root = __getattr__("PROJECT_ROOT")
        return root.joinpath(*_CHECKOUT_DEFAULTS[name]) if root is not None else None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _require_mesh_paths(arguments: argparse.Namespace) -> None:
    for flag, value in (("--grid", arguments.grid), ("--static", arguments.static)):
        if value is None:
            raise MpasPortError(
                f"{flag} was not given and there is no default outside a source "
                "checkout.  Mesh grid and static files are external assets: "
                "woof hex ships neither and has no general fetch path for them "
                "(verify-dycore downloads only its fixed small test mesh), so a "
                "guessed default would name a file that does not exist.  Obtain "
                "the pair from the MPAS-Atmosphere mesh downloads and pass both "
                "--grid and --static explicitly (see README, 'Assets you must "
                "supply')."
            )


def _mesh(arguments: argparse.Namespace):
    from .mesh import Mesh

    _require_mesh_paths(arguments)
    return Mesh.from_netcdf(arguments.grid, arguments.static)


def _version(arguments: argparse.Namespace) -> int:
    from . import DISTRIBUTION_NAME, __version__
    from .forecast_door import PROJECT_ROOT

    print(
        json.dumps(
            {
                "distribution": DISTRIBUTION_NAME,
                "version": __version__,
                "package": str(Path(__file__).resolve().parent),
                "source_checkout": (
                    str(PROJECT_ROOT) if PROJECT_ROOT is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _mesh_check(arguments: argparse.Namespace) -> int:
    from .dual_edge_admission import DualEdgeAdmissionError, admit_dual_edges
    from .oracle import sha256_file

    if getattr(arguments, "grid_only", False):
        # A culled regional grid exists before its static does (the culler
        # emits the grid first), and refusing to examine it until a static
        # exists would leave the one file the whole regional chain builds on
        # unvalidatable at the door users are told to run first.
        if arguments.grid is None:
            raise MpasPortError(
                "--grid was not given.  --grid-only validates one grid file "
                "and there is no default outside a source checkout; pass "
                "--grid explicitly."
            )
        from .mesh import Mesh

        mesh = Mesh.from_netcdf(arguments.grid)
        static_path = None
        static_sha = None
    else:
        mesh = _mesh(arguments)
        static_path = str(Path(arguments.static).resolve())
        static_sha = sha256_file(arguments.static)
    receipt = {
        "passed": True,
        "grid": str(Path(arguments.grid).resolve()),
        "grid_sha256": sha256_file(arguments.grid),
        "static": static_path,
        "static_sha256": static_sha,
        "dimensions": {
            name: int(mesh.dimensions[name])
            for name in ("nCells", "nEdges", "nVertices", "maxEdges")
        },
        "connectivity_indexing": mesh.provenance["connectivity_indexing"],
    }
    if mesh.is_regional:
        import numpy as np

        from .mesh import (
            REGIONAL_BOUNDARY_MASK_NAMES,
            regional_boundary_mask_digest,
        )

        mask_cell = np.asarray(mesh.arrays["bdyMaskCell"])
        receipt["regional"] = {
            "boundary_zone_width": int(
                max(
                    int(np.asarray(mesh.arrays[name]).max())
                    for name in REGIONAL_BOUNDARY_MASK_NAMES
                )
            ),
            "euler_characteristic": int(
                mesh.dimensions["nCells"]
                - mesh.dimensions["nEdges"]
                + mesh.dimensions["nVertices"]
            ),
            "ring_cell_counts": [
                int(np.count_nonzero(mask_cell == ring)) for ring in range(8)
            ],
            "bdy_mask_sha256": regional_boundary_mask_digest(mesh.arrays),
        }
    # THE BREAKAGE THIS PREVENTS: a pair that passes mesh-check and then dies
    # inside step 0 of every forecast.  The registered v15.150.38857 did
    # exactly that -- this receipt said "passed": true while its worst Voronoi
    # edge was 6.5 m -- and mesh-check is the door a user is told to run
    # BEFORE anything expensive touches the mesh.  The forecast bind applies
    # the same floor; agreeing here is what makes the earlier door worth
    # running.
    try:
        admission = admit_dual_edges(
            mesh.dvEdge,
            mesh.dcEdge,
            cells_on_edge=mesh.cellsOnEdge,
            cells_on_edge_base=0,
            mesh_name=str(
                Path(arguments.grid).name
                if static_path is None
                else Path(arguments.static).name
            ),
        )
    except DualEdgeAdmissionError as error:
        receipt["passed"] = False
        receipt["dual_edge_refusal"] = str(error)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    receipt["dual_edge_admission"] = admission.as_dict()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _oracle_gate(arguments: argparse.Namespace) -> int:
    from .oracle import run_m1_oracle

    if arguments.fixtures is None:
        raise MpasPortError(
            "--fixtures was not given and there is no default outside a source "
            "checkout.  The M1 oracle replays source-extracted Fortran fixtures "
            "that live in the port's own checkout, not in the installed wheel; "
            "without them there is nothing to replay.  Pass --fixtures pointing "
            "at the checkout's oracle directory."
        )
    report = run_m1_oracle(_mesh(arguments), arguments.fixtures)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _render(arguments: argparse.Namespace) -> int:
    from .render_door import run_render

    return run_render(arguments)


def _static_highres(arguments: argparse.Namespace) -> int:
    from .static_highres import run_cli

    return run_cli(arguments)


#: ``woof hex static-highres`` choices; held equal to
#: ``static_highres.TERRAIN_CHOICES``/``LANDUSE_CHOICES`` by a test.
STATIC_HIGHRES_TERRAIN = ("glo30", "none")
STATIC_HIGHRES_LANDUSE = ("cglc", "none")
STATIC_HIGHRES_MAX_SPACING_KM = 2.0


def _add_static_highres_parser(commands) -> None:
    parser = commands.add_parser(
        "static-highres",
        help="replace a hex static's 30-arc-second terrain (and optionally "
             "land use) with Copernicus GLO-30 / CGLC-MODIS-LCZ cell "
             "averages; run BEFORE the vertical build and mesh registration",
        description=(
            "Post-process an rw_mpas_static file for 50-100 m cells: "
            "area-average Copernicus DEM GLO-30 (~30 m) over each Voronoi "
            "cell into ter and, with --landuse cglc, take the per-cell area "
            "mode of CGLC-MODIS-LCZ (100 m) for ivgtyp/lu_index with "
            "landmask re-derived.  ORDER: run before `woof hex vertical` and "
            "`woof hex register`: the vertical build smooths ter into its "
            "height coordinate and the mesh-row registry pins the static's "
            "SHA-256, so both must see the post-processed file.  Orographic "
            "GWD statistics (var2d, con, oa*, ol*) are kept: they are "
            "30-arc-second box statistics by definition."))
    parser.add_argument("--static", type=Path, required=True, metavar="S",
                        help="the rw_mpas_static output to post-process")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        metavar="S2", help="the new static file to write")
    parser.add_argument("--terrain", choices=STATIC_HIGHRES_TERRAIN,
                        default="glo30",
                        help="terrain source (default glo30: Copernicus DEM "
                             "GLO-30); none keeps ter")
    parser.add_argument("--landuse", choices=STATIC_HIGHRES_LANDUSE,
                        default="none",
                        help="land-use source (cglc: CGLC-MODIS-LCZ 100 m, a "
                             "2.28 GB one-time download); default none keeps "
                             "ivgtyp/landmask")
    parser.add_argument("--max-spacing-km", type=float,
                        default=STATIC_HIGHRES_MAX_SPACING_KM, metavar="KM",
                        help="post-process only cells this fine or finer "
                             f"(default {STATIC_HIGHRES_MAX_SPACING_KM:g}); "
                             "coarser cells keep their 30-arc-second fields")
    parser.add_argument("--sample-m", type=float, default=None, metavar="M",
                        help="sample-lattice spacing in metres (default: "
                             "automatic, no coarser than the source pixel and "
                             "fine enough that every cell holds samples)")
    parser.add_argument("--cache-root", type=Path, default=None, metavar="DIR",
                        help="high-resolution tile cache (default: the "
                             "per-user woof highres-cache shared with the WRF "
                             "path)")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network; refuse when a needed "
                             "tile or raster is not already cached")
    parser.add_argument("--receipt", type=Path, default=None, metavar="FILE",
                        help="receipt path (default: <S2>.static-highres.json)")
    parser.add_argument("--clobber", action="store_true",
                        help="replace an existing -o file")
    parser.set_defaults(handler=_static_highres)
def _density(arguments: argparse.Namespace) -> int:
    from .density import run_density

    return run_density(arguments)


def _add_density_parser(commands) -> None:
    density = commands.add_parser(
        "density",
        help="write a woof-hex.density.v1 spacing raster from energy "
             "corridors and/or terrain gradient",
        description=(
            "Build a requested-spacing raster (woof-hex.density.v1, CF "
            "netCDF) for a raster-region mesh spec.  Sources combine by "
            "minimum spacing; the result is graded and limited so spacing "
            "changes by at most the generator's per-cell bound (3.06 %/cell "
            "by default).  Prints a JSON summary."),
    )
    density.add_argument("--sites", type=Path, default=None,
                         metavar="SITES.json",
                         help="woof-energy.sites.v1 document; fine spacing "
                              "within --corridor-km of its per-asset chains")
    density.add_argument("--assets", type=Path, default=None,
                         metavar="ASSETS.geojson",
                         help="woof-energy.assets.v1 GeoJSON; fine spacing "
                              "within --corridor-km of every asset")
    density.add_argument("--terrain-gradient", action="store_true",
                         help="refine where GLO-30 terrain is steep or "
                              "strongly curved (needs --bbox or --polygon)")
    density.add_argument("--bbox", default=None, metavar="W,S,E,N",
                         help="terrain region, degrees west,south,east,north "
                              "(write --bbox=-3.6,51.7,-3.3,51.9 when west is "
                              "negative)")
    density.add_argument("--polygon", type=Path, default=None,
                         metavar="FILE.geojson",
                         help="terrain region as a GeoJSON Polygon/"
                              "MultiPolygon")
    density.add_argument("--fine-km", type=float, required=True,
                         help="finest requested spacing, km")
    density.add_argument("--background-km", type=float, required=True,
                         help="spacing everywhere nothing asks for less, km")
    density.add_argument("--corridor-km", type=float, default=2.0,
                         help="corridor half-width around lines and sites, "
                              "km (default 2)")
    density.add_argument("--grade-percent-per-cell", type=float, default=None,
                         help="grade outward at this %%/cell instead of the "
                              "bound in force (refused above it)")
    density.add_argument("--allow-rough", action="store_true",
                         help="WORKAROUND: limit at the generator's 12.25 "
                              "%%/cell transition-band ceiling instead of the "
                              "3.06 %%/cell woof mesh smoothness bound")
    density.add_argument("--cells-per-finest", type=float, default=4.0,
                         help="target raster cells per finest spacing "
                              "(default 4)")
    density.add_argument("--max-raster-cells", type=int, default=30_000_000,
                         help="refuse rasters larger than this (default "
                              "30000000)")
    density.add_argument("--interp-tolerance", type=float, default=0.10,
                         help="when the size limit coarsens the raster, the "
                              "largest spacing error it may introduce, as a "
                              "fraction of the finest spacing (default 0.10)")
    terrain = density.add_argument_group("terrain gradient")
    terrain.add_argument("--terrain-fine-km", type=float, default=None,
                         help="finest spacing terrain asks for (default "
                              "--fine-km)")
    terrain.add_argument("--terrain-scale-km", type=float, default=None,
                         help="length the terrain is judged at; the DEM is "
                              "area-averaged to half of it (default the "
                              "terrain fine spacing)")
    terrain.add_argument("--slope-coarse-deg", type=float, default=5.0,
                         help="slope at and below which terrain asks for "
                              "nothing (default 5)")
    terrain.add_argument("--slope-fine-deg", type=float, default=25.0,
                         help="slope at and above which terrain asks for the "
                              "terrain fine spacing (default 25)")
    terrain.add_argument("--curvature-coarse", type=float, default=2.0e-4,
                         help="|laplacian of elevation|, 1/m, at and below "
                              "which terrain asks for nothing (default 2e-4)")
    terrain.add_argument("--curvature-fine", type=float, default=2.0e-3,
                         help="|laplacian of elevation|, 1/m, at and above "
                              "which terrain asks for the fine spacing "
                              "(default 2e-3)")
    terrain.add_argument("--cache-root", type=Path, default=None,
                         help="GLO-30 tile cache (default the per-user "
                              "highres cache)")
    terrain.add_argument("--offline", action="store_true",
                         help="read cached GLO-30 tiles only; refuse on a "
                              "missing tile")
    density.add_argument("-o", "--output", type=Path, required=True,
                         help="raster to write (netCDF)")
    density.set_defaults(handler=_density)


def _add_mesh_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grid", type=Path, default=__getattr__("DEFAULT_GRID"))
    parser.add_argument("--static", type=Path, default=__getattr__("DEFAULT_STATIC"))


def _door(module: str, adder: str) -> Callable[[argparse._SubParsersAction], None]:
    """A door's parser adder, imported when (and only when) that door's parser is built."""

    def add(commands: argparse._SubParsersAction) -> None:
        getattr(importlib.import_module(module, __package__), adder)(commands)

    return add


def _add_version_parser(commands: argparse._SubParsersAction) -> None:
    version = commands.add_parser(
        "version", help="report the installed distribution and version"
    )
    version.set_defaults(handler=_version)


def _add_verify_parsers(commands: argparse._SubParsersAction) -> None:
    from .verify_dycore_door import add_verify_dycore_parser
    add_verify_dycore_parser(commands)

    from .verify_mpas import add_verify_mpas_parser
    add_verify_mpas_parser(commands)
    from .verify_mpas import run_cached_dycore
    cached = commands.choices["verify-dycore"]
    cached.set_defaults(legacy_dycore_handler=cached.get_default("handler"), handler=run_cached_dycore)
    cached.add_argument("--cached-deck", type=Path, help="verify a pinned cached dycore forecast without rebuilding native")
    cached.add_argument("--receipt", type=Path)
    cached.add_argument("--ratchet", type=Path)
    cached.add_argument("--comparator", type=Path)


def _add_mesh_check_parser(commands: argparse._SubParsersAction) -> None:
    mesh_check = commands.add_parser("mesh-check", help="validate an MPAS mesh")
    _add_mesh_paths(mesh_check)
    mesh_check.add_argument(
        "--grid-only",
        action="store_true",
        help=(
            "validate the grid file alone (a culled regional grid exists "
            "before its static does); any --static is not read in this mode "
            "and the receipt records static: null"
        ),
    )
    mesh_check.set_defaults(handler=_mesh_check)


def _add_oracle_gate_parser(commands: argparse._SubParsersAction) -> None:
    oracle_gate = commands.add_parser(
        "oracle-gate", help="replay the source-extracted Fortran M1 fixtures"
    )
    _add_mesh_paths(oracle_gate)
    oracle_gate.add_argument("--fixtures", type=Path, default=__getattr__("DEFAULT_ORACLE"))
    oracle_gate.set_defaults(handler=_oracle_gate)


def _add_render_parser(commands: argparse._SubParsersAction) -> None:
    render = commands.add_parser(
        "render",
        help="MPAS history -> product PNGs through the Rust path "
             "(rw_mpas_convert + rw_wrfbatch)",
    )
    from .render_door import add_render_arguments

    add_render_arguments(render)
    render.set_defaults(handler=_render)


#: Every door, in ``--help`` order: (the commands it registers, its adder).
#: ``build_parser(command)`` builds only the row that registers ``command``;
#: tests/test_cli_lazy_doors.py holds each row to the commands its adder
#: really registers, and a row that disagrees falls back to every door.
#: Why the order is what it is:
#:
#: * ``doctor`` early: it is the command a fresh install should reach for.
#:   This distribution cannot carry the Rust engines, the CUDA runtime or the
#:   mesh assets, so a bare ``pip install`` lands something real that is not
#:   yet able to run; doctor names that gap with the command that closes it.
#: * ``mesh-plan`` before ``mesh-check``, the order a mesh happens in: a spec
#:   is asked whether it can be built, and only then is a built mesh validated
#:   (``rw_mpas_mesh --dry-run`` sizes a spec its own build then refuses, once
#:   on pre-run arithmetic and once after 711 s of relaxation).
#: * ``density`` beside ``mesh-plan`` (fork): a density raster is what a
#:   raster-region mesh spec is planned from.
#: * ``static-highres`` and ``vertical`` before cull (fork): static-highres
#:   rewrites the static's terrain, which the vertical build smooths and the
#:   mesh-row registry pins by SHA-256; the vertical artifact is minted on the
#:   GLOBAL parent and then cut with the grid and static (``cull
#:   --parent-vertical``).  ``register`` after cull: a cull row is registered
#:   from the cull's receipt.
#: * ``intermediate`` between cull and init, where a regional case meets it: a
#:   projected product (HRRR) is resampled onto a regular lat-lon intermediate
#:   the init and boundary engines can read.  The global lat-lon sources
#:   dispatch to woof.hex.met_intermediate_door (the Rust met_intermediate
#:   writer, no regridding); ICON global (icon-global, icon-global-ml) through
#:   a gdt101_remap stage (woof.hex.gdt101_door) first, and met-office-global
#:   to woof.hex.metoffice_intermediate, which takes the missing land state
#:   from the same-cycle GDAS.
#: * ``init-source`` beside init, because it ends in init: one table-driven
#:   path from any declared source to the same init door (a new source model
#:   is a table row, not a code path).  ``lbc`` after init: it consumes the
#:   culled init.  ``remap`` after lbc (fork): a state the last cycle ended
#:   on, carried onto the mesh the next cycle regenerated.
#: * ``forecast`` between init and render: an init is what a forecast starts
#:   from and history is what a render consumes.  ``moving`` right after it
#:   (a moving run is a chain of forecasts); ``pair`` with it (it IS the
#:   forecast door, run twice and differenced); ``swath`` after it and before
#:   render, the order a cascade meets them.
#: * ``adapt`` beside cycle (fork): criteria -> density raster -> hysteresis
#:   -> the next cycle's commands.
#: * ``physics-resources`` right after ``physics``: the MPAS suites it lists
#:   read the tables and device maths this command builds.
#: * ``products`` after render (it drives render), ``export`` after products
#:   (products without the drawing), ``add-nest`` beside export.
#: * ``global`` and the doors after it compose every door above from a row;
#:   ``regional`` is the limited-area counterpart of ``global``.
_DOORS: tuple[tuple[tuple[str, ...], Callable[[argparse._SubParsersAction], None]], ...] = (
    (("tc-verify",), _door(".tc_verify", "register_cli")),
    (("country",), _door(".country_door", "add_country_parser")),
    (("version",), _add_version_parser),
    (("verify-dycore", "verify-mpas"), _add_verify_parsers),
    (("doctor",), _door(".doctor", "add_doctor_parser")),
    (("mesh-plan",), _door(".mesh_plan_door", "add_mesh_plan_parser")),
    (("density",), _add_density_parser),
    (("mesh-check",), _add_mesh_check_parser),
    (("oracle-gate",), _add_oracle_gate_parser),
    (("static-highres",), _add_static_highres_parser),
    (("vertical",), _door(".vertical_door", "add_vertical_parser")),
    (("cull",), _door(".cull_door", "add_cull_parser")),
    (("register",), _door(".register_door", "add_register_parser")),
    (("intermediate",), _door(".hrrr_intermediate", "add_intermediate_parser")),
    (("init",), _door(".init_door", "add_init_parser")),
    (("surface-update",), _door(".init_door", "add_surface_update_parser")),
    (("init-source",), _door(".init_sources", "add_init_source_parser")),
    (("surface-series",), _door(".surface_series", "add_surface_series_parser")),
    (("lbc",), _door(".hrrr_intermediate", "add_lbc_parser")),
    (("remap",), _door(".remap_door", "add_remap_parser")),
    (("forecast",), _door(".forecast_door", "add_forecast_parser")),
    (("moving",), _door(".moving_door", "add_moving_parser")),
    (("score",), _door(".observation_scorecard", "add_score_parser")),
    (("moving-ensemble", "ensemble"), _door(".ensemble", "add_ensemble_parser")),
    (("ensemble-prepare",), _door(".ensemble_prepare", "add_ensemble_prepare_parser")),
    (("pair",), _door(".pair_door", "add_pair_parser")),
    (("swath",), _door(".swath.door", "add_swath_parser")),
    (("cycle",), _door(".cycle.door", "add_cycle_parser")),
    (("adapt",), _door(".adapt", "add_adapt_parser")),
    (("physics",), _door(".physics_door", "add_physics_parser")),
    (("physics-resources",), _door(".mpasphys.resources_build", "add_physics_resources_parser")),
    (("fetch-assets",), _door(".assets", "add_fetch_assets_parser")),
    (("render",), _add_render_parser),
    (("products",), _door(".products_door", "add_products_parser")),
    (("export",), _door(".export_door", "add_export_parser")),
    (("add-nest",), _door(".add_nest_door", "add_add_nest_parser")),
    (("global",), _door(".global_door", "add_global_parser")),
    (("s2s",), _door(".s2s_door", "add_s2s_parser")),
    (("run-plan",), _door(".global_door", "add_run_plan_parser")),
    (("sources",), _door(".global_door", "add_sources_parser")),
    (("chem-sources",), _door(".chem_emissions", "add_parser")),
    (("global-cycle",), _door(".global_cycle", "add_global_cycle_parser")),
    (("regional",), _door(".regional_door", "add_regional_parser")),
)


#: What a door that imports the engine while its parser is built says when the engine is absent. THE BREAKAGE
#: THIS PREVENTS (hex 0.4.10 gate, HXCI packaging job, 2026-10-29): the doors that take --output-format import
#: woof.output_formats as their parsers are built, so ``woof hex --help`` in an install without the engine
#: (the public CI packaging job installs the wheel with --no-deps) died with ModuleNotFoundError instead of
#: listing the doors.
ENGINE_MISSING = ("needs the woof engine, which is not installed: pip install --force-reinstall 'recast-woof' "
                  "(the engine and woof hex ship together in recast-woof)")


def _engine_missing(error: BaseException) -> bool:
    name = getattr(error, "name", None) or ""
    return name == "woof" or name.startswith("woof.")


def _needs_engine(arguments: argparse.Namespace) -> int:
    print(f"woof hex: the {arguments.command} door {ENGINE_MISSING}", file=sys.stderr)
    return 2


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """The ``woof hex`` parser: every door, or only the door that registers ``command``.

    A door's parser is the same either way (same adder, same ``--numerics``),
    so ``build_parser(command).parse_args(argv)`` equals
    ``build_parser().parse_args(argv)`` for any ``argv`` naming ``command``.
    An unknown ``command`` builds every door, so argparse's refusal still
    lists every choice.
    """

    rows = _DOORS
    if command is not None:
        rows = tuple(row for row in _DOORS if command in row[0]) or _DOORS
    parser = argparse.ArgumentParser(
        prog="woof hex",
        description=(
            "woof hex: front doors for the GPU-native global variable-resolution "
            "model core (a port of MPAS-Atmosphere v8.4.2)"
        ),
    )
    parser.add_argument("--backend", choices=("cuda", "amd"),
                        help="execution backend (default: CUDA; AMD is experimental)")
    commands = parser.add_subparsers(dest="command", required=True)
    for names, add in rows:
        try:
            add(commands)
        except ModuleNotFoundError as error:
            # Listing every door (``--help``, an unknown word) keeps a door whose parser needs the engine,
            # as a stub that refuses by name; a door built alone refuses the same way in main().
            if rows is not _DOORS or not _engine_missing(error):
                raise
            for name in names:
                if name not in commands.choices:
                    stub = commands.add_parser(name, help=f"(not available here: this door {ENGINE_MISSING})")
                    stub.set_defaults(handler=_needs_engine)
    if rows is not _DOORS and command not in commands.choices:
        return build_parser()
    _add_numerics_option(commands)
    return parser


def _command_word(argv: Sequence[str]) -> str | None:
    """The door an argument vector names (its first word), or None for ``--help`` and the like."""

    return str(argv[0]) if argv and not str(argv[0]).startswith("-") else None


NUMERICS_HELP = (
    "kernel numerics: 'fast' (the default) compiles with FMA contraction and flush-to-zero (IEEE division and "
    "square root kept; land surface keeps its subnormals); 'strict' compiles every kernel the way the bitwise "
    "proofs measured it and is the opt-in for byte-identity to native MPAS-A. Same key as the namelist's "
    "config_numerics and the WOOF_HEX_NUMERICS environment variable")


def _add_numerics_option(commands: argparse._SubParsersAction) -> None:
    """``--numerics {fast,strict}`` on every door (owner ruling 2026-10-16: strict is opt-in)."""
    for sub in commands.choices.values():
        if any("--numerics" in action.option_strings for action in sub._actions):
            continue
        sub.add_argument("--numerics", choices=("fast", "strict"), default=None, help=NUMERICS_HELP)


def _select_numerics(arguments: argparse.Namespace) -> None:
    """Fix the process's numerics before any kernel compiles; ranks and subprocesses inherit it."""
    from . import numerics
    text = None
    path = getattr(arguments, "physics_namelist", None)
    if path is not None:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = None
    chosen, source = numerics.resolve(getattr(arguments, "numerics", None), text)
    numerics.set_mode(chosen, source=source)
    arguments.numerics = chosen
    arguments.numerics_source = source


def _fold_stochastic_switches(arguments: argparse.Namespace) -> None:
    """``--sppt/--no-sppt`` and ``--skeb/--no-skeb`` override the same keys of ``--stochastic`` JSON."""
    switches = {name: getattr(arguments, name) for name in ("sppt", "skeb")
                if getattr(arguments, name, None) is not None}
    if not switches or not hasattr(arguments, "stochastic"):
        return
    import json
    current = json.loads(arguments.stochastic) if arguments.stochastic else {}
    if not isinstance(current, dict):
        raise ValueError("--stochastic must be a JSON object")
    current.update(switches)
    arguments.stochastic = json.dumps(current, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        from woof.backend import consume_backend_option
        argv = consume_backend_option(argv, local_backend_commands=("verify-dycore",))
    except ValueError as error:
        print(f"woof hex: {error}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as error:
        if not _engine_missing(error):
            raise
    if _command_word(argv) != "forecast":
        return _main_once(argv)
    # A storm-following segment in a resident moving process (woof.hex.mpasphys.moving_resident_v842): a stop that
    # handed its carrier over in memory continues here with the next segment's own argv, in this process.
    try:
        from .mpasphys import moving_resident_v842 as resident
    except ModuleNotFoundError as error:
        if not _engine_missing(error):
            raise
        return _main_once(argv)     # the forecast door refuses by name without the engine
    resident.begin_segment()
    rc = _main_once(argv)
    while rc == 0:
        following = resident.next_segment()
        if following is None:
            break
        try:
            rc = _main_once(following)
        except BaseException:
            resident.seal_handed("the continued segment raised")
            raise
        if rc != 0:
            resident.seal_handed(f"the continued segment returned {rc}")
    return rc


def _main_once(argv: Sequence[str]) -> int:
    try:
        parser = build_parser(_command_word(argv))
    except ModuleNotFoundError as error:
        if not _engine_missing(error):
            raise
        print(f"woof hex: the {_command_word(argv)} door {ENGINE_MISSING} ({error})", file=sys.stderr)
        return 2
    arguments = parser.parse_args(argv)
    try:
        if hasattr(arguments, "output_format"):
            from woof.output_formats import options, require_binary
            for row in options(arguments)[0]:
                if row["writer"]:
                    require_binary(row)
        _fold_stochastic_switches(arguments)
        _select_numerics(arguments)
        if arguments.command in {"init", "lbc"}:
            from .time_ledger import stage

            with stage(f"prep.{arguments.command}"):
                return int(arguments.handler(arguments))
        return int(arguments.handler(arguments))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"woof hex: {error}", file=sys.stderr)
        return 2


#: Door commands that hold the device.  Composing doors (global, regional, country) run these in a child
#: process so the CUDA context and every pooled byte end with the stage.  THE BREAKAGE THIS PREVENTS
#: (rolling 2.9.4 run, hex global-60km 24 h, 2026-10-21): the in-process forecast kept 11.6 GB of the card
#: through products, export and observation fetches for scoring, more than 10 min at 0 % GPU, so no other
#: job could admit on the card.  The stages after it are host work (Rust renderer, NetCDF export, scorer).
DEVICE_COMMANDS = frozenset({"forecast"})


def run_stage(argv: Sequence[str], env: dict[str, str] | None = None) -> int:
    """Run one door command for a composing door: device commands in a child process, the rest in-process."""

    argv = [str(item) for item in argv]
    if argv and argv[0] in DEVICE_COMMANDS:
        import os
        import subprocess
        child = {**os.environ, **(env or {})}
        return subprocess.run([sys.executable, "-m", "woof.hex.cli", *argv], env=child).returncode
    return int(main(argv))


def run_exit_handlers() -> None:
    """What the interpreter runs at a normal exit, in its order, without the module teardown after it.

    threading's exit hooks first (``threading._shutdown``: concurrent.futures stops its process pools and
    the kernel plans' compile pools drop their workers), then the ``atexit`` functions. THE BREAKAGE THIS
    PREVENTS (the hl-3 headline run, 2026-10-24): running ``atexit`` alone skipped the threading hooks, so
    multiprocessing's exit function joined an idle spawn compile worker that nothing had told to stop, and a
    finished forecast (receipt and products written) never exited.
    """
    import atexit
    import threading
    try:
        shutdown = getattr(threading, "_shutdown", None)
        if shutdown is not None:
            shutdown()
    finally:
        atexit._run_exitfuncs()


def device_process_exit(code: int) -> None:
    """End a device-command process without the interpreter's module teardown.

    THE BREAKAGE THIS PREVENTS (HEX-PHYSICS-PARITY run group 2 #6): a moving segment on the MPAS suite route
    finished, wrote its receipt ("completed": true) and its carrier, then died with SIGSEGV (rc -11) while the
    interpreter tore modules down: cupy's ``Event.__del__`` ran after the CUDA context was gone
    ("cudaErrorDeviceUninitialized: invalid device context").  The controller counted the finished segment as
    failed and launched a recovery.  Here the exit handlers run exactly as at a normal exit (kernel-cache and
    prewarm manifests, history writers), the streams are flushed, and the process ends with the door's code
    before any late finalizer can touch the device (the forked member slot in woof.hex.forecast_batch does the
    same).
    """
    import atexit
    import os
    try:
        run_exit_handlers()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(code))


def entry() -> int:
    """The console door: :func:`main`, with a device command's process ended by :func:`device_process_exit`."""
    argv = sys.argv[1:]
    code = main(argv)
    if argv and argv[0] in DEVICE_COMMANDS:
        device_process_exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(entry())
