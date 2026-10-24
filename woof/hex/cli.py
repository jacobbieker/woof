"""The ``woof hex`` console script: every front door this package ships."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .errors import MpasPortError
# ``.mesh`` and ``.oracle`` are imported inside the handlers that use them:
# they pull numpy/netCDF4, and the ``render`` door must run on a render node
# whose Python has neither -- its data path is entirely the two Rust
# binaries it drives.  ``.init_door`` and ``.forecast_door`` are stdlib-only
# at import; their netCDF4/cupy/driver needs are named runtime refusals.
from .cull_door import add_cull_parser
from .init_door import add_init_parser, add_surface_update_parser
# The checkout probe lives with the forecast door because that door is the
# one that cannot open without a checkout at all.  Stated once: two copies
# of "is this a checkout" would be two things to keep true, and the answer
# decides both this module's mesh defaults and that door's existence.
from .forecast_door import PROJECT_ROOT, add_forecast_parser
DEFAULT_GRID = (
    PROJECT_ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
    if PROJECT_ROOT is not None
    else None
)
DEFAULT_STATIC = (
    PROJECT_ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"
    if PROJECT_ROOT is not None
    else None
)
DEFAULT_ORACLE = (
    PROJECT_ROOT / "oracle" / "x1.2562" if PROJECT_ROOT is not None else None
)


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
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woof hex",
        description=(
            "woof hex: front doors for the GPU-native global variable-resolution "
            "model core (a port of MPAS-Atmosphere v8.4.2)"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    from .country_door import add_country_parser
    add_country_parser(commands)

    version = commands.add_parser(
        "version", help="report the installed distribution and version"
    )
    version.set_defaults(handler=_version)

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

    # First after `version`, because it is the command a fresh install
    # should reach for.  This distribution cannot carry the Rust engines,
    # the CUDA runtime or the mesh assets, so a bare `pip install` lands
    # something real that is not yet able to run: doctor is where that
    # gap gets NAMED, with the command that closes each half, instead of
    # a user meeting a refusal at the end of a long argument vector.
    from .doctor import add_doctor_parser

    add_doctor_parser(commands)

    # Before `mesh-check`, because that is the order a mesh happens in: a
    # spec is asked whether it can be built, and only then is a built mesh
    # validated.  The gap this closes is measured -- `rw_mpas_mesh --dry-run`
    # sizes a spec its own build then refuses, once on pre-run arithmetic and
    # once after 711 s of relaxation.
    from .mesh_plan_door import add_mesh_plan_parser

    add_mesh_plan_parser(commands)

    # Beside mesh-plan, because a density raster is what a raster-region
    # mesh spec is planned from.  The handler imports numpy/scipy; the
    # parser stays stdlib-only so the render door keeps working without them.
    _add_density_parser(commands)

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

    oracle_gate = commands.add_parser(
        "oracle-gate", help="replay the source-extracted Fortran M1 fixtures"
    )
    _add_mesh_paths(oracle_gate)
    oracle_gate.add_argument("--fixtures", type=Path, default=DEFAULT_ORACLE)
    oracle_gate.set_defaults(handler=_oracle_gate)

    # Before cull, vertical and init, because that is where it must run: it
    # rewrites the static's terrain, which the vertical build smooths and
    # the mesh-row registry pins by SHA-256.  The parser lives here and the
    # module is imported only by the handler: it pulls numpy, scipy and
    # netCDF4, and parsing must not.
    _add_static_highres_parser(commands)
    # Before cull, because the vertical artifact is minted on the GLOBAL
    # parent and then cut with the grid and static (`cull --parent-vertical`).
    # Deferred import: the door pulls netCDF4 and numpy at run time only.
    from .vertical_door import add_vertical_parser

    add_vertical_parser(commands)

    add_cull_parser(commands)

    # After cull, because a cull row is registered from the cull's receipt;
    # a global row is registered as soon as its pair exists.
    from .register_door import add_register_parser

    add_register_parser(commands)

    # Between cull and init, because that is where a regional case meets it:
    # a cull needs a regional meteorological source the init and boundary
    # engines can read, and a projected product (HRRR) is not one until it
    # is resampled onto a regular lat-lon intermediate.  The global lat-lon
    # sources (gfs, gdas, ecmwf-open-data, era5, aifs) share the command and
    # dispatch to woof.hex.met_intermediate_door, which drives the Rust
    # met_intermediate writer without regridding.  ICON global (icon-global,
    # icon-global-ml) dispatches there too, through a gdt101_remap stage
    # (woof.hex.gdt101_door) that writes its icosahedral mesh onto a window
    # over --grid first; met-office-global dispatches to
    # woof.hex.metoffice_intermediate, which reads the Met Office NetCDF
    # itself and takes the missing land state from the same-cycle GDAS.
    # Deferred import: the door pulls numpy and the
    # engine's ingest.
    from .hrrr_intermediate import add_intermediate_parser, add_lbc_parser

    add_intermediate_parser(commands)

    add_init_parser(commands)
    add_surface_update_parser(commands)

    # Beside init, because it ends in init: one table-driven path from any
    # declared source (GFS, IFS, AIFS, ENS/GEFS members, ERA5, ...) to the
    # same init door.  A new source model is a table row, not a code path.
    from .init_sources import add_init_source_parser

    add_init_source_parser(commands)
    from .surface_series import add_surface_series_parser
    add_surface_series_parser(commands)

    # After init, because it consumes the culled init: one boundary file per
    # intermediate valid time, through rw_mpas_lbc on its wps-intermediate
    # route.
    add_lbc_parser(commands)

    # After lbc, because it is the cycle-to-cycle step: a state the last
    # cycle ended on, carried onto the mesh the next cycle regenerated.
    from .remap_door import add_remap_parser

    add_remap_parser(commands)

    # Between init and render, because that is the order a user meets them:
    # an init is what a forecast starts from and history is what a render
    # consumes.  The door needs a source checkout and says so by name; it is
    # listed here unconditionally so that `woof hex --help` on a wheel
    # names the command and its one requirement, rather than hiding a
    # capability the distribution has.
    add_forecast_parser(commands)

    # Right after forecast, because a moving run is a chain of forecasts: the
    # tracker on the model's own frames, member preparation ahead of each move
    # and the balanced remap between members.  Stdlib-only at import.
    from .moving_door import add_moving_parser

    add_moving_parser(commands)

    from .observation_scorecard import add_score_parser

    add_score_parser(commands)

    from .ensemble import add_ensemble_parser
    from .ensemble_prepare import add_ensemble_prepare_parser
    add_ensemble_parser(commands)
    add_ensemble_prepare_parser(commands)

    # Immediately after forecast, because it IS the forecast door: `pair`
    # takes every forecast flag from `add_forecast_arguments`, runs that one
    # authored request twice -- once with a point-source table and once
    # without -- and differences the two.  The import is deferred for the
    # same reason `forecast` is not: this module is stdlib-only, but keeping
    # the two together here is what makes the adjacency visible in --help.
    from .pair_door import add_pair_parser

    add_pair_parser(commands)

    # After forecast and before render, because that is the order a cascade
    # meets them: a coarse forecast is what the placement reads, and the
    # swaths it places are what the next forecast and the render draw.  The
    # import is deferred: this door pulls numpy, netCDF4 and scipy, and the
    # render door must stay usable on a node whose Python has none of them.
    from .swath.door import add_swath_parser

    add_swath_parser(commands)

    from .cycle.door import add_cycle_parser

    add_cycle_parser(commands)

    # Beside cycle, because it is the step between two cycles when the mesh
    # follows the forecast: criteria -> density raster -> hysteresis -> the
    # next cycle's commands.  Stdlib-only at import; numpy, scipy and
    # netCDF4 are pulled inside the handler.
    from .adapt import add_adapt_parser

    add_adapt_parser(commands)

    # The named physics suites, resolved on the host: stdlib + json only.
    from .physics_door import add_physics_parser

    add_physics_parser(commands)

    # Right after `physics`: the MPAS suites it lists read tables and device
    # maths that this command builds from the install (stdlib at import).
    from .mpasphys.resources_build import add_physics_resources_parser

    add_physics_resources_parser(commands)
    from .assets import add_fetch_assets_parser
    add_fetch_assets_parser(commands)

    render = commands.add_parser(
        "render",
        help="MPAS history -> product PNGs through the Rust path "
             "(rw_mpas_convert + rw_wrfbatch)",
    )
    from .render_door import add_render_arguments

    add_render_arguments(render)
    render.set_defaults(handler=_render)

    # After render, because it drives render: the standard product set (maps
    # on every view a mesh has, soundings, meteograms) in one command, and
    # the forecast door's own last stage by default.
    from .products_door import add_products_parser

    add_products_parser(commands)

    # After products, because it is products without the drawing: the same
    # Rust converter, frames filed as NetCDF for any reader.
    from .export_door import add_export_parser

    add_export_parser(commands)

    # Last, because it composes every door above from a data row: mesh,
    # static, init-source, forecast, products, export and score in one
    # command, with its plan, status.json and events.jsonl.
    from .global_door import add_global_parser, add_run_plan_parser, add_sources_parser

    add_global_parser(commands)
    from .s2s_door import add_s2s_parser

    add_s2s_parser(commands)
    add_run_plan_parser(commands)
    add_sources_parser(commands)
    from .chem_emissions import add_parser as add_chem_sources_parser

    add_chem_sources_parser(commands)
    from .global_cycle import add_global_cycle_parser

    add_global_cycle_parser(commands)
    # The limited-area counterpart of `global`: mesh-plan --point --generate,
    # intermediate, init, lbc, the contract deck and forecast --lbc-dir from
    # one limited-area row and a point.
    from .regional_door import add_regional_parser

    add_regional_parser(commands)
    _add_numerics_option(commands)
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
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


if __name__ == "__main__":
    raise SystemExit(main())
