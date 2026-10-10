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
from .init_door import add_init_parser
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
                "woof hex ships neither and has no fetch path for them, so a "
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
            "model core (a port of MPAS-Atmosphere v8.4.1)"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    version = commands.add_parser(
        "version", help="report the installed distribution and version"
    )
    version.set_defaults(handler=_version)

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

    add_cull_parser(commands)

    # Between cull and init, because that is where a regional case meets it:
    # a cull needs a regional meteorological source the init and boundary
    # engines can read, and a projected product (HRRR) is not one until it
    # is resampled onto a regular lat-lon intermediate.  Deferred import:
    # the door pulls numpy and the engine's ingest.
    from .hrrr_intermediate import add_intermediate_parser, add_lbc_parser

    add_intermediate_parser(commands)

    add_init_parser(commands)

    # After init, because it consumes the culled init: one boundary file per
    # intermediate valid time, through rw_mpas_lbc on its wps-intermediate
    # route.
    add_lbc_parser(commands)

    # Between init and render, because that is the order a user meets them:
    # an init is what a forecast starts from and history is what a render
    # consumes.  The door needs a source checkout and says so by name; it is
    # listed here unconditionally so that `woof hex --help` on a wheel
    # names the command and its one requirement, rather than hiding a
    # capability the distribution has.
    add_forecast_parser(commands)

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

    render = commands.add_parser(
        "render",
        help="MPAS history -> product PNGs through the Rust path "
             "(rw_mpas_convert + rw_wrfbatch)",
    )
    from .render_door import add_render_arguments

    add_render_arguments(render)
    render.set_defaults(handler=_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"woof hex: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
