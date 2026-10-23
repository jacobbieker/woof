"""``woof fuel preview``: a domain's fire fuel map and fire-mesh slope.

Builds one domain's statics with a fuel request
(:mod:`woof.static.fuel`) and prints the Anderson-13 category histogram on
the atmospheric grid and on the fire mesh, with the fire-mesh terrain slope
statistics.  Read-only: it writes nothing unless ``--output`` asks.  The
configuration is not changed, and no configuration needs a fire setting
for the preview to run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


#: The refinement a preview builds at when neither the command line nor the
#: domain's own ``[[domain]] static`` fuel request names one.
DEFAULT_PREVIEW_RATIO = 4


def preview_fuel(config: Path, *, domain: int | None,
                 sr_x: int | None = None, sr_y: int | None = None,
                 landfire: Path | None = None,
                 landfire_kind: str | None = None,
                 device: str = "auto") -> tuple[dict, dict]:
    """``(summary, fields)`` for one domain of ``config``.

    The domain's own fuel request (``[[domain]] static fuel``), when it
    declares one, is what is built; a value given here replaces that
    value.  A domain without one is previewed at
    :data:`DEFAULT_PREVIEW_RATIO` on the land-use crosswalk unless told
    otherwise.
    """
    from woof.case_data import load_experiment_case
    from woof.static.build import GeogSelection, build_static
    from dataclasses import replace

    from woof.static.fuel import (FUEL_FIELDS, FuelStaticsRequest,
                                  build_fuel_fields, fuel_for, fuel_summary,
                                  with_fuel_statics)
    from woof.static.highres_production import overlay_active
    from woof.static.orographic import with_terrain_drag_statics
    from woof.static.projection import grids_from_projection_config

    exp, data = load_experiment_case(config, require_met_inputs=False)
    ids = [dc.grid_id for dc in exp.domains]
    grid_id = ids[0] if domain is None else int(domain)
    if grid_id not in ids:
        raise ValueError(
            f"--domain {grid_id} is not a domain of {config}; its domains "
            f"are {ids}")
    index = ids.index(grid_id)
    dc = exp.domains[index]
    grid = grids_from_projection_config(exp)[index]
    declared = fuel_for(getattr(data, "static_highres", None), grid_id)
    default = declared or FuelStaticsRequest(sr_x=DEFAULT_PREVIEW_RATIO,
                                             sr_y=DEFAULT_PREVIEW_RATIO)
    request = FuelStaticsRequest(
        sr_x=default.sr_x if sr_x is None else sr_x,
        sr_y=default.sr_y if sr_y is None else sr_y,
        landfire=default.landfire if landfire is None else landfire,
        # A raster named here is read as its own codes say unless its
        # kind is named too, never under the declared raster's kind.
        landfire_kind=(landfire_kind if landfire_kind is not None
                       else "auto" if landfire is not None
                       else default.landfire_kind),
        device=device)
    selection = GeogSelection.from_case_data(data, domain_id=grid_id)
    selection = with_fuel_statics(
        with_terrain_drag_statics(selection, dc.run), request)
    highres = getattr(data, "static_highres", None)
    overlay = overlay_active(highres, grid)
    report: dict = {}
    # Under an overlay the fuel is built once, from the overlaid fields.
    fields = build_static(
        grid, data.geog_root,
        selection=replace(selection, fuel=None) if overlay else selection,
        source_coverage_report=report)
    if overlay:
        from woof.static.highres_production import apply_highres_statics
        landuse_attrs = selection.landuse_global_attrs()
        # The seam's own fuel build is skipped (fuel=()): the preview builds
        # its request once, from what the overlay left, with its receipt.
        fields, overlay_receipt = apply_highres_statics(
            fields, grid, config=replace(highres, fuel=()),
            domain_id=grid_id, case_date=dc.start_time.date(),
            landuse_attrs=landuse_attrs)
        report = {}
        fields = {**fields, **build_fuel_fields(
            grid, fields, request, landuse_attrs=landuse_attrs,
            coverage_report=report)}
        report["fuel"]["after_highres_overlay"] = (
            isinstance(overlay_receipt, dict)
            and overlay_receipt.get("status") == "APPLIED")
    summary = {"config": str(config), "domain": grid_id,
               "nx": int(grid.e_we) - 1, "ny": int(grid.e_sn) - 1,
               "dx_m": float(grid.dx), "dy_m": float(grid.dy),
               "receipt": report["fuel"], **fuel_summary(fields)}
    return summary, {name: fields[name] for name in FUEL_FIELDS}


def _print_summary(summary: dict) -> None:
    receipt = summary["receipt"]
    print(f"fuel preview: domain {summary['domain']} of {summary['config']}, "
          f"{summary['nx']} x {summary['ny']} cells at "
          f"{summary['dx_m']:g} m; fire mesh "
          f"{receipt['sr_x']} x {receipt['sr_y']} per cell = "
          f"{summary['fire_mesh_shape'][1]} x "
          f"{summary['fire_mesh_shape'][0]}, built on the "
          f"{receipt['device'].upper()}")
    source = receipt["source"]
    if source == "landfire":
        landfire = receipt["landfire"]
        print(f"source: LANDFIRE {landfire['kind'].upper()} "
              f"{landfire['path']} ({landfire['sampled_cells']} fire cells "
              f"sampled, {landfire['landuse_filled_cells']} filled from "
              f"the {receipt['mminlu']} land-use crosswalk)")
    else:
        print(f"source: LU_INDEX through the {receipt['mminlu']} land-use "
              f"crosswalk ({receipt['crosswalk']}, sha256 "
              f"{receipt['crosswalk_sha256'][:12]})")
    for label, key in (("atmospheric grid", "atmospheric"),
                       ("fire mesh", "fire_mesh")):
        print(f"NFUEL_CAT on the {label}:")
        for row in summary[key]:
            print(f"  {row['category']:>2}  {row['name']:<32} "
                  f"{row['cells']:>9}  {row['percent']:6.2f}%")
    slope = summary["slope"]
    terrain = summary["terrain_m"]
    print(f"fire-mesh terrain ZSF: {terrain['min']:.1f} .. "
          f"{terrain['max']:.1f} m")
    print(f"slope dz/dx: {slope['dzdx_min']:+.4f} .. {slope['dzdx_max']:+.4f}"
          f"; dz/dy: {slope['dzdy_min']:+.4f} .. {slope['dzdy_max']:+.4f}")
    print(f"slope magnitude: mean {slope['magnitude_mean']:.4f}, p95 "
          f"{slope['magnitude_p95']:.4f}, max {slope['magnitude_max']:.4f} "
          f"({slope['max_degrees']:.1f} degrees)")


def fuel_main(args) -> int:
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr if args.json else sys.stdout):
        summary, fields = preview_fuel(
            args.config, domain=args.domain, sr_x=args.sr_x, sr_y=args.sr_y,
            landfire=args.landfire, landfire_kind=args.landfire_kind,
            device=args.device)
    if args.output is not None:
        import numpy as np
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Through a handle: np.savez given a path appends ".npz" to one
        # that lacks it, and the path printed below would not exist.
        with open(args.output, "wb") as stream:
            np.savez(stream, **fields)
        summary["output"] = str(args.output)
    if args.json:
        print(json.dumps(summary, indent=2, allow_nan=False))
    else:
        _print_summary(summary)
        if args.output is not None:
            print(f"fuel fields: {args.output}")
    return 0


def register_cli(subparsers) -> None:
    from woof.static.fuel import DEVICES, LANDFIRE_KINDS
    parser = subparsers.add_parser(
        "fuel", help="build and inspect a domain's fire fuel statics",
        description="Fire-model fuel statics: the Anderson-13 NFUEL_CAT map "
                    "(14 = no fuel) on the atmospheric grid and the refined "
                    "fire mesh, and the fire-mesh terrain slope.")
    commands = parser.add_subparsers(dest="fuel_command", required=True)
    preview = commands.add_parser(
        "preview", help="print a domain's fuel-category histogram and "
                        "fire-mesh slope statistics",
        description="Build one domain's statics with a fuel request and "
                    "print the NFUEL_CAT histogram on the atmospheric grid "
                    "and the fire mesh, with the slope of the fire-mesh "
                    "terrain.  The fuel comes from a LANDFIRE FBFM13/FBFM40 "
                    "GeoTIFF when --landfire names one, otherwise from "
                    "LU_INDEX through the land-use-to-fuel crosswalk.")
    preview.add_argument("config", type=Path, metavar="CONFIG",
                         help="experiment TOML with a [case_data] table")
    preview.add_argument("--domain", type=int, default=None, metavar="ID",
                         help="grid_id of the domain (default: the first)")
    preview.add_argument("--sr-x", type=int, default=None, metavar="N",
                         help="fire cells per atmospheric cell west-east "
                              "(WRF-SFIRE sr_x; default: the domain's own "
                              "[[domain]] static fuel request, else 4)")
    preview.add_argument("--sr-y", type=int, default=None, metavar="N",
                         help="fire cells per atmospheric cell south-north "
                              "(WRF-SFIRE sr_y; default: the domain's own "
                              "request, else 4)")
    preview.add_argument("--landfire", type=Path, default=None,
                         metavar="TIF",
                         help="local LANDFIRE FBFM13 or FBFM40 GeoTIFF")
    preview.add_argument("--landfire-kind", choices=LANDFIRE_KINDS,
                         default=None,
                         help="code set of --landfire (default: the domain's "
                              "own request, else auto, read from the codes)")
    preview.add_argument("--device", choices=DEVICES, default="auto",
                         help="where the fire mesh is built (default: auto, "
                              "the GPU when one is usable; both give the "
                              "same words)")
    preview.add_argument("--output", type=Path, default=None, metavar="NPZ",
                         help="also write NFUEL_CAT, NFUEL_CAT_FIRE, ZSF, "
                              "DZDXF and DZDYF to this NPZ")
    preview.add_argument("--json", action="store_true",
                         help="print the summary as JSON")
    preview.set_defaults(func=fuel_main)
    from woof.explain import add_explain_flag
    add_explain_flag(preview, nested=True)
