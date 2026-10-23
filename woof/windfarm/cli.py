"""``woof windfarm``: turbine layouts on a domain's grid."""

from __future__ import annotations

import json
from pathlib import Path


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "windfarm", help="wind-farm turbine layouts on a domain's grid",
        description="Read a turbine table and place it on one domain of an "
                    "experiment: the cells it occupies, the turbines in "
                    "each and the rated capacity: the placement a forecast "
                    "with windfarm_opt = 1 and this table as "
                    "windfarm_turbine_file runs.")
    commands = parser.add_subparsers(dest="windfarm_command", required=True)
    layout = commands.add_parser(
        "layout", help="place a turbine table on a domain and print the "
                       "cells, turbines per cell and rated capacity",
        description="Place every turbine of TURBINES (CSV, GeoJSON points "
                    "or a woof-energy sites.v1 document; columns lat, lon, "
                    "hub_height_m, rotor_diameter_m and curve) in its mass "
                    "cell of one CONFIG domain, resolve each curve through "
                    "the turbine library, and print one row per turbine "
                    "group (a run of consecutive, identical turbines of one "
                    "cell in table order, the order WRF reads them; a cell "
                    "listing A, B, A has three), with a total row under a "
                    "cell of several groups.  A turbine outside the domain, in its "
                    "outermost row or column, with no curve, with a curve "
                    "the library does not hold, or with its rotor bottom "
                    "below ground is refused by name.")
    layout.add_argument("config", type=Path, metavar="CONFIG",
                        help="experiment TOML whose domain the turbines are "
                             "placed on")
    layout.add_argument("turbines", type=Path, metavar="TURBINES",
                        help="turbine table: .csv, GeoJSON FeatureCollection "
                             "of points, or a sites.v1 JSON")
    layout.add_argument("--grid-id", type=int, default=1, metavar="N",
                        help="domain to place the turbines on (default: 1)")
    layout.add_argument("--curve", default=None, metavar="ID",
                        help="turbine-library curve for every turbine whose "
                             "row names none (required for a sites.v1 "
                             "document)")
    layout.add_argument("--library", type=Path, default=None, metavar="FILE",
                        help="turbine library JSON (default: the bundled "
                             "woof/data/energy/turbine_library.json)")
    layout.add_argument("--json", action="store_true",
                        help="print the summary as one JSON document "
                             "instead of a table")
    layout.set_defaults(func=layout_main)
    from woof.explain import add_explain_flag
    add_explain_flag(layout, nested=True)


def layout_table(layout) -> str:
    """The `woof windfarm layout` table: one row per group, a cell's
    groups under it, and a total row for a cell of several groups."""

    width = max([len("curve")] + [len(c) for c in layout.curve_ids])
    lines = [f"d{layout.grid_id:02d}: {layout.nx} x {layout.ny} cells of "
             f"{layout.dx_m:g} m",
             f"{'j':>5}  {'i':>5}  {'turbines':>8}  {'curve':<{width}}  "
             f"{'hub m':>6}  {'rotor m':>7}  {'MW':>8}"]
    for cell in layout.cells():
        for n, group in enumerate(cell["groups"]):
            where = (f"{cell['j']:>5}  {cell['i']:>5}" if n == 0
                     else " " * 12)
            lines.append(
                f"{where}  {group['turbines']:>8}  "
                f"{group['curve']:<{width}}  {group['hub_height_m']:>6.1f}  "
                f"{group['rotor_diameter_m']:>7.1f}  "
                f"{group['capacity_mw']:>8.2f}")
        if len(cell["groups"]) > 1:
            lines.append(
                f"{'':>12}  {cell['turbines']:>8}  {'(cell)':<{width}}  "
                f"{'':>6}  {'':>7}  {cell['capacity_mw']:>8.2f}  rotors "
                f"{cell['rotor_bottom_m']:.0f}-{cell['rotor_top_m']:.0f} m")
    lines.append("")
    lines.append(
        f"{layout.n_turbines} turbine(s) in {layout.n_groups} group(s) in "
        f"{layout.cells_occupied} cell(s), at most "
        f"{int(layout.turbine_count.max())} turbine(s) and "
        f"{int(layout.group_count.max())} group(s) per cell; rated "
        f"capacity {layout.rated_capacity_mw:.2f} MW; curves: "
        + ", ".join(layout.curve_ids))
    return "\n".join(lines)


def layout_main(args) -> int:
    from woof.windfarm.layout import layout_for_config

    layout = layout_for_config(args.config, args.turbines,
                               grid_id=args.grid_id, default_curve=args.curve,
                               library_path=args.library)
    if args.json:
        print(json.dumps(layout.summary(), indent=1))
    else:
        print(layout_table(layout))
    return 0
