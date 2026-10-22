#!/usr/bin/env python3
"""Write the oracle's decks: synthetic farms laid out by woof, as WRF reads them.

    python -m tools.windfarm_layout_wrf471_oracle.make_deck DECK_ROOT

One directory per case under DECK_ROOT, each holding what a WRF run of the
farm reads -- ``windturbines.txt`` and ``wind-turbine-N.tbl``, written by
:meth:`woof.windfarm.layout.WindFarmLayout.write_wrf_inputs` -- plus the
harness's ``grid.txt`` (WRF's config_flags and the float32 XLAT/XLONG of
the domain's first mass point) and ``queries.txt`` (dragcof calls), and
``deck.json``, the woof inputs the parity test rebuilds the layout from.

Every farm is synthetic: seeded random turbines over each domain's
interior, a block on exact half-cell projection coordinates (where
``nint`` rounds), and a cell holding several machines.  The domains cover
the three projections WRF's wind-farm code places on -- Lambert (secant in
both hemispheres, tangent), Mercator (tropics, across the dateline) and
polar stereographic (2-4 degrees from the north pole, and in the south
across the dateline) -- each on both sides of its standard longitude, and
nests: a three-level 12 -> 3 -> 1 km Lambert ladder and one-level Mercator
and polar nests, every nest placed from its own corner as WRF places it.
Between them the decks use every curve of the bundled turbine library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from woof.energy.turbine_library import load_turbine_library
from woof.static.projection import projection_class
from woof.windfarm import layout as wl
from woof.windfarm.fitch_tables import (FitchTableRefused, float32_word,
                                        table_rows)

F = np.float32

_LC_LADDER = {"map_proj": "lambert", "ref_lat": 35.2, "ref_lon": -97.9,
              "truelat1": 30.0, "truelat2": 60.0, "stand_lon": -97.9,
              "dx": 12000.0, "e_we": 61, "e_sn": 49}
_MERC = {"map_proj": "mercator", "ref_lat": 5.0, "ref_lon": 100.0,
         "truelat1": 5.0, "truelat2": 5.0, "stand_lon": 100.0,
         "dx": 9000.0, "e_we": 81, "e_sn": 61}
_PS_NH = {"map_proj": "polar", "ref_lat": 86.5, "ref_lon": -40.0,
          "truelat1": 60.0, "truelat2": 60.0, "stand_lon": -40.0,
          "dx": 6000.0, "e_we": 81, "e_sn": 81}

#: case id -> (root grid, nest chain [(i_parent_start, j_parent_start,
#: parent_grid_ratio, e_we, e_sn), ...], WRF's windfarm_tke_factor)
CASES = {
    "lc_nh_12km": ({"map_proj": "lambert", "ref_lat": 35.5, "ref_lon": -97.5,
                    "truelat1": 25.5, "truelat2": 45.5, "stand_lon": -97.5,
                    "dx": 12000.0, "e_we": 61, "e_sn": 49}, [], 1.0),
    "lc_ladder_d02_3km": (_LC_LADDER, [(21, 17, 4, 97, 73)], 0.25),
    "lc_ladder_d03_1km": (_LC_LADDER, [(21, 17, 4, 97, 73),
                                       (33, 25, 3, 97, 73)], 1.0),
    "lc_sh_3km": ({"map_proj": "lambert", "ref_lat": -30.0, "ref_lon": 140.0,
                   "truelat1": -20.0, "truelat2": -40.0, "stand_lon": 140.0,
                   "dx": 3000.0, "e_we": 81, "e_sn": 81}, [], 1.0),
    "lc_tangent_4km": ({"map_proj": "lambert", "ref_lat": 45.0,
                        "ref_lon": 10.0, "truelat1": 45.0, "truelat2": 45.0,
                        "stand_lon": 10.0, "dx": 4000.0, "e_we": 61,
                        "e_sn": 61}, [], 0.5),
    "merc_tropics_9km": (_MERC, [], 1.0),
    "merc_nest_d02_3km": (_MERC, [(27, 21, 3, 76, 61)], 0.5),
    "merc_dateline_6km": ({"map_proj": "mercator", "ref_lat": -17.8,
                           "ref_lon": 179.5, "truelat1": -17.8,
                           "truelat2": -17.8, "stand_lon": 179.5,
                           "dx": 6000.0, "e_we": 81, "e_sn": 61}, [], 1.0),
    "ps_nh_6km": (_PS_NH, [], 1.0),
    "ps_nh_nest_d02_2km": (_PS_NH, [(27, 27, 3, 79, 79)], 0.25),
    "ps_sh_dateline_6km": ({"map_proj": "polar", "ref_lat": -77.85,
                            "ref_lon": 179.0, "truelat1": -71.0,
                            "truelat2": -71.0, "stand_lon": 166.7,
                            "dx": 6000.0, "e_we": 81, "e_sn": 81}, [], 1.0),
}
RANDOM_TURBINES = 300
HALF_CELL_TURBINES = 24
SEED = 4711


def grid_from_spec(root: dict, nests=()):
    """The woof grid of one case: the root, then each nest in the chain."""

    cls = projection_class(root["map_proj"])
    grid = cls(ref_lat=root["ref_lat"], ref_lon=root["ref_lon"],
               truelat1=root["truelat1"], truelat2=root["truelat2"],
               stand_lon=root["stand_lon"], dx=root["dx"], dy=root["dx"],
               e_we=root["e_we"], e_sn=root["e_sn"])
    for i_start, j_start, ratio, e_we, e_sn in nests:
        grid = grid.nest(i_start, j_start, ratio, e_we, e_sn)
    return grid


def _hub(curve) -> float:
    floor = curve.rotor_diameter_m / 2.0 + 20.0
    hubs = [h for h in curve.hub_heights_m if h >= floor]
    return float(hubs[0]) if hubs else float(np.ceil(floor))


def _usable_curves(library) -> list:
    out = []
    for curve_id, curve in sorted(library.curves.items()):
        try:
            table_rows(curve)
        except FitchTableRefused:
            continue
        out.append(curve)
    return out


def _turbines(grid, curves, rng, case: str) -> list[dict]:
    nx, ny = grid.e_we - 1, grid.e_sn - 1
    rows = []

    def add(x, y, curve, hub=None, rotor=None):
        lat, lon = (float(v) for v in grid.ij_to_latlon(x, y))
        rows.append({"turbine_id": f"{case}-{len(rows):04d}", "lat": lat,
                     "lon": lon, "hub_height_m": hub or _hub(curve),
                     "rotor_diameter_m": rotor, "curve": curve.curve_id})

    # every curve once, then random positions cycling through them
    for n in range(RANDOM_TURBINES):
        x = rng.uniform(2.6, nx - 1.6)
        y = rng.uniform(2.6, ny - 1.6)
        add(x, y, curves[n % len(curves)])
    # exact half-cell projection coordinates: nint's rounding edge
    for n in range(HALF_CELL_TURBINES):
        x = float(rng.integers(3, nx - 3)) + 0.5
        y = float(rng.integers(3, ny - 3)) + (0.5 if n % 2 else 0.0)
        add(x, y, curves[n % len(curves)])
    # one cell holding two machines, one of them at hubs 0.2 m apart (each
    # its own WRF type, and its own group under the default exact grouping)
    cx, cy = nx // 2 + 0.1, ny // 2 + 0.1
    a, b = curves[0], curves[1 % len(curves)]
    for n in range(4):
        add(cx + 0.05 * n, cy, a, hub=_hub(a) + 0.2 * n)
    for n in range(3):
        add(cx, cy + 0.05 * n, b, hub=_hub(b) + 25.0)
    return rows


def _queries(layout, rng) -> list[tuple[int, float]]:
    """Per WRF type: below, at and past every row, midpoints, cut edges."""

    of_group, types = layout.wrf_types()
    kind = of_group[layout.turbine_group]
    out = []
    for t, (k, _, _) in enumerate(types):
        kt = int(np.flatnonzero(kind == t)[0]) + 1
        n = int(layout.tables.nval[k])
        ws = layout.tables.speeds_ms[k, :n]
        speeds = [F(0.0), F(0.5), F(30.0), F(40.0)]
        speeds += list(ws)
        speeds += list(((ws[:-1].astype(np.float64) + ws[1:]) / 2.0)
                       .astype(F))
        for edge in (ws[0], ws[-1]):
            speeds += [np.nextafter(edge, F(0.0)), np.nextafter(edge, F(99.0))]
        speeds += list(rng.uniform(0.0, 32.0, 16).astype(F))
        out += [(kt, float(v)) for v in speeds]
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    library = load_turbine_library()
    curves = _usable_curves(library)
    rng = np.random.default_rng(SEED)
    for c, (case, (spec, nests, cf)) in enumerate(CASES.items()):
        share = curves[c::len(CASES)]
        grid = grid_from_spec(spec, nests)
        rows = _turbines(grid, share, rng, case)
        turbines = [wl.turbine_from_row(r, n + 1) for n, r in enumerate(rows)]
        layout = wl.build_layout(turbines, grid, library=library)
        deck = root / case
        layout.write_wrf_inputs(deck)
        p = layout.projection
        word = float32_word
        (deck / "grid.txt").write_text(
            f"{p.code} {word(grid.truelat1)} {word(grid.truelat2)} "
            f"{word(grid.stand_lon)} {word(grid.dx)}\n"
            f"1 1 1 {grid.e_we}\n"
            f"{word(p.lat1)} {word(p.lon1)}\n", encoding="ascii")
        queries = _queries(layout, rng)
        (deck / "queries.txt").write_text(
            f"{word(cf)} {len(queries)}\n"
            + "".join(f"{kt} {word(v)}\n" for kt, v in queries),
            encoding="ascii")
        (deck / "deck.json").write_text(json.dumps({
            "case": case, "grid": spec, "nests": [list(n) for n in nests],
            "correction_factor": cf, "turbines": rows,
            "turbine_library_sha256": library.sha256,
        }, indent=1) + "\n", encoding="ascii")
        lats = layout.turbine_lat
        lons = layout.turbine_lon
        print(f"{case}: {layout.n_turbines} turbines in lat "
              f"{lats.min():.2f}..{lats.max():.2f}, lon {lons.min():.2f}.."
              f"{lons.max():.2f}; {layout.n_groups} groups, "
              f"{len(layout.wrf_types()[1])} WRF types, {len(queries)} "
              "dragcof queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
