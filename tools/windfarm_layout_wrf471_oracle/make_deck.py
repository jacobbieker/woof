#!/usr/bin/env python3
"""Write the oracle's decks: synthetic farms laid out by woof, as WRF reads them.

    python -m tools.windfarm_layout_wrf471_oracle.make_deck DECK_ROOT

One directory per case under DECK_ROOT, each holding what a WRF run of the
farm reads -- ``windturbines.txt`` and ``wind-turbine-N.tbl``, written by
:meth:`woof.windfarm.layout.WindFarmLayout.write_wrf_inputs` -- plus the
harness's ``grid.txt`` (WRF's config_flags and the float32 XLAT/XLONG of
the first mass point) and ``queries.txt`` (dragcof calls), and
``deck.json``, the woof inputs the parity test rebuilds the layout from.

Every farm is synthetic: seeded random turbines over each grid's interior,
a block of turbines on exact half-cell projection coordinates (where
``nint`` rounds), and cells holding several machines.  The four grids
cover Lambert secant in both hemispheres and the tangent arm at 12, 1, 3
and 4 km, and between them every curve of the bundled turbine library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from woof.energy.turbine_library import load_turbine_library
from woof.static.lambert import LambertGrid
from woof.windfarm import layout as wl
from woof.windfarm.fitch_tables import (FitchTableRefused, float32_word,
                                        table_rows)

F = np.float32

#: id, ref_lat, ref_lon, truelat1, truelat2, stand_lon, dx, e_we, e_sn,
#: correction_factor (WRF's windfarm_tke_factor)
GRIDS = (
    ("nh_secant_12km", 35.5, -97.5, 25.5, 45.5, -97.5, 12000.0, 61, 49, 1.0),
    ("nh_secant_1km", 35.205, -97.895, 30.0, 60.0, -97.9, 1000.0, 133, 103,
     0.25),
    ("sh_secant_3km", -30.0, 140.0, -20.0, -40.0, 140.0, 3000.0, 81, 81, 1.0),
    ("tangent_4km", 45.0, 10.0, 45.0, 45.0, 10.0, 4000.0, 61, 61, 0.5),
)
RANDOM_TURBINES = 400
HALF_CELL_TURBINES = 24
SEED = 4711


def _grid(spec) -> LambertGrid:
    _, lat, lon, t1, t2, std, dx, e_we, e_sn, _ = spec
    return LambertGrid(ref_lat=lat, ref_lon=lon, truelat1=t1, truelat2=t2,
                       stand_lon=std, dx=dx, dy=dx, e_we=e_we, e_sn=e_sn)


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
    # one cell holding two machines and a within-tolerance hub spread
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
        n = int(layout.tables.rows[k])
        ws = layout.tables.speed_ms[k, :n]
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
    chunk = -(-len(curves) // 3)
    shares = [curves[0:chunk], curves[chunk:2 * chunk], curves[2 * chunk:],
              [library.get("V112/3000"), library.get("V126/3000"),
               library.get("V90/3000")]]
    rng = np.random.default_rng(SEED)
    for spec, share in zip(GRIDS, shares):
        case, cf = spec[0], spec[-1]
        grid = _grid(spec)
        rows = _turbines(grid, share, rng, case)
        turbines = [wl.turbine_from_row(r, n + 1) for n, r in enumerate(rows)]
        layout = wl.build_layout(turbines, grid, library=library)
        deck = root / case
        layout.write_wrf_inputs(deck)
        p = layout.projection
        (deck / "grid.txt").write_text(
            f"1 {float32_word(p.truelat1)} {float32_word(p.truelat2)} {float32_word(p.stdlon)} "
            f"{float32_word(p.dx)}\n"
            f"1 1 1 {grid.e_we}\n"
            f"{float32_word(p.lat1)} {float32_word(p.lon1)}\n", encoding="ascii")
        queries = _queries(layout, rng)
        (deck / "queries.txt").write_text(
            f"{float32_word(cf)} {len(queries)}\n"
            + "".join(f"{kt} {float32_word(v)}\n" for kt, v in queries),
            encoding="ascii")
        (deck / "deck.json").write_text(json.dumps({
            "case": case,
            "grid": dict(zip(("ref_lat", "ref_lon", "truelat1", "truelat2",
                              "stand_lon", "dx", "e_we", "e_sn"), spec[1:9])),
            "correction_factor": cf,
            "turbines": rows,
            "turbine_library_sha256": library.sha256,
        }, indent=1) + "\n", encoding="ascii")
        print(f"{case}: {layout.n_turbines} turbines, {layout.n_groups} "
              f"groups, {len(layout.wrf_types()[1])} WRF types, "
              f"{len(queries)} dragcof queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
