"""Variable-resolution hex cases for testing and benchmarking, and their transition diagnostics.

WHY.  The hex cases the tests and benchmarks used (the option matrix case, the stability/perf regional
cases) are cut from the point ladder at its default grading: a fine core, then one doubling of spacing
before the cut.  None graded over several doublings, none was refined by a field (terrain) and none was a
global variable-resolution mesh, so nothing checked that a mesh whose spacing varies by a factor of four
or more runs as it should: a step limited by the smallest cell, scale-aware schemes reading each cell's own
spacing, and no noise imprinted where the spacing changes.

THE CASES are rows of ``components/hex/bench/varres-cases.v1.json`` (coordinates only):

* ``regional`` rows are ``woof hex regional`` arguments (``--ring-factor``/``--transition-factor``/
  ``--cull-pad-scale`` grade the cut; a ``density`` block builds a ``woof hex density`` raster first and
  hands it over with ``--density-raster``);
* the ``global`` row is ``woof hex mesh-plan --global-mesh`` arguments (the MPAS v8.4.2 suite route);
* a row may name ``references``: the same region on meshes without the transition, read beside it.

COMMANDS (pixi; mac ``-e mlx``, pb ``-e default``)::

    python tools/hex_varres.py list
    python tools/hex_varres.py args NAME [--work-dir DIR]      # the case's arguments, one per line
    python tools/hex_varres.py prep NAME --work-dir DIR [--cycle YYYY-MM-DDTHH] [--hours H] [-- EXTRA...]
    python tools/hex_varres.py analyze --grid GRID --history-dir DIR [--center LAT,LON] [--bins KM]
                                      [--json OUT] [--screen]

``prep`` runs every stage up to the forecast (``--skip forecast --skip products``); the forecast is then
``woof hex regional`` with the same arguments and ``--skip mesh --skip intermediate --skip init --skip lbc
--skip contract``, as the bench scripts run it.

THE DIAGNOSTICS (``analyze``), per radial bin from the centre (or per spacing class): the local cell spacing
``h`` (from ``areaCell``) and the per-cell spacing gradient ``g`` (the largest relative change to a
neighbour, the quantity the generator's transition-band gate bounds), the rms and max of ``w`` between
``--zlo`` and ``--zhi`` above ground, the 2-dx noise of ``w`` (``w`` minus its neighbours' mean) and its
ratio to rms ``w``, the rms horizontal divergence of the cell-centre wind in edge-flux form, the mean
accumulated and convective precipitation and the mean gravity-wave-drag surface stress.

THE SCREEN (``--screen``) is not a proof.  It flags a non-finite field, and a bin inside a transition band
(``g`` above :data:`TRANSITION_GRADIENT`) whose 2-dx noise of ``w`` exceeds :data:`IMPRINT_FACTOR` times the
larger of its two neighbouring bins' -- a transition that makes grid-scale noise rather than passing the flow
through.  Measured (2026-11-14, Edinburgh 6 h from GFS, CUDA): the graded 3 -> 20 km mesh (edi-vr) reads at
most 1.35 against its neighbours, the quasi-uniform 3 km mesh's own ladder band (edi-u3) 1.26 and the 12 km
mesh (edi-u12) 1.19.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "components" / "hex" / "bench" / "varres-cases.v1.json"
SCHEMA = "woof-hex.varres-cases/v1"
ANALYSIS_SCHEMA = "woof-hex.varres-analysis/v1"
#: MPAS sphere radius, m.
EARTH_RADIUS_M = 6371229.0
#: A bin is in a transition band when its median per-cell spacing gradient is above this (fraction).
TRANSITION_GRADIENT = 0.04
#: The screen's imprint threshold: transition-bin 2-dx noise over the larger neighbouring bin's.
IMPRINT_FACTOR = 2.0


def load_cases(path: Path = CASES) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    ids = [case["id"] for case in data["cases"]]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate case ids")
    for case in data["cases"]:
        for ref in case.get("references", ()):
            if ref not in ids:
                raise SystemExit(f"{path}: case {case['id']!r} names unknown reference {ref!r}")
    return data


def case_row(name: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = data or load_cases()
    for case in data["cases"]:
        if case["id"] == name:
            return case
    raise SystemExit(f"no case {name!r}; known: {', '.join(c['id'] for c in data['cases'])}")


def case_arguments(case: Mapping[str, Any], work_dir: Path | None = None) -> list[str]:
    """The door arguments of a case; a density raster is named at ``<work_dir>/<raster>``."""

    if case["kind"] == "global":
        return list(case["mesh_plan_args"])
    argv = list(case["regional_args"])
    density = case.get("density")
    if density is not None:
        if work_dir is None:
            raise SystemExit(f"case {case['id']!r} carries a density raster: pass --work-dir")
        argv += ["--density-raster", str(Path(work_dir) / density["raster"])]
    return argv


def prep_commands(case: Mapping[str, Any], *, work_dir: Path, cycle: str, source: str, hours: float,
                  extra: Sequence[str] = ()) -> list[list[str]]:
    """The ``python -m woof`` argv lists that prepare a case."""

    work_dir = Path(work_dir)
    commands: list[list[str]] = []
    density = case.get("density")
    if density is not None:
        commands.append(["hex", "density", *density["args"], "-o", str(work_dir / density["raster"])])
    if case["kind"] == "global":
        commands.append(["hex", "mesh-plan", *case_arguments(case), "--generate", "--run",
                         "--out-dir", str(work_dir / "global"), "--source", source, "--cycle", cycle,
                         "--hours", f"{hours:g}", *extra])
    else:
        commands.append(["hex", "regional", *case_arguments(case, work_dir), "--cycle", cycle,
                         "--source", source, "--hours", f"{hours:g}", "--work-dir", str(work_dir / case["id"]),
                         "--skip", "forecast", "--skip", "products", *extra])
    return commands


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def neighbour_table(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray, n_cells: int):
    """``(index, valid)``: a cell's neighbours (0-based, padded with 0) and which slots are real."""

    coc = np.asarray(cells_on_cell, dtype=np.int64)
    valid = (np.arange(coc.shape[1])[None, :] < np.asarray(n_edges_on_cell)[:, None]) & (coc >= 0) & (coc < n_cells)
    return np.where(valid, coc, 0), valid


def spacing_gradient(h: np.ndarray, index: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per cell, the largest relative spacing change to a neighbour (fraction per cell)."""

    return np.where(valid, np.abs(h[index] - h[:, None]) / ((h[index] + h[:, None]) / 2.0), 0.0).max(axis=1)


def two_dx_noise(field: np.ndarray, index: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``field`` minus the mean of its neighbours (cells first; any trailing level axis is kept)."""

    weights = valid.astype(float)
    count = np.maximum(weights.sum(axis=1), 1.0)
    gathered = field[index]
    if field.ndim > 1:
        weights = weights[..., None]
        count = count[:, None]
    return field - (gathered * weights).sum(axis=1) / count


def bin_statistics(key: np.ndarray, use: np.ndarray, *, h: np.ndarray, gradient: np.ndarray,
                   w: np.ndarray, w_mask: np.ndarray, noise: np.ndarray,
                   extra: Mapping[str, np.ndarray] | None = None, min_cells: int = 10) -> list[dict[str, Any]]:
    """Per ``key`` bin over cells where ``use``: spacing, gradient, rms/max ``w`` and 2-dx noise inside
    ``w_mask`` (cells x levels), and the mean of each ``extra`` cell field."""

    rows = []
    for k in np.unique(key[use]):
        s = use & (key == k)
        if s.sum() < min_cells:
            continue
        m = w_mask[s]
        if not m.any():
            continue
        w_s, n_s = w[s][m], noise[s][m]
        row = {"key": int(k), "cells": int(s.sum()), "h_km": float(np.median(h[s])),
               "gradient_pct": float(np.median(gradient[s]) * 100.0),
               "w_rms": float(np.sqrt(np.mean(w_s ** 2))), "w_max": float(np.abs(w_s).max()),
               "noise_rms": float(np.sqrt(np.mean(n_s ** 2)))}
        row["noise_over_w"] = row["noise_rms"] / max(row["w_rms"], 1e-30)
        for name, values in (extra or {}).items():
            row[name] = float(np.mean(values[s]))
        rows.append(row)
    return rows


def imprint_screen(rows: Sequence[Mapping[str, Any]], *, transition: float = TRANSITION_GRADIENT,
                   factor: float = IMPRINT_FACTOR) -> dict[str, Any]:
    """Transition bins whose 2-dx noise exceeds ``factor`` x the larger of their neighbouring bins'."""

    worst, flagged = 0.0, []
    for i in range(1, len(rows) - 1):
        row = rows[i]
        if row["gradient_pct"] / 100.0 <= transition:
            continue
        around = max(rows[i - 1]["noise_rms"], rows[i + 1]["noise_rms"])
        ratio = row["noise_rms"] / max(around, 1e-30)
        worst = max(worst, ratio)
        if ratio > factor:
            flagged.append({"key": row["key"], "h_km": row["h_km"], "gradient_pct": row["gradient_pct"],
                            "noise_ratio_to_neighbours": ratio})
    return {"transition_gradient": transition, "factor": factor, "worst_ratio": worst, "flagged": flagged}


def _divergence(u, v, c1, c2, cos_e, sin_e, dv, area, n_cells):
    flux = 0.5 * ((u[c1] + u[c2]) * cos_e[:, None] + (v[c1] + v[c2]) * sin_e[:, None]) * dv[:, None]
    out = np.zeros((n_cells, u.shape[1]))
    np.add.at(out, c1, flux)
    np.add.at(out, c2, -flux)
    return out / area[:, None]


def analyze(grid: Path, history_dir: Path, *, center: tuple[float, float] | None, bins_km: float = 25.0,
            zlo: float = 1000.0, zhi: float = 6000.0) -> dict[str, Any]:
    import netCDF4

    with netCDF4.Dataset(str(grid)) as g:
        area = np.asarray(g["areaCell"][:], dtype=float)
        dv = np.asarray(g["dvEdge"][:], dtype=float)
        if area.max() < 1.0:  # a unit-sphere grid
            area = area * EARTH_RADIUS_M ** 2
            dv = dv * EARTH_RADIUS_M
        n_cells = area.size
        index, valid = neighbour_table(g["cellsOnCell"][:] - 1, g["nEdgesOnCell"][:], n_cells)
        coe = np.asarray(g["cellsOnEdge"][:], dtype=np.int64) - 1
        angle = np.asarray(g["angleEdge"][:], dtype=float)
        lat = np.asarray(g["latCell"][:], dtype=float)
        lon = np.asarray(g["lonCell"][:], dtype=float)
    h = np.sqrt(area / (math.sqrt(3.0) / 2.0)) / 1000.0
    gradient = spacing_gradient(h, index, valid)
    edge_ok = (coe[:, 0] >= 0) & (coe[:, 1] >= 0) & (coe[:, 0] < n_cells) & (coe[:, 1] < n_cells)
    c1, c2 = coe[edge_ok, 0], coe[edge_ok, 1]
    cos_e, sin_e, dv_e = np.cos(angle[edge_ok]), np.sin(angle[edge_ok]), dv[edge_ok]
    on_rim = np.zeros(n_cells, dtype=bool)
    rim_edges = ~edge_ok
    for column in (0, 1):
        ids = coe[rim_edges, column]
        on_rim[ids[(ids >= 0) & (ids < n_cells)]] = True
    use = ~on_rim
    if center is not None:
        clat, clon = (math.radians(x) for x in center)
        distance = EARTH_RADIUS_M / 1000.0 * np.arccos(np.clip(
            np.sin(lat) * math.sin(clat) + np.cos(lat) * math.cos(clat) * np.cos(lon - clon), -1.0, 1.0))
        key = np.floor(distance / bins_km).astype(int)
        basis = {"kind": "radial", "center_deg": list(center), "bin_km": bins_km}
    else:
        key = np.floor(np.log(h / h.min()) / np.log(1.25)).astype(int)
        basis = {"kind": "spacing", "bin_ratio": 1.25}
    frames = sorted(glob.glob(str(Path(history_dir) / "*history*.nc")))
    if not frames:
        raise SystemExit(f"no history frames in {history_dir}")
    series, nonfinite = [], []
    for path in frames:
        with netCDF4.Dataset(path) as d:
            names = d.variables
            stamp = bytes(np.asarray(d["xtime"][0])).decode(errors="replace").strip("\x00 ")
            for name, var in names.items():
                if var.dtype.kind == "f" and not np.all(np.isfinite(np.asarray(var[:]))):
                    nonfinite.append(f"{Path(path).name}:{name}")
            w = np.asarray(d["w"][0], dtype=float)
            zgrid = np.asarray(d["zgrid"][:], dtype=float)
            u = np.asarray(d["u_zonal"][0], dtype=float)
            v = np.asarray(d["v_meridional"][0], dtype=float)
            zero = np.zeros(n_cells)
            rainc = np.asarray(d["rainc"][0], dtype=float) if "rainc" in names else zero
            rainnc = np.asarray(d["rainnc"][0], dtype=float) if "rainnc" in names else zero
            stress = (np.hypot(np.asarray(d["dusfcg"][0], dtype=float), np.asarray(d["dvsfcg"][0], dtype=float))
                      if "dusfcg" in names else zero)
        agl = zgrid - zgrid[:, :1]
        w_mask = (agl >= zlo) & (agl <= zhi)
        mid = 0.5 * (agl[:, 1:] + agl[:, :-1])
        div = _divergence(u, v, c1, c2, cos_e, sin_e, dv_e, area, n_cells)
        div_rms = np.sqrt(np.where((mid >= zlo) & (mid <= zhi), div ** 2, 0.0).sum(axis=1)
                          / np.maximum(((mid >= zlo) & (mid <= zhi)).sum(axis=1), 1))
        rows = bin_statistics(key, use, h=h, gradient=gradient, w=w, w_mask=w_mask,
                              noise=two_dx_noise(w, index, valid),
                              extra={"div_rms_per_s": div_rms, "precip_mm": rainc + rainnc,
                                     "conv_precip_mm": rainc, "gwd_stress_n_m2": stress})
        series.append({"time": stamp, "bins": rows, "screen": imprint_screen(rows)})
    return {"schema": ANALYSIS_SCHEMA, "grid": str(grid), "history_dir": str(history_dir), "basis": basis,
            "cells": int(n_cells), "h_km": [float(h.min()), float(h.max())],
            "max_gradient_pct": float(gradient.max() * 100.0), "nonfinite": nonfinite, "series": series}


def _table(result: Mapping[str, Any]) -> list[str]:
    last = result["series"][-1]
    basis = result["basis"]
    lines = [f"{result['history_dir']} {last['time']}: {result['cells']} cells, h {result['h_km'][0]:.2f}-"
             f"{result['h_km'][1]:.2f} km, steepest {result['max_gradient_pct']:.1f} %/cell, "
             f"non-finite: {', '.join(result['nonfinite']) or 'none'}",
             f"{'bin':>12} {'cells':>6} {'h km':>6} {'g %':>5} {'w rms':>7} {'w max':>6} {'2dx':>7} {'2dx/w':>6} "
             f"{'div e-5':>8} {'pr mm':>6} {'cu mm':>6} {'gwd':>7}"]
    for r in last["bins"]:
        label = (f"{r['key'] * basis['bin_km']:.0f}-{(r['key'] + 1) * basis['bin_km']:.0f} km"
                 if basis["kind"] == "radial" else f"h x{1.25 ** r['key']:.2f}")
        lines.append(f"{label:>12} {r['cells']:>6} {r['h_km']:6.2f} {r['gradient_pct']:5.1f} {r['w_rms']:7.4f} "
                     f"{r['w_max']:6.2f} {r['noise_rms']:7.4f} {r['noise_over_w']:6.2f} "
                     f"{r['div_rms_per_s'] * 1e5:8.2f} {r['precip_mm']:6.2f} {r['conv_precip_mm']:6.3f} "
                     f"{r['gwd_stress_n_m2']:7.4f}")
    screen = last["screen"]
    lines.append(f"screen: worst transition-bin 2-dx noise / neighbours {screen['worst_ratio']:.2f} "
                 f"(flag above {screen['factor']:g}); flagged {len(screen['flagged'])}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    p_args = sub.add_parser("args")
    p_args.add_argument("name")
    p_args.add_argument("--work-dir", type=Path)
    p_prep = sub.add_parser("prep")
    p_prep.add_argument("name")
    p_prep.add_argument("--work-dir", type=Path, required=True)
    p_prep.add_argument("--cycle")
    p_prep.add_argument("--source")
    p_prep.add_argument("--hours", type=float, default=6.0)
    p_prep.add_argument("--dry-run", action="store_true", help="print the commands only")
    p_prep.add_argument("extra", nargs="*", help="after --: more door arguments")
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--grid", type=Path, required=True)
    p_an.add_argument("--history-dir", type=Path, required=True)
    p_an.add_argument("--center")
    p_an.add_argument("--bins", type=float, default=25.0)
    p_an.add_argument("--zlo", type=float, default=1000.0)
    p_an.add_argument("--zhi", type=float, default=6000.0)
    p_an.add_argument("--json", type=Path)
    p_an.add_argument("--screen", action="store_true", help="exit 1 on a non-finite field or a flagged bin")
    a = parser.parse_args(argv)
    if a.command == "list":
        for case in load_cases()["cases"]:
            refs = f" (references: {', '.join(case['references'])})" if case.get("references") else ""
            print(f"{case['id']:<12} {case['kind']:<8} {case['about']}{refs}")
        return 0
    if a.command == "args":
        print("\n".join(case_arguments(case_row(a.name), a.work_dir)))
        return 0
    if a.command == "prep":
        data = load_cases()
        case = case_row(a.name, data)
        commands = prep_commands(case, work_dir=a.work_dir, cycle=a.cycle or data["cycle_default"],
                                 source=a.source or data["source_default"], hours=a.hours, extra=a.extra)
        for command in commands:
            print("python -m woof " + " ".join(command), flush=True)
            if not a.dry_run:
                Path(a.work_dir).mkdir(parents=True, exist_ok=True)
                code = subprocess.call([sys.executable, "-m", "woof", *command], cwd=str(REPO))
                if code:
                    return code
        return 0
    center = tuple(float(x) for x in a.center.split(",")) if a.center else None
    result = analyze(a.grid, a.history_dir, center=center, bins_km=a.bins, zlo=a.zlo, zhi=a.zhi)
    print("\n".join(_table(result)))
    if a.json:
        a.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if a.screen and (result["nonfinite"] or result["series"][-1]["screen"]["flagged"]):
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
