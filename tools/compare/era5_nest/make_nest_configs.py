#!/usr/bin/env python3
"""Emit the woof experiment TOMLs that reproduce Breeze's two ERA5 domains.

The domain wizard (``woof domain``) writes the config -- physics suite,
ERA5 route, fetch table, WPS namelist, time step -- and this script then
pins the three things the wizard chooses for itself and the comparison
needs fixed:

* the cell counts, to Breeze's ``Nx x Ny`` exactly (the wizard sizes a
  ``--point`` request to a budget, not to a count);
* the forcing area, recomputed for those counts with the wizard's own
  :func:`woof.domain_wizard._fetch_area`, so the boundary data still
  covers the grid plus the wizard's margin;
* optionally the radiation interval (``--radt-minutes``; Breeze solves
  RRTMGP hourly) and the restart cadence (``--no-restarts``; Breeze's
  timed runs write no checkpoint).

The 50 layers are Breeze's own interface heights mapped onto eta through a
standard-atmosphere column (:func:`breeze_cases.breeze_matched_eta`).

Usage (from the repository root)::

    python tools/compare/era5_nest/make_nest_configs.py --case mc3e \\
        --out-dir ~/wf-scratch/regional-benchmark/mc3e --hours 3 --vram-gib 6

Every patch is checked to hit exactly one line, so a wizard that changes
its output shape fails here instead of producing a silently different grid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import breeze_cases as cases  # noqa: E402

#: The verified woof suite closest to Breeze's nest physics.  Breeze: 1-moment
#: mixed-phase microphysics, all-sky RRTMGP, slab land with Monin-Obukhov
#: fluxes, no PBL, no cumulus.  This suite: Thompson (mp 8), RTE+RRTMGP, Noah
#: land with MM5 surface layer, YSU PBL, no cumulus.
DEFAULT_PHYSICS = "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1"


def wizard_argv(case: cases.Case, *, name: str, out: Path, hours: int,
                cadence: int, provider: str, physics: str,
                vram_gib: float) -> list[str]:
    eta, p_top = cases.breeze_matched_eta()
    lat, lon = case.center
    extent_km = max(case.nx, case.ny) * case.dx_m / 1000.0
    return [
        "domain",
        f"--point={lat:.4f},{lon:.4f}",
        "--projection", "lambert",
        "--point-extent-km", f"{extent_km:g}",
        "--root-dx", f"{case.dx_m / 1000.0:g}",
        "--eta-levels", ",".join(f"{value:.6f}" for value in eta),
        "--p-top-pa", f"{p_top:.0f}",
        "--source", "era5",
        "--era5-provider", provider,
        "--cadence", str(cadence),
        "--cycle", case.cycle,
        "--hours", str(hours),
        "--vram-gib", f"{vram_gib:g}",
        "--physics-profile", physics,
        "--history-interval", "3600",
        "--name", name,
        "--out", str(out),
    ]


def substitute_once(text: str, pattern: str, replacement: str,
                    what: str) -> str:
    new, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"{what}: expected exactly one match for "
                           f"{pattern!r}, found {count}")
    return new


def projection_of(toml_text: str) -> dict:
    import tomllib

    table = tomllib.loads(toml_text)["projection"]
    return {key: table[key] for key in
            ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2",
             "stand_lon")}


def fetch_area(projection: dict, nx: int, ny: int, dx_m: float) -> str:
    from woof.domain_wizard import _fetch_area

    south, west, north, east = _fetch_area(projection, nx, ny,
                                           root_dx_m=dx_m)
    return f"{south:.2f},{west:.2f},{north:.2f},{east:.2f}"


def patch_toml(text: str, case: cases.Case, *, area: str,
               radt_minutes: float | None, restarts: bool) -> str:
    text = substitute_once(text, r"^nx = \d+$", f"nx = {case.nx}", "nx")
    text = substitute_once(text, r"^ny = \d+$", f"ny = {case.ny}", "ny")
    text = substitute_once(text, r'^area = "[^"]*"$', f'area = "{area}"',
                           "fetch area")
    if radt_minutes is not None:
        text = substitute_once(text, r"^radt = [0-9.]+$",
                               f"radt = {radt_minutes:.1f}", "radt")
    if not restarts:
        text = substitute_once(text, r"^restart_interval_s = [0-9.]+$",
                               "restart_interval_s = 0.0", "restart interval")
    return text


def patch_namelist(text: str, case: cases.Case) -> str:
    text = substitute_once(text, r"^ e_we\s*=\s*\d+,$",
                           f" e_we              = {case.nx + 1},", "e_we")
    text = substitute_once(text, r"^ e_sn\s*=\s*\d+,$",
                           f" e_sn              = {case.ny + 1},", "e_sn")
    return text


def make(case: cases.Case, *, out_dir: Path, hours: int, cadence: int,
         provider: str, physics: str, vram_gib: float,
         radt_minutes: float | None, restarts: bool,
         python: str = sys.executable) -> Path:
    # Absolute: the wizard runs with cwd=out_dir and is handed --out.
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{case.key}_breeze_match"
    toml_path = out_dir / f"{name}.toml"
    argv = wizard_argv(case, name=name, out=toml_path, hours=hours,
                       cadence=cadence, provider=provider, physics=physics,
                       vram_gib=vram_gib)
    # The wizard runs in out_dir (its relative data paths land there) from
    # this checkout, whatever the caller's PYTHONPATH says.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    subprocess.run([python, "-m", "woof", *argv], check=True, cwd=out_dir,
                   env=env)
    text = toml_path.read_text(encoding="utf-8")
    area = fetch_area(projection_of(text), case.nx, case.ny, case.dx_m)
    toml_path.write_text(patch_toml(text, case, area=area,
                                    radt_minutes=radt_minutes,
                                    restarts=restarts), encoding="utf-8")
    namelist = out_dir / f"{name}.namelist.wps"
    namelist.write_text(patch_namelist(namelist.read_text(encoding="utf-8"),
                                       case), encoding="utf-8")
    summary = {
        "case": case.key,
        "config": str(toml_path),
        "nx": case.nx, "ny": case.ny, "nz": case.nz,
        "dx_m": case.dx_m,
        "cycle": case.cycle, "hours": hours, "cadence_h": cadence,
        "era5_provider": provider,
        "physics_profile": physics,
        "radt_minutes": radt_minutes,
        "restarts": restarts,
        "fetch_area": area,
        "breeze_spacing_km": cases.breeze_spacing_km(case),
        "ground_coverage": cases.ground_coverage_km(case),
        "wizard_argv": argv,
    }
    (out_dir / f"{name}.match.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return toml_path


def shrink(case: cases.Case, factor: int) -> cases.Case:
    """The same case on an ``Nx/factor x Ny/factor`` grid (same centre, dx)."""

    if factor < 1:
        raise ValueError("--shrink takes a whole factor >= 1")
    if factor == 1:
        return case
    nx, ny = case.nx // factor, case.ny // factor
    lat0, lon0 = case.center
    # The Breeze box shrinks with the counts, so its spacing is unchanged.
    half_lon = 0.5 * (case.lon_bounds[1] - case.lon_bounds[0]) * nx / case.nx
    half_lat = 0.5 * (case.lat_bounds[1] - case.lat_bounds[0]) * ny / case.ny
    return dataclasses.replace(
        case, key=f"{case.key}_shrink{factor}", nx=nx, ny=ny,
        lon_bounds=(lon0 - half_lon, lon0 + half_lon),
        lat_bounds=(lat0 - half_lat, lat0 + half_lat))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", choices=sorted(cases.CASES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int,
                        help="forecast length (default: the published run's)")
    parser.add_argument("--cadence", type=int, default=1,
                        help="ERA5 boundary spacing in hours (default 1, "
                             "Breeze's hourly parent)")
    parser.add_argument("--era5-provider", choices=("arco", "cds"),
                        default="arco")
    parser.add_argument("--physics-profile", default=DEFAULT_PHYSICS)
    parser.add_argument("--vram-gib", type=float, default=48.0,
                        help="card the wizard sizes for (default 48, the "
                             "RTX A6000 the timings run on)")
    parser.add_argument("--radt-minutes", type=float, default=60.0,
                        help="radiation interval; default 60 (Breeze's "
                             "hourly RRTMGP); pass 0 to keep the wizard's")
    parser.add_argument("--no-restarts", action="store_true",
                        help="write no restart files during the run")
    parser.add_argument("--shrink", type=int, default=1,
                        help="divide Nx and Ny by this factor at the same "
                             "centre and spacing: a reduced-domain "
                             "correctness run on a card the full case does "
                             "not fit (default 1, the published grid)")
    args = parser.parse_args(argv)
    case = shrink(cases.CASES[args.case], args.shrink)
    path = make(case, out_dir=args.out_dir.expanduser(),
                hours=args.hours or case.hours, cadence=args.cadence,
                provider=args.era5_provider, physics=args.physics_profile,
                vram_gib=args.vram_gib,
                radt_minutes=(args.radt_minutes or None),
                restarts=not args.no_restarts)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
