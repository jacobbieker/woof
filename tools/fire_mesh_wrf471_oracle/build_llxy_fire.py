#!/usr/bin/env python3
"""Fire-mesh cell centres through WRF's own map projection (module_llxy).

    python tools/fire_mesh_wrf471_oracle/build_llxy_fire.py WRF_ROOT

The LANDFIRE fuel reader (woof.static.fuel.landfire_pixel_indices) places
each fire-mesh cell centre through the grid's own ``ij_to_latlon``.  This
oracle places the same centres through WRF v4.7.1 share/module_llxy.F
``ij_to_latlon``.  That file is byte-identical to the v4.6.1 file
tools/llxy_wrf461_oracle builds (sha256 below), and this build uses the same
stubs, flags and driver (binary64, -O0, no contraction).  The parity test
holds woof's latitude and longitude words to WRF's at 0 ULP for every
projection woof grids use, in both hemispheres, and holds the pixel each
cell takes to the one WRF's words fall in.

A nested case places its child the way WPS geogrid does and woof's
``ProjectedGrid.nest`` does.  The parent's ij_to_latlon at the child's
(1, 1) mass point, ``(i_parent_start - 0.5) + 0.5/ratio``, becomes the
child's known point at (1, 1), with spacing ``dx/ratio``.  The oracle
computes that point itself in a first pass.

Queries are the fire-cell centres in 1-based mass coordinates,
``0.5 + (k + 0.5) / sr`` (woof.static.fuel.fire_mesh_axis).

Needs gfortran on PATH (``pixi exec -s gfortran -s python -- ...``).
Writes woof/data/fire/oracle/llxy_fire_mesh/{deck.json,fixture.csv}.
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from tools.llxy_wrf461_oracle.build_llxy import (  # noqa: E402
    FFLAGS, SOURCE_SHA256, STUBS)

OUT = REPO / "woof" / "data" / "fire" / "oracle" / "llxy_fire_mesh"
#: WRF v4.7.1 (tag v4.7.1); share/module_llxy.F there is byte-identical to
#: v4.6.1's, so the pin is the same SHA-256.
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

#: (id, code, ref_lat, ref_lon, stand_lon, truelat1, truelat2, nx, ny, dx,
#:  sr_x, sr_y).  code: 1 = PROJ_LC, 2 = PROJ_PS, 3 = PROJ_MERC.  The
#: reference point is the WPS default, the mass-grid centre (e_we/2,
#: e_sn/2).  Lambert secant and tangent in both hemispheres, Mercator north,
#: south and on the equator, polar stereographic over both poles.
CASES = (
    ("lc_nh_conus", 1, 39.7, -105.3, -105.0, 30.0, 60.0, 6, 5, 3000.0, 3, 3),
    ("lc_nh_boreal", 1, 62.5, 25.0, 24.0, 55.0, 70.0, 5, 4, 1000.0, 4, 2),
    ("lc_sh_secant", 1, -33.9, 151.2, 151.0, -30.0, -45.0, 5, 5, 2000.0, 3, 3),
    ("lc_sh_tangent", 1, -42.0, 147.0, 147.0, -42.0, -42.0, 4, 4, 1500.0,
     2, 2),
    ("merc_nh", 3, 19.6, -155.5, -155.5, 19.6, 19.6, 4, 4, 2000.0, 5, 5),
    ("merc_sh", 3, -8.5, 115.2, 115.2, -8.5, -8.5, 5, 4, 1000.0, 3, 3),
    ("merc_equator", 3, 0.5, -78.5, -78.5, 0.0, 0.0, 4, 5, 3000.0, 2, 2),
    ("ps_nh", 2, 64.8, -147.7, -147.7, 60.0, 60.0, 5, 5, 3000.0, 3, 3),
    ("ps_sh", 2, -77.85, 166.7, 166.7, -71.0, -71.0, 4, 4, 2000.0, 2, 2),
)
#: (id, parent id, i_parent_start, j_parent_start, ratio, nx, ny, sr).
NESTS = (
    ("nest_lc_nh", "lc_nh_conus", 2, 2, 3, 6, 6, 2),
    ("nest_ps_sh", "ps_sh", 2, 1, 2, 4, 4, 3),
)


def axis(n: int, sr: int) -> list[float]:
    return [0.5 + (k + 0.5) / sr for k in range(n * sr)]


def _word(text: str) -> float:
    return struct.unpack(">d", bytes.fromhex(text))[0]


class _Oracle:
    def __init__(self, build: Path, source: Path):
        self.build = build
        (build / "module_llxy.F").write_bytes(source.read_bytes())
        (build / "wrf_stubs.f90").write_text(STUBS)
        (build / "run_llxy.F90").write_bytes(
            (REPO / "tools" / "llxy_wrf461_oracle" / "run_llxy.F90")
            .read_bytes())
        flags = FFLAGS.split()
        for name in ("wrf_stubs.f90", "module_llxy.F", "run_llxy.F90"):
            subprocess.run(["gfortran", "-c", *flags, name], cwd=build,
                           check=True)
        subprocess.run(["gfortran", "-o", "run_llxy", "run_llxy.o",
                        "module_llxy.o", "wrf_stubs.o", "-lm"], cwd=build,
                       check=True)

    def run(self, deck: str) -> str:
        return subprocess.run(["./run_llxy"], cwd=self.build, input=deck,
                              capture_output=True, text=True,
                              check=True).stdout


def _case_line(cid, code, lat, lon, knowni, knownj, dx, stdlon, tl1, tl2):
    return (f"CASE {cid} {code} {lat!r} {lon!r} {knowni!r} {knownj!r} "
            f"{dx!r} {stdlon!r} {tl1!r} {tl2!r}")


def build_fixture(oracle: _Oracle) -> tuple[str, dict]:
    deck: dict = {"cases": {}}
    table = {case[0]: case for case in CASES}
    # pass 1: each nest's known point, through its parent's projection
    origin_lines = []
    for nid, pid, ips, jps, ratio, *_ in NESTS:
        (_, code, lat, lon, stdlon, tl1, tl2, nx, ny, dx, _, _) = table[pid]
        origin_lines.append(_case_line(f"{nid}_parent", code, lat, lon,
                                       (nx + 1) / 2.0, (ny + 1) / 2.0, dx,
                                       stdlon, tl1, tl2))
        xp = (ips - 0.5) + 0.5 / ratio
        yp = (jps - 0.5) + 0.5 / ratio
        origin_lines.append(f"IJLL {nid}_parent {xp!r} {yp!r}")
    origins = {}
    for line in oracle.run("\n".join(origin_lines) + "\n").splitlines():
        if line.startswith("IJLL"):
            _, cid, _, _, lat, lon = line.split()
            origins[cid[:-len("_parent")]] = (_word(lat), _word(lon))
    # pass 2: every fire-cell centre
    lines = []
    for (cid, code, lat, lon, stdlon, tl1, tl2, nx, ny, dx,
         sr_x, sr_y) in CASES:
        knowni, knownj = (nx + 1) / 2.0, (ny + 1) / 2.0
        lines.append(_case_line(cid, code, lat, lon, knowni, knownj, dx,
                                stdlon, tl1, tl2))
        lines += [f"IJLL {cid} {x!r} {y!r}"
                  for y in axis(ny, sr_y) for x in axis(nx, sr_x)]
        deck["cases"][cid] = {
            "code": code, "ref_lat": lat, "ref_lon": lon, "stand_lon": stdlon,
            "truelat1": tl1, "truelat2": tl2, "nx": nx, "ny": ny, "dx": dx,
            "sr_x": sr_x, "sr_y": sr_y, "known_x": knowni,
            "known_y": knownj}
    for nid, pid, ips, jps, ratio, nx, ny, sr in NESTS:
        (_, code, _, _, stdlon, tl1, tl2, _, _, dx, _, _) = table[pid]
        lat11, lon11 = origins[nid]
        lines.append(_case_line(nid, code, lat11, lon11, 1.0, 1.0,
                                dx / ratio, stdlon, tl1, tl2))
        lines += [f"IJLL {nid} {x!r} {y!r}"
                  for y in axis(ny, sr) for x in axis(nx, sr)]
        deck["cases"][nid] = {
            "parent": pid, "i_parent_start": ips, "j_parent_start": jps,
            "parent_grid_ratio": ratio, "code": code, "nx": nx, "ny": ny,
            "sr_x": sr, "sr_y": sr, "known_lat": lat11, "known_lon": lon11}
    return oracle.run("\n".join(lines) + "\n"), deck


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    source = Path(argv[1]) / "share" / "module_llxy.F"
    got = hashlib.sha256(source.read_bytes()).hexdigest()
    if got != SOURCE_SHA256:
        raise SystemExit(f"module_llxy.F sha256 {got} != {SOURCE_SHA256}")
    with tempfile.TemporaryDirectory() as build:
        out, deck = build_fixture(_Oracle(Path(build), source))
        compiler = subprocess.run(["gfortran", "--version"],
                                  capture_output=True, text=True,
                                  check=True).stdout.splitlines()[0]
    header = "\n".join([
        "# fire-mesh cell centres through WRF v4.7.1 share/module_llxy.F",
        f"# source_sha256: {SOURCE_SHA256} (byte-identical in v4.6.1)",
        f"# wrf_commit: {WRF_COMMIT} (tag v4.7.1)",
        f"# compiler: {compiler}",
        f"# flags: {FFLAGS}",
        "# rows: SETUP id code hemi cone rebydx rsw polei polej dlon | "
        "IJLL id i j lat lon (binary64 hex words)",
        "# regenerate: tools/fire_mesh_wrf471_oracle/build_llxy_fire.py",
    ]) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fixture.csv").write_text(header + out)
    (OUT / "deck.json").write_text(json.dumps(deck, indent=1) + "\n")
    # Receipts: the fixture's own hashes, and the harness that wrote it.
    (OUT / "FIXTURES.sha256").write_text("".join(
        f"{hashlib.sha256((OUT / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("deck.json", "fixture.csv")))
    harness = ("build_llxy_fire.py", "../llxy_wrf461_oracle/build_llxy.py",
               "../llxy_wrf461_oracle/run_llxy.F90")
    (OUT / "HARNESS.sha256").write_text("".join(
        f"{hashlib.sha256((HERE / rel).read_bytes()).hexdigest()}  {rel}\n"
        for rel in harness))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
