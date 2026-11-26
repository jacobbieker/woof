"""The WRF v4.8.1 source pin: what the fork's oracle lanes are re-pinned to.

The pin is the v4.8.1 tag commit plus its submodule commits (the gitlinks),
not the release asset: v4.8.1's ``v4.8.1.tar.gz`` and ``v4.8.1.zip`` ship
``phys/physics_mmm``, ``MYNN-EDMF``, ``MYNN-SFC``, ``GFL``, ``TEMPO``,
``noahmp`` and ``fire_behavior`` empty (v4.8.0's asset carried them), so a
tree that builds is a recursive clone of the tag.  The asset's sha256 is kept
as a secondary fact.

``source-v4.8.1.json`` beside this file records, for every WRF source path
the repository cites (``tools/``, ``woof/``, ``tests/``, ``docs/``; Noah-MP
v5 excluded on licence grounds), its sha256 in 4.8.1 and in 4.8.0, and the
4.8.0 -> 4.8.1 change ids of the paths that differ.  A lane whose cited
files are all byte-identical re-pins by label and digest only.

    python -m tools.wrf481_source.manifest build --tree481 WRF481 --tree480 WRF480
    python -m tools.wrf481_source.manifest lane tools/boulac_wrf471_oracle
    python -m tools.wrf481_source.manifest check --tree WRF481

``build`` rewrites the json from two source trees; ``lane`` lists a lane's
cited files with their 4.8.1 status; ``check`` verifies a tree against the
pin (run it on the oracle host before a build).  Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANIFEST = HERE / "source-v4.8.1.json"

#: Facts of the pin (WRF-4.8.1-PLAN.md section 1).
PIN = {
    "tag": "v4.8.1",
    "commit": "e209f84a1a811424990a8f2f373a0b77c0d35f0d",
    "tree": "a6fabccf8cab800467b2e8499ddbee64186fd2f4",
    "url": "https://github.com/wrf-model/WRF",
    "released": "2026-11-23",
    "license": "public domain",
    "submodules": {
        "phys/GFL": "873b414670323ae766ec4fe1f4b6b5ab891358ed",
        "phys/MYNN-EDMF": "8f8e8b59021249949c04d1686edc5a02ca325627",
        "phys/MYNN-SFC": "b66187f54a0c169e36592552634867474e603ac0",
        "phys/TEMPO": "881614f68aa65de8f383d56139162049baf600fa",
        "phys/physics_mmm": "d524bc9b8c8cf3e56dbe12f6dd0c728936c79153",
        "phys/noahmp": "e7b26ca47793edef99ce4dda1a9ac234cb352710",
        "phys/fire_behavior": "f456cd5f00ce1ea42e28486f6d970599e3626272",
    },
    "release_asset": {
        "name": "v4.8.1.tar.gz",
        "bytes": 104369508,
        "sha256": "777cd5a2b859387a44e45d31939aa515543885fdc8265c9c7f69dca6172f2316",
        "note": "secondary fact: the asset ships every submodule directory "
                "empty; outside them it equals the tag tree",
    },
    "previous": {
        "tag": "v4.8.0",
        "commit": "06d4240ae989cc3e50af412bb472df3d9048783c",
        "release_asset_sha256":
            "f3b997266379e1719d186c92471a8f44f8185e976244abde82145a7fddeaf418",
    },
}

#: The 4.8.0 -> 4.8.1 changes (WRF-4.8.1-PLAN.md section 2) by file.
CHANGES = {
    "dyn_em/solve_em.F": ["R1", "R6"],
    "dyn_em/module_small_step_em.F": ["R1"],
    "Registry/Registry.EM_COMMON": ["R1", "W2"],
    "dyn_em/module_big_step_utilities_em.F": ["R2"],
    "dyn_em/module_diffusion_em.F": ["R3"],
    "dyn_em/couple_or_uncouple_em.F": ["R4"],
    "dyn_em/start_em.F": ["R5"],
    "dyn_em/module_first_rk_step_part1.F": ["W2"],
    "share/input_wrf.F": ["W2"],
    "phys/module_microphysics_driver.F": ["W2"],
    "phys/module_physics_addtendc.F": ["W2"],
    "phys/module_bl_boulac.F": ["P1"],
    "phys/module_bl_keps.F": ["P2"],
    "phys/module_bl_shinhong.F": ["P3"],
    "phys/physics_mmm/bl_shinhong.F90": ["P3"],
    "phys/physics_mmm/bl_gwdo.F90": ["P4"],
    "phys/module_cu_gd.F": ["P5"],
    "phys/module_diag_hailcast.F": ["P6"],
    "phys/module_bl_fogdes.F": ["P7"],
    "phys/module_mp_nssl_2mom.F": ["P8"],
    "phys/module_mp_ntu.F": ["P9"],
    "phys/module_mp_udm.F": ["P10"],
    "phys/module_mp_wdm7.F": ["P11"],
    "phys/module_sf_ruclsm.F": ["P12"],
    "phys/module_surface_driver.F": ["P13"],
    "phys/module_ra_rrtmg_lwf.F": ["P14"],
    "phys/module_fr_fire_core.F": ["P15"],
    "chem/module_chem_utilities.F": ["C4"],
    "chem/module_optical_averaging.F": ["C5"],
    "chem/module_ctrans_grell.F": ["C6"],
    "chem/emissions_driver.F": ["C7"],
    "chem/module_emissions_anthropogenics.F": ["C7"],
    "chem/module_vash_settling.F": ["C7"],
    "chem/module_volc_chem.F": ["C7"],
    "chem/dry_dep_driver.F": ["C7"],
    "chem/mechanism_driver.F": ["C7"],
    "wrftladj/solve_em_ad.F": ["R1"],
}

#: A WRF source path as the repository cites it.
CITED = re.compile(
    r"(?<![A-Za-z0-9_/.-])((?:phys|dyn_em|share|frame|chem|hydro|main|var|"
    r"external|wrftladj|inc)/[A-Za-z0-9_./+-]*?\.(?:F90|F|f90|inc|c|h)"
    r"(?![A-Za-z0-9_])|Registry/Registry\.[A-Za-z_]+)")
#: Not opened, not hashed: Noah-MP v5 (UCAR licence; the fork pins v4).
EXCLUDED = ("phys/noahmp/",)
SEARCHED = ("tools", "woof", "tests", "docs")


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def cited_paths(paths=SEARCHED) -> list[str]:
    """Every WRF source path the given repository paths cite (git grep)."""
    out = subprocess.run(
        ["git", "grep", "-h", "-o", "-I", "-P", CITED.pattern, "--", *paths,
         ":!tools/rustwx",
         ":!tools/wrf481_source"], cwd=ROOT, capture_output=True, text=True)
    return sorted({p for p in out.stdout.split()
                   if not p.startswith(EXCLUDED)})


def build(tree481: Path, tree480: Path) -> dict:
    files = {}
    for rel in cited_paths():
        new, old = _sha(tree481 / rel), _sha(tree480 / rel)
        if new is None and old is None:
            continue
        row = {"sha256": new, "sha256_v4.8.0": old}
        if new != old:
            row["changes"] = CHANGES.get(rel, ["unclassified"])
        files[rel] = row
    return {"schema": "woof.wrf-source-pin.v1", **PIN, "files": files}


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def lane(lane_dir: str) -> list[tuple[str, str]]:
    """``(path, status)`` for each WRF path ``lane_dir`` cites."""
    files = load()["files"]
    rows = []
    for rel in cited_paths((lane_dir,)):
        row = files.get(rel)
        if row is None:
            status = "not in the pin (cited elsewhere only, or absent)"
        elif row["sha256"] is None:
            status = "absent in 4.8.1"
        elif row["sha256"] == row["sha256_v4.8.0"]:
            status = "identical 4.8.0 = 4.8.1"
        else:
            status = "differs: " + ",".join(row["changes"])
        rows.append((rel, status))
    return rows


def check(tree: Path) -> list[str]:
    """Paths whose bytes in ``tree`` differ from the pin (empty: matches)."""
    return [rel for rel, row in load()["files"].items()
            if row["sha256"] is not None and _sha(tree / rel) != row["sha256"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tree481", type=Path, required=True)
    b.add_argument("--tree480", type=Path, required=True)
    b.add_argument("--out", type=Path, default=MANIFEST)
    ln = sub.add_parser("lane")
    ln.add_argument("lane_dir")
    c = sub.add_parser("check")
    c.add_argument("--tree", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        table = build(args.tree481, args.tree480)
        args.out.write_text(json.dumps(table, indent=1, sort_keys=False) + "\n",
                            encoding="utf-8", newline="\n")
        changed = sum("changes" in row for row in table["files"].values())
        print(f"{len(table['files'])} cited paths, {changed} differ 4.8.0 -> "
              f"4.8.1 -> {args.out}")
        return 0
    if args.cmd == "lane":
        for rel, status in lane(args.lane_dir):
            print(f"{rel}: {status}")
        return 0
    bad = check(args.tree)
    for rel in bad:
        print(f"MISMATCH {rel}")
    print(f"{len(load()['files']) - len(bad)} pinned paths match" if not bad
          else f"{len(bad)} pinned paths differ from v4.8.1")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
