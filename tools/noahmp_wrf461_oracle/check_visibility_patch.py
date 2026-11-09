#!/usr/bin/env python3
"""Audit the Noah-MP leaf-visibility exposure patch.

The leaf oracles compile a *visibility-patched* copy of
``phys/module_sf_noahmplsm.F`` in a scratch directory.  This script is the
gate that makes that exposure auditable rather than a quiet source edit.  It
fails closed on anything that is not a ``private ::`` -> ``public ::``
accessibility-statement substitution.

Checks, all hard failures:

1. the patch file itself hashes to the pinned SHA-256;
2. the pristine source hashes to the SHA-256 already pinned in
   ``woof/core/noahmp_mynn_contract.py``;
3. the patched source hashes to the pinned patched SHA-256;
4. both files have the same number of lines;
5. every line that differs does so only by ``private`` -> ``public`` in an
   accessibility *statement*: same leading indentation, same text after
   ``::``, same symbol;
6. the set of lifted symbols is exactly the pinned 50-symbol list;
7. no ``PRIVATE`` entity *attribute* (``INTEGER, PRIVATE, PARAMETER ::``)
   was touched, so the module's own private constants stay private.

Usage:
    check_visibility_patch.py PRISTINE_SOURCE PATCHED_SOURCE PATCH_FILE
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


# patches/noahmp-lsm-leaf-visibility.patch.  One patch serves both pinned
# trees: its hunks (lines 27-87) are byte-identical in the two.
PATCH_SHA256 = "b16db586bbed70735b148618c794206ed0ff2988dfc8c9ee822b80added34a95"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from noahmp_pins import TREES  # noqa: E402
# The (pristine, patched) pair of each pinned tree.  PRISTINE_SHA256 and
# PATCHED_SHA256 name one pair (tests re-pin them to synthetic sources);
# ALTERNATE_PAIRS are the pinned trees, each accepted as a whole pair.
PRISTINE_SHA256 = TREES["wrf-4.7.1"]["lsm"]
PATCHED_SHA256 = TREES["wrf-4.7.1"]["patched"]
ALTERNATE_PAIRS = tuple((t["lsm"], t["patched"]) for t in TREES.values())

# module_sf_noahmplsm.F lines 30-84, in source order.
LIFTED_SYMBOLS = (
    "ATM", "PHENOLOGY", "PRECIP_HEAT", "ENERGY", "THERMOPROP", "CSNOW",
    "TDFCND", "RADIATION", "ALBEDO", "SNOW_AGE", "SNOWALB_BATS",
    "SNOWALB_CLASS", "GROUNDALB", "TWOSTREAM", "SURRAD", "VEGE_FLUX",
    "SFCDIF1", "SFCDIF2", "STOMATA", "CANRES", "ESAT", "RAGRB", "BARE_FLUX",
    "TSNOSOI", "HRT", "HSTEP", "ROSR12", "PHASECHANGE", "FRH2O", "WATER",
    "CANWATER", "SNOWWATER", "SNOWFALL", "COMBINE", "DIVIDE", "COMBO",
    "COMPACT", "SNOWH2O", "SOILWATER", "ZWTEQ", "INFIL", "SRT", "WDFCND1",
    "WDFCND2", "SSTEP", "GROUNDWATER", "SHALLOWWATERTABLE", "CARBON",
    "CO2FLUX", "ERROR",
)

_PRIVATE_STMT = re.compile(r"^( *)private( *::.*)$")
_PUBLIC_STMT = re.compile(r"^( *)public( *::.*)$")
_SYMBOL = re.compile(r"^ *:: *([A-Za-z_][A-Za-z0-9_]*) *$")
_PRIVATE_ATTR = re.compile(r",\s*PRIVATE\s*,", re.IGNORECASE)


class VisibilityPatchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(pristine: Path, patched: Path, patch: Path) -> tuple[str, ...]:
    digest = _sha256(patch)
    if digest != PATCH_SHA256:
        raise VisibilityPatchError(
            f"patch file {patch} has sha256 {digest}, expected {PATCH_SHA256}")
    pair = (_sha256(pristine), _sha256(patched))
    accepted = ((PRISTINE_SHA256, PATCHED_SHA256),) + ALTERNATE_PAIRS
    if pair not in accepted:
        raise VisibilityPatchError(
            f"pristine {pristine} / patched {patched} have sha256 "
            f"{pair[0]} / {pair[1]}, expected one of {accepted}")

    before = pristine.read_bytes().decode("utf-8", errors="strict").split("\n")
    after = patched.read_bytes().decode("utf-8", errors="strict").split("\n")
    if len(before) != len(after):
        raise VisibilityPatchError(
            f"line count changed: {len(before)} -> {len(after)}")

    lifted: list[str] = []
    for number, (old, new) in enumerate(zip(before, after), start=1):
        if old == new:
            if _PRIVATE_STMT.match(old) is not None:
                raise VisibilityPatchError(
                    f"line {number}: private statement survived the patch")
            continue
        old_match = _PRIVATE_STMT.match(old)
        new_match = _PUBLIC_STMT.match(new)
        if old_match is None or new_match is None:
            raise VisibilityPatchError(
                f"line {number} is not a visibility substitution:\n"
                f"  - {old!r}\n  + {new!r}")
        if old_match.group(1) != new_match.group(1):
            raise VisibilityPatchError(
                f"line {number}: indentation changed")
        if old_match.group(2) != new_match.group(2):
            raise VisibilityPatchError(
                f"line {number}: text after the keyword changed:\n"
                f"  - {old_match.group(2)!r}\n  + {new_match.group(2)!r}")
        symbol = _SYMBOL.match(old_match.group(2))
        if symbol is None:
            raise VisibilityPatchError(
                f"line {number}: cannot read a single symbol from "
                f"{old_match.group(2)!r}")
        lifted.append(symbol.group(1))

    if tuple(lifted) != LIFTED_SYMBOLS:
        extra = sorted(set(lifted) - set(LIFTED_SYMBOLS))
        missing = sorted(set(LIFTED_SYMBOLS) - set(lifted))
        raise VisibilityPatchError(
            "lifted symbol list does not match the pinned set "
            f"(extra={extra}, missing={missing}, "
            f"count={len(lifted)} vs {len(LIFTED_SYMBOLS)})")

    attr_before = [n for n, line in enumerate(before, 1)
                   if _PRIVATE_ATTR.search(line)]
    attr_after = [n for n, line in enumerate(after, 1)
                  if _PRIVATE_ATTR.search(line)]
    if attr_before != attr_after or not attr_before:
        raise VisibilityPatchError(
            "PRIVATE entity attributes were altered: "
            f"{attr_before} -> {attr_after}")

    return tuple(lifted)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    pristine, patched, patch = (Path(a) for a in argv[1:])
    try:
        lifted = audit(pristine, patched, patch)
    except VisibilityPatchError as error:
        print(f"visibility patch audit FAILED: {error}", file=sys.stderr)
        return 1
    print(f"visibility patch audit ok: {len(lifted)} accessibility statements "
          f"lifted, PRIVATE entity attributes untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
