"""Cut the two self-contained WRF routines the smoke oracle needs.

    python3 extract.py WRF_SOURCE_ROOT BUILD_DIR

``chem/module_aer_drydep.F`` and ``chem/module_gocart_settling.F`` cannot
be compiled whole outside WRF-Chem: their driver routines ``use``
module_configure, module_state_description and the MOSAIC / CAM-MAM /
GOCART data modules.  The two routines the oracle runs use none of them,
so each is cut verbatim -- every byte from its SUBROUTINE line to its END
SUBROUTINE line -- and wrapped in a module of its own:

    cut_zhang2001.F90        zhang2001_aer_drydepvel_1  (module_aer_drydep.F)
    cut_gocart_settling.F90  settling                   (module_gocart_settling.F)

The cut lines are located by their own text, which must occur exactly once,
and their sha256 is printed so a fixture can name the bytes it came from.
The source files themselves are pinned by SOURCES.sha256 (build.sh checks
them before this runs).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CUTS = (
    ("chem/module_aer_drydep.F", "smoke_cut_zhang2001", "cut_zhang2001.F90",
     "subroutine zhang2001_aer_drydepvel_1(",
     "end subroutine zhang2001_aer_drydepvel_1"),
    ("chem/module_gocart_settling.F", "smoke_cut_gocart_settling",
     "cut_gocart_settling.F90", "subroutine settling(",
     "END SUBROUTINE settling"),
)


def _one(lines: list[str], needle: str, path: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise SystemExit(f"{path}: {needle!r} found {len(hits)} times")
    return hits[0]


def main(source_root: str, build_dir: str) -> None:
    root, out = Path(source_root), Path(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path, module, target, first, last in CUTS:
        text = (root / path).read_text(encoding="latin-1")
        lines = text.splitlines(keepends=True)
        start, end = _one(lines, first, path), _one(lines, last, path)
        cut = "".join(lines[start:end + 1])
        (out / target).write_text(
            f"module {module}\nimplicit none\ncontains\n{cut}"
            f"end module {module}\n", encoding="latin-1")
        digest = hashlib.sha256(cut.encode("latin-1")).hexdigest()
        print(f"{target}: {path}:{start + 1}-{end + 1} sha256 {digest}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
