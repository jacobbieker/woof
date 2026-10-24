"""Cut WRF v4.7.1's wind-farm input reads out of module_wind_fitch.F verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

``init_module_wind_fitch`` (module_wind_fitch.F:341-525) reads the turbine
list and the turbine tables with list-directed READs, then places lat/lon
turbines with module_llxy and broadcasts with module_dm.  The placement is
held to WRF by ``tools/windfarm_layout_wrf471_oracle``; what this oracle
pins is the READS, which is where a decimal in a file becomes a REAL word:

* :369-375, the record count (``nt``);
* :393, the ``windfarm_ij = 1`` row read, ``ival, jval, nkind``;
* :399, the ``windfarm_ij = 0`` row read, ``lat, lon, nkind``;
* :491-510, the ``wind-turbine-N.tbl`` reads into turbws/turbtc/turbpw,
  hubheight, diameter, stc, npower, and cutin/cutout/stc2.

Each block's sha256 is checked, and the whole file's, before it is written
out as an INCLUDE file the harness (run_reader.F90) places where WRF has
it.  WRF v4.7.1 is tag v4.7.1, commit
f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
PATH = "phys/module_wind_fitch.F"
DIGEST = "83247add74780608e798bb62424533b18abf31f3da5aba6374de6fe2550060c3"
#: (include file, first line, last line, sha256 of the lines)
BLOCKS = (
    ("cut_count.inc", 369, 375,
     "cc8c41e1be9fde5fdbb263539f84a5fe1d2de63081ff5f570a3625cdefb059a8"),
    ("cut_read_ij.inc", 393, 393,
     "dfded8e85350a49fedf885b82af0208a4455bfec95f3565c6e8295f72d1bb37a"),
    ("cut_read_ll.inc", 399, 399,
     "7d91d1707092eb0b48190185c4ed576be38d8be2f506057e5eb088c448de065d"),
    ("cut_tables.inc", 491, 510,
     "ae6bf7973c85ac2d3fdf80dcade14d3151d9c9bb3c229d221861faccfa45e55f"),
)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__.splitlines()[2].strip())
    root, out = Path(args[0]), Path(args[1])
    raw = (root / PATH).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != DIGEST:
        raise SystemExit(f"{PATH} is not WRF v4.7.1 ({WRF_COMMIT}): sha256 "
                         f"{got}, expected {DIGEST}")
    lines = raw.split(b"\n")
    out.mkdir(parents=True, exist_ok=True)
    for name, first, last, digest in BLOCKS:
        text = b"\n".join(lines[first - 1:last]) + b"\n"
        sha = hashlib.sha256(text).hexdigest()
        if sha != digest:
            raise SystemExit(f"{PATH}:{first}-{last} is not the pinned block: "
                             f"sha256 {sha}, expected {digest}")
        (out / name).write_bytes(text)
    print(f"extracted {len(BLOCKS)} blocks into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
