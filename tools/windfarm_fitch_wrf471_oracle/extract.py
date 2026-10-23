"""Cut WRF v4.7.1's Fitch wind-farm routines out of module_wind_fitch.F verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

``phys/module_wind_fitch.F`` USEs module_llxy, module_dm and
module_configure for its initialisation routine (the windturbines.txt and
wind-turbine-N.tbl readers, map projections and MPI broadcasts), none of
which the drag itself touches.  This copies, after checking the sha256 of
the whole file and of each block, the parts the port transcribes into one
small Fortran module that compiles alone:

* module_wind_fitch.F:52-63, IMPLICIT NONE and the module's turbine state
  (nt, nkind, ival/jval, hubheight, diameter, stc, stc2, cutin, cutout,
  npower, the turbws/turbtc/turbpw tables, correction_factor), which the
  driver fills where init_module_wind_fitch would;
* module_wind_fitch.F:67-334, dragforce, turbine_area and dragcof;
* share/module_model_constants.F:72, the piconst parameter.

WRF v4.7.1 is tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186,
public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

#: (path, sha256 of the file, ((name, first, last, sha256 of the lines), ...))
FILES = (
    ("phys/module_wind_fitch.F",
     "83247add74780608e798bb62424533b18abf31f3da5aba6374de6fe2550060c3",
     (("state", 52, 63,
       "401873fdd71683c5714f47da410f792d377549d422e5d47d64443474e8d9f625"),
      ("routines", 67, 334,
       "7ef78ecae9bbdb12f3d6f358181e527df91140852f33edf8b12d510e0d205550"))),
    ("share/module_model_constants.F",
     "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
     (("piconst", 72, 72,
       "ae9c4d296b8b78ed27cede626cc52cf059fa45fcacc50f650ed680fe8388f247"),)),
)

_MODULE = """MODULE wrf_wind_fitch_inline
{state}{piconst}
CONTAINS

{routines}
END MODULE wrf_wind_fitch_inline
"""


def _block(lines: list[bytes], path: str, name: str, first: int, last: int,
           digest: str) -> str:
    text = b"\n".join(lines[first - 1:last]) + b"\n"
    got = hashlib.sha256(text).hexdigest()
    if got != digest:
        raise SystemExit(f"{path}:{first}-{last} ({name}) is not the pinned "
                         f"block: sha256 {got}, expected {digest}")
    return text.decode("ascii")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__.splitlines()[2].strip())
    root, out = Path(args[0]), Path(args[1])
    blocks = {}
    for path, digest, cuts in FILES:
        raw = (root / path).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != digest:
            raise SystemExit(f"{path} is not WRF v4.7.1 ({WRF_COMMIT}): "
                             f"sha256 {got}, expected {digest}")
        lines = raw.split(b"\n")
        for name, first, last, sha in cuts:
            blocks[name] = _block(lines, path, name, first, last, sha)
    out.mkdir(parents=True, exist_ok=True)
    header = (f"! Extracted verbatim from WRF v4.7.1 ({WRF_COMMIT}) by "
              "tools/windfarm_fitch_wrf471_oracle/extract.py.\n"
              "! Do not edit: regenerate.\n")
    (out / "wrf_wind_fitch_inline.F90").write_text(
        header + _MODULE.format(**blocks), encoding="ascii", newline="\n")
    print(f"extracted {len(blocks)} blocks into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
