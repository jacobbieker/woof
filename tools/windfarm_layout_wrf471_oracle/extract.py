#!/usr/bin/env python3
"""Cut the wind-farm statements the oracle runs out of module_wind_fitch.F.

``module_wind_fitch.F`` cannot be compiled outside WRF (it USEs
module_configure and module_dm), so the statements the layout must
reproduce are cut out of the pinned, byte-unmodified file into include
files and compiled inside the driver's own scaffolding.  Every cut is a
contiguous line range whose first and last lines are checked against the
text below, so a different file cannot be cut silently:

``wf_decls.inc``   54-63    module data: MAXVALS, nt, NKIND, ival/jval, the
                            per-turbine header arrays, turbws/turbtc/turbpw
``wf_dragcof.inc`` 277-334  SUBROUTINE dragcof (thrust, power, TKE coeffs)
``wf_place.inc``   399-477  one turbine of init_module_wind_fitch: read
                            lat, lon, type; map_set from XLAT/XLONG(its,jts);
                            latlon_to_ij; nint
``wf_tables.inc``  491-510  init_module_wind_fitch's table read
``wf_power1.inc``  190-191  dragforce's swept area and power1

    python3 extract.py WRF_SOURCE_ROOT BUILD_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

CUTS = {
    "wf_decls.inc": (54, 63, "INTEGER, PARAMETER :: MAXVALS  = 100",
                     "REAL :: correction_factor"),
    "wf_dragcof.inc": (277, 334, "SUBROUTINE dragcof(",
                       "END SUBROUTINE dragcof"),
    "wf_place.inc": (399, 477, "read(70,*)lat,lon,nkind(k)", "endif"),
    "wf_tables.inc": (491, 510, "turbws=0.", "132   continue"),
    "wf_power1.inc": (190, 191, "area=piconst/4.*diameter(kt)**2.",
                      "power1=0.5*1.23*speedhub**3.*area*powcof"),
}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source = Path(argv[1]) / "phys" / "module_wind_fitch.F"
    out = Path(argv[2])
    lines = source.read_text(encoding="ascii").split("\n")
    for name, (first, last, head, tail) in CUTS.items():
        block = lines[first - 1:last]
        if head not in block[0] or tail not in block[-1]:
            raise SystemExit(f"{source}:{first}-{last} is not the expected "
                             f"cut for {name}: {block[0]!r} .. {block[-1]!r}")
        (out / name).write_text("\n".join(block) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
