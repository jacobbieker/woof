"""Write a link stub for WRF's camuwshcu_driver (shcu_physics=2).

phys/module_shallowcu_driver.F USEs module_shcu_camuwshcu_driver, whose real
source pulls in the CAM support chain.  The Deng oracle never selects
shcu_physics=2, so the stub only has to give the driver the same explicit
interface: the subroutine statement and the dummy-argument declarations,
sliced VERBATIM out of the real file (header through the closing paren, then
the dummy declarations up to the first local REAL(r8) array), and no body.

    python gen_camuw_stub.py WRF_SOURCE_ROOT OUT.F
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(root: str, out: str) -> None:
    lines = Path(root, "phys", "module_shcu_camuwshcu_driver.F").read_text(
        encoding="utf-8").splitlines()
    start = next(i for i, s in enumerate(lines)
                 if s.strip().upper().startswith("SUBROUTINE CAMUWSHCU_DRIVER("))
    header_end = next(i for i in range(start, len(lines))
                      if lines[i].strip().startswith(")"))
    decl_start = next(i for i in range(header_end, len(lines))
                      if "is_CAMMGMP_used" in lines[i]
                      and "LOGICAL" in lines[i].upper())
    decl_end = next(i for i in range(decl_start, len(lines))
                    if lines[i].strip().upper().startswith(
                        "REAL(R8), DIMENSION(PCOLS, KTE, PCNST)"))
    body = (["MODULE module_shcu_camuwshcu_driver", "  IMPLICIT NONE",
             "  PRIVATE", "  PUBLIC :: camuwshcu_driver", "CONTAINS"]
            + lines[start:header_end + 1] + ["  IMPLICIT NONE"]
            + lines[decl_start:decl_end]
            + ["END SUBROUTINE camuwshcu_driver",
               "END MODULE module_shcu_camuwshcu_driver", ""])
    Path(out).write_text("\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
