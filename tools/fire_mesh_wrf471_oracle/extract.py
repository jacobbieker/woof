"""Cut WRF v4.7.1's fire-mesh slope and spacing statements out verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

module_fr_fire_util.F compiles whole (build.sh), and its interpolate_z2fire
refines the terrain.  Two other pieces sit inside driver routines that
cannot be compiled alone:

* module_fr_fire_model.F:194-200, the continue_at_boundary call that
  continues ZSF one node past the fire mesh before the gradients, and the
  centred differences of :207-208, ``(fp%zsf(i+1,j)-fp%zsf(i-1,j))/(2.*fdx)``
  and its y twin.  WRF v4.7.1 reads DZDXF/DZDYF from geo_em and evaluates
  these expressions to check them.  They are its only statement of the
  fire-mesh slope, so the oracle evaluates them as the slope itself;
* module_fr_fire_driver.F:566-567, the fire-mesh spacing ``dxf=dx/ir``.

This copies them into one small module, after checking the sha256 of each
whole file and of each cut, so the oracle compiles WRF's own statements and
nothing else.  The wrapper's dummy arguments carry WRF's names (fp%zsf,
fdx, ...), so the statements paste unmodified.

WRF v4.7.1 is tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186,
public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MODEL = ("phys/module_fr_fire_model.F",
         "f2e3932e088076508a8c7284dc5395770b93b8eb5ba5cccbb1a56775dddd184b")
DRIVER = ("phys/module_fr_fire_driver.F",
          "7662f29bb003697d08ccee585e272a3ed7362edc73bcef5ea8dc95d100d9e578")

#: (file, first line, last line, sha256 of the lines, LF-joined).
CONTINUE = (MODEL, 194, 200,
            "6381093e57447483321468cef009efd6e823adddcef42d8901afa1c9d446c76d")
GRADIENT = (MODEL, 207, 208,
            "767d7d59bb70471ec96106d6661c6fd3f657762a9ba5f55e228c1e2ad6aea205")
SPACING = (DRIVER, 566, 567,
           "d1185c27e2cc805e9bbed5d849b74dddae6453f128d65cc2cec10bb95e25c328")

#: The left-hand sides the gradient lines check against; the oracle takes
#: the right-hand side, the centred difference, as the slope.
_CHECKED = ("erri = fp%dzdxf(i,j) - ", "errj = fp%dzdyf(i,j) - ")

_MODULE = """MODULE wrf_fire_slope_cut
  USE module_fr_fire_util, ONLY: continue_at_boundary
  IMPLICIT NONE
  ! fp%zsf under the name module_fr_fire_model.F reads it by.
  TYPE fire_params_t
    REAL, POINTER :: zsf(:,:) => NULL()
  END TYPE fire_params_t
CONTAINS
  SUBROUTINE fire_slope(ifms,ifme,jfms,jfme, ifds,ifde,jfds,jfde, &
                        ifps,ifpe,jfps,jfpe, ifts,ifte,jfts,jfte, &
                        fdx, fdy, fp, dzdxf, dzdyf)
    INTEGER, INTENT(IN) :: ifms,ifme,jfms,jfme, ifds,ifde,jfds,jfde
    INTEGER, INTENT(IN) :: ifps,ifpe,jfps,jfpe, ifts,ifte,jfts,jfte
    REAL, INTENT(IN) :: fdx, fdy
    TYPE(fire_params_t), INTENT(INOUT) :: fp
    REAL, INTENT(OUT) :: dzdxf(ifms:ifme,jfms:jfme), dzdyf(ifms:ifme,jfms:jfme)
    INTEGER :: i, j, itso, iteo, jtso, jteo
    dzdxf = 0.
    dzdyf = 0.
! ---- module_fr_fire_model.F:194-200 (verbatim) ----
@CONTINUE@
! ---- module_fr_fire_model.F:207-208, right-hand sides (verbatim) ----
        do j=jfts,jfte
            do i=ifts,ifte
                dzdxf(i,j) = @DZDX@
                dzdyf(i,j) = @DZDY@
            enddo
        enddo
  END SUBROUTINE fire_slope

  SUBROUTINE fire_spacing(dx, dy, ir, jr, dxf, dyf)
    REAL, INTENT(IN) :: dx, dy
    INTEGER, INTENT(IN) :: ir, jr
    REAL, INTENT(OUT) :: dxf, dyf
! ---- module_fr_fire_driver.F:566-567 (verbatim) ----
@SPACING@
  END SUBROUTINE fire_spacing
END MODULE wrf_fire_slope_cut
"""


def _file(root: Path, spec) -> list[str]:
    rel, digest = spec
    data = (root / rel).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != digest:
        sys.exit(f"{rel}: sha256 {got} != pinned {digest}")
    return data.decode("utf-8").split("\n")


def _cut(root: Path, block) -> list[str]:
    spec, first, last, digest = block
    lines = _file(root, spec)[first - 1:last]
    got = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    if got != digest:
        sys.exit(f"{spec[0]}:{first}-{last}: sha256 {got} != pinned {digest}")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    root, out = Path(argv[1]), Path(argv[2])
    gradient = _cut(root, GRADIENT)
    rhs = []
    for line, checked in zip(gradient, _CHECKED):
        head, sep, tail = line.strip().partition(checked)
        if head or not sep:
            sys.exit(f"unexpected gradient line: {line!r}")
        rhs.append(tail)
    text = (_MODULE.replace("@CONTINUE@", "\n".join(_cut(root, CONTINUE)))
            .replace("@DZDX@", rhs[0]).replace("@DZDY@", rhs[1])
            .replace("@SPACING@", "\n".join(_cut(root, SPACING))))
    out.mkdir(parents=True, exist_ok=True)
    (out / "wrf_fire_slope_cut.F90").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
