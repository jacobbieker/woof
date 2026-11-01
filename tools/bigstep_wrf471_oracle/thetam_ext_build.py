"""Extract WRF v4.7.1's remaining use_theta_m branches, unchanged.

    thetam_ext_build.py WRF_SOURCE_ROOT OUT_DIR

Writes OUT_DIR/thetam_ext_exact.F90: module ``oracle_thetam_ext`` holding
the unmodified bodies of dyn_em/module_big_step_utilities_em.F's phy_prep
and phy_prep_part2 (whose ``use_theta_m == 1`` RTHFTEN block, :5323-5332,
woof answers with kernels/theta_m_cumulus.cu), with the file's own
VPOW/VPOWX preprocessor header, and vertical_diffusion_2's surface heat
flux -- the ``hflux: SELECT CASE`` block of dyn_em/module_diffusion_em.F
(:4286-4325), verbatim, in a wrapper declaring exactly the locals
vertical_diffusion_2 declares for it (heat_flux, cpm, the loop bounds it
sets at :4121-4124) -- which woof answers with kernels/smag2d_theta_m.cu.
module_model_constants.F and frame/libmassv.F compile whole and unmodified
(thetam_ext_build.sh).  Every source is pinned by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
SOURCES = {
    "dyn_em/module_big_step_utilities_em.F":
        "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
    "dyn_em/module_diffusion_em.F":
        "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "frame/libmassv.F":
        "449be0d252beba8c5a42ade42fceb2e582239d1a36db79ef3ad902f0ed82ede1",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}
BIG_STEP_ROUTINES = ("phy_prep", "phy_prep_part2")


def _pinned(root: Path, relative: str) -> str:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[relative]:
        raise SystemExit(f"{relative} differs from WRF v4.7.1 ({WRF_COMMIT})")
    return raw.decode()


def _routine(source: str, name: str, kind: str = "SUBROUTINE") -> str:
    match = re.search(
        rf"(?im)^[ \t]*{kind} {name}\s*\(.*?^[ \t]*END {kind} {name}[ \t]*$",
        source, re.S | re.M)
    if match is None:
        raise SystemExit(f"missing exact routine {name}")
    return match.group()


def _hflux_block(diffusion: str) -> str:
    """vertical_diffusion_2's ``hflux: SELECT CASE ... END SELECT hflux``."""
    body = _routine(diffusion, "vertical_diffusion_2")
    lines = body.splitlines()
    starts = [n for n, line in enumerate(lines)
              if line.strip().startswith("hflux: SELECT CASE")]
    ends = [n for n, line in enumerate(lines)
            if line.strip() == "END SELECT hflux"]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise SystemExit(f"expected one hflux block, got {starts} {ends}")
    block = "\n".join(lines[starts[0]:ends[0] + 1])
    if "config_flags%use_theta_m == 1" not in block:
        raise SystemExit("vertical_diffusion_2's hflux block has no theta_m branch")
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf_source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    for relative in SOURCES:
        _pinned(args.wrf_source, relative)
    big = _pinned(args.wrf_source, "dyn_em/module_big_step_utilities_em.F")
    diffusion = _pinned(args.wrf_source, "dyn_em/module_diffusion_em.F")
    header = big.split("MODULE module_big_step_utilities_em", 1)[0]
    if "# define VPOW  vspow" not in header:
        raise SystemExit("module_big_step_utilities_em.F header moved")
    routines = [_routine(big, name) for name in BIG_STEP_ROUTINES]
    block = _hflux_block(diffusion)
    wrapper = f"""
subroutine vd2_hflux ( config_flags, rt_tendf, moist, n_moist, theta, &
                       hfx, qfx, rho, dnw, &
                       ids, ide, jds, jde, kds, kde, &
                       ims, ime, jms, jme, kms, kme, &
                       its, ite, jts, jte, kts, kte )
   ! Wrapper declarations, as vertical_diffusion_2 declares them
   ! (module_diffusion_em.F:4004-4124); the statements are its hflux block.
   TYPE(grid_config_rec_type), INTENT(IN) :: config_flags
   INTEGER, INTENT(IN) :: n_moist
   INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: rt_tendf
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme, n_moist), INTENT(INOUT) :: moist
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) :: theta, rho
   REAL, DIMENSION(ims:ime, jms:jme), INTENT(INOUT) :: hfx, qfx
   REAL, DIMENSION(kms:kme), INTENT(IN) :: dnw
   INTEGER :: i, j
   INTEGER :: i_start, i_end, j_start, j_end
   REAL :: heat_flux
   REAL :: cpm
   i_start = its
   i_end   = MIN(ite,ide-1)
   j_start = jts
   j_end   = MIN(jte,jde-1)
{block}
end subroutine vd2_hflux
"""
    module = (header
              + "module oracle_thetam_ext\n"
              + "use module_model_constants\n"
              + "use module_state_description\n"
              + "use module_configure, only: grid_config_rec_type\n"
              + "use module_wrf_error\n"
              + "implicit none\n"
              + "contains\n"
              + "\n".join(routines) + "\n" + wrapper
              + "\nend module oracle_thetam_ext\n")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thetam_ext_exact.F90").write_text(module)


if __name__ == "__main__":
    main()
