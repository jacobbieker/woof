"""Extract WRF v4.7.1's use_theta_m branches, unchanged, for the theta_m oracle.

    thetam_build.py WRF_SOURCE_ROOT OUT_DIR

Writes OUT_DIR/thetam_exact.F90: module ``oracle_thetam`` holding the
unmodified bodies of dyn_em/module_big_step_utilities_em.F's
calc_p_rho_phi, phy_prep, moist_physics_prep_em, moist_physics_finish_em,
initialize_moist_old and conv_t_tendf_to_moist (with the file's own
VPOW/VPOWX preprocessor header), and real.exe's theta -> theta_m block of
dyn_em/module_initialize_real.F (the IF block at :4923-4931, verbatim, in a
wrapper whose ``grid`` carries only ``t_2``) with that file's
skip_middle_points_t.  module_avgflx_em.F, module_model_constants.F and
frame/libmassv.F compile whole and unmodified (thetam_build.sh).  Every
source is pinned by SHA-256 (wrf471_sha256 below).
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
    "dyn_em/module_initialize_real.F":
        "f79b37ec3e15c3d179873089bb92d2d2f6b804efc86032f0b4e47bdb14e12d9e",
    "dyn_em/module_avgflx_em.F":
        "224d293174f1a90f335fcbba69c461683d3aca7fe5061edc185f442fb2ced3e1",
    "frame/libmassv.F":
        "449be0d252beba8c5a42ade42fceb2e582239d1a36db79ef3ad902f0ed82ede1",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}
BIG_STEP_ROUTINES = ("calc_p_rho_phi", "phy_prep", "moist_physics_prep_em",
                     "moist_physics_finish_em", "initialize_moist_old",
                     "conv_t_tendf_to_moist")
REAL_BLOCK_START = ("IF ( ( config_flags%use_theta_m .EQ. 1 ) .AND. "
                    "(P_Qv .GE. PARAM_FIRST_SCALAR) ) THEN")


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


def _real_block(source: str) -> str:
    """The theta -> theta_m IF block that ends init_domain_rk, verbatim."""
    lines = source.splitlines()
    starts = [n for n, line in enumerate(lines)
              if line.strip() == REAL_BLOCK_START]
    # The block real.exe runs before its halos: the one preceded by the
    # "Turn dry potential temperature into moist potential temperature"
    # comment (the file has exactly one).
    starts = [n for n in starts if any(
        "Turn dry potential temperature into moist" in lines[m]
        for m in range(max(0, n - 6), n))]
    if len(starts) != 1:
        raise SystemExit(f"expected one real.exe theta_m block, got {starts}")
    start = starts[0]
    depth = 0
    for n in range(start, len(lines)):
        stripped = lines[n].strip().upper()
        if stripped.startswith("IF ") and stripped.endswith("THEN"):
            depth += 1
        elif stripped in ("END IF", "ENDIF"):
            depth -= 1
            if depth == 0:
                return "\n".join(lines[start:n + 1])
    raise SystemExit("unterminated real.exe theta_m block")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf_source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    for relative in SOURCES:
        _pinned(args.wrf_source, relative)
    big = _pinned(args.wrf_source, "dyn_em/module_big_step_utilities_em.F")
    real = _pinned(args.wrf_source, "dyn_em/module_initialize_real.F")
    header = big.split("MODULE module_big_step_utilities_em", 1)[0]
    if "# define VPOW  vspow" not in header:
        raise SystemExit("module_big_step_utilities_em.F header moved")
    routines = [_routine(big, name) for name in BIG_STEP_ROUTINES]
    skip = _routine(real, "skip_middle_points_t", "function")
    block = _real_block(real)
    wrapper = f"""
subroutine real_theta_m ( config_flags, grid, moist, n_moist, &
                          ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte )
   ! Wrapper declarations; the statements are module_initialize_real.F's.
   TYPE(grid_config_rec_type), INTENT(IN) :: config_flags
   TYPE(oracle_grid), INTENT(INOUT) :: grid
   INTEGER, INTENT(IN) :: n_moist
   INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                          ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte
   REAL, INTENT(IN) :: moist(ims:ime, kms:kme, jms:jme, n_moist)
   INTEGER :: i, j, k
   INTEGER, PARAMETER :: em_width = 5
   LOGICAL, PARAMETER :: hold_ups = .false.
{block}
end subroutine real_theta_m
"""
    module = (header
              + "module oracle_thetam\n"
              + "use module_model_constants\n"
              + "use module_state_description\n"
              + "use module_configure, only: grid_config_rec_type\n"
              + "use module_wrf_error\n"
              + "implicit none\n"
              + "type oracle_grid\n  real, allocatable :: t_2(:,:,:)\n"
              + "end type oracle_grid\n"
              + "contains\n"
              + "\n".join(routines) + "\n" + wrapper + "\n" + skip
              + "\nend module oracle_thetam\n")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thetam_exact.F90").write_text(module)


if __name__ == "__main__":
    main()
