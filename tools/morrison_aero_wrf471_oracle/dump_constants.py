#!/usr/bin/env python3
"""Write a copy of WRF v4.7.1 module_mp_morr_two_moment_aero.F with one
appended module procedure that prints every MORR_TWO_MOMENT_INIT_AERO word
the aerosol-aware Morrison scheme (mp_physics = 40) reads, as exact hex
words.

Only a CONTAINS-procedure is appended before END MODULE (module procedures
can read the module's PRIVATE state) and one PUBLIC line names it; no
physics line changes.  The scalar words are the mp_physics = 10 list of
tools/morrison_constants/dump_constants.py (make_header.py checks that
every one but DCS equals mp_physics = 10's word); the arrays are the
prescribed-aerosol activation parameters the init derives (the _pamdm
family) and the aerosol property table.

usage: dump_constants.py PRISTINE.F OUTPUT.F
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "morrison_constants"))
from dump_constants import INTS, REALS  # noqa: E402

SCALARS = REALS + ["aten_pamdm", "alogaten_pamdm", "alog2_pamdm",
                   "alog3_pamdm", "surften_pamdm", "third_pamdm",
                   "sq2_pamdm", "real_tiny", "t0"]
ARRAYS = ["alogsig_pamdm", "exp45logsig_pamdm", "argfactor_pamdm",
          "amcube_pamdm", "smcrit_pamdm", "lnsm_pamdm",
          "amcubefactor_pamdm", "smcritfactor_pamdm", "f1_pamdm",
          "f2_pamdm", "dryrad_aer", "density_aer", "hygro_aer",
          "dispersion_aer", "num_to_mass_aer"]


def proc():
    lines = ["      subroutine arwen_dump_morr_aero_constants(mode)",
             "      implicit none", "      integer, intent(in):: mode",
             "      integer :: n, s"]
    for n in SCALARS:
        lines.append(f"      write(*,'(A,1X,I0,1X,I0,1X,Z8.8)') 'R {n}', mode, "
                     f"1, transfer({n}, 0)")
    for n in ARRAYS:
        lines.append("      do n = 1, naer_cu")
        lines.append(f"      write(*,'(A,1X,I0,1X,I0,1X,Z8.8)') 'R {n}', mode, "
                     f"n, transfer({n}(n), 0)")
        lines.append("      enddo")
    # ccnfact_pamdm(sat, mode): flattened sat-major, index (m-1)*7 + sat.
    lines += ["      do n = 1, naer_cu", "      do s = 1, psat_pamdm",
              "      write(*,'(A,1X,I0,1X,I0,1X,Z8.8)') 'R ccnfact_pamdm', "
              "mode, (n-1)*psat_pamdm + s, transfer(ccnfact_pamdm(s,n), 0)",
              "      enddo", "      enddo"]
    for n in INTS:
        lines.append(f"      write(*,'(A,1X,I0,1X,I0,1X,I0)') 'I {n}', mode, 1, "
                     f"{n}")
    lines += ["      end subroutine arwen_dump_morr_aero_constants", ""]
    return "\n".join(lines)


def main(argv):
    src = open(argv[1], encoding="latin-1").read()
    anchor = "END MODULE module_mp_morr_two_moment_aero"
    if src.count(anchor) != 1:
        raise SystemExit("END MODULE anchor not unique")
    out = src.replace(anchor, "CONTAINS_PLACEHOLDER" + anchor)
    out = out.replace("CONTAINS_PLACEHOLDER", proc())
    pub = "   PUBLIC  ::  POLYSVP\n"
    if out.count(pub) != 1:
        raise SystemExit("PUBLIC POLYSVP anchor not unique")
    out = out.replace(pub, pub + "   PUBLIC  ::  arwen_dump_morr_aero_constants\n")
    open(argv[2], "w", encoding="latin-1", newline="\n").write(out)


if __name__ == "__main__":
    main(sys.argv)
