#!/usr/bin/env python3
"""Derive the WRF_HYDRO WATER harness from run_water.F90, mechanically.

Under -DWRF_HYDRO, WATER (module_sf_noahmplsm.F :5954-6261) takes two more
INOUT arguments, sfcheadrt (the ponded head WRF-Hydro hands the LSM, mm) and
WATBLED (the water-table depth for Hooghoudt tile drainage, unread with
opt_tdrn = 0).  This script adds them to run_water.F90's state, emits them
with the other inputs, gives each case a ponded head derived from its name
(several cases get none), and passes them in the WATER call.  Nothing else
in the harness changes; the script refuses to run if any anchor is missing
or appears twice.
"""

import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
edits = [
    ("  end type wstate\n",
     "     real    :: sfcheadrt, watbled   ! WRF_HYDRO\n  end type wstate\n"),
    ("  end subroutine emit_state\n",
     "    call emit_r (cname, stage, 'sfcheadrt', 0, s%sfcheadrt)\n"
     "    call emit_r (cname, stage, 'watbled', 0, s%watbled)\n"
     "  end subroutine emit_state\n"),
    ("    s = s_in\n",
     "    s = s_in\n"
     "    ! WRF_HYDRO: a ponded head per case (mm), zero for some\n"
     "    s%sfcheadrt = 0.25 * real(mod(len_trim(cname) * 37 + iachar(cname(1:1)), 23))\n"
     "    s%watbled = 0.0\n"),
    ("s%qevac, s%qdewc, s%acc_qinsur, s%acc_qseva, s%acc_etrani)",
     "s%qevac, s%qdewc, s%acc_qinsur, s%acc_qseva, s%acc_etrani, &\n"
     "                s%sfcheadrt, s%watbled)"),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"anchor not unique in {src}: {old!r}")
    s = s.replace(old, new)
open(out, "w").write(s)
