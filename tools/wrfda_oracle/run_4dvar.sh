#!/bin/bash
# WRFDA 4.7.1 4D-Var lanes on the tutorial's 4D-Var case (2008-02-05
# 12-18Z, hourly ob01..ob07.ascii, var4d_bin = 3600 so 7 slots), inside the
# oracle container (docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.0).  The 4dvar
# build (build_4dvar.sh, default the -dump copy) runs into $W/run/NAME:
#
#   lid  enable_identity: WRFPLUS's TL and AD models are the identity.  The
#        transfers (xatowrftl, wrftltoxa and their adjoints) and the
#        per-slot H' are all that is left, the gate for woof's
#        transfer4d (4D.1) and FourDObsTerm (4D.2).
#   l1   adiabatic: no physics in the nonlinear, TL or AD model
#        (mp/cu/bl/ra/sfclay/surface 0, diff_opt 0).
#   l0   the tutorial's namelist as shipped (warm rain, PBL 98, CU 98).
#
# Every lane: one outer loop, hybrid_opt = 0 and use_theta_m = 0 (the
# testdata are V3-style files), trajectory_io with var4d_detail_out so the
# TL perturbations at each slot, the adjoint forcings and the gradient at
# 0h are written to disk, and files.sha256 over every input and output.
#
# Layout under $W (default /d/scratch/wrfda) as run_tutorial.sh.
# usage: run_4dvar.sh lid|l1|l0 NAME
set -eux
LANE=${1:?usage: run_4dvar.sh lid|l1|l0 NAME}
NAME=${2:?usage: run_4dvar.sh lid|l1|l0 NAME}
case "$NAME" in */*|.*) echo "NAME must be a plain directory name" >&2; exit 2;; esac
case "$LANE" in lid|l1|l0) ;; *) echo "LANE must be lid, l1 or l0" >&2; exit 2;; esac
W=${W:-/d/scratch/wrfda}; W=$(cd "$W" && pwd); D=$W/data; R=$W/run/$NAME
S=$W/src/${TREE:-WRFDAV4.7.1-4dvar-dump}
PLUS=$W/src/WRFPLUSV4.7.1
export HOME=/tmp
H="2008-02-05_"
NL=$S/var/test/4dvar/namelist.input
[ -s "$NL" ] || { echo "$NL is missing: the 4D-Var tutorial namelist ships with WRFDA" >&2; exit 2; }
[ -x "$S/var/build/da_wrfvar.exe" ] || { echo "build $S first (build_4dvar.sh)" >&2; exit 2; }
rm -rf "$R"; mkdir -p "$R"; cd "$R"
# WRFPLUS's run tables (the nonlinear model's physics reads them)
for f in "$PLUS"/run/*; do b=$(basename "$f")
  case $b in namelist.input|*.exe|wrfinput_d01|wrfbdy_d01|README*) ;; *) ln -sf "$f" "$b";; esac
done
ln -sf "$S/var/build/da_wrfvar.exe" .
ln -sf "$S/run/LANDUSE.TBL" .
cp "${BE:-$D/be/be.dat}" be.dat
cp "$D/rc/2008020512/wrfinput_d01" fg
cp "$D/rc/2008020512/wrfbdy_d01" wrfbdy_d01
for n in 1 2 3 4 5 6 7; do h=$((11 + n)); cp "$D/ob/20080205$h/ob0$n.ascii" "ob0$n.ascii"; done
python3 - "$NL" "$LANE" > namelist.input <<'PY'
import re, sys
src, lane = sys.argv[1], sys.argv[2]
H = "2008-02-05_"
want = {
    "wrfvar1": {"var4d": ".true.", "var4d_bin": "3600", "var4d_lbc": ".false.", "write_increments": ".true."},
    "wrfvar6": {"max_ext_its": "1"},
    "wrfvar18": {"analysis_date": f"'{H}12:00:00.0000'"},
    "wrfvar21": {"time_window_min": f"'{H}12:00:00.0000'"},
    "wrfvar22": {"time_window_max": f"'{H}18:00:00.0000'"},
    "time_control": {"run_hours": "6", "start_hour": "12", "end_hour": "18", "force_use_old_data": ".true."},
    "perturbation": {"trajectory_io": ".true.", "var4d_detail_out": ".true.",
                     "enable_identity": ".true." if lane == "lid" else ".false."},
    "dynamics": {"hybrid_opt": "0", "use_theta_m": "0"},
}
if lane == "l1":
    want["physics"] = {k: "0" for k in ("mp_physics", "cu_physics", "bl_pbl_physics", "ra_lw_physics",
                                        "ra_sw_physics", "sf_sfclay_physics", "sf_surface_physics",
                                        "mp_physics_ad", "cu_physics_ad", "bl_pbl_physics_ad")}
    want["dynamics"]["diff_opt"] = "0"
lines = open(src).read().splitlines()
out, group, seen = [], None, set()
for line in lines:
    s = line.strip()
    m = re.match(r"&(\w+)", s)
    if m:
        group = m.group(1).lower()
        out.append(line)
        continue
    if s == "/" and group is not None:
        for k, v in want.get(group, {}).items():
            if (group, k) not in seen:
                out.append(f"{k}={v},")
                seen.add((group, k))
        group = None
        out.append(line)
        continue
    m = re.match(r"\s*(\w+)\s*=", line)
    if group in want and m and m.group(1).lower() in want[group]:
        k = m.group(1).lower()
        out.append(f"{k}={want[group][k]},")
        seen.add((group, k))
        continue
    out.append(line)
missing = [f"&{g} {k}" for g, kv in want.items() for k in kv if (g, k) not in seen]
for g in want:  # a group the tutorial namelist lacks
    if missing and any(m.startswith(f"&{g} ") for m in missing):
        out.append(f"&{g}")
        out += [f"{k}={v}," for k, v in want[g].items() if (g, k) not in seen]
        out.append("/")
print("\n".join(out))
PY
grep -qi "^var4d=.true.," namelist.input && grep -qi "^use_theta_m=0," namelist.input \
  && grep -qi "^var4d_detail_out=.true.," namelist.input \
  || { echo "the 4D-Var settings did not reach namelist.input" >&2; exit 2; }
status=0
/usr/bin/time -v ./da_wrfvar.exe > wrfda.out 2> time.txt || status=$?
grep -E "Elapsed|Maximum resident" time.txt
grep -q "WRF-Var completed successfully" rsl.out.0000 wrfda.out 2>/dev/null \
  || { echo "da_wrfvar.exe did not complete (exit $status)" >&2; exit 1; }
grep -E "Final cost|Original gradient|Final: " wrfda.out | tail -6
ls -la
sha256sum namelist.input fg wrfbdy_d01 be.dat ob0?.ascii woof_*.bin wrfvar_output \
  $(ls -d tl_d01_* af_d01_* gradient_d01_* auxhist*_d01_* nl_d01_* 2>/dev/null) > files.sha256
