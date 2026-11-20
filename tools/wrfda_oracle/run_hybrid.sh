#!/bin/bash
# WRFDA 4.7.1 hybrid 3DEnVar on the 3D-Var tutorial case (2008-02-05 12Z),
# inside the oracle container (the plan's V3.2 lane).  The ensemble is NE of
# the WRFDA gen_be sample forecasts ($W/genbe/gen_be_fc, the tutorial's
# domain; see run_gen_be.sh), taken as members in sorted order: not a real
# ensemble, but the same members for WRFDA and woof, which is what the
# comparison needs.  gen_be_ep2.exe (from link_gen_be.sh) writes the ep files
# (ep_format 1), then da_wrfvar.exe runs with
#   &wrfvar7  je_factor=JE
#   &wrfvar16 ensdim_alpha=NE, alpha_corr_scale=SCALE, alpha_vertloc_opt=2
# and writes be.vertloc.dat beside its log.
# Layout under $W (default /d/scratch/wrfda) as run_tutorial.sh.
# usage: [NE=10] [JE=2.0] [SCALE=200.0] run_hybrid.sh NAME
set -eux
NAME=${1:?usage: run_hybrid.sh NAME}
case "$NAME" in */*|.*) echo "NAME must be a plain directory name" >&2; exit 2;; esac
NE=${NE:-10}; JE=${JE:-2.0}; SCALE=${SCALE:-200.0}
case "$NE$JE$SCALE" in *[!0-9.]*) echo "NE, JE and SCALE must be numbers" >&2; exit 2;; esac
W=${W:-/d/scratch/wrfda}; W=$(cd "$W" && pwd); S=$W/src/${TREE:-WRFV4.7.1}; D=$W/data; R=$W/run/$NAME
rm -rf "$R"; mkdir -p "$R/members" "$R/ep"; cd "$R"
n=0
for f in $(ls "$W"/genbe/gen_be_fc/*/wrfout_d01_* | sort | head -n "$NE"); do
  n=$((n + 1)); ln -sf "$f" "members/wrfout.e$(printf %03d $n)"
done
[ "$n" -eq "$NE" ] || { echo "only $n forecasts for NE=$NE" >&2; exit 1; }
ls -l members > members.txt
(cd ep && "$W/genbe/bin/gen_be_ep2.exe" 2008020512 "$NE" "$R/members" wrfout > ../gen_be_ep2.log 2>&1)
rm -f ep/tmp.e*
ln -sf "$S/var/build/da_wrfvar.exe" .
ln -sf "$S/run/LANDUSE.TBL" .
cp "${BE:-$D/be/be.dat}" be.dat
cp "$D/ob/2008020512/ob.ascii" ob.ascii
cp "$D/rc/2008020512/wrfinput_d01" fg
NL=$S/var/test/tutorial/namelist.input
grep -q "^&wrfvar7" "$NL" && grep -q "^&wrfvar16" "$NL" && grep -q "^start_year=2008," "$NL" \
  || { echo "$NL lacks &wrfvar7, &wrfvar16 or start_year=2008," >&2; exit 2; }
sed "s/^&wrfvar7\$/\&wrfvar7\nje_factor=${JE},/" "$NL" \
  | sed "s/^&wrfvar16\$/\&wrfvar16\nensdim_alpha=${NE},\nalpha_corr_scale=${SCALE},\nalpha_vertloc_opt=2,/" \
  | sed "s/^start_year=2008,/force_use_old_data=.true.,\nstart_year=2008,/" > namelist.input
grep -q "^ensdim_alpha=${NE}," namelist.input && grep -q "^je_factor=${JE}," namelist.input \
  || { echo "the hybrid settings did not reach namelist.input" >&2; exit 2; }
status=0
/usr/bin/time -v ./da_wrfvar.exe > wrfda.out 2> time.txt || status=$?
grep -E "Elapsed|Maximum resident" time.txt
grep -q "WRF-Var completed successfully" rsl.out.0000 wrfda.out 2>/dev/null \
  || { echo "da_wrfvar.exe did not complete (exit $status)" >&2; exit 1; }
grep -E "jb_factor, je_factor|minimize_cg|Final cost" wrfda.out | tail -5
