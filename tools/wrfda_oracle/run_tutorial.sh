#!/bin/bash
# Run the WRFDA 3D-Var tutorial case (2008-02-05 12Z, 90x60x41, 60 km) inside
# the oracle container.  Layout under $W (default /d/scratch/wrfda):
#   src/<TREE>/   a built WRFV4.7.1 (TREE defaults to WRFV4.7.1; the dump
#                 copy made by apply_dump_patch.sh is WRFV4.7.1-dump)
#   data/         the unpacked WRFDAV3.9-testdata tarball
# DUMP=1 additionally requires woof_be.bin, woof_xb.bin, woof_u.bin and
#        woof_iv.bin in the run directory (use with the patched WRFV4.7.1-dump).
# BE=FILE uses FILE (a path inside the container) instead of the tutorial be.dat.
# usage: [DUMP=1] [BE=FILE] run_tutorial.sh NAME [extra wrfvar10 settings, e.g. test_transforms=true,]
set -eux
NAME=${1:?usage: run_tutorial.sh NAME [extra wrfvar10 settings]}
EXTRA=${2:-}
case "$NAME" in */*|.*) echo "NAME must be a plain directory name" >&2; exit 2;; esac
case "$EXTRA" in */*|*\&*|*\\*) echo "extra settings must not contain / & or \\" >&2; exit 2;; esac
W=${W:-/d/scratch/wrfda}; W=$(cd "$W" && pwd); S=$W/src/${TREE:-WRFV4.7.1}; D=$W/data; R=$W/run/$NAME
rm -rf "$R"; mkdir -p "$R"; cd "$R"
ln -sf "$S/var/build/da_wrfvar.exe" .
ln -sf "$S/run/LANDUSE.TBL" .
cp "${BE:-$D/be/be.dat}" be.dat
cp "$D/ob/2008020512/ob.ascii" ob.ascii
cp "$D/rc/2008020512/wrfinput_d01" fg
NL=$S/var/test/tutorial/namelist.input
grep -q "^test_transforms=false," "$NL" || { echo "$NL has no test_transforms=false, line" >&2; exit 2; }
grep -q "^start_year=2008," "$NL" || { echo "$NL has no start_year=2008, line" >&2; exit 2; }
sed "s/^test_transforms=false,/test_transforms=false,${EXTRA}/" "$NL" \
  | sed "s/^start_year=2008,/force_use_old_data=.true.,\nstart_year=2008,/" > namelist.input
status=0
/usr/bin/time -v ./da_wrfvar.exe > wrfda.out 2> time.txt || status=$?
grep -E "Elapsed|Maximum resident" time.txt
# da_wrfvar.exe can exit 0 after wrf_error_fatal; its own success line decides.
grep -q "WRF-Var completed successfully" rsl.out.0000 wrfda.out 2>/dev/null \
  || { echo "da_wrfvar.exe did not complete (exit $status)" >&2; exit 1; }
if [ -n "${DUMP:-}" ]; then
  for f in woof_be.bin woof_xb.bin woof_u.bin woof_iv.bin; do
    [ -s "$R/$f" ] || { echo "DUMP=1 but $R/$f is missing or empty" >&2; exit 1; }
  done
fi
