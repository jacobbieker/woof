#!/bin/bash
# Build the WRFDA 4.7.1 4D-Var oracle (docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.0)
# inside wrf-oracle:u2404: WRFPLUS 4.7.1, then WRFDA 4.7.1 configured with
# "4dvar" against it.  Both are GNU serial (configure option 17), so the
# runs are deterministic, and both promote reals to 8 bytes as WRFDA does.
#
# Steps, each idempotent on a fresh tree:
#   src      unpack the pinned tarball (checked by SHA-256) into
#            $W/src/WRFPLUSV4.7.1 and $W/src/WRFDAV4.7.1-4dvar
#   wrfplus  ./configure wrfplus; ./compile wrfplus  (wrf.exe is WRFPLUS)
#   wrfda    ./configure 4dvar with WRFPLUS_DIR; ./compile all_wrfvar
#   dump     copy the 4dvar tree to WRFDAV4.7.1-4dvar-dump, hook
#            woof_dump.inc in (apply_dump_patch.sh) and build it
# The pinned trees are never patched; only the -dump copy is.
# usage: build_4dvar.sh src|wrfplus|wrfda|dump
#   J (default 6): make jobs; RESUME=1 (wrfplus): keep the configured tree
#   and its objects, to finish a compile a killed compiler stopped
set -eux
STEP=${1:?usage: build_4dvar.sh src|wrfplus|wrfda|dump}
export HOME=/tmp NETCDF=/usr; JOBS=${J:-6}; unset J   # compile reads J as "-j N"
W=${W:-/d/scratch/wrfda}; W=$(cd "$W" && pwd); S=$W/src
TAR=$S/v4.7.1.tar.gz
TAR_SHA=11186188b033d26332e31769c1f7aff9349406920a7f72eeb256d5881f7223f4
PLUS=$S/WRFPLUSV4.7.1; DA=$S/WRFDAV4.7.1-4dvar
OPT=17   # GNU (gfortran/gcc), serial
here=$(cd "$(dirname "$0")" && pwd)
case "$STEP" in
src)
  echo "$TAR_SHA  $TAR" | sha256sum -c -
  for t in "$PLUS" "$DA"; do
    [ -e "$t" ] && { echo "$t exists; remove it to unpack again" >&2; exit 2; }
    x=$(mktemp -d "$S/.x.XXXXXX"); tar xzf "$TAR" -C "$x"; mv "$x/WRFV4.7.1" "$t"; rmdir "$x"
  done
  ;;
wrfplus)
  cd "$PLUS"
  if [ "${RESUME:-0}" = 1 ] && [ -s configure.wrf ]; then
    echo "RESUME=1: keeping the configured tree and its objects (a retry after a killed compile)"
  else
    ./clean -a >/dev/null 2>&1 || true
    printf "%s\n" "$OPT" | ./configure wrfplus
  fi
  grep -q -- "-fdefault-real-8" configure.wrf || { echo "WRFPLUS is not built with 8-byte reals" >&2; exit 1; }
  rm -f main/wrf.exe main/libwrflib.a
  /usr/bin/time -v ./compile -j "$JOBS" wrfplus > "$W/build-wrfplus.log" 2>&1 || true
  if grep -q "Killed signal terminated program" "$W/build-wrfplus.log"; then
    echo "a compiler was killed (out of memory?): retry with RESUME=1 and a smaller J" >&2
  fi
  [ -x main/wrf.exe ] && [ -s main/libwrflib.a ] || { echo "WRFPLUS did not build: see $W/build-wrfplus.log" >&2; exit 1; }
  ls -la main/wrf.exe main/libwrflib.a
  ;;
wrfda|dump)
  [ -s "$PLUS/main/libwrflib.a" ] || { echo "build WRFPLUS first (build_4dvar.sh wrfplus)" >&2; exit 2; }
  export WRFPLUS_DIR=$PLUS
  if [ "$STEP" = dump ]; then
    [ -x "$DA/var/build/da_wrfvar.exe" ] || { echo "build the pinned 4dvar tree first (build_4dvar.sh wrfda)" >&2; exit 2; }
    rm -rf "$DA-dump"; cp -a "$DA" "$DA-dump"; cd "$DA-dump"
    bash "$here/apply_dump_patch.sh" "$DA-dump"
    LOG=$W/build-WRFDAV4.7.1-4dvar-dump.log
  else
    cd "$DA"
    ./clean -a >/dev/null 2>&1 || true
    printf "%s\n" "$OPT" | ./configure 4dvar
    grep -q "VAR4D" configure.wrf || { echo "configure.wrf has no VAR4D: not a 4dvar build" >&2; exit 1; }
    LOG=$W/build-WRFDAV4.7.1-4dvar.log
  fi
  rm -f var/build/da_wrfvar.exe
  /usr/bin/time -v ./compile all_wrfvar > "$LOG" 2>&1 || true
  [ -x var/build/da_wrfvar.exe ] || { echo "da_wrfvar.exe was not linked: see $LOG" >&2; exit 1; }
  ls -la var/build/da_wrfvar.exe
  ;;
*) echo "STEP must be src, wrfplus, wrfda or dump" >&2; exit 2;;
esac
