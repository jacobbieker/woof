#!/bin/bash
# Build WRFDA 4.7.1 3D-Var (serial GNU, configure option 32) inside the
# oracle container.  TREE=WRFV4.7.1 for the pinned build (clean + configure),
# TREE=WRFV4.7.1-dump INCREMENTAL=1 for the dump copy after apply_dump_patch.sh.
set -ex
export HOME=/tmp NETCDF=/usr J=${J:-6}
W=${W:-/d/scratch/wrfda}
cd "$W/src/${TREE:-WRFV4.7.1}"
if [ -z "$INCREMENTAL" ]; then
  ./clean -a >/dev/null 2>&1 || true
  printf "32\n" | ./configure wrfda
fi
# WRF's compile exits 0 on failure, and an incremental build would leave the
# previous executable behind: remove it first so only a fresh link passes.
rm -f var/build/da_wrfvar.exe
LOG="$W/build-${TREE:-WRFV4.7.1}.log"
/usr/bin/time -v ./compile all_wrfvar > "$LOG" 2>&1 || true
# (Some gen_be, da_rad_diags and da_update_bc links fail in this
# configuration; make ignores them and da_wrfvar.exe does not need them.)
[ -x var/build/da_wrfvar.exe ] || { echo "da_wrfvar.exe was not linked: see $LOG" >&2; exit 1; }
ls -la var/build/da_wrfvar.exe
