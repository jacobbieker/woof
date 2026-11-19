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
/usr/bin/time -v ./compile all_wrfvar > "$W/build-${TREE:-WRFV4.7.1}.log" 2>&1
ls -la var/build/da_wrfvar.exe
