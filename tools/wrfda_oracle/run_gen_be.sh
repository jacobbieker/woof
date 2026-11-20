#!/bin/bash
# Run WRFDA 4.7.1's gen_be.ksh (NMC 24 h - 12 h, bin_type 5) on the WRFDA
# gen_be sample forecasts (gen_be_forecasts_200802.tar.gz, unpacked as
# $W/genbe/gen_be_fc), with the executables of link_gen_be.sh.  gen_be.ksh
# needs ksh: wrf-oracle:u2404 plus `apt-get install ksh` (wrf-oracle:u2404-ksh).
# usage: run_gen_be.sh CV_OPTIONS   (5 or 7) -> $W/genbe/run/cvN/be.dat
set -e
export HOME=/tmp
W=${W:-/d/scratch/wrfda}
export WRFVAR_DIR=$W/src/${TREE:-WRFV4.7.1}
export SCRIPTS_DIR=$WRFVAR_DIR/var/scripts
export BUILD_DIR=$W/genbe/bin
export RUN_GEN_BE_STAGE0=true RUN_GEN_BE_STAGE1=true RUN_GEN_BE_STAGE2=true RUN_GEN_BE_STAGE2A=true
export RUN_GEN_BE_STAGE3=true RUN_GEN_BE_STAGE4=true RUN_GEN_BE_DIAGS=true
case "${1:-}" in 5|7) ;; *) echo "usage: run_gen_be.sh CV_OPTIONS (5 or 7)" >&2; exit 2;; esac
export NL_CV_OPTIONS=$1 BIN_TYPE=5 CLEAN=false
export START_DATE=2008020400 END_DATE=2008020812
export NUM_LEVELS=40 BE_METHOD=NMC
export FC_DIR=$W/genbe/gen_be_fc
export RUN_DIR=$W/genbe/run/cv${NL_CV_OPTIONS}
export DOMAIN=01 FCST_RANGE1=24 FCST_RANGE2=12 INTERVAL=12 STRIDE=1 NUM_JOBS=${NUM_JOBS:-8}
export USE_RFi=true
# The forecast tarball must be the one the manifest pins (FC_TARBALL = the
# downloaded gen_be_forecasts_200802.tar.gz; skipped with a note if unset).
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")/../.." && pwd)/tests/data/wrfda471/manifest.json}
if [ -n "${FC_TARBALL:-}" ]; then
  [ -f "$MANIFEST" ] || { echo "manifest $MANIFEST not found" >&2; exit 1; }
  [ -f "$FC_TARBALL" ] || { echo "FC_TARBALL $FC_TARBALL not found" >&2; exit 1; }
  want=$(grep -A3 '"forecasts"' "$MANIFEST" | sed -n 's/.*"sha256": *"\([0-9a-f]*\)".*/\1/p' | head -1)
  got=$(sha256sum "$FC_TARBALL" | cut -d' ' -f1)
  [ "${#want}" -eq 64 ] || { echo "no valid forecast sha256 in $MANIFEST" >&2; exit 1; }
  [ "$want" = "$got" ] || { echo "forecast tarball SHA $got != manifest $want" >&2; exit 1; }
else
  echo "run_gen_be.sh: FC_TARBALL not set; forecast tarball SHA not verified" >&2
fi
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/../gen_be.cv${NL_CV_OPTIONS}.log"
rc=0
ksh "$SCRIPTS_DIR/gen_be/gen_be.ksh" > "$LOG" 2>&1 || rc=$?
# gen_be.ksh does not stop on a failed stage: check the exit status, the
# stage logs and logs' own error text, then each stage's expected output.
[ "$rc" -eq 0 ] || { echo "gen_be.ksh exited $rc (see $LOG)" >&2; exit 1; }
if grep -l -i -E "^ *(error|fatal)|forrtl|segmentation fault|Program received signal|STOP [1-9]|Abnormal termination|No such file" \
     "$RUN_DIR"/gen_be_*.log "$LOG" 2>/dev/null | grep -q .; then
  echo "gen_be stage failure; offending logs:" >&2
  grep -l -i -E "^ *(error|fatal)|forrtl|segmentation fault|Program received signal|STOP [1-9]|Abnormal termination|No such file" \
    "$RUN_DIR"/gen_be_*.log "$LOG" >&2
  exit 1
fi
need="gen_be_stage1.log gen_be_stage4_regional.log gen_be_diags.log"
[ "${NL_CV_OPTIONS}" = 5 ] && need="$need gen_be_stage2.log gen_be_stage2a.log gen_be_stage3.psi.log working/gen_be_stage2.dat"
for f in $need; do
  [ -s "$RUN_DIR/$f" ] || { echo "missing or empty stage output $RUN_DIR/$f" >&2; exit 1; }
done
# stage 4 runs its workers in the background and gen_be.ksh ignores their status: read their .out files
if find "$RUN_DIR/working/gen_be_stage4_regional.1" -name 'gen_be_stage4_regional_*.out' 2>/dev/null \
     | xargs -r grep -l -i -E "^ *(error|fatal)|forrtl|segmentation fault|Program received signal|STOP [1-9]|Abnormal termination|No such file" | grep -q .; then
  echo "a stage-4 worker reported an error (see $RUN_DIR/working/gen_be_stage4_regional.1/*/gen_be_stage4_regional_*.out)" >&2
  exit 1
fi
nsl=$(find "$RUN_DIR/working/gen_be_stage4_regional.1" -name 'sl_print.*' 2>/dev/null | wc -l)
[ "$nsl" -eq $((4 * NUM_LEVELS + 1)) ] || { echo "stage 4 wrote $nsl sl_print files, expected $((4 * NUM_LEVELS + 1))" >&2; exit 1; }
[ -s "$RUN_DIR/be.dat" ] || { echo "gen_be did not write $RUN_DIR/be.dat" >&2; exit 1; }
sha256sum "$RUN_DIR/be.dat"
