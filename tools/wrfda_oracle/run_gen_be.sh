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
export NL_CV_OPTIONS=${1:?usage: run_gen_be.sh CV_OPTIONS} BIN_TYPE=5
export START_DATE=2008020400 END_DATE=2008020812
export NUM_LEVELS=40 BE_METHOD=NMC
export FC_DIR=$W/genbe/gen_be_fc
export RUN_DIR=$W/genbe/run/cv${NL_CV_OPTIONS}
export DOMAIN=01 FCST_RANGE1=24 FCST_RANGE2=12 INTERVAL=12 STRIDE=1 NUM_JOBS=${NUM_JOBS:-8}
export USE_RFi=true
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
ksh "$SCRIPTS_DIR/gen_be/gen_be.ksh" > "$RUN_DIR/../gen_be.cv${NL_CV_OPTIONS}.log" 2>&1
[ -s "$RUN_DIR/be.dat" ] || { echo "gen_be did not write $RUN_DIR/be.dat" >&2; exit 1; }
sha256sum "$RUN_DIR/be.dat"
