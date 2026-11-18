#!/usr/bin/env bash
# Build the RRTMG-K column oracle against UNMODIFIED WRF 4.7.1 text.
#
# One arm, BUILD_DIR/stock/rad_k_column_oracle: -O0, no FP contraction (the
# 0 ULP referee), compiled whole from phys/module_ra_effective_radius.F,
# phys/module_ra_rrtmg_lwk.F and phys/module_ra_rrtmg_swk.F with
# -DBUILD_RRTMK=1 (the configure switch that turns RRTMG-K on), plus the
# legacy column oracle's verbatim driver extracts (radconst, calc_coszen,
# cal_cldfra1, the CAM ozone chain: tools/rrtmg_legacy_column_oracle/
# build_units.py).
#
# cpp state mirrors WRF's group build for this bundle: EM_CORE=1,
# WRF_CHEM=0, HWRF=0, RWORDSIZE=4 (kind_rb = FP32), -fconvert=big-endian
# (BYTESWAPIO; the RRTMG_*_DATA lookup reads); the oracle's own input and
# output files pin little-endian per unit.
#
# usage: build.sh WRF471_SOURCE_ROOT BUILD_DIR
set -euo pipefail
src=$(realpath "$1"); out=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
legacy="$here/../rrtmg_legacy_column_oracle"
PY=${PYTHON:-python3}
COMMON="-cpp -ffree-form -ffree-line-length-none -fconvert=big-endian \
  -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 -DBUILD_RRTMK=1 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
OPT="-O0 -ffp-contract=off"
FFLAGS="$OPT $COMMON"
d="$out/stock"; mkdir -p "$d"; cd "$d"
"$PY" "$legacy/build_units.py" stock "$src" "$out/fork-src" "$d"
gfortran -c $FFLAGS "$here/stub_wrf_k.F90"
gfortran -c $FFLAGS -I "$d" "$src/phys/module_ra_effective_radius.F"
gfortran -c $FFLAGS -I "$d" "$src/phys/module_ra_rrtmg_lwk.F"
gfortran -c $FFLAGS -I "$d" "$src/phys/module_ra_rrtmg_swk.F"
gfortran -c $FFLAGS -I "$d" raddrv_extract.F90
gfortran -c $FFLAGS -I "$d" "$here/rad_k_column_oracle.F90"
gfortran -o rad_k_column_oracle stub_wrf_k.o module_ra_effective_radius.o \
  module_ra_rrtmg_lwk.o module_ra_rrtmg_swk.o raddrv_extract.o rad_k_column_oracle.o
cp -f "$src/run/RRTMG_LW_DATA" "$src/run/RRTMG_SW_DATA" \
      "$src/run/ozone.formatted" "$src/run/ozone_lat.formatted" \
      "$src/run/ozone_plev.formatted" .
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
fma() { if command -v objdump > /dev/null; then objdump -d rad_k_column_oracle | grep -c -E 'vfn?m(add|sub)|fmadd|fmsub' || true; else echo "n/a"; fi; }
{ gfortran --version | head -1; uname -sm; echo "flags: $FFLAGS";
  sha "$here"/*.F90 "$here"/*.py "$here"/build.sh "$legacy"/build_units.py \
    "$src"/phys/module_ra_effective_radius.F "$src"/phys/module_ra_rrtmg_lwk.F \
    "$src"/phys/module_ra_rrtmg_swk.F RRTMG_LW_DATA RRTMG_SW_DATA \
    ozone.formatted ozone_lat.formatted ozone_plev.formatted raddrv_extract.F90;
  echo "fused multiply-add instructions in the executable: $(fma)";
} > build-receipt.txt
echo "built $d/rad_k_column_oracle"
