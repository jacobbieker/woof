#!/usr/bin/env bash
# Build the aerosol-aware Morrison (mp_physics = 40) oracle from
# byte-unmodified WRF v4.7.1, dump MORR_TWO_MOMENT_INIT_AERO's words and run
# the column sets aero_columns.py wrote.  CPU only.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR SETS_DIR
#
# Reference of record: gfortran 13.3 (Ubuntu 24.04) -O0 against glibc
# 2.39's libm, the toolchain of the mp_physics = 10 oracle
# (tools/morrison_wrf461_oracle, whose libm words morrison.cu carries).
# module_mp_morr_two_moment_aero.F calls expf, logf, powf and sqrtf (and
# log10f in POLYSVP's Goff-Gratch branches); the script refuses any other
# libm.  On pb (glibc 2.44) the record was made as
#
#   docker build -t morraero-oracle:u2404 - <<'EOF'
#   FROM ubuntu:24.04
#   RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install \
#       -y --no-install-recommends gfortran python3 python3-numpy binutils
#   EOF
#   docker run --rm -u $(id -u):$(id -g) -v $WRF:/wrf:ro -v $PWD:$PWD \
#       -w $PWD morraero-oracle:u2404 \
#       bash tools/morrison_aero_wrf471_oracle/build.sh /wrf BUILD SETS
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR SETS_DIR" >&2; exit 2; }
src=$(realpath "$1")
mkdir -p "$2"
build=$(realpath "$2")
sets=$(realpath "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
oracle10="$here/../morrison_wrf461_oracle"

# WRF v4.7.1 (f52c197ed39d12e087d02c50f412d90d418f6186).  The aero module
# is also byte-identical in v4.6.1.
sha256sum -c - <<EOF
d512aa16ad862b04c7278540a207a56aa10955e052bd9b9bc870e9a4f0812180  $src/phys/module_mp_morr_two_moment_aero.F
ca069a12cc149313456fa8ee08a827e0f542768b66ff6c1ece95a40ab208fb88  $src/phys/module_mp_radar.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  $src/share/module_model_constants.F
0cc7b2d71d8a93ff1f2b401324de8fd1ee70ba598f83d89254d6dbe8c31d0e4b  $src/phys/ccpp_kind_types.F
EOF
glibc=$(ldd --version | sed -n 1p)
case "$glibc" in
    *" 2.39"*) ;;
    *) echo "libm is not glibc 2.39 ($glibc); see the header" >&2; exit 4 ;;
esac

cd "$build"
f="-O0 -cpp -ffree-form -ffree-line-length-none"
gfortran -c $f -DRWORDSIZE=4 "$src/phys/ccpp_kind_types.F"
gfortran -c $f "$oracle10/stub_wrf_error.F90"
gfortran -c $f "$src/share/module_model_constants.F"
gfortran -c $f -I "$build" "$src/phys/module_mp_radar.F"
gfortran -c $f -DWRF_CHEM=0 -fallow-argument-mismatch -I "$build" \
    -fdump-tree-original \
    -o module_mp_morr_two_moment_aero.o \
    "$src/phys/module_mp_morr_two_moment_aero.F"
nm -u module_mp_morr_two_moment_aero.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q _ZGV undefined-O0.txt; then echo "libmvec symbol at -O0" >&2; exit 5; fi

# MORR_TWO_MOMENT_INIT_AERO's words, from a second object of the same
# source with the dump procedure appended (morrison_aero_constants.cuh).
python3 "$here/dump_constants.py" "$src/phys/module_mp_morr_two_moment_aero.F" \
    module_mp_morr_two_moment_aero_dump.F
# (gfortran reads .mod files from the working directory first, so the
# dump copy builds in its own directory.)
mkdir -p dump
(cd dump
 for m in ccpp_kind_types module_model_constants module_mp_radar \
          module_wrf_error; do cp "../$m.mod" .; done
 gfortran -c $f -DWRF_CHEM=0 -fallow-argument-mismatch \
     -o morr_aero_dump.o ../module_mp_morr_two_moment_aero_dump.F
 gfortran -c $f -o run_dump.o "$here/run_dump.F90"
 gfortran -o ../run_dump ../stub_wrf_error.o ../ccpp_kind_types.o \
     ../module_model_constants.o ../module_mp_radar.o morr_aero_dump.o \
     run_dump.o)
./run_dump | grep -E '^ ?[RDI] ' > constants-aero.txt

gfortran -c $f -I "$build" "$here/run_morrison_aero.F90"
gfortran -o run_morrison_aero stub_wrf_error.o ccpp_kind_types.o \
    module_model_constants.o module_mp_radar.o \
    module_mp_morr_two_moment_aero.o run_morrison_aero.o

python3 - "$sets/sets.json" > steps.txt <<'PY'
import json, sys
for name, info in json.load(open(sys.argv[1]))["sets"].items():
    print(name, info["steps"])
PY
while read -r name steps; do
    ./run_morrison_aero "$sets/$name.in.bin" "$sets/$name.wrf.bin" "$steps"
done < steps.txt

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $glibc"
    echo "# libm symbols of the -O0 object:"
    grep -E 'expf|powf|logf|log10f|sqrtf|erff' undefined-O0.txt || true
} > receipt.txt
sha256sum "$src/phys/module_mp_morr_two_moment_aero.F" \
    "$src/phys/module_mp_radar.F" "$src/share/module_model_constants.F" \
    "$here/run_morrison_aero.F90" "$sets"/*.in.bin "$sets"/*.wrf.bin \
    constants-aero.txt >> receipt.txt
cat receipt.txt
