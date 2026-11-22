#!/usr/bin/env bash
# Build and run the gridded-morphology (URB_PARAM / NUDAPT) oracle against the
# byte-unmodified WRF v4.7.1 phys/module_sf_urban.F.
#
#   bash tools/urban_wrf471_oracle/build_ucm_nudapt.sh WRF_SRC_DIR BUILD_DIR [OUT_DIR]
#
# WRF_SRC_DIR is the flat pinned copy build_ucm.sh takes (SOURCES.sha256 is
# checked before anything is built).  Two builds:
#   ref   -O0                          THE FIXTURE
#   snan  -O0 -finit-real=snan ...     negative control: ref must equal snan
#                                      byte for byte on every main-variant
#                                      row.  The NUDAPT arm's local BW is
#                                      written before it is read on every
#                                      reachable path; the `elseif (SW.lt.0.0
#                                      .or.BW.lt.0.0)` branch (:653-655) is
#                                      dead (its `if` fails only when its
#                                      `else` would compute the same thing),
#                                      and this proves no row reads BW unset.
#
# The column driver stores U1/V1 into U10/V10 before each call (see the
# driver header): WRF reads ATAN2(U10,V10) at :705 from an INTENT(OUT)
# argument the caller left holding the previous column's 10 m wind.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build_ucm_nudapt.sh WRF_SRC_DIR BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
out=$(realpath -m "${3:-$2/fixture}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

files=(module_sf_urban.F module_model_constants.F module_wrf_error.F
       URBPARM.TBL URBPARM_LCZ.TBL)
for f in "${files[@]}"; do
    want=$(awk -v n="$f" '$2 == n {print $1}' "$src/SOURCES.sha256")
    got=$(sha256sum "$src/$f" | cut -d' ' -f1)
    if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$f does not match SOURCES.sha256 (want '$want', got '$got')" >&2
        exit 3
    fi
done

mkdir -p "$build" "$out"
cd "$build"
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"

build_one() {   # $1 = tag, rest = extra flags
    local tag=$1; shift
    mkdir -p "$tag"
    ( cd "$tag"
      gfortran -c -O0 "$@" ${free} "$here/stub_ucm.F90"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} "$src/module_model_constants.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch "$src/module_wrf_error.F"
      gfortran -c -O0 "$@" -cpp ${defines} ${free} -fallow-argument-mismatch -I. "$src/module_sf_urban.F"
      gfortran -c -O0 "$@" ${free} -I. "$here/ucm_nudapt_oracle.F90"
      gfortran -o run_ucm_nudapt stub_ucm.o module_model_constants.o module_wrf_error.o \
          module_sf_urban.o ucm_nudapt_oracle.o
      cp "$src/URBPARM.TBL" "$src/URBPARM_LCZ.TBL" . )
}
build_one ref
build_one snan -finit-real=snan -finit-integer=-2147483647 -finit-logical=false

nm -u ref/module_sf_urban.o | sed 's/^ *//' | sort > undefined-O0-urban.txt
if grep -q '_ZGV' undefined-O0-urban.txt; then
    echo "-O0 module_sf_urban.o pulled in a libmvec vector symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 module_sf_urban.o libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|log10f|atanf|atan2f|sqrtf|sinf|cosf|tanf|asinf|acosf|__powisf2|_ZGV' \
        undefined-O0-urban.txt || true
} > "$out/libm-report.txt"

for tag in ref snan; do
    ( cd "$build/$tag"
      ./run_ucm_nudapt column nlcd main nudapt-nlcd-main
      ./run_ucm_nudapt column lcz main nudapt-lcz-main
      ./run_ucm_nudapt column nlcd probe nudapt-nlcd-probe
      for opt in 1 2 3; do
          ./run_ucm_nudapt init "$opt" 0 init-opt$opt-lcz0
          ./run_ucm_nudapt init "$opt" 1 init-opt$opt-lcz1
      done )
done
for name in nudapt-nlcd-main nudapt-lcz-main \
            init-opt1-lcz0 init-opt1-lcz1 init-opt2-lcz0 init-opt2-lcz1 \
            init-opt3-lcz0 init-opt3-lcz1; do
    if ! cmp -s "ref/$name.csv" "snan/$name.csv"; then
        echo "$name: -finit-real=snan changes the output; an uninitialised read reaches a row" >&2
        exit 6
    fi
done
for name in nudapt-nlcd-main nudapt-lcz-main nudapt-nlcd-probe; do
    cp "ref/$name.csv" "ref/$name-table.csv" "ref/$name-switches.csv" "$out/"
    gzip -n -9 -f "$out/$name.csv"
done
for opt in 1 2 3; do
    cp ref/init-opt$opt-lcz0.csv ref/init-opt$opt-lcz1.csv "$out/"
done
nprobe=$(paste -d'\n' "ref/nudapt-nlcd-probe.csv" "snan/nudapt-nlcd-probe.csv" | awk 'NR%2{a=$0;next}{if(a!=$0)n++}END{print n+0}')
echo "nudapt-nlcd-probe: rows differing ref vs snan-init: $nprobe" >> "$out/libm-report.txt"
( cd "$out" && sha256sum *.csv *.csv.gz > oracle-sha256sums.txt )
gfortran --version | head -1 > "$out/compiler.txt"
ldd --version | head -1 >> "$out/compiler.txt"
sha256sum "$here/ucm_nudapt_oracle.F90" "$here/build_ucm_nudapt.sh" "$here/stub_ucm.F90" \
    | sed "s#$here/##" > "$out/driver-sha256sums.txt"
echo "nudapt oracle written to $out"
