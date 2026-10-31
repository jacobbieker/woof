#!/usr/bin/env bash
# Build the Deng shallow cumulus (shcu_physics = 5) oracle against the
# byte-unmodified WRF v4.7.1 phys/module_shcu_deng.F and
# phys/module_shallowcu_driver.F, and run it on input files written by
# deng_columns.py.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin [INPUT.bin ...]
#
# gfortran is whatever is first on PATH (the receipts were made with
# `pixi exec -s gfortran`, conda-forge gfortran 16.2.0) and the C++ shim is
# built with the host's c++.  Two builds of the same unmodified sources:
#
#   O0    the reference: -O0 -ffp-contract=off, linked against libm_shim.o
#         (WOOF's transcription of glibc 2.39 expf/logf/powf).  This is the
#         0 ULP oracle WOOF is graded against.
#   snan  the same with every local REAL a signalling NaN and every local
#         INTEGER -2147483647 at entry (-finit-*).  An output that differs
#         between O0 and snan read memory the Fortran never wrote; the
#         receipt lists any such word and the gate never counts it.
#
# The pins are the files: every WRF source must hash to the value below.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin [...]" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd); shift
mkdir -p "$1"
build_dir=$(cd "$1" && pwd); shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kernels_dir=$(cd "${script_dir}/../../woof/core/kernels" && pwd)

sha() { shasum -a 256 "$1" | awk '{print $1}'; }
pin() {   # relative path, sha256
    local actual
    actual=$(sha "${source_root}/$1")
    if [[ "${actual}" != "$2" ]]; then
        echo "$1 is ${actual}, not the pinned $2" >&2
        exit 3
    fi
}
# WRF v4.7.1 (commit f52c197ed), LF line endings.
pin phys/module_shcu_deng.F         d03fd43559e1562bde415eb7a50906b993c6da3f5dc3e501f957b48031629645
pin phys/module_shallowcu_driver.F  068cf22b8c93cfac74a6223a72d61f4a2f890bc44e212da5d2c2a131c5ae271a
pin phys/module_shcu_grims.F        2b56ceb8d8c806179bb1e60b29174fd62ca8c4149fc487ecafa613ff937b131d
pin phys/module_shcu_nscv.F         d244722bce55d85a519edd305d78d575ecc8bd1309780cd4ed9c1c4b323067c7
pin share/module_model_constants.F  5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062

cd "${build_dir}"
python3 "${script_dir}/gen_camuw_stub.py" "${source_root}" camuw_stub.F
# WRF's EM build defines; -fconvert/-frecord-marker only touch unformatted
# record I/O and the driver streams little endian on purpose.
cppflags="-cpp -DEM_CORE=1 -DNMM_CORE=0 -DWRF_CHEM=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
common="-ffree-form -ffree-line-length-none -ffp-contract=off"

c++ -std=c++17 -O0 -ffp-contract=off -fno-fast-math -fno-builtin \
    -I "${kernels_dir}" -c "${script_dir}/libm_shim.cpp" -o libm_shim.o

build_one () {   # name, flags...
    local name=$1; shift
    mkdir -p "${name}"
    ( cd "${name}"
      gfortran -c "$@" ${common} ${cppflags} -o stub_wrf.o "${script_dir}/stub_wrf.F90"
      gfortran -c "$@" ${common} ${cppflags} -o module_model_constants.o \
          "${source_root}/share/module_model_constants.F"
      gfortran -c "$@" ${common} ${cppflags} -o camuw_stub.o ../camuw_stub.F
      gfortran -c "$@" ${common} ${cppflags} -o module_shcu_grims.o \
          "${source_root}/phys/module_shcu_grims.F"
      gfortran -c "$@" ${common} ${cppflags} -o module_shcu_nscv.o \
          "${source_root}/phys/module_shcu_nscv.F"
      gfortran -c "$@" ${common} ${cppflags} -o module_shcu_deng.o \
          "${source_root}/phys/module_shcu_deng.F"
      gfortran -c "$@" ${common} ${cppflags} -o module_shallowcu_driver.o \
          "${source_root}/phys/module_shallowcu_driver.F"
      gfortran -c -O0 ${common} -o run_deng_oracle.o "${script_dir}/run_deng_oracle.F90"
      gfortran -o run_deng_oracle run_deng_oracle.o module_shallowcu_driver.o \
          module_shcu_deng.o module_shcu_nscv.o module_shcu_grims.o camuw_stub.o \
          module_model_constants.o stub_wrf.o ../libm_shim.o
      nm -u module_shcu_deng.o module_shallowcu_driver.o 2>/dev/null \
          | sed 's/^ *//' | grep -v ':$' | sort -u > undefined.txt )
}

build_one O0 -O0
build_one snan -O0 -finit-real=snan -finit-integer=-2147483647 -finit-logical=false

# The float32 libm surface of the WRF objects must be exactly what the shim
# defines, and the executable must bind those calls to the shim.
for b in O0 snan; do
    calls=$(grep -E '^_?(exp|log|pow|sqrt|sin|cos|tan|atan|tgamma|lgamma|cbrt|expm1|log1p|log10|exp2|erf)f?$' \
        "${b}/undefined.txt" | sed 's/^_//' | sort -u | tr '\n' ' ')
    for f in ${calls}; do
        case "${f}" in
            expf|logf|powf|sinf) ;;
            *) echo "${b}: WRF objects call ${f}, which libm_shim does not pin" >&2
               exit 4 ;;
        esac
    done
    nm "${b}/run_deng_oracle" | grep -E ' T _?(expf|logf|powf|sinf)$' | wc -l | grep -q 4 || {
        echo "${b}: the executable does not define expf/logf/powf/sinf from the shim" >&2
        exit 5; }
done
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "c++:      $(c++ --version | head -1)"
    echo "host:     $(uname -sm)"
    echo "flags:    -O0 ${common} ${cppflags}"
    echo "float32 libm calls of the WRF objects (bound to libm_shim.o):"
    grep -E '^_?(expf|logf|powf|sinf)$' O0/undefined.txt | sed 's/^/  /'
} > compiler.txt
{
    for f in phys/module_shcu_deng.F phys/module_shallowcu_driver.F \
             phys/module_shcu_grims.F phys/module_shcu_nscv.F \
             share/module_model_constants.F; do
        echo "$(sha "${source_root}/${f}")  wrf-4.7.1/${f}"
    done
    for f in stub_wrf.F90 run_deng_oracle.F90 gen_camuw_stub.py libm_shim.cpp \
             deng_columns.py build.sh; do
        echo "$(sha "${script_dir}/${f}")  tools/shcu_deng_wrf471_oracle/${f}"
    done
    echo "$(sha "${kernels_dir}/glibc_flt32.cuh")  woof/core/kernels/glibc_flt32.cuh"
    echo "$(sha camuw_stub.F)  (generated) camuw_stub.F"
} > oracle-sha256sums.txt
cat compiler.txt

for input in "$@"; do
    stem=$(basename "${input}" .bin)
    abs=$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")
    for b in O0 snan; do
        ( cd "${b}" && ./run_deng_oracle "${abs}" "../${stem}-${b}.out" )
    done
    if cmp -s "${stem}-O0.out" "${stem}-snan.out"; then
        echo "${stem}: O0 and snan builds byte-identical"
    else
        echo "${stem}: O0 and snan DIFFER (uninitialised memory reaches an output)"
    fi
done
