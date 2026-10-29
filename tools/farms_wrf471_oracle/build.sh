#!/usr/bin/env bash
# Build and run the FARMS column oracle from byte-unmodified WRF v4.7.1
# (commit f52c197ed) phys/module_ra_farms.F.
#
#   FC=gfortran CC=cc CXX=c++ build.sh WRF_SOURCE_ROOT BUILD_DIR
#   (on the development Mac: FC="$(pixi exec -s gfortran -- which gfortran)")
#
# Writes into BUILD_DIR:
#   columns.npz, <scenario>.in.bin    the columns (make_columns.py)
#   <scenario>.out.bin                reference of record: -O0, no FMA
#                                     contraction, the binary32 libm calls
#                                     bound to woof's word-exact ports
#                                     (libm_ports.cpp via libm_shim.c)
#   <scenario>.syslibm.out.bin        evidence only: the same objects linked
#                                     against the host's own C library
#   receipt.txt                       sources, sha256s, compiler, flags,
#                                     libm symbols
# Then pack.py gathers the inputs and the reference into
# tests/data/oracles/farms/farms_wrf471.npz with its provenance.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
python=${PYTHON:-python3}
read -r -a fc <<< "${FC:-gfortran}"
read -r -a cc <<< "${CC:-cc}"
read -r -a cxx <<< "${CXX:-c++}"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# The two WRF files compiled, pinned to WRF v4.7.1 f52c197ed (module_ra_farms.F
# is byte-identical in v4.6.1 d66e442).
farms_src="${source_root}/phys/module_ra_farms.F"
const_src="${source_root}/share/module_model_constants.F"
pinned_farms="1b9e060d71dd7218c39e5a5724a7886307507a7cb85d7db33cd3ed52c0034a60"
pinned_const="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
[[ "$(sha "${farms_src}" | cut -d' ' -f1)" == "${pinned_farms}" ]] || {
    echo "module_ra_farms.F differs from WRF v4.7.1 f52c197ed" >&2; exit 3; }
[[ "$(sha "${const_src}" | cut -d' ' -f1)" == "${pinned_const}" ]] || {
    echo "module_model_constants.F differs from WRF v4.7.1 f52c197ed" >&2; exit 3; }

cd "${build_dir}"
f0=(-O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
c0=(-O2 -ffp-contract=off -fno-fast-math -fno-builtin)
"${fc[@]}" -c "${f0[@]}" -DEM_CORE=1 -o module_model_constants.o "${const_src}"
"${fc[@]}" -c "${f0[@]}" -o module_ra_farms.o "${farms_src}"
"${fc[@]}" -c "${f0[@]}" -o run_farms.o "${script_dir}/run_farms.F90"
"${cxx[@]}" -c -std=c++17 "${c0[@]}" -I "${repo_root}/woof/backend/host" \
    -I "${repo_root}/woof/core/kernels" -o libm_ports.o "${script_dir}/libm_ports.cpp"
"${cc[@]}" -c "${c0[@]}" -o libm_shim.o "${script_dir}/libm_shim.c"
objs=(run_farms.o module_ra_farms.o module_model_constants.o)
"${fc[@]}" -o run_farms "${objs[@]}" libm_shim.o libm_ports.o
"${fc[@]}" -o run_farms_syslibm "${objs[@]}"

# Every libm symbol the WRF object calls must be one the shim defines.
libm_calls=$(nm -u module_ra_farms.o | sed 's/^ *U *//; s/^_//' \
    | grep -E '^(exp|log|log10|pow|sin|cos|tan|acos|asin|atan|sinh|cosh|tanh|sqrt)f$' | sort -u)
defined=$(nm run_farms)
for sym in ${libm_calls}; do
    if ! grep -Eq " T _?${sym}\$" <<< "${defined}"; then
        echo "run_farms does not define ${sym}: the oracle would call the host libm" >&2
        exit 4
    fi
done
disasm=$(objdump -d module_ra_farms.o 2>/dev/null || true)
if grep -Eq 'vfn?m(add|sub)|fn?madd|fn?msub' <<< "${disasm}"; then
    echo "-O0 FARMS object contains a fused multiply-add" >&2; exit 4
fi

"${python}" "${script_dir}/make_columns.py" "${build_dir}" > columns.txt
for scen in $("${python}" -c "import numpy as n; print(' '.join(n.load('columns.npz')['scenarios']))"); do
    ./run_farms "${scen}.in.bin" "${scen}.out.bin"
    ./run_farms_syslibm "${scen}.in.bin" "${scen}.syslibm.out.bin"
done

{
    echo "wrf: WRF v4.7.1 commit f52c197ed39d12e087d02c50f412d90d418f6186"
    echo "host: $(uname -srm)"
    "${fc[@]}" --version | head -1
    "${cxx[@]}" --version | head -1
    echo "fortran flags: ${f0[*]} (module_model_constants.F with -DEM_CORE=1)"
    echo "libm port flags: -std=c++17 ${c0[*]}"
    echo "libm calls in module_ra_farms.o (bound to woof ports): $(echo ${libm_calls} | tr '\n' ' ')"
    cat columns.txt
    sha "${farms_src}" "${const_src}" "${script_dir}/run_farms.F90" \
        "${script_dir}/make_columns.py" "${script_dir}/libm_ports.cpp" \
        "${script_dir}/libm_shim.c" "${repo_root}/woof/backend/host/cuda_host_shim.h" \
        "${repo_root}/woof/core/kernels/glibc_flt32.cuh" \
        "${repo_root}/woof/core/kernels/glibc_trig_flt32.cuh" \
        "${repo_root}/woof/core/kernels/log10_flt32.cuh" \
        "${repo_root}/woof/core/kernels/sinhf_flt32.cuh"
    sha ./*.out.bin columns.npz
} > receipt.txt
cat receipt.txt
