#!/usr/bin/env bash
# Build and run the radiation-cloud oracle: WRF v4.7.1 module_radiation_driver.F
# lines sliced verbatim by gen_slices.py (the file is pinned there by
# SHA-256), gfortran -O0 -ffp-contract=off, float32 libm pinned to WOOF's
# glibc 2.39 transcription (tools/shcu_deng_wrf471_oracle/libm_shim.cpp, the
# same shim as the Deng oracle).
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin
set -euo pipefail
source_root=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
input=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kernels_dir=$(cd "${script_dir}/../../woof/core/kernels" && pwd)
cd "${build_dir}"
python3 "${script_dir}/gen_slices.py" "${source_root}" wrf_rad_cloud_slices.F90
flags="-O0 -ffp-contract=off -ffree-form -ffree-line-length-none"
c++ -std=c++17 -O0 -ffp-contract=off -fno-fast-math -fno-builtin \
    -I "${kernels_dir}" -c "${script_dir}/../shcu_deng_wrf471_oracle/libm_shim.cpp" \
    -o libm_shim.o
gfortran -c ${flags} "${script_dir}/stub_state.F90" -o stub_state.o
gfortran -c ${flags} wrf_rad_cloud_slices.F90 -o wrf_rad_cloud_slices.o
gfortran -c ${flags} "${script_dir}/run_rad_cloud.F90" -o run_rad_cloud.o
gfortran -o run_rad_cloud run_rad_cloud.o wrf_rad_cloud_slices.o stub_state.o libm_shim.o
nm -u wrf_rad_cloud_slices.o | sed 's/^ *//' | grep -E '^_?(exp|log|pow|sin|cos)f?$' \
    | sed 's/^_//' | sort -u > libm-calls.txt
for f in $(cat libm-calls.txt); do
    case "${f}" in expf|powf) ;; *) echo "slices call ${f}" >&2; exit 4 ;; esac
done
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "host:     $(uname -sm)"
    echo "flags:    ${flags}"
    echo "float32 libm calls (bound to libm_shim.o): $(tr '\n' ' ' < libm-calls.txt)"
    echo "$(shasum -a 256 "${source_root}/phys/module_radiation_driver.F" | awk '{print $1}')  wrf-4.7.1/phys/module_radiation_driver.F"
    echo "$(shasum -a 256 wrf_rad_cloud_slices.F90 | awk '{print $1}')  (generated) wrf_rad_cloud_slices.F90"
    for f in gen_slices.py run_rad_cloud.F90 stub_state.F90 rad_cloud_cases.py build.sh; do
        echo "$(shasum -a 256 "${script_dir}/${f}" | awk '{print $1}')  tools/cu_rad_feedback_oracle/${f}"
    done
} > receipt.txt
./run_rad_cloud "${input}" rad-cloud-O0.out
cat receipt.txt
