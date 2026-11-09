#!/usr/bin/env bash
# Build the varying-dt Kain-Fritsch driver oracle against the byte-unmodified
# WRF v4.7.1 phys/module_cu_kfeta.F and run it on a case file written by
# case.py.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR CASE.bin OUT.bin
#
# gfortran -O0 (the 0 ULP reference build of tools/kf_wrf461_oracle).  The
# KF source is the scheme that runs; module_cumulus_driver.F and
# module_physics_addtendc.F are pinned because run_kf_adaptive_oracle.F90
# transcribes their adaptive run_param test and advance_ppt.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR CASE.bin OUT.bin" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
case_file=$(realpath "$3")
out_file=$(realpath -m "$4")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

check () {   # relative path, sha256
    local actual
    actual=$(sha256sum "${source_root}/$1" | awk '{print $1}')
    if [[ "${actual}" != "$2" ]]; then
        echo "$1 is ${actual}, not the pinned v4.7.1 $2" >&2
        exit 3
    fi
}
check phys/module_cu_kfeta.F e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f
check phys/module_cumulus_driver.F d73eb4b670599e0bbf98a3661b5dbf17413aed01139d18fa1a84161578dc31e3
check phys/module_physics_addtendc.F 824929ccaf29d99670545cf5cea3e20f4344f06604d1d778cbdbdc2dd445fe44

mkdir -p "${build_dir}"
cd "${build_dir}"
common="-O0 -ffree-form -ffree-line-length-none"
gfortran -c ${common} -o stub_wrf_error.o \
    "${script_dir}/../kf_wrf461_oracle/stub_wrf_error.F90"
gfortran -c ${common} -cpp -o module_cu_kfeta.o \
    "${source_root}/phys/module_cu_kfeta.F"
gfortran -c ${common} -o run_kf_adaptive_oracle.o \
    "${script_dir}/run_kf_adaptive_oracle.F90"
gfortran -o run_kf_adaptive_oracle run_kf_adaptive_oracle.o \
    module_cu_kfeta.o stub_wrf_error.o
./run_kf_adaptive_oracle "${case_file}" "${out_file}"
