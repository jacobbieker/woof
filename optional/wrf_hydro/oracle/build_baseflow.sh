#!/usr/bin/env bash
# Build and run the WRF-Hydro groundwater bucket oracle (simp_gw_buck) from
# the WRF-Hydro tree WRF 4.7.1 ships under hydro/.  Part of the WRF-Hydro
# opt-in (see ../LICENSE): the pinned source is read at build time and none of
# it is copied into the tree.  Pin check, byte-for-byte cut with a pinned
# length, -O0, no MPP_LAND or HYDRO_D, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_baseflow.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
sources=(hydro/Routing/module_GW_baseflow.F90)
test -f "${source_root}/${sources[0]}"
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${pinned_commit}" ]]; then
    echo "WRF tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "module_GW_baseflow.F90 differs from the pinned commit" >&2
    exit 3
fi
mkdir -p "${build_dir}"
cd "${build_dir}"
free="-ffree-form -ffree-line-length-none"
awk '/^   subroutine simp_gw_buck\(/{on=1} on{print} on && /^   End subroutine simp_gw_buck$/{exit}' \
    "${source_root}/hydro/Routing/module_GW_baseflow.F90" > simp_gw_buck.inc
if [[ "$(grep -c '' simp_gw_buck.inc)" -ne 232 ]]; then
    echo "simp_gw_buck cut is $(grep -c '' simp_gw_buck.inc) lines, expected 232" >&2
    exit 6
fi
gfortran -c -O0 -cpp ${free} -I "${build_dir}" "${script_dir}/baseflow_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_baseflow.F90"
gfortran -o run_baseflow baseflow_harness.o run_baseflow.o
nm -u baseflow_harness.o | sed 's/^ *//' | sort > undefined-O0-baseflow.txt
if grep -q '_ZGV' undefined-O0-baseflow.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
grep -E 'exp' undefined-O0-baseflow.txt > baseflow-libm.txt || true
./run_baseflow . > run-baseflow-stdout.txt
sha256sum "${source_root}/${sources[0]}" simp_gw_buck.inc \
    "${script_dir}/baseflow_harness.F90" "${script_dir}/run_baseflow.F90" \
    "${script_dir}/build_baseflow.sh" \
    baseflow-static.csv baseflow-grids.csv baseflow-forcing.csv baseflow-steps.csv \
    | sed "s#${source_root}/#<wrf-4.7.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > baseflow-sha256sums.txt
echo "oracle built in ${build_dir}"
