#!/usr/bin/env bash
# Build and run the WRF-Hydro LSM <-> routing grid transform oracle
# (disaggregateDomain, aggregateDomain) from the WRF-Hydro tree WRF 4.6.1
# ships under hydro/.  Part of the WRF-Hydro opt-in (see ../LICENSE): the
# pinned source is read at build time and none of it is copied into the tree.
# Same discipline as build.sh: pin check, byte-for-byte cuts with pinned
# lengths, -O0, no MPP_LAND or HYDRO_D, no libmvec.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_grid_transform.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
sources=(hydro/utils/module_hydro_stop.F90 hydro/Routing/Noah_distr_routing.F90
         hydro/HYDRO_drv/module_HYDRO_drv.F90)
for rel in "${sources[@]}"; do test -f "${source_root}/${rel}"; done
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${pinned_commit}" ]]; then
    echo "WRF tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the WRF-Hydro sources differs from the pinned commit" >&2
    exit 3
fi
mkdir -p "${build_dir}"
cd "${build_dir}"
free="-ffree-form -ffree-line-length-none"

awk '/^subroutine disaggregateDomain\(/{on=1} on{print} on && /^end subroutine disaggregateDomain$/{exit}' \
    "${source_root}/hydro/Routing/Noah_distr_routing.F90" > disaggregate.inc
awk '/^      subroutine aggregateDomain\(did\)/{on=1} on{print} on && /^      end subroutine aggregateDomain/{exit}' \
    "${source_root}/hydro/HYDRO_drv/module_HYDRO_drv.F90" > aggregate.inc
lines="$(grep -c '' disaggregate.inc) $(grep -c '' aggregate.inc)"
if [[ "${lines}" != "346 277" ]]; then
    echo "cut lengths ${lines}, expected 346 277" >&2
    exit 6
fi

gfortran -c -O0 -cpp ${free} "${source_root}/hydro/utils/module_hydro_stop.F90"
gfortran -c -O0 -cpp ${free} -I "${build_dir}" "${script_dir}/grid_transform_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_grid_transform.F90"
gfortran -o run_grid_transform module_hydro_stop.o grid_transform_harness.o run_grid_transform.o
nm -u grid_transform_harness.o | sed 's/^ *//' | sort > undefined-O0-gt.txt
if grep -q '_ZGV' undefined-O0-gt.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
./run_grid_transform . > run-gt-stdout.txt
if grep -q "stopped due to the fatal error" run-gt-stdout.txt; then
    cat run-gt-stdout.txt >&2
    exit 7
fi
# the frozen-solid layer must have taken WRF's "SMC = SICE" path
grep -q "SMC = SICE, soil layer totally frozen" run-gt-stdout.txt
sha256sum "${sources[@]/#/${source_root}/}" disaggregate.inc aggregate.inc \
    "${script_dir}/grid_transform_harness.F90" "${script_dir}/run_grid_transform.F90" \
    "${script_dir}/build_grid_transform.sh" \
    disagg-lsm.csv disagg-rt.csv agg-rt.csv agg-lsm.csv \
    | sed "s#${source_root}/#<wrf-4.6.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > grid-transform-sha256sums.txt
echo "oracle built in ${build_dir}"
