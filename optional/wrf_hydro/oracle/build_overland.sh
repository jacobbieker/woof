#!/usr/bin/env bash
# Build and run the WRF-Hydro overland routing oracle (OverlandRouting,
# OV_RTNG, ROUTE_OVERLAND1 with GETMAX8DIR/GET8DIR, and ROUTE_OVERLAND2)
# from the WRF-Hydro tree
# WRF 4.7.1 ships under hydro/.  Part of the WRF-Hydro opt-in (see
# ../LICENSE): the pinned source is read at build time and none of it is
# copied into the tree.  Pin check, byte-for-byte cuts with pinned lengths,
# -O0, no MPP_LAND or HYDRO_D, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_overland.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
modules=(
    hydro/utils/module_hydro_stop.F90
    hydro/Routing/Overland/module_overland_control.F90
    hydro/Routing/Overland/module_overland_streams_and_lakes.F90
    hydro/Routing/Overland/module_overland_routing_properties.F90
    hydro/Routing/Overland/module_overland_mass_balance.F90
    hydro/Routing/Overland/module_overland.F90
)
sources=("${modules[@]}" hydro/Routing/Noah_distr_routing.F90
         hydro/Routing/Noah_distr_routing_overland.F90)
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
awk '/^SUBROUTINE GETMAX8DIR\(/{on=1} /^SUBROUTINE GET8DIR\(/{on=1} on{print} on && /^END SUBROUTINE GET(MAX)?8DIR$/{on=0}' \
    "${source_root}/hydro/Routing/Noah_distr_routing.F90" > getmax8dir.inc
if [[ "$(grep -c '' getmax8dir.inc)" -ne 62 ]]; then
    echo "GETMAX8DIR/GET8DIR cut is $(grep -c '' getmax8dir.inc) lines, expected 62" >&2
    exit 6
fi
for rel in "${modules[@]}"; do
    gfortran -c -O0 -cpp ${free} -I "${build_dir}" "${source_root}/${rel}"
done
gfortran -c -O0 -cpp ${free} -I "${build_dir}" -I "${source_root}/hydro/Routing" \
    "${script_dir}/overland_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_overland.F90"
gfortran -o run_overland ./*.o
nm -u overland_harness.o | sed 's/^ *//' | sort > undefined-O0-overland.txt
if grep -q '_ZGV' undefined-O0-overland.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
./run_overland . > run-overland-stdout.txt
if grep -q "stopped due to the fatal error" run-overland-stdout.txt; then
    cat run-overland-stdout.txt >&2
    exit 7
fi
sha256sum "${sources[@]/#/${source_root}/}" getmax8dir.inc \
    "${script_dir}/overland_harness.F90" "${script_dir}/run_overland.F90" \
    "${script_dir}/build_overland.sh" \
    overland-inputs.csv overland-steps.csv overland-totals.csv \
    overland2-inputs.csv overland2-steps.csv overland2-totals.csv \
    | sed "s#${source_root}/#<wrf-4.7.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > overland-sha256sums.txt
echo "oracle built in ${build_dir}"
