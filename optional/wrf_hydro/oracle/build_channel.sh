#!/usr/bin/env bash
# Build and run the WRF-Hydro reach channel routing oracle (MUSKING,
# SUBMUSKINGCUNGE and the reach branch of drive_CHANNEL) from the WRF-Hydro
# tree WRF 4.6.1 ships under hydro/.  Part of the WRF-Hydro opt-in (see
# ../LICENSE): the pinned source is read at build time and none of it is
# copied into the tree.  Pin check, byte-for-byte cuts with pinned lengths,
# -O0, no MPP_LAND or HYDRO_D, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_channel.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
sources=(hydro/utils/module_hydro_stop.F90 hydro/Routing/module_channel_routing.F90)
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
src="${source_root}/hydro/Routing/module_channel_routing.F90"
awk '/^real function MUSKING\(/{on=1} on{print} on && /^end function MUSKING$/{exit}' "${src}" > musking.inc
awk '/^subroutine SUBMUSKINGCUNGE\(/{on=1} on{print} on && /^END SUBROUTINE SUBMUSKINGCUNGE$/{exit}' "${src}" > submuskingcunge.inc
awk '/^   IF\(channel_option \.ne\. 3\) then/{on=1} on{print} on && /^   end do ! nsteps$/{exit}' "${src}" > reach_block.inc
lines="$(grep -c '' musking.inc) $(grep -c '' submuskingcunge.inc) $(grep -c '' reach_block.inc)"
if [[ "${lines}" != "24 390 162" ]]; then
    echo "cut lengths ${lines}, expected 24 390 162" >&2
    exit 6
fi
gfortran -c -O0 -cpp ${free} "${source_root}/hydro/utils/module_hydro_stop.F90"
gfortran -c -O0 -cpp ${free} -I "${build_dir}" "${script_dir}/channel_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_channel.F90"
gfortran -o run_channel module_hydro_stop.o channel_harness.o run_channel.o
nm -u channel_harness.o | sed 's/^ *//' | sort > undefined-O0-channel.txt
if grep -q '_ZGV' undefined-O0-channel.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
./run_channel . > run-channel-stdout.txt
if grep -q "stopped due to the fatal error\|Failure to converge" run-channel-stdout.txt; then
    cat run-channel-stdout.txt >&2
    exit 7
fi
sha256sum "${sources[@]/#/${source_root}/}" musking.inc submuskingcunge.inc reach_block.inc \
    "${script_dir}/channel_harness.F90" "${script_dir}/run_channel.F90" \
    "${script_dir}/build_channel.sh" \
    channel-network.csv channel-pixels.csv channel-forcing.csv channel-steps.csv \
    | sed "s#${source_root}/#<wrf-4.6.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > channel-sha256sums.txt
echo "oracle built in ${build_dir}"
