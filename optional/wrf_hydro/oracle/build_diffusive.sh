#!/usr/bin/env bash
# Build and run the WRF-Hydro diffusive-wave channel oracle (DIFFUSION, HEAD,
# AREAf, SGNf and the channel_option = 3 branch of drive_CHANNEL, with its
# level-pool lakes) and the level-pool reservoir oracle from the WRF-Hydro
# tree WRF 4.7.1 ships under hydro/.  Part of the WRF-Hydro opt-in (see
# ../LICENSE): the pinned source is read at build time and none of it is
# copied into the tree.  Pin check, byte-for-byte cuts with pinned lengths,
# the reservoir modules compiled whole, -O0, MPP_LAND (as WRF-Hydro is always
# built) on one process, no HYDRO_D or RESERVOIR_D, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_diffusive.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
sources=(hydro/utils/module_hydro_stop.F90 hydro/Routing/module_channel_routing.F90
         hydro/Routing/Reservoirs/module_reservoir.F90
         hydro/Routing/Reservoirs/Level_Pool/module_levelpool_properties.F90
         hydro/Routing/Reservoirs/Level_Pool/module_levelpool_state.F90
         hydro/Routing/Reservoirs/Level_Pool/module_levelpool.F90)
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
awk '/^real function DIFFUSION\(/{on=1} on{print} on && /^end function DIFFUSION$/{exit}' "${src}" > diffusion.inc
awk '/^     REAL FUNCTION AREAf\(/{on=1} on{print} on && /^     END FUNCTION AREAf$/{exit}' "${src}" > areaf.inc
awk '/^    REAL FUNCTION HEAD\(/{on=1} on{print} on && /^    END FUNCTION HEAD$/{exit}' "${src}" > head.inc
awk '/^     REAL FUNCTION SGNf\(/{on=1} on{print} on && /^     END FUNCTION SGNf$/{exit}' "${src}" > sgnf.inc
awk '/^   elseif\(channel_option \.eq\. 3\) then/{on=1} on{print} on && /^           CVOL  = CVOLTMP/{exit}' "${src}" > diffusive_block.inc
lines="$(grep -c '' diffusion.inc) $(grep -c '' areaf.inc) $(grep -c '' head.inc) $(grep -c '' sgnf.inc) $(grep -c '' diffusive_block.inc)"
if [[ "${lines}" != "75 4 54 12 409" ]]; then
    echo "cut lengths ${lines}, expected 75 4 54 12 409" >&2
    exit 6
fi
gfortran -c -O0 -cpp ${free} "${source_root}/hydro/utils/module_hydro_stop.F90"
objs=(module_hydro_stop.o)
for rel in Reservoirs/module_reservoir Reservoirs/Level_Pool/module_levelpool_properties \
           Reservoirs/Level_Pool/module_levelpool_state Reservoirs/Level_Pool/module_levelpool; do
    gfortran -c -O0 -cpp ${free} "${source_root}/hydro/Routing/${rel}.F90"
    objs+=("$(basename "${rel}").o")
done
gfortran -c -O0 -cpp -DMPP_LAND ${free} -I "${build_dir}" "${script_dir}/diffusive_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_diffusive.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_levelpool.F90"
gfortran -o run_diffusive "${objs[@]}" diffusive_harness.o run_diffusive.o
gfortran -o run_levelpool "${objs[@]}" run_levelpool.o
nm -u diffusive_harness.o module_levelpool.o | sed 's/^ *//' | sort -u > undefined-O0-diffusive.txt
if grep -q '_ZGV' undefined-O0-diffusive.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
./run_diffusive . > run-diffusive-stdout.txt
./run_levelpool . > run-levelpool-stdout.txt
if grep -q "stopped due to the fatal error\|FATAL ERROR\|Courant error\|ERROR" \
        run-diffusive-stdout.txt run-levelpool-stdout.txt; then
    cat run-diffusive-stdout.txt run-levelpool-stdout.txt >&2
    exit 7
fi
sha256sum "${sources[@]/#/${source_root}/}" diffusion.inc areaf.inc head.inc sgnf.inc diffusive_block.inc \
    "${script_dir}/diffusive_harness.F90" "${script_dir}/run_diffusive.F90" \
    "${script_dir}/run_levelpool.F90" "${script_dir}/build_diffusive.sh" \
    diffusive-network.csv diffusive-forcing.csv diffusive-steps.csv diffusive-lakes.csv \
    levelpool-cases.csv diffusive-gw-forcing.csv diffusive-gw-steps.csv diffusive-gw-lakes.csv \
    | sed "s#${source_root}/#<wrf-4.7.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > diffusive-sha256sums.txt
echo "oracle built in ${build_dir}"
