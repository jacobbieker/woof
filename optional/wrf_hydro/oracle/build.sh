#!/usr/bin/env bash
# Build and run the WRF-Hydro subsurface-routing oracle from the WRF-Hydro
# tree WRF 4.6.1 ships under hydro/ (WRF pinned at d66e442f...).
#
# Part of the WRF-Hydro opt-in (optional/wrf_hydro, see ../LICENSE): it reads
# UCAR's WRF-Hydro source at build time and copies none of it into the tree.
#
# Compiled from the pinned tree, unchanged: utils/module_hydro_stop.F90, the
# overland and subsurface data modules, and Noah_distr_routing_subsurface.F90
# (#included whole in subsurface_harness.F90).  GETSUB8 and GETSUB8DIR are cut
# byte for byte out of Noah_distr_routing.F90, whose other routines need the
# full routing build.  -O0, no MPP_LAND, no HYDRO_D, and a refusal on any
# libmvec symbol, as tools/noah_wrf461_oracle does.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

modules=(
    hydro/utils/module_hydro_stop.F90
    hydro/Routing/Overland/module_overland_control.F90
    hydro/Routing/Overland/module_overland_streams_and_lakes.F90
    hydro/Routing/Overland/module_overland_routing_properties.F90
    hydro/Routing/Overland/module_overland_mass_balance.F90
    hydro/Routing/Overland/module_overland.F90
    hydro/Routing/Subsurface/module_subsurface_state.F90
    hydro/Routing/Subsurface/module_subsurface_properties.F90
    hydro/Routing/Subsurface/module_subsurface_grid_transform.F90
    hydro/Routing/Subsurface/module_subsurface_static_data.F90
    hydro/Routing/Subsurface/module_subsurface_input.F90
    hydro/Routing/Subsurface/module_subsurface_output.F90
    hydro/Routing/Subsurface/module_subsurface.F90
)
sources=("${modules[@]}" hydro/Routing/Noah_distr_routing.F90
         hydro/Routing/Noah_distr_routing_subsurface.F90)
for rel in "${sources[@]}"; do test -f "${source_root}/${rel}"; done
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the WRF-Hydro sources differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
free="-ffree-form -ffree-line-length-none"

awk '
    /^SUBROUTINE GETSUB8\(/ { on = 1 }
    /^SUBROUTINE GETSUB8DIR\(/ { on = 1 }
    on { print }
    on && /^END SUBROUTINE GETSUB8(DIR)?$/ { on = 0 }
' "${source_root}/hydro/Routing/Noah_distr_routing.F90" > getsub8.inc
if [[ "$(grep -c '' getsub8.inc)" -ne 75 ]]; then
    echo "GETSUB8/GETSUB8DIR cut is $(grep -c '' getsub8.inc) lines, expected 75" >&2
    exit 6
fi

for rel in "${modules[@]}"; do
    gfortran -c -O0 -cpp ${free} -I "${build_dir}" "${source_root}/${rel}"
done
gfortran -c -O0 -cpp ${free} -I "${build_dir}" \
    -I "${source_root}/hydro/Routing" "${script_dir}/subsurface_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_subsurface.F90"
gfortran -o run_subsurface ./*.o

nm -u subsurface_harness.o | sed 's/^ *//' | sort > undefined-O0-harness.txt
if grep -q '_ZGV' undefined-O0-harness.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 subsurface_harness.o math symbols:"
    grep -E 'pow|exp|log|sqrt|_ZGV' undefined-O0-harness.txt || true
} > libm-report.txt
cat libm-report.txt

./run_subsurface . > run-stdout.txt
if grep -q "stopped due to the fatal error" run-stdout.txt; then
    echo "the routing stopped on a fatal error:" >&2
    cat run-stdout.txt >&2
    exit 7
fi

sha256sum "${sources[@]/#/${source_root}/}" getsub8.inc \
    "${script_dir}/subsurface_harness.F90" "${script_dir}/run_subsurface.F90" \
    "${script_dir}/build.sh" \
    subsurface-inputs.csv subsurface-steps.csv subsurface-totals.csv \
    | sed "s#${source_root}/#<wrf-4.6.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > oracle-sha256sums.txt
echo "oracle built in ${build_dir}"
