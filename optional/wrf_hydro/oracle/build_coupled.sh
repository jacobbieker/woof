#!/usr/bin/env bash
# Build and run the coupled WRF/WRF-Hydro oracle: WRF 4.7.1's own
# wrf_cpl_HYDRO (hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90) over the whole
# hydro/ tree its CMake builds (MPP, utils, IO, OrchestratorLayer,
# Debug_Utilities, Overland, Subsurface, Reservoirs, Data_Rec, Routing,
# HYDRO_drv), compiled whole and unchanged, MPP_LAND on one MPI process,
# -O0, no HYDRO_D, no nudging.  coupled_harness.F90 stands in for WRF's
# module_domain/module_configure and drives a few coupling cycles from
# coupled-inputs.txt (gen_coupled_inputs.py writes it and the synthetic
# domain into BUILD_DIR first).  Part of the WRF-Hydro opt-in (../LICENSE):
# the pinned source is read at build time and none of it is copied.
#
# Needs mpif90 and netCDF-Fortran (nf-config); the image used is
# noah-oracle:u2404 plus libnetcdff-dev and libopenmpi-dev.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_coupled.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
if [[ "$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || cat "${source_root}/.pinned-commit")" \
        != "${pinned_commit}" ]]; then
    echo "WRF tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
hydro="${source_root}/hydro"
dirs=(MPP utils IO OrchestratorLayer Debug_Utilities Routing/Overland Routing/Subsurface
      Routing/Reservoirs Routing/Reservoirs/Level_Pool
      Routing/Reservoirs/Persistence_Level_Pool_Hybrid Routing/Reservoirs/RFC_Forecasts
      Data_Rec Routing HYDRO_drv)
sources=()
for d in "${dirs[@]}"; do
    while read -r f; do sources+=("${d}/${f}"); done < <(
        grep -v '^ *#' "${hydro}/${d}/CMakeLists.txt" | grep -o '[A-Za-z0-9_]*\.F90')
done
sources+=(CPL/WRF_cpl/module_wrf_HYDRO.F90)
if [[ -d "${source_root}/.git" ]] && ! git -C "${source_root}" diff --quiet HEAD -- \
        "${sources[@]/#/hydro/}"; then
    echo "a WRF-Hydro source differs from the pinned commit" >&2
    exit 3
fi
mkdir -p "${build_dir}/obj"
cd "${build_dir}/obj"
flags=(-c -O0 -cpp -DMPP_LAND -ffree-form -ffree-line-length-none
       "-DWRF_HYDRO_VERSION=\"v5.4.0-wrf4.7.1\"" "-DNWM_VERSION=\"undefined\""
       $(nf-config --fflags) -I .)
mpif90 "${flags[@]}" "${script_dir}/coupled_wrf_stubs.F90"
pending=("${sources[@]}")
for pass in $(seq 1 12); do
    left=()
    for rel in "${pending[@]}"; do
        if ! mpif90 "${flags[@]}" "${hydro}/${rel}" > /dev/null 2> "$(basename "${rel}").err"; then
            left+=("${rel}")
        fi
    done
    [[ ${#left[@]} -eq 0 ]] && break
    if [[ ${#left[@]} -eq ${#pending[@]} ]]; then
        echo "no progress compiling: ${left[*]}" >&2
        cat "$(basename "${left[0]}").err" >&2
        exit 5
    fi
    pending=("${left[@]}")
done
mpif90 "${flags[@]}" "${script_dir}/coupled_harness.F90"
mpif90 -o ../run_coupled *.o $(nf-config --flibs)
nm -u ./*.o | sed 's/^ *//' | sort -u > ../undefined-O0-coupled.txt
if grep -q '_ZGV' ../undefined-O0-coupled.txt; then
    echo "-O0 build pulled in a libmvec symbol" >&2
    exit 4
fi
cd "${build_dir}"
cp "${hydro}/template/HYDRO/CHANPARM.TBL" CHANPARM.TBL.template
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --oversubscribe -np 1 \
    ./run_coupled > run-coupled-stdout.txt 2>&1 || { cat run-coupled-stdout.txt >&2; exit 7; }
if grep -q "FATAL ERROR\|stopped due to\|HYDRO_stop" run-coupled-stdout.txt; then
    cat run-coupled-stdout.txt >&2
    exit 7
fi
mkdir -p hashed
(cd "${hydro}" && sha256sum "${sources[@]}") | sed 's#^\([0-9a-f]*\)  #\1  <wrf-4.7.1-checkout>/hydro/#' > coupled-sha256sums.txt
sha256sum "${script_dir}/coupled_harness.F90" "${script_dir}/coupled_wrf_stubs.F90" \
    "${script_dir}/build_coupled.sh" \
    "${script_dir}/gen_coupled_inputs.py" coupled-inputs.txt coupled-steps.csv coupled-static.csv \
    Fulldom_hires.nc hydro.namelist \
    | sed "s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" >> coupled-sha256sums.txt
echo "oracle built in ${build_dir}"
