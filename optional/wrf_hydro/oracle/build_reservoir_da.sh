#!/usr/bin/env bash
# Build and run the lake_option 3 reservoir oracle: WRF 4.7.1's
# hydro/Routing/Reservoirs (module_reservoir, the utilities, the timeslice
# and RFC time-series readers, the level pool, the persistence/level-pool
# hybrid and the RFC-forecast reservoir, with their state and properties
# modules) compiled whole and unchanged, -O0, no RESERVOIR_D, driven by
# reservoir_da_harness.F90 over gen_reservoir_da_inputs.py's synthetic
# parameter file, timeslices and RFC series (written into BUILD_DIR first).
# The reservoir modules use no MPI; module_hydro_stop is built without
# MPP_LAND as in build_diffusive.sh.  Part of the WRF-Hydro opt-in
# (../LICENSE): the pinned source is read at build time, none of it copied.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_reservoir_da.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${pinned_commit}" ]]; then
    echo "WRF tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
r=hydro/Routing/Reservoirs
sources=(hydro/utils/module_hydro_stop.F90 ${r}/module_reservoir_utilities.F90
         ${r}/module_reservoir.F90 ${r}/module_reservoir_read_timeslice_data.F90
         ${r}/module_reservoir_read_rfc_time_series_data.F90
         ${r}/Level_Pool/module_levelpool_properties.F90 ${r}/Level_Pool/module_levelpool_state.F90
         ${r}/Level_Pool/module_levelpool.F90
         ${r}/Persistence_Level_Pool_Hybrid/module_persistence_levelpool_hybrid_properties.F90
         ${r}/Persistence_Level_Pool_Hybrid/module_persistence_levelpool_hybrid_state.F90
         ${r}/Persistence_Level_Pool_Hybrid/module_persistence_levelpool_hybrid.F90
         ${r}/RFC_Forecasts/module_rfc_forecasts_properties.F90
         ${r}/RFC_Forecasts/module_rfc_forecasts_state.F90
         ${r}/RFC_Forecasts/module_rfc_forecasts.F90)
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "a WRF-Hydro source differs from the pinned commit" >&2
    exit 3
fi
mkdir -p "${build_dir}/obj"
cd "${build_dir}/obj"
flags=(-c -O0 -cpp -ffree-form -ffree-line-length-none $(nf-config --fflags) -I .)
for rel in "${sources[@]}"; do gfortran "${flags[@]}" "${source_root}/${rel}"; done
gfortran "${flags[@]}" "${script_dir}/reservoir_da_harness.F90"
gfortran -o ../run_reservoir_da *.o $(nf-config --flibs)
nm -u ./*.o | sed 's/^ *//' | sort -u > ../undefined-O0-reservoir-da.txt
if grep -q '_ZGV' ../undefined-O0-reservoir-da.txt; then
    echo "-O0 build pulled in a libmvec symbol" >&2
    exit 4
fi
cd "${build_dir}"
./run_reservoir_da > run-reservoir-da-stdout.txt 2>&1 || { cat run-reservoir-da-stdout.txt >&2; exit 7; }
if grep -q "FATAL\|stopped due to\|ERROR" run-reservoir-da-stdout.txt; then
    cat run-reservoir-da-stdout.txt >&2
    exit 7
fi
(cd "${source_root}" && sha256sum "${sources[@]}") | sed 's#  #  <wrf-4.7.1-checkout>/#' > reservoir-da-sha256sums.txt
sha256sum "${script_dir}/reservoir_da_harness.F90" "${script_dir}/build_reservoir_da.sh" \
    "${script_dir}/gen_reservoir_da_inputs.py" reservoir-inputs.txt reservoir-steps.csv \
    | sed "s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" >> reservoir-da-sha256sums.txt
echo "oracle built in ${build_dir}"
