#!/usr/bin/env bash
# Build and run the WRF-Hydro 2-D groundwater oracle (GWBASESWCRT = 3:
# gw2d_ini, gwstep, gwSoilFlux, aggregateQsgw, HYDRO_drv's call of them and
# its channel-aquifer head correction)
# from the WRF-Hydro tree WRF 4.7.1 ships under hydro/.  Part of the
# WRF-Hydro opt-in (see ../LICENSE): the pinned source is read at build time
# and none of it is copied into the tree.  Pin check, byte-for-byte cuts with
# pinned lengths, -O0, MPP_LAND against one-process exchanges, no HYDRO_D,
# no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_gw2d.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd -P)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd -P)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="f52c197ed39d12e087d02c50f412d90d418f6186"
sources=(hydro/Routing/module_gw_gw2d.F90 hydro/Data_Rec/module_gw_gw2d_data.F90
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
cd "${build_dir}"
free="-ffree-form -ffree-line-length-none"
src="${source_root}/hydro/Routing/module_gw_gw2d.F90"
cut() {  # cut NAME FIRST-REGEX LAST-REGEX
    awk -v a="$2" -v b="$3" '$0 ~ a {on=1} on {print} on && $0 ~ b {exit}' "${src}" > "$1.inc"
}
cut gw2d_ini      '^   subroutine gw2d_ini\(did,dt,dx\)$'        '^   end subroutine gw2d_ini$'
cut gw2d_allocate '^   subroutine gw2d_allocate\(did, ix, jx, nsoil\)$' '^    end subroutine gw2d_allocate$'
cut gwstep        '^    subroutine gwstep\(ix, jx, dx, '          '^      end subroutine gwstep$'
cut scopy         '^      SUBROUTINE SCOPY \(NT, ARR, INCA, BRR, INCB\)$' '^      END SUBROUTINE SCOPY$'
cut gwsoilflux    '^subroutine gwSoilFlux\(did\)$'                '^end subroutine gwSoilFlux$'
cut darcygwsoil   '^subroutine darcyGwSoil\(Z, s, soil, q_darcy\)$' '^end subroutine darcyGwSoil$'
cut aggregateqsgw '^subroutine aggregateQsgw\(did\)$'             '^end subroutine aggregateQsgw$'
cut sub_tri_solv  '^      subroutine sub_tri_solv\('              '^      end  subroutine$'
awk '/^                if \(nlst\(did\)%gwsoilcpl \.GT\. 0\) THEN$/{on=1} on{print}
     on && /^                gw2d\(did\)%istep =  gw2d\(did\)%istep \+ 1$/{exit}' \
    "${source_root}/hydro/HYDRO_drv/module_HYDRO_drv.F90" > drv_gw2d.inc
awk '/^            if\(\(nlst\(did\)%gwBaseSwCRT == 3\) \.and\. \(nlst\(did\)%gwChanCondSw \.eq\. 1\)\) then$/{on=1} on{print}
     on && /^            endif$/{exit}' \
    "${source_root}/hydro/HYDRO_drv/module_HYDRO_drv.F90" > drv_gw2d_chan.inc
lines=""
for f in gw2d_ini gw2d_allocate gwstep scopy gwsoilflux darcygwsoil aggregateqsgw sub_tri_solv drv_gw2d drv_gw2d_chan; do
    lines="${lines} $(grep -c '' ${f}.inc)"
done
if [[ "${lines}" != " 58 64 760 16 102 67 39 83 26 9" ]]; then
    echo "cut lengths${lines}, expected 58 64 760 16 102 67 39 83 26 9" >&2
    exit 6
fi
gfortran -c -O0 -cpp ${free} "${source_root}/hydro/Data_Rec/module_gw_gw2d_data.F90"
gfortran -c -O0 -cpp -DMPP_LAND ${free} -I "${build_dir}" "${script_dir}/gw2d_harness.F90"
gfortran -c -O0 ${free} "${script_dir}/gw2d_mpp_stubs.f90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_gw2d.F90"
gfortran -o run_gw2d module_gw_gw2d_data.o gw2d_harness.o gw2d_mpp_stubs.o run_gw2d.o
nm -u gw2d_harness.o | sed 's/^ *//' | sort > undefined-O0-gw2d.txt
if grep -q '_ZGV' undefined-O0-gw2d.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
grep -E 'expf|powf|sqrtf|exp$|pow$' undefined-O0-gw2d.txt > gw2d-libm.txt || true
./run_gw2d . > run-gw2d-stdout.txt
sha256sum "${sources[@]/#/${source_root}/}" gw2d_ini.inc gw2d_allocate.inc gwstep.inc scopy.inc \
    gwsoilflux.inc darcygwsoil.inc aggregateqsgw.inc sub_tri_solv.inc drv_gw2d.inc drv_gw2d_chan.inc \
    "${script_dir}/gw2d_harness.F90" "${script_dir}/gw2d_mpp_stubs.f90" \
    "${script_dir}/run_gw2d.F90" "${script_dir}/build_gw2d.sh" \
    gw2d-static.csv gw2d-steps.csv gw2d-totals.csv gw2d-qsgw.csv gw2d-chan.csv \
    | sed "s#${source_root}/#<wrf-4.7.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/wrf_hydro/oracle/#" \
    > gw2d-sha256sums.txt
echo "oracle built in ${build_dir}"
