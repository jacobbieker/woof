#!/bin/sh
# Build the WRF v4.7.1 Noah-MP WATER oracle as WRF builds it with -DWRF_HYDRO
# (WRF_HYDRO=1 at configure): module_sf_noahmplsm.F's WATER then adds the
# ponded head WRF-Hydro returns to the LSM, QINSUR = QINSUR+sfcheadrt/DT*0.001
# (:6173-6175), and carries sfcheadrt and WATBLED through its interface.
#
# Usage:  build_water_hydro.sh <wrf-tree> <workdir>
#
# <wrf-tree> holds phys/module_sf_noahmplsm.F (WRF 4.7.1's phys/noahmp
# submodule at e5c0859874407859936739e8be8741f9aed369ee,
# src/module_sf_noahmplsm.F, the pinned sha256 below) and
# phys/module_sf_gecros.F.  WATER and everything it calls (:5954-6261 and the
# routines after it) are byte-identical to WRF 4.6.1's 848f54ad; the
# submodule's changes between the two are in ALBEDO, SFCDIF2's stability
# functions and the irrigation table reader.  The build is build_water.sh's at -O0 with
# -DWRF_HYDRO; the harness is run_water.F90 extended by
# patch_run_water_hydro.py with the two WRF_HYDRO arguments, a ponded head
# per case, and their rows in the fixture.  Same accessibility lift, same
# libmvec refusal.
set -eu
WRF_TREE=$1
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/noahmp_pins.sh"
noahmp_pins "$WRF_TREE/phys/module_sf_noahmplsm.F"
PRISTINE_SHA=$PRISTINE_LSM_SHA
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"
FCOPTIM="-O0"
mkdir -p "$WORK"
cd "$WORK"
GOT=$(sha256sum "$WRF_TREE/phys/module_sf_noahmplsm.F" | cut -d' ' -f1)
if [ "$GOT" != "$PRISTINE_SHA" ]; then
  echo "module_sf_noahmplsm.F is not the pinned file ($GOT)" >&2
  exit 3
fi
cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F
python3 "$HERE/visibility_patch_leaves.py" ./pristine.F --accept-sha256 "$PRISTINE_SHA" \
        --out ./module_sf_noahmplsm_public.F --check \
        --require-symbol WATER --require-symbol CANWATER \
        --require-symbol SNOWWATER --require-symbol SOILWATER
python3 "$HERE/patch_run_water_hydro.py" "$HERE/run_water.F90" ./run_water_hydro.F90
cat > wrf_stubs.f90 <<'STUB'
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp water oracle: WRF fatal: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal
subroutine wrf_error_fatal3(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'noahmp water oracle: WRF fatal: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal3
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') trim(msg)
end subroutine wrf_message
subroutine wrf_debug(lvl, msg)
  integer, intent(in) :: lvl
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
STUB
gfortran -c $FCBASE $FCOPTIM module_sf_gecros.F -o module_sf_gecros.o
gfortran -c $FCBASE $FCOPTIM -DWRF_HYDRO module_sf_noahmplsm_public.F -o module_sf_noahmplsm.o
gfortran -c -w -O2 wrf_stubs.f90 -o wrf_stubs.o
gfortran -c $FCBASE $FCOPTIM run_water_hydro.F90 -o run_water_hydro.o
gfortran -o run_water_hydro run_water_hydro.o module_sf_noahmplsm.o module_sf_gecros.o wrf_stubs.o -lm
UNDEF=$(nm -u module_sf_noahmplsm.o run_water_hydro.o | awk '{print $NF}' | sort -u)
if printf '%s\n' "$UNDEF" | grep -qE '^_ZGV'; then
  echo "libmvec symbol(s) present" >&2
  exit 4
fi
./run_water_hydro noahmp-water-hydro.csv noahmp-water-hydro-libm.csv
sha256sum "$WRF_TREE/phys/module_sf_noahmplsm.F" "$HERE/run_water.F90" \
    "$HERE/patch_run_water_hydro.py" "$HERE/build_water_hydro.sh" noahmp-water-hydro.csv \
    | sed "s#$WRF_TREE/#<$NOAHMP_TREE-tree>/#; s#$HERE/#tools/noahmp_wrf461_oracle/#" > noahmp-water-hydro-sha256sums.txt
echo "oracle built in $WORK"
