#!/usr/bin/env bash
# Build the RRTMG shortwave spectral oracle: sw_build.sh's fixture driver
# against WRF v4.6.1 RRTMG with spcvmc_sw's per-band accumulation made
# live and the per-g-point products recorded (spectral_patch.py).
#
# The longwave half builds lw_build.sh's extractor against rtrnmc with each
# band's flux term recorded, in BUILD_DIR/lw.
#
# usage: spectral_build.sh WRF_SOURCE_ROOT BUILD_DIR
#   then: BUILD_DIR/sw_spectral_driver sw_fixtures/columns_real.txt out.swd
#         BUILD_DIR/lw/lw_spectral_extract LW_INPUT_TXT OUT_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: spectral_build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lw_source="${source_root}/phys/module_ra_rrtmg_lw.F"

python "${script_dir}/spectral_patch.py" "${source_root}" "${build_dir}"
cd "${build_dir}"

FFLAGS="-O0 -ffp-contract=off -cpp -ffree-form -ffree-line-length-none -fconvert=big-endian \
  -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"

gfortran -c $FFLAGS "${script_dir}/sw_stub_wrf.F90"
gfortran -c $FFLAGS -I "${build_dir}" "${lw_source}"
gfortran -c $FFLAGS -I "${build_dir}" module_ra_rrtmg_sw_spectral.F
python "${script_dir}/aer1_extract.py" "${source_root}" "${build_dir}/aer1_authority.F90"
gfortran -c $FFLAGS -I "${build_dir}" aer1_authority.F90
gfortran -c $FFLAGS -I "${build_dir}" sw_spectral_driver.F90
gfortran -o sw_spectral_driver sw_stub_wrf.o module_ra_rrtmg_lw.o \
    module_ra_rrtmg_sw_spectral.o aer1_authority.o sw_spectral_driver.o

cp -f "${source_root}/run/RRTMG_SW_DATA" .
cp -f "${source_root}/run/RRTMG_LW_DATA" .

sha256sum "${lw_source}" "${source_root}/phys/module_ra_rrtmg_sw.F" \
    module_ra_rrtmg_sw_spectral.F sw_spectral_driver.F90 RRTMG_SW_DATA RRTMG_LW_DATA \
    "${script_dir}/sw_stub_wrf.F90" "${script_dir}/sw_fixture_driver.F90" \
    "${script_dir}/spectral_patch.py" > spectral-oracle-sha256sums.txt
{ gfortran --version | head -1; echo "flags: ${FFLAGS}"; } > spectral-compiler.txt

# ---- longwave (lw_build.sh's steps, patched module and extractor) -------
mkdir -p lw && cd lw
python3 "${script_dir}/lw_gen_dumpers.py" ../module_ra_rrtmg_lw_spectral.F lw_dump_state.F90
gfortran -c $FFLAGS "${script_dir}/lw_stub_wrf.F90"
gfortran -c $FFLAGS -I . ../module_ra_rrtmg_lw_spectral.F
gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/lw_binio.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none -I . lw_dump_state.F90
gfortran -c $FFLAGS -I . ../lw_spectral_extract.F90
gfortran -o lw_spectral_extract lw_stub_wrf.o module_ra_rrtmg_lw_spectral.o lw_binio.o \
    lw_dump_state.o lw_spectral_extract.o
cp -f "${source_root}/run/RRTMG_LW_DATA" .
sha256sum ../module_ra_rrtmg_lw_spectral.F ../lw_spectral_extract.F90 lw_dump_state.F90 \
    "${script_dir}/lw_extract.F90" "${script_dir}/lw_stub_wrf.F90" "${script_dir}/lw_binio.F90" \
    RRTMG_LW_DATA > lw-spectral-oracle-sha256sums.txt
cd ..
echo "spectral_build.sh: OK"
