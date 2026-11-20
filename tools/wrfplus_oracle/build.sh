#!/bin/bash
# Build one WRFPLUS oracle driver against a WRFPLUS 4.7.1 tree.
# Run inside the wrf-oracle:u2404 container (gfortran, netCDF):
#   build.sh <WRFPLUS tree> <driver.F90> <output exe>
# The tree is `./configure wrfplus` option 17 (serial GNU, RWORDSIZE 8)
# with FCOPTIM = -O0 -ffp-contract=off -fno-fast-math, then
# `./compile wrfplus` (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).
set -euo pipefail
W=$1; SRC=$2; EXE=$3
FFLAGS="-O0 -ffp-contract=off -fno-fast-math -fdefault-real-8 -ffree-line-length-none -fallow-argument-mismatch"
gfortran $FFLAGS -I"$W/main" -I"$W/frame" -I"$W/share" -I"$W/wrftladj" -I"$W/dyn_em" \
  -o "$EXE" "$SRC" \
  -Wl,--start-group "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" \
  "$W/external/io_grib_share/libio_grib_share.a" "$W/external/io_int/libwrfio_int.a" \
  -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" \
  -L"$W/external/io_netcdf" -lwrfio_nf -Wl,--end-group -lnetcdff -lnetcdf -lm
