#!/bin/bash
# Link WRFDA 4.7.1's gen_be executables from the pinned build's objects into
# $W/genbe/bin (the pinned tree itself is not written).  WRF's stock link line
# omits the netCDF Fortran libraries, so `./compile all_wrfvar` leaves gen_be
# unlinked; this adds them.  Run inside wrf-oracle:u2404:
#   docker run --rm -u $(id -u):$(id -g) -v /mnt/NVME2/woof-work:/d wrf-oracle:u2404 \
#     bash /d/scratch/wrfda/scripts/link_gen_be.sh
set -e
W=${W:-/d/scratch/wrfda}
T=$W/src/${TREE:-WRFV4.7.1}
B=$T/var/build
O=$W/genbe/bin
mkdir -p "$O"
cd "$B"
[ -x "$B/da_advance_time.exe" ] || { echo "link_gen_be.sh: $B/da_advance_time.exe is missing or not executable (gen_be stage 0 needs it; build WRFDA first)" >&2; exit 1; }
# gen_be_ep2 and gen_be_ensmean are the ensemble method's (be_method ENS) and
# optional: the NMC gate does not run them, so a missing object is skipped.
OPTIONAL="gen_be_ep2 gen_be_ensmean"
for p in gen_be_stage0_wrf gen_be_stage1 gen_be_stage2 gen_be_stage2a gen_be_stage3 gen_be_stage4_regional gen_be_diags $OPTIONAL; do
  if [ ! -f "$p.o" ]; then
    case " $OPTIONAL " in
      *" $p "*) echo "link_gen_be.sh: optional $p.o is not in $B; skipping $p.exe" >&2; continue ;;
      *) echo "link_gen_be.sh: required $p.o is not in $B (was WRFDA built?)" >&2; exit 1 ;;
    esac
  fi
  gfortran -o "$O/$p.exe" da_etkf.o da_blas.o da_lapack.o da_gen_be.o da_control.o da_wavelet.o da_be_spectral.o \
    module_wrf_error.o module_driver_constants.o module_domain_type.o module_streams.o module_symbols_util.o \
    module_utility.o da_memory.o da_reporting.o da_tools_serial.o module_ffts.o "$p.o" \
    -L"$T/external/io_netcdf" -lwrfio_nf -L"$T/external/fftpack/fftpack5" -lfftpack \
    -L"$T/external/esmf_time_f90" -lesmf_time -lnetcdff -lnetcdf
done
ln -sf "$B/da_advance_time.exe" "$O/da_advance_time.exe"
ls "$O"
