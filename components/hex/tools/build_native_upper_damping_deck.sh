#!/bin/bash
# Build the native upper-damping block deck (WOOF_HEX_NATIVE_UPPER_FIXTURE) from MPAS v8.4.2's unchanged
# source lines: the CAM viscosity floor and the horizontal Rayleigh damping of u in atm_compute_dyn_tend_work.
# Run inside the wrf-oracle container (fortran tree mounted at /f): build_native_upper_damping_deck.sh OUTDIR
set -euo pipefail
OUT=$1; mkdir -p $OUT; cd $OUT
F=/f/src/MPAS-8.4.2/src/core_atmosphere/dynamics/mpas_atm_time_integration.F
cam=$(grep -n "do k = nVertLevels-config_number_cam_damping_levels + 1, nVertLevels" $F | cut -d: -f1)
ray=$(grep -n "if (config_rayleigh_damp_u) then" $F | cut -d: -f1)
[ "$(echo $cam | wc -w)" = 1 ] && [ "$(echo $ray | wc -w)" = 1 ]
sed -n "${cam},$((cam+4))p" $F > cam.inc      # do k ... end do (the k loop of one cell)
end=$(awk -v s=$ray 'NR>s && /^      end if/ {print NR; exit}' $F)
sed -n "${ray},${end}p" $F > ray.inc
echo "CAM lines $cam-$((cam+4)), Rayleigh lines $ray-$end" > lines.txt
cat > deck.F <<'EOF'
program upperdeck
   implicit none
   integer, parameter :: RKIND = selected_real_kind(6)
   real (kind=RKIND), parameter :: seconds_per_day = 86400.0_RKIND
   integer, parameter :: nVertLevels = 59, nCells = 3, nEdges = 3
   integer :: k, iCell, iEdge, edgeSolveStart, edgeSolveEnd, i
   integer :: config_number_cam_damping_levels, config_number_rayleigh_damp_u_levels
   logical :: config_rayleigh_damp_u
   real (kind=RKIND) :: config_len_disp, config_mpas_cam_coef, config_rayleigh_damp_u_timescale_days
   real (kind=RKIND) :: visc2cam, rayleigh_coef_inverse
   real (kind=RKIND) :: eddy_visc_horz(nVertLevels,nCells), tend_u(nVertLevels,nEdges)
   real (kind=RKIND) :: rho_edge(nVertLevels,nEdges), u(nVertLevels,nEdges), rayleigh_damp_coef(nVertLevels)
   real (kind=RKIND) :: r(4*nVertLevels*nCells)
   integer :: seed(64)
   seed = 20261116
   call random_seed(put=seed(1:size(seed)))
   call random_number(r)
   i = 0
   do iCell = 1, nCells
      do k = 1, nVertLevels
         i = i + 1; eddy_visc_horz(k,iCell) = 4.0e5_RKIND*r(i)
      end do
   end do
   do iEdge = 1, nEdges
      do k = 1, nVertLevels
         i = i + 1; tend_u(k,iEdge) = 2.0e-3_RKIND*(r(i)-0.5_RKIND)
      end do
   end do
   do iEdge = 1, nEdges
      do k = 1, nVertLevels
         i = i + 1; rho_edge(k,iEdge) = 1.0e-3_RKIND + 0.1_RKIND*r(i)
      end do
   end do
   do iEdge = 1, nEdges
      do k = 1, nVertLevels
         i = i + 1; u(k,iEdge) = 100.0_RKIND*(r(i)-0.5_RKIND)
      end do
   end do
   open(10, file='upper-input.bin', access='stream', form='unformatted', status='replace')
   write(10) eddy_visc_horz, tend_u, rho_edge, u
   close(10)
   config_len_disp = 12000.0_RKIND; config_mpas_cam_coef = 2.0_RKIND
   config_number_cam_damping_levels = 8
   config_rayleigh_damp_u = .true.; config_number_rayleigh_damp_u_levels = 6
   config_rayleigh_damp_u_timescale_days = 5.0_RKIND
   edgeSolveStart = 1; edgeSolveEnd = 2
   do iCell = 1, nCells
#include "cam.inc"
   end do
#include "ray.inc"
   open(11, file='upper-output.bin', access='stream', form='unformatted', status='replace')
   write(11) eddy_visc_horz, tend_u
   close(11)
end program upperdeck
EOF
gfortran -O3 -ffree-line-length-none -ffree-form -cpp deck.F -o deck
./deck
gfortran --version | head -1 > compiler.txt
sha256sum $F cam.inc ray.inc upper-input.bin upper-output.bin > sha256.txt
cat lines.txt compiler.txt
