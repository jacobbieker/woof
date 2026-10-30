! Column driver for the FARMS oracle: reads one scenario's columns, calls
! WRF v4.7.1's Farms_driver (phys/module_ra_farms.F, compiled unmodified)
! exactly as the radiation driver does at module_radiation_driver.F:3064-3068
! (one tile, i over the columns, a single j row, kts = 1, kte = nz,
! kms = 1, kme = nz + 1), and writes its six irradiances and the four
! intent(inout) aerosol rows back.
!
!   run_farms IN.bin OUT.bin
!
! IN.bin (native-endian stream, written by make_columns.py):
!   int32 ncol, nz, has_reqc, has_reqi, has_reqs, aer_opt
!   real  julian
!   real  albedo(ncol), coszen_loc(ncol), aerssa2d(ncol), aerasy2d(ncol),
!         aod5502d(ncol), angexp2d(ncol)
!   real  p8w, rho, dz8w, qv, qi, qs, qc, re_cloud, re_ice, re_snow,
!         cldfra -- each (ncol, nz + 1), i fastest
! OUT.bin: real swdown2, swddir2, swddni2, swddif2, swdownc2, swddnic2,
!          aerssa2d, aerasy2d, aod5502d, angexp2d -- each (ncol)
program run_farms
  use module_ra_farms, only : farms_driver
  implicit none
  integer :: ncol, nz, kme, has_reqc, has_reqi, has_reqs, aer_opt, u
  real :: julian
  real, allocatable, dimension(:,:,:) :: p8w, rho, dz8w, qv, qi, qs, qc, &
      re_cloud, re_ice, re_snow, cldfra
  real, allocatable, dimension(:,:) :: albedo, coszen_loc, aerssa2d, aerasy2d, &
      aod5502d, angexp2d, swdown2, swddir2, swddni2, swddif2, swdownc2, swddnic2
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) ncol, nz, has_reqc, has_reqi, has_reqs, aer_opt
  read(u) julian
  kme = nz + 1
  allocate(albedo(ncol,1), coszen_loc(ncol,1), aerssa2d(ncol,1), aerasy2d(ncol,1), &
      aod5502d(ncol,1), angexp2d(ncol,1))
  allocate(swdown2(ncol,1), swddir2(ncol,1), swddni2(ncol,1), swddif2(ncol,1), &
      swdownc2(ncol,1), swddnic2(ncol,1))
  allocate(p8w(ncol,kme,1), rho(ncol,kme,1), dz8w(ncol,kme,1), qv(ncol,kme,1), &
      qi(ncol,kme,1), qs(ncol,kme,1), qc(ncol,kme,1), re_cloud(ncol,kme,1), &
      re_ice(ncol,kme,1), re_snow(ncol,kme,1), cldfra(ncol,kme,1))
  read(u) albedo, coszen_loc, aerssa2d, aerasy2d, aod5502d, angexp2d
  read(u) p8w, rho, dz8w, qv, qi, qs, qc, re_cloud, re_ice, re_snow, cldfra
  close(u)

  ! Sentinels: a column Farms_driver did not write would read back as -999.
  swdown2 = -999.0
  swddir2 = -999.0
  swddni2 = -999.0
  swddif2 = -999.0
  swdownc2 = -999.0
  swddnic2 = -999.0

  call farms_driver(1, ncol, 1, 1, 1, ncol, 1, 1, 1, kme, 1, nz,   &
      p8w, rho, dz8w, albedo, aer_opt, aerssa2d, aerasy2d, aod5502d, angexp2d, &
      coszen_loc, qv, qi, qs, qc, re_cloud, re_ice, re_snow,                 &
      julian, swdown2, swddir2, swddni2, swddif2, swdownc2, swddnic2,        &
      has_reqc, has_reqi, has_reqs, cldfra)

  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(u) swdown2, swddir2, swddni2, swddif2, swdownc2, swddnic2, &
      aerssa2d, aerasy2d, aod5502d, angexp2d
  close(u)
end program run_farms
