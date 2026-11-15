! Write colpkg_init_thermo's salinity profile (column/ice_colpkg.F90, compiled
! verbatim) for every ice-layer count 1..N with BL99 thermodynamics: the
! words of cos(pi*zn**(nsal/(msal+zn))) the oracle host's libm gives, which
! the GPU's correctly rounded cos must reproduce for each count it accepts.
!
!   run_init_thermo N OUTPUT.bin
!
! OUTPUT.bin is the profiles sprofile(1:n+1), n = 1..N, as little-endian
! binary64 words, one after the other.
program run_init_thermo
  use ice_kinds_mod
  use ice_colpkg, only: colpkg_init_thermo
  use ice_colpkg_shared, only: ktherm
  implicit none
  real(kind=dbl_kind), allocatable :: sprofile(:)
  character(len=1024) :: arg
  integer :: n, nMax, u

  call get_command_argument(1, arg)
  read(arg, *) nMax
  call get_command_argument(2, arg)
  ktherm = 1
  open(newunit=u, file=trim(arg), access='stream', form='unformatted', status='replace')
  do n = 1, nMax
     allocate(sprofile(n+1))
     call colpkg_init_thermo(n, sprofile)
     write(u) sprofile
     deallocate(sprofile)
  end do
  close(u)
end program run_init_thermo
