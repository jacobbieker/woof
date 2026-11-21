! Drive E3SM v3.2.0's prep_ice_merge (driver-mct/main/prep_ice_mod.F90, cut
! verbatim) over attribute vectors read from a stream.
!
!   run_ice_merge INPUT OUTPUT
!
! INPUT (little-endian): flux_epbalfact (r8); wav_ice_coup (8 characters);
! lsize; ncalls; then, for a2x, o2x, r2x, g2x, w2x and x2i in that order, the
! length of the colon-separated field list and the list.  Then, per call, each
! vector's rAttr(nflds, lsize) as r8 (x2i's too: prep_ice_merge does not zero
! it).  OUTPUT: x2i's rAttr after each call.
program run_ice_merge
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_flds_mod, only: wav_ice_coup
  use seq_comm_mct, only: logunit
  use prep_ice_merge_cut, only: prep_ice_merge
  implicit none
  integer, parameter :: nav = 6
  type(mct_aVect) :: av(nav)
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  character(len=8) :: wav
  real(r8) :: flux_epbalfact
  integer :: lsize, ncalls, rlen, i, icall, uin, uout

  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flux_epbalfact, wav, lsize, ncalls
  wav_ice_coup = trim(wav)
  do i = 1, nav
     read(uin) rlen
     allocate(character(len=rlen) :: rlist)
     if (rlen > 0) read(uin) rlist
     call mct_aVect_init(av(i), rList=rlist, lsize=lsize)
     deallocate(rlist)
  end do
  open(newunit=logunit, file='/dev/null', status='old', action='write')
  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  do icall = 1, ncalls
     do i = 1, nav
        if (mct_aVect_nRattr(av(i)) > 0) read(uin) av(i)%rAttr
     end do
     call prep_ice_merge(flux_epbalfact, av(1), av(2), av(3), av(4), av(5), av(6))
     write(uout) av(6)%rAttr
  end do
  close(uout)
  close(uin)
end program run_ice_merge
