! Drive E3SM v3.2.0's prep_ocn_merge (driver-mct/main/prep_ocn_mod.F90, cut
! verbatim) over attribute vectors read from a stream.
!
!   run_ocean_merge INPUT OUTPUT
!
! INPUT (little-endian): flux_epbalfact (r8); seq_flds_i2o_per_cat,
! rof2ocn_nutrients (int32 0/1); wav_ocn_coup (8 characters); lsize; ncalls;
! then, for a2x, i2x, r2x, w2x, g2x, xao, fractions and x2o in that order, the
! length of the colon-separated field list and the list.  Then, per call, each
! vector's rAttr(nflds, lsize) as r8.  OUTPUT: x2o's rAttr after each call.
program run_ocean_merge
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_flds_mod, only: wav_ocn_coup, rof2ocn_nutrients, seq_flds_i2o_per_cat
  use seq_comm_mct, only: logunit
  use prep_ocn_merge_cut, only: prep_ocn_merge
  implicit none
  integer, parameter :: nav = 8
  type(mct_aVect) :: av(nav)
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  character(len=8) :: wav
  real(r8) :: flux_epbalfact
  integer :: per_cat, nutrients, lsize, ncalls, rlen, i, icall, uin, uout

  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flux_epbalfact, per_cat, nutrients, wav, lsize, ncalls
  seq_flds_i2o_per_cat = per_cat /= 0
  rof2ocn_nutrients = nutrients /= 0
  wav_ocn_coup = trim(wav)
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
     call prep_ocn_merge(flux_epbalfact, av(1), av(2), av(3), av(4), av(5), av(6), av(7), av(8))
     write(uout) av(8)%rAttr
  end do
  close(uout)
  close(uin)
end program run_ocean_merge
