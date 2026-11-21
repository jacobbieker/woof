! Drive E3SM v3.2.0's ocean albedos, seq_flux_ocnalb_mct
! (driver-mct/main/seq_flux_mct.F90, cut verbatim), over attribute vectors
! read from a stream. One task; ncalls calls (the first one does the
! lookups and the checks).
!
!   run_flux_ocnalb INPUT OUTPUT
!
! INPUT (little-endian int32 unless named): flux_albav, lsize, ncalls;
! seq_flux_mct_albdif, seq_flux_mct_albdir (r8); then the flux grid's domain
! (its other fields; the coordinates are E3SM's lat:lon:hgt), a2x, fractions
! and xao, each as the length of its colon-separated field list and the
! list; then per call nextsw_cday (r8) and each vector's rAttr(nflds, lsize)
! (r8; the domain's only on the first call).
! OUTPUT: per call, xao's rAttr then the fractions' after the call.
! shr_sys_abort stops with status 7 and its message on the standard error.
program run_flux_ocnalb
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_comm_mct, only: CPLID
  use seq_infodata_mod, only: seq_infodata_type
  use component_type_mod, only: component_type
  use seq_flux_mct, only: seq_flux_ocnalb_mct, seq_flux_oracle_set_albedos
  implicit none
  include 'mpif.h'
  integer, parameter :: nav = 4
  type(mct_aVect), target :: av(nav)
  type(mct_gGrid), target :: dom
  type(seq_infodata_type) :: infodata
  type(component_type) :: ocn
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  integer :: flags(3), rlen, i, icall, uin, uout, ierr
  real(r8) :: albedos(2), cday

  call MPI_Init(ierr)
  call mct_world_init(1, MPI_COMM_WORLD, MPI_COMM_WORLD, CPLID)
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flags, albedos
  infodata % flux_albav = flags(1) /= 0
  call seq_flux_oracle_set_albedos(albedos(1), albedos(2))
  do i = 1, nav
     read(uin) rlen
     allocate(character(len=rlen) :: rlist)
     if (rlen > 0) read(uin) rlist
     if (i == 1) then
        call mct_gGrid_init(dom, CoordChars='lat:lon:hgt', OtherChars=rlist, lsize=flags(2))
     else
        call mct_aVect_init(av(i), rList=rlist, lsize=flags(2))
     end if
     deallocate(rlist)
  end do
  ocn % dom_cx => dom
  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  do icall = 1, flags(3)
     read(uin) cday
     infodata % nextsw_cday = cday
     if (icall == 1) read(uin) dom % data % rAttr
     do i = 2, nav
        if (mct_aVect_nRattr(av(i)) > 0) read(uin) av(i) % rAttr
     end do
     call seq_flux_ocnalb_mct(infodata, ocn, av(2), av(3), av(4))
     write(uout) av(4) % rAttr
     write(uout) av(3) % rAttr
  end do
  close(uout)
  close(uin)
  call MPI_Finalize(ierr)
end program run_flux_ocnalb
