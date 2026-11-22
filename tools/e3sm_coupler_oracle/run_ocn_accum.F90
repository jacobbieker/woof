! Drive E3SM v3.2.0's ocean-input accumulation, prep_ocn_accum and
! prep_ocn_accum_avg (driver-mct/main/prep_ocn_mod.F90, cut verbatim), over
! mct_aVect_accum and mct_aVect_avg (share/util/mct_mod.F90, cut verbatim)
! and MCT as it is. One task, one ocean instance.
!
!   run_ocn_accum INPUT OUTPUT
!
! INPUT (little-endian int32 unless named): lsize, nops; x2o's field list
! (its length and the colon-separated list); x2o's starting rAttr(nflds,
! lsize) (r8); then per operation its code, 1 for prep_ocn_accum (followed
! by x2o's rAttr, r8, which the merge would have written) or 2 for
! prep_ocn_accum_avg.
! The accumulator is set up as prep_ocn_init sets it up (x2o's fields,
! zeroed, a count of 0).
! OUTPUT: per operation the count (int32), the accumulator's rAttr and x2o's
! (r8) after it.
program run_ocn_accum
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_comm_mct, only: CPLID
  use component_type_mod, only: ocn
  use prep_ocn_mod, only: prep_ocn_accum, prep_ocn_accum_avg, x2oacc_ox, x2oacc_ox_cnt
  implicit none
  include 'mpif.h'
  type(mct_aVect), target :: x2o
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  integer :: sizes(2), rlen, iop, op, uin, uout, ierr

  call MPI_Init(ierr)
  call mct_world_init(1, MPI_COMM_WORLD, MPI_COMM_WORLD, CPLID)
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) sizes, rlen
  allocate(character(len=rlen) :: rlist)
  read(uin) rlist
  call mct_aVect_init(x2o, rList=rlist, lsize=sizes(1))
  read(uin) x2o % rAttr
  ocn(1) % x2c_cx => x2o
  ! prep_ocn_init's lines for the accumulator
  allocate(x2oacc_ox(1))
  call mct_avect_init(x2oacc_ox(1), x2o, sizes(1))
  call mct_aVect_zero(x2oacc_ox(1))
  x2oacc_ox_cnt = 0
  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  do iop = 1, sizes(2)
     read(uin) op
     if (op == 1) then
        read(uin) x2o % rAttr
        call prep_ocn_accum(timer='CPL:atmocnp_accum')
     else
        call prep_ocn_accum_avg(timer_accum='CPL:ocnprep_avg')
     end if
     write(uout) x2oacc_ox_cnt
     write(uout) x2oacc_ox(1) % rAttr
     write(uout) x2o % rAttr
  end do
  close(uout)
  close(uin)
  call MPI_Finalize(ierr)
end program run_ocn_accum
