! Drive E3SM v3.2.0's seq_frac_set (driver-mct/main/seq_frac_mct.F90, cut
! verbatim with seq_frac_check) and the mapping it calls (seq_map_map and
! seq_map_avNormArr, cut verbatim out of seq_map_mod.F90) on MCT as it is,
! over attribute vectors and mapping weights read from a stream. One task.
!
!   run_seq_frac INPUT OUTPUT
!
! INPUT (little-endian int32 unless named): atm_present, ice_present,
! ocn_present, seq_frac_dead (0/1); the ice-to-ocean mapper's kind and the
! ice-to-atmosphere mapper's kind (0 copy_only, 1 rearrange_only, 2 a sparse
! matrix); lsize_i, lsize_o, lsize_a; then i2x_i, dom_i's data, fractions_i,
! fractions_o and fractions_a, each as the length of its colon-separated
! field list, the list, and its rAttr(nflds, lsize) (r8); then, for each
! mapper of kind 2 (ice-to-ocean first), the element count and the elements'
! global rows, global columns (1-based) and weights (r8), in the mapping
! file's order.
! dom_i's list is its other fields; its coordinates are E3SM's, lat:lon:hgt,
! and its rAttr carries both. The two mappers are built as the coupler builds
! them on one task.
! OUTPUT: fractions_i, fractions_o and fractions_a's rAttr after the call.
! seq_frac_check's lines go to the coupler's log, the standard error; its
! abort stops with status 7 and no OUTPUT.
program run_seq_frac
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_comm_mct, only: CPLID
  use seq_infodata_mod, only: seq_infodata_type
  use seq_map_type_mod, only: seq_map
  use component_type_mod, only: component_type
  use prep_ocn_mod, only: mapper_SFi2o
  use prep_atm_mod, only: mapper_Fi2a
  use seq_frac_mct, only: seq_frac_set, seq_frac_oracle_set_dead
  implicit none
  include 'mpif.h'
  integer, parameter :: nav = 5
  type(mct_aVect), target :: av(nav)
  type(mct_gGrid), target :: dom_i
  type(mct_gsMap), target :: gsmap_i, gsmap_o, gsmap_a
  type(seq_infodata_type) :: infodata
  type(component_type) :: ice
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  integer :: flags(4), kinds(2), lsizes(3), rlen, i, uin, uout, ierr

  call MPI_Init(ierr)
  call mct_world_init(1, MPI_COMM_WORLD, MPI_COMM_WORLD, CPLID)
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flags, kinds, lsizes
  infodata % atm_present = flags(1) /= 0
  infodata % ice_present = flags(2) /= 0
  infodata % ocn_present = flags(3) /= 0
  call seq_frac_oracle_set_dead(flags(4) /= 0)
  ! i2x_i, dom_i%data, fractions_i (ice grid), fractions_o, fractions_a
  do i = 1, nav
     read(uin) rlen
     allocate(character(len=rlen) :: rlist)
     if (rlen > 0) read(uin) rlist
     if (i == 2) then
        call mct_gGrid_init(dom_i, CoordChars='lat:lon:hgt', OtherChars=rlist, lsize=lsizes(1))
        if (mct_aVect_nRattr(dom_i % data) > 0) read(uin) dom_i % data % rAttr
     else
        call mct_aVect_init(av(i), rList=rlist, lsize=lsizes(merge(1, i - 2, i == 1)))
        if (mct_aVect_nRattr(av(i)) > 0) read(uin) av(i) % rAttr
     end if
     deallocate(rlist)
  end do
  call mct_gsMap_init(gsmap_i, [1], [lsizes(1)], 0, MPI_COMM_WORLD, CPLID)
  call mct_gsMap_init(gsmap_o, [1], [lsizes(2)], 0, MPI_COMM_WORLD, CPLID)
  call mct_gsMap_init(gsmap_a, [1], [lsizes(3)], 0, MPI_COMM_WORLD, CPLID)
  allocate(mapper_SFi2o, mapper_Fi2a)
  call build_mapper(mapper_SFi2o, kinds(1), gsmap_i, gsmap_o)
  call build_mapper(mapper_Fi2a, kinds(2), gsmap_i, gsmap_a)
  close(uin)
  ice % dom_cx => dom_i
  ice % c2x_cx => av(1)

  call seq_frac_set(infodata, ice, av(5), av(3), av(4))

  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  do i = 3, nav
     if (mct_aVect_nRattr(av(i)) > 0) write(uout) av(i) % rAttr
  end do
  close(uout)
  call MPI_Finalize(ierr)

contains

  ! seq_map_init_rearrolap (rearrange_only), the same-grid copy (copy_only),
  ! or shr_mct_sMatPInitnc_mapfile's mct_sMatP_Init over the file's elements
  ! (shr_mct_sMatReaddnc keeps a task's rows in the file's order; one task
  ! keeps them all)
  subroutine build_mapper(mapper, kind, gsmap_s, gsmap_d)
    type(seq_map), intent(inout) :: mapper
    integer, intent(in) :: kind
    type(mct_gsMap), intent(in), target :: gsmap_s, gsmap_d
    type(mct_sMat) :: sMat
    integer :: n
    integer, allocatable :: rows(:), cols(:)
    real(r8), allocatable :: wgts(:)
    mapper % copy_only = kind == 0
    mapper % rearrange_only = kind == 1
    mapper % esmf_map = .false.
    mapper % nl_available = .false.
    mapper % nl_conservative = .false.
    mapper % gsmap_s => gsmap_s
    mapper % gsmap_d => gsmap_d
    mapper % mpicom = MPI_COMM_WORLD
    if (kind == 1) call mct_rearr_init(gsmap_s, gsmap_d, MPI_COMM_WORLD, mapper % rearr)
    if (kind == 2) then
       read(uin) n
       allocate(rows(n), cols(n), wgts(n))
       read(uin) rows, cols, wgts
       call mct_sMat_init(sMat, mct_gsMap_gsize(gsmap_d), mct_gsMap_gsize(gsmap_s), n)
       sMat % data % iAttr(mct_sMat_indexIA(sMat, 'grow'), 1:n) = rows
       sMat % data % iAttr(mct_sMat_indexIA(sMat, 'gcol'), 1:n) = cols
       sMat % data % rAttr(mct_sMat_indexRA(sMat, 'weight'), 1:n) = wgts
       call mct_sMatP_Init(mapper % sMatp, sMat, gsmap_s, gsmap_d, 0, MPI_COMM_WORLD, gsmap_s % comp_id)
       call mct_sMat_Clean(sMat)
    end if
  end subroutine build_mapper
end program run_seq_frac
