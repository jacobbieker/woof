! Drive E3SM v3.2.0's atmosphere-ocean flux, seq_flux_init_mct's mask and
! seq_flux_atmocn_mct (driver-mct/main/seq_flux_mct.F90, cut verbatim) over
! shr_flux_mod (share/util/shr_flux_mod.F90, compiled verbatim), on MCT as
! it is, over attribute vectors read from a stream. One task, one call.
!
!   run_flux_atmocn INPUT OUTPUT
!
! INPUT (little-endian int32 unless named): dead_comps, atm_flux_method
! (0 explicit, 1 implicit_stress), atm_gustiness, coldair_outbreak_mod,
! flux_max_iteration, ocn_surface_flux_scheme, flux_diurnal, wav_atm_coup
! (0 none, 1 twoway); lsize (the flux grid's), lsize_a2x, lsize_o2x;
! flux_convergence and seq_flux_atmocn_minwind (r8); then the flux grid's
! domain data (its other fields; the coordinates are E3SM's lat:lon:hgt),
! its fractions, a2x, o2x and xao, each as the length of its colon-separated
! field list, the list, and its rAttr(nflds, lsize) (r8).
! OUTPUT: xao's rAttr after the call. shr_sys_abort stops with status 7 and
! no OUTPUT; its message and the flux's log lines go to the standard error.
program run_flux_atmocn
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_comm_mct, only: CPLID
  use seq_flds_mod, only: atm_flux_method, atm_gustiness, wav_atm_coup
  use seq_infodata_mod, only: seq_infodata_type
  use component_type_mod, only: component_type
  use seq_flux_mct, only: seq_flux_init_mct, seq_flux_atmocn_mct, seq_flux_oracle_set_minwind
  implicit none
  include 'mpif.h'
  integer, parameter :: nav = 5
  type(mct_aVect), target :: av(nav), w2x
  type(mct_gGrid), target :: dom
  type(mct_gsMap), target :: gsmap
  type(seq_infodata_type) :: infodata
  type(component_type) :: comp
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  integer :: flags(8), lsizes(3), rlen, i, uin, uout, ierr
  real(r8) :: reals(2)

  call MPI_Init(ierr)
  call mct_world_init(1, MPI_COMM_WORLD, MPI_COMM_WORLD, CPLID)
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flags, lsizes, reals
  infodata % dead_comps = flags(1) /= 0
  atm_flux_method = merge('implicit_stress', 'explicit       ', flags(2) /= 0)
  atm_gustiness = flags(3) /= 0
  infodata % coldair_outbreak_mod = flags(4) /= 0
  infodata % flux_max_iteration = flags(5)
  infodata % ocn_surface_flux_scheme = flags(6)
  infodata % flux_diurnal = flags(7) /= 0
  wav_atm_coup = merge('twoway', 'none  ', flags(8) /= 0)
  infodata % flux_convergence = reals(1)
  call seq_flux_oracle_set_minwind(reals(2))
  ! the domain's data, the fractions (flux grid), a2x, o2x, xao
  do i = 1, nav
     read(uin) rlen
     allocate(character(len=rlen) :: rlist)
     if (rlen > 0) read(uin) rlist
     if (i == 1) then
        call mct_gGrid_init(dom, CoordChars='lat:lon:hgt', OtherChars=rlist, lsize=lsizes(1))
        if (mct_aVect_nRattr(dom % data) > 0) read(uin) dom % data % rAttr
     else
        call mct_aVect_init(av(i), rList=rlist, lsize=lsizes(merge(1, i - 1, i == 2 .or. i == 5)))
        if (mct_aVect_nRattr(av(i)) > 0 .and. mct_aVect_lsize(av(i)) > 0) read(uin) av(i) % rAttr
     end if
     deallocate(rlist)
  end do
  close(uin)
  call mct_gsMap_init(gsmap, [1], [lsizes(1)], 0, MPI_COMM_WORLD, CPLID)
  ! no wave model: the driver passes w2x as the coupler has it, with no points
  call mct_aVect_init(w2x, rList='Sw_Charn:Sw_Ustar:Sw_Z0', lsize=0)
  comp % dom_cx => dom
  comp % gsmap_cx => gsmap

  call seq_flux_init_mct(comp, av(2))
  call seq_flux_atmocn_mct(infodata, 0, 1800, av(3), av(4), av(5), w2x)

  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  if (mct_aVect_nRattr(av(5)) > 0) write(uout) av(5) % rAttr
  close(uout)
  call MPI_Finalize(ierr)
end program run_flux_atmocn
