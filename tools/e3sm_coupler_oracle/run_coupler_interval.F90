! Drive one E3SM v3.2.0 coupler interval: the coupler's own routines, cut
! verbatim and linked together, in cime_comp_mod's RASM_OPTION1 order with
! aoflux_grid 'ocn', over one task, one instance of each component, and an
! ocean and sea ice on the same grid (the coupler's ice-to-ocean and
! ocean-to-ice mappers copy_only). The atmosphere's export on the ocean grid
! and the components' new exports are scripted from the stream. Per coupler
! step:
!   cime_run_atmocn_setup: prep_ocn_calc_i2x_ox (ice_c2_ocn),
!     seq_flux_atmocn_mct, prep_ocn_mrg and prep_ocn_accum (ocn_prognostic),
!     seq_flux_ocnalb_mct;
!   ocean run alarm: cime_run_ocn_setup_send's prep_ocn_accum_avg
!     (ocn_prognostic);
!   sea-ice run alarm: cime_run_ice_setup_send's prep_ice_calc_o2x_ix
!     (ocn_c2_ice), prep_ice_calc_a2x_ix (atm_c2_ice) and prep_ice_mrg
!     (ice_prognostic); the sea ice runs: its new export from the stream;
!   cime_run_update_fractions: seq_frac_set;
!   ocean next alarm: the ocean's new export from the stream.
!
!   run_coupler_interval INPUT OUTPUT
!
! INPUT (little-endian int32 unless named): the flux's eight switches as
! run_flux_atmocn reads them (dead_comps, atm_flux_method, atm_gustiness,
! coldair_outbreak_mod, flux_max_iteration, ocn_surface_flux_scheme,
! flux_diurnal, wav_atm_coup); ocn_prognostic, ice_prognostic, ice_c2_ocn,
! ocn_c2_ice, atm_c2_ice, flux_albav, seq_frac_dead (0/1) and the
! ice-to-atmosphere mapper's kind (0 copy_only, 1 rearrange_only, 2 a sparse
! matrix); lsize (the ocean's and sea ice's), lsize_a, nsteps;
! flux_convergence, seq_flux_atmocn_minwind, seq_flux_mct_albdif,
! seq_flux_mct_albdir, nextsw_cday and flux_epbalfact (r8); then, each as
! the length of its colon-separated field list, the list and its rAttr(nflds,
! lsize) (r8): the domain's data (its other fields; the coordinates are
! E3SM's lat:lon:hgt), a2x_ox (no rAttr: it comes per step), o2x, i2x,
! xao_ox, fractions_o, fractions_i, fractions_a (lsize_a), r2x_ox, w2x_ox,
! g2x_ox, r2x_ix, g2x_ix, w2x_ix, x2o and x2i; then, for a mapper of kind 2,
! its element count and the elements' global rows, columns (1-based) and
! weights (r8). Per step: the ocean run, sea-ice run and ocean next alarms
! (0/1); nextsw_cday and flux_epbalfact (r8) and a flag (0/1) for new
! r2x_ox, w2x_ox, g2x_ox, r2x_ix, g2x_ix and w2x_ix rAttr, which follow
! when set (each that has fields); a2x_ox's rAttr, then the sea ice's new export (with its run alarm)
! and the ocean's (with its next alarm).
! i2x_ox, a2x_ix and o2x_ix take their sources' field lists and the
! accumulator x2o's, zeroed with a count of 0, as prep_ocn_init and
! prep_ice_init set them up.
! OUTPUT per step: xao_ox, the accumulator's count (int32) and rAttr, x2o,
! x2i, fractions_o, fractions_i and fractions_a (r8). shr_sys_abort stops with
! status 7 and its message on the standard error.
program run_coupler_interval
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use mct_mod
  use seq_comm_mct, only: CPLID
  use seq_flds_mod, only: atm_flux_method, atm_gustiness, wav_atm_coup
  use seq_infodata_mod, only: seq_infodata_type
  use seq_map_type_mod, only: seq_map
  use component_type_mod, only: ocn, ice
  use prep_ocn_mod, only: prep_ocn_mrg, prep_ocn_accum, prep_ocn_accum_avg, prep_ocn_calc_i2x_ox, &
       mapper_SFi2o, a2x_ox, r2x_ox, i2x_ox, g2x_ox, w2x_ox, x2oacc_ox, x2oacc_ox_cnt, x2o_average
  use prep_ice_mod, only: prep_ice_mrg, prep_ice_calc_a2x_ix, prep_ice_calc_o2x_ix, &
       mapper_SFo2i, a2x_ix, o2x_ix, g2x_ix, r2x_ix, w2x_ix
  use prep_atm_mod, only: mapper_Fi2a
  use seq_frac_mct, only: seq_frac_set, seq_frac_oracle_set_dead
  use seq_flux_mct, only: seq_flux_init_mct, seq_flux_atmocn_mct, seq_flux_ocnalb_mct, &
       seq_flux_oracle_set_minwind, seq_flux_oracle_set_albedos
  implicit none
  include 'mpif.h'
  ! the vectors the stream names, in its order
  integer, parameter :: A2X = 1, O2X = 2, I2X = 3, XAO = 4, FRO = 5, FRI = 6, FRA = 7, X2O = 14, X2I = 15
  integer, parameter :: nav = 15
  type(mct_aVect), target :: av(nav)
  type(mct_aVect), target :: xao_ox(1), fractions_ox(1)
  type(mct_gGrid), target :: dom
  type(mct_gsMap), target :: gsmap_o, gsmap_a
  type(seq_infodata_type) :: infodata
  character(len=4096) :: path_in, path_out
  character(len=:), allocatable :: rlist
  integer :: flags(8), switches(8), sizes(3), alarms(3), rlen, i, istep, uin, uout, ierr
  real(r8) :: reals(6), step_reals(2)
  integer :: iside
  logical :: ocn_prognostic, ice_prognostic, ice_c2_ocn, ocn_c2_ice, atm_c2_ice

  call MPI_Init(ierr)
  call mct_world_init(1, MPI_COMM_WORLD, MPI_COMM_WORLD, CPLID)
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) flags, switches, sizes, reals
  infodata % dead_comps = flags(1) /= 0
  atm_flux_method = merge('implicit_stress', 'explicit       ', flags(2) /= 0)
  atm_gustiness = flags(3) /= 0
  infodata % coldair_outbreak_mod = flags(4) /= 0
  infodata % flux_max_iteration = flags(5)
  infodata % ocn_surface_flux_scheme = flags(6)
  infodata % flux_diurnal = flags(7) /= 0
  wav_atm_coup = merge('twoway', 'none  ', flags(8) /= 0)
  ocn_prognostic = switches(1) /= 0
  ice_prognostic = switches(2) /= 0
  ice_c2_ocn = switches(3) /= 0
  ocn_c2_ice = switches(4) /= 0
  atm_c2_ice = switches(5) /= 0
  infodata % ocn_prognostic = ocn_prognostic
  infodata % flux_albav = switches(6) /= 0
  call seq_frac_oracle_set_dead(switches(7) /= 0)
  infodata % flux_convergence = reals(1)
  call seq_flux_oracle_set_minwind(reals(2))
  call seq_flux_oracle_set_albedos(reals(3), reals(4))
  infodata % nextsw_cday = reals(5)
  infodata % flux_epbalfact = reals(6)

  ! the domain's data, then the vectors
  read(uin) rlen
  allocate(character(len=rlen) :: rlist)
  if (rlen > 0) read(uin) rlist
  call mct_gGrid_init(dom, CoordChars='lat:lon:hgt', OtherChars=rlist, lsize=sizes(1))
  if (mct_aVect_nRattr(dom % data) > 0) read(uin) dom % data % rAttr
  deallocate(rlist)
  do i = 1, nav
     read(uin) rlen
     allocate(character(len=rlen) :: rlist)
     if (rlen > 0) read(uin) rlist
     call mct_aVect_init(av(i), rList=rlist, lsize=sizes(merge(2, 1, i == FRA)))
     if (i /= A2X .and. mct_aVect_nRattr(av(i)) > 0) read(uin) av(i) % rAttr
     deallocate(rlist)
  end do
  call mct_gsMap_init(gsmap_o, [1], [sizes(1)], 0, MPI_COMM_WORLD, CPLID)
  call mct_gsMap_init(gsmap_a, [1], [sizes(2)], 0, MPI_COMM_WORLD, CPLID)

  ! the components as the driver holds them: the ocean and the sea ice on one grid
  ocn(1) % dom_cx => dom
  ocn(1) % gsmap_cx => gsmap_o
  ocn(1) % c2x_cx => av(O2X)
  ocn(1) % x2c_cx => av(X2O)
  ice(1) % dom_cx => dom
  ice(1) % gsmap_cx => gsmap_o
  ice(1) % c2x_cx => av(I2X)
  ice(1) % x2c_cx => av(X2I)
  call mct_aVect_init(xao_ox(1), av(XAO), sizes(1))
  call mct_aVect_copy(av(XAO), xao_ox(1))
  call mct_aVect_init(fractions_ox(1), av(FRO), sizes(1))
  call mct_aVect_copy(av(FRO), fractions_ox(1))
  ! prep_ocn_init's and prep_ice_init's vectors and mappers
  allocate(a2x_ox(1), r2x_ox(1), i2x_ox(1), g2x_ox(1), w2x_ox(1), x2oacc_ox(1))
  call mct_aVect_init(a2x_ox(1), av(A2X), sizes(1))
  call mct_aVect_zero(a2x_ox(1))
  call mct_aVect_init(i2x_ox(1), av(I2X), sizes(1))
  call mct_aVect_zero(i2x_ox(1))
  call take(r2x_ox(1), av(8))
  call take(w2x_ox(1), av(9))
  call take(g2x_ox(1), av(10))
  call mct_aVect_init(x2oacc_ox(1), av(X2O), sizes(1))
  call mct_aVect_zero(x2oacc_ox(1))
  x2oacc_ox_cnt = 0
  x2o_average = .false.
  allocate(a2x_ix(1), o2x_ix(1), r2x_ix(1), g2x_ix(1), w2x_ix(1))
  call mct_aVect_init(a2x_ix(1), av(A2X), sizes(1))
  call mct_aVect_zero(a2x_ix(1))
  call mct_aVect_init(o2x_ix(1), av(O2X), sizes(1))
  call mct_aVect_zero(o2x_ix(1))
  call take(r2x_ix(1), av(11))
  call take(g2x_ix(1), av(12))
  call take(w2x_ix(1), av(13))
  allocate(mapper_SFi2o, mapper_SFo2i, mapper_Fi2a)
  call build_mapper(mapper_SFi2o, 0, gsmap_o, gsmap_o)
  call build_mapper(mapper_SFo2i, 0, gsmap_o, gsmap_o)
  call build_mapper(mapper_Fi2a, switches(8), gsmap_o, gsmap_a)

  call seq_flux_init_mct(ocn(1), fractions_ox(1))

  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  do istep = 1, sizes(3)
     read(uin) alarms
     ! this step's infodata (the atmosphere's nextsw_cday, flux_epbalfact) and,
     ! when flagged, the runoff, land-ice and wave exports to the ocean and sea ice
     read(uin) step_reals, iside
     infodata % nextsw_cday = step_reals(1)
     infodata % flux_epbalfact = step_reals(2)
     if (iside /= 0) then
        call read_side(r2x_ox(1))
        call read_side(w2x_ox(1))
        call read_side(g2x_ox(1))
        call read_side(r2x_ix(1))
        call read_side(g2x_ix(1))
        call read_side(w2x_ix(1))
     end if
     read(uin) a2x_ox(1) % rAttr
     ! cime_run_atmocn_setup
     if (ocn_prognostic .and. ice_c2_ocn) call prep_ocn_calc_i2x_ox(timer='CPL:atmocnp_ice2ocn')
     call seq_flux_atmocn_mct(infodata, 0, 1800, a2x_ox(1), av(O2X), xao_ox(1), w2x_ox(1))
     if (ocn_prognostic) then
        call prep_ocn_mrg(infodata, fractions_ox, xao_ox=xao_ox, timer_mrg='CPL:atmocnp_mrgx2o')
        call prep_ocn_accum(timer='CPL:atmocnp_accum')
     end if
     call seq_flux_ocnalb_mct(infodata, ocn(1), a2x_ox(1), fractions_ox(1), xao_ox(1))
     ! cime_run_ocn_setup_send
     if (alarms(1) /= 0 .and. ocn_prognostic) call prep_ocn_accum_avg(timer_accum='CPL:ocnprep_avg')
     ! cime_run_ice_setup_send, the sea ice's run and cime_run_ice_recv_post
     if (alarms(2) /= 0) then
        if (ice_prognostic) then
           if (ocn_c2_ice) call prep_ice_calc_o2x_ix(timer='CPL:iceprep_ocn2ice')
           if (atm_c2_ice) call prep_ice_calc_a2x_ix(a2x_ox, timer='CPL:iceprep_atm2ice')
           call prep_ice_mrg(infodata, timer_mrg='CPL:iceprep_mrgx2i')
        end if
        if (mct_aVect_nRattr(av(I2X)) > 0) read(uin) av(I2X) % rAttr
     end if
     ! cime_run_update_fractions
     call seq_frac_set(infodata, ice(1), av(FRA), av(FRI), fractions_ox(1))
     ! the ocean's run ends: cime_run_ocn_recv_post
     if (alarms(3) /= 0 .and. mct_aVect_nRattr(av(O2X)) > 0) read(uin) av(O2X) % rAttr
     write(uout) xao_ox(1) % rAttr
     write(uout) x2oacc_ox_cnt
     write(uout) x2oacc_ox(1) % rAttr
     write(uout) av(X2O) % rAttr
     write(uout) av(X2I) % rAttr
     write(uout) fractions_ox(1) % rAttr
     write(uout) av(FRI) % rAttr
     write(uout) av(FRA) % rAttr
  end do
  close(uout)
  close(uin)
  call MPI_Finalize(ierr)

contains

  ! a side export's new values, if it has fields
  subroutine read_side(v)
    type(mct_aVect), intent(inout) :: v
    if (mct_aVect_nRattr(v) > 0) read(uin) v % rAttr
  end subroutine read_side

  ! a vector the driver holds for an absent component: its list and values
  ! (MCT cannot initialise a vector off one with no fields: the list again)
  subroutine take(dst, src)
    type(mct_aVect), intent(inout) :: dst
    type(mct_aVect), intent(in) :: src
    if (mct_aVect_nRattr(src) == 0) then
       call mct_aVect_init(dst, rList='', lsize=sizes(1))
       return
    end if
    call mct_aVect_init(dst, src, sizes(1))
    call mct_aVect_copy(src, dst)
  end subroutine take

  ! as run_seq_frac.F90 builds them
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
end program run_coupler_interval
