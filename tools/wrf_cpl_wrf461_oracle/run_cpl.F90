program run_wrf_cpl_oracle
  ! Drive WRF v4.6.1's cpl_snd2 (what WRF hands an OASIS-coupled ocean every
  ! step: EVAP-PRECIP, SURF_NET_SOLAR, SURF_NET_NON-SOLAR, TAUX, TAUY,
  ! TAUMOD) and cpl_rcv (how WRF merges what the ocean hands back -- SST with
  ! the input SST under the coupling mask, UOCE and VOCE onto zero) over a row
  ! of independent columns, and dump every input and every output as CSV.
  ! Two external domains, so cpl_rcv's SUM over the mask and its per-domain
  ! accumulation order are both exercised.
  !
  ! Every value is either a word this program placed in an input array before
  ! the calls or a word cpl_snd2 / cpl_rcv placed in an output array.  The
  ! inputs are spread by integer hashes, not libm, plus named edge columns.
  use cpl_harness
  implicit none

  integer, parameter :: ncase = 64
  integer, parameter :: max_edom = 2
  type(domain), pointer :: grid
  real, dimension(ncase, max_edom, 1) :: cplmask
  real, dimension(ncase, 1) :: sst_input, sst_out, uoce_out, voce_out
  real, dimension(ncase, max_edom) :: rsst, ruoce, rvoce
  character(len=1024) :: out_path
  integer :: i, u

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) error stop 'usage: run_cpl OUT.csv'

  allocate(grid)
  grid%id = 1
  grid%dt = 60.0
  allocate(grid%qfx(ncase,1), grid%raincv(ncase,1), grid%rainncv(ncase,1))
  allocate(grid%gsw(ncase,1), grid%glw(ncase,1), grid%emiss(ncase,1))
  allocate(grid%sst(ncase,1), grid%lh(ncase,1), grid%hfx(ncase,1))
  allocate(grid%uoce(ncase,1), grid%voce(ncase,1), grid%ust(ncase,1))
  allocate(grid%u_phy(ncase,1,1), grid%v_phy(ncase,1,1), grid%rho(ncase,1,1))
  allocate(sent(ncase,1,nfld), received(ncase,1,max_edom))
  sent = -999.0

  do i = 1, ncase
    grid%qfx(i,1) = -5.0e-5 + 2.5e-4 * frac(i, 37)
    grid%raincv(i,1) = 4.0 * frac(i, 53) * merge(1.0, 0.0, mod(i, 3) == 0)
    grid%rainncv(i,1) = 6.0 * frac(i, 61) * merge(1.0, 0.0, mod(i, 4) /= 1)
    grid%gsw(i,1) = 1000.0 * frac(i, 41) * merge(1.0, 0.0, mod(i, 5) /= 0)
    grid%glw(i,1) = 150.0 + 300.0 * frac(i, 43)
    grid%emiss(i,1) = 0.95 + 0.05 * frac(i, 47)
    grid%sst(i,1) = 271.0 + 34.0 * frac(i, 59)
    grid%lh(i,1) = -50.0 + 650.0 * frac(i, 67)
    grid%hfx(i,1) = -100.0 + 400.0 * frac(i, 71)
    grid%uoce(i,1) = -2.0 + 4.0 * frac(i, 73)
    grid%voce(i,1) = -2.0 + 4.0 * frac(i, 79)
    grid%u_phy(i,1,1) = -30.0 + 60.0 * frac(i, 83)
    grid%v_phy(i,1,1) = -30.0 + 60.0 * frac(i, 89)
    grid%rho(i,1,1) = 1.0 + 0.35 * frac(i, 97)
    grid%ust(i,1) = 1.2 * frac(i, 101)
    sst_input(i,1) = 271.0 + 34.0 * frac(i, 103)
    rsst(i,1) = 271.0 + 34.0 * frac(i, 107)
    rsst(i,2) = 271.0 + 34.0 * frac(i, 109)
    ruoce(i,1) = -2.0 + 4.0 * frac(i, 113)
    ruoce(i,2) = -2.0 + 4.0 * frac(i, 127)
    rvoce(i,1) = -2.0 + 4.0 * frac(i, 131)
    rvoce(i,2) = -2.0 + 4.0 * frac(i, 137)
    ! Coupling masks: none, either domain alone, both, and partial overlaps,
    ! every pair summing to at most 1 as cpl_defdomain requires.
    select case (mod(i, 8))
    case (0); cplmask(i,1,1) = 0.0;  cplmask(i,2,1) = 0.0
    case (1); cplmask(i,1,1) = 1.0;  cplmask(i,2,1) = 0.0
    case (2); cplmask(i,1,1) = 0.0;  cplmask(i,2,1) = 1.0
    case (3); cplmask(i,1,1) = 0.5;  cplmask(i,2,1) = 0.5
    case (4); cplmask(i,1,1) = 0.3;  cplmask(i,2,1) = 0.2
    case (5); cplmask(i,1,1) = 0.25; cplmask(i,2,1) = 0.75
    case (6); cplmask(i,1,1) = 0.1;  cplmask(i,2,1) = 0.0
    case default; cplmask(i,1,1) = frac(i, 139); cplmask(i,2,1) = 1.0 - cplmask(i,1,1)
    end select
  end do

  ! Named edge columns.
  ! 1-2: no relative wind (atmosphere moving with the ocean), so WSPD takes
  !      its 1.e-7 floor and TAUX/TAUY are exactly zero.
  grid%u_phy(1,1,1) = grid%uoce(1,1); grid%v_phy(1,1,1) = grid%voce(1,1)
  grid%u_phy(2,1,1) = 0.0; grid%v_phy(2,1,1) = 0.0
  grid%uoce(2,1) = 0.0; grid%voce(2,1) = 0.0
  ! 3: relative wind below the floor, over a still ocean so float32 keeps
  !    the 3.e-8 offset.
  grid%uoce(3,1) = 0.0; grid%voce(3,1) = 0.0
  grid%u_phy(3,1,1) = 3.0e-8
  grid%v_phy(3,1,1) = 0.0
  ! 4: no friction velocity, so TAUMOD = 0.
  grid%ust(4,1) = 0.0
  ! 5: night, no precipitation, dewfall (negative QFX).
  grid%gsw(5,1) = 0.0; grid%raincv(5,1) = 0.0; grid%rainncv(5,1) = 0.0
  grid%qfx(5,1) = -3.0e-5
  ! 6: a cloudburst, EVAP-PRECIP strongly negative.
  grid%raincv(6,1) = 25.0; grid%rainncv(6,1) = 40.0
  ! 7: SST at freezing seawater, 8: tropical warm pool.
  grid%sst(7,1) = 271.35; grid%sst(8,1) = 305.0
  ! 9: storm-force wind over a strong current.
  grid%u_phy(9,1,1) = 45.0; grid%v_phy(9,1,1) = -38.0
  grid%uoce(9,1) = 2.5; grid%voce(9,1) = -1.5; grid%ust(9,1) = 2.4

  do i = 1, max_edom
    received(:,1,i) = rsst(:,i)
  end do
  call cpl_snd2( grid, max_edom, 1, ncase, 1, 1, 1, 1,   &
                 1, ncase, 1, 1, 1, 1,                   &
                 1, ncase, 1, 1, 1, 1 )
  call cpl_rcv( 1, 'SST', 1, ncase, 1, 1, 1, 1,          &
                1, ncase, 1, 1, 1, 1,                    &
                1, ncase, 1, 1, 1, 1,                    &
                max_edom, cplmask, sst_out, sst_input )
  do i = 1, max_edom
    received(:,1,i) = ruoce(:,i)
  end do
  call cpl_rcv( 1, 'UOCE', 1, ncase, 1, 1, 1, 1,         &
                1, ncase, 1, 1, 1, 1,                    &
                1, ncase, 1, 1, 1, 1,                    &
                max_edom, cplmask, uoce_out )
  do i = 1, max_edom
    received(:,1,i) = rvoce(:,i)
  end do
  call cpl_rcv( 1, 'VOCE', 1, ncase, 1, 1, 1, 1,         &
                1, ncase, 1, 1, 1, 1,                    &
                1, ncase, 1, 1, 1, 1,                    &
                max_edom, cplmask, voce_out )

  open(newunit=u, file=trim(out_path), status='replace', action='write')
  write(u, '(A)') 'case,dt,qfx,raincv,rainncv,gsw,glw,emiss,sst,lh,hfx,' //   &
    'u_phy,v_phy,uoce,voce,rho,ust,cplmask1,cplmask2,sst_input,' //            &
    'rcv_sst1,rcv_sst2,rcv_uoce1,rcv_uoce2,rcv_voce1,rcv_voce2,' //            &
    'evap_precip,surf_net_solar,surf_net_non_solar,taux,tauy,taumod,' //       &
    'sst_out,uoce_out,voce_out'
  do i = 1, ncase
    write(u, '(I0)', advance='no') i
    write(u, '(25(",",ES24.16E3))', advance='no')                              &
      grid%dt, grid%qfx(i,1), grid%raincv(i,1), grid%rainncv(i,1),             &
      grid%gsw(i,1), grid%glw(i,1), grid%emiss(i,1), grid%sst(i,1),            &
      grid%lh(i,1), grid%hfx(i,1), grid%u_phy(i,1,1), grid%v_phy(i,1,1),       &
      grid%uoce(i,1), grid%voce(i,1), grid%rho(i,1,1), grid%ust(i,1),          &
      cplmask(i,1,1), cplmask(i,2,1), sst_input(i,1),                          &
      rsst(i,1), rsst(i,2), ruoce(i,1), ruoce(i,2), rvoce(i,1), rvoce(i,2)
    write(u, '(9(",",ES24.16E3))')                                             &
      sent(i,1,1), sent(i,1,2), sent(i,1,3), sent(i,1,4), sent(i,1,5),         &
      sent(i,1,6), sst_out(i,1), uoce_out(i,1), voce_out(i,1)
  end do
  close(u)

contains

  ! A value in [0, 1): (i * p) mod 1009 / 1009, an integer hash with no libm.
  real function frac(i, p)
    integer, intent(in) :: i, p
    frac = real(mod(i * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_cpl_oracle
