! Zhang et al. (2001) particle dry deposition velocity: WRF v4.7.1's
! zhang2001_aer_drydepvel_1 (chem/module_aer_drydep.F:1650-1941, cut
! verbatim by extract.py) on a grid of cells that reaches every branch.
!
! One case per (zhang_optaa, modesurftype, ido_mom0, ido_mom3).  Within a
! case the cells walk every surface type (and two out of range on each
! side), every season (and one out of range on each side), dry and wet,
! and eight number-mode diameters from below the 1 nm clamp to above the
! 1 cm one; the other inputs are drawn from short lists by an integer hash
! of the cell index, each list holding values on both sides of the routine's
! "reality check" clamps.  airkinvisc and freepath are given (positive) on
! some cells and left for the routine to compute (zero or negative) on the
! others; both are written back after the call, as the routine sets them.
!
! Writes fixtures/zhang2001/<case>/ through oracle_io: the inputs, then
!   depvel_0 depvel_3 resist_s0 resist_s3 vsettl_0 vsettl_3
!   airkinvisc_out freepath_out
program run_zhang2001
  use oracle_io
  use smoke_cut_zhang2001
  implicit none
  character(len=1024) :: root

  call oracle_root(root)
  call one_case('zhang2001/optaa1_zhang', 1, 1, 1, 1)
  call one_case('zhang2001/optaa2_zhang', 2, 1, 1, 1)
  call one_case('zhang2001/optaa1_usgs', 1, 2, 1, 1)
  call one_case('zhang2001/optaa2_usgs', 2, 2, 1, 1)
  call one_case('zhang2001/mom0_only', 1, 1, 1, 0)
  call one_case('zhang2001/mom3_only', 2, 2, 0, 1)
  call one_case('zhang2001/no_moments', 1, 0, 0, 0)
  call oracle_close()

contains

  integer function pick(n, salt, m)
    integer, intent(in) :: n, salt, m
    integer(8) :: h
    h = int(n, 8) * 2654435761_8 + int(salt, 8) * 40503_8 + 12345_8
    h = ieor(h, ishft(h, -13))
    h = mod(h * 1103515245_8 + 12345_8, 2147483647_8)
    pick = int(mod(abs(h), int(m, 8))) + 1
  end function pick

  subroutine one_case(name, optaa, modesurf, ido0, ido3)
    character(len=*), intent(in) :: name
    integer, intent(in) :: optaa, modesurf, ido0, ido3
    real, parameter :: dgs(8) = (/ 5.0e-10, 2.0e-8, 1.0e-7, 3.0e-7, &
                                   1.0e-6, 4.0e-6, 2.0e-5, 2.0e-2 /)
    real, parameter :: lnsigs(6) = (/ -0.1, 0.0, 0.3, 0.6931472, 1.2, 1.8 /)
    real, parameter :: denss(6) = (/ 50.0, 1000.0, 1400.0, 1800.0, &
                                     2650.0, 6.0e4 /)
    real, parameter :: temps(6) = (/ 40.0, 230.0, 263.5, 288.15, 309.0, &
                                     360.0 /)
    real, parameter :: rhos(5) = (/ 0.005, 0.41, 0.9, 1.225, 6.0 /)
    real, parameter :: ustars(6) = (/ 1.0e-7, 0.05, 0.21, 0.4, 0.83, 200.0 /)
    real, parameter :: ras(6) = (/ 1.0e-3, 4.0, 18.5, 61.0, 333.0, 2.0e6 /)
    real, parameter :: kinvs(4) = (/ 0.0, -1.0, 1.46e-5, 2.9e-5 /)
    real, parameter :: paths(4) = (/ 0.0, -1.0, 6.5e-8, 1.3e-7 /)
    integer :: ntype, tlo, thi, n, it, is, iw, id, cell
    real, allocatable :: dg(:), lnsig(:), dens(:), temp(:), rho(:), ustar(:)
    real, allocatable :: ra(:), kinv_in(:), path_in(:), kinv_out(:)
    real, allocatable :: path_out(:)
    real, allocatable :: dv0(:), dv3(:), rs0(:), rs3(:), vs0(:), vs3(:)
    integer, allocatable :: season(:), stype(:), wet(:)

    if (modesurf >= 2) then
      tlo = -1; thi = 27
    else
      tlo = -1; thi = 16
    end if
    ntype = thi - tlo + 1
    n = ntype * 8 * 2 * 8
    allocate(dg(n), lnsig(n), dens(n), temp(n), rho(n), ustar(n), ra(n), &
             kinv_in(n), path_in(n), kinv_out(n), path_out(n), dv0(n), &
             dv3(n), rs0(n), rs3(n), vs0(n), vs3(n), season(n), stype(n), &
             wet(n))
    cell = 0
    do it = tlo, thi
      do is = -1, 6
        do iw = 0, 1
          do id = 1, 8
            cell = cell + 1
            stype(cell) = it
            season(cell) = is
            wet(cell) = iw
            dg(cell) = dgs(id)
            lnsig(cell) = lnsigs(pick(cell, 1, 6))
            dens(cell) = denss(pick(cell, 2, 6))
            temp(cell) = temps(pick(cell, 3, 6))
            rho(cell) = rhos(pick(cell, 4, 5))
            ustar(cell) = ustars(pick(cell, 5, 6))
            ra(cell) = ras(pick(cell, 6, 6))
            kinv_in(cell) = kinvs(pick(cell, 7, 4))
            path_in(cell) = paths(pick(cell, 8, 4))
          end do
        end do
      end do
    end do

    do cell = 1, n
      kinv_out(cell) = kinv_in(cell)
      path_out(cell) = path_in(cell)
      call zhang2001_aer_drydepvel_1( &
          dg(cell), lnsig(cell), dens(cell), &
          temp(cell), rho(cell), ustar(cell), ra(cell), &
          optaa, season(cell), &
          stype(cell), modesurf, wet(cell), &
          kinv_out(cell), path_out(cell), &
          ido0, ido3, &
          dv0(cell), dv3(cell), &
          rs0(cell), rs3(cell), &
          vs0(cell), vs3(cell))
    end do

    call oracle_open(name)
    call oracle_put('zhang_optaa', optaa)
    call oracle_put('modesurftype', modesurf)
    call oracle_put('ido_mom0', ido0)
    call oracle_put('ido_mom3', ido3)
    call oracle_put('dg_num', dg)
    call oracle_put('lnsig', lnsig)
    call oracle_put('dens_part', dens)
    call oracle_put('airtemp', temp)
    call oracle_put('airdens', rho)
    call oracle_put('ustar', ustar)
    call oracle_put('resist_a', ra)
    call oracle_put('iseason', season)
    call oracle_put('isurftype', stype)
    call oracle_put('iwetsurf', wet)
    call oracle_put('airkinvisc_in', kinv_in)
    call oracle_put('freepath_in', path_in)
    call oracle_put('depvel_0', dv0)
    call oracle_put('depvel_3', dv3)
    call oracle_put('resist_s0', rs0)
    call oracle_put('resist_s3', rs3)
    call oracle_put('vsettl_0', vs0)
    call oracle_put('vsettl_3', vs3)
    call oracle_put('airkinvisc_out', kinv_out)
    call oracle_put('freepath_out', path_out)
    deallocate(dg, lnsig, dens, temp, rho, ustar, ra, kinv_in, path_in, &
               kinv_out, path_out, dv0, dv3, rs0, rs3, vs0, vs3, season, &
               stype, wet)
  end subroutine one_case

end program run_zhang2001
