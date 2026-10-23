! WRF-SFIRE's fire smoke tracer emission (tracer_opt = 3), from WRF's own
! add_fire_tracer_emissions (phys/module_fr_fire_atm.F:15-106), compiled
! byte-unmodified.
!
! The routine's i loop is `do i=i_st,i_st` (module_fr_fire_atm.F:81): on a
! tile it emits into ONE atmospheric column, i = max(its, ids+1), and leaves
! the rest of the tile untouched.  So the driver calls it once per column,
! each call on a one-column tile its = ite = i with that column's fire-grid
! tile, which is the routine's own arithmetic for every column of the
! interior; the woof port applies it to every interior column at once
! (PROVENANCE.md, "held out").  j runs over max(jts, jds+1)..min(jte, jde-2)
! as WRF's does.  fire_smk_scheme = 0 only: scheme 1's tg_dist reads z_w
! before any statement sets it (module_fr_fire_atm.F:342).
!
! Writes fixtures/emission/caseNN/ (oracle_io.F90).
program run_emission
  use oracle_io
  use module_fr_fire_atm, only: add_fire_tracer_emissions
  implicit none
  character(len=1024) :: root
  integer, save :: seed = 20261022

  call oracle_root(root)
  call one_case('emission/case01', 6, 5, 4, 4, 4, 0.02, 0)
  call one_case('emission/case02', 7, 6, 3, 3, 2, 0.015, 1)
  call one_case('emission/case03', 5, 4, 5, 10, 10, 0.009, 2)
  call one_case('emission/case04', 4, 4, 2, 1, 1, 0.02, 3)

contains

  ! A uniform deviate in [0, 1) from a 31-bit LCG: the inputs are written
  ! out with the outputs, so their generator only has to be repeatable.
  real function uniform()
    seed = mod(1103515245 * int(seed, 8) + 12345, 2147483648_8)
    uniform = real(seed) / 2147483648.0
  end function uniform

  subroutine one_case(name, nx, ny, nz, ir, jr, fire_tracer_smoke, flavour)
    character(len=*), intent(in) :: name
    integer, intent(in) :: nx, ny, nz, ir, jr, flavour
    real, intent(in) :: fire_tracer_smoke
    real :: rho(nx, nz, ny), dz8w(nx, nz, ny), z_at_w(nx, nz, ny)
    real :: tracer(nx, nz, ny, 1), tracer_in(nx, nz, ny)
    real :: zs(nx, ny)
    real :: burnt_area_dt(nx * ir, ny * jr), fgip(nx * ir, ny * jr)
    integer :: i, j, k, ifts, ifte, ifme, jfme
    real :: u

    ifme = nx * ir
    jfme = ny * jr
    do j = 1, ny
      do k = 1, nz
        do i = 1, nx
          rho(i, k, j) = 1.25 - 0.1 * real(k - 1) - 0.05 * uniform()
          dz8w(i, k, j) = 20.0 + 40.0 * uniform() + 15.0 * real(k - 1)
          tracer(i, k, j, 1) = 0.0
          if (flavour >= 1) tracer(i, k, j, 1) = 1.0e-3 * uniform()
        end do
      end do
    end do
    tracer_in = tracer(:, :, :, 1)
    z_at_w = 0.0
    zs = 0.0
    do j = 1, jfme
      do i = 1, ifme
        u = uniform()
        burnt_area_dt(i, j) = 0.0
        ! Most fire cells unburnt, some partly, a few wholly burnt this step.
        if (u > 0.55) burnt_area_dt(i, j) = uniform() * 0.08
        if (u > 0.95) burnt_area_dt(i, j) = 1.0
        if (flavour == 2 .and. u > 0.3) burnt_area_dt(i, j) = uniform()
        fgip(i, j) = 0.1 + 30.0 * uniform()
        if (flavour == 3) fgip(i, j) = 1.0e-3 + 1.0e3 * uniform() ** 3
      end do
    end do

    do i = 2, nx - 1
      ifts = (i - 1) * ir + 1
      ifte = i * ir
      call add_fire_tracer_emissions(                                  &
           3, 1.0, 100.0, 100.0,                                       &
           1, ifme, 1, jfme,                                           &
           ifts, ifte, 1, jfme,                                        &
           1, nx + 1, 1, nz + 1, 1, ny + 1,                            &
           1, nx, 1, nz, 1, ny,                                        &
           i, i, 1, nz, 1, ny,                                         &
           rho, dz8w,                                                  &
           burnt_area_dt, fgip,                                        &
           tracer, fire_tracer_smoke,                                  &
           0, 0.0, 50.0, 0.0, zs, z_at_w)
    end do

    call oracle_open(name)
    call oracle_put('ir', ir)
    call oracle_put('jr', jr)
    call oracle_put('fire_tracer_smoke', fire_tracer_smoke)
    call oracle_put('rho', rho)
    call oracle_put('dz8w', dz8w)
    call oracle_put('burnt_area_dt', burnt_area_dt)
    call oracle_put('fgip', fgip)
    call oracle_put('tracer_in', tracer_in)
    call oracle_put('tracer_out', tracer(:, :, :, 1))
    call oracle_close()
  end subroutine one_case

end program run_emission
