! The fire-mesh oracle: WRF v4.7.1's own terrain refinement and slope.
!
! For each case, an atmospheric terrain ZS on an nx x ny mass grid is
! refined onto the nx*ir x ny*jr fire mesh by interpolate_z2fire
! (module_fr_fire_util.F, compiled whole and unmodified), as WRF-SFIRE
! calls it for every atmosphere-to-fire field (flag_z0 = 0: no clamp), on
! one tile that is the whole domain.  The slope is module_fr_fire_model.F's
! (wrf_fire_slope_cut, cut verbatim by extract.py) at the spacing of
! module_fr_fire_driver.F:566 (dxf = dx/ir).
!
! Case terrains reach every branch: even ratios (no fire node on a cell
! edge), odd ratios (every edge node written twice, the later write kept),
! unit ratio, unequal ratios and spacings, sea inside the domain, a shore
! on the domain edge (WRF's continuation goes below zero past it), and
! terrain below sea level, and two nested domains on their own grids.
! Written per case: zs (nx, ny), zsf, dzdxf and dzdyf (nx*ir, ny*jr) on the
! fire domain 1..ifde, 1..jfde, and the scalars ir, jr, dx, dy, dxf, dyf.
program run_fire_mesh
  use oracle_io
  implicit none
  character(len=1024) :: root

  call oracle_root(root)
  call run_case('hills_even', 7, 5, 4, 4, 1000.0, 1000.0, 1)
  call run_case('odd_ratio', 6, 4, 3, 3, 1500.0, 1500.0, 2)
  call run_case('coast', 8, 6, 5, 2, 333.3, 500.0, 3)
  call run_case('unit_ratio', 5, 6, 1, 1, 3000.0, 3000.0, 2)
  call run_case('fine_ratio', 4, 3, 10, 10, 1000.0, 1000.0, 2)
  call run_case('below_sea', 3, 2, 2, 7, 250.0, 125.0, 4)
  call run_case('coast_edge', 6, 5, 3, 4, 750.0, 900.0, 5)
  ! Nested domains: a child's own terrain on its own grid (the parent's
  ! terrain sampled at the child's mass points, (ips-0.5) + (i-0.5)/ratio
  ! in parent coordinates), at the child's spacing dx/ratio and its own
  ! fire-mesh ratio.
  call run_case('nest_ratio3', 9, 7, 4, 4, 1000.0 / 3.0, 1000.0 / 3.0, 6)
  call run_case('nest_ratio5', 10, 6, 3, 2, 750.0 / 5.0, 900.0 / 5.0, 7)
  call oracle_close()

contains

  subroutine terrain(kind, nx, ny, zs)
    integer, intent(in) :: kind, nx, ny
    real, intent(out) :: zs(nx, ny)
    integer :: i, j
    integer(8) :: state
    state = 20261022_8
    do j = 1, ny
      do i = 1, nx
        select case (kind)
        case (1)
          zs(i, j) = 400. + 300. * sin(0.9 * real(i)) * cos(0.7 * real(j))
        case (2)
          ! Park and Miller's minimal standard generator: no overflow in
          ! integer(8).
          state = mod(state * 48271_8, 2147483647_8)
          zs(i, j) = real(mod(state, 120000_8)) / 100.
        case (3)
          if (i <= 3) then
            zs(i, j) = 0.
          else
            zs(i, j) = 40. * real(i - 3) * real(i - 3) + 7. * real(j)
          end if
        case (5)
          ! the domain edge is the shore: WRF's continuation of the
          ! rising land goes below sea level past it
          if (i == 1) then
            zs(i, j) = 0.
          else
            zs(i, j) = 60. * real(i - 1) + 5. * real(j)
          end if
        case (6)
          ! child of hills_even at i_parent_start = j_parent_start = 2
          zs(i, j) = 400. + 300. * sin(0.9 * (1.5 + (real(i) - 0.5) / 3.)) &
                     * cos(0.7 * (1.5 + (real(j) - 0.5) / 3.))
        case (7)
          ! child of coast_edge at i_parent_start = 1, j_parent_start = 2:
          ! the parent's shore on the child's own west edge
          if (0.5 + (real(i) - 0.5) / 5. < 1.5) then
            zs(i, j) = 0.
          else
            zs(i, j) = 60. * (0.5 + (real(i) - 0.5) / 5. - 1.) &
                       + 5. * (1.5 + (real(j) - 0.5) / 5.)
          end if
        case default
          zs(i, j) = -400. + 37.5 * real(i) - 11.25 * real(j * j)
        end select
      end do
    end do
  end subroutine terrain

  subroutine run_case(name, nx, ny, ir, jr, dx, dy, kind)
    use module_fr_fire_util, only: interpolate_z2fire
    use wrf_fire_slope_cut, only: fire_params_t, fire_slope, fire_spacing
    character(len=*), intent(in) :: name
    integer, intent(in) :: nx, ny, ir, jr, kind
    real, intent(in) :: dx, dy
    real, allocatable :: zs(:,:), dzdxf(:,:), dzdyf(:,:)
    real, allocatable, target :: zsf(:,:)
    type(fire_params_t) :: fp
    integer :: ifde, jfde
    real :: dxf, dyf

    allocate(zs(nx, ny))
    call terrain(kind, nx, ny, zs)
    ifde = nx * ir
    jfde = ny * jr
    ! The fire memory holds the node past each end interpolate_z2fire and
    ! continue_at_boundary write.
    allocate(zsf(0:ifde+1, 0:jfde+1), dzdxf(0:ifde+1, 0:jfde+1), &
             dzdyf(0:ifde+1, 0:jfde+1))
    zsf = 0.
    call interpolate_z2fire(0,                 &
        1, nx, 1, ny,                          & ! atm domain
        1, nx, 1, ny,                          & ! atm memory
        1, nx, 1, ny,                          & ! atm patch
        1, nx, 1, ny,                          & ! atm tile
        1, ifde, 1, jfde,                      & ! fire domain
        0, ifde+1, 0, jfde+1,                  & ! fire memory
        1, ifde, 1, jfde,                      & ! fire tile
        ir, jr, zs, zsf, 0)
    call oracle_open(name)
    call oracle_put('zs', zs)
    call oracle_put('zsf', zsf(1:ifde, 1:jfde))
    call fire_spacing(dx, dy, ir, jr, dxf, dyf)
    fp%zsf => zsf
    call fire_slope(0, ifde+1, 0, jfde+1, 1, ifde, 1, jfde, &
                    1, ifde, 1, jfde, 1, ifde, 1, jfde, dxf, dyf, fp, &
                    dzdxf, dzdyf)
    call oracle_put('dzdxf', dzdxf(1:ifde, 1:jfde))
    call oracle_put('dzdyf', dzdyf(1:ifde, 1:jfde))
    call oracle_put('ir', ir)
    call oracle_put('jr', jr)
    call oracle_put('dx', dx)
    call oracle_put('dy', dy)
    call oracle_put('dxf', dxf)
    call oracle_put('dyf', dyf)
    call oracle_close()
  end subroutine run_case
end program run_fire_mesh
