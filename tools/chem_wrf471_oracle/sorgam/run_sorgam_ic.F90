! MADE/SORGAM initial conditions: aerosols_sorgam_init past its
! chem_in_opt = 1 return (chem/module_aerosols_sorgam.F:7525-7600), for
! aer_ic_opt = 1 (AER_IC_DEFAULT: aerosol sulfate from the gas, fixed
! small seeds) and 101 (AER_IC_PNNL: sorgam_init_aer_ic_pnnl's height
! profile, every one of its four height bands), on 4 x 2 columns of 30 mass
! levels.  As in WRF the arrays run to the staggered top (kme = kde = 31,
! z_at_w's w levels), whose level 31 the loop leaves at the epsilc / 1e8 fill.
program run_sorgam_ic
  use oracle_io
  use module_state_description
  use module_configure
  use module_data_sorgam
  use sorgam_extract, only: aerosols_sorgam_init
  implicit none
  integer, parameter :: nx = 4, ny = 2, nz = 31
  real :: chem(nx, nz, ny, num_chem), convfac(nx, nz, ny), z_at_w(nx, nz, ny)
  real :: pm25(nx, nz, ny), pm25w(nx, nz, ny), pm25ec(nx, nz, ny), sulf0(nx, nz, ny)
  logical :: is_aerosol(num_chem)
  integer :: i, j, k, opt, ic
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  do opt = 1, 2
    ic = merge(aer_ic_default, aer_ic_pnnl, opt == 1)
    chem = 7.  ! every row the IC does not set keeps this
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz
          ! w levels: 0, 230, 460 ... m (so PNNL's mid-layer heights cross
          ! 500, 1000 and 5000 m)
          z_at_w(i, k, j) = 230. * real(k - 1) * (1. + .1 * real(i - 1))
          convfac(i, k, j) = (1.e2 * exp(-z_at_w(i, k, j) / 7400.)) / (8.314510 * (290. - real(k)))
          sulf0(i, k, j) = 1.e-4 * real(i + 2 * j) * exp(-real(k) / 10.)
          chem(i, k, j, p_sulf) = sulf0(i, k, j)
        end do
      end do
    end do
    call aerosols_sorgam_init(chem, convfac, z_at_w, pm25, pm25w, pm25ec, 0, ic, is_aerosol, &
         1, nx + 1, 1, ny + 1, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, config_flags)
    ! (kde = kme = kte = 31: mass levels 1..30)
    write(name, '(A,I0)') 'aer_ic_opt', ic
    call oracle_open(trim(name))
    call oracle_put('sulf0', sulf0); call oracle_put('convfac', convfac)
    call oracle_put('z_at_w', z_at_w); call oracle_put('chem', chem)
    call oracle_close()
  end do
end program run_sorgam_ic
