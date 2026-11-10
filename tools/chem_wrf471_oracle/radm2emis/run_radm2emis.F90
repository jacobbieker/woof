! RADM2/SORGAM anthropogenic emissions (chem_opt = 2, emiss_opt = 3) on 8 x 2
! columns of 6 levels, in emissions_driver's order (chem/emissions_driver.F):
! add_anthropogenics (:1089, chem/module_emissions_anthropogenics.F) for the
! gases, then sorgam_addemiss (:1647, chem/module_aerosols_sorgam.F) for the
! aerosol mass and number rows.  aerosols_sorgam_init runs first for the
! module constants (chem_in_opt = 1 returns before touching chem).
!
! Three cases vary the step (dtstep) and the emission depth (kemit, up to the
! whole column).  Densities, layer depths and emission rates span several
! decades, from a deterministic integer generator, and some emission rates
! are exactly zero.  Every input and every output row is written.
program run_radm2emis
  use oracle_io
  use module_state_description
  use module_configure
  use module_data_sorgam
  use module_aerosols_sorgam, only: aerosols_sorgam_init, sorgam_addemiss
  use module_emissions_anthropogenics, only: add_anthropogenics
  implicit none
  integer, parameter :: nx = 8, ny = 2, nz = 6, ncase = 3
  real, parameter :: dts(ncase) = (/ 120., 30., 7.5 /)
  integer, parameter :: kemits(ncase) = (/ 4, 6, 1 /)
  real :: chem(nx, nz, ny, num_chem), chem0(nx, nz, ny, num_chem)
  real, dimension(nx, nz, ny) :: rho_phy, alt, dz8w, convfac, z_at_w, pm25, pm25w, &
       pm25ec, p8w, u_phy, v_phy
  real, allocatable :: emis_ant(:, :, :, :)
  real :: emis_aircraft(nx, 1, ny, num_emis_aircraft), ebu(nx, nz, ny, num_ebu)
  real, dimension(nx, ny) :: u10, v10, xland, slai, ust
  real :: smois(nx, 4, ny), erod(nx, ny, 3)
  integer :: ivgtyp(nx, ny), isltyp(nx, ny)
  logical :: is_aerosol(num_chem)
  integer :: i, j, k, n, c, kemit
  integer(8) :: seed
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  seed = 20261110_8
  do j = 1, ny
    do i = 1, nx
      c = i + (j - 1) * nx
      do k = 1, nz
        dz8w(i, k, j) = (20. + 480. * draw()) * (1. + .3 * real(k - 1))
        rho_phy(i, k, j) = 1.25 * exp(-real(k - 1) * dz8w(i, k, j) / 8000.) * (.6 + .5 * draw())
        alt(i, k, j) = (1. + .02 * draw()) / rho_phy(i, k, j)
        convfac(i, k, j) = 40.
        z_at_w(i, k, j) = 100. * real(k - 1)
        p8w(i, k, j) = 1.e5; u_phy(i, k, j) = 0.; v_phy(i, k, j) = 0.
        chem(i, k, j, 1) = 0.
        do n = param_first_scalar, num_chem
          chem(i, k, j, n) = 1.e-3 * draw()
        end do
        chem(i, k, j, p_nu0) = 1.e9 * draw(); chem(i, k, j, p_ac0) = 1.e8 * draw()
        chem(i, k, j, p_corn) = 1.e5 * draw()
        if (c == 5) chem(i, k, j, :) = 0.
      end do
    end do
  end do
  u10 = 0.; v10 = 0.; xland = 1.; slai = 0.; ust = 0.; smois = 0.; erod = 0.
  ivgtyp = 1; isltyp = 1; emis_aircraft = 0.; ebu = 0.
  call aerosols_sorgam_init(chem, convfac, z_at_w, pm25, pm25w, pm25ec, 1, 1, is_aerosol, &
       1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, config_flags)
  call oracle_open('init')
  call oracle_put('chem', chem)
  call oracle_put('rho_phy', rho_phy); call oracle_put('alt', alt)
  call oracle_put('dz8w', dz8w)
  call oracle_put('factnumn', factnumn); call oracle_put('factnuma', factnuma)
  call oracle_put('factnumc', factnumc); call oracle_put('anthfac', anthfac)
  call oracle_put('orgfac', orgfac); call oracle_put('so4fac', so4fac)
  call oracle_put('no3fac', no3fac)
  call oracle_close()
  chem0 = chem
  do c = 1, ncase
    kemit = kemits(c)
    config_flags%kemit = kemit
    allocate(emis_ant(nx, kemit, ny, num_emis_ant))
    emis_ant(:, :, :, 1) = 0.
    do n = param_first_scalar, num_emis_ant
      do j = 1, ny
        do k = 1, kemit
          do i = 1, nx
            ! gases in mol km-2 hr-1, aerosols in ug m-2 s-1; a tenth of the
            ! rates are exactly zero
            emis_ant(i, k, j, n) = 10. ** (4. * draw() - 2.) * merge(0., 1., draw() < .1)
          end do
        end do
      end do
    end do
    chem = chem0
    call add_anthropogenics(1, dts(c), dz8w, config_flags, rho_phy, alt,          &
         chem, emis_ant, emis_aircraft,                                          &
         1, nx + 1, 1, ny + 1, 1, nz + 1, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
    call sorgam_addemiss(1, dts(c), u10, v10, alt, dz8w, xland, chem, ebu,       &
         slai, ust, smois, ivgtyp, isltyp, emis_ant, 0, 0, kemit, 0, 4,          &
         config_flags%emiss_opt, 0, 1, p8w, u_phy, v_phy, rho_phy, 9.81, 1000.,  &
         erod,                                                                    &
         1, nx + 1, 1, ny + 1, 1, nz + 1, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
    write(name, '(A,I0)') 'case', c
    call oracle_open(trim(name))
    call oracle_put('dtstep', dts(c)); call oracle_put('kemit', kemit)
    call oracle_put('emis_ant', emis_ant)
    call oracle_put('chem', chem)
    call oracle_close()
    deallocate(emis_ant)
  end do

contains

  ! A 64-bit linear congruential generator (Knuth's MMIX constants), top 24
  ! bits as a real in [0, 1): deterministic on every compiler.
  real function draw()
    seed = seed * 6364136223846793005_8 + 1442695040888963407_8
    draw = real(ishft(seed, -40)) / 16777216.
  end function draw
end program run_radm2emis
