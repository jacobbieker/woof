program run_fitch
  ! WRF v4.7.1's Fitch wind-farm drag (module_wind_fitch.F dragforce, cut
  ! verbatim by extract.py) on one row of synthetic columns.  The module's
  ! turbine state is filled here where init_module_wind_fitch would fill it
  ! from windturbines-ij.txt and wind-turbine-N.tbl, then dragforce is called
  ! exactly as module_pbl_driver.F:2053-2063 calls it.  Writes, per case, the
  ! inputs, the turbine list, the curves and DU, DV, QKE, POWER through
  ! oracle_io: fixtures/<case>/.
  !
  ! One column per behaviour (i):
  !    1  one turbine, sheared and veering wind near 8 m/s at the hub
  !    2  three turbines of one type
  !    3  a mixed cell: four 3 MW at 100 m and two 8 MW at 150 m
  !    4  a rotor spanning many levels (8 m layers, 160 m rotor)
  !    5  a rotor within one level (320 m first layer, 90 m rotor)
  !    6  below cut-in
  !    7  above cut-out
  !    8  rated wind, on the thrust taper
  !    9  no turbine
  !   10  a mixed cell of three types at three hub heights
  !   11  a uniform 8 m/s column (Ct at a table node)
  ! Every hub lies above its column's first mass level: below it WRF reads
  ! an unassigned z1 (module_wind_fitch.F:163-164), which is held out.
  ! The turbine list interleaves the cells; within a cell each type's
  ! turbines are consecutive.
  use oracle_io
  use wrf_wind_fitch_inline
  implicit none
  integer, parameter :: nx = 11, ny = 1, nz = 50, nturb = 22, ntype = 3
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  real :: z_at_w(nx, nz + 1, ny), u(nx, nz + 1, ny), v(nx, nz + 1, ny)
  real :: dz(nx, nz + 1, ny), du(nx, nz + 1, ny), dv(nx, nz + 1, ny)
  real :: qke(nx, nz + 1, ny), power(nx, ny)
  real :: du0(nx, nz + 1, ny), dv0(nx, nz + 1, ny), qke0(nx, nz + 1, ny)
  real :: ws_out(ntype, maxvals2), tc_out(ntype, maxvals2), tp_out(ntype, maxvals2)
  integer :: nval(ntype), cells(nturb), kinds(nturb)
  real :: hubs(nturb)
  real :: dx, dt, factor, terrain, first, stretch, zc, s, alpha, theta, veer
  real :: speed(nx), shear(nx), dir(nx), surface(nx), dz1(nx), grow(nx)
  integer :: i, k, n, c
  character(len=1024) :: root

  call oracle_root(root)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 1; ime = nx; jms = 1; jme = ny; kms = 1; kme = nz + 1
  its = 1; ite = nx; jts = 1; jte = ny; kts = 1; kte = nz

  ! --- the curves (wind-turbine-N.tbl rows: speed, Ct, power kW) ----------
  turbws = 0.; turbtc = 0.; turbpw = 0.
  nval(1) = 23                               ! generic 3 MW, 120 m rotor
  do k = 1, nval(1)
    s = 2.0 + real(k)
    turbws(1, k) = s
    turbtc(1, k) = 0.8
    if (s > 12.0) turbtc(1, k) = 0.8 * (12.0 / s)**2
    turbpw(1, k) = min(3000.0, 3000.0 * ((s - 3.0) / 9.0)**3)
  end do
  nval(2) = 25                               ! generic 8 MW, 160 m rotor
  do k = 1, nval(2)
    s = 3.0 + real(k)
    turbws(2, k) = s
    turbtc(2, k) = 0.78
    if (s > 11.0) turbtc(2, k) = 0.78 * (11.0 / s)**2.2
    turbpw(2, k) = min(8000.0, 8000.0 * ((s - 4.0) / 7.0)**3)
  end do
  nval(3) = 10                               ! generic 2 MW, 90 m, uneven rows
  turbws(3, 1:10) = (/ 3.5, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 13.0, 16.0, 20.0 /)
  turbtc(3, 1:10) = (/ 0.86, 0.84, 0.82, 0.8, 0.79, 0.77, 0.7, 0.5, 0.3, 0.17 /)
  turbpw(3, 1:10) = (/ 20.0, 60.0, 180.0, 450.0, 850.0, 1350.0, 1850.0, &
                       2000.0, 2000.0, 2000.0 /)
  ws_out = turbws(1:ntype, :)
  tc_out = turbtc(1:ntype, :)
  tp_out = turbpw(1:ntype, :)

  ! --- the turbine list (windturbines-ij.txt: i, j, type) ------------------
  cells = (/ 3, 1, 3, 3, 2, 3, 2, 10, 3, 3, 4, 2, 4, 5, 10, 10, 6, 7, 8, 8, &
             10, 11 /)
  kinds = (/ 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 1, 2, 3, 1, 1, 1, 1, 1, 1, &
             2, 1 /)
  hubs = (/ 100., 100., 100., 100., 100., 100., 100., 80., 150., 150., &
            150., 100., 150., 170., 110., 110., 100., 100., 100., 100., &
            160., 100. /)
  nt = nturb
  allocate(nkind(nt), ival(nt, 1), jval(nt, 1))
  allocate(hubheight(nt), stc(nt), stc2(nt), cutin(nt), cutout(nt))
  allocate(diameter(nt), npower(nt))
  do n = 1, nt
    ival(n, 1) = cells(n)
    jval(n, 1) = 1
    nkind(n) = kinds(n)
    hubheight(n) = hubs(n)
    select case (kinds(n))
    case (1)
      diameter(n) = 120.; stc(n) = 0.15; npower(n) = 3000.
    case (2)
      diameter(n) = 160.; stc(n) = 0.1; npower(n) = 8000.
    case default
      diameter(n) = 90.; stc(n) = 0.12; npower(n) = 2000.
    end select
    ! init_module_wind_fitch.F:519-521
    cutin(n) = turbws(nkind(n), 1)
    cutout(n) = turbws(nkind(n), nval(nkind(n)))
    stc2(n) = turbtc(nkind(n), nval(nkind(n)))
  end do

  ! --- the columns ----------------------------------------------------------
  speed = (/ 8.0, 9.5, 10.0, 11.0, 9.0, 2.2, 27.0, 14.0, 7.0, 12.0, 8.0 /)
  shear = (/ 0.14, 0.2, 0.12, 0.25, 0.1, 0.14, 0.1, 0.16, 0.14, 0.3, 0.0 /)
  dir = (/ 0.4, 2.1, -1.0, 3.0, 0.9, 1.7, 4.4, -2.5, 0.0, 5.5, 0.4 /)
  surface = (/ 250., 0., 731.5, 12., 400., 90., 1500., 3.25, 60., 845., 250. /)
  dz1 = (/ 20., 24., 18., 8., 320., 20., 30., 15., 20., 12., 20. /)
  grow = (/ 1.08, 1.07, 1.09, 1.0, 1.05, 1.08, 1.06, 1.1, 1.08, 1.08, 1.08 /)
  call run_case('default', 1000.0, 10.0, 0.25)
  call run_case('coarse', 3000.0, 18.0, 1.0)

contains

  subroutine run_case(name, dx_case, dt_case, factor_case)
    character(len=*), intent(in) :: name
    real, intent(in) :: dx_case, dt_case, factor_case
    dx = dx_case; dt = dt_case; factor = factor_case
    correction_factor = factor
    u = 0.; v = 0.; dz = 0.
    do i = 1, nx
      z_at_w(i, 1, 1) = surface(i)
      first = dz1(i); stretch = grow(i)
      do k = 1, nz
        z_at_w(i, k + 1, 1) = z_at_w(i, k, 1) + first * stretch**(k - 1)
      end do
      do k = 1, nz
        ! dz8w as phy_prep forms it.
        dz(i, k, 1) = z_at_w(i, k + 1, 1) - z_at_w(i, k, 1)
        zc = 0.5 * (z_at_w(i, k + 1, 1) + z_at_w(i, k, 1)) - z_at_w(i, 1, 1)
        alpha = shear(i)
        veer = 0.0015 * real(i)
        if (i == 11) veer = 0.
        s = speed(i) * (zc / 100.0)**alpha
        theta = dir(i) + veer * zc
        u(i, k, 1) = s * cos(theta)
        v(i, k, 1) = s * sin(theta)
        ! The PBL scheme's tendencies, already in RUBLTEN/RVBLTEN and QKE.
        du0(i, k, 1) = 1.0e-4 * sin(0.37 * real(k + i))
        dv0(i, k, 1) = -2.0e-4 * cos(0.21 * real(k * i))
        qke0(i, k, 1) = 0.4 + 0.013 * real(k) + 0.002 * real(i)
      end do
    end do
    du = du0; dv = dv0; qke = qke0
    power = -1.0
    call dragforce(id=1, z_at_w=z_at_w, u=u, v=v, dx=dx, dz=dz, dt=dt, &
                   qke=qke, du=du, dv=dv, windfarm_opt=1, power=power, &
                   ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
                   ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
                   its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
    call oracle_open(name)
    call oracle_put('dx', dx)
    call oracle_put('dt', dt)
    call oracle_put('windfarm_tke_factor', factor)
    call oracle_put('piconst', piconst)
    call oracle_put('z_at_w', z_at_w)
    call oracle_put('u', u)
    call oracle_put('v', v)
    call oracle_put('dz', dz)
    call oracle_put('du_in', du0)
    call oracle_put('dv_in', dv0)
    call oracle_put('qke_in', qke0)
    call oracle_put('ival', ival(:, 1))
    call oracle_put('nkind', nkind)
    call oracle_put('hubheight', hubheight)
    call oracle_put('diameter', diameter)
    call oracle_put('stc', stc)
    call oracle_put('nval', nval)
    call oracle_put('turbws', ws_out)
    call oracle_put('turbtc', tc_out)
    call oracle_put('turbpw', tp_out)
    call oracle_put('du', du)
    call oracle_put('dv', dv)
    call oracle_put('qke', qke)
    call oracle_put('power', power)
    call oracle_close()
  end subroutine run_case

end program run_fitch
