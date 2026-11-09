! WOOF-OPTIN: wrf-hydro
! Oracle harness for the coupled WRF/WRF-Hydro exchange: WRF 4.7.1's own
! hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90 (wrf_cpl_HYDRO, compiled whole) and
! the whole hydro/ tree under it (HYDRO_ini, HYDRO_exe and everything they
! call), MPP_LAND on one process, driven over a few coupling cycles with LSM
! fields read from a file.  Part of the WRF-Hydro opt-in (../LICENSE); no
! WRF-Hydro source is copied here.
!
! The two WRF modules wrf_cpl_HYDRO uses are stood in for in
! coupled_wrf_stubs.F90 with just the members it reads: module_domain's grid
! (dimensions, ZS, IVGTYP, ISLTYP, TSLB, SMOIS, SH2O, INFXSRT, SOLDRAIN,
! SFCHEADRT, sf_surface_physics, num_nests) and domain_clock_get's current
! time, and module_configure's two type names.

program coupled_oracle
  use mpi
  use module_domain
  use module_WRF_HYDRO, only: wrf_cpl_HYDRO
  use module_rt_data, only: rt_domain
  use config_base, only: nlst
  implicit none
  type(domain) :: grid
  integer :: ix, jx, nsoil, ncycles, cyc, i, j, k, n, u, nl
  real :: hydro_dt
  integer :: bits, ierr

  call MPI_Init(ierr)               ! WRF has initialised MPI before wrf_cpl_HYDRO
  open(newunit=u, file="coupled-inputs.txt", status="old")
  read(u, *) ix, jx, nsoil, ncycles, hydro_dt
  grid%s_we = 1; grid%e_we = ix + 1; grid%s_sn = 1; grid%e_sn = jx + 1
  grid%num_soil_layers = nsoil
  grid%sf_surface_physics = 2
  grid%num_nests = 0
  allocate(grid%zs(nsoil), grid%ivgtyp(ix, jx), grid%isltyp(ix, jx))
  allocate(grid%tslb(ix, nsoil, jx), grid%smois(ix, nsoil, jx), grid%sh2o(ix, nsoil, jx))
  allocate(grid%infxsrt(ix, jx), grid%soldrain(ix, jx), grid%sfcheadrt(ix, jx))
  do k = 1, nsoil
     read(u, *) bits
     grid%zs(k) = transfer(bits, 1.0)
  end do
  do j = 1, jx
     do i = 1, ix
        read(u, *) grid%ivgtyp(i, j), grid%isltyp(i, j)
     end do
  end do
  grid%tslb = 285.0
  grid%sfcheadrt = 0.0
  open(newunit=n, file="coupled-steps.csv", status="replace")
  write(n, '(A)') "cycle,field,index,bits"
  do cyc = 1, ncycles
     do j = 1, jx
        do i = 1, ix
           do k = 1, nsoil
              read(u, *) bits
              grid%smois(i, k, j) = transfer(bits, 1.0)
              read(u, *) bits
              grid%sh2o(i, k, j) = transfer(bits, 1.0)
           end do
           read(u, *) bits
           grid%infxsrt(i, j) = transfer(bits, 1.0)
           read(u, *) bits
           grid%soldrain(i, j) = transfer(bits, 1.0)
        end do
     end do
     call wrf_cpl_HYDRO(hydro_dt, grid, 1, ix, 1, jx)
     do j = 1, jx
        do i = 1, ix
           do k = 1, nsoil
              call put(cyc, "smois", (k - 1) * ix * jx + (j - 1) * ix + i - 1, grid%smois(i, k, j))
              call put(cyc, "sh2o", (k - 1) * ix * jx + (j - 1) * ix + i - 1, grid%sh2o(i, k, j))
           end do
           call put(cyc, "sfcheadrt", (j - 1) * ix + i - 1, grid%sfcheadrt(i, j))
           call put(cyc, "infxsrt_rt", (j - 1) * ix + i - 1, rt_domain(1)%infxsrt(i, j))
        end do
     end do
     do j = 1, rt_domain(1)%jxrt
        do i = 1, rt_domain(1)%ixrt
           nl = (j - 1) * rt_domain(1)%ixrt + i - 1
           call put(cyc, "head_rt", nl, rt_domain(1)%overland%control%surface_water_head_routing(i, j))
           call put(cyc, "zwattablrt", nl, rt_domain(1)%subsurface%properties%zwattablrt(i, j))
           call put(cyc, "qstrmvolrt_acc", nl, rt_domain(1)%QSTRMVOLRT_ACC(i, j))
           call put(cyc, "qinflowbase", nl, rt_domain(1)%qinflowbase(i, j))
           call put(cyc, "to_lake", nl, rt_domain(1)%overland%streams_and_lakes%surface_water_to_lake(i, j))
        end do
     end do
     do i = 1, rt_domain(1)%nlinks
        call put(cyc, "qlink1", i - 1, rt_domain(1)%qlink(i, 1))
        call put(cyc, "hlink", i - 1, rt_domain(1)%hlink(i))
        call put(cyc, "cvol", i - 1, rt_domain(1)%cvol(i))
     end do
     do i = 1, rt_domain(1)%nlakes
        call put(cyc, "resht", i - 1, rt_domain(1)%resht(i))
        call put(cyc, "qlakeo", i - 1, rt_domain(1)%qlakeo(i))
     end do
     do i = 1, rt_domain(1)%numbasns
        call put(cyc, "z_gwsubbas", i - 1, rt_domain(1)%z_gwsubbas(i))
        call put(cyc, "qout_gwsubbas", i - 1, rt_domain(1)%qout_gwsubbas(i))
     end do
     call put(cyc, "dtct", 0, nlst(1)%dtct)
  end do
  close(n)
  close(u)
  ! the static network and terrain HYDRO_ini built, once
  open(newunit=n, file="coupled-static.csv", status="replace")
  write(n, '(A)') "field,index,value"
  do i = 1, rt_domain(1)%nlinks
     write(n, '(A,",",I0,",",I0)') "chanxi", i - 1, rt_domain(1)%chanxi(i)
     write(n, '(A,",",I0,",",I0)') "chanyj", i - 1, rt_domain(1)%chanyj(i)
     write(n, '(A,",",I0,",",I0)') "to_node", i - 1, rt_domain(1)%to_node(i)
     write(n, '(A,",",I0,",",I0)') "typel", i - 1, rt_domain(1)%typel(i)
     write(n, '(A,",",I0,",",I0)') "lakenode", i - 1, rt_domain(1)%lakenode(i)
     write(n, '(A,",",I0,",",I0)') "order", i - 1, rt_domain(1)%order(i)
     write(n, '(A,",",I0,",",I0)') "chanlen", i - 1, transfer(rt_domain(1)%chanlen(i), 1)
     write(n, '(A,",",I0,",",I0)') "node_area", i - 1, transfer(rt_domain(1)%node_area(i), 1)
     write(n, '(A,",",I0,",",I0)') "bw", i - 1, transfer(rt_domain(1)%bw(i), 1)
     write(n, '(A,",",I0,",",I0)') "zelev", i - 1, transfer(rt_domain(1)%zelev(i), 1)
  end do
  do j = 1, rt_domain(1)%jxrt
     do i = 1, rt_domain(1)%ixrt
        nl = (j - 1) * rt_domain(1)%ixrt + i - 1
        write(n, '(A,",",I0,",",I0)') "retention", nl, &
             transfer(rt_domain(1)%overland%properties%retention_depth(i, j), 1)
        write(n, '(A,",",I0,",",I0)') "lake_mask", nl, &
             rt_domain(1)%overland%streams_and_lakes%lake_mask(i, j)
        do k = 1, 8
           write(n, '(A,",",I0,",",I0)') "slope", (k - 1) * rt_domain(1)%ixrt * rt_domain(1)%jxrt + nl, &
                transfer(rt_domain(1)%overland%properties%surface_slope(i, j, k), 1)
        end do
        do k = 1, 3
           write(n, '(A,",",I0,",",I0)') "sdir", (k - 1) * rt_domain(1)%ixrt * rt_domain(1)%jxrt + nl, &
                rt_domain(1)%overland%properties%max_surface_slope_index(i, j, k)
        end do
     end do
  end do
  close(n)
  call MPI_Finalize(ierr)
contains
  subroutine put(cyc, name, idx, value)
    integer, intent(in) :: cyc, idx
    character(len=*), intent(in) :: name
    real, intent(in) :: value
    write(n, '(I0,",",A,",",I0,",",I0)') cyc, name, idx, transfer(value, 1)
  end subroutine
end program coupled_oracle

! nwmCheck (hydro/Routing/module_reservoir_routing.F90 :269-278) is an
! external subroutine that calls hydro_stop without the module; WRF's link
! leaves it unresolved.  Only an error path reaches it; it stops here too.
subroutine hydro_stop(msg)
  use module_hydro_stop, only: stop_with => HYDRO_stop
  character(len=*) :: msg
  call stop_with(msg)
end subroutine hydro_stop
