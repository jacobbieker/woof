! WOOF-OPTIN: wrf-hydro
! The two WRF modules wrf_cpl_HYDRO (hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90)
! uses, stood in for with only the members it reads, for the coupled oracle
! (build_coupled.sh, coupled_harness.F90).  Part of the WRF-Hydro opt-in.

module module_configure
  implicit none
  type grid_config_rec_type
     integer :: unused = 0
  end type
  type(grid_config_rec_type) :: model_config_rec
end module module_configure

module module_domain
  implicit none
  type domain
     integer :: s_we, e_we, s_sn, e_sn, num_soil_layers, sf_surface_physics, num_nests
     real, allocatable :: zs(:)
     integer, allocatable :: ivgtyp(:,:), isltyp(:,:)
     real, allocatable :: tslb(:,:,:), smois(:,:,:), sh2o(:,:,:)
     real, allocatable :: infxsrt(:,:), soldrain(:,:), sfcheadrt(:,:)
  end type
  character(len=19) :: harness_time = "2020-01-01_00:00:00"
contains
  subroutine domain_clock_get(grid, current_timestr)
    type(domain), intent(in) :: grid
    character(len=*), intent(out) :: current_timestr
    current_timestr = harness_time
  end subroutine
end module module_domain

