! Drive MPAS-Seaice v8.4.2's variational basis set-up on one block and write
! every field word back: the reference the host port
! (woof.hex.mpas_seaice.variational_basis_v842) is held against.
!
!   run_variational_basis INPUT.bin OUTPUT.bin LOG.txt
!
! INPUT.bin is written by woof.verify.mpas_seaice_variational_basis_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout.  The driver calls
! seaice_init_velocity_solver_variational (mpas_seaice_velocity_solver_variational.F,
! compiled verbatim, with the verbatim wachspress, pwl, variational_shared and
! numerics modules and seaice_mesh's routines cut verbatim; build.sh) once, as
! seaice_init_velocity_solver does under config_stress_divergence_scheme =
! 'variational', with the configs it passes.  MPAS_LOG_WARN messages go to
! LOG.txt; MPAS_LOG_CRIT is recorded there before the stop.

program run_variational_basis
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_warn_unit, stub_log_err_unit
  use seaice_velocity_solver_variational, only: seaice_init_velocity_solver_variational
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(MPAS_pool_type), pointer :: mesh, velocity_variational, boundary
  real(kind=RKIND) :: inputDt
  logical, pointer :: config_rotate_cartesian_grid, config_include_metric_terms
  character(len=StrKIND), pointer :: config_variational_basis, config_wachspress_integration_type
  integer, pointer :: config_wachspress_integration_order
  integer :: logUnit
  character(len=1024) :: inPath, outPath, logPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call get_command_argument(3, logPath)
  open(newunit=logUnit, file=trim(logPath), status='replace', action='write')
  stub_log_warn_unit = logUnit
  stub_log_err_unit = logUnit
  call read_stream(inPath, inputDt, pools, fields)
  mesh => pool_named(pools, 'mesh')
  velocity_variational => pool_named(pools, 'velocity_variational')
  boundary => pool_named(pools, 'boundary')
  call mpas_pool_get_config(pools % configs, 'config_rotate_cartesian_grid', config_rotate_cartesian_grid)
  call mpas_pool_get_config(pools % configs, 'config_include_metric_terms', config_include_metric_terms)
  call mpas_pool_get_config(pools % configs, 'config_variational_basis', config_variational_basis)
  call mpas_pool_get_config(pools % configs, 'config_wachspress_integration_type', config_wachspress_integration_type)
  call mpas_pool_get_config(pools % configs, 'config_wachspress_integration_order', config_wachspress_integration_order)
  call seaice_init_velocity_solver_variational(&
       mesh, &
       velocity_variational, &
       boundary, &
       config_rotate_cartesian_grid, &
       config_include_metric_terms, &
       config_variational_basis, &
       config_wachspress_integration_type, &
       config_wachspress_integration_order)
  close(logUnit)
  call write_stream(outPath, fields)
end program run_variational_basis
