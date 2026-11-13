! Drive one MPAS-Ocean v8.4.2 split-explicit step (ocn_time_integrator_split)
! on one block and write every field word back: the reference the GPU port
! is held against.
!
!   run_split_step INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_split_step_oracle in
! oracle_stream.F90's layout; a field's time level 2 is the entry NAME@2.
! ocn_time_integrator_split is cut verbatim out of
! mpas_ocn_time_integration_split.F (split_step_cut_head.F90); everything it
! calls is the real v8.4.2 module (the tendencies, diagnostics, equation of
! state, ALE thickness, tracer advection and mixing, vertical mixing, time
! averaging, the land-ice effective density) or a cut of it
! (mpas_reconstruct), less GM (stubs_split.F90).  The driver runs the _init
! routine of every module the step reaches, as ocn_forward_mode_init does,
! builds the block (its structs every pool, its dimensions the mesh pool's),
! computes the starting state's diagnostics (ocn_diagnostic_solve at time
! level 1, as MPAS's init does), sets nBtrSubcycles from the integer config of that name and takes the step.

program run_split_step
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_tendency
  use ocn_forcing
  use ocn_surface_bulk_forcing
  use ocn_thick_hadv
  use ocn_thick_vadv
  use ocn_thick_surface_flux
  use ocn_vel_coriolis
  use ocn_vel_vadv
  use ocn_vel_pressure_grad
  use ocn_vel_hmix
  use ocn_vel_forcing
  use ocn_tracer_advection
  use ocn_tracer_hmix
  use ocn_tracer_hmix_redi
  use ocn_tracer_surface_flux_to_tend
  use ocn_tracer_short_wave_absorption
  use ocn_equation_of_state
  use ocn_diagnostics
  use ocn_vmix
  use ocn_split_step_cut
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: dt
  integer :: err
  integer, pointer :: n
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  call stub_pool_put_subpool(domain % blocklist % structs, 'state', pool_named(pools, 'state'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'diagnostics', pool_named(pools, 'diagnostics'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'mesh', pool_named(pools, 'mesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'verticalMesh', pool_named(pools, 'verticalMesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'tend', pool_named(pools, 'tend'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'scratch', pool_named(pools, 'scratch'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'forcing', pool_named(pools, 'forcing'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'shortwave', pool_named(pools, 'swForcing'))

  call ocn_constants_init(pools % configs, pools % packages)
  err = 0
  call ocn_tendency_init(err)
  call ocn_forcing_init(err)
  call ocn_surface_bulk_forcing_init(err)
  call ocn_thick_hadv_init(err)
  call ocn_thick_vadv_init(err)
  call ocn_thick_surface_flux_init(err)
  call ocn_vel_coriolis_init(err)
  call ocn_vel_vadv_init(err)
  call ocn_vel_pressure_grad_init(err)
  call ocn_vel_hmix_init(err)
  call ocn_vel_forcing_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_advection_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_redi_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_surface_flux_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_short_wave_absorption_init(domain, err)
  if (err /= 0) stop 13
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostics_init(err)
  if (err /= 0) stop 13
  call ocn_vmix_init(domain, err)
  if (err /= 0) stop 13

  ! the diagnostics of the state the step starts from, as MPAS's init
  ! computes them before the first step
  call ocn_diagnostic_solve(dt, pool_named(pools, 'state'), pool_named(pools, 'forcing'), pool_named(pools, 'mesh'), &
       pool_named(pools, 'diagnostics'), pool_named(pools, 'scratch'), pool_named(pools, 'tracers'), 1)
  call mpas_pool_get_config(pools % configs, 'nBtrSubcycles', n)
  nBtrSubcycles = n
  call ocn_time_integrator_split(domain, dt)
  call write_stream(outPath, fields)
end program run_split_step
