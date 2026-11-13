! Drive MPAS-Ocean v8.4.2's ocn_tend_tracer on one block and write every
! field word back: the reference the GPU port is held against.
!
!   run_tend_tracer INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_tend_tracer_oracle in
! oracle_stream.F90's layout.  mpas_ocn_tendency.F is compiled verbatim
! against the real tracer modules (advection, mixing, surface flux,
! short-wave absorption, non-local flux), bulk forcing, diagnostics and
! equation of state; stubs_unported.F90 stands in for the rest (land-ice
! fluxes and frazil return when switched off).  The driver makes the
! _init calls ocn_forward_mode_init makes for these, then the call:
!
!   ocn_constants_init, ocn_surface_bulk_forcing_init, ocn_tracer_advection_init,
!   ocn_tracer_hmix_init, ocn_tracer_hmix_redi_init, ocn_tracer_surface_flux_init,
!   ocn_tracer_short_wave_absorption_init(domain, err)
!   ocn_tend_tracer(tend, state, forcing, diagnostics, mesh, swForcing, scratch, dt,
!                   activeTracersOnlyIn = config oracle_active_tracers_only, timeLevelIn = 1)

program run_tend_tracer
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_surface_bulk_forcing
  use ocn_tracer_advection
  use ocn_tracer_hmix
  use ocn_tracer_hmix_redi
  use ocn_tracer_surface_flux_to_tend
  use ocn_tracer_short_wave_absorption
  use ocn_tendency
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type) :: domain
  real(kind=RKIND) :: dt
  integer :: err
  logical, pointer :: activeOnly
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_surface_bulk_forcing_init(err)
  if (err /= 0) stop 13
  err = 0
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
  call mpas_pool_get_config(pools % configs, 'oracle_active_tracers_only', activeOnly)
  call ocn_tend_tracer(pool_named(pools, 'tend'), pool_named(pools, 'state'), pool_named(pools, 'forcing'), &
       pool_named(pools, 'diagnostics'), pool_named(pools, 'mesh'), pool_named(pools, 'swForcing'), &
       pool_named(pools, 'scratch'), dt, activeTracersOnlyIn=activeOnly, timeLevelIn=1)
  call write_stream(outPath, fields)
end program run_tend_tracer
