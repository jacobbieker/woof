! Drive MPAS-Seaice v8.4.2's column package wrappers (mpas_seaice_column.F
! and the column package, column/ice_*.F90, all compiled verbatim) on one block
! and write every field word back: the reference the GPU column port is held
! against.
!
!   run_column INPUT.bin OUTPUT.bin LOG.txt
!
! INPUT.bin is written by woof.verify.mpas_seaice_column_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout, on the framework
! stand-ins of tools/mpas_ocean_oracle/stubs.F90.  mpas_seaice_column.F is
! built with -Dprivate=public, which changes only what this driver may call
! and read (the tracer object).  The column package is double precision
! (dbl_kind) and mpas_seaice_column.F passes RKIND arrays to it, so the
! wrappers compile at RKIND 8 only; this oracle is built at RKIND 8.
!
! Every run starts as seaice_init does (mpas_seaice_initialize.F:47-125) with
! seaice_init_column_physics_package_parameters (the stand-in pointers of the
! packages that are off, the tracer object, the configs checked and handed to
! the column package).  Then, by oracle_stage:
!
!   init          initialize_coupler_fields, seaice_init_column_physics_
!                 package_variables, init_ice_state and seaice_reset_coupler_
!                 fluxes, in seaice_init's order (cut verbatim; the mesh,
!                 forcing, velocity-solver and advection inits and the halo
!                 exchanges are not part of it);
!   aggregate     seaice_column_aggregate;
!   update_state  seaice_column_update_state(domain, oracle_update_type,
!                 dt, oracle_ice_age_time_offset).
!
! LOG.txt gets the tracer object ('TOBJ name values...': nTracers, the
! physics tracer indices, parentIndex, firstAncestorMask (as integers),
! ancestorNumber, ancestorIndices, and the column package's nt_Tsfc and
! nt_fbri) and every warning and error the routines log.
program run_column
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_warn_unit, stub_log_err_unit
  use mpas_timekeeping
  use seaice_column
  use seaice_initialize, only: init_ice_state, initialize_coupler_fields
  use seaice_forcing, only: seaice_reset_coupler_fluxes
  use ice_colpkg_tracers, only: nt_Tsfc, nt_fbri
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: inputDt
  real(kind=RKIND), pointer :: iceAgeTimeOffset
  character(len=StrKIND), pointer :: stage, updateType, config_start_time, clockStep
  integer :: p, logUnit
  character(len=1024) :: inPath, outPath, logPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call get_command_argument(3, logPath)
  call read_stream(inPath, inputDt, pools, fields)
  open(newunit=logUnit, file=trim(logPath), status='replace', action='write')
  stub_log_warn_unit = logUnit
  stub_log_err_unit = logUnit
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  allocate(domain % dminfo)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  domain % blocklist % packages => pools % packages
  do p = 1, size(poolNames)
     call stub_pool_put_subpool(domain % blocklist % structs, trim(poolNames(p)), pool_named(pools, trim(poolNames(p))))
  end do
  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'oracle_clock_step', clockStep)
  call mpas_set_clock(domain % clock, config_start_time, clockStep)

  call seaice_init_column_physics_package_parameters(domain)
  call write_tracer_object()

  call mpas_pool_get_config(pools % configs, 'oracle_stage', stage)
  select case (trim(stage))
  case ('init')
     call initialize_coupler_fields(domain)
     call seaice_init_column_physics_package_variables(domain, domain % clock)
     call init_ice_state(domain, domain % clock)
     call seaice_reset_coupler_fluxes(domain)
  case ('aggregate')
     call seaice_column_aggregate(domain)
  case ('update_state')
     call mpas_pool_get_config(pools % configs, 'oracle_update_type', updateType)
     call mpas_pool_get_config(pools % configs, 'oracle_ice_age_time_offset', iceAgeTimeOffset)
     call seaice_column_update_state(domain, trim(updateType), inputDt, iceAgeTimeOffset)
  case default
     print *, 'run_column: unknown oracle_stage ', trim(stage)
     stop 2
  end select
  call write_stream(outPath, fields)
  close(logUnit)

contains

  subroutine write_ints(name, values)
    character(len=*), intent(in) :: name
    integer, dimension(:), intent(in) :: values
    integer :: i
    write(logUnit, '(a,1x,a)', advance='no') 'TOBJ', name
    do i = 1, size(values)
       write(logUnit, '(1x,i0)', advance='no') values(i)
    end do
    write(logUnit, '(a)') ''
  end subroutine write_ints

  subroutine write_tracer_object()
    integer :: k
    associate (t => ciceTracerObject)
      call write_ints('nTracers', [t % nTracers])
      call write_ints('index', [t % index_surfaceTemperature, t % index_iceEnthalpy, t % index_snowEnthalpy, &
           t % index_iceSalinity, t % index_iceAge, t % index_firstYearIceArea, t % index_levelIceArea, &
           t % index_levelIceVolume, t % index_pondArea, t % index_pondDepth, t % index_pondLidThickness, &
           t % index_aerosols])
      call write_ints('parentIndex', t % parentIndex)
      do k = 1, t % nBaseTracers
         call write_ints('firstAncestorMask', nint(t % firstAncestorMask(:,k)))
      end do
      call write_ints('ancestorNumber', t % ancestorNumber)
      do k = 1, t % nMaxAncestorTracers
         call write_ints('ancestorIndices', t % ancestorIndices(:,k))
      end do
      call write_ints('colpkg', [nt_Tsfc, nt_fbri])
    end associate
  end subroutine write_tracer_object

end program run_column
