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
!                 dt, oracle_ice_age_time_offset);
!   vertical_thermodynamics
!                 colpkg_init_thermo (which sets the column package's
!                 heat_capacity and l_brine, as seaice_init's
!                 init_column_thermodynamic_profiles does before any step),
!                 then column_vertical_thermodynamics(domain, clock, ierr),
!                 whose ierr is logged ('TOBJ IERR n'): 0, or
!                 SEAICE_ERROR_COL_VERT_THERM after an abort (its message and
!                 cell are logged too);
!   itd_thermodynamics
!                 colpkg_init_thermo and init_column_itd (colpkg_init_itd:
!                 categoryThicknessLimits and the column package's hi_min, as
!                 seaice_init_column_physics_package_variables runs them), then
!                 column_itd_thermodynamics(domain, clock, ierr), its ierr
!                 logged as above (SEAICE_ERROR_COL_ITD_THERM after an abort);
!   ridging       colpkg_init_thermo and init_column_itd as above, then
!                 column_ridging(domain, ierr), its ierr logged as above
!                 (SEAICE_ERROR_COL_RIDGING after an abort);
!   biogeochemistry
!                 with the vertical salinity, colpkg_init_thermo (l_brine)
!                 and colpkg_init_zsalinity's dts_b (as
!                 seaice_init_column_physics_package_variables leaves them:
!                 restart_zsal, so no tracer or Rayleigh criterion is
!                 touched), then column_biogeochemistry(domain, ierr) (the
!                 column biogeochemistry's step: with the brine height, its
!                 update), its ierr logged as above (SEAICE_ERROR_COL_BGC
!                 after an abort);
!   prep_radiation
!                 column_prep_radiation;
!   radiation     colpkg_init_thermo (heat_capacity), colpkg_init_orbit for
!                 the delta-Eddington shortwave (as
!                 seaice_init_column_shortwave calls it), then
!                 column_radiation(domain, clock, .false.);
!   init_shortwave
!                 colpkg_init_thermo, then seaice_init_column_shortwave(domain,
!                 clock) (the orbit for the delta-Eddington shortwave,
!                 column_radiation at initialisation and the albedos);
!   column_step   colpkg_init_thermo, init_column_itd and, for the
!                 delta-Eddington shortwave, colpkg_init_orbit (as seaice_init
!                 runs them), then oracle_n_steps times the column package's
!                 part of seaice_core_run (mpas_seaice_core.F,
!                 mpas_seaice_time_integration.F:42-174,240-256): from the
!                 second step on seaice_timestep_finalize's
!                 seaice_column_reinitialize_fluxes, then the clock advanced,
!                 seaice_column_reinitialize_diagnostics_thermodynamics and
!                 _bgc, seaice_column_predynamics_time_integration,
!                 config_dynamics_subcycle_number times
!                 seaice_column_reinitialize_diagnostics_dynamics and
!                 seaice_column_dynamics_time_integration (the velocity solver
!                 and the advection between them are the dynamics' step, not
!                 run here), and seaice_column_postdynamics_time_integration
!                 (the shortwave and seaice_column_coupling_prep).  The last
!                 step is not finalized, so the stream carries the fluxes the
!                 coupler reads.  Each step returns on ierr > 0 as
!                 seaice_timestep does; ierr is logged as above;
!   seaice_step   the column setup of column_step, then run_seaice_step.F90's
!                 (seaice_init_velocity_solver's time steps, seaice_init_evp,
!                 seaice_init_advection, seaice_initialize_time_diagnostics),
!                 then oracle_n_steps times seaice_core_run's loop: the clock
!                 advanced, seaice_timestep (cut verbatim, with the column
!                 package, the velocity solver and the incremental remap
!                 compiled verbatim: colpkg_ice_strength is the column
!                 package's own) and seaice_timestep_finalize, the last step's
!                 only under oracle_finalize_last.  On ierr > 0 the loop stops
!                 (seaice_check_critical_error would stop the model); ierr and
!                 the step are logged ('TOBJ IERR n', 'TOBJ STEP n');
!   ocean_mixed_layer
!                 seaice_column_ocean_mixed_layer (colpkg_atm_boundary for the
!                 open ocean, colpkg_ocn_mixed_layer, and the test ice shelf's
!                 zeroing), as seaice_column_coupling_prep runs it first under
!                 config_use_ocean_mixed_layer.
!
! The Registry struct alternative_atmos_forcing is longer than the stream's
! 24-character pool names; the stream spells it 'alternative_atmos_forcin'
! and the driver files that pool under the full name as well.
!
! LOG.txt gets the tracer object ('TOBJ name values...': nTracers, the
! physics tracer indices, parentIndex, firstAncestorMask (as integers),
! ancestorNumber, ancestorIndices, and the column package's nt_Tsfc and
! nt_fbri, and with the vertical salinity its nt_bgc_S as 'colpkg_zsal') and
! every warning and error the routines log.
program run_column
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_warn_unit, stub_log_err_unit
  use mpas_timekeeping
  use seaice_column
  use seaice_initialize, only: init_ice_state, initialize_coupler_fields
  use seaice_forcing, only: seaice_reset_coupler_fluxes
  use ice_colpkg_tracers, only: nt_Tsfc, nt_fbri, nt_bgc_S
  use ice_colpkg, only: colpkg_init_thermo, colpkg_init_orbit, colpkg_init_zsalinity
  use seaice_velocity_solver_constitutive_relation, only: seaice_init_evp
  use seaice_diagnostics, only: seaice_initialize_time_diagnostics
  use seaice_advection, only: seaice_init_advection
  use seaice_time_integration, only: seaice_timestep, seaice_timestep_finalize
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: inputDt
  real(kind=RKIND), pointer :: iceAgeTimeOffset
  character(len=StrKIND), pointer :: stage, updateType, config_start_time, clockStep, config_calendar_type
  character(len=StrKIND), pointer :: shortwaveType
  logical :: orbitAbort
  character(len=StrKIND) :: orbitMessage
  integer :: p, logUnit, ierr, iStep, iSubcycle
  integer, pointer :: nSteps, nSubcycles, nElasticSubcycles
  real(kind=RKIND), pointer :: config_dt, dynamicsTimeStep, elasticTimeStep
  logical, pointer :: finalizeLast
  logical, pointer :: useShortwave
  integer, pointer :: nIceLayers
  real(kind=RKIND), allocatable :: sprofile(:)
  logical, pointer :: useZSalinity
  integer, pointer :: nBioLayers, nCategoriesZ
  logical :: rayleighStandIn
  real(kind=RKIND) :: rayleighRealStandIn
  real(kind=RKIND) :: zsalStandIn(1,1)
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
  call stub_pool_put_subpool(domain % blocklist % structs, 'alternative_atmos_forcing', &
       pool_named(pools, 'alternative_atmos_forcin'))
  ! the framework sets the calendar before the clock (mpas_framework.F:111-116)
  call mpas_pool_get_config(pools % configs, 'config_calendar_type', config_calendar_type)
  call mpas_timekeeping_init(config_calendar_type)
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
  case ('vertical_thermodynamics')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     ierr = 0
     call column_vertical_thermodynamics(domain, domain % clock, ierr)
     call write_ints('IERR', [ierr])
  case ('itd_thermodynamics')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call init_column_itd(domain)
     ierr = 0
     call column_itd_thermodynamics(domain, domain % clock, ierr)
     call write_ints('IERR', [ierr])
  case ('ridging')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call init_column_itd(domain)
     ierr = 0
     call column_ridging(domain, ierr)
     call write_ints('IERR', [ierr])
  case ('biogeochemistry')
     call mpas_pool_get_config(pools % configs, 'config_use_vertical_zsalinity', useZSalinity)
     if (useZSalinity) then
        call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
        call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nBioLayers', nBioLayers)
        call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nCategories', nCategoriesZ)
        allocate(sprofile(nIceLayers + 1))
        call colpkg_init_thermo(nIceLayers, sprofile)
        call colpkg_init_zsalinity(nBioLayers, 0, .true., rayleighStandIn, rayleighRealStandIn, zsalStandIn, &
             1, nCategoriesZ, 0.0_RKIND)
     endif
     ierr = 0
     call column_biogeochemistry(domain, ierr)
     call write_ints('IERR', [ierr])
  case ('prep_radiation')
     call column_prep_radiation(domain)
  case ('radiation')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call mpas_pool_get_config(pools % configs, 'config_shortwave_type', shortwaveType)
     if (trim(shortwaveType) == 'dEdd') then
        call colpkg_init_orbit(orbitAbort, orbitMessage)
        if (orbitAbort) then
           print *, 'run_column: colpkg_init_orbit: ', trim(orbitMessage)
           stop 3
        end if
     end if
     call column_radiation(domain, domain % clock, .false.)
  case ('init_shortwave')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call seaice_init_column_shortwave(domain, domain % clock)
  case ('column_step')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call init_column_itd(domain)
     call mpas_pool_get_config(pools % configs, 'config_shortwave_type', shortwaveType)
     call mpas_pool_get_config(pools % configs, 'config_use_column_shortwave', useShortwave)
     if (useShortwave .and. trim(shortwaveType) == 'dEdd') then
        call colpkg_init_orbit(orbitAbort, orbitMessage)
        if (orbitAbort) then
           print *, 'run_column: colpkg_init_orbit: ', trim(orbitMessage)
           stop 3
        end if
     end if
     call mpas_pool_get_config(pools % configs, 'oracle_n_steps', nSteps)
     call mpas_pool_get_config(pools % configs, 'config_dynamics_subcycle_number', nSubcycles)
     ierr = 0
     steps: do iStep = 1, nSteps
        if (iStep > 1) call seaice_column_reinitialize_fluxes(domain)
        call mpas_advance_clock(domain % clock)
        call seaice_column_reinitialize_diagnostics_thermodynamics(domain)
        call seaice_column_reinitialize_diagnostics_bgc(domain)
        call seaice_column_predynamics_time_integration(domain, domain % clock, ierr)
        if (ierr > 0) exit steps
        do iSubcycle = 1, nSubcycles
           call seaice_column_reinitialize_diagnostics_dynamics(domain)
           call seaice_column_dynamics_time_integration(domain, domain % clock, ierr)
           if (ierr > 0) exit steps
        end do
        call seaice_column_postdynamics_time_integration(domain, domain % clock)
     end do steps
     call write_ints('IERR', [ierr])
  case ('seaice_step')
     call mpas_pool_get_dimension(pool_named(pools, 'mesh'), 'nIceLayers', nIceLayers)
     allocate(sprofile(nIceLayers + 1))
     call colpkg_init_thermo(nIceLayers, sprofile)
     call init_column_itd(domain)
     call mpas_pool_get_config(pools % configs, 'config_shortwave_type', shortwaveType)
     call mpas_pool_get_config(pools % configs, 'config_use_column_shortwave', useShortwave)
     if (useShortwave .and. trim(shortwaveType) == 'dEdd') then
        call colpkg_init_orbit(orbitAbort, orbitMessage)
        if (orbitAbort) then
           print *, 'run_column: colpkg_init_orbit: ', trim(orbitMessage)
           stop 3
        end if
     end if
     ! seaice_init_velocity_solver's time steps (mpas_seaice_velocity_solver.F:126-127)
     call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
     call mpas_pool_get_config(pools % configs, 'config_dynamics_subcycle_number', nSubcycles)
     call mpas_pool_get_config(pools % configs, 'config_elastic_subcycle_number', nElasticSubcycles)
     call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'dynamicsTimeStep', dynamicsTimeStep)
     call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'elasticTimeStep', elasticTimeStep)
     dynamicsTimeStep = config_dt / real(nSubcycles,RKIND)
     elasticTimeStep = dynamicsTimeStep / real(nElasticSubcycles,RKIND)
     call seaice_init_evp(domain)
     call seaice_init_advection(domain)
     call seaice_initialize_time_diagnostics(domain)
     call mpas_pool_get_config(pools % configs, 'oracle_n_steps', nSteps)
     call mpas_pool_get_config(pools % configs, 'oracle_finalize_last', finalizeLast)
     ierr = 0
     do iStep = 1, nSteps
        call mpas_advance_clock(domain % clock)
        call seaice_timestep(domain, domain % clock, iStep, ierr)
        if (ierr > 0) exit
        if (iStep < nSteps .or. finalizeLast) call seaice_timestep_finalize(domain)
     end do
     call write_ints('IERR', [ierr])
     call write_ints('STEP', [min(iStep, nSteps)])
  case ('ocean_mixed_layer')
     call seaice_column_ocean_mixed_layer(domain)
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
      call mpas_pool_get_config(pools % configs, 'config_use_vertical_zsalinity', useZSalinity)
      if (useZSalinity) call write_ints('colpkg_zsal', [nt_bgc_S])
    end associate
  end subroutine write_tracer_object

end program run_column
