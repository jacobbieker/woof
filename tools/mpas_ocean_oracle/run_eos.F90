! Drive MPAS-Ocean v8.4.2's equation of state and freezing temperature on one
! block and write every output word: the reference the GPU port is held
! against.
!
!   run_eos INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_eos_oracle (raw little-endian
! words, Fortran array order; reals are RKIND wide).  ocn_constants_init and
! ocn_equation_of_state_init read the configs (config_eos_type 'linear' or
! 'jm'), then ocn_equation_of_state_density runs the calls MPAS-Ocean makes
! (ocn_diagnostic_solve, ocn_vmix_coefs_rich), each into its own arrays, which
! start from the case's sentinel words:
!
!   1  k_displaced 0, 'relative', with both expansion coefficients (in situ)
!   2  k_displaced 0, 'relative', density only
!   3  k_displaced 1, 'absolute' (potential density)
!   4  k_displaced 1, 'relative' (displaced density)
!   5  k_displaced 0, 'surfaceDisplaced', with both coefficients
!   6  k_displaced nVertLevels, 'absolute'
!
! then ocn_freezing_temperature and ocn_freezing_temperature_salinity_deriv
! elementwise over the case's (salinity, pressure) pairs, open ocean and land-
! ice cavity.

program run_eos
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_equation_of_state
  implicit none

  character(len=*), parameter :: eosTypes(2) = [character(len=32) :: 'linear', 'jm']
  type(MPAS_pool_type), pointer :: mesh, configs, packages, scratch, state, tracersPool, diagnostics
  integer, pointer :: nVertLevels, indexT, indexS
  integer :: nCells, nTracers, eosType, err, u, iCell, k
  character(len=StrKIND), pointer :: eosTypeConfig
  real(kind=RKIND), pointer :: density0, densityref, alpha, beta, Tref, Sref, &
       open0, openS, openP, openPS, openRef, cav0, cavS, cavP, cavPS, cavRef
  real(kind=RKIND) :: reals(15)
  integer, dimension(:), pointer :: maxLevelCell
  real(kind=RKIND), dimension(:), pointer :: refBottomDepth
  real(kind=RKIND), dimension(:,:,:), pointer :: activeTracers
  real(kind=RKIND), dimension(:,:), pointer :: tracersSurfaceValue
  real(kind=RKIND), dimension(:,:), allocatable :: sentinel, freezeS, freezeP, &
       density1, alpha1, beta1, density2, density3, density4, density5, alpha5, beta5, density6, &
       tfOpen, tfCavity, dtfOpen, dtfCavity
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nVertLevels, indexT, indexS, eosTypeConfig, density0, densityref, alpha, beta, Tref, Sref, &
       open0, openS, openP, openPS, openRef, cav0, cavS, cavP, cavPS, cavRef)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nVertLevels, nTracers, indexT, indexS, eosType
  ! density0; linear densityref, alpha, beta, Tref, Sref; open-ocean then
  ! land-ice-cavity freezing coefficients 0, S, p, pS and reference pressure
  read(u) density0, reals(1:14)
  densityref = reals(1); alpha = reals(2); beta = reals(3); Tref = reals(4); Sref = reals(5)
  open0 = reals(6); openS = reals(7); openP = reals(8); openPS = reals(9); openRef = reals(10)
  cav0 = reals(11); cavS = reals(12); cavP = reals(13); cavPS = reals(14)
  read(u) cavRef
  eosTypeConfig = eosTypes(eosType)

  allocate(maxLevelCell(nCells), refBottomDepth(nVertLevels), activeTracers(nTracers,nVertLevels,nCells), &
       tracersSurfaceValue(nTracers,nCells), sentinel(nVertLevels,nCells), freezeS(nVertLevels,nCells), &
       freezeP(nVertLevels,nCells))
  read(u) maxLevelCell
  read(u) refBottomDepth, activeTracers, tracersSurfaceValue, sentinel, freezeS, freezeP
  close(u)

  allocate(density1, alpha1, beta1, density2, density3, density4, density5, alpha5, beta5, density6, &
       tfOpen, tfCavity, dtfOpen, dtfCavity, mold=sentinel)
  density1 = sentinel; alpha1 = sentinel; beta1 = sentinel; density2 = sentinel; density3 = sentinel
  density4 = sentinel; density5 = sentinel; alpha5 = sentinel; beta5 = sentinel; density6 = sentinel
  tfOpen = sentinel; tfCavity = sentinel; dtfOpen = sentinel; dtfCavity = sentinel

  allocate(mesh, configs, packages, scratch, state, tracersPool, diagnostics)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_array(mesh, "maxLevelCell", maxLevelCell)
  call stub_pool_put_array(mesh, "refBottomDepth", refBottomDepth)
  call stub_pool_put_dimension(tracersPool, "index_temperature", indexT)
  call stub_pool_put_dimension(tracersPool, "index_salinity", indexS)
  call stub_pool_put_array(tracersPool, "activeTracers", activeTracers)
  call stub_pool_put_subpool(state, "tracers", tracersPool)
  call stub_pool_put_array(diagnostics, "tracersSurfaceValue", tracersSurfaceValue)

  call stub_pool_put_config(configs, "config_density0", density0)
  call stub_pool_put_config(configs, "config_eos_type", eosTypeConfig)
  call stub_pool_put_config(configs, "config_eos_linear_densityref", densityref)
  call stub_pool_put_config(configs, "config_eos_linear_alpha", alpha)
  call stub_pool_put_config(configs, "config_eos_linear_beta", beta)
  call stub_pool_put_config(configs, "config_eos_linear_Tref", Tref)
  call stub_pool_put_config(configs, "config_eos_linear_Sref", Sref)
  call stub_pool_put_config(configs, "config_open_ocean_freezing_temperature_coeff_0", open0)
  call stub_pool_put_config(configs, "config_open_ocean_freezing_temperature_coeff_S", openS)
  call stub_pool_put_config(configs, "config_open_ocean_freezing_temperature_coeff_p", openP)
  call stub_pool_put_config(configs, "config_open_ocean_freezing_temperature_coeff_pS", openPS)
  call stub_pool_put_config(configs, "config_open_ocean_freezing_temperature_reference_pressure", openRef)
  call stub_pool_put_config(configs, "config_land_ice_cavity_freezing_temperature_coeff_0", cav0)
  call stub_pool_put_config(configs, "config_land_ice_cavity_freezing_temperature_coeff_S", cavS)
  call stub_pool_put_config(configs, "config_land_ice_cavity_freezing_temperature_coeff_p", cavP)
  call stub_pool_put_config(configs, "config_land_ice_cavity_freezing_temperature_coeff_pS", cavPS)
  call stub_pool_put_config(configs, "config_land_ice_cavity_freezing_temperature_reference_pressure", cavRef)

  call ocn_constants_init(configs, packages)
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 9

  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, 0, 'relative', density1, err, &
       alpha1, beta1, timeLevelIn=1)
  if (err /= 0) stop 9
  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, 0, 'relative', density2, err, &
       timeLevelIn=1)
  if (err /= 0) stop 9
  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, 1, 'absolute', density3, err, &
       timeLevelIn=1)
  if (err /= 0) stop 9
  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, 1, 'relative', density4, err, &
       timeLevelIn=1)
  if (err /= 0) stop 9
  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, 0, 'surfaceDisplaced', density5, &
       err, alpha5, beta5, 1)
  if (err /= 0) stop 9
  call ocn_equation_of_state_density(state, diagnostics, mesh, scratch, nCells, nVertLevels, 'absolute', density6, &
       err, timeLevelIn=1)
  if (err /= 0) stop 9

  do iCell = 1, nCells
    do k = 1, nVertLevels
      tfOpen(k,iCell) = ocn_freezing_temperature(freezeS(k,iCell), freezeP(k,iCell), .false.)
      tfCavity(k,iCell) = ocn_freezing_temperature(freezeS(k,iCell), freezeP(k,iCell), .true.)
      dtfOpen(k,iCell) = ocn_freezing_temperature_salinity_deriv(freezeS(k,iCell), freezeP(k,iCell), .false.)
      dtfCavity(k,iCell) = ocn_freezing_temperature_salinity_deriv(freezeS(k,iCell), freezeP(k,iCell), .true.)
    end do
  end do

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) density1, alpha1, beta1, density2, density3, density4, density5, alpha5, beta5, density6, &
       tfOpen, tfCavity, dtfOpen, dtfCavity
  close(u)

end program run_eos
