! The TUV-x photolysis oracle harness (tools/tuvx_oracle/build.sh).  Apache-2.0.
!
! One TUV-x core is built from the committed configuration with the column the host supplies: the
! height grid and the temperature, air, O2 and O3 profiles (TUV-x's from-host grid and profiles).
! Each case on standard input updates them and runs the core; the photolysis rate constants come
! back as binary64 words in hex, edge by reaction.  Once the core is built, the configuration-time pieces
! the port is graded on piece by piece follow: the wavelength grid's edges ('wedge') and the
! extraterrestrial flux on it ('etf', photon cm-2 s-1 per bin) and the Earth-Sun distance profile on the
! configuration's time grid ('esd', AU).  After every case: each reaction's cross
! section ('xs') and quantum yield ('qy') and the actinic flux's direct, diffuse-up and diffuse-down parts
! ('fdr', 'fup', 'fdn'), each (height edge, wavelength bin).  Then the radiative-transfer intermediates,
! from a spherical geometry and Lyman-alpha / Schumann-Runge calculator the harness builds over the core's own
! (updated) grids and profiles exactly as the core does: the geometry ('nid' per edge counted from the top,
! 'dsdh' (edge from the top, layer)), the air and O2 vertical and slant columns ('avc', 'asc', 'ovc', 'osc'),
! the O2 reaction's cross section with the bands applied ('lsx', (height edge, wavelength bin)), each
! radiator's layer optical depth after the run ('rod', radiator 1 air, 2 O2 (bands applied), 3 O3), and the
! radiators' accumulated layer optical depth, single-scattering albedo and asymmetry factor ('aod', 'ass',
! 'ag'), each (layer, wavelength bin).
!
! input:   config <path> / nlev <n> / then per case:
!          case / sza <deg> / esd <AU> / edges <n+1 km> / temperature <n+1 K> /
!          air <n+1 edge values> <n layer densities> / O2 ... / O3 ... / end
program tuvx_harness

  use musica_constants,     only : dk => musica_dk
  use musica_string,        only : string_t
  use tuvx_core,            only : core_t
  use tuvx_grid,            only : grid_t
  use tuvx_profile,         only : profile_t
  use tuvx_solver,          only : radiation_field_t
  use tuvx_grid_from_host,  only : grid_from_host_t, grid_updater_t
  use tuvx_grid_warehouse,  only : grid_warehouse_t
  use tuvx_profile_from_host, only : profile_from_host_t, profile_updater_t
  use tuvx_profile_warehouse, only : profile_warehouse_t
  use musica_config,        only : config_t
  use tuvx_la_sr_bands,     only : la_sr_bands_t
  use tuvx_spherical_geometry, only : spherical_geometry_t
  use tuvx_radiator,        only : radiator_t, radiator_state_t
  use tuvx_radiator_warehouse, only : radiator_warehouse_t

  implicit none

  character(len=4096) :: path
  character(len=32)   :: key
  integer :: nlev, i, j, ncase
  class(core_t), pointer :: core
  class(grid_t), pointer :: wavelength
  class(profile_t), pointer :: etf
  type(string_t) :: config_path
  type(grid_from_host_t), pointer :: height
  type(profile_from_host_t), pointer :: temperature, air, o2, o3
  type(grid_warehouse_t), pointer :: grids
  type(profile_warehouse_t), pointer :: profiles
  type(grid_updater_t) :: u_height
  type(profile_updater_t) :: u_temperature, u_air, u_o2, u_o3
  type(string_t), allocatable :: labels(:)
  real(dk) :: sza, esd
  real(dk), allocatable :: edges(:), tedge(:), aedge(:), alay(:), o2edge(:), o2lay(:), o3edge(:), o3lay(:)
  real(dk), allocatable :: rates(:,:), table(:,:)
  type(radiation_field_t) :: field
  type(config_t) :: config, o2_config
  type(spherical_geometry_t), pointer :: geometry
  type(la_sr_bands_t), pointer :: la_srb
  class(grid_warehouse_t), pointer :: core_grids
  class(profile_warehouse_t), pointer :: core_profiles
  class(radiator_warehouse_t), pointer :: core_radiators
  class(radiator_t), pointer :: radiator
  class(profile_t), pointer :: column_profile
  type(radiator_state_t) :: accumulated
  real(dk), allocatable :: avc(:), asc(:), ovc(:), osc(:)
  integer :: r, o2_reaction
  character(len=3), parameter :: radiator_names(3) = [ 'air', 'O2 ', 'O3 ' ]

  read(*,*) key, path
  read(*,*) key, nlev
  allocate( edges(nlev+1), tedge(nlev+1), aedge(nlev+1), alay(nlev), o2edge(nlev+1), o2lay(nlev), &
            o3edge(nlev+1), o3lay(nlev) )

  height => grid_from_host_t( "height", "km", nlev )
  grids => grid_warehouse_t( )
  call grids%add( height )
  temperature => profile_from_host_t( "temperature", "K", nlev )
  air => profile_from_host_t( "air", "molecule cm-3", nlev )
  o2 => profile_from_host_t( "O2", "molecule cm-3", nlev )
  o3 => profile_from_host_t( "O3", "molecule cm-3", nlev )
  profiles => profile_warehouse_t( )
  call profiles%add( temperature )
  call profiles%add( air )
  call profiles%add( o2 )
  call profiles%add( o3 )

  ncase = 0
  do
    read(*,*,end=10) key
    if( trim( key ) /= 'case' ) cycle
    read(*,*) key, sza
    read(*,*) key, esd
    read(*,*) key, edges
    read(*,*) key, tedge
    read(*,*) key, aedge, alay
    read(*,*) key, o2edge, o2lay
    read(*,*) key, o3edge, o3lay
    read(*,*) key
    if( ncase == 0 ) then
      ! the core takes its own copies, so the first column is set before it is built
      u_height = grid_updater_t( height )
      call u_height%update( edges = edges )
      call set( profile_updater_t( temperature ), tedge )
      call set( profile_updater_t( air ), aedge, alay )
      call set( profile_updater_t( o2 ), o2edge, o2lay )
      call set( profile_updater_t( o3 ), o3edge, o3lay )
      config_path = trim( path )
      core => core_t( config_path, grids, profiles )
      labels = core%photolysis_reaction_labels( )
      do j = 1, size( labels )
        write(*,'(a,i0,1x,a)') 'label ', j, labels(j)%val_
      end do
      wavelength => core%get_grid( "wavelength", "nm" )
      do i = 1, size( wavelength%edge_ )
        write(*,'(a,i0,1x,z16.16)') 'wedge ', i, transfer( wavelength%edge_(i), 0_8 )
      end do
      etf => core%get_profile( "extraterrestrial flux", "photon cm-2 s-1" )
      do i = 1, size( etf%mid_val_ )
        write(*,'(a,i0,1x,z16.16)') 'etf ', i, transfer( etf%mid_val_(i), 0_8 )
      end do
      deallocate( wavelength )
      column_profile => core%get_profile( "Earth-Sun distance", "AU" )
      do i = 1, size( column_profile%edge_val_ )
        write(*,'(a,i0,1x,z16.16)') 'esd ', i, transfer( column_profile%edge_val_(i), 0_8 )
      end do
      deallocate( column_profile )
      u_height = core%get_updater( height )
      u_temperature = core%get_updater( temperature )
      u_air = core%get_updater( air )
      u_o2 = core%get_updater( o2 )
      u_o3 = core%get_updater( o3 )
      allocate( rates( nlev+1, size( labels ) ) )
      core_grids => core%get_grid_warehouse( )
      core_profiles => core%get_profile_warehouse( )
      core_radiators => core%get_radiator_warehouse( )
      call config%from_file( trim( path ) )
      call config%get( 'O2 absorption', o2_config, 'harness' )
      geometry => spherical_geometry_t( core_grids )
      la_srb => la_sr_bands_t( o2_config, core_grids, core_profiles )
      o2_reaction = 0
      do j = 1, size( labels )
        if( labels(j)%val_ == 'O2+hv->O+O' ) o2_reaction = j
      end do
      allocate( avc( nlev ), asc( nlev+1 ), ovc( nlev ), osc( nlev+1 ) )
      allocate( accumulated%layer_G_( nlev, size( etf%mid_val_ ), 1 ) )
      deallocate( etf )
    end if
    call u_height%update( edges = edges )
    call u_temperature%update( edge_values = tedge )
    call u_air%update( edge_values = aedge, layer_densities = alay )
    call u_o2%update( edge_values = o2edge, layer_densities = o2lay )
    call u_o3%update( edge_values = o3edge, layer_densities = o3lay )
    call core%run( sza, esd, photolysis_rate_constants = rates )
    ncase = ncase + 1
    write(*,'(a,i0)') 'case ', ncase
    do j = 1, size( rates, 2 )
      do i = 1, size( rates, 1 )
        write(*,'(a,i0,1x,i0,1x,z16.16)') 'j ', j, i, transfer( rates(i,j), 0_8 )
      end do
    end do
    do j = 1, size( labels )
      table = core%get_photolysis_cross_section( labels(j) )
      call dump( 'xs', j, table )
      table = core%get_photolysis_quantum_yield( labels(j) )
      call dump( 'qy', j, table )
    end do
    field = core%get_radiation_field( )
    call dump( 'fdr', 0, field%fdr_ )
    call dump( 'fup', 0, field%fup_ )
    call dump( 'fdn', 0, field%fdn_ )
    call geometry%set_parameters( sza, core_grids )
    do i = 0, nlev
      write(*,'(a,1x,i0,1x,i0,1x,i0,1x,z16.16)') 'nid', 0, i+1, 1, transfer( real( geometry%nid_(i), dk ), 0_8 )
    end do
    call dump( 'dsdh', 0, geometry%dsdh_ )
    column_profile => core%get_profile( 'air', 'molecule cm-3' )
    call geometry%air_mass( column_profile%exo_layer_dens_, avc, asc )
    deallocate( column_profile )
    column_profile => core%get_profile( 'O2', 'molecule cm-3' )
    call geometry%air_mass( column_profile%exo_layer_dens_, ovc, osc )
    deallocate( column_profile )
    call dump( 'avc', 0, reshape( avc, [ nlev, 1 ] ) )
    call dump( 'asc', 0, reshape( asc, [ nlev+1, 1 ] ) )
    call dump( 'ovc', 0, reshape( ovc, [ nlev, 1 ] ) )
    call dump( 'osc', 0, reshape( osc, [ nlev+1, 1 ] ) )
    if( o2_reaction > 0 ) then
      table = core%get_photolysis_cross_section( labels(o2_reaction) )
      call la_srb%cross_section( core_grids, core_profiles, avc, asc, table, geometry )
      call dump( 'lsx', o2_reaction, table )
    end if
    do r = 1, size( radiator_names )
      radiator => core%get_radiator( trim( radiator_names(r) ) )
      call dump( 'rod', r, radiator%state_%layer_OD_ )
    end do
    call core_radiators%accumulate_states( accumulated )
    call dump( 'aod', 0, accumulated%layer_OD_ )
    call dump( 'ass', 0, accumulated%layer_SSA_ )
    call dump( 'ag', 0, accumulated%layer_G_(:,:,1) )
  end do
10 continue

contains

  subroutine dump( tag, j, values )
    character(len=*), intent(in) :: tag
    integer, intent(in) :: j
    real(dk), intent(in) :: values(:,:)
    integer :: i, k
    do k = 1, size( values, 2 )
      do i = 1, size( values, 1 )
        write(*,'(a,1x,i0,1x,i0,1x,i0,1x,z16.16)') tag, j, i, k, transfer( values(i,k), 0_8 )
      end do
    end do
  end subroutine dump

  subroutine set( updater, edge_values, layer_densities )
    type(profile_updater_t), intent(in) :: updater
    real(dk), intent(in) :: edge_values(:)
    real(dk), optional, intent(in) :: layer_densities(:)
    type(profile_updater_t) :: local
    local = updater
    call local%update( edge_values = edge_values, layer_densities = layer_densities )
  end subroutine set

end program tuvx_harness
