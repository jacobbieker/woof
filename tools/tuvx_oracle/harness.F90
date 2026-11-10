! The TUV-x photolysis oracle harness (tools/tuvx_oracle/build.sh).  Apache-2.0.
!
! One TUV-x core is built from the committed configuration with the column the host supplies: the
! height grid and the temperature, air, O2 and O3 profiles (TUV-x's from-host grid and profiles).
! Each case on standard input updates them and runs the core; the photolysis rate constants come
! back as binary64 words in hex, edge by reaction.
!
! input:   config <path> / nlev <n> / then per case:
!          case / sza <deg> / esd <AU> / edges <n+1 km> / temperature <n+1 K> /
!          air <n+1 edge values> <n layer densities> / O2 ... / O3 ... / end
program tuvx_harness

  use musica_constants,     only : dk => musica_dk
  use musica_string,        only : string_t
  use tuvx_core,            only : core_t
  use tuvx_grid_from_host,  only : grid_from_host_t, grid_updater_t
  use tuvx_grid_warehouse,  only : grid_warehouse_t
  use tuvx_profile_from_host, only : profile_from_host_t, profile_updater_t
  use tuvx_profile_warehouse, only : profile_warehouse_t

  implicit none

  character(len=4096) :: path
  character(len=32)   :: key
  integer :: nlev, i, j, ncase
  class(core_t), pointer :: core
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
  real(dk), allocatable :: rates(:,:)

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
      u_height = core%get_updater( height )
      u_temperature = core%get_updater( temperature )
      u_air = core%get_updater( air )
      u_o2 = core%get_updater( o2 )
      u_o3 = core%get_updater( o3 )
      allocate( rates( nlev+1, size( labels ) ) )
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
  end do
10 continue

contains

  subroutine set( updater, edge_values, layer_densities )
    type(profile_updater_t), intent(in) :: updater
    real(dk), intent(in) :: edge_values(:)
    real(dk), optional, intent(in) :: layer_densities(:)
    type(profile_updater_t) :: local
    local = updater
    call local%update( edge_values = edge_values, layer_densities = layer_densities )
  end subroutine set

end program tuvx_harness
