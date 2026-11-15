
  ! init_ice_state's other initial conditions and velocities are not part of
  ! the column oracle (the GPU port refuses them by name): they stop by name.

  subroutine init_ice_state_uniform_interior(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_state_uniform_interior is not part of the column oracle'
    stop 6
  end subroutine init_ice_state_uniform_interior

  subroutine init_ice_state_circle_of_ice(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_state_circle_of_ice is not part of the column oracle'
    stop 6
  end subroutine init_ice_state_circle_of_ice

  subroutine init_ice_state_special(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_state_special is not part of the column oracle'
    stop 6
  end subroutine init_ice_state_special

  subroutine init_ice_state_random_coverage(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_state_random_coverage is not part of the column oracle'
    stop 6
  end subroutine init_ice_state_random_coverage

  subroutine init_ice_velocity_ocean(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_velocity_ocean is not part of the column oracle'
    stop 6
  end subroutine init_ice_velocity_ocean

  subroutine init_ice_velocity_random(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_velocity_random is not part of the column oracle'
    stop 6
  end subroutine init_ice_velocity_random

  subroutine init_ice_velocity_special(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: init_ice_velocity_special is not part of the column oracle'
    stop 6
  end subroutine init_ice_velocity_special

end module seaice_initialize
