! The stand-in for MPAS-Ocean's CVMix module (mpas_ocn_vmix_cvmix.F), which
! links the external CVMix library: the vertical-mixing oracle compiles
! mpas_ocn_vmix.F against it.  Like the real routines, ocn_vmix_coefs_cvmix_build
! does nothing when config_use_cvmix is off; the stand-ins stop, naming the
! routine, when it is on.  The dummy declarations are copied from v8.4.2.
module ocn_vmix_cvmix
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  implicit none
  private
  public :: ocn_vmix_coefs_cvmix_build, ocn_vmix_cvmix_init

contains

  subroutine stop_if_on(name)
    character(len=*), intent(in) :: name
    logical, pointer :: config_use_cvmix
    call mpas_pool_get_config(ocnConfigs, 'config_use_cvmix', config_use_cvmix)
    if (config_use_cvmix) then
      print *, 'oracle stub: ', name, ' is not compiled here'
      stop 8
    end if
  end subroutine stop_if_on

  subroutine ocn_vmix_coefs_cvmix_build(meshPool, statePool, forcingPool, diagnosticsPool, err, timeLevelIn)
    type (mpas_pool_type), intent(in) :: meshPool
    integer, intent(in), optional :: timeLevelIn
    type (mpas_pool_type), intent(inout) :: statePool, forcingPool, diagnosticsPool
    integer, intent(out) :: err
    err = 0
    call stop_if_on('ocn_vmix_coefs_cvmix_build')
  end subroutine ocn_vmix_coefs_cvmix_build

  subroutine ocn_vmix_cvmix_init(domain, err)
    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
    call stop_if_on('ocn_vmix_cvmix_init')
  end subroutine ocn_vmix_cvmix_init

end module ocn_vmix_cvmix
