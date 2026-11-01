! What the cut MPAS routines USE, and no more: the kind (RKIND = 8, MPAS's
! default double-precision build, or with -DPN_SINGLE 4, the single build the
! hex dycore follows), an empty pool type, a pool whose get_* calls hand out
! module arrays the driver fills, a silent log, and nRelaxZone.
#ifdef PN_SINGLE
#define PN_RK 4
#else
#define PN_RK 8
#endif

module mpas_kind_types
   implicit none
   integer, parameter :: RKIND = PN_RK
   integer, parameter :: StrKIND = 512
end module mpas_kind_types

module mpas_derived_types
   implicit none
   type mpas_pool_type
      integer :: unused = 0
   end type mpas_pool_type
end module mpas_derived_types

module mpas_log
   use mpas_kind_types
   implicit none
   integer, parameter :: MPAS_LOG_WARN = 2, MPAS_LOG_ERR = 3, MPAS_LOG_CRIT = 4
contains
   subroutine mpas_log_write(message, messageType, realArgs, intArgs)
      character(len=*), intent(in) :: message
      integer, intent(in), optional :: messageType
      real(kind=RKIND), intent(in), optional :: realArgs(:)
      integer, intent(in), optional :: intArgs(:)
   end subroutine mpas_log_write
end module mpas_log

module pn_relax_constants
   implicit none
   ! mpas_atm_boundaries.F: integer, parameter :: nSpecZone = 2, nRelaxZone = 5
   integer, parameter :: nSpecZone = 2, nRelaxZone = 5
end module pn_relax_constants

module pn_store
   use mpas_kind_types
   implicit none
   integer, target :: vertexDegree = 3, nCells = 0, nVertices = 0
   logical, target :: on_a_sphere = .true.
   real(kind=RKIND), target :: sphere_radius = 1.0_RKIND
   real(kind=RKIND), target :: config_relax_zone_divdamp_coef = 6.0_RKIND
   real(kind=RKIND), dimension(:), allocatable, target :: xCell, yCell, zCell, &
      xVertex, yVertex, zVertex, dcEdge, dvEdge, invDcEdge, invDvEdge, invAreaCell, &
      invAreaTriangle, meshScalingRegionalCell, meshScalingRegionalEdge
   real(kind=RKIND), dimension(:,:), allocatable, target :: edgesOnCell_sign, &
      edgesOnVertex_sign, tend_u, tend_rho_zz, tend_theta_m, tend_w, ru, theta_m, rho_zz
   integer, dimension(:), allocatable, target :: nEdgesOnCell, bdyMaskCell, bdyMaskEdge
   integer, dimension(:,:), allocatable, target :: cellsOnVertex, verticesOnCell, &
      cellsOnEdge, verticesOnEdge, edgesOnCell, edgesOnVertex
end module pn_store

module mpas_pool_routines
   use mpas_kind_types
   use mpas_derived_types
   use pn_store
   implicit none
   private
   public :: mpas_pool_get_array, mpas_pool_get_dimension, mpas_pool_get_config
   interface mpas_pool_get_array
      module procedure get_r1, get_r2, get_i1, get_i2
   end interface
   interface mpas_pool_get_config
      module procedure get_config_l, get_config_r
   end interface
contains
   subroutine missing(name)
      character(len=*), intent(in) :: name
      write(0, *) 'pn stub pool: no array ', trim(name)
      stop 9
   end subroutine missing

   subroutine get_r1(pool, name, array, timeLevel)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      real(kind=RKIND), dimension(:), pointer :: array
      integer, intent(in), optional :: timeLevel
      select case (name)
      case ('xCell'); array => xCell
      case ('yCell'); array => yCell
      case ('zCell'); array => zCell
      case ('xVertex'); array => xVertex
      case ('yVertex'); array => yVertex
      case ('zVertex'); array => zVertex
      case ('dcEdge'); array => dcEdge
      case ('dvEdge'); array => dvEdge
      case ('invDcEdge'); array => invDcEdge
      case ('invDvEdge'); array => invDvEdge
      case ('invAreaCell'); array => invAreaCell
      case ('invAreaTriangle'); array => invAreaTriangle
      case ('meshScalingRegionalCell'); array => meshScalingRegionalCell
      case ('meshScalingRegionalEdge'); array => meshScalingRegionalEdge
      case default; call missing(name)
      end select
   end subroutine get_r1

   subroutine get_r2(pool, name, array, timeLevel)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      real(kind=RKIND), dimension(:,:), pointer :: array
      integer, intent(in), optional :: timeLevel
      ! The tend pool's u/rho_zz/theta_m/w, diag's ru, state's theta_m and
      ! rho_zz (time level 2): the names are unique across the pools used.
      select case (name)
      case ('edgesOnCell_sign'); array => edgesOnCell_sign
      case ('edgesOnVertex_sign'); array => edgesOnVertex_sign
      case ('u'); array => tend_u
      case ('w'); array => tend_w
      case ('ru'); array => ru
      case ('theta_m')
         if (present(timeLevel)) then
            array => theta_m
         else
            array => tend_theta_m
         end if
      case ('rho_zz')
         if (present(timeLevel)) then
            array => rho_zz
         else
            array => tend_rho_zz
         end if
      case default; call missing(name)
      end select
   end subroutine get_r2

   subroutine get_i1(pool, name, array, timeLevel)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      integer, dimension(:), pointer :: array
      integer, intent(in), optional :: timeLevel
      select case (name)
      case ('nEdgesOnCell'); array => nEdgesOnCell
      case ('bdyMaskCell'); array => bdyMaskCell
      case ('bdyMaskEdge'); array => bdyMaskEdge
      case default; call missing(name)
      end select
   end subroutine get_i1

   subroutine get_i2(pool, name, array, timeLevel)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      integer, dimension(:,:), pointer :: array
      integer, intent(in), optional :: timeLevel
      select case (name)
      case ('cellsOnVertex'); array => cellsOnVertex
      case ('verticesOnCell'); array => verticesOnCell
      case ('cellsOnEdge'); array => cellsOnEdge
      case ('verticesOnEdge'); array => verticesOnEdge
      case ('edgesOnCell'); array => edgesOnCell
      case ('edgesOnVertex'); array => edgesOnVertex
      case default; call missing(name)
      end select
   end subroutine get_i2

   subroutine mpas_pool_get_dimension(pool, name, value)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      integer, pointer :: value
      select case (name)
      case ('vertexDegree'); value => vertexDegree
      case ('nCells'); value => nCells
      case ('nVertices'); value => nVertices
      case default; call missing(name)
      end select
   end subroutine mpas_pool_get_dimension

   subroutine get_config_l(pool, name, value)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      logical, pointer :: value
      if (name /= 'on_a_sphere') call missing(name)
      value => on_a_sphere
   end subroutine get_config_l

   subroutine get_config_r(pool, name, value)
      type(mpas_pool_type), intent(in) :: pool
      character(len=*), intent(in) :: name
      real(kind=RKIND), pointer :: value
      select case (name)
      case ('sphere_radius'); value => sphere_radius
      case ('config_relax_zone_divdamp_coef'); value => config_relax_zone_divdamp_coef
      case default; call missing(name)
      end select
   end subroutine get_config_r
end module mpas_pool_routines
