! Drive the cut MPAS routines on the inputs make_fixture.py wrote.
!
!   run_oracle MODE DIR
!
! MODE bary  (RKIND 8): mpas_calculate_barycentric_weights_for_points
! MODE apply (RKIND 4): mpas_cells_to_points_using_baryweights, per level
! MODE relax (RKIND 4): atm_bdy_adjust_dynamics_relaxzone_tend on one level
!                       at a time, every active cell/edge at ring strength
!                       5 - d on the level d below the top (the hex top zone)
!
! DIR/dims.txt holds the sizes (list-directed); every array is DIR/<name>.bin,
! raw little-endian, Fortran order (C order of the transposed NumPy array);
! connectivity is 1-based.  Outputs are written as DIR/out_<name>.bin.
program run_oracle
   use mpas_kind_types
   use mpas_derived_types
   use pn_store
   use pn_mpas_geometry
#ifdef PN_SINGLE
   use pn_mpas_relax
#endif
   implicit none
   character(len=512) :: mode, dir
   type(mpas_pool_type) :: pool
   integer :: ierr

   call get_command_argument(1, mode)
   call get_command_argument(2, dir)
   select case (trim(mode))
#ifndef PN_SINGLE
   case ('bary'); call do_bary()
#else
   case ('apply'); call do_apply()
   case ('relax'); call do_relax()
#endif
   case default
      write(0, *) 'mode not built into this executable: ', trim(mode)
      stop 2
   end select

contains

   function path(name) result(p)
      character(len=*), intent(in) :: name
      character(len=1024) :: p
      p = trim(dir) // '/' // name // '.bin'
   end function path

   subroutine read_r(name, a)
      character(len=*), intent(in) :: name
      real(kind=RKIND), intent(out) :: a(*)
      integer :: u, n
      open(newunit=u, file=trim(path(name)), access='stream', form='unformatted', status='old')
      inquire(unit=u, size=n)
      read(u) a(1:n * 8 / storage_size(a(1)))
      close(u)
   end subroutine read_r

   subroutine read_i(name, a)
      character(len=*), intent(in) :: name
      integer, intent(out) :: a(*)
      integer :: u, n
      open(newunit=u, file=trim(path(name)), access='stream', form='unformatted', status='old')
      inquire(unit=u, size=n)
      read(u) a(1:n / 4)
      close(u)
   end subroutine read_i

   subroutine write_r(name, a, n)
      character(len=*), intent(in) :: name
      integer, intent(in) :: n
      real(kind=RKIND), intent(in) :: a(n)
      integer :: u
      open(newunit=u, file=trim(dir)//'/out_'//name//'.bin', access='stream', &
           form='unformatted', status='replace')
      write(u) a
      close(u)
   end subroutine write_r

   subroutine write_i(name, a, n)
      character(len=*), intent(in) :: name
      integer, intent(in) :: n
      integer, intent(in) :: a(n)
      integer :: u
      open(newunit=u, file=trim(dir)//'/out_'//name//'.bin', access='stream', &
           form='unformatted', status='replace')
      write(u) a
      close(u)
   end subroutine write_i

   subroutine read_dims(d, n)
      integer, intent(in) :: n
      integer, intent(out) :: d(n)
      integer :: u
      open(newunit=u, file=trim(dir)//'/dims.txt', status='old')
      read(u, *) d
      close(u)
   end subroutine read_dims

#ifndef PN_SINGLE
   subroutine do_bary()
      integer :: d(4), maxEdges, nPoints
      real(kind=RKIND) :: radius(1)
      real(kind=RKIND), allocatable :: xp(:), yp(:), zp(:), weights(:,:)
      integer, allocatable :: search(:), cells(:,:)
      call read_dims(d, 4)
      nCells = d(1); nVertices = d(2); maxEdges = d(3); nPoints = d(4)
      allocate(xCell(nCells), yCell(nCells), zCell(nCells))
      allocate(xVertex(nVertices), yVertex(nVertices), zVertex(nVertices))
      allocate(cellsOnVertex(3, nVertices), verticesOnCell(maxEdges, nCells), nEdgesOnCell(nCells))
      allocate(xp(nPoints), yp(nPoints), zp(nPoints), search(nPoints))
      allocate(cells(3, nPoints), weights(3, nPoints))
      call read_r('xCell', xCell); call read_r('yCell', yCell); call read_r('zCell', zCell)
      call read_r('xVertex', xVertex); call read_r('yVertex', yVertex); call read_r('zVertex', zVertex)
      call read_i('cellsOnVertex', cellsOnVertex); call read_i('verticesOnCell', verticesOnCell)
      call read_i('nEdgesOnCell', nEdgesOnCell)
      call read_r('xPoint', xp); call read_r('yPoint', yp); call read_r('zPoint', zp)
      call read_i('searchVertex', search)
      call read_r('sphere_radius', radius)
      sphere_radius = radius(1)
      on_a_sphere = .true.
      vertexDegree = 3
      call mpas_calculate_barycentric_weights_for_points(pool, xp, yp, zp, search, cells, weights, ierr)
      call write_i('baryCells', cells, 3 * nPoints)
      call write_r('baryWeights', weights, 3 * nPoints)
      print *, 'bary ierr', ierr, 'points', nPoints
   end subroutine do_bary
#else
   subroutine do_apply()
      integer :: d(3), nPoints, nLev, k
      integer, allocatable :: cells(:,:)
      real(kind=RKIND), allocatable :: weights(:,:), field(:,:), out(:,:)
      call read_dims(d, 3)
      nCells = d(1); nPoints = d(2); nLev = d(3)
      allocate(cells(3, nPoints), weights(3, nPoints), field(nCells, nLev), out(nPoints, nLev))
      call read_i('baryCells', cells); call read_r('baryWeights', weights)
      call read_r('field', field)
      vertexDegree = 3
      do k = 1, nLev
         call mpas_cells_to_points_using_baryweights(pool, cells, weights, field(:, k), out(:, k), ierr)
      end do
      call write_r('fieldPoints', out, nPoints * nLev)
      print *, 'apply ierr', ierr
   end subroutine do_apply

   subroutine do_relax()
      integer :: d(6), nEdges, maxEdges, nLev, k, depth
      real(kind=RKIND) :: scalars(2), dt
      integer, allocatable :: activeCell(:), activeEdge(:)
      real(kind=RKIND), allocatable :: all_ru(:,:), all_theta(:,:), all_rho(:,:)
      real(kind=RKIND), allocatable :: all_tu(:,:), all_tt(:,:), all_tr(:,:)
      real(kind=RKIND), allocatable :: all_dru(:,:), all_drt(:,:), all_drho(:,:)
      real(kind=RKIND), allocatable :: dru(:,:), drt(:,:), drho(:,:)
      call read_dims(d, 6)
      nCells = d(1); nEdges = d(2); nVertices = d(3); maxEdges = d(4)
      vertexDegree = d(5); nLev = d(6)
      allocate(dcEdge(nEdges), dvEdge(nEdges), invDcEdge(nEdges), invDvEdge(nEdges))
      allocate(invAreaCell(nCells), invAreaTriangle(nVertices))
      allocate(meshScalingRegionalCell(nCells), meshScalingRegionalEdge(nEdges))
      allocate(edgesOnCell_sign(maxEdges, nCells), edgesOnVertex_sign(vertexDegree, nVertices))
      allocate(cellsOnEdge(2, nEdges), verticesOnEdge(2, nEdges), edgesOnCell(maxEdges, nCells))
      allocate(edgesOnVertex(vertexDegree, nVertices), nEdgesOnCell(nCells))
      allocate(bdyMaskCell(nCells), bdyMaskEdge(nEdges), activeCell(nCells), activeEdge(nEdges))
      allocate(all_ru(nLev, nEdges), all_theta(nLev, nCells), all_rho(nLev, nCells))
      allocate(all_tu(nLev, nEdges), all_tt(nLev, nCells), all_tr(nLev, nCells))
      allocate(all_dru(nLev, nEdges), all_drt(nLev, nCells), all_drho(nLev, nCells))
      allocate(ru(1, nEdges), theta_m(1, nCells), rho_zz(1, nCells))
      allocate(tend_u(1, nEdges), tend_theta_m(1, nCells), tend_rho_zz(1, nCells), tend_w(1, nCells))
      allocate(dru(1, nEdges), drt(1, nCells), drho(1, nCells))
      call read_r('dcEdge', dcEdge); call read_r('dvEdge', dvEdge)
      call read_r('invDcEdge', invDcEdge); call read_r('invDvEdge', invDvEdge)
      call read_r('invAreaCell', invAreaCell); call read_r('invAreaTriangle', invAreaTriangle)
      call read_r('meshScalingRegionalCell', meshScalingRegionalCell)
      call read_r('meshScalingRegionalEdge', meshScalingRegionalEdge)
      call read_r('edgesOnCell_sign', edgesOnCell_sign)
      call read_r('edgesOnVertex_sign', edgesOnVertex_sign)
      call read_i('cellsOnEdge', cellsOnEdge); call read_i('verticesOnEdge', verticesOnEdge)
      call read_i('edgesOnCell', edgesOnCell); call read_i('edgesOnVertex', edgesOnVertex)
      call read_i('nEdgesOnCell', nEdgesOnCell)
      call read_i('activeCell', activeCell); call read_i('activeEdge', activeEdge)
      call read_r('ru', all_ru); call read_r('theta_m', all_theta); call read_r('rho_zz', all_rho)
      call read_r('tend_ru', all_tu); call read_r('tend_rt', all_tt); call read_r('tend_rho', all_tr)
      call read_r('ru_driving', all_dru); call read_r('rt_driving', all_drt)
      call read_r('rho_driving', all_drho)
      call read_r('scalars', scalars)
      dt = scalars(1)
      config_relax_zone_divdamp_coef = scalars(2)
      tend_w = 0.0_RKIND
      ! Level k (1-based) is depth = nLev - k below the top; its ring
      ! strength is 5 - depth.
      do k = 1, nLev
         depth = nLev - k
         where (activeCell /= 0)
            bdyMaskCell = 5 - depth
         elsewhere
            bdyMaskCell = 0
         end where
         where (activeEdge /= 0)
            bdyMaskEdge = 5 - depth
         elsewhere
            bdyMaskEdge = 0
         end where
         ru(1, :) = all_ru(k, :); theta_m(1, :) = all_theta(k, :); rho_zz(1, :) = all_rho(k, :)
         tend_u(1, :) = all_tu(k, :); tend_theta_m(1, :) = all_tt(k, :)
         tend_rho_zz(1, :) = all_tr(k, :)
         dru(1, :) = all_dru(k, :); drt(1, :) = all_drt(k, :); drho(1, :) = all_drho(k, :)
         call atm_bdy_adjust_dynamics_relaxzone_tend(pool, pool, pool, pool, pool, 1, dt, &
              dru, drt, drho, 1, nCells, 1, nEdges, 1, nCells, 1, nEdges)
         all_tu(k, :) = tend_u(1, :); all_tt(k, :) = tend_theta_m(1, :)
         all_tr(k, :) = tend_rho_zz(1, :)
      end do
      call write_r('tend_ru', all_tu, nLev * nEdges)
      call write_r('tend_rt', all_tt, nLev * nCells)
      call write_r('tend_rho', all_tr, nLev * nCells)
      print *, 'relax levels', nLev
   end subroutine do_relax
#endif
end program run_oracle
