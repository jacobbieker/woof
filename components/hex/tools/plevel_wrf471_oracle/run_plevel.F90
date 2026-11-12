! Drives WRF v4.7.1's byte-unmodified phys/module_diag_pld.F (pld) and phys/module_diag_zld.F (zld) on the
! columns of columns_extrapN.bin (written by make_columns.py) and writes the WRF outputs.
!
! One mesh cell is one WRF column.  Each column occupies two i points (identical data) and two j points, so
! WRF's staggered 0.5*(u(i)+u(i+1)) and 0.5*(v(j)+v(j+1)) are the cell value, as the mesh route's reconstructed
! winds are; only the first of the pair is read back.  p (hydrostatic) is unused at use_tot_or_hyd_p = 1.
!
!   run_plevel columns_extrapN.bin out.bin
! out.bin: float32 [8 fields][n_p][nc] (u v t rh ght s td q), then [9][n_z][nc] (u v t rh ght s td q p).
program run_plevel
  use module_diag_pld
  use module_diag_zld
  implicit none
  integer :: nc, nz, extrap, np_, nzl, i, j, k, c, f
  real, allocatable :: plev(:), zlev(:), mass(:,:,:), zg(:,:), pw2(:,:)
  real, allocatable :: u(:,:,:), v(:,:,:), w(:,:,:), t(:,:,:), qv(:,:,:), zp(:,:,:), zb(:,:,:), pp(:,:,:), pb(:,:,:), p(:,:,:), pw(:,:,:)
  real, allocatable :: m2(:,:), ht(:,:)
  real, allocatable :: opl(:,:,:,:), ozl(:,:,:,:)
  real, allocatable :: p_pl(:), z_zl(:)
  real, parameter :: g = 9.81
  character(len=512) :: infile, outfile
  integer :: ime, jme, kme, unit_in, unit_out
  real, allocatable :: pl_u(:,:,:), pl_v(:,:,:), pl_t(:,:,:), pl_rh(:,:,:), pl_ght(:,:,:), pl_s(:,:,:), pl_td(:,:,:), pl_q(:,:,:)
  real, allocatable :: zl_u(:,:,:), zl_v(:,:,:), zl_t(:,:,:), zl_rh(:,:,:), zl_ght(:,:,:), zl_s(:,:,:), zl_td(:,:,:), zl_q(:,:,:), zl_p(:,:,:)
  real, allocatable :: raw(:)
  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)
  open(newunit=unit_in, file=trim(infile), access='stream', form='unformatted', status='old')
  read(unit_in) nc, nz, extrap, np_, nzl
  allocate(plev(np_), zlev(nzl))
  read(unit_in) plev
  read(unit_in) zlev
  allocate(mass(nc,nz,6), zg(nc,nz+1), pw2(nc,nz+1))  ! numpy C order (level, cell): the cell index runs fastest
  do f = 1, 6
    read(unit_in) mass(:,:,f)
  end do
  read(unit_in) zg
  read(unit_in) pw2
  close(unit_in)
  ime = 2*nc + 1; jme = 2; kme = nz + 1
  allocate(u(ime,kme,jme), v(ime,kme,jme), w(ime,kme,jme), t(ime,kme,jme), qv(ime,kme,jme), zp(ime,kme,jme), &
           zb(ime,kme,jme), pp(ime,kme,jme), pb(ime,kme,jme), p(ime,kme,jme), pw(ime,kme,jme))
  allocate(m2(ime,jme), ht(ime,jme))
  u = 0.; v = 0.; w = 0.; t = 0.; qv = 0.; zp = 0.; zb = 0.; pp = 1.; pb = 1.; p = 1.; pw = 1.; m2 = 1.; ht = 0.
  do c = 1, nc
    do j = 1, 2
      do i = 2*c-1, 2*c
        do k = 1, nz
          u(i,k,j) = mass(c,k,1); v(i,k,j) = mass(c,k,2); t(i,k,j) = mass(c,k,3) - 300.
          qv(i,k,j) = mass(c,k,4); pp(i,k,j) = mass(c,k,5); pb(i,k,j) = mass(c,k,6)
          p(i,k,j) = mass(c,k,5) + mass(c,k,6)
        end do
        ! levels above the mass top carry the top level (never read where kte-2 bounds the loops)
        u(i,kme,j) = u(i,nz,j); v(i,kme,j) = v(i,nz,j); t(i,kme,j) = t(i,nz,j); qv(i,kme,j) = qv(i,nz,j)
        pp(i,kme,j) = pp(i,nz,j); pb(i,kme,j) = pb(i,nz,j); p(i,kme,j) = p(i,nz,j)
        do k = 1, nz + 1
          zb(i,k,j) = g * zg(c,k)
          pw(i,k,j) = pw2(c,k)
        end do
        ht(i,j) = zg(c,1)
      end do
    end do
  end do
  ! the column after the last pair supplies u(i+1) of the last pair's second point
  do j = 1, 2
    u(ime,:,j) = u(ime-1,:,j); v(ime,:,j) = v(ime-1,:,j)
  end do
  allocate(p_pl(np_), z_zl(nzl))
  allocate(pl_u(ime,np_,jme), pl_v(ime,np_,jme), pl_t(ime,np_,jme), pl_rh(ime,np_,jme), pl_ght(ime,np_,jme), &
           pl_s(ime,np_,jme), pl_td(ime,np_,jme), pl_q(ime,np_,jme))
  allocate(zl_u(ime,nzl,jme), zl_v(ime,nzl,jme), zl_t(ime,nzl,jme), zl_rh(ime,nzl,jme), zl_ght(ime,nzl,jme), &
           zl_s(ime,nzl,jme), zl_td(ime,nzl,jme), zl_q(ime,nzl,jme), zl_p(ime,nzl,jme))
  call pld(u,v,w,t,qv,zp,zb,pp,pb,p,pw, m2,m2,m2,m2,m2,m2, m2,m2, 1, extrap, -999., np_, np_, plev, &
           p_pl,pl_u,pl_v,pl_t,pl_rh,pl_ght,pl_s,pl_td,pl_q, &
           1,2*nc+1, 1,2, 1,kme, 1,ime, 1,jme, 1,kme, 1,ime, 1,jme, 1,kme)
  call zld(u,v,w,t,qv,zp,zb,pp,pb,p,pw, m2,m2,m2,m2,m2,m2, m2,m2, ht, 1, extrap, -999., nzl, nzl, zlev, &
           z_zl,zl_u,zl_v,zl_t,zl_rh,zl_ght,zl_s,zl_td,zl_q,zl_p, &
           1,2*nc+1, 1,2, 1,kme, 1,ime, 1,jme, 1,kme, 1,ime, 1,jme, 1,kme)
  open(newunit=unit_out, file=trim(outfile), access='stream', form='unformatted', status='replace')
  do f = 1, 8
    do k = 1, np_
      do c = 1, nc
        select case (f)
        case (1); write(unit_out) pl_u(2*c-1,k,1)
        case (2); write(unit_out) pl_v(2*c-1,k,1)
        case (3); write(unit_out) pl_t(2*c-1,k,1)
        case (4); write(unit_out) pl_rh(2*c-1,k,1)
        case (5); write(unit_out) pl_ght(2*c-1,k,1)
        case (6); write(unit_out) pl_s(2*c-1,k,1)
        case (7); write(unit_out) pl_td(2*c-1,k,1)
        case (8); write(unit_out) pl_q(2*c-1,k,1)
        end select
      end do
    end do
  end do
  do f = 1, 9
    do k = 1, nzl
      do c = 1, nc
        select case (f)
        case (1); write(unit_out) zl_u(2*c-1,k,1)
        case (2); write(unit_out) zl_v(2*c-1,k,1)
        case (3); write(unit_out) zl_t(2*c-1,k,1)
        case (4); write(unit_out) zl_rh(2*c-1,k,1)
        case (5); write(unit_out) zl_ght(2*c-1,k,1)
        case (6); write(unit_out) zl_s(2*c-1,k,1)
        case (7); write(unit_out) zl_td(2*c-1,k,1)
        case (8); write(unit_out) zl_q(2*c-1,k,1)
        case (9); write(unit_out) zl_p(2*c-1,k,1)
        end select
      end do
    end do
  end do
  close(unit_out)
end program run_plevel
