! woof's WRFDA 4.7.1 oracle dumps (docs/dev/WRFDA-3DVAR-PLAN.md, lanes O1, O2, O5, O7).
!
! Included into module da_wrfvar_top (da_main/da_wrfvar_top.f90) of a COPY of
! the pinned WRFDA tree by tools/wrfda_oracle/apply_dump_patch.sh; never into
! the pinned tree itself.  Each dump is a stream file of records
!   name (32 chars), kind ('r' real*8 | 'i' integer*4), rank (int4),
!   shape (rank x int4), values (column-major)
! in the build's own byte order (WRFDA's configure says -fconvert=big-endian),
! read by woof.da.var.oracle.read_dump.  Arrays are written over the tile
! (its:ite, jts:jte, kts:kte) unless the name says otherwise.

   subroutine woof_dump_open(u, fname)
      integer, intent(out) :: u
      character(len=*), intent(in) :: fname
      open(newunit=u, file=fname, form='unformatted', access='stream', status='replace')
      write(u) 'WOOFDUMP'
   end subroutine woof_dump_open

   subroutine woof_w0(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real, intent(in) :: a
      character(len=32) :: nm
      nm = name
      write(u) nm, 'r', 0, a
   end subroutine woof_w0

   subroutine woof_wi0(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      integer, intent(in) :: a
      character(len=32) :: nm
      nm = name
      write(u) nm, 'i', 0, a
   end subroutine woof_wi0

   subroutine woof_w1(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real, intent(in) :: a(:)
      character(len=32) :: nm
      nm = name
      write(u) nm, 'r', 1, shape(a), a
   end subroutine woof_w1

   subroutine woof_wi1(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      integer, intent(in) :: a(:)
      character(len=32) :: nm
      nm = name
      write(u) nm, 'i', 1, shape(a), a
   end subroutine woof_wi1

   subroutine woof_w2(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real, intent(in) :: a(:,:)
      character(len=32) :: nm
      nm = name
      write(u) nm, 'r', 2, shape(a), a
   end subroutine woof_w2

   subroutine woof_wi2(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      integer, intent(in) :: a(:,:)
      character(len=32) :: nm
      nm = name
      write(u) nm, 'i', 2, shape(a), a
   end subroutine woof_wi2

   subroutine woof_w3(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real, intent(in) :: a(:,:,:)
      character(len=32) :: nm
      nm = name
      write(u) nm, 'r', 3, shape(a), a
   end subroutine woof_w3

   subroutine woof_dump_be(be)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      type (be_type), intent(in) :: be
      integer :: u
      call woof_dump_open(u, 'woof_be.bin')
      call woof_wi0(u, 'cv%size', be%cv%size)
      call woof_wi0(u, 'cv%size_jb', be%cv%size_jb)
      call woof_wi1(u, 'cv_mz', be%cv_mz)
      call woof_dump_be_sub(u, 'v1', be%v1)
      call woof_dump_be_sub(u, 'v2', be%v2)
      call woof_dump_be_sub(u, 'v3', be%v3)
      call woof_dump_be_sub(u, 'v4', be%v4)
      call woof_dump_be_sub(u, 'v5', be%v5)
      if (cv_options /= 7) then
         call woof_w2(u, 'reg_psi_chi', be%reg_psi_chi)
         call woof_w3(u, 'reg_psi_t', be%reg_psi_t)
         call woof_w2(u, 'reg_psi_ps', be%reg_psi_ps)
      end if
      close(u)
   end subroutine woof_dump_be

   subroutine woof_dump_be_sub(u, tag, s)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      integer, intent(in) :: u
      character(len=*), intent(in) :: tag
      type (be_subtype), intent(in) :: s
      call woof_wi0(u, tag//'%mz', s%mz)
      if (s%mz > 0) then
         call woof_w2(u, tag//'%val', s%val)
         call woof_w3(u, tag//'%evec', s%evec)
         call woof_w1(u, tag//'%rf_alpha', s%rf_alpha)
         call woof_w1(u, tag//'%val_g', s%val_g)
         call woof_w2(u, tag//'%evec_g', s%evec_g)
      end if
   end subroutine woof_dump_be_sub

   subroutine woof_dump_xb(grid)
      type (domain), intent(in) :: grid
      integer :: u
      call woof_dump_open(u, 'woof_xb.bin')
      call woof_wi1(u, 'tile', (/its, ite, jts, jte, kts, kte, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme/))
      call woof_w0(u, 'ds', grid%xb%ds)
      call woof_w0(u, 'ptop', grid%xb%ptop)
      call woof_w0(u, 'ztop', grid%xb%ztop)
      call woof_w2(u, 'lat', grid%xb%lat(its:ite,jts:jte))
      call woof_w2(u, 'lon', grid%xb%lon(its:ite,jts:jte))
      call woof_w2(u, 'map_factor', grid%xb%map_factor(its:ite,jts:jte))
      call woof_w2(u, 'cori', grid%xb%cori(its:ite,jts:jte))
      call woof_w2(u, 'coefx', grid%xb%coefx(its:ite,jts:jte))
      call woof_w2(u, 'coefy', grid%xb%coefy(its:ite,jts:jte))
      call woof_w2(u, 'grid_box_area', grid%xb%grid_box_area(its:ite,jts:jte))
      call woof_w2(u, 'terr', grid%xb%terr(its:ite,jts:jte))
      call woof_w2(u, 'psac', grid%xb%psac(its:ite,jts:jte))
      call woof_w2(u, 'psfc', grid%xb%psfc(its:ite,jts:jte))
      call woof_w2(u, 'ztd', grid%xb%ztd(its:ite,jts:jte))
      call woof_w2(u, 'tpw', grid%xb%tpw(its:ite,jts:jte))
      call woof_w2(u, 'slp', grid%xb%slp(its:ite,jts:jte))
      call woof_w2(u, 'speed', grid%xb%speed(its:ite,jts:jte))
      call woof_w2(u, 'rough', grid%xb%rough(its:ite,jts:jte))
      call woof_w2(u, 'xland', grid%xb%xland(its:ite,jts:jte))
      call woof_w2(u, 'landmask', grid%xb%landmask(its:ite,jts:jte))
      call woof_w2(u, 'lanu', grid%xb%lanu(its:ite,jts:jte))
      call woof_w2(u, 'snow', grid%xb%snow(its:ite,jts:jte))
      call woof_w2(u, 'tsk', grid%xb%tsk(its:ite,jts:jte))
      call woof_w2(u, 'tgrn', grid%xb%tgrn(its:ite,jts:jte))
      call woof_w2(u, 'u10', grid%xb%u10(its:ite,jts:jte))
      call woof_w2(u, 'v10', grid%xb%v10(its:ite,jts:jte))
      call woof_w2(u, 't2', grid%xb%t2(its:ite,jts:jte))
      call woof_w2(u, 'q2', grid%xb%q2(its:ite,jts:jte))
      call woof_w3(u, 'u', grid%xb%u(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'v', grid%xb%v(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'w', grid%xb%w(its:ite,jts:jte,kts:kte+1))
      call woof_w3(u, 't', grid%xb%t(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'p', grid%xb%p(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'q', grid%xb%q(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'h', grid%xb%h(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'hf', grid%xb%hf(its:ite,jts:jte,kts:kte+1))
      call woof_w3(u, 'rho', grid%xb%rho(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'rh', grid%xb%rh(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'es', grid%xb%es(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'qs', grid%xb%qs(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'td', grid%xb%td(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'ref', grid%xb%ref(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'reflog', grid%xb%reflog(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'vertical_inner_product', grid%xb%vertical_inner_product(its:ite,jts:jte,kts:kte))
      call woof_w1(u, 'znu', grid%xb%znu(kts:kte))
      call woof_w1(u, 'znw', grid%xb%znw(kts:kte+1))
      call woof_w1(u, 'dn', grid%xb%dn(kts:kte))
      call woof_w1(u, 'dnw', grid%xb%dnw(kts:kte))
      call woof_w1(u, 'sigmah', grid%xb%sigmah(kts:kte))
      call woof_w1(u, 'sigmaf', grid%xb%sigmaf(kts:kte+1))
      call woof_w1(u, 'c1h', c1h(kts:kte))
      call woof_w1(u, 'c2h', c2h(kts:kte))
      call woof_w1(u, 'c1f', c1f(kts:kte+1))
      call woof_w1(u, 'c2f', c2f(kts:kte+1))
      close(u)
   end subroutine woof_dump_xb

   subroutine woof_dump_xa(u, grid)
      integer, intent(in) :: u
      type (domain), intent(in) :: grid
      call woof_w3(u, 'xa%u', grid%xa%u(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'xa%v', grid%xa%v(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'xa%t', grid%xa%t(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'xa%q', grid%xa%q(its:ite,jts:jte,kts:kte))
      call woof_w2(u, 'xa%psfc', grid%xa%psfc(its:ite,jts:jte))
      call woof_w3(u, 'xa%p', grid%xa%p(its:ite,jts:jte,kts:kte))
      call woof_w3(u, 'xa%rho', grid%xa%rho(its:ite,jts:jte,kts:kte))
      call woof_w2(u, 'xa%tpw', grid%xa%tpw(its:ite,jts:jte))
      call woof_w3(u, 'xa%ref', grid%xa%ref(its:ite,jts:jte,kts:kte))
   end subroutine woof_dump_xa

   subroutine woof_dump_vp(u, tag, vp, be)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      integer, intent(in) :: u
      character(len=*), intent(in) :: tag
      type (vp_type), intent(in) :: vp
      type (be_type), intent(in) :: be
      call woof_w3(u, tag//'%v1', vp%v1(its:ite,jts:jte,kts:kte))
      call woof_w3(u, tag//'%v2', vp%v2(its:ite,jts:jte,kts:kte))
      call woof_w3(u, tag//'%v3', vp%v3(its:ite,jts:jte,kts:kte))
      call woof_w3(u, tag//'%v4', vp%v4(its:ite,jts:jte,kts:kte))
      call woof_w3(u, tag//'%v5', vp%v5(its:ite,jts:jte,kts:kts))
   end subroutine woof_dump_vp

   ! O2: U v and U^T x for one random v and x, stage by stage, as
   ! da_transform_vtox and da_transform_vtox_adj run them (plus
   ! da_transform_xtoxa and its adjoint, which da_transform_vtoy adds).
   subroutine woof_dump_u(grid, cv_size, xbx, be)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      use da_vtox_transforms, only : da_transform_vtovv, da_transform_vtovv_adj, da_vertical_transform, &
         da_transform_vptox, da_transform_vptox_adj, da_transform_xtoxa, da_transform_xtoxa_adj
      type (domain), intent(inout) :: grid
      integer, intent(in) :: cv_size
      type (xbx_type), intent(in) :: xbx
      type (be_type), intent(in) :: be
      real :: cv1(1:cv_size), cv2(1:cv_size)
      integer :: u, seed_size
      integer, allocatable :: seed(:)
      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      seed = 20081205
      call random_seed(put=seed)
      call random_number(cv1)
      cv1 = cv1 - 0.5
      call woof_dump_open(u, 'woof_u.bin')
      call woof_w1(u, 'cv1', cv1)
      call da_zero_x(grid%xa)
      call da_transform_vtovv(grid, cv_size, be, cv1, grid%vv)
      call woof_dump_vp(u, 'vv', grid%vv, be)
      call da_vertical_transform(grid, 'u', be, grid%xb%vertical_inner_product, grid%vv, grid%vp)
      call woof_dump_vp(u, 'vp', grid%vp, be)
      call da_transform_vptox(grid, grid%vp, be, grid%ep)
      call woof_dump_vp(u, 'vp_bal', grid%vp, be)
      call da_transform_xtoxa(grid)
      call woof_dump_xa(u, grid)
      ! The adjoint, from an independent random x.
      call da_zero_x(grid%xa)
      call random_number(grid%xa%u(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%v(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%t(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%q(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%psfc(its:ite,jts:jte))
      call random_number(grid%xa%p(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%rho(its:ite,jts:jte,kts:kte))
      call random_number(grid%xa%tpw(its:ite,jts:jte))
      call random_number(grid%xa%ref(its:ite,jts:jte,kts:kte))
      grid%xa%u = grid%xa%u - 0.5
      grid%xa%v = grid%xa%v - 0.5
      grid%xa%t = grid%xa%t - 0.5
      grid%xa%q = (grid%xa%q - 0.5) * 1.0e-3
      grid%xa%psfc = (grid%xa%psfc - 0.5) * 100.0
      grid%xa%p = (grid%xa%p - 0.5) * 100.0
      grid%xa%rho = (grid%xa%rho - 0.5) * 1.0e-3
      grid%xa%tpw = grid%xa%tpw - 0.5
      grid%xa%ref = grid%xa%ref - 0.5
      call woof_dump_xa(u, grid)
      call da_transform_xtoxa_adj(grid)
      call woof_dump_xa(u, grid)
      call da_zero_vp_type(grid%vp)
      call da_transform_vptox_adj(grid, grid%vp, be, grid%ep)
      call woof_dump_vp(u, 'vp_adj', grid%vp, be)
      call da_zero_vp_type(grid%vv)
      call da_vertical_transform(grid, 'u_adj', be, grid%xb%vertical_inner_product, grid%vv, grid%vp)
      call woof_dump_vp(u, 'vv_adj', grid%vv, be)
      cv2 = 0.0
      call da_transform_vtovv_adj(grid, cv_size, be, cv2, grid%vv)
      call woof_w1(u, 'cv2', cv2)
      close(u)
      call da_zero_x(grid%xa)
      call da_zero_vp_type(grid%vp)
      call da_zero_vp_type(grid%vv)
   end subroutine woof_dump_u

   subroutine woof_dump_info(u, tag, iv, t)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      integer, intent(in) :: u, t
      character(len=*), intent(in) :: tag
      type (iv_type), intent(in) :: iv
      integer :: n, l
      n = iv%info(t)%nlocal
      call woof_wi0(u, tag//'%n', n)
      if (n <= 0) return
      l = iv%info(t)%max_lev
      call woof_wi1(u, tag//'%levels', iv%info(t)%levels(1:n))
      call woof_w2(u, tag//'%lat', iv%info(t)%lat(1:l,1:n))
      call woof_w2(u, tag//'%lon', iv%info(t)%lon(1:l,1:n))
      call woof_w1(u, tag//'%elv', iv%info(t)%elv(1:n))
      call woof_w2(u, tag//'%x', iv%info(t)%x(:,1:n))
      call woof_w2(u, tag//'%y', iv%info(t)%y(:,1:n))
      call woof_wi2(u, tag//'%i', iv%info(t)%i(:,1:n))
      call woof_wi2(u, tag//'%j', iv%info(t)%j(:,1:n))
      call woof_wi2(u, tag//'%k', iv%info(t)%k(1:l,1:n))
      call woof_w2(u, tag//'%dx', iv%info(t)%dx(:,1:n))
      call woof_w2(u, tag//'%dy', iv%info(t)%dy(:,1:n))
      call woof_w2(u, tag//'%dxm', iv%info(t)%dxm(:,1:n))
      call woof_w2(u, tag//'%dym', iv%info(t)%dym(:,1:n))
      call woof_w2(u, tag//'%dz', iv%info(t)%dz(1:l,1:n))
      call woof_w2(u, tag//'%dzm', iv%info(t)%dzm(1:l,1:n))
      call woof_w2(u, tag//'%zk', iv%info(t)%zk(1:l,1:n))
      call woof_wi2(u, tag//'%proc_domain', merge(1, 0, iv%info(t)%proc_domain(1:l,1:n)))
      call woof_wi1(u, tag//'%obs_global_index', iv%info(t)%obs_global_index(1:n))
   end subroutine woof_dump_info

   subroutine woof_dump_field(u, tag, f)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      integer, intent(in) :: u
      character(len=*), intent(in) :: tag
      type (field_type), intent(in) :: f(:)
      call woof_w1(u, tag//'%inv', f%inv)
      call woof_wi1(u, tag//'%qc', f%qc)
      call woof_w1(u, tag//'%error', f%error)
   end subroutine woof_dump_field

   subroutine woof_dump_sfc(u, tag, s)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      integer, intent(in) :: u
      character(len=*), intent(in) :: tag
      type (synop_type), intent(in) :: s(:)
      call woof_w1(u, tag//'%h', s%h)
      call woof_dump_field(u, tag//'%u', s%u)
      call woof_dump_field(u, tag//'%v', s%v)
      call woof_dump_field(u, tag//'%t', s%t)
      call woof_dump_field(u, tag//'%p', s%p)
      call woof_dump_field(u, tag//'%q', s%q)
   end subroutine woof_dump_sfc

   ! O5: the innovation vector after QC, per type, at full precision.
   ! Multi-level types are flattened obs by obs, levels in order
   ! (tag%levels says how many each observation has).
   subroutine woof_dump_iv(iv, fname)
      use da_define_structures, only : be_type, be_subtype, xbx_type, vp_type, iv_type, field_type, &
         synop_type, da_zero_x, da_zero_vp_type
      type (iv_type), intent(in) :: iv
      character(len=*), intent(in) :: fname
      integer :: u, n, k, lv, nn, t, ty
      real, allocatable :: h(:), p(:)
      type (field_type), allocatable :: f1(:), f2(:), f3(:), f4(:)
      call woof_dump_open(u, fname)
      call woof_dump_info(u, 'synop', iv, synop)
      if (iv%info(synop)%nlocal > 0) call woof_dump_sfc(u, 'synop', iv%synop(1:iv%info(synop)%nlocal))
      call woof_dump_info(u, 'metar', iv, metar)
      if (iv%info(metar)%nlocal > 0) call woof_dump_sfc(u, 'metar', iv%metar(1:iv%info(metar)%nlocal))
      call woof_dump_info(u, 'ships', iv, ships)
      if (iv%info(ships)%nlocal > 0) call woof_dump_sfc(u, 'ships', iv%ships(1:iv%info(ships)%nlocal))
      call woof_dump_info(u, 'buoy', iv, buoy)
      if (iv%info(buoy)%nlocal > 0) call woof_dump_sfc(u, 'buoy', iv%buoy(1:iv%info(buoy)%nlocal))
      call woof_dump_info(u, 'sonde_sfc', iv, sonde_sfc)
      if (iv%info(sonde_sfc)%nlocal > 0) call woof_dump_sfc(u, 'sonde_sfc', iv%sonde_sfc(1:iv%info(sonde_sfc)%nlocal))
      call woof_dump_info(u, 'gpspw', iv, gpspw)
      n = iv%info(gpspw)%nlocal
      if (n > 0) call woof_dump_field(u, 'gpspw%tpw', iv%gpspw(1:n)%tpw)
      call woof_dump_info(u, 'qscat', iv, qscat)
      n = iv%info(qscat)%nlocal
      if (n > 0) then
         call woof_w1(u, 'qscat%h', iv%qscat(1:n)%h)
         call woof_dump_field(u, 'qscat%u', iv%qscat(1:n)%u)
         call woof_dump_field(u, 'qscat%v', iv%qscat(1:n)%v)
      end if
      ! multi-level types: sound, airep, pilot, profiler, geoamv, polaramv, satem, gpsref
      do t = 1, 8
         select case (t)
         case (1); ty = sound
         case (2); ty = airep
         case (3); ty = pilot
         case (4); ty = profiler
         case (5); ty = geoamv
         case (6); ty = polaramv
         case (7); ty = satem
         case default; ty = gpsref
         end select
         call woof_dump_info(u, trim(obs_names(ty)), iv, ty)
         n = iv%info(ty)%nlocal
         if (n <= 0) cycle
         nn = sum(iv%info(ty)%levels(1:n))
         allocate(h(nn), p(nn), f1(nn), f2(nn), f3(nn), f4(nn))
         h = 0.0
         p = 0.0
         nn = 0
         do k = 1, n
            do lv = 1, iv%info(ty)%levels(k)
               nn = nn + 1
               select case (t)
               case (1)
                  h(nn) = iv%sound(k)%h(lv); p(nn) = iv%sound(k)%p(lv)
                  f1(nn) = iv%sound(k)%u(lv); f2(nn) = iv%sound(k)%v(lv)
                  f3(nn) = iv%sound(k)%t(lv); f4(nn) = iv%sound(k)%q(lv)
               case (2)
                  h(nn) = iv%airep(k)%h(lv); p(nn) = iv%airep(k)%p(lv)
                  f1(nn) = iv%airep(k)%u(lv); f2(nn) = iv%airep(k)%v(lv)
                  f3(nn) = iv%airep(k)%t(lv); f4(nn) = iv%airep(k)%q(lv)
               case (3)
                  h(nn) = iv%pilot(k)%h(lv); p(nn) = iv%pilot(k)%p(lv)
                  f1(nn) = iv%pilot(k)%u(lv); f2(nn) = iv%pilot(k)%v(lv)
               case (4)
                  h(nn) = iv%profiler(k)%h(lv); p(nn) = iv%profiler(k)%p(lv)
                  f1(nn) = iv%profiler(k)%u(lv); f2(nn) = iv%profiler(k)%v(lv)
               case (5)
                  p(nn) = iv%geoamv(k)%p(lv)
                  f1(nn) = iv%geoamv(k)%u(lv); f2(nn) = iv%geoamv(k)%v(lv)
               case (6)
                  p(nn) = iv%polaramv(k)%p(lv)
                  f1(nn) = iv%polaramv(k)%u(lv); f2(nn) = iv%polaramv(k)%v(lv)
               case (7)
                  p(nn) = iv%satem(k)%p(lv)
                  f1(nn) = iv%satem(k)%thickness(lv)
               case default
                  h(nn) = iv%gpsref(k)%h(lv)
                  f1(nn) = iv%gpsref(k)%ref(lv); f2(nn) = iv%gpsref(k)%p(lv)
                  f3(nn) = iv%gpsref(k)%t(lv); f4(nn) = iv%gpsref(k)%q(lv)
               end select
            end do
         end do
         ! h only where the type has heights, p only where it is a plain
         ! coordinate (gpsref's p, t, q are innovation fields, below)
         if (t <= 4 .or. t == 8) call woof_w1(u, trim(obs_names(ty))//'%h', h)
         if (t <= 7) call woof_w1(u, trim(obs_names(ty))//'%p', p)
         if (t == 7) then
            call woof_w1(u, 'satem%ref_p', iv%satem(1:n)%ref_p)
            call woof_dump_field(u, 'satem%thickness', f1)
         else if (t == 8) then
            call woof_dump_field(u, 'gpsref%ref', f1)
            call woof_dump_field(u, 'gpsref%p', f2)
            call woof_dump_field(u, 'gpsref%t', f3)
            call woof_dump_field(u, 'gpsref%q', f4)
         else
            call woof_dump_field(u, trim(obs_names(ty))//'%u', f1)
            call woof_dump_field(u, trim(obs_names(ty))//'%v', f2)
            if (t <= 2) then
               call woof_dump_field(u, trim(obs_names(ty))//'%t', f3)
               call woof_dump_field(u, trim(obs_names(ty))//'%q', f4)
            end if
         end if
         deallocate(h, p, f1, f2, f3, f4)
      end do
      close(u)
   end subroutine woof_dump_iv

   ! O7: the converged control vector and the analysis increment.
   subroutine woof_dump_final(grid, cv_size, xhat)
      type (domain), intent(in) :: grid
      integer, intent(in) :: cv_size
      real, intent(in) :: xhat(1:cv_size)
      integer :: u
      call woof_dump_open(u, 'woof_final.bin')
      call woof_w1(u, 'xhat', xhat)
      call woof_dump_xa(u, grid)
      close(u)
   end subroutine woof_dump_final
