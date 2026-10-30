! Driver for WRF v4.7.1's nest coupling operators as the partial-depth nest
! (Phase D3) uses them: interp_fcn_sint (+ SINT), copy_fcn, copy_fcnm,
! copy_fcni, smoother (sm121, smdsm), and relax_bdytend_core + spec_bdytend
! fed a TOP STRIP transposed onto WRF's south side.
!
! Inputs are generated here, deterministically (smooth fields plus a
! linear-congruential perturbation, at the magnitudes of WRF's coupled
! fields), and written with WRF's outputs through oracle_io in WRF's
! (i, k, j) order: the parity test reruns woof on the same words.  Only
! the DOMAIN part of each array is written; the halos the WRF routines
! dimension are filled with the same analytic field and are never read for
! the points compared (the child sits well inside the parent).
MODULE pn_gen
  IMPLICIT NONE
  INTEGER(8), SAVE :: lcg_state = 12345_8
CONTAINS
  REAL FUNCTION urand()
    ! Park-Miller minimal standard (48271), exact in 64-bit integers.
    lcg_state = MOD(lcg_state * 48271_8, 2147483647_8)
    urand = REAL(DBLE(lcg_state) / 2147483647.D0)
  END FUNCTION urand

  ! A coupled-field-like value: scale * (10 + smooth structure) + noise.
  REAL FUNCTION field_value(i, k, j, scale, noise)
    INTEGER, INTENT(IN) :: i, k, j
    REAL, INTENT(IN) :: scale, noise
    field_value = scale * (10.0 + 3.0 * SIN(0.37 * REAL(i) + 0.21 * REAL(k)) &
                  * COS(0.29 * REAL(j)) + 0.5 * SIN(1.3 * REAL(i + j))) &
                  + noise * scale * (urand() - 0.5)
  END FUNCTION field_value
END MODULE pn_gen

PROGRAM run_partial_nest
  USE oracle_io
  USE pn_gen
  USE module_configure
  USE wrf_bc
  IMPLICIT NONE
  CHARACTER(LEN=1024) :: root
  INTEGER, PARAMETER :: npx = 24, npy = 20, nz = 4, nri = 3, nrj = 3
  INTEGER, PARAMETER :: ipos = 7, jpos = 6
  INTEGER, PARAMETER :: ncx = 24, ncy = 18           ! child mass extents
  INTEGER, PARAMETER :: halo = 4
  INTEGER :: stag

  CALL oracle_root(root)
  ! sint: mass, x- and y-staggered.
  DO stag = 0, 2
    CALL sint_case(stag)
  END DO
  ! copy_fcn: mass, u, v; copy_fcnm, copy_fcni (mass).
  DO stag = 0, 2
    CALL copy_case(stag)
  END DO
  CALL copym_case()
  CALL copyi_case()
  ! smoother: sm121 and smdsm, mass and both staggerings.
  DO stag = 0, 2
    CALL smooth_case(1, stag)
    CALL smooth_case(2, stag)
  END DO
  ! the top strip
  CALL strip_case('strip_full_phi', 0, 12, 1.0E7, 8.0, 4.0)
  CALL strip_case('strip_half_u', 1, 11, 7.0E5, 8.0, 4.0)
  CALL strip_case('strip_half_theta', 1, 11, 2.8E6, 2.6666667, 4.0)
  CALL oracle_close()

CONTAINS

  SUBROUTINE extents(stag, xs, ys)
    INTEGER, INTENT(IN) :: stag
    INTEGER, INTENT(OUT) :: xs, ys
    xs = 0; ys = 0
    IF (stag == 1) xs = 1
    IF (stag == 2) ys = 1
  END SUBROUTINE extents

  CHARACTER(LEN=1) FUNCTION stagname(stag)
    INTEGER, INTENT(IN) :: stag
    stagname = 'm'
    IF (stag == 1) stagname = 'x'
    IF (stag == 2) stagname = 'y'
  END FUNCTION stagname

  SUBROUTINE sint_case(stag)
    INTEGER, INTENT(IN) :: stag
    INTEGER :: xs, ys, i, j, k, ex, ey, nex, ney
    REAL, ALLOCATABLE :: cfld(:,:,:), nfld(:,:,:)
    INTEGER, ALLOCATABLE :: imask(:,:)
    LOGICAL :: xstag, ystag
    CALL extents(stag, xs, ys)
    xstag = xs == 1; ystag = ys == 1
    ex = npx + xs; ey = npy + ys            ! parent extent of this stagger
    nex = ncx + xs; ney = ncy + ys          ! child extent
    ALLOCATE(cfld(1-halo:npx+1+halo, 1:nz, 1-halo:npy+1+halo))
    ALLOCATE(nfld(1-halo:ncx+1+halo, 1:nz, 1-halo:ncy+1+halo))
    ALLOCATE(imask(1-halo:ncx+1+halo, 1-halo:ncy+1+halo))
    DO j = LBOUND(cfld, 3), UBOUND(cfld, 3)
      DO k = 1, nz
        DO i = LBOUND(cfld, 1), UBOUND(cfld, 1)
          cfld(i, k, j) = field_value(i, k, j, 7.0E4, 0.6)
        END DO
      END DO
    END DO
    nfld = -999.0
    imask = 1
    CALL interp_fcn_sint(cfld, &
         1, npx+1, 1, nz+1, 1, npy+1, &
         LBOUND(cfld,1), UBOUND(cfld,1), 1, nz, LBOUND(cfld,3), UBOUND(cfld,3), &
         1, ex, 1, nz, 1, ey, &
         nfld, &
         1, ncx+1, 1, nz+1, 1, ncy+1, &
         LBOUND(nfld,1), UBOUND(nfld,1), 1, nz, LBOUND(nfld,3), UBOUND(nfld,3), &
         1, nex, 1, nz, 1, ney, &
         2, imask, xstag, ystag, ipos, jpos, nri, nrj)
    CALL oracle_open('sint_' // stagname(stag))
    CALL oracle_put('cfld', cfld(1:ex, 1:nz, 1:ey))
    CALL oracle_put('nfld', nfld(1:nex, 1:nz, 1:ney))
    DEALLOCATE(cfld, nfld, imask)
  END SUBROUTINE sint_case

  SUBROUTINE copy_case(stag)
    INTEGER, INTENT(IN) :: stag
    INTEGER :: xs, ys, i, j, k, ex, ey, nex, ney
    REAL, ALLOCATABLE :: cfld(:,:,:), nfld(:,:,:), cin(:,:,:)
    INTEGER, ALLOCATABLE :: imask(:,:)
    CALL extents(stag, xs, ys)
    ex = npx + xs; ey = npy + ys
    nex = ncx + xs; ney = ncy + ys
    ALLOCATE(cfld(1-halo:npx+1+halo, 1:nz, 1-halo:npy+1+halo))
    ALLOCATE(nfld(1-halo:ncx+1+halo, 1:nz, 1-halo:ncy+1+halo))
    ALLOCATE(imask(1-halo:ncx+1+halo, 1-halo:ncy+1+halo))
    DO j = LBOUND(cfld, 3), UBOUND(cfld, 3)
      DO k = 1, nz
        DO i = LBOUND(cfld, 1), UBOUND(cfld, 1)
          cfld(i, k, j) = field_value(i, k, j, 2.0, 0.1)
        END DO
      END DO
    END DO
    DO j = LBOUND(nfld, 3), UBOUND(nfld, 3)
      DO k = 1, nz
        DO i = LBOUND(nfld, 1), UBOUND(nfld, 1)
          nfld(i, k, j) = field_value(i, k, j, 1.7, 0.8)
        END DO
      END DO
    END DO
    imask = 1
    stub_spec_zone = 1
    ALLOCATE(cin, SOURCE=cfld)
    CALL copy_fcn(cfld, &
         1, npx+1, 1, nz+1, 1, npy+1, &
         LBOUND(cfld,1), UBOUND(cfld,1), 1, nz, LBOUND(cfld,3), UBOUND(cfld,3), &
         1, ex, 1, nz, 1, ey, &
         nfld, &
         1, ncx+1, 1, nz+1, 1, ncy+1, &
         LBOUND(nfld,1), UBOUND(nfld,1), 1, nz, LBOUND(nfld,3), UBOUND(nfld,3), &
         1, nex, 1, nz, 1, ney, &
         0, imask, xs == 1, ys == 1, ipos, jpos, nri, nrj)
    CALL oracle_open('copy_fcn_' // stagname(stag))
    CALL oracle_put('cfld_in', cin(1:ex, 1:nz, 1:ey))
    CALL oracle_put('nfld', nfld(1:nex, 1:nz, 1:ney))
    CALL oracle_put('cfld', cfld(1:ex, 1:nz, 1:ey))
    DEALLOCATE(cfld, nfld, cin, imask)
  END SUBROUTINE copy_case

  SUBROUTINE copym_case()
    INTEGER :: i, j, k
    REAL, ALLOCATABLE :: cfld(:,:,:), nfld(:,:,:), cin(:,:,:)
    INTEGER, ALLOCATABLE :: imask(:,:)
    ALLOCATE(cfld(1-halo:npx+1+halo, 1:nz, 1-halo:npy+1+halo))
    ALLOCATE(nfld(1-halo:ncx+1+halo, 1:nz, 1-halo:ncy+1+halo))
    ALLOCATE(imask(1-halo:ncx+1+halo, 1-halo:ncy+1+halo))
    DO j = LBOUND(cfld, 3), UBOUND(cfld, 3)
      DO k = 1, nz
        DO i = LBOUND(cfld, 1), UBOUND(cfld, 1)
          cfld(i, k, j) = field_value(i, k, j, 1.0, 0.1)
        END DO
      END DO
    END DO
    DO j = LBOUND(nfld, 3), UBOUND(nfld, 3)
      DO k = 1, nz
        DO i = LBOUND(nfld, 1), UBOUND(nfld, 1)
          nfld(i, k, j) = field_value(i, k, j, 3.0, 0.9)
        END DO
      END DO
    END DO
    imask = 1
    stub_spec_zone = 1
    ALLOCATE(cin, SOURCE=cfld)
    CALL copy_fcnm(cfld, &
         1, npx+1, 1, nz+1, 1, npy+1, &
         LBOUND(cfld,1), UBOUND(cfld,1), 1, nz, LBOUND(cfld,3), UBOUND(cfld,3), &
         1, npx, 1, nz, 1, npy, &
         nfld, &
         1, ncx+1, 1, nz+1, 1, ncy+1, &
         LBOUND(nfld,1), UBOUND(nfld,1), 1, nz, LBOUND(nfld,3), UBOUND(nfld,3), &
         1, ncx, 1, nz, 1, ncy, &
         0, imask, .FALSE., .FALSE., ipos, jpos, nri, nrj)
    CALL oracle_open('copy_fcnm')
    CALL oracle_put('cfld_in', cin(1:npx, 1:nz, 1:npy))
    CALL oracle_put('nfld', nfld(1:ncx, 1:nz, 1:ncy))
    CALL oracle_put('cfld', cfld(1:npx, 1:nz, 1:npy))
    DEALLOCATE(cfld, nfld, cin, imask)
  END SUBROUTINE copym_case

  SUBROUTINE copyi_case()
    INTEGER :: i, j, k
    INTEGER, ALLOCATABLE :: cfld(:,:,:), nfld(:,:,:), cin(:,:,:)
    INTEGER, ALLOCATABLE :: imask(:,:)
    ALLOCATE(cfld(1-halo:npx+1+halo, 1:nz, 1-halo:npy+1+halo))
    ALLOCATE(nfld(1-halo:ncx+1+halo, 1:nz, 1-halo:ncy+1+halo))
    ALLOCATE(imask(1-halo:ncx+1+halo, 1-halo:ncy+1+halo))
    DO j = LBOUND(cfld, 3), UBOUND(cfld, 3)
      DO k = 1, nz
        DO i = LBOUND(cfld, 1), UBOUND(cfld, 1)
          cfld(i, k, j) = -INT(100.0 * urand())
        END DO
      END DO
    END DO
    DO j = LBOUND(nfld, 3), UBOUND(nfld, 3)
      DO k = 1, nz
        DO i = LBOUND(nfld, 1), UBOUND(nfld, 1)
          nfld(i, k, j) = INT(1000.0 * urand())
        END DO
      END DO
    END DO
    imask = 1
    stub_spec_zone = 1
    ALLOCATE(cin, SOURCE=cfld)
    CALL copy_fcni(cfld, &
         1, npx+1, 1, nz+1, 1, npy+1, &
         LBOUND(cfld,1), UBOUND(cfld,1), 1, nz, LBOUND(cfld,3), UBOUND(cfld,3), &
         1, npx, 1, nz, 1, npy, &
         nfld, &
         1, ncx+1, 1, nz+1, 1, ncy+1, &
         LBOUND(nfld,1), UBOUND(nfld,1), 1, nz, LBOUND(nfld,3), UBOUND(nfld,3), &
         1, ncx, 1, nz, 1, ncy, &
         0, imask, .FALSE., .FALSE., ipos, jpos, nri, nrj)
    CALL oracle_open('copy_fcni')
    CALL oracle_put('cfld_in', cin(1:npx, 1:nz, 1:npy))
    CALL oracle_put('nfld', nfld(1:ncx, 1:nz, 1:ncy))
    CALL oracle_put('cfld', cfld(1:npx, 1:nz, 1:npy))
    DEALLOCATE(cfld, nfld, cin, imask)
  END SUBROUTINE copyi_case

  SUBROUTINE smooth_case(option, stag)
    INTEGER, INTENT(IN) :: option, stag
    INTEGER :: xs, ys, i, j, k, ex, ey
    REAL, ALLOCATABLE :: cfld(:,:,:), cin(:,:,:)
    CHARACTER(LEN=32) :: name
    CALL extents(stag, xs, ys)
    ex = npx + xs; ey = npy + ys
    ALLOCATE(cfld(1-halo:npx+1+halo, 1:nz, 1-halo:npy+1+halo))
    DO j = LBOUND(cfld, 3), UBOUND(cfld, 3)
      DO k = 1, nz
        DO i = LBOUND(cfld, 1), UBOUND(cfld, 1)
          cfld(i, k, j) = field_value(i, k, j, 3.0E2, 0.3)
        END DO
      END DO
    END DO
    ALLOCATE(cin, SOURCE=cfld)
    stub_feedback = 1
    stub_spec_zone = 1
    stub_smooth_option = option
    CALL smoother(cfld, &
         1, npx+1, 1, nz+1, 1, npy+1, &
         LBOUND(cfld,1), UBOUND(cfld,1), 1, nz, LBOUND(cfld,3), UBOUND(cfld,3), &
         1, ex, 1, nz, 1, ey, &
         1, ncx+1, 1, nz+1, 1, ncy+1, &
         1-halo, ncx+1+halo, 1, nz, 1-halo, ncy+1+halo, &
         1, ncx + xs, 1, nz, 1, ncy + ys, &
         xs == 1, ys == 1, ipos, jpos, nri, nrj)
    WRITE(name, '(A,I1,A)') 'smoother', option, '_' // stagname(stag)
    CALL oracle_open(TRIM(name))
    CALL oracle_put('cfld_in', cin(1:ex, 1:nz, 1:ey))
    CALL oracle_put('cfld', cfld(1:ex, 1:nz, 1:ey))
    DEALLOCATE(cfld, cin)
  END SUBROUTINE smooth_case

  ! The top strip on WRF's south side.  WRF field(i, k, j): i is our x,
  ! WRF's carried k is our y, WRF's j is strip row d + 1 (row 0 = the
  ! interface).  half = 1 (u, v, theta): row 0 is the parent's level above
  ! the interface, which the child reads as its own target -- field row 1
  ! is top_val row 0 and top_tend row 0 is zero.
  SUBROUTINE strip_case(case, half, nzf, scale, dtbc, dt)
    CHARACTER(LEN=*), INTENT(IN) :: case
    INTEGER, INTENT(IN) :: half, nzf
    REAL, INTENT(IN) :: scale, dtbc, dt
    INTEGER, PARAMETER :: nxf = 9, nyf = 5, width = 7, spec_zone = 1
    INTEGER, PARAMETER :: relax_zone = 6, jj = 20
    REAL :: field(nxf, nyf + 1, jj), tend(nxf, nyf + 1, jj), tend_in(nxf, nyf + 1, jj)
    REAL :: ys(nxf, nyf + 1, width), tys(nxf, nyf + 1, width)
    REAL :: ye(nxf, nyf + 1, width), tye(nxf, nyf + 1, width)
    REAL :: xs(jj, nyf + 1, width), xe(jj, nyf + 1, width)
    REAL :: txs(jj, nyf + 1, width), txe(jj, nyf + 1, width)
    REAL :: fcx(width), gcx(width), f, g, numer, denom
    INTEGER :: i, j, k, loop
    TYPE(grid_config_rec_type) :: config_flags
    config_flags%periodic_x = .TRUE.
    ! WRF's nested lbc_fcx_gcx (module_bc_em.F), default-REAL left to right.
    fcx = 0.0; gcx = 0.0
    DO loop = spec_zone + 1, spec_zone + relax_zone
      IF (loop > width) EXIT
      numer = REAL(spec_zone + relax_zone - loop)
      denom = REAL(relax_zone - 1)
      f = 0.1 / dt
      g = 1.0 / dt
      g = g / 50.0
      f = f * numer
      fcx(loop) = f / denom
      g = g * numer
      gcx(loop) = g / denom
    END DO
    DO j = 1, jj
      DO k = 1, nyf + 1
        DO i = 1, nxf
          field(i, k, j) = field_value(i, j, k, scale, 0.05)
          tend_in(i, k, j) = 1.0E-3 * scale * (urand() - 0.5)
        END DO
      END DO
    END DO
    DO j = 1, width
      DO k = 1, nyf + 1
        DO i = 1, nxf
          ys(i, k, j) = field(i, k, j) * (1.0 + 2.0E-3 * (urand() - 0.5))
          tys(i, k, j) = 1.0E-4 * scale * (urand() - 0.5)
        END DO
      END DO
    END DO
    IF (half == 1) THEN
      tys(:, :, 1) = 0.0
      field(:, :, 1) = ys(:, :, 1)
    END IF
    ye = 0.0; tye = 0.0; xs = 0.0; xe = 0.0; txs = 0.0; txe = 0.0
    tend = tend_in
    CALL relax_bdytend_core(field, tend, xs, xe, ys, ye, txs, txe, tys, tye, &
         't', config_flags, width, spec_zone, relax_zone, dtbc, fcx, gcx, &
         1, nxf + 1, 1, jj + 1, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1)
    CALL spec_bdytend(tend, xs, xe, ys, ye, txs, txe, tys, tye, &
         't', config_flags, width, spec_zone, &
         1, nxf + 1, 1, jj + 1, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1, &
         1, nxf, 1, jj, 1, nyf + 1)
    CALL oracle_open(case)
    CALL oracle_put('half', half)
    CALL oracle_put('nzf', nzf)
    CALL oracle_put('dtbc', dtbc)
    CALL oracle_put('dt', dt)
    CALL oracle_put('fcx', fcx)
    CALL oracle_put('gcx', gcx)
    CALL oracle_put('field', field(:, 1:nyf, 1:width))
    CALL oracle_put('tend_in', tend_in(:, 1:nyf, 1:width))
    CALL oracle_put('top_val', ys(:, 1:nyf, :))
    CALL oracle_put('top_tend', tys(:, 1:nyf, :))
    CALL oracle_put('tend', tend(:, 1:nyf, 1:width))
  END SUBROUTINE strip_case

END PROGRAM run_partial_nest
