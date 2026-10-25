! Driver for WRF v4.7.1's vertical grid nesting (vert_refine_method = 2).
!
! For each case it builds a parent ladder and a nest ladder, takes their
! hybrid coefficients from WRF's own compute_vcoord_1d_coeffs and their
! dn/rdn/cf1..cfn1 from WRF's own vert_cor_vertical_nesting_arbitrary lines
! (wrf_dn_cf.inc), forms the reference log-pressure heights with
! force_domain_em_part2's lines (wrf_alt.inc), fills a small intermediate
! grid (the parent's columns, WRF-coupled) and runs force_domain_em_part2's
! vertical-refinement block (wrf_rebalance.inc): the vertical interpolation
! of every carried field, then the rebalance of t_2, QV and ph_2.
!
! Every array the port needs is written, before and after, by oracle_io in
! WRF's (i, k, j) order.  Nothing here computes a model quantity: the
! fields are synthetic but smooth profiles of the kind a parent carries.
MODULE vert_nest_types
  IMPLICIT NONE
  TYPE ladder_t
    REAL, ALLOCATABLE :: znw(:), znu(:), dnw(:), rdnw(:), dn(:), rdn(:)
    REAL, ALLOCATABLE :: fnp(:), fnm(:)
    REAL, ALLOCATABLE :: c1f(:), c2f(:), c3f(:), c4f(:)
    REAL, ALLOCATABLE :: c1h(:), c2h(:), c3h(:), c4h(:)
    REAL :: cf1, cf2, cf3, cfn, cfn1
    REAL :: p_top, tiso, t00, tlp, p00, p_strat, tlp_strat
    LOGICAL :: this_is_an_ideal_run = .FALSE.
    INTEGER :: vert_refine_method = 2
    INTEGER :: s_vert, e_vert, grid_id = 2
  END TYPE ladder_t
  TYPE grid_t
    REAL, ALLOCATABLE :: t_2(:,:,:), u_2(:,:,:), w_2(:,:,:), ph_2(:,:,:)
    REAL, ALLOCATABLE :: pb(:,:,:), t_init(:,:,:), alb(:,:,:), phb(:,:,:)
    REAL, ALLOCATABLE :: mub(:,:), mu_2(:,:), ht(:,:)
    INTEGER :: hypsometric_opt
    INTEGER :: sm31, sm32, sm33
  END TYPE grid_t
  TYPE config_t
    INTEGER :: use_theta_m = 0
  END TYPE config_t
END MODULE vert_nest_types

MODULE vert_nest_blocks
  USE vert_nest_types
  USE wrf_vert_nest_interp
  USE wrf_vcoord
  IMPLICIT NONE
  REAL, PARAMETER :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  REAL, PARAMETER :: cv = cp-r_d, cvpm = -cv/cp, p1000mb = 100000.
  REAL, PARAMETER :: t0 = 300., rvovrd = r_v/r_d
  ! WRF's 4-D moist array: index 1 is unused, the first scalar is 2.
  INTEGER, PARAMETER :: P_QV = 2, PARAM_FIRST_SCALAR = 2, num_moist = 2
CONTAINS

  ! vert_cor_vertical_nesting_arbitrary: dn..cfn1, then the coefficients.
  SUBROUTINE build_ladder(nest_in, znw, hybrid_opt, etac)
    TYPE(ladder_t), TARGET, INTENT(INOUT) :: nest_in
    REAL, INTENT(IN) :: znw(:), etac
    INTEGER, INTENT(IN) :: hybrid_opt
    TYPE(ladder_t), POINTER :: nest
    INTEGER :: k, kde_n
    REAL :: cof1, cof2
    REAL :: ht(1, 1)
    nest => nest_in
    kde_n = SIZE(znw)
    ALLOCATE(nest%znw(kde_n), nest%znu(kde_n), nest%dnw(kde_n), &
             nest%rdnw(kde_n), nest%dn(kde_n), nest%rdn(kde_n), &
             nest%fnp(kde_n), nest%fnm(kde_n), nest%c1f(kde_n), &
             nest%c2f(kde_n), nest%c3f(kde_n), nest%c4f(kde_n), &
             nest%c1h(kde_n), nest%c2h(kde_n), nest%c3h(kde_n), &
             nest%c4h(kde_n))
    nest%znw = znw
    nest%dnw = 0.; nest%rdnw = 0.; nest%dn = 0.; nest%rdn = 0.
    nest%fnp = 0.; nest%fnm = 0.; nest%c1h = 0.; nest%c2h = 0.
    nest%c3h = 0.; nest%c4h = 0.
    nest%s_vert = 1
    nest%e_vert = kde_n
#include "wrf_dn_cf.inc"
    ht = 0.
    CALL compute_vcoord_1d_coeffs(ht, etac, nest%znw, hybrid_opt, &
         r_d, g, p1000mb, nest%p_top, nest%p00, nest%t00, nest%tlp, &
         1, 2, 1, 2, 1, kde_n, 1, 1, 1, 1, 1, kde_n, &
         1, 1, 1, 1, 1, kde_n, nest%znu, &
         nest%c1f, nest%c2f, nest%c3f, nest%c4f, &
         nest%c1h, nest%c2h, nest%c3h, nest%c4h)
  END SUBROUTINE build_ladder

  ! force_domain_em_part2:73-109, the reference log-pressure heights.
  SUBROUTINE reference_heights(pgrid, ngrid, ckde, nkde, alt_w_c, alt_u_c, &
                               alt_w_n, alt_u_n)
    TYPE(ladder_t), INTENT(IN) :: pgrid, ngrid
    INTEGER, INTENT(IN) :: ckde, nkde
    REAL, INTENT(OUT) :: alt_w_c(ckde), alt_u_c(ckde+1)
    REAL, INTENT(OUT) :: alt_w_n(nkde), alt_u_n(nkde+1)
    REAL :: p_top_m, p_surf_m, mu_m, hsca_m, pre_c, pre_n
    INTEGER :: k
#include "wrf_alt.inc"
  END SUBROUTINE reference_heights

  ! force_domain_em_part2:124-303 on an intermediate grid of the parent's
  ! columns: vertical interpolation, then the rebalance.
  SUBROUTINE vertical_block(grid, ngrid, pgrid, config_flags, moist, &
                            ids, ide, jds, jde, kds, kde, &
                            ims, ime, jms, jme, kms, kme, &
                            ips, ipe, jps, jpe, kps, kpe, &
                            alt_w_c, alt_u_c, alt_w_n, alt_u_n)
    TYPE(grid_t), INTENT(INOUT) :: grid
    TYPE(ladder_t), INTENT(IN) :: ngrid, pgrid
    TYPE(config_t), INTENT(IN) :: config_flags
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, &
                           ims, ime, jms, jme, kms, kme, &
                           ips, ipe, jps, jpe, kps, kpe
    REAL, INTENT(INOUT) :: moist(ims:ime, kms:kme, jms:jme, num_moist)
    REAL, INTENT(IN) :: alt_w_c(pgrid%s_vert:pgrid%e_vert)
    REAL, INTENT(IN) :: alt_u_c(pgrid%s_vert:pgrid%e_vert+1)
    REAL, INTENT(IN) :: alt_w_n(ngrid%s_vert:ngrid%e_vert)
    REAL, INTENT(IN) :: alt_u_n(ngrid%s_vert:ngrid%e_vert+1)
    REAL, DIMENSION(:,:,:), ALLOCATABLE :: p, al
    REAL :: pfu, pfd, phm, temp, qvf, qvf1, qvf2
    INTEGER :: i, j, k, kk, itrace
#include "wrf_rebalance.inc"
  END SUBROUTINE vertical_block
END MODULE vert_nest_blocks

PROGRAM run_vert_nest
  USE oracle_io
  USE vert_nest_types
  USE vert_nest_blocks
  IMPLICIT NONE
  CHARACTER(LEN=1024) :: root
  CALL oracle_root(root)
  CALL run_case('refine_h2', 40, 1.4, 79, 2.2, 2)
  CALL run_case('refine_h1', 40, 1.4, 79, 2.2, 1)
  CALL run_case('coarsen_h2', 60, 1.3, 30, 1.6, 2)
  CALL run_case('same_h2', 40, 1.4, 40, 1.4, 2)
  CALL oracle_close()
CONTAINS

  SUBROUTINE run_case(name, nzp, pp, nzn, pn, hypsometric_opt)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: nzp, nzn, hypsometric_opt
    REAL, INTENT(IN) :: pp, pn
    INTEGER, PARAMETER :: nx = 5, ny = 4
    TYPE(ladder_t) :: pgrid, ngrid
    TYPE(grid_t) :: grid
    TYPE(config_t) :: config_flags
    REAL, ALLOCATABLE :: znw_p(:), znw_n(:), moist(:,:,:,:)
    REAL, ALLOCATABLE :: alt_w_c(:), alt_u_c(:), alt_w_n(:), alt_u_n(:)
    REAL, ALLOCATABLE :: u_in(:,:,:), w_in(:,:,:), t_in(:,:,:)
    REAL, ALLOCATABLE :: qv_in(:,:,:), ph_in(:,:,:)
    INTEGER :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
    INTEGER :: ips, ipe, jps, jpe, kps, kpe, i, j, k, ckde, nkde
    REAL :: x, mut, znu, wave

    ALLOCATE(znw_p(nzp+1), znw_n(nzn+1))
    DO k = 1, nzp+1
      x = REAL(k-1)/REAL(nzp)
      znw_p(k) = 1. - x**pp
    END DO
    DO k = 1, nzn+1
      x = REAL(k-1)/REAL(nzn)
      znw_n(k) = 1. - x**pn
    END DO
    CALL set_base(pgrid)
    CALL set_base(ngrid)
    CALL build_ladder(pgrid, znw_p, 2, 0.2)
    CALL build_ladder(ngrid, znw_n, 2, 0.2)
    ckde = nzp + 1
    nkde = nzn + 1
    ALLOCATE(alt_w_c(ckde), alt_u_c(ckde+1), alt_w_n(nkde), alt_u_n(nkde+1))
    CALL reference_heights(pgrid, ngrid, ckde, nkde, alt_w_c, alt_u_c, &
                           alt_w_n, alt_u_n)

    ! The intermediate grid: the parent's columns, the nest's vertical
    ! extent (sized to hold the parent's levels too), WRF's (i, k, j).
    ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nkde
    ims = ids; ime = ide; jms = jds; jme = jde; kms = 1
    kme = MAX(ckde, nkde) + 1
    ips = ids; ipe = ide; jps = jds; jpe = jde; kps = kds; kpe = kde
    grid%hypsometric_opt = hypsometric_opt
    grid%sm31 = ims; grid%sm32 = kms; grid%sm33 = jms
    ALLOCATE(grid%t_2(ims:ime,kms:kme,jms:jme), &
             grid%u_2(ims:ime,kms:kme,jms:jme), &
             grid%w_2(ims:ime,kms:kme,jms:jme), &
             grid%ph_2(ims:ime,kms:kme,jms:jme), &
             grid%pb(ims:ime,kms:kme,jms:jme), &
             grid%t_init(ims:ime,kms:kme,jms:jme), &
             grid%alb(ims:ime,kms:kme,jms:jme), &
             grid%phb(ims:ime,kms:kme,jms:jme), &
             grid%mub(ims:ime,jms:jme), grid%mu_2(ims:ime,jms:jme), &
             grid%ht(ims:ime,jms:jme), moist(ims:ime,kms:kme,jms:jme,num_moist))
    grid%t_2 = 0.; grid%u_2 = 0.; grid%w_2 = 0.; grid%ph_2 = 0.
    grid%pb = 0.; grid%t_init = 0.; grid%alb = 0.; grid%phb = 0.
    moist = 0.
    DO j = jms, jme
      DO i = ims, ime
        grid%mub(i,j) = 95000. - 300.*REAL(i) - 200.*REAL(j)
        grid%mu_2(i,j) = 150.*SIN(0.7*REAL(i) + 0.3*REAL(j))
        grid%ht(i,j) = 50.*REAL(i) + 80.*REAL(j) + 0.5
        mut = grid%mub(i,j) + grid%mu_2(i,j)
        wave = SIN(1.3*REAL(i) - 0.9*REAL(j))
        ! WRF's coupling (couple_or_uncouple_em): x*(c1*mut + c2).
        DO k = 1, nzp
          znu = pgrid%znu(k)
          grid%t_2(i,k,j) = (290. + 45.*(1.-znu)**1.6 + 0.3*wave - t0) &
                            * (pgrid%c1h(k)*mut + pgrid%c2h(k))
          moist(i,k,j,P_QV) = (0.014*znu**3.5 + 1.e-6 + 2.e-4*wave*znu) &
                              * (pgrid%c1h(k)*mut + pgrid%c2h(k))
          grid%u_2(i,k,j) = (5. + 25.*(1.-znu) + 2.*wave) &
                            * (pgrid%c1h(k)*mut + pgrid%c2h(k))
        END DO
        DO k = 1, nzp+1
          grid%w_2(i,k,j) = 0.05*wave*SIN(3.14159*pgrid%znw(k)) &
                            * (pgrid%c1f(k)*mut + pgrid%c2f(k))
          grid%ph_2(i,k,j) = 10.*wave*(1.-pgrid%znw(k)) &
                             * (pgrid%c1f(k)*mut + pgrid%c2f(k))
        END DO
      END DO
    END DO
    u_in = grid%u_2; w_in = grid%w_2; t_in = grid%t_2
    qv_in = moist(:,:,:,P_QV); ph_in = grid%ph_2

    CALL vertical_block(grid, ngrid, pgrid, config_flags, moist, &
                        ids, ide, jds, jde, kds, kde, &
                        ims, ime, jms, jme, kms, kme, &
                        ips, ipe, jps, jpe, kps, kpe, &
                        alt_w_c, alt_u_c, alt_w_n, alt_u_n)

    CALL oracle_open(name)
    CALL put_ladder('p_', pgrid)
    CALL put_ladder('n_', ngrid)
    CALL oracle_put('alt_w_c', alt_w_c)
    CALL oracle_put('alt_u_c', alt_u_c)
    CALL oracle_put('alt_w_n', alt_w_n)
    CALL oracle_put('alt_u_n', alt_u_n)
    CALL oracle_put('hypsometric_opt', hypsometric_opt)
    CALL oracle_put('use_theta_m', config_flags%use_theta_m)
    CALL oracle_put('mub', grid%mub)
    CALL oracle_put('mu_2', grid%mu_2)
    CALL oracle_put('ht', grid%ht)
    CALL oracle_put('u_in', u_in)
    CALL oracle_put('w_in', w_in)
    CALL oracle_put('t_in', t_in)
    CALL oracle_put('qv_in', qv_in)
    CALL oracle_put('ph_in', ph_in)
    CALL oracle_put('u_out', grid%u_2)
    CALL oracle_put('w_out', grid%w_2)
    CALL oracle_put('t_out', grid%t_2)
    CALL oracle_put('qv_out', moist(:,:,:,P_QV))
    CALL oracle_put('ph_out', grid%ph_2)
    CALL oracle_put('pb_out', grid%pb)
    CALL oracle_put('alb_out', grid%alb)
    CALL oracle_put('phb_out', grid%phb)
  END SUBROUTINE run_case

  SUBROUTINE set_base(lad)
    TYPE(ladder_t), INTENT(INOUT) :: lad
    lad%p_top = 5000.
    lad%tiso = 200.
    lad%t00 = 290.
    lad%tlp = 50.
    lad%p00 = 100000.
    lad%p_strat = 0.
    lad%tlp_strat = -11.
  END SUBROUTINE set_base

  SUBROUTINE put_ladder(prefix, lad)
    CHARACTER(LEN=*), INTENT(IN) :: prefix
    TYPE(ladder_t), INTENT(IN) :: lad
    CALL oracle_put(prefix // 'znw', lad%znw)
    CALL oracle_put(prefix // 'dnw', lad%dnw)
    CALL oracle_put(prefix // 'rdnw', lad%rdnw)
    CALL oracle_put(prefix // 'rdn', lad%rdn)
    CALL oracle_put(prefix // 'c1f', lad%c1f)
    CALL oracle_put(prefix // 'c2f', lad%c2f)
    CALL oracle_put(prefix // 'c3f', lad%c3f)
    CALL oracle_put(prefix // 'c4f', lad%c4f)
    CALL oracle_put(prefix // 'c1h', lad%c1h)
    CALL oracle_put(prefix // 'c2h', lad%c2h)
    CALL oracle_put(prefix // 'c3h', lad%c3h)
    CALL oracle_put(prefix // 'c4h', lad%c4h)
    CALL oracle_put(prefix // 'cf', (/lad%cf1, lad%cf2, lad%cf3, lad%cfn, &
                                      lad%cfn1/))
    CALL oracle_put(prefix // 'base', (/lad%p_top, lad%tiso, lad%t00, &
                                        lad%tlp, lad%p00, lad%p_strat, &
                                        lad%tlp_strat/))
  END SUBROUTINE put_ladder
END PROGRAM run_vert_nest
