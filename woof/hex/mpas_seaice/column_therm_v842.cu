// MPAS-Seaice v8.4.2: the column package's vertical thermodynamics
// (column_vertical_thermodynamics -> colpkg_step_therm1) with the Bitz and
// Lipscomb (1999) thermodynamics.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     1064-1666  column_vertical_thermodynamics
//     3286-3307  get_day_of_year (the host passes the day)
//   src/core_seaice/column/ice_colpkg.F90:
//     1719-2281  colpkg_step_therm1
//     3414-3513  colpkg_atm_boundary
//   src/core_seaice/column/ice_atmo.F90:
//     46-366     atmo_boundary_layer
//   src/core_seaice/column/ice_therm_vertical.F90:
//     52-450     thermo_vertical
//     461-620    frzmlt_bottom_lateral
//     631-996    init_vertical_profile
//     1006-1601  thickness_changes
//     1612-1709  freeboard
//     1719-1790  adjust_enthalpy
//     1801-1918  conservation_check_vthermo
//     1930-1989  update_state_vthermo
//   src/core_seaice/column/ice_therm_bl99.F90:
//     53-811     temperature_changes
//     823-914    conductivity
//     994-1228   get_matrix_elements_calc_Tsfc
//     1242-1447  get_matrix_elements_know_Tsfc
//     1457-1498  tridiag_solver
//   src/core_seaice/column/ice_therm_shared.F90:
//     47-73      calculate_Tin_from_qin
//     81-197     surface_heat_flux, dsurface_heat_flux_dTsf
//   src/core_seaice/column/ice_flux_colpkg.F90:
//     30-155     merge_fluxes
//     169-283    set_sfcflux
//   src/core_seaice/column/ice_age.F90:23-34 increment_age
//   src/core_seaice/column/ice_firstyear.F90:36-62 update_FYarea
// (Elizabeth C. Hunke, William H. Lipscomb, C. M. Bitz, Adrian K. Turner,
// Andrew Roberts, David Schroeder, Nicole Jeffery and the CICE Consortium).
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// Appended to column_v842.cu (its rounding macros, constants and layout).
// The MPAS build has no CCSMCOUPLED, so temperature_changes absorbs with
// frac = 0.9 and dTemp = 0.02. x**4 and x**3 are libgcc's __powidf2, the call
// gfortran -O0 makes ((x*x)*(x*x) and x*(x*x)); deltaT**m2 is pow (glibc_pow),
// exp and log glibc_exp and glibc_log. atmo_boundary_layer's psimhu calls
// glibc's binary64 atan, which is not correctly rounded and is LGPL: this
// unit compiles only with the opt-in transcription prepended
// (WOOF_GLIBC239_TRIG64, g239::g239_atan; column_therm_v842.py refuses
// otherwise). Constant expressions gfortran folds (log(zref/zTrf),
// vonkar/log(zref/iceruf)) arrive as arguments, folded on the host.
//
// One thread is one owned cell; COL_NILYR (prepended) is nIceLayers and the
// one snow layer is nSnowLayers (column_v842 refuses more).

#ifdef WOOF_GLIBC239_TRIG64

#define COL_NSLYR 1
#define COL_NMAT (COL_NILYR + COL_NSLYR + 1)

// ice_constants_colpkg (cice), ice_therm_shared and ice_therm_bl99
#define LVAP 2.501e6
#define LSUB 2.835e6
#define KSNO 0.30
#define KICE 2.03
#define HS_MIN 1.0e-4
#define STEFAN_BOLTZMANN 567.0e-10
#define EMISSIVITY 0.95
#define QQQICE 11637800.0
#define TTTICE 5897.8
#define CP_AIR 1005.0
#define CP_WV 1.81e3
#define ZVIR 0.606
#define VONKAR 0.4
#define GRAVIT 9.80616
#define ICE_REF_SALINITY 4.0
#define SECDAY 86400.0
#define PIH (0.5 * PI)
#define FERRMAX 1.0e-3
#define TMIN (-100.0)
#define BETAK 0.13
#define KIMIN 0.10

// abort codes (the stop_label colpkg_step_therm1 hands back)
#define VT_OK 0
#define VT_TSN_HIGH 1      // init_vertical_profile: Starting thermo, zTsn > Tmax
#define VT_TSN_LOW 2       // init_vertical_profile: Starting thermo, zTsn < Tmin
#define VT_ZSIN_LOW 3      // init_vertical_profile: Starting zSin < min_salin, layer
#define VT_TIN_HIGH 4      // init_vertical_profile: Starting thermo, T > Tmax, layer
#define VT_TIN_LOW 5       // init_vertical_profile: Starting thermo, T < Tmin, layer
#define VT_NO_CONVERGE 6   // temperature_changes: Thermo iteration does not converge
#define VT_CONSERVATION 7  // conservation_check_vthermo: Thermo energy conservation error

__device__ __forceinline__ double powi4(double x) { double t = M(x, x); return M(t, t); }
__device__ __forceinline__ double powi3(double x) { return M(x, M(x, x)); }
// gfortran SIGN(a, b) for reals: |a| with b's sign bit
__device__ __forceinline__ double f_sign(double a, double b) { return copysign(fabs(a), b); }

// The colpkg parameters column_vertical_thermodynamics runs under.
struct vt_params {
    double dt, yday, ustar_min, rdn_ice, al2;
    int ncat, l_brine, calc_tsfc, calc_strair, highfreq, natmiter, conduct_mu71, tr_iage, tr_fy;
};

// ice_therm_shared: calculate_Tin_from_qin
__device__ double calculate_Tin_from_qin(double qin, double Tmltk, int l_brine)
{
    if (l_brine) {
        double aa1 = CP_ICE;
        double bb1 = S(S(M(S(CP_OCN, CP_ICE), Tmltk), D(qin, RHOI)), lfresh());
        double cc1 = M(lfresh(), Tmltk);
        return col_min(D(S(-bb1, SQRT(S(M(bb1, bb1), M(M(4.0, aa1), cc1)))), M(2.0, aa1)), Tmltk);
    }
    return D(A(lfresh(), D(qin, RHOI)), CP_ICE);
}

// ice_therm_shared: surface_heat_flux
__device__ void surface_heat_flux(double Tsf, double fswsfc, double rhoa, double flw, double potT, double Qa,
                                  double shcoef, double lhcoef, double *flwoutn, double *fsensn, double *flatn,
                                  double *fsurfn)
{
    double TsfK = A(Tsf, TFFRESH);
    double tmpvar = D(1.0, TsfK);
    double qsat = M(QQQICE, glibc_exp(-M(TTTICE, tmpvar)));
    double Qsfc = D(qsat, rhoa);
    double flwdabs = M(EMISSIVITY, flw);
    *flwoutn = -M(M(EMISSIVITY, STEFAN_BOLTZMANN), powi4(TsfK));
    *fsensn = M(shcoef, S(potT, TsfK));
    *flatn = M(lhcoef, S(Qa, Qsfc));
    *fsurfn = A(A(A(A(fswsfc, flwdabs), *flwoutn), *fsensn), *flatn);
}

// ice_therm_shared: dsurface_heat_flux_dTsf
__device__ void dsurface_heat_flux_dTsf(double Tsf, double rhoa, double shcoef, double lhcoef,
                                        double *dfsurfn_dTsf, double *dflwoutn_dTsf, double *dfsensn_dTsf,
                                        double *dflatn_dTsf)
{
    double TsfK = A(Tsf, TFFRESH);
    double tmpvar = D(1.0, TsfK);
    double qsat = M(QQQICE, glibc_exp(-M(TTTICE, tmpvar)));
    double dQsfc_dTsf = M(M(M(TTTICE, tmpvar), tmpvar), D(qsat, rhoa));
    *dflwoutn_dTsf = -M(M(M(EMISSIVITY, STEFAN_BOLTZMANN), 4.0), powi3(TsfK));
    *dfsensn_dTsf = -shcoef;
    *dflatn_dTsf = -M(lhcoef, dQsfc_dTsf);
    *dfsurfn_dTsf = A(A(*dflwoutn_dTsf, *dfsensn_dTsf), *dflatn_dTsf);
}

// ice_atmo: psimhu and psixhu (statement functions)
__device__ __forceinline__ double psimhu(double xd)
{
    return A(S(glibc_log(D(M(A(1.0, M(xd, A(2.0, xd))), A(1.0, M(xd, xd))), 8.0)),
               M(2.0, g239::g239_atan(xd))), PIH);
}
__device__ __forceinline__ double psixhu(double xd)
{
    return M(2.0, glibc_log(D(A(1.0, M(xd, xd)), 2.0)));
}

// ice_atmo: atmo_boundary_layer for sfctype 'ice' without form drag (refused),
// as colpkg_atm_boundary calls it (uvel = uVelocity, vvel = 0: its second
// "if (present(uvel)) worku = uvel" never sets workv).
__device__ void atmo_boundary_layer_ice(const vt_params &p, double Tsf, double potT, double uatm, double vatm,
                                        double wind, double zlvl, double Qa, double rhoa, double *strx,
                                        double *stry, double *Tref, double *Qref, double *delt, double *delq,
                                        double *lhcoef, double *shcoef, double *Cdn_atm,
                                        double *Cdn_atm_ratio_n, double uvel, double vvel, double *Uref)
{
    const double cpvir = S(D(CP_WV, CP_AIR), 1.0);
    const double zTrf = 2.0;
    double al2 = p.al2;
    double umin = p.highfreq ? 0.5 : 1.0;
    *Tref = 0.0; *Qref = 0.0; *Uref = 0.0; *delt = 0.0; *delq = 0.0; *shcoef = 0.0; *lhcoef = 0.0;
    double qqq = QQQICE, TTT = TTTICE, Lheat = LSUB;
    double vmag;
    if (p.highfreq) {
        double du = S(uatm, uvel), dv = S(vatm, vvel);
        vmag = col_max(umin, SQRT(A(M(du, du), M(dv, dv))));
    } else {
        vmag = col_max(umin, wind);
    }
    double rdn = p.rdn_ice;
    *Cdn_atm = M(rdn, rdn);
    double TsfK = A(Tsf, TFFRESH);
    double qsat = M(qqq, glibc_exp(-D(TTT, TsfK)));
    double ssq = D(qsat, rhoa);
    double thva = M(potT, A(1.0, M(ZVIR, Qa)));
    *delt = S(potT, TsfK);
    *delq = S(Qa, ssq);
    double alz = glibc_log(D(zlvl, 10.0));
    double cp = M(CP_AIR, A(1.0, M(cpvir, ssq)));
    double rhn = rdn, ren = rdn;
    double ustar = M(rdn, vmag);
    double tstar = M(rhn, *delt);
    double qstar = M(ren, *delq);
    double hol = 0.0, stable = 0.0, xqq, psimh, psixh = 0.0, psimhs, rd = 0.0, rh = 0.0, re = 0.0;
    for (int k = 1; k <= p.natmiter; ++k) {
        hol = D(M(M(M(VONKAR, GRAVIT), zlvl), A(D(tstar, thva), D(qstar, A(D(1.0, ZVIR), Qa)))),
                M(ustar, ustar));
        hol = f_sign(col_min(fabs(hol), 10.0), hol);
        stable = A(0.5, f_sign(0.5, hol));
        xqq = col_max(SQRT(fabs(S(1.0, M(16.0, hol)))), 1.0);
        xqq = SQRT(xqq);
        psimhs = -A(A(M(0.7, hol), M(M(0.75, S(hol, 14.3)), glibc_exp(-M(0.35, hol)))), 10.7);
        psimh = A(M(psimhs, stable), M(S(1.0, stable), psimhu(xqq)));
        psixh = A(M(psimhs, stable), M(S(1.0, stable), psixhu(xqq)));
        rd = D(rdn, A(1.0, M(D(rdn, VONKAR), S(alz, psimh))));
        rh = D(rhn, A(1.0, M(D(rhn, VONKAR), S(alz, psixh))));
        re = D(ren, A(1.0, M(D(ren, VONKAR), S(alz, psixh))));
        ustar = M(rd, vmag);
        tstar = M(rh, *delt);
        qstar = M(re, *delq);
    }
    if (p.calc_strair) {
        *strx = 0.0;
        *stry = 0.0;
        if (p.highfreq) {
            double tau = M(M(rhoa, rd), rd);
            double du = S(uatm, uvel), dv = S(vatm, vvel);
            *strx = M(M(tau, SQRT(A(M(du, du), M(dv, dv)))), du);
            *stry = M(M(tau, SQRT(A(M(du, du), M(dv, dv)))), dv);
        } else {
            double tau = M(M(rhoa, ustar), rd);
            *strx = M(tau, uatm);
            *stry = M(tau, vatm);
        }
        *Cdn_atm_ratio_n = D(D(M(rd, rd), rdn), rdn);
    }
    *shcoef = A(M(M(M(rhoa, ustar), cp), rh), 1.0);
    *lhcoef = M(M(M(rhoa, ustar), Lheat), re);
    hol = D(M(hol, zTrf), zlvl);
    xqq = col_max(1.0, SQRT(fabs(S(1.0, M(16.0, hol)))));
    xqq = SQRT(xqq);
    double psix2 = A(-M(M(5.0, hol), stable), M(S(1.0, stable), psixhu(xqq)));
    double fac = M(D(rh, VONKAR), A(S(A(alz, al2), psixh), psix2));
    *Tref = S(potT, M(*delt, fac));
    *Tref = S(*Tref, M(P01, zTrf));
    fac = M(D(re, VONKAR), A(S(A(alz, al2), psixh), psix2));
    *Qref = S(Qa, M(*delq, fac));
    if (p.highfreq) {
        double du = S(uatm, uvel), dv = S(vatm, vvel);
        *Uref = D(M(SQRT(A(M(du, du), M(dv, dv))), rd), rdn);
    } else {
        *Uref = D(M(vmag, rd), rdn);
    }
}

// ice_therm_bl99: conductivity (one snow layer)
__device__ void conductivity(const vt_params &p, int l_snow, double hilyr, double hslyr, const double *zTin,
                             const double *zSin, double *kh)
{
    double kslyr[COL_NSLYR], kilyr[COL_NILYR];
    for (int k = 0; k < COL_NSLYR; ++k) kslyr[k] = KSNO;
    if (p.conduct_mu71) {
        for (int k = 0; k < COL_NILYR; ++k) {
            kilyr[k] = A(KICE, D(M(BETAK, zSin[k]), col_min(-PUNY, zTin[k])));
            kilyr[k] = col_max(kilyr[k], KIMIN);
        }
    } else {
        for (int k = 0; k < COL_NILYR; ++k) {
            kilyr[k] = D(M(A(S(2.11, M(0.011, zTin[k])), D(M(0.09, zSin[k]), col_min(-PUNY, zTin[k]))), RHOI),
                         917.0);
            kilyr[k] = col_max(kilyr[k], KIMIN);
        }
    }
    // kh is 1-based in the Fortran: kh[i] here is kh(i+1)
    if (l_snow) {
        kh[0] = D(M(2.0, kslyr[0]), hslyr);
        kh[COL_NSLYR] = D(M(M(2.0, kslyr[COL_NSLYR - 1]), kilyr[0]),
                          A(M(kslyr[COL_NSLYR - 1], hilyr), M(kilyr[0], hslyr)));
    } else {
        kh[0] = 0.0;
        kh[COL_NSLYR] = D(M(2.0, kilyr[0]), hilyr);
    }
    kh[COL_NSLYR + COL_NILYR] = D(M(2.0, kilyr[COL_NILYR - 1]), hilyr);
    for (int k = 2; k <= COL_NSLYR; ++k) {
        kh[k - 1] = l_snow ? D(M(M(2.0, kslyr[k - 2]), kslyr[k - 1]), M(A(kslyr[k - 2], kslyr[k - 1]), hslyr))
                           : 0.0;
    }
    for (int k = 2; k <= COL_NILYR; ++k) {
        kh[k + COL_NSLYR - 1] = D(M(M(2.0, kilyr[k - 2]), kilyr[k - 1]), M(A(kilyr[k - 2], kilyr[k - 1]), hilyr));
    }
}

// ice_therm_bl99: get_matrix_elements_calc_Tsfc (1-based rows as kr-1)
__device__ void matrix_calc_tsfc(int l_snow, int l_cold, double Tsf, double Tbot, double fsurfn,
                                 double dfsurf_dT, const double *Tin_init, const double *Tsn_init,
                                 const double *kh, const double *Sswabs, const double *Iswabs, const double *etai,
                                 const double *etas, double *sbdiag, double *diag, double *spdiag, double *rhs)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    for (int k = 1; k <= nslyr + 1; ++k) {
        sbdiag[k - 1] = 0.0; diag[k - 1] = 1.0; spdiag[k - 1] = 0.0; rhs[k - 1] = 0.0;
    }
    int k, ki, kr;
    if (l_cold) {
        k = l_snow ? 1 : 1 + nslyr;
        kr = k;
        sbdiag[kr - 1] = 0.0;
        diag[kr - 1] = S(dfsurf_dT, kh[k - 1]);
        spdiag[kr - 1] = kh[k - 1];
        rhs[kr - 1] = S(M(dfsurf_dT, Tsf), fsurfn);
    }
    if (l_snow) {
        if (l_cold) {
            sbdiag[1] = -M(etas[0], kh[0]);
            spdiag[1] = -M(etas[0], kh[1]);
            diag[1] = A(1.0, M(etas[0], A(kh[0], kh[1])));
            rhs[1] = A(Tsn_init[0], M(etas[0], Sswabs[0]));
        } else {
            sbdiag[1] = 0.0;
            spdiag[1] = -M(etas[0], kh[1]);
            diag[1] = A(1.0, M(etas[0], A(kh[0], kh[1])));
            rhs[1] = A(A(Tsn_init[0], M(M(etas[0], kh[0]), Tsf)), M(etas[0], Sswabs[0]));
        }
    }
    for (k = 2; k <= nslyr; ++k) {
        kr = k + 1;
        if (l_snow) {
            sbdiag[kr - 1] = -M(etas[k - 1], kh[k - 1]);
            spdiag[kr - 1] = -M(etas[k - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etas[k - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(Tsn_init[k - 1], M(etas[k - 1], Sswabs[k - 1]));
        }
    }
    if (nilyr > 1) {
        ki = 1; k = ki + nslyr; kr = k + 1;
        if (l_snow || l_cold) {
            sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
            spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1]));
        } else {
            sbdiag[kr - 1] = 0.0;
            spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k - 1]), Tsf));
        }
        ki = nilyr; k = ki + nslyr; kr = k + 1;
        sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
        spdiag[kr - 1] = 0.0;
        diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
        rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k]), Tbot));
    } else {
        ki = 1; k = ki + nslyr; kr = k + 1;
        if (l_snow || l_cold) {
            sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
            spdiag[kr - 1] = 0.0;
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k]), Tbot));
        } else {
            sbdiag[kr - 1] = 0.0;
            spdiag[kr - 1] = 0.0;
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k - 1]), Tsf)),
                            M(M(etai[ki - 1], kh[k]), Tbot));
        }
    }
    for (ki = 2; ki <= nilyr - 1; ++ki) {
        k = ki + nslyr; kr = k + 1;
        sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
        spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
        diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
        rhs[kr - 1] = A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1]));
    }
}

// ice_therm_bl99: get_matrix_elements_know_Tsfc
__device__ void matrix_know_tsfc(int l_snow, double Tbot, const double *Tin_init, const double *Tsn_init,
                                 const double *kh, const double *Sswabs, const double *Iswabs, const double *etai,
                                 const double *etas, double *sbdiag, double *diag, double *spdiag, double *rhs,
                                 double fcondtopn)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    for (int k = 1; k <= nslyr + 1; ++k) {
        sbdiag[k - 1] = 0.0; diag[k - 1] = 1.0; spdiag[k - 1] = 0.0; rhs[k - 1] = 0.0;
    }
    int k, ki, kr;
    if (l_snow) {
        sbdiag[1] = 0.0;
        spdiag[1] = -M(etas[0], kh[1]);
        diag[1] = A(1.0, M(etas[0], kh[1]));
        rhs[1] = A(A(Tsn_init[0], M(etas[0], Sswabs[0])), M(etas[0], fcondtopn));
    }
    for (k = 2; k <= nslyr; ++k) {
        kr = k + 1;
        if (l_snow) {
            sbdiag[kr - 1] = -M(etas[k - 1], kh[k - 1]);
            spdiag[kr - 1] = -M(etas[k - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etas[k - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(Tsn_init[k - 1], M(etas[k - 1], Sswabs[k - 1]));
        }
    }
    if (nilyr > 1) {
        ki = 1; k = ki + nslyr; kr = k + 1;
        if (l_snow) {
            sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
            spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1]));
        } else {
            sbdiag[kr - 1] = 0.0;
            spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
            diag[kr - 1] = A(1.0, M(etai[ki - 1], kh[k]));
            rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(etai[ki - 1], fcondtopn));
        }
        ki = nilyr; k = ki + nslyr; kr = k + 1;
        sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
        spdiag[kr - 1] = 0.0;
        diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
        rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k]), Tbot));
    } else {
        ki = 1; k = ki + nslyr; kr = k + 1;
        if (l_snow) {
            sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
            spdiag[kr - 1] = 0.0;
            diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
            rhs[kr - 1] = A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(M(etai[ki - 1], kh[k]), Tbot));
        } else {
            sbdiag[kr - 1] = 0.0;
            spdiag[kr - 1] = 0.0;
            diag[kr - 1] = A(1.0, M(etai[ki - 1], kh[k]));
            rhs[kr - 1] = A(A(A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1])), M(etai[ki - 1], fcondtopn)),
                            M(M(etai[ki - 1], kh[k]), Tbot));
        }
    }
    for (ki = 2; ki <= nilyr - 1; ++ki) {
        k = ki + nslyr; kr = k + 1;
        sbdiag[kr - 1] = -M(etai[ki - 1], kh[k - 1]);
        spdiag[kr - 1] = -M(etai[ki - 1], kh[k]);
        diag[kr - 1] = A(1.0, M(etai[ki - 1], A(kh[k - 1], kh[k])));
        rhs[kr - 1] = A(Tin_init[ki - 1], M(etai[ki - 1], Iswabs[ki - 1]));
    }
}

// ice_therm_bl99: tridiag_solver
__device__ void tridiag_solver(int nmat, const double *sbdiag, const double *diag, const double *spdiag,
                               const double *rhs, double *xout)
{
    double wgamma[COL_NMAT];
    double wbeta = diag[0];
    xout[0] = D(rhs[0], wbeta);
    for (int k = 2; k <= nmat; ++k) {
        wgamma[k - 1] = D(spdiag[k - 2], wbeta);
        wbeta = S(diag[k - 1], M(sbdiag[k - 1], wgamma[k - 1]));
        xout[k - 1] = D(S(rhs[k - 1], M(sbdiag[k - 1], xout[k - 2])), wbeta);
    }
    for (int k = nmat - 1; k >= 1; --k) xout[k - 1] = S(xout[k - 1], M(wgamma[k], xout[k]));
}

// ice_therm_bl99: temperature_changes; returns VT_OK or VT_NO_CONVERGE
__device__ int temperature_changes(const vt_params &p, double rhoa, double flw, double potT, double Qa,
                                   double shcoef, double lhcoef, double *fswsfc, double *fswint, double *Sswabs,
                                   double *Iswabs, double hilyr, double hslyr, double *zqin, double *zTin,
                                   double *zqsn, double *zTsn, const double *zSin, double *Tsf, double Tbot,
                                   double *fsensn, double *flatn, double *flwoutn, double *fsurfn,
                                   double *fcondtopn, double *fcondbot, double einit)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    const int nitermax = 100;
    const double Tsf_errmax = 5.0e-4;
    const double dt = p.dt;
    int converged = 0, l_snow = 0, l_cold = 1;
    *fcondbot = 0.0;
    double dTsf_prev = 0.0, dTi1_prev = 0.0, dfsens_dT = 0.0, dflat_dT = 0.0, dflwout_dT = 0.0, einex = 0.0;
    double dt_rhoi_hlyr = D(dt, M(RHOI, hilyr));
    if (hslyr > D(HS_MIN, (double)nslyr)) l_snow = 1;
    double Tsn_init[COL_NSLYR], Tsn_start[COL_NSLYR], etas[COL_NSLYR];
    double Tin_init[COL_NILYR], Tin_start[COL_NILYR], dTmat[COL_NILYR], dqmat[COL_NILYR], Tmlts[COL_NILYR],
        etai[COL_NILYR];
    int reduce_kh[COL_NILYR];
    double sbdiag[COL_NMAT], diag[COL_NMAT], spdiag[COL_NMAT], rhs[COL_NMAT], Tmat[COL_NMAT], kh[COL_NMAT];
    for (int k = 0; k < nslyr; ++k) {
        Tsn_init[k] = zTsn[k];
        Tsn_start[k] = zTsn[k];
        etas[k] = l_snow ? D(dt, M(M(RHOS, CP_ICE), hslyr)) : 0.0;
    }
    for (int k = 0; k < nilyr; ++k) {
        Tin_init[k] = zTin[k];
        Tin_start[k] = zTin[k];
        Tmlts[k] = -M(zSin[k], DEPRESST);
    }
    conductivity(p, l_snow, hilyr, hslyr, zTin, zSin, kh);
    double frac = 0.9, dTemp = 0.02;
    double ci, dswabs;
    for (int k = 0; k < nilyr; ++k) {
        double Iswabs_tmp = 0.0;
        if (Tin_init[k] <= S(Tmlts[k], dTemp)) {
            if (p.l_brine) {
                ci = S(CP_ICE, D(M(lfresh(), Tmlts[k]), M(Tin_init[k], Tin_init[k])));
                Iswabs_tmp = col_min(Iswabs[k], D(M(M(frac, S(Tmlts[k], Tin_init[k])), ci), dt_rhoi_hlyr));
            } else {
                ci = CP_ICE;
                Iswabs_tmp = col_min(Iswabs[k], D(M(M(frac, -Tin_init[k]), ci), dt_rhoi_hlyr));
            }
        }
        if (Iswabs_tmp < PUNY) Iswabs_tmp = 0.0;
        dswabs = col_min(S(Iswabs[k], Iswabs_tmp), *fswint);
        *fswsfc = A(*fswsfc, dswabs);
        *fswint = S(*fswint, dswabs);
        Iswabs[k] = Iswabs_tmp;
    }
    for (int k = 0; k < nslyr; ++k) {
        if (l_snow) {
            double Sswabs_tmp = 0.0;
            if (Tsn_init[k] <= -dTemp) {
                Sswabs_tmp = col_min(Sswabs[k], D(-M(frac, Tsn_init[k]), etas[k]));
            }
            if (Sswabs_tmp < PUNY) Sswabs_tmp = 0.0;
            dswabs = col_min(S(Sswabs[k], Sswabs_tmp), *fswint);
            *fswsfc = A(*fswsfc, dswabs);
            *fswint = S(*fswint, dswabs);
            Sswabs[k] = Sswabs_tmp;
        }
    }
    converged = 0;
    double Tsf_start = 0.0, dTsf = 0.0, dTi1, dfsurf_dT, avg_Tsi, enew, avg_Tsf, ferr = 0.0;
    for (int niter = 1; niter <= nitermax; ++niter) {
        if (converged) continue;
        converged = 1;
        dfsurf_dT = 0.0;
        avg_Tsi = 0.0;
        enew = 0.0;
        einex = 0.0;
        for (int k = 0; k < nilyr; ++k) {
            ci = p.l_brine ? S(CP_ICE, D(M(lfresh(), Tmlts[k]), M(zTin[k], Tin_init[k]))) : CP_ICE;
            etai[k] = D(dt_rhoi_hlyr, ci);
        }
        if (p.calc_tsfc) {
            surface_heat_flux(*Tsf, *fswsfc, rhoa, flw, potT, Qa, shcoef, lhcoef, flwoutn, fsensn, flatn, fsurfn);
            dsurface_heat_flux_dTsf(*Tsf, rhoa, shcoef, lhcoef, &dfsurf_dT, &dflwout_dT, &dfsens_dT, &dflat_dT);
            *fcondtopn = l_snow ? M(kh[0], S(*Tsf, zTsn[0])) : M(kh[nslyr], S(*Tsf, zTin[0]));
            if (*Tsf >= 0.0 && *fsurfn < *fcondtopn) *Tsf = -PUNY;
            Tsf_start = *Tsf;
            l_cold = *Tsf < 0.0 ? 1 : 0;
            matrix_calc_tsfc(l_snow, l_cold, *Tsf, Tbot, *fsurfn, dfsurf_dT, Tin_init, Tsn_init, kh, Sswabs,
                             Iswabs, etai, etas, sbdiag, diag, spdiag, rhs);
        } else {
            matrix_know_tsfc(l_snow, Tbot, Tin_init, Tsn_init, kh, Sswabs, Iswabs, etai, etas, sbdiag, diag,
                             spdiag, rhs, *fcondtopn);
        }
        int nmat = nslyr + nilyr + 1;
        tridiag_solver(nmat, sbdiag, diag, spdiag, rhs, Tmat);
        if (p.calc_tsfc) {
            if (l_cold) {
                *Tsf = l_snow ? Tmat[0] : Tmat[nslyr];
            } else {
                *Tsf = 0.0;
            }
            dTsf = S(*Tsf, Tsf_start);
            avg_Tsf = 0.0;
            if (*Tsf > PUNY) {
                *Tsf = 0.0;
                dTsf = -Tsf_start;
                if (p.l_brine) avg_Tsi = 1.0;
                converged = 0;
            } else if (niter > 1 && Tsf_start <= -PUNY && fabs(dTsf) > PUNY && fabs(dTsf_prev) > PUNY &&
                       D(-dTsf, A(dTsf_prev, M(PUNY, PUNY))) > 0.5) {
                if (p.l_brine) {
                    avg_Tsf = 1.0;
                    avg_Tsi = 1.0;
                }
                dTsf = M(0.5, dTsf);
                converged = 0;
            }
            *Tsf = A(*Tsf, M(M(avg_Tsf, 0.5), S(Tsf_start, *Tsf)));
        }
        for (int k = 0; k < nslyr; ++k) {
            zTsn[k] = l_snow ? Tmat[k + 1] : 0.0;
            if (p.l_brine) zTsn[k] = col_min(zTsn[k], 0.0);
            zTsn[k] = A(zTsn[k], M(M(avg_Tsi, 0.5), S(Tsn_start[k], zTsn[k])));
            zqsn[k] = -M(RHOS, S(lfresh(), M(CP_ICE, zTsn[k])));
            enew = A(enew, M(hslyr, zqsn[k]));
            Tsn_start[k] = zTsn[k];
        }
        for (int k = 0; k < nilyr; ++k) { dTmat[k] = 0.0; dqmat[k] = 0.0; reduce_kh[k] = 0; }
        for (int k = 0; k < nilyr; ++k) {
            zTin[k] = Tmat[k + 1 + nslyr];
            if (p.l_brine && zTin[k] > S(Tmlts[k], PUNY)) {
                dTmat[k] = S(zTin[k], Tmlts[k]);
                dqmat[k] = M(M(RHOI, dTmat[k]), S(CP_ICE, D(M(lfresh(), Tmlts[k]), M(zTin[k], zTin[k]))));
                zTin[k] = Tmlts[k];
                reduce_kh[k] = 1;
            }
            if (k == 0 && !p.calc_tsfc) {
                dTi1 = S(zTin[k], Tin_start[k]);
                if (niter > 1 && fabs(dTi1) > PUNY && fabs(dTi1_prev) > PUNY &&
                    D(-dTi1, A(dTi1_prev, M(PUNY, PUNY))) > 0.5) {
                    if (p.l_brine) avg_Tsi = 1.0;
                    dTi1 = M(0.5, dTi1);
                    converged = 0;
                }
                dTi1_prev = dTi1;
            }
            zTin[k] = A(zTin[k], M(M(avg_Tsi, 0.5), S(Tin_start[k], zTin[k])));
            if (p.l_brine) {
                zqin[k] = -M(RHOI, S(A(M(CP_ICE, S(Tmlts[k], zTin[k])), M(lfresh(), S(1.0, D(Tmlts[k], zTin[k])))),
                                     M(CP_OCN, Tmlts[k])));
            } else {
                zqin[k] = -M(RHOI, A(-M(CP_ICE, zTin[k]), lfresh()));
            }
            enew = A(enew, M(hilyr, zqin[k]));
            einex = A(einex, M(hilyr, dqmat[k]));
            Tin_start[k] = zTin[k];
        }
        if (p.calc_tsfc) {
            if (fabs(dTsf) > Tsf_errmax) converged = 0;
            *fsurfn = A(*fsurfn, M(dTsf, dfsurf_dT));
            *fcondtopn = l_snow ? M(kh[0], S(*Tsf, zTsn[0])) : M(kh[nslyr], S(*Tsf, zTin[0]));
            if (*Tsf >= 0.0 && *fsurfn < *fcondtopn) converged = 0;
            dTsf_prev = dTsf;
        }
        *fcondbot = M(kh[nslyr + nilyr], S(zTin[nilyr - 1], Tbot));
        *fcondbot = A(*fcondbot, D(einex, dt));
        ferr = fabs(S(D(S(enew, einit), dt), A(S(*fcondtopn, *fcondbot), *fswint)));
        if (ferr > M(0.9, FERRMAX)) {
            converged = 0;
            for (int k = 0; k < nilyr; ++k) {
                if (reduce_kh[k] && dqmat[k] > 0.0) {
                    frac = col_max(M(0.5, S(1.0, D(ferr, fabs(S(*fcondtopn, *fcondbot))))), P1);
                    kh[k + nslyr + 1] = M(kh[k + nslyr + 1], frac);
                    kh[k + nslyr] = kh[k + nslyr + 1];
                }
            }
        }
    }
    if (!converged) return VT_NO_CONVERGE;
    if (p.calc_tsfc) {
        *flwoutn = A(*flwoutn, M(dTsf_prev, dflwout_dT));
        *fsensn = A(*fsensn, M(dTsf_prev, dfsens_dT));
        *flatn = A(*flatn, M(dTsf_prev, dflat_dT));
    }
    return VT_OK;
}

// ice_therm_vertical: init_vertical_profile (BL99, heat capacity)
__device__ int init_vertical_profile(const vt_params &p, double aicen, double vicen, double vsnon, double *hin,
                                     double *hilyr, double *hsn, double *hslyr, double *zqin, double *zTin,
                                     double *zqsn, double *zTsn, const double *zSin, double *einit)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    double rnslyr = (double)nslyr;
    int tsno_high = 0, tice_high = 0, tsno_low = 0, tice_low = 0;
    double Tmax, Tmlts;
    *einit = 0.0;
    *hin = D(vicen, aicen);
    *hsn = D(vsnon, aicen);
    *hilyr = D(*hin, (double)nilyr);
    *hslyr = D(*hsn, rnslyr);
    for (int k = 0; k < nslyr; ++k) {
        if (*hslyr > D(HS_MIN, rnslyr)) {
            Tmax = D(M(M(-zqsn[k], PUNY), rnslyr), M(M(RHOS, CP_ICE), vsnon));
        } else {
            zqsn[k] = -M(RHOS, lfresh());
            Tmax = PUNY;
        }
        zTsn[k] = D(A(lfresh(), D(zqsn[k], RHOS)), CP_ICE);
        if (zTsn[k] > Tmax) tsno_high = 1;
        else if (zTsn[k] < TMIN) tsno_low = 1;
    }
    if (tsno_high) {
        for (int k = 0; k < nslyr; ++k) {
            Tmax = *hslyr > D(HS_MIN, rnslyr) ? D(M(M(-zqsn[k], PUNY), rnslyr), M(M(RHOS, CP_ICE), vsnon)) : PUNY;
            if (zTsn[k] > Tmax) return VT_TSN_HIGH;
        }
    }
    if (tsno_low) {
        for (int k = 0; k < nslyr; ++k) if (zTsn[k] < TMIN) return VT_TSN_LOW;
    }
    for (int k = 0; k < nslyr; ++k) {
        if (zTsn[k] > 0.0) {
            zTsn[k] = 0.0;
            zqsn[k] = -M(RHOS, lfresh());
        }
        *einit = A(*einit, M(*hslyr, zqsn[k]));
    }
    for (int k = 0; k < nilyr; ++k) {
        if (zSin[k] < S(MIN_SALIN, PUNY)) return VT_ZSIN_LOW;
        Tmlts = -M(zSin[k], DEPRESST);
        zTin[k] = calculate_Tin_from_qin(zqin[k], Tmlts, p.l_brine);
        Tmax = p.l_brine ? Tmlts : D(M(-zqin[k], PUNY), M(M(RHOS, CP_ICE), vicen));
        if (zTin[k] > Tmax) tice_high = 1;
        else if (zTin[k] < TMIN) tice_low = 1;
        if (tice_high) {
            Tmax = p.l_brine ? Tmlts : D(M(-zqin[k], PUNY), M(M(RHOS, CP_ICE), vicen));
            if (zTin[k] > Tmax) return VT_TIN_HIGH;
        }
        if (tice_low) {
            if (zTin[k] < TMIN) return VT_TIN_LOW;
        }
        if (zTin[k] > 0.0) {
            zTin[k] = 0.0;
            zqin[k] = -M(RHOI, lfresh());
        }
        *einit = A(*einit, M(*hilyr, zqin[k]));
    }
    return VT_OK;
}

// ice_therm_vertical: adjust_enthalpy over nlyr layers (z1, z2 have nlyr+1 entries)
__device__ void adjust_enthalpy(int nlyr, const double *z1, const double *z2, double hlyr, double hn, double *qn)
{
    double hq[COL_NILYR > COL_NSLYR ? COL_NILYR : COL_NSLYR];
    double rhlyr = 0.0;
    if (hn > PUNY) rhlyr = D(1.0, hlyr);
    for (int k2 = 0; k2 < nlyr; ++k2) hq[k2] = 0.0;
    int k1 = 1, k2 = 1;
    while (k1 <= nlyr && k2 <= nlyr) {
        double hovlp = S(col_min(z1[k1], z2[k2]), col_max(z1[k1 - 1], z2[k2 - 1]));
        hovlp = col_max(hovlp, 0.0);
        hq[k2 - 1] = A(hq[k2 - 1], M(hovlp, qn[k1 - 1]));
        if (z1[k1] > z2[k2]) ++k2;
        else ++k1;
    }
    for (int k = 0; k < nlyr; ++k) qn[k] = M(hq[k], rhlyr);
}

// ice_therm_vertical: freeboard
__device__ void freeboard(double *snoice, double *hin, double *hsn, double *zqin, const double *zqsn,
                          double *dzi, double *dzs, double *dsnow)
{
    double dhin = 0.0, dhsn = 0.0, hqs = 0.0;
    double wk1 = S(*hsn, D(M(*hin, S(RHOW, RHOI)), RHOS));
    if (wk1 > PUNY && *hsn > PUNY) {
        dhsn = col_min(D(M(wk1, RHOI), RHOW), *hsn);
        dhin = D(M(dhsn, RHOS), RHOI);
    }
    for (int k = COL_NSLYR; k >= 1; --k) {
        if (dhin > PUNY) {
            double dhs = col_min(dhsn, dzs[k - 1]);
            *hsn = S(*hsn, dhs);
            *dsnow = S(*dsnow, dhs);
            dzs[k - 1] = S(dzs[k - 1], dhs);
            dhsn = S(dhsn, dhs);
            dhsn = col_max(dhsn, 0.0);
            hqs = A(hqs, M(dhs, zqsn[k - 1]));
        }
    }
    if (dhin > PUNY) {
        wk1 = A(dzi[0], dhin);
        *hin = A(*hin, dhin);
        zqin[0] = D(A(M(dzi[0], zqin[0]), hqs), wk1);
        dzi[0] = wk1;
        *snoice = A(*snoice, dhin);
    }
}

// ice_therm_vertical: thickness_changes (BL99, heat capacity)
__device__ void thickness_changes(const vt_params &p, double *efinal, double *hin, double *hilyr, double *hsn,
                                  double *hslyr, double *zqin, double *zqsn, double fbot, double Tbot,
                                  double flatn, double fsurfn, double fcondtopn, double fcondbot, double fsnow,
                                  double *hsn_new, double *fhocnn, double *evapn, double *meltt, double *melts,
                                  double *meltb, double *congel, double *snoice, double *mlt_onset,
                                  double *frz_onset, const double *zSin, double *dsnow)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    const double dt = p.dt, yday = p.yday;
    const double qbotmax = -M(M(0.5, RHOI), lfresh());
    double dzi[COL_NILYR], dzs[COL_NSLYR], qm[COL_NILYR], qmlt[COL_NILYR];
    double zi1[COL_NILYR + 1], zi2[COL_NILYR + 1], zs1[COL_NSLYR + 1], zs2[COL_NSLYR + 1];
    double emlt_atm = 0.0, emlt_ocn = 0.0, dhi, dhs, wk1, esub, econ, etop_mlt, ebot_mlt, ebot_gro, qbot, qsub,
        hqtot, hstot, zqsnew, Tmlts;
    *hsn_new = 0.0;
    for (int k = 0; k < nilyr; ++k) dzi[k] = *hilyr;
    for (int k = 0; k < nslyr; ++k) dzs[k] = *hslyr;
    for (int k = 0; k < nilyr; ++k) {
        qmlt[k] = 0.0;
        qm[k] = S(zqin[k], qmlt[k]);
        emlt_atm = 0.0;
        emlt_ocn = 0.0;
    }
    if (!p.l_brine) {
        for (int k = 0; k < nslyr; ++k) {
            double Ts = D(A(lfresh(), D(zqsn[k], RHOS)), CP_ICE);
            if (Ts > 0.0) {
                dhs = D(M(M(CP_ICE, Ts), dzs[k]), lfresh());
                dzs[k] = S(dzs[k], dhs);
                zqsn[k] = -M(RHOS, lfresh());
            }
        }
        for (int k = 0; k < nilyr; ++k) {
            double Ti = D(A(lfresh(), D(zqin[k], RHOI)), CP_ICE);
            if (Ti > 0.0) {
                dhi = D(M(M(CP_ICE, Ti), dzi[k]), lfresh());
                dzi[k] = S(dzi[k], dhi);
                zqin[k] = -M(RHOI, lfresh());
            }
        }
    }
    wk1 = M(-flatn, dt);
    esub = col_max(wk1, 0.0);
    econ = col_min(wk1, 0.0);
    wk1 = M(S(fsurfn, fcondtopn), dt);
    etop_mlt = col_max(wk1, 0.0);
    wk1 = M(S(fcondbot, fbot), dt);
    ebot_mlt = col_max(wk1, 0.0);
    ebot_gro = col_min(wk1, 0.0);
    *evapn = 0.0;
    if (*hsn > PUNY) {
        dhs = D(econ, S(zqsn[0], M(RHOS, LVAP)));
        dzs[0] = A(dzs[0], dhs);
        *evapn = A(*evapn, M(dhs, RHOS));
    } else {
        dhi = D(econ, S(qm[0], M(RHOI, LVAP)));
        dzi[0] = A(dzi[0], dhi);
        *evapn = A(*evapn, M(dhi, RHOI));
        emlt_atm = S(emlt_atm, M(qmlt[0], dhi));
    }
    Tmlts = -M(zSin[nilyr - 1], DEPRESST);
    if (p.l_brine) {
        qbot = -M(RHOI, S(A(M(CP_ICE, S(Tmlts, Tbot)), M(lfresh(), S(1.0, D(Tmlts, Tbot)))), M(CP_OCN, Tmlts)));
        qbot = col_min(qbot, qbotmax);
    } else {
        qbot = -M(RHOI, A(-M(CP_ICE, Tbot), lfresh()));
    }
    dhi = D(ebot_gro, qbot);
    hqtot = A(M(dzi[nilyr - 1], zqin[nilyr - 1]), M(dhi, qbot));
    hstot = 0.0;
    dzi[nilyr - 1] = A(dzi[nilyr - 1], dhi);
    if (dzi[nilyr - 1] > PUNY) {
        zqin[nilyr - 1] = D(hqtot, dzi[nilyr - 1]);
        qmlt[nilyr - 1] = 0.0;
        qm[nilyr - 1] = S(zqin[nilyr - 1], qmlt[nilyr - 1]);
    }
    *congel = A(*congel, dhi);
    if (dhi > PUNY && *frz_onset < PUNY) *frz_onset = yday;
    for (int k = 0; k < nslyr; ++k) {
        qsub = S(zqsn[k], M(RHOS, LVAP));
        dhs = col_max(-dzs[k], D(esub, qsub));
        dzs[k] = A(dzs[k], dhs);
        esub = S(esub, M(dhs, qsub));
        esub = col_max(esub, 0.0);
        *evapn = A(*evapn, M(dhs, RHOS));
        dhs = col_max(-dzs[k], D(etop_mlt, zqsn[k]));
        dzs[k] = A(dzs[k], dhs);
        etop_mlt = S(etop_mlt, M(dhs, zqsn[k]));
        etop_mlt = col_max(etop_mlt, 0.0);
        if (dhs < -PUNY && *mlt_onset < PUNY) *mlt_onset = yday;
        *melts = S(*melts, dhs);
    }
    for (int k = 0; k < nilyr; ++k) {
        qsub = S(qm[k], M(RHOI, LVAP));
        dhi = col_max(-dzi[k], D(esub, qsub));
        dzi[k] = A(dzi[k], dhi);
        esub = S(esub, M(dhi, qsub));
        esub = col_max(esub, 0.0);
        *evapn = A(*evapn, M(dhi, RHOI));
        emlt_ocn = S(emlt_ocn, M(qmlt[k], dhi));
        if (qm[k] < 0.0) {
            dhi = col_max(-dzi[k], D(etop_mlt, qm[k]));
        } else {
            qm[k] = 0.0;
            dhi = -dzi[k];
        }
        emlt_ocn = S(emlt_ocn, M(col_max(zqin[k], qmlt[k]), dhi));
        dzi[k] = A(dzi[k], dhi);
        etop_mlt = col_max(S(etop_mlt, M(dhi, qm[k])), 0.0);
        if (dhi < -PUNY && *mlt_onset < PUNY) *mlt_onset = yday;
        *meltt = S(*meltt, dhi);
    }
    for (int k = nilyr - 1; k >= 0; --k) {
        if (qm[k] < 0.0) {
            dhi = col_max(-dzi[k], D(ebot_mlt, qm[k]));
        } else {
            qm[k] = 0.0;
            dhi = -dzi[k];
        }
        emlt_ocn = S(emlt_ocn, M(col_max(zqin[k], qmlt[k]), dhi));
        dzi[k] = A(dzi[k], dhi);
        ebot_mlt = col_max(S(ebot_mlt, M(dhi, qm[k])), 0.0);
        *meltb = S(*meltb, dhi);
    }
    for (int k = nslyr - 1; k >= 0; --k) {
        dhs = col_max(-dzs[k], D(ebot_mlt, zqsn[k]));
        dzs[k] = A(dzs[k], dhs);
        ebot_mlt = S(ebot_mlt, M(dhs, zqsn[k]));
        ebot_mlt = col_max(ebot_mlt, 0.0);
    }
    *fhocnn = A(fbot, D(A(A(esub, etop_mlt), ebot_mlt), dt));
    if (fsnow > 0.0) {
        *hsn_new = M(D(fsnow, RHOS), dt);
        zqsnew = -M(RHOS, lfresh());
        hstot = A(dzs[0], *hsn_new);
        if (hstot > 0.0) {
            zqsn[0] = D(A(M(dzs[0], zqsn[0]), M(*hsn_new, zqsnew)), hstot);
            zqsn[0] = col_min(zqsn[0], -M(RHOS, lfresh()));
            dzs[0] = hstot;
        }
    }
    *hin = 0.0;
    *hsn = 0.0;
    for (int k = 0; k < nilyr; ++k) *hin = A(*hin, dzi[k]);
    for (int k = 0; k < nslyr; ++k) {
        *hsn = A(*hsn, dzs[k]);
        *dsnow = S(A(*dsnow, dzs[k]), *hslyr);
    }
    freeboard(snoice, hin, hsn, zqin, zqsn, dzi, dzs, dsnow);
    if (*hin > 0.0) {
        *hilyr = D(*hin, (double)nilyr);
    } else {
        *hin = 0.0;
        *hilyr = 0.0;
    }
    if (*hsn > 0.0) {
        *hslyr = D(*hsn, (double)nslyr);
    } else {
        *hsn = 0.0;
        *hslyr = 0.0;
    }
    zi1[0] = 0.0;
    zi1[nilyr] = *hin;
    zi2[0] = 0.0;
    zi2[nilyr] = *hin;
    for (int k = 1; k <= nilyr - 1; ++k) {
        zi1[k] = A(zi1[k - 1], dzi[k - 1]);
        zi2[k] = A(zi2[k - 1], *hilyr);
    }
    adjust_enthalpy(nilyr, zi1, zi2, *hilyr, *hin, zqin);
    if (nslyr > 1) {
        zs1[0] = 0.0;
        zs1[nslyr] = *hsn;
        zs2[0] = 0.0;
        zs2[nslyr] = *hsn;
        for (int k = 1; k <= nslyr - 1; ++k) {
            zs1[k] = A(zs1[k - 1], dzs[k - 1]);
            zs2[k] = A(zs2[k - 1], *hslyr);
        }
        adjust_enthalpy(nslyr, zs1, zs2, *hslyr, *hsn, zqsn);
    }
    *efinal = -M(*evapn, LVAP);
    *evapn = D(*evapn, dt);
    for (int k = 0; k < nslyr; ++k) *efinal = A(*efinal, M(*hslyr, zqsn[k]));
    for (int k = 0; k < nilyr; ++k) *efinal = A(*efinal, M(*hilyr, zqin[k]));
    emlt_atm = 0.0;
    emlt_ocn = 0.0;
    *fhocnn = A(*fhocnn, D(emlt_ocn, dt));
    *efinal = A(*efinal, emlt_atm);
}

// ice_therm_vertical: thermo_vertical (BL99 with heat capacity; no topo ponds)
__device__ int thermo_vertical(const vt_params &p, double *aicen, double *vicen, double *vsnon, double *Tsf,
                               const double *zSin, double *zqin, double *zqsn, double *iage, double flw,
                               double potT, double Qa, double rhoa, double fsnow, double fbot, double Tbot,
                               double lhcoef, double shcoef, double *fswsfc, double *fswint, double *Sswabs,
                               double *Iswabs, double *fsurfn, double *fcondtopn, double *fsensn, double *flatn,
                               double *flwoutn, double *evapn, double *freshn, double *fsaltn, double *fhocnn,
                               double *meltt, double *melts, double *meltb, double *congel, double *snoice,
                               double *mlt_onset, double *frz_onset, double *dsnow)
{
    const int nslyr = COL_NSLYR, nilyr = COL_NILYR;
    const double dt = p.dt;
    double hilyr, hslyr, hin, hsn, hsn_new, worki, works, fcondbot, einit, efinal, einter, fadvocn;
    double zTin[COL_NILYR], zTsn[COL_NSLYR];
    *flwoutn = 0.0;
    *evapn = 0.0;
    *freshn = 0.0;
    *fsaltn = 0.0;
    *fhocnn = 0.0;
    fadvocn = 0.0;
    *meltt = 0.0;
    *meltb = 0.0;
    *melts = 0.0;
    *congel = 0.0;
    *snoice = 0.0;
    *dsnow = 0.0;
    if (p.calc_tsfc) {
        *fsensn = 0.0;
        *flatn = 0.0;
        *fsurfn = 0.0;
        *fcondtopn = 0.0;
    }
    int status = init_vertical_profile(p, *aicen, *vicen, *vsnon, &hin, &hilyr, &hsn, &hslyr, zqin, zTin, zqsn,
                                       zTsn, zSin, &einit);
    if (status) return status;
    worki = hin;
    works = hsn;
    status = temperature_changes(p, rhoa, flw, potT, Qa, shcoef, lhcoef, fswsfc, fswint, Sswabs, Iswabs, hilyr,
                                 hslyr, zqin, zTin, zqsn, zTsn, zSin, Tsf, Tbot, fsensn, flatn, flwoutn, fsurfn,
                                 fcondtopn, &fcondbot, einit);
    if (status) return status;
    einter = 0.0;
    for (int k = 0; k < nslyr; ++k) einter = A(einter, M(hslyr, zqsn[k]));
    for (int k = 0; k < nilyr; ++k) einter = A(einter, M(hilyr, zqin[k]));
    thickness_changes(p, &efinal, &hin, &hilyr, &hsn, &hslyr, zqin, zqsn, fbot, Tbot, *flatn, *fsurfn, *fcondtopn,
                      fcondbot, fsnow, &hsn_new, fhocnn, evapn, meltt, melts, meltb, congel, snoice, mlt_onset,
                      frz_onset, zSin, dsnow);
    // conservation_check_vthermo
    double einp = M(S(S(S(A(S(*fsurfn, *flatn), *fswint), *fhocnn), M(fsnow, lfresh())), fadvocn), dt);
    double ferr = D(fabs(S(S(efinal, einit), einp)), dt);
    if (ferr > FERRMAX) return VT_CONSERVATION;
    double dhi = S(hin, worki);
    double dhs = S(S(hsn, works), hsn_new);
    *freshn = S(A(*freshn, *evapn), D(A(M(RHOI, dhi), M(RHOS, dhs)), dt));
    *fsaltn = S(*fsaltn, D(M(M(M(RHOI, dhi), ICE_REF_SALINITY), 0.001), dt));
    *fhocnn = A(*fhocnn, fadvocn);
    // update_state_vthermo (Tf is Tbot)
    if (hin <= 0.0) {
        *aicen = 0.0;
        *vicen = 0.0;
        *vsnon = 0.0;
        *Tsf = Tbot;
        for (int k = 0; k < nilyr; ++k) zqin[k] = 0.0;
        for (int k = 0; k < nslyr; ++k) zqsn[k] = 0.0;
    } else {
        *vicen = M(*aicen, hin);
        *vsnon = M(*aicen, hsn);
    }
    (void)iage;
    return VT_OK;
}

// column_vertical_thermodynamics: one owned cell (iCell < nCellsSolve).
// status[iCell] gets the abort code (0 when the cell ran through).
extern "C" __global__ void seaice_column_vertical_thermodynamics_v842(
    int *status, double dt_, double yday, double ustar_min, double rdn_ice, double al2, int ncat_, int flags,
    int natmiter, int nSolve,
    // mesh, icestate, tracers_aggregate
    const double *latCell, double *iceAreaCellInitial, double *iceAreaCategoryInitial,
    double *iceVolumeCategoryInitial, double *snowVolumeCategoryInitial, const double *iceAreaCell,
    // tracers
    double *iceAreaCategory, double *iceVolumeCategory, double *snowVolumeCategory, double *surfaceTemperature,
    double *snowEnthalpy, double *iceEnthalpy, const double *iceSalinity, double *iceAge, double *firstYearIceArea,
    // velocity_solver
    const double *uVelocity, double *airStressCellU, double *airStressCellV, const double *oceanStressCellU,
    const double *oceanStressCellV,
    // atmos_coupling, atmos_forcing
    const double *uAirVelocity, const double *vAirVelocity, const double *airLevelHeight,
    const double *airSpecificHumidity, const double *airDensity, const double *airPotentialTemperature,
    const double *snowfallRate, const double *longwaveDown, double *atmosReferenceTemperature2m,
    double *atmosReferenceHumidity2m, double *atmosReferenceSpeed10m, const double *windSpeed,
    // alternative_atmos_forcing (calc_Tsfc = .false.)
    const double *latentHeatFluxCouple, const double *sensibleHeatFluxCouple, const double *surfaceHeatFluxCouple,
    const double *surfaceConductiveFluxCouple,
    // ocean_coupling
    const double *seaSurfaceTemperature, const double *seaSurfaceSalinity, const double *seaFreezingTemperature,
    const double *freezingMeltingPotential,
    // drag
    double *airOceanDragCoefficientRatio, const double *oceanDragCoefficient, double *airDragCoefficient,
    // ocean_fluxes
    double *oceanHeatFluxIceBottom, double *oceanFreshWaterFlux, double *oceanSaltFlux, double *oceanHeatFlux,
    double *oceanShortwaveFlux,
    // melt_growth_rates
    double *lateralIceMeltFraction, double *surfaceIceMelt, double *surfaceIceMeltCategory, double *basalIceMelt,
    double *basalIceMeltCategory, double *snowMelt, double *snowMeltCategory, double *congelation,
    double *congelationCategory, double *snowiceFormation, double *snowiceFormationCategory,
    double *snowThicknessChangeCategory,
    // atmos_fluxes
    double *surfaceHeatFlux, double *surfaceHeatFluxCategory, double *surfaceConductiveFlux,
    double *surfaceConductiveFluxCategory, double *longwaveUp, double *sensibleHeatFlux,
    double *sensibleHeatFluxCategory, double *latentHeatFlux, double *latentHeatFluxCategory,
    double *evaporativeWaterFlux,
    // shortwave
    double *surfaceShortwaveFlux, double *interiorShortwaveFlux, const double *penetratingShortwaveFlux,
    double *absorbedShortwaveFlux, double *absorbedShortwaveSnowLayer, double *absorbedShortwaveIceLayer,
    // diagnostics
    double *meltOnset, double *freezeOnset)
{
    const int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nSolve) return;
    vt_params p;
    p.dt = dt_; p.yday = yday; p.ustar_min = ustar_min; p.rdn_ice = rdn_ice; p.al2 = al2; p.ncat = ncat_;
    p.natmiter = natmiter;
    p.l_brine = (flags >> 0) & 1; p.calc_tsfc = (flags >> 1) & 1; p.calc_strair = (flags >> 2) & 1;
    p.highfreq = (flags >> 3) & 1; p.conduct_mu71 = (flags >> 4) & 1;
    p.tr_iage = (flags >> 5) & 1; p.tr_fy = (flags >> 6) & 1;
    const int ncat = p.ncat, nilyr = COL_NILYR, nslyr = COL_NSLYR;
    const int c = iCell;
    const double dt = p.dt;
    iceAreaCellInitial[c] = iceAreaCell[c];
    for (int n = 0; n < ncat; ++n) {
        iceAreaCategoryInitial[c * ncat + n] = iceAreaCategory[c * ncat + n];
        iceVolumeCategoryInitial[c * ncat + n] = iceVolumeCategory[c * ncat + n];
        snowVolumeCategoryInitial[c * ncat + n] = snowVolumeCategory[c * ncat + n];
    }
    const int lmask_n = latCell[c] > 0.0;
    const int lmask_s = !lmask_n;
    // colpkg_step_therm1
    const double aice = iceAreaCell[c];
    double *vicen = iceVolumeCategory + c * ncat, *vsnon = snowVolumeCategory + c * ncat;
    double *zqin_all = iceEnthalpy + c * ncat * nilyr, *zqsn_all = snowEnthalpy + c * ncat * nslyr;
    // frzmlt_bottom_lateral
    double rside = 0.0, Tbot = seaFreezingTemperature[c], fbot = 0.0;
    const double frzmlt = freezingMeltingPotential[c];
    if (aice > PUNY && frzmlt < 0.0) {
        const double floediam = 300.0, floeshape = 0.66, m1 = 1.6e-6, m2 = 1.36;
        double fside = 0.0;
        double deltaT = col_max(S(seaSurfaceTemperature[c], Tbot), 0.0);
        double sx = oceanStressCellU[c], sy = oceanStressCellV[c];
        double ustar = SQRT(D(SQRT(A(M(sx, sx), M(sy, sy))), RHOW));
        ustar = col_max(ustar, p.ustar_min);
        // fbot_xfer_type 'constant' (MPAS allows 'Cdn_ocn' only with form drag, refused)
        double cpchr = -M(M(CP_OCN, RHOW), 0.006);
        fbot = M(M(cpchr, deltaT), ustar);
        fbot = col_max(fbot, frzmlt);
        double wlat = M(m1, glibc_pow(deltaT, m2));
        rside = D(M(M(wlat, dt), PI), M(floeshape, floediam));
        rside = col_max(0.0, col_min(rside, 1.0));
        for (int n = 0; n < ncat; ++n) {
            double etot = 0.0;
            for (int k = 0; k < nslyr; ++k) etot = A(etot, D(M(zqsn_all[n * nslyr + k], vsnon[n]), (double)nslyr));
            for (int k = 0; k < nilyr; ++k) etot = A(etot, D(M(zqin_all[n * nilyr + k], vicen[n]), (double)nilyr));
            fside = A(fside, D(M(rside, etot), dt));
        }
        double xtmp = D(frzmlt, A(A(fbot, fside), PUNY));
        xtmp = col_min(xtmp, 1.0);
        fbot = M(fbot, xtmp);
        rside = M(rside, xtmp);
    }
    oceanHeatFluxIceBottom[c] = fbot;
    lateralIceMeltFraction[c] = rside;
    for (int n = 0; n < ncat; ++n) {
        const int cn = c * ncat + n;
        snowMeltCategory[cn] = 0.0;
        surfaceIceMeltCategory[cn] = 0.0;
        basalIceMeltCategory[cn] = 0.0;
        congelationCategory[cn] = 0.0;
        snowiceFormationCategory[cn] = 0.0;
        snowThicknessChangeCategory[cn] = 0.0;
        double Trefn = 0.0, Qrefn = 0.0, Urefn = 0.0, lhcoef = 0.0, shcoef = 0.0, worka = 0.0, workb = 0.0;
        double strairxn = 0.0, strairyn = 0.0, Cdn_atm_ratio_n = 0.0, fswabsn = 0.0, flwoutn = 0.0, evapn = 0.0,
               freshn = 0.0, fsaltn = 0.0, fhocnn = 0.0;
        const double aicen_init = iceAreaCategoryInitial[cn];
        if (aicen_init > PUNY) {
            atmo_boundary_layer_ice(p, surfaceTemperature[cn], airPotentialTemperature[c], uAirVelocity[c],
                                    vAirVelocity[c], windSpeed[c], airLevelHeight[c], airSpecificHumidity[c],
                                    airDensity[c], &strairxn, &strairyn, &Trefn, &Qrefn, &worka, &workb, &lhcoef,
                                    &shcoef, &airDragCoefficient[c], &Cdn_atm_ratio_n, uVelocity[c], 0.0,
                                    &Urefn);
            if (p.tr_iage) iceAge[cn] = A(iceAge[cn], dt);
            if (p.tr_fy) {
                // update_FYarea
                if (p.yday >= 259.0 && p.yday < A(259.0, D(dt, SECDAY))) {
                    if (lmask_n) firstYearIceArea[cn] = 0.0;
                }
                if (p.yday >= 75.0 && p.yday < A(75.0, D(dt, SECDAY))) {
                    if (lmask_s) firstYearIceArea[cn] = 0.0;
                }
            }
            if (!p.calc_tsfc) {
                // set_sfcflux (raicen = 1). colpkg_step_therm1 passes fcondtopn_f where
                // set_sfcflux takes fsurfn_f and fsurfn_f where it takes fcondtopn_f
                // (ice_colpkg.F90:2095-2101 against ice_flux_colpkg.F90:169-177), so fsurfn
                // is the coupled conductive flux and fcondtopn the coupled surface flux.
                surfaceHeatFluxCategory[cn] = M(surfaceConductiveFluxCouple[cn], 1.0);
                surfaceConductiveFluxCategory[cn] = M(surfaceHeatFluxCouple[cn], 1.0);
                latentHeatFluxCategory[cn] = M(latentHeatFluxCouple[cn], 1.0);
                sensibleHeatFluxCategory[cn] = M(sensibleHeatFluxCouple[cn], 1.0);
            }
            double meltt_n = 0.0, melts_n = 0.0, meltb_n = 0.0, congel_n = 0.0, snoice_n = 0.0, dsnow_n = 0.0;
            int st = thermo_vertical(p, &iceAreaCategory[cn], &vicen[n], &vsnon[n], &surfaceTemperature[cn],
                                     iceSalinity + cn * nilyr, zqin_all + n * nilyr, zqsn_all + n * nslyr,
                                     &iceAge[cn], longwaveDown[c], airPotentialTemperature[c],
                                     airSpecificHumidity[c], airDensity[c], snowfallRate[c], fbot, Tbot, lhcoef,
                                     shcoef, &surfaceShortwaveFlux[cn], &interiorShortwaveFlux[cn],
                                     absorbedShortwaveSnowLayer + cn * nslyr, absorbedShortwaveIceLayer + cn * nilyr,
                                     &surfaceHeatFluxCategory[cn], &surfaceConductiveFluxCategory[cn],
                                     &sensibleHeatFluxCategory[cn], &latentHeatFluxCategory[cn], &flwoutn, &evapn,
                                     &freshn, &fsaltn, &fhocnn, &meltt_n, &melts_n, &meltb_n, &congel_n,
                                     &snoice_n, &meltOnset[c], &freezeOnset[c], &dsnow_n);
            surfaceIceMeltCategory[cn] = meltt_n;
            snowMeltCategory[cn] = melts_n;
            basalIceMeltCategory[cn] = meltb_n;
            congelationCategory[cn] = congel_n;
            snowiceFormationCategory[cn] = snoice_n;
            snowThicknessChangeCategory[cn] = dsnow_n;
            if (st) {
                status[c] = st;
                return;
            }
            fswabsn = A(A(surfaceShortwaveFlux[cn], interiorShortwaveFlux[cn]), penetratingShortwaveFlux[cn]);
            // merge_fluxes
            const double a = aicen_init;
            airStressCellU[c] = A(airStressCellU[c], M(strairxn, a));
            airStressCellV[c] = A(airStressCellV[c], M(strairyn, a));
            airOceanDragCoefficientRatio[c] = A(airOceanDragCoefficientRatio[c], M(Cdn_atm_ratio_n, a));
            surfaceHeatFlux[c] = A(surfaceHeatFlux[c], M(surfaceHeatFluxCategory[cn], a));
            surfaceConductiveFlux[c] = A(surfaceConductiveFlux[c], M(surfaceConductiveFluxCategory[cn], a));
            sensibleHeatFlux[c] = A(sensibleHeatFlux[c], M(sensibleHeatFluxCategory[cn], a));
            latentHeatFlux[c] = A(latentHeatFlux[c], M(latentHeatFluxCategory[cn], a));
            absorbedShortwaveFlux[c] = A(absorbedShortwaveFlux[c], M(fswabsn, a));
            longwaveUp[c] = A(longwaveUp[c], M(S(flwoutn, M(S(1.0, EMISSIVITY), longwaveDown[c])), a));
            evaporativeWaterFlux[c] = A(evaporativeWaterFlux[c], M(evapn, a));
            atmosReferenceTemperature2m[c] = A(atmosReferenceTemperature2m[c], M(Trefn, a));
            atmosReferenceHumidity2m[c] = A(atmosReferenceHumidity2m[c], M(Qrefn, a));
            atmosReferenceSpeed10m[c] = A(atmosReferenceSpeed10m[c], M(Urefn, a));
            oceanFreshWaterFlux[c] = A(oceanFreshWaterFlux[c], M(freshn, a));
            oceanSaltFlux[c] = A(oceanSaltFlux[c], M(fsaltn, a));
            oceanHeatFlux[c] = A(oceanHeatFlux[c], M(fhocnn, a));
            oceanShortwaveFlux[c] = A(oceanShortwaveFlux[c], M(penetratingShortwaveFlux[cn], a));
            surfaceIceMelt[c] = A(surfaceIceMelt[c], M(meltt_n, a));
            basalIceMelt[c] = A(basalIceMelt[c], M(meltb_n, a));
            snowMelt[c] = A(snowMelt[c], M(melts_n, a));
            congelation[c] = A(congelation[c], M(congel_n, a));
            snowiceFormation[c] = A(snowiceFormation[c], M(snoice_n, a));
        }
    }
    status[c] = VT_OK;
}

#endif  // WOOF_GLIBC239_TRIG64
