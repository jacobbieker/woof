// MPAS-Seaice v8.4.2: the column biogeochemistry's brine height, vertical
// salinity and vertical tracers' nonreactive tracer (column_biogeochemistry ->
// colpkg_biogeochemistry with tr_brine, without the vertical or the skeletal
// biochemistry).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     2781-3272  column_biogeochemistry
//   src/core_seaice/column/ice_colpkg.F90:
//     4884-5262  colpkg_biogeochemistry
//   src/core_seaice/column/ice_brine.F90:
//     17-31      maxhbr, viscos, a1-a3, b1-b2
//     46-127     preflushing_changes
//     136-322    compute_microS_mushy
//     326-446    prepare_hbrine
//     461-579    update_hbrine
//     841-943    calculate_drho
//   src/core_seaice/column/ice_zbgc_shared.F90:
//     110-133    exp_h, k_o, thinS, viscos_dynamic, Dm, Ra_c
//     370-524    regrid_stationary
//   src/core_seaice/column/ice_algae.F90:
//     29-340     zbio
//     747-1417   z_biogeochemistry
//     2159-2223  thin_ice_flux
//     2232-2429  compute_FCT_matrix
//     2438-2559  compute_FCT_corr
//     2568-2611  tridiag_solverz
//     2617-2710  check_conservation_FCT
//   src/core_seaice/column/ice_aerosol.F90:
//     447-826    update_snow_bgc
//   src/core_seaice/column/ice_zbgc.F90:
//     528-653    merge_bgc_fluxes
// (remap_zbgc and calculate_Tin_from_qin are in column_zbgc_shared_v842.cu)
//   src/core_seaice/column/ice_mushy_physics.F90:
//     29-110     the Assur (1958) liquidus constants
//     361-409    temperature_mush
//   src/core_seaice/column/ice_colpkg_shared.F90:410 rhosi
// (Nicole Jeffery, Elizabeth C. Hunke, Adrian K. Turner, LANL, and the CICE
// Consortium). Copyright (c) 2013-2019, Los Alamos National Security, LLC
// (LANS) and the University Corporation for Atmospheric Research (UCAR);
// BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is
// not the version available from LANS and UCAR.
//
// Appended to column_v842.cu (its rounding macros, constants, layout and
// liquid_fraction) and column_zbgc_shared_v842.cu. x**2 and x**3 (bTin**2, bTin**3, iphin**exp_h,
// bphin**exp_h, hbr_old**2) are libgcc's __powidf2, the call gfortran -O0
// makes: x*x and x*(x*x). exp is the shared glibc transcription (glibc_exp,
// glibc_flt64.cuh); sqrt the correctly rounded instruction. MAX and MIN of
// three are gfortran's pairwise from the left; MINVAL a running minimum.
//
// Upstream behaviour kept word for word (each also in the module docstring):
// calculate_drho's mm loop assigns rho_2a(k) the same expression twice (or
// not at all); compute_microS_mushy compares the brine height hbr_old with
// the critical Rayleigh number Ra_c (hbr_old >= Ra_c, a length against a
// dimensionless number) to switch on gravity drainage; remap_zbgc would read
// trdr(0) if its first donor point lay below the first receiver point, which
// the CICE-to-biology remap colpkg_biogeochemistry makes (hinS = hice) never
// does: trdr[0] here is 0.
//
// One thread is one owned cell; COL_NILYR and COL_NBLYR (prepended) are
// nIceLayers and nBioLayers. Arrays are 1-based where the Fortran's are.

#define BG_MAXHBR 1.25
#define BG_VISCOS 2.1e-6
#define BG_A1 (-21.4)
#define BG_A2 (-0.886)
#define BG_A3 (-0.012)
#define BG_B1 1000.0
#define BG_B2 0.8
#define BG_K_O 3.e-8
#define BG_THINS 0.05
#define BG_VISCOS_DYNAMIC 2.2
#define BG_DM 1.0e-9
#define BG_RA_C 0.05
#define BG_RHOSI 940.0
#define BG_GRAVIT 9.80616
#define BG_SMIN 0.01
#define BG_DH_MIN 0.001

#define BG_OK 0
#define BG_FBRI 1      // ice_brine preflushing: fbri <= c0
#define BG_REMAP 2     // ice: remap_layers_bgc error

#define NB2 (COL_NBLYR + 2)
#define NB1 (COL_NBLYR + 1)

__device__ __forceinline__ double bg_powi3(double x) { return M(x, M(x, x)); }
__device__ __forceinline__ double bg_sign(double a, double b) { return copysign(fabs(a), b); }

// ice_mushy_physics: temperature_mush, its parameters folded as gfortran folds them
__device__ double temperature_mush(double zqin, double zSin)
{
    const double lf = lfresh();
    const double az1p = D(AZ1_LIQ, 1000.0), bz1p = D(BZ1_LIQ, 1000.0);
    const double az2p = D(AZ2_LIQ, 1000.0), bz2p = D(BZ2_LIQ, 1000.0);
    const double dcp = S(M(RHOW, CP_OCN), M(RHOI, CP_ICE));
    const double AS1 = M(az1p, dcp), AC1 = M(M(RHOI, CP_ICE), AZ1_LIQ);
    const double BS1 = A(M(A(1.0, bz1p), dcp), M(M(RHOI, lf), az1p));
    const double BQ1 = -AZ1_LIQ;
    const double BC1 = S(M(M(RHOI, CP_ICE), BZ1_LIQ), M(M(RHOI, lf), AZ1_LIQ));
    const double CS1 = M(M(RHOI, lf), A(1.0, bz1p));
    const double CQ1 = -BZ1_LIQ;
    const double CC1 = -M(M(RHOI, lf), BZ1_LIQ);
    const double AS2 = M(az2p, dcp), AC2 = M(M(RHOI, CP_ICE), AZ2_LIQ);
    const double BS2 = A(M(A(1.0, bz2p), dcp), M(M(RHOI, lf), az2p));
    const double BQ2 = -AZ2_LIQ;
    const double BC2 = S(M(M(RHOI, CP_ICE), BZ2_LIQ), M(M(RHOI, lf), AZ2_LIQ));
    const double CS2 = M(M(RHOI, lf), A(1.0, bz2p));
    const double CQ2 = -BZ2_LIQ;
    const double CC2 = -M(M(RHOI, lf), BZ2_LIQ);
    const double D_liq = M(D(A(A(1.0, M(az1p, TB_LIQ)), bz1p), A(M(AZ1_LIQ, TB_LIQ), BZ1_LIQ)),
                           A(M(S(M(CP_OCN, RHOW), M(CP_ICE, RHOI)), TB_LIQ), M(lf, RHOI)));
    const double E_liq = S(M(M(CP_ICE, RHOI), TB_LIQ), M(lf, RHOI));
    const double F1 = D(-M(M(1000.0, CP_OCN), RHOW), AZ1_LIQ), G1 = -1000.0;
    const double H1 = D(-M(M(BZ1_LIQ, CP_OCN), RHOW), AZ1_LIQ);
    const double F2 = D(-M(M(1000.0, CP_OCN), RHOW), AZ2_LIQ), G2 = -1000.0;
    const double H2 = D(-M(M(BZ2_LIQ, CP_OCN), RHOW), AZ2_LIQ);
    const double I_liq = D(1.0, M(CP_OCN, RHOW));

    double S_low = zSin < SB_LIQ ? 1.0 : 0.0;
    double q0 = A(M(A(D(M(F1, zSin), A(G1, zSin)), H1), S_low),
                  M(A(D(M(F2, zSin), A(G2, zSin)), H2), S(1.0, S_low)));
    double q_melt = zqin > q0 ? 1.0 : 0.0;
    double qb = A(M(D_liq, zSin), E_liq);
    double t_high = zqin > qb ? 1.0 : 0.0;
    double t_low = S(1.0, t_high);
    double Aq = A(M(A(M(AS1, zSin), AC1), t_high), M(A(M(AS2, zSin), AC2), t_low));
    double Bq = A(M(A(A(M(BS1, zSin), M(BQ1, zqin)), BC1), t_high),
                  M(A(A(M(BS2, zSin), M(BQ2, zqin)), BC2), t_low));
    double Cq = A(M(A(A(M(CS1, zSin), M(CQ1, zqin)), CC1), t_high),
                  M(A(A(M(CS2, zSin), M(CQ2, zqin)), CC2), t_low));
    double zTin = D(A(-Bq, SQRT(col_max(S(M(Bq, Bq), M(M(4.0, Aq), Cq)), PUNY))), M(2.0, Aq));
    return A(M(M(q_melt, zqin), I_liq), M(S(1.0, q_melt), zTin));
}

// preflushing_changes; returns BG_FBRI when fbri <= 0 on entry (the rest still runs)
__device__ int preflushing_changes(double aicen, double vicen, double vsnon, double meltb, double meltt,
                                   double congel, double snoice, double *hice_old, double *dhice, double *fbri,
                                   double *dhbr_top, double *dhbr_bot, double *hbr_old, double *hin,
                                   double *hsn, int firstice)
{
    int stop = *fbri <= 0.0 ? BG_FBRI : BG_OK;
    *hin = D(vicen, aicen);
    *hsn = D(vsnon, aicen);
    double hin_old = col_max(0.0, S(S(A(A(*hin, meltb), meltt), congel), snoice));
    *dhice = S(hin_old, *hice_old);
    *dhbr_top = S(S(meltt, snoice), *dhice);
    *dhbr_bot = S(congel, meltb);
    if (*hice_old < PUNY || hin_old < PUNY || firstice) {
        *hice_old = *hin;
        *dhbr_top = 0.0;
        *dhbr_bot = 0.0;
        *dhice = 0.0;
        *fbri = 1.0;
    }
    *hbr_old = M(*fbri, *hice_old);
    return stop;
}

// remap_zbgc from the CICE grid to the biology grid at hinS = hice, as
// compute_microS_mushy makes it
__device__ __forceinline__ int remap_cice_to_bio(const double *src, double hice, double hinS, const double *cgrid,
                                                 const double *bgrid, double *out)
{
    return remap_zbgc(src, 0, COL_NILYR, COL_NBLYR, hice, hinS, cgrid, bgrid, MIN_SALIN, out);
}

// prepare_hbrine (1-based arrays)
__device__ void prepare_hbrine(const double *bSin, const double *bTin, double *iTin, double *brine_sal,
                               double *brine_rho, double *ibrine_sal, double *ibrine_rho, double *sice_rho,
                               double *bphin, double *iphin, double *kperm, double *bphi_min,
                               const double *i_grid, double sss)
{
    double kin[NB1 + 1];
    double igrm, igrp, rigr;
    *sice_rho = 0.0;
    for (int k = 1; k <= NB1; ++k) {
        igrm = k == 1 ? 0.0 : S(i_grid[k], i_grid[k - 1]);
        double t = bTin[k];
        brine_sal[k] = A(A(M(BG_A1, t), M(BG_A2, M(t, t))), M(BG_A3, bg_powi3(t)));
        brine_rho[k] = A(BG_B1, M(BG_B2, brine_sal[k]));
        bphin[k] = col_max(PUNY, D(M(bSin[k], BG_RHOSI), M(brine_sal[k], brine_rho[k])));
        bphin[k] = col_min(1.0, bphin[k]);
        kin[k] = M(BG_K_O, bg_powi3(bphin[k]));
        *sice_rho = A(*sice_rho, M(A(M(RHOI, S(1.0, bphin[k])), M(brine_rho[k], bphin[k])), igrm));
    }
    brine_sal[NB2] = sss;
    brine_rho[NB2] = RHOW;
    bphin[NB2] = 1.0;
    ibrine_sal[1] = brine_sal[2];
    ibrine_sal[NB1] = brine_sal[NB2];
    ibrine_rho[1] = brine_rho[2];
    ibrine_rho[NB1] = brine_rho[NB2];
    iTin[1] = bTin[2];
    iTin[NB1] = bTin[NB1];
    iphin[1] = bphin[2];
    iphin[NB1] = bphin[NB1];
    double k_min = kin[2];                       // minval(kin(2:nblyr+1)), nblyr >= 1
    for (int k = 3; k <= NB1; ++k)
        if (kin[k] < k_min) k_min = kin[k];
    *kperm = 0.0;
    double ktemp = 0.0;
    *bphi_min = bphin[1];
    for (int k = 2; k <= COL_NBLYR; ++k) {
        if (k_min > 0.0) {
            ktemp = A(ktemp, D(1.0, kin[k]));
            *kperm = k_min;
        }
        igrp = S(i_grid[k + 1], i_grid[k]);
        igrm = S(i_grid[k], i_grid[k - 1]);
        rigr = D(1.0, S(i_grid[k + 1], i_grid[k - 1]));
        ibrine_sal[k] = M(A(M(brine_sal[k + 1], igrp), M(brine_sal[k], igrm)), rigr);
        ibrine_rho[k] = M(A(M(brine_rho[k + 1], igrp), M(brine_rho[k], igrm)), rigr);
        iTin[k] = M(A(M(bTin[k + 1], igrp), M(bTin[k], igrm)), rigr);
        iphin[k] = col_max(PUNY, M(A(M(bphin[k + 1], igrp), M(bphin[k], igrm)), rigr));
        iphin[k] = col_min(1.0, iphin[k]);
    }
    if (k_min > 0.0) {
        ktemp = A(ktemp, D(1.0, kin[NB1]));
        *kperm = D((double)COL_NBLYR, ktemp);
    }
}

// calculate_drho (1-based arrays)
__device__ void calculate_drho(const double *i_grid, const double *b_grid, const double *brine_rho,
                               const double *ibrine_rho, double *drho)
{
    const int nblyr = COL_NBLYR;
    double rho_a[NB1 + 1], rho_2a[NB1 + 1], rho_b[NB1 + 1], rho_2b[NB1 + 1];
    for (int k = 1; k <= NB1; ++k) { rho_a[k] = 0.0; rho_2a[k] = 0.0; rho_b[k] = 0.0; rho_2b[k] = 0.0; drho[k] = 0.0; }
    for (int k = 1; k <= NB1; ++k) {
        if (k == 2) {
            rho_a[2] = D(A(M(brine_rho[2], b_grid[2]),
                           M(M(A(ibrine_rho[2], brine_rho[2]), 0.5), S(i_grid[2], b_grid[2]))), i_grid[2]);
            rho_b[2] = brine_rho[2];
        } else if (k > 2 && k < nblyr + 1) {
            rho_a[k] = D(A(A(M(rho_a[k - 1], i_grid[k - 1]),
                             M(M(A(ibrine_rho[k - 1], brine_rho[k]), 0.5), S(b_grid[k], i_grid[k - 1]))),
                           M(M(A(ibrine_rho[k], brine_rho[k]), 0.5), S(i_grid[k], b_grid[k]))), i_grid[k]);
            rho_b[k] = brine_rho[k];
        } else {
            rho_a[nblyr + 1] = D(A(A(M(rho_a[nblyr], i_grid[nblyr]),
                                     M(M(A(ibrine_rho[nblyr], brine_rho[nblyr + 1]), 0.5),
                                       S(b_grid[nblyr + 1], i_grid[nblyr]))),
                                   M(brine_rho[nblyr + 1], S(i_grid[nblyr + 1], b_grid[nblyr + 1]))),
                                 i_grid[nblyr + 1]);
            rho_a[1] = brine_rho[2];
            rho_b[nblyr + 1] = brine_rho[nblyr + 1];
            rho_b[1] = brine_rho[2];
        }
    }
    for (int k = 1; k <= NB1; ++k) {
        if (k == 1) {
            rho_2a[1] = D(A(M(rho_a[1], b_grid[2]), M(M(0.5, A(brine_rho[2], ibrine_rho[2])),
                                                     S(i_grid[2], b_grid[2]))), i_grid[2]);
            rho_2b[1] = brine_rho[2];
        } else {
            int mstop = 2 * (k - 1) + 1, mstart;
            if (mstop < nblyr + 1) {
                rho_2a[k] = rho_a[mstop];
                mstart = 2;
                mstop = 1;
            } else {
                mstart = nblyr + 2;
                mstop = nblyr + 3;
            }
            for (int mm = mstart; mm <= mstop; ++mm)
                rho_2a[k] = D(M(A(rho_a[nblyr + 1], M(RHOW, S(M(2.0, i_grid[k]), 1.0))), 0.5), i_grid[k]);
            rho_2b[k] = brine_rho[k + 1];
        }
        drho[k] = col_max(S(rho_b[k], rho_2b[k]),
                          col_max(col_max(0.0, M(2.0, S(rho_a[k], rho_2a[k]))),
                                  D(M(2.0, S(brine_rho[k], brine_rho[k + 1])), (double)nblyr)));
    }
}

// compute_microS_mushy (bgrid, igrid 1-based; cgrid 0-based as the Fortran's cgrid(1:))
__device__ int compute_microS_mushy(const double *bgrid, const double *cgrid0, const double *bgrid0,
                                    const double *igrid, const double *qice, const double *sice, double hice_old,
                                    double *hbr_old, double sss, double sst, double *bTin, double *bphin,
                                    double *kperm, double *bphi_min, double *sice_rho, double *iDin, double l_sk,
                                    double *iphin_out)
{
    double trtmp_s[COL_NBLYR + 1], trtmp_q[COL_NBLYR + 1];
    double bSin[NB2 + 1], bqin[NB2 + 1], brine_sal[NB2 + 1], brine_rho[NB2 + 1];
    double iTin[NB1 + 1], iphin[NB1 + 1], ibrine_rho[NB1 + 1], ibrine_sal[NB1 + 1], drho[NB1 + 1];
    for (int k = 0; k <= COL_NBLYR; ++k) { trtmp_s[k] = 0.0; trtmp_q[k] = 0.0; }
    for (int k = 1; k <= NB1; ++k) iDin[k] = 0.0;
    *hbr_old = col_min(*hbr_old, M(BG_MAXHBR, hice_old));
    double hinc_old = hice_old;
    if (remap_cice_to_bio(sice, hinc_old, hinc_old, cgrid0, bgrid0, trtmp_s)) return BG_REMAP;
    if (remap_cice_to_bio(qice, hinc_old, hinc_old, cgrid0, bgrid0, trtmp_q)) return BG_REMAP;
    for (int k = 1; k <= COL_NBLYR; ++k) {
        bqin[k + 1] = col_min(0.0, trtmp_q[k]);
        bSin[k + 1] = col_max(BG_SMIN, trtmp_s[k]);
        bTin[k + 1] = temperature_mush(bqin[k + 1], bSin[k + 1]);
        bphin[k + 1] = liquid_fraction(bTin[k + 1], bSin[k + 1]);
    }
    bSin[1] = bSin[2];
    bTin[1] = bTin[2];
    bphin[1] = bphin[2];
    bphin[NB2] = 1.0;
    bSin[NB2] = sss;
    bTin[NB2] = sst;
    bphin[NB2] = 1.0;
    prepare_hbrine(bSin, bTin, iTin, brine_sal, brine_rho, ibrine_sal, ibrine_rho, sice_rho, bphin, iphin,
                   kperm, bphi_min, igrid, sss);
    calculate_drho(igrid, bgrid, brine_rho, ibrine_rho, drho);
    for (int k = 1; k <= NB1; ++k) iphin_out[k] = iphin[k];
    for (int k = 2; k <= NB1; ++k) {
        double hb2 = M(*hbr_old, *hbr_old);
        double ikin = M(BG_K_O, bg_powi3(iphin[k]));
        iDin[k] = D(M(iphin[k], BG_DM), hb2);
        if (*hbr_old >= BG_RA_C)
            iDin[k] = A(iDin[k], D(M(D(M(M(l_sk, ikin), BG_GRAVIT), BG_VISCOS_DYNAMIC), drho[k]), hb2));
    }
    return BG_OK;
}

// update_hbrine
__device__ void update_hbrine(double meltt, double melts, double dt, double hin, double hsn, double hin_old,
                              double *hbr, double hbr_old, double *fbri, double *dhS_top, double dhS_bottom,
                              double kperm, double *bphi_min, double *darcy_V, double bphin, double aice0,
                              double *dh_direct)
{
    double hbrocn = 0.0, hbrocn_new = 0.0, hbrmin, dhbr, darcy_coeff, dhrunoff, dhflood, dhbr_hin;
    *darcy_V = 0.0;
    *dh_direct = 0.0;
    double h_ocn = A(M(D(BG_RHOSI, RHOW), hin), M(D(RHOS, RHOW), hsn));
    if (hbr_old > BG_THINS && hin_old > BG_THINS && hin > BG_THINS) {
        hbrmin = BG_THINS;
        *dhS_top = -D(M(col_max(0.0, col_min(S(hin_old, hbr_old), meltt)), RHOI), RHOW);
        *dhS_top = S(*dhS_top, D(M(col_max(0.0, melts), RHOS), RHOW));
        dhbr = S(dhS_bottom, *dhS_top);
        *hbr = col_max(PUNY, A(hbr_old, dhbr));
        hbrocn = S(*hbr, h_ocn);
        darcy_coeff = col_max(0.0, D(M(kperm, BG_GRAVIT), M(BG_VISCOS, hbr_old)));
        if (hbrocn > 0.0 && *hbr > BG_THINS) {
            *bphi_min = bphin;
            dhrunoff = -M(*dhS_top, aice0);
            hbrocn = col_max(0.0, S(hbrocn, dhrunoff));
            hbrocn_new = M(hbrocn, glibc_exp(-M(D(darcy_coeff, *bphi_min), dt)));
            *hbr = col_max(hbrmin, A(h_ocn, hbrocn_new));
            hbrocn_new = S(*hbr, h_ocn);
            *darcy_V = -bg_sign(M(D(S(hbrocn, hbrocn_new), dt), *bphi_min), hbrocn);
            *dhS_top = A(S(*dhS_top, D(M(*darcy_V, dt), *bphi_min)), dhrunoff);
            *dh_direct = dhrunoff;
        } else if (hbrocn < 0.0 && *hbr > BG_THINS) {
            hbrocn_new = M(hbrocn, glibc_exp(-M(D(darcy_coeff, *bphi_min), dt)));
            dhflood = M(col_max(0.0, S(hbrocn_new, hbrocn)), aice0);
            *hbr = col_max(hbrmin, A(h_ocn, hbrocn_new));
            *darcy_V = -bg_sign(M(D(A(S(hbrocn, hbrocn_new), dhflood), dt), *bphi_min), hbrocn);
            *dhS_top = S(S(*dhS_top, D(M(*darcy_V, dt), *bphi_min)), dhflood);
            *dh_direct = -dhflood;
        }
    } else {
        hbrmin = col_min(BG_THINS, hin);
        *hbr = col_max(hbrmin, S(A(hbr_old, dhS_bottom), *dhS_top));
        dhbr_hin = S(*hbr, h_ocn);
        if (fabs(dhbr_hin) > BG_DH_MIN) *hbr = col_max(hbrmin, A(h_ocn, bg_sign(BG_DH_MIN, dhbr_hin)));
        *dhS_top = A(S(hbr_old, *hbr), dhS_bottom);
    }
    *fbri = D(*hbr, hin);
}

// ---------------------------------------------------------------------------
// the vertical salinity (solve_zsal): compute_microS and ice_zsalinity
// ---------------------------------------------------------------------------

#define BG_SALT_LOSS 0.4
#define BG_MAX_SALIN 200.0
#define BG_LAPIDUS_G 0.3
#define BG_MAXSALIN_STOP 3     // bSin(k) > max_salin

// (trcrn(nt_bgc_S:)) are the category's tracers, written in place; 1-based
// bgrid, igrid and profiles, 0-based cgrid0/bgrid0. Its sloss and the local
// Rayleigh it derives are never read. Returns BG_REMAP on remap_zbgc's stop.
__device__ int compute_microS(const double *bgrid0, const double *cgrid0, const double *igrid, double *sice,
                              const double *qice, double tsfc, double *zsal, double hice_old, double *hbr_old,
                              double sss, double sst, double *bTin, double *iTin, double *bphin, double *kperm,
                              double *bphi_min, int firstice, double *bSin, double *brine_sal, double *brine_rho,
                              double *iphin, double *ibrine_rho, double *ibrine_sal, double *sice_rho)
{
    double trtmp[COL_NBLYR + 1], trtmp0[COL_NILYR];
    for (int k = 1; k <= NB2; ++k) { bTin[k] = 0.0; bSin[k] = 0.0; }
    for (int k = 0; k <= COL_NBLYR; ++k) trtmp[k] = 0.0;
    const double surface_S = MIN_SALIN;
    const double hinc_old = hice_old;
    if (firstice) {
        for (int k = 0; k < COL_NILYR; ++k) sice[k] = M(sss, BG_SALT_LOSS);
        if (remap_zbgc(sice, 0, COL_NILYR, COL_NBLYR, hinc_old, hinc_old, cgrid0, bgrid0, surface_S, trtmp))
            return BG_REMAP;
        for (int k = 1; k <= COL_NBLYR; ++k) {
            zsal[k - 1] = col_max(MIN_SALIN, trtmp[k]);
            bSin[k + 1] = col_max(MIN_SALIN, zsal[k - 1]);
        }
        bSin[1] = bSin[2];
        bSin[NB2] = sss;
    } else if (*hbr_old > M(BG_MAXHBR, hice_old)) {
        if (remap_zbgc(zsal, 0, COL_NBLYR, COL_NBLYR, *hbr_old, M(BG_MAXHBR, hinc_old), bgrid0, bgrid0, surface_S,
                       trtmp))
            return BG_REMAP;
        for (int k = 1; k <= COL_NBLYR; ++k) {
            bSin[k + 1] = col_max(MIN_SALIN, trtmp[k]);
            zsal[k - 1] = bSin[k + 1];
        }
        bSin[1] = bSin[2];
        bSin[NB2] = sss;
        *hbr_old = M(BG_MAXHBR, hinc_old);
    } else {
        for (int k = 1; k <= COL_NBLYR; ++k) {
            zsal[k - 1] = col_max(MIN_SALIN, zsal[k - 1]);
            bSin[k + 1] = zsal[k - 1];
        }
        bSin[1] = bSin[2];
        bSin[NB2] = sss;
    }
    for (int k = 0; k < COL_NILYR; ++k) trtmp0[k] = bg_Tin_from_qin(qice[k], -M(sice[k], DEPRESST));
    for (int k = 0; k <= COL_NBLYR; ++k) trtmp[k] = 0.0;
    if (remap_zbgc(trtmp0, 0, COL_NILYR, COL_NBLYR, hinc_old, *hbr_old, cgrid0, bgrid0, surface_S, trtmp))
        return BG_REMAP;
    for (int k = 1; k <= COL_NBLYR; ++k) bTin[k + 1] = col_min(-M(bSin[k + 1], DEPRESST), trtmp[k]);
    bTin[1] = col_min(-M(MIN_SALIN, DEPRESST), M(A(bTin[2], tsfc), 0.5));
    bTin[NB2] = sst;
    prepare_hbrine(bSin, bTin, iTin, brine_sal, brine_rho, ibrine_sal, ibrine_rho, sice_rho, bphin, iphin,
                   kperm, bphi_min, igrid, sss);
    return BG_OK;
}

struct zsal_params {
    double dts_b, l_skS, grid_oS, l_sk;
};

// solve_S_dt (ice_zsalinity.F90:435-885), 1-based arrays; the salt
// conservation check is off (test_conservation), and what only it reads
// (Ci_s, Vi_s, the boundary and molecular fluxes, the corrections) is not
// carried. calc_salt_fluxes (891-1023) is folded in: only fzsaln and
// fzsaln_g leave it.
__device__ int solve_S_dt(int *cflag_out, int nint, double dts, double *bSin, const double *bTin, double *bphin,
                          double *iphin, const double *igrid, const double *bgrid, double *ikin, double hbri_old,
                          double hbrin, double hice_old, double *iDin, double darcy_V, const double *brine_sal,
                          int *Rayleigh, int first_ice, double sss, double dt, double dht, double dhb,
                          const double *brine_rho, const double *ibrine_sal, const double *ibrine_rho,
                          double *fzsaln, double *fzsaln_g, const double *S_bot, const zsal_params &zp)
{
    const int nblyr = COL_NBLYR;
    int l_stop = BG_OK, cflag = 0;
    double iDin_p[NB1 + 1], dSbdx[NB1 + 1], drho[NB1 + 1], Ui_s[NB1 + 1], ivel[NB1 + 1];
    double Din_p[NB2 + 1], Sintemp[NB2 + 1], pre_sin[NB2 + 1], pre_sinb[NB2 + 1], bgrid_temp[NB2 + 1];
    double Q_s[NB2 + 1], C_s[NB2 + 1], V_s[NB2 + 1], U_s[NB2 + 1], F_s[NB2 + 1];
    double vel[COL_NBLYR + 1], lapA[COL_NBLYR + 1], lapB[COL_NBLYR + 1], lapidus_diff[COL_NBLYR + 1];
    for (int k = 1; k <= NB1; ++k) { iDin_p[k] = 0.0; Ui_s[k] = 0.0; ivel[k] = 0.0; dSbdx[k] = 0.0; }
    for (int k = 1; k <= NB2; ++k) { Din_p[k] = 0.0; V_s[k] = 0.0; U_s[k] = 0.0; Q_s[k] = 0.0; C_s[k] = 0.0; F_s[k] = 0.0; }
    for (int k = 1; k <= nblyr; ++k) { lapA[k] = 1.0; lapB[k] = 1.0; vel[k] = 0.0; }
    lapA[nblyr] = 0.0;
    lapB[1] = 0.0;
    double dh = 0.0, dbgrid = 2.0, lapidus = 0.0;
    *fzsaln = 0.0;
    *fzsaln_g = 0.0;
    calculate_drho(igrid, bgrid, brine_rho, ibrine_rho, drho);
    const double h2 = M(hbri_old, hbri_old);
    for (int k = 2; k <= NB1; ++k) {
        iDin_p[k] = D(M(D(M(M(BG_K_O, BG_GRAVIT), zp.l_skS), BG_VISCOS_DYNAMIC), drho[k]), h2);
        Din_p[k] = D(A(M(iDin_p[k], S(igrid[k], bgrid[k])), M(iDin_p[k - 1], S(bgrid[k], igrid[k - 1]))),
                     S(igrid[k], igrid[k - 1]));
    }
    for (int k = 1; k <= NB2; ++k) bgrid_temp[k] = bgrid[k];
    Din_p[NB2] = iDin_p[NB1];
    if (!*Rayleigh && hbrin < BG_RA_C) {
        for (int k = 1; k <= NB2; ++k) Din_p[k] = 0.0;
        for (int k = 1; k <= NB1; ++k) iDin_p[k] = 0.0;
    } else {
        *Rayleigh = 1;
    }
    if (hbri_old > BG_THINS && hbrin > BG_THINS && hice_old > BG_THINS && !first_ice) {
        cflag = 1;
        bgrid_temp[1] = 0.0;
        bgrid_temp[NB2] = 1.0;
        dbgrid = D(igrid[2], S(bgrid_temp[2], bgrid_temp[1]));
        lapidus = D(BG_LAPIDUS_G, M((double)nblyr, (double)nblyr));
        ivel[1] = D(dht, hbri_old);
        U_s[1] = M(D(ivel[1], dt), dts);
        Ui_s[1] = U_s[1];
        F_s[1] = D(D(M(M(D(M(brine_rho[2], brine_sal[2]), BG_RHOSI), darcy_V), dts), hbri_old), bSin[1]);
        ivel[NB1] = D(dhb, hbri_old);
        Ui_s[NB1] = M(D(ivel[NB1], dt), dts);
        dSbdx[nblyr] = D(S(M(ibrine_sal[NB1], ibrine_rho[NB1]), M(ibrine_sal[nblyr], ibrine_rho[nblyr])),
                         S(igrid[NB1], igrid[nblyr]));
        C_s[NB1] = D(M(D(M(D(D(BG_DM, brine_sal[NB1]), brine_rho[NB1]), dts), h2),
                       S(M(ibrine_sal[NB1], ibrine_rho[NB1]), M(ibrine_sal[nblyr], ibrine_rho[nblyr]))),
                     S(igrid[NB1], igrid[nblyr]));
        F_s[NB1] = D(D(M(darcy_V, dts), hbri_old), bphin[NB1]);
        F_s[NB2] = D(D(M(darcy_V, dts), hbri_old), bphin[NB2]);
        vel[nblyr] = D(S(M(bgrid[NB1], dhb), M(S(bgrid[NB1], 1.0), dht)), hbri_old);
        U_s[NB1] = M(D(vel[nblyr], dt), dts);
        V_s[NB1] = M(M(M(D(Din_p[NB1], BG_RHOSI), bg_powi3(D(D(BG_RHOSI, brine_sal[NB1]), brine_rho[NB1]))), dts),
                     dSbdx[nblyr]);
        const double bot = A(S(bgrid[NB2], bgrid[NB1]), D(zp.grid_oS, hbri_old));
        dSbdx[NB1] = D(S(M(brine_sal[NB2], brine_rho[NB2]), M(brine_sal[NB1], brine_rho[NB1])), bot);
        C_s[NB2] = D(M(D(M(D(D(BG_DM, brine_sal[NB2]), brine_rho[NB2]), dts), h2),
                       S(M(brine_sal[NB2], brine_rho[NB2]), M(brine_sal[NB1], brine_rho[NB1]))), bot);
        U_s[NB2] = M(D(ivel[NB1], dt), dts);
        V_s[NB2] = M(M(M(D(Din_p[NB2], BG_RHOSI), bg_powi3(D(bphin[NB1], bSin[NB2]))), dts), dSbdx[NB1]);
        dh = M(D(S(dhb, dht), dt), dts);
        for (int k = 2; k <= nblyr; ++k) {
            ivel[k] = D(S(M(igrid[k], dhb), M(S(igrid[k], 1.0), dht)), hbri_old);
            Ui_s[k] = M(D(ivel[k], dt), dts);
            dSbdx[k - 1] = D(S(M(ibrine_sal[k], ibrine_rho[k]), M(ibrine_sal[k - 1], ibrine_rho[k - 1])),
                             S(igrid[k], igrid[k - 1]));
            F_s[k] = D(D(M(darcy_V, dts), hbri_old), bphin[k]);
            C_s[k] = D(M(D(M(D(D(BG_DM, brine_sal[k]), brine_rho[k]), dts), h2),
                         S(M(ibrine_sal[k], ibrine_rho[k]), M(ibrine_sal[k - 1], ibrine_rho[k - 1]))),
                       S(igrid[k], igrid[k - 1]));
            vel[k - 1] = D(S(M(bgrid[k], dhb), M(S(bgrid[k], 1.0), dht)), hbri_old);
            U_s[k] = M(D(vel[k - 1], dt), dts);
            V_s[k] = M(M(M(D(Din_p[k], BG_RHOSI), bg_powi3(D(D(BG_RHOSI, brine_sal[k]), brine_rho[k]))), dts),
                       dSbdx[k - 1]);
            C_s[2] = 0.0;
            V_s[2] = 0.0;
        }
        for (int m = 1; m <= nint; ++m) {
            for (int k = 1; k <= NB2; ++k) { Sintemp[k] = bSin[k]; pre_sin[k] = bSin[k]; pre_sinb[k] = bSin[k]; }
            double Ssum_old = M(bSin[NB1], S(igrid[NB1], igrid[nblyr]));
            for (int k = 2; k <= nblyr; ++k) {
                Ssum_old = A(Ssum_old, M(bSin[k], S(igrid[k], igrid[k - 1])));
                pre_sin[k] = A(bSin[k],
                               D(S(A(S(A(M(Ui_s[k], S(bSin[k + 1], bSin[k])), M(V_s[k + 1], bg_powi3(bSin[k + 1]))),
                                       M(V_s[k], bg_powi3(bSin[k]))),
                                     M(A(C_s[k + 1], F_s[k + 1]), bSin[k + 1])),
                                   M(A(C_s[k], F_s[k]), bSin[k])),
                                 S(bgrid_temp[k + 1], bgrid_temp[k])));
            }
            pre_sin[NB1] = A(bSin[NB1],
                             D(S(A(S(A(M(Ui_s[NB1], S(bSin[NB2], bSin[NB1])), M(V_s[NB2], bg_powi3(bSin[NB2]))),
                                     M(V_s[NB1], bg_powi3(bSin[NB1]))),
                                   M(A(C_s[NB2], F_s[NB2]), bSin[NB2])),
                                 M(A(C_s[NB1], F_s[NB1]), bSin[NB1])),
                               S(bgrid_temp[NB2], bgrid_temp[NB1])));
            pre_sinb[2] = M(0.5, A(A(bSin[2], pre_sin[2]),
                                   D(S(A(S(A(M(Ui_s[1], S(pre_sin[2], pre_sin[1])), M(V_s[2], bg_powi3(pre_sin[2]))),
                                           M(V_s[1], bg_powi3(pre_sin[1]))),
                                         M(A(C_s[2], F_s[2]), pre_sin[2])),
                                       M(A(C_s[1], F_s[1]), pre_sin[1])),
                                     S(bgrid_temp[2], bgrid_temp[1]))));
            for (int k = NB1; k >= 3; --k) {
                pre_sinb[k] = M(0.5, A(A(bSin[k], pre_sin[k]),
                                       D(S(A(S(A(M(Ui_s[k - 1], S(pre_sin[k], pre_sin[k - 1])),
                                                 M(V_s[k], bg_powi3(pre_sin[k]))),
                                               M(V_s[k - 1], bg_powi3(pre_sin[k - 1]))),
                                             M(A(C_s[k], F_s[k]), pre_sin[k])),
                                           M(A(C_s[k - 1], F_s[k - 1]), pre_sin[k - 1])),
                                         S(bgrid_temp[k], bgrid_temp[k - 1]))));
                Q_s[k] = A(A(A(M(V_s[k], M(pre_sin[k], pre_sin[k])), U_s[k]), C_s[k]), F_s[k]);
            }
            Q_s[2] = A(A(A(M(V_s[2], M(pre_sin[2], pre_sin[2])), U_s[2]), C_s[2]), F_s[2]);
            Q_s[1] = A(A(A(M(V_s[1], M(pre_sin[2], pre_sin[2])), Ui_s[1]), C_s[1]), F_s[1]);
            Q_s[NB2] = A(A(A(M(V_s[NB2], M(pre_sin[NB1], pre_sin[NB1])), Ui_s[NB1]), C_s[NB2]), F_s[NB2]);
            double Ssum_new = 0.0;
            for (int k = 2; k <= NB1; ++k) {
                const double gp = S(bgrid_temp[k + 1], bgrid_temp[k]), gm = S(bgrid_temp[k], bgrid_temp[k - 1]);
                lapidus_diff[k - 1] = M(D(lapidus, S(igrid[k], igrid[k - 1])),
                                        S(D(M(M(lapA[k - 1], fabs(S(Q_s[k + 1], Q_s[k]))),
                                              S(fabs(pre_sinb[k + 1]), fabs(pre_sinb[k]))), M(gp, gp)),
                                          D(M(M(lapB[k - 1], fabs(S(Q_s[k], Q_s[k - 1]))),
                                              S(fabs(pre_sinb[k]), fabs(pre_sinb[k - 1]))), M(gm, gm))));
                bSin[k] = A(pre_sinb[k], lapidus_diff[k - 1]);
                if (bSin[k] < MIN_SALIN) {
                    bSin[k] = MIN_SALIN;
                } else if (bSin[k] > -D(bTin[k], DEPRESST)) {
                    bSin[k] = -D(bTin[k], DEPRESST);
                } else if (bSin[k] > BG_MAX_SALIN) {
                    l_stop = BG_MAXSALIN_STOP;
                }
                if (k == NB1) bSin[NB2] = A(M(S_bot[1], bSin[NB1]), M(S_bot[2], bSin[NB2]));
                Ssum_new = A(Ssum_new, M(bSin[k], S(igrid[k], igrid[k - 1])));
            }
            // calc_salt_fluxes
            const double dhtmp = S(1.0, D(dh, hbri_old));
            const double hin_next = A(hbri_old, M((double)m, dh));
            const double hin_old = A(hbri_old, M(S((double)m, 1.0), dh));
            const double s3 = bg_powi3(Sintemp[NB1]);
            const double fluxg_b = M(0.5, A(M(M(M(dhtmp, dbgrid), V_s[NB2]), s3),
                                            M(V_s[NB1], S(bg_powi3(pre_sin[NB1]), M(M(dhtmp, S(dbgrid, 1.0)), s3)))));
            *fzsaln_g = S(*fzsaln_g, D(M(M(M(hin_next, fluxg_b), BG_RHOSI), 0.001), dts));
            *fzsaln = A(*fzsaln, D(M(M(S(M(Ssum_old, hin_old), M(Ssum_new, hin_next)), BG_RHOSI), 0.001), dts));
        }
    } else {                                          // add/melt ice only
        double sum_old = 0.0, sum_new = 0.0;
        const double dh_dt = S(hbrin, hbri_old);
        double dS_dt = 0.0;
        if (dh_dt > 0.0) {
            dS_dt = M(M(sss, dh_dt), BG_SALT_LOSS);
            for (int k = 2; k <= NB1; ++k) bSin[k] = col_max(MIN_SALIN, D(A(M(bSin[k], hbri_old), dS_dt), hbrin));
        } else {
            for (int k = 2; k <= NB1; ++k) {
                sum_old = A(sum_old, M(M(bSin[k], hbri_old), S(igrid[k], igrid[k - 1])));
                bSin[k] = col_max(MIN_SALIN, M(bSin[k], S(1.0, D(fabs(dh_dt), hbri_old))));
                sum_new = A(sum_new, M(M(bSin[k], hbrin), S(igrid[k], igrid[k - 1])));
            }
        }
        *fzsaln = D(M(M(BG_RHOSI, A(S(sum_old, sum_new), dS_dt)), 0.001), dt);
        *fzsaln_g = 0.0;
    }
    for (int k = 1; k <= NB1; ++k) { iDin[k] = 0.0; iphin[k] = 1.0; ikin[k] = 0.0; }
    for (int k = 1; k <= NB1; ++k) {
        if (k < NB1)
            bphin[k + 1] = col_min(1.0, col_max(PUNY, D(M(bSin[k + 1], BG_RHOSI), M(brine_sal[k + 1], brine_rho[k + 1]))));
        if (k == 1) {
            bphin[1] = col_min(1.0, col_max(PUNY, D(M(bSin[1], BG_RHOSI), M(brine_sal[1], brine_rho[1]))));
            iphin[1] = bphin[2];
        } else if (k == NB1) {
            iphin[NB1] = bphin[NB1];
        } else {
            iphin[k] = col_min(1.0, col_max(0.0, A(M(D(S(bphin[k + 1], bphin[k]), S(bgrid[k + 1], bgrid[k])),
                                                     S(igrid[k], bgrid[k])), bphin[k])));
        }
        ikin[k] = M(BG_K_O, bg_powi3(iphin[k]));
    }
    for (int k = 2; k <= NB1; ++k) {
        iDin[k] = D(M(iphin[k], BG_DM), h2);
        if (cflag && *Rayleigh && hbrin >= BG_RA_C)
            iDin[k] = A(iDin[k], D(M(D(M(M(zp.l_sk, ikin[k]), BG_GRAVIT), BG_VISCOS_DYNAMIC), drho[k]), h2));
    }
    *cflag_out = cflag;
    return l_stop;
}

// zsalinity -> solve_zsalinity (ice_zsalinity.F90:38-425), with
// column_sum_zsal and merge_zsal_fluxes. n is the 0-based category; zsal,
// qice and sice the category's tracers; bSin, bTin, bphin, brine_sal and
// brine_rho (1..NB2) and iphin, ibrine_sal and ibrine_rho (1..NB1) what
// compute_microS and update_hbrine left. aicen_init is merge_zsal_fluxes'
// weight, as colpkg_biogeochemistry passes it.
__device__ int zsalinity(int n, double dt, const double *cgrid0, const double *bgrid0, const double *bgrid,
                         const double *igrid, double *zsal, const double *qice, double *sice, double fbri,
                         double *bSin, double *bTin, double *bphin, double *iphin, double *ikin, double hbr_old,
                         double hbrin, double hin, double hin_old, double *iDin, double darcy_V, double *brine_sal,
                         double *brine_rho, const double *ibrine_sal, const double *ibrine_rho, double dh_direct,
                         int *Rayleigh_criteria, int first_ice, double sss, double sst, double dh_top, double dh_bot,
                         double *fzsal, double *fzsal_g, double bphi_min, double vicen, double aicen_init,
                         double *zsal_tot, const zsal_params &zp)
{
    double fzsaln, fzsaln_g, S_bot[3];
    int stop;
    {   // solve_zsalinity
        double dts = zp.dts_b;
        int nint = (int)D(dt, dts);
        if (nint < 1) nint = 1;
        dts = D(dt, (double)nint);
        const double surface_S = MIN_SALIN;
        int Rayleigh = 1;
        if (n == 0 && hbr_old < BG_RA_C) Rayleigh = *Rayleigh_criteria;
        if (A(dh_bot, M(darcy_V, dt)) > 0.0) {
            bSin[NB2] = sss;
            bTin[NB2] = sst;
            brine_sal[NB2] = sss;
            brine_rho[NB2] = RHOW;
            bphin[NB2] = 1.0;
            S_bot[1] = 0.0;
            S_bot[2] = 1.0;
        } else {
            bSin[NB2] = bSin[NB1];
            bTin[NB2] = bTin[NB1];
            bphin[NB2] = iphin[NB1];
            S_bot[1] = 1.0;
            S_bot[2] = 0.0;
        }
        if (fabs(dh_top) > PUNY && fabs(darcy_V) > PUNY) {
            const double inner = A(S(M(M(D(M(brine_rho[2], brine_sal[2]), BG_RHOSI), darcy_V), dt),
                                     M(S(A(dh_top, D(M(darcy_V, dt), bphi_min)), dh_direct), MIN_SALIN)),
                                   M(col_max(0.0, -dh_direct), sss));
            bSin[1] = col_max(MIN_SALIN, -D(inner, dh_top));
            brine_sal[1] = brine_sal[2];
            brine_rho[1] = brine_rho[2];
            bphin[1] = bphi_min;
        } else {
            bSin[1] = MIN_SALIN;
        }
        int cflag;
        stop = solve_S_dt(&cflag, nint, dts, bSin, bTin, bphin, iphin, igrid, bgrid, ikin, hbr_old, hbrin, hin_old,
                          iDin, darcy_V, brine_sal, &Rayleigh, first_ice, sss, dt, dh_top, dh_bot, brine_rho,
                          ibrine_sal, ibrine_rho, &fzsaln, &fzsaln_g, S_bot, zp);
        if (!stop) {
            if (n == 0) *Rayleigh_criteria = Rayleigh;
            double trtmp[COL_NILYR + 1];
            for (int k = 0; k <= COL_NILYR; ++k) trtmp[k] = 0.0;
            for (int k = 1; k <= COL_NBLYR; ++k) zsal[k - 1] = bSin[k + 1];
            // remap_zbgc's stop here is not checked: the salinities are updated from
            // the untouched zeros and colpkg_biogeochemistry returns on it
            stop = remap_zbgc(zsal, 1, COL_NILYR, COL_NBLYR, hin, hbrin, cgrid0, bgrid0, surface_S, trtmp);
            const double Tmax = -M(A(MIN_SALIN, PUNY), DEPRESST);
            for (int k = 0; k < COL_NILYR; ++k) {
                const double Ttemp = col_min(Tmax, bg_Tin_from_qin(qice[k], -M(sice[k], DEPRESST)));
                sice[k] = col_min(-D(Ttemp, DEPRESST), col_max(MIN_SALIN, trtmp[k + 1]));
            }
        }
    }
    // column_sum_zsal and merge_zsal_fluxes (ice_zsalinity.F90:1111-1177)
    double zsal_totn = 0.0;
    for (int k = 0; k < COL_NBLYR; ++k)
        zsal_totn = A(zsal_totn, D(M(M(M(BG_RHOSI, zsal[k]), fbri), vicen), (double)COL_NBLYR));
    *zsal_tot = A(*zsal_tot, zsal_totn);
    *fzsal = A(*fzsal, M(fzsaln, aicen_init));
    *fzsal_g = A(*fzsal_g, M(fzsaln_g, aicen_init));
    return stop;
}

// ---------------------------------------------------------------------------
// the vertical tracers (z_tracers): zbio with the nonreactive tracer alone
// (nbtrcr = 1, PON; no vertical biochemistry, z-aerosols or dust iron)
// ---------------------------------------------------------------------------

#define ZB_HS_MIN 1.e-4            // ice_constants_colpkg hs_min
#define ZB_KSCAVZ 0.03             // ice_zbgc_shared kscavz(1)
#define ZB_N_VOL 0.04e-12          // z_biogeochemistry N_vol
#define ZB_NG_TO_MMOL 0.0140067    // z_biogeochemistry Ng_to_mmol
#define BG_ZBGC_INIT 4             // zbgc initialization error
#define BG_NEGATIVE_BGC 5          // negative bgc
#define BG_FCT_POSITIVITY 6        // check_conservation_FCT's l_stop (no stop label of its own)
// zpar: colpkg_init_zbgc's per-tracer parameters, grid_o and z_biogeochemistry's
// folded constants (f_v is the default-real literal 0.7854 widened)
#define ZP_TYPE 0
#define ZP_TAU_RET 1
#define ZP_TAU_REL 2
#define ZP_FRAC_INIT 3
#define ZP_INIT_FRAC 4
#define ZP_GRID_O 5
#define ZP_F_V 6
#define ZP_V_C 7
#define ZP_S_COL 8
#define ZP_P_B 9
#define ZP_V_ALG 10

// ice_aerosol.F90 update_snow_bgc for one tracer, nslyr = 1. trc is the
// tracer's nblyr + 3 layers (trcrn(bio_index:)): the snow's at trc[NB1] and
// trc[NB1 + 1]. The conservation and negative-aerosol warnings are not kept.
__device__ void update_snow_bgc(double dt, double melts, double snoice, double fsnow, double *trc,
                                double aice_old, double vsno_old, double vsnon, double aicen,
                                double flux_bio_atm, double *zbgc_snow, double *zbgc_atm, double *flux_bio,
                                int new_category)
{
    double a1 = 0.0, a2 = 0.0;
    *zbgc_snow = 0.0;
    *zbgc_atm = 0.0;
    // aice_old = 0 (ice new to the category) is unguarded upstream: with the vertical
    // thermodynamics the step never reaches it (aicen > puny there implies aicen_init >
    // puny), but without them aicen_init is never refreshed, so a category the last
    // step's ITD opened gives 0/0 here and a NaN flux; new_category (the opt-in
    // correction bgc_snow_new_category) takes such a category to have had no snow
    double hs_old = new_category && aice_old <= PUNY ? 0.0 : D(vsno_old, aice_old);
    double hslyr_old = D(hs_old, 1.0);
    double dzssl = col_min(D(hslyr_old, 2.0), ZB_HS_SSL);
    double dzint = S(hs_old, dzssl);
    double ar, hs;
    if (aicen > 0.0) {
        ar = D(1.0, aicen);
        hs = M(vsnon, ar);
    } else {
        ar = 1.0;
        hs = D(vsnon, aice_old);
    }
    const double dhs_melts = -melts;
    const double dhs_snoice = D(M(snoice, RHOI), RHOS);
    const double dhs_evap = S(hs, A(S(A(hs_old, dhs_melts), dhs_snoice), M(D(fsnow, RHOS), dt)));
    double *s1 = trc + NB1, *s2 = trc + NB1 + 1;
    if (vsno_old <= PUNY || vsnon <= PUNY) {
        *flux_bio = A(*flux_bio, D(A(M(*s1, dzssl), M(*s2, dzint)), dt));
        *s1 = 0.0;
        *s2 = 0.0;
        *zbgc_atm = A(*zbgc_atm, M(flux_bio_atm, dt));
        return;
    }
    a1 = M(*s1, dzssl);
    a2 = M(*s2, dzint);
    dzint = A(dzint, col_min(A(dzssl, dhs_evap), 0.0));
    dzssl = col_max(A(dzssl, dhs_evap), 0.0);
    if (-dhs_melts > PUNY) {
        double sloss1 = 0.0, sloss2 = 0.0;
        if (dzssl > PUNY) sloss1 = D(M(M(ZB_KSCAVZ, a1), col_min(-dhs_melts, dzssl)), dzssl);
        a1 = S(a1, sloss1);
        if (dzint > PUNY) sloss2 = D(M(M(ZB_KSCAVZ, a2), col_max(S(-dhs_melts, dzssl), 0.0)), dzint);
        a2 = S(a2, sloss2);
        *zbgc_snow = A(*zbgc_snow, A(sloss1, sloss2));
        dzint = A(dzint, col_min(A(dzssl, dhs_melts), 0.0));
        dzssl = col_max(A(dzssl, dhs_melts), 0.0);
        if (dzssl <= PUNY) {
            a2 = A(a1, a2);
            a1 = 0.0;
            dzssl = col_max(dzssl, 0.0);
        }
        if (dzint <= PUNY) {
            *zbgc_snow = A(*zbgc_snow, col_max(0.0, A(a1, a2)));
            a1 = 0.0;
            a2 = 0.0;
            dzint = col_max(dzint, 0.0);
        }
    }
    if (fsnow > 0.0) dzssl = A(dzssl, M(D(fsnow, RHOS), dt));
    if (dhs_snoice > PUNY) {
        double sloss1 = 0.0, sloss2 = 0.0;
        if (dzint > PUNY) sloss2 = D(M(col_min(dhs_snoice, dzint), a2), dzint);
        a2 = S(a2, sloss2);
        if (dzssl > PUNY) sloss1 = D(M(col_max(S(dhs_snoice, dzint), 0.0), a1), dzssl);
        a1 = S(a1, sloss1);
        *zbgc_snow = A(*zbgc_snow, A(sloss2, sloss1));
        dzssl = S(dzssl, col_max(S(dhs_snoice, dzint), 0.0));
        dzint = col_max(S(dzint, dhs_snoice), 0.0);
    }
    hs = aicen > 0.0 ? M(vsnon, ar) : 0.0;
    if (hs >= ZB_HS_MIN) a1 = A(a1, M(flux_bio_atm, dt));
    else *zbgc_atm = A(*zbgc_atm, M(flux_bio_atm, dt));
    if (dzssl <= PUNY) {
        a2 = A(a2, a1);
        a1 = 0.0;
    }
    if (dzint <= PUNY) {
        *zbgc_snow = A(*zbgc_snow, col_max(0.0, a2));
        a2 = 0.0;
    }
    const double hslyr = D(hs, 1.0);
    const double dzssl_new = col_min(D(hslyr, 2.0), ZB_HS_SSL);
    const double dzint_new = S(hs, dzssl_new);
    if (hs > ZB_HS_MIN) {
        double dznew = col_min(S(dzssl_new, dzssl), 0.0);
        double sloss1 = 0.0;
        if (dzssl > PUNY) sloss1 = D(M(dznew, a1), dzssl);
        dznew = col_max(S(dzssl_new, dzssl), 0.0);
        if (dzint > PUNY) sloss1 = A(sloss1, D(M(a2, dznew), dzint));
        a1 = A(a1, sloss1);
        a2 = S(a2, sloss1);
    } else {
        *zbgc_snow = A(*zbgc_snow, col_max(0.0, A(a1, a2)));
        a1 = 0.0;
        a2 = 0.0;
    }
    if (vsnon > PUNY) {
        *s1 = D(a1, dzssl_new);
        *s2 = D(a2, dzint_new);
    } else {
        *zbgc_snow = A(A(*zbgc_snow, a1), a2);
        *s1 = 0.0;
        *s2 = 0.0;
    }
    if (a1 < -PUNY || a2 < -PUNY) {
        *s1 = col_max(*s1, 0.0);
        *s2 = col_max(*s2, 0.0);
    }
}

// ice_algae.F90 compute_FCT_matrix (1-based arrays; its top and bottom sources
// and sinks feed only check_conservation_FCT's warnings and are dropped)
__device__ void compute_FCT_matrix(const double *C_in, double dt, const double *bgrid, double darcyV, double dhtop,
                                   double dhbot, const double *iphin_N, const double *iDin, double hbri_old,
                                   double atm_add, const double *bphin_N, double C_top, double C_bot, double grid_o,
                                   double *sbdiag, double *diag, double *spdiag, double *rhs, double *D_sbdiag,
                                   double *D_spdiag, double *ML)
{
    const int nb = COL_NBLYR;
    double kvector1[NB1], kvectorn1[NB1];
    double K_diag[NB1 + 1], K_sp[NB1 + 1], K_sb[NB1 + 1], S_diag[NB1 + 1], S_sp[NB1 + 1], S_sb[NB1 + 1];
    double D_diag[NB1 + 1], iDin_phi[NB1 + 1], Q_top[NB1 + 1], Q_bot[NB1 + 1];
    for (int k = 1; k <= nb; ++k) {
        kvector1[k] = 0.0;
        kvectorn1[k] = 1.0;
    }
    kvector1[1] = 1.0;
    kvectorn1[1] = 0.0;
    const double zspace = D(1.0, (double)nb);
    for (int k = 1; k <= NB1; ++k) {
        ML[k] = zspace;
        K_diag[k] = 0.0; D_diag[k] = 0.0; D_spdiag[k] = 0.0; D_sbdiag[k] = 0.0; K_sp[k] = 0.0; K_sb[k] = 0.0;
        S_diag[k] = 0.0; S_sp[k] = 0.0; S_sb[k] = 0.0; iDin_phi[k] = 0.0;
    }
    ML[1] = D(zspace, 2.0);
    ML[NB1] = D(zspace, 2.0);
    iDin_phi[1] = 0.0;
    iDin_phi[NB1] = D(iDin[NB1], iphin_N[NB1]);
    iDin_phi[nb] = M(0.5, A(D(iDin[NB1], iphin_N[NB1]), D(iDin[nb], iphin_N[nb])));
    double vel = A(D(S(M(bgrid[2], dhbot), M(S(bgrid[2], 1.0), dhtop)), dt), D(darcyV, bphin_N[2]));
    K_diag[1] = D(M(0.5, vel), hbri_old);
    double dphi_dx = D(S(iphin_N[NB1], iphin_N[nb]), zspace);
    vel = A(D(S(M(bgrid[NB1], dhbot), M(S(bgrid[NB1], 1.0), dhtop)), dt), D(darcyV, bphin_N[NB1]));
    vel = D(vel, hbri_old);
    double vel2 = A(D(D(dhbot, hbri_old), dt), D(darcyV, hbri_old));
    K_diag[NB1] = A(S(col_min(0.0, vel2), D(iDin_phi[NB1], A(zspace, D(grid_o, hbri_old)))),
                    M(0.5, A(-vel, M(D(iDin_phi[nb], bphin_N[NB1]), dphi_dx))));
    for (int k = 1; k <= nb - 1; ++k) {
        vel = A(D(S(M(bgrid[k + 1], dhbot), M(S(bgrid[k + 1], 1.0), dhtop)), dt), D(darcyV, bphin_N[k + 1]));
        iDin_phi[k + 1] = M(0.5, A(D(iDin[k + 2], iphin_N[k + 2]), D(iDin[k + 1], iphin_N[k + 1])));
        dphi_dx = D(S(iphin_N[k + 1], iphin_N[k]), zspace);
        K_sp[k] = M(0.5, S(D(vel, hbri_old), M(D(iDin_phi[k], bphin_N[k + 1]), dphi_dx)));
        vel = A(D(S(M(bgrid[k + 1], dhbot), M(S(bgrid[k + 1], 1.0), dhtop)), dt), D(darcyV, bphin_N[k + 1]));
        dphi_dx = 0.0;
        dphi_dx = D(M(kvectorn1[k], S(iphin_N[k + 1], iphin_N[k])), zspace);
        K_sb[k + 1] = -M(0.5, S(D(vel, hbri_old), M(D(iDin_phi[k], bphin_N[k + 1]), dphi_dx)));
        K_diag[k] = A(M(K_diag[1], kvector1[k]), M(A(K_sp[k], K_sb[k]), kvectorn1[k]));
        S_diag[k + 1] = -D(A(iDin_phi[k], iDin_phi[k + 1]), zspace);
        S_sp[k] = D(iDin_phi[k], zspace);
        S_sb[k + 1] = D(iDin_phi[k], zspace);
    }
    vel = A(D(S(M(bgrid[NB1], dhbot), M(S(bgrid[NB1], 1.0), dhtop)), dt), D(darcyV, bphin_N[NB1]));
    dphi_dx = D(S(iphin_N[NB1], iphin_N[nb]), zspace);
    K_sp[nb] = M(0.5, S(D(vel, hbri_old), M(D(iDin_phi[nb], bphin_N[NB1]), dphi_dx)));
    vel = A(D(S(M(bgrid[NB1], dhbot), M(S(bgrid[NB1], 1.0), dhtop)), dt), D(darcyV, bphin_N[NB1]));
    dphi_dx = D(M(kvectorn1[nb], S(iphin_N[NB1], iphin_N[nb])), zspace);
    K_sb[NB1] = -M(0.5, S(D(vel, hbri_old), M(D(iDin_phi[nb], bphin_N[NB1]), dphi_dx)));
    K_diag[nb] = A(K_sp[nb], K_sb[nb]);
    S_diag[NB1] = -D(iDin_phi[nb], zspace);
    S_sp[nb] = D(iDin_phi[nb], zspace);
    S_sb[NB1] = D(iDin_phi[nb], zspace);
    for (int k = 1; k <= nb; ++k) {
        D_spdiag[k] = col_max(col_max(-K_sp[k], 0.0), -K_sb[k + 1]);
        D_sbdiag[k + 1] = D_spdiag[k];
    }
    for (int k = 1; k <= NB1; ++k) D_diag[k] = S(S(D_diag[k], D_spdiag[k]), D_sbdiag[k]);
    vel2 = -A(D(D(dhtop, hbri_old), dt), D(D(darcyV, bphin_N[1]), hbri_old));
    for (int k = 1; k <= NB1; ++k) { Q_top[k] = 0.0; Q_bot[k] = 0.0; }
    Q_top[1] = col_max(0.0, A(M(vel2, C_top), D(atm_add, dt)));
    vel = A(D(D(dhbot, hbri_old), dt), D(darcyV, hbri_old));
    Q_bot[NB1] = A(col_max(0.0, M(vel, C_bot)), D(M(iDin_phi[NB1], C_bot), A(zspace, D(grid_o, hbri_old))));
    for (int k = 1; k <= NB1; ++k) {
        spdiag[k] = -M(dt, A(A(D_spdiag[k], K_sp[k]), S_sp[k]));
        sbdiag[k] = -M(dt, A(A(D_sbdiag[k], K_sb[k]), S_sb[k]));
        diag[k] = S(ML[k], M(dt, A(A(D_diag[k], K_diag[k]), S_diag[k])));
        rhs[k] = A(A(M(ML[k], C_in[k]), M(dt, Q_top[k])), M(dt, Q_bot[k]));
    }
}

// ice_algae.F90 tridiag_solverz, nmat = nblyr + 1 (1-based)
__device__ void tridiag_solverz(const double *sbdiag, const double *diag, const double *spdiag, const double *rhs,
                                double *xout)
{
    double wgamma[NB1 + 1];
    double wbeta = diag[1];
    xout[1] = D(rhs[1], wbeta);
    for (int k = 2; k <= NB1; ++k) {
        wgamma[k] = D(spdiag[k - 1], wbeta);
        wbeta = S(diag[k], M(sbdiag[k], wgamma[k]));
        xout[k] = D(S(rhs[k], M(sbdiag[k], xout[k - 1])), wbeta);
    }
    for (int k = NB1 - 1; k >= 1; --k) xout[k] = S(xout[k], M(wgamma[k + 1], xout[k + 1]));
}

// ice_algae.F90 compute_FCT_corr (1-based)
__device__ void compute_FCT_corr(const double *C_in, double *C_low, double dt, const double *D_sbdiag,
                                 const double *D_spdiag, const double *ML)
{
    const int nb = COL_NBLYR;
    const double zspace = D(1.0, (double)nb);
    double M_sp[NB1 + 1], M_sb[NB1 + 1], F_diag[NB1 + 1], F_sp[NB1 + 1], F_sb[NB1 + 1];
    double Pplus[NB1 + 1], Pminus[NB1 + 1], Qplus[NB1 + 1], Qminus[NB1 + 1], Rplus[NB1 + 1], Rminus[NB1 + 1];
    double a_sp[NB1 + 1], a_sb[NB1 + 1];
    for (int k = 1; k <= NB1; ++k) {
        M_sp[k] = D(zspace, 6.0);
        M_sb[k] = D(zspace, 6.0);
        F_diag[k] = 0.0;
        F_sp[k] = 0.0;
        F_sb[k] = 0.0;
    }
    M_sp[NB1] = 0.0;
    M_sb[1] = 0.0;
    for (int k = 1; k <= nb; ++k) {
        F_sp[k] = A(D(M(M_sp[k], A(S(S(C_low[k], C_in[k]), C_low[k + 1]), C_in[k + 1])), dt),
                    M(D_spdiag[k], S(C_low[k], C_low[k + 1])));
        F_sb[k + 1] = A(D(M(M_sb[k + 1], A(S(S(C_low[k + 1], C_in[k + 1]), C_low[k]), C_in[k])), dt),
                        M(D_sbdiag[k + 1], S(C_low[k + 1], C_low[k])));
        if (M(F_sp[k], S(C_low[k], C_low[k + 1])) > 0.0) F_sp[k] = 0.0;
        if (M(F_sb[k + 1], S(C_low[k + 1], C_low[k])) > 0.0) F_sb[k + 1] = 0.0;
    }
    int nonzero = 0;                                   // maxval(abs(F_spdiag)) > c0
    for (int k = 1; k <= NB1; ++k)
        if (fabs(F_sp[k]) > 0.0) nonzero = 1;
    if (!nonzero) return;
    for (int k = 1; k <= NB1; ++k) { a_sp[k] = 0.0; a_sb[k] = 0.0; }
    Pplus[1] = col_max(0.0, F_sp[1]);
    Pminus[1] = col_min(0.0, F_sp[1]);
    Pplus[NB1] = col_max(0.0, F_sb[NB1]);
    Pminus[NB1] = col_min(0.0, F_sb[NB1]);
    Qplus[1] = col_max(0.0, S(C_low[2], C_low[1]));
    Qminus[1] = col_min(0.0, S(C_low[2], C_low[1]));
    Qplus[NB1] = col_max(0.0, S(C_low[nb], C_low[NB1]));
    Qminus[NB1] = col_min(0.0, S(C_low[nb], C_low[NB1]));
    Rplus[1] = col_min(1.0, D(D(M(ML[1], Qplus[1]), dt), A(Pplus[1], PUNY)));
    Rminus[1] = col_min(1.0, D(D(M(ML[1], Qminus[1]), dt), S(Pminus[1], PUNY)));
    Rplus[NB1] = col_min(1.0, D(D(M(ML[NB1], Qplus[NB1]), dt), A(Pplus[NB1], PUNY)));
    Rminus[NB1] = col_min(1.0, D(D(M(ML[NB1], Qminus[NB1]), dt), S(Pminus[NB1], PUNY)));
    for (int k = 2; k <= nb; ++k) {
        Pplus[k] = A(col_max(0.0, F_sp[k]), col_max(0.0, F_sb[k]));
        Pminus[k] = A(col_min(0.0, F_sp[k]), col_min(0.0, F_sb[k]));
        Qplus[k] = col_max(0.0, col_max(S(C_low[k + 1], C_low[k]), S(C_low[k - 1], C_low[k])));
        Qminus[k] = col_min(0.0, col_min(S(C_low[k + 1], C_low[k]), S(C_low[k - 1], C_low[k])));
        Rplus[k] = col_min(1.0, D(D(M(ML[k], Qplus[k]), dt), A(Pplus[k], PUNY)));
        Rminus[k] = col_min(1.0, D(D(M(ML[k], Qminus[k]), dt), S(Pminus[k], PUNY)));
    }
    for (int k = 1; k <= nb; ++k) {
        a_sp[k] = col_min(Rminus[k], Rplus[k + 1]);
        if (F_sp[k] > 0.0) a_sp[k] = col_min(Rplus[k], Rminus[k + 1]);
        a_sb[k + 1] = col_min(Rminus[k + 1], Rplus[k]);
        if (F_sb[k + 1] > 0.0) a_sb[k + 1] = col_min(Rplus[k + 1], Rminus[k]);
    }
    F_diag[1] = M(a_sp[1], F_sp[1]);
    F_diag[NB1] = M(a_sb[NB1], F_sb[NB1]);
    C_low[1] = A(C_low[1], D(M(dt, F_diag[1]), ML[1]));
    C_low[NB1] = A(C_low[NB1], D(M(dt, F_diag[NB1]), ML[NB1]));
    for (int k = 2; k <= nb; ++k) {
        F_diag[k] = A(M(a_sp[k], F_sp[k]), M(a_sb[k], F_sb[k]));
        C_low[k] = A(C_low[k], D(M(dt, F_diag[k]), ML[k]));
    }
}

// ice_zbgc_shared.F90 regrid_stationary with the bottom melt and growth and
// its flux (1-based C, igrid); returns BG_REMAP on remap_zbgc's stop
__device__ int zb_regrid_stationary(double *C, double hbri_old, double hbri, double dt, double top_conc,
                                    const double *igrid, double *flux_bio, double meltb, double congel)
{
    double zsp[NB1 + 1], trtmp0[NB2 + 1], trtmp[NB2 + 1];
    for (int k = 1; k <= NB1; ++k) zsp[k] = D(1.0, (double)COL_NBLYR);
    zsp[1] = M(0.5, zsp[1]);
    zsp[NB1] = zsp[1];
    for (int k = 0; k <= NB2; ++k) { trtmp0[k] = 0.0; trtmp[k] = 0.0; }
    double sum_i = 0.0, sum_f = 0.0;
    for (int k = 1; k <= NB1; ++k) sum_i = A(sum_i, M(C[k], zsp[k]));
    if (hbri_old > 0.0)
        for (int k = 1; k <= NB1; ++k) trtmp0[NB2 - k] = D(C[k], hbri_old);    // reverse order
    double htemp = 0.0, hice = 0.0, hbio = 0.0;
    int nr = 0;
    if (meltb > 0.0) {
        htemp = S(hbri_old, meltb);
        nr = 0;
        hice = hbri_old;
        hbio = htemp;
    } else if (congel > 0.0) {
        htemp = A(hbri_old, congel);
        nr = 1;
        hice = htemp;
        hbio = hbri_old;
    } else if (hbri > hbri_old) {
        htemp = hbri;
        nr = 1;
        hice = htemp;
        hbio = hbri_old;
    }
    if (htemp > 0.0) {
        if (remap_zbgc(trtmp0 + 1, nr, NB1, NB1, hice, hbio, igrid, igrid, top_conc, trtmp)) return BG_REMAP;
        for (int k = 0; k <= NB2; ++k) trtmp0[k] = 0.0;
        for (int k = 1; k <= NB1; ++k) trtmp0[NB2 - k] = trtmp[k];
        for (int k = 1; k <= NB1; ++k) {
            C[k] = M(trtmp0[k], htemp);
            sum_f = A(sum_f, M(C[k], zsp[k]));
        }
        if (congel > 0.0 && top_conc <= 0.0 && fabs(S(sum_i, sum_f)) > PUNY) {
            const double dflux = S(sum_i, sum_f);
            sum_f = 0.0;
            for (int k = 1; k <= NB1; ++k) {
                C[k] = col_max(0.0, A(C[k], dflux));
                sum_f = A(sum_f, M(C[k], zsp[k]));
            }
        }
        *flux_bio = A(*flux_bio, D(S(sum_i, sum_f), dt));
    }
    return BG_OK;
}

// ice_algae.F90 thin_ice_flux (1-based Cin); it sets the flux
__device__ void thin_ice_flux(double hin, double hin_old, double *Cin, double *flux_o_tot, double source, double dt,
                              double ocean_bio)
{
    const double zspace = D(1.0, (double)COL_NBLYR);
    double dC = 0.0, sum_bio = 0.0;
    const double dh = S(hin, hin_old);
    if (dh <= 0.0) {
        sum_bio = M(M(D(A(Cin[1], Cin[NB1]), hin_old), zspace), 0.5);
        Cin[1] = M(D(Cin[1], hin_old), hin);
        Cin[NB1] = M(D(Cin[NB1], hin_old), hin);
        for (int k = 2; k <= COL_NBLYR; ++k) {
            sum_bio = A(sum_bio, M(D(Cin[k], hin_old), zspace));
            Cin[k] = A(M(D(Cin[k], hin_old), hin), dC);
        }
    } else {
        dC = M(dh, ocean_bio);
        for (int k = 1; k <= NB1; ++k) Cin[k] = A(Cin[k], dC);
    }
    *flux_o_tot = A(S(-D(M(dh, sum_bio), dt), D(dC, dt)), D(source, dt));
}

// gfortran's MAXVAL over bphin_N(2:nblyr+1): the first element not NaN, then strict >
__device__ double zb_maxval(const double *x, int lo, int hi)
{
    int k = lo;
    while (k <= hi && x[k] != x[k]) ++k;
    if (k > hi) return __longlong_as_double(0x7ff8000000000000LL);
    double r = x[k];
    for (++k; k <= hi; ++k)
        if (x[k] > r) r = x[k];
    return r;
}

// ice_algae.F90 z_biogeochemistry for one tracer without solve_zbgc: trc its
// nblyr + 3 layers, frac its mobile fraction, flux_bio zbio's flux_bion.
// bphin (nblyr + 2), iphin, iDin, bgrid, igrid and icgrid are 1-based; fswthrul
// and zfswin 0-based. Returns the stop (BG_*), 0 without.
__device__ int z_biogeochemistry(double dt, int first_ice, double aicen, double vicen, double hice_old,
                                 double ocean_bio, double *flux_bio, const double *bphin, const double *iphin,
                                 double *trc, double *frac, const double *iDin, const double *fswthrul,
                                 double *zfswin, double dh_top, double dh_bot, double hbri, double hbri_old,
                                 double darcy_V, const double *bgrid, const double *igrid, const double *icgrid,
                                 double bphi_min, double zbgc_snow, double zbgc_atm, double dh_direct, double meltb,
                                 double congel, const double *zp)
{
    const double zspace = D(1.0, (double)COL_NBLYR);
    double bphin_N[NB2 + 1], iphin_N[NB1 + 1], in_init[NB1 + 1], cons[NB1 + 1];
    for (int k = 1; k <= NB1; ++k) {
        bphin_N[NB2] = 1.0;
        bphin_N[k] = bphin[k];
        iphin_N[k] = iphin[k];
        bphin_N[1] = bphi_min;
        double *t = trc + k - 1;
        if (first_ice) {
            *t = M(ocean_bio, zp[ZP_INIT_FRAC]);
            in_init[k] = M(*t, hbri_old);
        } else if (fabs(*t) < PUNY) {
            *t = 0.0;
            in_init[k] = 0.0;
        } else {
            in_init[k] = M(*t, hbri_old);
        }
        if (*t < 0.0) return BG_ZBGC_INIT;
    }
    const double hin = D(vicen, aicen);
    const double hin_old = hice_old;
    const double phi_max = zb_maxval(bphin_N, 2, NB1);
    // Sat_conc = f_s*f_a*f_v*(c1-phi_max)/V_c*S_col/P_b*N_vol*V_alg/Ng_to_mmol (f_s = f_a = 1)
    const double sat_conc = D(M(M(D(M(D(M(zp[ZP_F_V], S(1.0, phi_max)), zp[ZP_V_C]), zp[ZP_S_COL]), zp[ZP_P_B]),
                                  ZB_N_VOL), zp[ZP_V_ALG]), ZB_NG_TO_MMOL);
    // nlt_bgc_N(1) = 0 without algae: every tracer takes the first branch
    double rtau_rel, rtau_ret;
    if (hin_old > hin) {
        rtau_rel = D(1.0, zp[ZP_TAU_REL]);
        rtau_ret = 0.0;
    } else {
        rtau_ret = D(1.0, zp[ZP_TAU_RET]);
        rtau_rel = 0.0;
    }
    const double dhtop = dh_top, dhbot = dh_bot, darcyV = darcy_V;
    double C_top = M(in_init[1], *frac);
    const double added = A(A(zbgc_snow, zbgc_atm), 0.0);       // + dust_Fe(m) = 0
    const double source = fabs(added);
    const double dhflood = col_max(0.0, -dh_direct);
    double atm_add = 0.0;
    const double top = A(dhtop, M(D(darcyV, bphin_N[1]), dt));
    if (top < -PUNY)
        C_top = M(D(added, fabs(A(top, PUNY))), hbri_old);
    else if (top >= -PUNY && fabs(A(added, M(M(ocean_bio, bphin_N[1]), dhflood))) > PUNY)
        atm_add = A(fabs(added), M(M(ocean_bio, bphin_N[1]), dhflood));
    else
        atm_add = fabs(added);
    const double C_bot = M(M(ocean_bio, hbri_old), iphin_N[NB1]);
    // the shortwave onto the bio grid
    double trtmp[NB1 + 1];
    for (int k = 0; k <= NB1; ++k) trtmp[k] = 0.0;
    for (int k = 0; k < NB1; ++k) zfswin[k] = 0.0;
    if (remap_zbgc(fswthrul, 0, COL_NILYR + 1, NB1, hin, hbri, icgrid, igrid, 0.0, trtmp)) return BG_REMAP;
    for (int k = 1; k <= NB1; ++k) zfswin[k - 1] = trtmp[k];
    const double mobile = zp[ZP_TYPE] >= 0.0 ? 1.0 : 0.0;
    for (int k = 1; k <= NB1; ++k) cons[k] = in_init[k];
    if (hbri_old > BG_THINS && hbri > BG_THINS) {
        double icm[NB1 + 1], ics[NB1 + 1], initcons[NB1 + 1], biocons[NB1 + 1], low[NB1 + 1];
        double sbdiag[NB1 + 1], diag[NB1 + 1], spdiag[NB1 + 1], rhs[NB1 + 1];
        double D_sbdiag[NB1 + 1], D_spdiag[NB1 + 1], ML[NB1 + 1];
        for (int k = 1; k <= NB1; ++k) {
            icm[k] = M(in_init[k], *frac);
            ics[k] = M(mobile, S(in_init[k], icm[k]));
            const double dmobile = M(mobile, A(M(icm[k], S(glibc_exp(-M(dt, rtau_ret)), 1.0)),
                                               M(ics[k], S(1.0, glibc_exp(-M(dt, rtau_rel))))));
            icm[k] = col_max(0.0, A(icm[k], dmobile));
            ics[k] = col_max(0.0, S(ics[k], dmobile));
            if (D(ics[k], hbri_old) > sat_conc) {
                icm[k] = S(A(icm[k], ics[k]), M(sat_conc, hbri_old));
                ics[k] = M(sat_conc, hbri_old);
            }
            initcons[k] = icm[k];
            biocons[k] = icm[k];
        }
        compute_FCT_matrix(initcons, dt, bgrid, darcyV, dhtop, dhbot, iphin_N, iDin, hbri_old, atm_add, bphin_N,
                           C_top, C_bot, zp[ZP_GRID_O], sbdiag, diag, spdiag, rhs, D_sbdiag, D_spdiag, ML);
        tridiag_solverz(sbdiag, diag, spdiag, rhs, biocons);
        // check_conservation_FCT: the flux, the low order solution, and its stop
        // on a negative one (its conservation failure only warns)
        double c_init = M(M(A(initcons[1], initcons[NB1]), zspace), 0.5);
        double c_new = M(M(A(biocons[1], biocons[NB1]), zspace), 0.5);
        low[1] = biocons[1];
        low[NB1] = biocons[NB1];
        for (int k = 2; k <= COL_NBLYR; ++k) {
            c_init = A(c_init, M(initcons[k], zspace));
            c_new = A(c_new, M(biocons[k], zspace));
            low[k] = biocons[k];
        }
        *flux_bio = D(A(S(c_init, c_new), source), dt);
        for (int k = 1; k <= NB1; ++k)
            if (low[k] < 0.0) return BG_FCT_POSITIVITY;
        compute_FCT_corr(initcons, biocons, dt, D_sbdiag, D_spdiag, ML);
        int stop = zb_regrid_stationary(ics, hbri_old, hbri, dt, 0.0, igrid, flux_bio, meltb, congel);
        if (stop) return stop;
        for (int k = 1; k <= NB1; ++k) cons[k] = A(biocons[k], ics[k]);
        double sum_new = D(M(A(biocons[1], biocons[NB1]), zspace), 2.0);
        double sum_tot = D(M(A(cons[1], cons[NB1]), zspace), 2.0);
        for (int k = 2; k <= COL_NBLYR; ++k) {
            sum_new = A(sum_new, M(biocons[k], zspace));
            sum_tot = A(sum_tot, M(cons[k], zspace));
        }
        *frac = zp[ZP_FRAC_INIT];
        if (sum_tot > 0.0 && mobile > 0.0) *frac = D(sum_new, sum_tot);
        // the FCT failure (sum_old against sum_new, negative profiles) only warns
    } else {
        thin_ice_flux(hbri, hbri_old, cons, flux_bio, source, dt, ocean_bio);
    }
    for (int k = 1; k <= NB1; ++k) {
        double bio_tmp = M(A(D(D(cons[k], hbri), iphin_N[k]), 0.0), iphin_N[k]);    // + react = 0
        if (fabs(bio_tmp) < PUNY) bio_tmp = 0.0;
        else if (bio_tmp > 1.0e6) {}                                                 // a warning only
        else if (bio_tmp < 0.0) return BG_NEGATIVE_BGC;
        trc[k - 1] = col_max(0.0, bio_tmp);
        if (ocean_bio <= 0.0 && *flux_bio < 0.0) *flux_bio = col_max(0.0, *flux_bio);
    }
    return BG_OK;
}

// ice_zbgc.F90 merge_bgc_fluxes for one tracer without solve_zbgc
__device__ void merge_bgc_fluxes(double dt, double aicen, double vicen, double vsnon, const double *trc, double fbri,
                                 double flux_bion, double zbgc_snown, double zbgc_atmn, double *flux_bio,
                                 double *zbgc_snow, double *zbgc_atm, double *ice_bio_net, double *snow_bio_net)
{
    double zsp[NB1 + 1];
    for (int k = 1; k <= NB1; ++k) zsp[k] = D(1.0, (double)COL_NBLYR);
    zsp[1] = D(0.5, (double)COL_NBLYR);
    zsp[NB1] = D(0.5, (double)COL_NBLYR);
    for (int k = 1; k <= NB1; ++k) *ice_bio_net = A(*ice_bio_net, M(M(M(trc[k - 1], fbri), vicen), zsp[k]));
    const double dvssl = col_min(M(0.5, vsnon), M(ZB_HS_SSL, aicen));
    const double dvint = S(vsnon, dvssl);
    *snow_bio_net = A(A(*snow_bio_net, M(trc[NB1], dvssl)), M(trc[NB1 + 1], dvint));
    *flux_bio = A(*flux_bio, M(flux_bion, aicen));
    *zbgc_snow = A(*zbgc_snow, D(M(zbgc_snown, aicen), dt));
    *zbgc_atm = A(*zbgc_atm, D(M(zbgc_atmn, aicen), dt));
}

// column_biogeochemistry: one thread per owned cell. flags: bit 0 tr_brine,
// bit 1 solve_zsal, bit 2 z_tracers (with the brine height; the nonreactive
// tracer alone), bit 3 the correction bgc_snow_new_category. fbri is brineFraction and zsal_out verticalSalinity, ztr_conc
// verticalNonreactiveConc (nblyr + 3 per category) and ztr_frac mobileFraction
// (nzbgc per category) (or dummies without them).
extern "C" __global__ void seaice_column_biogeochemistry_v842(
    int *__restrict__ status, double dt, double l_sk, int ncat, int flags, int nCellsSolve,
    const double *__restrict__ bgrid0, const double *__restrict__ igrid0, const double *__restrict__ cgrid0,
    const double *__restrict__ aicen_init, const double *__restrict__ vicen_init,
    const double *__restrict__ aicen, const double *__restrict__ vicen, const double *__restrict__ vsnon,
    const double *__restrict__ meltb, const double *__restrict__ meltt, const double *__restrict__ congel,
    const double *__restrict__ snoice, const double *__restrict__ melts,
    const double *__restrict__ sst, const double *__restrict__ sss, const double *__restrict__ aice0,
    const double *__restrict__ qice, double *__restrict__ sice,
    double *__restrict__ fbri_out, double *__restrict__ hin_old, int *__restrict__ newly,
    double *__restrict__ rayleigh, double *__restrict__ dhbr_top, double *__restrict__ dhbr_bot,
    double *__restrict__ bTiz, double *__restrict__ bphi, double *__restrict__ iDi,
    double *__restrict__ darcy_V, double *__restrict__ sice_rho, double *__restrict__ hbri,
    double *__restrict__ atm_bc, double *__restrict__ atm_dust, double *__restrict__ total_skeletal,
    double *__restrict__ bio_sw_cell,
    double dts_b, double l_skS, double grid_oS, const double *__restrict__ tsfc, double *__restrict__ zsal_out,
    double *__restrict__ iki, double *__restrict__ zsal_tot, double *__restrict__ fzsal,
    double *__restrict__ fzsal_g,
    double *__restrict__ ztr_conc, double *__restrict__ ztr_frac, int nzbgc, const double *__restrict__ zpar,
    int bio_o, const double *__restrict__ ocean_conc, const double *__restrict__ fsnow,
    const double *__restrict__ vsno_init, const double *__restrict__ fswthrul, const double *__restrict__ icgrid0,
    double *__restrict__ zfswin, double *__restrict__ flux_bio, double *__restrict__ fbio_snoice,
    double *__restrict__ fbio_atmice, double *__restrict__ ice_bio_net, double *__restrict__ snow_bio_net,
    const double *__restrict__ flux_bio_atm)
{
    const int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const int tr_brine = flags & 1, solve_zsal = (flags >> 1) & 1, z_tracers = (flags >> 2) & 1;
    const int snow_new_category = (flags >> 3) & 1;
    const zsal_params zp = {dts_b, l_skS, grid_oS, l_sk};
    // the uncoupled build's deposition (#else of #ifdef coupled)
    atm_bc[iCell * 2 + 0] = 1.e-12;
    atm_bc[iCell * 2 + 1] = 1.e-12;
    for (int k = 0; k < 4; ++k) atm_dust[iCell * 4 + k] = 1.e-13;
    // 1-based grids
    double bgrid[NB2 + 1], igrid[NB1 + 1];
    for (int k = 1; k <= NB2; ++k) bgrid[k] = bgrid0[k - 1];
    for (int k = 1; k <= NB1; ++k) igrid[k] = igrid0[k - 1];
    double icgrid[COL_NILYR + 2];
    for (int k = 1; k <= COL_NILYR + 1; ++k) icgrid[k] = icgrid0[k - 1];
    // colpkg_biogeochemistry's iphin: compute_microS_mushy's, kept across categories
    double iphin_c[NB1 + 1];
    for (int k = 0; k <= NB1; ++k) iphin_c[k] = 0.0;
    const double ocean_bio = z_tracers ? ocean_conc[(size_t)iCell * nzbgc + bio_o] : 0.0;
    int first_ice[64];
    double fbri[64];
    for (int n = 0; n < ncat; ++n) {
        first_ice[n] = newly[iCell * ncat + n] == 1;
        fbri[n] = tr_brine ? fbri_out[iCell * ncat + n] : 0.0;
    }
    int Rayleigh = rayleigh[iCell] > 0.5;
    double hbri_c = hbri[iCell];
    int l_stop = BG_OK;
    for (int n = 0; n < ncat; ++n) {
        const int c = iCell * ncat + n;
        double *zs = zsal_out + (size_t)c * COL_NBLYR;
        double *si = sice + (size_t)c * COL_NILYR;
        hin_old[c] = 0.0;
        if (aicen_init[c] > PUNY) {
            hin_old[c] = D(vicen_init[c], aicen_init[c]);
        } else {
            first_ice[n] = 1;
            if (tr_brine) fbri[n] = 1.0;
            if (n == 0) Rayleigh = 0;
            if (z_tracers) ztr_frac[(size_t)c * nzbgc] = zpar[ZP_FRAC_INIT];
            if (solve_zsal) for (int k = 0; k < COL_NBLYR; ++k) zs[k] = 0.0;
        }
        if (aicen[c] > PUNY) {
            dhbr_top[c] = 0.0;
            dhbr_bot[c] = 0.0;
            if (tr_brine) {
                if (fbri[n] <= 0.0) fbri[n] = 1.0;
                double dhice = 0.0, hbr_old, hin, hsn, hbrin = 0.0, kavg = 0.0, bphi_o = 0.0, dh_direct;
                l_stop = preflushing_changes(aicen[c], vicen[c], vsnon[c], meltb[c], meltt[c], congel[c],
                                             snoice[c], &hin_old[c], &dhice, &fbri[n], &dhbr_top[c],
                                             &dhbr_bot[c], &hbr_old, &hin, &hsn, first_ice[n]);
                if (l_stop) { hbri[iCell] = hbri_c; status[iCell] = l_stop; return; }
                double *idi = iDi + (size_t)c * NB1 - 1;          // 1-based
                double bT[NB2 + 1], bp[NB2 + 1], rho = sice_rho[c];
                for (int k = 1; k <= NB2; ++k) { bT[k] = bTiz[(size_t)c * NB2 + k - 1]; bp[k] = bphi[(size_t)c * NB2 + k - 1]; }
                double bSin[NB2 + 1], brine_sal[NB2 + 1], brine_rho[NB2 + 1];
                double iTin[NB1 + 1], iphin[NB1 + 1], ibrine_rho[NB1 + 1], ibrine_sal[NB1 + 1];
                if (solve_zsal) {
                    l_stop = compute_microS(bgrid0, cgrid0, igrid, si, qice + (size_t)c * COL_NILYR,
                                            tsfc[c], zs, hin_old[c], &hbr_old, sss[iCell], sst[iCell], bT, iTin,
                                            bp, &kavg, &bphi_o, first_ice[n], bSin, brine_sal, brine_rho, iphin,
                                            ibrine_rho, ibrine_sal, &rho);
                    if (l_stop) { hbri[iCell] = hbri_c; status[iCell] = l_stop; return; }
                    for (int k = 1; k <= NB2; ++k) { bTiz[(size_t)c * NB2 + k - 1] = bT[k]; bphi[(size_t)c * NB2 + k - 1] = bp[k]; }
                    sice_rho[c] = rho;
                } else {
                    for (int k = 1; k <= NB1; ++k) idi[k] = 0.0;
                    // colpkg_biogeochemistry does not check compute_microS_mushy's l_stop: on a
                    // remap stop the profiles keep their words, update_hbrine runs with
                    // kavg = bphi_o = 0, and the next ice category's preflushing_changes clears it
                    l_stop = compute_microS_mushy(bgrid, cgrid0, bgrid0, igrid, qice + (size_t)c * COL_NILYR,
                                                  si, hin_old[c], &hbr_old, sss[iCell], sst[iCell], bT, bp, &kavg,
                                                  &bphi_o, &rho, idi, l_sk, iphin_c);
                    if (!l_stop) {
                        for (int k = 1; k <= NB2; ++k) { bTiz[(size_t)c * NB2 + k - 1] = bT[k]; bphi[(size_t)c * NB2 + k - 1] = bp[k]; }
                        sice_rho[c] = rho;
                    }
                }
                update_hbrine(meltt[c], melts[c], dt, hin, hsn, hin_old[c], &hbrin, hbr_old, &fbri[n],
                              &dhbr_top[c], dhbr_bot[c], kavg, &bphi_o, &darcy_V[c], bp[2], aice0[iCell],
                              &dh_direct);
                hbri_c = A(hbri_c, M(hbrin, aicen[c]));
                if (solve_zsal) {
                    double *ik = iki + (size_t)c * NB1 - 1;       // 1-based
                    l_stop = zsalinity(n, dt, cgrid0, bgrid0, bgrid, igrid, zs, qice + (size_t)c * COL_NILYR, si,
                                       fbri[n], bSin, bT, bp, iphin, ik, hbr_old, hbrin, hin, hin_old[c], idi,
                                       darcy_V[c], brine_sal, brine_rho, ibrine_sal, ibrine_rho, dh_direct,
                                       &Rayleigh, first_ice[n], sss[iCell], sst[iCell], dhbr_top[c], dhbr_bot[c],
                                       &fzsal[iCell], &fzsal_g[iCell], bphi_o, vicen[c], aicen_init[c],
                                       &zsal_tot[iCell], zp);
                    for (int k = 1; k <= NB2; ++k) { bTiz[(size_t)c * NB2 + k - 1] = bT[k]; bphi[(size_t)c * NB2 + k - 1] = bp[k]; }
                    if (l_stop) { hbri[iCell] = hbri_c; status[iCell] = l_stop; return; }
                }
                if (z_tracers) {
                    // zbio (z_tracers is carried with the brine height only): its l_stop is
                    // z_biogeochemistry's, which clears compute_microS_mushy's
                    double *trc = ztr_conc + (size_t)c * (NB1 + 2);
                    double zsn, zan, flux_sno = 0.0, flux_bion = 0.0;
                    update_snow_bgc(dt, melts[c], snoice[c], fsnow[iCell], trc, aicen_init[c], vsno_init[c],
                                    vsnon[c], aicen[c], flux_bio_atm[(size_t)iCell * nzbgc], &zsn, &zan,
                                    &flux_sno, snow_new_category);
                    l_stop = z_biogeochemistry(dt, first_ice[n], aicen[c], vicen[c], hin_old[c], ocean_bio,
                                               &flux_bion, bp, iphin_c, trc, ztr_frac + (size_t)c * nzbgc, idi,
                                               fswthrul + (size_t)c * (COL_NILYR + 1), zfswin + (size_t)c * NB1,
                                               dhbr_top[c], dhbr_bot[c], hbrin, hbr_old, darcy_V[c], bgrid, igrid,
                                               icgrid, bphi_o, zsn, zan, dh_direct, meltb[c], congel[c], zpar);
                    flux_bion = A(flux_bion, flux_sno);
                    if (l_stop) { hbri[iCell] = hbri_c; status[iCell] = l_stop; return; }
                    const size_t o = (size_t)iCell * nzbgc;
                    merge_bgc_fluxes(dt, aicen[c], vicen[c], vsnon[c], trc, fbri[n], flux_bion, zsn, zan,
                                     &flux_bio[o], &fbio_snoice[o], &fbio_atmice[o], &ice_bio_net[o],
                                     &snow_bio_net[o]);
                }
            }
            first_ice[n] = 0;
        }
    }
    hbri[iCell] = hbri_c;
    // the stop of the cell's last ice category reaches the wrapper, which terminates MPAS
    if (l_stop) { status[iCell] = l_stop; return; }
    // get_cice_tracer_array_category and the wrapper's tail
    for (int n = 0; n < ncat; ++n) {
        if (tr_brine) fbri_out[iCell * ncat + n] = fbri[n];
        newly[iCell * ncat + n] = first_ice[n] ? 1 : 0;
    }
    total_skeletal[iCell] = 0.0;
    for (int k = 0; k < NB1; ++k) bio_sw_cell[(size_t)iCell * NB1 + k] = 0.0;
    if (z_tracers)
        for (int n = 0; n < ncat; ++n)
            for (int k = 0; k < NB1; ++k)
                bio_sw_cell[(size_t)iCell * NB1 + k] = A(bio_sw_cell[(size_t)iCell * NB1 + k],
                                                         M(zfswin[((size_t)iCell * ncat + n) * NB1 + k],
                                                           aicen[iCell * ncat + n]));
    // only ever zeroed: a criterion solve_S_dt switches on is not stored (it changes no
    // other word: drainage needs hbrin > thinS = Ra_c, where solve_S_dt switches it on)
    if (!Rayleigh) rayleigh[iCell] = 0.0;
}
