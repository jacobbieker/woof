// MPAS-Seaice v8.4.2: the column biogeochemistry's brine height
// (column_biogeochemistry -> colpkg_biogeochemistry with tr_brine and without
// the vertical salinity, the vertical or the skeletal biochemistry).
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
//     167-333    remap_zbgc (the CICE grid to the biology grid, nr0 = 0)
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
// liquid_fraction). x**2 and x**3 (bTin**2, bTin**3, iphin**exp_h,
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

// remap_zbgc from the CICE grid to the biology grid (nr0 = 0): trtmp(it+kr-1)
// is out[kr], kr = 1..nblyr, an untouched one 0 as trtmp's initial value
__device__ int remap_cice_to_bio(const double *src, double hice, double hinS, const double *cgrid,
                                 const double *bgrid, double *out)
{
    if (hinS < 0.0 || hice < 0.0) return BG_REMAP;
    const int n_nd = COL_NILYR, n_nr = COL_NBLYR, n_plus = 2;
    double dgrid[COL_NILYR + 3], rgrid[COL_NBLYR + 3], tracer[COL_NILYR + 3];
    double trgrid[COL_NBLYR + COL_NILYR + 4], trdr[COL_NBLYR + COL_NILYR + 4];
    for (int k = 0; k < COL_NBLYR + COL_NILYR + 4; ++k) { trgrid[k] = 0.0; trdr[k] = 0.0; }
    dgrid[1] = col_min(col_min(A(-hice, hinS), A(-hinS, hice)), 0.0);
    dgrid[COL_NILYR + 2] = col_min(hinS, hice);
    tracer[1] = src[0];
    tracer[COL_NILYR + 2] = src[COL_NILYR - 1];
    rgrid[COL_NBLYR + 2] = col_min(hinS, hice);
    // ice_grid(kd) = cgrid(kd+1) = cgrid[kd]; bio_grid(kr) = bgrid(kr+1) = bgrid[kr] (0-based)
    if (hice > hinS) {
        rgrid[1] = 0.0;
        for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = M(bgrid[kr], hinS);
        for (int kd = 1; kd <= n_nd; ++kd) {
            dgrid[kd + 1] = A(M(S(cgrid[kd], 1.0), hice), hinS);
            tracer[kd + 1] = src[kd - 1];
        }
    } else {
        rgrid[1] = A(-hinS, hice);
        for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = A(M(S(bgrid[kr], 1.0), hinS), hice);
        for (int kd = 1; kd <= n_nd; ++kd) {
            dgrid[kd + 1] = M(cgrid[kd], hice);
            tracer[kd + 1] = src[kd - 1];
        }
    }
    int kdr = 0, kdi = 1;
    for (int kr = 1; kr <= n_nr; ++kr) {
        const int last = n_nd + n_plus;
        for (int kd = kdi; kd <= last; ++kd) {
            if (dgrid[kd] < rgrid[kr + 1]) {
                kdr = kdr + 1;
                trgrid[kdr] = dgrid[kd];
                trdr[kdr] = tracer[kd];
            } else if (dgrid[kd] > rgrid[kr + 1]) {
                kdr = kdr + 1;
                kdi = kd;
                trgrid[kdr] = rgrid[kr + 1];
                out[kr] = A(trdr[kdr - 1], D(M(S(rgrid[kr + 1], trgrid[kdr - 1]), S(tracer[kd], trdr[kdr - 1])),
                                            S(dgrid[kd], trgrid[kdr - 1])));
                trdr[kdr] = out[kr];
                break;
            } else {
                kdr = kdr + 1;
                kdi = kd + 1;
                trgrid[kdr] = rgrid[kr + 1];
                out[kr] = tracer[kd];
                trdr[kdr] = tracer[kd];
                break;
            }
        }
    }
    return BG_OK;
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
                                    double *kperm, double *bphi_min, double *sice_rho, double *iDin, double l_sk)
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
                              double kperm, double *bphi_min, double *darcy_V, double bphin, double aice0)
{
    double hbrocn = 0.0, hbrocn_new = 0.0, hbrmin, dhbr, darcy_coeff, dhrunoff, dhflood, dhbr_hin;
    *darcy_V = 0.0;
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
        } else if (hbrocn < 0.0 && *hbr > BG_THINS) {
            hbrocn_new = M(hbrocn, glibc_exp(-M(D(darcy_coeff, *bphi_min), dt)));
            dhflood = M(col_max(0.0, S(hbrocn_new, hbrocn)), aice0);
            *hbr = col_max(hbrmin, A(h_ocn, hbrocn_new));
            *darcy_V = -bg_sign(M(D(A(S(hbrocn, hbrocn_new), dhflood), dt), *bphi_min), hbrocn);
            *dhS_top = S(S(*dhS_top, D(M(*darcy_V, dt), *bphi_min)), dhflood);
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

// column_biogeochemistry: one thread per owned cell. flags: bit 0 tr_brine.
// fbri is brineFraction (or a dummy without the brine height).
extern "C" __global__ void seaice_column_biogeochemistry_v842(
    int *__restrict__ status, double dt, double l_sk, int ncat, int flags, int nCellsSolve,
    const double *__restrict__ bgrid0, const double *__restrict__ igrid0, const double *__restrict__ cgrid0,
    const double *__restrict__ aicen_init, const double *__restrict__ vicen_init,
    const double *__restrict__ aicen, const double *__restrict__ vicen, const double *__restrict__ vsnon,
    const double *__restrict__ meltb, const double *__restrict__ meltt, const double *__restrict__ congel,
    const double *__restrict__ snoice, const double *__restrict__ melts,
    const double *__restrict__ sst, const double *__restrict__ sss, const double *__restrict__ aice0,
    const double *__restrict__ qice, const double *__restrict__ sice,
    double *__restrict__ fbri_out, double *__restrict__ hin_old, int *__restrict__ newly,
    double *__restrict__ rayleigh, double *__restrict__ dhbr_top, double *__restrict__ dhbr_bot,
    double *__restrict__ bTiz, double *__restrict__ bphi, double *__restrict__ iDi,
    double *__restrict__ darcy_V, double *__restrict__ sice_rho, double *__restrict__ hbri,
    double *__restrict__ atm_bc, double *__restrict__ atm_dust, double *__restrict__ total_skeletal,
    double *__restrict__ bio_sw_cell)
{
    const int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const int tr_brine = flags & 1;
    // the uncoupled build's deposition (#else of #ifdef coupled)
    atm_bc[iCell * 2 + 0] = 1.e-12;
    atm_bc[iCell * 2 + 1] = 1.e-12;
    for (int k = 0; k < 4; ++k) atm_dust[iCell * 4 + k] = 1.e-13;
    // 1-based grids
    double bgrid[NB2 + 1], igrid[NB1 + 1];
    for (int k = 1; k <= NB2; ++k) bgrid[k] = bgrid0[k - 1];
    for (int k = 1; k <= NB1; ++k) igrid[k] = igrid0[k - 1];
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
        hin_old[c] = 0.0;
        if (aicen_init[c] > PUNY) {
            hin_old[c] = D(vicen_init[c], aicen_init[c]);
        } else {
            first_ice[n] = 1;
            if (tr_brine) fbri[n] = 1.0;
            if (n == 0) Rayleigh = 0;
        }
        if (aicen[c] > PUNY) {
            dhbr_top[c] = 0.0;
            dhbr_bot[c] = 0.0;
            if (tr_brine) {
                if (fbri[n] <= 0.0) fbri[n] = 1.0;
                double dhice = 0.0, hbr_old, hin, hsn, hbrin = 0.0, kavg = 0.0, bphi_o = 0.0;
                l_stop = preflushing_changes(aicen[c], vicen[c], vsnon[c], meltb[c], meltt[c], congel[c],
                                             snoice[c], &hin_old[c], &dhice, &fbri[n], &dhbr_top[c],
                                             &dhbr_bot[c], &hbr_old, &hin, &hsn, first_ice[n]);
                if (l_stop) { hbri[iCell] = hbri_c; status[iCell] = l_stop; return; }
                double *idi = iDi + (size_t)c * NB1 - 1;          // 1-based
                for (int k = 1; k <= NB1; ++k) idi[k] = 0.0;
                double bT[NB2 + 1], bp[NB2 + 1], rho = sice_rho[c];
                for (int k = 1; k <= NB2; ++k) { bT[k] = bTiz[(size_t)c * NB2 + k - 1]; bp[k] = bphi[(size_t)c * NB2 + k - 1]; }
                // colpkg_biogeochemistry does not check compute_microS_mushy's l_stop: on a
                // remap stop the profiles keep their words, update_hbrine runs with
                // kavg = bphi_o = 0, and the next ice category's preflushing_changes clears it
                l_stop = compute_microS_mushy(bgrid, cgrid0, bgrid0, igrid, qice + (size_t)c * COL_NILYR,
                                              sice + (size_t)c * COL_NILYR, hin_old[c], &hbr_old, sss[iCell],
                                              sst[iCell], bT, bp, &kavg, &bphi_o, &rho, idi, l_sk);
                if (!l_stop) {
                    for (int k = 1; k <= NB2; ++k) { bTiz[(size_t)c * NB2 + k - 1] = bT[k]; bphi[(size_t)c * NB2 + k - 1] = bp[k]; }
                    sice_rho[c] = rho;
                }
                update_hbrine(meltt[c], melts[c], dt, hin, hsn, hin_old[c], &hbrin, hbr_old, &fbri[n],
                              &dhbr_top[c], dhbr_bot[c], kavg, &bphi_o, &darcy_V[c], bp[2], aice0[iCell]);
                hbri_c = A(hbri_c, M(hbrin, aicen[c]));
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
    if (!Rayleigh) rayleigh[iCell] = 0.0;
}
