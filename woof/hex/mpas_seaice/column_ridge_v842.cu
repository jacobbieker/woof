// MPAS-Seaice v8.4.2: the column package's mechanical redistribution
// (column_ridging -> colpkg_step_ridge): the ridging participation and
// redistribution, the ridging iterations, and the ITD cleanup after them.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     2498-2767  column_ridging
//   src/core_seaice/column/ice_colpkg.F90:
//     2981-3161  colpkg_step_ridge
//   src/core_seaice/column/ice_mechred.F90:
//     80-594     ridge_ice
//     606-629    asum_ridging
//     637-719    ridge_prep
//     738-1009   ridge_itd
//     1027-1535  ridge_shift
// (William H. Lipscomb, Elizabeth C. Hunke, Adrian K. Turner and the CICE
// Consortium). Copyright (c) 2013-2019, Los Alamos National Security, LLC
// (LANS) and the University Corporation for Atmospheric Research (UCAR);
// BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is
// not the version available from LANS and UCAR.
//
// Appended to column_v842.cu and column_itd_v842.cu (the rounding macros,
// constants, the tracer layout, colpkg_compute_tracers and cleanup_itd).
// l_conservation_check is .false. (no conservation checks); the aerosols,
// the biogeochemistry and the zero-layer thermodynamics are refused by the
// host (column_ridge_v842.py), so tr_aero and tr_brine are .false. exp is
// glibc_exp (glibc_flt64.cuh, the MIT transcription of glibc's binary64 exp);
// gfortran folds c1/(c1 - exp(-astari)), whose exp(-20) every correctly
// rounded exp gives. Fortran's MIN and MAX are gfortran's comparisons
// (rdg_min, rdg_max: the first argument on a tie), SIGN is copysign.
//
// One thread is one owned cell, with column_itd_v842.cu's layout: trcrn of
// cell c is trcrnAll[(c*ncat + n)*nTracers + it] and a(nCategories, nCells)
// is a[c*ncat + n].

#define RDG_NITERMAX 20
#define RDG_BIG 1.0e8          // ridge_prep's hin_max(ncat)
#define RDG_CS 0.25            // Cs = p25
#define RDG_FSNOWRDG 0.5       // fsnowrdg = p5
#define RDG_GSTAR 0.15         // Gstar = p15
#define RDG_ASTAR 0.05         // astar = p05
#define RDG_MAXRAFT 1.0        // maxraft = c1
#define RDG_HSTAR 25.0         // Hstar = c25

// abort codes: the stop labels in column_ridge_v842.STOP_LABELS
#define RDG_OK 0
#define RDG_NITER 1            // ridge_ice: Exceeded max number of ridging iterations
#define RDG_AICE0_NEG 2        // Ridging error: aice0 < 0
#define RDG_ARDG_GT 3          // Ridging error: ardg > aicen
#define RDG_AREA_GT1 4         // ridge_ice: total area > 1
#define RDG_CLEANUP 5          // ice: ITD cleanup error in colpkg_step_ridge

// gfortran's MIN/MAX: mvar = a; if (b < mvar || isnan(mvar)) mvar = b
__device__ __forceinline__ double rdg_min(double a, double b) { return (b < a || isnan(a)) ? b : a; }
__device__ __forceinline__ double rdg_max(double a, double b) { return (b > a || isnan(a)) ? b : a; }

// ice_mechred.F90:606-629 asum_ridging
__device__ double rdg_asum(int ncat, const double *aicen, double aice0)
{
    double asum = aice0;
    for (int n = 0; n < ncat; ++n) asum = A(asum, aicen[n]);
    return asum;
}

// ice_mechred.F90:738-1009 ridge_itd. apartic[0..ncat] is apartic(0:ncat);
// the per-category arrays are 0-based (category n at n-1).
__device__ void rdg_ridge_itd(int ncat, double aice0, const double *aicen, const double *vicen, int krdg_partic,
                              int krdg_redist, double mu_rdg, double *aksum, double *apartic, double *hrmin,
                              double *hrmax, double *hrexp, double *krdg, double *aparticn, double *krdgn,
                              double *mraft)
{
    const double gstari = D(1.0, RDG_GSTAR);   // parameters gfortran folds
    const double astari = D(1.0, RDG_ASTAR);
    double G[ITD_MAXCAT + 2];                  // G[n + 1] is Gsum(n), n = -1..ncat
    G[0] = 0.0;
    if (aice0 > PUNY) G[1] = aice0;
    else G[1] = G[0];
    apartic[0] = 0.0;
    for (int n = 1; n <= ncat; ++n) {
        G[n + 1] = 1.0;
        apartic[n] = 0.0;
        hrmin[n - 1] = 0.0;
        hrmax[n - 1] = 0.0;
        hrexp[n - 1] = 0.0;
        krdg[n - 1] = 1.0;
        if (aicen[n - 1] > PUNY) G[n + 1] = A(G[n], aicen[n - 1]);
        else G[n + 1] = G[n];
    }
    double work = D(1.0, G[ncat + 1]);
    for (int n = 0; n <= ncat; ++n) G[n + 1] = M(G[n + 1], work);

    if (krdg_partic == 0) {
        for (int n = 0; n <= ncat; ++n) {
            if (G[n + 1] < RDG_GSTAR) {
                apartic[n] = M(M(gstari, S(G[n + 1], G[n])), S(2.0, M(gstari, A(G[n], G[n + 1]))));
            } else if (G[n] < RDG_GSTAR) {
                apartic[n] = M(M(gstari, S(RDG_GSTAR, G[n])), S(2.0, M(gstari, A(RDG_GSTAR, G[n]))));
            }
        }
    } else if (krdg_partic == 1) {
        const double xtmp = D(1.0, S(1.0, glibc_exp(-astari)));
        G[0] = M(glibc_exp(-M(G[0], astari)), xtmp);
        for (int n = 0; n <= ncat; ++n) {
            G[n + 1] = M(glibc_exp(-M(G[n + 1], astari)), xtmp);
            apartic[n] = S(G[n], G[n + 1]);
        }
    }

    if (krdg_redist == 0) {
        for (int i = 0; i < ncat; ++i) {
            if (aicen[i] > PUNY) {
                double hi = D(vicen[i], aicen[i]);
                hrmin[i] = rdg_min(M(2.0, hi), A(hi, RDG_MAXRAFT));
                hrmax[i] = M(2.0, SQRT(M(RDG_HSTAR, hi)));
                hrmax[i] = rdg_max(hrmax[i], A(hrmin[i], PUNY));
                double hrmean = M(0.5, A(hrmin[i], hrmax[i]));
                krdg[i] = D(hrmean, hi);
            }
        }
    } else {
        for (int i = 0; i < ncat; ++i) {
            if (aicen[i] > PUNY) {
                double hi = D(vicen[i], aicen[i]);
                hi = rdg_max(hi, PUNY);
                hrmin[i] = rdg_min(M(2.0, hi), A(hi, RDG_MAXRAFT));
                hrexp[i] = M(mu_rdg, SQRT(hi));
                krdg[i] = D(A(hrmin[i], hrexp[i]), hi);
                mraft[i] = rdg_max(0.0, copysign(1.0, S(A(hi, RDG_MAXRAFT), hrmin[i])));
                double xtmp = M(mraft[i], S(D(A(M(2.0, hi), hrexp[i]), hi), krdg[i]));
                mraft[i] = rdg_max(0.0, copysign(1.0, S(PUNY, fabs(xtmp))));
            }
        }
    }

    double ak = apartic[0];
    for (int n = 1; n <= ncat; ++n) ak = A(ak, M(apartic[n], S(1.0, D(1.0, krdg[n - 1]))));
    *aksum = ak;
    for (int i = 0; i < ncat; ++i) {
        aparticn[i] = apartic[i + 1];
        krdgn[i] = krdg[i];
    }
}

// ice_mechred.F90:1027-1535 ridge_shift (no aerosols, no brine). Returns the
// abort code. aredistn and vredistn are written only for the thinnest
// category's ridges under the exponential redistribution, as v8.4.2 does.
__device__ int rdg_ridge_shift(const ItdTracers &T, double dt, int ncat, const double *hin_max, double *aicen,
                               double *trcrn, double *vicen, double *vsnon, double *aice0, int krdg_redist,
                               double aksum, const double *apartic, const double *hrmin, const double *hrmax,
                               const double *hrexp, const double *krdg, double closing_net, double *opning,
                               double *ardg1, double *ardg2, double *virdg, double *aopen, double *ardg1nn,
                               double *ardg2nn, double *virdgnn, double *msnow_mlt, double *esnow_mlt,
                               double *mpond, int topo, double *aredistn, double *vredistn, double *atrcrn)
{
    const int nt = T.nt;
    const int *ix = T.idx;
    double aicen_init[ITD_MAXCAT], vicen_init[ITD_MAXCAT], vsnon_init[ITD_MAXCAT];
    for (int i = 0; i < ncat; ++i) {
        aicen_init[i] = aicen[i];
        vicen_init[i] = vicen[i];
        vsnon_init[i] = vsnon[i];
        const double *tr = trcrn + i * nt;
        for (int it = 0; it < nt; ++it) {
            atrcrn[i * nt + it] = M(tr[it], A(A(M(T.base[it], aicen[i]), M(T.base[nt + it], vicen[i])),
                                              M(T.base[2 * nt + it], vsnon[i])));
            for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                int ntr = T.strata[itl * nt + it] - 1;
                atrcrn[i * nt + it] = M(atrcrn[i * nt + it], tr[ntr]);
            }
        }
    }

    double closing_gross = D(closing_net, aksum);
    if (apartic[0] > 0.0) {
        double wk1 = M(M(apartic[0], closing_gross), dt);
        if (wk1 > *aice0) {
            double tmpfac = D(*aice0, wk1);
            closing_gross = M(closing_gross, tmpfac);
            *opning = M(*opning, tmpfac);
        }
    }
    for (int n = 1; n <= ncat; ++n) {
        if (aicen[n - 1] > PUNY && apartic[n] > 0.0) {
            double wk1 = M(M(apartic[n], closing_gross), dt);
            if (wk1 > aicen[n - 1]) {
                double tmpfac = D(aicen[n - 1], wk1);
                closing_gross = M(closing_gross, tmpfac);
                *opning = M(*opning, tmpfac);
            }
        }
    }

    *aice0 = A(S(*aice0, M(M(apartic[0], closing_gross), dt)), M(*opning, dt));
    if (*aice0 < -PUNY) {
        return RDG_AICE0_NEG;
    } else if (*aice0 < 0.0) {
        *aice0 = 0.0;
    }
    *aopen = M(*opning, dt);

    for (int n = 1; n <= ncat; ++n) {
        const int i = n - 1;
        if (!(aicen_init[i] > PUNY && apartic[n] > 0.0 && closing_gross > 0.0)) continue;
        const double *tr = trcrn + i * nt;
        double ardg1n = M(M(apartic[n], closing_gross), dt);
        if (ardg1n > A(aicen_init[i], PUNY)) {
            return RDG_ARDG_GT;
        } else {
            ardg1n = rdg_min(aicen_init[i], ardg1n);
        }
        double ardg2n = D(ardg1n, krdg[i]);
        double afrac = D(ardg1n, aicen_init[i]);
        double virdgn = M(vicen_init[i], afrac);
        double vsrdgn = M(vsnon_init[i], afrac);
        aicen[i] = S(aicen[i], ardg1n);
        vicen[i] = S(vicen[i], virdgn);
        vsnon[i] = S(vsnon[i], vsrdgn);
        *ardg1 = A(*ardg1, ardg1n);
        *ardg2 = A(*ardg2, ardg2n);
        *virdg = A(*virdg, virdgn);
        ardg1nn[i] = ardg1n;
        ardg2nn[i] = ardg2n;
        virdgnn[i] = virdgn;

        *msnow_mlt = A(*msnow_mlt, M(M(RHOS, vsrdgn), S(1.0, RDG_FSNOWRDG)));
        if (topo) *mpond = A(*mpond, M(M(ardg1n, tr[ix[TI_APND]]), tr[ix[TI_HPND]]));

        double dhr = S(hrmax[i], hrmin[i]);
        double dhr2 = S(M(hrmax[i], hrmax[i]), M(hrmin[i], hrmin[i]));

        for (int k = 0; k < ITD_NSLYR; ++k) {
            double esrdgn = D(M(vsrdgn, tr[ix[TI_QSNO] + k]), (double)ITD_NSLYR);
            *esnow_mlt = A(*esnow_mlt, M(esrdgn, S(1.0, RDG_FSNOWRDG)));
        }

        for (int it = 0; it < nt; ++it) {
            double trfactor[3] = {M(T.base[it], ardg1n), M(T.base[nt + it], virdgn), M(T.base[2 * nt + it], vsrdgn)};
            double work = 0.0;
            for (int k = 0; k < 3; ++k) work = A(work, M(trfactor[k], tr[it]));
            for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                int ntr = T.strata[itl * nt + it] - 1;
                work = M(work, tr[ntr]);
            }
            atrcrn[i * nt + it] = S(atrcrn[i * nt + it], work);
        }

        for (int nr = 1; nr <= ncat; ++nr) {
            const int j = nr - 1;
            double farea, fvol;
            if (krdg_redist == 0) {
                double hL, hR;
                if (hrmin[i] >= hin_max[nr] || hrmax[i] <= hin_max[nr - 1]) {
                    hL = 0.0;
                    hR = 0.0;
                } else {
                    hL = rdg_max(hrmin[i], hin_max[nr - 1]);
                    hR = rdg_min(hrmax[i], hin_max[nr]);
                }
                farea = D(S(hR, hL), dhr);
                fvol = D(S(M(hR, hR), M(hL, hL)), dhr2);
            } else {
                double hi1 = hrmin[i];
                double hexp = hrexp[i];
                if (nr < ncat) {
                    if (hi1 >= hin_max[nr]) {
                        farea = 0.0;
                        fvol = 0.0;
                    } else {
                        double hL = rdg_max(hi1, hin_max[nr - 1]);
                        double hR = hin_max[nr];
                        double expL = glibc_exp(-D(S(hL, hi1), hexp));
                        double expR = glibc_exp(-D(S(hR, hi1), hexp));
                        farea = S(expL, expR);
                        fvol = D(S(M(A(hL, hexp), expL), M(A(hR, hexp), expR)), A(hi1, hexp));
                    }
                } else {
                    double hL = rdg_max(hi1, hin_max[nr - 1]);
                    double expL = glibc_exp(-D(S(hL, hi1), hexp));
                    farea = expL;
                    fvol = D(M(A(hL, hexp), expL), A(hi1, hexp));
                }
                if (n == 1) {
                    aredistn[j] = M(farea, ardg2n);
                    vredistn[j] = M(fvol, virdgn);
                }
            }

            aicen[j] = A(aicen[j], M(farea, ardg2n));
            vicen[j] = A(vicen[j], M(fvol, virdgn));
            vsnon[j] = A(vsnon[j], M(M(fvol, vsrdgn), RDG_FSNOWRDG));

            for (int it = 0; it < nt; ++it) {
                double trfactor[3] = {0.0, 0.0, 0.0};
                if (it != ix[TI_ALVL] && it != ix[TI_VLVL]) {
                    trfactor[0] = M(M(T.base[it], ardg2n), farea);
                    trfactor[1] = M(M(T.base[nt + it], virdgn), fvol);
                    trfactor[2] = M(M(M(T.base[2 * nt + it], vsrdgn), fvol), RDG_FSNOWRDG);
                }
                double work = 0.0;
                for (int k = 0; k < 3; ++k) work = A(work, M(trfactor[k], tr[it]));
                for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                    int ntr = T.strata[itl * nt + it] - 1;
                    if (ntr == ix[TI_FBRI]) work = M(work, tr[ntr]);   // brine fraction only
                    else work = 0.0;
                }
                atrcrn[j * nt + it] = A(atrcrn[j * nt + it], work);
            }
        }
    }

    for (int i = 0; i < ncat; ++i)
        itd_compute_tracers(T, atrcrn + i * nt, aicen[i], vicen[i], vsnon[i], trcrn + i * nt);
    return RDG_OK;
}

// ice_mechred.F90:80-594 ridge_ice (with ridge_prep). Returns the abort
// code; *nrepeat counts its 'Repeat ridging' warnings.
__device__ int rdg_ridge_ice(const ItdTracers &T, double dt, int ndtd, int ncat, double *hin_max, double rdg_conv,
                             double rdg_shear, double *aicen, double *trcrn, double *vicen, double *vsnon,
                             double *aice0, int krdg_partic, int krdg_redist, double mu_rdg, double *dardg1dt,
                             double *dardg2dt, double *dvirdgdt, double *opening, double *fpond, double *fresh,
                             double *fhocn, int topo, double *aparticn, double *krdgn, double *aredistn,
                             double *vredistn, double *dardg1ndt, double *dardg2ndt, double *dvirdgndt,
                             double *araftn, double *vraftn, double *atrcrn, int *nrepeat)
{
    double msnow_mlt = 0.0, esnow_mlt = 0.0, mpond = 0.0;
    double ardg1 = 0.0, ardg2 = 0.0, virdg = 0.0, aopen = 0.0;
    double ardg1n[ITD_MAXCAT], ardg2n[ITD_MAXCAT], virdgn[ITD_MAXCAT], mraftn[ITD_MAXCAT];
    for (int i = 0; i < ncat; ++i) {
        ardg1n[i] = 0.0;
        ardg2n[i] = 0.0;
        virdgn[i] = 0.0;
        mraftn[i] = 0.0;
    }

    double asum = rdg_asum(ncat, aicen, *aice0);

    // ridge_prep
    hin_max[ncat] = RDG_BIG;
    double closing_net = A(M(RDG_CS, rdg_shear), rdg_conv);
    double divu_adv = D(S(1.0, asum), dt);
    if (divu_adv < 0.0) closing_net = rdg_max(closing_net, -divu_adv);
    double opning = A(closing_net, divu_adv);

    for (int niter = 1; niter <= RDG_NITERMAX; ++niter) {
        double aksum, apartic[ITD_MAXCAT + 1], hrmin[ITD_MAXCAT], hrmax[ITD_MAXCAT], hrexp[ITD_MAXCAT],
            krdg[ITD_MAXCAT];
        rdg_ridge_itd(ncat, *aice0, aicen, vicen, krdg_partic, krdg_redist, mu_rdg, &aksum, apartic, hrmin, hrmax,
                      hrexp, krdg, aparticn, krdgn, mraftn);
        int stop = rdg_ridge_shift(T, dt, ncat, hin_max, aicen, trcrn, vicen, vsnon, aice0, krdg_redist, aksum,
                                   apartic, hrmin, hrmax, hrexp, krdg, closing_net, &opning, &ardg1, &ardg2,
                                   &virdg, &aopen, ardg1n, ardg2n, virdgn, &msnow_mlt, &esnow_mlt, &mpond, topo,
                                   aredistn, vredistn, atrcrn);
        if (stop) return stop;

        asum = rdg_asum(ncat, aicen, *aice0);
        bool iterate_ridging;
        if (fabs(S(asum, 1.0)) < PUNY) {
            iterate_ridging = false;
            closing_net = 0.0;
            opning = 0.0;
        } else {
            iterate_ridging = true;
            divu_adv = D(S(1.0, asum), dt);
            closing_net = rdg_max(0.0, -divu_adv);
            opning = rdg_max(0.0, divu_adv);
        }
        if (iterate_ridging) ++*nrepeat;
        else break;
        if (niter == RDG_NITERMAX) return RDG_NITER;
    }

    double dti = D(1.0, dt);
    *dardg1dt = M(ardg1, dti);
    *dardg2dt = M(ardg2, dti);
    *dvirdgdt = M(virdg, dti);
    *opening = M(aopen, dti);
    for (int i = 0; i < ncat; ++i) {
        dardg1ndt[i] = M(ardg1n[i], dti);
        dardg2ndt[i] = M(ardg2n[i], dti);
        dvirdgndt[i] = M(virdgn[i], dti);
        araftn[i] = M(mraftn[i], ardg2n[i]);
        vraftn[i] = M(mraftn[i], virdgn[i]);
    }

    dti = D(1.0, M((double)ndtd, dt));
    *fresh = A(*fresh, M(msnow_mlt, dti));
    *fhocn = A(*fhocn, M(esnow_mlt, dti));
    *fpond = S(*fpond, mpond);

    if (fabs(S(asum, 1.0)) > PUNY) return RDG_AREA_GT1;
    return RDG_OK;
}

// column_ridging -> colpkg_step_ridge, one owned cell per thread. MPAS's
// categoryThicknessLimits(ncat) is ridge_prep's 1e8 for every cell MPAS
// reaches: the host writes it.
extern "C" __global__ void seaice_column_ridging_v842(
    int *status, int *nwarn, int *nrepeat, double dt, int ndtd, double mu_rdg, int krdg_partic, int krdg_redist,
    int ncat, int nt, int flags, int nCellsSolve, const int *tidx, const int *trcr_depend,
    const double *trcr_base, const int *n_trcr_strata, const int *nt_strata, const double *categoryThicknessLimits,
    double *trcrnAll, double *atrcrnAll, double *iceAreaCategory, double *iceVolumeCategory,
    double *snowVolumeCategory, double *openWaterArea, double *iceAreaCell, const double *ridgeConvergence,
    const double *ridgeShear, double *areaLossRidge, double *areaGainRidge, double *iceVolumeRidged,
    double *openingRateRidge, double *pondFreshWaterFlux, double *oceanFreshWaterFlux, double *oceanSaltFlux,
    double *oceanHeatFlux, double *ridgeParticipationFunction, double *ratioRidgeThicknessToIce,
    double *fractionNewRidgeArea, double *fractionNewRidgeVolume, double *areaLossRidgeCategory,
    double *areaGainRidgeCategory, double *iceVolumeRidgedCategory, double *raftingIceArea,
    double *raftingIceVolume, int *newlyFormedIce)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    ItdTracers T;
    T.nt = nt;
    T.depend = trcr_depend;
    T.base = trcr_base;
    T.nstrata = n_trcr_strata;
    T.strata = nt_strata;
    for (int i = 0; i < TI_COUNT; ++i) T.idx[i] = tidx[i];

    double aicen[ITD_MAXCAT], vicen[ITD_MAXCAT], vsnon[ITD_MAXCAT], hin_max[ITD_MAXCAT + 1];
    int first_ice[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) {
        aicen[n] = iceAreaCategory[c * ncat + n];
        vicen[n] = iceVolumeCategory[c * ncat + n];
        vsnon[n] = snowVolumeCategory[c * ncat + n];
        first_ice[n] = newlyFormedIce[c * ncat + n] == 1;
    }
    for (int n = 0; n <= ncat; ++n) hin_max[n] = categoryThicknessLimits[n];
    double *trcrn = trcrnAll + (size_t)c * ncat * nt;
    double *atrcrn = atrcrnAll + (size_t)c * ncat * nt;
    double aice0 = openWaterArea[c], aice = iceAreaCell[c];
    double fpond = pondFreshWaterFlux[c], fresh = oceanFreshWaterFlux[c], fsalt = oceanSaltFlux[c];
    double fhocn = oceanHeatFlux[c];
    double dardg1dt = areaLossRidge[c], dardg2dt = areaGainRidge[c], dvirdgdt = iceVolumeRidged[c];
    double opening = openingRateRidge[c];
    int warnings = 0, repeats = 0;
    const size_t cc = (size_t)c * ncat;

    int code = rdg_ridge_ice(T, dt, ndtd, ncat, hin_max, ridgeConvergence[c], ridgeShear[c], aicen, trcrn, vicen,
                             vsnon, &aice0, krdg_partic, krdg_redist, mu_rdg, &dardg1dt, &dardg2dt, &dvirdgdt,
                             &opening, &fpond, &fresh, &fhocn, flags & ITD_TR_POND_TOPO,
                             ridgeParticipationFunction + cc, ratioRidgeThicknessToIce + cc,
                             fractionNewRidgeArea + cc, fractionNewRidgeVolume + cc, areaLossRidgeCategory + cc,
                             areaGainRidgeCategory + cc, iceVolumeRidgedCategory + cc, raftingIceArea + cc,
                             raftingIceVolume + cc, atrcrn, &repeats);
    if (code == RDG_OK) {
        double dtt = M(dt, (double)ndtd);
        if (itd_cleanup_itd(T, dtt, ncat, hin_max, aicen, trcrn, vicen, vsnon, &aice0, &aice, flags, first_ice,
                            &fpond, &fresh, &fsalt, &fhocn, atrcrn, &warnings))
            code = RDG_CLEANUP;
    }

    for (int n = 0; n < ncat; ++n) {
        iceAreaCategory[c * ncat + n] = aicen[n];
        iceVolumeCategory[c * ncat + n] = vicen[n];
        snowVolumeCategory[c * ncat + n] = vsnon[n];
        newlyFormedIce[c * ncat + n] = first_ice[n] ? 1 : 0;
    }
    openWaterArea[c] = aice0;
    iceAreaCell[c] = aice;
    pondFreshWaterFlux[c] = fpond;
    oceanFreshWaterFlux[c] = fresh;
    oceanSaltFlux[c] = fsalt;
    oceanHeatFlux[c] = fhocn;
    areaLossRidge[c] = dardg1dt;
    areaGainRidge[c] = dardg2dt;
    iceVolumeRidged[c] = dvirdgdt;
    openingRateRidge[c] = opening;
    status[c] = code;
    nwarn[c] = warnings;
    nrepeat[c] = repeats;
}

// ice_strength's column branch (mpas_seaice_velocity_solver.F:1359-1379)
// with colpkg_ice_strength (ice_colpkg.F90:3298-3410), one owned cell per
// thread: icePressure is zeroed, then, where the stress is solved, the
// strength of Rothrock (1975) (kstrength = 1: ridge_itd's participation and
// ridges, the potential-energy change per category, Cf*Cp/aksum) or of
// Hibler (1979) (kstrength = 0). Cf and Cp are the column constants'
// (ice_constants_colpkg), not config_ratio_ridging_work_to_PE, which v8.4.2
// does not read. asum_ridging's sum is computed and unused, as in v8.4.2.
extern "C" __global__ void seaice_column_ice_strength_v842(
    double *icePressure, const int *solveStress, const double *iceAreaCell, const double *iceVolumeCell,
    const double *openWaterArea, const double *iceAreaCategory, const double *iceVolumeCategory, int ncat,
    int kstrength, int krdg_partic, int krdg_redist, double mu_rdg, int nCellsSolve)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    icePressure[c] = 0.0;
    if (solveStress[c] != 1) return;
    const double *aicen = iceAreaCategory + c * ncat;
    const double *vicen = iceVolumeCategory + c * ncat;
    double strength = 0.0;
    if (kstrength == 1) {
        double aksum;
        double apartic[ITD_MAXCAT + 1], hrmin[ITD_MAXCAT], hrmax[ITD_MAXCAT], hrexp[ITD_MAXCAT], krdg[ITD_MAXCAT];
        double aparticn[ITD_MAXCAT], krdgn[ITD_MAXCAT], mraft[ITD_MAXCAT];
        (void)rdg_asum(ncat, aicen, openWaterArea[c]);
        rdg_ridge_itd(ncat, openWaterArea[c], aicen, vicen, krdg_partic, krdg_redist, mu_rdg, &aksum, apartic,
                      hrmin, hrmax, hrexp, krdg, aparticn, krdgn, mraft);
        const double p333 = D(1.0, 3.0);
        for (int n = 1; n <= ncat; ++n) {
            if (aicen[n - 1] > PUNY && apartic[n] > 0.0) {
                double hi = D(vicen[n - 1], aicen[n - 1]);
                double h2rdg;
                if (krdg_redist == 0) {
                    // x**3 is __powidf2's x*(x*x)
                    double xmax = hrmax[n - 1], xmin = hrmin[n - 1];
                    h2rdg = D(M(p333, S(M(xmax, M(xmax, xmax)), M(xmin, M(xmin, xmin)))), S(xmax, xmin));
                } else if (krdg_redist == 1) {
                    h2rdg = A(A(M(hrmin[n - 1], hrmin[n - 1]), M(M(2.0, hrmin[n - 1]), hrexp[n - 1])),
                              M(M(2.0, hrexp[n - 1]), hrexp[n - 1]));
                } else {
                    continue;
                }
                double dh2rdg = A(-M(hi, hi), D(h2rdg, krdg[n - 1]));
                strength = A(strength, M(apartic[n], dh2rdg));
            }
        }
        // Cp = 0.5*gravit*(rhow-rhoi)*rhoi/rhow; Cf*Cp folded
        const double cp_pe = D(M(M(M(0.5, 9.80616), S(RHOW, RHOI)), RHOI), RHOW);
        strength = D(M(M(17.0, cp_pe), strength), aksum);
    } else {
        // Pstar*vice*exp(-Cstar*(c1-aice))
        strength = M(M(2.75e4, iceVolumeCell[c]), glibc_exp(-M(20.0, S(1.0, iceAreaCell[c]))));
    }
    icePressure[c] = strength;
}
