// MPAS-Seaice v8.4.2: the column package's ITD thermodynamics
// (column_itd_thermodynamics -> colpkg_step_therm2): the linear remapping of
// the thickness distribution, new ice in open water, lateral melt and the
// ITD cleanup.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     1680-1976  column_itd_thermodynamics
//     5595-5741  get_cice_physics_tracer_array_category
//   src/core_seaice/column/ice_colpkg.F90:
//     2290-2539  colpkg_step_therm2
//   src/core_seaice/column/ice_colpkg_tracers.F90:
//     157-246    colpkg_compute_tracers
//   src/core_seaice/column/ice_therm_itd.F90:
//     69-666     linear_itd
//     678-741    fit_line
//     750-819    update_vertical_tracers
//     829-985    lateral_melt
//     1007-1503  add_new_ice
//   src/core_seaice/column/ice_itd.F90:
//     53-81      aggregate_area
//     89-277     rebin
//     291-335    reduce_area
//     344-658    shift_ice
//     747-1010   cleanup_itd
//     1019-1323  zap_small_areas
//     1327-1409  zap_snow
//     1413-1546  zap_snow_temperature
//   src/core_seaice/column/ice_zbgc.F90:
//     29-253     add_new_ice_bgc
//     259-324    lateral_melt_bgc
//     332-521    adjust_tracer_profile
//   src/core_seaice/column/ice_zbgc_shared.F90:
//     370-524    regrid_stationary
// (William H. Lipscomb, Elizabeth C. Hunke, C. M. Bitz, Adrian K. Turner,
// Nicole Jeffery and the CICE Consortium). Copyright (c) 2013-2019, Los
// Alamos National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// Appended to column_v842.cu (its rounding macros, constants, layout and the
// mushy liquidus and enthalpy) and column_zbgc_shared_v842.cu (remap_zbgc,
// calculate_Tin_from_qin). The column package's l_conservation_check is
// .false. (no conservation checks), print_diags .false.; of the
// biogeochemistry the brine height is carried (tr_brine: nt_fbri,
// tidx[TI_FBRI] >= 0) and the vertical salinity (tidx[TI_BGCS]): inert as
// v8.4.2 runs it (its solve_zsal is never set), solved under the opt-in
// correction zsal_solve (ITD_SOLVE_ZSAL: the zsal branches of add_new_ice,
// add_new_ice_bgc, lateral_melt_bgc and zap_small_areas); the rest, the
// aerosols and the zero-layer thermodynamics are refused by the host
// (column_itd_v842.py). x**2 is x*x.
//
// One thread is one owned cell. The category tracer array of each owned cell
// (set_cice_tracer_array_category: seaice_gather_tracer_v842) is
// trcrn[(iCell*ncat + n)*nTracers + it], scattered back afterwards
// (seaice_scatter_category_tracer_v842). Tracer indices arrive 0-based (-1
// when absent); category and donor indices below are Fortran's, 1-based.
// COL_NILYR (prepended) is nIceLayers; the one snow layer is nSnowLayers.

#define ITD_MAXCAT 32
#define ITD_NSLYR 1
#define ICE_REF_SAL 4.0
#define P001 0.001
#define HS_MIN_ITD 1.0e-4
#define TMIN_ITD (-100.0)
#define HFRAZILMIN 0.05
#define BIGNUM 1.0e30
#define PHI_INIT 0.75
#define DSIN0_FRAZIL 3.0

// abort codes: the stop labels in column_itd_v842.STOP_LABELS
#define ITD_OK 0
#define ITD_NEG_DAICE 1      // shift_ice: negative daice
#define ITD_NEG_DVICE 2      // shift_ice: negative dvice
#define ITD_DAICE_GT 3       // shift_ice: daice > aicen
#define ITD_DVICE_GT 4       // shift_ice: dvice > vicen
#define ITD_AICE_BOUNDS 5    // aggregate ice area out of bounds
#define ITD_ZAP_NEGATIVE 6   // Zap ice: negative ice area
#define ITD_ZAP_EXCESS 7     // Zap ice: excess ice area
#define ITD_ZSAL_REMAP 8     // regrid_stationary's remap_zbgc: ice: remap_layers_bgc error
#define ITD_REMAPPED 256     // status bit: linear_itd ran (hin_max(ncat) = 999.9)

// the tracer indices (tidx), 0-based, -1 when the tracer is off
#define TI_TSFC 0
#define TI_QICE 1
#define TI_QSNO 2
#define TI_SICE 3
#define TI_IAGE 4
#define TI_FY 5
#define TI_ALVL 6
#define TI_VLVL 7
#define TI_APND 8
#define TI_HPND 9
#define TI_FBRI 10
#define TI_BGCS 11
#define TI_COUNT 12

// flags
#define ITD_TR_IAGE 1
#define ITD_TR_FY 2
#define ITD_TR_LVL 4
#define ITD_TR_POND_CESM 8
#define ITD_TR_POND_LVL 16
#define ITD_TR_POND_TOPO 32
#define ITD_UPDATE_OCN_F 64
#define ITD_HEAT_CAPACITY 128
#define ITD_SOLVE_ZSAL 256     // zsal_solve: the column package's solve_zsal

#define ITD_RHOSI 940.0
#define ITD_SALT_LOSS 0.4
#define ITD_HS_SSL 0.040        // ice_colpkg_shared hs_ssl: the snow surface layer (m)

// 1/3 and 2/3 (p333 = c1/c3, p666 = c2/c3: gfortran folds each division)
__device__ __forceinline__ double p333() { return D(1.0, 3.0); }
__device__ __forceinline__ double p666() { return D(2.0, 3.0); }

struct ItdTracers {
    int nt;
    const int *depend;      // trcr_depend (parentIndex)
    const double *base;     // trcr_base, C-order (3, nt)
    const int *nstrata;     // n_trcr_strata
    const int *strata;      // nt_strata, C-order (2, nt), 1-based
    int idx[TI_COUNT];
    int zsal;               // solve_zsal (ITD_SOLVE_ZSAL)
    int nblyr;              // nBioLayers
    const double *bgrid;    // biologyGrid: bgrid(k) is bgrid[k - 1]
    const double *cgrid;    // verticalGrid: cgrid(k) is cgrid[k - 1]
    // the vertical tracers (z_tracers) with the nonreactive tracer alone: nbtrcr
    // (0 or 1), its first tracer bio_index(1) (0-based: nblyr + 1 ice layers, then the
    // snow's two), interfaceBiologyGrid (igrid(k) is igrid[k]: 1-based), the
    // column package's zbgc_init_frac(1), and the cell's ocean concentration
    // (oceanBioConcentrationsUsed(1)) and flux (oceanBioFluxes(1, iCell))
    int nbio;
    int bio;
    const double *igrid;
    double init_frac;
    double ocean_bio;
    double *flux_bio;
};

// ice_itd.F90:53-81 aggregate_area
__device__ void itd_aggregate_area(int ncat, const double *aicen, double *aice, double *aice0)
{
    double s = 0.0;
    for (int n = 0; n < ncat; ++n) s = A(s, aicen[n]);
    *aice = s;
    *aice0 = col_max(S(1.0, s), 0.0);
}

// ice_colpkg_tracers.F90:157-246 colpkg_compute_tracers for one category
__device__ void itd_compute_tracers(const ItdTracers &T, const double *atrcrn, double aicen, double vicen,
                                    double vsnon, double *trcrn)
{
    const int nt = T.nt;
    for (int it = 0; it < nt; ++it) {
        double divisor[3] = {M(T.base[it], aicen), M(T.base[nt + it], vicen), M(T.base[2 * nt + it], vsnon)};
        if (T.depend[it] == 0) {
            if (aicen > PUNY) {
                trcrn[it] = D(atrcrn[it], aicen);
            } else {
                trcrn[it] = 0.0;
                if (it == T.idx[TI_TSFC]) trcrn[it] = TOCNFRZ;
            }
        } else {
            double work = 0.0;
            for (int k = 0; k < 3; ++k)
                if (divisor[k] > 0.0) work = D(atrcrn[it], divisor[k]);
            trcrn[it] = work;
            for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                int ntr = T.strata[itl * nt + it] - 1;
                if (trcrn[ntr] > 0.0) trcrn[it] = D(trcrn[it], trcrn[ntr]);
                else trcrn[it] = 0.0;
            }
            if (vicen <= 0.0 && it == T.idx[TI_FBRI]) trcrn[it] = 1.0;
        }
    }
}

// ice_itd.F90:344-658 shift_ice. donor, daice and dvice are per boundary
// (1..ncat-1, stored 0-based); donor values are 1-based categories. Returns
// the abort code.
__device__ int itd_shift_ice(const ItdTracers &T, int ncat, double *aicen, double *trcrn, double *vicen,
                             double *vsnon, double *hicen, const int *donor, double *daice, double *dvice,
                             double *atrcrn)
{
    const int nt = T.nt;
    for (int n = 0; n < ncat; ++n) {
        for (int it = 0; it < nt; ++it) {
            double w = M(trcrn[n * nt + it], A(A(M(T.base[it], aicen[n]), M(T.base[nt + it], vicen[n])),
                                               M(T.base[2 * nt + it], vsnon[n])));
            for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                int ntr = T.strata[itl * nt + it] - 1;
                w = M(w, trcrn[n * nt + ntr]);
            }
            atrcrn[n * nt + it] = w;
        }
    }
    int nd = 0;
    for (int n = 1; n <= ncat - 1; ++n) {
        bool daice_negative = false, dvice_negative = false, daice_greater_aicen = false,
             dvice_greater_vicen = false;
        if (donor[n - 1] > 0) {
            nd = donor[n - 1];
            if (daice[n - 1] < 0.0) {
                if (daice[n - 1] > -M(PUNY, aicen[nd - 1])) { daice[n - 1] = 0.0; dvice[n - 1] = 0.0; }
                else daice_negative = true;
            }
            if (dvice[n - 1] < 0.0) {
                if (dvice[n - 1] > -M(PUNY, vicen[nd - 1])) { daice[n - 1] = 0.0; dvice[n - 1] = 0.0; }
                else dvice_negative = true;
            }
            if (daice[n - 1] > M(aicen[nd - 1], S(1.0, PUNY))) {
                if (daice[n - 1] < M(aicen[nd - 1], A(1.0, PUNY))) { daice[n - 1] = aicen[nd - 1]; dvice[n - 1] = vicen[nd - 1]; }
                else daice_greater_aicen = true;
            }
            if (dvice[n - 1] > M(vicen[nd - 1], S(1.0, PUNY))) {
                if (dvice[n - 1] < M(vicen[nd - 1], A(1.0, PUNY))) { daice[n - 1] = aicen[nd - 1]; dvice[n - 1] = vicen[nd - 1]; }
                else dvice_greater_vicen = true;
            }
        }
        if (daice_negative && donor[n - 1] > 0 && daice[n - 1] <= -M(PUNY, aicen[nd - 1])) return ITD_NEG_DAICE;
        if (dvice_negative && donor[n - 1] > 0 && dvice[n - 1] <= -M(PUNY, vicen[nd - 1])) return ITD_NEG_DVICE;
        if (daice_greater_aicen && donor[n - 1] > 0) {
            nd = donor[n - 1];
            if (daice[n - 1] >= M(aicen[nd - 1], A(1.0, PUNY))) return ITD_DAICE_GT;
        }
        if (dvice_greater_vicen && donor[n - 1] > 0) {
            nd = donor[n - 1];
            if (dvice[n - 1] >= M(vicen[nd - 1], A(1.0, PUNY))) return ITD_DVICE_GT;
        }
        // a positive daice always has a donor (linear_itd and rebin zero both together)
        if (daice[n - 1] > 0.0 && donor[n - 1] > 0) {
            nd = donor[n - 1];
            double worka = D(daice[n - 1], aicen[nd - 1]);
            int nr = (nd == n) ? nd + 1 : n;
            aicen[nd - 1] = S(aicen[nd - 1], daice[n - 1]);
            aicen[nr - 1] = A(aicen[nr - 1], daice[n - 1]);
            vicen[nd - 1] = S(vicen[nd - 1], dvice[n - 1]);
            vicen[nr - 1] = A(vicen[nr - 1], dvice[n - 1]);
            double dvsnow = M(vsnon[nd - 1], worka);
            vsnon[nd - 1] = S(vsnon[nd - 1], dvsnow);
            vsnon[nr - 1] = A(vsnon[nr - 1], dvsnow);
            double workb = dvsnow;
            for (int it = 0; it < nt; ++it) {
                double datrcr = M(trcrn[(nd - 1) * nt + it],
                                  A(A(M(T.base[it], daice[n - 1]), M(T.base[nt + it], dvice[n - 1])), M(T.base[2 * nt + it], workb)));
                for (int itl = 0; itl < T.nstrata[it]; ++itl) {
                    int ntr = T.strata[itl * nt + it] - 1;
                    datrcr = M(datrcr, trcrn[(nd - 1) * nt + ntr]);
                }
                atrcrn[(nd - 1) * nt + it] = S(atrcrn[(nd - 1) * nt + it], datrcr);
                atrcrn[(nr - 1) * nt + it] = A(atrcrn[(nr - 1) * nt + it], datrcr);
            }
        }
    }
    for (int n = 0; n < ncat; ++n) {
        hicen[n] = aicen[n] > PUNY ? D(vicen[n], aicen[n]) : 0.0;
        itd_compute_tracers(T, atrcrn + n * nt, aicen[n], vicen[n], vsnon[n], trcrn + n * nt);
    }
    return ITD_OK;
}

// ice_itd.F90:89-277 rebin. Its shift_ice status is the last call's, which
// cleanup_itd never reads (zap_small_areas resets l_stop).
__device__ int itd_rebin(const ItdTracers &T, int ncat, double *aicen, double *trcrn, double *vicen,
                         double *vsnon, const double *hin_max, double *atrcrn)
{
    int stop = ITD_OK;
    int donor[ITD_MAXCAT];
    double daice[ITD_MAXCAT], dvice[ITD_MAXCAT], hicen[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) {
        donor[n] = 0;
        daice[n] = 0.0;
        dvice[n] = 0.0;
        hicen[n] = aicen[n] > PUNY ? D(vicen[n], aicen[n]) : 0.0;
    }
    if (aicen[0] > PUNY) {
        if (hicen[0] <= hin_max[0] && hin_max[0] > 0.0) {
            aicen[0] = D(vicen[0], hin_max[0]);
            hicen[0] = hin_max[0];
        }
    }
    for (int n = 1; n <= ncat - 1; ++n) {
        if (aicen[n - 1] > PUNY && hicen[n - 1] > hin_max[n]) {
            donor[n - 1] = n;
            daice[n - 1] = aicen[n - 1];
            dvice[n - 1] = vicen[n - 1];
            stop = itd_shift_ice(T, ncat, aicen, trcrn, vicen, vsnon, hicen, donor, daice, dvice, atrcrn);
            donor[n - 1] = 0;
            daice[n - 1] = 0.0;
            dvice[n - 1] = 0.0;
        }
    }
    for (int n = ncat - 1; n >= 1; --n) {
        if (aicen[n] > PUNY && hicen[n] <= hin_max[n]) {
            donor[n - 1] = n + 1;
            daice[n - 1] = aicen[n];
            dvice[n - 1] = vicen[n];
            stop = itd_shift_ice(T, ncat, aicen, trcrn, vicen, vsnon, hicen, donor, daice, dvice, atrcrn);
            donor[n - 1] = 0;
            daice[n - 1] = 0.0;
            dvice[n - 1] = 0.0;
        }
    }
    return stop;
}

// ice_therm_itd.F90:678-741 fit_line
__device__ void itd_fit_line(double aicen, double hice, double hbL, double hbR, double *g0, double *g1,
                             double *hL, double *hR)
{
    if (aicen > PUNY && S(hbR, hbL) > PUNY) {
        double l = hbL, r = hbR;
        double h13 = M(p333(), A(M(2.0, l), r));
        double h23 = M(p333(), A(l, M(2.0, r)));
        if (hice < h13) r = S(M(3.0, hice), M(2.0, l));
        else if (hice > h23) l = S(M(3.0, hice), M(2.0, r));
        double dhr = D(1.0, S(r, l));
        double wk1 = M(M(6.0, aicen), dhr);
        double wk2 = M(S(hice, l), dhr);
        *g0 = M(wk1, S(p666(), wk2));
        *g1 = M(M(M(2.0, dhr), wk1), S(wk2, 0.5));
        *hL = l;
        *hR = r;
    } else {
        *g0 = 0.0;
        *g1 = 0.0;
        *hL = 0.0;
        *hR = 0.0;
    }
}

// ice_therm_itd.F90:69-666 linear_itd (l_conservation_check is .false.).
// hin_max(ncat) is set to 999.9 by the caller. Returns the abort code.
__device__ int itd_linear_itd(const ItdTracers &T, int ncat, const double *hin_max, const double *aicen_init,
                              const double *vicen_init, double *aicen, double *trcrn, double *vicen,
                              double *vsnon, double *aice, double *aice0, double *fpond, int flags, double hi_min,
                              double *atrcrn)
{
    const int nt = T.nt;
    double hbnew[ITD_MAXCAT + 1], g0[ITD_MAXCAT], g1[ITD_MAXCAT], hL[ITD_MAXCAT], hR[ITD_MAXCAT];
    double hicen[ITD_MAXCAT], hicen_init[ITD_MAXCAT], dhicen[ITD_MAXCAT], daice[ITD_MAXCAT],
        dvice[ITD_MAXCAT];
    int donor[ITD_MAXCAT];
    const bool topo = flags & ITD_TR_POND_TOPO;
    for (int n = 0; n < ncat; ++n) {
        donor[n] = 0;
        daice[n] = 0.0;
        dvice[n] = 0.0;
    }
    bool remap_flag = true;
    for (int n = 0; n < ncat; ++n) {
        hicen_init[n] = aicen_init[n] > PUNY ? D(vicen_init[n], aicen_init[n]) : 0.0;
        if (aicen[n] > PUNY) {
            hicen[n] = D(vicen[n], aicen[n]);
            dhicen[n] = S(hicen[n], hicen_init[n]);
        } else {
            hicen[n] = 0.0;
            dhicen[n] = 0.0;
        }
    }
    hbnew[0] = hin_max[0];
    for (int n = 1; n <= ncat - 1; ++n) {
        double hi0 = hicen_init[n - 1], hi1 = hicen_init[n];
        if (hi0 > PUNY && hi1 > PUNY) {
            double slope = D(S(dhicen[n], dhicen[n - 1]), S(hi1, hi0));
            hbnew[n] = A(A(hin_max[n], dhicen[n - 1]), M(slope, S(hin_max[n], hi0)));
        } else if (hi0 > PUNY) {
            hbnew[n] = A(hin_max[n], dhicen[n - 1]);
        } else if (hi1 > PUNY) {
            hbnew[n] = A(hin_max[n], dhicen[n]);
        } else {
            hbnew[n] = hin_max[n];
        }
        if (aicen[n - 1] > PUNY && hicen[n - 1] >= hbnew[n]) remap_flag = false;
        else if (aicen[n] > PUNY && hicen[n] <= hbnew[n]) remap_flag = false;
        if (hbnew[n] > hin_max[n + 1]) remap_flag = false;
        if (hbnew[n] < hin_max[n - 1]) remap_flag = false;
    }
    if (aicen[ncat - 1] > PUNY) hbnew[ncat] = S(M(3.0, hicen[ncat - 1]), M(2.0, hbnew[ncat - 1]));
    else hbnew[ncat] = hin_max[ncat];
    hbnew[ncat] = col_max(hbnew[ncat], hin_max[ncat - 1]);

    if (remap_flag) {
        itd_fit_line(aicen[0], hicen_init[0], hbnew[0], hin_max[1], &g0[0], &g1[0], &hL[0], &hR[0]);
        if (aicen[0] > PUNY) {
            double dh0 = dhicen[0];
            if (dh0 < 0.0) {
                dh0 = col_min(-dh0, hin_max[1]);
                double etamax = S(col_min(dh0, hR[0]), hL[0]);
                if (etamax > 0.0) {
                    double x1 = etamax;
                    double x2 = M(M(0.5, etamax), etamax);
                    double da0 = A(M(g1[0], x2), M(g0[0], x1));
                    double damax = M(aicen[0], S(1.0, D(hicen[0], hicen_init[0])));
                    da0 = col_min(da0, damax);
                    hicen[0] = D(M(hicen[0], aicen[0]), S(aicen[0], da0));
                    aicen[0] = S(aicen[0], da0);
                    if (topo)
                        *fpond = S(*fpond, M(M(da0, trcrn[T.idx[TI_APND]]), trcrn[T.idx[TI_HPND]]));
                }
            } else {
                hbnew[0] = col_min(dh0, hin_max[1]);
            }
        }
        for (int n = 1; n <= ncat; ++n) {
            itd_fit_line(aicen[n - 1], hicen[n - 1], hbnew[n - 1], hbnew[n], &g0[n - 1], &g1[n - 1], &hL[n - 1],
                         &hR[n - 1]);
            donor[n - 1] = 0;
            daice[n - 1] = 0.0;
            dvice[n - 1] = 0.0;
        }
        for (int n = 1; n <= ncat - 1; ++n) {
            double etamin, etamax;
            if (hbnew[n] > hin_max[n]) {
                etamin = S(col_max(hin_max[n], hL[n - 1]), hL[n - 1]);
                etamax = S(col_min(hbnew[n], hR[n - 1]), hL[n - 1]);
                donor[n - 1] = n;
            } else {
                etamin = 0.0;
                etamax = S(col_min(hin_max[n], hR[n]), hL[n]);
                donor[n - 1] = n + 1;
            }
            if (etamax > etamin) {
                double x1 = S(etamax, etamin);
                double wk1 = M(etamin, etamin);
                double wk2 = M(etamax, etamax);
                double x2 = M(0.5, S(wk2, wk1));
                wk1 = M(wk1, etamin);
                wk2 = M(wk2, etamax);
                double x3 = M(p333(), S(wk2, wk1));
                int nd = donor[n - 1] - 1;
                daice[n - 1] = A(M(g1[nd], x2), M(g0[nd], x1));
                dvice[n - 1] = A(A(M(g1[nd], x3), M(g0[nd], x2)), M(daice[n - 1], hL[nd]));
            }
            int nd = donor[n - 1] - 1;
            if (daice[n - 1] < M(aicen[nd], PUNY)) {
                daice[n - 1] = 0.0;
                dvice[n - 1] = 0.0;
                donor[n - 1] = 0;
            }
            if (dvice[n - 1] < M(vicen[nd], PUNY)) {
                daice[n - 1] = 0.0;
                dvice[n - 1] = 0.0;
                donor[n - 1] = 0;
            }
            if (daice[n - 1] > M(aicen[nd], S(1.0, PUNY))) {
                daice[n - 1] = aicen[nd];
                dvice[n - 1] = vicen[nd];
            }
            if (dvice[n - 1] > M(vicen[nd], S(1.0, PUNY))) {
                daice[n - 1] = aicen[nd];
                dvice[n - 1] = vicen[nd];
            }
        }
        // maintain qsno negative definiteness
        const double rhosL = M(RHOS, lfresh());
        for (int n = 0; n < ncat; ++n)
            for (int k = 0; k < ITD_NSLYR; ++k)
                trcrn[n * nt + T.idx[TI_QSNO] + k] = A(trcrn[n * nt + T.idx[TI_QSNO] + k], rhosL);
        int stop = itd_shift_ice(T, ncat, aicen, trcrn, vicen, vsnon, hicen, donor, daice, dvice, atrcrn);
        if (stop) return stop;   // the snow enthalpies still shifted, as MPAS returns
        for (int n = 0; n < ncat; ++n)
            for (int k = 0; k < ITD_NSLYR; ++k)
                trcrn[n * nt + T.idx[TI_QSNO] + k] = S(trcrn[n * nt + T.idx[TI_QSNO] + k], rhosL);
        if (hi_min > 0.0 && aicen[0] > PUNY && hicen[0] < hi_min) {
            double da0 = M(aicen[0], S(1.0, D(hicen[0], hi_min)));
            aicen[0] = S(aicen[0], da0);
            hicen[0] = hi_min;
            if (topo) *fpond = S(*fpond, M(M(da0, trcrn[T.idx[TI_APND]]), trcrn[T.idx[TI_HPND]]));
        }
    }
    itd_aggregate_area(ncat, aicen, aice, aice0);
    return ITD_OK;
}

// ice_therm_itd.F90:750-819 update_vertical_tracers (mushy)
__device__ void itd_update_vertical_tracers(double *trc, double h1, double h2, double trc0)
{
    const double rnilyr = (double)COL_NILYR;
    double trc2[COL_NILYR];
    for (int k2 = 1; k2 <= COL_NILYR; ++k2) {
        double t2 = 0.0;
        double z2a = D(M((double)(k2 - 1), h2), rnilyr);
        double z2b = D(M((double)k2, h2), rnilyr);
        for (int k1 = 1; k1 <= COL_NILYR; ++k1) {
            double z1a = D(M((double)(k1 - 1), h1), rnilyr);
            double z1b = D(M((double)k1, h1), rnilyr);
            double overlap = col_max(S(col_min(z1b, z2b), col_max(z1a, z2a)), 0.0);
            t2 = A(t2, M(overlap, trc[k1 - 1]));
        }
        double overlap = col_max(S(col_min(h2, z2b), col_max(h1, z2a)), 0.0);
        t2 = A(t2, M(overlap, trc0));
        trc2[k2 - 1] = D(M(rnilyr, t2), h2);
    }
    for (int k = 0; k < COL_NILYR; ++k) trc[k] = trc2[k];
}

// ice_zbgc_shared.F90:370-524 regrid_stationary as adjust_tracer_profile calls
// it: n points c[1..n] (1-based) on grid (grid[k] is the grid's k-th point), no
// bottom melt or growth, its flux (fluxb) and the sums that feed only it
// dropped. The vertical salinity's call is nblyr - 1 intervals (n = nblyr) on
// bgrid(2:nblyr+1); a vertical tracer's nblyr intervals (n = nblyr + 1) on
// igrid. Returns ITD_ZSAL_REMAP on remap_zbgc's stop.
__device__ int itd_regrid_points(double *c, int n, const double *grid, double hbri_old, double hbri,
                                 double top_conc)
{
    double trtmp0[ZB_NBMAX + 3], trtmp[ZB_NBMAX + 3];
    for (int k = 0; k < ZB_NBMAX + 3; ++k) {
        trtmp0[k] = 0.0;
        trtmp[k] = 0.0;
    }
    if (hbri_old > 0.0)
        for (int k = 1; k <= n; ++k) trtmp0[n + 1 - k] = D(c[k], hbri_old);   // reverse order
    if (hbri > hbri_old) {
        // nr = 1, hice = htemp = hbri, hbio = hbri_old
        const double htemp = hbri;
        if (htemp > 0.0) {
            if (remap_zbgc(trtmp0 + 1, 1, n, n, htemp, hbri_old, grid, grid, top_conc, trtmp))
                return ITD_ZSAL_REMAP;
            for (int k = 0; k < ZB_NBMAX + 3; ++k) trtmp0[k] = 0.0;
            for (int k = 1; k <= n; ++k) trtmp0[n + 1 - k] = trtmp[k];
            for (int k = 1; k <= n; ++k) c[k] = M(trtmp0[k], htemp);
        }
    }
    return ITD_OK;
}

__device__ int itd_regrid_stationary(const ItdTracers &T, double *c, double hbri_old, double hbri, double top_conc)
{
    return itd_regrid_points(c, T.nblyr, T.bgrid, hbri_old, hbri, top_conc);
}

// ice_zbgc.F90:332-521 adjust_tracer_profile on one category's tracers tr:
// the vertical salinity under solve_zsal and the vertical tracer (nbtrcr =
// T.nbio). location 1: frazil at the bottom (regrid_stationary); 0: frazil
// throughout. Under solve_zsal the ice salinity is remapped from the new
// vertical salinity; that remap_zbgc's stop is not checked (its receivers
// stay 0). Returns regrid_stationary's stop.
__device__ int itd_adjust_tracer_profile(const ItdTracers &T, double aicen, double vbrin, double vicen, double *tr,
                                         double vtmp, double vsurp, double sss, int location)
{
    const int nbl = T.nblyr;
    double *zsal = tr + T.idx[TI_BGCS];
    double *zbio = tr + T.bio;
    double src[ZB_NBMAX], out[COL_NILYR + 2];
    if (!(vbrin > 0.0)) return ITD_OK;
    if (location == 1) {
        const double hbri = vbrin, hbri_old = vtmp;
        double cs[ZB_NBMAX + 2];
        if (T.zsal) {
            const double top_conc = M(sss, ITD_SALT_LOSS);
            for (int k = 1; k <= nbl; ++k) cs[k] = M(zsal[k - 1], hbri_old);
            int stop = itd_regrid_stationary(T, cs, hbri_old, hbri, top_conc);
            if (stop) return stop;
            for (int k = 1; k <= nbl; ++k) {
                zsal[k - 1] = D(cs[k], hbri);
                src[k - 1] = zsal[k - 1];
            }
        }
        if (T.nbio) {
            const double top_conc = M(T.ocean_bio, T.init_frac);
            for (int k = 1; k <= nbl + 1; ++k) cs[k] = M(zbio[k - 1], hbri_old);
            int stop = itd_regrid_points(cs, nbl + 1, T.igrid, hbri_old, hbri, top_conc);
            if (stop) return stop;
            for (int k = 1; k <= nbl + 1; ++k) zbio[k - 1] = D(cs[k], hbri);
        }
    } else {
        for (int k = 1; k <= nbl + 1; ++k) {
            if (T.zsal && k < nbl + 1) {
                zsal[k - 1] = D(A(M(zsal[k - 1], vtmp), M(M(sss, ITD_SALT_LOSS), vsurp)), vbrin);
                src[k - 1] = zsal[k - 1];
            }
            if (T.nbio)
                zbio[k - 1] = D(A(M(zbio[k - 1], vtmp), M(M(T.ocean_bio, T.init_frac), vsurp)), vbrin);
        }
    }
    if (T.zsal) {
        double hinS_new = 0.0, hin = 0.0;
        if (aicen > 0.0) {
            hinS_new = D(vbrin, aicen);
            hin = D(vicen, aicen);
        }
        for (int k = 0; k < COL_NILYR + 2; ++k) out[k] = 0.0;
        (void)remap_zbgc(src, 1, COL_NILYR, nbl, hin, hinS_new, T.cgrid, T.bgrid, MIN_SALIN, out);
        for (int k = 1; k <= COL_NILYR; ++k) tr[T.idx[TI_SICE] + k - 1] = out[k];
    }
    return ITD_OK;
}

// ice_zbgc.F90:29-253 add_new_ice_bgc with the brine height
// (l_conservation_check .false.), the vertical salinity under solve_zsal and
// the vertical tracer (nbtrcr = T.nbio): the frazil's tracer taken from the
// ocean flux, then adjust_tracer_profile for nltrcr > 0 (v8.4.2 counts the
// vertical salinity in nltrcr unsolved too, where adjust_tracer_profile
// changes nothing); under solve_zsal the first category's surface temperature
// from its new top-layer enthalpy and salinity under no snow. aicen_init and
// vicen_init are add_new_ice's copies at its entry, aicen and vicen the
// updated state; vi0_init is add_new_ice's first frazil volume, vi0new its
// final one and vsnon1 the first category's snow. Returns
// regrid_stationary's stop.
__device__ int itd_add_new_ice_bgc(const ItdTracers &T, int ncat, double dt, const double *aicen_init,
                                   const double *vicen_init, const double *aicen, const double *vicen, double vsnon1,
                                   double vi0_init, double vi0new, double hsurp, double sss, double *trcrn)
{
    const int nt = T.nt;
    const int fb = T.idx[TI_FBRI];
    const bool layered = T.zsal || T.nbio;
    double vbrin[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) vbrin[n] = M(trcrn[n * nt + fb], vicen_init[n]);
    if (T.nbio) *T.flux_bio = S(*T.flux_bio, M(M(D(vi0_init, dt), T.ocean_bio), T.init_frac));
    for (int n = 0; n < ncat; ++n) {
        if (hsurp > 0.0) {
            double vtmp = vbrin[n];
            double vsurp = M(hsurp, aicen_init[n]);
            vbrin[n] = A(vbrin[n], vsurp);
            double vice_new = A(vicen_init[n], vsurp);
            if (vicen[n] > 0.0) trcrn[n * nt + fb] = D(vbrin[n], vicen[n]);
            else trcrn[n * nt + fb] = 1.0;
            if (layered) {
                int stop = itd_adjust_tracer_profile(T, aicen_init[n], vbrin[n], vice_new, trcrn + n * nt, vtmp,
                                                     vsurp, sss, 1);
                if (stop) return stop;
            }
        }
    }
    if (vi0new > 0.0) {
        double vbri1 = vbrin[0];
        vbrin[0] = A(vbrin[0], vi0new);
        if (vicen[0] > 0.0) trcrn[fb] = D(vbrin[0], vicen[0]);
        else trcrn[fb] = 1.0;
        if (layered) {
            int stop = itd_adjust_tracer_profile(T, aicen[0], vbrin[0], vicen[0], trcrn, vbri1, vi0new, sss, 0);
            if (stop) return stop;
            if (T.zsal && vsnon1 <= 0.0) {
                double Tmlts = M(-trcrn[T.idx[TI_SICE]], DEPRESST);
                trcrn[T.idx[TI_TSFC]] = bg_Tin_from_qin(trcrn[T.idx[TI_QICE]], Tmlts);
            }
        }
    }
    return ITD_OK;
}

// ice_therm_itd.F90:1007-1503 add_new_ice (no conservation check, no
// aerosols; of the biogeochemistry the brine height and the vertical
// salinity, itd_add_new_ice_bgc). Returns add_new_ice_bgc's stop.
__device__ int itd_add_new_ice(const ItdTracers &T, int ncat, double dt, const double *hin_max, int ktherm,
                               double *aicen, double *trcrn, double *vicen, double vsnon1, double *aice0,
                               double aice, double frzmlt, double *frazil, double *frz_onset, double yday,
                               int flags, double *fresh, double *fsalt, double Tf, double sss,
                               const double *salinz, double *frazil_diag, double *fzsal)
{
    const int nt = T.nt;
    const int *ix = T.idx;
    const double rnilyr = (double)COL_NILYR;
    (void)rnilyr;
    double aicen_init[ITD_MAXCAT], vicen_init[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) {
        aicen_init[n] = aicen[n];
        vicen_init[n] = vicen[n];
    }
    double hi0max = ncat > 1 ? M(hin_max[1], 0.9) : BIGNUM;
    double Sprofile[COL_NILYR];
    double Si0new = 0.0, qi0new;
    if (ktherm == 2) {
        if (sss > M(2.0, DSIN0_FRAZIL)) Si0new = S(sss, DSIN0_FRAZIL);
        else Si0new = D(M(sss, sss), M(4.0, DSIN0_FRAZIL));
        for (int k = 0; k < COL_NILYR; ++k) Sprofile[k] = Si0new;
        double Ti = col_min(liquidus_temperature_mush(D(Si0new, PHI_INIT)), -P1);
        qi0new = enthalpy_mush(Ti, Si0new);
    } else {
        for (int k = 0; k < COL_NILYR; ++k) Sprofile[k] = salinz[k];
        qi0new = -M(RHOI, lfresh());
    }
    double fnew = col_max(frzmlt, 0.0);
    double vi0new = -D(M(fnew, dt), qi0new);
    const double vi0_init = vi0new;
    *frazil = vi0new;
    if (T.zsal) *fzsal = S(*fzsal, M(M(M(D(M(ITD_RHOSI, vi0new), dt), P001), sss), ITD_SALT_LOSS));
    if (*frazil > PUNY && *frz_onset < PUNY) *frz_onset = yday;
    if (flags & ITD_UPDATE_OCN_F) {
        if (ktherm <= 1) {
            double dfresh = -D(M(RHOI, vi0new), dt);
            double dfsalt = M(M(ICE_REF_SAL, P001), dfresh);
            *fresh = A(*fresh, dfresh);
            *fsalt = A(*fsalt, dfsalt);
        }
    } else if (ktherm == 2) {
        double vi0tmp = D(M(fnew, dt), M(RHOI, lfresh()));
        double dfresh = -D(M(RHOI, S(vi0new, vi0tmp)), dt);
        double dfsalt = M(M(ICE_REF_SAL, P001), dfresh);
        *fresh = A(*fresh, dfresh);
        *fsalt = A(*fsalt, dfsalt);
        *frazil_diag = S(*frazil, vi0tmp);
    }
    double hsurp = 0.0, ai0new = 0.0;
    if (vi0new > 0.0) {
        if (*aice0 > PUNY) {
            double hi0new = col_max(D(vi0new, *aice0), HFRAZILMIN);
            if (hi0new > hi0max && A(*aice0, PUNY) < 1.0) {
                hi0new = hi0max;
                ai0new = *aice0;
                double vsurp = S(vi0new, M(ai0new, hi0new));
                hsurp = D(vsurp, aice);
                vi0new = M(ai0new, hi0new);
            } else {
                ai0new = D(vi0new, hi0new);
            }
        } else {
            hsurp = D(vi0new, aice);
            vi0new = 0.0;
        }
    }
    if (hsurp > 0.0) {
        for (int n = 0; n < ncat; ++n) {
            double *tr = trcrn + n * nt;
            double vsurp = M(hsurp, aicen[n]);
            double vtmp = A(vicen[n], vsurp);
            if ((flags & ITD_TR_IAGE) && vtmp > PUNY)
                tr[ix[TI_IAGE]] = D(A(M(tr[ix[TI_IAGE]], vicen[n]), M(dt, vsurp)), vtmp);
            if ((flags & ITD_TR_LVL) && vicen[n] > PUNY)
                tr[ix[TI_VLVL]] = D(A(M(tr[ix[TI_VLVL]], vicen[n]), M(tr[ix[TI_ALVL]], vsurp)), vtmp);
            vicen[n] = vtmp;
            if (ktherm == 2) {
                vsurp = M(hsurp, aicen[n]);
                vtmp = S(vicen[n], vsurp);
                if (vicen[n] > 0.0) {
                    itd_update_vertical_tracers(tr + ix[TI_QICE], vtmp, vicen[n], qi0new);
                    itd_update_vertical_tracers(tr + ix[TI_SICE], vtmp, vicen[n], Si0new);
                }
            } else {
                for (int k = 0; k < COL_NILYR; ++k) {
                    vsurp = M(hsurp, aicen[n]);
                    vtmp = S(vicen[n], vsurp);
                    if (vicen[n] > 0.0) {
                        tr[ix[TI_QICE] + k] = D(A(M(tr[ix[TI_QICE] + k], vtmp), M(qi0new, vsurp)), vicen[n]);
                        if (!T.zsal)
                            tr[ix[TI_SICE] + k] = D(A(M(tr[ix[TI_SICE] + k], vtmp), M(Sprofile[k], vsurp)),
                                                    vicen[n]);
                    }
                }
            }
        }
    }
    if (vi0new > 0.0) {
        double *tr = trcrn;
        double area1 = aicen[0];
        double vice1 = vicen[0];
        aicen[0] = A(aicen[0], ai0new);
        *aice0 = S(*aice0, ai0new);
        vicen[0] = A(vicen[0], vi0new);
        tr[ix[TI_TSFC]] = D(A(M(tr[ix[TI_TSFC]], area1), M(Tf, ai0new)), aicen[0]);
        tr[ix[TI_TSFC]] = col_min(tr[ix[TI_TSFC]], 0.0);
        if (flags & ITD_TR_FY) {
            tr[ix[TI_FY]] = D(A(M(tr[ix[TI_FY]], area1), ai0new), aicen[0]);
            tr[ix[TI_FY]] = col_min(tr[ix[TI_FY]], 1.0);
        }
        if (vicen[0] > PUNY) {
            if (flags & ITD_TR_IAGE) tr[ix[TI_IAGE]] = D(A(M(tr[ix[TI_IAGE]], vice1), M(dt, vi0new)), vicen[0]);
            double alvl = 0.0;
            if (flags & ITD_TR_LVL) {
                alvl = tr[ix[TI_ALVL]];
                tr[ix[TI_ALVL]] = D(A(M(tr[ix[TI_ALVL]], area1), ai0new), aicen[0]);
                tr[ix[TI_VLVL]] = D(A(M(tr[ix[TI_VLVL]], vice1), vi0new), vicen[0]);
            }
            if ((flags & ITD_TR_POND_CESM) || (flags & ITD_TR_POND_TOPO)) {
                tr[ix[TI_APND]] = D(M(tr[ix[TI_APND]], area1), aicen[0]);
            } else if (flags & ITD_TR_POND_LVL) {
                // level ponds need the level-ice tracer (column_v842 refuses them without it)
                if (tr[ix[TI_ALVL]] > PUNY)
                    tr[ix[TI_APND]] = D(M(M(tr[ix[TI_APND]], alvl), area1), M(tr[ix[TI_ALVL]], aicen[0]));
            }
        }
        for (int k = 0; k < COL_NILYR; ++k) {
            if (vicen[0] > 0.0) {
                tr[ix[TI_QICE] + k] = D(A(M(tr[ix[TI_QICE] + k], vice1), M(qi0new, vi0new)), vicen[0]);
                if (!T.zsal)
                    tr[ix[TI_SICE] + k] = D(A(M(tr[ix[TI_SICE] + k], vice1), M(Sprofile[k], vi0new)), vicen[0]);
            }
        }
    }
    // tr_brine .or. nbtrcr > 0 (ice_therm_itd.F90:1491)
    if (ix[TI_FBRI] >= 0)
        return itd_add_new_ice_bgc(T, ncat, dt, aicen_init, vicen_init, aicen, vicen, vsnon1, vi0_init, vi0new,
                                   hsurp, sss, trcrn);
    return ITD_OK;
}

// ice_therm_itd.F90:829-985 lateral_melt (no aerosols; of the
// biogeochemistry, the vertical tracer's snow and lateral_melt_bgc: the
// vertical salinity under solve_zsal and the vertical tracer's ice layers)
__device__ void itd_lateral_melt(const ItdTracers &T, double dt, int ncat, double *fpond, double *fresh,
                                 double *fsalt, double *fhocn, double rside, double *meltl, double *aicen,
                                 double *vicen, double *vsnon, const double *trcrn, int flags, double *fzsal)
{
    const int nt = T.nt;
    const int *ix = T.idx;
    if (!(rside > 0.0)) return;
    double vicen_init[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) {
        const double *tr = trcrn + n * nt;
        double dfresh = D(M(A(M(RHOS, vsnon[n]), M(RHOI, vicen[n])), rside), dt);
        double dfsalt = D(M(M(M(M(RHOI, vicen[n]), ICE_REF_SAL), P001), rside), dt);
        *fresh = A(*fresh, dfresh);
        *fsalt = A(*fsalt, dfsalt);
        if (flags & ITD_TR_POND_TOPO) {
            double dfpond = M(M(M(aicen[n], tr[ix[TI_APND]]), tr[ix[TI_HPND]]), rside);
            *fpond = S(*fpond, dfpond);
        }
        *meltl = A(*meltl, M(vicen[n], rside));
        vicen_init[n] = vicen[n];
        aicen[n] = M(aicen[n], S(1.0, rside));
        vicen[n] = M(vicen[n], S(1.0, rside));
        vsnon[n] = M(vsnon[n], S(1.0, rside));
        for (int k = 0; k < COL_NILYR; ++k) {
            double dfhocn = D(M(D(M(tr[ix[TI_QICE] + k], rside), dt), vicen[n]), (double)COL_NILYR);
            *fhocn = A(*fhocn, dfhocn);
        }
        for (int k = 0; k < ITD_NSLYR; ++k) {
            double dfhocn = D(M(D(M(tr[ix[TI_QSNO] + k], rside), dt), vsnon[n]), (double)ITD_NSLYR);
            *fhocn = A(*fhocn, dfhocn);
        }
        if (T.nbio) {
            // the snow tracers, on the melted category's area and snow
            const double dvssl = col_min(M(0.5, vsnon[n]), M(ITD_HS_SSL, aicen[n]));
            const double dvint = S(vsnon[n], dvssl);
            const int sn = T.bio + T.nblyr + 1;
            *T.flux_bio = A(*T.flux_bio, D(M(A(M(tr[sn], dvssl), M(tr[sn + 1], dvint)), rside), dt));
        }
    }
    // ice_zbgc.F90:259-324 lateral_melt_bgc (solve_zsal .or. z_tracers; nbltrcr = nbtrcr)
    if (T.zsal) {
        const double zspace = D(1.0, (double)T.nblyr);
        for (int n = 0; n < ncat; ++n) {
            const double *tr = trcrn + n * nt;
            for (int k = 0; k < T.nblyr; ++k)
                *fzsal = A(*fzsal, D(M(M(M(M(M(M(ITD_RHOSI, tr[ix[TI_FBRI]]), vicen_init[n]), P001), zspace),
                                         tr[ix[TI_BGCS] + k]), rside), dt));
        }
    }
    if (T.nbio) {
        const double zspace = D(1.0, (double)T.nblyr);
        for (int n = 0; n < ncat; ++n) {
            const double *tr = trcrn + n * nt;
            for (int k = 0; k <= T.nblyr; ++k)
                *T.flux_bio = A(*T.flux_bio, D(M(M(M(M(tr[ix[TI_FBRI]], vicen_init[n]), zspace), tr[T.bio + k]),
                                                 rside), dt));
        }
    }
}

// ice_itd.F90:291-335 reduce_area
__device__ void itd_reduce_area(double hin_max, double *aicen, double *vicen, double aicen_init,
                                double vicen_init)
{
    double hi0 = 0.0;
    if (aicen_init > 0.0) hi0 = D(vicen_init, aicen_init);
    double hi1 = 0.0;
    if (*aicen > 0.0) hi1 = D(*vicen, *aicen);
    if (hi1 <= hin_max && hin_max > 0.0) {
        *aicen = D(*vicen, hin_max);
        hi1 = hin_max;
    }
    if (*aicen > 0.0) {
        double dhi = S(hi1, hi0);
        if (dhi < 0.0) {
            hi1 = D(*vicen, *aicen);
            *aicen = D(M(2.0, *vicen), A(hi1, hi0));
        }
    }
}

// ice_itd.F90:1327-1409 zap_snow (no aerosols) on category n's tracers; the
// vertical tracer's snow into dflux_bio on aicen, the category's area as the
// caller passes it (zap_small_areas zeroes it first)
__device__ void itd_zap_snow(const ItdTracers &T, double dt, double *tr, double *vsnon, double *dfresh,
                             double *dfhocn, double aicen, double *dflux_bio)
{
    if (T.nbio) {
        const double dvssl = col_min(M(0.5, *vsnon), M(ITD_HS_SSL, aicen));
        const double dvint = S(*vsnon, dvssl);
        const int sn = T.bio + T.nblyr + 1;
        *dflux_bio = A(*dflux_bio, D(A(M(tr[sn], dvssl), M(tr[sn + 1], dvint)), dt));
    }
    for (int k = 0; k < ITD_NSLYR; ++k) {
        double xtmp = D(M(D(tr[T.idx[TI_QSNO] + k], dt), *vsnon), (double)ITD_NSLYR);
        *dfhocn = A(*dfhocn, xtmp);
        tr[T.idx[TI_QSNO] + k] = 0.0;
    }
    double xtmp = D(M(RHOS, *vsnon), dt);
    *dfresh = A(*dfresh, xtmp);
    *vsnon = 0.0;
}

// ice_itd.F90:1019-1323 zap_small_areas (no aerosols; tr_brine resets the
// brine fraction to 1; solve_zsal's salt flux dfzsal; the vertical tracer's
// zapped ice layers, zap_small_bgc, and snow into dflux_bio). Returns the
// abort code.
__device__ int itd_zap_small_areas(const ItdTracers &T, double dt, int ncat, double *aice, double *aice0,
                                   double *aicen, double *trcrn, double *vicen, double *vsnon, double *dfpond,
                                   double *dfresh, double *dfsalt, double *dfhocn, int flags, int *first_ice,
                                   double *dfzsal, double *dflux_bio)
{
    const int nt = T.nt;
    const int *ix = T.idx;
    const bool topo = flags & ITD_TR_POND_TOPO;
    for (int n = 0; n < ncat; ++n) {
        double *tr = trcrn + n * nt;
        if (aicen[n] < -PUNY) {
            return ITD_ZAP_NEGATIVE;
        } else if (fabs(aicen[n]) != 0.0 && fabs(aicen[n]) <= PUNY) {
            if (topo) {
                double xtmp = M(M(aicen[n], tr[ix[TI_APND]]), tr[ix[TI_HPND]]);
                *dfpond = S(*dfpond, xtmp);
            }
            if (T.zsal)
                for (int it = 0; it < T.nblyr; ++it) {
                    double xtmp = D(D(M(M(M(M(ITD_RHOSI, tr[ix[TI_FBRI]]), vicen[n]), P001), tr[ix[TI_BGCS] + it]),
                                      (double)T.nblyr), dt);
                    *dfzsal = A(*dfzsal, xtmp);
                }
            if (T.nbio) {
                // ice_zbgc_shared.F90:339-364 zap_small_bgc on the brine's layer volumes
                const int blevels = T.nblyr + 1;
                const double bv = M(D(vicen[n], (double)T.nblyr), tr[ix[TI_FBRI]]);
                for (int k = 1; k <= blevels; ++k) {
                    const double bvol = (k == 1 || k == blevels) ? M(0.5, bv) : bv;
                    *dflux_bio = A(*dflux_bio, D(M(tr[T.bio + k - 1], bvol), dt));
                }
            }
            for (int k = 0; k < COL_NILYR; ++k) {
                double xtmp = D(M(D(tr[ix[TI_QICE] + k], dt), vicen[n]), (double)COL_NILYR);
                *dfhocn = A(*dfhocn, xtmp);
                tr[ix[TI_QICE] + k] = 0.0;
            }
            double xtmp = D(M(RHOI, vicen[n]), dt);
            *dfresh = A(*dfresh, xtmp);
            xtmp = D(M(M(M(RHOI, vicen[n]), ICE_REF_SAL), P001), dt);
            *dfsalt = A(*dfsalt, xtmp);
            *aice0 = A(*aice0, aicen[n]);
            aicen[n] = 0.0;
            vicen[n] = 0.0;
            tr[ix[TI_TSFC]] = TOCNFRZ;
            itd_zap_snow(T, dt, tr, &vsnon[n], dfresh, dfhocn, aicen[n], dflux_bio);
            for (int it = 1; it < nt; ++it) tr[it] = it == ix[TI_FBRI] ? 1.0 : 0.0;   // tr_brine: fbri = 1
            first_ice[n] = 1;
        }
    }
    if (*aice > A(1.0, PUNY)) {
        return ITD_ZAP_EXCESS;
    } else if (*aice > 1.0 && *aice < A(1.0, PUNY)) {
        for (int n = 0; n < ncat; ++n) {
            double *tr = trcrn + n * nt;
            double excess = S(*aice, 1.0);
            if (topo) {
                double xtmp = D(M(M(M(aicen[n], tr[ix[TI_APND]]), tr[ix[TI_HPND]]), excess), *aice);
                *dfpond = S(*dfpond, xtmp);
            }
            for (int k = 0; k < COL_NILYR; ++k) {
                double xtmp = D(D(M(D(M(tr[ix[TI_QICE] + k], vicen[n]), (double)COL_NILYR), excess), *aice), dt);
                *dfhocn = A(*dfhocn, xtmp);
            }
            for (int k = 0; k < ITD_NSLYR; ++k) {
                double xtmp = D(D(M(D(M(tr[ix[TI_QSNO] + k], vsnon[n]), (double)ITD_NSLYR), excess), *aice), dt);
                *dfhocn = A(*dfhocn, xtmp);
            }
            double xtmp = D(D(M(A(M(RHOI, vicen[n]), M(RHOS, vsnon[n])), excess), *aice), dt);
            *dfresh = A(*dfresh, xtmp);
            xtmp = D(D(M(M(M(M(RHOI, vicen[n]), ICE_REF_SAL), P001), excess), *aice), dt);
            *dfsalt = A(*dfsalt, xtmp);
            if (T.zsal) {
                const double fbri = tr[ix[TI_FBRI]];
                for (int k = 0; k < T.nblyr; ++k) {
                    xtmp = D(D(M(M(D(M(M(M(ITD_RHOSI, fbri), vicen[n]), P001), (double)T.nblyr),
                                     tr[ix[TI_BGCS] + k]), excess), *aice), dt);
                    *dfzsal = A(*dfzsal, xtmp);
                }
                if (vicen[n] > M(vicen[n], fbri)) {
                    xtmp = D(M(M(M(D(M(S(vicen[n], M(vicen[n], fbri)), excess), *aice), P001), ITD_RHOSI),
                               MIN_SALIN), dt);
                    *dfzsal = A(*dfzsal, xtmp);
                }
            }
            aicen[n] = M(aicen[n], D(1.0, *aice));
            vicen[n] = M(vicen[n], D(1.0, *aice));
            vsnon[n] = M(vsnon[n], D(1.0, *aice));
        }
        *aice = 1.0;
        *aice0 = 0.0;
    }
    return ITD_OK;
}

// ice_itd.F90:1413-1546 zap_snow_temperature; counts its warnings (one per
// out-of-bounds snow layer)
__device__ void itd_zap_snow_temperature(const ItdTracers &T, double dt, int ncat, int heat_capacity,
                                         const double *aicen, double *trcrn, double *vsnon, double *dfresh,
                                         double *dfhocn, int *nwarn, double *dflux_bio)
{
    const int nt = T.nt;
    const double rnslyr = (double)ITD_NSLYR;
    for (int n = 0; n < ncat; ++n) {
        double *tr = trcrn + n * nt;
        bool l_zap = false;
        if (aicen[n] > PUNY) {
            double hsn = D(vsnon[n], aicen[n]);
            for (int k = 0; k < ITD_NSLYR; ++k) {
                double zqsn, Tmax;
                if (hsn > HS_MIN_ITD && heat_capacity) {
                    zqsn = tr[T.idx[TI_QSNO] + k];
                    Tmax = -D(M(M(zqsn, PUNY), rnslyr), M(M(RHOS, CP_ICE), vsnon[n]));
                } else {
                    zqsn = -M(RHOS, lfresh());
                    Tmax = PUNY;
                }
                double zTsn = D(A(lfresh(), D(zqsn, RHOS)), CP_ICE);
                if (zTsn < TMIN_ITD || zTsn > Tmax) {
                    l_zap = true;
                    ++*nwarn;
                }
            }
        }
        if (l_zap) itd_zap_snow(T, dt, tr, &vsnon[n], dfresh, dfhocn, aicen[n], dflux_bio);
    }
}

// ice_itd.F90:747-1010 cleanup_itd (limit_aice; no zerolayer_check: the zero
// layer is refused). Returns the abort code.
__device__ int itd_cleanup_itd(const ItdTracers &T, double dt, int ncat, const double *hin_max, double *aicen,
                               double *trcrn, double *vicen, double *vsnon, double *aice0, double *aice,
                               int flags, int *first_ice, double *fpond, double *fresh, double *fsalt,
                               double *fhocn, double *atrcrn, int *nwarn, double *fzsal)
{
    double dfpond = 0.0, dfresh = 0.0, dfsalt = 0.0, dfhocn = 0.0, dfzsal = 0.0, dflux_bio = 0.0;
    itd_aggregate_area(ncat, aicen, aice, aice0);
    if (*aice > A(1.0, PUNY) || *aice < -PUNY) return ITD_AICE_BOUNDS;
    if (*aice > PUNY) itd_rebin(T, ncat, aicen, trcrn, vicen, vsnon, hin_max, atrcrn);
    int stop = itd_zap_small_areas(T, dt, ncat, aice, aice0, aicen, trcrn, vicen, vsnon, &dfpond, &dfresh,
                                   &dfsalt, &dfhocn, flags, first_ice, &dfzsal, &dflux_bio);
    if (stop) return stop;
    itd_zap_snow_temperature(T, dt, ncat, flags & ITD_HEAT_CAPACITY, aicen, trcrn, vsnon, &dfresh, &dfhocn,
                             nwarn, &dflux_bio);
    *fpond = A(*fpond, dfpond);
    *fresh = A(*fresh, dfresh);
    *fsalt = A(*fsalt, dfsalt);
    *fhocn = A(*fhocn, dfhocn);
    // fzsal is present (MPAS passes zSalinityFlux); v8.4.2's dfzsal is 0 without solve_zsal
    if (T.zsal) *fzsal = A(*fzsal, dfzsal);
    // flux_bio is present (MPAS passes oceanBioFluxes) for nbtrcr tracers
    if (T.nbio) *T.flux_bio = A(*T.flux_bio, dflux_bio);
    return ITD_OK;
}

// column_itd_thermodynamics -> colpkg_step_therm2, one owned cell per thread.
// Field layout as column_v842.cu: a(nCategories, nCells) is a[iCell*ncat + n].
extern "C" __global__ void seaice_column_itd_thermodynamics_v842(
    int *status, int *nwarn, double dt, double yday, double hi_min, int ncat, int nt, int flags, int ktherm,
    int kitd, int nCellsSolve, const int *tidx, const int *trcr_depend, const double *trcr_base,
    const int *n_trcr_strata, const int *nt_strata, const double *categoryThicknessLimits, double *trcrnAll,
    double *atrcrnAll, double *iceAreaCategory, double *iceVolumeCategory, double *snowVolumeCategory,
    const double *iceAreaCategoryInitial, const double *iceVolumeCategoryInitial, double *openWaterArea,
    double *iceAreaCell, const double *seaFreezingTemperature, const double *seaSurfaceSalinity,
    const double *initialSalinityProfile, const double *lateralIceMeltFraction, double *lateralIceMelt,
    const double *freezingMeltingPotential, double *frazilFormation, const double *rainfallRate,
    double *pondFreshWaterFlux, double *oceanFreshWaterFlux, double *oceanSaltFlux, double *oceanHeatFlux,
    double *frazilGrowthDiagnostic, double *freezeOnset, int *newlyFormedIce, int nblyr, const double *biologyGrid,
    const double *verticalGrid, double *zSalinityFlux, int nbio, int bio0, const double *interfaceBiologyGrid,
    double zbgc_init_frac, const double *oceanBioConcentrations, int nzbgc, int bio_o, double *oceanBioFluxes)
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
    T.zsal = (flags & ITD_SOLVE_ZSAL) != 0;
    T.nblyr = nblyr;
    T.bgrid = biologyGrid;
    T.cgrid = verticalGrid;
    // the vertical tracer (nbio = 0: dummies, never read)
    double igrid[ZB_NBMAX + 2];
    T.nbio = nbio;
    T.bio = bio0;
    T.init_frac = zbgc_init_frac;
    T.ocean_bio = 0.0;
    T.flux_bio = oceanBioFluxes;
    T.igrid = igrid;
    if (nbio) {
        for (int k = 1; k <= nblyr + 1; ++k) igrid[k] = interfaceBiologyGrid[k - 1];
        T.ocean_bio = oceanBioConcentrations[(size_t)c * nzbgc + bio_o];
        T.flux_bio = oceanBioFluxes + (size_t)c * nzbgc;
    }

    double aicen[ITD_MAXCAT], vicen[ITD_MAXCAT], vsnon[ITD_MAXCAT], aicen_init[ITD_MAXCAT],
        vicen_init[ITD_MAXCAT], hin_max[ITD_MAXCAT + 1];
    int first_ice[ITD_MAXCAT];
    for (int n = 0; n < ncat; ++n) {
        aicen[n] = iceAreaCategory[c * ncat + n];
        vicen[n] = iceVolumeCategory[c * ncat + n];
        vsnon[n] = snowVolumeCategory[c * ncat + n];
        aicen_init[n] = iceAreaCategoryInitial[c * ncat + n];
        vicen_init[n] = iceVolumeCategoryInitial[c * ncat + n];
        first_ice[n] = newlyFormedIce[c * ncat + n] == 1;
    }
    for (int n = 0; n <= ncat; ++n) hin_max[n] = categoryThicknessLimits[n];
    double *trcrn = trcrnAll + (size_t)c * ncat * nt;
    double *atrcrn = atrcrnAll + (size_t)c * ncat * nt;
    double aice0 = openWaterArea[c], aice = iceAreaCell[c];
    double fpond = pondFreshWaterFlux[c], fresh = oceanFreshWaterFlux[c], fsalt = oceanSaltFlux[c];
    double fhocn = oceanHeatFlux[c], meltl = lateralIceMelt[c], frazil = frazilFormation[c];
    double frazil_diag = frazilGrowthDiagnostic[c], frz_onset = freezeOnset[c];
    double fzsal = T.zsal ? zSalinityFlux[c] : 0.0;
    int warnings = 0;
    int code = ITD_OK;
    int remapped = 0;

    fresh = A(fresh, M(rainfallRate[c], aice));
    itd_aggregate_area(ncat, aicen, &aice, &aice0);
    if (kitd == 1 && aice > PUNY) {
        // linear_itd's hin_max(ncat) = 999.9 lands in categoryThicknessLimits, MPAS's
        // array: the host writes it once (ITD_REMAPPED) for the cells MPAS reaches
        hin_max[ncat] = 999.9;
        remapped = ITD_REMAPPED;
        code = itd_linear_itd(T, ncat, hin_max, aicen_init, vicen_init, aicen, trcrn, vicen, vsnon, &aice,
                              &aice0, &fpond, flags, hi_min, atrcrn);
    }
    if (code == ITD_OK)
        code = itd_add_new_ice(T, ncat, dt, hin_max, ktherm, aicen, trcrn, vicen, vsnon[0], &aice0, aice,
                               freezingMeltingPotential[c], &frazil, &frz_onset, yday, flags, &fresh, &fsalt,
                               seaFreezingTemperature[c], seaSurfaceSalinity[c],
                               initialSalinityProfile + (size_t)c * (COL_NILYR + 1), &frazil_diag, &fzsal);
    if (code == ITD_OK) {
        itd_lateral_melt(T, dt, ncat, &fpond, &fresh, &fsalt, &fhocn, lateralIceMeltFraction[c], &meltl, aicen,
                         vicen, vsnon, trcrn, flags, &fzsal);
        if (ncat == 1) itd_reduce_area(hin_max[0], &aicen[0], &vicen[0], aicen_init[0], vicen_init[0]);
        code = itd_cleanup_itd(T, dt, ncat, hin_max, aicen, trcrn, vicen, vsnon, &aice0, &aice, flags, first_ice,
                               &fpond, &fresh, &fsalt, &fhocn, atrcrn, &warnings, &fzsal);
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
    lateralIceMelt[c] = meltl;
    frazilFormation[c] = frazil;
    frazilGrowthDiagnostic[c] = frazil_diag;
    freezeOnset[c] = frz_onset;
    if (T.zsal) zSalinityFlux[c] = fzsal;
    status[c] = code | remapped;
    nwarn[c] = warnings;
}

// get_cice_physics_tracer_array_category for one tracer field:
// field(sub+1:sub+layers, n, iCell) = trcrn(first:..., n)
extern "C" __global__ void seaice_scatter_category_tracer_v842(double *field, const double *trcrn, int first,
                                                               int layers, int sub, int fieldLayers,
                                                               int nTracers, int nCategories, int nCellsSolve)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nCellsSolve * nCategories * layers) return;
    int l = t % layers;
    int cn = t / layers;
    field[cn * fieldLayers + sub + l] = trcrn[cn * nTracers + first + l];
}
