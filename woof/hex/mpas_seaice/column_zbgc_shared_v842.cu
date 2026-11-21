// MPAS-Seaice v8.4.2: the column package's shared z-grid routines, which the
// column biogeochemistry (column_bgc_v842.cu) and the ITD thermodynamics and
// ridging under zsal_solve (column_itd_v842.cu) both call.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/column/ice_zbgc_shared.F90:
//     167-333    remap_zbgc
//   src/core_seaice/column/ice_therm_shared.F90:
//     47-73      calculate_Tin_from_qin (l_brine)
// (Nicole Jeffery, Elizabeth C. Hunke, Adrian K. Turner, LANL, and the CICE
// Consortium). Copyright (c) 2013-2019, Los Alamos National Security, LLC
// (LANS) and the University Corporation for Atmospheric Research (UCAR);
// BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is
// not the version available from LANS and UCAR.
//
// Appended to column_v842.cu (its rounding macros and constants). COL_NILYR is
// prepended; ZB_NBMAX bounds the biology layers: COL_NBLYR where the includer
// fixes nBioLayers at compile time, else MAX_BIO_LAYERS (32, column_v842.py).

#ifdef COL_NBLYR
#define ZB_NBMAX COL_NBLYR
#else
#define ZB_NBMAX 32
#endif
#define ZB_REMAP_STOP 2     // ice: remap_layers_bgc error

// remap_zbgc (ice_zbgc_shared.F90:167-333). nr0 = 0: the donor's nlyrn layers
// on ice_grid (hice) onto the receiver's nbyrn layers on bio_grid (hinS);
// nr0 = 1: the donor's nbyrn layers on bio_grid (hinS) onto nlyrn layers on
// ice_grid (hice), the ice above the brine at S_min. src[0..] are the donor
// layers (trcrn(it+kd-1)); ice_grid[k] and bio_grid[k] are ice_grid(k) and
// bio_grid(k) (cgrid(k+1), bgrid(k+1): the 0-based grids); out[kr] is
// trtmp(it+kr-1), kr = 1..receivers, an untouched one keeping its word.
// Returns ZB_REMAP_STOP on a negative thickness (out untouched).
__device__ int remap_zbgc(const double *src, int nr0, int nlyrn, int nbyrn, double hice, double hinS,
                          const double *ice_grid, const double *bio_grid, double S_min, double *out)
{
    if (hinS < 0.0 || hice < 0.0) return ZB_REMAP_STOP;
    double dgrid[ZB_NBMAX + COL_NILYR + 4], rgrid[ZB_NBMAX + COL_NILYR + 4], tracer[ZB_NBMAX + COL_NILYR + 4];
    double trgrid[ZB_NBMAX + COL_NILYR + 4], trdr[ZB_NBMAX + COL_NILYR + 4];
    for (int k = 0; k < ZB_NBMAX + COL_NILYR + 4; ++k) { trgrid[k] = 0.0; trdr[k] = 0.0; }
    int n_nd, n_nr, n_plus;
    if (nr0 == 0) {                                   // cice to bio
        n_nd = nlyrn;
        n_nr = nbyrn;
        n_plus = 2;
        dgrid[1] = col_min(col_min(A(-hice, hinS), A(-hinS, hice)), 0.0);
        dgrid[nlyrn + 2] = col_min(hinS, hice);
        tracer[1] = src[0];
        tracer[nlyrn + 2] = src[nlyrn - 1];
        rgrid[nbyrn + 2] = col_min(hinS, hice);
        if (hice > hinS) {
            rgrid[1] = 0.0;
            for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = M(bio_grid[kr], hinS);
            for (int kd = 1; kd <= n_nd; ++kd) {
                dgrid[kd + 1] = A(M(S(ice_grid[kd], 1.0), hice), hinS);
                tracer[kd + 1] = src[kd - 1];
            }
        } else {
            rgrid[1] = A(-hinS, hice);
            for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = A(M(S(bio_grid[kr], 1.0), hinS), hice);
            for (int kd = 1; kd <= n_nd; ++kd) {
                dgrid[kd + 1] = M(ice_grid[kd], hice);
                tracer[kd + 1] = src[kd - 1];
            }
        }
    } else {                                          // bio to cice
        n_nd = nbyrn;
        n_nr = nlyrn;
        if (hice > hinS) {                            // add S_min to the top layer
            n_plus = 3;
            tracer[1] = S_min;
            tracer[2] = S_min;
            rgrid[1] = A(-hice, hinS);
            rgrid[nlyrn + n_plus - 1] = hinS;
            for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = A(M(S(ice_grid[kr], 1.0), hice), hinS);
            dgrid[1] = A(-hice, hinS);
            dgrid[2] = M(S(hinS, hice), 0.5);
            dgrid[nbyrn + n_plus] = hinS;
            tracer[nbyrn + n_plus] = src[nbyrn - 1];
            for (int kd = 1; kd <= n_nd; ++kd) {
                dgrid[kd + 2] = M(bio_grid[kd], hinS);
                tracer[kd + 2] = src[kd - 1];
            }
            const double half = M(0.5, S(dgrid[n_plus + 1], dgrid[n_plus]));
            tracer[n_plus] = D(A(M(S_min, S(hice, hinS)), M(M(tracer[n_plus], 0.5), S(dgrid[n_plus + 1], dgrid[n_plus]))),
                               A(S(hice, hinS), half));
            tracer[1] = tracer[n_plus];
            tracer[2] = tracer[n_plus];
        } else {
            n_plus = 2;
            tracer[1] = src[0];
            tracer[nbyrn + 2] = src[nbyrn - 1];
            dgrid[1] = S(hice, hinS);
            dgrid[nbyrn + 2] = hice;
            rgrid[nlyrn + 2] = hice;
            rgrid[1] = 0.0;
            for (int kd = 1; kd <= n_nd; ++kd) {
                dgrid[kd + 1] = A(M(S(bio_grid[kd], 1.0), hinS), hice);
                tracer[kd + 1] = src[kd - 1];
            }
            for (int kr = 1; kr <= n_nr; ++kr) rgrid[kr + 1] = M(ice_grid[kr], hice);
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
    return 0;
}

// ice_therm_shared: calculate_Tin_from_qin with l_brine (the vertical salinity
// runs under BL99, where colpkg_init_thermo sets l_brine)
__device__ double bg_Tin_from_qin(double qin, double Tmltk)
{
    double aa1 = CP_ICE;
    double bb1 = S(S(M(S(CP_OCN, CP_ICE), Tmltk), D(qin, RHOI)), lfresh());
    double cc1 = M(lfresh(), Tmltk);
    return col_min(D(S(-bb1, SQRT(S(M(bb1, bb1), M(M(4.0, aa1), cc1)))), M(2.0, aa1)), Tmltk);
}

// compute_microS (ice_brine.F90:591-834): sice (trcrn(nt_sice:)) and zsal
