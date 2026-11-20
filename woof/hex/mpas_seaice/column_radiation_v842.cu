// MPAS-Seaice v8.4.2: the column package's shortwave radiation without the
// delta-Eddington solution: column_prep_radiation (the shortwave scaling),
// column_radiation with the CCSM3 shortwave (shortwave_ccsm3: the ccsm3 or
// constant albedos and absorbed_solar) or with a coupled surface temperature
// (no shortwave), and seaice_init_column_shortwave's albedo aggregation.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     435-647    seaice_init_column_shortwave
//     1990-2104  column_prep_radiation
//     2118-2484  column_radiation
//   src/core_seaice/column/ice_colpkg.F90:
//     2547-2639  colpkg_prep_radiation
//     2649-2972  colpkg_step_radiation
//   src/core_seaice/column/ice_shortwave.F90:
//     72-256     shortwave_ccsm3
//     262-392    compute_albedos
//     398-488    constant_albedos
//     494-660    absorbed_solar
// (William H. Lipscomb, Elizabeth C. Hunke, Adrian K. Turner and the CICE
// Consortium). Copyright (c) 2013-2019, Los Alamos National Security, LLC
// (LANS) and the University Corporation for Atmospheric Research (UCAR);
// BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is
// not the version available from LANS and UCAR.
//
// Appended to column_v842.cu (the rounding macros and constants). exp is
// glibc_exp (glibc_flt64.cuh, the MIT transcription of glibc's binary64 exp).
// compute_albedos (config_albedo_type = 'ccsm3') calls glibc's binary64 atan,
// which is not correctly rounded and is LGPL: that branch compiles only with
// the opt-in transcription prepended (WOOF_GLIBC239_TRIG64, g239::g239_atan;
// column_radiation_v842.py refuses otherwise). Fortran's MIN and MAX are
// gfortran's comparisons (sw_min, sw_max: the first argument on a tie).
//
// v8.4.2 passes ncat as absorbed_solar's nilyr (ice_shortwave.F90:236): the
// penetrating light is spread over ncat layers of thickness hi/ncat, and the
// ice layers below the ncat-th keep 0. The host refuses ncat > nilyr, where
// absorbed_solar writes past the category's layers. The opt-in correction
// ccsm3_absorbed_solar_layers (SW_FIX_LAYERS) passes nilyr.
//
// One thread is one owned cell: a(nCategories, nCells) is a[c*ncat + n] and
// a(nLayers, nCategories, nCells) is a[(c*ncat + n)*nLayers + k].

#define SW_ALBOCN 0.06          // albocn
#define SW_SNOWPATCH 0.02       // snowpatch
#define SW_AWTVDR 0.00318       // awtvdr
#define SW_AWTIDR 0.00182       // awtidr
#define SW_AWTVDF 0.63282       // awtvdf
#define SW_AWTIDF 0.36218       // awtidf
#define SW_KAPPAV 1.4           // kappav
#define SW_TIMELT 0.0           // Timelt
#define SW_I0VIS 0.70           // absorbed_solar's i0vis
#define SW_DT_MELT 1.0          // compute_albedos' dT_melt = c1
#define SW_DALB_MLT (-0.075)    // dalb_mlt
#define SW_DALB_MLTV (-0.1)     // dalb_mltv = -p1
#define SW_DALB_MLTI (-0.15)    // dalb_mlti = -p15
#define SW_WARMICE 0.68         // constant_albedos
#define SW_COLDICE 0.70
#define SW_WARMSNOW 0.77
#define SW_COLDSNOW 0.81

// flags (column_radiation_v842._flags)
#define SW_HEAT_CAPACITY 1      // colpkg_init_thermo's heat_capacity (not the zero layer)
#define SW_ALBEDO_CONSTANT 2    // albedo_type == 'constant'
#define SW_CALC_TSFC 4          // calc_Tsfc
#define SW_POND_TOPO 8          // tr_pond_topo
#define SW_FIX_LAYERS 16        // opt-in correction ccsm3_absorbed_solar_layers: nilyr, not ncat

// gfortran's MIN/MAX: mvar = a; if (b < mvar || isnan(mvar)) mvar = b
__device__ __forceinline__ double sw_min(double a, double b) { return (b < a || isnan(a)) ? b : a; }
__device__ __forceinline__ double sw_max(double a, double b) { return (b > a || isnan(a)) ? b : a; }

// ice_shortwave.F90:262-392 compute_albedos. The snow albedos are intent(out)
// but only set where hs > puny: elsewhere they keep the caller's albocn.
__device__ void sw_compute_albedos(double aicen, double vicen, double vsnon, double Tsfcn, double albicev,
                                   double albicei, double albsnowv, double albsnowi, double ahmax,
                                   double *alvdrni, double *alidrni, double *alvdfni, double *alidfni,
                                   double *alvdrns, double *alidrns, double *alvdfns, double *alidfns,
                                   double *alvdrn, double *alidrn, double *alvdfn, double *alidfn,
                                   double *albin, double *albsn)
{
#ifdef WOOF_GLIBC239_TRIG64
    double fhtan = g239::g239_atan(M(ahmax, 4.0));
    double hi = D(vicen, aicen);
    double hs = D(vsnon, aicen);
    double fh = sw_min(D(g239::g239_atan(M(hi, 4.0)), fhtan), 1.0);
    double albo = M(SW_ALBOCN, S(1.0, fh));
    *alvdfni = A(M(albicev, fh), albo);
    *alidfni = A(M(albicei, fh), albo);
    double dTs = S(SW_TIMELT, Tsfcn);
    double fT = sw_min(S(D(dTs, SW_DT_MELT), 1.0), 0.0);
    *alvdfni = S(*alvdfni, M(SW_DALB_MLT, fT));
    *alidfni = S(*alidfni, M(SW_DALB_MLT, fT));
    *alvdfni = sw_max(*alvdfni, SW_ALBOCN);
    *alidfni = sw_max(*alidfni, SW_ALBOCN);
    if (hs > PUNY) {
        *alvdfns = albsnowv;
        *alidfns = albsnowi;
        *alvdfns = S(*alvdfns, M(SW_DALB_MLTV, fT));
        *alidfns = S(*alidfns, M(SW_DALB_MLTI, fT));
    }
    *alvdrni = *alvdfni;
    *alidrni = *alidfni;
    *alvdrns = *alvdfns;
    *alidrns = *alidfns;
    double asnow;
    if (hs > PUNY) asnow = D(hs, A(hs, SW_SNOWPATCH));
    else asnow = 0.0;
    *alvdfn = A(M(*alvdfni, S(1.0, asnow)), M(*alvdfns, asnow));
    *alidfn = A(M(*alidfni, S(1.0, asnow)), M(*alidfns, asnow));
    *alvdrn = A(M(*alvdrni, S(1.0, asnow)), M(*alvdrns, asnow));
    *alidrn = A(M(*alidrni, S(1.0, asnow)), M(*alidrns, asnow));
    *albin = A(A(A(M(SW_AWTVDR, *alvdrni), M(SW_AWTIDR, *alidrni)), M(SW_AWTVDF, *alvdfni)),
               M(SW_AWTIDF, *alidfni));
    *albsn = A(A(A(M(SW_AWTVDR, *alvdrns), M(SW_AWTIDR, *alidrns)), M(SW_AWTVDF, *alvdfns)),
               M(SW_AWTIDF, *alidfns));
#else
    __trap();   // the host refuses config_albedo_type = 'ccsm3' without the opt-in atan
#endif
}

// ice_shortwave.F90:398-488 constant_albedos
__device__ void sw_constant_albedos(double aicen, double vsnon, double Tsfcn,
                                    double *alvdrni, double *alidrni, double *alvdfni, double *alidfni,
                                    double *alvdrns, double *alidrns, double *alvdfns, double *alidfns,
                                    double *alvdrn, double *alidrn, double *alvdfn, double *alidfn,
                                    double *albin, double *albsn)
{
    double hs = D(vsnon, aicen);
    if (hs > PUNY) {
        if (Tsfcn >= -2.0 * PUNY) {      // -c2*puny, folded
            *alvdfn = SW_WARMSNOW;
            *alidfn = SW_WARMSNOW;
        } else {
            *alvdfn = SW_COLDSNOW;
            *alidfn = SW_COLDSNOW;
        }
    } else {
        if (Tsfcn >= -2.0 * PUNY) {
            *alvdfn = SW_WARMICE;
            *alidfn = SW_WARMICE;
        } else {
            *alvdfn = SW_COLDICE;
            *alidfn = SW_COLDICE;
        }
    }
    *alvdrn = *alvdfn;
    *alidrn = *alidfn;
    *alvdrni = *alvdrn;
    *alidrni = *alidrn;
    *alvdrns = *alvdrn;
    *alidrns = *alidrn;
    *alvdfni = *alvdfn;
    *alidfni = *alidfn;
    *alvdfns = *alvdfn;
    *alidfns = *alidfn;
    *albin = A(A(A(M(SW_AWTVDR, *alvdrni), M(SW_AWTIDR, *alidrni)), M(SW_AWTVDF, *alvdfni)),
               M(SW_AWTIDF, *alidfni));
    *albsn = A(A(A(M(SW_AWTVDR, *alvdrns), M(SW_AWTIDR, *alidrns)), M(SW_AWTVDF, *alvdfns)),
               M(SW_AWTIDF, *alidfns));
}

// ice_shortwave.F90:494-660 absorbed_solar, with nilyr = nlay (v8.4.2 passes ncat)
__device__ void sw_absorbed_solar(bool heat_capacity, int nlay, double aicen, double vicen, double vsnon,
                                  double swvdr, double swvdf, double swidr, double swidf,
                                  double alvdrni, double alvdfni, double alidrni, double alidfni,
                                  double alvdrns, double alvdfns, double alidrns, double alidfns,
                                  double *fswsfc, double *fswint, double *fswthru, double *fswpenl, double *Iswabs)
{
    double trantop = 0.0, tranbot = 0.0;
    double hs = D(vsnon, aicen);
    double asnow;
    if (hs > PUNY) asnow = D(hs, A(hs, SW_SNOWPATCH));
    else asnow = 0.0;
    double swabsv = A(M(swvdr, A(M(S(1.0, alvdrni), S(1.0, asnow)), M(S(1.0, alvdrns), asnow))),
                      M(swvdf, A(M(S(1.0, alvdfni), S(1.0, asnow)), M(S(1.0, alvdfns), asnow))));
    double swabsi = A(M(swidr, A(M(S(1.0, alidrni), S(1.0, asnow)), M(S(1.0, alidrns), asnow))),
                      M(swidf, A(M(S(1.0, alidfni), S(1.0, asnow)), M(S(1.0, alidfns), asnow))));
    double swabs = A(swabsv, swabsi);
    double fswpenvdr = M(M(M(swvdr, S(1.0, alvdrni)), S(1.0, asnow)), SW_I0VIS);
    double fswpenvdf = M(M(M(swvdf, S(1.0, alvdfni)), S(1.0, asnow)), SW_I0VIS);
    double fswpen = A(fswpenvdr, fswpenvdf);
    *fswsfc = S(swabs, fswpen);
    trantop = 1.0;
    for (int k = 1; k <= nlay; ++k) {
        double hi = D(vicen, aicen);
        double hilyr = D(hi, (double)nlay);
        tranbot = glibc_exp(-M(M(SW_KAPPAV, hilyr), (double)k));
        Iswabs[k - 1] = M(fswpen, S(trantop, tranbot));
        trantop = tranbot;
        if (k == 1) {
            fswpenl[k - 1] = fswpen;
            fswpenl[k] = M(fswpen, tranbot);
        } else {
            fswpenl[k] = M(fswpen, tranbot);
        }
    }
    *fswthru = M(fswpen, tranbot);
    *fswint = S(fswpen, *fswthru);
    if (!heat_capacity) {
        *fswsfc = A(*fswsfc, *fswint);
        *fswint = 0.0;
        Iswabs[0] = 0.0;
    }
}

// colpkg_step_radiation (calc_Tsfc: shortwave_ccsm3; else no shortwave) for
// one owned cell, called from column_radiation. apndn and ipndn are read
// only with topo ponds and a coupled surface temperature; zbion
// (bioTracerShortwave) is zeroed when it is allocated (nbio > 0).
extern "C" __global__ void seaice_column_radiation_v842(
    int ncat, int nilyr, int nslyr, int nbio, int flags, double albicev, double albicei, double albsnowv,
    double albsnowi, double ahmax, double hp1, int nCellsSolve,
    const double *__restrict__ aicenAll, const double *__restrict__ vicenAll,
    const double *__restrict__ vsnonAll, const double *__restrict__ TsfcnAll,
    const double *__restrict__ apndnAll, const double *__restrict__ ipndnAll,
    const double *__restrict__ swvdrAll, const double *__restrict__ swvdfAll,
    const double *__restrict__ swidrAll, const double *__restrict__ swidfAll,
    double *__restrict__ coszenAll,
    double *__restrict__ alvdrnAll, double *__restrict__ alvdfnAll,
    double *__restrict__ alidrnAll, double *__restrict__ alidfnAll,
    double *__restrict__ fswsfcnAll, double *__restrict__ fswintnAll, double *__restrict__ fswthrunAll,
    double *__restrict__ fswpenlnAll, double *__restrict__ IswabsnAll, double *__restrict__ SswabsnAll,
    double *__restrict__ albicenAll, double *__restrict__ albsnonAll, double *__restrict__ apeffnAll,
    double *__restrict__ zbionAll)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    const bool heat_capacity = (flags & SW_HEAT_CAPACITY) != 0;
    const double *aicen = aicenAll + (size_t)c * ncat, *vicen = vicenAll + (size_t)c * ncat;
    const double *vsnon = vsnonAll + (size_t)c * ncat, *Tsfcn = TsfcnAll + (size_t)c * ncat;
    double *alvdrn = alvdrnAll + (size_t)c * ncat, *alvdfn = alvdfnAll + (size_t)c * ncat;
    double *alidrn = alidrnAll + (size_t)c * ncat, *alidfn = alidfnAll + (size_t)c * ncat;
    double *fswsfcn = fswsfcnAll + (size_t)c * ncat, *fswintn = fswintnAll + (size_t)c * ncat;
    double *fswthrun = fswthrunAll + (size_t)c * ncat;
    double *fswpenln = fswpenlnAll + (size_t)c * ncat * (nilyr + 1);
    double *Iswabsn = IswabsnAll + (size_t)c * ncat * nilyr;
    double *Sswabsn = SswabsnAll + (size_t)c * ncat * nslyr;
    double swvdr = swvdrAll[c], swvdf = swvdfAll[c], swidr = swidrAll[c], swidf = swidfAll[c];

    for (int n = 0; n < ncat; ++n) {
        alvdrn[n] = 0.0;
        alidrn[n] = 0.0;
        alvdfn[n] = 0.0;
        alidfn[n] = 0.0;
        fswsfcn[n] = 0.0;
        fswintn[n] = 0.0;
        fswthrun[n] = 0.0;
    }
    for (int i = 0; i < ncat * (nilyr + 1); ++i) fswpenln[i] = 0.0;
    for (int i = 0; i < ncat * nilyr; ++i) Iswabsn[i] = 0.0;
    for (int i = 0; i < ncat * nslyr; ++i) Sswabsn[i] = 0.0;
    if (nbio > 0)
        for (int i = 0; i < ncat * nbio; ++i) zbionAll[(size_t)c * ncat * nbio + i] = 0.0;

    if (flags & SW_CALC_TSFC) {
        // ice_shortwave.F90:72-256 shortwave_ccsm3
        double *albin = albicenAll + (size_t)c * ncat, *albsn = albsnonAll + (size_t)c * ncat;
        coszenAll[c] = 0.5;
        for (int n = 0; n < ncat; ++n) {
            for (int k = 0; k < nslyr; ++k) Sswabsn[n * nslyr + k] = 0.0;
            double alvdrni = SW_ALBOCN, alidrni = SW_ALBOCN, alvdfni = SW_ALBOCN, alidfni = SW_ALBOCN;
            double alvdrns = SW_ALBOCN, alidrns = SW_ALBOCN, alvdfns = SW_ALBOCN, alidfns = SW_ALBOCN;
            alvdrn[n] = SW_ALBOCN;
            alidrn[n] = SW_ALBOCN;
            alvdfn[n] = SW_ALBOCN;
            alidfn[n] = SW_ALBOCN;
            albin[n] = 0.0;
            albsn[n] = 0.0;
            fswsfcn[n] = 0.0;
            fswintn[n] = 0.0;
            fswthrun[n] = 0.0;
            for (int k = 0; k <= nilyr; ++k) fswpenln[n * (nilyr + 1) + k] = 0.0;
            for (int k = 0; k < nilyr; ++k) Iswabsn[n * nilyr + k] = 0.0;
            if (aicen[n] > PUNY) {
                if (flags & SW_ALBEDO_CONSTANT)
                    sw_constant_albedos(aicen[n], vsnon[n], Tsfcn[n], &alvdrni, &alidrni, &alvdfni, &alidfni,
                                        &alvdrns, &alidrns, &alvdfns, &alidfns, &alvdrn[n], &alidrn[n],
                                        &alvdfn[n], &alidfn[n], &albin[n], &albsn[n]);
                else
                    sw_compute_albedos(aicen[n], vicen[n], vsnon[n], Tsfcn[n], albicev, albicei, albsnowv,
                                       albsnowi, ahmax, &alvdrni, &alidrni, &alvdfni, &alidfni, &alvdrns,
                                       &alidrns, &alvdfns, &alidfns, &alvdrn[n], &alidrn[n], &alvdfn[n],
                                       &alidfn[n], &albin[n], &albsn[n]);
                sw_absorbed_solar(heat_capacity, (flags & SW_FIX_LAYERS) ? nilyr : ncat, aicen[n], vicen[n], vsnon[n], swvdr, swvdf, swidr, swidf,
                                  alvdrni, alvdfni, alidrni, alidfni, alvdrns, alvdfns, alidrns, alidfns,
                                  &fswsfcn[n], &fswintn[n], &fswthrun[n], &fswpenln[n * (nilyr + 1)],
                                  &Iswabsn[n * nilyr]);
            }
        }
    } else {
        if (flags & SW_POND_TOPO) {
            const double *apndn = apndnAll + (size_t)c * ncat, *ipndn = ipndnAll + (size_t)c * ncat;
            double *apeffn = apeffnAll + (size_t)c * ncat;
            for (int n = 0; n < ncat; ++n) {
                apeffn[n] = 0.0;
                if (aicen[n] > PUNY) {
                    if (M(apndn[n], aicen[n]) > PUNY && ipndn[n] < hp1) apeffn[n] = apndn[n];
                    else apeffn[n] = 0.0;
                    if (apndn[n] < PUNY) apeffn[n] = 0.0;
                }
            }
        }
        for (int n = 0; n < ncat; ++n) {
            alvdrn[n] = 0.0;
            alidrn[n] = 0.0;
            alvdfn[n] = 0.0;
            alidfn[n] = 0.0;
            fswsfcn[n] = 0.0;
            fswintn[n] = 0.0;
            fswthrun[n] = 0.0;
        }
        for (int i = 0; i < ncat * nilyr; ++i) Iswabsn[i] = 0.0;
        for (int i = 0; i < ncat * nslyr; ++i) Sswabsn[i] = 0.0;
    }
}

// ice_colpkg.F90:2547-2639 colpkg_prep_radiation, called from column_prep_radiation
extern "C" __global__ void seaice_column_prep_radiation_v842(
    int ncat, int nilyr, int nslyr, int nCellsSolve, const double *__restrict__ aiceAll,
    const double *__restrict__ aicenAll, const double *__restrict__ swvdrAll,
    const double *__restrict__ swvdfAll, const double *__restrict__ swidrAll,
    const double *__restrict__ swidfAll, const double *__restrict__ alvdrAll,
    const double *__restrict__ alvdfAll, const double *__restrict__ alidrAll,
    const double *__restrict__ alidfAll, double *__restrict__ scaleAll,
    double *__restrict__ fswsfcnAll, double *__restrict__ fswintnAll, double *__restrict__ fswthrunAll,
    double *__restrict__ fswpenlnAll, double *__restrict__ SswabsnAll, double *__restrict__ IswabsnAll)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    const double *aicen = aicenAll + (size_t)c * ncat;
    double scale_factor = scaleAll[c];
    if (aiceAll[c] > 0.0 && scale_factor > PUNY) {
        double netsw = A(A(A(M(swvdrAll[c], S(1.0, alvdrAll[c])), M(swvdfAll[c], S(1.0, alvdfAll[c]))),
                           M(swidrAll[c], S(1.0, alidrAll[c]))),
                         M(swidfAll[c], S(1.0, alidfAll[c])));
        scale_factor = D(netsw, scale_factor);
    } else {
        scale_factor = 1.0;
    }
    scaleAll[c] = scale_factor;
    for (int n = 0; n < ncat; ++n) {
        if (aicen[n] > PUNY) {
            size_t cn = (size_t)c * ncat + n;
            fswsfcnAll[cn] = M(scale_factor, fswsfcnAll[cn]);
            fswintnAll[cn] = M(scale_factor, fswintnAll[cn]);
            fswthrunAll[cn] = M(scale_factor, fswthrunAll[cn]);
            for (int k = 0; k <= nilyr; ++k)
                fswpenlnAll[cn * (nilyr + 1) + k] = M(scale_factor, fswpenlnAll[cn * (nilyr + 1) + k]);
            for (int k = 0; k < nslyr; ++k)
                SswabsnAll[cn * nslyr + k] = M(scale_factor, SswabsnAll[cn * nslyr + k]);
            for (int k = 0; k < nilyr; ++k)
                IswabsnAll[cn * nilyr + k] = M(scale_factor, IswabsnAll[cn * nilyr + k]);
        }
    }
}

// mpas_seaice_column.F:588-641 seaice_init_column_shortwave's aggregation,
// after column_radiation. bareIceAlbedoCell, snowAlbedoCell, pondAlbedoCell
// and effectivePondAreaCell are not reset: they add onto their words.
// config_do_restart is .false. (refused).
extern "C" __global__ void seaice_init_column_shortwave_aggregate_v842(
    int ncat, int nCellsSolve, const double *__restrict__ areaAll, const double *__restrict__ coszenAll,
    const double *__restrict__ alvdrnAll, const double *__restrict__ alvdfnAll,
    const double *__restrict__ alidrnAll, const double *__restrict__ alidfnAll,
    const double *__restrict__ albicenAll, const double *__restrict__ albsnonAll,
    const double *__restrict__ albpndnAll, const double *__restrict__ apeffnAll,
    const double *__restrict__ swvdrAll, const double *__restrict__ swvdfAll,
    const double *__restrict__ swidrAll, const double *__restrict__ swidfAll,
    double *__restrict__ alvdrCell, double *__restrict__ alvdfCell, double *__restrict__ alidrCell,
    double *__restrict__ alidfCell, double *__restrict__ albiceCell, double *__restrict__ albsnoCell,
    double *__restrict__ albpndCell, double *__restrict__ apeffCell, double *__restrict__ alvdrArea,
    double *__restrict__ alvdfArea, double *__restrict__ alidrArea, double *__restrict__ alidfArea,
    double *__restrict__ scaleAll)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    alvdrCell[c] = 0.0;
    alvdfCell[c] = 0.0;
    alidrCell[c] = 0.0;
    alidfCell[c] = 0.0;
    for (int n = 0; n < ncat; ++n) {
        size_t cn = (size_t)c * ncat + n;
        double a = areaAll[cn];
        if (a > PUNY) {
            alvdrCell[c] = A(alvdrCell[c], M(alvdrnAll[cn], a));
            alvdfCell[c] = A(alvdfCell[c], M(alvdfnAll[cn], a));
            alidrCell[c] = A(alidrCell[c], M(alidrnAll[cn], a));
            alidfCell[c] = A(alidfCell[c], M(alidfnAll[cn], a));
            if (coszenAll[c] > PUNY) {
                albiceCell[c] = A(albiceCell[c], M(albicenAll[cn], a));
                albsnoCell[c] = A(albsnoCell[c], M(albsnonAll[cn], a));
                albpndCell[c] = A(albpndCell[c], M(albpndnAll[cn], a));
            }
            apeffCell[c] = A(apeffCell[c], M(apeffnAll[cn], a));
        }
    }
    alvdrArea[c] = alvdrCell[c];
    alvdfArea[c] = alvdfCell[c];
    alidrArea[c] = alidrCell[c];
    alidfArea[c] = alidfCell[c];
    scaleAll[c] = A(A(A(M(swvdrAll[c], S(1.0, alvdrArea[c])), M(swvdfAll[c], S(1.0, alvdfArea[c]))),
                      M(swidrAll[c], S(1.0, alidrArea[c]))),
                    M(swidfAll[c], S(1.0, alidfArea[c])));
}
