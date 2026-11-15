// MPAS-Seaice v8.4.2: the column package's (colpkg's) initialisation and
// aggregation, and the MPAS wrappers that drive them at seaice_init.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/column/ice_colpkg.F90:
//     82-262     colpkg_init_itd
//     326-379    colpkg_init_thermo
//     416-489    colpkg_init_trcr
//     1599-1618  colpkg_liquidus_temperature
//     1698-1707  colpkg_enthalpy_snow
//     3170-3280  colpkg_aggregate
//   src/core_seaice/column/ice_colpkg_tracers.F90:
//     157-246    colpkg_compute_tracers
//   src/core_seaice/column/ice_mushy_physics.F90:
//     29-110     the Assur (1958) liquidus constants
//     199-211    enthalpy_snow
//     236-306    liquidus_brine_salinity_mush, liquidus_temperature_mush,
//                enthalpy_mush
//     453-468    liquid_fraction
//   src/core_seaice/column/constants/cice/ice_constants_colpkg.F90 and
//   ice_colpkg_shared.F90 (saltmax, min_salin): the constants
//   src/core_seaice/shared/mpas_seaice_column.F:
//     295-420    init_column_itd, init_column_thermodynamic_profiles
//     758-800    init_column_level_ice_tracers
//     3413-3620  seaice_column_update_state, seaice_column_aggregate
//     5435-5600  set_cice_physics_tracer_array_category
//     5917-6079  get_cice_physics_tracer_array_cell
//     10940-10990 init_column_history_variables
//   src/core_seaice/shared/mpas_seaice_initialize.F:
//     434-531    init_ice_state_uniform_ice
//     545-850    init_ice_cice_default, initial_category_areas_and_volumes
//     1309-1377  init_ice_velocity_uniform
//     1707-1944  initialize_coupler_fields
// (Adrian K. Turner, LANL; Elizabeth C. Hunke, William H. Lipscomb, Bruce P.
// Briegleb, Nicole Jeffery and the CICE Consortium). Copyright (c) 2013-2019,
// Los Alamos National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// The column package is double precision (dbl_kind) and mpas_seaice_column.F
// passes it RKIND arrays: this is the RKIND 8 build only. Every operation is
// an explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction). A constant expression of parameters is folded by gfortran
// with each operation rounded to nearest, the word the same operation makes
// here at run time. MAX(a, b) and MIN(a, b) are gfortran's (a > b ? a : b and
// a < b ? a : b), sum() an accumulator from 0, x**2 x*x. A leading unary
// minus negates the whole product. sqrt is the correctly rounded instruction;
// tanh is cr_tanh (cr_tanh_flt64.cuh, glibc 2.44's), pow glibc_pow and cos
// uw_cos (glibc_flt64.cuh), prepended by column_v842.module_source.
//
// Layout: a Fortran array a(d1, d2, nCells) is a[(iCell*d2 + j)*d1 + i]
// (C-order (nCells + 1, d2, d1)), 0-based iCell; MPAS's missing-cell element
// is the last.

#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)

__device__ __forceinline__ double col_max(double a, double b) { return a > b ? a : b; }
__device__ __forceinline__ double col_min(double a, double b) { return a < b ? a : b; }

// ice_constants_colpkg (cice) and ice_colpkg_shared
#define RHOS 330.0
#define RHOI 917.0
#define RHOW 1026.0
#define CP_ICE 2106.0
#define CP_OCN 4218.0
#define DEPRESST 0.054
#define TOCNFRZ (-1.8)
#define TFFRESH 273.15
#define TSMELT 0.0
#define PUNY 1.0e-11
#define PI 3.14159265358979323846
#define SALTMAX 3.2
#define MIN_SALIN 0.1
#define P01 0.01
#define P1 0.1
// Lfresh = Lsub - Lvap
__device__ __forceinline__ double lfresh() { return S(2.835e6, 2.501e6); }
// seaiceDegreesToRadians = pii / 180.0_RKIND
__device__ __forceinline__ double deg2rad() { return D(3.141592653589793, 180.0); }

// ---------------------------------------------------------------------------
// ice_mushy_physics: the Assur (1958) liquidus
// ---------------------------------------------------------------------------

#define AZ1_LIQ (-18.48)
#define BZ1_LIQ 0.0
#define AZ2_LIQ (-10.3085)
#define BZ2_LIQ 62.4
#define TB_LIQ (-7.6362968855167352)
#define SB_LIQ 123.66702800276086

__device__ double liquidus_brine_salinity_mush(double zTin)
{
    double az1p = D(AZ1_LIQ, 1000.0), bz1p = D(BZ1_LIQ, 1000.0);
    double az2p = D(AZ2_LIQ, 1000.0), bz2p = D(BZ2_LIQ, 1000.0);
    (void)az1p; (void)az2p;
    double J1 = D(BZ1_LIQ, AZ1_LIQ), K1 = D(1.0, 1000.0), L1 = D(A(1.0, bz1p), AZ1_LIQ);
    double J2 = D(BZ2_LIQ, AZ2_LIQ), K2 = D(1.0, 1000.0), L2 = D(A(1.0, bz2p), AZ2_LIQ);
    double t_high = zTin > TB_LIQ ? 1.0 : 0.0;
    double lsubzero = zTin <= 0.0 ? 1.0 : 0.0;
    double Sbr = A(M(D(A(zTin, J1), A(M(K1, zTin), L1)), t_high),
                   M(D(A(zTin, J2), A(M(K2, zTin), L2)), S(1.0, t_high)));
    return M(Sbr, lsubzero);
}

__device__ double liquidus_temperature_mush(double Sbr)
{
    double M1 = AZ1_LIQ, N1 = -D(AZ1_LIQ, 1000.0), O1 = D(-BZ1_LIQ, AZ1_LIQ);
    double M2 = AZ2_LIQ, N2 = -D(AZ2_LIQ, 1000.0), O2 = D(-BZ2_LIQ, AZ2_LIQ);
    double t_high = Sbr <= SB_LIQ ? 1.0 : 0.0;
    return A(M(A(D(Sbr, A(M1, M(N1, Sbr))), O1), t_high),
             M(A(D(Sbr, A(M2, M(N2, Sbr))), O2), S(1.0, t_high)));
}

__device__ double liquid_fraction(double zTin, double zSin)
{
    double Sbr = col_max(liquidus_brine_salinity_mush(zTin), PUNY);
    return D(zSin, col_max(Sbr, zSin));
}

__device__ double enthalpy_mush(double zTin, double zSin)
{
    double phi = liquid_fraction(zTin, zSin);
    double cc = S(M(CP_OCN, RHOW), M(CP_ICE, RHOI));
    return S(A(M(M(phi, cc), zTin), M(M(RHOI, CP_ICE), zTin)), M(M(S(1.0, phi), RHOI), lfresh()));
}

// enthalpy_snow: -rhos * (-cp_ice * zTsn + Lfresh)
__device__ double enthalpy_snow(double zTsn)
{
    return -M(RHOS, A(-M(CP_ICE, zTsn), lfresh()));
}

// colpkg_liquidus_temperature
__device__ double colpkg_liquidus_temperature(double Sin, int ktherm)
{
    if (ktherm == 2) return liquidus_temperature_mush(Sin);
    return -M(DEPRESST, Sin);
}

// ---------------------------------------------------------------------------
// whole-array assignments
// ---------------------------------------------------------------------------

extern "C" __global__ void seaice_column_fill_v842(double *a, double value, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = value;
}

extern "C" __global__ void seaice_column_copy_v842(double *dst, const double *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// initialize_coupler_fields
// ---------------------------------------------------------------------------

// windSpeed = sqrt(uAirVelocity**2 + vAirVelocity**2), the whole arrays
extern "C" __global__ void seaice_wind_speed_v842(double *windSpeed, const double *u, const double *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) windSpeed[i] = SQRT(A(M(u[i], u[i]), M(v[i], v[i])));
}

// seaFreezingTemperature(iCell) = colpkg_liquidus_temperature(seaSurfaceSalinity(iCell)), iCell = 1, nCells
extern "C" __global__ void seaice_sea_freezing_temperature_v842(double *Tf, const double *sss, int nCells,
                                                                int ktherm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nCells) Tf[i] = colpkg_liquidus_temperature(sss[i], ktherm);
}

// ---------------------------------------------------------------------------
// colpkg_init_itd (one thread): hin_max(0:ncat) and hi_min
// ---------------------------------------------------------------------------

extern "C" __global__ void colpkg_init_itd_v842(double *hin_max, double *hi_min_out, int *status, int ncat,
                                                int kcatbound, int kitd)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    status[0] = 0;
    double rncat = (double)ncat;
    double d1 = D(3.0, rncat);
    double d2 = D(0.5, rncat);
    double b1 = P1;
    double b2 = 3.0;
    double b3 = col_max(M(rncat, S(rncat, 1.0)), D(M(2.0, b2), b1));
    double hi_min = P01;
    if (kcatbound == -1) {
        hin_max[0] = 0.0;
        hin_max[1] = 100.0;
    } else if (kcatbound == 0) {
        double cc1, cc2, cc3;
        if (kitd == 1) {
            cc1 = D(3.0, rncat);
            cc2 = M(15.0, cc1);
            cc3 = 3.0;
            hin_max[0] = 0.0;
        } else {
            hi_min = P1;
            cc1 = col_max(D(1.1, rncat), hi_min);
            cc2 = M(25.0, cc1);
            cc3 = 2.25;
            hin_max[0] = hi_min;
        }
        for (int n = 1; n <= ncat; ++n) {
            double x1 = D((double)(n - 1), rncat);
            hin_max[n] = A(A(hin_max[n - 1], cc1), M(cc2, A(1.0, g64_tanh::cr_tanh(M(cc3, S(x1, 1.0))))));
        }
    } else if (kcatbound == 1) {
        hin_max[0] = 0.0;
        for (int n = 1; n <= ncat; ++n) {
            double rn = (double)n;
            hin_max[n] = M(rn, A(d1, M(S(rn, 1.0), d2)));
        }
    } else if (kcatbound == 2) {
        const double wmo5[5] = {0.30, 0.70, 1.20, 2.00, 999.0};
        const double wmo6[6] = {0.15, 0.30, 0.70, 1.20, 2.00, 999.0};
        const double wmo7[7] = {0.10, 0.15, 0.30, 0.70, 1.20, 2.00, 999.0};
        const double *wmo;
        if (ncat == 5) wmo = wmo5;
        else if (ncat == 6) wmo = wmo6;
        else if (ncat == 7) wmo = wmo7;
        else { status[0] = 1; hi_min_out[0] = hi_min; return; }
        hin_max[0] = 0.0;
        for (int n = 1; n <= ncat; ++n) hin_max[n] = wmo[n - 1];
    } else if (kcatbound == 3) {
        hin_max[0] = 0.0;
        for (int n = 1; n <= ncat; ++n) {
            double rn = (double)n;
            hin_max[n] = M(b1, A(rn, D(M(M(b3, rn), A(rn, 1.0)), M(M(2.0, rncat), S(rncat, 1.0)))));
        }
    }
    hi_min_out[0] = hi_min;
}

// ---------------------------------------------------------------------------
// colpkg_init_thermo (one thread): sprofile(1:nilyr+1)
// ---------------------------------------------------------------------------

extern "C" __global__ void colpkg_init_thermo_v842(double *sprofile, int nilyr, int l_brine)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double nsal = 0.407, msal = 0.573;
    if (l_brine) {
        for (int k = 1; k <= nilyr; ++k) {
            double zn = D(S((double)k, 0.5), (double)nilyr);
            double s = M(D(SALTMAX, 2.0), S(1.0, uw_cos(M(PI, glibc_pow(zn, D(nsal, A(msal, zn)))))));
            sprofile[k - 1] = col_max(s, MIN_SALIN);
        }
        sprofile[nilyr] = SALTMAX;
    } else {
        for (int k = 0; k <= nilyr; ++k) sprofile[k] = 0.0;
    }
}

// init_column_thermodynamic_profiles: the owned cells' profiles, initialSalinityProfile(nIceLayers+1, nCells)
extern "C" __global__ void seaice_init_thermodynamic_profiles_v842(double *salinity, double *melting,
                                                                   const double *sprofile, int nCellsSolve,
                                                                   int nIceLayers, int ktherm)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    int np1 = nIceLayers + 1;
    for (int k = 0; k < np1; ++k) {
        salinity[c * np1 + k] = sprofile[k];
        melting[c * np1 + k] = colpkg_liquidus_temperature(sprofile[k], ktherm);
    }
}

// ---------------------------------------------------------------------------
// init_ice_state's initial conditions
// ---------------------------------------------------------------------------

// init_ice_state_uniform_ice: its nCellsSolve pointer is the nCells dimension
extern "C" __global__ void seaice_init_uniform_ice_v842(double *area, double *volume, double *snow, double *tsfc,
                                                        double *iceAreaCell, double *surfaceTemperatureCell,
                                                        const double *latCell, double north, double south,
                                                        double initialArea, double initialVolume,
                                                        double initialSnow, int nCells, int nCategories)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;
    bool ice = latCell[c] > M(north, deg2rad()) || latCell[c] < M(south, deg2rad());
    double sum = 0.0;
    for (int n = 0; n < nCategories; ++n) {
        int i = c * nCategories + n;
        area[i] = ice ? initialArea : 0.0;
        volume[i] = ice ? initialVolume : 0.0;
        snow[i] = ice ? initialSnow : 0.0;
        tsfc[i] = ice ? -1.0 : 0.0;
        sum = A(sum, area[i]);
    }
    iceAreaCell[c] = sum;
    surfaceTemperatureCell[c] = -20.15;
}

// initial_category_areas_and_volumes (one thread), with the column package on
extern "C" __global__ void seaice_initial_category_areas_v842(double *categoryArea, double *categoryThickness,
                                                              const double *limits, int nCategories)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double thicknessWithLargestArea = 3.0;
    for (int n = 0; n < nCategories - 1; ++n)
        categoryThickness[n] = M(0.5, A(limits[n], limits[n + 1]));
    categoryThickness[nCategories - 1] = A(limits[nCategories - 1], 1.0);
    double sum = 0.0;
    for (int n = 0; n < nCategories; ++n) {
        double h = categoryThickness[n];
        categoryArea[n] = col_max(0.0, S(M(M(2.0, thicknessWithLargestArea), h), M(h, h)));
        sum = A(sum, categoryArea[n]);
    }
    for (int n = 0; n < nCategories; ++n)
        categoryArea[n] = D(categoryArea[n], A(sum, D(PUNY, (double)nCategories)));
}

// colpkg_init_trcr for one category
__device__ void colpkg_init_trcr(double Tair, double Tf, const double *Sprofile, const double *Tprofile,
                                 double *Tsfc_out, int nilyr, int nslyr, double *qin, double *qsn, int ktherm,
                                 int calc_Tsfc, int heat_capacity)
{
    double Tsfc = Tf;
    if (calc_Tsfc) Tsfc = col_min(TSMELT, S(Tair, TFFRESH));
    *Tsfc_out = Tsfc;
    if (heat_capacity) {
        for (int k = 1; k <= nilyr; ++k) {
            double slope = S(Tf, Tsfc);
            double Ti = A(Tsfc, D(M(slope, S((double)k, 0.5)), (double)nilyr));
            if (ktherm == 2) {
                qin[k - 1] = enthalpy_mush(Ti, Sprofile[k - 1]);
            } else {
                double Tp = Tprofile[k - 1];
                qin[k - 1] = -M(RHOI, S(A(M(CP_ICE, S(Tp, Ti)), M(lfresh(), S(1.0, D(Tp, Ti)))), M(CP_OCN, Tp)));
            }
        }
        for (int k = 1; k <= nslyr; ++k) {
            double Ti = col_min(0.0, Tsfc);
            qsn[k - 1] = -M(RHOS, S(lfresh(), M(CP_ICE, Ti)));
        }
    } else {
        qin[0] = -M(RHOI, lfresh());
        qsn[0] = -M(RHOS, lfresh());
    }
}

// init_ice_cice_default over the owned cells
extern "C" __global__ void seaice_init_cice_default_v842(
    double *area, double *volume, double *snow, double *tsfc, double *iceEnthalpy, double *snowEnthalpy,
    double *iceSalinity, const double *airTemperature, const double *sst, const double *Tf, const int *landIceMask,
    const double *latCell, const double *salinityProfile, const double *meltingProfile,
    const double *categoryArea, const double *categoryThickness, double north, double south, int nCellsSolve,
    int nCategories, int nIceLayers, int nSnowLayers, int ktherm, int calc_Tsfc, int heat_capacity)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    const double initialCategorySnowThickness = 0.2;
    int np1 = nIceLayers + 1;
    if (sst[c] <= A(Tf[c], 0.2) &&
        (latCell[c] > M(north, deg2rad()) || latCell[c] < M(south, deg2rad())) &&
        landIceMask[c] == 0) {
        for (int n = 0; n < nCategories; ++n) {
            int i = c * nCategories + n;
            area[i] = categoryArea[n];
            volume[i] = M(categoryArea[n], categoryThickness[n]);
            snow[i] = col_min(M(area[i], initialCategorySnowThickness), M(0.2, volume[i]));
            colpkg_init_trcr(airTemperature[c], Tf[c], salinityProfile + c * np1, meltingProfile + c * np1,
                             tsfc + i, nIceLayers, nSnowLayers, iceEnthalpy + i * nIceLayers,
                             snowEnthalpy + i * nSnowLayers, ktherm, calc_Tsfc, heat_capacity);
        }
    } else {
        double qsn = enthalpy_snow(0.0);
        for (int n = 0; n < nCategories; ++n) {
            int i = c * nCategories + n;
            area[i] = 0.0;
            volume[i] = 0.0;
            snow[i] = 0.0;
            tsfc[i] = Tf[c];
            snowEnthalpy[i * nSnowLayers] = qsn;
        }
    }
    for (int n = 0; n < nCategories; ++n)
        for (int k = 0; k < nIceLayers; ++k)
            iceSalinity[(c * nCategories + n) * nIceLayers + k] = salinityProfile[c * np1 + k];
}

// init_ice_velocity_uniform over nVertices
extern "C" __global__ void seaice_init_velocity_uniform_v842(double *u, double *v, const int *interiorVertex,
                                                             const int *landIceMaskVertex, double u0, double v0,
                                                             int nVertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVertices) return;
    if (interiorVertex[i] == 1 && landIceMaskVertex[i] == 0) {
        u[i] = u0;
        v[i] = v0;
    } else {
        u[i] = 0.0;
        v[i] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// seaice_column_aggregate
// ---------------------------------------------------------------------------

// set_cice_physics_tracer_array_category for one tracer field: trcrn(first:first+layers-1, :) of every owned
// cell from field(sub+1:sub+layers, :, iCell); trcrn is (nCellsSolve, nCategories, nTracers)
extern "C" __global__ void seaice_gather_tracer_v842(double *trcrn, const double *field, int first, int layers,
                                                     int sub, int fieldLayers, int nTracers, int nCategories,
                                                     int nCellsSolve)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nCellsSolve * nCategories * layers) return;
    int l = t % layers;
    int cn = t / layers;
    trcrn[cn * nTracers + first + l] = field[cn * fieldLayers + sub + l];
}

// get_cice_physics_tracer_array_cell for one tracer field: field(sub+1:sub+layers, iCell) = trcr(first:...)
extern "C" __global__ void seaice_scatter_cell_tracer_v842(double *field, const double *trcr, int first,
                                                           int layers, int sub, int fieldLayers, int nTracers,
                                                           int nCellsSolve)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nCellsSolve * layers) return;
    int l = t % layers;
    int c = t / layers;
    field[c * fieldLayers + sub + l] = trcr[c * nTracers + first + l];
}

// colpkg_aggregate with colpkg_compute_tracers, per owned cell. trcr_depend = parentIndex, trcr_base =
// firstAncestorMask (C-order (3, nTracers)), n_trcr_strata = ancestorNumber, nt_strata = ancestorIndices
// (C-order (2, nTracers), 1-based)
extern "C" __global__ void colpkg_aggregate_v842(
    double *aiceCell, double *viceCell, double *vsnoCell, double *aice0Cell, double *trcrAll, double *atrcrAll,
    const double *trcrnAll, const double *aicen, const double *vicen, const double *vsnon,
    const int *trcr_depend, const double *trcr_base, const int *n_trcr_strata, const int *nt_strata,
    int ntrcr, int nt_Tsfc, int nt_fbri, int nCellsSolve, int ncat)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    const double *trcrn = trcrnAll + (size_t)c * ncat * ntrcr;
    double *trcr = trcrAll + (size_t)c * ntrcr;
    double *atrcr = atrcrAll + (size_t)c * ntrcr;
    double aice0 = 1.0, aice = 0.0, vice = 0.0, vsno = 0.0;
    (void)aice0;
    for (int it = 0; it < ntrcr; ++it) atrcr[it] = 0.0;
    for (int n = 0; n < ncat; ++n) {
        double a = aicen[c * ncat + n], v = vicen[c * ncat + n], s = vsnon[c * ncat + n];
        aice = A(aice, a);
        vice = A(vice, v);
        vsno = A(vsno, s);
        for (int it = 0; it < ntrcr; ++it) {
            double atrcrn = M(trcrn[n * ntrcr + it],
                              A(A(M(trcr_base[it], a), M(trcr_base[ntrcr + it], v)), M(trcr_base[2 * ntrcr + it], s)));
            for (int itl = 0; itl < n_trcr_strata[it]; ++itl) {
                int ntr = nt_strata[itl * ntrcr + it];
                atrcrn = M(atrcrn, trcrn[n * ntrcr + ntr - 1]);
            }
            atrcr[it] = A(atrcr[it], atrcrn);
        }
    }
    aice0 = col_max(S(1.0, aice), 0.0);
    // colpkg_compute_tracers
    for (int it = 0; it < ntrcr; ++it) {
        double divisor[3] = {M(trcr_base[it], aice), M(trcr_base[ntrcr + it], vice), M(trcr_base[2 * ntrcr + it], vsno)};
        if (trcr_depend[it] == 0) {
            if (aice > PUNY) {
                trcr[it] = D(atrcr[it], aice);
            } else {
                trcr[it] = 0.0;
                if (it + 1 == nt_Tsfc) trcr[it] = TOCNFRZ;
            }
        } else {
            double work = 0.0;
            for (int k = 0; k < 3; ++k)
                if (divisor[k] > 0.0) work = D(atrcr[it], divisor[k]);
            trcr[it] = work;
            for (int itl = 0; itl < n_trcr_strata[it]; ++itl) {
                int ntr = nt_strata[itl * ntrcr + it];
                if (trcr[ntr - 1] > 0.0) trcr[it] = D(trcr[it], trcr[ntr - 1]);
                else trcr[it] = 0.0;
            }
            if (vice <= 0.0 && it + 1 == nt_fbri) trcr[it] = 1.0;
        }
    }
    aiceCell[c] = aice;
    viceCell[c] = vice;
    vsnoCell[c] = vsno;
    aice0Cell[c] = aice0;
}

// seaice_column_update_state's tendencies over the owned cells
extern "C" __global__ void seaice_column_update_state_v842(double *areaTend, double *volumeTend, double *ageTend,
                                                           const double *areaCell, const double *volumeCell,
                                                           const double *ageCell, double dt, double ageOffset,
                                                           int useAge, int nCellsSolve)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    areaTend[c] = D(S(areaCell[c], areaTend[c]), dt);
    volumeTend[c] = D(S(volumeCell[c], volumeTend[c]), dt);
    if (useAge) {
        if (ageOffset > 0.0) {
            if (ageCell[c] > 0.0) ageTend[c] = D(S(S(ageCell[c], ageTend[c]), ageOffset), dt);
        } else {
            ageTend[c] = D(S(ageCell[c], ageTend[c]), dt);
        }
    }
}
