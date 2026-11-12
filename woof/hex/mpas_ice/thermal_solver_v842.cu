// MALI v8.4.2: the vertical thermal solver and the basal melt of grounded and
// floating ice.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_thermal.F:
//     276-296    li_thermal_init's dsigmaTerm
//     642-1384   li_thermal_solver
//     1403-1584  li_basal_melt_floating_ice
//     1598-1844  heat_dissipation_sia
//     1859-1966  basal_friction
//     1980-2170  li_temperature_to_enthalpy, li_enthalpy_to_temperature
//     2190-2618  temperature_matrix_elements, enthalpy_matrix_elements
//     2632-3114  basal_melt_grounded_ice, basal_melt_floating_ice
//     3128-3190  pressure_melting_point[_column]
//     3265-3307  tridiag_solver
//   src/core_landice/shared/mpas_li_mask.F: li_mask_is_floating_ice
//   with li_constants' cp_ice, latent_heat_ice, rho_water, gravity,
//   iceMeltingPointPressureDependence, kelvin_to_celsius, iceConductivity,
//   oceanFreezingTempSurface, oceanFreezingTempDepthDependence, scyr and pii.
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off: left to right,
// parentheses kept, no contraction; x**2 is x*x). min(a, b) and max(a, b)
// are a < b ? a : b and a > b ? a : b. rhoi and rhoo are the module's copies
// of config_ice_density and config_ocean_density that li_thermal_init makes.
// RKIND = 4 (MALI's default build) unless MPAS_RKIND8 is defined. The libm
// calls are the shared transcriptions the unit is prepended with: real powers
// are glibc's powf (gfk_pow, glibc_flt32.cuh) or pow (glibc_pow,
// glibc_flt64.cuh); sin is glibc's sinf (glibc_sinf, glibc_trig_flt32.cuh)
// and tanh glibc's tanhf or tanh, CORE-MATH's correctly rounded ones
// (sfire_tanhf, sfire_libm.cuh; g64_tanh::cr_tanh, cr_tanh_flt64.cuh). The
// RKIND 8 unit carries no sin: 'seroussi' is refused there.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// a(maxEdges, nCells) is a[iCell*maxEdges + i]; cellsOnEdge(2, nEdges) is
// cellsOnEdge[2*iEdge + j]. Connectivity is 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_ABS(x) fabs(x)
#define LI_POW(x,y) glibc_pow((x),(y))
#define LI_TANH(x) g64_tanh::cr_tanh(x)
#define LI_C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_ABS(x) fabsf(x)
#define LI_POW(x,y) gfk_pow((x),(y))
#define LI_SIN(x) glibc_sinf(x)
#define LI_TANH(x) sfire_tanhf(x)
#define LI_C(x) x##f
#endif

#define LI_CP_ICE LI_C(2009.0)
#define LI_LATENT_HEAT_ICE LI_C(335.0e3)
#define LI_RHO_WATER LI_C(1000.0)
#define LI_GRAVITY LI_C(9.80616)
#define LI_PMP_DEPENDENCE LI_C(9.7456e-8)     // iceMeltingPointPressureDependence
#define LI_K2C LI_C(273.15)                   // kelvin_to_celsius
#define LI_ICE_CONDUCTIVITY LI_C(2.1)
#define LI_OCEAN_FREEZING_SURFACE LI_C(-1.92)
#define LI_OCEAN_FREEZING_DEPTH LI_C(-7.53e-4)
#define LI_SCYR LI_C(31536000.0)
#define LI_PII LI_C(3.141592653589793)
#define LI_MAX_WATERFRAC LI_C(0.01)           // basal_melt_grounded_ice maxWaterfrac
#define LI_EPS11 LI_C(1.0e-11)
#define LI_MASK_FLOATING 4                    // li_mask_ValueFloating
#define LI_MAX_LEVELS 128

__device__ __forceinline__ rk li_max(rk a, rk b) { return a > b ? a : b; }
__device__ __forceinline__ rk li_min(rk a, rk b) { return a < b ? a : b; }
__device__ __forceinline__ bool li_floating(int mask) { return (mask & LI_MASK_FLOATING) == LI_MASK_FLOATING; }

// pressure_melting_point(depth) (deg C)
__device__ __forceinline__ rk li_pmp(rk rhoi, rk depth) {
    return -M(M(M(LI_PMP_DEPENDENCE, rhoi), LI_GRAVITY), depth);
}
// pressure_melting_point_column: sigma(k) of the column
__device__ __forceinline__ rk li_pmp_column(rk rhoi, rk thickness, rk sigma) {
    return -M(M(M(M(LI_PMP_DEPENDENCE, rhoi), LI_GRAVITY), thickness), sigma);
}
// li_thermal_init's dsigmaTerm(k, 1) and dsigmaTerm(k, 2), k 0-based
__device__ __forceinline__ rk li_dsigma1(const rk *iS, const rk *cS, int k) {
    return D((rk)1.0, M(S(iS[k + 1], iS[k]), S(cS[k], k == 0 ? iS[0] : cS[k - 1])));
}
__device__ __forceinline__ rk li_dsigma2(const rk *iS, const rk *cS, int k, int nLevels) {
    return D((rk)1.0, M(S(iS[k + 1], iS[k]), S(k == nLevels - 1 ? iS[nLevels] : cS[k + 1], cS[k])));
}

// li_basal_melt_floating_ice for 'none' (option 0), 'constant' (2), 'mismip'
// (3) and 'seroussi' (4), after the masks; one thread per cell over nCells.
// 'file' (1) is never launched: the routine returns before touching anything.
// iceMask is thickness > config_thermal_thickness over the owned cells.
extern "C" __global__ void li_basal_melt_floating_v842(
    rk *floatingBasalMassBal, const rk *thickness, const int *cellMask, const rk *xCell, const rk *lowerSurface,
    const rk *bedTopography, int option, rk thermalThickness, rk seaLevel, rk bmltFlux, rk bmltXlimit,
    rk serAmplitude, rk serPeriod, rk serPhase, rk daysSinceStart, rk rhoi, int nCells, int nCellsSolve)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk value = (rk)0.0;
    if (iCell < nCellsSolve) {
        const bool iceMask = thickness[iCell] > thermalThickness;
        const bool floating = li_floating(cellMask[iCell]);
        const bool openOcean = bedTopography[iCell] < seaLevel && !iceMask;
        if (option == 2) {
            const rk rate = D(bmltFlux, M(rhoi, LI_LATENT_HEAT_ICE));
            if ((floating || openOcean) && LI_ABS(xCell[iCell]) >= bmltXlimit) value = -rate;
        }
        else if (option == 3) {
            if (floating || openOcean) {
                const rk omega = D(LI_C(0.20), LI_SCYR);   // bmlt_float_omega
                const rk hCavity = S(lowerSurface[iCell], bedTopography[iCell]);
                const rk zDraft = S(lowerSurface[iCell], seaLevel);
                value = M(M(-omega, LI_TANH(D(hCavity, LI_C(75.0)))), li_max(S(LI_C(-100.0), zDraft), (rk)0.0));
            }
        }
#ifndef MPAS_RKIND8
        else if (option == 4) {
            const rk twoPi = M((rk)2.0, LI_PII);
            const rk interceptSer = A(LI_C(-100.0), M(serAmplitude, LI_SIN(A(
                M(D(twoPi, serPeriod), D(daysSinceStart, LI_C(365.0))), M(twoPi, serPhase)))));
            if (floating && lowerSurface[iCell] < LI_C(-10.0)) {
                const rk zDraft = S(lowerSurface[iCell], seaLevel);
                value = li_max(M(LI_C(-1.0), LI_C(50.0)),
                               li_min((rk)0.0, M(LI_C(0.088), S(li_max(zDraft, LI_C(-650.0)), interceptSer))));
            }
        }
#endif
    }
    if (option == 2 || option == 3) value = M(value, rhoi);
    else if (option == 4) value = D(M(value, rhoi), LI_SCYR);
    floatingBasalMassBal[iCell] = value;
}

// heat_dissipation_sia, the edge loop: one thread per edge over nEdges.
extern "C" __global__ void li_heat_dissipation_edge_v842(
    rk *heatDissipationEdge, const rk *thickness, const rk *flowParamA, const rk *slopeEdge,
    const rk *layerCenterSigma, const int *cellsOnEdge, rk rhoi, rk flowLawExponent, int nEdges, int nCells,
    int nLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int iCell1 = cellsOnEdge[2 * iEdge], iCell2 = cellsOnEdge[2 * iEdge + 1];
    rk *out = heatDissipationEdge + iEdge * nLevels;
    if (iCell1 >= 1 && iCell1 <= nCells && iCell2 >= 1 && iCell2 <= nCells) {
        const int c1 = iCell1 - 1, c2 = iCell2 - 1;
        const rk thicknessEdge = M((rk)0.5, A(thickness[c1], thickness[c2]));
        const rk slope = LI_ABS(slopeEdge[iEdge]);
        const rk exponent = A(flowLawExponent, (rk)1.0);
        for (int k = 0; k < nLevels; ++k) {
            const rk flowParamAEdge = M((rk)0.5, A(flowParamA[c1 * nLevels + k], flowParamA[c2 * nLevels + k]));
            const rk base = M(M(M(M(layerCenterSigma[k], rhoi), LI_GRAVITY), thicknessEdge), slope);
            out[k] = M(M((rk)2.0, flowParamAEdge), LI_POW(base, exponent));
        }
    } else {
        for (int k = 0; k < nLevels; ++k) out[k] = (rk)0.0;
    }
}

// heat_dissipation_sia, the average to cells and the conversion to deg/s:
// one thread per cell over nCells.
extern "C" __global__ void li_heat_dissipation_cell_v842(
    rk *heatDissipation, const rk *heatDissipationEdge, const rk *dcEdge, const rk *dvEdge, const rk *areaCell,
    const int *nEdgesOnCell, const int *edgesOnCell, rk rhoi, int nCells, int nLevels, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk *hd = heatDissipation + iCell * nLevels;
    for (int k = 0; k < nLevels; ++k) hd[k] = (rk)0.0;
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk weightEdge = D(M(M(LI_C(0.25), dcEdge[iEdge]), dvEdge[iEdge]), areaCell[iCell]);
        for (int k = 0; k < nLevels; ++k) hd[k] = A(hd[k], M(weightEdge, heatDissipationEdge[iEdge * nLevels + k]));
    }
    const rk heatCapacity = M(rhoi, LI_CP_ICE);
    for (int k = 0; k < nLevels; ++k) hd[k] = D(hd[k], heatCapacity);
}

// basal_friction: one thread per cell over nCells.
extern "C" __global__ void li_basal_friction_v842(
    rk *basalFrictionFlux, const rk *betaSolve, const rk *basalSpeed, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    basalFrictionFlux[iCell] = M(M(betaSolve[iCell], M(basalSpeed[iCell], basalSpeed[iCell])), LI_SCYR);
}

// tridiag_solver(a, b, c, x, y) on n unknowns; x may not alias the others.
__device__ void li_tridiag(const rk *a, const rk *b, const rk *c, rk *x, const rk *y, rk *aa, rk *bb, int n)
{
    aa[0] = D(c[0], b[0]);
    bb[0] = D(y[0], b[0]);
    for (int i = 1; i < n; ++i) {
        aa[i] = D(c[i], S(b[i], M(a[i], aa[i - 1])));
        bb[i] = D(S(y[i], M(a[i], bb[i - 1])), S(b[i], M(a[i], aa[i - 1])));
    }
    x[n - 1] = bb[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = S(bb[i], M(aa[i], x[i + 1]));
}

// li_thermal_solver's column work after the masks, the heat dissipation and
// the basal friction, through basal_melt_grounded_ice and the floating-ice
// transfer: one thread per cell over nCells. thermalSolver 1 'temperature',
// 2 'enthalpy'. airLapse: config_surface_air_temperature_source 'lapse'.
// The halo cells only see the whole-array conversions to Celsius and back
// and the zeroed grounded basal mass balance. errorFlag is set where an owned
// column leaves [-100, 100] deg C (the Fortran's MPAS_LOG_CRIT abort).
extern "C" __global__ void li_thermal_column_v842(
    rk *temperature, rk *waterfrac, rk *enthalpy, rk *surfaceTemperature, rk *basalTemperature,
    rk *surfaceAirTemperature, rk *surfaceConductiveFlux, rk *basalConductiveFlux, rk *groundedBasalMassBal,
    rk *floatingBasalMassBal, int *errorFlag, const rk *heatDissipation, const rk *basalHeatFlux,
    const rk *basalFrictionFlux, const rk *thickness, const rk *upperSurface, const rk *basalWaterThickness,
    const int *cellMask, const rk *layerCenterSigma, const rk *layerInterfaceSigma, rk deltat, rk rhoi, rk rhoo,
    rk thermalThickness, int airLapse, rk airValue, rk lapseRate, int thermalSolver, int calculateBmb,
    int nCells, int nCellsSolve, int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nLevels;
    rk *T = temperature + base;
    rk *wf = waterfrac + base;
    rk *E = enthalpy + base;
    const rk *hd = heatDissipation + base;
    const rk *cS = layerCenterSigma;
    const rk *iS = layerInterfaceSigma;
    if (airLapse) surfaceAirTemperature[iCell] = S(airValue, M(lapseRate, upperSurface[iCell]));
    for (int k = 0; k < nLevels; ++k) T[k] = S(T[k], LI_K2C);
    rk sat = S(surfaceAirTemperature[iCell], LI_K2C);
    rk sfcT = S(surfaceTemperature[iCell], LI_K2C);
    rk basT = S(basalTemperature[iCell], LI_K2C);
    rk gbmb = (rk)0.0;

    if (iCell < nCellsSolve) {
        const rk h = thickness[iCell];
        const bool iceMask = h > thermalThickness;
        const bool floating = li_floating(cellMask[iCell]);
        const int n = nLevels;
        rk subd[LI_MAX_LEVELS + 2], diag[LI_MAX_LEVELS + 2], supd[LI_MAX_LEVELS + 2], rhs[LI_MAX_LEVELS + 2];
        rk sol[LI_MAX_LEVELS + 2], aa[LI_MAX_LEVELS + 2], bb[LI_MAX_LEVELS + 2];
        if (iceMask) {
            sfcT = li_min((rk)0.0, sat);
            if (floating) {
                const rk depth = D(M(h, rhoi), rhoo);
                basT = A(LI_OCEAN_FREEZING_SURFACE, M(LI_OCEAN_FREEZING_DEPTH, depth));
            }
            const rk pmpTemperatureBed = li_pmp(rhoi, h);
            if (thermalSolver == 2) {
                // li_temperature_to_enthalpy
                for (int k = 0; k < n; ++k) {
                    const rk pmp = li_pmp_column(rhoi, h, cS[k]);
                    E[k] = A(M(M(M(S((rk)1.0, wf[k]), rhoi), LI_CP_ICE), T[k]),
                             M(M(wf[k], LI_RHO_WATER), A(M(LI_CP_ICE, pmp), LI_LATENT_HEAT_ICE)));
                }
                // enthalpy_matrix_elements
                rk diffusivity[LI_MAX_LEVELS + 1];
                const rk diffusivityCold = D(LI_ICE_CONDUCTIVITY, M(rhoi, LI_CP_ICE));
                const rk diffusivityTemperate = D(diffusivityCold, LI_C(100.0));
                const rk surfaceEnthalpy = M(M(sfcT, rhoi), LI_CP_ICE);
                const rk basalEnthalpy = M(M(basT, rhoi), LI_CP_ICE);
                const rk tiny = M(M(LI_C(1.e-20), LI_RHO_WATER), LI_LATENT_HEAT_ICE);
                rk enthalpyTempPrev = surfaceEnthalpy;
                for (int k = 0; k <= n; ++k) {   // Fortran k + 1
                    rk dEnthalpy, enthalpyTemp;
                    if (k == 0) dEnthalpy = S(E[0], surfaceEnthalpy);
                    else if (k == n) dEnthalpy = S(basalEnthalpy, E[n - 1]);
                    else dEnthalpy = S(E[k], E[k - 1]);
                    if (k == n) enthalpyTemp = basalEnthalpy;
                    else enthalpyTemp = M(M(M(S((rk)1.0, wf[k]), rhoi), LI_CP_ICE), T[k]);
                    const rk dEnthalpyTemp = S(enthalpyTemp, enthalpyTempPrev);
                    enthalpyTempPrev = enthalpyTemp;
                    rk avgFactor;
                    if (LI_ABS(dEnthalpy) > tiny) {
                        avgFactor = li_max((rk)0.0, D(dEnthalpyTemp, dEnthalpy));
                        avgFactor = li_min((rk)1.0, avgFactor);
                    } else {
                        avgFactor = (rk)0.0;
                    }
                    diffusivity[k] = A(M(avgFactor, diffusivityCold), M(S((rk)1.0, avgFactor), diffusivityTemperate));
                }
                subd[0] = (rk)0.0; diag[0] = (rk)1.0; supd[0] = (rk)0.0; rhs[0] = surfaceEnthalpy;
                const rk factor = D(deltat, M(h, h));
                for (int k = 0; k < n; ++k) {
                    subd[k + 1] = -M(M(factor, diffusivity[k]), li_dsigma1(iS, cS, k));
                    supd[k + 1] = -M(M(factor, diffusivity[k + 1]), li_dsigma2(iS, cS, k, n));
                    diag[k + 1] = S(S((rk)1.0, subd[k + 1]), supd[k + 1]);
                    rhs[k + 1] = A(E[k], M(M(M(hd[k], deltat), rhoi), LI_CP_ICE));
                }
                if (floating) {
                    subd[n + 1] = (rk)0.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0; rhs[n + 1] = basalEnthalpy;
                } else if (LI_ABS(S(T[n - 1], li_pmp_column(rhoi, h, cS[n - 1]))) < LI_C(0.001)) {
                    subd[n + 1] = (rk)-1.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0; rhs[n + 1] = (rk)0.0;
                } else if (LI_ABS(S(basT, pmpTemperatureBed)) < LI_C(0.001)) {
                    subd[n + 1] = (rk)0.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0;
                    rhs[n + 1] = M(M(pmpTemperatureBed, rhoi), LI_CP_ICE);
                } else {
                    const rk dsigmaBot = S((rk)1.0, cS[n - 1]);
                    subd[n + 1] = (rk)-1.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0;
                    rhs[n + 1] = D(M(M(M(M(A(basalFrictionFlux[iCell], basalHeatFlux[iCell]), dsigmaBot), h), rhoi),
                                     LI_CP_ICE), LI_ICE_CONDUCTIVITY);
                }
                li_tridiag(subd, diag, supd, sol, rhs, aa, bb, n + 2);
                rk sfcE = sol[0];
                for (int k = 0; k < n; ++k) E[k] = sol[k + 1];
                rk basE = sol[n + 1];
                const rk denthTop = S(E[0], sfcE);
                const rk denthBot = S(basE, E[n - 1]);
                surfaceConductiveFlux[iCell] = -D(M(D(diffusivity[0], h), denthTop), cS[0]);
                basalConductiveFlux[iCell] = -D(M(D(diffusivity[n], h), denthBot), S((rk)1.0, cS[n - 1]));
                // li_enthalpy_to_temperature with the surface and basal enthalpies
                const rk pmpEnthalpyBed = M(M(pmpTemperatureBed, rhoi), LI_CP_ICE);
                for (int k = 0; k < n; ++k) {
                    const rk pmpTemperature = li_pmp_column(rhoi, h, cS[k]);
                    const rk pmpEnthalpy = M(M(pmpTemperature, rhoi), LI_CP_ICE);
                    if (E[k] >= pmpEnthalpy) {
                        T[k] = pmpTemperature;
                        wf[k] = D(S(E[k], pmpEnthalpy), A(M(M(S(LI_RHO_WATER, rhoi), LI_CP_ICE), pmpTemperature),
                                                          M(LI_RHO_WATER, LI_LATENT_HEAT_ICE)));
                    } else {
                        T[k] = D(E[k], M(rhoi, LI_CP_ICE));
                        wf[k] = (rk)0.0;
                    }
                }
                if (sfcE >= (rk)0.0) { sfcT = (rk)0.0; sfcE = (rk)0.0; }
                else sfcT = D(sfcE, M(rhoi, LI_CP_ICE));
                if (basE >= pmpEnthalpyBed) { basT = pmpTemperatureBed; basE = pmpEnthalpyBed; }
                else basT = D(basE, M(rhoi, LI_CP_ICE));
            } else {
                // temperature_matrix_elements
                subd[0] = (rk)0.0; diag[0] = (rk)1.0; supd[0] = (rk)0.0; rhs[0] = sfcT;
                const rk factor = D(D(M(deltat, LI_ICE_CONDUCTIVITY), M(rhoi, LI_CP_ICE)), M(h, h));
                for (int k = 0; k < n; ++k) {
                    subd[k + 1] = -M(factor, li_dsigma1(iS, cS, k));
                    supd[k + 1] = -M(factor, li_dsigma2(iS, cS, k, n));
                    diag[k + 1] = S(S((rk)1.0, subd[k + 1]), supd[k + 1]);
                    rhs[k + 1] = A(T[k], M(hd[k], deltat));
                }
                if (floating) {
                    subd[n + 1] = (rk)0.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0; rhs[n + 1] = basT;
                } else if (LI_ABS(S(basT, pmpTemperatureBed)) < LI_C(0.001)) {
                    subd[n + 1] = (rk)0.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0; rhs[n + 1] = pmpTemperatureBed;
                } else {
                    const rk dsigmaBot = S((rk)1.0, cS[n - 1]);
                    subd[n + 1] = (rk)-1.0; diag[n + 1] = (rk)1.0; supd[n + 1] = (rk)0.0;
                    rhs[n + 1] = D(M(M(A(basalFrictionFlux[iCell], basalHeatFlux[iCell]), dsigmaBot), h),
                                   LI_ICE_CONDUCTIVITY);
                }
                li_tridiag(subd, diag, supd, sol, rhs, aa, bb, n + 2);
                sfcT = sol[0];
                for (int k = 0; k < n; ++k) T[k] = sol[k + 1];
                basT = sol[n + 1];
                const rk dTtop = S(T[0], sfcT);
                const rk dTbot = S(basT, T[n - 1]);
                surfaceConductiveFlux[iCell] = D(M(-D(LI_ICE_CONDUCTIVITY, h), dTtop), cS[0]);
                basalConductiveFlux[iCell] = D(M(-D(LI_ICE_CONDUCTIVITY, h), dTbot), S((rk)1.0, cS[n - 1]));
            }

            // basal_melt_grounded_ice
            if (!floating) {
                rk netBasalFlux = A(A(basalFrictionFlux[iCell], basalHeatFlux[iCell]), basalConductiveFlux[iCell]);
                if (LI_ABS(netBasalFlux) < LI_EPS11) netBasalFlux = (rk)0.0;
                if (thermalSolver == 2)
                    gbmb = D(-netBasalFlux, S(M(LI_LATENT_HEAT_ICE, rhoi), E[n - 1]));
                else
                    gbmb = D(-netBasalFlux, M(LI_LATENT_HEAT_ICE, rhoi));
            }
            if (thermalSolver == 2) {
                for (int k = 0; k < n; ++k) {
                    if (wf[k] > LI_MAX_WATERFRAC) {
                        const rk excessWater = M(M(S(wf[k], LI_MAX_WATERFRAC), h), S(iS[k + 1], iS[k]));
                        gbmb = S(gbmb, D(excessWater, deltat));
                        wf[k] = LI_MAX_WATERFRAC;
                    }
                }
            } else {
                for (int k = 0; k < n; ++k) {
                    const rk pmp = li_pmp_column(rhoi, h, cS[k]);
                    if (T[k] > pmp) {
                        const rk layerThickness = M(h, S(iS[k + 1], iS[k]));
                        const rk meltEnergy = M(M(M(rhoi, LI_CP_ICE), S(T[k], pmp)), layerThickness);
                        gbmb = S(gbmb, D(meltEnergy, M(M(rhoi, LI_LATENT_HEAT_ICE), deltat)));
                        T[k] = pmp;
                    }
                }
            }
            if (!floating) {
                basT = li_min(basT, pmpTemperatureBed);
                if (gbmb > (rk)0.0 && basalWaterThickness[iCell] == (rk)0.0 && basT >= pmpTemperatureBed)
                    basT = S(pmpTemperatureBed, LI_C(0.01));
            }
        } else {   // not thermally active
            sfcT = (rk)0.0;
            basT = (rk)0.0;
            for (int k = 0; k < n; ++k) { T[k] = (rk)0.0; wf[k] = (rk)0.0; E[k] = (rk)0.0; }
        }
        gbmb = M(gbmb, rhoi);
        if (!calculateBmb) gbmb = (rk)0.0;
    }

    for (int k = 0; k < nLevels; ++k) T[k] = A(T[k], LI_K2C);
    surfaceAirTemperature[iCell] = A(sat, LI_K2C);
    surfaceTemperature[iCell] = A(sfcT, LI_K2C);
    basalTemperature[iCell] = A(basT, LI_K2C);
    if (iCell < nCellsSolve) {
        const rk maxtempThreshold = A(LI_C(100.0), LI_K2C);
        const rk mintempThreshold = A(LI_C(-100.0), LI_K2C);
        for (int k = 0; k < nLevels; ++k)
            if (T[k] > maxtempThreshold || T[k] < mintempThreshold) *errorFlag = 1;
        if (thickness[iCell] > thermalThickness && li_floating(cellMask[iCell]) && gbmb != (rk)0.0) {
            floatingBasalMassBal[iCell] = A(floatingBasalMassBal[iCell], gbmb);
            gbmb = (rk)0.0;
        }
    }
    groundedBasalMassBal[iCell] = gbmb;
}
