// MPAS-Ocean v8.4.2: frazil ice formation (the tendencies and the accumulated mass).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_frazil_forcing.F (ocn_frazil_forcing_build_arrays,
//     ocn_frazil_forcing_layer_thickness, ocn_frazil_forcing_active_tracers)
//   src/core_ocean/shared/mpas_ocn_equation_of_state.F (ocn_freezing_temperature)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off). Fortran MIN/MAX are
// chosen on the words as integer selects (gfortran -O0: MAX(a, b) is b when
// b > a, else a). One thread per column: the column's frazil sums run down
// from the deepest frazil level, as the Fortran's.
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nl + k]; activeTracers
// (nTracers, nVertLevels, nCells+1) is t[(iCell*nl + k)*nTracers + i];
// tracer indices are 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
__device__ __forceinline__ rk fz_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk fz_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
__device__ __forceinline__ rk fz_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk fz_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif
#define GRAVITY C(9.80616)
__device__ __forceinline__ int fz_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ocn_freezing_temperature: c[0..4] = coeff_0, coeff_S, coeff_p, coeff_pS, reference_pressure
__device__ __forceinline__ rk fz_freezing(rk salinity, rk pressure, const rk *c)
{
    const rk pressureOffset = fz_max(S(pressure, c[4]), C(0.0));
    return A(A(A(c[0], M(c[1], salinity)), M(c[2], pressureOffset)), M(M(c[3], pressureOffset), salinity));
}

// ocn_frazil_forcing_build_arrays' column loop, cells 1..nCellsArray(2). The
// three tendencies are zeroed over the whole array first (the caller).
// p = heat_of_fusion, ice_density, fractional_thickness_limit, maximum_depth,
//     specific_heat_sea_water, sea_ice_reference_salinity,
//     land_ice_reference_salinity, maximum_freezing_temperature, rho_sw, dt;
// freeze = the open ocean's five coefficients, then the land-ice cavity's.
extern "C" __global__ void ocn_frazil_build_arrays_v842(
    rk *frazilLayerThicknessTendency, rk *frazilTemperatureTendency, rk *frazilSalinityTendency,
    rk *accumulatedFrazilIceMassNew, rk *accumulatedFrazilIceSalinityNew, rk *accumulatedLandIceFrazilMassNew,
    const rk *accumulatedFrazilIceMassOld, const rk *accumulatedFrazilIceSalinityOld,
    const rk *accumulatedLandIceFrazilMassOld, const rk *layerThickness, const rk *ssh, const rk *activeTracers,
    const rk *zMid, const rk *density, const rk *pressure, const int *landIceMask, const int *maxLevelCell,
    const rk *p, const rk *freeze, int nCells, int nl, int nTracers, int indexT, int indexS, int inOpenOcean,
    int underLandIceOn, int withLandIceMask)
{
    const int iCell = fz_thread();
    if (iCell >= nCells) return;
    const rk heatOfFusion = p[0], iceDensity = p[1], thicknessLimit = p[2], maximumDepth = p[3];
    const rk cpSw = p[4], seaIceSalinity = p[5], landIceSalinity = p[6], maximumFreezing = p[7];
    const rk rhoSw = p[8], dt = p[9];
    const bool underLandIce = withLandIceMask && landIceMask[iCell] == 1;
    if (underLandIce && !underLandIceOn) return;
    if (!underLandIce && !inOpenOcean) return;
    const int mlc = maxLevelCell[iCell];
    const rk *T = activeTracers + (size_t)iCell * nl * nTracers + (indexT - 1);
    const rk *Sal = activeTracers + (size_t)iCell * nl * nTracers + (indexS - 1);
    const rk *h = layerThickness + (size_t)iCell * nl;

    // the deepest level shallower than the maximum depth below the surface
    int kBottomFrazil = mlc;
    const rk depthLimit = A(-ssh[iCell], maximumDepth);
    for (int k = mlc; k >= 1; --k) {
        if (-zMid[(size_t)iCell * nl + k - 1] < depthLimit) {
            kBottomFrazil = k;
            break;
        }
    }
    rk columnTemperatureMin = C(1.0e30);
    for (int k = 1; k <= kBottomFrazil; ++k) {
        const rk t = T[(size_t)(k - 1) * nTracers];
        if (t < columnTemperatureMin) columnTemperatureMin = t;
    }
    if (columnTemperatureMin > maximumFreezing) return;

    const rk *coeffs = underLandIce ? freeze + 5 : freeze;
    const rk fusionIce = M(heatOfFusion, iceDensity);
    const rk cpRho = M(cpSw, rhoSw);
    rk sumThickness = C(0.0), sumSalt = C(0.0);
    for (int k = kBottomFrazil; k >= 1; --k) {
        const size_t at = (size_t)iCell * nl + k - 1;
        const rk t = T[(size_t)(k - 1) * nTracers], s = Sal[(size_t)(k - 1) * nTracers];
        const rk freezing = fz_freezing(s, pressure[at], coeffs);
        const rk potential = M(M(M(h[k - 1], cpSw), rhoSw), S(t, freezing));
        const rk freezingEnergy = fz_max(C(0.0), -potential);
        const rk meltingEnergy = fz_max(C(0.0), potential);
        if (freezingEnergy > C(0.0)) {
            rk frazilSalinity = fz_min(underLandIce ? landIceSalinity : seaIceSalinity, s);
            rk newThickness = D(freezingEnergy, fusionIce);
            newThickness = fz_min(newThickness, M(h[k - 1], thicknessLimit));
            const rk newSalt = M(newThickness, frazilSalinity);
            frazilLayerThicknessTendency[at] = -D(D(M(newThickness, iceDensity), density[at]), dt);
            frazilSalinityTendency[at] = -D(newSalt, dt);
            frazilTemperatureTendency[at] = D(D(M(M(newThickness, heatOfFusion), iceDensity), cpRho), dt);
            sumThickness = A(sumThickness, newThickness);
            sumSalt = A(sumSalt, newSalt);
        } else if (sumThickness > C(0.0)) {
            rk melted = D(meltingEnergy, fusionIce);
            melted = fz_min(melted, sumThickness);
            melted = fz_min(melted, M(h[k - 1], thicknessLimit));
            const rk meltedSalt = M(melted, D(sumSalt, sumThickness));
            frazilLayerThicknessTendency[at] = D(D(M(melted, iceDensity), density[at]), dt);
            frazilSalinityTendency[at] = D(meltedSalt, dt);
            frazilTemperatureTendency[at] = -D(D(M(M(melted, heatOfFusion), iceDensity), cpRho), dt);
            sumSalt = S(sumSalt, meltedSalt);
            sumThickness = S(sumThickness, melted);
        }
    }
    accumulatedFrazilIceMassNew[iCell] = A(accumulatedFrazilIceMassOld[iCell], M(sumThickness, iceDensity));
    accumulatedFrazilIceSalinityNew[iCell] = A(accumulatedFrazilIceSalinityOld[iCell], sumSalt);
    if (underLandIce)
        accumulatedLandIceFrazilMassNew[iCell] = A(accumulatedLandIceFrazilMassOld[iCell],
                                                   M(sumThickness, iceDensity));
}

// config_frazil_use_surface_pressure: the weight of the accumulated frazil, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_frazil_surface_pressure_v842(
    rk *frazilSurfacePressure, const rk *accumulatedFrazilIceMassNew, int nCells)
{
    const int iCell = fz_thread();
    if (iCell >= nCells) return;
    frazilSurfacePressure[iCell] = M(accumulatedFrazilIceMassNew[iCell], GRAVITY);
}

// ocn_frazil_forcing_layer_thickness, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_frazil_thickness_tend_v842(
    rk *tend, const rk *frazilLayerThicknessTendency, const int *maxLevelCell, int nCells, int nl)
{
    const int iCell = fz_thread();
    if (iCell >= nCells) return;
    for (int k = 0; k < maxLevelCell[iCell]; ++k) {
        const size_t at = (size_t)iCell * nl + k;
        tend[at] = A(tend[at], frazilLayerThicknessTendency[at]);
    }
}

// ocn_frazil_forcing_active_tracers, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_frazil_tracers_tend_v842(
    rk *tend, const rk *frazilTemperatureTendency, const rk *frazilSalinityTendency, const int *maxLevelCell,
    int nCells, int nl, int nTracers, int indexT, int indexS)
{
    const int iCell = fz_thread();
    if (iCell >= nCells) return;
    for (int k = 0; k < maxLevelCell[iCell]; ++k) {
        const size_t at = (size_t)iCell * nl + k;
        rk *t = tend + at * nTracers;
        t[indexT - 1] = A(t[indexT - 1], frazilTemperatureTendency[at]);
        t[indexS - 1] = A(t[indexS - 1], frazilSalinityTendency[at]);
    }
}
