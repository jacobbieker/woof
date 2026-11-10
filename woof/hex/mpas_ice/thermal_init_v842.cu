// MPAS land-ice (MALI) v8.4.2: the thermal initialisation and the
// temperature, enthalpy and pressure-melting-point conversions.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_thermal.F:92-521     (li_thermal_init)
//   src/core_landice/mode_forward/mpas_li_thermal.F:537-624    (li_init_linear_temperature_in_column)
//   src/core_landice/mode_forward/mpas_li_thermal.F:1980-2170  (li_temperature_to_enthalpy, li_enthalpy_to_temperature)
//   src/core_landice/mode_forward/mpas_li_thermal.F:3128-3249  (pressure_melting_point[_column],
//                                                              li_compute_pressure_melting_point_fields)
//   src/core_landice/shared/mpas_li_constants.F:41-63          (cp_ice, latent_heat_ice, rho_water, gravity,
//                                                              iceMeltingPointPressureDependence, kelvin_to_celsius)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 unless MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). rhoi is the module's copy of config_ice_density that
// li_thermal_init makes.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_SQRT(x) __dsqrt_rn(x)
#define LI_ABS(x) fabs(x)
#define LI_POW(x,y) glibc_pow((x),(y))
#define LI_EXP(x) glibc_exp(x)
#define LI_C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_SQRT(x) __fsqrt_rn(x)
#define LI_ABS(x) fabsf(x)
#define LI_POW(x,y) gfk_pow((x),(y))
#define LI_EXP(x) gfk_exp(x)
#define LI_C(x) x##f
#endif

#define LI_CP_ICE LI_C(2009.0)
#define LI_LATENT_HEAT_ICE LI_C(335.0e3)
#define LI_RHO_WATER LI_C(1000.0)
#define LI_GRAVITY LI_C(9.80616)
#define LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE LI_C(9.7456e-8)
#define LI_KELVIN_TO_CELSIUS LI_C(273.15)
#define LI_PMPT_OFFSET LI_C(2.0)   // li_init_linear_temperature_in_column pmpt_offset

__device__ __forceinline__ rk li_min(rk a, rk b) { return a < b ? a : b; }

// pressure_melting_point(depth): - c * rhoi * gravity * depth (deg C)
__device__ __forceinline__ rk li_pmp(rk rhoi, rk depth) {
    return -M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), depth);
}
// pressure_melting_point_column: - c * rhoi * gravity * thickness * sigma(k)
__device__ __forceinline__ rk li_pmp_column(rk rhoi, rk thickness, rk sigma) {
    return -M(M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), thickness), sigma);
}

// li_thermal_init's per-cell work, one thread per cell over nCells.
// airOption: 1 'constant', 2 'lapse', 0 'file' (left as read).
// heatFluxOption: 1 'constant', 0 'file'.
// initOption: 1 'sfc_air_temperature', 2 'linear', 3 'file', 0 restart.
extern "C" __global__ void li_thermal_init_v842(
    rk *temperature, rk *waterfrac, rk *surfaceAirTemperature, rk *surfaceTemperature,
    rk *basalTemperature, rk *basalHeatFlux, const rk *thickness, const rk *upperSurface,
    const rk *layerCenterSigma, rk rhoi, int airOption, rk airValue, rk lapseRate,
    int heatFluxOption, rk heatFluxValue, int initOption, int nCells, int nCellsSolve, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nVertLevels;
    if (airOption == 1) surfaceAirTemperature[iCell] = airValue;
    else if (airOption == 2) surfaceAirTemperature[iCell] = S(airValue, M(lapseRate, upperSurface[iCell]));
    if (heatFluxOption == 1) basalHeatFlux[iCell] = heatFluxValue;
    const rk sat = surfaceAirTemperature[iCell];
    const rk h = thickness[iCell];
    if (initOption == 3) {
        for (int k = 0; k < nVertLevels; ++k) waterfrac[base + k] = LI_C(0.0);
    } else if (initOption == 1) {
        if (iCell < nCellsSolve) {
            surfaceTemperature[iCell] = li_min(sat, LI_KELVIN_TO_CELSIUS);
            for (int k = 0; k < nVertLevels; ++k)
                temperature[base + k] = li_min(sat, A(li_pmp_column(rhoi, h, layerCenterSigma[k]),
                                                      LI_KELVIN_TO_CELSIUS));
            basalTemperature[iCell] = li_min(sat, A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS));
        }
        for (int k = 0; k < nVertLevels; ++k) waterfrac[base + k] = LI_C(0.0);
    } else if (initOption == 2) {
        for (int k = 0; k < nVertLevels; ++k) temperature[base + k] = LI_KELVIN_TO_CELSIUS;
        if (iCell < nCellsSolve) {
            // li_init_linear_temperature_in_column
            const rk sT = li_min(sat, LI_KELVIN_TO_CELSIUS);
            const rk pmptempBed = A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS);
            const rk bT = S(pmptempBed, LI_PMPT_OFFSET);
            surfaceTemperature[iCell] = sT;
            basalTemperature[iCell] = bT;
            for (int k = 0; k < nVertLevels; ++k) {
                const rk pmptemp = A(li_pmp_column(rhoi, h, layerCenterSigma[k]), LI_KELVIN_TO_CELSIUS);
                const rk t = A(sT, M(S(bT, sT), layerCenterSigma[k]));
                temperature[base + k] = li_min(t, S(pmptemp, LI_PMPT_OFFSET));
                waterfrac[base + k] = LI_C(0.0);
            }
        }
    }
}

// li_compute_pressure_melting_point_fields, one thread per cell (K)
extern "C" __global__ void li_pressure_melting_point_fields_v842(
    rk *pmpTemperature, rk *basalPmpTemperature, const rk *thickness, const rk *layerCenterSigma,
    rk rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk h = thickness[iCell];
    for (int k = 0; k < nVertLevels; ++k)
        pmpTemperature[iCell * nVertLevels + k] = A(li_pmp_column(rhoi, h, layerCenterSigma[k]), LI_KELVIN_TO_CELSIUS);
    basalPmpTemperature[iCell] = A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS);
}

// li_temperature_to_enthalpy, one thread per cell
extern "C" __global__ void li_temperature_to_enthalpy_v842(
    rk *enthalpy, const rk *temperature, const rk *waterfrac, const rk *thickness,
    const rk *layerCenterSigma, rk rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk h = thickness[iCell];
    for (int k = 0; k < nVertLevels; ++k) {
        const int p = iCell * nVertLevels + k;
        const rk pmp = li_pmp_column(rhoi, h, layerCenterSigma[k]);
        enthalpy[p] = A(M(M(M(S(LI_C(1.0), waterfrac[p]), rhoi), LI_CP_ICE), temperature[p]),
                        M(M(waterfrac[p], LI_RHO_WATER), A(M(LI_CP_ICE, pmp), LI_LATENT_HEAT_ICE)));
    }
}

// li_enthalpy_to_temperature with the surface and basal enthalpies, one thread per cell
extern "C" __global__ void li_enthalpy_to_temperature_v842(
    rk *enthalpy, rk *temperature, rk *waterfrac, rk *surfaceEnthalpy, rk *surfaceTemperature,
    rk *basalEnthalpy, rk *basalTemperature, const rk *thickness, const rk *layerCenterSigma,
    rk rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk h = thickness[iCell];
    const rk pmpTemperatureBed = li_pmp(rhoi, h);
    const rk pmpEnthalpyBed = M(M(pmpTemperatureBed, rhoi), LI_CP_ICE);
    for (int k = 0; k < nVertLevels; ++k) {
        const int p = iCell * nVertLevels + k;
        const rk pmpTemperature = li_pmp_column(rhoi, h, layerCenterSigma[k]);
        const rk pmpEnthalpy = M(M(pmpTemperature, rhoi), LI_CP_ICE);
        if (enthalpy[p] >= pmpEnthalpy) {   // temperate ice
            temperature[p] = pmpTemperature;
            waterfrac[p] = D(S(enthalpy[p], pmpEnthalpy),
                             A(M(M(S(LI_RHO_WATER, rhoi), LI_CP_ICE), pmpTemperature),
                               M(LI_RHO_WATER, LI_LATENT_HEAT_ICE)));
        } else {                            // cold ice
            temperature[p] = D(enthalpy[p], M(rhoi, LI_CP_ICE));
            waterfrac[p] = LI_C(0.0);
        }
    }
    if (surfaceEnthalpy[iCell] >= LI_C(0.0)) {
        surfaceTemperature[iCell] = LI_C(0.0);
        surfaceEnthalpy[iCell] = LI_C(0.0);
    } else {
        surfaceTemperature[iCell] = D(surfaceEnthalpy[iCell], M(rhoi, LI_CP_ICE));
    }
    if (basalEnthalpy[iCell] >= pmpEnthalpyBed) {
        basalTemperature[iCell] = pmpTemperatureBed;
        basalEnthalpy[iCell] = pmpEnthalpyBed;
    } else {
        basalTemperature[iCell] = D(basalEnthalpy[iCell], M(rhoi, LI_CP_ICE));
    }
}
