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
// RKIND = 4. Every operation is an explicit round-to-nearest intrinsic in the
// Fortran's evaluation order (gfortran -O0 -ffp-contract=off). rhoi is the
// module's copy of config_ice_density that li_thermal_init makes.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k].

#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

#define LI_CP_ICE 2009.0f
#define LI_LATENT_HEAT_ICE 335.0e3f
#define LI_RHO_WATER 1000.0f
#define LI_GRAVITY 9.80616f
#define LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE 9.7456e-8f
#define LI_KELVIN_TO_CELSIUS 273.15f
#define LI_PMPT_OFFSET 2.0f   // li_init_linear_temperature_in_column pmpt_offset

__device__ __forceinline__ float li_min(float a, float b) { return a < b ? a : b; }

// pressure_melting_point(depth): - c * rhoi * gravity * depth (deg C)
__device__ __forceinline__ float li_pmp(float rhoi, float depth) {
    return -M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), depth);
}
// pressure_melting_point_column: - c * rhoi * gravity * thickness * sigma(k)
__device__ __forceinline__ float li_pmp_column(float rhoi, float thickness, float sigma) {
    return -M(M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), thickness), sigma);
}

// li_thermal_init's per-cell work, one thread per cell over nCells.
// airOption: 1 'constant', 2 'lapse', 0 'file' (left as read).
// heatFluxOption: 1 'constant', 0 'file'.
// initOption: 1 'sfc_air_temperature', 2 'linear', 3 'file', 0 restart.
extern "C" __global__ void li_thermal_init_v842(
    float *temperature, float *waterfrac, float *surfaceAirTemperature, float *surfaceTemperature,
    float *basalTemperature, float *basalHeatFlux, const float *thickness, const float *upperSurface,
    const float *layerCenterSigma, float rhoi, int airOption, float airValue, float lapseRate,
    int heatFluxOption, float heatFluxValue, int initOption, int nCells, int nCellsSolve, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nVertLevels;
    if (airOption == 1) surfaceAirTemperature[iCell] = airValue;
    else if (airOption == 2) surfaceAirTemperature[iCell] = S(airValue, M(lapseRate, upperSurface[iCell]));
    if (heatFluxOption == 1) basalHeatFlux[iCell] = heatFluxValue;
    const float sat = surfaceAirTemperature[iCell];
    const float h = thickness[iCell];
    if (initOption == 3) {
        for (int k = 0; k < nVertLevels; ++k) waterfrac[base + k] = 0.0f;
    } else if (initOption == 1) {
        if (iCell < nCellsSolve) {
            surfaceTemperature[iCell] = li_min(sat, LI_KELVIN_TO_CELSIUS);
            for (int k = 0; k < nVertLevels; ++k)
                temperature[base + k] = li_min(sat, A(li_pmp_column(rhoi, h, layerCenterSigma[k]),
                                                      LI_KELVIN_TO_CELSIUS));
            basalTemperature[iCell] = li_min(sat, A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS));
        }
        for (int k = 0; k < nVertLevels; ++k) waterfrac[base + k] = 0.0f;
    } else if (initOption == 2) {
        for (int k = 0; k < nVertLevels; ++k) temperature[base + k] = LI_KELVIN_TO_CELSIUS;
        if (iCell < nCellsSolve) {
            // li_init_linear_temperature_in_column
            const float sT = li_min(sat, LI_KELVIN_TO_CELSIUS);
            const float pmptempBed = A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS);
            const float bT = S(pmptempBed, LI_PMPT_OFFSET);
            surfaceTemperature[iCell] = sT;
            basalTemperature[iCell] = bT;
            for (int k = 0; k < nVertLevels; ++k) {
                const float pmptemp = A(li_pmp_column(rhoi, h, layerCenterSigma[k]), LI_KELVIN_TO_CELSIUS);
                const float t = A(sT, M(S(bT, sT), layerCenterSigma[k]));
                temperature[base + k] = li_min(t, S(pmptemp, LI_PMPT_OFFSET));
                waterfrac[base + k] = 0.0f;
            }
        }
    }
}

// li_compute_pressure_melting_point_fields, one thread per cell (K)
extern "C" __global__ void li_pressure_melting_point_fields_v842(
    float *pmpTemperature, float *basalPmpTemperature, const float *thickness, const float *layerCenterSigma,
    float rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const float h = thickness[iCell];
    for (int k = 0; k < nVertLevels; ++k)
        pmpTemperature[iCell * nVertLevels + k] = A(li_pmp_column(rhoi, h, layerCenterSigma[k]), LI_KELVIN_TO_CELSIUS);
    basalPmpTemperature[iCell] = A(li_pmp(rhoi, h), LI_KELVIN_TO_CELSIUS);
}

// li_temperature_to_enthalpy, one thread per cell
extern "C" __global__ void li_temperature_to_enthalpy_v842(
    float *enthalpy, const float *temperature, const float *waterfrac, const float *thickness,
    const float *layerCenterSigma, float rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const float h = thickness[iCell];
    for (int k = 0; k < nVertLevels; ++k) {
        const int p = iCell * nVertLevels + k;
        const float pmp = li_pmp_column(rhoi, h, layerCenterSigma[k]);
        enthalpy[p] = A(M(M(M(S(1.0f, waterfrac[p]), rhoi), LI_CP_ICE), temperature[p]),
                        M(M(waterfrac[p], LI_RHO_WATER), A(M(LI_CP_ICE, pmp), LI_LATENT_HEAT_ICE)));
    }
}

// li_enthalpy_to_temperature with the surface and basal enthalpies, one thread per cell
extern "C" __global__ void li_enthalpy_to_temperature_v842(
    float *enthalpy, float *temperature, float *waterfrac, float *surfaceEnthalpy, float *surfaceTemperature,
    float *basalEnthalpy, float *basalTemperature, const float *thickness, const float *layerCenterSigma,
    float rhoi, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const float h = thickness[iCell];
    const float pmpTemperatureBed = li_pmp(rhoi, h);
    const float pmpEnthalpyBed = M(M(pmpTemperatureBed, rhoi), LI_CP_ICE);
    for (int k = 0; k < nVertLevels; ++k) {
        const int p = iCell * nVertLevels + k;
        const float pmpTemperature = li_pmp_column(rhoi, h, layerCenterSigma[k]);
        const float pmpEnthalpy = M(M(pmpTemperature, rhoi), LI_CP_ICE);
        if (enthalpy[p] >= pmpEnthalpy) {   // temperate ice
            temperature[p] = pmpTemperature;
            waterfrac[p] = D(S(enthalpy[p], pmpEnthalpy),
                             A(M(M(S(LI_RHO_WATER, rhoi), LI_CP_ICE), pmpTemperature),
                               M(LI_RHO_WATER, LI_LATENT_HEAT_ICE)));
        } else {                            // cold ice
            temperature[p] = D(enthalpy[p], M(rhoi, LI_CP_ICE));
            waterfrac[p] = 0.0f;
        }
    }
    if (surfaceEnthalpy[iCell] >= 0.0f) {
        surfaceTemperature[iCell] = 0.0f;
        surfaceEnthalpy[iCell] = 0.0f;
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
