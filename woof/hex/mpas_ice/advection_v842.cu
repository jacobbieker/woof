// MALI v8.4.2: first-order upwind thickness and tracer advection, the
// surface and basal mass balance, and the vertical remap.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_advection.F:
//     74-517    li_advection_thickness_tracers ('fo' and 'none')
//     612-772   apply_mass_balance
//     788-940   tracer_setup
//     955-1060  tracer_finish
//     1076-1378 advect_thickness_tracers_upwind
//     1522-1671 vertical_remap
//   src/core_landice/shared/mpas_li_setup.F:553-609 li_calculate_layerThickness
//   src/core_landice/shared/mpas_li_mask.F: li_mask_is_ice, _floating_ice,
//     _grounded_ice (the logical forms)
//   with li_constants' cp_ice, kelvin_to_celsius and scyr.
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// Every operation is an explicit round-to-nearest intrinsic in the
// Fortran's evaluation order (gfortran -O0 -ffp-contract=off: left to right,
// parentheses kept, no contraction). sum(a, 1) is gfortran's inline sum, an
// accumulator from 0 in index order. sign(1.0, x) is copysign(1.0, x).
// maxTracersAdvect is 1 (Registry.xml), so every tracer array holds one
// tracer. RKIND = 4 (MALI's default build) unless MPAS_RKIND8 is defined.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// a(1, nVertLevels, nCells) is the same; a(maxEdges, nCells) is
// a[iCell*maxEdges + i]; cellsOnEdge(2, nEdges) is cellsOnEdge[2*iEdge + j].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_SIGN1(x) copysign(1.0, (x))
#define LI_CP_ICE 2009.0                   // cp_ice = 2009.0_RKIND
#define LI_K2C 273.15                      // kelvin_to_celsius = 273.15_RKIND
#define LI_SCYR 31536000.0                 // scyr = 31536000.0_RKIND
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_SIGN1(x) copysignf(1.0f, (x))
#define LI_CP_ICE 2009.0f
#define LI_K2C 273.15f
#define LI_SCYR 31536000.0f
#endif

#define LI_MASK_ICE 32                     // li_mask_ValueIce
#define LI_MASK_FLOATING 4                 // li_mask_ValueFloating
#define LI_MAX_LEVELS 128

__device__ __forceinline__ rk li_max(rk a, rk b) { return a > b ? a : b; }
__device__ __forceinline__ rk li_min(rk a, rk b) { return a < b ? a : b; }

// li_calculate_layerThickness: one thread per cell, over nCells.
extern "C" __global__ void li_calculate_layer_thickness_v842(
    rk *layerThickness, const rk *thickness, const rk *layerThicknessFractions, int nCells, int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    for (int k = 0; k < nLevels; ++k)
        layerThickness[iCell * nLevels + k] = M(thickness[iCell], layerThicknessFractions[k]);
}

// tracer_setup after the enthalpy conversion (or not): one thread per cell,
// over nCells. mode 1 'temperature': the tracer is the temperature, the new
// surface ice min(T_air, 273.15) and the new basal ice the basal temperature.
// mode 2 'enthalpy': the tracer is the enthalpy, and the new ice's tracers are
// those temperatures times config_ice_density*cp_ice. mode 0 (no tracer
// advection, or no thermal solver): the tracer is 0 and the new ice's
// tracers are not set (the model leaves its scratch array as it was).
extern "C" __global__ void li_tracer_setup_v842(
    rk *tracers, rk *surfaceTracers, rk *basalTracers, const rk *temperature, const rk *enthalpy,
    const rk *surfaceAirTemperature, const rk *basalTemperature, rk rhoi, int mode, int nCells, int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    for (int k = 0; k < nLevels; ++k) {
        const int p = iCell * nLevels + k;
        tracers[p] = mode == 1 ? temperature[p] : (mode == 2 ? enthalpy[p] : (rk)0.0);
    }
    const rk surface = li_min(surfaceAirTemperature[iCell], LI_K2C);
    if (mode == 1) {
        surfaceTracers[iCell] = surface;
        basalTracers[iCell] = basalTemperature[iCell];
    } else if (mode == 2) {
        surfaceTracers[iCell] = M(M(surface, rhoi), LI_CP_ICE);
        basalTracers[iCell] = M(M(basalTemperature[iCell], rhoi), LI_CP_ICE);
    }
}

// advect_thickness_tracers_upwind: one thread per cell, over nCellsSolve.
// Reads the old layer thickness and tracer, writes the new.
extern "C" __global__ void li_advect_upwind_v842(
    rk *layerThicknessNew, rk *tracersNew, const rk *layerThicknessOld, const rk *tracersOld,
    const rk *layerNormalVelocity, const rk *layerThicknessEdge, const rk *dvEdge, const rk *areaCell,
    const int *nEdgesOnCell, const int *edgesOnCell, const int *edgeSignOnCell, const int *cellsOnEdge,
    rk dt, int nCellsSolve, int nLevels, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const rk invAreaCell = D((rk)1.0, areaCell[iCell]);
    for (int k = 0; k < nLevels; ++k) {
        rk thicknessTendency = (rk)0.0, tracerTendency = (rk)0.0;
        for (int e = 0; e < nEdgesOnCell[iCell]; ++e) {
            const int iEdge = edgesOnCell[iCell * maxEdges + e] - 1;
            const int iCell1 = cellsOnEdge[2 * iEdge] - 1;
            const int iCell2 = cellsOnEdge[2 * iEdge + 1] - 1;
            const rk vel = layerNormalVelocity[iEdge * nLevels + k];
            const rk flux = M(M(vel, dvEdge[iEdge]), layerThicknessEdge[iEdge * nLevels + k]);
            const rk signedFlux = M(M((rk)edgeSignOnCell[iCell * maxEdges + e], flux), invAreaCell);
            thicknessTendency = A(thicknessTendency, signedFlux);
            const rk velSign = LI_SIGN1(vel);
            const rk edgeTracer = li_max(M(velSign, tracersOld[iCell1 * nLevels + k]),
                                         M(M(velSign, (rk)-1.0), tracersOld[iCell2 * nLevels + k]));
            tracerTendency = A(tracerTendency, M(signedFlux, edgeTracer));
        }
        const int p = iCell * nLevels + k;
        const rk newThicknessTracer = A(M(layerThicknessOld[p], tracersOld[p]), M(tracerTendency, dt));
        const rk newThickness = A(layerThicknessOld[p], M(thicknessTendency, dt));
        layerThicknessNew[p] = newThickness;
        tracersNew[p] = newThickness > (rk)0.0 ? D(newThicknessTracer, newThickness) : (rk)0.0;
    }
}

// thickness = sum(layerThickness, 1), with (when dynamicThickening is given)
// dynamicThickening = (sum(layerThickness, 1) - thickness) / dt * scyr
// first: one thread per cell, over nCells.
extern "C" __global__ void li_column_thickness_v842(
    rk *thickness, rk *dynamicThickening, const rk *layerThickness, rk dt, int withThickening, int nCells,
    int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk total = (rk)0.0;
    for (int k = 0; k < nLevels; ++k) total = A(total, layerThickness[iCell * nLevels + k]);
    if (withThickening) dynamicThickening[iCell] = M(D(S(total, thickness[iCell]), dt), LI_SCYR);
    thickness[iCell] = total;
}

// The two where blocks before apply_mass_balance: one thread per cell.
extern "C" __global__ void li_mass_balance_masks_v842(
    rk *sfcMassBal, rk *basalMassBal, const rk *bedTopography, const rk *groundedBasalMassBal,
    const rk *floatingBasalMassBal, const int *cellMask, rk seaLevel, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int mask = cellMask[iCell];
    const bool ice = (mask & LI_MASK_ICE) == LI_MASK_ICE;
    const bool floating = (mask & LI_MASK_FLOATING) == LI_MASK_FLOATING;
    if (sfcMassBal[iCell] > (rk)0.0 && bedTopography[iCell] < seaLevel && !ice) sfcMassBal[iCell] = (rk)0.0;
    if (!floating && ice) basalMassBal[iCell] = groundedBasalMassBal[iCell];
    else if (floating) basalMassBal[iCell] = floatingBasalMassBal[iCell];
    else if (!ice) basalMassBal[iCell] = (rk)0.0;
}

// apply_mass_balance: one thread per cell, over nCells. Accumulation adds a
// layer of new ice at the surface (base) and mixes its tracer in; ablation
// melts whole layers from the surface (base) down (up) and then part of one.
extern "C" __global__ void li_apply_mass_balance_v842(
    rk *layerThickness, rk *tracers, const rk *sfcMassBal, const rk *basalMassBal, const rk *surfaceTracers,
    const rk *basalTracers, rk dt, rk rhoi, int nCells, int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk *h = layerThickness + iCell * nLevels;
    rk *t = tracers + iCell * nLevels;
    const rk sfc = sfcMassBal[iCell];
    if (sfc > (rk)0.0) {
        const rk accum = D(M(sfc, dt), rhoi);
        const rk product = A(M(h[0], t[0]), M(accum, surfaceTracers[iCell]));
        h[0] = A(h[0], accum);
        t[0] = D(product, h[0]);
    } else if (sfc < (rk)0.0) {
        rk ablat = -D(M(sfc, dt), rhoi);
        for (int k = 0; k < nLevels; ++k) {
            if (ablat > h[k]) {
                ablat = S(ablat, h[k]);
                h[k] = (rk)0.0;
                t[k] = (rk)0.0;
            } else {
                h[k] = S(h[k], ablat);
                break;
            }
        }
    }
    const rk basal = basalMassBal[iCell];
    const int b = nLevels - 1;
    if (basal > (rk)0.0) {
        const rk accum = D(M(basal, dt), rhoi);
        const rk product = A(M(h[b], t[b]), M(accum, basalTracers[iCell]));
        h[b] = A(h[b], accum);
        t[b] = D(product, h[b]);
    } else if (basal < (rk)0.0) {
        rk ablat = -D(M(basal, dt), rhoi);
        for (int k = b; k >= 0; --k) {
            if (ablat > h[k]) {
                ablat = S(ablat, h[k]);
                h[k] = (rk)0.0;
                t[k] = (rk)0.0;
            } else {
                h[k] = S(h[k], ablat);
                break;
            }
        }
    }
}

// vertical_remap: one thread per cell with ice, over nCells. The layers go
// back to layerThicknessFractions of the thickness, and the tracer is the
// overlap-weighted mean of the old layers' over each new one.
extern "C" __global__ void li_vertical_remap_v842(
    rk *layerThickness, rk *tracers, const rk *thickness, const rk *layerThicknessFractions,
    const rk *layerInterfaceSigma, int nCells, int nLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk thk = thickness[iCell];
    if (!(thk > (rk)0.0)) return;
    rk *h = layerThickness + iCell * nLevels;
    rk *t = tracers + iCell * nLevels;
    rk sigmaIn[LI_MAX_LEVELS + 1], hTsum[LI_MAX_LEVELS];
    sigmaIn[0] = (rk)0.0;
    for (int k = 1; k < nLevels; ++k) sigmaIn[k] = A(sigmaIn[k - 1], D(h[k - 1], thk));
    sigmaIn[nLevels] = (rk)1.0;
    for (int k = 0; k < nLevels; ++k) h[k] = M(layerThicknessFractions[k], thk);
    for (int k2 = 0; k2 < nLevels; ++k2) {
        rk sum = (rk)0.0;
        for (int k1 = 0; k1 < nLevels; ++k1) {
            const rk zhi = li_min(sigmaIn[k1 + 1], layerInterfaceSigma[k2 + 1]);
            const rk zlo = li_max(sigmaIn[k1], layerInterfaceSigma[k2]);
            const rk overlap = M(li_max(S(zhi, zlo), (rk)0.0), thk);
            sum = A(sum, M(t[k1], overlap));
        }
        hTsum[k2] = sum;
    }
    for (int k = 0; k < nLevels; ++k) t[k] = D(hTsum[k], h[k]);
}

// Whole-array copy (a(:,:) = b(:,:)) over n words.
extern "C" __global__ void li_copy_v842(rk *dst, const rk *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
