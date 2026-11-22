// MPAS-Ocean v8.4.2: land-ice fluxes ('standalone' and 'coupled' modes) and
// the effective density in land ice.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_diagnostics.F (ocn_compute_land_ice_flux_input_fields)
//   src/core_ocean/shared/mpas_ocn_surface_land_ice_fluxes.F (build_arrays with
//     compute_melt_fluxes, _thick, _vel, _active_tracers)
//   src/core_ocean/shared/mpas_ocn_effective_density_in_land_ice.F
//   src/core_ocean/shared/mpas_ocn_time_average_coupled.F (the land-ice means)
//   src/core_ocean/shared/mpas_ocn_equation_of_state.F (ocn_freezing_temperature
//     and its salinity derivative, in the land-ice cavity)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); sqrt is the correctly
// rounded instruction gfortran emits and x**2 is x*x. Fortran MIN/MAX are
// chosen on the words as integer selects (MAX(a, b) is b when b > a, else a).
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nl + k]; a two-member var_array
// (landIceBoundaryLayerTracers, landIceInterfaceTracers,
// landIceTracerTransferVelocities) is a[iCell*2 + i]; activeTracers
// (nTracers, nVertLevels, nCells+1) is t[(iCell*nl + k)*nTracers + i];
// cellsOnEdge/cellsOnCell hold 1-based cells (nCells+1 is the missing cell);
// indices are 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define C(x) x
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define C(x) x##f
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif
#define GRAVITY C(9.80616)
__device__ __forceinline__ int li_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ocn_freezing_temperature and its salinity derivative in the land-ice
// cavity: c[0..4] = coeff_0, coeff_S, coeff_p, coeff_pS, reference_pressure
__device__ __forceinline__ rk li_offset(rk pressure, const rk *c)
{
    return li_max(S(pressure, c[4]), C(0.0));
}
__device__ __forceinline__ rk li_freezing(rk salinity, rk pressure, const rk *c)
{
    const rk pressureOffset = li_offset(pressure, c);
    return A(A(A(c[0], M(c[1], salinity)), M(c[2], pressureOffset)), M(M(c[3], pressureOffset), salinity));
}
__device__ __forceinline__ rk li_freezing_deriv(rk pressure, const rk *c)
{
    return A(c[1], M(c[3], li_offset(pressure, c)));
}

// ---------------------------------------------------------------- the input fields (diagnostics)

// the land-ice mask from the fraction, cells 1..nCells (every mode but 'off')
// is diagnostics_v842's ocn_diag_land_ice_mask_v842.

// top drag on edges 1..nEdges
extern "C" __global__ void ocn_li_top_drag_v842(
    rk *topDrag, const rk *kineticEnergyCell, const rk *landIceFraction, const rk *normalVelocity,
    const int *cellsOnEdge, int nEdges, int nl, rk rhoSw, rk topDragCoeff)
{
    const int iEdge = li_thread();
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    const rk velocityMagnitude = SQRT(A(kineticEnergyCell[(size_t)cell1 * nl], kineticEnergyCell[(size_t)cell2 * nl]));
    const rk landIceEdgeFraction = M(C(0.5), A(landIceFraction[cell1], landIceFraction[cell2]));
    topDrag[iEdge] = -M(M(M(M(rhoSw, landIceEdgeFraction), topDragCoeff), velocityMagnitude),
                        normalVelocity[(size_t)iEdge * nl]);
}

// the top drag's magnitude and the friction velocity, cells 1..nCells
extern "C" __global__ void ocn_li_friction_v842(
    rk *topDragMagnitude, rk *landIceFrictionVelocity, const rk *kineticEnergyCell, const rk *landIceFraction,
    int nCells, int nl, rk rhoSw, rk topDragCoeff, rk rmsTidalVelocity)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    const rk ke = kineticEnergyCell[(size_t)iCell * nl];
    topDragMagnitude[iCell] = M(M(M(M(rhoSw, landIceFraction[iCell]), C(2.0)), topDragCoeff), ke);
    landIceFrictionVelocity[iCell] = SQRT(M(topDragCoeff, A(M(C(2.0), ke), M(rmsTidalVelocity, rmsTidalVelocity))));
}

// the boundary layer's mean temperature and salinity, cells 1..nCells, into
// the two scratch arrays
extern "C" __global__ void ocn_li_boundary_layer_v842(
    rk *blTempScratch, rk *blSaltScratch, const rk *layerThickness, const rk *activeTracers,
    const int *maxLevelCell, int nCells, int nl, int nTracers, int indexT, int indexS, rk boundaryLayerThickness)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    rk blThickness = C(0.0), t = C(0.0), s = C(0.0);
    for (int k = 0; k < maxLevelCell[iCell]; ++k) {
        const size_t at = (size_t)iCell * nl + k;
        const rk dz = li_min(layerThickness[at], S(boundaryLayerThickness, blThickness));
        if (dz <= C(0.0)) break;
        t = A(t, M(activeTracers[at * nTracers + indexT - 1], dz));
        s = A(s, M(activeTracers[at * nTracers + indexS - 1], dz));
        blThickness = A(blThickness, dz);
    }
    if (blThickness > C(0.0)) {
        t = D(t, blThickness);
        s = D(s, blThickness);
    }
    blTempScratch[iCell] = t;
    blSaltScratch[iCell] = s;
}

// landIceBoundaryLayerTracers, with the neighbours' weight when it is
// positive, cells 1..nCells
extern "C" __global__ void ocn_li_boundary_layer_tracers_v842(
    rk *landIceBoundaryLayerTracers, const rk *blTempScratch, const rk *blSaltScratch, const int *cellsOnCell,
    const int *nEdgesOnCell, const int *cellMask, int nCells, int nl, int maxEdges, int indexBLT, int indexBLS,
    rk neighborWeight)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    rk *bl = landIceBoundaryLayerTracers + (size_t)iCell * 2;
    bl[indexBLT - 1] = blTempScratch[iCell];
    bl[indexBLS - 1] = blSaltScratch[iCell];
    if (neighborWeight > C(0.0)) {
        rk weightSum = C(1.0);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int cell2 = cellsOnCell[(size_t)iCell * maxEdges + i] - 1;
            const rk mask = (rk)cellMask[(size_t)cell2 * nl];
            bl[indexBLT - 1] = A(bl[indexBLT - 1], M(M(mask, neighborWeight), blTempScratch[cell2]));
            bl[indexBLS - 1] = A(bl[indexBLS - 1], M(M(mask, neighborWeight), blSaltScratch[cell2]));
            weightSum = A(weightSum, M(mask, neighborWeight));
        }
        bl[0] = D(bl[0], weightSum);
        bl[1] = D(bl[1], weightSum);
    }
}

// Jenkins: the transfer velocities from the friction velocity, cells 1..nCells
extern "C" __global__ void ocn_li_jenkins_transfer_v842(
    rk *landIceTracerTransferVelocities, const rk *landIceFrictionVelocity, int nCells, int indexHeatTrans,
    int indexSaltTrans, rk heatCoefficient, rk saltCoefficient)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    rk *tv = landIceTracerTransferVelocities + (size_t)iCell * 2;
    tv[indexHeatTrans - 1] = M(landIceFrictionVelocity[iCell], heatCoefficient);
    tv[indexSaltTrans - 1] = M(landIceFrictionVelocity[iCell], saltCoefficient);
}

// the land-ice attenuation coefficient under land ice, cells 1..nCells
extern "C" __global__ void ocn_li_attenuation_v842(
    rk *surfaceFluxAttenuationCoefficient, const int *landIceMask, int nCells, rk attenuation)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    if (landIceMask[iCell] == 1) surfaceFluxAttenuationCoefficient[iCell] = attenuation;
}

// ---------------------------------------------------------------- build_arrays ('standalone')

// p = rho_sw, cp_sw, latent_heat_fusion_mks, ISOMIP gammaT, cp_land_ice;
// cavity = the land-ice cavity's five freezing coefficients

// ISOMIP, cells 1..nCellsSolve. v8.4.2 passes the boundary-layer temperature
// as the freezing point's salinity (an oracle quirk): indexFreezeS is indexBLT
// for that word-exact default, indexBLS for the corrected opt-in
// (woof_land_ice_isomip_freezing_salinity), which takes the freezing point at
// the boundary layer's salinity.
extern "C" __global__ void ocn_li_isomip_v842(
    rk *landIceInterfaceTracers, rk *landIceFreshwaterFlux, rk *landIceHeatFlux, rk *heatFluxToLandIce,
    const rk *landIceBoundaryLayerTracers, const rk *landIcePressure, const rk *landIceFraction,
    const int *landIceMask, const rk *p, const rk *cavity, int nCells, int indexBLT, int indexIT, int indexFreezeS)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    if (landIceMask[iCell] == 0) return;
    const rk rhoSw = p[0], cpSw = p[1], latent = p[2], gammaT = p[3];
    const rk blt = landIceBoundaryLayerTracers[(size_t)iCell * 2 + indexBLT - 1];
    const rk it = li_freezing(landIceBoundaryLayerTracers[(size_t)iCell * 2 + indexFreezeS - 1],
                              landIcePressure[iCell], cavity);
    landIceInterfaceTracers[(size_t)iCell * 2 + indexIT - 1] = it;
    const rk freshwaterFlux = -M(M(M(rhoSw, gammaT), D(cpSw, latent)), S(it, blt));
    landIceFreshwaterFlux[iCell] = M(landIceFraction[iCell], freshwaterFlux);
    const rk heatFlux = M(cpSw, A(M(freshwaterFlux, it), M(M(rhoSw, gammaT), S(it, blt))));
    landIceHeatFlux[iCell] = M(landIceFraction[iCell], heatFlux);
    heatFluxToLandIce[iCell] = C(0.0);
}

// Jenkins: compute_melt_fluxes without the ice temperature (nu = 0), then the
// fluxes scaled by the land-ice fraction, cells 1..nCellsSolve
extern "C" __global__ void ocn_li_jenkins_melt_v842(
    rk *landIceInterfaceTracers, rk *landIceFreshwaterFlux, rk *landIceHeatFlux, rk *heatFluxToLandIce,
    const rk *landIceBoundaryLayerTracers, const rk *landIceTracerTransferVelocities, const rk *landIcePressure,
    const rk *landIceFraction, const int *landIceMask, const rk *p, const rk *cavity, int nCells, int indexBLT,
    int indexBLS, int indexHeatTrans, int indexSaltTrans, int indexIT, int indexIS)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    if (landIceMask[iCell] == 0) return;
    const rk rhoSw = p[0], cpSw = p[1], latent = p[2], cpLandIce = p[4];
    const rk Tlatent = D(latent, cpSw);
    const rk nu = C(0.0), iceDeltaT = C(0.0);
    const rk *bl = landIceBoundaryLayerTracers + (size_t)iCell * 2;
    const rk *tv = landIceTracerTransferVelocities + (size_t)iCell * 2;
    const rk oceanT = bl[indexBLT - 1], oceanS = bl[indexBLS - 1];
    const rk heatTV = tv[indexHeatTrans - 1], saltTV = tv[indexSaltTrans - 1];
    const rk T0 = li_freezing(C(0.0), landIcePressure[iCell], cavity);
    const rk dTf_dS = li_freezing_deriv(landIcePressure[iCell], cavity);
    const rk ratio = D(saltTV, heatTV);
    const rk a = -M(dTf_dS, A(C(1.0), nu));
    const rk b = S(A(S(M(ratio, Tlatent), M(nu, iceDeltaT)), oceanT), T0);
    const rk c = -M(M(ratio, Tlatent), li_max(oceanS, C(0.0)));
    const rk interfaceS = li_max(D(-M(C(2.0), c), A(b, SQRT(S(M(b, b), M(M(C(4.0), a), c))))), C(0.001));
    const rk interfaceT = A(M(dTf_dS, interfaceS), T0);
    const rk freshwaterFlux = M(M(rhoSw, saltTV), S(D(oceanS, interfaceS), C(1.0)));
    const rk oceanHeatFlux = M(cpSw, S(M(freshwaterFlux, interfaceT), M(M(rhoSw, heatTV), S(oceanT, interfaceT))));
    const rk iceHeatFlux = -M(M(cpLandIce, freshwaterFlux), interfaceT);
    rk *it = landIceInterfaceTracers + (size_t)iCell * 2;
    it[indexIT - 1] = interfaceT;
    it[indexIS - 1] = interfaceS;
    const rk fraction = landIceFraction[iCell];
    landIceFreshwaterFlux[iCell] = M(fraction, freshwaterFlux);
    landIceHeatFlux[iCell] = M(fraction, oceanHeatFlux);
    heatFluxToLandIce[iCell] = M(fraction, iceHeatFlux);
}

// the accumulated land-ice mass and heat at time level 2, cells 1..nCellsSolve
extern "C" __global__ void ocn_li_accumulate_v842(
    rk *accumulatedLandIceMassNew, rk *accumulatedLandIceHeatNew, const rk *accumulatedLandIceMassOld,
    const rk *accumulatedLandIceHeatOld, const rk *landIceFreshwaterFlux, const rk *heatFluxToLandIce, int nCells,
    rk dt)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    accumulatedLandIceMassNew[iCell] = S(accumulatedLandIceMassOld[iCell], M(dt, landIceFreshwaterFlux[iCell]));
    accumulatedLandIceHeatNew[iCell] = A(accumulatedLandIceHeatOld[iCell], M(dt, heatFluxToLandIce[iCell]));
}

// ---------------------------------------------------------------- the tendencies' terms

// ocn_surface_land_ice_fluxes_thick, cells 1..nCells
extern "C" __global__ void ocn_li_thick_v842(
    rk *surfaceThicknessFlux, const rk *landIceFreshwaterFlux, int nCells, rk rhoSw)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    surfaceThicknessFlux[iCell] = A(surfaceThicknessFlux[iCell], D(landIceFreshwaterFlux[iCell], rhoSw));
}

// ocn_surface_land_ice_fluxes_vel: edges 1..nEdges, then cells 1..nCells
extern "C" __global__ void ocn_li_vel_edges_v842(rk *surfaceStress, const rk *topDrag, int nEdges)
{
    const int iEdge = li_thread();
    if (iEdge >= nEdges) return;
    surfaceStress[iEdge] = A(surfaceStress[iEdge], topDrag[iEdge]);
}
extern "C" __global__ void ocn_li_vel_cells_v842(rk *surfaceStressMagnitude, const rk *topDragMagnitude, int nCells)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    surfaceStressMagnitude[iCell] = A(surfaceStressMagnitude[iCell], topDragMagnitude[iCell]);
}

// ocn_surface_land_ice_fluxes_active_tracers: the first tracer's surface flux, cells 1..nCells
extern "C" __global__ void ocn_li_tracers_v842(
    rk *tracersSurfaceFlux, const rk *landIceHeatFlux, int nCells, int nTracers, rk rhoSw, rk cpSw)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    rk *f = tracersSurfaceFlux + (size_t)iCell * nTracers;
    f[0] = A(f[0], D(landIceHeatFlux[iCell], M(rhoSw, cpSw)));
}

// ---------------------------------------------------------------- 'coupled'

// ocn_effective_density_in_land_ice_update: the scratch, cells 1..nCells
extern "C" __global__ void ocn_li_effective_density_scratch_v842(
    rk *effectiveDensityScratch, const rk *landIcePressure, const rk *sshNew, const rk *effectiveDensityCur,
    const int *landIceMask, int nCells)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    if (landIceMask[iCell] == 1)
        effectiveDensityScratch[iCell] = D(-landIcePressure[iCell], M(sshNew[iCell], GRAVITY));
    else
        effectiveDensityScratch[iCell] = effectiveDensityCur[iCell];
}

// the neighbourhood mean, cells 1..nCells
extern "C" __global__ void ocn_li_effective_density_v842(
    rk *effectiveDensityNew, const rk *effectiveDensityScratch, const int *cellsOnCell, const int *nEdgesOnCell,
    const int *cellMask, int nCells, int nl, int maxEdges)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    rk weightSum = C(1.0);
    rk value = effectiveDensityScratch[iCell];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int cell2 = cellsOnCell[(size_t)iCell * maxEdges + i] - 1;
        const rk mask = (rk)cellMask[(size_t)cell2 * nl];
        value = A(value, M(mask, effectiveDensityScratch[cell2]));
        weightSum = A(weightSum, mask);
    }
    effectiveDensityNew[iCell] = D(value, weightSum);
}

// ocn_time_average_coupled_accumulate's land-ice means, cells 1..nCells:
// (avg * n + x) / (n + 1), n the integer count converted to RKIND
extern "C" __global__ void ocn_li_coupled_means_v842(
    rk *avgLandIceBoundaryLayerTracers, rk *avgLandIceTracerTransferVelocities, rk *avgEffectiveDensityInLandIce,
    const rk *landIceBoundaryLayerTracers, const rk *landIceTracerTransferVelocities,
    const rk *effectiveDensityInLandIce, int nCells, int nAccumulated)
{
    const int iCell = li_thread();
    if (iCell >= nCells) return;
    const rk n = (rk)nAccumulated, n1 = (rk)(nAccumulated + 1);
    for (int i = 0; i < 2; ++i) {
        const size_t at = (size_t)iCell * 2 + i;
        avgLandIceBoundaryLayerTracers[at] = D(A(M(avgLandIceBoundaryLayerTracers[at], n),
                                                 landIceBoundaryLayerTracers[at]), n1);
    }
    for (int i = 0; i < 2; ++i) {
        const size_t at = (size_t)iCell * 2 + i;
        avgLandIceTracerTransferVelocities[at] = D(A(M(avgLandIceTracerTransferVelocities[at], n),
                                                     landIceTracerTransferVelocities[at]), n1);
    }
    avgEffectiveDensityInLandIce[iCell] = D(A(M(avgEffectiveDensityInLandIce[iCell], n),
                                              effectiveDensityInLandIce[iCell]), n1);
}
