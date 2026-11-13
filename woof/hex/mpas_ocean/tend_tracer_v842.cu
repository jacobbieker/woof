// MPAS-Ocean v8.4.2: the tracer-tendency terms ocn_tend_tracer adds beyond
// advection and horizontal mixing.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_tendency.F (ocn_tend_tracer: the thickness flux)
//   src/core_ocean/shared/mpas_ocn_surface_bulk_forcing.F
//     (ocn_surface_bulk_forcing_active_tracers)
//   src/core_ocean/shared/mpas_ocn_equation_of_state.F (ocn_freezing_temperature)
//   src/core_ocean/shared/mpas_ocn_tracer_surface_flux_to_tend.F (ocn_tracer_surface_flux_tend)
//   src/core_ocean/shared/mpas_ocn_tracer_short_wave_absorption_jerlov.F
//     (ocn_tracer_short_wave_absorption_jerlov_tend, ocn_get_jerlov_fraction)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off). Jerlov's exp is glibc's
// expf (gfk_exp, glibc_flt32.cuh) at RKIND 4 and exp (glibc_exp,
// glibc_flt64.cuh) at RKIND 8, prepended by tend_tracer_v842.module_source.
// Fortran MIN/MAX are chosen on the words as integer selects.
//
// Layout: tracers(nTracers, nVertLevels, nCells+1) is
// t[(iCell*nl + k)*nTracers + i]; surface fluxes(nTracers, nCells+1) are
// f[iCell*nTracers + i]; a(nVertLevels, n+1) is a[i*nl + k]. Tracer indices
// are 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define EXP(a) glibc_exp(a)
#define C(x) x
__device__ __forceinline__ rk tt_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk tt_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define EXP(a) gfk_exp(a)
#define C(x) x##f
__device__ __forceinline__ rk tt_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk tt_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif

#define LATENT_HEAT_FUSION_MKS C(3.337e5)
__device__ __forceinline__ int tt_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// normalThicknessFlux = normalTransportVelocity * layerThicknessEdge, edges 1..nEdges, every level
extern "C" __global__ void ocn_tt_thickness_flux_v842(
    rk *normalThicknessFlux, const rk *normalTransportVelocity, const rk *layerThicknessEdge, int n)
{
    const int t = tt_thread();
    if (t >= n) return;
    normalThicknessFlux[t] = M(normalTransportVelocity[t], layerThicknessEdge[t]);
}

// ocn_surface_bulk_forcing_active_tracers, cells 1..nCellsArray(3)
extern "C" __global__ void ocn_tt_bulk_forcing_active_tracers_v842(
    rk *surfaceFlux, rk *surfaceFluxRunoff, rk *surfaceFluxRemoved, rk *penetrativeTemperatureFlux,
    const rk *tracers, const rk *layerThickness, const rk *latentHeatFlux, const rk *sensibleHeatFlux,
    const rk *longWaveHeatFluxUp, const rk *longWaveHeatFluxDown, const rk *seaIceHeatFlux, const rk *rainFlux,
    const rk *snowFlux, const rk *shortWaveHeatFlux, const rk *evaporationFlux, const rk *seaIceFreshWaterFlux,
    const rk *seaIceSalinityFlux, const rk *iceRunoffFlux, const rk *riverRunoffFlux, int nCells, int nl,
    int nTracers, int indexT, int indexS, int thicknessFluxOn, rk dt, rk hfluxFactor, rk sfluxFactor, rk rhoSw,
    rk freezeCoeff0, rk freezeCoeffS, rk freezeCoeffP, rk freezeCoeffPS, rk freezeReferencePressure)
{
    const int iCell = tt_thread();
    if (iCell >= nCells) return;
    const int it = iCell * nTracers + indexT - 1, is = iCell * nTracers + indexS - 1;
    const rk T1 = tracers[(iCell * nl) * nTracers + indexT - 1];
    const rk S1 = tracers[(iCell * nl) * nTracers + indexS - 1];
    surfaceFlux[it] = A(surfaceFlux[it],
        M(S(A(A(A(A(latentHeatFlux[iCell], sensibleHeatFlux[iCell]), longWaveHeatFluxUp[iCell]),
                  longWaveHeatFluxDown[iCell]), seaIceHeatFlux[iCell]),
            M(A(snowFlux[iCell], iceRunoffFlux[iCell]), LATENT_HEAT_FUSION_MKS)), hfluxFactor));
    const rk requiredSalt = -D(M(M(seaIceSalinityFlux[iCell], sfluxFactor), dt), layerThickness[iCell * nl]);
    const rk allowedSalt = tt_min(C(4.0), S1);
    if (allowedSalt < requiredSalt) {
        surfaceFluxRemoved[is] = A(surfaceFluxRemoved[is],
            M(M(S(C(1.0), D(allowedSalt, requiredSalt)), seaIceSalinityFlux[iCell]), sfluxFactor));
        surfaceFlux[is] = A(surfaceFlux[is], M(M(D(allowedSalt, requiredSalt), seaIceSalinityFlux[iCell]), sfluxFactor));
    } else {
        surfaceFlux[is] = A(surfaceFlux[is], M(seaIceSalinityFlux[iCell], sfluxFactor));
    }
    if (thicknessFluxOn) {
        surfaceFlux[it] = A(surfaceFlux[it], D(M(A(rainFlux[iCell], evaporationFlux[iCell]), T1), rhoSw));
        surfaceFluxRunoff[it] = D(M(riverRunoffFlux[iCell], tt_max(T1, C(0.0))), rhoSw);
        // ocn_freezing_temperature(salinity, pressure = 0, open ocean)
        const rk pressureOffset = tt_max(S(C(0.0), freezeReferencePressure), C(0.0));
        const rk freezing = A(A(A(freezeCoeff0, M(freezeCoeffS, S1)), M(freezeCoeffP, pressureOffset)),
                              M(M(freezeCoeffPS, pressureOffset), S1));
        surfaceFlux[it] = A(surfaceFlux[it], D(M(seaIceFreshWaterFlux[iCell], freezing), rhoSw));
    }
    penetrativeTemperatureFlux[iCell] = M(shortWaveHeatFlux[iCell], hfluxFactor);
}

// ocn_tracer_surface_flux_tend's sum for one flux, cells 1..nCellsArray(1):
// each level its fraction, what remains to the bottom level
extern "C" __global__ void ocn_tt_surface_flux_v842(
    rk *tend, const rk *surfaceFlux, const rk *fractionAbsorbed, const int *maxLevelCell, int nCells, int nl,
    int nTracers)
{
    const int iCell = tt_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell];
    rk remaining = C(1.0);
    for (int k = 0; k < top; ++k) {
        const rk fraction = fractionAbsorbed[iCell * nl + k];
        remaining = S(remaining, fraction);
        for (int i = 0; i < nTracers; ++i) {
            const int at = (iCell * nl + k) * nTracers + i;
            tend[at] = A(tend[at], M(surfaceFlux[iCell * nTracers + i], fraction));
        }
    }
    if (top > 0 && remaining > C(0.0)) {
        for (int i = 0; i < nTracers; ++i) {
            const int at = (iCell * nl + top - 1) * nTracers + i;
            tend[at] = A(tend[at], M(surfaceFlux[iCell * nTracers + i], remaining));
        }
    }
}

// ocn_get_jerlov_fraction
__device__ __forceinline__ rk jerlov_fraction(rk depth, rk rfac, rk depth1, rk depth2) {
    if (-depth < C(-200.0)) return C(0.0);
    return A(M(rfac, EXP(-D(depth, depth1))), M(S(C(1.0), rfac), EXP(-D(depth, depth2))));
}

// ocn_tracer_short_wave_absorption_jerlov_tend, cells 1..nCellsArray(3): one
// thread per column (the weights run down it)
extern "C" __global__ void ocn_tt_jerlov_v842(
    rk *tend, rk *penetrativeTemperatureFluxOBL, rk *weights, const rk *penetrativeTemperatureFlux,
    const rk *layerThickness, const int *maxLevelCell, int nCells, int nl, int nTracers, int indexT, rk rfac,
    rk depth1, rk depth2, rk buoyancyDepth)
{
    const int iCell = tt_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell];
    rk *w = weights + iCell * (nl + 1);
    w[0] = C(1.0);
    rk depth = C(0.0);
    for (int k = 0; k < top; ++k) {
        depth = A(depth, layerThickness[iCell * nl + k]);
        w[k + 1] = jerlov_fraction(depth, rfac, depth1, depth2);
        const int at = (iCell * nl + k) * nTracers + indexT - 1;
        tend[at] = A(tend[at], M(penetrativeTemperatureFlux[iCell], S(w[k], w[k + 1])));
    }
    depth = C(0.0);
    int kf = 1;   // the Fortran's k after the loop: the exit level, or maxLevelCell+1
    for (kf = 1; kf <= top; ++kf) {
        depth = A(depth, layerThickness[iCell * nl + kf - 1]);
        if (depth > buoyancyDepth) break;
    }
    const int depLev = (kf == top || kf == 1) ? 2 : kf;
    penetrativeTemperatureFluxOBL[iCell] = M(penetrativeTemperatureFlux[iCell], w[depLev - 1]);
}
