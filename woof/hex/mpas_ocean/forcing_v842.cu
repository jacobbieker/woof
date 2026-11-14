// MPAS-Ocean v8.4.2: the momentum forcing, bulk forcing and thickness surface
// flux terms that ocn_tend_vel and ocn_tend_thick add.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_surface_bulk_forcing.F:139-306,320-334
//     (ocn_surface_bulk_forcing_vel, _thick, _init)
//   src/core_ocean/shared/mpas_ocn_forcing.F:191-196 (ocn_forcing_transmission)
//   src/core_ocean/shared/mpas_ocn_vel_forcing_surface_stress.F (ocn_vel_forcing_surface_stress_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_forcing_explicit_bottom_drag.F (ocn_vel_forcing_explicit_bottom_drag_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_forcing_rayleigh.F (ocn_vel_forcing_rayleigh_tend, _init)
//   src/core_ocean/shared/mpas_ocn_thick_surface_flux.F (ocn_thick_surface_flux_tend, _init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions, as tend_v842.cu. Explicit round-to-nearest
// operations in the Fortran's order (gfortran -O0 -ffp-contract=off). The
// libm calls gfortran makes are the repository's shared glibc
// transcriptions, prepended by forcing_v842.module_source:
//   RKIND 4: expf -> gfk_exp (glibc_flt32.cuh), cosf/sinf -> glibc_cosf/
//            glibc_sinf (glibc_trig_flt32.cuh);
//   RKIND 8: exp -> glibc_exp (glibc_flt64.cuh). There is no permissive
//            double cos/sin equal to glibc's: the bulk wind stress's kernel
//            is built at RKIND 8 only with the opt-in LGPL transcription of
//            glibc 2.39's (WOOF_GLIBC239_TRIG64, g239::g239_cos/_sin), and
//            forcing_v842.py refuses it otherwise.
// sqrt is the correctly rounded instruction gfortran emits; x**2 is x*x.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define EXP(a) glibc_exp(a)
#ifdef WOOF_GLIBC239_TRIG64
#define BW_COS(a) g239::g239_cos(a)
#define BW_SIN(a) g239::g239_sin(a)
#endif
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define EXP(a) gfk_exp(a)
#define BW_COS(a) glibc_cosf(a)
#define BW_SIN(a) glibc_sinf(a)
#endif

// ocn_forcing_transmission: exp(max(z/attenuationCoefficient, -100)).
// max keeps -100 unless z/att is larger (maxss/maxsd).
__device__ __forceinline__ rk ocn_transmission(rk z, rk att)
{
    const rk x = D(z, att);
    return EXP(x > (rk)-100.0 ? x : (rk)-100.0);
}

#if !defined(MPAS_RKIND8) || defined(WOOF_GLIBC239_TRIG64)
// ocn_surface_bulk_forcing_vel, edge loop (nEdgesArray(4)): add the rotated
// cell-mean wind stress to surfaceStress.
extern "C" __global__ void ocn_bulk_wind_stress_edges_v842(
    rk *surfaceStress, const rk *windStressZonal, const rk *windStressMeridional, const rk *angleEdge,
    const int *cellsOnEdge, int nEdges)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int c1 = cellsOnEdge[iEdge * 2] - 1, c2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const rk zonalAverage = M((rk)0.5, A(windStressZonal[c1], windStressZonal[c2]));
    const rk meridionalAverage = M((rk)0.5, A(windStressMeridional[c1], windStressMeridional[c2]));
    surfaceStress[iEdge] = A(A(surfaceStress[iEdge], M(BW_COS(angleEdge[iEdge]), zonalAverage)),
                             M(BW_SIN(angleEdge[iEdge]), meridionalAverage));
}

// ocn_surface_bulk_forcing_vel, cell loop (nCellsArray(3)): the stress magnitude
extern "C" __global__ void ocn_bulk_wind_stress_cells_v842(
    rk *surfaceStressMagnitude, const rk *windStressZonal, const rk *windStressMeridional, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk tz = windStressZonal[iCell], tm = windStressMeridional[iCell];
    surfaceStressMagnitude[iCell] = A(surfaceStressMagnitude[iCell], SQRT(A(M(tz, tz), M(tm, tm))));
}
#endif

// ocn_surface_bulk_forcing_thick (nCellsArray(3))
extern "C" __global__ void ocn_bulk_thickness_flux_v842(
    rk *surfaceThicknessFlux, rk *surfaceThicknessFluxRunoff, const rk *snowFlux, const rk *rainFlux,
    const rk *evaporationFlux, const rk *seaIceFreshWaterFlux, const rk *iceRunoffFlux, const rk *riverRunoffFlux,
    rk rhoSw, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    surfaceThicknessFlux[iCell] = A(surfaceThicknessFlux[iCell],
        D(A(A(A(A(snowFlux[iCell], rainFlux[iCell]), evaporationFlux[iCell]), seaIceFreshWaterFlux[iCell]),
            iceRunoffFlux[iCell]), rhoSw));
    surfaceThicknessFluxRunoff[iCell] = D(riverRunoffFlux[iCell], rhoSw);
}

// ocn_vel_forcing_surface_stress_tend: one thread per edge (nEdgesArray(1))
extern "C" __global__ void ocn_vel_forcing_surface_stress_v842(
    rk *tend, const rk *surfaceFluxAttenuationCoefficient, const rk *surfaceStress, const rk *layerThicknessEdge,
    const int *maxLevelEdgeTop, const int *cellsOnEdge, const int *edgeMask, rk rhoSw, int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int c1 = cellsOnEdge[iEdge * 2] - 1, c2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const int base = iEdge * nVertLevels;
    const int kmax = maxLevelEdgeTop[iEdge];
    const rk ss = surfaceStress[iEdge];
    const rk att = M((rk)0.5, A(surfaceFluxAttenuationCoefficient[c1], surfaceFluxAttenuationCoefficient[c2]));
    rk zTop = (rk)0.0;
    rk tTop = ocn_transmission(zTop, att);
    rk remainingStress = (rk)1.0;
    for (int k = 0; k < kmax; ++k) {
        const rk h = layerThicknessEdge[base + k];
        const rk zBot = S(zTop, h);
        const rk tBot = ocn_transmission(zBot, att);
        remainingStress = S(remainingStress, S(tTop, tBot));
        tend[base + k] = A(tend[base + k], D(D(M(M((rk)edgeMask[base + k], ss), S(tTop, tBot)), rhoSw), h));
        zTop = zBot;
        tTop = tBot;
    }
    if (kmax > 0 && remainingStress > (rk)0.0) {
        const int p = base + kmax - 1;
        tend[p] = A(tend[p], D(D(M(M((rk)edgeMask[p], ss), remainingStress), rhoSw), layerThicknessEdge[p]));
    }
}

// ocn_vel_forcing_explicit_bottom_drag_tend: one thread per edge, at level
// maxLevelEdgeTop (which the caller has checked is at least 1)
extern "C" __global__ void ocn_vel_forcing_explicit_bottom_drag_v842(
    rk *tend, const rk *normalVelocity, const rk *kineticEnergyCell, const rk *layerThicknessEdge,
    const int *maxLevelEdgeTop, const int *cellsOnEdge, rk coef, int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int c1 = cellsOnEdge[iEdge * 2] - 1, c2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const int k = maxLevelEdgeTop[iEdge] - 1;
    const int p = iEdge * nVertLevels + k;
    const rk drag = SQRT(A(kineticEnergyCell[c1 * nVertLevels + k], kineticEnergyCell[c2 * nVertLevels + k]));
    tend[p] = S(tend[p], D(M(M(coef, drag), normalVelocity[p]), layerThicknessEdge[p]));
}

// ocn_vel_forcing_rayleigh_tend: one thread per edge. The bottom term, as the
// Fortran writes it, uses rayleighDampingCoef (not the bottom coefficient) and
// runs only when the interior friction is on; bottom != 0 asks for it.
extern "C" __global__ void ocn_vel_forcing_rayleigh_v842(
    rk *tend, const rk *normalVelocity, const int *maxLevelEdgeTop, rk coef, int bottom,
    int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int base = iEdge * nVertLevels;
    const int kmax = maxLevelEdgeTop[iEdge];
    for (int k = 0; k < kmax; ++k)
        tend[base + k] = S(tend[base + k], M(coef, normalVelocity[base + k]));
    if (bottom) {
        const int p = base + kmax - 1;
        tend[p] = S(tend[p], M(coef, normalVelocity[p]));
    }
}

// ocn_thick_surface_flux_tend: one thread per cell (nCellsArray(1))
extern "C" __global__ void ocn_thick_surface_flux_v842(
    rk *tend, const rk *transmissionCoefficients, const rk *transmissionCoefficientsRunoff,
    const rk *surfaceThicknessFlux, const rk *surfaceThicknessFluxRunoff, const int *maxLevelCell,
    int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nVertLevels;
    const int kmax = maxLevelCell[iCell];
    const rk f = surfaceThicknessFlux[iCell], fr = surfaceThicknessFluxRunoff[iCell];
    rk remainingFlux = (rk)1.0, remainingFluxRunoff = (rk)1.0;
    for (int k = 0; k < kmax; ++k) {
        const rk tc = transmissionCoefficients[base + k], tcr = transmissionCoefficientsRunoff[base + k];
        remainingFlux = S(remainingFlux, tc);
        remainingFluxRunoff = S(remainingFluxRunoff, tcr);
        tend[base + k] = A(A(tend[base + k], M(f, tc)), M(fr, tcr));
    }
    if (kmax > 0 && remainingFlux > (rk)0.0)
        tend[base + kmax - 1] = A(tend[base + kmax - 1], M(remainingFlux, f));
    if (kmax > 0 && remainingFluxRunoff > (rk)0.0)
        tend[base + kmax - 1] = A(tend[base + kmax - 1], M(remainingFluxRunoff, fr));
}

// ocn_forcing_build_fraction_absorbed_array (mpas_ocn_forcing.F), cells
// 1..nCellsArray(2): each layer's share of the surface flux, the transmission
// at its top less that at its bottom; for the runoff with its own attenuation
extern "C" __global__ void ocn_fraction_absorbed_v842(
    rk *fraction, const rk *layerThickness, const rk *attenuation, const int *maxLevelCell, int nCells, int nl)
{
    const int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk zTop = (rk)0.0;
    rk tTop = ocn_transmission(zTop, attenuation[iCell]);
    for (int k = 0; k < maxLevelCell[iCell]; ++k) {
        const rk zBot = S(zTop, layerThickness[iCell * nl + k]);
        const rk tBot = ocn_transmission(zBot, attenuation[iCell]);
        fraction[iCell * nl + k] = S(tTop, tBot);
        zTop = zBot;
        tTop = tBot;
    }
}
