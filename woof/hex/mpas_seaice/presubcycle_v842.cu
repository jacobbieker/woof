// MPAS-Seaice v8.4.2: the velocity solver's pre-subcycle routines (aggregate
// mass and area, the stress and velocity masks, the new-ice velocities, the
// Hibler ice strength, the air stress and the surface tilt) and seaice_mesh's
// cell-to-vertex interpolation.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_velocity_solver.F:
//     31-36     the module constants (seaiceAreaMinimum, seaiceMassMinimum)
//     610-677   aggregate_mass_and_area
//     691-869   calculation_masks
//     883-981   stress_calculation_mask
//     995-1072  velocity_calculation_mask
//     1086-1249 new_ice_velocities
//     1263-1430 ice_strength (the Hibler branch)
//     1444-1573 air_stress
//     1587-1650 constant_air_stress
//     1819-2135 surface_tilt, surface_tilt_geostrophic,
//               surface_tilt_ssh_gradient, no_surface_tilt
//   src/core_seaice/shared/mpas_seaice_mesh.F:
//     2742-2874 seaice_interpolate_cell_to_vertex (the #if 1 area weighting)
//   with seaice_constants' densities, gravity and Hibler constants
//   (mpas_seaice_constants.F, column/constants/cice/ice_constants_colpkg.F90).
// (Adrian K. Turner, LANL, 2013-2015). Copyright (c) 2013-2019, Los Alamos
// National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction). sum(a(1,:,iCell)) is gfortran's inline sum: an accumulator
// from 0 adding the categories in index order. x**2 is x*x; sqrt is the
// correctly rounded instruction. A leading unary minus negates the whole
// product (-a*b*c is -(a*b*c)); negation is exact and round-to-nearest is
// symmetric, so the words are those of either grouping.
//
// exp is the libm call gfortran makes, taken from the repository's shared
// glibc transcriptions, prepended by presubcycle_v842.module_source:
// expf -> gfk_exp (glibc_flt32.cuh) at RKIND 4, exp -> glibc_exp
// (glibc_flt64.cuh, the MIT Arm optimized-routines transcription) at RKIND 8.
//
// Constants: a literal written _RKIND is decimal rounded once to RKIND (the
// f-suffixed literal at RKIND 4). A seaice_constants value is a colpkg
// dbl_kind parameter converted to RKIND (the double literal cast).
//
// Layout: a Fortran array a(maxEdges, nCells) is a[iCell*maxEdges + i];
// a(1, nCategories, nCells) is a[iCell*nCategories + n].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define EXP(a) glibc_exp(a)
#define SI_AREA_MIN 0.001                  // seaiceAreaMinimum = 0.001_RKIND
#define SI_MASS_MIN 0.01                   // seaiceMassMinimum = 0.01_RKIND
#define SI_AIR_COEFF 0.0012                // airStressCoeff = 0.0012_RKIND
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define EXP(a) gfk_exp(a)
#define SI_AREA_MIN 0.001f
#define SI_MASS_MIN 0.01f
#define SI_AIR_COEFF 0.0012f
#endif

// seaice_constants (dbl_kind colpkg parameters) as RKIND parameters
#define SI_RHOI ((rk)917.0)                // seaiceDensityIce = rhoi
#define SI_RHOS ((rk)330.0)                // seaiceDensitySnow = rhos
#define SI_GRAVITY ((rk)9.80616)           // seaiceGravity = gravit
#define SI_PSTAR ((rk)2.75e4)              // seaiceIceStrengthConstantHiblerP = Pstar
#define SI_CSTAR ((rk)20.0)                // seaiceIceStrengthConstantHiblerC = Cstar

// Whole-array assignments (a = b, a = 0.0_RKIND) over n words.
extern "C" __global__ void seaice_copy_real_v842(rk *dst, const rk *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void seaice_copy_int_v842(int *dst, const int *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void seaice_zero_real_v842(rk *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (rk)0.0;
}

// aggregate_mass_and_area: one thread per cell, over nCells (the routine
// binds its nCellsSolve pointer to the nCells dimension).
extern "C" __global__ void seaice_aggregate_mass_and_area_v842(
    rk *iceAreaCell, rk *iceVolumeCell, rk *snowVolumeCell, rk *totalMassCell,
    const rk *iceAreaCategory, const rk *iceVolumeCategory, const rk *snowVolumeCategory,
    int nCells, int nCategories)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk area = (rk)0.0, ice = (rk)0.0, snow = (rk)0.0;
    for (int n = 0; n < nCategories; ++n) area = A(area, iceAreaCategory[iCell * nCategories + n]);
    for (int n = 0; n < nCategories; ++n) ice = A(ice, iceVolumeCategory[iCell * nCategories + n]);
    for (int n = 0; n < nCategories; ++n) snow = A(snow, snowVolumeCategory[iCell * nCategories + n]);
    iceAreaCell[iCell] = area;
    iceVolumeCell[iCell] = ice;
    snowVolumeCell[iCell] = snow;
    totalMassCell[iCell] = A(M(ice, SI_RHOI), M(snow, SI_RHOS));
}

// seaice_interpolate_cell_to_vertex (the compiled #if 1 branch): one thread
// per vertex, over nVerticesSolve. The area-weighted mean of the
// vertexDegree cells around the vertex; the accumulators start from 0.
extern "C" __global__ void seaice_interpolate_cell_to_vertex_v842(
    rk *variableVertex, const rk *variableCell, const rk *areaCell, const int *cellsOnVertex,
    int nVerticesSolve, int vertexDegree)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve) return;
    rk value = (rk)0.0, totalArea = (rk)0.0;
    for (int k = 0; k < vertexDegree; ++k) {
        const int iCell = cellsOnVertex[iVertex * vertexDegree + k] - 1;
        value = A(value, M(areaCell[iCell], variableCell[iCell]));
        totalArea = A(totalArea, areaCell[iCell]);
    }
    variableVertex[iVertex] = D(value, totalArea);
}

__device__ __forceinline__ bool seaice_has_ice(rk area, rk mass, int landIce)
{
    return area > (rk)SI_AREA_MIN && mass > (rk)SI_MASS_MIN && landIce == 0;
}

// stress_calculation_mask: one thread per cell, over nCells. A cell is
// solved when it or one of its nEdgesOnCell neighbours holds ice.
extern "C" __global__ void seaice_stress_calculation_mask_v842(
    int *solveStress, const rk *iceAreaCellInitial, const rk *totalMassCell, const int *landIceMask,
    const int *nEdgesOnCell, const int *cellsOnCell, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    int solve = 0;
    if (seaice_has_ice(iceAreaCellInitial[iCell], totalMassCell[iCell], landIceMask[iCell])) {
        solve = 1;
    } else {
        for (int k = 0; k < nEdgesOnCell[iCell]; ++k) {
            const int n = cellsOnCell[iCell * maxEdges + k] - 1;
            if (seaice_has_ice(iceAreaCellInitial[n], totalMassCell[n], landIceMask[n])) {
                solve = 1;
                break;
            }
        }
    }
    solveStress[iCell] = solve;
}

// velocity_calculation_mask: one thread per vertex. Solved vertices within
// nVerticesSolve; the halo vertices are 0.
extern "C" __global__ void seaice_velocity_calculation_mask_v842(
    int *solveVelocity, const int *interiorVertex, const int *landIceMaskVertex, const rk *iceAreaVertex,
    const rk *totalMassVertex, int nVerticesSolve, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices) return;
    int solve = 0;
    if (iVertex < nVerticesSolve && interiorVertex[iVertex] == 1 && landIceMaskVertex[iVertex] == 0 &&
        iceAreaVertex[iVertex] > (rk)SI_AREA_MIN && totalMassVertex[iVertex] > (rk)SI_MASS_MIN) {
        solve = 1;
    }
    solveVelocity[iVertex] = solve;
}

// new_ice_velocities, the vertex loop: one thread per vertex, over
// nVerticesSolve. Newly solved vertices start at the ocean velocity;
// unsolved ones are zeroed with their stress divergence and ocean stress.
extern "C" __global__ void seaice_new_ice_velocities_v842(
    rk *uVelocity, rk *vVelocity, rk *stressDivergenceU, rk *stressDivergenceV, rk *oceanStressU,
    rk *oceanStressV, const rk *uOceanVelocityVertex, const rk *vOceanVelocityVertex,
    const int *solveVelocity, const int *solveVelocityPrevious, int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve) return;
    if (solveVelocity[iVertex] == 1) {
        if (solveVelocityPrevious[iVertex] == 0) {
            uVelocity[iVertex] = uOceanVelocityVertex[iVertex];
            vVelocity[iVertex] = vOceanVelocityVertex[iVertex];
        }
    } else {
        uVelocity[iVertex] = (rk)0.0;
        vVelocity[iVertex] = (rk)0.0;
        stressDivergenceU[iVertex] = (rk)0.0;
        stressDivergenceV[iVertex] = (rk)0.0;
        oceanStressU[iVertex] = (rk)0.0;
        oceanStressV[iVertex] = (rk)0.0;
    }
}

// ice_strength, the Hibler branch (no column vertical thermodynamics): one
// thread per cell, over nCellsSolve.
//   icePressure = P* * iceVolumeCell * exp(-C* * (1 - iceAreaCell))
extern "C" __global__ void seaice_ice_strength_hibler_v842(
    rk *icePressure, const rk *iceVolumeCell, const rk *iceAreaCell, const int *solveStress, int nCellsSolve)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    if (solveStress[iCell] == 1) {
        const rk arg = -M(SI_CSTAR, S((rk)1.0, iceAreaCell[iCell]));
        icePressure[iCell] = M(M(SI_PSTAR, iceVolumeCell[iCell]), EXP(arg));
    } else {
        icePressure[iCell] = (rk)0.0;
    }
}

// constant_air_stress: one thread per cell, over nCellsSolve.
//   windSpeed = sqrt(u**2 + v**2)
//   airStressCellU = airDensity * windSpeed * airStressCoeff * u * iceAreaCell
extern "C" __global__ void seaice_constant_air_stress_v842(
    rk *airStressCellU, rk *airStressCellV, const rk *uAirVelocity, const rk *vAirVelocity,
    const rk *airDensity, const rk *iceAreaCell, int nCellsSolve)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const rk u = uAirVelocity[iCell], v = vAirVelocity[iCell];
    const rk windSpeed = SQRT(A(M(u, u), M(v, v)));
    const rk scale = M(M(airDensity[iCell], windSpeed), (rk)SI_AIR_COEFF);
    airStressCellU[iCell] = M(M(scale, u), iceAreaCell[iCell]);
    airStressCellV[iCell] = M(M(scale, v), iceAreaCell[iCell]);
}

// surface_tilt_geostrophic: one thread per vertex, over nVerticesSolve.
extern "C" __global__ void seaice_surface_tilt_geostrophic_v842(
    rk *surfaceTiltForceU, rk *surfaceTiltForceV, const rk *fVertex, const rk *totalMassVertex,
    const rk *uOceanVelocityVertex, const rk *vOceanVelocityVertex, const int *solveVelocity,
    int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve) return;
    if (solveVelocity[iVertex] == 1) {
        const rk fm = M(fVertex[iVertex], totalMassVertex[iVertex]);
        surfaceTiltForceU[iVertex] = -M(fm, vOceanVelocityVertex[iVertex]);
        surfaceTiltForceV[iVertex] = M(fm, uOceanVelocityVertex[iVertex]);
    } else {
        surfaceTiltForceU[iVertex] = (rk)0.0;
        surfaceTiltForceV[iVertex] = (rk)0.0;
    }
}

// surface_tilt_ssh_gradient, after the tilt is interpolated to the vertices:
// one thread per vertex, over nVerticesSolve.
extern "C" __global__ void seaice_surface_tilt_ssh_gradient_v842(
    rk *surfaceTiltForceU, rk *surfaceTiltForceV, const rk *totalMassVertex,
    const rk *seaSurfaceTiltVertexU, const rk *seaSurfaceTiltVertexV, const int *solveVelocity,
    int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve) return;
    if (solveVelocity[iVertex] == 1) {
        const rk gm = M(SI_GRAVITY, totalMassVertex[iVertex]);
        surfaceTiltForceU[iVertex] = -M(gm, seaSurfaceTiltVertexU[iVertex]);
        surfaceTiltForceV[iVertex] = -M(gm, seaSurfaceTiltVertexV[iVertex]);
    } else {
        surfaceTiltForceU[iVertex] = (rk)0.0;
        surfaceTiltForceV[iVertex] = (rk)0.0;
    }
}
