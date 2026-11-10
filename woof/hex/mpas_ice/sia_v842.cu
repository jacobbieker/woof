// MPAS land-ice (MALI) v8.4.2: the shallow-ice approximation velocity solver.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_sia.F:228-442           (li_sia_solve)
//   src/core_landice/mode_forward/mpas_li_diagnostic_vars.F:381-509 (li_calculate_flowParamA)
//   src/core_landice/shared/mpas_li_setup.F:484-536                 (li_cells_to_vertices_1dfield_using_kiteAreas)
//   src/core_landice/shared/mpas_li_mask.F:738-757                  (li_mask_is_dynamic_ice, _int)
//   src/core_landice/shared/mpas_li_constants.F:49,53               (gravity, idealGasConstant)
//   src/operators/mpas_geometry_utils.F:775-823                     (mpas_cells_to_points_using_baryweights)
//   src/operators/mpas_vector_operations.F:381-438                  (mpas_tangential_vector_1d)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 (MPAS's default single precision) unless MPAS_RKIND8 is defined.
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off). The real powers
// x**(n-1), x**(n+1), (rhoi*g)**n become powf (pow at RKIND 8), and exp
// becomes expf (exp), in the reference object. Here they are gfk_pow and
// gfk_exp from the shared glibc float32 transcription
// (woof/core/kernels/glibc_flt32.cuh, prepended through
// woof.hex.mpasphys.kessler_v842.shared_math_header), or glibc_pow and
// glibc_exp from the binary64 one (glibc_flt64.cuh), both Arm
// optimized-routines under MIT.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// cellsOnEdge(2, nEdges) is c[iEdge*2 + i]; baryCellsOnVertex(3, nVertices) is
// b[iVertex*3 + i]; edgesOnEdge(maxEdges2, nEdges) is e[iEdge*maxEdges2 + j].
// Connectivity is MPAS's 1-based indices.

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

#define LI_GRAVITY LI_C(9.80616)              // li_constants gravity
#define LI_IDEAL_GAS_CONSTANT LI_C(8.314)     // li_constants idealGasConstant
#define LI_MASK_VALUE_DYNAMIC_ICE 2      // li_mask_ValueDynamicIce

// li_mask_is_dynamic_ice_int: iand(mask, 2) / 2
__device__ __forceinline__ int li_dynamic_int(int mask) {
    return (mask & LI_MASK_VALUE_DYNAMIC_ICE) / LI_MASK_VALUE_DYNAMIC_ICE;
}
// li_mask_is_dynamic_ice (logical): iand(mask, 2) == 2
__device__ __forceinline__ bool li_dynamic(int mask) {
    return (mask & LI_MASK_VALUE_DYNAMIC_ICE) == LI_MASK_VALUE_DYNAMIC_ICE;
}

// li_calculate_flowParamA, one thread per cell, followed by the closing
// flowParamA = flowParamA * config_enhancementFactor.
// option: 1 'constant', 2 'PB1982', 3 'CP2010'.
extern "C" __global__ void li_calculate_flowParamA_v842(
    rk *flowParamA, const rk *temperature, const rk *thickness,
    const rk *layerCenterSigma, int option, rk defaultFlowParamA,
    rk enhancementFactor, rk iceDensity, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nVertLevels;
    if (option == 1) {
        for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = defaultFlowParamA;
    } else if (thickness[iCell] > LI_C(0.0)) {
        const rk pressureMeltPointSlope = option == 2 ? LI_C(9.7456e-8) : LI_C(7.0e-8);
        for (int k = 0; k < nVertLevels; ++k) {
            const rk pressureTerm = M(M(M(M(pressureMeltPointSlope, thickness[iCell]), iceDensity),
                                           LI_GRAVITY), layerCenterSigma[k]);
            rk temperatureCorrected = A(temperature[base + k], pressureTerm);
            temperatureCorrected = LI_C(273.15) < temperatureCorrected ? LI_C(273.15) : temperatureCorrected;
            temperatureCorrected = LI_C(223.15) > temperatureCorrected ? LI_C(223.15) : temperatureCorrected;
            if (option == 2) {
                rk A0, Q;
                if (temperatureCorrected > LI_C(263.15)) { A0 = LI_C(1.733e3); Q = LI_C(139.0e3); }
                else { A0 = LI_C(3.613e-13); Q = LI_C(60.0e3); }
                flowParamA[base + k] = M(A0, LI_EXP(D(M(-LI_C(1.0), Q),
                                                       M(LI_IDEAL_GAS_CONSTANT, temperatureCorrected))));
            } else {
                const rk transitionTemperatureCorrected = A(LI_C(263.15), pressureTerm);
                const rk A0 = LI_C(3.5e-25);
                const rk Q = temperatureCorrected > LI_C(263.15) ? LI_C(115.0e3) : LI_C(6.0e4);
                flowParamA[base + k] = M(A0, LI_EXP(M(D(M(-LI_C(1.0), Q), LI_IDEAL_GAS_CONSTANT),
                    S(D(LI_C(1.0), temperatureCorrected), D(LI_C(1.0), transitionTemperatureCorrected)))));
            }
        }
    } else {
        for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = LI_C(0.0);  // non-ice cells get 0
    }
    for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = M(flowParamA[base + k], enhancementFactor);
}

// mpas_cells_to_points_using_baryweights (option 1) or
// li_cells_to_vertices_1dfield_using_kiteAreas (option 2), one thread per vertex
extern "C" __global__ void li_upper_surface_vertex_v842(
    rk *upperSurfaceVertex, const rk *upperSurface, const int *baryCellsOnVertex,
    const rk *baryWeightsOnVertex, const int *cellsOnVertex, const rk *kiteAreasOnVertex,
    int option, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices) return;
    const int b = iVertex * 3;
    if (option == 1) {
        upperSurfaceVertex[iVertex] = A(A(M(baryWeightsOnVertex[b], upperSurface[baryCellsOnVertex[b] - 1]),
                                          M(baryWeightsOnVertex[b + 1], upperSurface[baryCellsOnVertex[b + 1] - 1])),
                                        M(baryWeightsOnVertex[b + 2], upperSurface[baryCellsOnVertex[b + 2] - 1]));
    } else {
        rk fVertexAccum = LI_C(0.0), weightAccum = LI_C(0.0);
        for (int iCell = 0; iCell < 3; ++iCell) {
            const int cellIndex = cellsOnVertex[b + iCell] - 1;
            rk baryweight = LI_C(0.0);
            for (int iCell2 = 0; iCell2 < 3; ++iCell2)
                if (iCell2 != iCell) baryweight = A(baryweight, M(LI_C(0.5), kiteAreasOnVertex[b + iCell2]));
            fVertexAccum = A(fVertexAccum, M(baryweight, upperSurface[cellIndex]));
            weightAccum = A(weightAccum, kiteAreasOnVertex[b + iCell]);
        }
        upperSurfaceVertex[iVertex] = D(fVertexAccum, weightAccum);
    }
}

// tangentSlopeEdge for the configured method, then slopeEdge, one thread per
// edge over all nEdges. option: 1 or 2 from the vertex surface, 3
// 'from_normal_slope' (mpas_tangential_vector_1d with includeHalo).
extern "C" __global__ void li_slope_edge_v842(
    rk *tangentSlopeEdge, rk *slopeEdge, const rk *normalSlopeEdge,
    const rk *upperSurfaceVertex, const int *verticesOnEdge, const rk *dvEdge,
    const int *edgeMask, const int *nEdgesOnEdge, const int *edgesOnEdge,
    const rk *weightsOnEdge, int option, int nEdges, int maxEdges2)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    rk t;
    if (option == 3) {
        t = LI_C(0.0);
        const int n = nEdgesOnEdge[iEdge];
        for (int i = 0; i < n; ++i) {
            const int eoe = edgesOnEdge[iEdge * maxEdges2 + i] - 1;
            t = A(t, M(weightsOnEdge[iEdge * maxEdges2 + i], normalSlopeEdge[eoe]));
        }
    } else if (li_dynamic(edgeMask[iEdge])) {
        t = D(S(upperSurfaceVertex[verticesOnEdge[iEdge * 2] - 1],
                upperSurfaceVertex[verticesOnEdge[iEdge * 2 + 1] - 1]), dvEdge[iEdge]);
    } else {
        t = LI_C(0.0);
    }
    tangentSlopeEdge[iEdge] = t;
    const rk ns = normalSlopeEdge[iEdge];
    slopeEdge[iEdge] = LI_SQRT(A(M(ns, ns), M(t, t)));
}

// The SIA velocity profile, li_sia_solve's edge loop over nEdgesSolve, one
// thread per edge. normalVelocity is (nVertInterfaces, nEdges).
extern "C" __global__ void li_sia_velocity_v842(
    rk *normalVelocity, const rk *flowParamA, const rk *thickness,
    const rk *slopeEdge, const rk *normalSlopeEdge, const rk *layerInterfaceSigma,
    const int *cellsOnEdge, const int *edgeMask, const int *cellMask,
    rk iceDensity, rk n, int nEdgesSolve, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdgesSolve) return;
    const int nVertInterfaces = nVertLevels + 1;
    rk *velocity = normalVelocity + iEdge * nVertInterfaces;
    if (!li_dynamic(edgeMask[iEdge])) {
        for (int k = 0; k < nVertInterfaces; ++k) velocity[k] = LI_C(0.0);  // zero velocity on non-dynamic edges
        return;
    }
    const rk positionIndependentFactor = -M(LI_C(0.5), LI_POW(M(iceDensity, LI_GRAVITY), n));
    const rk nMinus1 = S(n, LI_C(1.0)), nPlus1 = A(n, LI_C(1.0));
    const int cell1 = cellsOnEdge[iEdge * 2] - 1;
    const int cell2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const int cell1IsDynamic = li_dynamic_int(cellMask[cell1]);
    const int cell2IsDynamic = li_dynamic_int(cellMask[cell2]);
    const rk thicknessEdge = M(A(thickness[cell1], thickness[cell2]), LI_C(0.5));
    const rk levelIndependentFactor = M(M(LI_POW(slopeEdge[iEdge], nMinus1), normalSlopeEdge[iEdge]),
                                           LI_POW(thicknessEdge, nPlus1));
    velocity[nVertInterfaces - 1] = LI_C(0.0);  // Assume no sliding
    for (int iLevel = nVertInterfaces - 2; iLevel >= 0; --iLevel) {
        const rk flwaLevelEdge = D(A(M(flowParamA[cell1 * nVertLevels + iLevel], (rk)cell1IsDynamic),
                                        M(flowParamA[cell2 * nVertLevels + iLevel], (rk)cell2IsDynamic)),
                                      (rk)(cell1IsDynamic + cell2IsDynamic));
        velocity[iLevel] = A(velocity[iLevel + 1],
            M(M(M(positionIndependentFactor, levelIndependentFactor), flwaLevelEdge),
              S(LI_POW(layerInterfaceSigma[iLevel], nPlus1), LI_POW(layerInterfaceSigma[iLevel + 1], nPlus1))));
    }
}
