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
// RKIND = 4 (MPAS's default single precision). Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). The real powers x**(n-1), x**(n+1), (rhoi*g)**n become
// powf, and exp becomes expf, in the reference object. Here they are
// gfk_pow and gfk_exp from the shared glibc float32 transcription (Arm
// optimized-routines, MIT, woof/core/kernels/glibc_flt32.cuh), which the
// caller prepends (woof.hex.mpasphys.kessler_v842.shared_math_header).
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// cellsOnEdge(2, nEdges) is c[iEdge*2 + i]; baryCellsOnVertex(3, nVertices) is
// b[iVertex*3 + i]; edgesOnEdge(maxEdges2, nEdges) is e[iEdge*maxEdges2 + j].
// Connectivity is MPAS's 1-based indices.

#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

#define LI_GRAVITY 9.80616f              // li_constants gravity
#define LI_IDEAL_GAS_CONSTANT 8.314f     // li_constants idealGasConstant
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
    float *flowParamA, const float *temperature, const float *thickness,
    const float *layerCenterSigma, int option, float defaultFlowParamA,
    float enhancementFactor, float iceDensity, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * nVertLevels;
    if (option == 1) {
        for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = defaultFlowParamA;
    } else if (thickness[iCell] > 0.0f) {
        const float pressureMeltPointSlope = option == 2 ? 9.7456e-8f : 7.0e-8f;
        for (int k = 0; k < nVertLevels; ++k) {
            const float pressureTerm = M(M(M(M(pressureMeltPointSlope, thickness[iCell]), iceDensity),
                                           LI_GRAVITY), layerCenterSigma[k]);
            float temperatureCorrected = A(temperature[base + k], pressureTerm);
            temperatureCorrected = 273.15f < temperatureCorrected ? 273.15f : temperatureCorrected;
            temperatureCorrected = 223.15f > temperatureCorrected ? 223.15f : temperatureCorrected;
            if (option == 2) {
                float A0, Q;
                if (temperatureCorrected > 263.15f) { A0 = 1.733e3f; Q = 139.0e3f; }
                else { A0 = 3.613e-13f; Q = 60.0e3f; }
                flowParamA[base + k] = M(A0, gfk_exp(D(M(-1.0f, Q),
                                                       M(LI_IDEAL_GAS_CONSTANT, temperatureCorrected))));
            } else {
                const float transitionTemperatureCorrected = A(263.15f, pressureTerm);
                const float A0 = 3.5e-25f;
                const float Q = temperatureCorrected > 263.15f ? 115.0e3f : 6.0e4f;
                flowParamA[base + k] = M(A0, gfk_exp(M(D(M(-1.0f, Q), LI_IDEAL_GAS_CONSTANT),
                    S(D(1.0f, temperatureCorrected), D(1.0f, transitionTemperatureCorrected)))));
            }
        }
    } else {
        for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = 0.0f;  // non-ice cells get 0
    }
    for (int k = 0; k < nVertLevels; ++k) flowParamA[base + k] = M(flowParamA[base + k], enhancementFactor);
}

// mpas_cells_to_points_using_baryweights (option 1) or
// li_cells_to_vertices_1dfield_using_kiteAreas (option 2), one thread per vertex
extern "C" __global__ void li_upper_surface_vertex_v842(
    float *upperSurfaceVertex, const float *upperSurface, const int *baryCellsOnVertex,
    const float *baryWeightsOnVertex, const int *cellsOnVertex, const float *kiteAreasOnVertex,
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
        float fVertexAccum = 0.0f, weightAccum = 0.0f;
        for (int iCell = 0; iCell < 3; ++iCell) {
            const int cellIndex = cellsOnVertex[b + iCell] - 1;
            float baryweight = 0.0f;
            for (int iCell2 = 0; iCell2 < 3; ++iCell2)
                if (iCell2 != iCell) baryweight = A(baryweight, M(0.5f, kiteAreasOnVertex[b + iCell2]));
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
    float *tangentSlopeEdge, float *slopeEdge, const float *normalSlopeEdge,
    const float *upperSurfaceVertex, const int *verticesOnEdge, const float *dvEdge,
    const int *edgeMask, const int *nEdgesOnEdge, const int *edgesOnEdge,
    const float *weightsOnEdge, int option, int nEdges, int maxEdges2)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    float t;
    if (option == 3) {
        t = 0.0f;
        const int n = nEdgesOnEdge[iEdge];
        for (int i = 0; i < n; ++i) {
            const int eoe = edgesOnEdge[iEdge * maxEdges2 + i] - 1;
            t = A(t, M(weightsOnEdge[iEdge * maxEdges2 + i], normalSlopeEdge[eoe]));
        }
    } else if (li_dynamic(edgeMask[iEdge])) {
        t = D(S(upperSurfaceVertex[verticesOnEdge[iEdge * 2] - 1],
                upperSurfaceVertex[verticesOnEdge[iEdge * 2 + 1] - 1]), dvEdge[iEdge]);
    } else {
        t = 0.0f;
    }
    tangentSlopeEdge[iEdge] = t;
    const float ns = normalSlopeEdge[iEdge];
    slopeEdge[iEdge] = __fsqrt_rn(A(M(ns, ns), M(t, t)));
}

// The SIA velocity profile, li_sia_solve's edge loop over nEdgesSolve, one
// thread per edge. normalVelocity is (nVertInterfaces, nEdges).
extern "C" __global__ void li_sia_velocity_v842(
    float *normalVelocity, const float *flowParamA, const float *thickness,
    const float *slopeEdge, const float *normalSlopeEdge, const float *layerInterfaceSigma,
    const int *cellsOnEdge, const int *edgeMask, const int *cellMask,
    float iceDensity, float n, int nEdgesSolve, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdgesSolve) return;
    const int nVertInterfaces = nVertLevels + 1;
    float *velocity = normalVelocity + iEdge * nVertInterfaces;
    if (!li_dynamic(edgeMask[iEdge])) {
        for (int k = 0; k < nVertInterfaces; ++k) velocity[k] = 0.0f;  // zero velocity on non-dynamic edges
        return;
    }
    const float positionIndependentFactor = -M(0.5f, gfk_pow(M(iceDensity, LI_GRAVITY), n));
    const float nMinus1 = S(n, 1.0f), nPlus1 = A(n, 1.0f);
    const int cell1 = cellsOnEdge[iEdge * 2] - 1;
    const int cell2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const int cell1IsDynamic = li_dynamic_int(cellMask[cell1]);
    const int cell2IsDynamic = li_dynamic_int(cellMask[cell2]);
    const float thicknessEdge = M(A(thickness[cell1], thickness[cell2]), 0.5f);
    const float levelIndependentFactor = M(M(gfk_pow(slopeEdge[iEdge], nMinus1), normalSlopeEdge[iEdge]),
                                           gfk_pow(thicknessEdge, nPlus1));
    velocity[nVertInterfaces - 1] = 0.0f;  // Assume no sliding
    for (int iLevel = nVertInterfaces - 2; iLevel >= 0; --iLevel) {
        const float flwaLevelEdge = D(A(M(flowParamA[cell1 * nVertLevels + iLevel], (float)cell1IsDynamic),
                                        M(flowParamA[cell2 * nVertLevels + iLevel], (float)cell2IsDynamic)),
                                      (float)(cell1IsDynamic + cell2IsDynamic));
        velocity[iLevel] = A(velocity[iLevel + 1],
            M(M(M(positionIndependentFactor, levelIndependentFactor), flwaLevelEdge),
              S(gfk_pow(layerInterfaceSigma[iLevel], nPlus1), gfk_pow(layerInterfaceSigma[iLevel + 1], nPlus1))));
    }
}
