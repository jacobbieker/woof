// MPAS land-ice (MALI) v8.4.2: the masks, geometry and layer diagnostics the
// model computes before thickness advection.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/shared/mpas_li_mask.F:30-40,239-626,700-935 (li_calculate_mask, the mask values and predicates)
//   src/core_landice/mode_forward/mpas_li_advection.F:533-591   (li_update_geometry)
//   src/core_landice/mode_forward/mpas_li_advection.F:1397-1503 (li_layer_normal_velocity)
//   src/core_landice/shared/mpas_li_setup.F:553-609             (li_calculate_layerThickness)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4. Every operation is an explicit round-to-nearest intrinsic in the
// Fortran's evaluation order (gfortran -O0 -ffp-contract=off).
//
// li_calculate_mask runs its cell passes in sequence. Each pass reads the
// neighbours' bits that earlier passes settled and sets bits that no later
// read in the same pass depends on. So each pass here is one launch, reading
// the previous pass's mask (maskIn) and writing its own (maskOut). Only the
// SIA and 'none' velocity solvers' paths are carried; the higher-order
// solvers' Albany bits are refused by the driver.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// cellsOnCell(maxEdges, nCells) is c[iCell*maxEdges + j]. Connectivity is
// MPAS's 1-based indices.

#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

#define LI_ICE 32
#define LI_DYNAMIC_ICE 2
#define LI_FLOATING 4
#define LI_MARGIN 8
#define LI_DYNAMIC_MARGIN 16
#define LI_INITIAL_ICE_EXTENT 1
#define LI_ALBANY_ACTIVE 64
#define LI_GROUNDING_LINE 256

__device__ __forceinline__ bool li_is(int mask, int value) { return (mask & value) == value; }
__device__ __forceinline__ bool li_is_grounded(int mask) {
    return !li_is(mask, LI_FLOATING) && li_is(mask, LI_ICE);
}

// the first passes: keep the initial-extent bit; ice where thickness > 0;
// floating where ice and (rhoi/rhoo*H) <= (sea level - bed)
extern "C" __global__ void li_mask_cell_state_v842(
    int *cellMask, const float *thickness, const float *bedTopography,
    float iceDensity, float oceanDensity, float seaLevel, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int mask = cellMask[i] & LI_INITIAL_ICE_EXTENT;
    const float h = thickness[i];
    if (h > 0.0f) mask |= LI_ICE;
    if (li_is(mask, LI_ICE) && M(D(iceDensity, oceanDensity), h) <= S(seaLevel, bedTopography[i]))
        mask |= LI_FLOATING;
    cellMask[i] = mask;
}

// ice cells with a non-ice neighbour are margin; then the dynamic bit where
// thickness > config_dynamic_thickness
extern "C" __global__ void li_mask_margin_v842(
    int *maskOut, const int *maskIn, const float *thickness, const int *nEdgesOnCell,
    const int *cellsOnCell, float dynamicThickness, int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int mask = maskIn[i];
    if (li_is(mask, LI_ICE)) {
        bool isMargin = false;
        for (int j = 0; j < nEdgesOnCell[i]; ++j)
            isMargin = isMargin || !li_is(maskIn[cellsOnCell[i * maxEdges + j] - 1], LI_ICE);
        if (isMargin) mask |= LI_MARGIN;
    }
    if (thickness[i] > dynamicThickness) mask |= LI_DYNAMIC_ICE;
    maskOut[i] = mask;
}

// floating margin cells next to open ocean and thinner than 0.95 of their
// thinnest non-margin ice neighbour lose the dynamic bit
extern "C" __global__ void li_mask_calving_front_v842(
    int *maskOut, const int *maskIn, const float *thickness, const float *bedTopography,
    const int *nEdgesOnCell, const int *cellsOnCell, float seaLevel, int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int mask = maskIn[i];
    if (li_is(mask, LI_MARGIN) && li_is(mask, LI_FLOATING)) {
        bool openOceanNeighbor = false;
        float thinnestNeighborHeight = 1.0e36f;
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            const int n = cellsOnCell[i * maxEdges + j] - 1;
            if (thickness[n] > 0.0f && !li_is(maskIn[n], LI_MARGIN))
                thinnestNeighborHeight = thinnestNeighborHeight < thickness[n] ? thinnestNeighborHeight : thickness[n];
            if (bedTopography[n] < seaLevel && !li_is(maskIn[n], LI_ICE)) openOceanNeighbor = true;
        }
        if (openOceanNeighbor && thickness[i] < M(0.95f, thinnestNeighborHeight))
            mask &= ~LI_DYNAMIC_ICE;
    }
    maskOut[i] = mask;
}

// dynamic cells with a non-dynamic neighbour are dynamic margin; grounded
// cells with a floating or open-ocean neighbour are grounding line
extern "C" __global__ void li_mask_dynamic_margin_grounding_line_v842(
    int *maskOut, const int *maskIn, const float *bedTopography, const int *nEdgesOnCell,
    const int *cellsOnCell, float seaLevel, int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int mask = maskIn[i];
    const int n = nEdgesOnCell[i];
    if (li_is(mask, LI_DYNAMIC_ICE)) {
        bool isMargin = false;
        for (int j = 0; j < n; ++j)
            isMargin = isMargin || !li_is(maskIn[cellsOnCell[i * maxEdges + j] - 1], LI_DYNAMIC_ICE);
        if (isMargin) mask |= LI_DYNAMIC_MARGIN;
    }
    if (li_is_grounded(mask)) {
        for (int j = 0; j < n; ++j) {
            const int c = cellsOnCell[i * maxEdges + j] - 1;
            if (li_is(maskIn[c], LI_FLOATING) || (bedTopography[c] < seaLevel && !li_is(maskIn[c], LI_ICE)))
                mask |= LI_GROUNDING_LINE;
        }
    }
    maskOut[i] = mask;
}

// vertexMask from the vertexDegree cells around each vertex
extern "C" __global__ void li_mask_vertex_v842(
    int *vertexMask, const int *cellMask, const int *cellsOnVertex, int nVertices, int vertexDegree)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVertices) return;
    bool hasIce = false, hasNoIce = false, hasDynamic = false, hasNoDynamic = false;
    bool floating = false, grounded = false;
    for (int j = 0; j < vertexDegree; ++j) {
        const int m = cellMask[cellsOnVertex[i * vertexDegree + j] - 1];
        hasIce = hasIce || li_is(m, LI_ICE);
        hasNoIce = hasNoIce || !li_is(m, LI_ICE);
        hasDynamic = hasDynamic || li_is(m, LI_DYNAMIC_ICE);
        hasNoDynamic = hasNoDynamic || !li_is(m, LI_DYNAMIC_ICE);
        floating = floating || li_is(m, LI_FLOATING);
        grounded = grounded || li_is_grounded(m);
    }
    int mask = 0;
    if (hasIce) mask |= LI_ICE;
    if (hasDynamic) mask |= LI_DYNAMIC_ICE;
    if (floating) mask |= LI_FLOATING;
    if (floating && grounded) mask |= LI_GROUNDING_LINE;
    if (hasIce && hasNoIce) mask |= LI_MARGIN;
    if (hasDynamic && hasNoDynamic) mask |= LI_DYNAMIC_MARGIN;
    vertexMask[i] = mask;
}

// edgeMask from the two cells on each edge (a dynamic edge also carries the
// Albany-active bit, whatever the solver)
extern "C" __global__ void li_mask_edge_v842(
    int *edgeMask, const int *cellMask, const int *cellsOnEdge, int nEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nEdges) return;
    bool hasIce = false, hasNoIce = false, hasDynamic = false, hasNoDynamic = false;
    bool floating = false, grounded = false;
    for (int j = 0; j < 2; ++j) {
        const int m = cellMask[cellsOnEdge[i * 2 + j] - 1];
        hasIce = hasIce || li_is(m, LI_ICE);
        hasNoIce = hasNoIce || !li_is(m, LI_ICE);
        hasDynamic = hasDynamic || li_is(m, LI_DYNAMIC_ICE);
        hasNoDynamic = hasNoDynamic || !li_is(m, LI_DYNAMIC_ICE);
        floating = floating || li_is(m, LI_FLOATING);
        grounded = grounded || li_is_grounded(m);
    }
    int mask = 0;
    if (hasIce) mask |= LI_ICE;
    if (hasDynamic) mask |= LI_DYNAMIC_ICE | LI_ALBANY_ACTIVE;
    if (floating) mask |= LI_FLOATING;
    if (floating && grounded) mask |= LI_GROUNDING_LINE;
    if (hasIce && hasNoIce) mask |= LI_MARGIN;
    if (hasDynamic && hasNoDynamic) mask |= LI_DYNAMIC_MARGIN;
    edgeMask[i] = mask;
}

// li_update_geometry and li_calculate_layerThickness, one thread per cell
extern "C" __global__ void li_update_geometry_layer_thickness_v842(
    float *upperSurface, float *lowerSurface, float *layerThickness, const float *thickness,
    const float *bedTopography, const int *cellMask, const float *layerThicknessFractions,
    float iceDensity, float oceanDensity, float seaLevel, int nCells, int nVertLevels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const float h = thickness[i];
    const float lower = li_is(cellMask[i], LI_FLOATING) ? S(seaLevel, M(h, D(iceDensity, oceanDensity)))
                                                        : bedTopography[i];
    lowerSurface[i] = lower;
    upperSurface[i] = A(lower, h);
    for (int k = 0; k < nVertLevels; ++k)
        layerThickness[i * nVertLevels + k] = M(h, layerThicknessFractions[k]);
}

// li_layer_normal_velocity: the layer-midpoint velocity on every solved edge
// and each edge's smallest advective CFL time step (the driver takes the
// minimum over edges)
extern "C" __global__ void li_layer_normal_velocity_v842(
    float *layerNormalVelocity, float *edgeMinDt, const float *normalVelocity, const float *dcEdge,
    int nEdgesSolve, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdgesSolve) return;
    const float bigNumber = 1.0e16f;
    float minDt = bigNumber;
    for (int k = 0; k < nVertLevels; ++k) {
        const float v = M(0.5f, A(normalVelocity[iEdge * (nVertLevels + 1) + k],
                                  normalVelocity[iEdge * (nVertLevels + 1) + k + 1]));
        layerNormalVelocity[iEdge * nVertLevels + k] = v;
        const float maxAllowableDt = fabsf(v) > 0.0f ? D(M(0.5f, dcEdge[iEdge]), fabsf(v)) : bigNumber;
        minDt = minDt < maxAllowableDt ? minDt : maxAllowableDt;
    }
    edgeMinDt[iEdge] = minDt;
}

// the minimum over edges, in the Fortran's edge order (one thread)
extern "C" __global__ void li_min_dt_v842(float *minOfMaxAllowableDt, const float *edgeMinDt, int nEdgesSolve)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float m = 1.0e16f;
    for (int i = 0; i < nEdgesSolve; ++i) m = m < edgeMinDt[i] ? m : edgeMinDt[i];
    *minOfMaxAllowableDt = m;
}
