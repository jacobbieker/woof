// MPAS-Ocean v8.4.2: the vertical (ALE) transport through the top of each layer.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_diagnostics.F (ocn_vert_transport_velocity_top)
//   src/core_ocean/shared/mpas_ocn_thick_ale.F (ocn_ALE_thickness)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); an integer sign times a
// real is the sign converted, then multiplied. Fortran MIN/MAX are chosen on
// the words as integer selects. Each kernel runs one thread per column, the
// Fortran's sums in its order.
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nl + k]; vertAleTransportTop
// (nVertLevelsP1, nCells+1) is w[iCell*(nl+1) + k]; restingThickness as
// layerThickness. Connectivity is 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
__device__ __forceinline__ rk vt_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk vt_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
__device__ __forceinline__ rk vt_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk vt_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif
__device__ __forceinline__ int vt_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// the thickness-flux divergence of each layer and the column's (edges outer,
// levels inner, as the Fortran sums them), then the projected SSH; cells
// 1..nCellsArray(2)
extern "C" __global__ void ocn_vt_divergence_v842(
    rk *divHu, rk *projectedSSH, const rk *layerThicknessEdge, const rk *normalVelocity, const rk *oldSSH,
    const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *maxLevelEdgeBot, int nCells, int nl, int maxEdges, rk dt)
{
    const int iCell = vt_thread();
    if (iCell >= nCells) return;
    rk *div = divHu + iCell * nl;
    for (int k = 0; k < nl; ++k) div[k] = C(0.0);
    rk btr = C(0.0);
    const rk invAreaCell = D(C(1.0), areaCell[iCell]);
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk sign = (rk)edgeSignOnCell[iCell * maxEdges + i];
        for (int k = 0; k < maxLevelEdgeBot[iEdge]; ++k) {
            const rk flux = M(M(M(M(layerThicknessEdge[iEdge * nl + k], normalVelocity[iEdge * nl + k]),
                                  dvEdge[iEdge]), sign), invAreaCell);
            div[k] = S(div[k], flux);
            btr = S(btr, flux);
        }
    }
    projectedSSH[iCell] = S(oldSSH[iCell], M(dt, btr));
}

// ocn_ALE_thickness, cells 1..nCells (z-star, then the min/max thickness)
extern "C" __global__ void ocn_vt_ale_thickness_v842(
    rk *aleThickness, rk *work, const rk *ssh, const rk *restingThickness, const rk *vertCoordMovementWeights,
    const int *maxLevelCell, int nCells, int nl, int minMax, rk maxThicknessFactor, rk minThickness)
{
    const int iCell = vt_thread();
    if (iCell >= nCells) return;
    const int kMax = maxLevelCell[iCell];
    const rk *rest = restingThickness + iCell * nl;
    rk *ale = aleThickness + iCell * nl;
    rk *sshAle = work + iCell * 4 * nl, *prelim = sshAle + nl, *down = prelim + nl, *up = down + nl;
    rk thicknessSum = C(1e-14);
    for (int k = 0; k < kMax; ++k) {
        sshAle[k] = M(M(ssh[iCell], vertCoordMovementWeights[k]), rest[k]);
        thicknessSum = A(thicknessSum, M(vertCoordMovementWeights[k], rest[k]));
    }
    for (int k = 0; k < kMax; ++k) {
        sshAle[k] = D(sshAle[k], thicknessSum);
        ale[k] = A(rest[k], sshAle[k]);
    }
    if (!minMax) return;
    for (int k = 0; k < kMax; ++k) prelim[k] = ale[k];
    rk remainder = C(0.0);
    for (int k = 0; k < kMax; ++k) {
        const rk t = vt_max(vt_min(A(prelim[k], remainder), M(maxThicknessFactor, rest[k])), minThickness);
        down[k] = S(t, prelim[k]);
        remainder = S(remainder, down[k]);
    }
    up[kMax - 1] = C(0.0);
    for (int k = 0; k < kMax; ++k) prelim[k] = A(prelim[k], down[k]);
    for (int k = kMax - 2; k >= 0; --k) {
        const rk t = vt_max(vt_min(A(prelim[k], remainder), M(maxThicknessFactor, rest[k])), minThickness);
        up[k] = S(t, prelim[k]);
        remainder = S(remainder, up[k]);
    }
    up[0] = A(up[0], remainder);
    for (int k = 0; k < kMax; ++k) ale[k] = A(A(ale[k], down[k]), up[k]);
}

// vertAleTransportTop, cells 1..nCellsArray(2), integrated up from the bottom
extern "C" __global__ void ocn_vt_transport_top_v842(
    rk *vertAleTransportTop, const rk *divHu, const rk *aleThickness, const rk *oldLayerThickness,
    const int *maxLevelCell, int nCells, int nl, rk dt)
{
    const int iCell = vt_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell];
    rk *w = vertAleTransportTop + iCell * (nl + 1);
    w[0] = C(0.0);
    w[top] = C(0.0);
    for (int k = top - 1; k >= 1; --k)
        w[k] = S(S(w[k + 1], divHu[iCell * nl + k]),
                 D(S(aleThickness[iCell * nl + k], oldLayerThickness[iCell * nl + k]), dt));
}
