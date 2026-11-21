// MPAS-Ocean v8.4.2: the Gent-McWilliams bolus velocity and the isoneutral
// slopes Redi mixing reads (ocn_gm_compute_Bolus_velocity).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_gm.F (ocn_gm_compute_Bolus_velocity,
//   tridiagonal_solve, ocn_gm_init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); MIN and MAX keep their
// first argument unless the second is strictly smaller (larger), as gfortran's
// do. A thread owns one edge or cell column, and each kernel is one of the
// routine's loops (or loops over the same extent that read nothing another
// column of the same loop writes), launched in the routine's order.
//
// Layout: a(nVertLevels, nEdges+1) is a[iEdge*nl + k] and a(nVertLevelsP1,
// nEdges+1) is a[iEdge*(nl+1) + k] (cells alike); a(maxEdges, nCells+1) is
// a[iCell*maxEdges + i]. Connectivity is 1-based.
//
// The routine fills five edge scratch fields and one cell one with
// huge(0D0) before use: RKIND 8 keeps DBL_MAX, RKIND 4 converts it to +Inf
// (v8.4.2 assigns the double constant to the RKIND 4 array; gfortran accepts
// that only with -fno-range-check, which the oracle builds this file with).

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define C(x) x
#define HUGE_D0 __longlong_as_double(0x7fefffffffffffffLL)
__device__ __forceinline__ rk gm_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk gm_max(rk a, rk b) {
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
#define HUGE_D0 __int_as_float(0x7f800000)
__device__ __forceinline__ rk gm_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk gm_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif
#define EPS_GM C(1.0e-12)

__device__ __forceinline__ int gm_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// the opening fills, edges 1..nEdges+1, levels 1..nVertLevels: the scratch
// fields huge(0D0), the slope, tapering and bolus velocity zero
extern "C" __global__ void ocn_gm_init_edges_v842(
    rk *gradDensityEdge, rk *gradDensityTopOfEdge, rk *dDensityDzTopOfEdge, rk *gradZMidEdge,
    rk *gradZMidTopOfEdge, rk *relativeSlopeTopOfEdge, rk *relativeSlopeTapering, rk *normalGMBolusVelocity,
    int nEdgesP1, int nl)
{
    const int t = gm_thread();
    if (t >= nEdgesP1 * nl) return;
    const int iEdge = t / nl, k = t % nl, p = iEdge * (nl + 1) + k;
    gradDensityEdge[t] = HUGE_D0;
    gradDensityTopOfEdge[p] = HUGE_D0;
    dDensityDzTopOfEdge[p] = HUGE_D0;
    gradZMidEdge[t] = HUGE_D0;
    gradZMidTopOfEdge[p] = HUGE_D0;
    relativeSlopeTopOfEdge[p] = C(0.0);
    relativeSlopeTapering[p] = C(0.0);
    normalGMBolusVelocity[t] = C(0.0);
}

// cells 1..nCells+1, levels 1..nVertLevels
extern "C" __global__ void ocn_gm_init_cells_v842(
    rk *dDensityDzTopOfCell, rk *k33, rk *relativeSlopeTopOfCell, rk *relativeSlopeTaperingCell, int nCellsP1,
    int nl)
{
    const int t = gm_thread();
    if (t >= nCellsP1 * nl) return;
    const int p = (t / nl) * (nl + 1) + t % nl;
    dDensityDzTopOfCell[p] = HUGE_D0;
    k33[p] = C(0.0);
    relativeSlopeTopOfCell[p] = C(0.0);
    relativeSlopeTaperingCell[p] = C(0.0);
}

// dDensityDzTopOfCell, cells 1..nCellsArray(3)
extern "C" __global__ void ocn_gm_ddensity_dz_cell_v842(
    rk *dDensityDzTopOfCell, const rk *displacedDensity, const rk *density, const rk *zMid,
    const int *maxLevelCell, int nCells, int nl)
{
    const int iCell = gm_thread();
    if (iCell >= nCells) return;
    const int mlc = maxLevelCell[iCell], q = iCell * nl;
    rk *out = dDensityDzTopOfCell + iCell * (nl + 1);
    for (int k = 1; k < mlc; ++k) {
        const rk rtmp = D(S(displacedDensity[q + k - 1], density[q + k]), S(zMid[q + k - 1], zMid[q + k]));
        out[k] = gm_min(rtmp, -EPS_GM);
    }
    out[0] = C(0.0);
    out[mlc] = C(0.0);
}

// edges 1..nEdgesArray(3): dDensityDzTopOfEdge, gradDensityEdge and
// gradZMidEdge, their values at the layer interfaces, the constant-z
// density gradient, the relative slope and its clip to [-1, 1]
extern "C" __global__ void ocn_gm_edge_slope_v842(
    rk *dDensityDzTopOfEdge, rk *gradDensityEdge, rk *gradZMidEdge, rk *gradDensityTopOfEdge,
    rk *gradZMidTopOfEdge, rk *gradDensityConstZTopOfEdge, rk *relativeSlopeTopOfEdge,
    const rk *dDensityDzTopOfCell, const rk *density, const rk *zMid, const rk *layerThicknessEdge,
    const rk *dcEdge, const int *cellsOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int iEdge = gm_thread();
    if (iEdge >= nEdges) return;
    const int l1 = nl + 1, mlet = maxLevelEdgeTop[iEdge];
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    const int pe = iEdge * l1, qe = iEdge * nl;
    for (int k = 0; k <= mlet; ++k)
        dDensityDzTopOfEdge[pe + k] = M(C(0.5), A(dDensityDzTopOfCell[cell1 * l1 + k],
                                                   dDensityDzTopOfCell[cell2 * l1 + k]));
    for (int k = 0; k < mlet; ++k) {
        gradDensityEdge[qe + k] = D(S(density[cell2 * nl + k], density[cell1 * nl + k]), dcEdge[iEdge]);
        gradZMidEdge[qe + k] = D(S(zMid[cell2 * nl + k], zMid[cell1 * nl + k]), dcEdge[iEdge]);
    }
    if (mlet >= 1) {
        for (int k = 1; k < mlet; ++k) {
            const rk h1 = layerThicknessEdge[qe + k - 1], h2 = layerThicknessEdge[qe + k];
            gradDensityTopOfEdge[pe + k] = D(A(M(h2, gradDensityEdge[qe + k - 1]), M(h1, gradDensityEdge[qe + k])),
                                             A(h1, h2));
            gradZMidTopOfEdge[pe + k] = D(A(M(h2, gradZMidEdge[qe + k - 1]), M(h1, gradZMidEdge[qe + k])),
                                          A(h1, h2));
        }
        gradDensityTopOfEdge[pe] = gradDensityEdge[qe];
        gradDensityTopOfEdge[pe + mlet] = gradDensityEdge[qe + mlet - 1];
        gradZMidTopOfEdge[pe] = gradZMidEdge[qe];
        gradZMidTopOfEdge[pe + mlet] = gradZMidEdge[qe + mlet - 1];
        for (int k = 0; k <= mlet; ++k)
            gradDensityConstZTopOfEdge[pe + k] = S(gradDensityTopOfEdge[pe + k],
                                                   M(dDensityDzTopOfEdge[pe + k], gradZMidTopOfEdge[pe + k]));
    }
    rk *slope = relativeSlopeTopOfEdge + pe;
    for (int k = 0; k < l1; ++k) slope[k] = C(0.0);
    for (int k = 1; k < mlet; ++k)
        slope[k] = -D(gradDensityTopOfEdge[pe + k], gm_min(dDensityDzTopOfEdge[pe + k], -EPS_GM));
    slope[0] = slope[1];
    slope[mlet] = slope[(mlet > 1 ? mlet : 1) - 1];
    for (int k = 0; k < nl; ++k) slope[k] = gm_max(gm_min(slope[k], C(1.0)), C(-1.0));
}

// cells 1..nCells+1: the slope averaged to cells (cells 1..nCellsArray(2)),
// k33 zeroed (1..nCellsArray(last)), the tapering's working copy of the slope
// (every cell), the five smoothing passes, the tapering (maximum slope,
// surface and bottom layers) and k33 (1..nCellsArray(2)), and k33 disabled
// (1..nCellsArray(last))
extern "C" __global__ void ocn_gm_cell_slope_v842(
    rk *relativeSlopeTopOfCell, rk *relativeSlopeTaperingCell, rk *k33, rk *areaCellSum,
    const rk *relativeSlopeTopOfEdge, const rk *dcEdge, const rk *dvEdge, const rk *zMid,
    const rk *boundaryLayerDepth, const int *nEdgesOnCell, const int *edgesOnCell, const int *maxLevelEdgeTop,
    const int *maxLevelCell, int nCellsP1, int nCells2, int nCellsLast, int nl, int maxEdges,
    rk maxRelativeSlope, int surfaceTapering, rk surfaceExtent, int bottomTapering, rk bottomDepth,
    int disableK33)
{
    const int iCell = gm_thread();
    if (iCell >= nCellsP1) return;
    const int l1 = nl + 1, mlc = maxLevelCell[iCell], pc = iCell * l1, qc = iCell * nl;
    rk *sc = relativeSlopeTopOfCell + pc, *tc = relativeSlopeTaperingCell + pc, *kk = k33 + pc;
    if (iCell < nCells2) {
        rk *sum = areaCellSum + qc;
        for (int k = 0; k < nl; ++k) sum[k] = C(1.0e-34);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
            const rk areaEdge = M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]);
            for (int k = 0; k < maxLevelEdgeTop[iEdge]; ++k) {
                const rk s = relativeSlopeTopOfEdge[iEdge * l1 + k];
                sc[k] = A(sc[k], M(areaEdge, M(s, s)));
                sum[k] = A(sum[k], areaEdge);
            }
        }
        for (int k = 0; k < mlc; ++k) sc[k] = SQRT(D(sc[k], sum[k]));
    }
    if (iCell < nCellsLast)
        for (int k = 0; k < l1; ++k) kk[k] = C(0.0);
    for (int k = 0; k < l1; ++k) tc[k] = sc[k];
    if (iCell < nCells2) {
        for (int iter = 0; iter < 5; ++iter) {
            tc[0] = C(0.0);
            for (int k = mlc - 1; k < nl; ++k) tc[k] = C(0.0);
            for (int k = 1; k < mlc - 1; ++k) {
                const rk rtmp = A(sc[k - 1], sc[k + 1]);
                const rk stmp = M(C(2.0), sc[k]);
                tc[k] = D(A(rtmp, stmp), C(4.0));
            }
            for (int k = 0; k < l1; ++k) sc[k] = tc[k];
        }
        for (int k = 0; k < mlc; ++k) tc[k] = gm_min(C(1.0), D(maxRelativeSlope, A(sc[k], EPS_GM)));
        const rk *z = zMid + qc;
        if (surfaceTapering) {
            const rk depth = gm_max(surfaceExtent, A(boundaryLayerDepth[iCell], EPS_GM));
            for (int k = 0; k < mlc; ++k) {
                rk rtmp = -D(z[k], depth);
                rtmp = gm_max(C(0.0), rtmp);
                rtmp = gm_min(C(1.0), rtmp);
                tc[k] = M(rtmp, tc[k]);
            }
        }
        if (bottomTapering) {
            for (int k = 0; k < mlc; ++k) {
                rk rtmp = D(S(z[k], z[mlc - 1]), A(bottomDepth, EPS_GM));
                rtmp = gm_max(C(0.0), rtmp);
                rtmp = gm_min(C(1.0), rtmp);
                tc[k] = M(rtmp, tc[k]);
            }
        }
        for (int k = 0; k < l1; ++k) kk[k] = C(0.0);
        for (int k = 1; k < mlc; ++k) {
            const rk ts = M(tc[k], sc[k]);
            kk[k] = M(ts, ts);
        }
    }
    if (disableK33 && iCell < nCellsLast)
        for (int k = 0; k < l1; ++k) kk[k] = C(0.0);
}

// relativeSlopeTapering, edges 1..nEdgesArray(3)
extern "C" __global__ void ocn_gm_edge_tapering_v842(
    rk *relativeSlopeTapering, const rk *relativeSlopeTaperingCell, const int *cellsOnEdge,
    const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int iEdge = gm_thread();
    if (iEdge >= nEdges) return;
    const int l1 = nl + 1;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    for (int k = 0; k < maxLevelEdgeTop[iEdge]; ++k)
        relativeSlopeTapering[iEdge * l1 + k] = M(C(0.5), A(relativeSlopeTaperingCell[cell1 * l1 + k],
                                                             relativeSlopeTaperingCell[cell2 * l1 + k]));
}

// edges 1..nEdgesArray(3): the stream function's tridiagonal system and its
// solve (work: a, b, c, r, bTemp, rTemp, x per edge, nl words each), then the
// bolus velocity
extern "C" __global__ void ocn_gm_stream_function_v842(
    rk *gmStreamFuncTopOfEdge, rk *normalGMBolusVelocity, rk *work, const rk *BruntVaisalaFreqTop,
    const rk *layerThicknessEdge, const rk *gradDensityConstZTopOfEdge, const int *cellsOnEdge,
    const int *maxLevelEdgeTop, int nEdges, int nl, rk gravWaveSpeed, rk kappa, rk gravity, rk rhoSw)
{
    const int iEdge = gm_thread();
    if (iEdge >= nEdges) return;
    const int l1 = nl + 1, mlet = maxLevelEdgeTop[iEdge], pe = iEdge * l1, qe = iEdge * nl;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    const rk *h = layerThicknessEdge + qe;
    rk *psi = gmStreamFuncTopOfEdge + pe;
    for (int k = 0; k < l1; ++k) psi[k] = C(0.0);
    if (mlet >= 3) {
        rk *a = work + (size_t)iEdge * 7 * nl, *b = a + nl, *c = b + nl, *r = c + nl;
        rk *bTemp = r + nl, *rTemp = bTemp + nl, *x = rTemp + nl;
        const rk g2 = M(gravWaveSpeed, gravWaveSpeed);
        const rk rhsCoef = D(M(kappa, gravity), rhoSw);
        // rows k = 2..maxLevelEdgeTop (Fortran levels); f is the 0-based level k-1
        for (int f = 1; f < mlet; ++f) {
            rk bv = M(C(0.5), A(BruntVaisalaFreqTop[cell1 * nl + f], BruntVaisalaFreqTop[cell2 * nl + f]));
            bv = gm_max(bv, C(0.0));
            if (f > 1) a[f - 2] = D(D(M(C(2.0), g2), h[f - 1]), A(h[f - 1], h[f]));
            b[f - 1] = S(-D(M(C(2.0), g2), M(h[f - 1], h[f])), bv);
            if (f < mlet - 1) c[f - 1] = D(D(M(C(2.0), g2), h[f]), A(h[f - 1], h[f]));
            r[f - 1] = M(rhsCoef, gradDensityConstZTopOfEdge[pe + f]);
        }
        const int n = mlet - 1;
        bTemp[0] = b[0];
        rTemp[0] = r[0];
        for (int i = 1; i < n; ++i) {
            const rk m = D(a[i - 1], bTemp[i - 1]);
            bTemp[i] = S(b[i], M(m, c[i - 1]));
            rTemp[i] = S(r[i], M(m, rTemp[i - 1]));
        }
        x[n - 1] = D(rTemp[n - 1], bTemp[n - 1]);
        for (int i = n - 2; i >= 0; --i) x[i] = D(S(rTemp[i], M(c[i], x[i + 1])), bTemp[i]);
        for (int i = 0; i < n; ++i) psi[1 + i] = x[i];
    }
    for (int k = 0; k < mlet; ++k) normalGMBolusVelocity[qe + k] = D(S(psi[k], psi[k + 1]), h[k]);
}

// gmStreamFuncTopOfCell, cells 1..nCellsArray(1)
extern "C" __global__ void ocn_gm_stream_function_cell_v842(
    rk *gmStreamFuncTopOfCell, const rk *gmStreamFuncTopOfEdge, const rk *dcEdge, const rk *dvEdge,
    const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell, const int *maxLevelEdgeTop, int nCells,
    int nl, int maxEdges)
{
    const int iCell = gm_thread();
    if (iCell >= nCells) return;
    const int l1 = nl + 1;
    rk *out = gmStreamFuncTopOfCell + iCell * nl;
    for (int k = 0; k < nl; ++k) out[k] = C(0.0);
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk areaEdge = M(M(C(0.25), dcEdge[iEdge]), dvEdge[iEdge]);
        const rk *psi = gmStreamFuncTopOfEdge + iEdge * l1;
        for (int k = 0; k < maxLevelEdgeTop[iEdge]; ++k)
            out[k] = A(out[k], M(M(C(0.5), A(psi[k], psi[k + 1])), areaEdge));
    }
    for (int k = 0; k < nl; ++k) out[k] = D(out[k], areaCell[iCell]);
}
