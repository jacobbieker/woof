// MPAS-Ocean v8.4.2: horizontal tracer mixing, del2, del4 and Redi.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_tracer_hmix.F (ocn_tracer_hmix_tend, _init)
//   src/core_ocean/shared/mpas_ocn_tracer_hmix_del2.F (ocn_tracer_hmix_del2_tend, _init)
//   src/core_ocean/shared/mpas_ocn_tracer_hmix_del4.F (ocn_tracer_hmix_del4_tend, _init)
//   src/core_ocean/shared/mpas_ocn_tracer_hmix_redi.F (ocn_tracer_hmix_redi_tend, _init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); an integer sign times a
// real is the sign converted, then multiplied. A thread owns one (cell,
// level, tracer) word and sums its edges in the Fortran's order.
//
// Layout: tracers(nTracers, nVertLevels, nCells+1) is
// t[(iCell*nl + k)*nTracers + i]; a(nVertLevels, nEdges+1) is a[iEdge*nl + k];
// a(maxEdges, nCells+1) is a[iCell*maxEdges + i]. Connectivity is 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
#endif

__device__ __forceinline__ int hmix_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// del2, cells 1..nCellsArray(1)
extern "C" __global__ void ocn_tracer_hmix_del2_v842(
    rk *tend, const rk *tracers, const rk *layerThicknessEdge, const rk *meshScalingDel2, const rk *dvEdge,
    const rk *dcEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *cellsOnEdge, const int *maxLevelEdgeTop, int nCells, int nl, int nTracers,
    int maxEdges, rk eddyDiff2)
{
    const int t = hmix_thread();
    if (t >= nCells * nl * nTracers) return;
    const int iTracer = t % nTracers, k = (t / nTracers) % nl, iCell = t / (nTracers * nl);
    const rk invAreaCell = D(C(1.0), areaCell[iCell]);
    rk value = tend[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeTop[iEdge]) continue;
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        const rk r_tmp = D(M(M(meshScalingDel2[iEdge], eddyDiff2), dvEdge[iEdge]), dcEdge[iEdge]);
        const rk turb = S(tracers[(cell2 * nl + k) * nTracers + iTracer],
                          tracers[(cell1 * nl + k) * nTracers + iTracer]);
        const rk flux = M(M(layerThicknessEdge[iEdge * nl + k], turb), r_tmp);
        value = S(value, M(M((rk)edgeSignOnCell[iCell * maxEdges + i], flux), invAreaCell));
    }
    tend[t] = value;
}

// del4's first Laplacian, cells 1..nCellsArray(2) (every level and tracer
// zeroed first; an edge level with edgeMask 0 adds nothing)
extern "C" __global__ void ocn_tracer_hmix_del4_laplacian_v842(
    rk *delsq, const rk *tracers, const rk *layerThicknessEdge, const rk *dvEdge, const rk *dcEdge,
    const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell, const int *edgeSignOnCell,
    const int *cellsOnEdge, const int *maxLevelEdgeTop, const int *edgeMask, int nCells, int nl, int nTracers,
    int maxEdges)
{
    const int t = hmix_thread();
    if (t >= nCells * nl * nTracers) return;
    const int iTracer = t % nTracers, k = (t / nTracers) % nl, iCell = t / (nTracers * nl);
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    rk value = C(0.0);
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeTop[iEdge] || edgeMask[iEdge * nl + k] == 0) continue;
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        const rk invdcEdge = D(dvEdge[iEdge], dcEdge[iEdge]);
        const rk h = layerThicknessEdge[iEdge * nl + k];
        const rk r1 = M(M(invdcEdge, h), tracers[(cell1 * nl + k) * nTracers + iTracer]);
        const rk r2 = M(M(invdcEdge, h), tracers[(cell2 * nl + k) * nTracers + iTracer]);
        value = S(value, M(M((rk)edgeSignOnCell[iCell * maxEdges + i], S(r2, r1)), invAreaCell1));
    }
    delsq[t] = value;
}

// del4's tendency, cells 1..nCellsArray(1)
extern "C" __global__ void ocn_tracer_hmix_del4_v842(
    rk *tend, const rk *delsq, const rk *meshScalingDel4, const rk *dvEdge, const rk *dcEdge, const rk *areaCell,
    const int *nEdgesOnCell, const int *edgesOnCell, const int *edgeSignOnCell, const int *cellsOnEdge,
    const int *maxLevelEdgeTop, const int *edgeMask, int nCells, int nl, int nTracers, int maxEdges, rk eddyDiff4)
{
    const int t = hmix_thread();
    if (t >= nCells * nl * nTracers) return;
    const int iTracer = t % nTracers, k = (t / nTracers) % nl, iCell = t / (nTracers * nl);
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    rk value = tend[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeTop[iEdge] || edgeMask[iEdge * nl + k] == 0) continue;
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        const rk invdcEdge = D(M(M(meshScalingDel4[iEdge], dvEdge[iEdge]), eddyDiff4), dcEdge[iEdge]);
        const rk turb = S(delsq[(cell2 * nl + k) * nTracers + iTracer], delsq[(cell1 * nl + k) * nTracers + iTracer]);
        const rk flux = M(turb, invdcEdge);
        value = A(value, M(M((rk)edgeSignOnCell[iCell * maxEdges + i], flux), invAreaCell1));
    }
    tend[t] = value;
}

// ---------------------------------------------------------------------------
// Redi isoneutral mixing, ocn_tracer_hmix_redi_tend
//   (src/core_ocean/shared/mpas_ocn_tracer_hmix_redi.F, same copyright and terms).
// The Fortran loops over tracers, reusing its scratch fields; every word a
// tracer reads is either written for that tracer or the opening zero, so
// each tracer here has its own zeroed scratch slice and the tracers run
// together. Scratch slices: s[(iTracer*rows + row)*ld + k] (ld nl or nl+1).

// dTracerdZTopOfCell, cells 1..nCellsArray(2), one thread per (cell, tracer)
extern "C" __global__ void ocn_tracer_hmix_redi_dtdz_cell_v842(
    rk *dTracerdZTopOfCell, const rk *tracers, const rk *zMid, const int *maxLevelCell, int nCells, int rowsC,
    int nl, int nTracers)
{
    const int t = hmix_thread();
    if (t >= nCells * nTracers) return;
    const int iTracer = t % nTracers, iCell = t / nTracers;
    const int mlc = maxLevelCell[iCell], l1 = nl + 1;
    rk *dz = dTracerdZTopOfCell + ((size_t)iTracer * rowsC + iCell) * l1;
    for (int k = 1; k < mlc; ++k)
        dz[k] = D(S(tracers[(iCell * nl + k - 1) * nTracers + iTracer], tracers[(iCell * nl + k) * nTracers + iTracer]),
                  S(zMid[iCell * nl + k - 1], zMid[iCell * nl + k]));
    dz[0] = C(0.0);
    dz[mlc] = C(0.0);
}

// gradTracerEdge, dTracerdZTopOfEdge, gradTracerTopOfEdge, edges 1..nEdgesArray(2),
// one thread per (edge, tracer)
extern "C" __global__ void ocn_tracer_hmix_redi_edge_v842(
    rk *gradTracerEdge, rk *dTracerdZTopOfEdge, rk *gradTracerTopOfEdge, const rk *dTracerdZTopOfCell,
    const rk *tracers, const rk *layerThicknessEdge, const rk *dcEdge, const int *cellsOnEdge,
    const int *maxLevelEdgeTop, int nEdges, int rowsC, int rowsE, int nl, int nTracers)
{
    const int t = hmix_thread();
    if (t >= nEdges * nTracers) return;
    const int iTracer = t % nTracers, iEdge = t / nTracers;
    const int mlet = maxLevelEdgeTop[iEdge], l1 = nl + 1;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    rk *ge = gradTracerEdge + ((size_t)iTracer * rowsE + iEdge) * nl;
    rk *dze = dTracerdZTopOfEdge + ((size_t)iTracer * rowsE + iEdge) * l1;
    rk *gt = gradTracerTopOfEdge + ((size_t)iTracer * rowsE + iEdge) * l1;
    const rk *dzc = dTracerdZTopOfCell + (size_t)iTracer * rowsC * l1;
    const rk *h = layerThicknessEdge + iEdge * nl;
    for (int k = 0; k < mlet; ++k)
        ge[k] = D(S(tracers[(cell2 * nl + k) * nTracers + iTracer], tracers[(cell1 * nl + k) * nTracers + iTracer]),
                  dcEdge[iEdge]);
    for (int k = 0; k < mlet; ++k) dze[k] = M(C(0.5), A(dzc[cell1 * l1 + k], dzc[cell2 * l1 + k]));
    dze[mlet] = C(0.0);
    for (int k = 1; k < mlet; ++k) {
        const rk h1 = h[k - 1], h2 = h[k];
        gt[k] = D(A(M(h2, ge[k - 1]), M(h1, ge[k])), A(h1, h2));
    }
    // v8.4.2 takes the top interface from level 2
    gt[0] = ge[1];
    gt[mlet] = ge[(mlet > 1 ? mlet : 1) - 1];
}

// gradHTracerSlopedTopOfCell over areaCellSum, cells 1..nCellsArray(1),
// one thread per (cell, tracer)
extern "C" __global__ void ocn_tracer_hmix_redi_sloped_cell_v842(
    rk *gradHTracerSlopedTopOfCell, rk *areaCellSum, const rk *gradTracerTopOfEdge,
    const rk *relativeSlopeTopOfEdge, const rk *relativeSlopeTapering, const rk *dcEdge, const rk *dvEdge,
    const int *nEdgesOnCell, const int *edgesOnCell, const int *maxLevelEdgeTop, const int *maxLevelCell,
    int nCells, int rowsC, int rowsE, int nl, int nTracers, int maxEdges)
{
    const int t = hmix_thread();
    if (t >= nCells * nTracers) return;
    const int iTracer = t % nTracers, iCell = t / nTracers;
    const int l1 = nl + 1;
    rk *g = gradHTracerSlopedTopOfCell + ((size_t)iTracer * rowsC + iCell) * l1;
    rk *acs = areaCellSum + ((size_t)iTracer * rowsC + iCell) * nl;
    const rk *gt = gradTracerTopOfEdge + (size_t)iTracer * rowsE * l1;
    for (int k = 0; k < nl; ++k) acs[k] = C(1.0e-34);
    for (int k = 0; k < l1; ++k) g[k] = C(0.0);
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk areaEdge = M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]);
        for (int k = 0; k < maxLevelEdgeTop[iEdge]; ++k) {
            const rk r_tmp = M(M(M(areaEdge, relativeSlopeTapering[iEdge * l1 + k]),
                                 relativeSlopeTopOfEdge[iEdge * l1 + k]), gt[iEdge * l1 + k]);
            g[k] = A(g[k], r_tmp);
            acs[k] = A(acs[k], areaEdge);
        }
    }
    for (int k = 0; k < maxLevelCell[iCell]; ++k) g[k] = D(g[k], acs[k]);
}

// Redi's three horizontal terms on tend, cells 1..nCellsArray(1), one thread
// per (cell, level, tracer) word: term 1 (for every tracer before any other
// term in the Fortran, which a word does not see), term 2, term 3
extern "C" __global__ void ocn_tracer_hmix_redi_tend_v842(
    rk *tend, const rk *tracers, const rk *layerThicknessEdge, const rk *dTracerdZTopOfEdge,
    const rk *gradHTracerSlopedTopOfCell, const rk *relativeSlopeTopOfEdge, const rk *relativeSlopeTapering,
    const rk *dvEdge, const rk *dcEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *cellsOnEdge, const int *maxLevelEdgeTop, const int *maxLevelCell,
    int nCells, int rowsC, int rowsE, int nl, int nTracers, int maxEdges, rk kappa, int term1, int term2, int term3)
{
    const int t = hmix_thread();
    if (t >= nCells * nl * nTracers) return;
    const int iTracer = t % nTracers, k = (t / nTracers) % nl, iCell = t / (nTracers * nl);
    const int l1 = nl + 1;
    rk value = tend[t];
    if (term1) {
        const rk invAreaCell = D(C(1.0), areaCell[iCell]);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
            if (k >= maxLevelEdgeTop[iEdge]) continue;
            const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
            const rk r_tmp = D(M(kappa, dvEdge[iEdge]), dcEdge[iEdge]);
            const rk turb = S(tracers[(cell2 * nl + k) * nTracers + iTracer],
                              tracers[(cell1 * nl + k) * nTracers + iTracer]);
            const rk flux = M(M(layerThicknessEdge[iEdge * nl + k], turb), r_tmp);
            value = S(value, M(M((rk)edgeSignOnCell[iCell * maxEdges + i], flux), invAreaCell));
        }
    }
    if (term2) {
        const rk invAreaCell = D(C(1.0), areaCell[iCell]);
        const rk *dze = dTracerdZTopOfEdge + (size_t)iTracer * rowsE * l1;
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
            if (k >= maxLevelEdgeTop[iEdge]) continue;
            const int p = iEdge * l1 + k;
            const rk s_tmpU = M(M(relativeSlopeTapering[p], relativeSlopeTopOfEdge[p]), dze[p]);
            const rk s_tmpD = M(M(relativeSlopeTapering[p + 1], relativeSlopeTopOfEdge[p + 1]), dze[p + 1]);
            rk flux = M(M(C(0.5), dvEdge[iEdge]), A(s_tmpU, s_tmpD));
            flux = M(flux, layerThicknessEdge[iEdge * nl + k]);
            value = A(value, M(M(M((rk)edgeSignOnCell[iCell * maxEdges + i], kappa), flux), invAreaCell));
        }
    }
    if (term3) {
        const int mlc = maxLevelCell[iCell];
        if (k < mlc) {
            const rk *g = gradHTracerSlopedTopOfCell + ((size_t)iTracer * rowsC + iCell) * l1;
            const rk upper = k == 0 ? C(0.0) : g[k];
            const rk lower = k + 1 == mlc ? C(0.0) : g[k + 1];
            value = A(value, M(kappa, S(upper, lower)));
        }
    }
    tend[t] = value;
}
