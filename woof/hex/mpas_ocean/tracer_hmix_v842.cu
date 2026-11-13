// MPAS-Ocean v8.4.2: horizontal tracer mixing, del2 and del4.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_tracer_hmix.F (ocn_tracer_hmix_tend, _init)
//   src/core_ocean/shared/mpas_ocn_tracer_hmix_del2.F (ocn_tracer_hmix_del2_tend, _init)
//   src/core_ocean/shared/mpas_ocn_tracer_hmix_del4.F (ocn_tracer_hmix_del4_tend, _init)
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
