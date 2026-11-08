// MPAS-Ocean v8.4.2: four tendency terms of the momentum and thickness equations.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_vel_coriolis.F:74-187,204-241 (ocn_vel_coriolis_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_vadv.F:74-168,183-213     (ocn_vel_vadv_tend, _init)
//   src/core_ocean/shared/mpas_ocn_thick_hadv.F:72-168,183-207   (ocn_thick_hadv_tend, _init)
//   src/core_ocean/shared/mpas_ocn_thick_vadv.F:72-147,162-186   (ocn_thick_vadv_tend, _init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off: left to right, parentheses kept, no contraction). An
// integer factor (RK4On, edgeMask, edgeSignOnCell) is converted to real first,
// as Fortran's mixed-mode rule does.
//
// Layout: a Fortran array a(nVertLevels, nEdges) is a[iEdge*nVertLevels + k];
// cellsOnEdge(2, nEdges) is c[iEdge*2 + i]; edgesOnEdge(maxEdges2, nEdges) is
// e[iEdge*maxEdges2 + j]; vertAleTransportTop(nVertLevels+1, nCells) is
// w[iCell*(nVertLevels+1) + k]. Connectivity is MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#endif

// ocn_vel_coriolis_tend: one thread per edge. rk4On is RK4On (1 for 'RK4',
// 0 for 'split_explicit' / 'unsplit_explicit').
extern "C" __global__ void ocn_vel_coriolis_tend_v842(
    rk *tend, const rk *normalizedRelativeVorticityEdge,
    const rk *normalizedPlanetaryVorticityEdge, const rk *layerThicknessEdge,
    const rk *normalVelocity, const rk *kineticEnergyCell,
    const int *maxLevelEdgeTop, const int *nEdgesOnEdge, const int *cellsOnEdge,
    const int *edgesOnEdge, const rk *weightsOnEdge, const rk *dcEdge, const int *edgeMask,
    int rk4On, int nEdges, int nVertLevels, int maxEdges2)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[iEdge * 2] - 1;
    const int cell2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const rk invLength = D((rk)1.0, dcEdge[iEdge]);
    const rk r4 = (rk)rk4On;
    const int kmax = maxLevelEdgeTop[iEdge];
    const int nj = nEdgesOnEdge[iEdge];
    for (int k = 0; k < kmax; ++k) {
        rk q = (rk)0.0;
        const int p = iEdge * nVertLevels + k;
        for (int j = 0; j < nj; ++j) {
            const int eoe = edgesOnEdge[iEdge * maxEdges2 + j] - 1;
            const rk edgeWeight = weightsOnEdge[iEdge * maxEdges2 + j];
            const int pe = eoe * nVertLevels + k;
            const rk workVorticity = M((rk)0.5,
                A(A(A(normalizedRelativeVorticityEdge[p], M(r4, normalizedPlanetaryVorticityEdge[p])),
                    normalizedRelativeVorticityEdge[pe]),
                  M(r4, normalizedPlanetaryVorticityEdge[pe])));
            q = A(q, M(M(M(edgeWeight, normalVelocity[pe]), workVorticity), layerThicknessEdge[pe]));
        }
        tend[p] = A(tend[p], M((rk)edgeMask[p],
            S(q, M(S(kineticEnergyCell[cell2 * nVertLevels + k], kineticEnergyCell[cell1 * nVertLevels + k]),
                   invLength))));
    }
}

// ocn_vel_vadv_tend: one thread per edge. w_dudzTopEdge(1) is the zero the
// routine sets once and never overwrites; w_dudzTopEdge(maxLevelEdgeTop+1) is
// the zero it sets per edge.
extern "C" __global__ void ocn_vel_vadv_tend_v842(
    rk *tend, const rk *normalVelocity, const rk *layerThicknessEdge,
    const rk *vertAleTransportTop, const int *maxLevelEdgeTop, const int *cellsOnEdge,
    const int *edgeMask, int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[iEdge * 2] - 1;
    const int cell2 = cellsOnEdge[iEdge * 2 + 1] - 1;
    const int kmax = maxLevelEdgeTop[iEdge];
    const int base = iEdge * nVertLevels;
    const int nw = nVertLevels + 1;
    // w_dudzTopEdge at the top interface of layer k (0-based k = Fortran k-1)
    rk wTop = (rk)0.0;
    for (int k = 0; k < kmax; ++k) {
        rk wBot = (rk)0.0;
        if (k + 1 < kmax) {
            const int kk = k + 1;  // Fortran k = kk+1, from 2 to maxLevelEdgeTop
            const rk vertAleTransportTopEdge = M((rk)0.5,
                A(vertAleTransportTop[cell1 * nw + kk], vertAleTransportTop[cell2 * nw + kk]));
            wBot = D(M(vertAleTransportTopEdge, S(normalVelocity[base + kk - 1], normalVelocity[base + kk])),
                     M((rk)0.5, A(layerThicknessEdge[base + kk - 1], layerThicknessEdge[base + kk])));
        }
        // edgeMask * 0.5: integer times a default-real literal, exact in either kind
        tend[base + k] = S(tend[base + k], M((rk)((float)edgeMask[base + k] * 0.5f), A(wTop, wBot)));
        wTop = wBot;
    }
}

// ocn_thick_hadv_tend: one thread per cell
extern "C" __global__ void ocn_thick_hadv_tend_v842(
    rk *tend, const rk *normalVelocity, const rk *layerThicknessEdge,
    const int *maxLevelEdgeBot, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const rk *dvEdge, const rk *areaCell,
    int nCells, int nVertLevels, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk invAreaCell = D((rk)1.0, areaCell[iCell]);
    const int n = nEdgesOnCell[iCell];
    for (int i = 0; i < n; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk sign = (rk)edgeSignOnCell[iCell * maxEdges + i];
        const int kmax = maxLevelEdgeBot[iEdge];
        for (int k = 0; k < kmax; ++k) {
            const rk flux = M(M(normalVelocity[iEdge * nVertLevels + k], dvEdge[iEdge]),
                              layerThicknessEdge[iEdge * nVertLevels + k]);
            const int p = iCell * nVertLevels + k;
            tend[p] = A(tend[p], M(M(sign, flux), invAreaCell));
        }
    }
}

// ocn_thick_vadv_tend: one thread per cell
extern "C" __global__ void ocn_thick_vadv_tend_v842(
    rk *tend, const rk *vertAleTransportTop, const int *maxLevelCell, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int nw = nVertLevels + 1;
    const int kmax = maxLevelCell[iCell];
    for (int k = 0; k < kmax; ++k) {
        const int p = iCell * nVertLevels + k;
        tend[p] = S(A(tend[p], vertAleTransportTop[iCell * nw + k + 1]), vertAleTransportTop[iCell * nw + k]);
    }
}
