// MPAS-Ocean v8.4.2: the split-explicit step's stage 2, the barotropic
// velocity and SSH, explicitly subcycled.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F
//   (ocn_time_integrator_split, "Stage 2: Barotropic velocity (2D) prediction")
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); an integer mask or sign
// times a real is the integer converted, then multiplied. Fortran MIN is
// chosen on the words as an integer select. The host passes the scalars the
// Fortran forms once per expression with the same rounding: dt/nBtrSubcycles,
// 1.0 - gam1, 1 - gam2, 1.0 - gam3 (all exact conversions, one rounding).
//
// Layout: a(nVertLevels, nEdges+1) is a[iEdge*nl + k]; cellsOnEdge(2, .) is
// coe[iEdge*2 + j]; edgesOnEdge and weightsOnEdge (maxEdges2, nEdges+1) are
// [iEdge*me2 + i]; edgesOnCell and edgeSignOnCell (maxEdges, nCells+1) are
// [iCell*me + i]. Connectivity is 1-based. One thread per edge or cell;
// the SSH corrector's cell loop reads the SSH it is overwriting and runs on
// one thread, in the Fortran's cell order.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
__device__ __forceinline__ rk bt_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
__device__ __forceinline__ rk bt_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
#endif
__device__ __forceinline__ int bt_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// unsplit_explicit: the barotropic velocity zero, the new velocity the
// baroclinic one and the transport velocity masked; edges 1..nEdges
extern "C" __global__ void ocn_btr_unsplit_v842(
    rk *btrNew, rk *normalVelocityNew, rk *normalTransportVelocity, const rk *bclNew, const rk *bolus,
    const int *edgeMask, int nEdges, int nl)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    btrNew[e] = C(0.0);
    for (int k = 0; k < nl; ++k) {
        const int i = e * nl + k;
        normalVelocityNew[i] = bclNew[i];
        normalTransportVelocity[i] = M((rk)edgeMask[i], A(bclNew[i], bolus[i]));
    }
}

// the subcycle's start: the SSH and barotropic velocity copied into the
// old subcycle level, the running sum started, the thickness flux zeroed
extern "C" __global__ void ocn_btr_init_cells_v842(rk *sshSubCur, const rk *sshCur, int nCells)
{
    const int c = bt_thread();
    if (c < nCells) sshSubCur[c] = sshCur[c];
}

extern "C" __global__ void ocn_btr_init_edges_v842(
    rk *uSubCur, rk *btrNew, rk *thicknessFlux, rk *forcing, const rk *btrCur, int nEdges, int filterBtrMode)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    if (filterBtrMode) forcing[e] = C(0.0);
    uSubCur[e] = btrCur[e];
    btrNew[e] = btrCur[e];
    thicknessFlux[e] = C(0.0);
}

// the velocity predictor (corrector = 0: uPerp the old level, the SSH the
// old level's) and corrector (corrector = 1: uPerp btrvel_temp, the SSH
// (1 - gam2)*old + gam2*new); edges 1..nEdges
extern "C" __global__ void ocn_btr_velocity_v842(
    rk *uNew, const rk *uCur, const rk *uPerp, const rk *sshCur, const rk *sshNew, const rk *forcing,
    const rk *fEdge, const rk *dcEdge, const rk *weightsOnEdge, const int *edgesOnEdge, const int *nEdgesOnEdge,
    const int *cellsOnEdge, const int *edgeMask, int nEdges, int nl, int me2, int corrector,
    rk dtn, rk gravity, rk omg2, rk g2)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    const int tempMask = edgeMask[e * nl];
    const int cell1 = cellsOnEdge[e * 2] - 1, cell2 = cellsOnEdge[e * 2 + 1] - 1;
    rk coriolis = C(0.0);
    for (int i = 0; i < nEdgesOnEdge[e]; ++i) {
        const int eoe = edgesOnEdge[e * me2 + i] - 1;
        coriolis = A(coriolis, M(M(weightsOnEdge[e * me2 + i], uPerp[eoe]), fEdge[eoe]));
    }
    rk ssh1, ssh2;
    if (corrector) {
        ssh1 = A(M(omg2, sshCur[cell1]), M(g2, sshNew[cell1]));
        ssh2 = A(M(omg2, sshCur[cell2]), M(g2, sshNew[cell2]));
    } else {
        ssh1 = sshCur[cell1];
        ssh2 = sshCur[cell2];
    }
    uNew[e] = M((rk)tempMask,
                A(uCur[e], M(dtn, A(S(coriolis, D(M(gravity, S(ssh2, ssh1)), dcEdge[e])), forcing[e]))));
}

// the corrector's copy of the new velocity, edges 1..nEdges+1
extern "C" __global__ void ocn_btr_copy_v842(rk *temp, const rk *uNew, int n)
{
    const int e = bt_thread();
    if (e < n) temp[e] = uNew[e];
}

// an edge's thickness flux: the edge SSH from the old level (weighted = 0)
// or the gam2 mix of old and new (weighted = 1)
__device__ __forceinline__ rk bt_flux(
    int e, const rk *uCur, const rk *uNew, const rk *sshCur, const rk *sshNew, const rk *bottomDepth,
    const int *cellsOnEdge, int weighted, rk omg2, rk g2, rk omgv, rk gv)
{
    const int cell1 = cellsOnEdge[e * 2] - 1, cell2 = cellsOnEdge[e * 2 + 1] - 1;
    rk sshEdge;
    if (weighted) {
        const rk ssh1 = A(M(omg2, sshCur[cell1]), M(g2, sshNew[cell1]));
        const rk ssh2 = A(M(omg2, sshCur[cell2]), M(g2, sshNew[cell2]));
        sshEdge = M(C(0.5), A(ssh1, ssh2));
    } else {
        sshEdge = M(C(0.5), A(sshCur[cell1], sshCur[cell2]));
    }
    const rk thicknessSum = A(sshEdge, bt_min(bottomDepth[cell1], bottomDepth[cell2]));
    return M(A(M(omgv, uCur[e]), M(gv, uNew[e])), thicknessSum);
}

__device__ __forceinline__ void bt_ssh_cell(
    int c, rk *sshTend, rk *sshNew, const rk *sshCur, const rk *uCur, const rk *uNew, const rk *bottomDepth,
    const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *cellsOnEdge, int me, int weighted, rk omg2, rk g2, rk omgv, rk gv,
    rk dtn)
{
    sshTend[c] = C(0.0);
    for (int i = 0; i < nEdgesOnCell[c]; ++i) {
        const int e = edgesOnCell[c * me + i] - 1;
        const rk flux = bt_flux(e, uCur, uNew, sshCur, sshNew, bottomDepth, cellsOnEdge, weighted, omg2, g2,
                                omgv, gv);
        sshTend[c] = A(sshTend[c], M(M((rk)edgeSignOnCell[c * me + i], flux), dvEdge[e]));
    }
    sshNew[c] = A(sshCur[c], D(M(dtn, sshTend[c]), areaCell[c]));
}

// the SSH predictor, cells 1..nCells (reads only the old SSH)
extern "C" __global__ void ocn_btr_ssh_predict_v842(
    rk *sshTend, rk *sshNew, const rk *sshCur, const rk *uCur, const rk *uNew, const rk *bottomDepth,
    const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *cellsOnEdge, int nCells, int me, rk g1, rk omg1, rk dtn)
{
    const int c = bt_thread();
    if (c >= nCells) return;
    bt_ssh_cell(c, sshTend, sshNew, sshCur, uCur, uNew, bottomDepth, dvEdge, areaCell, nEdgesOnCell,
                edgesOnCell, edgeSignOnCell, cellsOnEdge, me, 0, C(0.0), C(0.0), omg1, g1, dtn);
}

// the SSH corrector, cells 1..nCells in order on one thread: a cell's SSH
// mix reads its neighbours' new SSH as far as the loop has written it
extern "C" __global__ void ocn_btr_ssh_correct_v842(
    rk *sshTend, rk *sshNew, const rk *sshCur, const rk *uCur, const rk *uNew, const rk *bottomDepth,
    const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *cellsOnEdge, int nCells, int me, rk omg2, rk g2, rk omg3, rk g3,
    rk dtn)
{
    if (bt_thread() != 0) return;
    for (int c = 0; c < nCells; ++c)
        bt_ssh_cell(c, sshTend, sshNew, sshCur, uCur, uNew, bottomDepth, dvEdge, areaCell, nEdgesOnCell,
                    edgesOnCell, edgeSignOnCell, cellsOnEdge, me, 1, omg2, g2, omg3, g3, dtn);
}

// the thickness flux summed over the subcycles, edges 1..nEdges
extern "C" __global__ void ocn_btr_flux_sum_v842(
    rk *thicknessFlux, const rk *uCur, const rk *uNew, const rk *sshCur, const rk *sshNew, const rk *bottomDepth,
    const int *cellsOnEdge, int nEdges, int weighted, rk omg2, rk g2, rk omgv, rk gv)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    thicknessFlux[e] = A(thicknessFlux[e],
                         bt_flux(e, uCur, uNew, sshCur, sshNew, bottomDepth, cellsOnEdge, weighted, omg2, g2,
                                 omgv, gv));
}

// the running sum of the barotropic velocity, edges 1..nEdges
extern "C" __global__ void ocn_btr_accumulate_v842(rk *btrNew, const rk *uSubNew, int nEdges)
{
    const int e = bt_thread();
    if (e < nEdges) btrNew[e] = A(btrNew[e], uSubNew[e]);
}

// the sums normalised, edges 1..nEdgesArray(1)
extern "C" __global__ void ocn_btr_normalize_v842(rk *thicknessFlux, rk *btrNew, int nEdges, int nSum)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    thicknessFlux[e] = D(thicknessFlux[e], (rk)nSum);
    btrNew[e] = D(btrNew[e], (rk)(nSum + 1));
}

// the transport velocity with the correction (F - sum h u)/sum h, edges
// 1..nEdgesArray(config_num_halos)
extern "C" __global__ void ocn_btr_transport_v842(
    rk *normalTransportVelocity, const rk *btrNew, const rk *bclNew, const rk *bolus,
    const rk *layerThicknessEdge, const rk *thicknessFlux, const int *maxLevelEdgeTop, const int *edgeMask,
    int nEdges, int nl, int useVelocityCorrection)
{
    const int e = bt_thread();
    if (e >= nEdges) return;
    const int b = e * nl;
    rk fluxSum = M(layerThicknessEdge[b], A(A(btrNew[e], bclNew[b]), bolus[b]));
    rk thicknessSum = layerThicknessEdge[b];
    for (int k = 1; k < maxLevelEdgeTop[e]; ++k) {
        fluxSum = A(fluxSum, M(layerThicknessEdge[b + k], A(A(btrNew[e], bclNew[b + k]), bolus[b + k])));
        thicknessSum = A(thicknessSum, layerThicknessEdge[b + k]);
    }
    const rk correction = M((rk)useVelocityCorrection, D(S(thicknessFlux[e], fluxSum), thicknessSum));
    for (int k = 0; k < nl; ++k)
        normalTransportVelocity[b + k] = M((rk)edgeMask[b + k],
                                           A(A(A(btrNew[e], bclNew[b + k]), bolus[b + k]), correction));
}
