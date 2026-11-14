// MPAS-Ocean v8.4.2: the forward mode's cold start.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F
//     (ocn_time_integration_split_init: the barotropic and baroclinic split
//     of the initial velocity)
//   src/core_ocean/shared/mpas_ocn_init_routines.F (ocn_init_routines_block:
//     the missing cell's and the land levels' sentinels, the transport velocity)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off). Layout as
// split_step_v842.cu; connectivity is 1-based.

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
__device__ __forceinline__ int fw_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// unsplit_explicit's cold start, edges 1..nEdges: no barotropic velocity
extern "C" __global__ void ocn_fw_unsplit_init_v842(rk *btr, rk *bcl, const rk *u, int nEdges, int nl)
{
    const int e = fw_thread();
    if (e >= nEdges) return;
    btr[e] = C(0.0);
    for (int k = 0; k < nl; ++k) bcl[e * nl + k] = u[e * nl + k];
}

// config_filter_btr_mode's cold start, cells 1..nCells: the top layer the reference's
extern "C" __global__ void ocn_fw_filter_thickness_v842(rk *h, rk top, int nCells, int nl)
{
    const int c = fw_thread();
    if (c < nCells) h[c * nl] = top;
}

// split_explicit's cold start, edges 1..nEdges: the barotropic velocity the
// thickness-weighted mean over the edge's levels, the baroclinic the rest,
// zero below maxLevelEdgeTop
extern "C" __global__ void ocn_fw_split_init_v842(
    rk *btr, rk *bcl, rk *u, const rk *h, const int *cellsOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int e = fw_thread();
    if (e >= nEdges) return;
    const int c1 = cellsOnEdge[e * 2] - 1, c2 = cellsOnEdge[e * 2 + 1] - 1;
    const int top = maxLevelEdgeTop[e];
    rk he = M(C(0.5), A(h[c1 * nl], h[c2 * nl]));
    rk flux = M(he, u[e * nl]);
    rk thick = he;
    for (int k = 1; k < top; ++k) {
        he = M(C(0.5), A(h[c1 * nl + k], h[c2 * nl + k]));
        flux = A(flux, M(he, u[e * nl + k]));
        thick = A(thick, he);
    }
    btr[e] = D(flux, thick);
    for (int k = 0; k < top; ++k) bcl[e * nl + k] = S(u[e * nl + k], btr[e]);
    for (int k = top; k < nl; ++k) {
        bcl[e * nl + k] = C(0.0);
        u[e * nl + k] = C(0.0);
    }
}

// ocn_init_routines_block, edges 1..nEdges: the velocity zero between the
// edge's top and bottom levels, the sentinel below, and the transport velocity
extern "C" __global__ void ocn_fw_init_edges_v842(
    rk *u, rk *transport, const int *maxLevelEdgeTop, const int *maxLevelEdgeBot, int nEdges, int nl)
{
    const int e = fw_thread();
    if (e >= nEdges) return;
    for (int k = maxLevelEdgeTop[e]; k < maxLevelEdgeBot[e]; ++k) u[e * nl + k] = C(0.0);
    for (int k = maxLevelEdgeBot[e]; k < nl; ++k) u[e * nl + k] = C(-1.0e34);
    for (int k = 0; k < nl; ++k) transport[e * nl + k] = u[e * nl + k];
}

// ocn_init_routines_block, cells 1..nCells: the tracers' sentinel below maxLevelCell
extern "C" __global__ void ocn_fw_init_cells_v842(rk *tracers, const int *maxLevelCell, int nCells, int nl, int nt)
{
    const int c = fw_thread();
    if (c >= nCells) return;
    for (int k = maxLevelCell[c]; k < nl; ++k)
        for (int i = 0; i < nt; ++i) tracers[(c * nl + k) * nt + i] = C(-1.0e34);
}
