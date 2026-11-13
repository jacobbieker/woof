// MPAS-Ocean v8.4.2: the split-explicit step's own loops (the pieces it calls
// are their own units).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F
//     (ocn_time_integrator_split: the preparation, stage 1's baroclinic
//     iterations, stage 3's updates, the surface fields)
//   src/core_ocean/shared/mpas_ocn_diagnostics.F (ocn_fuperp)
//   src/operators/mpas_vector_reconstruction.F (mpas_reconstruct_1d/_2d)
//   src/core_ocean/shared/mpas_ocn_time_average_coupled.F
//     (ocn_time_average_coupled_accumulate)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); an integer times a real
// is the integer converted, then multiplied; Fortran MAX is chosen on the
// words as an integer select. The sphere's reconstruction calls glibc's
// cosf/sinf at RKIND 4 (glibc_trig_flt32.cuh) and, behind the opt-in
// WOOF_GLIBC239_TRIG64, glibc 2.39's cos/sin at RKIND 8.
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nl + k]; tracers (nTracers,
// nVertLevels, nCells+1) are t[(iCell*nl + k)*nt + i]; vertAleTransportTop
// (nVertLevels+1, nCells+1) is w[iCell*(nl+1) + k]; coeffs_reconstruct
// (3, maxEdges, nCells+1) is c[(iCell*me + i)*3 + j]; (2, nCells+1) is
// a[iCell*2 + j]. Connectivity is 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
__device__ __forceinline__ rk ss_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#ifdef WOOF_GLIBC239_TRIG64
#define SS_COS(x) g239::g239_cos(x)
#define SS_SIN(x) g239::g239_sin(x)
#else
#define SS_COS(x) (x)
#define SS_SIN(x) (x)
#endif
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
__device__ __forceinline__ rk ss_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#define SS_COS(x) glibc_cosf(x)
#define SS_SIN(x) glibc_sinf(x)
#endif
__device__ __forceinline__ int ss_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// the preparation, edges 1..nEdges: the baroclinic velocity, the new level
extern "C" __global__ void ocn_ss_prep_edges_v842(
    rk *bclCur, rk *uNew, rk *bclNew, const rk *uCur, const rk *btrCur, int nEdges, int nl)
{
    const int e = ss_thread();
    if (e >= nEdges) return;
    for (int k = 0; k < nl; ++k) {
        const int i = e * nl + k;
        bclCur[i] = S(uCur[i], btrCur[e]);
        uNew[i] = uCur[i];
        bclNew[i] = bclCur[i];
    }
}

// the preparation, cells 1..nCells: SSH, thickness and tracers copied to the
// new level, the vertical transport zeroed down to maxLevelCell
extern "C" __global__ void ocn_ss_prep_cells_v842(
    rk *sshNew, rk *hNew, rk *w, rk *trNew, const rk *sshCur, const rk *hCur, const rk *trCur,
    const int *maxLevelCell, int nCells, int nl, int nt)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    sshNew[c] = sshCur[c];
    for (int k = 0; k < maxLevelCell[c]; ++k) {
        hNew[c * nl + k] = hCur[c * nl + k];
        w[c * (nl + 1) + k] = C(0.0);
    }
    for (int k = 0; k < maxLevelCell[c]; ++k)
        for (int i = 0; i < nt; ++i) trNew[(c * nl + k) * nt + i] = trCur[(c * nl + k) * nt + i];
}

// ocn_fuperp: f times the baroclinic velocity's perpendicular, edges 1..nEdgesSolve
extern "C" __global__ void ocn_ss_fuperp_v842(
    rk *u, const rk *bcl, const rk *weightsOnEdge, const rk *fEdge, const int *edgesOnEdge,
    const int *nEdgesOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl, int me2)
{
    const int e = ss_thread();
    if (e >= nEdges) return;
    for (int k = 0; k < maxLevelEdgeTop[e]; ++k) {
        rk sum = C(0.0);
        for (int j = 0; j < nEdgesOnEdge[e]; ++j) {
            const int eoe = edgesOnEdge[e * me2 + j] - 1;
            sum = A(sum, M(M(weightsOnEdge[e * me2 + j], bcl[eoe * nl + k]), fEdge[eoe]));
        }
        u[e * nl + k] = sum;
    }
}

// one baroclinic iteration on the linear Coriolis term, edges 1..nEdgesArray(1):
// the predicted baroclinic velocity, its thickness-weighted mean as the
// barotropic forcing, the new baroclinic velocity the average of old and predicted
extern "C" __global__ void ocn_ss_bcl_v842(
    rk *bclNew, rk *forcing, rk *uTemp, const rk *bclCur, const rk *tend, const rk *uNew, const rk *sshNew,
    const rk *layerThicknessEdge, const rk *dcEdge, const int *cellsOnEdge, const int *maxLevelEdgeTop,
    int nEdges, int nl, int split, rk dt, rk gravity)
{
    const int e = ss_thread();
    if (e >= nEdges) return;
    const int cell1 = cellsOnEdge[e * 2] - 1, cell2 = cellsOnEdge[e * 2 + 1] - 1;
    rk *ut = uTemp + (size_t)e * nl;
    const int b = e * nl, top = maxLevelEdgeTop[e];
    for (int k = 0; k < nl; ++k) ut[k] = C(0.0);
    for (int k = 0; k < top; ++k)
        ut[k] = A(bclCur[b + k],
                  M(dt, A(A(tend[b + k], uNew[b + k]),
                          D(M(M((rk)split, gravity), S(sshNew[cell2], sshNew[cell1])), dcEdge[e]))));
    rk sum = M(layerThicknessEdge[b], ut[0]);
    rk thick = layerThicknessEdge[b];
    for (int k = 1; k < top; ++k) {
        sum = A(sum, M(layerThicknessEdge[b + k], ut[k]));
        thick = A(thick, layerThicknessEdge[b + k]);
    }
    forcing[e] = D(D(M((rk)split, sum), thick), dt);
    for (int k = 0; k < top; ++k)
        bclNew[b + k] = M(C(0.5), S(A(bclCur[b + k], ut[k]), M(dt, forcing[e])));
}

// stage 3 before the last iteration, cells 1..nCells: thickness and the
// active tracers half-way to their predictions
extern "C" __global__ void ocn_ss_mid_cells_v842(
    rk *hNew, rk *trNew, const rk *hCur, const rk *hTend, const rk *trCur, const rk *trTend,
    const int *maxLevelCell, int nCells, int nl, int nt, int iStart, int iEnd, rk dt)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    for (int k = 0; k < maxLevelCell[c]; ++k) {
        const int j = c * nl + k;
        const rk tempH = A(hCur[j], M(dt, hTend[j]));
        hNew[j] = M(C(0.5), A(hCur[j], tempH));
        for (int i = iStart - 1; i < iEnd; ++i) {
            const int t = j * nt + i;
            const rk temp = D(A(M(trCur[t], hCur[j]), M(dt, trTend[t])), tempH);
            trNew[t] = M(C(0.5), A(trCur[t], temp));
        }
    }
}

// stage 3 before the last iteration, edges 1..nEdges: the new velocity the
// masked sum of barotropic and baroclinic
extern "C" __global__ void ocn_ss_mid_edges_v842(
    rk *uNew, const rk *btrNew, const rk *bclNew, const int *edgeMask, int nEdges, int nl)
{
    const int e = ss_thread();
    if (e >= nEdges) return;
    for (int k = 0; k < nl; ++k)
        uNew[e * nl + k] = M((rk)edgeMask[e * nl + k], A(btrNew[e], bclNew[e * nl + k]));
}

// the last iteration, cells 1..nCells: the new thickness
extern "C" __global__ void ocn_ss_final_thickness_v842(
    rk *hNew, const rk *hCur, const rk *hTend, const int *maxLevelCell, int nCells, int nl, rk dt)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    for (int k = 0; k < maxLevelCell[c]; ++k)
        hNew[c * nl + k] = A(hCur[c * nl + k], M(dt, hTend[c * nl + k]));
}

// the active-tracer budgets per unit thickness: a (n, nVertLevels, nCells+1)
// array divided by the new thickness down to maxLevelCell (n = 1 for the
// short-wave tendency)
extern "C" __global__ void ocn_ss_budget_v842(
    rk *a, const rk *hNew, const int *maxLevelCell, int nCells, int nl, int n)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    for (int k = 0; k < maxLevelCell[c]; ++k)
        for (int i = 0; i < n; ++i) a[(c * nl + k) * n + i] = D(a[(c * nl + k) * n + i], hNew[c * nl + k]);
}

// the last iteration's tracers, cells 1..nCells, with the salinity floor
extern "C" __global__ void ocn_ss_final_tracers_v842(
    rk *trNew, const rk *trCur, const rk *trTend, const rk *hCur, const rk *hNew, const int *maxLevelCell,
    int nCells, int nl, int nt, int iSalinity, rk dt)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    for (int k = 0; k < maxLevelCell[c]; ++k) {
        const int j = c * nl + k;
        for (int i = 0; i < nt; ++i)
            trNew[j * nt + i] = D(A(M(trCur[j * nt + i], hCur[j]), M(dt, trTend[j * nt + i])), hNew[j]);
    }
    if (iSalinity > 0)
        for (int k = 0; k < maxLevelCell[c]; ++k) {
            const int t = (c * nl + k) * nt + iSalinity - 1;
            trNew[t] = ss_max(C(0.001), trNew[t]);
        }
}

// the last iteration's velocity, edges 1..nEdges, down to maxLevelEdgeTop
extern "C" __global__ void ocn_ss_final_velocity_v842(
    rk *uNew, const rk *btrNew, const rk *bclNew, const rk *bclCur, const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int e = ss_thread();
    if (e >= nEdges) return;
    for (int k = 0; k < maxLevelEdgeTop[e]; ++k) {
        const int i = e * nl + k;
        uNew[i] = S(A(btrNew[e], M(C(2.0), bclNew[i])), bclCur[i]);
    }
}

// mpas_reconstruct (2-D, or 1-D with nl = 1): the cell vector from the edge
// normals, then its zonal and meridional components, cells 1..nCells
extern "C" __global__ void ocn_ss_reconstruct_v842(
    rk *x, rk *y, rk *z, rk *zonal, rk *meridional, const rk *u, const rk *coeffs, const int *nEdgesOnCell,
    const int *edgesOnCell, const rk *latCell, const rk *lonCell, int nCells, int nl, int me, int onSphere)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    const int b = c * nl;
    for (int k = 0; k < nl; ++k) {
        x[b + k] = C(0.0);
        y[b + k] = C(0.0);
        z[b + k] = C(0.0);
    }
    for (int i = 0; i < nEdgesOnCell[c]; ++i) {
        const int e = edgesOnCell[c * me + i] - 1;
        const rk *cf = coeffs + (c * me + i) * 3;
        for (int k = 0; k < nl; ++k) {
            x[b + k] = A(x[b + k], M(cf[0], u[e * nl + k]));
            y[b + k] = A(y[b + k], M(cf[1], u[e * nl + k]));
            z[b + k] = A(z[b + k], M(cf[2], u[e * nl + k]));
        }
    }
    if (onSphere) {
        const rk clat = SS_COS(latCell[c]), slat = SS_SIN(latCell[c]);
        const rk clon = SS_COS(lonCell[c]), slon = SS_SIN(lonCell[c]);
        for (int k = 0; k < nl; ++k) {
            zonal[b + k] = A(-M(x[b + k], slon), M(y[b + k], clon));
            meridional[b + k] = A(-M(A(M(x[b + k], clon), M(y[b + k], slon)), slat), M(z[b + k], clat));
        }
    } else {
        for (int k = 0; k < nl; ++k) {
            zonal[b + k] = x[b + k];
            meridional[b + k] = y[b + k];
        }
    }
}

// the coupler's surface fields, cells 1..nCells
extern "C" __global__ void ocn_ss_surface_v842(
    rk *surfaceVelocity, rk *sshGradient, const rk *velocityZonal, const rk *velocityMeridional,
    const rk *gradZonal, const rk *gradMeridional, int nCells, int nl, int iSVZ, int iSVM, int iSGZ, int iSGM)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    surfaceVelocity[c * 2 + iSVZ - 1] = velocityZonal[c * nl];
    surfaceVelocity[c * 2 + iSVM - 1] = velocityMeridional[c * nl];
    sshGradient[c * 2 + iSGZ - 1] = gradZonal[c];
    sshGradient[c * 2 + iSGM - 1] = gradMeridional[c];
}

// ocn_time_average_coupled_accumulate's running means, cells 1..nCells
extern "C" __global__ void ocn_ss_time_average_v842(
    rk *avgTracers, rk *avgSSHGradient, rk *avgSurfaceVelocity, const rk *tracersSurfaceValue,
    const rk *gradZonal, const rk *gradMeridional, const rk *surfaceVelocity, int nCells, int nt, int iAvgT,
    int iAvgSGZ, int iAvgSGM, int nAccumulated, rk t0Kelvin)
{
    const int c = ss_thread();
    if (c >= nCells) return;
    const rk n = (rk)nAccumulated, n1 = (rk)(nAccumulated + 1);
    for (int i = 0; i < nt; ++i)
        avgTracers[c * nt + i] = A(M(avgTracers[c * nt + i], n), tracersSurfaceValue[c * nt + i]);
    avgTracers[c * nt + iAvgT - 1] = A(avgTracers[c * nt + iAvgT - 1], t0Kelvin);
    for (int i = 0; i < nt; ++i) avgTracers[c * nt + i] = D(avgTracers[c * nt + i], n1);
    avgSSHGradient[c * 2 + iAvgSGZ - 1] = D(A(M(avgSSHGradient[c * 2 + iAvgSGZ - 1], n), gradZonal[c]), n1);
    avgSSHGradient[c * 2 + iAvgSGM - 1] = D(A(M(avgSSHGradient[c * 2 + iAvgSGM - 1], n), gradMeridional[c]), n1);
    for (int j = 0; j < 2; ++j)
        avgSurfaceVelocity[c * 2 + j] = D(A(M(avgSurfaceVelocity[c * 2 + j], n), surfaceVelocity[c * 2 + j]), n1);
}
