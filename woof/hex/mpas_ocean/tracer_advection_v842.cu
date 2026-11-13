// MPAS-Ocean v8.4.2: monotone (flux-corrected) tracer advection.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_tracer_advection_mono.F
//     (ocn_tracer_advection_mono_tend, ocn_tracer_advection_mono_init)
//   src/core_ocean/shared/mpas_ocn_tracer_advection.F (ocn_tracer_advection_tend, _init)
//   src/operators/mpas_tracer_advection_helpers.F
//     (mpas_tracer_advection_vflux4, mpas_tracer_advection_vflux3)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); an integer times a real
// is the integer converted, then multiplied, and SIGN(1, x) copies x's sign
// bit. Fortran MAX/MIN are chosen on the words as integer selects (MAX(a, b,
// c) is MAX(MAX(a, b), c)), so no compiler turns them into IEEE min/max. A
// tie of signed zeros, and SIGN of a zero flux, only ever decide the sign of
// a zero term added to others here, so no output word depends on either.
//
// One kernel per loop nest of the Fortran, one tracer at a time; a thread
// owns a (cell or edge, level) word where the Fortran sums over neighbours
// only, and a column where it recurs in the vertical.
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nl + k], a(nVertLevelsP1, ...)
// a[iCell*(nl+1) + k]; tracers(nTracers, nVertLevels, nCells+1) is
// t[(iCell*nl + k)*nTracers + i]; a(n, nEdges+1) of coefficients is
// a[iEdge*n + i]. Connectivity is 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
__device__ __forceinline__ rk fmaxw(rk a, rk b) {
    return __longlong_as_double(a > b ? __double_as_longlong(a) : __double_as_longlong(b));
}
__device__ __forceinline__ rk fminw(rk a, rk b) {
    return __longlong_as_double(a < b ? __double_as_longlong(a) : __double_as_longlong(b));
}
__device__ __forceinline__ rk sign1(rk x) {
    return __longlong_as_double((__double_as_longlong(x) & (long long)0x8000000000000000ULL) |
                                __double_as_longlong(1.0));
}
__device__ __forceinline__ rk absw(rk x) { return fabs(x); }
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
__device__ __forceinline__ rk fmaxw(rk a, rk b) {
    return __int_as_float(a > b ? __float_as_int(a) : __float_as_int(b));
}
__device__ __forceinline__ rk fminw(rk a, rk b) {
    return __int_as_float(a < b ? __float_as_int(a) : __float_as_int(b));
}
__device__ __forceinline__ rk sign1(rk x) {
    return __int_as_float((__float_as_int(x) & (int)0x80000000u) | __float_as_int(1.0f));
}
__device__ __forceinline__ rk absw(rk x) { return fabsf(x); }
#endif

#define EPS C(1.e-10)
__device__ __forceinline__ int adv_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// mpas_tracer_advection_vflux4 / _vflux3
__device__ __forceinline__ rk vflux4(rk q_im2, rk q_im1, rk q_i, rk q_ip1, rk w) {
    return D(M(w, S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2))), C(12.0));
}
__device__ __forceinline__ rk vflux3(rk q_im2, rk q_im1, rk q_i, rk q_ip1, rk w, rk coef) {
    return D(S(M(w, S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2))),
               M(M(coef, absw(w)), S(S(q_ip1, q_im2), M(C(3.0), S(q_i, q_im1))))), C(12.0));
}

// the provisional thickness after horizontal transport, its inverse, and the
// inverse of the new thickness; cells 1..nCells, levels to maxLevelCell
extern "C" __global__ void ocn_adv_mono_thickness_v842(
    rk *hProv, rk *hProvInv, rk *hNewInv, const rk *layerThickness, const rk *normalThicknessFlux, const rk *w,
    const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell, const int *edgeSignOnCell,
    const int *maxLevelCell, int nCells, int nl, int maxEdges, rk dt)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k >= maxLevelCell[iCell]) return;
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    rk h = layerThickness[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        h = A(h, M(M(M(M(dt, invAreaCell1), dvEdge[iEdge]), (rk)edgeSignOnCell[iCell * maxEdges + i]),
                   normalThicknessFlux[iEdge * nl + k]));
    }
    hProv[t] = h;
    hProvInv[t] = D(C(1.0), h);
    hNewInv[t] = D(C(1.0), A(S(h, M(dt, w[iCell * (nl + 1) + k])), M(dt, w[iCell * (nl + 1) + k + 1])));
}

// tracer_cur from the tracer, the work arrays zeroed; cells 1..nCells, every level
extern "C" __global__ void ocn_adv_mono_start_v842(
    rk *tracerCur, rk *workTend, rk *fluxIncoming, rk *fluxOutgoing, const rk *tracers, int nCells, int nl,
    int nTracers, int iTracer)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    tracerCur[t] = tracers[t * nTracers + iTracer];
    workTend[t] = C(0.0);
    fluxIncoming[t] = C(0.0);
    fluxOutgoing[t] = C(0.0);
}

// the horizontal neighbours' extrema; cells 1..nCellsArray(2)
extern "C" __global__ void ocn_adv_mono_horizontal_bounds_v842(
    rk *tracerMin, rk *tracerMax, const rk *tracerCur, const int *nEdgesOnCell, const int *cellsOnCell,
    const int *maxLevelCell, int nCells, int nl, int maxEdges)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    const int top = maxLevelCell[iCell];
    if (k >= top) return;
    rk tmax = tracerCur[t], tmin = tracerCur[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int neighbour = cellsOnCell[iCell * maxEdges + i] - 1;
        const int levels = top < maxLevelCell[neighbour] ? top : maxLevelCell[neighbour];
        if (k < levels) {
            tmax = fmaxw(tmax, tracerCur[neighbour * nl + k]);
            tmin = fminw(tmin, tracerCur[neighbour * nl + k]);
        }
    }
    tracerMax[t] = tmax;
    tracerMin[t] = tmin;
}

// the high-order horizontal flux less the upwind flux; edges 1..nEdgesArray(3)
extern "C" __global__ void ocn_adv_mono_horizontal_flux_v842(
    rk *highOrderFlux, const rk *tracerCur, const rk *normalThicknessFlux, const rk *advCoefs, const rk *advCoefs3rd,
    const int *nAdvCellsForEdge, const int *advCellsForEdge, const int *highOrderAdvectionMask, const rk *dvEdge,
    const int *cellsOnEdge, const int *maxLevelCell, const int *maxLevelEdgeTop, int nEdges, int nl,
    int nAdvectionCells, rk coef3rd)
{
    const int t = adv_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    const rk F = normalThicknessFlux[t];
    const rk mask = (rk)highOrderAdvectionMask[t];
    rk flux = C(0.0);
    for (int i = 0; i < nAdvCellsForEdge[iEdge]; ++i) {
        const int iCell = advCellsForEdge[iEdge * nAdvectionCells + i] - 1;
        if (k < maxLevelCell[iCell]) {
            rk weight = M(mask, A(advCoefs[iEdge * nAdvectionCells + i],
                                  M(M(coef3rd, sign1(F)), advCoefs3rd[iEdge * nAdvectionCells + i])));
            weight = M(F, weight);
            flux = A(flux, M(weight, tracerCur[iCell * nl + k]));
        }
    }
    if (k < maxLevelEdgeTop[iEdge]) {
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        const rk weight = M(M((rk)((highOrderAdvectionMask[t] + 1) & 1), M(dvEdge[iEdge], C(0.5))), F);
        flux = A(flux, M(weight, A(tracerCur[cell1 * nl + k], tracerCur[cell2 * nl + k])));
        const rk upwind = M(dvEdge[iEdge], A(M(fmaxw(C(0.0), F), tracerCur[cell1 * nl + k]),
                                             M(fminw(C(0.0), F), tracerCur[cell2 * nl + k])));
        flux = S(flux, upwind);
    }
    highOrderFlux[iEdge * (nl + 1) + k] = flux;
}

// the upwind tendency, the incoming and outgoing antidiffusive fluxes, then
// the horizontal limiter's factors; cells 1..nCellsArray(2)
extern "C" __global__ void ocn_adv_mono_horizontal_limiter_v842(
    rk *workTend, rk *fluxIncoming, rk *fluxOutgoing, const rk *highOrderFlux, const rk *tracerCur,
    const rk *tracerMin, const rk *tracerMax, const rk *layerThickness, const rk *hProvInv,
    const rk *normalThicknessFlux, const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell,
    const int *edgesOnCell, const int *edgeSignOnCell, const int *cellsOnEdge, const int *maxLevelEdgeTop,
    const int *maxLevelCell, int nCells, int nl, int maxEdges, rk dt)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    rk work = workTend[t], fin = fluxIncoming[t], fout = fluxOutgoing[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeTop[iEdge]) continue;
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        const rk signedFactor = M((rk)edgeSignOnCell[iCell * maxEdges + i], invAreaCell1);
        const rk F = normalThicknessFlux[iEdge * nl + k];
        const rk upwind = M(dvEdge[iEdge], A(M(fmaxw(C(0.0), F), tracerCur[cell1 * nl + k]),
                                             M(fminw(C(0.0), F), tracerCur[cell2 * nl + k])));
        const rk hof = highOrderFlux[iEdge * (nl + 1) + k];
        work = A(work, M(signedFactor, upwind));
        fout = A(fout, fminw(C(0.0), M(signedFactor, hof)));
        fin = A(fin, fmaxw(C(0.0), M(signedFactor, hof)));
    }
    if (k < maxLevelCell[iCell]) {
        const rk mass = M(tracerCur[t], layerThickness[t]);
        const rk minNew = M(A(mass, M(dt, A(work, fout))), hProvInv[t]);
        const rk maxNew = M(A(mass, M(dt, A(work, fin))), hProvInv[t]);
        const rk upwindNew = M(A(mass, M(dt, work)), hProvInv[t]);
        rk scale = D(S(tracerMax[t], upwindNew), A(S(maxNew, upwindNew), EPS));
        fin = fminw(C(1.0), fmaxw(C(0.0), scale));
        scale = D(S(upwindNew, tracerMin[t]), A(S(upwindNew, minNew), EPS));
        fout = fminw(C(1.0), fmaxw(C(0.0), scale));
    }
    workTend[t] = work;
    fluxIncoming[t] = fin;
    fluxOutgoing[t] = fout;
}

// the limited antidiffusive flux; edges 1..nEdgesArray(2)
extern "C" __global__ void ocn_adv_mono_horizontal_limit_v842(
    rk *highOrderFlux, const rk *fluxIncoming, const rk *fluxOutgoing, const int *cellsOnEdge,
    const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int t = adv_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    if (k >= maxLevelEdgeTop[iEdge]) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    rk *slot = highOrderFlux + iEdge * (nl + 1) + k;
    const rk flux = *slot;
    *slot = A(M(fmaxw(C(0.0), flux), fminw(fluxOutgoing[cell1 * nl + k], fluxIncoming[cell2 * nl + k])),
              M(fminw(C(0.0), flux), fminw(fluxIncoming[cell1 * nl + k], fluxOutgoing[cell2 * nl + k])));
}

// the horizontal update: tracer_cur and the tendency (and its budget);
// cells 1..nCellsArray(1)
extern "C" __global__ void ocn_adv_mono_horizontal_update_v842(
    rk *tracerCur, rk *workTend, rk *tend, rk *budget, const rk *highOrderFlux, const rk *layerThickness,
    const rk *hProvInv, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, const int *maxLevelEdgeTop, const int *maxLevelCell, int nCells, int nl,
    int maxEdges, int nTracers, int iTracer, int withBudget, rk dt)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    rk work = workTend[t];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeTop[iEdge]) continue;
        const rk signedFactor = M(invAreaCell1, (rk)edgeSignOnCell[iCell * maxEdges + i]);
        work = A(work, M(signedFactor, highOrderFlux[iEdge * (nl + 1) + k]));
    }
    workTend[t] = work;
    if (k < maxLevelCell[iCell]) {
        tracerCur[t] = M(A(M(tracerCur[t], layerThickness[t]), M(dt, work)), hProvInv[t]);
        tend[t * nTracers + iTracer] = A(tend[t * nTracers + iTracer], work);
        if (withBudget) budget[t * nTracers + iTracer] = work;
    }
}

// the vertical high-order flux less the upwind flux, the vertical extrema and
// the antidiffusive fluxes in and out of each layer; one thread per column of
// cells 1..nCellsArray(2). highOrderFlux (nVertLevels+1 per cell) and
// workTend start at zero.
extern "C" __global__ void ocn_adv_mono_vertical_flux_v842(
    rk *highOrderFlux, rk *workTend, rk *tracerMin, rk *tracerMax, rk *fluxIncoming, rk *fluxOutgoing,
    const rk *tracerCur, const rk *hProv, const rk *w, const int *maxLevelCell, int nCells, int nl, int vertOrder,
    rk coef3rd)
{
    const int iCell = adv_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell], np1 = nl + 1;
    const rk *cur = tracerCur + iCell * nl, *hp = hProv + iCell * nl, *ww = w + iCell * np1;
    rk *hof = highOrderFlux + iCell * np1, *work = workTend + iCell * nl;
    rk *tmax = tracerMax + iCell * nl, *tmin = tracerMin + iCell * nl;
    tmax[0] = fmaxw(cur[0], cur[1]);
    tmin[0] = fminw(cur[0], cur[1]);
    int k = top < 2 ? top : 2;
    if (k < 1) k = 1;
    if (k >= 2) {
        const int kk = k - 1;   // 0-based level k
        const rk sum = A(hp[kk], hp[kk - 1]);
        const rk weightK = D(hp[kk - 1], sum), weightKm1 = D(hp[kk], sum);
        hof[kk] = M(ww[kk], A(M(weightK, cur[kk]), M(weightKm1, cur[kk - 1])));
        tmax[kk] = fmaxw(fmaxw(cur[kk - 1], cur[kk]), cur[kk + 1]);
        tmin[kk] = fminw(fminw(cur[kk - 1], cur[kk]), cur[kk + 1]);
    }
    for (int kf = 3; kf <= top - 1; ++kf) {
        const int kk = kf - 1;
        if (vertOrder == 4) {
            hof[kk] = vflux4(cur[kk - 2], cur[kk - 1], cur[kk], cur[kk + 1], ww[kk]);
        } else if (vertOrder == 3) {
            hof[kk] = vflux3(cur[kk - 2], cur[kk - 1], cur[kk], cur[kk + 1], ww[kk], coef3rd);
        } else {
            const rk sum = A(hp[kk], hp[kk - 1]);
            const rk weightK = D(hp[kk - 1], sum), weightKm1 = D(hp[kk], sum);
            hof[kk] = M(ww[kk], A(M(weightK, cur[kk]), M(weightKm1, cur[kk - 1])));
        }
        tmax[kk] = fmaxw(fmaxw(cur[kk - 1], cur[kk]), cur[kk + 1]);
        tmin[kk] = fminw(fminw(cur[kk - 1], cur[kk]), cur[kk + 1]);
    }
    {
        const int kk = top - 1;   // k = max(1, maxLevelCell) >= 2 here
        const rk sum = A(hp[kk], hp[kk - 1]);
        const rk weightK = D(hp[kk - 1], sum), weightKm1 = D(hp[kk], sum);
        hof[kk] = M(ww[kk], A(M(weightK, cur[kk]), M(weightKm1, cur[kk - 1])));
        tmax[kk] = fmaxw(cur[kk], cur[kk - 1]);
        tmin[kk] = fminw(cur[kk], cur[kk - 1]);
    }
    for (int kf = 2; kf <= top; ++kf) {
        const int kk = kf - 1;
        const rk upwind = A(M(fminw(C(0.0), ww[kk]), cur[kk - 1]), M(fmaxw(C(0.0), ww[kk]), cur[kk]));
        work[kk - 1] = A(work[kk - 1], upwind);
        work[kk] = S(work[kk], upwind);
        hof[kk] = S(hof[kk], upwind);
    }
    for (int kk = 0; kk < top; ++kk) {
        fluxIncoming[iCell * nl + kk] = S(fmaxw(C(0.0), hof[kk + 1]), fminw(C(0.0), hof[kk]));
        fluxOutgoing[iCell * nl + kk] = S(fminw(C(0.0), hof[kk + 1]), fmaxw(C(0.0), hof[kk]));
    }
}

// the vertical limiter's factors; cells 1..nCellsArray(2)
extern "C" __global__ void ocn_adv_mono_vertical_limiter_v842(
    rk *fluxIncoming, rk *fluxOutgoing, const rk *workTend, const rk *tracerCur, const rk *tracerMin,
    const rk *tracerMax, const rk *hProv, const rk *hNewInv, const int *maxLevelCell, int nCells, int nl, rk dt)
{
    const int t = adv_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k >= maxLevelCell[iCell]) return;
    const rk mass = M(tracerCur[t], hProv[t]);
    const rk minNew = M(A(mass, M(dt, A(workTend[t], fluxOutgoing[t]))), hNewInv[t]);
    const rk maxNew = M(A(mass, M(dt, A(workTend[t], fluxIncoming[t]))), hNewInv[t]);
    const rk upwindNew = M(A(mass, M(dt, workTend[t])), hNewInv[t]);
    rk scale = D(S(tracerMax[t], upwindNew), A(S(maxNew, upwindNew), EPS));
    fluxIncoming[t] = fminw(C(1.0), fmaxw(C(0.0), scale));
    scale = D(S(upwindNew, tracerMin[t]), A(S(upwindNew, minNew), EPS));
    fluxOutgoing[t] = fminw(C(1.0), fmaxw(C(0.0), scale));
}

// the limited vertical flux and the vertical update of the tendency (and its
// budget); one thread per column of cells 1..nCellsArray(1)
extern "C" __global__ void ocn_adv_mono_vertical_update_v842(
    rk *highOrderFlux, rk *workTend, rk *tend, rk *budget, const rk *fluxIncoming, const rk *fluxOutgoing,
    const int *maxLevelCell, int nCells, int nl, int nTracers, int iTracer, int withBudget)
{
    const int iCell = adv_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell], np1 = nl + 1;
    rk *hof = highOrderFlux + iCell * np1, *work = workTend + iCell * nl;
    const rk *fin = fluxIncoming + iCell * nl, *fout = fluxOutgoing + iCell * nl;
    for (int kk = 1; kk < top; ++kk) {
        const rk flux = hof[kk];
        hof[kk] = A(M(fmaxw(C(0.0), flux), fminw(fout[kk], fin[kk - 1])),
                    M(fminw(C(0.0), flux), fminw(fout[kk - 1], fin[kk])));
    }
    for (int kk = 0; kk < top; ++kk) {
        work[kk] = A(work[kk], S(hof[kk + 1], hof[kk]));
        const int at = (iCell * nl + kk) * nTracers + iTracer;
        tend[at] = A(tend[at], work[kk]);
        if (withBudget) budget[at] = work[kk];
    }
}
