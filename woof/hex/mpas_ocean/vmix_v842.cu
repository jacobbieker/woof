// MPAS-Ocean v8.4.2: vertical mixing, the implicit step.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_vmix.F (ocn_vmix_implicit, ocn_vmix_coefs,
//     ocn_vel_vmix_tend_implicit, ocn_tracer_vmix_tend_implicit,
//     tridiagonal_solve, tridiagonal_solve_mult, ocn_vmix_init)
//   src/core_ocean/shared/mpas_ocn_vmix_coefs_const.F
//   src/core_ocean/shared/mpas_ocn_vmix_coefs_rich.F
//     (ocn_vmix_coefs_rich_build, ocn_vmix_get_rich_numbers, the coefficients)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); x**2 is x*x, sqrt the
// correctly rounded instruction, and an integer literal (1, 2) the same
// value as a real. The tridiagonal solves run one thread per column.
//
// Layout: a(nVertLevels[P1], n+1) is a[i*nl(+1) + k]; tracers(nTracers,
// nVertLevels, nCells+1) is t[(iCell*nl + k)*nTracers + i]. Connectivity is
// 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define C(x) x##f
#endif
#define SENTINEL_DEFAULT ((rk)-1e34f)   // a default (single precision) literal in v8.4.2
__device__ __forceinline__ int vm_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// a(k, i) += value for the rows 1..n, every level (ocn_vmix_coefs_const)
extern "C" __global__ void ocn_vmix_add_const_v842(rk *field, int n, int levels, rk value)
{
    const int t = vm_thread();
    if (t >= n * levels) return;
    field[t] = A(field[t], value);
}

// ocn_vmix_get_rich_numbers' first loop: ddensityTopOfCell, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_vmix_ddensity_cell_v842(
    rk *ddensityTopOfCell, const rk *displacedDensity, const rk *density, const int *maxLevelCell, int nCells, int nl)
{
    const int t = vm_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelCell[iCell]) return;
    ddensityTopOfCell[iCell * (nl + 1) + k] = S(displacedDensity[iCell * nl + k - 1], density[iCell * nl + k]);
}

// the edge values: ddensityTopOfEdge, du2TopOfEdge, edges 1..nEdgesArray(2)
extern "C" __global__ void ocn_vmix_rich_edge_v842(
    rk *ddensityTopOfEdge, rk *du2TopOfEdge, const rk *ddensityTopOfCell, const rk *normalVelocity,
    const int *cellsOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl)
{
    const int t = vm_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelEdgeTop[iEdge]) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    ddensityTopOfEdge[iEdge * (nl + 1) + k] = D(A(ddensityTopOfCell[cell1 * (nl + 1) + k],
                                                  ddensityTopOfCell[cell2 * (nl + 1) + k]), C(2.0));
    const rk du = S(normalVelocity[iEdge * nl + k - 1], normalVelocity[iEdge * nl + k]);
    du2TopOfEdge[iEdge * (nl + 1) + k] = M(du, du);
}

// du2TopOfCell, cells 1..nCellsArray(1): its edges' shears down to maxLevelEdgeBot
extern "C" __global__ void ocn_vmix_du2_cell_v842(
    rk *du2TopOfCell, const rk *du2TopOfEdge, const rk *dcEdge, const rk *dvEdge, const rk *areaCell,
    const int *nEdgesOnCell, const int *edgesOnCell, const int *maxLevelEdgeBot, int nCells, int nl, int maxEdges)
{
    const int t = vm_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k < 1) return;
    const rk invAreaCell = D(C(1.0), areaCell[iCell]);
    rk value = du2TopOfCell[iCell * (nl + 1) + k];
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        if (k >= maxLevelEdgeBot[iEdge]) continue;
        value = A(value, M(M(M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]), du2TopOfEdge[iEdge * (nl + 1) + k]),
                           invAreaCell));
    }
    du2TopOfCell[iCell * (nl + 1) + k] = value;
}

// RiTopOfEdge, edges 1..nEdgesArray(2)
extern "C" __global__ void ocn_vmix_ri_edge_v842(
    rk *RiTopOfEdge, const rk *ddensityTopOfEdge, const rk *du2TopOfEdge, const rk *layerThicknessEdge,
    const int *maxLevelEdgeTop, int nEdges, int nl, rk coef)
{
    const int t = vm_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelEdgeTop[iEdge]) return;
    const int at = iEdge * (nl + 1) + k;
    RiTopOfEdge[at] = D(M(M(coef, ddensityTopOfEdge[at]),
                          A(layerThicknessEdge[iEdge * nl + k - 1], layerThicknessEdge[iEdge * nl + k])),
                        A(du2TopOfEdge[at], C(1e-20)));
}

// RiTopOfCell, cells 1..nCellsArray(1)
extern "C" __global__ void ocn_vmix_ri_cell_v842(
    rk *RiTopOfCell, const rk *ddensityTopOfCell, const rk *du2TopOfCell, const rk *layerThickness,
    const int *maxLevelCell, int nCells, int nl, rk coef)
{
    const int t = vm_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelCell[iCell]) return;
    const int at = iCell * (nl + 1) + k;
    RiTopOfCell[at] = D(M(M(coef, ddensityTopOfCell[at]),
                          A(layerThickness[iCell * nl + k - 1], layerThickness[iCell * nl + k])),
                        A(du2TopOfCell[at], C(1e-20)));
}

// ocn_vel_vmix_coefs_rich, edges 1..nEdgesSolve
extern "C" __global__ void ocn_vmix_rich_visc_v842(
    rk *vertViscTopOfEdge, const rk *RiTopOfEdge, const int *maxLevelEdgeTop, int nEdges, int nl, rk richMix,
    rk bkrdVisc, rk convectiveVisc)
{
    const int t = vm_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelEdgeTop[iEdge]) return;
    const int at = iEdge * (nl + 1) + k;
    const rk ri = RiTopOfEdge[at];
    if (ri > C(0.0)) {
        const rk x = A(C(1.0), M(C(5.0), ri));
        rk v = A(A(vertViscTopOfEdge[at], bkrdVisc), D(richMix, M(x, x)));
        if (v > convectiveVisc) v = convectiveVisc;
        vertViscTopOfEdge[at] = v;
    } else {
        vertViscTopOfEdge[at] = convectiveVisc;
    }
}

// ocn_tracer_vmix_coefs_rich, cells 1..nCellsSolve
extern "C" __global__ void ocn_vmix_rich_diff_v842(
    rk *vertDiffTopOfCell, const rk *RiTopOfCell, const int *maxLevelCell, int nCells, int nl, rk richMix,
    rk bkrdDiff, rk bkrdVisc, rk convectiveDiff)
{
    const int t = vm_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    if (k < 1 || k >= maxLevelCell[iCell]) return;
    const int at = iCell * (nl + 1) + k;
    const rk ri = RiTopOfCell[at];
    if (ri > C(0.0)) {
        const rk x = A(C(1.0), M(C(5.0), ri));
        rk d = A(A(vertDiffTopOfCell[at], bkrdDiff), D(A(bkrdVisc, D(richMix, M(x, x))), x));
        if (d > convectiveDiff) d = convectiveDiff;
        vertDiffTopOfCell[at] = d;
    } else {
        vertDiffTopOfCell[at] = convectiveDiff;
    }
}

// ocn_vel_vmix_tend_implicit, edges 1..nEdgesArray(1): one thread per edge.
// work holds A, B, C, bTemp, rTemp, x for the edge (6*nl words).
extern "C" __global__ void ocn_vmix_vel_implicit_v842(
    rk *normalVelocity, rk *layerThicknessEdge, rk *work, const rk *layerThickness, const rk *vertViscTopOfEdge,
    const rk *kineticEnergyCell, const int *cellsOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl, rk dt,
    rk bottomDrag)
{
    const int iEdge = vm_thread();
    if (iEdge >= nEdges) return;
    const int N = maxLevelEdgeTop[iEdge];
    if (N <= 0) return;
    rk *Aa = work + iEdge * 6 * nl, *Bb = Aa + nl, *Cc = Bb + nl, *bT = Cc + nl, *rT = bT + nl, *x = rT + nl;
    rk *u = normalVelocity + iEdge * nl, *hE = layerThicknessEdge + iEdge * nl;
    const rk *visc = vertViscTopOfEdge + iEdge * (nl + 1);
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    Aa[0] = C(0.0);
    for (int k = 0; k < N; ++k)
        hE[k] = M(C(0.5), A(layerThickness[cell1 * nl + k], layerThickness[cell2 * nl + k]));
    for (int k = 1; k < N; ++k)
        Aa[k] = D(D(M(M(C(-2.0), dt), visc[k]), A(hE[k - 1], hE[k])), hE[k]);
    for (int k = 0; k < N - 1; ++k)
        Cc[k] = D(D(M(M(C(-2.0), dt), visc[k + 1]), A(hE[k], hE[k + 1])), hE[k]);
    if (N > 1) Bb[0] = S(C(1.0), Cc[0]);
    for (int k = 1; k < N - 1; ++k) Bb[k] = S(S(C(1.0), Aa[k]), Cc[k]);
    // the Fortran's k after "do k = 2, N-1": N, or 2 when that loop is empty with N = 1
    const int kd = (N > 2 ? N : 2) - 1;
    Bb[N - 1] = A(S(C(1.0), Aa[N - 1]),
                  D(M(M(dt, bottomDrag), SQRT(A(kineticEnergyCell[cell1 * nl + kd], kineticEnergyCell[cell2 * nl + kd]))),
                    hE[kd]));
    // tridiagonal_solve(A(2:N), B, C(1:N-1), u, x, N)
    bT[0] = Bb[0];
    rT[0] = u[0];
    for (int i = 1; i < N; ++i) {
        const rk m = D(Aa[i], bT[i - 1]);
        bT[i] = S(Bb[i], M(m, Cc[i - 1]));
        rT[i] = S(u[i], M(m, rT[i - 1]));
    }
    x[N - 1] = D(rT[N - 1], bT[N - 1]);
    for (int i = N - 2; i >= 0; --i) x[i] = D(S(rT[i], M(Cc[i], x[i + 1])), bT[i]);
    for (int k = 0; k < N; ++k) u[k] = x[k];
    for (int k = N; k < nl; ++k) u[k] = C(0.0);
}

// ocn_tracer_vmix_tend_implicit, cells 1..nCellsArray(1): one thread per
// cell; work holds A, B, C, bTemp (4*nl words) and rTemp, x (2*nl*nTracers)
extern "C" __global__ void ocn_vmix_tracer_implicit_v842(
    rk *tracers, rk *work, const rk *layerThickness, const rk *vertDiffTopOfCell, const int *maxLevelCell, int nCells,
    int nl, int nTracers, rk dt)
{
    const int iCell = vm_thread();
    if (iCell >= nCells) return;
    const int N = maxLevelCell[iCell];
    rk *Aa = work + iCell * (4 + 2 * nTracers) * nl, *Bb = Aa + nl, *Cc = Bb + nl, *bT = Cc + nl;
    rk *rT = bT + nl, *x = rT + nl * nTracers;
    const rk *h = layerThickness + iCell * nl, *diff = vertDiffTopOfCell + iCell * (nl + 1);
    rk *tr = tracers + iCell * nl * nTracers;
    Aa[0] = C(0.0);
    for (int k = 1; k < N; ++k) Aa[k] = D(D(M(M(C(-2.0), dt), diff[k]), A(h[k - 1], h[k])), h[k]);
    for (int k = 0; k < N - 1; ++k) Cc[k] = D(D(M(M(C(-2.0), dt), diff[k + 1]), A(h[k], h[k + 1])), h[k]);
    if (N > 0) Cc[N - 1] = C(0.0);
    for (int k = 0; k < N; ++k) Bb[k] = S(S(C(1.0), Aa[k]), Cc[k]);
    if (N > 0) {
        // tridiagonal_solve_mult(A(2:N), B, C(1:N-1), tracers, x, N, nVertLevels, nTracers)
        bT[0] = Bb[0];
        for (int j = 0; j < nTracers; ++j) rT[j] = tr[j];
        for (int i = 1; i < N; ++i) {
            const rk m = D(Aa[i], bT[i - 1]);
            bT[i] = S(Bb[i], M(m, Cc[i - 1]));
            for (int j = 0; j < nTracers; ++j)
                rT[i * nTracers + j] = S(tr[i * nTracers + j], M(m, rT[(i - 1) * nTracers + j]));
        }
        for (int j = 0; j < nTracers; ++j) x[(N - 1) * nTracers + j] = D(rT[(N - 1) * nTracers + j], bT[N - 1]);
        for (int i = N - 2; i >= 0; --i)
            for (int j = 0; j < nTracers; ++j)
                x[i * nTracers + j] = D(S(rT[i * nTracers + j], M(Cc[i], x[(i + 1) * nTracers + j])), bT[i]);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < nTracers; ++j) tr[i * nTracers + j] = x[i * nTracers + j];
    }
    for (int k = (N > 0 ? N : 0); k < nl; ++k)
        for (int j = 0; j < nTracers; ++j) tr[k * nTracers + j] = SENTINEL_DEFAULT;
}

// the vertical mixing budget: (new - old) / dt over every level of cells 1..nCellsArray(1)
extern "C" __global__ void ocn_vmix_budget_v842(rk *budget, const rk *tracers, int n, rk dt)
{
    const int t = vm_thread();
    if (t >= n) return;
    budget[t] = D(S(tracers[t], budget[t]), dt);
}
