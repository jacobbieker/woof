// MPAS-Ocean v8.4.2: ocn_diagnostic_solve.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_diagnostics.F
//     (ocn_diagnostic_solve, ocn_diagnostics_init,
//      ocn_compute_land_ice_flux_input_fields with land ice pressure only)
//   src/core_ocean/shared/mpas_ocn_diagnostics_routines.F
//     (ocn_relativeVorticity_circulation)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off); x**2 is x*x, sqrt the
// correctly rounded instruction, and an integer sign times a real is the
// sign converted, then multiplied. Each kernel runs the extent its Fortran
// loop runs (diagnostics_v842.py passes nCellsArray(i) ...). A thread owns one
// (element, level) word where the Fortran's sums run over neighbours only, so
// every word is summed in the Fortran's order; column recurrences run one
// thread per column.
//
// Layout: a(nVertLevels, nCells+1) is a[iCell*nVertLevels + k] (levels nl+1
// for the nVertLevelsP1 fields); a(n, nCells+1) of connectivity is
// a[iCell*n + i]; activeTracers(nTracers, nVertLevels, nCells+1) is
// t[(iCell*nVertLevels + k)*nTracers + i]. Connectivity is 1-based, and
// nCells+1 (nEdges+1, nVertices+1) is MPAS's missing-neighbour element.

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

#define GRAVITY C(9.80616)            // mpas_constants gravity
// -1e34 is a default (single precision) literal in ocn_diagnostic_solve's
// sentinels: its RKIND 8 word is the binary32 word widened
#define SENTINEL_DEFAULT ((rk)-1e34f)

__device__ __forceinline__ int diag_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// Fortran FRACTION(x): x's significand, x * 2**-exponent(x), in [0.5, 1)
// with x's sign (0 for 0); on the words, for the finite normal values the
// KPP surface layer gives it
#ifdef MPAS_RKIND8
__device__ __forceinline__ rk diag_fraction(rk x) {
    if (x == 0.0) return x;
    const unsigned long long w = (unsigned long long)__double_as_longlong(x);
    return __longlong_as_double((long long)((w & 0x800fffffffffffffULL) | 0x3fe0000000000000ULL));
}
#else
__device__ __forceinline__ rk diag_fraction(rk x) {
    if (x == 0.0f) return x;
    const unsigned int w = (unsigned int)__float_as_int(x);
    return __int_as_float((int)((w & 0x807fffffu) | 0x3f000000u));
}
#endif

// layerThicknessEdge: every level of edges 1..nEdges to -1e34, then the
// two-cell mean down to maxLevelEdgeTop
extern "C" __global__ void ocn_diag_layer_thickness_edge_v842(
    rk *layerThicknessEdge, const rk *layerThickness, const int *cellsOnEdge, const int *maxLevelEdgeTop,
    int nEdges, int nl)
{
    const int i = diag_thread();
    if (i >= nEdges * nl) return;
    const int iEdge = i / nl, k = i % nl;
    rk value = C(-1.0e34);
    if (k < maxLevelEdgeTop[iEdge]) {
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        value = M(C(0.5), A(layerThickness[cell1 * nl + k], layerThickness[cell2 * nl + k]));
    }
    layerThicknessEdge[i] = value;
}

// the missing-neighbour sentinels: normalVelocity(:, nEdges+1),
// layerThickness(:, nCells+1), activeTracers(T and S, :, nCells+1)
extern "C" __global__ void ocn_diag_sentinels_v842(
    rk *normalVelocity, rk *layerThickness, rk *tracers, int nEdges, int nCells, int nl, int nTracers,
    int indexT, int indexS)
{
    const int k = diag_thread();
    if (k >= nl) return;
    normalVelocity[nEdges * nl + k] = SENTINEL_DEFAULT;
    layerThickness[nCells * nl + k] = SENTINEL_DEFAULT;
    tracers[(nCells * nl + k) * nTracers + indexT - 1] = SENTINEL_DEFAULT;
    tracers[(nCells * nl + k) * nTracers + indexS - 1] = SENTINEL_DEFAULT;
}

// ocn_relativeVorticity_circulation over vertices 1..nVertices
extern "C" __global__ void ocn_diag_vorticity_circulation_v842(
    rk *relativeVorticity, rk *circulation, const rk *normalVelocity, const rk *dcEdge, const rk *areaTriangle,
    const int *edgesOnVertex, const int *edgeSignOnVertex, const int *maxLevelVertexBot, int nVertices, int nl,
    int vertexDegree)
{
    const int t = diag_thread();
    if (t >= nVertices * nl) return;
    const int iVertex = t / nl, k = t % nl;
    rk circ = C(0.0), vort = C(0.0);
    if (k < maxLevelVertexBot[iVertex]) {
        const rk invAreaTri1 = D(C(1.0), areaTriangle[iVertex]);
        for (int i = 0; i < vertexDegree; ++i) {
            const int iEdge = edgesOnVertex[iVertex * vertexDegree + i] - 1;
            const rk sign = (rk)edgeSignOnVertex[iVertex * vertexDegree + i];
            const rk r_tmp = M(dcEdge[iEdge], normalVelocity[iEdge * nl + k]);
            circ = A(circ, M(sign, r_tmp));
            vort = A(vort, M(M(sign, r_tmp), invAreaTri1));
        }
    }
    circulation[t] = circ;
    relativeVorticity[t] = vort;
}

// a kite-weighted vertex-to-cell mean, over cells 1..nCells, down to
// maxLevelCell (allLevels: every level) -- relativeVorticityCell,
// normalizedRelativeVorticityCell and kineticEnergyVertexOnCells
extern "C" __global__ void ocn_diag_vertex_to_cell_v842(
    rk *out, const rk *vertexField, const rk *kiteAreasOnVertex, const rk *areaCell, const int *nEdgesOnCell,
    const int *kiteIndexOnCell, const int *verticesOnCell, const int *maxLevelCell, int nCells, int nl,
    int maxEdges, int vertexDegree, int allLevels)
{
    const int t = diag_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    rk sum = C(0.0);
    if (allLevels || k < maxLevelCell[iCell]) {
        const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int j = kiteIndexOnCell[iCell * maxEdges + i] - 1;
            const int iVertex = verticesOnCell[iCell * maxEdges + i] - 1;
            sum = A(sum, M(M(kiteAreasOnVertex[iVertex * vertexDegree + j], vertexField[iVertex * nl + k]),
                           invAreaCell1));
        }
    }
    out[t] = sum;
}

// divergence, kineticEnergyCell and the three layer-thickness flux
// divergences of each (cell, level), cells 1..nCellsArray(2)
extern "C" __global__ void ocn_diag_divergence_v842(
    rk *divergence, rk *kineticEnergyCell, rk *divHu, rk *divHuTransport, rk *divHuGMBolus,
    const rk *normalVelocity, const rk *normalTransportVelocity, const rk *normalGMBolusVelocity,
    const rk *layerThicknessEdge, const rk *dcEdge, const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell,
    const int *edgesOnCell, const int *edgeSignOnCell, const int *maxLevelCell, int nCells, int nl, int maxEdges)
{
    const int t = diag_thread();
    if (t >= nCells * nl) return;
    const int iCell = t / nl, k = t % nl;
    rk div = C(0.0), ke = C(0.0), hu = C(0.0), huT = C(0.0), huGM = C(0.0);
    if (k < maxLevelCell[iCell]) {
        const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
            const rk sign = (rk)edgeSignOnCell[iCell * maxEdges + i];
            const rk dc = dcEdge[iEdge], dv = dvEdge[iEdge];
            const rk u = normalVelocity[iEdge * nl + k];
            const rk h = layerThicknessEdge[iEdge * nl + k];
            const rk r_tmp = M(M(dv, u), invAreaCell1);
            div = S(div, M(sign, r_tmp));
            hu = S(hu, M(M(h, sign), r_tmp));
            ke = A(ke, M(M(M(C(0.25), r_tmp), dc), u));
            huT = S(huT, M(M(M(M(h, sign), dv), normalTransportVelocity[iEdge * nl + k]), invAreaCell1));
            huGM = S(huGM, M(M(M(M(h, sign), dv), normalGMBolusVelocity[iEdge * nl + k]), invAreaCell1));
        }
    }
    divergence[t] = div;
    kineticEnergyCell[t] = ke;
    divHu[t] = hu;
    divHuTransport[t] = huT;
    divHuGMBolus[t] = huGM;
}

// the vertical velocities, integrated up from maxLevelCell+1, where only
// vertVelocityTop is set to 0 (v8.4.2 leaves the other two's words there)
extern "C" __global__ void ocn_diag_vertical_velocity_v842(
    rk *vertVelocityTop, rk *vertTransportVelocityTop, rk *vertGMBolusVelocityTop, const rk *divHu,
    const rk *divHuTransport, const rk *divHuGMBolus, const int *maxLevelCell, int nCells, int nl)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell], np1 = nl + 1;
    rk *w = vertVelocityTop + iCell * np1, *wT = vertTransportVelocityTop + iCell * np1,
       *wGM = vertGMBolusVelocityTop + iCell * np1;
    w[top] = C(0.0);
    for (int k = top - 1; k >= 0; --k) {
        w[k] = S(w[k + 1], divHu[iCell * nl + k]);
        wT[k] = S(wT[k + 1], divHuTransport[iCell * nl + k]);
        wGM[k] = S(wGM[k + 1], divHuGMBolus[iCell * nl + k]);
    }
}

// tangentialVelocity, edges 1..nEdgesArray(2)
extern "C" __global__ void ocn_diag_tangential_velocity_v842(
    rk *tangentialVelocity, const rk *normalVelocity, const rk *weightsOnEdge, const int *nEdgesOnEdge,
    const int *edgesOnEdge, const int *maxLevelEdgeTop, int nEdges, int nl, int maxEdges2)
{
    const int t = diag_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    rk v = C(0.0);
    if (k < maxLevelEdgeTop[iEdge]) {
        for (int i = 0; i < nEdgesOnEdge[iEdge]; ++i) {
            const int eoe = edgesOnEdge[iEdge * maxEdges2 + i] - 1;
            v = A(v, M(weightsOnEdge[iEdge * maxEdges2 + i], normalVelocity[eoe * nl + k]));
        }
    }
    tangentialVelocity[t] = v;
}

// kineticEnergyVertex, vertices 1..nVertices, every level
extern "C" __global__ void ocn_diag_kinetic_energy_vertex_v842(
    rk *kineticEnergyVertex, const rk *normalVelocity, const rk *dcEdge, const rk *dvEdge, const rk *areaTriangle,
    const int *edgesOnVertex, int nVertices, int nl, int vertexDegree)
{
    const int t = diag_thread();
    if (t >= nVertices * nl) return;
    const int iVertex = t / nl, k = t % nl;
    rk ke = C(0.0);
    for (int i = 0; i < vertexDegree; ++i) {
        const int iEdge = edgesOnVertex[iVertex * vertexDegree + i] - 1;
        const rk r_tmp = D(M(M(dcEdge[iEdge], dvEdge[iEdge]), C(0.25)), areaTriangle[iVertex]);
        const rk u = normalVelocity[iEdge * nl + k];
        ke = A(ke, M(r_tmp, M(u, u)));
    }
    kineticEnergyVertex[t] = ke;
}

// kineticEnergyCell = 5/8 of the cell value + 3/8 of the vertex value, cells
// 1..nCells (every level)
extern "C" __global__ void ocn_diag_kinetic_energy_blend_v842(
    rk *kineticEnergyCell, const rk *kineticEnergyVertexOnCells, int n)
{
    const int t = diag_thread();
    if (t >= n) return;
    kineticEnergyCell[t] = A(M(C(0.625), kineticEnergyCell[t]), M(C(0.375), kineticEnergyVertexOnCells[t]));
}

// the normalized relative and planetary vorticity at vertices 1..nVerticesArray(3),
// down to maxLevelVertexBot
extern "C" __global__ void ocn_diag_normalized_vorticity_vertex_v842(
    rk *nrvVertex, rk *npvVertex, const rk *relativeVorticity, const rk *fVertex, const rk *layerThickness,
    const rk *kiteAreasOnVertex, const rk *areaTriangle, const int *cellsOnVertex, const int *maxLevelVertexBot,
    int nVertices, int nl, int vertexDegree)
{
    const int t = diag_thread();
    if (t >= nVertices * nl) return;
    const int iVertex = t / nl, k = t % nl;
    if (k >= maxLevelVertexBot[iVertex]) return;
    const rk invAreaTri1 = D(C(1.0), areaTriangle[iVertex]);
    rk h = C(0.0);
    for (int i = 0; i < vertexDegree; ++i) {
        const int iCell = cellsOnVertex[iVertex * vertexDegree + i] - 1;
        h = A(h, M(layerThickness[iCell * nl + k], kiteAreasOnVertex[iVertex * vertexDegree + i]));
    }
    h = M(h, invAreaTri1);
    nrvVertex[t] = D(relativeVorticity[t], h);
    npvVertex[t] = D(fVertex[iVertex], h);
}

// the normalized vorticities at edges 1..nEdgesArray(3), down to maxLevelEdgeBot
extern "C" __global__ void ocn_diag_normalized_vorticity_edge_v842(
    rk *nrvEdge, rk *npvEdge, const rk *nrvVertex, const rk *npvVertex, const int *verticesOnEdge,
    const int *maxLevelEdgeBot, int nEdges, int nl)
{
    const int t = diag_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    rk r = C(0.0), p = C(0.0);
    if (k < maxLevelEdgeBot[iEdge]) {
        const int v1 = verticesOnEdge[2 * iEdge] - 1, v2 = verticesOnEdge[2 * iEdge + 1] - 1;
        r = M(C(0.5), A(nrvVertex[v1 * nl + k], nrvVertex[v2 * nl + k]));
        p = M(C(0.5), A(npvVertex[v1 * nl + k], npvVertex[v2 * nl + k]));
    }
    nrvEdge[t] = r;
    npvEdge[t] = p;
}

// APVM, edges 1..nEdgesArray(2): the vorticity gradients (normal down to
// maxLevelEdgeTop, tangential down to maxLevelEdgeBot), then the upwinding
// down to maxLevelEdgeBot. Below maxLevelEdgeTop v8.4.2 reads a normal
// gradient it never wrote (its scratch field's words); here it is 0, and
// normalVelocity, which multiplies it, is 0 there.
extern "C" __global__ void ocn_diag_apvm_v842(
    rk *nrvEdge, const rk *nrvCell, const rk *nrvVertex, const rk *normalVelocity, const rk *tangentialVelocity,
    const rk *dcEdge, const rk *dvEdge, const int *cellsOnEdge, const int *verticesOnEdge,
    const int *maxLevelEdgeTop, const int *maxLevelEdgeBot, int nEdges, int nl, rk apvmDt)
{
    const int t = diag_thread();
    if (t >= nEdges * nl) return;
    const int iEdge = t / nl, k = t % nl;
    if (k >= maxLevelEdgeBot[iEdge]) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    const int v1 = verticesOnEdge[2 * iEdge] - 1, v2 = verticesOnEdge[2 * iEdge + 1] - 1;
    rk gradNormal = C(0.0);
    if (k < maxLevelEdgeTop[iEdge])
        gradNormal = M(S(nrvCell[cell2 * nl + k], nrvCell[cell1 * nl + k]), D(C(1.0), dcEdge[iEdge]));
    const rk gradTangential = M(S(nrvVertex[v2 * nl + k], nrvVertex[v1 * nl + k]), D(C(1.0), dvEdge[iEdge]));
    nrvEdge[t] = S(nrvEdge[t], M(apvmDt, A(M(normalVelocity[t], gradNormal),
                                           M(tangentialVelocity[t], gradTangential))));
}

// MontgomeryPotential, cells 1..nCells: one thread per column
extern "C" __global__ void ocn_diag_montgomery_v842(
    rk *montgomeryPotential, const rk *density, const rk *layerThickness, const rk *bottomDepth, int nCells, int nl)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    const rk *rho = density + iCell * nl, *h = layerThickness + iCell * nl;
    rk *mp = montgomeryPotential + iCell * nl;
    rk column = C(0.0);
    for (int k = 0; k < nl; ++k) column = A(column, h[k]);
    rk pTop = C(0.0);
    mp[0] = M(GRAVITY, A(bottomDepth[iCell], column));
    for (int k = 1; k < nl; ++k) {
        pTop = A(pTop, M(M(rho[k - 1], GRAVITY), h[k - 1]));
        mp[k] = A(mp[k - 1], M(pTop, S(D(C(1.0), rho[k]), D(C(1.0), rho[k - 1]))));
    }
}

// pressure, zMid, zTop and ssh, cells 1..nCells: one thread per column
extern "C" __global__ void ocn_diag_pressure_zmid_v842(
    rk *pressure, rk *zMid, rk *zTop, rk *ssh, const rk *density, const rk *layerThickness, const rk *bottomDepth,
    const rk *atmosphericPressure, const rk *seaIcePressure, const rk *frazilSurfacePressure,
    const rk *landIcePressure, const int *maxLevelCell, int nCells, int nl, int withFrazil, int withLandIce)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    const rk *rho = density + iCell * nl, *h = layerThickness + iCell * nl;
    rk *p = pressure + iCell * nl, *zm = zMid + iCell * nl, *zt = zTop + iCell * nl;
    rk p1 = A(atmosphericPressure[iCell], seaIcePressure[iCell]);
    if (withFrazil) p1 = A(p1, frazilSurfacePressure[iCell]);
    if (withLandIce) p1 = A(p1, landIcePressure[iCell]);
    p[0] = A(p1, M(M(M(rho[0], GRAVITY), C(0.5)), h[0]));
    const int top = maxLevelCell[iCell];
    for (int k = 1; k < top; ++k)
        p[k] = A(p[k - 1], M(M(C(0.5), GRAVITY), A(M(rho[k - 1], h[k - 1]), M(rho[k], h[k]))));
    const int kb = top - 1;
    const rk zMidBottom = A(-bottomDepth[iCell], M(C(0.5), h[kb]));
    const rk zTopBottom = A(-bottomDepth[iCell], h[kb]);
    for (int k = kb; k < nl; ++k) {
        zm[k] = zMidBottom;
        zt[k] = zTopBottom;
    }
    for (int k = kb - 1; k >= 0; --k) {
        zm[k] = A(zm[k + 1], M(C(0.5), A(h[k + 1], h[k])));
        zt[k] = A(zt[k + 1], h[k]);
    }
    ssh[iCell] = zt[0];
}

// BruntVaisalaFreqTop, then RiTopOfCell, cells 1..nCellsArray(3): one thread per column
extern "C" __global__ void ocn_diag_stability_v842(
    rk *BruntVaisalaFreqTop, rk *RiTopOfCell, const rk *displacedDensity, const rk *density, const rk *zMid,
    const rk *normalVelocity, const rk *dcEdge, const rk *dvEdge, const rk *areaCell, const int *nEdgesOnCell,
    const int *edgesOnCell, const int *maxLevelCell, int nCells, int nl, int maxEdges, rk coef)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell];
    rk *bv = BruntVaisalaFreqTop + iCell * nl, *ri = RiTopOfCell + iCell * (nl + 1);
    const rk *zm = zMid + iCell * nl;
    bv[0] = C(0.0);
    for (int k = 1; k < top; ++k)
        bv[k] = D(M(coef, S(displacedDensity[iCell * nl + k - 1], density[iCell * nl + k])), S(zm[k - 1], zm[k]));
    for (int k = 0; k <= nl; ++k) ri[k] = C(100.0);
    const rk invAreaCell1 = D(C(1.0), areaCell[iCell]);
    for (int k = 1; k < top; ++k) {
        rk shearSquared = C(0.0);
        for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
            const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
            const rk factor = M(M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]), invAreaCell1);
            const rk du = S(normalVelocity[iEdge * nl + k - 1], normalVelocity[iEdge * nl + k]);
            shearSquared = A(shearSquared, M(factor, M(du, du)));
        }
        rk shearMean = SQRT(shearSquared);
        shearMean = D(shearMean, S(zm[k - 1], zm[k]));
        ri[k] = D(bv[k], A(M(shearMean, shearMean), C(1.0e-10)));
    }
    ri[0] = ri[1];
}

// tracersSurfaceValue(:, iCell) = activeTracers(:, 1, iCell), cells 1..nCellsArray(3)
extern "C" __global__ void ocn_diag_tracers_surface_value_v842(
    rk *tracersSurfaceValue, const rk *tracers, int nCells, int nl, int nTracers)
{
    const int t = diag_thread();
    if (t >= nCells * nTracers) return;
    const int iCell = t / nTracers, i = t % nTracers;
    tracersSurfaceValue[t] = tracers[(iCell * nl) * nTracers + i];
}

// normalVelocitySurfaceLayer(iEdge) = normalVelocity(1, iEdge), edges 1..n
extern "C" __global__ void ocn_diag_surface_velocity_v842(
    rk *normalVelocitySurfaceLayer, const rk *normalVelocity, int nEdges, int nl)
{
    const int iEdge = diag_thread();
    if (iEdge >= nEdges) return;
    normalVelocitySurfaceLayer[iEdge] = normalVelocity[iEdge * nl];
}

// KPP's surface-layer averages of the tracers, cells 1..nCellsArray(1): one
// thread per column. v8.4.2 weighs the partial layer by FRACTION(rSurfaceLayer),
// the significand, not the fractional part.
extern "C" __global__ void ocn_diag_kpp_cells_v842(
    rk *tracersSurfaceLayerValue, rk *indexSurfaceLayerDepth, const rk *tracers, const rk *layerThickness,
    const int *maxLevelCell, int nCells, int nl, int nTracers, rk surfaceLayerDepth)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    const int top = maxLevelCell[iCell];
    const rk *h = layerThickness + iCell * nl;
    const rk *tr = tracers + iCell * nl * nTracers;
    rk *out = tracersSurfaceLayerValue + iCell * nTracers;
    rk sum = C(0.0), rSurfaceLayer = C(0.0);
    indexSurfaceLayerDepth[iCell] = (rk)-9.e30f;
    for (int k = 0; k < top; ++k) {
        sum = A(sum, h[k]);
        rSurfaceLayer = (rk)top;
        if (sum > surfaceLayerDepth) {
            sum = S(sum, h[k]);
            rSurfaceLayer = A((rk)k, D(S(surfaceLayerDepth, sum), h[k]));
            indexSurfaceLayerDepth[iCell] = rSurfaceLayer;
            break;
        }
    }
    const int whole = (int)rSurfaceLayer;
    int kp = whole + 1;
    if (kp > top) kp = top;
    const rk frac = diag_fraction(rSurfaceLayer);
    for (int i = 0; i < nTracers; ++i) {
        rk value = C(0.0);
        for (int k = 0; k < whole; ++k) value = A(value, M(tr[k * nTracers + i], h[k]));
        out[i] = D(A(value, M(M(frac, tr[(kp - 1) * nTracers + i]), h[kp - 1])), surfaceLayerDepth);
    }
}

// KPP's surface-layer normal velocity, edges 1..nEdgesArray(1)
extern "C" __global__ void ocn_diag_kpp_edges_v842(
    rk *normalVelocitySurfaceLayer, const rk *normalVelocity, const rk *layerThicknessEdge,
    const int *maxLevelEdgeTop, int nEdges, int nl, rk surfaceLayerDepth)
{
    const int iEdge = diag_thread();
    if (iEdge >= nEdges) return;
    const int top = maxLevelEdgeTop[iEdge];
    const rk *h = layerThicknessEdge + iEdge * nl, *u = normalVelocity + iEdge * nl;
    rk sum = C(0.0), rSurfaceLayer = (rk)(top < 1 ? top : 1);
    for (int k = 0; k < top; ++k) {
        rSurfaceLayer = (rk)(k + 1);
        sum = A(sum, h[k]);
        if (sum > surfaceLayerDepth) {
            sum = S(sum, h[k]);
            rSurfaceLayer = A((rk)k, D(S(surfaceLayerDepth, sum), h[k]));
            break;
        }
    }
    const int whole = (int)rSurfaceLayer;
    rk value = C(0.0);
    for (int k = 0; k < whole; ++k) value = A(value, M(u[k], h[k]));
    const int kp = whole + 1;
    if (kp <= top) value = D(A(value, M(M(diag_fraction(rSurfaceLayer), u[kp - 1]), h[kp - 1])), surfaceLayerDepth);
    normalVelocitySurfaceLayer[iEdge] = value;
}

// the attenuation coefficients, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_diag_attenuation_v842(
    rk *coefficient, rk *coefficientRunoff, int nCells, rk value, rk valueRunoff)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    coefficient[iCell] = value;
    coefficientRunoff[iCell] = valueRunoff;
}

// ocn_compute_land_ice_flux_input_fields with the land ice pressure package
// alone: the mask from the fraction, cells 1..nCells
extern "C" __global__ void ocn_diag_land_ice_mask_v842(
    rk *landIceFraction, int *landIceMask, int nCells)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    if (landIceFraction[iCell] >= C(0.5)) {
        landIceMask[iCell] = 1;
    } else {
        landIceMask[iCell] = 0;
        landIceFraction[iCell] = C(0.0);
    }
}

// pressureAdjustedSSH, cells 1..nCellsArray(2)
extern "C" __global__ void ocn_diag_pressure_adjusted_ssh_v842(
    rk *pressureAdjustedSSH, const rk *ssh, const rk *seaIcePressure, const rk *landIceDraft, int nCells,
    int withLandIce, rk gravityRhoSw)
{
    const int iCell = diag_thread();
    if (iCell >= nCells) return;
    rk value = A(ssh[iCell], D(seaIcePressure[iCell], gravityRhoSw));
    if (withLandIce) value = S(value, landIceDraft[iCell]);
    pressureAdjustedSSH[iCell] = value;
}

// gradSSH, edges 1..nEdgesArray(2)
extern "C" __global__ void ocn_diag_grad_ssh_v842(
    rk *gradSSH, const rk *pressureAdjustedSSH, const rk *dcEdge, const int *cellsOnEdge, const int *edgeMask,
    int nEdges, int nl)
{
    const int iEdge = diag_thread();
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    gradSSH[iEdge] = D(M((rk)edgeMask[iEdge * nl], S(pressureAdjustedSSH[cell2], pressureAdjustedSSH[cell1])),
                       dcEdge[iEdge]);
}
