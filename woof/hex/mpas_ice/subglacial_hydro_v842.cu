// MPAS land-ice (MALI) v8.4.2: the subglacial hydrology model (li_SGH_solve):
// the distributed sheet with till storage and, optionally, channels.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_subglacial_hydro.F:206-586 (li_SGH_solve), 666-886
//   (calc_edge_quantities), 888-1115 (check_timestep's per-edge limits), 1117-1279 (calc_pressure),
//   1281-1361 (calc_pressure_diag_vars), 1363-1536 (update_channel), 1538-1631 (evolve_channel),
//   1633-1755 (shmip_timevarying_forcing)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 unless MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). Real powers are glibc's powf/pow (the shared
// transcriptions the driver prepends), x**3 is (x*x)*x as gfortran expands
// it, and sin/cos (the SHMIP forcings, RKIND 4 only) are glibc's sinf/cosf.
// Fortran MIN/MAX return the first argument only if it compares less/greater
// and the second otherwise, so a tie of signed zeros returns the second: the
// operand order the x86-64 oracle's gfortran -O0 gives this module's minss/
// maxss (measured on its fixture at max(0, waterThickness)). The upwind sheet
// thickness's MAX keeps the first argument instead (li_max_first), as
// calculate_layerThicknessEdge's identical expression does. The choice is
// made on the words, as an integer select, so no compiler turns it into an
// IEEE min/max.
//
// Layout: a (nVertLevels, nCells) field is a[iCell*nVertLevels + k];
// edgesOnCell(maxEdges, nCells) is e[iCell*maxEdges + j]. Connectivity is
// MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_SQRT(x) __dsqrt_rn(x)
#define LI_ABS(x) fabs(x)
#define LI_SIGN1(x) copysign(1.0, (x))
#define LI_POW(x,y) glibc_pow((x),(y))
#define LI_SIN(x) (x)
#define LI_COS(x) (x)
#define LI_C(x) x
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __longlong_as_double(a < b ? __double_as_longlong(a) : __double_as_longlong(b));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __longlong_as_double(a > b ? __double_as_longlong(a) : __double_as_longlong(b));
}
__device__ __forceinline__ rk li_max_first(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_SQRT(x) __fsqrt_rn(x)
#define LI_ABS(x) fabsf(x)
#define LI_SIGN1(x) copysignf(1.0f, (x))
#define LI_POW(x,y) gfk_pow((x),(y))
#define LI_SIN(x) glibc_sinf(x)
#define LI_COS(x) glibc_cosf(x)
#define LI_C(x) x##f
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __int_as_float(a < b ? __float_as_int(a) : __float_as_int(b));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __int_as_float(a > b ? __float_as_int(a) : __float_as_int(b));
}
__device__ __forceinline__ rk li_max_first(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif

#define LI_RHO_WATER LI_C(1000.0)          // li_constants rho_water
#define LI_GRAVITY LI_C(9.80616)           // li_constants gravity
#define LI_LATENT_HEAT_ICE LI_C(335.0e3)   // li_constants latent_heat_ice
#define LI_CP_FRESHWATER LI_C(4.188e3)     // li_constants cp_freshwater
#define LI_PMP_DEPENDENCE LI_C(9.7456e-8)  // li_constants iceMeltingPointPressureDependence
#define LI_PII LI_C(3.141592653589793)     // li_constants pii

#define LI_ICE 32
#define LI_FLOATING 4
#define LI_GROUNDING_LINE 256

__device__ __forceinline__ bool li_is(int mask, int value) { return (mask & value) == value; }
__device__ __forceinline__ int li_grounded(int mask) { return (!li_is(mask, LI_FLOATING) && li_is(mask, LI_ICE)) ? 1 : 0; }
__device__ __forceinline__ rk li_cube(rk x) { return M(M(x, x), x); }

// basalMeltInput from 'thermal' (1) or 'basal_heat' (2); 'file' (0) keeps it
extern "C" __global__ void li_sgh_basal_melt_input_v842(
    rk *basalMeltInput, const rk *groundedBasalMassBal, const rk *basalHeatFlux, const rk *basalFrictionFlux,
    int option, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (option == 1) basalMeltInput[i] = M(LI_C(-1.0), groundedBasalMassBal[i]);
    else if (option == 2) basalMeltInput[i] = D(A(basalHeatFlux[i], basalFrictionFlux[i]), LI_LATENT_HEAT_ICE);
}

// shmip_timevarying_forcing: experiment C (kind 1, amplitude ra) or D (kind 2,
// temperature offset dT)
extern "C" __global__ void li_sgh_shmip_forcing_v842(
    rk *basalMeltInput, rk *externalWaterInput, const rk *areaCell, const rk *upperSurface, rk daysSinceStart,
    int kind, rk ra, rk dT, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (kind == 1) {
        rk b = LI_C(0.0);
        if (externalWaterInput[i] > LI_C(0.0))
            b = M(D(LI_C(0.9), areaCell[i]),
                  S(LI_C(1.0), M(ra, LI_SIN(M(M(LI_C(2.0), LI_PII), daysSinceStart)))));
        if (b < LI_C(0.0)) b = LI_C(0.0);
        basalMeltInput[i] = M(A(b, LI_C(7.93e-11)), LI_C(1000.0));
    } else {
        const rk t0 = A(S(M(LI_C(-16.0), LI_COS(M(D(M(LI_C(2.0), LI_PII), (rk)365.0f), daysSinceStart))),
                          LI_C(5.0)), dT);
        rk e = M(M(A(M(upperSurface[i], LI_C(-0.0075)), t0), D(LI_C(0.01), LI_C(86400.0))), LI_C(1000.0));
        if (e < LI_C(0.0)) e = LI_C(0.0);
        externalWaterInput[i] = e;
    }
}

// the till: storage from melt, drained at Cd, grounded only, within [0, till max]
extern "C" __global__ void li_sgh_till_v842(
    rk *tillWaterThickness, rk *tillWaterThicknessOld, const rk *basalMeltInput, const rk *externalWaterInput,
    const int *cellMask, rk deltatSGH, rk drainage, rk tillMax, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    rk w = tillWaterThickness[i];
    tillWaterThicknessOld[i] = w;
    w = A(w, M(deltatSGH, S(D(A(basalMeltInput[i], externalWaterInput[i]), LI_RHO_WATER), drainage)));
    w = M(w, (rk)li_grounded(cellMask[i]));
    w = li_min(w, tillMax);
    tillWaterThickness[i] = li_max(LI_C(0.0), w);
}

// calc_edge_quantities, first loop: edge water thickness and the normal slopes
extern "C" __global__ void li_sgh_edge_normals_v842(
    rk *waterThicknessEdge, rk *hydropotentialBaseSlopeNormal, rk *waterPressureSlopeNormal,
    const rk *waterThickness, const rk *hydropotentialBase, const rk *waterPressure, const int *cellMask,
    const int *cellsOnEdge, const rk *dcEdge, int nEdges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdges) return;
    const int c1 = cellsOnEdge[2 * e] - 1, c2 = cellsOnEdge[2 * e + 1] - 1;
    const int n = li_grounded(cellMask[c1]) + li_grounded(cellMask[c2]);
    waterThicknessEdge[e] = n > 0 ? D(A(waterThickness[c1], waterThickness[c2]), (rk)n) : LI_C(0.0);
    hydropotentialBaseSlopeNormal[e] = D(S(hydropotentialBase[c2], hydropotentialBase[c1]), dcEdge[e]);
    waterPressureSlopeNormal[e] = D(S(waterPressure[c2], waterPressure[c1]), dcEdge[e]);
}

// the tangent slope: from the vertex field on ice edges (option 1, 2), or
// mpas_tangential_vector_1d without the halo (3: nEdgesSolve, zero beyond)
extern "C" __global__ void li_sgh_tangent_v842(
    rk *tangent, const rk *normal, const rk *vertexField, const int *verticesOnEdge, const rk *dvEdge,
    const int *edgeMask, const int *nEdgesOnEdge, const int *edgesOnEdge, const rk *weightsOnEdge, int option,
    int nEdges, int nEdgesSolve, int maxEdges2)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdges) return;
    rk t = LI_C(0.0);
    if (option == 3) {
        if (e < nEdgesSolve)
            for (int i = 0; i < nEdgesOnEdge[e]; ++i)
                t = A(t, M(weightsOnEdge[e * maxEdges2 + i], normal[edgesOnEdge[e * maxEdges2 + i] - 1]));
    } else if (li_is(edgeMask[e], LI_ICE)) {
        t = D(S(vertexField[verticesOnEdge[2 * e] - 1], vertexField[verticesOnEdge[2 * e + 1] - 1]), dvEdge[e]);
    }
    tangent[e] = t;
}

// calc_edge_quantities, the rest: conductivity, diffusivity, velocity, the
// upwind and diffusive sheet fluxes
extern "C" __global__ void li_sgh_edge_fluxes_v842(
    rk *gradMagPhiEdge, rk *effectiveConducEdge, rk *diffusivity, rk *waterVelocity, rk *waterThicknessEdgeUpwind,
    rk *waterFluxAdvec, rk *waterFluxDiffu, rk *waterFlux, const rk *normal, const rk *tangent,
    const rk *waterThicknessEdge, const rk *waterThickness, const int *cellMask, const int *waterFluxMask,
    const int *cellsOnEdge, const rk *dcEdge, rk conducCoeff, rk alpha, rk beta, int nEdges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdges) return;
    const rk n = normal[e], t = tangent[e];
    const rk grad = LI_SQRT(A(M(n, n), M(t, t)));
    gradMagPhiEdge[e] = grad;
    const rk wte = waterThicknessEdge[e];
    rk k = LI_C(0.0);
    if (!(grad < LI_C(0.01)))
        k = M(M(conducCoeff, LI_POW(wte, S(alpha, LI_C(1.0)))), LI_POW(grad, S(beta, LI_C(2.0))));
    effectiveConducEdge[e] = k;
    const rk diff = M(M(M(LI_RHO_WATER, LI_GRAVITY), k), wte);
    diffusivity[e] = diff;
    const int c1 = cellsOnEdge[2 * e] - 1, c2 = cellsOnEdge[2 * e + 1] - 1;
    rk v = M(M(LI_C(-1.0), k), n);
    const rk velSign = LI_SIGN1(v);
    const rk up = li_max_first(M(velSign, waterThickness[c1]), M(M(velSign, LI_C(-1.0)), waterThickness[c2]));
    waterThicknessEdgeUpwind[e] = up;
    rk advec = M(v, up);
    rk diffu = LI_C(0.0);
    if (li_grounded(cellMask[c1]) + li_grounded(cellMask[c2]) >= 2)
        diffu = D(M(M(LI_C(-1.0), diff), S(waterThickness[c2], waterThickness[c1])), dcEdge[e]);
    if (waterFluxMask[e] == 2) {
        advec = LI_C(0.0);
        diffu = LI_C(0.0);
        v = LI_C(0.0);
    }
    waterVelocity[e] = v;
    waterFluxAdvec[e] = advec;
    waterFluxDiffu[e] = diffu;
    waterFlux[e] = A(advec, diffu);
}

// the reconstructed cell velocity, zero off grounded ice
extern "C" __global__ void li_sgh_mask_cell_velocity_v842(rk *x, rk *y, const int *cellMask, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk g = (rk)li_grounded(cellMask[i]);
    x[i] = M(x[i], g);
    y[i] = M(y[i], g);
}

// update_channel over every edge (the channel's A and N averages over nEdgesSolve)
extern "C" __global__ void li_sgh_update_channel_v842(
    rk *channelDischarge, rk *channelArea, rk *channelVelocity, rk *channelDiffusivity, rk *channelMelt,
    rk *channelPressureFreeze, rk *channelOpeningRate, rk *flowParamAChannel, rk *channelEffectivePressure,
    rk *channelClosingRate, rk *channelChangeRate, const rk *gradMagPhiEdge, const rk *normal,
    const rk *waterFlux, const rk *waterPressureSlopeNormal, const int *waterFluxMask, const int *edgeMask,
    const rk *flowParamA, const rk *effectivePressure, const int *cellsOnEdge, rk kc, rk alphaC, rk betaC,
    rk creep, rk width, rk rhoi, int includePressureMelt, int nEdges, int nEdgesSolve, int nVertLevels)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdges) return;
    const rk grad = gradMagPhiEdge[e], n = normal[e];
    rk area = channelArea[e];
    rk q = LI_C(0.0);
    if (!(grad < LI_C(0.01)))
        q = M(M(M(M(LI_C(-1.0), kc), LI_POW(area, alphaC)), LI_POW(grad, S(betaC, LI_C(2.0)))), n);
    if (waterFluxMask[e] == 2) {
        q = LI_C(0.0);
        area = LI_C(0.0);
    }
    const int em = edgeMask[e];
    if (!(li_grounded(em) || li_is(em, LI_GROUNDING_LINE))) {
        area = LI_C(0.0);
        q = LI_C(0.0);
    }
    channelArea[e] = area;
    channelDischarge[e] = q;
    channelVelocity[e] = D(q, A(area, LI_C(1.0e-12)));
    rk cd = LI_C(0.0);
    if (!(grad < LI_C(0.01)))
        cd = LI_ABS(M(M(M(M(M(LI_RHO_WATER, LI_GRAVITY), area), kc), LI_POW(area, S(alphaC, LI_C(1.0)))),
                      LI_POW(grad, S(betaC, LI_C(2.0)))));
    channelDiffusivity[e] = cd;
    const rk melt = D(A(LI_ABS(M(q, n)), LI_ABS(M(M(waterFlux[e], n), width))), LI_LATENT_HEAT_ICE);
    channelMelt[e] = melt;
    const rk freeze = D(M(M(M(M(M(LI_C(-1.0), LI_PMP_DEPENDENCE), LI_CP_FRESHWATER), LI_RHO_WATER),
                            A(q, M(waterFlux[e], width))), waterPressureSlopeNormal[e]), LI_LATENT_HEAT_ICE);
    channelPressureFreeze[e] = freeze;
    rk opening = includePressureMelt ? D(S(melt, freeze), rhoi) : D(melt, rhoi);
    if (e < nEdgesSolve) {
        const int c1 = cellsOnEdge[2 * e] - 1, c2 = cellsOnEdge[2 * e + 1] - 1;
        flowParamAChannel[e] = M(LI_C(0.5), A(flowParamA[c1 * nVertLevels + nVertLevels - 1],
                                             flowParamA[c2 * nVertLevels + nVertLevels - 1]));
        channelEffectivePressure[e] = M(LI_C(0.5), A(effectivePressure[c1], effectivePressure[c2]));
    }
    rk closing = M(M(M(creep, area), flowParamAChannel[e]), li_cube(channelEffectivePressure[e]));
    if (waterFluxMask[e] == 2) {
        opening = LI_C(0.0);
        closing = LI_C(0.0);
    }
    channelOpeningRate[e] = opening;
    channelClosingRate[e] = closing;
    channelChangeRate[e] = S(opening, closing);
}

// check_timestep's per-edge limits over nEdgesSolve (the driver takes the minima)
extern "C" __global__ void li_sgh_edge_limits_v842(
    rk *advec, rk *diffu, rk *pressure, rk *advecChannel, rk *diffuChannel, const rk *dcEdge,
    const rk *waterVelocity, const rk *diffusivity, const rk *channelVelocity, const rk *channelDiffusivity,
    rk porosity, int channels, int nEdgesSolve)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdgesSolve) return;
    const rk dc = dcEdge[e], dc2 = M(dc, dc);
    advec[e] = D(dc, A(LI_ABS(waterVelocity[e]), LI_C(1.0e-12)));
    diffu[e] = D(dc2, A(diffusivity[e], LI_C(1.0e-12)));
    pressure[e] = D(M(porosity, dc2), A(M(LI_C(2.0), diffusivity[e]), LI_C(1.0e-12)));
    if (channels) {
        advecChannel[e] = D(dc, A(LI_ABS(channelVelocity[e]), LI_C(1.0e-12)));
        diffuChannel[e] = D(dc2, A(channelDiffusivity[e], LI_C(1.0e-12)));
    }
}

// the minimum of n words in the Fortran's order (one thread)
extern "C" __global__ void li_sgh_min_v842(rk *out, const rk *words, int n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    rk m = words[0];
    for (int i = 1; i < n; ++i) m = li_min(m, words[i]);
    out[0] = m;
}

// the sheet's flux divergence over the owned grounded cells, zero elsewhere
extern "C" __global__ void li_sgh_divergence_v842(
    rk *divergence, const rk *waterFlux, const rk *dvEdge, const rk *areaCell, const int *cellMask,
    const int *nEdgesOnCell, const int *edgesOnCell, const int *edgeSignOnCell, int nCells, int nCellsSolve,
    int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    rk d = LI_C(0.0);
    if (i < nCellsSolve) {
        if (li_grounded(cellMask[i]))
            for (int j = 0; j < nEdgesOnCell[i]; ++j) {
                const int e = edgesOnCell[i * maxEdges + j] - 1;
                d = S(d, M(M(waterFlux[e], dvEdge[e]), (rk)edgeSignOnCell[i * maxEdges + j]));
            }
        d = D(d, areaCell[i]);
    }
    divergence[i] = d;
}

// evolve_channel's cell loop. As written, channelAreaChangeCell keeps the
// last edge's half-channel change rather than a sum.
extern "C" __global__ void li_sgh_channel_cells_v842(
    rk *divergenceChannel, rk *channelAreaChangeCell, const rk *channelDischarge, const rk *channelChangeRate,
    const rk *dcEdge, const rk *areaCell, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *edgeSignOnCell, int nCells, int nCellsSolve, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    rk d = LI_C(0.0), c = LI_C(0.0);
    if (i < nCellsSolve) {
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            const int e = edgesOnCell[i * maxEdges + j] - 1;
            d = S(d, M(channelDischarge[e], (rk)edgeSignOnCell[i * maxEdges + j]));
            c = M(M(channelChangeRate[e], dcEdge[e]), LI_C(0.5));
        }
        d = D(d, areaCell[i]);
        c = D(c, areaCell[i]);
    }
    divergenceChannel[i] = d;
    channelAreaChangeCell[i] = c;
}

// evolve_channel: the channel area over every edge
extern "C" __global__ void li_sgh_channel_area_v842(
    rk *channelArea, const rk *channelChangeRate, rk deltatSGH, int nEdges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEdges) return;
    channelArea[e] = li_max(LI_C(1.0e-8), A(M(channelChangeRate[e], deltatSGH), channelArea[e]));
}

// calc_pressure and calc_pressure_diag_vars over every cell. pressureCalc 1
// 'cavity', 2 'overburden'.
extern "C" __global__ void li_sgh_pressure_v842(
    rk *waterPressure, rk *waterPressureOld, rk *waterPressureTendency, rk *openingRate, rk *closingRate,
    rk *zeroOrderSum, rk *effectivePressure, rk *hydropotentialBase, rk *hydropotential,
    const rk *basalSpeed, const rk *waterThickness, const rk *flowParamA, const rk *basalMeltInput,
    const rk *externalWaterInput, const rk *tillWaterThickness, const rk *tillWaterThicknessOld,
    const rk *divergence, const rk *divergenceChannel, const rk *channelAreaChangeCell, const rk *thickness,
    const rk *bedTopography, const int *cellMask, rk bedRough, rk bedRoughMax, rk creep, rk porosity,
    rk rhoi, rk rhoo, rk seaLevel, rk deltatSGH, int pressureCalc, int nCells, int nVertLevels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk w = waterThickness[i];
    const rk opening = li_max(LI_C(0.0), M(M(bedRough, basalSpeed[i]), S(bedRoughMax, w)));
    openingRate[i] = opening;
    const rk closing = M(M(M(creep, flowParamA[i * nVertLevels + nVertLevels - 1]), li_cube(effectivePressure[i])), w);
    closingRate[i] = closing;
    const rk z = S(A(S(closing, opening), D(A(basalMeltInput[i], externalWaterInput[i]), LI_RHO_WATER)),
                   D(S(tillWaterThickness[i], tillWaterThicknessOld[i]), deltatSGH));
    zeroOrderSum[i] = z;
    const rk old = waterPressure[i];
    waterPressureOld[i] = old;
    const int mask = cellMask[i];
    const rk overburden = M(M(rhoi, LI_GRAVITY), thickness[i]);
    rk p;
    if (li_is(mask, LI_FLOATING)) {
        p = overburden;
    } else if (!li_is(mask, LI_ICE)) {
        p = LI_C(0.0);
    } else if (pressureCalc == 1) {
        p = A(D(M(M(M(S(S(S(z, divergence[i]), divergenceChannel[i]), channelAreaChangeCell[i]), LI_RHO_WATER),
                    LI_GRAVITY), deltatSGH), porosity), old);
    } else {
        p = overburden;
    }
    p = li_max(LI_C(0.0), p);
    p = li_min(p, overburden);
    if (li_is(mask, LI_FLOATING) || (!li_is(mask, LI_ICE) && bedTopography[i] < seaLevel))
        p = M(M(rhoo, LI_GRAVITY), S(seaLevel, bedTopography[i]));
    waterPressure[i] = p;
    waterPressureTendency[i] = D(S(p, old), deltatSGH);
    // calc_pressure_diag_vars
    rk n = S(overburden, p);
    if (!li_is(mask, LI_ICE)) n = LI_C(0.0);
    effectivePressure[i] = n;
    const rk base = A(M(M(LI_RHO_WATER, LI_GRAVITY), bedTopography[i]), p);
    hydropotentialBase[i] = base;
    hydropotential[i] = A(base, M(M(LI_RHO_WATER, LI_GRAVITY), w));
}

// the sheet's thickness update over every cell
extern "C" __global__ void li_sgh_water_thickness_v842(
    rk *waterThickness, rk *waterThicknessOld, rk *waterThicknessTendency, rk *divergence,
    const rk *basalMeltInput, const rk *externalWaterInput, const rk *divergenceChannel,
    const rk *channelAreaChangeCell, const rk *tillWaterThickness, const rk *tillWaterThicknessOld,
    const int *cellMask, rk deltatSGH, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk old = waterThickness[i];
    waterThicknessOld[i] = old;
    const rk inner = S(S(S(S(D(A(basalMeltInput[i], externalWaterInput[i]), LI_RHO_WATER), divergence[i]),
                           divergenceChannel[i]), channelAreaChangeCell[i]),
                       D(S(tillWaterThickness[i], tillWaterThicknessOld[i]), deltatSGH));
    const rk g = (rk)li_grounded(cellMask[i]);
    rk w = M(A(old, M(deltatSGH, inner)), g);
    w = li_max(LI_C(0.0), w);
    waterThickness[i] = w;
    divergence[i] = M(divergence[i], g);
    waterThicknessTendency[i] = D(S(w, old), deltatSGH);
}
