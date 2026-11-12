// MPAS land-ice (MALI) v8.4.2: the forward-Euler time step's own arithmetic,
// between the routines the other MALI units port.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_time_integration_fe.F:222-498 (prepare_advection, set_timestep's
//                                                                       inputs), 520-773 (advection_solver),
//                                                                       879-952 (calculate_layerThicknessEdge)
//   src/core_landice/mode_forward/mpas_li_diagnostic_vars.F:78-338 (li_calculate_diagnostic_vars,
//                                                                    li_calculate_apparent_diffusivity)
//   src/core_landice/mode_forward/mpas_li_velocity.F:224-666 (li_velocity_solve for 'sia' and 'none'),
//                                                    879-984 (calculate_strain_rates, remove_fast_ice)
//   src/core_landice/shared/mpas_li_setup.F:626-739 (li_compute_gradient_2d)
//   src/operators/mpas_vector_reconstruction.F:205-339,489-588 (mpas_reconstruct_2d, mpas_reconstruct_1d)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 unless MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). The RKIND 4 unit is prepended with glibc's cosf, sinf
// and atanf (glibc_trig_flt32.cuh); at RKIND 8 the driver refuses the sphere's
// reconstruction (glibc's cos and sin) and the strain-rate angle (glibc's
// atan): neither binary64 function is correctly rounded or permissively
// licensed, so their words cannot be reproduced here. Under the opt-in
// (WOOF_GLIBC239_LIBM64=1, a source checkout only) the r8 unit is prepended
// with the LGPL transcription of glibc's (WOOF_GLIBC239_TRIG64,
// tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh) and nothing is refused.
//
// Layout: a Fortran array a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// coeffs_reconstruct(3, maxEdges, nCells) is c[(iCell*maxEdges + i)*3 + d].
// Connectivity is MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_SQRT(x) __dsqrt_rn(x)
#define LI_ABS(x) fabs(x)
#define LI_SIGN1(x) copysign(1.0, (x))
#ifdef WOOF_GLIBC239_TRIG64
#define LI_COS(x) g239::g239_cos(x)
#define LI_SIN(x) g239::g239_sin(x)
#define LI_ATAN(x) g239::g239_atan(x)
#else
#define LI_COS(x) (x)
#define LI_SIN(x) (x)
#define LI_ATAN(x) (x)
#endif
#define LI_C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_SQRT(x) __fsqrt_rn(x)
#define LI_ABS(x) fabsf(x)
#define LI_SIGN1(x) copysignf(1.0f, (x))
#define LI_COS(x) glibc_cosf(x)
#define LI_SIN(x) glibc_sinf(x)
#define LI_ATAN(x) glibc_atanf(x)
#define LI_C(x) x##f
#endif

#define LI_SCYR LI_C(31536000.0)          // li_constants scyr
#define LI_ICE 32
#define LI_DYNAMIC_ICE 2
#define LI_FLOATING 4
#define LI_DYNAMIC_MARGIN 16

__device__ __forceinline__ bool li_is(int mask, int value) { return (mask & value) == value; }
__device__ __forceinline__ bool li_is_grounded_ice(int mask) {
    return !li_is(mask, LI_FLOATING) && li_is(mask, LI_ICE);
}
// Fortran MIN/MAX as the x86-64 oracle's gfortran -O0 evaluates them here
// (minss/maxss with the first argument kept unless the second compares
// less/greater; measured on the oracle's own toolchain at
// calculate_layerThicknessEdge, where max(+0, -0) is +0 and max(-0, +0) is
// -0). The choice is made on the words, as an integer select: a floating
// conditional of this shape may be compiled to min/max.f32, which order -0
// below +0 (the Metal translator compiles it so, as NVCC can).
#ifdef MPAS_RKIND8
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif

// mpas_reconstruct_1d over the owned cells (nCellsSolve), the edge field u
extern "C" __global__ void li_reconstruct_1d_v842(
    rk *x, rk *y, rk *z, rk *zonal, rk *meridional, const rk *u, const rk *coeffs,
    const int *nEdgesOnCell, const int *edgesOnCell, const rk *latCell, const rk *lonCell,
    int onSphere, int nCellsSolve, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    rk rx = LI_C(0.0), ry = LI_C(0.0), rz = LI_C(0.0);
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const rk ue = u[edgesOnCell[iCell * maxEdges + i] - 1];
        const rk *c = coeffs + (iCell * maxEdges + i) * 3;
        rx = A(rx, M(c[0], ue));
        ry = A(ry, M(c[1], ue));
        rz = A(rz, M(c[2], ue));
    }
    x[iCell] = rx;
    y[iCell] = ry;
    z[iCell] = rz;
    if (onSphere) {
        const rk clat = LI_COS(latCell[iCell]), slat = LI_SIN(latCell[iCell]);
        const rk clon = LI_COS(lonCell[iCell]), slon = LI_SIN(lonCell[iCell]);
        zonal[iCell] = A(-M(rx, slon), M(ry, clon));
        meridional[iCell] = A(-M(A(M(rx, clon), M(ry, slon)), slat), M(rz, clat));
    } else {
        zonal[iCell] = rx;
        meridional[iCell] = ry;
    }
}

// mpas_reconstruct_2d over the owned cells: the first nVertLevels levels of
// the (nVertInterfaces)-level velocity, as MALI calls it (the bottom
// interface keeps its words)
extern "C" __global__ void li_reconstruct_2d_v842(
    rk *x, rk *y, rk *z, rk *zonal, rk *meridional, const rk *u, const rk *coeffs,
    const int *nEdgesOnCell, const int *edgesOnCell, const rk *latCell, const rk *lonCell,
    int onSphere, int nCellsSolve, int maxEdges, int nVertLevels, int stride)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    rk *cx = x + iCell * stride, *cy = y + iCell * stride, *cz = z + iCell * stride;
    for (int k = 0; k < nVertLevels; ++k) {
        cx[k] = LI_C(0.0);
        cy[k] = LI_C(0.0);
        cz[k] = LI_C(0.0);
    }
    for (int i = 0; i < nEdgesOnCell[iCell]; ++i) {
        const rk *ue = u + (edgesOnCell[iCell * maxEdges + i] - 1) * stride;
        const rk *c = coeffs + (iCell * maxEdges + i) * 3;
        for (int k = 0; k < nVertLevels; ++k) {
            cx[k] = A(cx[k], M(c[0], ue[k]));
            cy[k] = A(cy[k], M(c[1], ue[k]));
            cz[k] = A(cz[k], M(c[2], ue[k]));
        }
    }
    rk *zo = zonal + iCell * stride, *me = meridional + iCell * stride;
    if (onSphere) {
        const rk clat = LI_COS(latCell[iCell]), slat = LI_SIN(latCell[iCell]);
        const rk clon = LI_COS(lonCell[iCell]), slon = LI_SIN(lonCell[iCell]);
        for (int k = 0; k < nVertLevels; ++k) {
            zo[k] = A(-M(cx[k], slon), M(cy[k], clon));
            me[k] = A(-M(A(M(cx[k], clon), M(cy[k], slon)), slat), M(cz[k], clat));
        }
    } else {
        for (int k = 0; k < nVertLevels; ++k) {
            zo[k] = cx[k];
            me[k] = cy[k];
        }
    }
}

// li_calculate_apparent_diffusivity's cell loop over nCells: the apparent
// diffusivity and each cell's diffusive CFL step (the driver takes the
// minimum). slopeX/slopeY are the reconstructed normal slope; uZonal and
// uMeridional the velocity's (nVertInterfaces levels)
extern "C" __global__ void li_apparent_diffusivity_v842(
    rk *apparentDiffusivity, rk *cellDt, const rk *slopeX, const rk *slopeY, const rk *uZonal,
    const rk *uMeridional, const rk *layerThickness, const int *cellMask, const int *nEdgesOnCell,
    const rk *dcEdge, int nCells, int nVertLevels)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const rk bigNumber = LI_C(1.0e16), smallNumber = LI_C(1.0e-36);
    const int nvi = nVertLevels + 1;
    const rk s1 = slopeX[iCell], s2 = slopeY[iCell];
    const rk magnitude = A(LI_SQRT(A(M(s1, s1), M(s2, s2))), smallNumber);
    const rk *u1 = uZonal + iCell * nvi, *u2 = uMeridional + iCell * nvi;
    rk max1 = u1[0], max2 = u2[0];
    for (int k = 1; k < nvi; ++k) {
        max1 = li_max(max1, u1[k]);
        max2 = li_max(max2, u2[k]);
    }
    rk diffusivity;
    if (magnitude < LI_C(1.0e-4) && li_max(max1, max2) < LI_C(3.18e-8)) {
        diffusivity = LI_C(0.0);
    } else {
        rk flux = LI_C(0.0);
        for (int k = 0; k < nVertLevels; ++k) {
            const rk v1 = M(A(u1[k], u1[k + 1]), LI_C(0.5));
            const rk v2 = M(A(u2[k], u2[k + 1]), LI_C(0.5));
            flux = A(flux, D(M(S(M(M(LI_C(-1.0), s1), v1), M(s2, v2)), layerThickness[iCell * nVertLevels + k]),
                             magnitude));
        }
        diffusivity = D(LI_ABS(flux), magnitude);
    }
    apparentDiffusivity[iCell] = diffusivity;
    rk dt = bigNumber;
    const int mask = cellMask[iCell];
    if (li_is_grounded_ice(mask) && li_is(mask, LI_DYNAMIC_ICE)) {
        // minval(dcEdge(1:nEdgesOnCell(iCell))), as written: the first
        // nEdgesOnCell edges of the mesh, not the cell's
        rk dCell = dcEdge[0];
        for (int i = 1; i < nEdgesOnCell[iCell]; ++i) dCell = li_min(dCell, dcEdge[i]);
        dt = D(M(LI_C(0.5), M(dCell, dCell)), A(diffusivity, smallNumber));
    }
    cellDt[iCell] = dt;
}

// the minimum of n words in the Fortran's order (one thread): min(a, b) is
// order-free on finite words, and this keeps the first of equal ones
extern "C" __global__ void li_min_v842(rk *out, const rk *words, rk start, int n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    rk m = start;
    for (int i = 0; i < n; ++i) m = li_min(m, words[i]);
    out[0] = m;
}

// calculate_layerThicknessEdge: the upwind layer thickness on every edge
extern "C" __global__ void li_layer_thickness_edge_v842(
    rk *layerThicknessEdge, const rk *layerThickness, const rk *normalVelocity, const int *cellsOnEdge,
    int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    for (int k = 0; k < nVertLevels; ++k) {
        const rk velSign = LI_SIGN1(normalVelocity[iEdge * (nVertLevels + 1) + k]);
        layerThicknessEdge[iEdge * nVertLevels + k] =
            li_max(M(velSign, layerThickness[cell1 * nVertLevels + k]),
                   M(M(velSign, LI_C(-1.0)), layerThickness[cell2 * nVertLevels + k]));
    }
}

// advection_solver: thickness below zero is set to zero
extern "C" __global__ void li_clamp_thickness_v842(rk *thickness, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (thickness[i] < LI_C(0.0)) thickness[i] = LI_C(0.0);
}

// li_velocity_solve: the normal surface slope on dynamic edges
extern "C" __global__ void li_normal_slope_edge_v842(
    rk *normalSlopeEdge, const rk *upperSurface, const int *edgeMask, const int *cellsOnEdge,
    const rk *dcEdge, int nEdges)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    if (li_is(edgeMask[iEdge], LI_DYNAMIC_ICE)) {
        const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
        normalSlopeEdge[iEdge] = D(S(upperSurface[cell1], upperSurface[cell2]), dcEdge[iEdge]);
    } else {
        normalSlopeEdge[iEdge] = LI_C(0.0);
    }
}

// li_velocity_solve: no velocity on an 'uphill' dynamic-margin edge
extern "C" __global__ void li_uphill_margin_v842(
    rk *normalVelocity, const int *edgeMask, const int *cellMask, const rk *upperSurface,
    const int *cellsOnEdge, int nEdgesSolve, int nVertInterfaces)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdgesSolve) return;
    if (!li_is(edgeMask[iEdge], LI_DYNAMIC_MARGIN)) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    if ((li_is(cellMask[cell1], LI_DYNAMIC_ICE) && upperSurface[cell2] > upperSurface[cell1]) ||
        (li_is(cellMask[cell2], LI_DYNAMIC_ICE) && upperSurface[cell1] > upperSurface[cell2])) {
        for (int k = 0; k < nVertInterfaces; ++k) normalVelocity[iEdge * nVertInterfaces + k] = LI_C(0.0);
    }
}

// li_velocity_solve with config_velocity_solver 'none': the normal velocity
// from the cells' reconstructed velocity
extern "C" __global__ void li_velocity_from_cells_v842(
    rk *normalVelocity, const rk *uX, const rk *uY, const rk *edgeNormalVectors, const int *cellsOnEdge,
    int nEdges, int nVertInterfaces)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int cell1 = cellsOnEdge[2 * iEdge] - 1, cell2 = cellsOnEdge[2 * iEdge + 1] - 1;
    const rk n1 = edgeNormalVectors[3 * iEdge], n2 = edgeNormalVectors[3 * iEdge + 1];
    for (int k = 0; k < nVertInterfaces; ++k) {
        const rk xv = M(LI_C(0.5), A(uX[cell1 * nVertInterfaces + k], uX[cell2 * nVertInterfaces + k]));
        const rk yv = M(LI_C(0.5), A(uY[cell1 * nVertInterfaces + k], uY[cell2 * nVertInterfaces + k]));
        normalVelocity[iEdge * nVertInterfaces + k] = A(M(xv, n1), M(yv, n2));
    }
}

// the surface and basal speeds over every cell
extern "C" __global__ void li_speeds_v842(
    rk *surfaceSpeed, rk *basalSpeed, const rk *uX, const rk *uY, int nCells, int nVertInterfaces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk *x = uX + i * nVertInterfaces, *y = uY + i * nVertInterfaces;
    const int b = nVertInterfaces - 1;
    surfaceSpeed[i] = LI_SQRT(A(M(x[0], x[0]), M(y[0], y[0])));
    basalSpeed[i] = LI_SQRT(A(M(x[b], x[b]), M(y[b], y[b])));
}

// li_compute_gradient_2d of the surface level of a (nVertInterfaces)-level
// cell field (stride words apart)
extern "C" __global__ void li_gradient_2d_v842(
    rk *xGrad, rk *yGrad, const rk *field, const rk *coeffs, const int *nEdgesOnCell, const int *edgesOnCell,
    const int *cellsOnCell, const int *cellsOnEdge, const rk *dcEdge, int nCells, int maxEdges, int stride)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk gx = LI_C(0.0), gy = LI_C(0.0);
    for (int j = 0; j < nEdgesOnCell[iCell]; ++j) {
        const int neighbour = cellsOnCell[iCell * maxEdges + j];
        rk normalGrad = LI_C(0.0);
        if (neighbour >= 1 && neighbour <= nCells) {
            const int iEdge = edgesOnCell[iCell * maxEdges + j] - 1;
            const rk sign = (iCell + 1 == cellsOnEdge[2 * iEdge]) ? LI_C(1.0) : LI_C(-1.0);
            normalGrad = D(M(sign, S(field[(neighbour - 1) * stride], field[iCell * stride])), dcEdge[iEdge]);
        }
        const rk *c = coeffs + (iCell * maxEdges + j) * 3;
        gx = A(gx, M(c[0], normalGrad));
        gy = A(gy, M(c[1], normalGrad));
    }
    xGrad[iCell] = gx;
    yGrad[iCell] = gy;
}

// calculate_strain_rates' principal strain rates and (withTheta) angle
extern "C" __global__ void li_strain_rates_v842(
    rk *eTheta, rk *eMax, rk *eMin, const rk *exx, const rk *eyy, const rk *exy, const rk *eyx, int withTheta,
    int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (withTheta) eTheta[i] = M(LI_C(0.5), LI_ATAN(D(A(exy[i], eyx[i]), A(S(exx[i], eyy[i]), LI_C(1.0e-42)))));
    const rk half = M(LI_C(0.5), S(exx[i], eyy[i]));
    const rk quarter = M(LI_C(0.25), A(exy[i], eyx[i]));
    const rk mean = M(LI_C(0.5), A(exx[i], eyy[i]));
    eMax[i] = A(mean, LI_SQRT(A(M(half, half), M(quarter, quarter))));
    // eMin's expression is evaluated anew, in the same order
    const rk half2 = M(LI_C(0.5), S(exx[i], eyy[i]));
    const rk quarter2 = M(LI_C(0.25), A(exy[i], eyx[i]));
    eMin[i] = S(M(LI_C(0.5), A(exx[i], eyy[i])), LI_SQRT(A(M(half2, half2), M(quarter2, quarter2))));
}

// remove_fast_ice: floating ice faster than 100 km/yr calves
extern "C" __global__ void li_remove_fast_ice_v842(
    rk *thickness, rk *calvingThickness, const rk *surfaceSpeed, const int *cellMask, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (li_is(cellMask[i], LI_FLOATING) && surfaceSpeed[i] > LI_C(0.00318471337)) {
        calvingThickness[i] = A(calvingThickness[i], thickness[i]);
        thickness[i] = LI_C(0.0);
    }
}

// li_calculate_diagnostic_vars: dH/dt (m/yr) and the next step's old thickness
extern "C" __global__ void li_dhdt_v842(rk *dHdt, rk *thicknessOld, const rk *thickness, rk deltat, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    dHdt[i] = M(D(S(thickness[i], thicknessOld[i]), deltat), LI_SCYR);
    thicknessOld[i] = thickness[i];
}
