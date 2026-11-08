// MPAS-Ocean v8.4.2: the horizontal pressure gradient and the horizontal
// momentum mixing (del2, Leith, del4) tendencies.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_vel_pressure_grad.F:80-411,625-664
//     (ocn_vel_pressure_grad_tend, pGrad_Jacobian_common_level, _init)
//   src/core_ocean/shared/mpas_ocn_vel_hmix_del2.F:81-194,385-415 (ocn_vel_hmix_del2_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_hmix_leith.F:84-204,219-243 (ocn_vel_hmix_leith_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_hmix_del4.F:82-279,506-534 (ocn_vel_hmix_del4_tend, _init)
//   src/core_ocean/shared/mpas_ocn_vel_hmix.F:82-217 (ocn_vel_hmix_tend, _init)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both of MPAS's precisions, as tend_v842.cu: RKIND = 4 by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction). An integer factor (edgeMask, edgeSign*) is converted to real
// first. A leading unary minus (-a*b) negates the product; negation is exact,
// so where it is applied does not change a word.
//
// Layout as tend_v842.cu: a Fortran array a(nVertLevels, nX) is
// a[iX*nVertLevels + k]; x(2, nEdges) is x[iEdge*2 + i]; edgesOnVertex
// (vertexDegree, nVertices) is e[iVertex*vertexDegree + i]; tracers(nTracers,
// nVertLevels, nCells) is t[(iCell*nVertLevels + k)*nTracers + iTracer].
// Connectivity is MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define ABS(a) fabs(a)
// mpas_constants pii and gfortran's folded sqrt(3.0_RKIND), as RKIND words
#define PII 3.141592653589793
#define SQRT3 1.7320508075688772
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define ABS(a) fabsf(a)
#define PII 3.14159274f
#define SQRT3 1.73205078f
#endif

// config_pressure_gradient_type codes (pgrad_hmix_v842.PGRAD_TYPES)
#define PGRAD_PRESSURE_AND_ZMID 1
#define PGRAD_MONTGOMERY 2
#define PGRAD_MONTGOMERY_AND_DENSITY 3
#define PGRAD_JACOBIAN_FROM_DENSITY 4
#define PGRAD_JACOBIAN_FROM_TS 5

// pGrad_Jacobian_common_level at one level k >= 2 (0-based kk = k-1 >= 1):
// JacobianDxDs(k) from columns rho1/rho2, z1/z2.
__device__ __forceinline__ rk jacobian_common_level(const rk *rho1, const rk *rho2, int rs,
                                                    const rk *z1, const rk *z2, int kk, rk gamma)
{
    const rk z1k = z1[kk], z1m = z1[kk - 1], z2k = z2[kk], z2m = z2[kk - 1];
    const rk area = M((rk)0.5, S(A(S(z1m, z1k), z2m), z2k));
    const rk zStar = D(S(M(z2m, z1m), M(z2k, z1k)), S(A(S(z2m, z2k), z1m), z1k));
    const rk zC = M((rk)0.25, A(A(A(z1k, z1m), z2k), z2m));
    const rk zGamma = A(M(S((rk)1.0, gamma), zStar), M(gamma, zC));
    const rk rhoL = D(A(M(rho1[kk * rs], S(z1m, zGamma)), M(rho1[(kk - 1) * rs], S(zGamma, z1k))), S(z1m, z1k));
    const rk rhoR = D(A(M(rho2[kk * rs], S(z2m, zGamma)), M(rho2[(kk - 1) * rs], S(zGamma, z2k))), S(z2m, z2k));
    return M(area, S(rhoL, rhoR));
}

// ocn_vel_pressure_grad_tend: one thread per edge. density0Inv = 1/rho_sw and
// gdensity0Inv = gravity/rho_sw are the RKIND words ocn_vel_pressure_grad_init
// forms.
extern "C" __global__ void ocn_vel_pressure_grad_tend_v842(
    rk *tend, const rk *pressure, const rk *montgomeryPotential, const rk *zMid, const rk *density,
    const rk *potentialDensity, const rk *tracers, const rk *thermalExpansion, const rk *salineContraction,
    const int *maxLevelEdgeTop, const int *cellsOnEdge, const rk *dcEdge, const int *edgeMask,
    int pgradType, rk density0Inv, rk gdensity0Inv, rk gamma, int indexT, int indexS, int nTracers,
    int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int nl = nVertLevels;
    const int c1 = (cellsOnEdge[iEdge * 2] - 1) * nl;
    const int c2 = (cellsOnEdge[iEdge * 2 + 1] - 1) * nl;
    const rk invdcEdge = D((rk)1.0, dcEdge[iEdge]);
    const int kmax = maxLevelEdgeTop[iEdge];
    const int base = iEdge * nl;

    if (pgradType == PGRAD_PRESSURE_AND_ZMID) {
        for (int k = 0; k < kmax; ++k) {
            const rk x = S(-M(density0Inv, S(pressure[c2 + k], pressure[c1 + k])),
                           M(M(M(gdensity0Inv, (rk)0.5), A(density[c1 + k], density[c2 + k])),
                             S(zMid[c2 + k], zMid[c1 + k])));
            tend[base + k] = A(tend[base + k], M(M((rk)edgeMask[base + k], invdcEdge), x));
        }
    } else if (pgradType == PGRAD_MONTGOMERY) {
        for (int k = 0; k < kmax; ++k) {
            const rk x = -S(montgomeryPotential[c2 + k], montgomeryPotential[c1 + k]);
            tend[base + k] = A(tend[base + k], M(M((rk)edgeMask[base + k], invdcEdge), x));
        }
    } else if (pgradType == PGRAD_MONTGOMERY_AND_DENSITY) {
        for (int k = 0; k < kmax; ++k) {
            const rk x = A(-S(montgomeryPotential[c2 + k], montgomeryPotential[c1 + k]),
                           M(M((rk)0.5, A(pressure[c1 + k], pressure[c2 + k])),
                             S(D((rk)1.0, potentialDensity[c2 + k]), D((rk)1.0, potentialDensity[c1 + k]))));
            tend[base + k] = A(tend[base + k], M(M((rk)edgeMask[base + k], invdcEdge), x));
        }
    } else {
        // Jacobian_from_density / Jacobian_from_TS: layer 1 from pressure and
        // zMid, whatever maxLevelEdgeTop is (the Fortran sets k = 1 outside
        // the level loop), then the Jacobian sum down to maxLevelEdgeTop
        const rk x = S(-M(density0Inv, S(pressure[c2], pressure[c1])),
                       M(M(M(gdensity0Inv, (rk)0.5), A(density[c1], density[c2])), S(zMid[c2], zMid[c1])));
        rk pGrad = M(M((rk)edgeMask[base], invdcEdge), x);
        tend[base] = A(tend[base], pGrad);
        const rk *z1 = zMid + c1, *z2 = zMid + c2;
        for (int kk = 1; kk < kmax; ++kk) {
            rk jacobianDxDs;
            if (pgradType == PGRAD_JACOBIAN_FROM_DENSITY) {
                jacobianDxDs = jacobian_common_level(density + c1, density + c2, 1, z1, z2, kk, gamma);
            } else {
                const rk *t1 = tracers + c1 * nTracers, *t2 = tracers + c2 * nTracers;
                const rk jTz = jacobian_common_level(t1 + (indexT - 1), t2 + (indexT - 1), nTracers, z1, z2, kk, gamma);
                const rk jSz = jacobian_common_level(t1 + (indexS - 1), t2 + (indexS - 1), nTracers, z1, z2, kk, gamma);
                const rk alpha = M((rk)0.25,
                    A(A(A(M(density[c1 + kk], thermalExpansion[c1 + kk]),
                          M(density[c1 + kk - 1], thermalExpansion[c1 + kk - 1])),
                        M(density[c2 + kk], thermalExpansion[c2 + kk])),
                      M(density[c2 + kk - 1], thermalExpansion[c2 + kk - 1])));
                const rk beta = M((rk)0.25,
                    A(A(A(M(density[c1 + kk], salineContraction[c1 + kk]),
                          M(density[c1 + kk - 1], salineContraction[c1 + kk - 1])),
                        M(density[c2 + kk], salineContraction[c2 + kk])),
                      M(density[c2 + kk - 1], salineContraction[c2 + kk - 1])));
                jacobianDxDs = A(-M(alpha, jTz), M(beta, jSz));
            }
            pGrad = A(pGrad, M(M(gdensity0Inv, jacobianDxDs), invdcEdge));
            tend[base + kk] = A(tend[base + kk], pGrad);
        }
    }
}

// ocn_vel_hmix_del2_tend (leith = 0) and ocn_vel_hmix_leith_tend (leith = 1):
// one thread per edge. coef is config_mom_del2 or config_Leith_parameter;
// leithDx and visc2Max are config_Leith_dx and config_Leith_visc2_max.
extern "C" __global__ void ocn_vel_hmix_del2_leith_tend_v842(
    rk *tend, rk *viscosity, const rk *divergence, const rk *relativeVorticity,
    const int *maxLevelEdgeTop, const int *cellsOnEdge, const int *verticesOnEdge,
    const rk *meshScaling, const rk *dcEdge, const rk *dvEdge, const int *edgeMask,
    int leith, rk coef, rk leithDx, rk visc2Max, int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int nl = nVertLevels;
    const int c1 = (cellsOnEdge[iEdge * 2] - 1) * nl, c2 = (cellsOnEdge[iEdge * 2 + 1] - 1) * nl;
    const int v1 = (verticesOnEdge[iEdge * 2] - 1) * nl, v2 = (verticesOnEdge[iEdge * 2 + 1] - 1) * nl;
    const rk invLength1 = D((rk)1.0, dcEdge[iEdge]);
    const rk invLength2 = D((rk)1.0, dvEdge[iEdge]);
    const int kmax = maxLevelEdgeTop[iEdge];
    const int base = iEdge * nl;
    for (int k = 0; k < kmax; ++k) {
        const rk dRv = S(relativeVorticity[v2 + k], relativeVorticity[v1 + k]);
        const rk uDiffusion = S(M(S(divergence[c2 + k], divergence[c1 + k]), invLength1), M(dRv, invLength2));
        rk visc2;
        if (leith) {
            // x**3 is libgcc's __powisf2/__powidf2(x, 3) at -O0: x*(x*x)
            const rk p = D(M(M(coef, leithDx), meshScaling[iEdge]), (rk)PII);
            visc2 = M(M(M(M(p, M(p, p)), ABS(dRv)), invLength1), (rk)SQRT3);
            // min(visc2, max): minss/minsd keeps visc2 unless max is smaller
            visc2 = visc2 < visc2Max ? visc2 : visc2Max;
        } else {
            visc2 = M(coef, meshScaling[iEdge]);
        }
        tend[base + k] = A(tend[base + k], M(M((rk)edgeMask[base + k], visc2), uDiffusion));
        viscosity[base + k] = A(viscosity[base + k], visc2);
    }
}

// ocn_vel_hmix_del4_tend, step 1 (every edge, nEdgesArray(3)): delsq_u,
// zero below maxLevelEdgeTop.
extern "C" __global__ void ocn_vel_hmix_del4_delsq_u_v842(
    rk *delsqU, const rk *divergence, const rk *relativeVorticity, const int *maxLevelEdgeTop,
    const int *cellsOnEdge, const int *verticesOnEdge, const rk *dcEdge, const rk *dvEdge,
    int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int nl = nVertLevels;
    const int c1 = (cellsOnEdge[iEdge * 2] - 1) * nl, c2 = (cellsOnEdge[iEdge * 2 + 1] - 1) * nl;
    const int v1 = (verticesOnEdge[iEdge * 2] - 1) * nl, v2 = (verticesOnEdge[iEdge * 2 + 1] - 1) * nl;
    const rk invDcEdge = D((rk)1.0, dcEdge[iEdge]);
    const rk dv = dvEdge[iEdge], dcq = M((rk)0.25, dcEdge[iEdge]);
    // max(dvEdge, 0.25*dcEdge): maxss/maxsd keeps 0.25*dcEdge unless dvEdge is larger
    const rk invDvEdge = D((rk)1.0, dv > dcq ? dv : dcq);
    const int kmax = maxLevelEdgeTop[iEdge];
    const int base = iEdge * nl;
    for (int k = 0; k < nl; ++k) {
        delsqU[base + k] = k < kmax
            ? S(M(S(divergence[c2 + k], divergence[c1 + k]), invDcEdge),
                M(S(relativeVorticity[v2 + k], relativeVorticity[v1 + k]), invDvEdge))
            : (rk)0.0;
    }
}

// step 2 (nVerticesArray(2)): delsq_relativeVorticity
extern "C" __global__ void ocn_vel_hmix_del4_delsq_vorticity_v842(
    rk *delsqRelativeVorticity, const rk *delsqU, const int *maxLevelVertexTop, const int *edgesOnVertex,
    const int *edgeSignOnVertex, const rk *dcEdge, const rk *areaTriangle,
    int nVertices, int nVertLevels, int vertexDegree)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices) return;
    const int nl = nVertLevels;
    const int base = iVertex * nl;
    for (int k = 0; k < nl; ++k) delsqRelativeVorticity[base + k] = (rk)0.0;
    const rk invAreaTri1 = D((rk)1.0, areaTriangle[iVertex]);
    const int kmax = maxLevelVertexTop[iVertex];
    for (int i = 0; i < vertexDegree; ++i) {
        const int iEdge = edgesOnVertex[iVertex * vertexDegree + i] - 1;
        const rk sign = (rk)edgeSignOnVertex[iVertex * vertexDegree + i];
        for (int k = 0; k < kmax; ++k) {
            delsqRelativeVorticity[base + k] = A(delsqRelativeVorticity[base + k],
                M(M(M(sign, dcEdge[iEdge]), delsqU[iEdge * nl + k]), invAreaTri1));
        }
    }
}

// step 3 (nCellsArray(2)): delsq_divergence
extern "C" __global__ void ocn_vel_hmix_del4_delsq_divergence_v842(
    rk *delsqDivergence, const rk *delsqU, const int *maxLevelCell, const int *nEdgesOnCell,
    const int *edgesOnCell, const int *edgeSignOnCell, const rk *dvEdge, const rk *areaCell,
    int nCells, int nVertLevels, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int nl = nVertLevels;
    const int base = iCell * nl;
    for (int k = 0; k < nl; ++k) delsqDivergence[base + k] = (rk)0.0;
    const rk invAreaCell1 = D((rk)1.0, areaCell[iCell]);
    const int kmax = maxLevelCell[iCell];
    const int n = nEdgesOnCell[iCell];
    for (int i = 0; i < n; ++i) {
        const int iEdge = edgesOnCell[iCell * maxEdges + i] - 1;
        const rk sign = (rk)edgeSignOnCell[iCell * maxEdges + i];
        for (int k = 0; k < kmax; ++k) {
            delsqDivergence[base + k] = S(delsqDivergence[base + k],
                M(M(M(sign, dvEdge[iEdge]), delsqU[iEdge * nl + k]), invAreaCell1));
        }
    }
}

// step 4 (nEdgesArray(1)): the biharmonic tendency
extern "C" __global__ void ocn_vel_hmix_del4_tend_v842(
    rk *tend, const rk *delsqDivergence, const rk *delsqRelativeVorticity, const int *maxLevelEdgeTop,
    const int *cellsOnEdge, const int *verticesOnEdge, const rk *meshScalingDel4, const rk *dcEdge,
    const rk *dvEdge, const int *edgeMask, rk momDel4, rk divFactor, int nEdges, int nVertLevels)
{
    int iEdge = blockIdx.x * blockDim.x + threadIdx.x;
    if (iEdge >= nEdges) return;
    const int nl = nVertLevels;
    const int c1 = (cellsOnEdge[iEdge * 2] - 1) * nl, c2 = (cellsOnEdge[iEdge * 2 + 1] - 1) * nl;
    const int v1 = (verticesOnEdge[iEdge * 2] - 1) * nl, v2 = (verticesOnEdge[iEdge * 2 + 1] - 1) * nl;
    const rk invDcEdge = D((rk)1.0, dcEdge[iEdge]);
    const rk invDvEdge = D((rk)1.0, dvEdge[iEdge]);
    const rk rTmp = M(momDel4, meshScalingDel4[iEdge]);
    const int kmax = maxLevelEdgeTop[iEdge];
    const int base = iEdge * nl;
    for (int k = 0; k < kmax; ++k) {
        const rk uDiffusion = S(M(M(divFactor, S(delsqDivergence[c2 + k], delsqDivergence[c1 + k])), invDcEdge),
                                M(S(delsqRelativeVorticity[v2 + k], delsqRelativeVorticity[v1 + k]), invDvEdge));
        tend[base + k] = S(tend[base + k], M(M((rk)edgeMask[base + k], uDiffusion), rTmp));
    }
}
