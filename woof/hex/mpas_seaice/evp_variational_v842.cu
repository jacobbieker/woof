// MPAS-Seaice v8.4.2: EVP internal stress, variational discretisation.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_velocity_solver_constitutive_relation.F:28-49,64-294
//   src/core_seaice/shared/mpas_seaice_velocity_solver_variational.F:271-651,665-806
// (Adrian K. Turner, LANL, 2013-2015). Copyright (c) 2013-2019, Los Alamos
// National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined (the column package is
// double, so production MPAS-Seaice is built double). Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off: left to right, parentheses kept, no contraction); x**2 is
// x*x as gfortran expands it. max/min are written as the SSE maxss/minss the
// reference object executes (operand order read from its disassembly), so a
// NaN or signed-zero operand takes the reference's word. A literal written
// _RKIND is RL(): decimal rounded once to RKIND.
//
// Layout: a Fortran array a(maxEdges, nCells) is a[iCell*maxEdges + i]; a
// basis array b(maxEdges, maxEdges, nCells) is b[(iCell*maxEdges + j)*maxEdges + i];
// cellsOnVertex(vertexDegree, nVertices) is c[iVertex*vertexDegree + i].
// Connectivity is MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define FABS(a) fabs(a)
#define RL(x) x                              // x_RKIND
#define SI_HUGE 1.7976931348623157e308       // huge(1.0_RKIND)
#define SI_QNAN __longlong_as_double(0x7ff8000000000000LL)
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define FABS(a) fabsf(a)
#define RL(x) x##f                           // x_RKIND: decimal rounded once to binary32
#define SI_HUGE 3.40282347e38f               // huge(1.0_RKIND)
#define SI_QNAN __int_as_float(0x7fc00000)
#endif

// constitutive_relation.F:29-46 (module parameters)
#define SI_ECCENTRICITY_SQUARED RL(4.0)          // eccentricity**2, eccentricity = 2.0
#define SI_DAMPING_TIMESCALE_PARAMETER RL(0.36)
#define SI_PUNY RL(1.0e-11)
#define SI_DAMPING_RATIO_DENOMINATOR RL(0.86)
#define SI_DAMPING_RATIO RL(5.5e-3)

// SSE maxss/maxsd(dst=a, src=b): a > b ? a : b (b when either is NaN)
__device__ __forceinline__ rk si_maxss(rk a, rk b) { return a > b ? a : b; }
// SSE minss/minsd(dst=a, src=b): a < b ? a : b (b when either is NaN)
__device__ __forceinline__ rk si_minss(rk a, rk b) { return a < b ? a : b; }

// seaice_init_evp, constitutive_relation.F:64-136, for one block on one rank.
// consts[0] dampingTimescale, consts[1] numericalInertiaCoefficient,
// consts[2] evpDampingCriterion (set by the model, read by no v8.4.2 routine).
extern "C" __global__ void seaice_init_evp_v842(
    const rk *dvEdge, int nEdgesSolve, rk dynamicsTimeStep,
    rk elasticTimeStep, rk *consts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    rk dampingTimescale = M(SI_DAMPING_TIMESCALE_PARAMETER, dynamicsTimeStep);
    rk evpDampingCriterion = D(M(RL(1230.0), dampingTimescale),
                                  M(elasticTimeStep, elasticTimeStep));
    // minval(dvEdge(1:nEdgesSolve)) as gfortran evaluates it: the first
    // element not above huge() starts the scan (NaN if none is; huge() for an
    // empty section), then a strict less-than scan
    rk mv = SI_HUGE;
    int i = 0;
    if (nEdgesSolve > 0) {
        mv = SI_QNAN;
        for (; i < nEdgesSolve; ++i) {
            if (dvEdge[i] <= SI_HUGE) { mv = dvEdge[i]; ++i; break; }
        }
        for (; i < nEdgesSolve; ++i) {
            if (dvEdge[i] < mv) mv = dvEdge[i];
        }
    }
    rk dvEdgeMin = si_minss(mv, RL(1.0e30));
    rk dvEdgeMinGlobal = dvEdgeMin;  // mpas_dmpar_min_real on one rank
    rk gamma = M(M(RL(0.25), RL(1.0e11)), dynamicsTimeStep);
    rk numericalInertiaCoefficient =
        D(M(M(M(RL(2.0), SI_DAMPING_RATIO_DENOMINATOR), SI_DAMPING_RATIO), gamma),
          M(dvEdgeMinGlobal, dvEdgeMinGlobal));
    consts[0] = dampingTimescale;
    consts[1] = numericalInertiaCoefficient;
    consts[2] = evpDampingCriterion;
}

// seaice_strain_tensor_variational, variational.F:271-370: one thread per cell
extern "C" __global__ void seaice_strain_tensor_variational_v842(
    rk *strain11, rk *strain22, rk *strain12,
    const rk *uVelocity, const rk *vVelocity,
    const rk *basisGradientU, const rk *basisGradientV,
    const rk *tanLatVertexRotatedOverRadius, const int *solveStress,
    const int *nEdgesOnCell, const int *verticesOnCell, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells || solveStress[iCell] != 1) return;
    const int base = iCell * maxEdges;
    for (int k = 0; k < maxEdges; ++k) {
        strain11[base + k] = RL(0.0);
        strain22[base + k] = RL(0.0);
        strain12[base + k] = RL(0.0);
    }
    const int n = nEdgesOnCell[iCell];
    for (int iGradientVertex = 0; iGradientVertex < n; ++iGradientVertex) {
        rk s11 = RL(0.0), s22 = RL(0.0), s12 = RL(0.0);
        const int gbase = (base + iGradientVertex) * maxEdges;
        for (int iBasisVertex = 0; iBasisVertex < n; ++iBasisVertex) {
            const int iVertex = verticesOnCell[base + iBasisVertex] - 1;
            const rk u = uVelocity[iVertex], v = vVelocity[iVertex];
            const rk gU = basisGradientU[gbase + iBasisVertex];
            const rk gV = basisGradientV[gbase + iBasisVertex];
            s11 = A(s11, M(u, gU));
            s22 = A(s22, M(v, gV));
            s12 = A(s12, M(RL(0.5), A(M(u, gV), M(v, gU))));
        }
        // metric terms
        const int jVertex = verticesOnCell[base + iGradientVertex] - 1;
        const rk t = tanLatVertexRotatedOverRadius[jVertex];
        s11 = S(s11, M(vVelocity[jVertex], t));
        s12 = A(s12, M(M(uVelocity[jVertex], t), RL(0.5)));
        strain11[base + iGradientVertex] = s11;
        strain22[base + iGradientVertex] = s22;
        strain12[base + iGradientVertex] = s12;
    }
}

// seaice_evp_constitutive_relation (constitutive_relation.F:150-216) and
// seaice_evp_constitutive_relation_revised (:230-294), one stress point
__device__ __forceinline__ void si_evp_constitutive_relation(
    rk *stress11, rk *stress22, rk *stress12,
    rk strain11, rk strain22, rk strain12,
    rk icePressure, rk *replacementPressure,
    rk dtElastic, rk dampingTimescale, bool revised)
{
    // convert from stress11 to stress1 etc
    const rk strainDivergence = A(strain11, strain22);
    const rk strainTension    = S(strain11, strain22);
    const rk strainShearing   = M(strain12, RL(2.0));
    rk stress1 = A(*stress11, *stress22);
    rk stress2 = S(*stress11, *stress22);
    // perform the constituitive relation
    const rk Delta = SQRT(A(M(strainDivergence, strainDivergence),
        D(A(M(strainTension, strainTension), M(strainShearing, strainShearing)),
          SI_ECCENTRICITY_SQUARED)));
    rk pressureCoefficient = D(icePressure, si_maxss(Delta, SI_PUNY));
    *replacementPressure = M(pressureCoefficient, Delta);
    rk denominator;
    if (!revised) {
        pressureCoefficient = D(M(pressureCoefficient, dtElastic), M(RL(2.0), dampingTimescale));
        denominator = A(RL(1.0), D(M(RL(0.5), dtElastic), dampingTimescale));
    } else {
        pressureCoefficient = D(M(M(pressureCoefficient, RL(2.0)), SI_DAMPING_RATIO),
                                SI_DAMPING_RATIO_DENOMINATOR);
        denominator = A(RL(1.0), D(M(RL(2.0), SI_DAMPING_RATIO), SI_DAMPING_RATIO_DENOMINATOR));
    }
    stress1 = D(A(stress1, M(pressureCoefficient, S(strainDivergence, Delta))), denominator);
    stress2 = D(A(stress2, M(D(pressureCoefficient, SI_ECCENTRICITY_SQUARED), strainTension)),
                denominator);
    *stress12 = D(A(*stress12, M(M(D(pressureCoefficient, SI_ECCENTRICITY_SQUARED),
                                   strainShearing), RL(0.5))), denominator);
    // convert back
    *stress11 = M(RL(0.5), A(stress1, stress2));
    *stress22 = M(RL(0.5), S(stress1, stress2));
}

// seaice_stress_tensor_variational, variational.F:384-507: one thread per cell.
// consts[0] is seaice_init_evp's dampingTimescale.
extern "C" __global__ void seaice_stress_tensor_variational_v842(
    rk *stress11, rk *stress22, rk *stress12,
    const rk *strain11, const rk *strain22, const rk *strain12,
    const rk *icePressure, rk *replacementPressure,
    const int *solveStress, const int *nEdgesOnCell, const rk *consts,
    rk dtElastic, int revisedEVP, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * maxEdges;
    if (!revisedEVP) {
        for (int k = 0; k < maxEdges; ++k) replacementPressure[base + k] = RL(0.0);
    }
    if (solveStress[iCell] != 1) return;
    const rk dampingTimescale = consts[0];
    const int n = nEdgesOnCell[iCell];
    for (int iVertexOnCell = 0; iVertexOnCell < n; ++iVertexOnCell) {
        const int p = base + iVertexOnCell;
        si_evp_constitutive_relation(&stress11[p], &stress22[p], &stress12[p],
            strain11[p], strain22[p], strain12[p], icePressure[iCell],
            &replacementPressure[p], dtElastic, dampingTimescale, revisedEVP != 0);
    }
}

// seaice_stress_divergence_variational, variational.F:521-651: one thread per vertex
extern "C" __global__ void seaice_stress_divergence_variational_v842(
    rk *stressDivergenceU, rk *stressDivergenceV,
    const rk *stress11, const rk *stress22, const rk *stress12,
    const rk *basisIntegralsU, const rk *basisIntegralsV,
    const rk *basisIntegralsMetric, const rk *tanLatVertexRotatedOverRadius,
    const int *cellVerticesAtVertex, const int *solveVelocity,
    const int *nEdgesOnCell, const int *cellsOnVertex, const rk *areaTriangle,
    int nVertices, int vertexDegree, int maxEdges)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices || solveVelocity[iVertex] != 1) return;
    rk divU = RL(0.0), divV = RL(0.0);
    const rk t = tanLatVertexRotatedOverRadius[iVertex];
    for (int iSurroundingCell = 0; iSurroundingCell < vertexDegree; ++iSurroundingCell) {
        const int iCell = cellsOnVertex[iVertex * vertexDegree + iSurroundingCell] - 1;
        const int iVelocityVertex = cellVerticesAtVertex[iVertex * vertexDegree + iSurroundingCell] - 1;
        rk cellU = RL(0.0), cellV = RL(0.0);
        const int base = iCell * maxEdges;
        const int bbase = (base + iVelocityVertex) * maxEdges;
        const int n = nEdgesOnCell[iCell];
        for (int iStressVertex = 0; iStressVertex < n; ++iStressVertex) {
            const rk s11 = stress11[base + iStressVertex];
            const rk s22 = stress22[base + iStressVertex];
            const rk s12 = stress12[base + iStressVertex];
            const rk bU = basisIntegralsU[bbase + iStressVertex];
            const rk bV = basisIntegralsV[bbase + iStressVertex];
            const rk bM = basisIntegralsMetric[bbase + iStressVertex];
            // normal terms
            cellU = S(S(cellU, M(s11, bU)), M(s12, bV));
            cellV = S(S(cellV, M(s22, bV)), M(s12, bU));
            // metric terms
            cellU = S(cellU, M(M(s12, bM), t));
            cellV = A(cellV, M(M(s11, bM), t));
        }
        divU = A(divU, cellU);
        divV = A(divV, cellV);
    }
    stressDivergenceU[iVertex] = D(divU, areaTriangle[iVertex]);
    stressDivergenceV[iVertex] = D(divV, areaTriangle[iVertex]);
}

// seaice_final_divergence_shear_variational, variational.F:665-806: one thread
// per cell. The ridging parameters (useColumnPackage) take the unscaled
// divergence; the closing unit conversion applies to every cell.
extern "C" __global__ void seaice_final_divergence_shear_variational_v842(
    rk *divergence, rk *shear, rk *ridgeConvergence, rk *ridgeShear,
    const rk *strain11, const rk *strain22, const rk *strain12,
    const int *solveStress, const int *nEdgesOnCell, int useColumnPackage,
    int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk div = RL(0.0), shr = RL(0.0), deltaAverage = RL(0.0);
    const bool solve = solveStress[iCell] == 1;
    if (solve) {
        rk strainDivergenceSum = RL(0.0), strainTensionSum = RL(0.0), strainShearingSum = RL(0.0);
        const int base = iCell * maxEdges;
        const int n = nEdgesOnCell[iCell];
        for (int iVertexOnCell = 0; iVertexOnCell < n; ++iVertexOnCell) {
            const rk e11 = strain11[base + iVertexOnCell];
            const rk e22 = strain22[base + iVertexOnCell];
            const rk strainDivergence = A(e11, e22);
            const rk strainTension    = S(e11, e22);
            const rk strainShearing   = M(strain12[base + iVertexOnCell], RL(2.0));
            const rk Delta = SQRT(A(M(strainDivergence, strainDivergence),
                D(A(M(strainTension, strainTension), M(strainShearing, strainShearing)),
                  SI_ECCENTRICITY_SQUARED)));
            strainDivergenceSum = A(strainDivergenceSum, strainDivergence);
            strainTensionSum    = A(strainTensionSum, strainTension);
            strainShearingSum   = A(strainShearingSum, strainShearing);
            deltaAverage        = A(deltaAverage, Delta);
        }
        const rk rn = (rk)n;
        div = D(strainDivergenceSum, rn);
        shr = D(SQRT(A(M(strainTensionSum, strainTensionSum),
                             M(strainShearingSum, strainShearingSum))), rn);
        deltaAverage = D(deltaAverage, rn);
    }
    if (useColumnPackage) {
        if (solve) {
            ridgeConvergence[iCell] = -si_minss(RL(0.0), div);
            ridgeShear[iCell] = M(RL(0.5), S(deltaAverage, FABS(div)));
        } else {
            ridgeConvergence[iCell] = RL(0.0);
            ridgeShear[iCell] = RL(0.0);
        }
    }
    // units - for comparison to CICE
    divergence[iCell] = M(M(div, RL(100.0)), RL(86400.0));
    shear[iCell] = M(M(shr, RL(100.0)), RL(86400.0));
}
