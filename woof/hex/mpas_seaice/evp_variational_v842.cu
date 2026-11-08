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
// RKIND = 4 (MPAS's default single precision). Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off: left to right, parentheses kept, no contraction); x**2 is
// x*x as gfortran expands it. max/min are written as the SSE maxss/minss the
// reference object executes (operand order read from its disassembly), so a
// NaN or signed-zero operand takes the reference's word.
//
// Layout: a Fortran array a(maxEdges, nCells) is a[iCell*maxEdges + i]; a
// basis array b(maxEdges, maxEdges, nCells) is b[(iCell*maxEdges + j)*maxEdges + i];
// cellsOnVertex(vertexDegree, nVertices) is c[iVertex*vertexDegree + i].
// Connectivity is MPAS's 1-based indices.

#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

// constitutive_relation.F:29-46 (module parameters)
#define SI_ECCENTRICITY_SQUARED 4.0f          // eccentricity**2, eccentricity = 2.0
#define SI_DAMPING_TIMESCALE_PARAMETER 0.36f
#define SI_PUNY 1.0e-11f
#define SI_DAMPING_RATIO_DENOMINATOR 0.86f
#define SI_DAMPING_RATIO 5.5e-3f

// SSE maxss(dst=a, src=b): a > b ? a : b (b when either is NaN)
__device__ __forceinline__ float si_maxss(float a, float b) { return a > b ? a : b; }
// SSE minss(dst=a, src=b): a < b ? a : b (b when either is NaN)
__device__ __forceinline__ float si_minss(float a, float b) { return a < b ? a : b; }

// seaice_init_evp, constitutive_relation.F:64-136, for one block on one rank.
// consts[0] dampingTimescale, consts[1] numericalInertiaCoefficient,
// consts[2] evpDampingCriterion (set by the model, read by no v8.4.2 routine).
extern "C" __global__ void seaice_init_evp_v842(
    const float *dvEdge, int nEdgesSolve, float dynamicsTimeStep,
    float elasticTimeStep, float *consts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float dampingTimescale = M(SI_DAMPING_TIMESCALE_PARAMETER, dynamicsTimeStep);
    float evpDampingCriterion = D(M(1230.0f, dampingTimescale),
                                  M(elasticTimeStep, elasticTimeStep));
    // minval(dvEdge(1:nEdgesSolve)) as gfortran evaluates it: the first
    // element not above huge() starts the scan (NaN if none is; huge() for an
    // empty section), then a strict less-than scan
    float mv = 3.40282347e38f;
    int i = 0;
    if (nEdgesSolve > 0) {
        mv = __int_as_float(0x7fc00000);
        for (; i < nEdgesSolve; ++i) {
            if (dvEdge[i] <= 3.40282347e38f) { mv = dvEdge[i]; ++i; break; }
        }
        for (; i < nEdgesSolve; ++i) {
            if (dvEdge[i] < mv) mv = dvEdge[i];
        }
    }
    float dvEdgeMin = si_minss(mv, 1.0e30f);
    float dvEdgeMinGlobal = dvEdgeMin;  // mpas_dmpar_min_real on one rank
    float gamma = M(M(0.25f, 1.0e11f), dynamicsTimeStep);
    float numericalInertiaCoefficient =
        D(M(M(M(2.0f, SI_DAMPING_RATIO_DENOMINATOR), SI_DAMPING_RATIO), gamma),
          M(dvEdgeMinGlobal, dvEdgeMinGlobal));
    consts[0] = dampingTimescale;
    consts[1] = numericalInertiaCoefficient;
    consts[2] = evpDampingCriterion;
}

// seaice_strain_tensor_variational, variational.F:271-370: one thread per cell
extern "C" __global__ void seaice_strain_tensor_variational_v842(
    float *strain11, float *strain22, float *strain12,
    const float *uVelocity, const float *vVelocity,
    const float *basisGradientU, const float *basisGradientV,
    const float *tanLatVertexRotatedOverRadius, const int *solveStress,
    const int *nEdgesOnCell, const int *verticesOnCell, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells || solveStress[iCell] != 1) return;
    const int base = iCell * maxEdges;
    for (int k = 0; k < maxEdges; ++k) {
        strain11[base + k] = 0.0f;
        strain22[base + k] = 0.0f;
        strain12[base + k] = 0.0f;
    }
    const int n = nEdgesOnCell[iCell];
    for (int iGradientVertex = 0; iGradientVertex < n; ++iGradientVertex) {
        float s11 = 0.0f, s22 = 0.0f, s12 = 0.0f;
        const int gbase = (base + iGradientVertex) * maxEdges;
        for (int iBasisVertex = 0; iBasisVertex < n; ++iBasisVertex) {
            const int iVertex = verticesOnCell[base + iBasisVertex] - 1;
            const float u = uVelocity[iVertex], v = vVelocity[iVertex];
            const float gU = basisGradientU[gbase + iBasisVertex];
            const float gV = basisGradientV[gbase + iBasisVertex];
            s11 = A(s11, M(u, gU));
            s22 = A(s22, M(v, gV));
            s12 = A(s12, M(0.5f, A(M(u, gV), M(v, gU))));
        }
        // metric terms
        const int jVertex = verticesOnCell[base + iGradientVertex] - 1;
        const float t = tanLatVertexRotatedOverRadius[jVertex];
        s11 = S(s11, M(vVelocity[jVertex], t));
        s12 = A(s12, M(M(uVelocity[jVertex], t), 0.5f));
        strain11[base + iGradientVertex] = s11;
        strain22[base + iGradientVertex] = s22;
        strain12[base + iGradientVertex] = s12;
    }
}

// seaice_evp_constitutive_relation (constitutive_relation.F:150-216) and
// seaice_evp_constitutive_relation_revised (:230-294), one stress point
__device__ __forceinline__ void si_evp_constitutive_relation(
    float *stress11, float *stress22, float *stress12,
    float strain11, float strain22, float strain12,
    float icePressure, float *replacementPressure,
    float dtElastic, float dampingTimescale, bool revised)
{
    // convert from stress11 to stress1 etc
    const float strainDivergence = A(strain11, strain22);
    const float strainTension    = S(strain11, strain22);
    const float strainShearing   = M(strain12, 2.0f);
    float stress1 = A(*stress11, *stress22);
    float stress2 = S(*stress11, *stress22);
    // perform the constituitive relation
    const float Delta = __fsqrt_rn(A(M(strainDivergence, strainDivergence),
        D(A(M(strainTension, strainTension), M(strainShearing, strainShearing)),
          SI_ECCENTRICITY_SQUARED)));
    float pressureCoefficient = D(icePressure, si_maxss(Delta, SI_PUNY));
    *replacementPressure = M(pressureCoefficient, Delta);
    float denominator;
    if (!revised) {
        pressureCoefficient = D(M(pressureCoefficient, dtElastic), M(2.0f, dampingTimescale));
        denominator = A(1.0f, D(M(0.5f, dtElastic), dampingTimescale));
    } else {
        pressureCoefficient = D(M(M(pressureCoefficient, 2.0f), SI_DAMPING_RATIO),
                                SI_DAMPING_RATIO_DENOMINATOR);
        denominator = A(1.0f, D(M(2.0f, SI_DAMPING_RATIO), SI_DAMPING_RATIO_DENOMINATOR));
    }
    stress1 = D(A(stress1, M(pressureCoefficient, S(strainDivergence, Delta))), denominator);
    stress2 = D(A(stress2, M(D(pressureCoefficient, SI_ECCENTRICITY_SQUARED), strainTension)),
                denominator);
    *stress12 = D(A(*stress12, M(M(D(pressureCoefficient, SI_ECCENTRICITY_SQUARED),
                                   strainShearing), 0.5f)), denominator);
    // convert back
    *stress11 = M(0.5f, A(stress1, stress2));
    *stress22 = M(0.5f, S(stress1, stress2));
}

// seaice_stress_tensor_variational, variational.F:384-507: one thread per cell.
// consts[0] is seaice_init_evp's dampingTimescale.
extern "C" __global__ void seaice_stress_tensor_variational_v842(
    float *stress11, float *stress22, float *stress12,
    const float *strain11, const float *strain22, const float *strain12,
    const float *icePressure, float *replacementPressure,
    const int *solveStress, const int *nEdgesOnCell, const float *consts,
    float dtElastic, int revisedEVP, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const int base = iCell * maxEdges;
    if (!revisedEVP) {
        for (int k = 0; k < maxEdges; ++k) replacementPressure[base + k] = 0.0f;
    }
    if (solveStress[iCell] != 1) return;
    const float dampingTimescale = consts[0];
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
    float *stressDivergenceU, float *stressDivergenceV,
    const float *stress11, const float *stress22, const float *stress12,
    const float *basisIntegralsU, const float *basisIntegralsV,
    const float *basisIntegralsMetric, const float *tanLatVertexRotatedOverRadius,
    const int *cellVerticesAtVertex, const int *solveVelocity,
    const int *nEdgesOnCell, const int *cellsOnVertex, const float *areaTriangle,
    int nVertices, int vertexDegree, int maxEdges)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices || solveVelocity[iVertex] != 1) return;
    float divU = 0.0f, divV = 0.0f;
    const float t = tanLatVertexRotatedOverRadius[iVertex];
    for (int iSurroundingCell = 0; iSurroundingCell < vertexDegree; ++iSurroundingCell) {
        const int iCell = cellsOnVertex[iVertex * vertexDegree + iSurroundingCell] - 1;
        const int iVelocityVertex = cellVerticesAtVertex[iVertex * vertexDegree + iSurroundingCell] - 1;
        float cellU = 0.0f, cellV = 0.0f;
        const int base = iCell * maxEdges;
        const int bbase = (base + iVelocityVertex) * maxEdges;
        const int n = nEdgesOnCell[iCell];
        for (int iStressVertex = 0; iStressVertex < n; ++iStressVertex) {
            const float s11 = stress11[base + iStressVertex];
            const float s22 = stress22[base + iStressVertex];
            const float s12 = stress12[base + iStressVertex];
            const float bU = basisIntegralsU[bbase + iStressVertex];
            const float bV = basisIntegralsV[bbase + iStressVertex];
            const float bM = basisIntegralsMetric[bbase + iStressVertex];
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
    float *divergence, float *shear, float *ridgeConvergence, float *ridgeShear,
    const float *strain11, const float *strain22, const float *strain12,
    const int *solveStress, const int *nEdgesOnCell, int useColumnPackage,
    int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    float div = 0.0f, shr = 0.0f, deltaAverage = 0.0f;
    const bool solve = solveStress[iCell] == 1;
    if (solve) {
        float strainDivergenceSum = 0.0f, strainTensionSum = 0.0f, strainShearingSum = 0.0f;
        const int base = iCell * maxEdges;
        const int n = nEdgesOnCell[iCell];
        for (int iVertexOnCell = 0; iVertexOnCell < n; ++iVertexOnCell) {
            const float e11 = strain11[base + iVertexOnCell];
            const float e22 = strain22[base + iVertexOnCell];
            const float strainDivergence = A(e11, e22);
            const float strainTension    = S(e11, e22);
            const float strainShearing   = M(strain12[base + iVertexOnCell], 2.0f);
            const float Delta = __fsqrt_rn(A(M(strainDivergence, strainDivergence),
                D(A(M(strainTension, strainTension), M(strainShearing, strainShearing)),
                  SI_ECCENTRICITY_SQUARED)));
            strainDivergenceSum = A(strainDivergenceSum, strainDivergence);
            strainTensionSum    = A(strainTensionSum, strainTension);
            strainShearingSum   = A(strainShearingSum, strainShearing);
            deltaAverage        = A(deltaAverage, Delta);
        }
        const float rn = (float)n;
        div = D(strainDivergenceSum, rn);
        shr = D(__fsqrt_rn(A(M(strainTensionSum, strainTensionSum),
                             M(strainShearingSum, strainShearingSum))), rn);
        deltaAverage = D(deltaAverage, rn);
    }
    if (useColumnPackage) {
        if (solve) {
            ridgeConvergence[iCell] = -si_minss(0.0f, div);
            ridgeShear[iCell] = M(0.5f, S(deltaAverage, fabsf(div)));
        } else {
            ridgeConvergence[iCell] = 0.0f;
            ridgeShear[iCell] = 0.0f;
        }
    }
    // units - for comparison to CICE
    divergence[iCell] = M(M(div, 100.0f), 86400.0f);
    shear[iCell] = M(M(shr, 100.0f), 86400.0f);
}
