// MPAS-Seaice v8.4.2: EVP internal stress, weak-form discretisation.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_velocity_solver_weak.F:239-380,394-507,521-668,682-777
//   src/core_seaice/shared/mpas_seaice_velocity_solver.F:2149-2308 (init_subcycle_variables, weak),
//     3026-3046,3074-3108 (principal_stresses_driver, weak, and principal_stresses)
// (Adrian K. Turner, LANL, 2013-2015). Copyright (c) 2013-2019, Los Alamos
// National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// This unit follows evp_variational_v842.cu in one source (weak_v842.py's
// module_source): the precision macros, the module parameters and
// si_evp_constitutive_relation are that file's. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction); x**2 is x*x.
//
// The metric terms' tan(latCellRotated) and tan(latVertexRotated) are the
// host's (weak_v842.metric_tangents: the C library's tan at RKIND, or the
// correctly rounded one), passed in as arrays: the latitudes are set at init
// and never change, so each word is the one v8.4.2 computes every subcycle.
//
// Layout: normalVectorPolygon(2, maxEdges, nCells) is
// n[(iCell*maxEdges + i)*2 + k]; normalVectorTriangle(2, vertexDegree,
// nVertices) is n[(iVertex*vertexDegree + i)*2 + k]; verticesOnEdge(2,
// nEdges) is v[iEdge*2 + k]. Connectivity is MPAS's 1-based indices.

// seaice_strain_tensor_weak, weak.F:239-380: one thread per cell, cells
// 1..nCells (sphereRadius: sphere_radius, or 1 on a plane where it is 0)
extern "C" __global__ void seaice_strain_tensor_weak_v842(
    rk *strain11, rk *strain22, rk *strain12,
    const rk *uVelocity, const rk *vVelocity, const rk *normalVectorPolygon,
    const rk *tanLatCellRotated, const int *solveStress,
    const int *nEdgesOnCell, const int *verticesOnCell, const int *edgesOnCell,
    const int *verticesOnEdge, const rk *dvEdge, const rk *areaCell,
    rk sphereRadius, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    rk s11 = RL(0.0), s22 = RL(0.0), s12 = RL(0.0);
    if (solveStress[iCell] == 1) {
        rk uCellCentre = RL(0.0), vCellCentre = RL(0.0);
        const int base = iCell * maxEdges;
        const int n = nEdgesOnCell[iCell];
        for (int iEdgeOnCell = 0; iEdgeOnCell < n; ++iEdgeOnCell) {
            // cell centre velocities
            const int iVertex = verticesOnCell[base + iEdgeOnCell] - 1;
            uCellCentre = A(uCellCentre, uVelocity[iVertex]);
            vCellCentre = A(vCellCentre, vVelocity[iVertex]);
            // interpolated edge velocity
            const int iEdge = edgesOnCell[base + iEdgeOnCell] - 1;
            rk uVelocityEdge = RL(0.0), vVelocityEdge = RL(0.0);
            for (int iVertexOnEdge = 0; iVertexOnEdge < 2; ++iVertexOnEdge) {
                const int jVertex = verticesOnEdge[iEdge * 2 + iVertexOnEdge] - 1;
                uVelocityEdge = A(uVelocityEdge, uVelocity[jVertex]);
                vVelocityEdge = A(vVelocityEdge, vVelocity[jVertex]);
            }
            uVelocityEdge = D(uVelocityEdge, RL(2.0));
            vVelocityEdge = D(vVelocityEdge, RL(2.0));
            // summation over edges
            const rk n1 = normalVectorPolygon[(base + iEdgeOnCell) * 2];
            const rk n2 = normalVectorPolygon[(base + iEdgeOnCell) * 2 + 1];
            const rk dv = dvEdge[iEdge];
            s11 = A(s11, M(M(uVelocityEdge, n1), dv));
            s22 = A(s22, M(M(vVelocityEdge, n2), dv));
            s12 = A(s12, M(M(RL(0.5), A(M(uVelocityEdge, n2), M(vVelocityEdge, n1))), dv));
        }
        const rk rn = (rk)n;
        uCellCentre = D(uCellCentre, rn);
        vCellCentre = D(vCellCentre, rn);
        s11 = D(s11, areaCell[iCell]);
        s22 = D(s22, areaCell[iCell]);
        s12 = D(s12, areaCell[iCell]);
        // metric terms
        const rk t = tanLatCellRotated[iCell];
        s11 = S(s11, D(M(vCellCentre, t), sphereRadius));
        s12 = A(s12, D(M(M(uCellCentre, t), RL(0.5)), sphereRadius));
    }
    strain11[iCell] = s11;
    strain22[iCell] = s22;
    strain12[iCell] = s12;
}

// seaice_stress_tensor_weak, weak.F:394-507: one thread per cell, cells
// 1..nCells. consts[0] is seaice_init_evp's dampingTimescale.
extern "C" __global__ void seaice_stress_tensor_weak_v842(
    rk *stress11, rk *stress22, rk *stress12,
    const rk *strain11, const rk *strain22, const rk *strain12,
    const rk *icePressure, rk *replacementPressure, const int *solveStress,
    const rk *consts, rk dtElastic, int revisedEVP, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    if (solveStress[iCell] == 1) {
        si_evp_constitutive_relation(&stress11[iCell], &stress22[iCell], &stress12[iCell],
            strain11[iCell], strain22[iCell], strain12[iCell], icePressure[iCell],
            &replacementPressure[iCell], dtElastic, consts[0], revisedEVP != 0);
    } else {
        stress11[iCell] = RL(0.0);
        stress22[iCell] = RL(0.0);
        stress12[iCell] = RL(0.0);
    }
}

// seaice_stress_divergence_weak, weak.F:521-668: one thread per vertex,
// vertices 1..nVertices (the routine's nVerticesSolve is the nVertices
// dimension)
extern "C" __global__ void seaice_stress_divergence_weak_v842(
    rk *stressDivergenceU, rk *stressDivergenceV,
    const rk *stress11, const rk *stress22, const rk *stress12,
    const rk *normalVectorTriangle, const rk *tanLatVertexRotated,
    const int *solveVelocity, const int *cellsOnVertex, const int *edgesOnVertex,
    const int *cellsOnEdge, const rk *areaTriangle, const rk *dcEdge,
    rk sphereRadius, int nVertices, int vertexDegree)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices) return;
    rk divU = RL(0.0), divV = RL(0.0);
    if (solveVelocity[iVertex] == 1) {
        rk stress11Vertex = RL(0.0), stress22Vertex = RL(0.0), stress12Vertex = RL(0.0);
        for (int iVertexDegree = 0; iVertexDegree < vertexDegree; ++iVertexDegree) {
            // vertex stresses
            const int iCell = cellsOnVertex[iVertex * vertexDegree + iVertexDegree] - 1;
            stress11Vertex = A(stress11Vertex, stress11[iCell]);
            stress22Vertex = A(stress22Vertex, stress22[iCell]);
            stress12Vertex = A(stress12Vertex, stress12[iCell]);
            // interpolated edge stress
            const int iEdge = edgesOnVertex[iVertex * vertexDegree + iVertexDegree] - 1;
            rk stress11Edge = RL(0.0), stress22Edge = RL(0.0), stress12Edge = RL(0.0);
            for (int iCellOnEdge = 0; iCellOnEdge < 2; ++iCellOnEdge) {
                const int jCell = cellsOnEdge[iEdge * 2 + iCellOnEdge] - 1;
                stress11Edge = A(stress11Edge, stress11[jCell]);
                stress22Edge = A(stress22Edge, stress22[jCell]);
                stress12Edge = A(stress12Edge, stress12[jCell]);
            }
            stress11Edge = D(stress11Edge, RL(2.0));
            stress22Edge = D(stress22Edge, RL(2.0));
            stress12Edge = D(stress12Edge, RL(2.0));
            const int p = (iVertex * vertexDegree + iVertexDegree) * 2;
            const rk n1 = normalVectorTriangle[p], n2 = normalVectorTriangle[p + 1];
            const rk dc = dcEdge[iEdge];
            divU = A(divU, M(A(M(stress11Edge, n1), M(stress12Edge, n2)), dc));
            divV = A(divV, M(A(M(stress22Edge, n2), M(stress12Edge, n1)), dc));
        }
        const rk rd = (rk)vertexDegree;
        stress11Vertex = D(stress11Vertex, rd);
        stress22Vertex = D(stress22Vertex, rd);
        stress12Vertex = D(stress12Vertex, rd);
        divU = D(divU, areaTriangle[iVertex]);
        divV = D(divV, areaTriangle[iVertex]);
        // metric terms
        const rk t = tanLatVertexRotated[iVertex];
        divU = S(divU, D(M(M(t, stress12Vertex), RL(2.0)), sphereRadius));
        divV = A(divV, D(M(t, S(stress11Vertex, stress22Vertex)), sphereRadius));
    }
    stressDivergenceU[iVertex] = divU;
    stressDivergenceV[iVertex] = divV;
}

// the Delta of seaice_final_divergence_shear_weak's loop for one cell
__device__ __forceinline__ rk si_weak_delta(rk e11, rk e22, rk e12)
{
    const rk strainDivergence = A(e11, e22);
    const rk strainTension    = S(e11, e22);
    const rk strainShearing   = M(e12, RL(2.0));
    return SQRT(A(M(strainDivergence, strainDivergence),
        D(A(M(strainTension, strainTension), M(strainShearing, strainShearing)),
          SI_ECCENTRICITY_SQUARED)));
}

// seaice_final_divergence_shear_weak, weak.F:682-777: one thread per cell,
// cells 1..nCellsSolve; divergence and shear stay in s^-1 (no unit
// conversion, unlike the variational routine). v8.4.2 assigns the whole
// Delta array in each pass of its loop (`Delta = sqrt(...)`), so after it
// every element holds the last cell's value, which ridgeShear then reads:
// perCellDelta = 0 keeps that; 1 (the opt-in weak_ridge_delta correction)
// uses each cell's own Delta.
extern "C" __global__ void seaice_final_divergence_shear_weak_v842(
    rk *divergence, rk *shear, rk *ridgeConvergence, rk *ridgeShear,
    const rk *strain11, const rk *strain22, const rk *strain12,
    int useColumnPackage, int perCellDelta, int nCellsSolve)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const rk e11 = strain11[iCell], e22 = strain22[iCell];
    const rk strainDivergence = A(e11, e22);
    const rk strainTension    = S(e11, e22);
    const rk strainShearing   = M(strain12[iCell], RL(2.0));
    divergence[iCell] = strainDivergence;
    shear[iCell] = SQRT(A(M(strainTension, strainTension), M(strainShearing, strainShearing)));
    if (useColumnPackage) {
        const int j = perCellDelta ? iCell : nCellsSolve - 1;
        const rk Delta = si_weak_delta(strain11[j], strain22[j], strain12[j]);
        ridgeConvergence[iCell] = -si_minss(RL(0.0), strainDivergence);
        ridgeShear[iCell] = M(RL(0.5), S(Delta, FABS(strainDivergence)));
    }
}

// init_subcycle_variables (weak), velocity_solver.F:2269-2303: the strains
// zeroed whole (n: the arrays' length) and the stresses of the cells
// 1..nCellsLoop (nCells) not solved for zeroed
extern "C" __global__ void seaice_init_subcycle_cells_weak_v842(
    rk *strain11, rk *strain22, rk *strain12, rk *stress11, rk *stress22, rk *stress12,
    const int *solveStress, int n, int nCellsLoop)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= n) return;
    strain11[iCell] = RL(0.0);
    strain22[iCell] = RL(0.0);
    strain12[iCell] = RL(0.0);
    if (iCell < nCellsLoop && solveStress[iCell] != 1) {
        stress11[iCell] = RL(0.0);
        stress22[iCell] = RL(0.0);
        stress12[iCell] = RL(0.0);
    }
}
