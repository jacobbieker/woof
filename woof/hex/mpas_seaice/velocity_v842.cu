// MPAS-Seaice v8.4.2: the EVP velocity solve (ocean stress, Coriolis, the
// velocity update, the principal stresses and the final ocean stress).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_velocity_solver.F:
//     1664-1710 coriolis_force_coefficient
//     1724-1805 ocean_stress
//     2149-2308 init_subcycle_variables (the variational branch)
//     2499-2579 ocean_stress_coefficient
//     2593-2707 solve_velocity
//     2721-2841 solve_velocity_revised
//     2942-3109 principal_stresses_driver (variational), principal_stresses
//     3123-3347 ocean_stress_final
//   src/core_seaice/shared/mpas_seaice_mesh.F:
//     2888-2958 seaice_interpolate_vertex_to_cell
//   with the module constants at 31-36 and seaice_constants' density and
//   drag (mpas_seaice_constants.F, column/constants/cice/ice_constants_colpkg.F90).
// (Adrian K. Turner, LANL, 2013-2015). Copyright (c) 2013-2019, Los Alamos
// National Security, LLC (LANS) and the University Corporation for
// Atmospheric Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction); x**2 is x*x as gfortran expands it; sign(1.0, x) is
// copysign(1.0, x), as gfortran compiles it (so sign(1.0, -0.0) is -1).
//
// Constants: a literal written _RKIND is decimal rounded once to RKIND (the
// f-suffixed literal at RKIND 4). A seaice_constants value is a colpkg
// dbl_kind parameter converted to RKIND (the double literal cast).
//
// Layout: a Fortran array a(maxEdges, nCells) is a[iCell*maxEdges + i].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define SI_SIGN1(x) copysign(1.0, (x))
#define SI_BIG 1.0e30                      // 1.0e30_RKIND
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define SI_SIGN1(x) copysignf(1.0f, (x))
#define SI_BIG 1.0e30f                     // 1.0e30_RKIND
#endif

// velocity_solver.F:31-36
#define SI_SIN_TURN ((rk)0.0)              // sinOceanTurningAngle
#define SI_COS_TURN ((rk)1.0)              // cosOceanTurningAngle
// seaice_constants: rhow, dragio, puny (dbl_kind) as RKIND parameters
#define SI_RHOW ((rk)1026.0)
#define SI_DRAGIO ((rk)0.00536)
#define SI_PUNY ((rk)1.0e-11)

// coriolis_force_coefficient: one thread per vertex.
extern "C" __global__ void seaice_coriolis_force_coefficient_v842(
    rk *totalMassVertexfVertex, const rk *totalMassVertex, const rk *fVertex, int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve) return;
    totalMassVertexfVertex[iVertex] = M(totalMassVertex[iVertex], fVertex[iVertex]);
}

// ocean_stress: one thread per vertex. Without config_use_ocean_stress the
// whole array (nVertices) is zeroed.
extern "C" __global__ void seaice_ocean_stress_v842(
    rk *oceanStressU, rk *oceanStressV, const rk *uOceanVelocityVertex, const rk *vOceanVelocityVertex,
    const rk *fVertex, const int *solveVelocity, int useOceanStress, int nVerticesSolve, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (!useOceanStress) {
        if (iVertex < nVertices) {
            oceanStressU[iVertex] = (rk)0.0;
            oceanStressV[iVertex] = (rk)0.0;
        }
        return;
    }
    if (iVertex >= nVerticesSolve) return;
    if (solveVelocity[iVertex] == 1) {
        const rk sgn = SI_SIGN1(fVertex[iVertex]);
        const rk uo = uOceanVelocityVertex[iVertex];
        const rk vo = vOceanVelocityVertex[iVertex];
        oceanStressU[iVertex] = S(M(uo, SI_COS_TURN), M(M(vo, SI_SIN_TURN), sgn));
        oceanStressV[iVertex] = A(M(M(uo, SI_SIN_TURN), sgn), M(vo, SI_COS_TURN));
    } else {
        oceanStressU[iVertex] = (rk)0.0;
        oceanStressV[iVertex] = (rk)0.0;
    }
}

// init_subcycle_variables, the vertex part: one thread per vertex.
extern "C" __global__ void seaice_init_subcycle_vertices_v842(
    rk *stressDivergenceU, rk *stressDivergenceV, rk *uVelocity, rk *vVelocity, rk *oceanStressCoeff,
    const int *solveVelocity, int nVerticesSolve, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVertices) return;
    stressDivergenceU[iVertex] = (rk)0.0;
    stressDivergenceV[iVertex] = (rk)0.0;
    if (iVertex < nVerticesSolve && solveVelocity[iVertex] != 1) {
        uVelocity[iVertex] = (rk)0.0;
        vVelocity[iVertex] = (rk)0.0;
        oceanStressCoeff[iVertex] = (rk)0.0;
    }
}

// init_subcycle_variables, the variational cell part: one thread per cell.
// Every strain word is zeroed; the stresses of cells not solved are zeroed.
extern "C" __global__ void seaice_init_subcycle_cells_variational_v842(
    rk *strain11, rk *strain22, rk *strain12, rk *stress11, rk *stress22, rk *stress12,
    const int *solveStress, int nCells, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    const bool zeroStress = solveStress[iCell] != 1;
    for (int i = 0; i < maxEdges; ++i) {
        const int p = iCell * maxEdges + i;
        strain11[p] = (rk)0.0;
        strain22[p] = (rk)0.0;
        strain12[p] = (rk)0.0;
        if (zeroStress) {
            stress11[p] = (rk)0.0;
            stress22[p] = (rk)0.0;
            stress12[p] = (rk)0.0;
        }
    }
}

// ocean_stress_coefficient: one thread per vertex. Without
// config_use_ocean_stress the whole array is zeroed; with it, vertices not
// solved keep their words.
extern "C" __global__ void seaice_ocean_stress_coefficient_v842(
    rk *oceanStressCoeff, const rk *uOceanVelocityVertex, const rk *vOceanVelocityVertex,
    const rk *uVelocity, const rk *vVelocity, const rk *iceAreaVertex, const int *solveVelocity,
    int useOceanStress, int nVerticesSolve, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (!useOceanStress) {
        if (iVertex < nVertices) oceanStressCoeff[iVertex] = (rk)0.0;
        return;
    }
    if (iVertex >= nVerticesSolve || solveVelocity[iVertex] != 1) return;
    const rk du = S(uOceanVelocityVertex[iVertex], uVelocity[iVertex]);
    const rk dv = S(vOceanVelocityVertex[iVertex], vVelocity[iVertex]);
    oceanStressCoeff[iVertex] = M(M(M(SI_DRAGIO, SI_RHOW), iceAreaVertex[iVertex]),
                                  SQRT(A(M(du, du), M(dv, dv))));
}

// solve_velocity (revised = 0, elasticTimeStep) and solve_velocity_revised
// (revised = 1, dynamicsTimeStep and numericalInertiaCoefficient, read from
// inertia[0]): one thread per vertex, the 2x2 implicit solve.
extern "C" __global__ void seaice_solve_velocity_v842(
    rk *uVelocity, rk *vVelocity, const rk *uVelocityInitial, const rk *vVelocityInitial,
    const rk *totalMassVertex, const rk *totalMassVertexfVertex,
    const rk *stressDivergenceU, const rk *stressDivergenceV, const rk *airStressVertexU,
    const rk *airStressVertexV, const rk *surfaceTiltForceU, const rk *surfaceTiltForceV,
    const rk *oceanStressU, const rk *oceanStressV, const rk *oceanStressCoeff, const int *solveVelocity,
    const rk *inertia, rk timeStep, int revised, int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve || solveVelocity[iVertex] != 1) return;
    const rk mass = totalMassVertex[iVertex];
    const rk massF = totalMassVertexfVertex[iVertex];
    const rk coeff = oceanStressCoeff[iVertex];
    const rk sgn = SI_SIGN1(massF);
    const rk u = uVelocity[iVertex];
    const rk v = vVelocity[iVertex];
    rk l11, l22, inertiaU, inertiaV;
    if (!revised) {
        l11 = A(D(mass, timeStep), M(coeff, SI_COS_TURN));
        l22 = A(D(mass, timeStep), M(coeff, SI_COS_TURN));
        inertiaU = D(M(mass, u), timeStep);
        inertiaV = D(M(mass, v), timeStep);
    } else {
        const rk nic = inertia[0];
        l11 = A(M(A(nic, (rk)1.0), D(mass, timeStep)), M(coeff, SI_COS_TURN));
        l22 = A(M(A(nic, (rk)1.0), D(mass, timeStep)), M(coeff, SI_COS_TURN));
        inertiaU = D(M(mass, A(M(nic, u), uVelocityInitial[iVertex])), timeStep);
        inertiaV = D(M(mass, A(M(nic, v), vVelocityInitial[iVertex])), timeStep);
    }
    const rk l12 = S(-massF, M(M(coeff, SI_SIN_TURN), sgn));
    const rk l21 = A(massF, M(M(coeff, SI_SIN_TURN), sgn));
    const rk r1 = A(A(A(A(stressDivergenceU[iVertex], airStressVertexU[iVertex]), surfaceTiltForceU[iVertex]),
                      M(coeff, oceanStressU[iVertex])), inertiaU);
    const rk r2 = A(A(A(A(stressDivergenceV[iVertex], airStressVertexV[iVertex]), surfaceTiltForceV[iVertex]),
                      M(coeff, oceanStressV[iVertex])), inertiaV);
    const rk den = S(M(l11, l22), M(l12, l21));
    uVelocity[iVertex] = D(S(M(l22, r1), M(l12, r2)), den);
    vVelocity[iVertex] = D(S(M(l11, r2), M(l21, r1)), den);
}

// principal_stresses_driver, variational branch: one thread per cell, over
// the nEdgesOnCell vertices of the first nCellsSolve cells.
extern "C" __global__ void seaice_principal_stresses_variational_v842(
    rk *principalStress1, rk *principalStress2, const rk *stress11, const rk *stress22,
    const rk *stress12, const rk *replacementPressure, const int *nEdgesOnCell, int nCellsSolve,
    int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    const int n = nEdgesOnCell[iCell];
    for (int i = 0; i < n; ++i) {
        const int p = iCell * maxEdges + i;
        const rk rp = replacementPressure[p];
        if (rp > SI_PUNY) {
            const rk s11 = stress11[p];
            const rk s22 = stress22[p];
            const rk s12 = stress12[p];
            const rk t = A(s11, s22);
            const rk sqrtContents = A(S(M(t, t), M(M((rk)4.0, s11), s22)), M((rk)4.0, M(s12, s12)));
            const rk root = SQRT(sqrtContents);
            const rk p1 = A(M((rk)0.5, A(s11, s22)), M((rk)0.5, root));
            const rk p2 = S(M((rk)0.5, A(s11, s22)), M((rk)0.5, root));
            principalStress1[p] = D(p1, rp);
            principalStress2[p] = D(p2, rp);
        } else {
            principalStress1[p] = SI_BIG;
            principalStress2[p] = SI_BIG;
        }
    }
}

// ocean_stress_final, the vertex loop (after ocean_stress_coefficient): one
// thread per vertex. With config_use_ocean_stress, the stress on the ice per
// unit ice area over nVerticesSolve (0 where not solved); without it, the
// whole array (nVertices) is zeroed.
extern "C" __global__ void seaice_ocean_stress_final_v842(
    rk *oceanStressU, rk *oceanStressV, const rk *oceanStressCoeff, const rk *uOceanVelocityVertex,
    const rk *vOceanVelocityVertex, const rk *uVelocity, const rk *vVelocity, const rk *iceAreaVertex,
    const rk *fVertex, const int *solveVelocity, int useOceanStress, int nVerticesSolve, int nVertices)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (!useOceanStress) {
        if (iVertex < nVertices) {
            oceanStressU[iVertex] = (rk)0.0;
            oceanStressV[iVertex] = (rk)0.0;
        }
        return;
    }
    if (iVertex >= nVerticesSolve) return;
    if (solveVelocity[iVertex] == 1) {
        const rk sgn = SI_SIGN1(fVertex[iVertex]);
        const rk coeff = oceanStressCoeff[iVertex];
        const rk du = S(uOceanVelocityVertex[iVertex], uVelocity[iVertex]);
        const rk dv = S(vOceanVelocityVertex[iVertex], vVelocity[iVertex]);
        const rk su = M(coeff, S(M(du, SI_COS_TURN), M(M(dv, SI_SIN_TURN), sgn)));
        const rk sv = M(coeff, A(M(dv, SI_COS_TURN), M(M(du, SI_SIN_TURN), sgn)));
        oceanStressU[iVertex] = D(su, iceAreaVertex[iVertex]);
        oceanStressV[iVertex] = D(sv, iceAreaVertex[iVertex]);
    } else {
        oceanStressU[iVertex] = (rk)0.0;
        oceanStressV[iVertex] = (rk)0.0;
    }
}

// seaice_interpolate_vertex_to_cell: one thread per cell, over nCellsSolve.
// The areaTriangle-weighted mean over the cell's nEdgesOnCell vertices,
// counting interior vertices only: real(interiorVertex) is a default
// (single) real, exactly 0 or 1 at either precision. The accumulators start
// from 0; a cell with no interior vertex keeps the accumulated sum.
extern "C" __global__ void seaice_interpolate_vertex_to_cell_v842(
    rk *variableCell, const rk *variableVertex, const rk *areaTriangle, const int *interiorVertex,
    const int *nEdgesOnCell, const int *verticesOnCell, int nCellsSolve, int maxEdges)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCellsSolve) return;
    rk value = (rk)0.0, totalArea = (rk)0.0;
    for (int k = 0; k < nEdgesOnCell[iCell]; ++k) {
        const int iVertex = verticesOnCell[iCell * maxEdges + k] - 1;
        const rk interior = (rk)(float)interiorVertex[iVertex];
        value = A(value, M(M(areaTriangle[iVertex], variableVertex[iVertex]), interior));
        totalArea = A(totalArea, M(areaTriangle[iVertex], interior));
    }
    if (totalArea > (rk)0.0) value = D(value, totalArea);
    variableCell[iCell] = value;
}

// ocean_stress_final, after the cell interpolation: the vertex stress back
// to per unit grid area on solved vertices (nVerticesSolve).
extern "C" __global__ void seaice_ocean_stress_final_rescale_v842(
    rk *oceanStressU, rk *oceanStressV, const rk *iceAreaVertex, const int *solveVelocity, int nVerticesSolve)
{
    int iVertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (iVertex >= nVerticesSolve || solveVelocity[iVertex] != 1) return;
    oceanStressU[iVertex] = M(oceanStressU[iVertex], iceAreaVertex[iVertex]);
    oceanStressV[iVertex] = M(oceanStressV[iVertex], iceAreaVertex[iVertex]);
}

// ocean_stress_final without config_use_ocean_stress: the cell stress zeroed
// whole (nCells).
extern "C" __global__ void seaice_zero_cell_stress_v842(rk *oceanStressCellU, rk *oceanStressCellV, int nCells)
{
    int iCell = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCell >= nCells) return;
    oceanStressCellU[iCell] = (rk)0.0;
    oceanStressCellV[iCell] = (rk)0.0;
}
