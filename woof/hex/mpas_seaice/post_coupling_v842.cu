// MPAS-Seaice v8.4.2: what the sea ice does to the coupler's fields after each
// import (post_atmospheric_coupling, post_oceanic_coupling).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_forcing.F:
//     585-677   post_atmospheric_coupling
//     1503-1591 post_oceanic_coupling
//   src/core_seaice/shared/mpas_seaice_mesh.F:
//     seaice_latlon_vector_rotation_forward and the routines it calls
//     (seaice_grid_rotation_forward, seaice_latlon_from_xyz,
//     latlon_vector_to_xyz_vector, xyz_vector_rotation_forward,
//     xyz_vector_to_latlon_vector)
// (Adrian K. Turner, LANL). Copyright (c) 2013-2019, Los Alamos National
// Security, LLC (LANS) and the University Corporation for Atmospheric Research
// (UCAR); BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt.
// This is not the version available from LANS and UCAR.
//
// real(8) only. Every operation is an explicit round-to-nearest intrinsic in
// the Fortran's evaluation order (gfortran -O0 -ffp-contract=off); x**2 is
// x*x. The rotation's sin, cos, asin and atan2 are the host libm's: their
// words come in as trig (8 per cell: sin and cos of latCell, of lonCell, of
// the rotated grid's latitude and of its longitude), so no libm runs here.
// One thread per cell; no cell reads another. The rotation writes its output
// over its input (the Fortran passes the same actual twice), which is safe
// because every input is read before the first output is written.

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))

// seaice_latlon_vector_rotation_forward with its trigonometric words
__device__ void rotate_forward(r8& up, r8& vp, r8 u, r8 v, const r8* t, int rotate)
{
    const r8 sLat = t[0], cLat = t[1], sLon = t[2], cLon = t[3];
    const r8 sLatp = t[4], cLatp = t[5], sLonp = t[6], cLonp = t[7];
    // latlon_vector_to_xyz_vector on the geographical grid
    const r8 vx = S(M(-u, sLon), M(M(v, sLat), cLon));
    const r8 vy = S(M(u, cLon), M(M(v, sLat), sLon));
    const r8 vz = M(v, cLat);
    // xyz_vector_rotation_forward
    r8 vxp = vx, vyp = vy, vzp = vz;
    if (rotate) {
        vxp = -vz;
        vyp = vy;
        vzp = vx;
    }
    // xyz_vector_to_latlon_vector on the rotated grid
    up = A(M(-sLonp, vxp), M(cLonp, vyp));
    vp = A(A(M(M(-sLatp, cLonp), vxp), M(M(-sLatp, sLonp), vyp)), M(cLatp, vzp));
}

// post_atmospheric_coupling's loop over 1..nCellsSolve
extern "C" __global__ void seaice_post_atmospheric_coupling_v842(
    int nCellsSolve, r8* __restrict__ uAir, r8* __restrict__ vAir, r8* __restrict__ windSpeed,
    r8* __restrict__ shortwaveDown, const r8* __restrict__ swVisDir, const r8* __restrict__ swVisDif,
    const r8* __restrict__ swIRDir, const r8* __restrict__ swIRDif, const r8* __restrict__ trig, int rotate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCellsSolve) return;
    const r8 u = uAir[i], v = vAir[i];
    windSpeed[i] = __dsqrt_rn(A(M(u, u), M(v, v)));
    r8 up, vp;
    rotate_forward(up, vp, u, v, trig + (size_t)i * 8, rotate);
    uAir[i] = up;
    vAir[i] = vp;
    shortwaveDown[i] = A(A(A(swVisDir[i], swVisDif[i]), swIRDir[i]), swIRDif[i]);
}

// post_oceanic_coupling's loop over 1..nCells
extern "C" __global__ void seaice_post_oceanic_coupling_v842(
    int nCells, r8* __restrict__ uOcean, r8* __restrict__ vOcean, r8* __restrict__ tiltU,
    r8* __restrict__ tiltV, const r8* __restrict__ trig, int rotate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const r8* t = trig + (size_t)i * 8;
    r8 up, vp;
    rotate_forward(up, vp, uOcean[i], vOcean[i], t, rotate);
    uOcean[i] = up;
    vOcean[i] = vp;
    rotate_forward(up, vp, tiltU[i], tiltV[i], t, rotate);
    tiltU[i] = up;
    tiltV[i] = vp;
}
