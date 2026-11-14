// MPAS-Seaice v8.4.2: incremental remap advection (planar meshes).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_advection_incremental_remap.F:
//     9309-9387 volume_to_thickness, thickness_to_volume
// (William Lipscomb, LANL, 2015). Copyright (c) 2015, Los Alamos National
// Security, LLC (LANS) and the University Corporation for Atmospheric
// Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction); x**2 is x*x as gfortran expands it. A literal written _RKIND
// is decimal rounded once to RKIND (the f-suffixed literal at RKIND 4).
//
// Layout: a Fortran array a(d1, d2, ..., dn) is C-order (dn, ..., d2, d1); a
// tracer t(nLayers, nCategories, nCells+1) is t[(iCell*nCategories + iCat)*nLayers + iLayer].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define C(x) x
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define C(x) x##f
#endif

__device__ __forceinline__ int ir_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ---------------------------------------------------------------------------
// volume_to_thickness (9309-9343) and thickness_to_volume (9356-9385) over a
// whole (nCategories, nCells+1) array, as seaice_run_advection_incremental_remap
// calls them on the ice and snow volume.
extern "C" __global__ void seaice_ir_volume_to_thickness_v842(rk *volume, const rk *area, int n)
{
    const int i = ir_thread();
    if (i >= n) return;
    if (area[i] > C(0.0)) {
        volume[i] = D(volume[i], area[i]);
    } else {
        volume[i] = C(0.0);
    }
}

extern "C" __global__ void seaice_ir_thickness_to_volume_v842(rk *thickness, const rk *area, int n)
{
    const int i = ir_thread();
    if (i >= n) return;
    thickness[i] = M(area[i], thickness[i]);
}
