// MPAS land-ice (MALI) v8.4.2: the forward mode's own per-cell initialisation
// (li_calculate_mask_init and li_SGH_init).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/shared/mpas_li_mask.F:186-224 (li_calculate_mask_init)
//   src/core_landice/mode_forward/mpas_li_subglacial_hydro.F:75-191 (li_SGH_init),
//   1281-1361 (calc_pressure_diag_vars)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 unless MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). Fortran MIN/MAX return the first argument only if it
// compares less/greater and the second otherwise (so a tie of signed zeros
// returns the second), chosen on the words as an integer select so no
// compiler turns it into an IEEE min/max: the convention of
// subglacial_hydro_v842.cu, whose clamps these are.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define LI_C(x) x
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __longlong_as_double(a < b ? __double_as_longlong(a) : __double_as_longlong(b));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __longlong_as_double(a > b ? __double_as_longlong(a) : __double_as_longlong(b));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define LI_C(x) x##f
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __int_as_float(a < b ? __float_as_int(a) : __float_as_int(b));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __int_as_float(a > b ? __float_as_int(a) : __float_as_int(b));
}
#endif

#define LI_RHO_WATER LI_C(1000.0)          // li_constants rho_water
#define LI_GRAVITY LI_C(9.80616)           // li_constants gravity
#define LI_INITIAL_ICE_EXTENT 1            // li_mask_ValueInitialIceExtent
#define LI_ICE 32
#define LI_FLOATING 4

__device__ __forceinline__ bool li_is(int mask, int value) { return (mask & value) == value; }

// li_calculate_mask_init on a cold start: cellMask = 0, then the initial
// ice extent's bit where there is ice
extern "C" __global__ void li_mask_init_v842(int *cellMask, const rk *thickness, int nCells)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int mask = 0;
    if (thickness[i] > LI_C(0.0)) mask = mask | LI_INITIAL_ICE_EXTENT;
    cellMask[i] = mask;
}

// li_SGH_init's clamps of the input state and its ocean-pressure override,
// on the cellMask it is given (the model's: li_calculate_mask_init's alone)
extern "C" __global__ void li_sgh_init_v842(
    rk *waterThickness, rk *tillWaterThickness, rk *waterPressure, const rk *thickness,
    const rk *bedTopography, const int *cellMask, rk rhoi, rk rhoo, rk seaLevel, rk tillMax, int nCells)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    waterThickness[i] = li_max(LI_C(0.0), waterThickness[i]);
    rk till = li_max(LI_C(0.0), tillWaterThickness[i]);
    tillWaterThickness[i] = li_min(tillMax, till);
    rk p = li_max(LI_C(0.0), waterPressure[i]);
    p = li_min(p, M(M(rhoi, LI_GRAVITY), thickness[i]));
    const int mask = cellMask[i];
    if (li_is(mask, LI_FLOATING) || (!li_is(mask, LI_ICE) && bedTopography[i] < seaLevel))
        p = M(M(rhoo, LI_GRAVITY), S(seaLevel, bedTopography[i]));
    waterPressure[i] = p;
}

// calc_pressure_diag_vars
extern "C" __global__ void li_sgh_pressure_diag_v842(
    rk *effectivePressure, rk *hydropotentialBase, rk *hydropotential, const rk *waterPressure,
    const rk *waterThickness, const rk *thickness, const rk *bedTopography, const int *cellMask, rk rhoi,
    int nCells)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk p = waterPressure[i];
    rk n = S(M(M(rhoi, LI_GRAVITY), thickness[i]), p);
    if (!li_is(cellMask[i], LI_ICE)) n = LI_C(0.0);
    effectivePressure[i] = n;
    const rk base = A(M(M(LI_RHO_WATER, LI_GRAVITY), bedTopography[i]), p);
    hydropotentialBase[i] = base;
    hydropotential[i] = A(base, M(M(LI_RHO_WATER, LI_GRAVITY), waterThickness[i]));
}
