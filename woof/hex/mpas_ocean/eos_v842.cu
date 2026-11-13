// MPAS-Ocean v8.4.2: the equation of state and the freezing temperature.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_ocean/shared/mpas_ocn_equation_of_state_linear.F
//     (ocn_equation_of_state_linear_density)
//   src/core_ocean/shared/mpas_ocn_equation_of_state_jm.F
//     (ocn_equation_of_state_jm_density)
//   src/core_ocean/shared/mpas_ocn_equation_of_state.F
//     (ocn_freezing_temperature, ocn_freezing_temperature_salinity_deriv)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// One source for both precisions (RKIND 4 unless MPAS_RKIND8 is defined).
// Every operation is an explicit round-to-nearest intrinsic in the Fortran's
// evaluation order (gfortran -O0 -ffp-contract=off: left to right, the
// parentheses kept); x**2 is x*x and sqrt the correctly rounded instruction.
// JM's reference pressure calls exp: glibc's expf (gfk_exp, glibc_flt32.cuh)
// at RKIND 4 and glibc's exp (glibc_exp, glibc_flt64.cuh) at RKIND 8, the
// shared transcriptions eos_v842.module_source prepends. Fortran MIN/MAX are
// chosen on the words, as integer selects, so no compiler turns them into an
// IEEE min/max: the first argument unless the second compares less/greater.
//
// Layout: a(nVertLevels, nCells) is a[iCell*nVertLevels + k];
// activeTracers(nTracers, nVertLevels, nCells) is
// t[(iCell*nVertLevels + k)*nTracers + i]; tracersSurfaceValue(nTracers,
// nCells) is s[iCell*nTracers + i]. Tracer indices are 1-based.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define EXP(a) glibc_exp(a)
#define C(x) x
__device__ __forceinline__ rk eos_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk eos_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define EXP(a) gfk_exp(a)
#define C(x) x##f
__device__ __forceinline__ rk eos_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk eos_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif

// displacement types, as the driver passes them
#define EOS_ABSOLUTE 0
#define EOS_RELATIVE 1
#define EOS_SURFACE_DISPLACED 2

// JM's polynomial coefficients (mpas_ocn_equation_of_state_jm.F)
#define unt0 C(999.842594)
#define unt1 C(6.793952e-2)
#define unt2 C(-9.095290e-3)
#define unt3 C(1.001685e-4)
#define unt4 C(-1.120083e-6)
#define unt5 C(6.536332e-9)
#define uns1t0 C(0.824493)
#define uns1t1 C(-4.0899e-3)
#define uns1t2 C(7.6438e-5)
#define uns1t3 C(-8.2467e-7)
#define uns1t4 C(5.3875e-9)
#define unsqt0 C(-5.72466e-3)
#define unsqt1 C(1.0227e-4)
#define unsqt2 C(-1.6546e-6)
#define uns2t0 C(4.8314e-4)
#define bup0s0t0 C(1.965933e+4)
#define bup0s0t1 C(1.444304e+2)
#define bup0s0t2 C(-1.706103)
#define bup0s0t3 C(9.648704e-3)
#define bup0s0t4 C(-4.190253e-5)
#define bup0s1t0 C(5.284855e+1)
#define bup0s1t1 C(-3.101089e-1)
#define bup0s1t2 C(6.283263e-3)
#define bup0s1t3 C(-5.084188e-5)
#define bup0sqt0 C(3.886640e-1)
#define bup0sqt1 C(9.085835e-3)
#define bup0sqt2 C(-4.619924e-4)
#define bup1s0t0 C(3.186519)
#define bup1s0t1 C(2.212276e-2)
#define bup1s0t2 C(-2.984642e-4)
#define bup1s0t3 C(1.956415e-6)
#define bup1s1t0 C(6.704388e-3)
#define bup1s1t1 C(-1.847318e-4)
#define bup1s1t2 C(2.059331e-7)
#define bup1sqt0 C(1.480266e-4)
#define bup2s0t0 C(2.102898e-4)
#define bup2s0t1 C(-1.202016e-5)
#define bup2s0t2 C(1.394680e-7)
#define bup2s1t0 C(-2.040237e-6)
#define bup2s1t1 C(6.128773e-8)
#define bup2s1t2 C(6.207323e-10)

// pRefEOS(k): the reference pressure at level k's mid-depth, one thread per level
extern "C" __global__ void ocn_eos_jm_reference_pressure_v842(
    rk *pRefEOS, const rk *refBottomDepth, int nVertLevels)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nVertLevels) return;
    const rk depth = k == 0 ? M(C(0.5), refBottomDepth[0])
                            : M(C(0.5), A(refBottomDepth[k], refBottomDepth[k - 1]));
    pRefEOS[k] = A(A(M(C(0.059808), S(EXP(-M(C(0.025), depth)), C(1.0))), M(C(0.100766), depth)),
                   M(C(2.28405e-7), M(depth, depth)));
}

// The level whose reference pressure level k (0-based) takes: k itself when
// k_displaced is 0, else max(min(k + kd, nVertLevels), 1) ('relative') or
// max(min(kd, nVertLevels), 1) ('absolute'), 1-based in the Fortran
__device__ __forceinline__ int eos_reference_level(int k, int kDisplaced, int type, int nVertLevels)
{
    if (kDisplaced == 0) return k;
    int kTest = type == EOS_RELATIVE ? k + 1 + kDisplaced : kDisplaced;
    if (kTest > nVertLevels) kTest = nVertLevels;
    if (kTest < 1) kTest = 1;
    return kTest - 1;
}

// ocn_equation_of_state_jm_density, one thread per (cell, level) of the
// cells' active columns (k < maxLevelCell); the levels below keep their words
extern "C" __global__ void ocn_eos_jm_density_v842(
    rk *density, rk *thermalExpansionCoeff, rk *salineContractionCoeff, const rk *tracers,
    const rk *tracersSurfaceValue, const rk *pRefEOS, const int *maxLevelCell, int nCells, int nVertLevels,
    int nTracers, int indexT, int indexS, int kDisplaced, int type, int withAlpha, int withBeta)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells * nVertLevels) return;
    const int iCell = i / nVertLevels, k = i % nVertLevels;
    if (k >= maxLevelCell[iCell]) return;
    // surfaceDisplaced runs as 'relative' with k_displaced 0 on the surface values
    const int kd = type == EOS_SURFACE_DISPLACED ? 0 : kDisplaced;
    const rk p = pRefEOS[eos_reference_level(k, kd, type, nVertLevels)];
    const rk p2 = M(p, p);
    const rk tracerTemp = type == EOS_SURFACE_DISPLACED ? tracersSurfaceValue[iCell * nTracers + indexT - 1]
                                                        : tracers[i * nTracers + indexT - 1];
    const rk tracerSalt = type == EOS_SURFACE_DISPLACED ? tracersSurfaceValue[iCell * nTracers + indexS - 1]
                                                        : tracers[i * nTracers + indexS - 1];
    const rk SQ = eos_max(eos_min(tracerSalt, C(42.0)), C(0.0));
    const rk TQ = eos_max(eos_min(tracerTemp, C(40.0)), C(-2.0));
    const rk SQR = SQRT(SQ);
    const rk T2 = M(TQ, TQ);
    const rk WORK1 = A(A(uns1t0, M(uns1t1, TQ)), M(A(A(uns1t2, M(uns1t3, TQ)), M(uns1t4, T2)), T2));
    const rk WORK2 = M(SQR, A(A(unsqt0, M(unsqt1, TQ)), M(unsqt2, T2)));
    const rk RHO_S = A(A(M(unt1, TQ), M(A(A(unt2, M(unt3, TQ)), M(A(unt4, M(unt5, TQ)), T2)), T2)),
                       M(A(A(M(uns2t0, SQ), WORK1), WORK2), SQ));
    const rk WORK3 = A(A(A(A(bup0s1t0, M(bup0s1t1, TQ)), M(A(bup0s1t2, M(bup0s1t3, TQ)), T2)),
                         M(p, A(A(bup1s1t0, M(bup1s1t1, TQ)), M(bup1s1t2, T2)))),
                       M(p2, A(A(bup2s1t0, M(bup2s1t1, TQ)), M(bup2s1t2, T2))));
    const rk WORK4 = M(SQR, A(A(A(bup0sqt0, M(bup0sqt1, TQ)), M(bup0sqt2, T2)), M(bup1sqt0, p)));
    const rk BULK_MOD = A(A(A(A(A(bup0s0t0, M(bup0s0t1, TQ)),
                                M(A(A(bup0s0t2, M(bup0s0t3, TQ)), M(bup0s0t4, T2)), T2)),
                              M(p, A(A(bup1s0t0, M(bup1s0t1, TQ)), M(A(bup1s0t2, M(bup1s0t3, TQ)), T2)))),
                            M(p2, A(A(bup2s0t0, M(bup2s0t1, TQ)), M(bup2s0t2, T2)))),
                          M(SQ, A(WORK3, WORK4)));
    const rk DENOMK = D(C(1.0), S(BULK_MOD, p));
    const rk rho = M(M(A(unt0, RHO_S), BULK_MOD), DENOMK);
    density[i] = rho;
    if (withAlpha) {
        const rk DRDT0 = A(A(A(unt1, M(M(C(2.0), unt2), TQ)),
                             M(A(A(M(C(3.0), unt3), M(M(C(4.0), unt4), TQ)), M(M(C(5.0), unt5), T2)), T2)),
                           M(A(A(A(uns1t1, M(M(C(2.0), uns1t2), TQ)),
                                 M(A(M(C(3.0), uns1t3), M(M(C(4.0), uns1t4), TQ)), T2)),
                               M(A(unsqt1, M(M(C(2.0), unsqt2), TQ)), SQR)), SQ));
        const rk DKDT = A(A(A(A(A(bup0s0t1, M(M(C(2.0), bup0s0t2), TQ)),
                                M(A(M(C(3.0), bup0s0t3), M(M(C(4.0), bup0s0t4), TQ)), T2)),
                              M(p, A(A(bup1s0t1, M(M(C(2.0), bup1s0t2), TQ)), M(M(C(3.0), bup1s0t3), T2)))),
                            M(p2, A(bup2s0t1, M(M(C(2.0), bup2s0t2), TQ)))),
                          M(SQ, A(A(A(A(A(bup0s1t1, M(M(C(2.0), bup0s1t2), TQ)), M(M(C(3.0), bup0s1t3), T2)),
                                      M(p, A(bup1s1t1, M(M(C(2.0), bup1s1t2), TQ)))),
                                    M(p2, A(bup2s1t1, M(M(C(2.0), bup2s1t2), TQ)))),
                                  M(SQR, A(bup0sqt1, M(M(C(2.0), bup0sqt2), TQ))))));
        const rk DRHODT = M(DENOMK, S(M(DRDT0, BULK_MOD), M(M(M(p, A(unt0, RHO_S)), DKDT), DENOMK)));
        thermalExpansionCoeff[i] = -D(DRHODT, rho);
    }
    if (withBeta) {
        const rk DRDS0 = A(A(M(M(C(2.0), uns2t0), SQ), WORK1), M(C(1.5), WORK2));
        const rk DKDS = A(WORK3, M(C(1.5), WORK4));
        const rk DRHODS = M(DENOMK, S(M(DRDS0, BULK_MOD), M(M(M(p, A(unt0, RHO_S)), DKDS), DENOMK)));
        salineContractionCoeff[i] = D(DRHODS, rho);
    }
}

// ocn_equation_of_state_linear_density: the same threads; every displacement
// type is the same formula, on the surface values for 'surfaceDisplaced'
extern "C" __global__ void ocn_eos_linear_density_v842(
    rk *density, rk *thermalExpansionCoeff, rk *salineContractionCoeff, const rk *tracers,
    const rk *tracersSurfaceValue, const int *maxLevelCell, int nCells, int nVertLevels, int nTracers, int indexT,
    int indexS, int type, int withAlpha, int withBeta, rk densityref, rk alpha, rk beta, rk Tref, rk Sref)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells * nVertLevels) return;
    const int iCell = i / nVertLevels, k = i % nVertLevels;
    if (k >= maxLevelCell[iCell]) return;
    const rk T = type == EOS_SURFACE_DISPLACED ? tracersSurfaceValue[iCell * nTracers + indexT - 1]
                                               : tracers[i * nTracers + indexT - 1];
    const rk Sa = type == EOS_SURFACE_DISPLACED ? tracersSurfaceValue[iCell * nTracers + indexS - 1]
                                                : tracers[i * nTracers + indexS - 1];
    const rk rho = A(S(densityref, M(alpha, S(T, Tref))), M(beta, S(Sa, Sref)));
    density[i] = rho;
    if (withAlpha) thermalExpansionCoeff[i] = D(alpha, rho);
    if (withBeta) salineContractionCoeff[i] = D(beta, rho);
}

// ocn_freezing_temperature (deriv 0) or ocn_freezing_temperature_salinity_deriv
// (deriv 1), elementwise; pressureOffset = max(pressure - reference, 0)
extern "C" __global__ void ocn_freezing_temperature_v842(
    rk *out, const rk *salinity, const rk *pressure, int n, int deriv, rk coeff0, rk coeffS, rk coeffP,
    rk coeffPS, rk referencePressure)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const rk pressureOffset = eos_max(S(pressure[i], referencePressure), C(0.0));
    if (deriv) {
        out[i] = A(coeffS, M(coeffPS, pressureOffset));
    } else {
        out[i] = A(A(A(coeff0, M(coeffS, salinity[i])), M(coeffP, pressureOffset)),
                   M(M(coeffPS, pressureOffset), salinity[i]));
    }
}
