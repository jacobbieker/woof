// E3SM v3.2.0's ocean-input accumulation: the sums of prep_ocn_accum
// (driver-mct/main/prep_ocn_mod.F90:404-433, through mct_avect_accum,
// share/util/mct_mod.F90:649-778) and the average of prep_ocn_accum_avg
// (prep_ocn_mod.F90:437-465, through mct_aVect_avg, mct_mod.F90:794-832).
// real(r8); each operation an explicit round-to-nearest intrinsic
// (gfortran -O0 -ffp-contract=off).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f). driver-mct (cpl7) and the
// share code: University Corporation for Atmospheric Research and UChicago
// Argonne, LLC; E3SM: Copyright 2024, UChicago Argonne, LLC. BSD-3-Clause,
// the terms in licenses/LICENSE-E3SM-BSD-3-Clause.txt. This is not the
// version available from the E3SM project.
//
// The accumulator and x2o hold the same fields in the same order (prep_ocn_init
// builds the one from the other), so every word is its own: one thread per word.

typedef double r8;

// mct_avect_accum: aVout = aVout + avotmp, avotmp being aVin copied over zeros
extern "C" __global__ void ocn_accum_add(r8* __restrict__ acc, const r8* __restrict__ x2o, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = __dadd_rn(acc[i], x2o[i]);
}

// mct_aVect_avg: aVect = aVect * ravg, ravg = 1/counter (rounded once, on the host)
extern "C" __global__ void ocn_accum_scale(r8* __restrict__ acc, r8 ravg, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = __dmul_rn(acc[i], ravg);
}
