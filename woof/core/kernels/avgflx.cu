// woof/core/kernels/avgflx.cu
//
// WRF v4.7.1 ``do_avgflx_em = 1``: the history-interval mean of the
// mass-coupled wind fluxes (Registry/registry.avgflx AVGFLX_RUM/RVM/WWM),
// dyn_em/module_avgflx_em.F SUBROUTINE upd_avgflx, called by solve_em after
// the Runge-Kutta loop (solve_em.F:3536-3556) on the last RK stage's
// sumflux fluxes ru_m, rv_m and ww_m:
//
//   local_count = real(avgflx_count)
//   avgflx_rum = (local_count*avgflx_rum + ru_m)/(local_count+1.)
//
// One float32 rounding per Fortran operation, in Fortran's order (the unit
// compiles --fmad=false --ftz=false).  ``count`` is a one-word device
// counter (avgflx_count) read by every thread; the caller increments it
// after the launch, stream ordered, as solve_em does after its tile loop.
// A negative count is a pending zero_avgflx (the first step after a history
// write): the mean reads as WRF's zeroed word and the count as 0, so a -0.0
// flux averages to +0.0 exactly as it does in WRF.  One thread per element
// of one flux field (the caller launches the u-, v- and w-staggered fields).

extern "C" __global__
void upd_avgflx(float* __restrict__ avg, const float* __restrict__ flux,
                const int* __restrict__ count, long long n)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int c = count[0];
    float a = avg[idx];
    if (c < 0) {            // zero_avgflx, then avgflx_count = 0
        a = 0.0f;
        c = 0;
    }
    const float local_count = (float)c;
    avg[idx] = __fdiv_rn(__fadd_rn(__fmul_rn(local_count, a), flux[idx]),
                         __fadd_rn(local_count, 1.0f));
}
