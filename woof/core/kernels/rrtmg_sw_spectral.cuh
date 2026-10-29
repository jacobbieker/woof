// RRTMG shortwave spectral output kernels, appended to the shortwave unit
// by both of its compiles: woof.core.rrtmg_sw.CudaSW (WRF) and
// woof.hex.mpasphys.rrtmg_sw_v842.MPASShortwave (MPAS).  Needs the unit's
// real, AD, MU, RSW_NGPT and RSW_NBND.

// ---------------------------------------------------------------------------
// Spectral output (rrtmg_spectral_output): the six full-spectrum sums of
// rsw_spc_accum_b resolved by band or by g-point.  WRF's spcvmc_sw carries
// the band accumulation as inactive code (module_ra_rrtmg_sw.F "Accumulate
// spectral fluxes over bands - inactive"): per band, the g-points in
// ascending iw, zincflx(iw)*flux.  These kernels are separate launches that
// read the same buffers, so the broadband words are untouched.
// Output quantity order q: fu, fd, cu, cd, fddir, cddir.
// ---------------------------------------------------------------------------

__device__ __forceinline__ const real* rsw_spectral_input(
    int q, const real* zfu, const real* zfd, const real* zcu, const real* zcd,
    const real* ztdbt_nodel, const real* ztdbtc_nodel)
{
    return q == 0 ? zfu : q == 1 ? zfd : q == 2 ? zcu : q == 3 ? zcd
         : q == 4 ? ztdbt_nodel : ztdbtc_nodel;
}

// thread -> (q, col, K, band); out (6, ncol, nlayers+1, 14)
extern "C" __global__
void rsw_spc_band_accum_b(int ncol, int nlayers,
                          const int* __restrict__ ngb,
                          const real* __restrict__ zincflx,   // (ncol, 112)
                          const real* __restrict__ zcd,       // (ncol, n1, 112)
                          const real* __restrict__ zcu,
                          const real* __restrict__ zfd, const real* __restrict__ zfu,
                          const real* __restrict__ ztdbt_nodel,
                          const real* __restrict__ ztdbtc_nodel,
                          real* out)
{
    long long n1 = (long long)nlayers + 1;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 6LL * ncol * n1 * RSW_NBND) return;
    int band = (int)(idx % RSW_NBND);
    long long rest = idx / RSW_NBND;
    int K = (int)(rest % n1);
    rest /= n1;
    int col = (int)(rest % ncol);
    int q = (int)(rest / ncol);
    const real* in = rsw_spectral_input(q, zfu, zfd, zcu, zcd, ztdbt_nodel, ztdbtc_nodel);
    size_t row = ((size_t)col * n1 + (size_t)(nlayers - K)) * RSW_NGPT;
    const real* w = zincflx + (size_t)col * RSW_NGPT;
    real acc = 0.0f;
    for (int iw = 0; iw < RSW_NGPT; ++iw)
        if (ngb[iw] - 16 == band)
            acc = AD(acc, MU(w[iw], in[row + iw]));
    out[idx] = acc;
}

// thread -> (q, col, K, iw); out (6, ncol, nlayers+1, 112): zincflx(iw)*flux
extern "C" __global__
void rsw_spc_gpt_b(int ncol, int nlayers,
                   const real* __restrict__ zincflx,
                   const real* __restrict__ zcd, const real* __restrict__ zcu,
                   const real* __restrict__ zfd, const real* __restrict__ zfu,
                   const real* __restrict__ ztdbt_nodel,
                   const real* __restrict__ ztdbtc_nodel,
                   real* out)
{
    long long n1 = (long long)nlayers + 1;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 6LL * ncol * n1 * RSW_NGPT) return;
    int iw = (int)(idx % RSW_NGPT);
    long long rest = idx / RSW_NGPT;
    int K = (int)(rest % n1);
    rest /= n1;
    int col = (int)(rest % ncol);
    int q = (int)(rest / ncol);
    const real* in = rsw_spectral_input(q, zfu, zfd, zcu, zcd, ztdbt_nodel, ztdbtc_nodel);
    size_t row = ((size_t)col * n1 + (size_t)(nlayers - K)) * RSW_NGPT;
    out[idx] = MU(zincflx[(size_t)col * RSW_NGPT + iw], in[row + iw]);
}
