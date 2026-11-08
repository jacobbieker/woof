// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's implicit vertical diffusion of temperature and salinity,
// src/OCE/TRA/trazdf.F90 (tra_zdf, tra_zdf_imp), for a linear free surface,
// no iso-neutral slopes, no adaptive-implicit vertical advection, no
// mass-flux convection and no double diffusion: the matrix is built from avt
// for temperature and reused for salinity, as tra_zdf_imp does when
// ln_zdfddm is off.  Double, NEMO's expressions in NEMO's order.  Fields are
// (jpk, jpj, jpi) C order with NEMO's one-cell halo; one thread per interior
// column.

#define ZDF_KMAX 128
#define Z3(k) ((size_t)((k) - 1) * n2 + c)

extern "C" __global__
void nemo_tra_zdf(const double* __restrict__ avt, const double* __restrict__ e3t,
                  const double* __restrict__ e3w, const double* __restrict__ tmask,
                  const double* __restrict__ tb, const double* __restrict__ sb,
                  const double* __restrict__ trhs, const double* __restrict__ srhs,
                  double* __restrict__ ta, double* __restrict__ sa, double p2dt,
                  int salt_floor, int nx, int ny, int jpk)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nx - 2, nj = ny - 2;
    if (t >= ni * nj || jpk >= ZDF_KMAX) return;
    int i = t % ni + 2, j = t / ni + 2;
    size_t n2 = (size_t)nx * ny, c = (size_t)(j - 1) * nx + (i - 1);
    int jpkm1 = jpk - 1;
    double zwi[ZDF_KMAX], zwt[ZDF_KMAX], zwd[ZDF_KMAX], zws[ZDF_KMAX];
    for (int k = 2; k <= jpk; ++k) zwt[k] = avt[Z3(k)];
    zwt[1] = 0.0;
    for (int k = 1; k <= jpkm1; ++k) {
        zwi[k] = - p2dt * zwt[k] / e3w[Z3(k)];
        zws[k] = - p2dt * zwt[k + 1] / e3w[Z3(k + 1)];
        zwd[k] = e3t[Z3(k)] - ( zwi[k] + zws[k] );
    }
    zwt[1] = zwd[1];
    for (int k = 2; k <= jpkm1; ++k)
        zwt[k] = zwd[k] - zwi[k] * zws[k - 1] / zwt[k - 1];
    for (int jn = 0; jn < 2; ++jn) {
        const double* b = jn ? sb : tb;
        const double* r = jn ? srhs : trhs;
        double* a = jn ? sa : ta;
        a[Z3(1)] = e3t[Z3(1)] * b[Z3(1)] + p2dt * e3t[Z3(1)] * r[Z3(1)];
        for (int k = 2; k <= jpkm1; ++k) {
            double zrhs = e3t[Z3(k)] * b[Z3(k)] + p2dt * e3t[Z3(k)] * r[Z3(k)];
            a[Z3(k)] = zrhs - zwi[k] / zwt[k - 1] * a[Z3(k - 1)];
        }
        a[Z3(jpkm1)] = a[Z3(jpkm1)] / zwt[jpkm1] * tmask[Z3(jpkm1)];
        for (int k = jpk - 2; k >= 1; --k)
            a[Z3(k)] = ( a[Z3(k)] - zws[k] * a[Z3(k + 1)] ) / zwt[k] * tmask[Z3(k)];
    }
    // tra_zdf (:71-73): salinity below zero is reset to 0.1 unless S-EOS
    // runs with rn_b0 = 0.
    if (salt_floor)
        for (int k = 1; k <= jpk; ++k)
            if (sa[Z3(k)] < 0.0) sa[Z3(k)] = 0.1;
}
