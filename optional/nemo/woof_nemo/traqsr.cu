// WOOF-OPTIN: nemo
// Derived from NEMO 5.0.2 (CeCILL v2.0); see optional/nemo/LICENSE and
// NOTICE.md.  Not part of the default build or of any wheel.
//
// NEMO's penetrative solar radiation with the 2-band scheme (ln_qsr_2bd),
// src/OCE/TRA/traqsr.F90: qsr_ext_lev, qsr_2BD and tra_qsr's leapfrog (MLF,
// no key_RK3) update, for a linear free surface (key_linssh) on a 3-D
// vertical grid (key_vco_3d).  NEMO computes in double (wp = dp): every
// operation here is a double in NEMO's order with round-to-nearest
// intrinsics, EXP and LOG are glibc's (glibc_exp / glibc_log), as the -O0
// reference calls exp and log.  3-D fields are (jpk, jpj, jpi) C order, i.e.
// NEMO's (jpi, jpj, jpk) memory order.

// qsr_ext_lev (traqsr.F90:1143-1213, the non-key_vco_1d branch): the deepest
// level the light reaches at 10.e-15 precision.  A domain reduction with
// MAXVAL/MINVAL/SUM; one thread.  MINVAL over an empty mask is HUGE(1._wp).
extern "C" __global__
void nemo_qsr_ext_lev(const double* __restrict__ gdepw0, const double* __restrict__ e3t0,
                      const double* __restrict__ tmask, const double* __restrict__ wmask,
                      double pL, double pfr, double rho0_rcp, double rDt,
                      int* __restrict__ klev_out, int jpk, int n2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double zprec = 10.e-15, zQmax = 1000.0;
    double zcoef = __ddiv_rn(__dmul_rn(zprec, rho0_rcp),
                             __dmul_rn(__dmul_rn(rDt, zQmax), pfr));
    int jpkm1 = jpk - 1;
    int klev = jpkm1;
    for (int jk = jpkm1; jk >= 1; --jk) {
        size_t lk = (size_t)(jk - 1) * n2, lk1 = (size_t)jk * n2;
        double tsum = 0.0;
        for (int c = 0; c < n2; ++c) tsum = __dadd_rn(tsum, tmask[lk + c]);
        if (tsum > 0.0) {
            double zdw = -1.7976931348623157e308;
            double ze3t = 1.7976931348623157e308;
            for (int c = 0; c < n2; ++c) {
                double w = __dmul_rn(gdepw0[lk1 + c], wmask[lk + c]);
                if (w > zdw) zdw = w;
                if (wmask[lk1 + c] == 1.0 && e3t0[lk + c] < ze3t) ze3t = e3t0[lk + c];
            }
            double zhext = -__dmul_rn(pL, glibc_log(__dmul_rn(zcoef, ze3t)));
            if (zdw >= zhext) klev = jk;
        } else {
            klev = jk;
        }
    }
    klev_out[0] = klev;
}

// One tra_qsr call with np_2BD (traqsr.F90:105-252 and qsr_2BD :627-712), one
// thread per column: the leapfrog swap of the heat-content trend, qsr_2BD's
// two-band attenuation, the temperature trend over the top nksr levels and
// fraqsr_1lev.  first = (kt == nit000) with no restart: z1_2 = 1 and a zero
// before-trend.
extern "C" __global__
void nemo_tra_qsr_2bd(const double* __restrict__ qsr, const double* __restrict__ gdepw,
                      const double* __restrict__ e3t, const double* __restrict__ wmask,
                      double* __restrict__ qsr_hc, double* __restrict__ qsr_hc_b,
                      double* __restrict__ ta_rhs, double* __restrict__ fraqsr_1lev,
                      double rn_abs, double r1_si0, double r1_si1, double r1_rho0_rcp,
                      int nk0, int nkV, int nksr, int first, int jpk, int n2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n2) return;
#define L3(k) ((size_t)((k) - 1) * n2 + c)
    double z1_2;
    if (first) {
        z1_2 = 1.0;
        for (int jk = 1; jk <= jpk; ++jk) qsr_hc_b[L3(jk)] = 0.0;
    } else {
        z1_2 = 0.5;
        for (int jk = 1; jk <= jpk; ++jk) qsr_hc_b[L3(jk)] = qsr_hc[L3(jk)];
    }
    // qsr_2BD
    double zz0 = __dmul_rn(rn_abs, r1_rho0_rcp);
    double zz1 = __dmul_rn(__dsub_rn(1.0, rn_abs), r1_rho0_rcp);
    double g1 = gdepw[L3(1)];
    double zatt = __dadd_rn(__dmul_rn(zz0, glibc_exp(-__dmul_rn(g1, r1_si0))),
                            __dmul_rn(zz1, glibc_exp(-__dmul_rn(g1, r1_si1))));
    double q = qsr[c];
    for (int jk = 1; jk <= nk0; ++jk) {
        double g = gdepw[L3(jk + 1)];
        double zzatt = __dmul_rn(__dadd_rn(__dmul_rn(zz0, glibc_exp(-__dmul_rn(g, r1_si0))),
                                           __dmul_rn(zz1, glibc_exp(-__dmul_rn(g, r1_si1)))),
                                 wmask[L3(jk + 1)]);
        qsr_hc[L3(jk)] = __dmul_rn(q, __dsub_rn(zatt, zzatt));
        zatt = zzatt;
    }
    for (int jk = nk0 + 1; jk <= nkV; ++jk) {
        double g = gdepw[L3(jk + 1)];
        double zzatt = __dmul_rn(__dmul_rn(zz1, glibc_exp(-__dmul_rn(g, r1_si1))),
                                 wmask[L3(jk + 1)]);
        qsr_hc[L3(jk)] = __dmul_rn(q, __dsub_rn(zatt, zzatt));
        zatt = zzatt;
    }
    // tra_qsr, MLF: the trend over the top nksr levels and fraqsr_1lev
    for (int jk = 1; jk <= nksr; ++jk)
        ta_rhs[L3(jk)] = __dadd_rn(ta_rhs[L3(jk)],
            __ddiv_rn(__dmul_rn(z1_2, __dadd_rn(qsr_hc_b[L3(jk)], qsr_hc[L3(jk)])),
                      e3t[L3(jk)]));
    double zz0m = __dmul_rn(r1_rho0_rcp, q);
    if (zz0m != 0.0) fraqsr_1lev[c] = __ddiv_rn(qsr_hc[L3(1)], zz0m);
    else fraqsr_1lev[c] = 1.0;
#undef L3
}
