// Tangent-linear and adjoint of woof's strict calc_coefs
// (woof/core/kernels/acoustic.cu): c2a = gamma*p/alt (WRF small_step_prep)
// and the LU factors a/alpha/gam of the implicit w'' system (WRF
// calc_coef_w).  The oracle for the factors is WRFPLUS 4.7.1
// g_/a_calc_coef_w (wrftladj/module_small_step_em_{tl,ad}.F); see
// docs/dev/P3A-WRF-DRYCORE-ADJOINT.md.
//
// wreal is binary64 unless the unit is built with -DWRFPLUS_REAL=float.
// The base-state statements keep the strict primal's operation order, so
// the float instantiation of calc_coef_w_tl's base state gives the primal's
// words.
//
// Layout (k, j, i).  p, alt, c2a at half levels (nz, ny, nx); a, alpha,
// gam, cqw at w levels (nz+1, ny, nx); mup (ny, nx).  One thread per
// column, every column.  As in the primal, a[0] and alpha[0] are not
// written (their perturbations and adjoints pass through); a[1], gam[0]
// and gam[nz] are written as constants.
//
// Active inputs: p, alt, mup, cqw (when moist_cq).  Outputs: c2a, a,
// alpha, gam.  Passive: rdn, rdnw, c1h, c2h, c1f, c2f, mub2d.

#ifndef WRFPLUS_REAL
#define WRFPLUS_REAL double
#endif
typedef WRFPLUS_REAL wreal;

#ifndef WV_LEV
#define WV_LEV 129
#endif

#define WG ((wreal)9.81)
#define WGAMMA ((wreal)1.4)

#define CW_PASSIVE                                                          \
    const wreal* __restrict__ rdn, const wreal* __restrict__ rdnw,         \
    const wreal* __restrict__ c1h, const wreal* __restrict__ c2h,          \
    const wreal* __restrict__ c1f, const wreal* __restrict__ c2f,          \
    const wreal* __restrict__ mub2d

#define CW_GEOM int nz, int ny, int nx, int moist_cq, int top_lid,          \
    wreal dtau, wreal epssm

// The cqw factor: cqw on a moist column, else exactly 1 (and inactive).
#define CQ(k) (moist_cq ? cqw[(size_t)(k) * st + c] : (wreal)1.0)

extern "C" __global__
void calc_coef_w_tl(wreal* __restrict__ c2a, wreal* __restrict__ a,
                    wreal* __restrict__ alpha, wreal* __restrict__ gam,
                    wreal* __restrict__ c2a_d, wreal* __restrict__ a_d,
                    wreal* __restrict__ alpha_d, wreal* __restrict__ gam_d,
                    const wreal* __restrict__ p, const wreal* __restrict__ alt,
                    const wreal* __restrict__ mup, const wreal* __restrict__ cqw,
                    const wreal* __restrict__ p_d, const wreal* __restrict__ alt_d,
                    const wreal* __restrict__ mup_d, const wreal* __restrict__ cqw_d,
                    CW_PASSIVE, int nl, CW_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    size_t st = (size_t)ny * nx;
    wreal mut = mub2d[c] + mup[c];
    wreal mut_d = mup_d[c];
    wreal cof = (wreal)0.5 * dtau * WG * ((wreal)1.0 + epssm);
    cof *= cof;

    wreal cl[WV_LEV], cd[WV_LEV];
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        cl[k] = WGAMMA * p[h] / alt[h];
        cd[k] = WGAMMA * (p_d[h] - p[h] / alt[h] * alt_d[h]) / alt[h];
    }
    wreal al_[WV_LEV], ald[WV_LEV], gl[WV_LEV], gd[WV_LEV];
    gl[0] = 0; gd[0] = 0;
    al_[1] = 0; ald[1] = 0;                   // a[1]: w[0] is a BC
    for (int k = 2; k < nz; ++k) {
        wreal h = c1h[k - 1] * mut + c2h[k - 1], f = c1f[k - 1] * mut + c2f[k - 1];
        wreal cq = CQ(k);
        wreal cqd = moist_cq ? cqw_d[(size_t)k * st + c] : (wreal)0.0;
        if (moist_cq)
            al_[k] = -cq * cof * rdn[k] * rdnw[k - 1] * cl[k - 1] / (h * f);
        else
            al_[k] = -cof * rdn[k] * rdnw[k - 1] * cl[k - 1] / (h * f);
        wreal g1 = -cof * rdn[k] * rdnw[k - 1] / (h * f);
        ald[k] = g1 * (cqd * cl[k - 1] + cq * cd[k - 1])
               - al_[k] * mut_d * (c1h[k - 1] / h + c1f[k - 1] / f);
    }
    if (top_lid) {
        al_[nz] = -(wreal)0.0; ald[nz] = 0;
    } else {
        wreal h = c1h[nz - 1] * mut + c2h[nz - 1], f = c1f[nz - 1] * mut + c2f[nz - 1];
        al_[nz] = -(wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]) * cl[nz - 1] / (h * f);
        ald[nz] = -(wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]) * cd[nz - 1] / (h * f)
                - al_[nz] * mut_d * (c1h[nz - 1] / h + c1f[nz - 1] / f);
    }
    for (int k = 1; k < nz; ++k) {
        wreal chm_k = c1h[k] * mut + c2h[k];
        wreal chm_km = c1h[k - 1] * mut + c2h[k - 1];
        wreal cfm_k = c1f[k] * mut + c2f[k];
        wreal cfm_kp = c1f[k + 1] * mut + c2f[k + 1];
        wreal cq = CQ(k);
        wreal cqd = moist_cq ? cqw_d[(size_t)k * st + c] : (wreal)0.0;
        wreal b, cc;
        if (moist_cq) {
            b = (wreal)1.0 + cq * cof * rdn[k]
                * (rdnw[k] * cl[k] / (chm_k * cfm_k)
                   + rdnw[k - 1] * cl[k - 1] / (chm_km * cfm_k));
            cc = -cq * cof * rdn[k] * rdnw[k] * cl[k] / (chm_k * cfm_kp);
        } else {
            b = (wreal)1.0 + cof * rdn[k]
                * (rdnw[k] * cl[k] / (chm_k * cfm_k)
                   + rdnw[k - 1] * cl[k - 1] / (chm_km * cfm_k));
            cc = -cof * rdn[k] * rdnw[k] * cl[k] / (chm_k * cfm_kp);
        }
        wreal t1 = rdnw[k] * cl[k] / (chm_k * cfm_k);
        wreal t2 = rdnw[k - 1] * cl[k - 1] / (chm_km * cfm_k);
        wreal t1d = rdnw[k] * cd[k] / (chm_k * cfm_k)
                  - t1 * mut_d * (c1h[k] / chm_k + c1f[k] / cfm_k);
        wreal t2d = rdnw[k - 1] * cd[k - 1] / (chm_km * cfm_k)
                  - t2 * mut_d * (c1h[k - 1] / chm_km + c1f[k] / cfm_k);
        wreal bd = cof * rdn[k] * (cqd * (t1 + t2) + cq * (t1d + t2d));
        wreal g0 = -cof * rdn[k] * rdnw[k] / (chm_k * cfm_kp);
        wreal ccd = g0 * (cqd * cl[k] + cq * cd[k])
                  - cc * mut_d * (c1h[k] / chm_k + c1f[k + 1] / cfm_kp);
        wreal den = b - al_[k] * gl[k - 1];
        wreal dend = bd - ald[k] * gl[k - 1] - al_[k] * gd[k - 1];
        wreal alv = (wreal)1.0 / den;
        wreal alvd = -dend * alv * alv;
        alpha_d[(size_t)k * st + c] = alvd;
        if (nl) alpha[(size_t)k * st + c] = alv;
        gl[k] = cc * alv;
        gd[k] = ccd * alv + cc * alvd;
    }
    {
        wreal h = c1h[nz - 1] * mut + c2h[nz - 1], f = c1f[nz] * mut + c2f[nz];
        wreal kk = (wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]);
        wreal t = kk * cl[nz - 1] / (h * f);
        wreal b_top = (wreal)1.0 + (wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1])
                      * cl[nz - 1] / ((c1h[nz - 1] * mut + c2h[nz - 1])
                                      * (c1f[nz] * mut + c2f[nz]));
        wreal bd = kk * cd[nz - 1] / (h * f)
                 - t * mut_d * (c1h[nz - 1] / h + c1f[nz] / f);
        wreal den = b_top - al_[nz] * gl[nz - 1];
        wreal dend = bd - ald[nz] * gl[nz - 1] - al_[nz] * gd[nz - 1];
        wreal alv = (wreal)1.0 / den;
        alpha_d[(size_t)nz * st + c] = -dend * alv * alv;
        if (nl) alpha[(size_t)nz * st + c] = alv;
        gl[nz] = 0; gd[nz] = 0;
    }
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        c2a_d[h] = cd[k];
        if (nl) c2a[h] = cl[k];
    }
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        gam_d[f] = gd[k];
        if (nl) gam[f] = gl[k];
        if (k >= 1) {
            a_d[f] = ald[k];
            if (nl) a[f] = al_[k];
        }
    }
}

// Adjoint.  On entry c2a_a, a_a, alpha_a, gam_a hold the output adjoints;
// the written entries' adjoints are consumed (zeroed) and those of p, alt,
// mup and cqw are added.
extern "C" __global__
void calc_coef_w_ad(wreal* __restrict__ c2a_a, wreal* __restrict__ a_a,
                    wreal* __restrict__ alpha_a, wreal* __restrict__ gam_a,
                    const wreal* __restrict__ p, const wreal* __restrict__ alt,
                    const wreal* __restrict__ mup, const wreal* __restrict__ cqw,
                    wreal* __restrict__ p_a, wreal* __restrict__ alt_a,
                    wreal* __restrict__ mup_a, wreal* __restrict__ cqw_a,
                    CW_PASSIVE, CW_GEOM)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WV_LEV) return;
    size_t st = (size_t)ny * nx;
    wreal mut = mub2d[c] + mup[c];
    wreal cof = (wreal)0.5 * dtau * WG * ((wreal)1.0 + epssm);
    cof *= cof;

    // Forward recompute.
    wreal cl[WV_LEV], al_[WV_LEV], gl[WV_LEV], alv[WV_LEV];
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        cl[k] = WGAMMA * p[h] / alt[h];
    }
    gl[0] = 0; al_[1] = 0;
    for (int k = 2; k < nz; ++k) {
        wreal h = c1h[k - 1] * mut + c2h[k - 1], f = c1f[k - 1] * mut + c2f[k - 1];
        al_[k] = -CQ(k) * cof * rdn[k] * rdnw[k - 1] * cl[k - 1] / (h * f);
    }
    if (top_lid) {
        al_[nz] = 0;
    } else {
        wreal h = c1h[nz - 1] * mut + c2h[nz - 1], f = c1f[nz - 1] * mut + c2f[nz - 1];
        al_[nz] = -(wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]) * cl[nz - 1] / (h * f);
    }
    for (int k = 1; k < nz; ++k) {
        wreal chm_k = c1h[k] * mut + c2h[k], chm_km = c1h[k - 1] * mut + c2h[k - 1];
        wreal cfm_k = c1f[k] * mut + c2f[k], cfm_kp = c1f[k + 1] * mut + c2f[k + 1];
        wreal b = (wreal)1.0 + CQ(k) * cof * rdn[k]
                * (rdnw[k] * cl[k] / (chm_k * cfm_k)
                   + rdnw[k - 1] * cl[k - 1] / (chm_km * cfm_k));
        wreal cc = -CQ(k) * cof * rdn[k] * rdnw[k] * cl[k] / (chm_k * cfm_kp);
        alv[k] = (wreal)1.0 / (b - al_[k] * gl[k - 1]);
        gl[k] = cc * alv[k];
    }

    // Reverse.
    wreal acl[WV_LEV], aa[WV_LEV], ag[WV_LEV];
    for (int k = 0; k < nz; ++k) acl[k] = c2a_a[(size_t)k * st + c];
    for (int k = 0; k <= nz; ++k) {
        ag[k] = gam_a[(size_t)k * st + c];
        aa[k] = k >= 1 ? a_a[(size_t)k * st + c] : (wreal)0.0;
    }
    wreal amut = 0;
    {   // top row; gam[nz] = 0 is a constant.
        wreal h = c1h[nz - 1] * mut + c2h[nz - 1], f = c1f[nz] * mut + c2f[nz];
        wreal kk = (wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]);
        wreal t = kk * cl[nz - 1] / (h * f);
        wreal den = (wreal)1.0 + t - al_[nz] * gl[nz - 1];
        wreal av = (wreal)1.0 / den;
        wreal aden = -alpha_a[(size_t)nz * st + c] * av * av;
        aa[nz] += -aden * gl[nz - 1];
        ag[nz - 1] += -aden * al_[nz];
        acl[nz - 1] += aden * kk / (h * f);
        amut += -aden * t * (c1h[nz - 1] / h + c1f[nz] / f);
    }
    for (int k = nz - 1; k >= 1; --k) {
        size_t fk = (size_t)k * st + c;
        wreal chm_k = c1h[k] * mut + c2h[k], chm_km = c1h[k - 1] * mut + c2h[k - 1];
        wreal cfm_k = c1f[k] * mut + c2f[k], cfm_kp = c1f[k + 1] * mut + c2f[k + 1];
        wreal cq = CQ(k);
        wreal g0 = -cof * rdn[k] * rdnw[k] / (chm_k * cfm_kp);
        wreal cc = g0 * cq * cl[k];
        wreal aal = ag[k] * cc + alpha_a[fk];
        wreal acc = ag[k] * alv[k];
        wreal aden = -aal * alv[k] * alv[k];
        aa[k] += -aden * gl[k - 1];
        ag[k - 1] += -aden * al_[k];
        // cc
        if (moist_cq) cqw_a[fk] += acc * g0 * cl[k];
        acl[k] += acc * g0 * cq;
        amut += -acc * cc * (c1h[k] / chm_k + c1f[k + 1] / cfm_kp);
        // b
        wreal t1 = rdnw[k] * cl[k] / (chm_k * cfm_k);
        wreal t2 = rdnw[k - 1] * cl[k - 1] / (chm_km * cfm_k);
        if (moist_cq) cqw_a[fk] += aden * cof * rdn[k] * (t1 + t2);
        wreal as = aden * cq * cof * rdn[k];
        acl[k] += as * rdnw[k] / (chm_k * cfm_k);
        amut += -as * t1 * (c1h[k] / chm_k + c1f[k] / cfm_k);
        acl[k - 1] += as * rdnw[k - 1] / (chm_km * cfm_k);
        amut += -as * t2 * (c1h[k - 1] / chm_km + c1f[k] / cfm_k);
    }
    if (!top_lid) {
        wreal h = c1h[nz - 1] * mut + c2h[nz - 1], f = c1f[nz - 1] * mut + c2f[nz - 1];
        acl[nz - 1] += -aa[nz] * (wreal)2.0 * cof * (rdnw[nz - 1] * rdnw[nz - 1]) / (h * f);
        amut += -aa[nz] * al_[nz] * (c1h[nz - 1] / h + c1f[nz - 1] / f);
    }
    for (int k = 2; k < nz; ++k) {
        wreal h = c1h[k - 1] * mut + c2h[k - 1], f = c1f[k - 1] * mut + c2f[k - 1];
        wreal g1 = -cof * rdn[k] * rdnw[k - 1] / (h * f);
        if (moist_cq) cqw_a[(size_t)k * st + c] += aa[k] * g1 * cl[k - 1];
        acl[k - 1] += aa[k] * g1 * CQ(k);
        amut += -aa[k] * al_[k] * (c1h[k - 1] / h + c1f[k - 1] / f);
    }
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        p_a[h] += acl[k] * WGAMMA / alt[h];
        alt_a[h] += -acl[k] * cl[k] / alt[h];
        c2a_a[h] = 0;
    }
    mup_a[c] += amut;
    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        gam_a[f] = 0;
        if (k >= 1) { a_a[f] = 0; alpha_a[f] = 0; }
    }
}
