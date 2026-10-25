// WRF v4.7.1 vertical grid nesting (vert_refine_method = 2): a child on
// its own eta ladder under the parent's p_top, hybrid_opt and etac.
//
//   nest_vert_interp     one column of a parent field onto the child's
//                        levels: share/interp_fcn.F vert_interp_vert_nesting
//                        (:1910-1965, half levels, with the surface and top
//                        points extrapolated by the parent's cf1..cfn1) and
//                        vert_interp_vert_nesting_w (:1969-2013, full
//                        levels), both through inter_wrf_copy (:2073-2106)
//                        and coeff_mon_wrf_copy (:2111-2152), the monotone
//                        cubic Hermite in the reference log-pressure height.
//   nest_vert_rebalance  external/RSL_LITE/force_domain_em_part2.F:139-300:
//                        after the vertical interpolation, t_2 and QV are
//                        uncoupled with the CHILD's column mass, the child's
//                        base state (pb, t_init, alb, phb) is rebuilt on the
//                        parent's columns, the perturbation pressure and
//                        inverse density are integrated down from the top,
//                        ph_2 is rebuilt hydrostatically, and the three are
//                        coupled again with the child's coefficients.
//
// One thread owns one column.  Fields are (nlev, ncol) slabs: level k of
// column c at k*ncol + c, the layout woof's (nz, ny, nx) arrays have when
// (ny, nx) is flattened.
//
// The cubic's segment search (inter_wrf_copy's double loop) depends only on
// the two reference columns, which are the same for every column, so the
// host resolves it once (woof/core/nest_vertical.py) and passes seg[i], the
// interval of child point i; the cubic's coefficients for that interval are
// formed here, statement for statement as coeff_mon_wrf_copy forms them for
// the whole column, so no per-level array is held.
//
// Every product and sum is WRF's, in WRF's order; Fortran's LOG and real
// ** are glibc's logf/powf (gfk_log/gfk_pow, glibc_flt32.cuh), SIGN(1.,x)
// is copysignf.  Compiled -fmad=false --ftz=false
// (woof/core/nest_vertical.py) so nothing is fused and nothing flushed.

#define NV_IDX(k, c) ((long long)(k) * (long long)ncol + (long long)(c))

// pro_c(m) of the column: the parent's values with, on half levels, the
// extrapolated surface (m = 0) and top (m = nsrc + 1) points of
// vert_interp_vert_nesting (:1945-1953).
__device__ __forceinline__ float nv_pro(const float* __restrict__ src,
                                         int m, int nsrc, int wstag,
                                         long long ncol, int c,
                                         float cf1, float cf2, float cf3,
                                         float cfn, float cfn1)
{
    if (wstag) return src[NV_IDX(m, c)];
    if (m == 0) {
        float a = cf1 * src[NV_IDX(0, c)];
        float b = cf2 * src[NV_IDX(1, c)];
        float s = a + b;
        float d = cf3 * src[NV_IDX(2, c)];
        return s + d;
    }
    if (m == nsrc + 1) {
        float a = cfn * src[NV_IDX(nsrc - 1, c)];
        float b = cfn1 * src[NV_IDX(nsrc - 2, c)];
        return a + b;
    }
    return src[NV_IDX(m - 1, c)];
}

// coeff_mon_wrf_copy's yp(m) (:2124-2140) of a column of n points.
__device__ __forceinline__ float nv_slope(const float* __restrict__ x,
                                           const float* __restrict__ src,
                                           int m, int n, int nsrc, int wstag,
                                           long long ncol, int c,
                                           float cf1, float cf2, float cf3,
                                           float cfn, float cfn1)
{
    // s(i) = (y(i+1)-y(i)) / h(i), h(i) = x(i+1)-x(i)
    if (m == 0) {
        float h0 = x[1] - x[0];
        float y0 = nv_pro(src, 0, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
        float y1 = nv_pro(src, 1, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
        return (y1 - y0) / h0;              // p(1) = s(1), yp(1) = p(1)
    }
    if (m == n - 1) {
        float h = x[n - 1] - x[n - 2];
        float ya = nv_pro(src, n - 2, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
        float yb = nv_pro(src, n - 1, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
        return (yb - ya) / h;               // p(n) = s(n-1)
    }
    float hm = x[m] - x[m - 1];
    float hp = x[m + 1] - x[m];
    float ym = nv_pro(src, m - 1, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
    float y0 = nv_pro(src, m, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
    float yp = nv_pro(src, m + 1, nsrc, wstag, ncol, c, cf1, cf2, cf3, cfn, cfn1);
    float sm = (y0 - ym) / hm;
    float sp = (yp - y0) / hp;
    // p(i) = (s(i-1)*h(i)+s(i)*h(i-1)) / (h(i-1)+h(i))
    float t1 = sm * hp;
    float t2 = sp * hm;
    float num = t1 + t2;
    float den = hm + hp;
    float p = num / den;
    // yp(i) = (sign(1.,s(i-1))+sign(1.,s(i)))*min(abs(s(i-1)),abs(s(i)),0.5*abs(p(i)))
    float sg = copysignf(1.0f, sm) + copysignf(1.0f, sp);
    float hp5 = 0.5f * fabsf(p);
    float mn = fminf(fminf(fabsf(sm), fabsf(sp)), hp5);
    return sg * mn;
}

extern "C" __global__ void nest_vert_interp(
    const float* __restrict__ src, float* __restrict__ dst,
    const float* __restrict__ alt_c, const float* __restrict__ alt_n,
    const int* __restrict__ seg, int nsrc, int ndst, int ncol, int wstag,
    float cf1, float cf2, float cf3, float cfn, float cfn1)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    // Points on the coarse and the fine 1-D grids: half levels carry the
    // two extrapolated end points, full levels do not.
    const int n_c = wstag ? nsrc : nsrc + 2;
    const int n_n = wstag ? ndst : ndst + 2;
    const long long nc = (long long)ncol;
    for (int k = 0; k < ndst; ++k) {
        // child level k is point i of the fine grid
        const int i = wstag ? k : k + 1;
        float value;
        if (i == n_n - 1) {
            // pro_n(kde_n) = pro_c(kde_c) (:2103)
            value = nv_pro(src, n_c - 1, nsrc, wstag, nc, c, cf1, cf2, cf3, cfn, cfn1);
        } else {
            const int j = seg[i];
            if (j < 0) {
                value = __int_as_float(0x7fc00000);
            } else {
                float h = alt_c[j + 1] - alt_c[j];
                float yj = nv_pro(src, j, nsrc, wstag, nc, c, cf1, cf2, cf3, cfn, cfn1);
                float yj1 = nv_pro(src, j + 1, nsrc, wstag, nc, c, cf1, cf2, cf3, cfn, cfn1);
                float s = (yj1 - yj) / h;
                float ypj = nv_slope(alt_c, src, j, n_c, nsrc, wstag, nc, c,
                                     cf1, cf2, cf3, cfn, cfn1);
                float ypj1 = nv_slope(alt_c, src, j + 1, n_c, nsrc, wstag, nc, c,
                                      cf1, cf2, cf3, cfn, cfn1);
                // a(i) = (yp(i)+yp(i+1)-2.*s(i))/(h(i)*h(i))
                float a = ((ypj + ypj1) - 2.0f * s) / (h * h);
                // b(i) = (3.*s(i)-2.*yp(i)-yp(i+1))/h(i)
                float b = ((3.0f * s - 2.0f * ypj) - ypj1) / h;
                float cc = ypj;
                float d = yj;
                float p = alt_n[i] - alt_c[j];
                // pro_n(i) = p*( p*(a(j)*p+b(j))+c(j)) + d(j)
                float inner = a * p + b;
                float mid = p * inner + cc;
                value = p * mid + d;
            }
        }
        dst[NV_IDX(k, c)] = value;
    }
}

// force_domain_em_part2.F:139-300 on one column.  t and qv (nz) and ph
// (nz+1) are the vertically interpolated COUPLED fields and are rebalanced
// in place; pb, alb (nz), phb (nz+1) and al (nz) are written as WRF's
// intermediate grid carries them.  Arrays c1h..c4h, dnw, rdnw, rdn are the
// child's (nz), c1f..c4f its (nz+1); mub, mu, ht the parent's columns.
extern "C" __global__ void nest_vert_rebalance(
    float* __restrict__ t, float* __restrict__ qv, float* __restrict__ ph,
    float* __restrict__ pb, float* __restrict__ alb,
    float* __restrict__ phb, float* __restrict__ al,
    const float* __restrict__ mub, const float* __restrict__ mu,
    const float* __restrict__ ht,
    const float* __restrict__ c1h, const float* __restrict__ c2h,
    const float* __restrict__ c3h, const float* __restrict__ c4h,
    const float* __restrict__ c1f, const float* __restrict__ c2f,
    const float* __restrict__ c3f, const float* __restrict__ c4f,
    const float* __restrict__ dnw, const float* __restrict__ rdnw,
    const float* __restrict__ rdn,
    int nz, int ncol, int hypsometric_opt, int use_theta_m, int ideal,
    float p_top, float tiso, float t00, float tlp, float p00,
    float p_strat, float tlp_strat,
    float g, float t0, float p1000mb, float rcp, float rdp, float cvpm,
    float rvovrd)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const long long nc = (long long)ncol;
    const float MUB = mub[c];
    const float MU2 = mu[c];
#define I(k) ((long long)(k) * nc + (long long)c)
    // Uncouple t_2 and QV with the child's column mass (:154-161).
    for (int k = 0; k < nz; ++k) {
        float m = (c1h[k] * MUB + c2h[k]) + (c1h[k] * MU2);
        t[I(k)] = t[I(k)] / m;
        qv[I(k)] = qv[I(k)] / m;
    }
    // Base state on the child's levels (:163-178).
    for (int k = 0; k < nz; ++k) {
        float pbk = (c3h[k] * MUB + c4h[k]) + p_top;
        pb[I(k)] = pbk;
        float t_init;
        if (!ideal) {
            float temp = fmaxf(tiso, t00 + tlp * gfk_log(pbk / p00));
            if (pbk < p_strat) {
                temp = tiso + tlp_strat * gfk_log(pbk / p_strat);
            }
            t_init = temp * gfk_pow(p00 / pbk, rcp) - t0;
        } else {
            t_init = 0.0f;  // ideal runs are refused by the host
        }
        alb[I(k)] = (rdp * (t_init + t0)) * gfk_pow(pbk / p1000mb, cvpm);
    }
    // Base geopotential from the terrain (:183-201).
    phb[I(0)] = ht[c] * g;
    if (hypsometric_opt == 1) {
        for (int kk = 1; kk <= nz; ++kk) {
            const int k = kk - 1;
            float w = dnw[k] * (c1h[k] * MUB + c2h[k]);
            phb[I(kk)] = phb[I(k)] - w * alb[I(k)];
        }
    } else {
        for (int k = 1; k <= nz; ++k) {
            float pfu = (c3f[k] * MUB + c4f[k]) + p_top;
            float pfd = (c3f[k - 1] * MUB + c4f[k - 1]) + p_top;
            float phm = (c3h[k - 1] * MUB + c4h[k - 1]) + p_top;
            phb[I(k)] = phb[I(k - 1)] + (alb[I(k - 1)] * phm) * gfk_log(pfd / pfu);
        }
    }
    // Perturbation pressure and inverse density from the top down
    // (:209-245).  The top level reads QV twice, as WRF does.
    float pnext;
    {
        const int kk = nz - 1;
        const int k = kk + 1;
        float qvf1 = 0.5f * (qv[I(kk)] + qv[I(kk)]);
        float qvf2 = 1.0f / (1.0f + qvf1);
        qvf1 = qvf1 * qvf2;
        float a = c1f[k] * MU2;
        float b = qvf1 * (c1f[k] * MUB + c2f[k]);
        float pk = -(((0.5f * (a + b)) / rdnw[kk]) / qvf2);
        float qvf = use_theta_m == 0 ? 1.0f + rvovrd * qv[I(kk)] : 1.0f;
        float tk = (rdp * (t[I(kk)] + t0)) * qvf;
        al[I(kk)] = tk * gfk_pow((pk + pb[I(kk)]) / p1000mb, cvpm) - alb[I(kk)];
        pnext = pk;
    }
    for (int kk = nz - 2; kk >= 0; --kk) {
        const int k = kk + 1;
        float qvf1 = 0.5f * (qv[I(kk)] + qv[I(kk + 1)]);
        float qvf2 = 1.0f / (1.0f + qvf1);
        qvf1 = qvf1 * qvf2;
        float a = c1f[k] * MU2;
        float b = qvf1 * (c1f[k] * MUB + c2f[k]);
        float pk = pnext - (((a + b) / qvf2) / rdn[kk + 1]);
        float qvf = use_theta_m == 0 ? 1.0f + rvovrd * qv[I(kk)] : 1.0f;
        float tk = (rdp * (t[I(kk)] + t0)) * qvf;
        al[I(kk)] = tk * gfk_pow((pk + pb[I(kk)]) / p1000mb, cvpm) - alb[I(kk)];
        pnext = pk;
    }
    // ph_2 from the hydrostatic equation (:250-277).
    if (hypsometric_opt == 1) {
        for (int kk = 1; kk <= nz; ++kk) {
            const int k = kk - 1;
            float m = (c1h[k] * MUB + c2h[k]) + (c1h[k] * MU2);
            float first = m * al[I(kk - 1)];
            float second = (c1h[k] * MU2) * alb[I(kk - 1)];
            ph[I(kk)] = ph[I(kk - 1)] - dnw[kk - 1] * (first + second);
        }
    } else {
        ph[I(0)] = phb[I(0)];
        for (int k = 1; k <= nz; ++k) {
            float mt = MUB + MU2;
            float pfu = (c3f[k] * mt + c4f[k]) + p_top;
            float pfd = (c3f[k - 1] * mt + c4f[k - 1]) + p_top;
            float phm = (c3h[k - 1] * mt + c4h[k - 1]) + p_top;
            ph[I(k)] = ph[I(k - 1)]
                       + ((alb[I(k - 1)] + al[I(k - 1)]) * phm) * gfk_log(pfd / pfu);
        }
        for (int k = 0; k <= nz; ++k) {
            ph[I(k)] = ph[I(k)] - phb[I(k)];
        }
    }
    // Couple again with the child's coefficients (:285-300).
    for (int k = 0; k <= nz; ++k) {
        float m = (c1f[k] * MUB + c2f[k]) + (c1f[k] * MU2);
        ph[I(k)] = ph[I(k)] * m;
    }
    for (int k = 0; k < nz; ++k) {
        float m = (c1h[k] * MUB + c2h[k]) + (c1h[k] * MU2);
        t[I(k)] = t[I(k)] * m;
        qv[I(k)] = qv[I(k)] * m;
    }
#undef I
}
