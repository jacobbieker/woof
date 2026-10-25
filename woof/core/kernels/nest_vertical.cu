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

// ---------------------------------------------------------------------------
// Two-way feedback from a nest on its own eta ladder, limited in height
// (woof/core/nest_feedback.py).  Not WRF's: WRF v4.7.1 refuses feedback with
// vertical refinement (share/module_check_a_mundo.F:2719-2737,
// "vert_refine_method=2 only works with feedback = 0"), so these are woof's
// own operators, held to their binary64/binary32 NumPy twins at max ULP 0.
//
//   nest_fb_remap     one CHILD column from the child's ladder onto the
//                     parent's, both ladders placed in the child column's own
//                     dry-pressure coordinate pd = c3f*MU + c4f (p_top, shared,
//                     cancels; the bottom edge is MU and the top 0 by
//                     construction).  mode 0: the layer integral
//                     Q_k = sum_j overlap(j, k) * src_j, the mass-coupled
//                     content of parent layer k (conservative: the overlaps
//                     of a child layer sum to its own mass).  mode 1: an
//                     interface value, linear in pd between the child
//                     interfaces that bracket the parent's.  mode 2: the
//                     column's mass in each parent layer, the divisor that
//                     turns a restricted content back into a mean.
//   nest_fb_blend     parent = parent + w_k * (fed - parent), level by level:
//                     w = 1 copies, w = 0 leaves the parent's word alone.
//
// MU at a u (v) point is the mean of the two mass columns beside it, the
// domain edge taking its one neighbour.  Every sum is binary64, in increasing
// child-layer order; compiled -fmad=false --ftz=false.

__device__ __forceinline__ double fb_mu(const float* __restrict__ mub,
                                        const float* __restrict__ mup,
                                        int ny, int nx, int jf, int i_f,
                                        int xstag, int ystag)
{
    int ja = jf, jb = jf, ia = i_f, ib = i_f;
    if (xstag) { ia = i_f - 1 < 0 ? 0 : i_f - 1; ib = i_f > nx - 1 ? nx - 1 : i_f; }
    if (ystag) { ja = jf - 1 < 0 ? 0 : jf - 1; jb = jf > ny - 1 ? ny - 1 : jf; }
    long long a = (long long)ja * nx + ia;
    long long b = (long long)jb * nx + ib;
    double ma = (double)mub[a] + (double)mup[a];
    if (a == b) return ma;
    double mb = (double)mub[b] + (double)mup[b];
    return 0.5 * (ma + mb);
}

__device__ __forceinline__ double fb_edge(const float* __restrict__ c3,
                                          const float* __restrict__ c4,
                                          int k, int nz, double mu)
{
    if (k <= 0) return mu;
    if (k >= nz) return 0.0;
    double a = (double)c3[k] * mu;
    return a + (double)c4[k];
}

extern "C" __global__ void nest_fb_remap(
    const float* __restrict__ src, float* __restrict__ out,
    const float* __restrict__ mub, const float* __restrict__ mup,
    const float* __restrict__ c3c, const float* __restrict__ c4c,
    const float* __restrict__ c3p, const float* __restrict__ c4p,
    int nzc, int nzp, int ny, int nx, int nyf, int nxf,
    int xstag, int ystag, int mode)
{
    long long ncol = (long long)nyf * nxf;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    int jf = (int)(c / nxf), i_f = (int)(c % nxf);
    double mu = fb_mu(mub, mup, ny, nx, jf, i_f, xstag, ystag);
    if (mode == 0) {
        int j0 = 0;
        for (int k = 0; k < nzp; ++k) {
            double bot = fb_edge(c3p, c4p, k, nzp, mu);
            double top = fb_edge(c3p, c4p, k + 1, nzp, mu);
            double acc = 0.0;
            for (int j = j0; j < nzc; ++j) {
                double cb = fb_edge(c3c, c4c, j, nzc, mu);
                if (cb <= top) break;
                double ct = fb_edge(c3c, c4c, j + 1, nzc, mu);
                double ov = fmin(bot, cb) - fmax(top, ct);
                if (ov > 0.0) {
                    double term = ov * (double)src[(long long)j * ncol + c];
                    acc = acc + term;
                }
            }
            while (j0 < nzc && fb_edge(c3c, c4c, j0 + 1, nzc, mu) >= top) ++j0;
            out[(long long)k * ncol + c] = (float)acc;
        }
    } else if (mode == 2) {
        for (int k = 0; k < nzp; ++k) {
            double bot = fb_edge(c3p, c4p, k, nzp, mu);
            double top = fb_edge(c3p, c4p, k + 1, nzp, mu);
            out[(long long)k * ncol + c] = (float)(bot - top);
        }
    } else {
        int j = 0;
        for (int k = 0; k <= nzp; ++k) {
            double x = fb_edge(c3p, c4p, k, nzp, mu);
            while (j < nzc - 1 && fb_edge(c3c, c4c, j + 1, nzc, mu) >= x) ++j;
            double lo = fb_edge(c3c, c4c, j, nzc, mu);
            double hi = fb_edge(c3c, c4c, j + 1, nzc, mu);
            double f = (lo - x) / (lo - hi);
            double a = (double)src[(long long)j * ncol + c];
            double b = (double)src[(long long)(j + 1) * ncol + c];
            double d = b - a;
            double s = f * d;
            out[(long long)k * ncol + c] = (float)(a + s);
        }
    }
}

extern "C" __global__ void nest_fb_blend(
    float* __restrict__ parent, const float* __restrict__ fed,
    const float* __restrict__ w, int nlev, int nyf, int nxf,
    int i0, int j0, int ni, int nj)
{
    long long n = (long long)nlev * nj * ni;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int k = (int)(c / ((long long)nj * ni));
    long long r = c % ((long long)nj * ni);
    int jf = j0 + (int)(r / ni), i_f = i0 + (int)(r % ni);
    long long idx = ((long long)k * nyf + jf) * nxf + i_f;
    float wk = w[k];
    if (wk == 1.0f) {
        parent[idx] = fed[idx];
    } else if (wk != 0.0f) {
        float p = parent[idx];
        float d = fed[idx] - p;
        float s = wk * d;
        parent[idx] = p + s;
    }
}

// nest_fb_ph_column: the height-limited blend of ph, one parent column of the
// blend window.  Geopotential is a vertical integral, so a level-by-level
// blend would hand the column above the limit a thickness its mass does not
// have.  Instead the column is rebuilt upward from the ground, layer by
// layer, on TOTAL geopotential (phb + ph):
//   fed layer (w = 1)    the restricted child thickness,
//   untouched (w = 0)    the parent's own thickness scaled by the change of
//                        its layer mass, dm(MU_new) / dm(MU_old): the layer's
//                        inverse density -- hence its p, theta, q -- is kept,
//   taper (0 < w < 1)    d_old + w * (d_child - d_old).
// The ground point takes the interface blend of the perturbation.
extern "C" __global__ void nest_fb_ph_column(
    float* __restrict__ ph, const float* __restrict__ fed,
    const float* __restrict__ phb, const float* __restrict__ mub,
    const float* __restrict__ mup_old, const float* __restrict__ mup_new,
    const float* __restrict__ c3p, const float* __restrict__ c4p,
    const float* __restrict__ wl, const float* __restrict__ wi,
    int nz, int ny, int nx, int phb_column, int i0, int j0, int ni, int nj)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (long long)ni * nj) return;
    int jf = j0 + (int)(c / ni), i_f = i0 + (int)(c % ni);
    long long plane = (long long)ny * nx;
    long long at = (long long)jf * nx + i_f;
    double mb = (double)mub[at];
    double mu_old = mb + (double)mup_old[at];
    double mu_new = mb + (double)mup_new[at];
#define PHB(k) ((double)(phb_column ? phb[(k)] : phb[(long long)(k) * plane + at]))
    float p0 = ph[at];
    float r0 = fed[at];
    float g0;
    float w0 = wi[0];
    if (w0 == 1.0f) g0 = r0;
    else if (w0 == 0.0f) g0 = p0;
    else { float d = r0 - p0; float s = w0 * d; g0 = p0 + s; }
    double tot = PHB(0) + (double)g0;
    double old_lo = PHB(0) + (double)p0;
    double fed_lo = PHB(0) + (double)r0;
    ph[at] = g0;
    for (int k = 0; k < nz; ++k) {
        long long up = (long long)(k + 1) * plane + at;
        double old_hi = PHB(k + 1) + (double)ph[up];
        double fed_hi = PHB(k + 1) + (double)fed[up];
        double dm_old = fb_edge(c3p, c4p, k, nz, mu_old)
                        - fb_edge(c3p, c4p, k + 1, nz, mu_old);
        double dm_new = fb_edge(c3p, c4p, k, nz, mu_new)
                        - fb_edge(c3p, c4p, k + 1, nz, mu_new);
        double d_old = (old_hi - old_lo) * (dm_new / dm_old);
        double d_fed = fed_hi - fed_lo;
        float w = wl[k];
        double d;
        if (w == 1.0f) d = d_fed;
        else if (w == 0.0f) d = d_old;
        else { double e = d_fed - d_old; double s = (double)w * e; d = d_old + s; }
        tot = tot + d;
        ph[up] = (float)(tot - PHB(k + 1));
        old_lo = old_hi;
        fed_lo = fed_hi;
    }
#undef PHB
}
